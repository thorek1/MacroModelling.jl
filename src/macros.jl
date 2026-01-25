const all_available_algorithms = [:first_order, :second_order, :pruned_second_order, :third_order, :pruned_third_order]


"""
$(SIGNATURES)
Parses the model equations and assigns them to an object.

# Arguments
- `𝓂`: name of the object to be created containing the model information.
- `ex`: equations

# Optional arguments to be placed between `𝓂` and `ex`
- `max_obc_horizon` [Default: `40`, Type: `Int`]: maximum length of anticipated shocks and corresponding unconditional forecast horizon over which the occasionally binding constraint is to be enforced. Increase this number if no solution is found to enforce the constraint.

Variables must be defined with their time subscript in square brackets.
Endogenous variables can have the following:
- present: `c[0]`
- non-stochastic steady state: `c[ss]` instead of `ss` any of the following is also a valid flag for the non-stochastic steady state: `ss`, `stst`, `steady`, `steadystate`, `steady_state`, and the parser is case-insensitive (`SS` or `sTst` will work as well).
- past: `c[-1]` or any negative Integer: e.g. `c[-12]`
- future: `c[1]` or any positive Integer: e.g. `c[16]` or `c[+16]`
Signed integers are recognised and parsed as such.

Exogenous variables (shocks) can have the following:
- present: `eps_z[x]` instead of `x` any of the following is also a valid flag for exogenous variables: `ex`, `exo`, `exogenous`, and the parser is case-insensitive (`Ex` or `exoGenous` will work as well).
- past: `eps_z[x-1]`
- future: `eps_z[x+1]`

Parameters enter the equations without square brackets.

If an equation contains a `max` or `min` operator, the default dynamic (first order) solution of the model will enforce the occasionally binding constraint. This enforcement can be disabled by setting `ignore_obc = true` in the relevant function calls.

# Examples
```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end
```

# Programmatic model writing

Parameters and variables can be indexed using curly braces: e.g. `c{H}[0]`, `eps_z{F}[x]`, or `α{H}`.

`for` loops can be used to write models programmatically. They can either be used to generate expressions where the time index or the index in curly braces is iterated over:
- generate equation with different indices in curly braces: `for co in [H,F] C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1] end = for co in [H,F] Y{co}[0] end`
- generate multiple equations with different indices in curly braces: `for co in [H, F] K{co}[0] = (1-delta{co}) * K{co}[-1] + S{co}[0] end`
- generate equation with different time indices: `Y_annual[0] = for lag in -3:0 Y[lag] end` or `R_annual[0] = for operator = :*, lag in -3:0 R[lag] end`

# Returns
- `Nothing`. The macro creates the model `𝓂` in the calling scope.
"""
macro model(𝓂,ex...)
    # parse options
    verbose = false
    precompile = false
    max_obc_horizon = 40

    for exp in ex[1:end-1]
        postwalk(x -> 
            x isa Expr ?
                x.head == :(=) ?  
                    x.args[1] == :verbose && x.args[2] isa Bool ?
                        verbose = x.args[2] :
                    x.args[1] == :precompile && x.args[2] isa Bool ?
                        precompile = x.args[2] :
                    x.args[1] == :max_obc_horizon && x.args[2] isa Int ?
                        max_obc_horizon = x.args[2] :
                    begin
                        @warn "Invalid option `$(x.args[1])` ignored. See docs: `?@model` for valid options."
                        x
                    end :
                x :
            x,
        exp)
    end

    # create data containers
    parameters = []
    parameter_values = Vector{Float64}(undef,0)

    ss_calib_list = []
    par_calib_list = []
    
    # NSSS struct fields
    NSSS_solve_blocks_in_place = ss_solve_block[]
    NSSS_solver_cache = CircularBuffer{Vector{Vector{Float64}}}(500)
    NSSS_solve_func = x->x
    NSSS_check_func = x->x
    NSSS_custom_function = nothing
    NSSS_∂equations_∂parameters = zeros(0,0)
    NSSS_∂equations_∂parameters_func = x->x
    NSSS_∂equations_∂SS_and_pars = zeros(0,0)
    NSSS_∂equations_∂SS_and_pars_func = x->x
    NSSS_dependencies = nothing

    original_equations = []
    calibration_equations = []
    calibration_equations_parameters = []

    bounds = Dict{Symbol,Tuple{Float64,Float64}}()

    dyn_equations = []

    ➕_vars = []
    ss_and_aux_equations = []
    ss_equations = []
    aux_vars_created = Set()

    unique_➕_eqs = Dict{Union{Expr,Symbol},Expr}()

    ss_equations_with_aux_variables = Int[]
    dyn_eq_aux_ind = Int[]

    model_ex = parse_for_loops(ex[end])
    
    model_ex = resolve_if_expr(model_ex::Expr)::Expr

    model_ex = remove_nothing(model_ex::Expr)::Expr

    model_ex = parse_occasionally_binding_constraints(model_ex::Expr, max_obc_horizon = max_obc_horizon)::Expr
    
    # obc_shock_bounds = Tuple{Symbol, Bool, Float64}[]

    # write down dynamic equations and add auxiliary variables for leads and lags > 1
    for (i,arg) in enumerate(model_ex.args)
        if isa(arg,Expr)
            # write down dynamic equations
            t_ex = postwalk(x -> 
                x isa Expr ? 
                    x.head == :(=) ? 
                        Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                        x.head == :ref ?
                            occursin(r"^(x|ex|exo|exogenous){1}$"i,string(x.args[2])) ?
                                begin
                                    Symbol(string(x.args[1]) * "₍ₓ₎") 
                                end :
                            occursin(r"^(x|ex|exo|exogenous){1}(?=(\s{1}(\-|\+){1}\s{1}\d+$))"i,string(x.args[2])) ?
                                x.args[2].args[1] == :(+) ?
                                    begin
                                        k = x.args[2].args[3]
                
                                        while k > 2 # create auxiliary dynamic equation for exogenous variables with lead > 1
                                            if Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₀₎") ∈ aux_vars_created
                                                break
                                            else
                                                push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₀₎"))
                    
                                                push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(k - 2))) * "⁾₍₁₎")))
                                                push!(dyn_eq_aux_ind,length(dyn_equations))
                                                
                                                k -= 1
                                            end
                                        end

                                        if Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₀₎") ∉ aux_vars_created && k > 1
                                            push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₀₎"))
                    
                                            push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₀₎"), Symbol(string(x.args[1]) * "₍₁₎")))
                                            push!(dyn_eq_aux_ind,length(dyn_equations))
                                        end

                                        if Symbol(string(x.args[1]) * "₍₀₎") ∉ aux_vars_created
                                            push!(aux_vars_created,Symbol(string(x.args[1]) * "₍₀₎"))
                                            
                                            push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "₍₀₎"),Symbol(string(x.args[1]) * "₍ₓ₎")))
                                            push!(dyn_eq_aux_ind,length(dyn_equations))
                                        end

                                        if x.args[2].args[3] > 1
                                            Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(x.args[2].args[3] - 1))) * "⁾₍₁₎")
                                        else
                                            Symbol(string(x.args[1]) * "₍₁₎")
                                        end
                                    end :
                                x.args[2].args[1] == :(-) ?
                                    begin
                                        k = - x.args[2].args[3]
                    
                                        while k < -2 # create auxiliary dynamic equations for exogenous variables with lag < -1
                                            if Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(k + 1))) * "⁾₍₀₎") ∈ aux_vars_created
                                                break
                                            else
                                                push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(k + 1))) * "⁾₍₀₎"))
                    
                                                push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(k + 1))) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(k + 2))) * "⁾₍₋₁₎")))
                                                push!(dyn_eq_aux_ind,length(dyn_equations))
                                                
                                                k += 1
                                            end
                                        end
                    
                                        if Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(k + 1))) * "⁾₍₀₎") ∉ aux_vars_created && k < -1
                                        
                                            push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(k + 1))) * "⁾₍₀₎"))
                    
                                            push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(k + 1))) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "₍₋₁₎")))
                                            push!(dyn_eq_aux_ind,length(dyn_equations))
                                        end
                                        
                                        if Symbol(string(x.args[1]) * "₍₀₎") ∉ aux_vars_created
                                            push!(aux_vars_created,Symbol(string(x.args[1]) * "₍₀₎"))
                                            
                                            push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "₍₀₎"),Symbol(string(x.args[1]) * "₍ₓ₎")))
                                            push!(dyn_eq_aux_ind,length(dyn_equations))
                                        end

                                        if  - x.args[2].args[3] < -1
                                            Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(x.args[2].args[3] - 1))) * "⁾₍₋₁₎")
                                        else
                                            Symbol(string(x.args[1]) * "₍₋₁₎")
                                        end
                                    end :
                                x.args[1] : 
                            occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                                begin
                                    Symbol(string(x.args[1]) * "₍ₛₛ₎") 
                                end :
                            x.args[2] isa Int ? 
                                x.args[2] > 1 ? 
                                    begin
                                        k = x.args[2]

                                        while k > 2 # create auxiliary dynamic equations for endogenous variables with lead > 1
                                            if Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₀₎") ∈ aux_vars_created
                                                break
                                            else
                                                push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₀₎"))

                                                push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(k - 2))) * "⁾₍₁₎")))
                                                push!(dyn_eq_aux_ind,length(dyn_equations))
                                                
                                                k -= 1
                                            end
                                        end

                                        if Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₀₎") ∉ aux_vars_created
                                            push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₀₎"))

                                            push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "₍₁₎")))
                                            push!(dyn_eq_aux_ind,length(dyn_equations))
                                        end
                                        Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(abs(x.args[2] - 1))) * "⁾₍₁₎")
                                    end :
                                1 >= x.args[2] >= 0 ? 
                                    begin
                                        Symbol(string(x.args[1]) * "₍" * sub(string(x.args[2])) * "₎")
                                    end :  
                                -1 <= x.args[2] < 0 ? 
                                    begin
                                        Symbol(string(x.args[1]) * "₍₋" * sub(string(x.args[2])) * "₎")
                                    end :
                                x.args[2] < -1 ?  # create auxiliary dynamic equations for endogenous variables with lag < -1
                                    begin
                                        k = x.args[2]

                                        while k < -2
                                            if Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(k + 1))) * "⁾₍₀₎") ∈ aux_vars_created
                                                break
                                            else
                                                push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(k + 1))) * "⁾₍₀₎"))

                                                push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(k + 1))) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(k + 2))) * "⁾₍₋₁₎")))
                                                push!(dyn_eq_aux_ind,length(dyn_equations))
                                                
                                                k += 1
                                            end
                                        end

                                        if Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(k + 1))) * "⁾₍₀₎") ∉ aux_vars_created
                                            push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(k + 1))) * "⁾₍₀₎"))

                                            push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(k + 1))) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "₍₋₁₎")))
                                            push!(dyn_eq_aux_ind,length(dyn_equations))
                                        end

                                        Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(abs(x.args[2] + 1))) * "⁾₍₋₁₎")
                                    end :
                            x.args[1] :
                        x.args[1] : 
                    unblock(x) : 
                x,
            model_ex.args[i])

            push!(dyn_equations,unblock(t_ex))
            
            
            # write down ss equations
            eqs = postwalk(x -> 
                x isa Expr ? 
                    x.head == :(=) ? 
                        Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                            x.head == :ref ?
                                occursin(r"^(x|ex|exo|exogenous){1}"i,string(x.args[2])) ? 0 : # set shocks to zero and remove time scripts
                        x.args[1] :
                    x.head == :call ?
                        x.args[1] == :* ?
                            x.args[2] isa Int ?
                                x.args[3] isa Int ?
                                    x :
                                Expr(:call, :*, x.args[3:end]..., x.args[2]) : # 2beta => beta * 2 
                            x :
                        x :
                    x :
                x,
            model_ex.args[i])
            push!(ss_equations,flatten(unblock(eqs)))

            # write down ss equations including nonnegativity auxiliary variables
            # find nonegative variables, parameters, or terms
            eqs = postwalk(x -> 
                x isa Expr ? 
                    x.head == :(=) ? 
                        Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                            x.head == :ref ?
                                occursin(r"^(x|ex|exo|exogenous){1}"i,string(x.args[2])) ? 0 : # set shocks to zero and remove time scripts
                        x : 
                    x.head == :call ?
                        x.args[1] == :* ?
                            x.args[2] isa Int ?
                                x.args[3] isa Int ?
                                    x :
                                Expr(:call, :*, x.args[3:end]..., x.args[2]) : # 2beta => beta * 2 
                            x :
                        x.args[1] ∈ [:^] ?
                            !(x.args[3] isa Int) ?
                                x.args[2] isa Symbol ? # nonnegative parameters 
                                        begin
                                            bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 1e12)) : (eps(), 1e12)
                                            x
                                        end :
                                x.args[2].head == :ref ?
                                    x.args[2].args[1] isa Symbol ? # nonnegative variables 
                                        begin
                                            bounds[x.args[2].args[1]] = haskey(bounds, x.args[2].args[1]) ? (max(bounds[x.args[2].args[1]][1], eps()), min(bounds[x.args[2].args[1]][2], 1e12)) : (eps(), 1e12)
                                            x
                                        end :
                                    x :
                                x.args[2].head == :call ? # nonnegative expressions
                                    begin
                                        if precompile
                                            replacement = x.args[2]
                                        else
                                            replacement = simplify(x.args[2])
                                        end

                                        if !(replacement isa Int) # check if the nonnegative term is just a constant
                                            if haskey(unique_➕_eqs, x.args[2])
                                                replacement = unique_➕_eqs[x.args[2]]
                                            else 
                                                lb = eps()
                                                ub = 1e12

                                                # push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(ub,max(lb,$(x.args[2])))))
                                                push!(ss_and_aux_equations, Expr(:call,:-, :($(Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)+1))),0))), x.args[2]))

                                                bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], lb), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], ub)) : (lb, ub)

                                                push!(ss_equations_with_aux_variables,length(ss_and_aux_equations))

                                                push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                                replacement = Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)))),0)

                                                unique_➕_eqs[x.args[2]] = replacement
                                            end
                                        end

                                        :($(replacement) ^ $(x.args[3]))
                                    end :
                                x :
                            x :
                        x.args[2] isa Float64 ?
                            x :
                        x.args[1] ∈ [:log] ?
                            x.args[2] isa Symbol ? # nonnegative parameters 
                                begin
                                    bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 1e12)) : (eps(), 1e12)
                                    x
                                end :
                            x.args[2].head == :ref ?
                                x.args[2].args[1] isa Symbol ? # nonnegative variables 
                                    begin
                                        bounds[x.args[2].args[1]] = haskey(bounds, x.args[2].args[1]) ? (max(bounds[x.args[2].args[1]][1], eps()), min(bounds[x.args[2].args[1]][2], 1e12)) : (eps(), 1e12)
                                        x
                                    end :
                                x :
                            x.args[2].head == :call ? # nonnegative expressions
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int) # check if the nonnegative term is just a constant
                                        if haskey(unique_➕_eqs, x.args[2])
                                            replacement = unique_➕_eqs[x.args[2]]
                                        else
                                            lb = eps()
                                            ub = 1e12

                                            # push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(ub,max(lb,$(x.args[2])))))
                                            push!(ss_and_aux_equations, Expr(:call,:-, :($(Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)+1))),0))), x.args[2]))

                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], lb), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], ub)) : (lb, ub)

                                            push!(ss_equations_with_aux_variables,length(ss_and_aux_equations))

                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)))),0)

                                            unique_➕_eqs[x.args[2]] = replacement
                                        end
                                    end
                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x :
                        x.args[1] ∈ [:norminvcdf, :norminv, :qnorm] ?
                            x.args[2] isa Symbol ? # nonnegative parameters 
                                begin
                                    bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 1-eps())) : (eps(), 1-eps())
                                    x
                                end :
                            x.args[2].head == :ref ?
                                x.args[2].args[1] isa Symbol ? # nonnegative variables 
                                    begin
                                        bounds[x.args[2].args[1]] = haskey(bounds, x.args[2].args[1]) ? (max(bounds[x.args[2].args[1]][1], eps()), min(bounds[x.args[2].args[1]][2], 1-eps())) : (eps(), 1-eps())
                                        x
                                    end :
                                x :
                            x.args[2].head == :call ? # nonnegative expressions
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int) # check if the nonnegative term is just a constant
                                        if haskey(unique_➕_eqs, x.args[2])
                                            replacement = unique_➕_eqs[x.args[2]]
                                        else
                                            lb = eps()
                                            ub = 1-eps()

                                            # push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(ub,max(lb,$(x.args[2])))))
                                            push!(ss_and_aux_equations, Expr(:call,:-, :($(Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)+1))),0))), x.args[2]))

                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], lb), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], ub)) : (lb, ub)

                                            push!(ss_equations_with_aux_variables,length(ss_and_aux_equations))

                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)))),0)

                                            unique_➕_eqs[x.args[2]] = replacement
                                        end
                                    end
                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x :
                        x.args[1] ∈ [:exp] ?
                            x.args[2] isa Symbol ? # have exp terms bound so they dont go to Inf
                                begin
                                    bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], -1e12), min(bounds[x.args[2]][2], 600)) : (-1e12, 600)
                                    x
                                end :
                            x.args[2].head == :ref ?
                                x.args[2].args[1] isa Symbol ? # have exp terms bound so they dont go to Inf
                                    begin
                                        bounds[x.args[2].args[1]] = haskey(bounds, x.args[2].args[1]) ? (max(bounds[x.args[2].args[1]][1], -1e12), min(bounds[x.args[2].args[1]][2], 600)) : (-1e12, 600)
                                        x
                                    end :
                                x :
                            x.args[2].head == :call ? # nonnegative expressions
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int) # check if the nonnegative term is just a constant
                                        if haskey(unique_➕_eqs, x.args[2])
                                            replacement = unique_➕_eqs[x.args[2]]
                                        else
                                            lb = -1e12
                                            ub = 600

                                            # push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(ub,max(lb,$(x.args[2])))))
                                            push!(ss_and_aux_equations, Expr(:call,:-, :($(Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)+1))),0))), x.args[2]))

                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], lb), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], ub)) : (lb, ub)

                                            push!(ss_equations_with_aux_variables,length(ss_and_aux_equations))

                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)))),0)

                                            unique_➕_eqs[x.args[2]] = replacement
                                        end
                                    end
                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x :
                        x.args[1] ∈ [:erfcinv] ?
                            x.args[2] isa Symbol ? # nonnegative parameters 
                                begin
                                    bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 2-eps())) : (eps(), 2-eps())
                                    x
                                end :
                            x.args[2].head == :ref ?
                                x.args[2].args[1] isa Symbol ? # nonnegative variables 
                                    begin
                                        bounds[x.args[2].args[1]] = haskey(bounds, x.args[2].args[1]) ? (max(bounds[x.args[2].args[1]][1], eps()), min(bounds[x.args[2].args[1]][2], 2-eps())) : (eps(), 2-eps())
                                        x
                                    end :
                                x :
                            x.args[2].head == :call ? # nonnegative expressions
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int) # check if the nonnegative term is just a constant
                                        if haskey(unique_➕_eqs, x.args[2])
                                            replacement = unique_➕_eqs[x.args[2]]
                                        else
                                            lb = eps()
                                            ub = 2-eps()

                                            # push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(ub,max(lb,$(x.args[2])))))
                                            push!(ss_and_aux_equations, Expr(:call,:-, :($(Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)+1))),0))), x.args[2]))

                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], lb), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], ub)) : (lb, ub)

                                            push!(ss_equations_with_aux_variables,length(ss_and_aux_equations))

                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)))),0)

                                            unique_➕_eqs[x.args[2]] = replacement
                                        end
                                    end
                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x :
                        x :
                    x :
                x,
            model_ex.args[i])
            push!(ss_and_aux_equations,unblock(eqs))
        end
    end

    # go through changed SS equations including nonnegative auxiliary variables
    ss_aux_equations = Expr[]

    # tag vars and pars in changed SS equations
    var_list_aux_SS = []
    ss_list_aux_SS = []
    par_list_aux_SS = []

    var_future_list_aux_SS = []
    var_present_list_aux_SS = []
    var_past_list_aux_SS = []

    # # label all variables parameters and exogenous variables and timings for changed SS equations including nonnegativity auxiliary variables
    for (idx,eq) in enumerate(ss_and_aux_equations)
        var_tmp = Set()
        ss_tmp = Set()
        par_tmp = Set()
        var_future_tmp = Set()
        var_present_tmp = Set()
        var_past_tmp = Set()

        # remove terms multiplied with 0
        eq = postwalk(x -> 
            x isa Expr ? 
                x.head == :call ? 
                    x.args[1] == :* ?
                        any(x.args[2:end] .== 0) ? 
                            0 :
                        x :
                    x :
                x :
            x,
        eq)

        # label all variables parameters and exogenous variables and timings for individual equations
        postwalk(x -> 
            x isa Expr ? 
                x.head == :call ? 
                    for i in 2:length(x.args)
                        x.args[i] isa Symbol ? 
                            occursin(r"^(ss|stst|steady|steadystate|steady_state|x|ex|exo|exogenous){1}$"i,string(x.args[i])) ? 
                                x :
                            push!(par_tmp,x.args[i]) : 
                        x
                    end :
                x.head == :ref ? 
                    x.args[2] isa Int ? 
                        x.args[2] == 0 ? 
                            push!(var_present_tmp,x.args[1]) : 
                        x.args[2] > 0 ? 
                            push!(var_future_tmp,x.args[1]) : 
                        x.args[2] < 0 ? 
                            push!(var_past_tmp,x.args[1]) : 
                        x :
                    occursin(r"^(x|ex|exo|exogenous){1}(?=(\s{1}\-{1}\s{1}\d+$))"i,string(x.args[2])) ?
                        push!(var_past_tmp,x.args[1]) : 
                    occursin(r"^(x|ex|exo|exogenous){1}(?=(\s{1}\+{1}\s{1}\d+$))"i,string(x.args[2])) ?
                        push!(var_future_tmp,x.args[1]) : 
                    occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                        push!(ss_tmp,x.args[1]) :
                    x : 
                x :
            x,
        eq)

        var_tmp = union(var_future_tmp,var_present_tmp,var_past_tmp)
        
        push!(var_list_aux_SS,var_tmp)
        push!(ss_list_aux_SS,ss_tmp)
        push!(par_list_aux_SS,par_tmp)
        push!(var_future_list_aux_SS,var_future_tmp)
        push!(var_present_list_aux_SS,var_present_tmp)
        push!(var_past_list_aux_SS,var_past_tmp)


        # write down SS equations including nonnegativity auxiliary variables
        prs_ex = convert_to_ss_equation(eq)
        
        if idx ∈ ss_equations_with_aux_variables
            if precompile
                ss_aux_equation = Expr(:call,:-,unblock(prs_ex).args[2],unblock(prs_ex).args[3]) 
            else
                ss_aux_equation = Expr(:call,:-,unblock(prs_ex).args[2],simplify(unblock(prs_ex).args[3])) # simplify RHS if nonnegative auxiliary variable
            end
        else
            if precompile
                ss_aux_equation = unblock(prs_ex)
            else
                ss_aux_equation = simplify(unblock(prs_ex))
            end
        end
        
        if ss_aux_equation isa Symbol 
            push!(ss_aux_equations, Expr(:call,:-,ss_aux_equation,0))
        else#if !(ss_aux_equation isa Int)
            push!(ss_aux_equations, ss_aux_equation)
        end
    end

    # go through dynamic equations and label
    # create timings
    dyn_var_future_list  = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₁₎" => "")),x)),collect.(match_pattern.(get_symbols.(dyn_equations),r"₍₁₎")))
    dyn_var_present_list = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₀₎" => "")),x)),collect.(match_pattern.(get_symbols.(dyn_equations),r"₍₀₎")))
    dyn_var_past_list    = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₋₁₎"=> "")),x)),collect.(match_pattern.(get_symbols.(dyn_equations),r"₍₋₁₎")))
    dyn_exo_list         = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),x)),collect.(match_pattern.(get_symbols.(dyn_equations),r"₍ₓ₎")))
    dyn_ss_list          = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₛₛ₎" => "")),x)),collect.(match_pattern.(get_symbols.(dyn_equations),r"₍ₛₛ₎")))

    all_symbols = reduce(union,collect.(get_symbols.(dyn_equations)))
    parameters_in_equations = sort(collect(setdiff(all_symbols,match_pattern(all_symbols,r"₎$"))))
    
    dyn_var_future  =  sort(collect(reduce(union,dyn_var_future_list)))
    dyn_var_present =  sort(collect(reduce(union,dyn_var_present_list)))
    dyn_var_past    =  sort(collect(reduce(union,dyn_var_past_list)))
    dyn_var_ss      =  sort(collect(reduce(union,dyn_ss_list)))

    all_dyn_vars        = union(dyn_var_future, dyn_var_present, dyn_var_past)

    @assert length(setdiff(dyn_var_ss, all_dyn_vars)) == 0 "The following variables are (and cannot be) defined only in steady state (`[ss]`): $(setdiff(dyn_var_ss, all_dyn_vars))"

    all_vars = union(all_dyn_vars, dyn_var_ss)

    present_only              = sort(setdiff(dyn_var_present,union(dyn_var_past,dyn_var_future)))
    future_not_past           = sort(setdiff(dyn_var_future, dyn_var_past))
    past_not_future           = sort(setdiff(dyn_var_past, dyn_var_future))
    mixed                     = sort(setdiff(dyn_var_present, union(present_only, future_not_past, past_not_future)))
    future_not_past_and_mixed = sort(union(future_not_past,mixed))
    past_not_future_and_mixed = sort(union(past_not_future,mixed))
    present_but_not_only      = sort(setdiff(dyn_var_present,present_only))
    mixed_in_past             = sort(intersect(dyn_var_past, mixed))
    not_mixed_in_past         = sort(setdiff(dyn_var_past,mixed_in_past))
    mixed_in_future           = sort(intersect(dyn_var_future, mixed))
    exo                       = sort(collect(reduce(union,dyn_exo_list)))
    var                       = sort(dyn_var_present)
    aux_tmp                   = sort(filter(x->occursin(r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾",string(x)), dyn_var_present))
    aux                       = sort(aux_tmp[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∉ exo, aux_tmp)])
    exo_future                = dyn_var_future[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∈ exo, dyn_var_future)]
    exo_present               = dyn_var_present[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∈ exo, dyn_var_present)]
    exo_past                  = dyn_var_past[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∈ exo, dyn_var_past)]

    nPresent_only              = length(present_only)
    nMixed                     = length(mixed)
    nFuture_not_past_and_mixed = length(future_not_past_and_mixed)
    nPast_not_future_and_mixed = length(past_not_future_and_mixed)
    nPresent_but_not_only      = length(present_but_not_only)
    nVars                      = length(all_vars)
    nExo                       = length(collect(exo))

    present_only_idx              = indexin(present_only,var)
    present_but_not_only_idx      = indexin(present_but_not_only,var)
    future_not_past_and_mixed_idx = indexin(future_not_past_and_mixed,var)
    past_not_future_and_mixed_idx = indexin(past_not_future_and_mixed,var)
    mixed_in_future_idx           = indexin(mixed_in_future,dyn_var_future)
    mixed_in_past_idx             = indexin(mixed_in_past,dyn_var_past)
    not_mixed_in_past_idx         = indexin(not_mixed_in_past,dyn_var_past)
    past_not_future_idx           = indexin(past_not_future,var)

    reorder       = indexin(var, [present_only; past_not_future; future_not_past_and_mixed])
    dynamic_order = indexin(present_but_not_only, [past_not_future; future_not_past_and_mixed])

    @assert length(intersect(union(var,exo),parameters_in_equations)) == 0 "Parameters and variables cannot have the same name. This is the case for: " * repr(sort([intersect(union(var,exo),parameters_in_equations)...]))

    # Check that no variable, shock, or parameter names conflict with SymPyWorkspace reserved names
    reserved_conflicts_vars = intersect(var, SYMPYWORKSPACE_RESERVED_NAMES)
    reserved_conflicts_exo = intersect(exo, SYMPYWORKSPACE_RESERVED_NAMES)
    reserved_conflicts_params = intersect(parameters_in_equations, SYMPYWORKSPACE_RESERVED_NAMES)
    
    @assert length(reserved_conflicts_vars) == 0 "The following variable names are reserved and cannot be used: " * repr(sort([reserved_conflicts_vars...]))
    @assert length(reserved_conflicts_exo) == 0 "The following shock names are reserved and cannot be used: " * repr(sort([reserved_conflicts_exo...]))
    @assert length(reserved_conflicts_params) == 0 "The following parameter names are reserved and cannot be used: " * repr(sort([reserved_conflicts_params...]))

    @assert !any(isnothing, future_not_past_and_mixed_idx) "The following variables appear in the future only (and should at least appear in the present as well): $(setdiff(future_not_past_and_mixed, var)))"

    @assert !any(isnothing, past_not_future_and_mixed_idx) "The following variables appear in the past only (and should at least appear in the present as well): $(setdiff(future_not_past_and_mixed, var)))"

    aux_future_tmp  = sort(filter(x->occursin(r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾",string(x)), dyn_var_future))
    aux_future      = aux_future_tmp[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∉ exo, aux_future_tmp)]

    aux_past_tmp    = sort(filter(x->occursin(r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾",string(x)), dyn_var_past))
    aux_past        = aux_past_tmp[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∉ exo, aux_past_tmp)]

    aux_present_tmp = sort(filter(x->occursin(r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾",string(x)), dyn_var_present))
    aux_present     = aux_present_tmp[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∉ exo, aux_present_tmp)]

    vars_in_ss_equations = sort(collect(setdiff(reduce(union, get_symbols.(ss_aux_equations)), parameters_in_equations)))
    vars_in_ss_equations_no_aux = setdiff(vars_in_ss_equations, ➕_vars)

    dyn_future_list =   match_pattern.(get_symbols.(dyn_equations),r"₍₁₎")
    dyn_present_list =  match_pattern.(get_symbols.(dyn_equations),r"₍₀₎")
    dyn_past_list =     match_pattern.(get_symbols.(dyn_equations),r"₍₋₁₎")
    dyn_exo_list =      match_pattern.(get_symbols.(dyn_equations),r"₍ₓ₎")

    T = post_model_macro(
                max_obc_horizon,
                # present_only,
                # future_not_past,
                # past_not_future,
                # mixed,
                future_not_past_and_mixed,
                past_not_future_and_mixed,
                # present_but_not_only,
                # mixed_in_past,
                # not_mixed_in_past,
                # mixed_in_future,

                var,

                parameters_in_equations,

                exo,
                exo_past,
                exo_present,
                exo_future,

                aux,
                aux_past,
                aux_present,
                aux_future,

                ➕_vars,

                nPresent_only,
                nMixed,
                nFuture_not_past_and_mixed,
                nPast_not_future_and_mixed,
                # nPresent_but_not_only,
                nVars,
                nExo,

                present_only_idx,
                present_but_not_only_idx,
                future_not_past_and_mixed_idx,
                not_mixed_in_past_idx,
                past_not_future_and_mixed_idx,
                mixed_in_past_idx,
                mixed_in_future_idx,
                past_not_future_idx,

                reorder,
                dynamic_order,
                vars_in_ss_equations,
                vars_in_ss_equations_no_aux,

                dyn_var_future_list,
                dyn_var_present_list,
                dyn_var_past_list,
                dyn_ss_list,
                dyn_exo_list,

                dyn_future_list,
                dyn_present_list,
                dyn_past_list,

                var_list_aux_SS,
                ss_list_aux_SS,
                par_list_aux_SS,
                var_future_list_aux_SS,
                var_present_list_aux_SS,
                var_past_list_aux_SS,
                ss_equations_with_aux_variables)

    ℂ = Constants(T)

    𝓦 = Workspaces()


    # write down original equations as written down in model block
    for (i,arg) in enumerate(model_ex.args)
        if isa(arg,Expr)
            prs_exx = postwalk(x -> 
                x isa Expr ? 
                    unblock(x) : 
                x,
            model_ex.args[i])
            push!(original_equations,unblock(prs_exx))
        end
    end
    
    single_dyn_vars_equations = findall(length.(vcat.(collect.(dyn_var_future_list),
                                                      collect.(dyn_var_present_list),
                                                      collect.(dyn_var_past_list),
                                                    #   collect.(dyn_ss_list), # needs to be dynamic after all
                                                      collect.(dyn_exo_list))) .== 1)
                                                    
    @assert length(single_dyn_vars_equations) == 0 "Equations must contain more than 1 dynamic variable. This is not the case for: " * repr([original_equations[indexin(single_dyn_vars_equations,setdiff(1:length(dyn_equations),dyn_eq_aux_ind .- 1))]...])
    
    duplicate_equations = []
    for item in unique(dyn_equations)
        indices = findall(x -> x == item, dyn_equations)
        if length(indices) > 1
            push!(duplicate_equations, indices)
        end
    end
    
    @assert length(duplicate_equations) == 0 "The following equations appear more than once (and should only appear once): \n" * join(["$(original_equations[eq_idxs[1]])" for eq_idxs in duplicate_equations], "\n")
    
    # default_optimizer = nlboxsolve
    # default_optimizer = Optimisers.Adam
    # default_optimizer = NLopt.LN_BOBYQA
    
    #assemble data container
    model_name = string(𝓂)
    quote
        global $𝓂 =  ℳ(
                        $model_name,
                        # $default_optimizer,
                        # sort(collect($parameters_in_equations)),
                        $parameter_values,

                        non_stochastic_steady_state(
                            $NSSS_solve_blocks_in_place,
                            $NSSS_dependencies
                        ),

                        equations($original_equations, $dyn_equations, $ss_equations, $ss_aux_equations, Expr[], $calibration_equations, Expr[], Symbol[]), 

                        caches(
                            zeros(0,0), # jacobian
                            zeros(0,0), # jacobian_parameters
                            zeros(0,0), # jacobian_SS_and_pars
                            zeros(0,0), # hessian
                            zeros(0,0), # hessian_parameters
                            zeros(0,0), # hessian_SS_and_pars
                            zeros(0,0), # third_order_derivatives
                            zeros(0,0), # third_order_derivatives_parameters
                            zeros(0,0), # third_order_derivatives_SS_and_pars
                            zeros(0,0), # first_order_solution_matrix
                            zeros(0,0), # qme_solution
                            Float64[],  # second_order_stochastic_steady_state
                            SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), # second_order_solution
                            Float64[],  # pruned_second_order_stochastic_steady_state
                            Float64[],  # third_order_stochastic_steady_state
                            SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), # third_order_solution
                            Float64[],  # pruned_third_order_stochastic_steady_state
                            Float64[],  # non_stochastic_steady_state
                            $NSSS_solver_cache,  # solver_cache
                            $NSSS_∂equations_∂parameters,  # ∂equations_∂parameters
                            $NSSS_∂equations_∂SS_and_pars,  # ∂equations_∂SS_and_pars
                        ),
                        # (x->x, SparseMatrixCSC{Float64, Int64}(ℒ.I, 0, 0), 𝒟.prepare_jacobian(x->x, 𝒟.AutoForwardDiff(), [0]), SparseMatrixCSC{Float64, Int64}(ℒ.I, 0, 0)), # third_order_derivatives
                        # ([], SparseMatrixCSC{Float64, Int64}(ℒ.I, 0, 0)), # model_jacobian
                        # ([], Int[], zeros(1,1)), # model_jacobian
                        # # x->x, # model_jacobian_parameters
                        # ([], SparseMatrixCSC{Float64, Int64}(ℒ.I, 0, 0)), # model_jacobian_SS_and_pars_vars
                        # # FWrap{Tuple{Vector{Float64}, Vector{Number}, Vector{Float64}}, SparseMatrixCSC{Float64}}(model_jacobian),
                        # ([], SparseMatrixCSC{Float64, Int64}(ℒ.I, 0, 0)),#x->x, # model_hessian
                        # ([], SparseMatrixCSC{Float64, Int64}(ℒ.I, 0, 0)), # model_hessian_SS_and_pars_vars
                        # ([], SparseMatrixCSC{Float64, Int64}(ℒ.I, 0, 0)),#x->x, # model_third_order_derivatives
                        # ([], SparseMatrixCSC{Float64, Int64}(ℒ.I, 0, 0)),#x->x, # model_third_order_derivatives_SS_and_pars_vars

                        # $T,

                        $ℂ,
                        $𝓦,

                        model_functions(
                            $NSSS_solve_func,
                            $NSSS_check_func,
                            $NSSS_custom_function,
                            $NSSS_∂equations_∂parameters_func, # NSSS_∂equations_∂parameters
                            $NSSS_∂equations_∂SS_and_pars_func, # NSSS_∂equations_∂SS_and_pars
                            x->x, # jacobian
                            x->x, # jacobian_parameters
                            x->x, # jacobian_SS_and_pars
                            x->x, # hessian
                            x->x, # hessian_parameters
                            x->x, # hessian_SS_and_pars
                            x->x, # third_order_derivatives
                            x->x, # third_order_derivatives_parameters
                            x->x, # third_order_derivatives_SS_and_pars
                            (x,y)->nothing, # first_order_state_update
                            (x,y)->nothing, # first_order_state_update_obc
                            (x,y)->nothing, # second_order_state_update
                            (x,y)->nothing, # second_order_state_update_obc
                            (x,y)->nothing, # pruned_second_order_state_update
                            (x,y)->nothing, # pruned_second_order_state_update_obc
                            (x,y)->nothing, # third_order_state_update
                            (x,y)->nothing, # third_order_state_update_obc
                            (x,y)->nothing, # pruned_third_order_state_update
                            (x,y)->nothing, # pruned_third_order_state_update_obc
                            x->x, # obc_violation
                        ),

                        solution(
                            # Set([:first_order]),
                            Set(all_available_algorithms),
                            true,
                            false
                        ),

                        # Dict{Vector{Symbol}, timings}() # estimation_helper
                    );
    end
end






"""
$(SIGNATURES)
Adds parameter values and calibration equations to the previously defined model. Allows to provide an initial guess for the non-stochastic steady state (NSSS).

# Arguments
- `𝓂`: name of the object previously created containing the model information.
- `ex`: parameter, parameters values, and calibration equations

Parameters can be defined in either of the following ways:
- plain number: `δ = 0.02`
- expression containing numbers: `δ = 1/50`
- expression containing other parameters: `δ = 2 * std_z` in this case it is irrelevant if `std_z` is defined before or after. The definitions including other parameters are treated as a system of equations and solved accordingly.
- expressions containing a target parameter and an equations with endogenous variables in the non-stochastic steady state, and other parameters, or numbers: `k[ss] / (4 * q[ss]) = 1.5 | δ` or `α | 4 * q[ss] = δ * k[ss]` in this case the target parameter will be solved simultaneously with the non-stochastic steady state using the equation defined with it.

# Optional arguments to be placed between `𝓂` and `ex`
- `guess` [Type: `Dict{Symbol, <:Real}, Dict{String, <:Real}}`]: Guess for the non-stochastic steady state. The keys must be the variable (and calibrated parameters) names and the values the guesses. Missing values are filled with standard starting values.
- $STEADY_STATE_FUNCTION®
- `verbose` [Default: `false`, Type: `Bool`]: print more information about how the non-stochastic steady state is solved
- `silent` [Default: `false`, Type: `Bool`]: do not print any information
- `symbolic` [Default: `false`, Type: `Bool`]: try to solve the non-stochastic steady state symbolically and fall back to a numerical solution if not possible
- `perturbation_order` [Default: `1`, Type: `Int`]: take derivatives only up to the specified order at this stage. When working with higher order perturbation later on, respective derivatives will be taken at that stage.
- `simplify` [Default: `true`, Type: `Bool`]: whether to eliminate redundant variables and simplify the non-stochastic steady state (NSSS) problem. Setting this to `false` can speed up the process, but might make it harder to find the NSSS. If the model does not parse at all (at step 1 or 2), setting this option to `false` might solve it.
- `ss_solver_parameters_algorithm` [Default: `:ESCH`, Type: `Symbol`]: global optimization routine used when searching for steady-state solver parameters after an initial failure; choose `:ESCH` (evolutionary) or `:SAMIN` (simulated annealing). `:SAMIN` is available only when Optim.jl is loaded.
- `ss_solver_parameters_maxtime` [Default: `120.0`, Type: `Real`]: time budget in seconds for the steady-state solver parameter search when `ss_solver_parameters_algorithm` is invoked

# Delayed parameter definition
Not all parameters need to be defined in the `@parameters` macro. Calibration equations using the `|` syntax and parameters defined as functions of other parameters must be declared here, but simple parameter value assignments (e.g., `α = 0.5`) can be deferred and provided later by passing them to any function that accepts the `parameters` argument (e.g., [`get_irf`](@ref), [`get_steady_state`](@ref), [`simulate`](@ref)). 

**Parameter ordering:** When some parameters are not defined in `@parameters`, the final parameter vector follows a specific order: first come the parameters defined in `@parameters` (in their declaration order), followed by any missing parameters (in alphabetical order). This ordering is important when passing parameter values by position rather than by name in subsequent function calls.

# Examples
```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC verbose = true begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

@model RBC_calibrated begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC_calibrated verbose = true guess = Dict(:k => 3) begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    k[ss] / q[ss] = 2.5 | α
    β = 0.95
end
```

# Programmatic model writing
Variables and parameters indexed with curly braces can be either referenced specifically (e.g. `c{H}[ss]`) or generally (e.g. `alpha`). If they are referenced generally the parse assumes all instances (indices) are meant. For example, in a model where `alpha` has two indices `H` and `F`, the expression `alpha = 0.3` is interpreted as two expressions: `alpha{H} = 0.3` and `alpha{F} = 0.3`. The same goes for calibration equations.

# Returns
- `Nothing`. The macro assigns parameter values and calibration equations to `𝓂` in the calling scope.
"""
macro parameters(𝓂,ex...)
    calib_equations = []
    calib_equations_no_var = []
    calib_values_no_var = []
    
    calib_parameters_no_var = Symbol[]
    
    calib_eq_parameters = Symbol[]
    calib_equations_list = Expr[]
    
    ss_calib_list = []
    par_calib_list = []
    
    
    calib_equations_no_var_list = []
    
    ss_no_var_calib_list = []
    par_no_var_calib_list = []
    par_no_var_calib_rhs_list = []
    
    calib_parameters = Symbol[]
    calib_values = Float64[]

    par_defined_more_than_once = Set()
    
    bounded_vars = []

    # parse options
    verbose = false
    silent = false
    symbolic = false
    precompile = false
    report_missing_parameters = true
    perturbation_order = 1
    guess = Dict{Symbol,Float64}()
    simplify = true
    steady_state_function = nothing
    ss_solver_parameters_algorithm = :ESCH
    ss_solver_parameters_maxtime = 120.0

    for exp in ex[1:end-1]
        postwalk(x -> 
            x isa Expr ?
                x.head == :(=) ?  
                    (x.args[1] == :symbolic && x.args[2] isa Bool) ?
                        symbolic = x.args[2] :
                    (x.args[1] == :verbose && x.args[2] isa Bool) ?
                        verbose = x.args[2] :
                    (x.args[1] == :silent && x.args[2] isa Bool) ?
                        silent = x.args[2] :
                    (x.args[1] == :report_missing_parameters && x.args[2] isa Bool) ?
                        report_missing_parameters = x.args[2] :
                    (x.args[1] == :precompile && x.args[2] isa Bool) ?
                        precompile = x.args[2] :
                    (x.args[1] == :perturbation_order && x.args[2] isa Int) ?
                        perturbation_order = x.args[2] :
                    (x.args[1] == :guess && (isa(eval(x.args[2]), Dict{Symbol, <:Real}) || isa(eval(x.args[2]), Dict{String, <:Real}))) ?
                        guess = x.args[2] :
                    (x.args[1] == :ss_solver_parameters_algorithm && (x.args[2] isa Symbol || (x.args[2] isa QuoteNode && x.args[2].value isa Symbol))) ?
                        ss_solver_parameters_algorithm = x.args[2] isa QuoteNode ? x.args[2].value : x.args[2] :
                    (x.args[1] == :simplify && x.args[2] isa Bool) ?
                        simplify = x.args[2] :
                    (x.args[1] == :steady_state_function && x.args[2] isa Symbol) ? # allow Symbol, anonymous fn, or any callable expr
                        steady_state_function = esc(x.args[2]) :
                    (x.args[1] == :ss_solver_parameters_maxtime && x.args[2] isa Real) ?
                        ss_solver_parameters_maxtime = x.args[2] :
                    begin
                        @warn "Invalid option `$(x.args[1])` ignored. See docs: `?@parameters` for valid options."
                        x
                    end :
                x :
            x,
        exp)
    end
    
    @assert ss_solver_parameters_algorithm ∈ [:ESCH, :SAMIN] "ss_solver_parameters_algorithm must be :ESCH or :SAMIN. Got $ss_solver_parameters_algorithm. Using default :ESCH."
    
    parameter_definitions = replace_indices(ex[end])

    # parse parameter inputs
    # label all variables parameters and exogenous variables and timings across all equations
    postwalk(x -> 
        x isa Expr ?
            x.head == :(=) ? 
                x.args[1] isa Symbol ?
                    typeof(x.args[2]) ∈ [Int, Float64] ?
                        begin # normal calibration by setting values of parameters
                            push!(calib_values,x.args[2])
                            if x.args[1] ∈ union(union(calib_parameters,calib_parameters_no_var),calib_eq_parameters) push!(par_defined_more_than_once,x.args[1]) end 
                            push!(calib_parameters,x.args[1]) 
                        end :
                    x.args[2] isa Symbol ?
                        begin # normal calibration by setting values of parameters
                            push!(calib_values_no_var,unblock(x.args[2]))
                            if x.args[1] ∈ union(union(calib_parameters,calib_parameters_no_var),calib_eq_parameters) push!(par_defined_more_than_once,x.args[1]) end
                            push!(calib_parameters_no_var,x.args[1])
                        end :
                    x.args[2].args[1] == :| ?
                        x :
                    begin # normal calibration by setting values of parameters
                        push!(calib_values_no_var,unblock(x.args[2]))
                        if x.args[1] ∈ union(union(calib_parameters,calib_parameters_no_var),calib_eq_parameters) push!(par_defined_more_than_once,x.args[1]) end
                        push!(calib_parameters_no_var,x.args[1])
                    end :
                x.args[1].args[1] == :| ?
                    begin # calibration by targeting SS values (conditional parameter at the beginning)
                        if x.args[1].args[2] ∈ union(union(calib_parameters,calib_parameters_no_var),calib_eq_parameters) push!(par_defined_more_than_once,x.args[1].args[2]) end
                        push!(calib_eq_parameters,x.args[1].args[2])
                        push!(calib_equations,Expr(:(=),x.args[1].args[3], unblock(x.args[2])))
                    end :
                x :
            x.head == :comparison ? 
                push!(bounded_vars,x) :
            x.head == :call ?
                issubset([x.args[1]], [:(<) :(>) :(<=) :(>=)]) ?
                    push!(bounded_vars,x) :
                x :
            x :
        x,
    parameter_definitions)



    postwalk(x -> 
        x isa Expr ?
            x.head == :(=) ? 
                typeof(x.args[2]) ∈ [Int, Float64] ?
                    x :
                x.args[1] isa Symbol ?# || x.args[1] isa Expr ? # this doesn't work really well yet
                    x.args[2] isa Expr ?
                        x.args[2].args[1] == :| ? # capture this case: b_star = b_share * y[ss] | b_star
                            begin # this is calibration by targeting SS values (conditional parameter at the end)
                                if x.args[2].args[end] ∈ union(union(calib_parameters,calib_parameters_no_var),calib_eq_parameters) push!(par_defined_more_than_once, x.args[2].args[end]) end
                                push!(calib_eq_parameters,x.args[2].args[end])#.args[end])
                                push!(calib_equations,Expr(:(=),x.args[1], unblock(x.args[2].args[2])))#.args[2])))
                            end :
                            x :
                        x :
                x.args[2].head == :block ?
                    x.args[1].args[1] == :| ?
                        x :
                    x.args[2].args[2].args[1] == :| ?
                        begin # this is calibration by targeting SS values (conditional parameter at the end)
                            if x.args[2].args[end].args[end] ∈ union(union(calib_parameters,calib_parameters_no_var),calib_eq_parameters) push!(par_defined_more_than_once, x.args[2].args[end].args[end]) end
                            push!(calib_eq_parameters,x.args[2].args[end].args[end])
                            push!(calib_equations,Expr(:(=),x.args[1], unblock(x.args[2].args[2].args[2])))
                        end :
                    begin 
                        @warn "Invalid parameter input ignored: " * repr(x)
                        x
                    end :
                x.args[2].head == :call ?
                    x.args[1].args[1] == :| ?
                            x :
                    begin # this is calibration by targeting SS values (conditional parameter at the end)
                        if x.args[2].args[end] ∈ union(union(calib_parameters,calib_parameters_no_var),calib_eq_parameters) push!(par_defined_more_than_once, x.args[2].args[end]) end
                        push!(calib_eq_parameters, x.args[2].args[end])
                        push!(calib_equations, Expr(:(=),x.args[1], unblock(x.args[2].args[2])))
                    end :
                x :
            x :
        x,
    parameter_definitions)
    
    @assert length(par_defined_more_than_once) == 0 "Parameters can only be defined once. This is not the case for: " * repr([par_defined_more_than_once...])
    
    # Check that no parameter names conflict with SymPyWorkspace reserved names
    all_params = union(calib_parameters, calib_parameters_no_var, calib_eq_parameters)
    reserved_conflicts_params = intersect(all_params, SYMPYWORKSPACE_RESERVED_NAMES)
    @assert length(reserved_conflicts_params) == 0 "The following parameter names are reserved and cannot be used: " * repr(sort([reserved_conflicts_params...]))
    
    # evaluate inputs where they are of the type: log(1/3) (no variables but need evaluation to become a Float64)
    for (i, v) in enumerate(calib_values_no_var)
        out = try eval(v) catch e end
        if out isa Float64
            push!(calib_parameters, calib_parameters_no_var[i])
            push!(calib_values, out)
        else
            push!(calib_equations_no_var, Expr(:(=),calib_parameters_no_var[i], calib_values_no_var[i]))
        end
    end
    
    calib_parameters_no_var = setdiff(calib_parameters_no_var, calib_parameters)
    
    for (i, cal_eq) in enumerate(calib_equations)
        ss_tmp = Set{Symbol}()
        par_tmp = Set()
    
        # parse SS variables
        postwalk(x -> 
            x isa Expr ? 
                x.head == :ref ?
                    occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                        push!(ss_tmp,x.args[1]) :
                    x : 
                x :
            x,
        cal_eq)
    
        # separate out parameters
        postwalk(x -> 
            x isa Symbol ? 
                occursin(r"^(\+|\-|\*|\/|\^|ss|stst|steady|steadystate|steady_state){1}$"i,string(x)) ?
                    x :
                    begin
                        diffed = intersect(setdiff([x], ss_tmp), get_symbols(cal_eq))
                        if !isempty(diffed)
                            push!(par_tmp,diffed[1])
                        end
                    end :
            x,
        cal_eq)
    
        push!(ss_calib_list,ss_tmp)
        push!(par_calib_list,par_tmp)
        
        # write down calibration equations
        prs_ex = postwalk(x -> 
            x isa Expr ? 
                x.head == :(=) ? 
                    Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                        x.head == :ref ?
                            occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ? # K[ss] => K
                        x.args[1] : 
                    x : 
                x.head == :call ?
                    x.args[1] == :* ?
                        x.args[2] isa Int ?
                            x.args[3] isa Int ?
                                x :
                            :($(x.args[3]) * $(x.args[2])) : # 2Π => Π*2 (the former doesn't work with sympy)
                        x :
                    x :
                unblock(x) : 
            x,
            cal_eq)
        push!(calib_equations_list,unblock(prs_ex))
    end
    
    # parse calibration equations without a variable present: eta = Pi_bar /2 (Pi_bar is also a parameter)
    for (i, cal_eq) in enumerate(calib_equations_no_var)
        ss_tmp = Set()
        par_tmp = Set()
    
        # parse SS variables
        postwalk(x -> 
            x isa Expr ? 
                x.head == :ref ?
                    occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                        push!(ss_tmp,x.args[1]) :
                    x : 
                x :
            x,
            cal_eq)
    
        # get SS variables per non_linear_solved_vals
        postwalk(x -> 
        x isa Symbol ? 
            occursin(r"^(\+|\-|\*|\/|\^|ss|stst|steady|steadystate|steady_state){1}$"i,string(x)) ?
                x :
                begin
                    diffed = setdiff([x],ss_tmp)
                    if !isempty(diffed)
                        push!(par_tmp,diffed[1])
                    end
                end :
        x,
        cal_eq)
    
        push!(ss_no_var_calib_list,ss_tmp)
        push!(par_no_var_calib_list, setdiff(par_tmp,calib_parameters))
        push!(par_no_var_calib_rhs_list, intersect(par_tmp,calib_parameters))
        
        # write down calibration equations
        prs_ex = postwalk(x -> 
            x isa Expr ? 
                x.head == :ref ?
                    occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                    x.args[1] : 
                x : 
                x.head == :call ?
                    x.args[1] == :* ?
                        x.args[2] isa Int ?
                            x.args[3] isa Int ?
                                x :
                            :($(x.args[3]) * $(x.args[2])) :
                        x :
                    x :
                unblock(x) : 
            x,
            cal_eq)
        push!(calib_equations_no_var_list,unblock(prs_ex))
    end
    
    # arrange calibration equations where they use parameters defined in parameters block so that they appear in right order (Pi_bar is defined before it is used later on: eta = Pi_bar / 2)
    if length(calib_equations_no_var_list) > 0
        incidence_matrix = fill(0,length(calib_parameters_no_var),length(calib_parameters_no_var))
        
        for i in 1:length(calib_parameters_no_var)
            for k in 1:length(calib_parameters_no_var)
                incidence_matrix[i,k] = collect(calib_parameters_no_var)[i] ∈ collect(par_no_var_calib_list)[k]
            end
        end
        
        Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(incidence_matrix))
        
        @assert length(Q) == n_blocks "Check the parameter definitions. They are either incomplete or have more than only the defined parameter on the LHS."
        
        calib_equations_no_var_list = calib_equations_no_var_list[Q]
    end
    


    #parse bounds
    bounds = Dict{Symbol,Tuple{Float64,Float64}}()

    for bound in bounded_vars
        postwalk(x -> 
        x isa Expr ?
            x.head == :comparison ? 
                x.args[2] == :(<) ?
                    x.args[4] == :(<) ?
                        begin
                            bounds[x.args[3]] = haskey(bounds, x.args[3]) ? (max(bounds[x.args[3]][1], x.args[1]+eps(Float32)), min(bounds[x.args[3]][2], x.args[5]-eps(Float32))) : (x.args[1]+eps(Float32), x.args[5]-eps(Float32))
                        end :
                    x.args[4] == :(<=) ?
                        begin
                            bounds[x.args[3]] = haskey(bounds, x.args[3]) ? (max(bounds[x.args[3]][1], x.args[1]+eps(Float32)), min(bounds[x.args[3]][2], x.args[5])) : (x.args[1]+eps(Float32), x.args[5])
                        end :
                    x :
                x.args[2] == :(<=) ?
                    x.args[4] == :(<) ?
                        begin
                            bounds[x.args[3]] = haskey(bounds, x.args[3]) ? (max(bounds[x.args[3]][1], x.args[1]), min(bounds[x.args[3]][2], x.args[5]-eps(Float32))) : (x.args[1], x.args[5]-eps(Float32))
                        end :
                    x.args[4] == :(<=) ?
                        begin
                            bounds[x.args[3]] = haskey(bounds, x.args[3]) ? (max(bounds[x.args[3]][1], x.args[1]), min(bounds[x.args[3]][2], x.args[5])) : (x.args[1], x.args[5])
                        end :
                    x :

                x.args[2] == :(>) ?
                    x.args[4] == :(>) ?
                        begin
                            bounds[x.args[3]] = haskey(bounds, x.args[3]) ? (max(bounds[x.args[3]][1], x.args[5]+eps(Float32)), min(bounds[x.args[3]][2], x.args[1]-eps(Float32))) : (x.args[5]+eps(Float32), x.args[1]-eps(Float32))
                        end :
                    x.args[4] == :(>=) ?
                        begin
                            bounds[x.args[3]] = haskey(bounds, x.args[3]) ? (max(bounds[x.args[3]][1], x.args[5]+eps(Float32)), min(bounds[x.args[3]][2], x.args[1])) : (x.args[5]+eps(Float32), x.args[1])
                        end :
                    x :
                x.args[2] == :(>=) ?
                    x.args[4] == :(>) ?
                        begin
                            bounds[x.args[3]] = haskey(bounds, x.args[3]) ? (max(bounds[x.args[3]][1], x.args[5]), min(bounds[x.args[3]][2], x.args[1]-eps(Float32))) : (x.args[5], x.args[1]-eps(Float32))
                        end :
                    x.args[4] == :(>=) ?
                        begin
                            bounds[x.args[3]] = haskey(bounds, x.args[3]) ? (max(bounds[x.args[3]][1], x.args[5]), min(bounds[x.args[3]][2], x.args[1])) : (x.args[5], x.args[1])
                        end :
                    x :
                x :

            x.head ==  :call ? 
                x.args[1] == :(<) ?
                    x.args[2] isa Symbol ? 
                        begin
                            bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], -1e12+rand()), min(bounds[x.args[2]][2], x.args[3]-eps(Float32))) : (-1e12+rand(), x.args[3]-eps(Float32))
                        end :
                    x.args[3] isa Symbol ? 
                        begin
                            bounds[x.args[3]] = haskey(bounds, x.args[3]) ? (max(bounds[x.args[3]][1], x.args[2]+eps(Float32)), min(bounds[x.args[3]][2], 1e12+rand())) : (x.args[2]+eps(Float32), 1e12+rand())
                        end :
                    x :
                x.args[1] == :(>) ?
                    x.args[2] isa Symbol ? 
                        begin
                            bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], x.args[3]+eps(Float32)), min(bounds[x.args[2]][2], 1e12+rand())) : (x.args[3]+eps(Float32), 1e12+rand())
                        end :
                    x.args[3] isa Symbol ? 
                        begin
                            bounds[x.args[3]] = haskey(bounds, x.args[3]) ? (max(bounds[x.args[3]][1], -1e12+rand()), min(bounds[x.args[3]][2], x.args[2]-eps(Float32))) : (-1e12+rand(), x.args[2]-eps(Float32))
                        end :
                    x :
                x.args[1] == :(>=) ?
                    x.args[2] isa Symbol ? 
                        begin
                            bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], x.args[3]), min(bounds[x.args[2]][2], 1e12+rand())) : (x.args[3], 1e12+rand())
                        end :
                    x.args[3] isa Symbol ? 
                        begin
                            bounds[x.args[3]] = haskey(bounds, x.args[3]) ? (max(bounds[x.args[3]][1], -1e12+rand()), min(bounds[x.args[3]][2], x.args[2])) : (-1e12+rand(), x.args[2])
                        end :
                    x :
                x.args[1] == :(<=) ?
                    x.args[2] isa Symbol ? 
                        begin
                            bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], -1e12+rand()), min(bounds[x.args[2]][2], x.args[3])) : (-1e12+rand(), x.args[3])
                        end :
                    x.args[3] isa Symbol ? 
                        begin
                            bounds[x.args[3]] = haskey(bounds, x.args[3]) ? (max(bounds[x.args[3]][1], x.args[2]), min(bounds[x.args[3]][2],1e12+rand())) : (x.args[2],1e12+rand())
                        end :
                    x :
                x :
            x :
        x,bound)
    end

    return quote
        mod = @__MODULE__

        if any(contains.(string.(mod.$𝓂.constants.post_model_macro.var), "ᵒᵇᶜ"))
            push!($calib_parameters, :activeᵒᵇᶜshocks)
            push!($calib_values, 0)
        end

        calib_parameters, calib_values = expand_indices($calib_parameters, $calib_values, [mod.$𝓂.constants.post_model_macro.parameters_in_equations; mod.$𝓂.constants.post_model_macro.var])
        calib_eq_parameters, calib_equations_list, ss_calib_list, par_calib_list = expand_calibration_equations($calib_eq_parameters, $calib_equations_list, $ss_calib_list, $par_calib_list, [mod.$𝓂.constants.post_model_macro.parameters_in_equations; mod.$𝓂.constants.post_model_macro.var])
        calib_parameters_no_var, calib_equations_no_var_list = expand_indices($calib_parameters_no_var, $calib_equations_no_var_list, [mod.$𝓂.constants.post_model_macro.parameters_in_equations; mod.$𝓂.constants.post_model_macro.var])
        
        # Calculate missing parameters instead of asserting
        # Include parameters from:
        # 1. par_calib_list - parameters used in calibration equations (e.g., K_ss in "K[ss] = K_ss | beta")
        # 2. parameters_in_equations - parameters used in model equations
        # 3. par_no_var_calib_list - parameters used in parameter definitions (e.g., rho{H}{H} in "rho{F}{F} = rho{H}{H}")
        # Subtract:
        # 1. calib_parameters - parameters with explicit values (e.g., "α = 0.5")
        # 2. calib_parameters_no_var - parameters defined as functions of other parameters (e.g., "α = alpha_param")
        # 3. calib_eq_parameters - parameters determined by calibration equations (e.g., "beta" in "K[ss] = K_ss | beta")
        # Start with directly required parameters
        all_required_params = union(
            reduce(union, par_calib_list, init = Set{Symbol}()),
            reduce(union, $par_no_var_calib_rhs_list, init = Set{Symbol}()),
            Set{Symbol}(mod.$𝓂.constants.post_model_macro.parameters_in_equations)
        )
        
        # Add parameters from parameter definitions, but only if the target parameter is needed
        # This handles the case where parameter X = f(Y, Z) but X is not used in the model.
        # In that case, Y and Z should not be required either.
        # We need to check if target is in all_required_params OR in calib_eq_parameters (parameters used in calibration equations)
        par_no_var_calib_filtered = mapreduce(i -> $par_no_var_calib_list[i], union, findall(target_param -> target_param ∈ all_required_params, calib_parameters_no_var), init = Set{Symbol}())
        
        all_required_params = union(all_required_params, par_no_var_calib_filtered)
        
        defined_params = union(
            Set{Symbol}(calib_parameters),
            Set{Symbol}(calib_parameters_no_var),
            Set{Symbol}(calib_eq_parameters)
        )

        ignored_params = collect(setdiff(defined_params, all_required_params))

        if !isempty(ignored_params) @warn "Parameters not part of the model are ignored: $ignored_params" end

        missing_params_unsorted = collect(setdiff(all_required_params, defined_params))
        missing_params = sort(missing_params_unsorted)
        
        has_missing_parameters = length(missing_params) > 0

        guess_dict = mod.$𝓂.constants.post_parameters_macro.guess
        if isa($guess, Dict{String, <:Real})
            guess_dict = Dict{Symbol, Float64}()
            for (key, value) in $guess
                if key isa String
                    key = replace_indices(key)
                end
                guess_dict[replace_indices(key)] = value
            end
        elseif isa($guess, Dict{Symbol, <:Real})
            guess_dict = $guess
        end

        bounds_dict = copy(mod.$𝓂.constants.post_parameters_macro.bounds)
        for (k,v) in $bounds
            bounds_dict[k] = haskey(bounds_dict, k) ? (max(bounds_dict[k][1], v[1]), min(bounds_dict[k][2], v[2])) : (v[1], v[2])
        end
        
        invalid_bounds = Symbol[]

        for (k,v) in bounds_dict
            if v[1] >= v[2]
                push!(invalid_bounds, k)
            end
        end

        @assert isempty(invalid_bounds) "Invalid bounds: " * repr(invalid_bounds)
        
        mod.$𝓂.constants.post_parameters_macro = post_parameters_macro(
            calib_parameters_no_var,
            $precompile,
            $simplify,
            guess_dict,
            ss_calib_list,
            par_calib_list,
            # $ss_no_var_calib_list,
            # $par_no_var_calib_list,
            bounds_dict,
        )

        # Update equations struct with calibration fields
        mod.$𝓂.equations.calibration = calib_equations_list
        mod.$𝓂.equations.calibration_no_var = calib_equations_no_var_list
        mod.$𝓂.equations.calibration_parameters = calib_eq_parameters
    
        # Keep calib_parameters in declaration order, append missing_params at end
        # This preserves declaration order for estimation and method of moments
        all_params = vcat(calib_parameters, missing_params)
        all_values = vcat(calib_values, fill(NaN, length(missing_params)))

        defined_params_idx = indexin(setdiff(intersect(all_params, defined_params), ignored_params), collect(all_params))

        mod.$𝓂.constants.post_complete_parameters = update_post_complete_parameters(
            mod.$𝓂.constants.post_complete_parameters;
            parameters = all_params[defined_params_idx],
            missing_parameters = missing_params,
        )
        mod.$𝓂.parameter_values = all_values[defined_params_idx]
        # mod.$𝓂.solution.outdated_NSSS = true
        
        # Store precompile and simplify flag in model container
        
        # Set custom steady state function if provided
        # if !isnothing($steady_state_function)
        set_custom_steady_state_function!(mod.$𝓂, $steady_state_function)
        # end

        mod.$𝓂.solution.functions_written = false

        # time_symbolics = @elapsed 
        # time_rm_red_SS_vars = @elapsed
        if !isnothing($steady_state_function)
            write_ss_check_function!(mod.$𝓂)
        else
            if !has_missing_parameters
                set_up_steady_state_solver!(mod.$𝓂, verbose = $verbose, silent = $silent, avoid_solve = !$simplify, symbolic = $symbolic)
            end
        end

        if !has_missing_parameters
            opts = merge_calculation_options(verbose = $verbose)
            
            SS_and_pars, solution_error, found_solution = solve_steady_state!(mod.$𝓂, opts, $(QuoteNode(ss_solver_parameters_algorithm)), $ss_solver_parameters_maxtime, silent = $silent)
            
            write_symbolic_derivatives!(mod.$𝓂; perturbation_order = $perturbation_order, silent = $silent)

            mod.$𝓂.solution.functions_written = true
        end

        if has_missing_parameters && $report_missing_parameters
            @warn "Model has been set up with incomplete parameter definitions. Missing parameters: $(missing_params). The non-stochastic steady state and perturbation solution cannot be computed until all parameters are defined. Provide missing parameter values via the `parameters` keyword argument in functions like `get_irf`, `get_SS`, `simulate`, etc."
        end

        if !$silent && $report_missing_parameters Base.show(mod.$𝓂) end
        nothing
    end
end
