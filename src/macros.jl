
import MacroTools: unblock, postwalk, @capture

const all_available_algorithms = [:linear_time_iteration, :riccati, :first_order, :quadratic_iteration, :binder_pesaran, :second_order, :pruned_second_order, :third_order, :pruned_third_order]


"""
$(SIGNATURES)
Parses the model equations and assigns them to an object.

# Arguments
- `𝓂`: name of the object to be created containing the model information.
- `ex`: equations

# Optional arguments to be placed between `𝓂` and `ex`
- `max_obc_horizon` [Default: `40`, Type: `Int`]: maximum length of anticipated shocks and corresponding unconditional forecast horizon over which the occasionally binding constraint is to be enforced. Increase this number if no solution is found to enforce the constraint.

Variables must be defined with their time subscript in squared brackets.
Endogenous variables can have the following:
- present: `c[0]`
- non-stcohastic steady state: `c[ss]` instead of `ss` any of the following is also a valid flag for the non-stochastic steady state: `ss`, `stst`, `steady`, `steadystate`, `steady_state`, and the parser is case-insensitive (`SS` or `sTst` will work as well).
- past: `c[-1]` or any negative Integer: e.g. `c[-12]`
- future: `c[1]` or any positive Integer: e.g. `c[16]` or `c[+16]`
Signed integers are recognised and parsed as such.

Exogenous variables (shocks) can have the following:
- present: `eps_z[x]` instead of `x` any of the following is also a valid flag for exogenous variables: `ex`, `exo`, `exogenous`, and the parser is case-insensitive (`Ex` or `exoGenous` will work as well).
- past: `eps_z[x-1]`
- future: `eps_z[x+1]`

Parameters enter the equations without squared brackets.

If an equation contains a `max` or `min` operator, then the default dynamic (first order) solution of the model will enforce the occasionally binding constraint. You can choose to ignore it by setting `ignore_obc = true` in the relevant function calls.

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

`for` loops can be used to write models programmatically. They can either be used to generate expressions where you iterate over the time index or the index in curly braces:
- generate equation with different indices in curly braces: `for co in [H,F] C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1] end = for co in [H,F] Y{co}[0] end`
- generate multiple equations with different indices in curly braces: `for co in [H, F] K{co}[0] = (1-delta{co}) * K{co}[-1] + S{co}[0] end`
- generate equation with different time indices: `Y_annual[0] = for lag in -3:0 Y[lag] end` or `R_annual[0] = for operator = :*, lag in -3:0 R[lag] end`
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
                        @warn "Invalid options." 
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
    
    solved_vars = [] 
    solved_vals = []
    
    ss_solve_blocks = []
    
    NSSS_solver_cache = CircularBuffer{Vector{Vector{Float64}}}(500)
    SS_solve_func = x->x
    SS_dependencies = nothing

    original_equations = []
    calibration_equations = []
    calibration_equations_parameters = []

    bounds = Dict{Symbol,Tuple{Float64,Float64}}()

    dyn_equations = []

    ➕_vars = []
    ss_and_aux_equations = []
    aux_vars_created = Set()

    unique_➕_eqs = Dict{Union{Expr,Symbol},Expr}()

    ss_eq_aux_ind = Int[]
    dyn_eq_aux_ind = Int[]

    model_ex = parse_for_loops(ex[end])

    model_ex = parse_occasionally_binding_constraints(model_ex, max_obc_horizon = max_obc_horizon)
    
    # obc_shock_bounds = Tuple{Symbol, Bool, Float64}[]

    # write down dynamic equations and add auxilliary variables for leads and lags > 1
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
                
                                        while k > 2 # create auxilliary dynamic equation for exogenous variables with lead > 1
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
                    
                                        while k < -2 # create auxilliary dynamic equations for exogenous variables with lag < -1
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

                                        while k > 2 # create auxilliary dynamic equations for endogenous variables with lead > 1
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
                                x.args[2] < -1 ?  # create auxilliary dynamic equations for endogenous variables with lag < -1
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
            
            # write down ss equations including nonnegativity auxilliary variables
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

                                                push!(ss_eq_aux_ind,length(ss_and_aux_equations))

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

                                            push!(ss_eq_aux_ind,length(ss_and_aux_equations))

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

                                            push!(ss_eq_aux_ind,length(ss_and_aux_equations))

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
                                    bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], -1e12), min(bounds[x.args[2]][2], 700)) : (-1e12, 700)
                                    x
                                end :
                            x.args[2].head == :ref ?
                                x.args[2].args[1] isa Symbol ? # have exp terms bound so they dont go to Inf
                                    begin
                                        bounds[x.args[2].args[1]] = haskey(bounds, x.args[2].args[1]) ? (max(bounds[x.args[2].args[1]][1], -1e12), min(bounds[x.args[2].args[1]][2], 700)) : (-1e12, 700)
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
                                            ub = 700

                                            # push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(ub,max(lb,$(x.args[2])))))
                                            push!(ss_and_aux_equations, Expr(:call,:-, :($(Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)+1))),0))), x.args[2]))

                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], lb), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], ub)) : (lb, ub)

                                            push!(ss_eq_aux_ind,length(ss_and_aux_equations))

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

                                            push!(ss_eq_aux_ind,length(ss_and_aux_equations))

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

    # go through changed SS equations including nonnegative auxilliary variables
    ss_aux_equations = []

    # tag vars and pars in changed SS equations
    var_list_aux_SS = []
    ss_list_aux_SS = []
    par_list_aux_SS = []

    var_future_list_aux_SS = []
    var_present_list_aux_SS = []
    var_past_list_aux_SS = []

    # # label all variables parameters and exogenous variables and timings for changed SS equations including nonnegativity auxilliary variables
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


        # write down SS equations including nonnegativity auxilliary variables
        prs_ex = convert_to_ss_equation(eq)
        
        if idx ∈ ss_eq_aux_ind
            if precompile
                ss_aux_equation = Expr(:call,:-,unblock(prs_ex).args[2],unblock(prs_ex).args[3]) 
            else
                ss_aux_equation = Expr(:call,:-,unblock(prs_ex).args[2],simplify(unblock(prs_ex).args[3])) # simplify RHS if nonnegative auxilliary variable
            end
        else
            if precompile
                ss_aux_equation = unblock(prs_ex)
            else
                ss_aux_equation = simplify(unblock(prs_ex))
            end
        end
        ss_aux_equation_expr = if ss_aux_equation isa Symbol Expr(:call,:-,ss_aux_equation,0) else ss_aux_equation end

        push!(ss_aux_equations,ss_aux_equation_expr)
    end

    # go through dynamic equations and label
    # create timings
    dyn_var_future_list  = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₁₎" => "")),x)),collect.(match_pattern.(get_symbols.(dyn_equations),r"₍₁₎")))
    dyn_var_present_list = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₀₎" => "")),x)),collect.(match_pattern.(get_symbols.(dyn_equations),r"₍₀₎")))
    dyn_var_past_list    = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₋₁₎"=> "")),x)),collect.(match_pattern.(get_symbols.(dyn_equations),r"₍₋₁₎")))
    dyn_exo_list         = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),x)),collect.(match_pattern.(get_symbols.(dyn_equations),r"₍ₓ₎")))
    dyn_ss_list          = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₛₛ₎" => "")),x)),collect.(match_pattern.(get_symbols.(dyn_equations),r"₍ₛₛ₎")))

    all_symbols = reduce(union,collect.(get_symbols.(dyn_equations)))
    parameters_in_equations = sort(setdiff(all_symbols,match_pattern(all_symbols,r"₎$")))
    
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
    aux                       = aux_tmp[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∉ exo, aux_tmp)]
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


    T = timings(present_only,
                future_not_past,
                past_not_future,
                mixed,
                future_not_past_and_mixed,
                past_not_future_and_mixed,
                present_but_not_only,
                mixed_in_past,
                not_mixed_in_past,
                mixed_in_future,
                exo,
                var,
                aux,
                exo_present,

                nPresent_only,
                nMixed,
                nFuture_not_past_and_mixed,
                nPast_not_future_and_mixed,
                nPresent_but_not_only,
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
                dynamic_order)


    aux_future_tmp  = sort(filter(x->occursin(r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾",string(x)), dyn_var_future))
    aux_future      = aux_future_tmp[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∉ exo, aux_future_tmp)]

    aux_past_tmp    = sort(filter(x->occursin(r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾",string(x)), dyn_var_past))
    aux_past        = aux_past_tmp[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∉ exo, aux_past_tmp)]

    aux_present_tmp = sort(filter(x->occursin(r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾",string(x)), dyn_var_present))
    aux_present     = aux_present_tmp[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∉ exo, aux_present_tmp)]

    vars_in_ss_equations  = setdiff(reduce(union,get_symbols.(ss_aux_equations)),parameters_in_equations)


    dyn_future_list =   match_pattern.(get_symbols.(dyn_equations),r"₍₁₎")
    dyn_present_list =  match_pattern.(get_symbols.(dyn_equations),r"₍₀₎")
    dyn_past_list =     match_pattern.(get_symbols.(dyn_equations),r"₍₋₁₎")
    dyn_exo_list =      match_pattern.(get_symbols.(dyn_equations),r"₍ₓ₎")

    # println(ss_aux_equations)
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
                        sort(collect($exo)), 
                        sort(collect($parameters_in_equations)), 

                        $parameters,
                        $parameters,
                        $parameter_values,

                        sort($aux),
                        sort(collect($aux_present)), 
                        sort(collect($aux_future)), 
                        sort(collect($aux_past)), 

                        sort(collect($exo_future)), 
                        sort(collect($exo_present)), 
                        sort(collect($exo_past)), 

                        sort(collect($vars_in_ss_equations)),
                        sort($var), 
                        
                        $ss_calib_list,
                        $par_calib_list,

                        $ss_calib_list, #no_var_
                        $par_calib_list, #no_var_

                        $ss_aux_equations,
                        $var_list_aux_SS,
                        $ss_list_aux_SS,
                        $par_list_aux_SS,
                        $var_future_list_aux_SS,
                        $var_present_list_aux_SS,
                        $var_past_list_aux_SS,

                        $dyn_var_future_list,
                        $dyn_var_present_list,
                        $dyn_var_past_list, 
                        $dyn_ss_list,
                        $dyn_exo_list,

                        $dyn_future_list,
                        $dyn_present_list,
                        $dyn_past_list, 

                        $solved_vars, 
                        $solved_vals, 

                        $ss_solve_blocks,
                        $NSSS_solver_cache,
                        $SS_solve_func,
                        $SS_dependencies,

                        $➕_vars,
                        $ss_eq_aux_ind,
                        $dyn_equations,
                        $original_equations, 

                        $calibration_equations, #no_var_

                        $calibration_equations, 
                        $calibration_equations_parameters,

                        $bounds,

                        x->x,
                        # FWrap{Tuple{Vector{Float64}, Vector{Number}, Vector{Float64}}, SparseMatrixCSC{Float64}}(model_jacobian),
                        [],#x->x,
                        [],#x->x,

                        $T,

                        Expr[],
                        # $obc_shock_bounds,
                        $max_obc_horizon,
                        x->x,

                        [
                            # solver_parameters(eps(), eps(), 250, 
                            # 2.9912988764832833, 0.8725, 0.0027, 0.028948770826150612, 8.04, 4.076413176215408, 0.06375413238034794, 0.24284340766769424, 0.5634017580097571, 0.009549630552246828, 0.6342888355132347, 0.5275522227754195, 1.0, 0.06178989216048817, 0.5234277812131813, 0.422, 0.011209254402846185, 0.5047, 0.6020757011698457, 0.897,
                            # 1, 0.0, 2),

                            solver_parameters(eps(), eps(), 250, 
                            1.0242323883590136, 0.5892723157762478, 0.0006988523559835617, 0.009036867721330505, 0.14457591298892497, 1.3282546133453548, 0.7955753778741823, 1.7661485851863441e-6, 2.6206711939142943e-7, 7.052160321659248e-12, 1.06497513443326e-6, 5.118937128189348, 90.94952163302091, 3.1268025435012207e-13, 1.691251847378593, 0.5455751102495228, 0.1201767636895742, 0.0007802908980930664, 0.011310267585075185, 1.0032972640942657, 
                            1, 0.0, 2),

                            solver_parameters(eps(), eps(), 250, 
                            1.2472903868878749, 0.7149401846020106, 0.0034717544971213966, 0.0008409477479813854, 0.24599133854242075, 1.7996260724902138, 0.2399133704286251, 0.728108158144521, 0.03250298738504968, 0.003271716521926188, 0.5319194600339338, 2.1541622462034, 7.751722474870615, 0.08193253023289011, 1.52607969046303, 0.0002086811131899754, 0.005611466658864538, 0.018304952326087726, 0.0024888171138406773, 0.9061879299736817, 
                            1, 0.0, 2),

                            solver_parameters(eps(), eps(), 250, 
                            1.9479518608134938, 0.02343520604394183, 5.125002799990568, 0.02387522857907376, 0.2239226474715968, 4.889172213411495, 1.747880258818237, 2.8683242331457, 0.938229356687311, 1.4890887655876235, 1.6261504814901664, 11.26863249187599, 36.05486169712279, 6.091535897587629, 11.73936761697657, 3.189349432626493, 0.21045178305336348, 0.17122196312330415, 13.251662547139363, 5.282429995876679, 
                            1, 0.0, 2),

                            solver_parameters(eps(), eps(), 250, 
                            2.9912988764832833, 0.8725, 0.0027, 0.028948770826150612, 8.04, 4.076413176215408, 0.06375413238034794, 0.24284340766769424, 0.5634017580097571, 0.009549630552246828, 0.6342888355132347, 0.5275522227754195, 1.0, 0.06178989216048817, 0.5234277812131813, 0.422, 0.011209254402846185, 0.5047, 0.6020757011698457, 0.7688, 
                            1, 0.0, 2)

                        ],

                        solution(
                            perturbation(   perturbation_solution(SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), (x,y)->nothing, nothing),
                                            perturbation_solution(SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), (x,y)->nothing, nothing),
                                            perturbation_solution(SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), (x,y)->nothing, nothing),
                                            second_order_perturbation_solution(SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), [], (x,y)->nothing, nothing),
                                            second_order_perturbation_solution(SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), [], (x,y)->nothing, nothing),
                                            third_order_perturbation_solution(SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), [], (x,y)->nothing, nothing),
                                            third_order_perturbation_solution(SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), [], (x,y)->nothing, nothing),
                                            auxilliary_indices(Int[],Int[],Int[],Int[],Int[]),
                                            second_order_auxilliary_matrices(SparseMatrixCSC{Int, Int64}(ℒ.I,0,0),SparseMatrixCSC{Int, Int64}(ℒ.I,0,0),SparseMatrixCSC{Int, Int64}(ℒ.I,0,0),SparseMatrixCSC{Int, Int64}(ℒ.I,0,0)),
                                            third_order_auxilliary_matrices(SparseMatrixCSC{Int, Int64}(ℒ.I,0,0),SparseMatrixCSC{Int, Int64}(ℒ.I,0,0),SparseMatrixCSC{Int, Int64}(ℒ.I,0,0),SparseMatrixCSC{Int, Int64}(ℒ.I,0,0),SparseMatrixCSC{Int, Int64}(ℒ.I,0,0),SparseMatrixCSC{Int, Int64}(ℒ.I,0,0),SparseMatrixCSC{Int, Int64}(ℒ.I,0,0),SparseMatrixCSC{Int, Int64}(ℒ.I,0,0),SparseMatrixCSC{Int, Int64}(ℒ.I,0,0),SparseMatrixCSC{Int, Int64}(ℒ.I,0,0),SparseMatrixCSC{Int, Int64}(ℒ.I,0,0),SparseMatrixCSC{Int, Int64}(ℒ.I,0,0),SparseMatrixCSC{Int, Int64}(ℒ.I,0,0))
                            ),
                            Float64[], 
                            Set([:first_order]),
                            Set(all_available_algorithms),
                            true,
                            false
                        )
                    );
    end
end






"""
$(SIGNATURES)
Adds parameter values and calibration equations to the previously defined model.

# Arguments
- `𝓂`: name of the object previously created containing the model information.
- `ex`: parameter, parameters values, and calibration equations

Parameters can be defined in either of the following ways:
- plain number: `δ = 0.02`
- expression containing numbers: `δ = 1/50`
- expression containing other parameters: `δ = 2 * std_z` in this case it is irrelevant if `std_z` is defined before or after. The definitons including other parameters are treated as a system of equaitons and solved accordingly.
- expressions containing a target parameter and an equations with endogenous variables in the non-stochastic steady state, and other parameters, or numbers: `k[ss] / (4 * q[ss]) = 1.5 | δ` or `α | 4 * q[ss] = δ * k[ss]` in this case the target parameter will be solved simultaneaously with the non-stochastic steady state using the equation defined with it.

# Optional arguments to be placed between `𝓂` and `ex`
- `verbose` [Default: `false`, Type: `Bool`]: print more information about how the non stochastic steady state is solved
- `silent` [Default: `false`, Type: `Bool`]: do not print any information
- `symbolic` [Default: `false`, Type: `Bool`]: try to solve the non stochastic steady state symbolically and fall back to a numerical solution if not possible
- `perturbation_order` [Default: `1`, Type: `Int`]: take derivatives only up to the specified order at this stage. In case you want to work with higher order perturbation later on, respective derivatives will be taken at that stage.



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
```

# Programmatic model writing

Variables and parameters indexed with curly braces can be either referenced specifically (e.g. `c{H}[ss]`) or generally (e.g. `alpha`). If they are referenced generaly the parse assumes all instances (indices) are meant. For example, in a model where `alpha` has two indices `H` and `F`, the expression `alpha = 0.3` is interpreted as two expressions: `alpha{H} = 0.3` and `alpha{F} = 0.3`. The same goes for calibration equations.
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
    
    calib_parameters = Symbol[]
    calib_values = Float64[]

    par_defined_more_than_once = Set()
    
    bounded_vars = []

    # parse options
    verbose = false
    silent = false
    symbolic = false
    precompile = false
    perturbation_order = 1

    for exp in ex[1:end-1]
        postwalk(x -> 
            x isa Expr ?
                x.head == :(=) ?  
                    x.args[1] == :symbolic && x.args[2] isa Bool ?
                        symbolic = x.args[2] :
                    x.args[1] == :verbose && x.args[2] isa Bool ?
                        verbose = x.args[2] :
                    x.args[1] == :silent && x.args[2] isa Bool ?
                        silent = x.args[2] :
                    x.args[1] == :precompile && x.args[2] isa Bool ?
                        precompile = x.args[2] :
                    x.args[1] == :perturbation_order && x.args[2] isa Int ?
                        perturbation_order = x.args[2] :
                    begin
                        @warn "Invalid options." 
                        x
                    end :
                x :
            x,
        exp)
    end

    parameter_definitions = replace_indices(ex[end])

    # parse parameter inputs
    # label all variables parameters and exogenous vairables and timings across all equations
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
                x.args[1] isa Symbol ?# || x.args[1] isa Expr ? #this doesnt work really well yet
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
                        push!(calib_eq_parameters,x.args[2].args[end])
                        push!(calib_equations,Expr(:(=),x.args[1], unblock(x.args[2].args[2])))
                    end :
                x :
            x :
        x,
    parameter_definitions)
    
    @assert length(par_defined_more_than_once) == 0 "Parameters can only be defined once. This is not the case for: " * repr([par_defined_more_than_once...])
    
    # evaluate inputs where they are of the type: log(1/3) (no variables but need evaluation to becoe a Float64)
    for (i, v) in enumerate(calib_values_no_var)
        out = try eval(v) catch e end
        if out isa Float64
            push!(calib_parameters, calib_parameters_no_var[i])
            push!(calib_values, out)
        else
            push!(calib_equations_no_var, Expr(:(=),calib_parameters_no_var[i], calib_values_no_var[i]))
        end
    end
    
    calib_parameters_no_var = setdiff(calib_parameters_no_var,calib_parameters)
    
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
                        diffed = setdiff([x],ss_tmp)
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
                            :($(x.args[3]) * $(x.args[2])) : # 2Π => Π*2 (the former doesnt work with sympy)
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
        push!(par_no_var_calib_list,setdiff(par_tmp,calib_parameters))
        
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

    # println($m)
    return quote
        mod = @__MODULE__

        if any(contains.(string.(mod.$𝓂.var), "ᵒᵇᶜ"))
            push!($calib_parameters, :activeᵒᵇᶜshocks)
            push!($calib_values, 0)
        end

        calib_parameters, calib_values = expand_indices($calib_parameters, $calib_values, [mod.$𝓂.parameters_in_equations; mod.$𝓂.var])
        calib_eq_parameters, calib_equations_list, ss_calib_list, par_calib_list = expand_calibration_equations($calib_eq_parameters, $calib_equations_list, $ss_calib_list, $par_calib_list, [mod.$𝓂.parameters_in_equations; mod.$𝓂.var])
        calib_parameters_no_var, calib_equations_no_var_list = expand_indices($calib_parameters_no_var, $calib_equations_no_var_list, [mod.$𝓂.parameters_in_equations; mod.$𝓂.var])

        @assert length(setdiff(setdiff(setdiff(union(reduce(union, par_calib_list,init = []),mod.$𝓂.parameters_in_equations),calib_parameters),calib_parameters_no_var),calib_eq_parameters)) == 0 "Undefined parameters: " * repr([setdiff(setdiff(setdiff(union(reduce(union,par_calib_list,init = []),mod.$𝓂.parameters_in_equations),calib_parameters),calib_parameters_no_var),calib_eq_parameters)...])

        for (k,v) in $bounds
            mod.$𝓂.bounds[k] = haskey(mod.$𝓂.bounds, k) ? (max(mod.$𝓂.bounds[k][1], v[1]), min(mod.$𝓂.bounds[k][2], v[2])) : (v[1], v[2])
        end
             
        invalid_bounds = Symbol[]

        for (k,v) in mod.$𝓂.bounds
            if v[1] >= v[2]
                push!(invalid_bounds, k)
            end
        end

        @assert isempty(invalid_bounds) "Invalid bounds: " * repr(invalid_bounds)
        
        mod.$𝓂.ss_calib_list = ss_calib_list
        mod.$𝓂.par_calib_list = par_calib_list
    
        mod.$𝓂.ss_no_var_calib_list = $ss_no_var_calib_list
        mod.$𝓂.par_no_var_calib_list = $par_no_var_calib_list
    
        mod.$𝓂.parameters = calib_parameters
        mod.$𝓂.parameter_values = calib_values
        mod.$𝓂.calibration_equations = calib_equations_list
        mod.$𝓂.parameters_as_function_of_parameters = calib_parameters_no_var
        mod.$𝓂.calibration_equations_no_var = calib_equations_no_var_list
        mod.$𝓂.calibration_equations_parameters = calib_eq_parameters
        # mod.$𝓂.solution.outdated_NSSS = true

        # time_symbolics = @elapsed 
        # time_rm_red_SS_vars = @elapsed 
        if !$precompile 
            start_time = time()

            if !$silent print("Remove redundant variables in non stochastic steady state problem:\t") end

            symbolics = create_symbols_eqs!(mod.$𝓂)
            remove_redundant_SS_vars!(mod.$𝓂, symbolics) 

            if !$silent println(round(time() - start_time, digits = 3), " seconds") end


            start_time = time()
    
            if !$silent print("Set up non stochastic steady state problem:\t") end

            solve_steady_state!(mod.$𝓂, $symbolic, symbolics, verbose = $verbose) # 2nd argument is SS_symbolic

            mod.$𝓂.obc_violation_equations = write_obc_violation_equations(mod.$𝓂)
            
            set_up_obc_violation_function!(mod.$𝓂)

            if !$silent println(round(time() - start_time, digits = 3), " seconds") end
        else
            start_time = time()
        
            if !$silent print("Set up non stochastic steady state problem:\t") end

            solve_steady_state!(mod.$𝓂, verbose = $verbose)

            if !$silent println(round(time() - start_time, digits = 3), " seconds") end
        end

        start_time = time()

        if !$silent
            if $perturbation_order == 1
                print("Take symbolic derivatives up to first order:\t")
            elseif $perturbation_order == 2
                print("Take symbolic derivatives up to second order:\t")
            elseif $perturbation_order == 3
                print("Take symbolic derivatives up to third order:\t")
            end
        end

        # time_dynamic_derivs = @elapsed 
        write_functions_mapping!(mod.$𝓂, $perturbation_order)

        mod.$𝓂.solution.outdated_algorithms = Set(all_available_algorithms)
        
        if !$silent
            println(round(time() - start_time, digits = 3), " seconds")
        end

        start_time = time()

        mod.$𝓂.solution.functions_written = true

        if !$precompile
            if !$silent 
                print("Find non stochastic steady state:\t") 
            end
            # time_SS_real_solve = @elapsed 
            SS_and_pars, (solution_error, iters) = mod.$𝓂.SS_solve_func(mod.$𝓂.parameter_values, mod.$𝓂, $verbose, true, mod.$𝓂.solver_parameters)
            
            select_fastest_SS_solver_parameters!(mod.$𝓂)

            found_solution = true

            if solution_error > 1e-12
                # start_time = time()
                found_solution = find_SS_solver_parameters!(mod.$𝓂)
                # println("Find SS solver parameters which solve for the NSSS:\t",round(time() - start_time, digits = 3), " seconds")
            end
            
            if !found_solution
                @warn "Could not find non-stochastic steady state. Consider setting bounds on variables or calibrated parameters in the `@parameters` section (e.g. `k > 10`)."
            end

            if !$silent 
                println(round(time() - start_time, digits = 3), " seconds") 
            end

            mod.$𝓂.solution.non_stochastic_steady_state = SS_and_pars
            mod.$𝓂.solution.outdated_NSSS = false
        end

        if !$silent Base.show(mod.$𝓂) end
        nothing
    end
end