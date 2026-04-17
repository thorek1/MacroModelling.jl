# Pure-function equation processing helpers used by both the equation
# modification pipeline and (potentially) the model macros.
#
# `process_model_equations` reproduces the work the `@model` macro performs on
# its equation block, returning a `post_model_macro` struct and an `equations`
# struct so the model state can be updated without re-invoking the macro.
#
# `process_parameter_definitions` reproduces the work the `@parameters` macro
# performs on the parameter block. It takes a `post_model_macro` describing the
# current model (used for variable name lookups, index expansion, etc.) and
# returns the pieces needed to update `post_parameters_macro`, the equations
# struct's calibration fields, and `post_complete_parameters`.

"""
    process_model_equations(model_block::Expr, max_obc_horizon::Int, precompile::Bool)

Parse a `@model`-style equation block and return `(T, equations_struct)` where
`T::post_model_macro` is the parsed model structure and `equations_struct::equations`
is a freshly constructed equations container with dynamic, steady-state and
original equations populated. Calibration fields on the returned equations
struct are left empty and must be populated by
`process_parameter_definitions` before the model can be solved.
"""
function process_model_equations(model_block_in::Expr, max_obc_horizon::Int, precompile::Bool)
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

    model_ex = parse_for_loops(model_block_in)
    
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
    I_nPast                    = ℒ.I(nPast_not_future_and_mixed)

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
                I_nPast,
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

    ℂ = Constants(T)
    𝓦 = Workspaces()

    ss_aux_eqs_vec = Expr[e for e in ss_aux_equations]
    dyn_eqs_vec = Expr[e for e in dyn_equations]
    ss_eqs_vec = Expr[e for e in ss_equations]
    orig_eqs_vec = Expr[e for e in original_equations]
    calib_eqs_vec = Expr[e for e in calibration_equations]

    equations_struct = equations(
        orig_eqs_vec,
        dyn_eqs_vec,
        ss_eqs_vec,
        ss_aux_eqs_vec,
        Expr[],            # obc_violation
        calib_eqs_vec,     # calibration (filled later by @parameters)
        Expr[],            # calibration_no_var
        Symbol[],          # calibration_parameters
        Expr[],            # calibration_original
    )

    return T, equations_struct, ℂ, 𝓦
end
