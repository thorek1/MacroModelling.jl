function preprocess_model_block(model_block::Expr, max_obc_horizon::Int)
    model_ex = parse_for_loops(model_block)
    model_ex = resolve_if_expr(model_ex::Expr)::Expr
    model_ex = remove_nothing(model_ex::Expr)::Expr
    return parse_occasionally_binding_constraints(model_ex::Expr, max_obc_horizon = max_obc_horizon)::Expr
end

function build_dynamic_and_ss_equations!(
    model_ex::Expr,
    dyn_equations,
    ss_equations,
    aux_vars_created,
    dyn_eq_aux_ind,
)
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
        end
    end
end

function create_nonnegativity_auxiliaries!(
    model_ex::Expr,
    ss_and_aux_equations,
    ➕_vars,
    bounds,
    ss_equations_with_aux_variables,
    unique_➕_eqs,
    precompile,
)
    # write down ss equations including nonnegativity auxiliary variables
    # find nonegative variables, parameters, or terms
    for (i,arg) in enumerate(model_ex.args)
        if isa(arg,Expr)
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
end

function process_auxiliary_ss_equations!(
    ss_and_aux_equations,
    ss_aux_equations,
    var_list_aux_SS,
    ss_list_aux_SS,
    par_list_aux_SS,
    var_future_list_aux_SS,
    var_present_list_aux_SS,
    var_past_list_aux_SS,
    ss_equations_with_aux_variables,
    precompile,
)
    # go through changed SS equations including nonnegativity auxiliary variables
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
end

function extract_dyn_symbol_lists(dyn_equations)
    dyn_var_future_list  = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₁₎" => "")),x)),collect.(match_pattern.(get_symbols.(dyn_equations),r"₍₁₎")))
    dyn_var_present_list = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₀₎" => "")),x)),collect.(match_pattern.(get_symbols.(dyn_equations),r"₍₀₎")))
    dyn_var_past_list    = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₋₁₎"=> "")),x)),collect.(match_pattern.(get_symbols.(dyn_equations),r"₍₋₁₎")))
    dyn_exo_list         = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),x)),collect.(match_pattern.(get_symbols.(dyn_equations),r"₍ₓ₎")))
    dyn_ss_list          = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₛₛ₎" => "")),x)),collect.(match_pattern.(get_symbols.(dyn_equations),r"₍ₛₛ₎")))
    return dyn_var_future_list, dyn_var_present_list, dyn_var_past_list, dyn_exo_list, dyn_ss_list
end

function process_model_equations(
    model_block::Expr,
    max_obc_horizon::Int,
    precompile::Bool,
)
    model_ex = preprocess_model_block(model_block, max_obc_horizon)

    bounds = Dict{Symbol,Tuple{Float64,Float64}}()
    dyn_equations = []
    ➕_vars = []
    ss_and_aux_equations = []
    ss_equations = []
    ss_aux_equations = Expr[]
    aux_vars_created = Set()
    unique_➕_eqs = Dict{Union{Expr,Symbol},Expr}()
    ss_equations_with_aux_variables = Int[]
    dyn_eq_aux_ind = Int[]

    build_dynamic_and_ss_equations!(model_ex, dyn_equations, ss_equations, aux_vars_created, dyn_eq_aux_ind)
    create_nonnegativity_auxiliaries!(model_ex, ss_and_aux_equations, ➕_vars, bounds, ss_equations_with_aux_variables, unique_➕_eqs, precompile)

    # go through changed SS equations including nonnegativity auxiliary variables
    var_list_aux_SS = []
    ss_list_aux_SS = []
    par_list_aux_SS = []

    var_future_list_aux_SS = []
    var_present_list_aux_SS = []
    var_past_list_aux_SS = []

    process_auxiliary_ss_equations!(
        ss_and_aux_equations,
        ss_aux_equations,
        var_list_aux_SS,
        ss_list_aux_SS,
        par_list_aux_SS,
        var_future_list_aux_SS,
        var_present_list_aux_SS,
        var_past_list_aux_SS,
        ss_equations_with_aux_variables,
        precompile,
    )

    # go through dynamic equations and label
    # create timings
    dyn_var_future_list, dyn_var_present_list, dyn_var_past_list, dyn_exo_list, dyn_ss_list = extract_dyn_symbol_lists(dyn_equations)

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

    original_equations = []
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

    equations_struct = equations(
        original_equations,
        dyn_equations,
        ss_equations,
        ss_aux_equations,
        Expr[],
        Expr[],
        Expr[],
        Symbol[],
    )

    return T, equations_struct
end

function parse_parameter_definitions_raw(parameter_block::Expr)
    calib_equations = Expr[]
    calib_equations_no_var = Expr[]
    calib_values_no_var = Any[]
    calib_parameters_no_var = Symbol[]
    calib_eq_parameters = Symbol[]
    calib_equations_list = Expr[]

    ss_calib_list = []
    par_calib_list = []

    calib_equations_no_var_list = Expr[]

    ss_no_var_calib_list = []
    par_no_var_calib_list = []
    par_no_var_calib_rhs_list = []

    calib_parameters = Symbol[]
    calib_values = Float64[]

    par_defined_more_than_once = Set{Symbol}()

    bounded_vars = []

    parameter_definitions = replace_indices(parameter_block)

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

    return (
        calib_parameters = calib_parameters,
        calib_values = calib_values,
        calib_eq_parameters = calib_eq_parameters,
        calib_equations_list = calib_equations_list,
        ss_calib_list = ss_calib_list,
        par_calib_list = par_calib_list,
        calib_parameters_no_var = calib_parameters_no_var,
        calib_equations_no_var_list = calib_equations_no_var_list,
        par_no_var_calib_list = par_no_var_calib_list,
        par_no_var_calib_rhs_list = par_no_var_calib_rhs_list,
        bounds = bounds,
    )
end

function build_guess_dict(guess)
    guess_dict = Dict{Symbol, Float64}()
    if isa(guess, Dict{String, <:Real})
        for (key, value) in guess
            if key isa String
                key = replace_indices(key)
            end
            guess_dict[replace_indices(key)] = value
        end
    elseif isa(guess, Dict{Symbol, <:Real})
        guess_dict = guess
    end
    return guess_dict
end

function process_parameter_definitions(
    parameter_block::Expr,
    post_model_macro::post_model_macro
)
    parsed = parse_parameter_definitions_raw(parameter_block)

    calib_parameters = parsed.calib_parameters
    calib_values = parsed.calib_values
    calib_eq_parameters = parsed.calib_eq_parameters
    calib_equations_list = parsed.calib_equations_list
    ss_calib_list = parsed.ss_calib_list
    par_calib_list = parsed.par_calib_list
    calib_parameters_no_var = parsed.calib_parameters_no_var
    calib_equations_no_var_list = parsed.calib_equations_no_var_list
    par_no_var_calib_list = parsed.par_no_var_calib_list
    par_no_var_calib_rhs_list = parsed.par_no_var_calib_rhs_list

    if any(contains.(string.(post_model_macro.var), "ᵒᵇᶜ"))
        push!(calib_parameters, :activeᵒᵇᶜshocks)
        push!(calib_values, 0)
    end

    calib_parameters, calib_values = expand_indices(calib_parameters, calib_values, [post_model_macro.parameters_in_equations; post_model_macro.var])
    calib_eq_parameters, calib_equations_list, ss_calib_list, par_calib_list = expand_calibration_equations(calib_eq_parameters, calib_equations_list, ss_calib_list, par_calib_list, [post_model_macro.parameters_in_equations; post_model_macro.var])
    calib_parameters_no_var, calib_equations_no_var_list = expand_indices(calib_parameters_no_var, calib_equations_no_var_list, [post_model_macro.parameters_in_equations; post_model_macro.var])

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
        reduce(union, par_no_var_calib_rhs_list, init = Set{Symbol}()),
        Set{Symbol}(post_model_macro.parameters_in_equations)
    )

    # Add parameters from parameter definitions, but only if the target parameter is needed
    # This handles the case where parameter X = f(Y, Z) but X is not used in the model.
    # In that case, Y and Z should not be required either.
    # We need to check if target is in all_required_params OR in calib_eq_parameters (parameters used in calibration equations)
    par_no_var_calib_filtered = mapreduce(i -> par_no_var_calib_list[i], union, findall(target_param -> target_param ∈ all_required_params, calib_parameters_no_var), init = Set{Symbol}())

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

    bounds_dict = Dict{Symbol,Tuple{Float64,Float64}}()
    for (k,v) in parsed.bounds
        bounds_dict[k] = haskey(bounds_dict, k) ? (max(bounds_dict[k][1], v[1]), min(bounds_dict[k][2], v[2])) : (v[1], v[2])
    end

    invalid_bounds = Symbol[]

    for (k,v) in bounds_dict
        if v[1] >= v[2]
            push!(invalid_bounds, k)
        end
    end

    @assert isempty(invalid_bounds) "Invalid bounds: " * repr(invalid_bounds)

    # Keep calib_parameters in declaration order, append missing_params at end
    # This preserves declaration order for estimation and method of moments
    all_params = vcat(calib_parameters, missing_params)
    all_values = vcat(calib_values, fill(NaN, length(missing_params)))

    defined_params_idx = indexin(setdiff(intersect(all_params, defined_params), ignored_params), collect(all_params))

    return (
        equations = (
            calibration = calib_equations_list,
            calibration_no_var = calib_equations_no_var_list,
            calibration_parameters = calib_eq_parameters,
        ),
        calib_parameters_no_var = calib_parameters_no_var,
        ss_calib_list = ss_calib_list,
        par_calib_list = par_calib_list,
        bounds = bounds_dict,
        parameters = all_params[defined_params_idx],
        missing_parameters = missing_params,
        parameter_values = all_values[defined_params_idx],
    )
end
