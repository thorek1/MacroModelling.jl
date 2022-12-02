
import MacroTools: postwalk, unblock


"""
$(SIGNATURES)
Parses the model equations and assigns them to an object.

# Arguments
- `𝓂`: name of the object to be created containing the model information.
- `ex`: equations

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
"""
macro model(𝓂,ex)
    # create data containers
    exo = Set()
    aux = Set()
    var = Set()
    par = Set()
    ss = Set()

    dyn_ss_past = Set()
    dyn_ss_future = Set()
    dyn_ss_present = Set()

    var_future = Set()
    var_present = Set()
    var_past = Set()

    aux_future = Set()
    aux_present = Set()
    aux_past = Set()

    exo_future = Set()
    exo_present = Set()
    exo_past = Set()

    parameters = []
    parameter_values = Vector{Float64}(undef,0)

    exo_list = []
    ss_list = []

    ss_calib_list = []
    par_calib_list = []
    var_list = []
    dynamic_variables_list = []
    dynamic_variables_future_list = []
    # var_redundant_list = nothing
    # var_redundant_calib_list = nothing
    # var_solved_list = nothing
    # var_solved_calib_list = nothing
    # var_remaining_list = []
    par_list = []
    var_future_list = []
    var_present_list = []
    var_past_list = []


    dyn_shift_var_future_list = []
    dyn_shift_var_present_list = []
    dyn_shift_var_past_list = []

    dyn_shift2_var_past_list = []

    dyn_var_future_list = []
    dyn_var_present_list = []
    dyn_var_past_list = []
    dyn_exo_list = []
    dyn_ss_list = []

    solved_vars = [] 
    solved_vals = []

    non_linear_solved_vars = []
    non_linear_solved_vals = []

    # solved_sub_vals = []
    # solved_sub_values = []
    ss_solve_blocks = []
    ss_solve_blocks_optim = []
    SS_init_guess = Vector{Float64}(undef,0)
    SS_solve_func = nothing
    nonlinear_solution_helper = nothing
    SS_dependencies = nothing

    ss_equations = []
    equations = []
    calibration_equations = []
    calibration_equations_parameters = []

    bounds = []

    bounded_vars = []
    lower_bounds = []
    upper_bounds = []

    t_future_equations = []
    t_past_equations = []
    t_present_equations = []
    dyn_equations = []
    dyn_equations_future = []

    # label all variables parameters and exogenous vairables and timings across all equations
    postwalk(x -> 
            x isa Expr ? 
                x.head == :ref ? 
                    x.args[2] isa Int ? 
                        x.args[2] == 0 ? 
                            push!(var_present,x.args[1]) : 
                        x.args[2] > 1 ? 
                            begin
                                time_idx = x.args[2]

                                while time_idx > 1
                                    push!(aux_future,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(time_idx - 1)) * "⁾"))
                                    push!(aux_present,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(time_idx - 1)) * "⁾"))
                                    
                                    time_idx -= 1
                                end

                                push!(var_future,x.args[1])
                            end : 
                        1 >= x.args[2] > 0 ? 
                            push!(var_future,x.args[1]) : 
                        -1 <= x.args[2] < 0 ? 
                            push!(var_past,x.args[1]) : 
                        x.args[2] < -1 ? 
                            begin
                                time_idx = x.args[2]

                                while time_idx < -1
                                    push!(aux_past,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(time_idx + 1)) * "⁾"))
                                    push!(aux_present,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(time_idx + 1)) * "⁾"))

                                    time_idx += 1
                                end

                                push!(var_past,x.args[1])
                            end : 
                        x :
                    # issubset([x.args[2]],[:x :ex :exo :exogenous]) ?
                    occursin(r"^(x|ex|exo|exogenous){1}$"i,string(x.args[2])) ?
                        push!(exo,x.args[1]) :
                    # issubset([x.args[2]],[:ss :SS :ℳ :StSt :steady :steadystate :steady_state :Steady_State]) ?
                    occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                        push!(ss,x.args[1]) :
                    x : 
                x.head == :call ? 
                    for i in 2:length(x.args)
                        x.args[i] isa Symbol ? 
                            occursin(r"^(ss|stst|steady|steadystate|steady_state|x|ex|exo|exogenous){1}$"i,string(x.args[i])) ? 
                                x :
                            push!(par,x.args[i]) : 
                        x
                    end : 
                x :
            x,
    ex)



    #throw errors if variables are not matched

    # var = collect(union(var_future,var_present,var_past))

    # aux = collect(union(aux_future,aux_present,aux_past))
    nonnegativity_aux_vars = []
    ss_equations_aux = []
    aux_vars_created = Set()

    # write down SS equations and go by equation
    for (i,arg) in enumerate(ex.args)
        if isa(arg,Expr)

            # label all variables parameters and exogenous vairables and timings for individual equations
            exo_tmp = Set()
            var_tmp = Set()
            par_tmp = Set()
            ss_tmp = Set()
            var_future_tmp = Set()
            var_present_tmp = Set()
            var_past_tmp = Set()
            var_dyn_tmp = Set()

            
            var_shift_dyn_future_tmp = Set()
            var_shift_dyn_present_tmp = Set()
            var_shift_dyn_past_tmp = Set()

            var_dyn_future_tmp = Set()
            var_dyn_present_tmp = Set()
            var_dyn_past_tmp = Set()
            var_shift_dyn_future_tmp = Set()
            var_shift_dyn_present_tmp = Set()
            var_shift_dyn_past_tmp = Set()

            var_shift2_dyn_past_tmp = Set()

            ss_dyn_tmp = Set()
            exo_dyn_tmp = Set()

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
                                begin
                                    push!(var_present_tmp,x.args[1])
                                    push!(var_dyn_present_tmp,Symbol(string(x.args[1]) * "₍₀₎"))
                                    push!(var_shift_dyn_present_tmp,Symbol(string(x.args[1]) * "₍₁₎"))
                                    # push!(var_dyn_tmp,Symbol(string(x.args[1]) * "₍₀₎"))
                                end : 
                            x.args[2] > 0 ? 
                                begin
                                    push!(var_future_tmp,x.args[1])
                                    push!(var_dyn_future_tmp,Symbol(string(x.args[1]) * "₍₁₎"))
                                    push!(var_shift_dyn_future_tmp,Symbol(string(x.args[1]) * "₍₂₎"))
                                    # push!(var_dyn_tmp,Symbol(string(x.args[1]) * "₍₁₎"))
                                end : 
                            x.args[2] < 0 ? 
                                begin
                                    push!(var_past_tmp,x.args[1])
                                    push!(var_dyn_past_tmp,Symbol(string(x.args[1]) * "₍₋₁₎"))
                                    push!(var_shift_dyn_past_tmp,Symbol(string(x.args[1]) * "₍₀₎"))
                                    push!(var_shift2_dyn_past_tmp,Symbol(string(x.args[1]) * "₍₁₎"))
                                    # push!(var_dyn_tmp,Symbol(string(x.args[1]) * "₍₋₁₎"))
                                end : 
                            x :
                        # issubset([x.args[2]],[:x :ex :exo :exogenous]) ?
                        occursin(r"^(x|ex|exo|exogenous){1}$"i,string(x.args[2])) ?
                            begin
                                push!(exo_tmp,x.args[1])
                                push!(exo_dyn_tmp,Symbol(string(x.args[1]) * "₍ₓ₎"))
                                # push!(var_dyn_tmp,Symbol(string(x.args[1]) * "₍ₓ₎"))
                            end : 
                        occursin(r"^(x|ex|exo|exogenous){1}(?=(\s{1}\-{1}\s{1}\d+$))"i,string(x.args[2])) ?
                            begin
                                push!(var_past_tmp,x.args[1])
                                push!(var_dyn_past_tmp,Symbol(string(x.args[1]) * "₍₋₁₎"))
                                push!(var_shift_dyn_past_tmp,Symbol(string(x.args[1]) * "₍₀₎"))
                                push!(var_shift2_dyn_past_tmp,Symbol(string(x.args[1]) * "₍₁₎"))
                            end : 
                        occursin(r"^(x|ex|exo|exogenous){1}(?=(\s{1}\+{1}\s{1}\d+$))"i,string(x.args[2])) ?
                            begin
                                push!(var_future_tmp,x.args[1])
                                push!(var_dyn_future_tmp,Symbol(string(x.args[1]) * "₍₁₎"))
                                push!(var_shift_dyn_future_tmp,Symbol(string(x.args[1]) * "₍₂₎"))
                            end : 
                        occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                            begin
                                push!(ss_tmp,x.args[1])
                                push!(ss_dyn_tmp,Symbol(string(x.args[1]) * "₍ₛₛ₎"))
                                # push!(var_dyn_tmp,Symbol(string(x.args[1]) * "₍ₛₛ₎"))
                            end :
                        x : 
                    x :
                x,
            ex.args[i])




            var_tmp = union(var_future_tmp,var_present_tmp,var_past_tmp)
            var_dyn_tmp = union(var_dyn_future_tmp,var_dyn_present_tmp,var_dyn_past_tmp,ss_dyn_tmp,exo_dyn_tmp)
            var_shift_dyn_tmp = union(var_shift_dyn_future_tmp,var_shift_dyn_present_tmp,var_shift_dyn_past_tmp,ss_dyn_tmp)

            push!(dynamic_variables_list,var_dyn_tmp)
            push!(dynamic_variables_future_list,var_shift_dyn_tmp)
            push!(var_list,var_tmp)
            push!(var_future_list,var_future_tmp)
            push!(var_present_list,var_present_tmp)
            push!(var_past_list,var_past_tmp)
            push!(exo_list,exo_tmp)
            push!(ss_list,ss_tmp)

            push!(dyn_shift_var_future_list,var_shift_dyn_future_tmp)
            push!(dyn_shift_var_present_list,var_shift_dyn_present_tmp)
            push!(dyn_shift_var_past_list,var_shift_dyn_past_tmp)

            push!(dyn_shift2_var_past_list,var_shift2_dyn_past_tmp)

            push!(dyn_var_future_list,var_dyn_future_tmp)
            push!(dyn_var_present_list,var_dyn_present_tmp)
            push!(dyn_var_past_list,var_dyn_past_tmp)
            push!(dyn_exo_list,exo_dyn_tmp)
            push!(dyn_ss_list,ss_dyn_tmp)

            push!(par_list,par_tmp)

            
            t_ex = postwalk(x -> 
                x isa Expr ? 
                    x.head == :(=) ? 
                        Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                        x.head == :ref ?
                            # issubset([x.args[2]],[:x :ex :exo :exogenous]) ? 
                            occursin(r"^(x|ex|exo|exogenous){1}$"i,string(x.args[2])) ?
                                begin
                                    push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍ₓ₎"))
                                    Symbol(string(x.args[1]) * "₍ₓ₎") 
                                end :
                            occursin(r"^(x|ex|exo|exogenous){1}(?=(\s{1}(\-|\+){1}\s{1}\d+$))"i,string(x.args[2])) ?  
                            # occursin(r"\+{1}\s{1}\d+$",string(x.args[2])) ?
                            x.args[2].args[1] == :(+) ?
                                begin
                                    # k = parse(Int16,match(r"\-{1}\s{1}\d+$",string(x.args[2])).match)
                                    k = x.args[2].args[3]
            
                                    while k > 2
                                        push!(exo_future,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾"))
                                        push!(exo_present,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾"))

                                        if Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎") ∈ aux_vars_created
                                            break
                                        else
                                            push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"))
                
                                            push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"))
                                            push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₁₎"))
                
                                            push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 2)) * "⁾₍₁₎")))
                                            
                                            k -= 1
                                        end
                                    end

                                    if Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎") ∉ aux_vars_created && k > 1
                                        push!(exo_future,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾"))
                                        push!(exo_present,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾"))

                                        push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"))
                
                                        push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"))
                                        push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₁₎"))
                
                                        push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "₍₁₎")))
                                    end

                                    if Symbol(string(x.args[1]) * "₍₀₎") ∉ aux_vars_created
                                        push!(aux_vars_created,Symbol(string(x.args[1]) * "₍₀₎"))
                                        
                                        push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "₍₀₎"),Symbol(string(x.args[1]) * "₍ₓ₎")))
                                    end

                                    push!(exo,Symbol(string(x.args[1])))
                                    push!(exo_future,Symbol(string(x.args[1])))
                                    push!(exo_present,Symbol(string(x.args[1])))
                                    
                                    push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍₀₎"))
                                    push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍ₓ₎"))
                                    
                                    push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍₁₎"))

                                    if x.args[2].args[3] > 1
                                        Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(x.args[2].args[3] - 1)) * "⁾₍₁₎")
                                    else
                                        Symbol(string(x.args[1]) * "₍₁₎")
                                    end
                                end :
                            # occursin(r"\-{1}\s{1}\d+$",string(x.args[2])) ?
                            x.args[2].args[1] == :(-) ?
                                begin
                                    # k = parse(Int16,match(r"\-{1}\s{1}\d+$",string(x.args[2])).match)
                                    k = - x.args[2].args[3]
                
                                    while k < -2
                                        push!(exo_past,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾"))
                                        push!(exo_present,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾"))

                                        if Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎") ∈ aux_vars_created
                                            break
                                        else
                                            push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"))
                
                                            push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"))
                                            push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₋₁₎"))
                
                                            push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 2)) * "⁾₍₋₁₎")))
                                            
                                            k += 1
                                        end
                                    end
                
                                    if Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎") ∉ aux_vars_created && k < -1
                                        push!(exo_past,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾"))
                                        push!(exo_present,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾"))

                                        push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"))
                
                                        push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"))
                                        push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₋₁₎"))
                
                                        push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "₍₋₁₎")))
                                    end
                                    
                                    if Symbol(string(x.args[1]) * "₍₀₎") ∉ aux_vars_created
                                        push!(aux_vars_created,Symbol(string(x.args[1]) * "₍₀₎"))
                                        
                                        push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "₍₀₎"),Symbol(string(x.args[1]) * "₍ₓ₎")))

                                    end

                                    push!(exo,Symbol(string(x.args[1])))
                                    push!(exo_past,Symbol(string(x.args[1])))
                                    push!(exo_present,Symbol(string(x.args[1])))
                                    
                                    push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍₀₎"))
                                    push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍ₓ₎"))
                                    
                                    push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍₋₁₎"))
                                    
                                    if  - x.args[2].args[3] < -1
                                        Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(x.args[2].args[3] - 1)) * "⁾₍₋₁₎")
                                    else
                                        Symbol(string(x.args[1]) * "₍₋₁₎")
                                    end
                                end :
                            x.args[1] : 
                            occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                                begin
                                    push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍ₛₛ₎"))
                                    Symbol(string(x.args[1]) * "₍ₛₛ₎") 
                                end :
                            x.args[2] isa Int ? 
                                x.args[2] > 1 ? 
                                    begin
                                        k = x.args[2]

                                        while k > 2
                                            if Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎") ∈ aux_vars_created
                                                break
                                            else
                                                push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"))

                                                push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"))
                                                push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₁₎"))

                                                push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 2)) * "⁾₍₁₎")))
                                                
                                                k -= 1
                                            end
                                        end

                                        if Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎") ∉ aux_vars_created
                                            push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"))

                                            push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"))
                                            push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₁₎"))

                                            push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "₍₁₎")))
                                        end

                                        push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍₁₎"))
                                        Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(x.args[2] - 1)) * "⁾₍₁₎")
                                    end :
                                1 >= x.args[2] >= 0 ? 
                                    begin
                                        push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍" * sub(string(x.args[2])) * "₎"))
                                        Symbol(string(x.args[1]) * "₍" * sub(string(x.args[2])) * "₎")
                                    end :  
                                -1 <= x.args[2] < 0 ? 
                                    begin
                                        push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍₋" * sub(string(x.args[2])) * "₎"))
                                        Symbol(string(x.args[1]) * "₍₋" * sub(string(x.args[2])) * "₎")
                                    end :
                                x.args[2] < -1 ? 
                                    begin
                                        k = x.args[2]

                                        while k < -2
                                            if Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎") ∈ aux_vars_created
                                                break
                                            else
                                                push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"))

                                                push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"))
                                                push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₋₁₎"))

                                                push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 2)) * "⁾₍₋₁₎")))
                                                
                                                k += 1
                                            end
                                        end

                                        if Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎") ∉ aux_vars_created
                                            push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"))

                                            push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"))
                                            push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₋₁₎"))

                                            push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "₍₋₁₎")))
                                        end

                                        push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍₋₁₎"))
                                        Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(x.args[2] + 1)) * "⁾₍₋₁₎")
                                    end :
                            x.args[1] :
                        x.args[1] : 
                    unblock(x) : 
                x,
            ex.args[i])
            # println(t_ex)
            push!(dyn_equations,unblock(t_ex))
            # println(dyn_equations)
            # println(dyn_ss_present)




            t_ex = postwalk(x -> 
                x isa Expr ? 
                    x.head == :(=) ? 
                        Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                        x.head == :ref ?
                            # issubset([x.args[2]],[:x :ex :exo :exogenous]) ? 0 :
                            occursin(r"^(x|ex|exo|exogenous){1}$"i,string(x.args[2])) ? 0 :
                            # issubset([x.args[2]],[:ss :SS :ℳ :StSt :steady :steadystate :steady_state :Steady_State]) ?
                            occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                                begin
                                    # push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍ₛₛ₎"))
                                    Symbol(string(x.args[1]) * "₍ₛₛ₎") 
                                end :
                            x.args[2] isa Int ? 
                                1 >= x.args[2] >= 0 ? 
                                    begin
                                        push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍" * sub(string(x.args[2] + 1)) * "₎"))
                                        Symbol(string(x.args[1]) * "₍" * sub(string(x.args[2] + 1)) * "₎")
                                    end :  
                                -1 <= x.args[2] < 0 ? 
                                    begin
                                        push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍₋" * sub(string(x.args[2] + 1)) * "₎"))
                                        Symbol(string(x.args[1]) * "₍₋" * sub(string(x.args[2] + 1)) * "₎")
                                    end :
                                x.args[2] < -1 ? 
                                    begin
                                        k = x.args[2]
                                        # while i < -1
                                        #     push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(i + 1)) * "⁾₍₋₁₎"))
                                        #     push!(dyn_equations,:(Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(i + 1)) * "⁾₍₀₎") = Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(i + 1)) * "⁾₍₀₎")))
                                        #     i += 1
                                        # end
                                        # push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₁₎"), Symbol(string(x.args[1]) * "₍₀₎")))
                                        push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₁₎"))
                                        push!(dyn_ss_present,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"))
                                        push!(dyn_ss_present,Symbol(string(x.args[1]) * "₍" * sub(string(k + 2)) * "₎"))
                                        Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎") # review this. timing might be off here.
                                    end :
                            x.args[1] :
                        x.args[1] : 
                    unblock(x) : 
                x,
            ex.args[i])
            # println(t_ex)
            push!(dyn_equations_future,unblock(t_ex))
            

            # write down SS equations
            prs_ex = postwalk(x -> 
                x isa Expr ? 
                    x.head == :(=) ? 
                        Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                            x.head == :ref ?
                                # issubset([x.args[2]],[:x :ex :exo :exogenous]) ? 0 : #set shocks to zero
                                occursin(r"^(x|ex|exo|exogenous){1}"i,string(x.args[2])) ? 0 :
                        x.args[1] : 
                    x.head == :call ?
                        x.args[1] == :* ?
                            x.args[2] isa Int ?
                                x.args[3] isa Int ?
                                    x :
                                :($(x.args[3]) * $(x.args[2])) :
                            x :
                        x.args[1] ∈ [:^, :log] ?
                            x.args[2] isa Symbol ?
                                begin
                                    if length(intersect(bounded_vars,[x.args[2]])) == 0
                                        push!(lower_bounds,eps())
                                        push!(upper_bounds,Inf)
                                        push!(bounded_vars,x.args[2]) 
                                    end
                                    x
                                end :
                            x.args[2].head == :call ?
                                begin
                                    push!(lower_bounds,eps())
                                    push!(upper_bounds,Inf)
                                    push!(bounded_vars,:($(Symbol("nonnegativity_auxilliary" * sub(string(1))))))
                                    push!(ss_equations_aux,Expr(:call,:-, Symbol("nonnegativity_auxilliary" * sub(string(1))) ,x.args[2])) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier i the code
                                    push!(nonnegativity_aux_vars,:($(Symbol("nonnegativity_auxilliary" * sub(string(length(nonnegativity_aux_vars)+1))))))
                                    :($(Symbol("nonnegativity_auxilliary" * sub(string(1))))^$(x.args[3]))
                                end :
                            x :
                        x :
                    unblock(x) : 
                x,
            ex.args[i])
            # println(ex.args[i])
            # println(unblock(prs_ex))
            # println(unblock(prs_ex))
            push!(ss_equations,unblock(prs_ex))
        end
    end

    # keep normal names as you write them in model block
    for (i,arg) in enumerate(ex.args)
        if isa(arg,Expr)
            prs_exx = postwalk(x -> 
                x isa Expr ? 
                    unblock(x) : 
                x,
            ex.args[i])
            push!(equations,unblock(prs_exx))
        end
    end

    var = collect(union(var_future,var_present,var_past))

    aux = collect(union(aux_future,aux_present,aux_past))

    creator = true

    default_optimizer = NLopt.LD_LBFGS
    # default_optimizer = NLopt.LN_BOBYQA

    dynamic_variables = collect(union(dyn_ss_past,dyn_ss_future,dyn_ss_present))
    #assemble data container of containers
    model_name = string(𝓂)
    quote
        # println($ss_equations)
       global $𝓂 =  ℳ(
                        $model_name,
                        $default_optimizer,
                        sort(collect($exo)), 
                        sort(collect($par)), 

                        $parameters,
                        # $par_values,
                        $parameter_values,

                        collect($ss),
                        collect($dynamic_variables),
                        collect($dyn_ss_past),
                        collect($dyn_ss_present),
                        collect($dyn_ss_future),

                        sort($aux),
                        sort(collect($aux_present)), 
                        sort(collect($aux_future)), 
                        sort(collect($aux_past)), 

                        sort(collect($exo_future)), 
                        sort(collect($exo_present)), 
                        sort(collect($exo_past)), 

                        sort($var), 
                        sort(collect($var_present)), 
                        sort(collect($var_future)), 
                        sort(collect($var_past)), 

                        $exo_list,
                        $var_list,
                        $dynamic_variables_list,
                        $dynamic_variables_future_list,

                        $ss_calib_list,
                        $par_calib_list,
                        $ss_list,
                        # $var_solved_list,
                        # $var_solved_calib_list,
                        # $var_redundant_list,
                        # $var_redundant_calib_list,
                        $par_list,
                        $var_future_list,
                        $var_present_list,
                        $var_past_list, 

                        $dyn_shift_var_future_list,
                        $dyn_shift_var_present_list,
                        $dyn_shift_var_past_list, 

                        $dyn_shift2_var_past_list, 

                        $dyn_var_future_list,
                        $dyn_var_present_list,
                        $dyn_var_past_list, 
                        $dyn_ss_list,
                        $dyn_exo_list,

                        $solved_vars, 
                        $solved_vals, 

                        $non_linear_solved_vars,
                        $non_linear_solved_vals,
                        # $solved_sub_vals,
                        # $solved_sub_values,
                        $ss_solve_blocks,
                        $ss_solve_blocks_optim,
                        $SS_init_guess,
                        $SS_solve_func,
                        $nonlinear_solution_helper,
                        $SS_dependencies,

                        $nonnegativity_aux_vars,
                        $ss_equations, 
                        $t_future_equations,
                        # :(function t_future_deriv($($var_future...),$($dyn_ss_future...),$($par...))
                        #     [$($t_future_equations...)]
                        # end),
                        $t_past_equations,
                        # :(function t_past_deriv($($var_past...),$($dyn_ss_past...),$($par...))
                        #     [$($t_past_equations...)]
                        # end),
                        $t_present_equations,
                        # :(function t_present_deriv($($var_present...),$($dyn_ss_present...),$($par...))
                        #     [$($t_present_equations...)]
                        # end),
                        $dyn_equations,
                        $dyn_equations_future,
                        $equations, 
                        $calibration_equations, 
                        $calibration_equations_parameters,

                        $bounds,
                        $bounded_vars,
                        $lower_bounds,
                        $upper_bounds,
                        $creator,

                        x->x,
                        # [],
                        # [],

                        timings([],
                                [],
                                [],
                                [],
                                [],
                                [],
                                [],
                                [],
                                [],
                                [],
                                [],
                                [],
                                [],
                                [],
                                
                                Int(0),
                                Int(0),
                                Int(0),
                                Int(0),
                                Int(0),
                                Int(0),
                                Int(0),

                                [],
                                [],
                                [],
                                [],
                                [],
                                [],
                                [],
                                [],
                                [],
                                []
                            ),

                            solution(
                                perturbation(  perturbation_solution(SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), x->x),
                                                perturbation_solution(SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), x->x),
                                                higher_order_perturbation_solution(Matrix{Float64}(undef,0,0), [],x->x),
                                                higher_order_perturbation_solution(Matrix{Float64}(undef,0,0), [],x->x)
                                ),
                                ComponentVector(nothing = 0.0),
                                Set([:first_order]),
                                true,
                                true,
                                false,
                                false
                            ),
                            # nothing
                            symbolics(
                                [],
                                [],
                                [],
                                
                                [],
                                [],
                                [],
                            
                                [],
                            
                                [],
                                [],
                                [],
                                [],
                                [],
                            
                                [],
                                [],
                                [],
                                [],
                                [],
                                [],
                                [],
                            
                                [],
                            
                                [],
                                [],
                                [],
                            
                                Set(),
                                Set(),
                                Set(),
                                Set(),
                            
                                [],
                                [],
                            
                                [],
                                [],
                                [],
                                []
                            )
                    );
        # nothing
    end
end






"""
$(SIGNATURES)
Adds parameter values and calibration equations to the previously defined model.

# Arguments
- `𝓂`: name of the object previously created containing the model information.
- `ex`: parameter, parameters values, and calibration equations

# Examples
```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end
```
"""
macro parameters(𝓂,ex)
    calib_equations = []
    calib_equations_list = []
    calib_eq_parameters = []

    ss_calib_list = []
    par_calib_list = []

    calib_parameters = []
    calib_values = []
    
    bounds = []

    # var_solved_list = []
    # var_redundant_list = []
    

    # label all variables parameters and exogenous vairables and timings across all equations
    postwalk(x -> 
        x isa Expr ?
            x.head == :(=) ? 
                x.args[1] isa Symbol ?
                    begin # this is normal calibration by setting values of parameters
                        push!(calib_values,x.args[2])
                        push!(calib_parameters,x.args[1])
                    end :
                begin # this is calibration by targeting SS values
                    push!(calib_eq_parameters,x.args[1].args[2])
                    push!(calib_equations,Expr(:(=),x.args[1].args[3], unblock(x.args[2])))
                end :
            x.head == :comparison ? 
                push!(bounds,x) :
            x.head == :call ?
                issubset([x.args[1]], [:(<) :(>) :(<=) :(>=)]) ?
                    push!(bounds,x) :
                x :
            x :
        x,
    ex)

    for i in 1:length(calib_equations)
        ss_tmp = Set()
        par_tmp = Set()

        # push!(var_solved_list,Set())
        # push!(var_redundant_list,Set())

        postwalk(x -> 
            x isa Expr ? 
                x.head == :ref ?
                    # issubset([x.args[2]],[:ss :SS :ℳ :StSt :steady :steadystate :steady_state :Steady_State]) ?
                    occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                        push!(ss_tmp,x.args[1]) :
                    x : 
                x :
            x,
        calib_equations[i])


        postwalk(x -> 
        x isa Symbol ? 
            # issubset([x],[:(-) :(+) :(*) :(/) :(^) :ss :SS :ℳ :StSt :steady :steadystate :steady_state :Steady_State]) ?
            occursin(r"^(\+|\-|\*|\/|\^|ss|stst|steady|steadystate|steady_state){1}$"i,string(x)) ?
                x :
                begin
                    diffed = setdiff([x],ss_tmp)
                    # println("pre",diffed)
                    # filter!(isempty,diffed)
                    # println(diffed)
                    if !isempty(diffed)
                        push!(par_tmp,diffed[1])
                    end
                end :
        x,
        calib_equations[i])

        push!(ss_calib_list,ss_tmp)
        # if par_tmp == Set([Symbol[]])
        #     push!(par_calib_list,Set())
        # else
            push!(par_calib_list,par_tmp)
        # end
        
        # write down calibration equations
        prs_ex = postwalk(x -> 
            x isa Expr ? 
                x.head == :(=) ? 
                    Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                        x.head == :ref ?
                            # issubset([x.args[2]],[:ss :SS :ℳ :StSt :steady :steadystate :steady_state :Steady_State]) ? 
                            occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                        # split(string(x.args[1]), "₍")[1] :
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
        calib_equations[i])
        push!(calib_equations_list,unblock(prs_ex))
    end
    # println(unblock(prs_ex))
    # push!(ss_equations,unblock(prs_ex))
    # map(xx -> postwalk(x -> 
    # x
    
    # ,xx),calib_equations)
    bounded_vars = []
    upper_bounds = []
    lower_bounds = []

    #parse bounds
    for bound in bounds
        postwalk(x -> 
        x isa Expr ?
            x.head == :comparison ? 
                x.args[2] == :(<) ?
                    x.args[4] == :(<) ?
                        begin
                            push!(bounded_vars,x.args[3]) 
                            push!(lower_bounds,x.args[1]+eps()) 
                            push!(upper_bounds,x.args[5]-eps()) 
                        end :
                    x.args[4] == :(<=) ?
                        begin
                            push!(bounded_vars,x.args[3]) 
                            push!(lower_bounds,x.args[1]+eps()) 
                            push!(upper_bounds,x.args[5]) 
                        end :
                    x :
                x.args[2] == :(<=) ?
                    x.args[4] == :(<) ?
                        begin
                            push!(bounded_vars,x.args[3]) 
                            push!(lower_bounds,x.args[1]) 
                            push!(upper_bounds,x.args[5]-eps()) 
                        end :
                    x.args[4] == :(<=) ?
                        begin
                            push!(bounded_vars,x.args[3]) 
                            push!(lower_bounds,x.args[1]) 
                            push!(upper_bounds,x.args[5]) 
                        end :
                    x :

                x.args[2] == :(>) ?
                    x.args[4] == :(>) ?
                        begin
                            push!(bounded_vars,x.args[3]) 
                            push!(lower_bounds,x.args[5]+eps()) 
                            push!(upper_bounds,x.args[1]-eps()) 
                        end :
                    x.args[4] == :(>=) ?
                        begin
                            push!(bounded_vars,x.args[3]) 
                            push!(lower_bounds,x.args[5]+eps()) 
                            push!(upper_bounds,x.args[1]) 
                        end :
                    x :
                x.args[2] == :(>=) ?
                    x.args[4] == :(>) ?
                        begin
                            push!(bounded_vars,x.args[3]) 
                            push!(lower_bounds,x.args[5]) 
                            push!(upper_bounds,x.args[1]-eps()) 
                        end :
                    x.args[4] == :(>=) ?
                        begin
                            push!(bounded_vars,x.args[3]) 
                            push!(lower_bounds,x.args[5]) 
                            push!(upper_bounds,x.args[1]) 
                        end :
                    x :
                x :

            x.head ==  :call ? 
                x.args[1] == :(<) ?
                    x.args[2] isa Symbol ? 
                        begin
                            push!(bounded_vars,x.args[2]) 
                            push!(upper_bounds,x.args[3]-eps()) 
                            push!(lower_bounds,-Inf) 
                        end :
                    x.args[3] isa Symbol ? 
                        begin
                            push!(bounded_vars,x.args[3]) 
                            push!(lower_bounds,x.args[2]+eps()) 
                            push!(upper_bounds,Inf) 
                        end :
                    x :
                x.args[1] == :(>) ?
                    x.args[2] isa Symbol ? 
                        begin
                            push!(bounded_vars,x.args[2]) 
                            push!(lower_bounds,x.args[3]+eps()) 
                            push!(upper_bounds,Inf) 
                        end :
                    x.args[3] isa Symbol ? 
                        begin
                            push!(bounded_vars,x.args[3]) 
                            push!(upper_bounds,x.args[2]-eps()) 
                            push!(lower_bounds,-Inf) 
                        end :
                    x :
                x.args[1] == :(>=) ?
                    x.args[2] isa Symbol ? 
                        begin
                            push!(bounded_vars,x.args[2]) 
                            push!(lower_bounds,x.args[3]) 
                            push!(upper_bounds,Inf) 
                        end :
                    x.args[3] isa Symbol ? 
                        begin
                            push!(bounded_vars,x.args[3]) 
                            push!(upper_bounds,x.args[2])
                            push!(lower_bounds,-Inf) 
                        end :
                    x :
                x.args[1] == :(<=) ?
                    x.args[2] isa Symbol ? 
                        begin
                            push!(bounded_vars,x.args[2]) 
                            push!(upper_bounds,x.args[3]) 
                            push!(lower_bounds,-Inf) 
                        end :
                    x.args[3] isa Symbol ? 
                        begin
                            push!(bounded_vars,x.args[3]) 
                            push!(lower_bounds,x.args[2]) 
                            push!(upper_bounds,Inf) 
                        end :
                    x :
                x :
            x :
        x,bound)
    end
    # :(print($𝓂))


    
    quote
        global $𝓂.bounds = $bounds

        global $𝓂.bounded_vars = $bounded_vars
        global $𝓂.lower_bounds = $lower_bounds
        global $𝓂.upper_bounds = $upper_bounds

        global $𝓂.ss_calib_list = $ss_calib_list
        global $𝓂.par_calib_list = $par_calib_list

        global $𝓂.parameters = $calib_parameters
        global $𝓂.parameter_values = $calib_values
        global $𝓂.calibration_equations = $calib_equations_list
        global $𝓂.calibration_equations_parameters = $calib_eq_parameters
        global $𝓂.solution.outdated = true
        global $𝓂.solution.NSSS_outdated = true
        global $𝓂.solution.functions_written = false
        nothing
    end
end