
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
    NSSS_solver_cache = CircularBuffer{Vector{Vector{Float64}}}(500)
    SS_solve_func = nothing
    nonlinear_solution_helper = nothing
    SS_dependencies = nothing

    ss_equations = []
    equations = []
    calibration_equations = []
    calibration_equations_parameters = []

    bounds⁺ = Set()

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
    ss_and_aux_equations = []
    aux_vars_created = Set()

    # write down SS equations and go by equation
    for (i,arg) in enumerate(ex.args)
        if isa(arg,Expr)

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

            # write down dynamic equations
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



            # write down dynamic equations shifted one period into future
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
                                occursin(r"^(x|ex|exo|exogenous){1}"i,string(x.args[2])) ? 0 :
                        x.args[1] : 
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
            ex.args[i])
            push!(ss_equations,unblock(prs_ex))




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
                                :($(x.args[3]) * $(x.args[2])) :
                            x :
                        x.args[1] ∈ [:^] ?
                            !(x.args[3] isa Int) ?
                                x.args[2] isa Symbol ? # nonnegative parameters 
                                        begin
                                            # if length(intersect(bounded_vars,[x.args[2]])) == 0
                                                # push!(lower_bounds,eps())
                                                # push!(upper_bounds,Inf)
                                                push!(bounds⁺,x.args[2]) 
                                            # end
                                            x
                                        end :
                                x.args[2].head == :ref ?
                                    x.args[2].args[1] isa Symbol ? # nonnegative variables 
                                        begin
                                            # if length(intersect(bounded_vars,[x.args[2].args[1]])) == 0
                                                # push!(lower_bounds,eps())
                                                # push!(upper_bounds,Inf)
                                                push!(bounds⁺,x.args[2].args[1]) 
                                            # end
                                            x
                                        end :
                                    x :
                                x.args[2].head == :call ? # nonnegative expressions
                                    begin
                                        # push!(lower_bounds,eps())
                                        # push!(upper_bounds,Inf)
                                        push!(bounds⁺,:($(Symbol("nonnegativity_auxilliary" * sub(string(length(nonnegativity_aux_vars)+1))))))
                                        
                                        push!(ss_and_aux_equations, Expr(:call,:-, :($(Expr(:ref,Symbol("nonnegativity_auxilliary" * sub(string(length(nonnegativity_aux_vars)+1))),0))), x.args[2])) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier i the code
                                        push!(nonnegativity_aux_vars,Symbol("nonnegativity_auxilliary" * sub(string(length(nonnegativity_aux_vars)+1))))
                                        :($(Expr(:ref,Symbol("nonnegativity_auxilliary" * sub(string(length(nonnegativity_aux_vars)))),0)) ^ $(x.args[3]))
                                    end :
                                x :
                            x :
                        x.args[1] ∈ [:log, :norminvcdf, :erfcinv, :qnorm, :norminv] ?
                            x.args[2] isa Symbol ? # nonnegative parameters 
                                begin
                                    # if length(intersect(bounded_vars,[x.args[2]])) == 0
                                        # push!(lower_bounds,eps())
                                        # push!(upper_bounds,Inf)
                                        push!(bounds⁺,x.args[2]) 
                                    # end
                                    x
                                end :
                            x.args[2].head == :ref ?
                                x.args[2].args[1] isa Symbol ? # nonnegative variables 
                                    begin
                                        # if length(intersect(bounded_vars,[x.args[2].args[1]])) == 0
                                            # push!(lower_bounds,eps())
                                            # push!(upper_bounds,Inf)
                                            push!(bounds⁺,x.args[2].args[1]) 
                                        # end
                                        x
                                    end :
                                x :
                            x.args[2].head == :call ? # nonnegative expressions
                                begin
                                    # push!(lower_bounds,eps())
                                    # push!(upper_bounds,Inf)
                                    push!(bounds⁺,:($(Symbol("nonnegativity_auxilliary" * sub(string(length(nonnegativity_aux_vars)+1))))))
                                    
                                    push!(ss_and_aux_equations, Expr(:call,:-, :($(Expr(:ref,Symbol("nonnegativity_auxilliary" * sub(string(length(nonnegativity_aux_vars)+1))),0))), x.args[2])) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier i the code
                                    push!(nonnegativity_aux_vars,Symbol("nonnegativity_auxilliary" * sub(string(length(nonnegativity_aux_vars)+1))))
                                    :($(Expr(:call, x.args[1], Expr(:ref,Symbol("nonnegativity_auxilliary" * sub(string(length(nonnegativity_aux_vars)))),0))))
                                end :
                            x :
                        x :
                    x :
                x,
            ex.args[i])

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
    for eq in ss_and_aux_equations
        var_tmp = Set()
        ss_tmp = Set()
        par_tmp = Set()
        var_future_tmp = Set()
        var_present_tmp = Set()
        var_past_tmp = Set()

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
        prs_ex = postwalk(x -> 
            x isa Expr ? 
                x.head == :(=) ? 
                    Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                        x.head == :ref ?
                            occursin(r"^(x|ex|exo|exogenous){1}"i,string(x.args[2])) ? 0 :
                    x.args[1] : 
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
        eq)
        
        push!(ss_aux_equations,unblock(prs_ex))

    end

    var = collect(union(var_future,var_present,var_past))

    aux = collect(union(aux_future,aux_present,aux_past))


    # create timings
    var = sort(union(var, aux, exo_past, exo_future))
    var_past = sort(collect(union(var_past, aux_past, exo_past)))
    var_future = sort(collect(union(var_future, aux_future, exo_future)))

    present_only              = sort(setdiff(var,union(var_past,var_future)))
    future_not_past           = sort(setdiff(var_future, var_past))
    past_not_future           = sort(setdiff(var_past, var_future))
    mixed                     = sort(setdiff(var, union(present_only, future_not_past, past_not_future)))
    future_not_past_and_mixed = sort(union(future_not_past,mixed))
    past_not_future_and_mixed = sort(union(past_not_future,mixed))
    present_but_not_only      = sort(setdiff(var,present_only))
    mixed_in_past             = sort(intersect(var_past, mixed))
    not_mixed_in_past         = sort(setdiff(var_past,mixed_in_past))
    mixed_in_future           = sort(intersect(var_future, mixed))
    exo                       = sort(collect(exo))
    var                       = sort(var)
    aux                       = sort(aux)
    exo_present               = sort(collect(exo_present))

    nPresent_only              = length(present_only)
    nMixed                     = length(mixed)
    nFuture_not_past_and_mixed = length(future_not_past_and_mixed)
    nPast_not_future_and_mixed = length(past_not_future_and_mixed)
    nPresent_but_not_only      = length(present_but_not_only)
    nVars                      = length(var)
    nExo                       = length(collect(exo))

    present_only_idx              = indexin(present_only,var)
    present_but_not_only_idx      = indexin(present_but_not_only,var)
    future_not_past_and_mixed_idx = indexin(future_not_past_and_mixed,var)
    past_not_future_and_mixed_idx = indexin(past_not_future_and_mixed,var)
    mixed_in_future_idx           = indexin(mixed_in_future,var_future)
    mixed_in_past_idx             = indexin(mixed_in_past,var_past)
    not_mixed_in_past_idx         = indexin(not_mixed_in_past,var_past)
    past_not_future_idx           = indexin(past_not_future,var)

    reorder       = map(x->(getindex(1:nVars, x .== [present_only..., past_not_future..., future_not_past_and_mixed...]))[1], var)
    dynamic_order = map(x->(getindex(1:nPresent_but_not_only, x .== [past_not_future..., future_not_past_and_mixed...]))[1], present_but_not_only)

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



    var_future = reduce(union,var_future_list_aux_SS)
    var_present = reduce(union,var_present_list_aux_SS)
    var_past = reduce(union,var_past_list_aux_SS)
      
    var = collect(setdiff(union(var_future,var_present,var_past),nonnegativity_aux_vars))

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


    default_optimizer = NLopt.LD_LBFGS
    # default_optimizer = Optimisers.ADAM
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

                        $ss_calib_list, #no_var_
                        $par_calib_list, #no_var_

                        $ss_list,

                        $ss_aux_equations,
                        $var_list_aux_SS,
                        $ss_list_aux_SS,
                        $par_list_aux_SS,
                        $var_future_list_aux_SS,
                        $var_present_list_aux_SS,
                        $var_past_list_aux_SS,

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
                        $NSSS_solver_cache,
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

                        $calibration_equations, #no_var_

                        $calibration_equations, 
                        $calibration_equations_parameters,

                        collect($bounds⁺),
                        $bounded_vars,
                        $lower_bounds,
                        $upper_bounds,

                        x->x,

                        $T,

                        solution(
                            perturbation(  perturbation_solution(SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), x->x),
                                            perturbation_solution(SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), x->x),
                                            higher_order_perturbation_solution(Matrix{Float64}(undef,0,0), [],x->x),
                                            higher_order_perturbation_solution(Matrix{Float64}(undef,0,0), [],x->x)
                            ),
                            ComponentVector(nothing = 0.0),
                            Set([:first_order]),
                            Set([:linear_time_iteration, :riccati, :first_order, :second_order, :third_order]),
                            true,
                            false,
                            false
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
    calib_equations_no_var = []
    calib_values_no_var = []
    
    calib_parameters_no_var = []
    
    
    calib_eq_parameters = []
    calib_equations_list = []
    
    ss_calib_list = []
    par_calib_list = []
    
    
    calib_equations_no_var_list = []
    
    ss_no_var_calib_list = []
    par_no_var_calib_list = []
    
    calib_parameters = []
    calib_values = []
    
    
    bounds = []
    
    # label all variables parameters and exogenous vairables and timings across all equations
    postwalk(x -> 
        x isa Expr ?
            x.head == :(=) ? 
                x.args[1] isa Symbol ?
                    typeof(x.args[2]) ∈ [Int, Float64] ?
                        begin # this is normal calibration by setting values of parameters
                            push!(calib_values,x.args[2])
                            push!(calib_parameters,x.args[1])
                        end :
                    begin # this is normal calibration by setting values of parameters
                        # push!(calib_equations_no_var,Expr(:(=),x.args[1], unblock(x.args[2])))
                        push!(calib_values_no_var,unblock(x.args[2]))
                        push!(calib_parameters_no_var,x.args[1])
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
    
        push!(ss_calib_list,ss_tmp)
        push!(par_calib_list,par_tmp)
        
        # write down calibration equations
        prs_ex = postwalk(x -> 
            x isa Expr ? 
                x.head == :(=) ? 
                    Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
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
        push!(calib_equations_list,unblock(prs_ex))
    end
    
    
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
    bounded_vars = []
    upper_bounds = []
    lower_bounds = []

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

    
    return quote
        Main.$𝓂.bounded_vars = $bounded_vars
        Main.$𝓂.lower_bounds = $lower_bounds
        Main.$𝓂.upper_bounds = $upper_bounds

        Main.$𝓂.ss_calib_list = $ss_calib_list
        Main.$𝓂.par_calib_list = $par_calib_list

        Main.$𝓂.ss_no_var_calib_list = $ss_no_var_calib_list
        Main.$𝓂.par_no_var_calib_list = $par_no_var_calib_list

        Main.$𝓂.parameters = $calib_parameters
        Main.$𝓂.parameter_values = $calib_values
        Main.$𝓂.calibration_equations = $calib_equations_list
        Main.$𝓂.parameters_as_function_of_parameters = $calib_parameters_no_var
        Main.$𝓂.calibration_equations_no_var = $calib_equations_no_var_list
        Main.$𝓂.calibration_equations_parameters = $calib_eq_parameters
        Main.$𝓂.solution.outdated_NSSS = true
        Main.$𝓂.solution.functions_written = false
        nothing
    end
end