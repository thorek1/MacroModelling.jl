using MacroTools



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

# ex = :(delta | c[ss]/y[ss] = 1 - I_K_ratio)
ex = :(c[ss]/y[ss] + 100 = 1 - I_K_ratio / 5 | delta)
# ex = :(delta = .9)
dump(ex)
MacroTools.postwalk(x -> 
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
                x.args[1].args[1] == :| ?
                    begin # this is calibration by targeting SS values
                        push!(calib_eq_parameters,x.args[1].args[2])
                        push!(calib_equations,Expr(:(=),x.args[1].args[3], unblock(x.args[2])))
                    end :
                begin # this is calibration by targeting SS values
                    push!(calib_eq_parameters,x.args[2].args[end].args[end])
                    push!(calib_equations,Expr(:(=),x.args[1], unblock(x.args[2].args[2].args[2])))
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