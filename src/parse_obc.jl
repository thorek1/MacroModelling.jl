# import SymPyPythonCall as SPyPyC
# import MacroTools
# import MacroModelling: get_symbols

# # eq = :(0 = max(r̄ - r[0], r̂[0] - r[0]))
# eq = :(0 = max(r̄ - r[0], r̂[0] - r[0]))
# using MacroTools

# function transform_expression(expr::Expr)
#     # Dictionary to store the transformations for reversing
#     reverse_transformations = Dict{Symbol, Expr}()

#     # Counter for generating unique placeholders
#     unique_counter = Ref(0)

#     # Step 1: Replace min/max calls and record their original form
#     function replace_min_max(expr)
#         if expr isa Expr && expr.head == :call && (expr.args[1] == :min || expr.args[1] == :max)
#             # Replace min/max functions with a placeholder
#             # placeholder = Symbol("minimal__P", unique_counter[])
#             placeholder = :minmax__P
#             unique_counter[] += 1

#             # Store the original min/max call for reversal
#             reverse_transformations[placeholder] = expr

#             return placeholder
#         else
#             return expr
#         end
#     end

#     # Step 2: Transform :ref fields in the rest of the expression
#     function transform_ref_fields(expr)
#         if expr isa Expr && expr.head == :ref && isa(expr.args[1], Symbol)
#             # Handle :ref expressions
#             if isa(expr.args[2], Number) || isa(expr.args[2], Symbol)
#                 new_symbol = Symbol(expr.args[1], "_", expr.args[2])
#             else
#                 # Generate a unique placeholder for complex :ref
#                 unique_counter[] += 1
#                 placeholder = Symbol("__placeholder", unique_counter[])
#                 new_symbol = placeholder
#             end

#             # Record the reverse transformation
#             reverse_transformations[new_symbol] = expr

#             return new_symbol
#         else
#             return expr
#         end
#     end


#     # Replace equality sign with minus
#     function replace_equality_with_minus(expr)
#         if expr isa Expr && expr.head == :(=)
#             return Expr(:call, :-, expr.args...)
#         else
#             return expr
#         end
#     end

#     # Apply transformations
#     expr = MacroTools.postwalk(replace_min_max, expr)
#     expr = MacroTools.postwalk(transform_ref_fields, expr)
#     transformed_expr = MacroTools.postwalk(replace_equality_with_minus, expr)
    
#     return transformed_expr, reverse_transformations
# end


# function reverse_transformation(transformed_expr::Expr, reverse_dict::Dict{Symbol, Expr})
#     # Function to replace the transformed symbols with their original form
#     function revert_symbol(expr)
#         if expr isa Symbol && haskey(reverse_dict, expr)
#             return reverse_dict[expr]
#         else
#             return expr
#         end
#     end

#     # Revert the expression using MacroTools.postwalk
#     reverted_expr = MacroTools.postwalk(revert_symbol, transformed_expr)

#     return reverted_expr
# end


# function transform_obc(ex::Expr)
#     transformed_expr, reverse_dict = transform_expression(ex)

#     for symbs in get_symbols(transformed_expr)
#         eval(:($symbs = SPyPyC.symbols($(string(symbs)), real = true, finite = true)))
#     end

#     soll = try SPyPyC.solve(eval(transformed_expr), minmax__P)
#     catch
#     end

#     if length(soll) == 1
#         sorted_minmax = Expr(:call, reverse_dict[:minmax__P].args[1], :($(reverse_dict[:minmax__P].args[2]) - $(Meta.parse(string(soll[1])))),  :($(reverse_dict[:minmax__P].args[3]) - $(Meta.parse(string(soll[1])))))
#         return Expr(:call, :(=), 0, reverse_transformation(sorted_minmax, reverse_dict))
#     else
#         @error "Occasionally binding constraint not well-defined. See documentation for examples."
#     end
# end

# using BenchmarkTools
# @benchmark transform_obc(eq)





# original_expr = reverse_transformation(transformed_expr, reverse_dict)






#     # create symbols in module scope
#     symbols_in_dynamic_equations = reduce(union,get_symbols.(𝓂.dyn_equations))

#     symbols_in_dynamic_equations_wo_subscripts = Symbol.(replace.(string.(symbols_in_dynamic_equations),r"₍₋?(₀|₁|ₛₛ|ₓ)₎$"=>""))

#     symbols_in_ss_equations = reduce(union,get_symbols.(𝓂.ss_aux_equations))

#     symbols_in_equation = union(𝓂.parameters_in_equations,𝓂.parameters,𝓂.parameters_as_function_of_parameters,symbols_in_dynamic_equations,symbols_in_dynamic_equations_wo_subscripts,symbols_in_ss_equations)#,𝓂.dynamic_variables_future)

#     l_bnds = Dict(𝓂.bounded_vars .=> 𝓂.lower_bounds)
#     u_bnds = Dict(𝓂.bounded_vars .=> 𝓂.upper_bounds)

#     symbols_pos = []
#     symbols_neg = []
#     symbols_none = []

#     for symb in symbols_in_equation
#         if symb in 𝓂.bounded_vars
#             if l_bnds[symb] >= 0
#                 push!(symbols_pos, symb)
#             elseif u_bnds[symb] <= 0
#                 push!(symbols_neg, symb)
#             else 
#                 push!(symbols_none, symb)
#             end
#         else
#             push!(symbols_none, symb)
#         end
#     end

#     for pos in symbols_pos
#         eval(:($pos = SPyPyC.symbols($(string(pos)), real = true, finite = true, positive = true)))
#     end
#     for neg in symbols_neg
#         eval(:($neg = SPyPyC.symbols($(string(neg)), real = true, finite = true, negative = true)))
#     end
#     for none in symbols_none
#         eval(:($none = SPyPyC.symbols($(string(none)), real = true, finite = true)))
#     end

#     symbolics(map(x->eval(:($x)),𝓂.ss_aux_equations),
#                 map(x->eval(:($x)),𝓂.dyn_equations),
#                 # map(x->eval(:($x)),𝓂.dyn_equations_future),

#                 # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift_var_present_list),
#                 # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift_var_past_list),
#                 # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift_var_future_list),

#                 # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift2_var_past_list),

#                 map(x->Set(eval(:([$(x...)]))),𝓂.dyn_var_present_list),
#                 map(x->Set(eval(:([$(x...)]))),𝓂.dyn_var_past_list),
#                 map(x->Set(eval(:([$(x...)]))),𝓂.dyn_var_future_list),
#                 # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_ss_list),
#                 map(x->Set(eval(:([$(x...)]))),𝓂.dyn_exo_list),

#                 # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_exo_future_list),
#                 # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_exo_present_list),
#                 # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_exo_past_list),

#                 map(x->Set(eval(:([$(x...)]))),𝓂.dyn_future_list),
#                 map(x->Set(eval(:([$(x...)]))),𝓂.dyn_present_list),
#                 map(x->Set(eval(:([$(x...)]))),𝓂.dyn_past_list),

#                 map(x->Set(eval(:([$(x...)]))),𝓂.var_present_list_aux_SS),
#                 map(x->Set(eval(:([$(x...)]))),𝓂.var_past_list_aux_SS),
#                 map(x->Set(eval(:([$(x...)]))),𝓂.var_future_list_aux_SS),
#                 map(x->Set(eval(:([$(x...)]))),𝓂.ss_list_aux_SS),

#                 map(x->Set(eval(:([$(x...)]))),𝓂.var_list_aux_SS),
#                 # map(x->Set(eval(:([$(x...)]))),𝓂.dynamic_variables_list),
#                 # map(x->Set(eval(:([$(x...)]))),𝓂.dynamic_variables_future_list),
#                 map(x->Set(eval(:([$(x...)]))),𝓂.par_list_aux_SS),

#                 map(x->eval(:($x)),𝓂.calibration_equations),
#                 map(x->eval(:($x)),𝓂.calibration_equations_parameters),
#                 # map(x->eval(:($x)),𝓂.parameters),

#                 # Set(eval(:([$(𝓂.var_present...)]))),
#                 # Set(eval(:([$(𝓂.var_past...)]))),
#                 # Set(eval(:([$(𝓂.var_future...)]))),
#                 Set(eval(:([$(𝓂.vars_in_ss_equations...)]))),
#                 Set(eval(:([$(𝓂.var...)]))),
#                 Set(eval(:([$(𝓂.➕_vars...)]))),

#                 map(x->Set(eval(:([$(x...)]))),𝓂.ss_calib_list),
#                 map(x->Set(eval(:([$(x...)]))),𝓂.par_calib_list),

#                 [Set() for _ in 1:length(𝓂.ss_aux_equations)],
#                 # [Set() for _ in 1:length(𝓂.calibration_equations)],
#                 # [Set() for _ in 1:length(𝓂.ss_aux_equations)],
#                 # [Set() for _ in 1:length(𝓂.calibration_equations)]
#                 )
# end



# function remove_redundant_SS_vars!(𝓂::ℳ, Symbolics::symbolics)
#     ss_equations = Symbolics.ss_equations

#     # check variables which appear in two time periods. they might be redundant in steady state
#     redundant_vars = intersect.(
#         union.(
#             intersect.(Symbolics.var_future_list,Symbolics.var_present_list),
#             intersect.(Symbolics.var_future_list,Symbolics.var_past_list),
#             intersect.(Symbolics.var_present_list,Symbolics.var_past_list),
#             intersect.(Symbolics.ss_list,Symbolics.var_present_list),
#             intersect.(Symbolics.ss_list,Symbolics.var_past_list),
#             intersect.(Symbolics.ss_list,Symbolics.var_future_list)
#         ),
#     Symbolics.var_list)
#     redundant_idx = getindex(1:length(redundant_vars), (length.(redundant_vars) .> 0) .& (length.(Symbolics.var_list) .> 1))

#     for i in redundant_idx
#         for var_to_solve_for in redundant_vars[i]
#             soll = try SPyPyC.solve(ss_equations[i],var_to_solve_for)
#             catch
#             end
            
#             if isnothing(soll)
#                 continue
#             end
            
#             if length(soll) == 0 || soll == SPyPyC.Sym[0] # take out variable if it is redundant from that euation only
#                 push!(Symbolics.var_redundant_list[i],var_to_solve_for)
#                 ss_equations[i] = ss_equations[i].subs(var_to_solve_for,1).replace(SPyPyC.Sym(ℯ),exp(1)) # replace euler constant as it is not translated to julia properly
#             end

#         end
#     end