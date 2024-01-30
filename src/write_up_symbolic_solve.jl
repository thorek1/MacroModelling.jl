using MacroModelling
include("../models/FS2000.jl")

import BlockTriangularForm
import SymPyPythonCall as SPyPyC
import MacroTools: postwalk
import MacroModelling: create_symbols_eqs!, remove_redundant_SS_vars!, solve_steady_state!, get_symbols
𝓂 = FS2000

symbolics = create_symbols_eqs!(𝓂)

remove_redundant_SS_vars!(𝓂, symbolics) 

# solve_steady_state!(RBC_baseline, true, symbolics, verbose = true)

symbolic_SS = true
Symbolics = symbolics
verbose = false


unknowns = union(Symbolics.vars_in_ss_equations,Symbolics.calibration_equations_parameters)

@assert length(unknowns) <= length(Symbolics.ss_equations) + length(Symbolics.calibration_equations) "Unable to solve steady state. More unknowns than equations."

incidence_matrix = fill(0,length(unknowns),length(unknowns))

eq_list = vcat(union.(setdiff.(union.(Symbolics.var_list,
                                    Symbolics.ss_list),
                                Symbolics.var_redundant_list),
                        Symbolics.par_list),
                union.(Symbolics.ss_calib_list,
                        Symbolics.par_calib_list))


for i in 1:length(unknowns)
    for k in 1:length(unknowns)
        incidence_matrix[i,k] = collect(unknowns)[i] ∈ collect(eq_list)[k]
    end
end

Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(incidence_matrix))
R̂ = []
for i in 1:n_blocks
    [push!(R̂, n_blocks - i + 1) for ii in R[i]:R[i+1] - 1]
end
push!(R̂,1)

vars = hcat(P, R̂)'
eqs = hcat(Q, R̂)'

# @assert all(eqs[1,:] .> 0) "Could not solve system of steady state and calibration equations for: " * repr([collect(Symbol.(unknowns))[vars[1,eqs[1,:] .< 0]]...]) # repr([vcat(Symbolics.ss_equations,Symbolics.calibration_equations)[-eqs[1,eqs[1,:].<0]]...])
@assert all(eqs[1,:] .> 0) "Could not solve system of steady state and calibration equations. Number of redundant euqations: " * repr(sum(eqs[1,:] .< 0)) * ". Try defining some steady state values as parameters (e.g. r[ss] -> r̄). Nonstationary variables are not supported as of now." # repr([vcat(Symbolics.ss_equations,Symbolics.calibration_equations)[-eqs[1,eqs[1,:].<0]]...])

n = n_blocks

ss_equations = vcat(Symbolics.ss_equations,Symbolics.calibration_equations)# .|> SPyPyC.Sym
# println(ss_equations)

SS_solve_func = []

atoms_in_equations = Set{Symbol}()
atoms_in_equations_list = []
relevant_pars_across = []
NSSS_solver_cache_init_tmp = []

min_max_errors = []

n_block = 1


while n > 0 
    if length(eqs[:,eqs[2,:] .== n]) == 2
        var_to_solve_for = collect(unknowns)[vars[:,vars[2,:] .== n][1]]

        eq_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1]]

        # eliminate min/max from equations if solving for variables inside min/max. set to the variable we solve for automatically
        parsed_eq_to_solve_for = eq_to_solve |> string |> Meta.parse

        minmax_fixed_eqs = postwalk(x -> 
            x isa Expr ?
                x.head == :call ? 
                    x.args[1] ∈ [:Max,:Min] ?
                        Symbol(var_to_solve_for) ∈ get_symbols(x.args[2]) ?
                            x.args[2] :
                        Symbol(var_to_solve_for) ∈ get_symbols(x.args[3]) ?
                            x.args[3] :
                        x :
                    x :
                x :
            x,
        parsed_eq_to_solve_for)

        if parsed_eq_to_solve_for != minmax_fixed_eqs
            [push!(atoms_in_equations, a) for a in setdiff(get_symbols(parsed_eq_to_solve_for), get_symbols(minmax_fixed_eqs))]
            push!(min_max_errors,:(solution_error += abs($parsed_eq_to_solve_for)))
            eq_to_solve = eval(minmax_fixed_eqs)
        end

        soll = try SPyPyC.solve(eq_to_solve,var_to_solve_for)
        catch
        end
        
        if isnothing(soll)
            # println("Could not solve single variables case symbolically.")
            println("Failed finding solution symbolically for: ",var_to_solve_for," in: ",eq_to_solve)
            # solve numerically
            continue
        # elseif PythonCall.pyconvert(Bool,soll[1].is_number)

        elseif soll[1].is_number == true
            # ss_equations = ss_equations.subs(var_to_solve,soll[1])
            ss_equations = [eq.subs(var_to_solve_for,soll[1]) for eq in ss_equations]
            
            push!(𝓂.solved_vars,Symbol(var_to_solve_for))
            push!(𝓂.solved_vals,Meta.parse(string(soll[1])))

            if (𝓂.solved_vars[end] ∈ 𝓂.➕_vars) 
                push!(SS_solve_func,:($(𝓂.solved_vars[end]) = max(eps(),$(𝓂.solved_vals[end]))))
            else
                push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
            end

            push!(atoms_in_equations_list,[])
        else

            push!(𝓂.solved_vars,Symbol(var_to_solve_for))
            push!(𝓂.solved_vals,Meta.parse(string(soll[1])))
            
            # atoms = reduce(union,soll[1].atoms())

            [push!(atoms_in_equations, Symbol(a)) for a in soll[1].atoms()]
            push!(atoms_in_equations_list, Set(union(setdiff(get_symbols(parsed_eq_to_solve_for), get_symbols(minmax_fixed_eqs)),Symbol.(soll[1].atoms()))))

            # println(atoms_in_equations)
            # push!(atoms_in_equations, soll[1].atoms())

            if (𝓂.solved_vars[end] ∈ 𝓂.➕_vars) 
                push!(SS_solve_func,:($(𝓂.solved_vars[end]) = min(max($(𝓂.lower_bounds[indexin([𝓂.solved_vars[end]],𝓂.bounded_vars)][1]),$(𝓂.solved_vals[end])),$(𝓂.upper_bounds[indexin([𝓂.solved_vars[end]],𝓂.bounded_vars)][1]))))
            else
                push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
            end
        end
        n -= 1
    else 
        break
    end
end


# multi var solve

        # push!(single_eqs,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
        # solve symbolically
    # else
        vars_to_solve = collect(unknowns)[vars[:,vars[2,:] .== n][1,:]]

        eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

        numerical_sol = false
        
        # write some code that tries solving first all equations for all variables and then solves for one variable less and one equation less. note that the subset of equations and variables to solve for must satisfy the incidence_matrix_subset. that is solving the subset of equations for the subset of variables must be possible. if not, then solve numerically. if it is possible, then solve symbolically and then check if the solution is valid for the other equations. 
        incidence_matrix_subset = incidence_matrix[vars[:,vars[2,:] .== n][1,:], eqs[:,eqs[2,:] .== n][1,:]]

        using Combinatorics

        # Assuming eqs_to_solve is already defined
        all_combinations = []
        for n in length(eqs_to_solve):-1:2
            push!(all_combinations, combinations(1:length(eqs_to_solve), n))
        end
        combos = all_combinations[2]|>collect

        indices_to_select_from = findall([sum(incidence_matrix_subset[:,combos[1]],dims = 2)...] .> 0)
        indices_that_need_to_be_there = findall([sum(incidence_matrix_subset[:,setdiff(1:length(eqs_to_solve),combos[1])],dims = 2)...] .> 0)

        selected_indices = combinations(indices_to_select_from, 4)
        not_picked_indices = [setdiff(indices_that_need_to_be_there, i) for i in selected_indices]

        


        
        all_combinations = []
        for n in length(eqs_to_solve)-1:-1:4
            for eq_combo in combinations(1:length(eqs_to_solve), n)
                indices_to_select_from = findall([sum(incidence_matrix_subset[:,eq_combo],dims = 2)...] .> 0)

                indices_that_need_to_be_there = findall([sum(incidence_matrix_subset[:,setdiff(1:length(eqs_to_solve),eq_combo)],dims = 2)...] .> 0) 

                for var_combo in combinations(indices_to_select_from, n)

                    unmatched_vars = setdiff(indices_that_need_to_be_there, var_combo)

                    if length(unmatched_vars) <= length(eqs_to_solve) - n
                        
                        println("Solving for: ", [var_combo], " in: ", [eq_combo])

                        if length(eqs_to_solve) - n < n
                            soll_rest = try SPyPyC.solve(eqs_to_solve[setdiff(1:length(eqs_to_solve),eq_combo)], vars_to_solve[setdiff(1:length(eqs_to_solve),var_combo)])
                            catch
                            end

                            if isnothing(soll_rest) || length(soll_rest) == 0 break end

                            soll = try SPyPyC.solve(eqs_to_solve[eq_combo], vars_to_solve[var_combo])
                            catch
                            end
                            
                            if isnothing(soll) || length(soll) == 0 break end
                        else
                            soll = try SPyPyC.solve(eqs_to_solve[eq_combo], vars_to_solve[var_combo])
                            catch
                            end

                            if isnothing(soll) || length(soll) == 0 break end

                            soll_rest = try SPyPyC.solve(eqs_to_solve[setdiff(1:length(eqs_to_solve),eq_combo)], vars_to_solve[setdiff(1:length(eqs_to_solve),var_combo)])
                            catch
                            end

                            if isnothing(soll_rest) || length(soll_rest) == 0 break end
                        end

                        push!(all_combinations, (vars_to_solve[setdiff(1:length(eqs_to_solve),var_combo)],
                                                vcat(vars_to_solve[var_combo], vars_to_solve[setdiff(1:length(eqs_to_solve),var_combo)]), 
                                                vcat(collect(soll[1])..., collect(soll_rest[1])...)))

                        if length(all_combinations) > 0
                            break
                        end
                    end
                end
                
                if length(all_combinations) > 0
                    break
                end
            end
        end

        all_combinations[1][1]
        all_combinations[1][2]
        all_combinations[1][3]
        all_combinations[1][3][1]|>collect
        all_combinations[1][4][1]|>collect

        using BenchmarkTools

        @benchmark soll = try SPyPyC.solve(eqs_to_solve[[1, 2, 3, 4]], vars_to_solve[[1, 2, 3, 4]])
        catch
        end


        @benchmark soll_rest = try SPyPyC.solve(eqs_to_solve[setdiff(1:length(eqs_to_solve),[1, 2, 3, 4])], vars_to_solve[setdiff(1:length(eqs_to_solve),[1, 2, 3, 4])])
        catch
        end


        soll = try SPyPyC.solve(ss_equations, unknowns)
        catch
        end

#         if isnothing(soll)
#             if verbose
#                 println("Failed finding solution symbolically for: ", unknowns, " in: ", ss_equations, ". Solving numerically.")
#             end
#             numerical_sol = true
#         else
#             for n = length(unknowns):-1:2
#                 vars_to_solve = unknowns[1:n]
#                 eqs_to_solve = ss_equations[1:n]
                
#                 soll = try SPyPyC.solve(eqs_to_solve, vars_to_solve)
#                 catch
#                 end
                
#                 if isnothing(soll) || length(soll) == 0 || length(intersect((union(SPyPyC.free_symbols.(soll[1])...) .|> SPyPyC.:↓), (vars_to_solve .|> SPyPyC.:↓))) > 0
#                     if verbose
#                         println("Failed finding solution symbolically for: ", vars_to_solve, " in: ", eqs_to_solve, ". Solving numerically.")
#                     end
#                     numerical_sol = true
#                 else
#                     # Code for successful symbolic solution
#                 end
#             end
#         end
        














#         # if symbolic_SS
#             soll = try SPyPyC.solve(eqs_to_solve,vars_to_solve)
#             # soll = try solve(SPyPyC.Sym(eqs_to_solve),var_order)#,check=false,force = true,manual=true)
#             catch
#             end

#             if isnothing(soll)
#                 if verbose
#                     println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
#                 end
#                 numerical_sol = true
#                 # continue
#             elseif length(soll) == 0
#                 if verbose
#                     println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
#                 end
#                 numerical_sol = true
#                 # continue
#             # elseif length(intersect(vars_to_solve,reduce(union,map(x->x.atoms(),collect(soll[1]))))) > 0
#             elseif length(intersect((union(SPyPyC.free_symbols.(soll[1])...) .|> SPyPyC.:↓),(vars_to_solve .|> SPyPyC.:↓))) > 0
#             # elseif length(intersect(union(SPyPyC.free_symbols.(soll[1])...),vars_to_solve)) > 0
#                 if verbose
#                     println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
#                 end
#                 numerical_sol = true
#                 # println("Could not solve for: ",intersect(var_list,reduce(union,map(x->x.atoms(),solll)))...)
#                 # break_ind = true
#                 # break
#             else
#                 if verbose
#                     println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " symbolically.")
#                 end
#                 # relevant_pars = reduce(union,vcat(𝓂.par_list,𝓂.par_calib_list)[eqs[:,eqs[2,:] .== n][1,:]])
#                 # relevant_pars = reduce(union,map(x->x.atoms(),collect(soll[1])))
#                 atoms = reduce(union,map(x->x.atoms(),collect(soll[1])))
#                 # println(atoms)
#                 [push!(atoms_in_equations, Symbol(a)) for a in atoms]
                
#                 for (k, vars) in enumerate(vars_to_solve)
#                     push!(𝓂.solved_vars,Symbol(vars))
#                     push!(𝓂.solved_vals,Meta.parse(string(soll[1][k]))) #using convert(Expr,x) leads to ugly expressions

#                     push!(atoms_in_equations_list, Set(Symbol.(soll[1][k].atoms())))
#                     # push!(atoms_in_equations_list, Set(Symbol.(soll[vars].atoms()))) # to be fixed
#                     push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
#                 end
#             end


#         end
            
#         # try symbolically and use numerical if it does not work
#         if numerical_sol || !symbolic_SS
#             if !symbolic_SS && verbose
#                 println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " numerically.")
#             end
            
#             push!(𝓂.solved_vars,Symbol.(vars_to_solve))
#             push!(𝓂.solved_vals,Meta.parse.(string.(eqs_to_solve)))

#             syms_in_eqs = Set()

#             for i in eqs_to_solve
#                 # push!(syms_in_eqs, Symbol.(PythonCall.pystr.(i.atoms()))...)
#                 push!(syms_in_eqs, Symbol.(SPyPyC.:↓(SPyPyC.free_symbols(i)))...)
#             end

#             # println(syms_in_eqs)
#             push!(atoms_in_equations_list,setdiff(syms_in_eqs, 𝓂.solved_vars[end]))

#             calib_pars = []
#             calib_pars_input = []
#             relevant_pars = reduce(union,vcat(𝓂.par_list_aux_SS,𝓂.par_calib_list)[eqs[:,eqs[2,:] .== n][1,:]])
#             relevant_pars_across = union(relevant_pars_across,relevant_pars)
            
#             iii = 1
#             for parss in union(𝓂.parameters,𝓂.parameters_as_function_of_parameters)
#                 # valss   = 𝓂.parameter_values[i]
#                 if :($parss) ∈ relevant_pars
#                     push!(calib_pars,:($parss = parameters_and_solved_vars[$iii]))
#                     push!(calib_pars_input,:($parss))
#                     iii += 1
#                 end
#             end


#             guess = []
#             result = []
#             sorted_vars = sort(𝓂.solved_vars[end])
#             # sorted_vars = sort(setdiff(𝓂.solved_vars[end],𝓂.➕_vars))
#             for (i, parss) in enumerate(sorted_vars) 
#                 push!(guess,:($parss = guess[$i]))
#                 # push!(guess,:($parss = undo_transformer(guess[$i])))
#                 push!(result,:($parss = sol[$i]))
#             end

            
#             # separate out auxilliary variables (nonnegativity)
#             nnaux = []
#             nnaux_linear = []
#             nnaux_error = []
#             push!(nnaux_error, :(aux_error = 0))
#             solved_vals = []
            
#             eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]


#             other_vrs_eliminated_by_sympy = Set()

#             for (i,val) in enumerate(𝓂.solved_vals[end])
#                 if typeof(val) ∈ [Symbol,Float64,Int]
#                     push!(solved_vals,val)
#                 else
#                     if eq_idx_in_block_to_solve[i] ∈ 𝓂.ss_equations_with_aux_variables
#                         val = vcat(𝓂.ss_aux_equations,𝓂.calibration_equations)[eq_idx_in_block_to_solve[i]]
#                         push!(nnaux,:($(val.args[2]) = max(eps(),$(val.args[3]))))
#                         push!(other_vrs_eliminated_by_sympy, val.args[2])
#                         push!(nnaux_linear,:($val))
#                         push!(nnaux_error, :(aux_error += min(eps(),$(val.args[3]))))
#                     else
#                         push!(solved_vals,postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))
#                     end
#                 end
#             end

#             # println(other_vrs_eliminated_by_sympy)
#             # sort nnaux vars so that they enter in right order. avoid using a variable before it is declared
#             # println(nnaux)
#             if length(nnaux) > 1
#                 all_symbols = map(x->x.args[1],nnaux) #relevant symbols come first in respective equations

#                 nn_symbols = map(x->intersect(all_symbols,x), get_symbols.(nnaux))
                
#                 inc_matrix = fill(0,length(all_symbols),length(all_symbols))

#                 for i in 1:length(all_symbols)
#                     for k in 1:length(nn_symbols)
#                         inc_matrix[i,k] = collect(all_symbols)[i] ∈ collect(nn_symbols)[k]
#                     end
#                 end

#                 QQ, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(inc_matrix))

#                 nnaux = nnaux[QQ]
#                 nnaux_linear = nnaux_linear[QQ]
#             end



#             other_vars = []
#             other_vars_input = []
#             # other_vars_inverse = []
#             other_vrs = intersect( setdiff( union(𝓂.var, 𝓂.calibration_equations_parameters, 𝓂.➕_vars),
#                                                 sort(𝓂.solved_vars[end]) ),
#                                     union(syms_in_eqs, other_vrs_eliminated_by_sympy, setdiff(reduce(union, get_symbols.(nnaux), init = []), map(x->x.args[1],nnaux)) ) )

#             # println(intersect( setdiff( union(𝓂.var, 𝓂.calibration_equations_parameters, 𝓂.➕_vars), sort(𝓂.solved_vars[end]) ), union(syms_in_eqs, other_vrs_eliminated_by_sympy ) ))
#             # println(other_vrs)
#             for var in other_vrs
#                 # var_idx = findfirst(x -> x == var, union(𝓂.var,𝓂.calibration_equations_parameters))
#                 push!(other_vars,:($(var) = parameters_and_solved_vars[$iii]))
#                 push!(other_vars_input,:($(var)))
#                 iii += 1
#                 # push!(other_vars_inverse,:(𝓂.SS_init_guess[$var_idx] = $(var)))
#             end

#             # augment system for bound constraint violations
#             # aug_lag = []
#             # aug_lag_penalty = []
#             # push!(aug_lag_penalty, :(bound_violation_penalty = 0))

#             # for varpar in intersect(𝓂.bounded_vars,union(other_vrs,sorted_vars,relevant_pars))
#             #     i = indexin([varpar],𝓂.bounded_vars)
#             #     push!(aug_lag,:($varpar = min(max($varpar,$(𝓂.lower_bounds[i...])),$(𝓂.upper_bounds[i...]))))
#             #     push!(aug_lag_penalty,:(bound_violation_penalty += max(0,$(𝓂.lower_bounds[i...]) - $varpar) + max(0,$varpar - $(𝓂.upper_bounds[i...]))))
#             # end


#             # add it also to output from optimisation, in case you use optimiser without bounds
#             # aug_lag_results = []

#             # for varpar in intersect(𝓂.bounded_vars,sorted_vars)
#             #     i = indexin([varpar],𝓂.bounded_vars)
#             #     push!(aug_lag_results,:($varpar = min(max($varpar,𝓂.lower_bounds[$i...]),𝓂.upper_bounds[$i...])))
#             # end

#             # funcs_no_transform = :(function block(parameters_and_solved_vars::Vector{Float64}, guess::Vector{Float64})
#             #         # if guess isa Tuple guess = guess[1] end
#             #         # guess = undo_transformer(guess) 
#             #         # println(guess)
#             #         $(guess...) 
#             #         $(calib_pars...) # add those variables which were previously solved and are used in the equations
#             #         $(other_vars...) # take only those that appear in equations - DONE

#             #         # $(aug_lag...)
#             #         # $(nnaux_linear...)
#             #         return [$(solved_vals...),$(nnaux_linear...)]
#             #     end)

# # println(solved_vals)
#             funcs = :(function block(parameters_and_solved_vars::Vector, guess::Vector)
#                     # if guess isa Tuple guess = guess[1] end
#                     # guess = undo_transformer(guess,lbs,ubs, option = transformer_option) 
#                     # println(guess)
#                     $(guess...) 
#                     $(calib_pars...) # add those variables which were previously solved and are used in the equations
#                     $(other_vars...) # take only those that appear in equations - DONE

#                     # $(aug_lag...)
#                     # $(nnaux...)
#                     # $(nnaux_linear...)
#                     return [$(solved_vals...),$(nnaux_linear...)]
#                 end)

#             # push!(solved_vals,:(aux_error))
#             # push!(solved_vals,:(bound_violation_penalty))

#             #funcs_optim = :(function block(guess::Vector{Float64},transformer_parameters_and_solved_vars::Tuple{Vector{Float64},Int})
#                 #guess = undo_transformer(guess,option = transformer_parameters_and_solved_vars[2])
#                 #parameters_and_solved_vars = transformer_parameters_and_solved_vars[1]
#             #  $(guess...) 
#             # $(calib_pars...) # add those variables which were previously solved and are used in the equations
#             #   $(other_vars...) # take only those that appear in equations - DONE

#                 # $(aug_lag_penalty...)
#                 # $(aug_lag...)
#                 # $(nnaux...) # not needed because the aux vars are inputs
#                 # $(nnaux_error...)
#                 #return sum(abs2,[$(solved_vals...),$(nnaux_linear...)])
#             #end)
        
#             push!(NSSS_solver_cache_init_tmp,fill(0.897,length(sorted_vars)))
#             push!(NSSS_solver_cache_init_tmp,[Inf])

#             # WARNING: infinite bounds are transformed to 1e12
#             lbs = []
#             ubs = []
            
#             limit_boundaries = 1e12

#             for i in vcat(sorted_vars, calib_pars_input, other_vars_input)
#                 if i ∈ 𝓂.bounded_vars
#                     push!(lbs,𝓂.lower_bounds[i .== 𝓂.bounded_vars][1] == -Inf ? -limit_boundaries+rand() : 𝓂.lower_bounds[i .== 𝓂.bounded_vars][1])
#                     push!(ubs,𝓂.upper_bounds[i .== 𝓂.bounded_vars][1] ==  Inf ?  limit_boundaries-rand() : 𝓂.upper_bounds[i .== 𝓂.bounded_vars][1])
#                 else
#                     push!(lbs,-limit_boundaries+rand())
#                     push!(ubs,limit_boundaries+rand())
#                 end
#             end

#             push!(SS_solve_func,:(params_and_solved_vars = [$(calib_pars_input...),$(other_vars_input...)]))

#             push!(SS_solve_func,:(lbs = [$(lbs...)]))
#             push!(SS_solve_func,:(ubs = [$(ubs...)]))
            
#             # push!(SS_solve_func,:(𝓂.SS_init_guess = initial_guess))
#             # push!(SS_solve_func,:(f = OptimizationFunction(𝓂.ss_solve_blocks_optim[$(n_block)], Optimization.AutoForwardDiff())))
#             # push!(SS_solve_func,:(inits = max.(lbs,min.(ubs,𝓂.SS_init_guess[$([findfirst(x->x==y,union(𝓂.var,𝓂.calibration_equations_parameters)) for y in sorted_vars])]))))
#             # push!(SS_solve_func,:(closest_solution = 𝓂.NSSS_solver_cache[findmin([sum(abs2,pars[end] - params_flt) for pars in 𝓂.NSSS_solver_cache])[2]]))
#             # push!(SS_solve_func,:(inits = [transformer(max.(lbs,min.(ubs, closest_solution[$(n_block)] ))),closest_solution[end]]))
#             # push!(SS_solve_func,:(inits = max.(lbs[1:length(closest_solution[$(n_block)])], min.(ubs[1:length(closest_solution[$(n_block)])], closest_solution[$(n_block)]))))                    
#             push!(SS_solve_func,:(inits = [max.(lbs[1:length(closest_solution[$(2*(n_block-1)+1)])], min.(ubs[1:length(closest_solution[$(2*(n_block-1)+1)])], closest_solution[$(2*(n_block-1)+1)])), closest_solution[$(2*n_block)]]))

#             # push!(SS_solve_func,:(println([$(calib_pars_input...),$(other_vars_input...)])))

#             if VERSION >= v"1.9"
#                 push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷())))
#             else
#                 push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver())))
#             end

#             push!(SS_solve_func,:(solution = block_solver_AD(params_and_solved_vars,
#                                                                     $(n_block), 
#                                                                     𝓂.ss_solve_blocks[$(n_block)], 
#                                                                     # 𝓂.ss_solve_blocks_no_transform[$(n_block)], 
#                                                                     # f, 
#                                                                     inits,
#                                                                     lbs, 
#                                                                     ubs,
#                                                                     solver_parameters,
#                                                                     # fail_fast_solvers_only = fail_fast_solvers_only,
#                                                                     cold_start,
#                                                                     verbose)))
            
#             # push!(SS_solve_func,:(solution = block_solver_RD([$(calib_pars_input...),$(other_vars_input...)])))#, 
#                     # $(n_block), 
#                     # 𝓂.ss_solve_blocks[$(n_block)], 
#                     # # 𝓂.SS_optimizer, 
#                     # f, 
#                     # inits,
#                     # lbs, 
#                     # ubs,
#                     # fail_fast_solvers_only = fail_fast_solvers_only,
#                     # verbose = verbose)))
#             push!(SS_solve_func,:(iters += solution[2][2])) 
#             push!(SS_solve_func,:(solution_error += solution[2][1])) 
#             push!(SS_solve_func,:(sol = solution[1]))
#             # push!(SS_solve_func,:(solution_error += sum(abs2,𝓂.ss_solve_blocks[$(n_block)]([$(calib_pars_input...),$(other_vars_input...)],solution))))
#             # push!(SS_solve_func,:(sol = solution))

#             # push!(SS_solve_func,:(println(sol))) 

#             push!(SS_solve_func,:($(result...)))   
#             # push!(SS_solve_func,:($(aug_lag_results...))) 

#             # push!(SS_solve_func,:(NSSS_solver_cache_tmp = []))
#             # push!(SS_solve_func,:(push!(NSSS_solver_cache_tmp, typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol))))
#             push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol)]))
#             push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(params_and_solved_vars) == Vector{Float64} ? params_and_solved_vars : ℱ.value.(params_and_solved_vars)]))

#             push!(𝓂.ss_solve_blocks,@RuntimeGeneratedFunction(funcs))
#             # push!(𝓂.ss_solve_blocks_no_transform,@RuntimeGeneratedFunction(funcs_no_transform))
#             # push!(𝓂.ss_solve_blocks_optim,@RuntimeGeneratedFunction(funcs_optim))
            
#             n_block += 1
#         end
#     end
#     n -= 1
# end