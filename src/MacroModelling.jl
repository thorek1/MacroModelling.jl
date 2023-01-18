module MacroModelling


import DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF, TYPEDSIGNATURES, TYPEDFIELDS
using StatsFuns, SpecialFunctions
import SymPy: @vars, solve, subs, free_symbols
import SymPy
import ForwardDiff as ℱ 
import SparseArrays: SparseMatrixCSC, sparse, spzeros, droptol!, sparsevec, spdiagm, findnz#, sparse!
import LinearAlgebra as ℒ
using Optimization, OptimizationNLopt, NLboxsolve
import BlockTriangularForm
import Subscripts: super, sub
import IterativeSolvers as ℐ
import DataStructures: CircularBuffer
using LinearMaps
using ComponentArrays
using ImplicitDifferentiation
# using NamedArrays
using AxisKeys
using ChainRulesCore

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

# Type definitions
Symbol_input = Union{Symbol,Vector{Symbol},Matrix{Symbol},Tuple{Symbol,Vararg{Symbol}}}

# Imports
include("common_docstrings.jl")
include("structures.jl")
include("macros.jl")
include("get_functions.jl")
include("plotting.jl")


export @model, @parameters, solve!
export plot_irfs, plot_irf, plot_IRF, plot, plot_simulations, plot_solution
export plot_conditional_variance_decomposition, plot_forecast_error_variance_decomposition, plot_fevd
export get_irfs, get_irf, get_IRF, simulate
export get_solution, get_first_order_solution, get_perturbation_solution
export get_steady_state, get_SS, get_non_stochastic_steady_state, get_stochastic_steady_state
export get_moments, get_covariance, get_standard_deviation, get_variance, get_var, get_std, get_cov, var, std, cov
export get_autocorrelation, get_correlation, get_variance_decomposition, get_corr, get_autocorr, get_var_decomp, corr, autocorr
export get_fevd, fevd, get_forecast_error_variance_decomposition, get_conditional_variance_decomposition
export calculate_jacobian, calculate_hessian, calculate_third_order_derivatives
export calculate_first_order_solution, calculate_second_order_solution, calculate_third_order_solution#, calculate_jacobian_manual, calculate_jacobian_sparse, calculate_jacobian_threaded
export calculate_kalman_filter_loglikelihood


# Internal
export irf, girf, block_solver, block_solver_AD
# export riccati_forward, block_solver, remove_redundant_SS_vars!, write_parameters_input!


# StatsFuns
norminvcdf(p::Number) = -erfcinv(2*p) * sqrt2
norminv(p::Number) = norminvcdf(p)
pnorm(p::Number) = normcdf(p)
dnorm(p::Number) = normpdf(p)
qnorm(p::Number) = norminvcdf(p)





Base.show(io::IO, 𝓂::ℳ) = println(io, 
                "Model:      ", 𝓂.model_name, 
                "\nVariables", 
                "\n Total:     ", 𝓂.timings.nVars - length(𝓂.exo_present) - length(𝓂.aux),
                "\n States:    ", length(setdiff(𝓂.timings.past_not_future_and_mixed,𝓂.aux_present)),
                "\n Jumpers:   ", length(setdiff(setdiff(𝓂.timings.future_not_past_and_mixed,𝓂.aux_present,𝓂.timings.mixed),𝓂.aux_future)),
                "\n Auxiliary: ",length(𝓂.exo_present) + length(𝓂.aux),
                "\nShocks:     ", 𝓂.timings.nExo,
                "\nParameters: ", length(𝓂.par),
                if 𝓂.calibration_equations == Expr[]
                    ""
                else
                    "\nCalibration equations: " * repr(length(𝓂.calibration_equations))
                end,
                # "\n¹: including auxilliary variables"
                # "\nVariable bounds (upper,lower,any): ",sum(𝓂.upper_bounds .< Inf),", ",sum(𝓂.lower_bounds .> -Inf),", ",length(𝓂.bounds),
                # "\nNon-stochastic-steady-state found: ",!𝓂.solution.outdated_NSSS
                )


function get_symbols(ex)
    par = Set()
    postwalk(x ->   
    x isa Expr ? 
        x.head == :(=) ?
            for i in x.args
                i isa Symbol ? 
                    push!(par,i) :
                x
            end :
        x.head == :call ? 
            for i in 2:length(x.args)
                x.args[i] isa Symbol ? 
                    push!(par,x.args[i]) : 
                x
            end : 
        x : 
    x, ex)
    return par
end


function match_pattern(strings::Union{Set,Vector}, pattern::Regex)
    return filter(r -> match(pattern, string(r)) != nothing, strings)
end



# function get_symbols(ex)
#     list = Set()
#     postwalk(x -> x isa Symbol ? push!(list, x) : x, ex)
#     return list
# end

function create_symbols_eqs!(𝓂::ℳ)
    # create symbols in module scope
    symbols_in_equation = union(𝓂.var,𝓂.par,𝓂.parameters,𝓂.parameters_as_function_of_parameters,𝓂.exo,𝓂.dynamic_variables,𝓂.nonnegativity_auxilliary_vars)#,𝓂.dynamic_variables_future)
    l_bnds = Dict(𝓂.bounded_vars .=> 𝓂.lower_bounds)
    u_bnds = Dict(𝓂.bounded_vars .=> 𝓂.upper_bounds)

    symbols_pos = []
    symbols_neg = []
    symbols_none = []

    for symb in symbols_in_equation
        if symb in 𝓂.bounded_vars
            if l_bnds[symb] >= 0
                push!(symbols_pos, symb)
            elseif u_bnds[symb] <= 0
                push!(symbols_neg, symb)
            end
        else
            push!(symbols_none, symb)
        end
    end

    expr =  quote
                @vars $(symbols_pos...)  real = true finite = true positive = true
                @vars $(symbols_neg...)  real = true finite = true negative = true 
                @vars $(symbols_none...) real = true finite = true 
            end

    eval(expr)

    symbolics(map(x->eval(:($x)),𝓂.ss_aux_equations),
                map(x->eval(:($x)),𝓂.dyn_equations),
                # map(x->eval(:($x)),𝓂.dyn_equations_future),

                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift_var_present_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift_var_past_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift_var_future_list),

                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift2_var_past_list),

                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_var_present_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_var_past_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_var_future_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_ss_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_exo_list),

                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_exo_future_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_exo_present_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_exo_past_list),

                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_future_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_present_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_past_list),

                map(x->Set(eval(:([$(x...)]))),𝓂.var_present_list_aux_SS),
                map(x->Set(eval(:([$(x...)]))),𝓂.var_past_list_aux_SS),
                map(x->Set(eval(:([$(x...)]))),𝓂.var_future_list_aux_SS),
                map(x->Set(eval(:([$(x...)]))),𝓂.ss_list_aux_SS),

                map(x->Set(eval(:([$(x...)]))),𝓂.var_list_aux_SS),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dynamic_variables_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dynamic_variables_future_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.par_list_aux_SS),

                map(x->eval(:($x)),𝓂.calibration_equations),
                map(x->eval(:($x)),𝓂.calibration_equations_parameters),
                # map(x->eval(:($x)),𝓂.parameters),

                # Set(eval(:([$(𝓂.var_present...)]))),
                # Set(eval(:([$(𝓂.var_past...)]))),
                # Set(eval(:([$(𝓂.var_future...)]))),
                Set(eval(:([$(𝓂.var...)]))),
                Set(eval(:([$(𝓂.nonnegativity_auxilliary_vars...)]))),

                map(x->Set(eval(:([$(x...)]))),𝓂.ss_calib_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.par_calib_list),

                [Set() for _ in 1:length(𝓂.ss_aux_equations)],
                # [Set() for _ in 1:length(𝓂.calibration_equations)],
                # [Set() for _ in 1:length(𝓂.ss_aux_equations)],
                # [Set() for _ in 1:length(𝓂.calibration_equations)]
                )
end



function remove_redundant_SS_vars!(𝓂::ℳ, symbolics::symbolics)
    ss_equations = symbolics.ss_equations

    # check variables which appear in two time periods. they might be redundant in steady state
    redundant_vars = intersect.(
        union.(
            intersect.(symbolics.var_future_list,symbolics.var_present_list),
            intersect.(symbolics.var_future_list,symbolics.var_past_list),
            intersect.(symbolics.var_present_list,symbolics.var_past_list),
            intersect.(symbolics.ss_list,symbolics.var_present_list),
            intersect.(symbolics.ss_list,symbolics.var_past_list),
            intersect.(symbolics.ss_list,symbolics.var_future_list)
        ),
    symbolics.var_list)
    redundant_idx = getindex(1:length(redundant_vars), (length.(redundant_vars) .> 0) .& (length.(symbolics.var_list) .> 1))

    for i in redundant_idx
        for var_to_solve in redundant_vars[i]
            soll = try solve(ss_equations[i],var_to_solve)
            catch
            end
            
            if isnothing(soll)
                continue
            end
            
            if length(soll) == 0 || soll == SymPy.Sym[0] # take out variable if it is redundant from that euation only
                push!(symbolics.var_redundant_list[i],var_to_solve)
                ss_equations[i] = ss_equations[i].subs(var_to_solve,1)
            end

        end
    end

end




function solve_steady_state!(𝓂::ℳ, symbolic_SS, symbolics::symbolics; verbose = false)
    unknowns = union(symbolics.var,symbolics.nonnegativity_auxilliary_vars,symbolics.calibration_equations_parameters)

    @assert length(unknowns) <= length(symbolics.ss_equations) + length(symbolics.calibration_equations) "Unable to solve steady state. More unknowns than equations."

    incidence_matrix = fill(0,length(unknowns),length(unknowns))

    eq_list = union(union.(setdiff.(union.(symbolics.var_list,
                                           symbolics.ss_list),
                                    symbolics.var_redundant_list),
                            symbolics.par_list),
                    union.(symbolics.ss_calib_list,
                            symbolics.par_calib_list))


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

    n = n_blocks

    ss_equations = vcat(symbolics.ss_equations,symbolics.calibration_equations) .|> SymPy.Sym
    # println(ss_equations)

    SS_solve_func = []

    atoms_in_equations = Set()
    atoms_in_equations_list = []
    relevant_pars_across = []
    NSSS_solver_cache_init_tmp = []

    n_block = 1

    while n > 0 
        if length(eqs[:,eqs[2,:] .== n]) == 2
            var_to_solve = collect(unknowns)[vars[:,vars[2,:] .== n][1]]

            soll = try solve(ss_equations[eqs[:,eqs[2,:] .== n][1]],var_to_solve)
            catch
            end

            if isnothing(soll)
                # println("Could not solve single variables case symbolically.")
                println("Failed finding solution symbolically for: ",var_to_solve," in: ",ss_equations[eqs[:,eqs[2,:] .== n][1]])
                # solve numerically
                continue
            elseif soll[1].is_number
                # ss_equations = ss_equations.subs(var_to_solve,soll[1])
                ss_equations = [eq.subs(var_to_solve,soll[1]) for eq in ss_equations]
                
                push!(𝓂.solved_vars,Symbol(var_to_solve))
                push!(𝓂.solved_vals,Meta.parse(string(soll[1])))

                if (𝓂.solved_vars[end] ∈ 𝓂.nonnegativity_auxilliary_vars) 
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = max(eps(),$(𝓂.solved_vals[end]))))
                else
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
                end

                push!(atoms_in_equations_list,[])
            else

                push!(𝓂.solved_vars,Symbol(var_to_solve))
                push!(𝓂.solved_vals,Meta.parse(string(soll[1])))
                
                # atoms = reduce(union,soll[1].atoms())
                [push!(atoms_in_equations, a) for a in soll[1].atoms()]
                push!(atoms_in_equations_list, Set(Symbol.(soll[1].atoms())))
                # println(atoms_in_equations)
                # push!(atoms_in_equations, soll[1].atoms())

                if (𝓂.solved_vars[end] ∈ 𝓂.nonnegativity_auxilliary_vars) 
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = max(eps(),$(𝓂.solved_vals[end]))))
                else
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
                end
            end

            # push!(single_eqs,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
            # solve symbolically
        else

            vars_to_solve = collect(unknowns)[vars[:,vars[2,:] .== n][1,:]]

            eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

            numerical_sol = false
            
            if symbolic_SS
                soll = try solve(SymPy.Sym(eqs_to_solve),vars_to_solve)
                # soll = try solve(Sym(eqs_to_solve),var_order)#,check=false,force = true,manual=true)
                catch
                end

                # println(soll)
                if isnothing(soll)
                    if verbose
                        println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
                    end
                    numerical_sol = true
                    # continue
                elseif length(soll) == 0
                    if verbose
                        println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
                    end
                    numerical_sol = true
                    # continue
                elseif length(intersect(vars_to_solve,reduce(union,map(x->x.atoms(),collect(soll[1]))))) > 0
                    if verbose
                        println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
                    end
                    numerical_sol = true
                    # println("Could not solve for: ",intersect(var_list,reduce(union,map(x->x.atoms(),solll)))...)
                    # break_ind = true
                    # break
                else
                    if verbose
                        println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " symbolically.")
                    end
                    # relevant_pars = reduce(union,vcat(𝓂.par_list,𝓂.par_calib_list)[eqs[:,eqs[2,:] .== n][1,:]])
                    # relevant_pars = reduce(union,map(x->x.atoms(),collect(soll[1])))
                    atoms = reduce(union,map(x->x.atoms(),collect(soll[1])))
                    # println(atoms)
                    [push!(atoms_in_equations, a) for a in atoms]
                    
                    for (k, vars) in enumerate(vars_to_solve)
                        push!(𝓂.solved_vars,Symbol(vars))
                        push!(𝓂.solved_vals,Meta.parse(string(soll[1][k]))) #using convert(Expr,x) leads to ugly expressions

                        push!(atoms_in_equations_list, Set(Symbol.(soll[1][k].atoms())))
                        push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
                    end
                end


            end
                
            # try symbolically and use numerical if it does not work
            if numerical_sol || !symbolic_SS
                if !symbolic_SS && verbose
                    println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " numerically.")
                end
                push!(𝓂.solved_vars,Symbol.(collect(unknowns)[vars[:,vars[2,:] .== n][1,:]]))
                push!(𝓂.solved_vals,Meta.parse.(string.(ss_equations[eqs[:,eqs[2,:] .== n][1,:]])))
                
                syms_in_eqs = Set(Symbol.(SymPy.Sym(ss_equations[eqs[:,eqs[2,:] .== n][1,:]]).atoms()))
                push!(atoms_in_equations_list,setdiff(syms_in_eqs, 𝓂.solved_vars[end]))

                calib_pars = []
                calib_pars_input = []
                relevant_pars = reduce(union,vcat(𝓂.par_list_aux_SS,𝓂.par_calib_list)[eqs[:,eqs[2,:] .== n][1,:]])
                relevant_pars_across = union(relevant_pars_across,relevant_pars)
                
                iii = 1
                for parss in union(𝓂.parameters,𝓂.parameters_as_function_of_parameters)
                    # valss   = 𝓂.parameter_values[i]
                    if :($parss) ∈ relevant_pars
                        push!(calib_pars,:($parss = parameters_and_solved_vars[$iii]))
                        push!(calib_pars_input,:($parss))
                        iii += 1
                    end
                end


                guess = []
                result = []
                sorted_vars = sort(setdiff(𝓂.solved_vars[end],𝓂.nonnegativity_auxilliary_vars))
                for (i, parss) in enumerate(sorted_vars) 
                    push!(guess,:($parss = guess[$i]))
                    # push!(guess,:($parss = undo_transformer(guess[$i])))
                    push!(result,:($parss = sol[$i]))
                end

                other_vars = []
                other_vars_input = []
                # other_vars_inverse = []
                other_vrs = intersect(setdiff(union(𝓂.var,𝓂.calibration_equations_parameters),sort(𝓂.solved_vars[end])),syms_in_eqs)
                
                for var in other_vrs
                    var_idx = findfirst(x -> x == var, union(𝓂.var,𝓂.calibration_equations_parameters))
                    push!(other_vars,:($(var) = parameters_and_solved_vars[$iii]))
                    push!(other_vars_input,:($(var)))
                    iii += 1
                    # push!(other_vars_inverse,:(𝓂.SS_init_guess[$var_idx] = $(var)))
                end
                
                # separate out auxilliary variables (nonnegativity)
                nnaux = []
                nnaux_linear = []
                nnaux_error = []
                push!(nnaux_error, :(aux_error = 0))
                solved_vals = []
                
                for val in 𝓂.solved_vals[end]
                    if (val.args[1] == :+ && val.args[3] ∈ 𝓂.nonnegativity_auxilliary_vars) 
                        push!(nnaux,:($(val.args[3]) = max(eps(),-$(val.args[2]))))
                        push!(nnaux_linear,:($(val.args[3]) = -$(val.args[2])))
                        push!(nnaux_error, :(aux_error += min(0.0,-$(val.args[2]))))
                    elseif (val.args[1] == :- && val.args[2] ∈ 𝓂.nonnegativity_auxilliary_vars) 
                        push!(nnaux,:($(val.args[2]) = max(eps(),$(val.args[3]))))
                        push!(nnaux_linear,:($(val.args[2]) = $(val.args[3])))
                        push!(nnaux_error, :(aux_error += min(0.0,$(val.args[3]))))
                    else
                        push!(solved_vals,postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))
                    end
                end

                # sort nnaux vars so that they enter in right order. avoid using a variable before it is declared
                if length(nnaux) > 1

                    nn_symbols = map(x->intersect(𝓂.nonnegativity_auxilliary_vars,x), get_symbols.(nnaux))

                    all_symbols = reduce(vcat,nn_symbols) |> Set

                    inc_matrix = fill(0,length(all_symbols),length(all_symbols))


                    for i in 1:length(all_symbols)
                        for k in 1:length(all_symbols)
                            inc_matrix[i,k] = collect(all_symbols)[i] ∈ collect(nn_symbols)[k]
                        end
                    end

                    QQ, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(inc_matrix))

                    nnaux = nnaux[QQ]
                    nnaux_linear = nnaux_linear[QQ]
                end

                # augment system for bound constraint violations
                # aug_lag = []
                # aug_lag_penalty = []
                # push!(aug_lag_penalty, :(bound_violation_penalty = 0))

                # for varpar in intersect(𝓂.bounded_vars,union(other_vrs,sorted_vars,relevant_pars))
                #     i = indexin([varpar],𝓂.bounded_vars)
                #     push!(aug_lag,:($varpar = min(max($varpar,$(𝓂.lower_bounds[i...])),$(𝓂.upper_bounds[i...]))))
                #     push!(aug_lag_penalty,:(bound_violation_penalty += max(0,$(𝓂.lower_bounds[i...]) - $varpar) + max(0,$varpar - $(𝓂.upper_bounds[i...]))))
                # end


                # add it also to output from optimisation, in case you use optimiser without bounds
                # aug_lag_results = []

                # for varpar in intersect(𝓂.bounded_vars,sorted_vars)
                #     i = indexin([varpar],𝓂.bounded_vars)
                #     push!(aug_lag_results,:($varpar = min(max($varpar,𝓂.lower_bounds[$i...]),𝓂.upper_bounds[$i...])))
                # end

                funcs_no_transform = :(function block(parameters_and_solved_vars::Vector{Float64}, guess::Vector{Float64})
                        # if guess isa Tuple guess = guess[1] end
                        # guess = undo_transformer(guess) 
                        # println(guess)
                        $(guess...) 
                        $(calib_pars...) # add those variables which were previously solved and are used in the equations
                        $(other_vars...) # take only those that appear in equations - DONE

                        # $(aug_lag...)
                        $(nnaux_linear...)
                        return [$(solved_vals...)]
                    end)


                funcs = :(function block(parameters_and_solved_vars::Vector{Float64}, guess::Vector{Float64})
                        # if guess isa Tuple guess = guess[1] end
                        guess = undo_transformer.(guess) 
                        # println(guess)
                        $(guess...) 
                        $(calib_pars...) # add those variables which were previously solved and are used in the equations
                        $(other_vars...) # take only those that appear in equations - DONE

                        # $(aug_lag...)
                        $(nnaux_linear...)
                        return [$(solved_vals...)]
                    end)

                push!(solved_vals,:(aux_error))
                # push!(solved_vals,:(bound_violation_penalty))

                funcs_optim = :(function block(guess::Vector{Float64},parameters_and_solved_vars::Vector{Float64})
                    guess = undo_transformer.(guess)
                    $(guess...) 
                    $(calib_pars...) # add those variables which were previously solved and are used in the equations
                    $(other_vars...) # take only those that appear in equations - DONE

                    # $(aug_lag_penalty...)
                    # $(aug_lag...)
                    $(nnaux...)
                    $(nnaux_error...)
                    return sum(abs2,[$(solved_vals...)])
                end)
            
                # 𝓂.SS_init_guess = [fill(.9,length(𝓂.var)); fill(.5, length(𝓂.calibration_equations_parameters))]

                push!(NSSS_solver_cache_init_tmp,fill(.9,length(sorted_vars)))

                # WARNING: infinite bounds are transformed to 1e12
                lbs = []
                ubs = []
                
                limit_boundaries = 1e12

                for i in sorted_vars
                    if i ∈ 𝓂.bounded_vars
                        push!(lbs,𝓂.lower_bounds[i .== 𝓂.bounded_vars][1] == -Inf ? -limit_boundaries : 𝓂.lower_bounds[i .== 𝓂.bounded_vars][1])
                        push!(ubs,𝓂.upper_bounds[i .== 𝓂.bounded_vars][1] ==  Inf ?  limit_boundaries : 𝓂.upper_bounds[i .== 𝓂.bounded_vars][1])
                    else
                        push!(lbs,-limit_boundaries)
                        push!(ubs,limit_boundaries)
                    end
                end
                push!(SS_solve_func,:(lbs = [$(lbs...)]))
                push!(SS_solve_func,:(ubs = [$(ubs...)]))
                # push!(SS_solve_func,:(𝓂.SS_init_guess = initial_guess))
                push!(SS_solve_func,:(f = OptimizationFunction(𝓂.ss_solve_blocks_optim[$(n_block)], Optimization.AutoForwardDiff())))
                # push!(SS_solve_func,:(inits = max.(lbs,min.(ubs,𝓂.SS_init_guess[$([findfirst(x->x==y,union(𝓂.var,𝓂.calibration_equations_parameters)) for y in sorted_vars])]))))
                # push!(SS_solve_func,:(closest_solution = 𝓂.NSSS_solver_cache[findmin([sum(abs2,pars[end] - params_flt) for pars in 𝓂.NSSS_solver_cache])[2]]))
                # push!(SS_solve_func,:(inits = [transformer(max.(lbs,min.(ubs, closest_solution[$(n_block)] ))),closest_solution[end]]))
                push!(SS_solve_func,:(inits = max.(lbs,min.(ubs, closest_solution[$(n_block)]))))
                push!(SS_solve_func,:(block_solver_RD = block_solver_AD([$(calib_pars_input...),$(other_vars_input...)],
                                                                        $(n_block), 
                                                                        𝓂.ss_solve_blocks[$(n_block)], 
                                                                        𝓂.ss_solve_blocks_no_transform[$(n_block)], 
                                                                        f, 
                                                                        inits,
                                                                        lbs, 
                                                                        ubs,
                                                                        fail_fast_solvers_only = fail_fast_solvers_only,
                                                                        verbose = verbose)))
                
                push!(SS_solve_func,:(solution = block_solver_RD([$(calib_pars_input...),$(other_vars_input...)])))#, 
                        # $(n_block), 
                        # 𝓂.ss_solve_blocks[$(n_block)], 
                        # # 𝓂.SS_optimizer, 
                        # f, 
                        # inits,
                        # lbs, 
                        # ubs,
                        # fail_fast_solvers_only = fail_fast_solvers_only,
                        # verbose = verbose)))
                # push!(SS_solve_func,:(solution_error += solution[2])) 
                # push!(SS_solve_func,:(sol = solution[1]))
                push!(SS_solve_func,:(solution_error += sum(abs2,𝓂.ss_solve_blocks_no_transform[$(n_block)]([$(calib_pars_input...),$(other_vars_input...)],solution))))
                push!(SS_solve_func,:(sol = solution))

                # push!(SS_solve_func,:(println(sol))) 

                push!(SS_solve_func,:($(result...)))   
                # push!(SS_solve_func,:($(aug_lag_results...))) 

                # push!(SS_solve_func,:(NSSS_solver_cache_tmp = []))
                # push!(SS_solve_func,:(push!(NSSS_solver_cache_tmp, typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol))))
                push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol)]))

                push!(𝓂.ss_solve_blocks,@RuntimeGeneratedFunction(funcs))
                push!(𝓂.ss_solve_blocks_no_transform,@RuntimeGeneratedFunction(funcs_no_transform))
                push!(𝓂.ss_solve_blocks_optim,@RuntimeGeneratedFunction(funcs_optim))
                
                n_block += 1
            end
        end
        n -= 1
    end

    push!(NSSS_solver_cache_init_tmp,fill(Inf,length(𝓂.parameters)))
    push!(𝓂.NSSS_solver_cache,NSSS_solver_cache_init_tmp)

    unknwns = Symbol.(collect(unknowns))

    parameters_only_in_par_defs = Set()
    # add parameters from parameter definitions
    if length(𝓂.calibration_equations_no_var) > 0
		atoms = reduce(union,get_symbols.(𝓂.calibration_equations_no_var))
	    [push!(atoms_in_equations, a) for a in atoms]
	    [push!(parameters_only_in_par_defs, a) for a in atoms]
	end
    
    𝓂.par = union(𝓂.par,setdiff(parameters_only_in_par_defs,𝓂.parameters_as_function_of_parameters))
    
    parameters_in_equations = []

    for (i, parss) in enumerate(𝓂.parameters) 
        if parss ∈ union(Symbol.(atoms_in_equations),relevant_pars_across)
            push!(parameters_in_equations,:($parss = params[$i]))
        end
    end
    
    dependencies = []
    for (i, a) in enumerate(atoms_in_equations_list)
        push!(dependencies,𝓂.solved_vars[i] => intersect(a, union(𝓂.var,𝓂.parameters)))
    end

    push!(dependencies,:SS_relevant_calibration_parameters => intersect(reduce(union,atoms_in_equations_list),𝓂.parameters))

    𝓂.SS_dependencies = dependencies
    

    
    dyn_exos = []
    for dex in union(𝓂.exo_past,𝓂.exo_future)
        push!(dyn_exos,:($dex = 0))
    end

    push!(SS_solve_func,:($(dyn_exos...)))
    
    # push!(SS_solve_func,:(push!(NSSS_solver_cache_tmp, params_scaled_flt)))
    push!(SS_solve_func,:(if length(NSSS_solver_cache_tmp) == 0 NSSS_solver_cache_tmp = [params_scaled_flt] else NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp...,params_scaled_flt] end))
    
    push!(SS_solve_func,:(current_best = sum(abs2,𝓂.NSSS_solver_cache[end][end] - params_flt)))

    push!(SS_solve_func,:(for pars in 𝓂.NSSS_solver_cache
                                latest = sum(abs2,pars[end] - params_flt)
                                if latest <= current_best
                                    current_best = latest
                                end
                            end))

    push!(SS_solve_func,:(if (current_best > eps(Float32)) && (solution_error < eps(Float64)) 
                                # println(NSSS_solver_cache_tmp)
                                push!(𝓂.NSSS_solver_cache, NSSS_solver_cache_tmp) 
                                solved_scale = scale
                            end))
    # push!(SS_solve_func,:(if length(𝓂.NSSS_solver_cache) > 100 popfirst!(𝓂.NSSS_solver_cache) end))
    
    # push!(SS_solve_func,:(SS_init_guess = ([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)])))

    # push!(SS_solve_func,:(𝓂.SS_init_guess = typeof(SS_init_guess) == Vector{Float64} ? SS_init_guess : ℱ.value.(SS_init_guess)))

    # push!(SS_solve_func,:(return ComponentVector([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]))))


    # fix parameter bounds
    par_bounds = []

    for varpar in intersect(𝓂.bounded_vars, 𝓂.parameters)
        i = indexin([varpar],𝓂.bounded_vars)
        push!(par_bounds, :($varpar = min(max($varpar,$(𝓂.lower_bounds[i...])),$(𝓂.upper_bounds[i...]))))
    end


    solve_exp = :(function solve_SS(parameters::Vector{Real}, 𝓂::ℳ, fail_fast_solvers_only::Bool, verbose::Bool)
                    params_flt = typeof(parameters) == Vector{Float64} ? parameters : ℱ.value.(parameters)
                    current_best = sum(abs2,𝓂.NSSS_solver_cache[end][end] - params_flt)
                    closest_solution_init = 𝓂.NSSS_solver_cache[end]
                    for pars in 𝓂.NSSS_solver_cache
                        latest = sum(abs2,pars[end] - params_flt)
                        if latest <= current_best
                            current_best = latest
                            closest_solution_init = pars
                        end
                    end
                    solved_scale = 0
                    range_length = fail_fast_solvers_only ? [1] : [ 1, 2, 4, 8,16,32]
                    for r in range_length
                        for scale in range(0,1,r+1)[2:end]
                            if scale <= solved_scale continue end
                            current_best = sum(abs2,𝓂.NSSS_solver_cache[end][end] - params_flt)
                            closest_solution = 𝓂.NSSS_solver_cache[end]
                            for pars in 𝓂.NSSS_solver_cache
                                latest = sum(abs2,pars[end] - params_flt)
                                if latest <= current_best
                                    current_best = latest
                                    closest_solution = pars
                                end
                            end
                            params = all(isfinite.(closest_solution_init[end])) && parameters != closest_solution_init[end] ? scale * parameters + (1 - scale) * closest_solution_init[end] : parameters
                            params_scaled_flt = typeof(params) == Vector{Float64} ? params : ℱ.value.(params)
                            $(parameters_in_equations...)
                            $(par_bounds...)
                            $(𝓂.calibration_equations_no_var...)
                            NSSS_solver_cache_tmp = []
                            solution_error = 0.0
                            $(SS_solve_func...)
                            if scale == 1
                                # return ComponentVector([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...])), solution_error
                                return [$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)] , solution_error
                            end
                        end
                    end
                end)

    𝓂.SS_solve_func = @RuntimeGeneratedFunction(solve_exp)

    return nothing
end

# transformation of NSSS problem
function transformer(x)
    # return asinh.(asinh.(asinh.(x)))
    return asinh(asinh(x))
    # return asinh(x)
    # return x
end

function undo_transformer(x)
    # return sinh.(sinh.(sinh.(x)))
    return sinh(sinh(x))
    # return sinh(x)
    # return x
end

block_solver_AD(parameters_and_solved_vars::Vector{<: Number}, 
    n_block::Int, 
    ss_solve_blocks::Function, 
    ss_solve_blocks_no_transform::Function, 
    f::OptimizationFunction, 
    guess::Vector{Float64}, 
    lbs::Vector{Float64}, 
    ubs::Vector{Float64};
    tol = eps(Float64),
    maxtime = 120,
    starting_points = [.9, 1, 1.1, .75, 1.5, -.5, 2, .25],
    fail_fast_solvers_only = true,
    verbose = false) = ImplicitFunction(x -> block_solver(x,
                                                            n_block, 
                                                            ss_solve_blocks,
                                                            f,
                                                            guess,
                                                            lbs,
                                                            ubs;
                                                            tol = tol,
                                                            maxtime = maxtime,
                                                            starting_points = starting_points,
                                                            fail_fast_solvers_only = fail_fast_solvers_only,
                                                            verbose = verbose)[1],  
                                        ss_solve_blocks_no_transform)

function block_solver(parameters_and_solved_vars::Vector{Float64}, 
                        n_block::Int, 
                        ss_solve_blocks::Function, 
                        # SS_optimizer, 
                        f::OptimizationFunction, 
                        guess::Vector{Float64}, 
                        lbs::Vector{Float64}, 
                        ubs::Vector{Float64};
                        tol = eps(Float64),
                        maxtime = 120,
                        starting_points = [.9, 1, 1.1, .75, 1.5, -.5, 2, .25],
                        fail_fast_solvers_only = true,
                        verbose = false)
    
    sol_values = guess
    sol_minimum  = sum(abs2,ss_solve_blocks(parameters_and_solved_vars,transformer.(sol_values)))

    # try NLboxsolve first
    if (sol_minimum > tol)# | (maximum(abs,ss_solve_blocks(sol_values,parameters_and_solved_vars)) > tol))
        SS_optimizer = nlboxsolve

        previous_sol_init = max.(lbs,min.(ubs, sol_values))
        sol_new = try SS_optimizer(x->ss_solve_blocks(parameters_and_solved_vars, x),transformer.(previous_sol_init),transformer.(lbs),transformer.(ubs),method = :nk) catch e end

        if isnothing(sol_new)
            sol_minimum = Inf
            sol_values = zero(sol_values)
        else
            sol_minimum = isnan(sum(abs2,sol_new.fzero)) ? Inf : sum(abs2,sol_new.fzero)
            sol_values = undo_transformer.(sol_new.zero)
        end

        if (sol_minimum < tol) && verbose
            println("Block: ",n_block," - Solved using ",string(SS_optimizer)," and previous best non-converged solution; maximum residual = ",maximum(abs,ss_solve_blocks(parameters_and_solved_vars,transformer.(sol_values))))
        end

        if !fail_fast_solvers_only
            # if the previous non-converged best guess as a starting point does not work, try the standard starting points
            for starting_point in starting_points
                if sol_minimum > tol
                    standard_inits = max.(lbs,min.(ubs, fill(starting_point,length(guess))))
                    sol_new = try SS_optimizer(x->ss_solve_blocks(parameters_and_solved_vars, x),transformer.(standard_inits),transformer.(lbs),transformer.(ubs),method = :nk) catch e end
                    
                    if isnothing(sol_new)
                        sol_minimum = Inf
                        sol_values = zero(sol_values)
                    elseif (isnan(sum(abs2,sol_new.fzero)) ? Inf : sum(abs2,sol_new.fzero)) < sol_minimum
                        sol_minimum = isnan(sum(abs2,sol_new.fzero)) ? Inf : sum(abs2,sol_new.fzero)
                        sol_values = undo_transformer.(sol_new.zero)

                        if sol_minimum < tol && verbose
                            println("Block: ",n_block," - Solved using ",string(SS_optimizer)," and starting point: ",starting_point,"; maximum residual = ",maximum(abs,ss_solve_blocks(parameters_and_solved_vars,transformer.(sol_values))))
                        end
                    end

                else 
                    break
                end
            end

            # if the the standard starting point doesnt work try the provided guess
            if sol_minimum > tol
                sol_new = try SS_optimizer(x->ss_solve_blocks(parameters_and_solved_vars,x),transformer.(guess),transformer.(lbs),transformer.(ubs),method = :nk) catch e end
                if isnothing(sol_new)
                    sol_minimum = Inf
                    sol_values = zero(sol_values)
                elseif (isnan(sum(abs2,sol_new.fzero)) ? Inf : sum(abs2,sol_new.fzero)) < sol_minimum
                    sol_minimum = isnan(sum(abs2,sol_new.fzero)) ? Inf : sum(abs2,sol_new.fzero)
                    sol_values = undo_transformer.(sol_new.zero)

                    if (sol_minimum < tol) && verbose
                        println("Block: ",n_block," - Solved using ",string(SS_optimizer)," and initial guess; maximum residual = ",maximum(abs,ss_solve_blocks(parameters_and_solved_vars,transformer.(sol_values))))
                    end
                end
            end
        end
    elseif verbose
        println("Block: ",n_block," - Solved using previous solution; maximum residual = ",maximum(abs,ss_solve_blocks(parameters_and_solved_vars,transformer.(sol_values))))
    end

    optimizers = fail_fast_solvers_only ? [NLopt.LD_LBFGS] : [NLopt.LD_LBFGS, NLopt.LN_BOBYQA, NLopt.LN_PRAXIS, NLopt.LD_SLSQP, NLopt.LN_SBPLX]

    # cycle through NLopt solvers
    for SS_optimizer in optimizers
        if (sol_minimum > tol)# | (maximum(abs,ss_solve_blocks(sol_values,parameters_and_solved_vars)) > tol))

            previous_sol_init = max.(lbs,min.(ubs, sol_values))
            prob = OptimizationProblem(f, transformer.(previous_sol_init), parameters_and_solved_vars, lb = transformer.(lbs), ub = transformer.(ubs))
            sol_new = solve(prob, SS_optimizer(), local_maxtime = maxtime, maxtime = maxtime)

            if sol_new.minimum < sol_minimum
                sol_minimum = sol_new.minimum
                sol_values = undo_transformer.(sol_new.u)

                if (sol_minimum < tol) && verbose
                    println("Block: ",n_block," - Solved using ",string(SS_optimizer)," and previous best non-converged solution; maximum residual = ",maximum(abs,ss_solve_blocks(parameters_and_solved_vars,transformer.(sol_values))))
                end
            end

            if !fail_fast_solvers_only
                # if the previous non-converged best guess as a starting point does not work, try the standard starting point
                for starting_point in starting_points
                    if (sol_minimum > tol)# | (maximum(abs,ss_solve_blocks(sol_values, parameters_and_solved_vars)) > tol)
                        standard_inits = max.(lbs,min.(ubs, fill(starting_point,length(guess))))
                        prob = OptimizationProblem(f, transformer.(standard_inits), parameters_and_solved_vars, lb = transformer.(lbs), ub = transformer.(ubs))
                        sol_new = solve(prob, SS_optimizer(), local_maxtime = maxtime, maxtime = maxtime)

                        if sol_new.minimum < sol_minimum
                            sol_minimum = sol_new.minimum
                            sol_values = undo_transformer.(sol_new.u)

                            if (sol_minimum < tol) && verbose
                                println("Block: ",n_block," - Solved using ",string(SS_optimizer)," and starting point: ",starting_point,"; maximum residual = ",maximum(abs,ss_solve_blocks(parameters_and_solved_vars,transformer.(sol_values))))
                            end
                        end

                    else 
                        break
                    end
                end

                # if the the standard starting point doesnt work try the provided guess
                if (sol_minimum > tol)# | (maximum(abs,ss_solve_blocks(sol_values, parameters_and_solved_vars)) > tol)
                    prob = OptimizationProblem(f, transformer.(guess), parameters_and_solved_vars, lb = transformer.(lbs), ub = transformer.(ubs))
                    sol_new = solve(prob, SS_optimizer(), local_maxtime = maxtime, maxtime = maxtime)
                    if sol_new.minimum < sol_minimum
                        sol_minimum  = sol_new.minimum
                        sol_values = undo_transformer.(sol_new.u)

                        if (sol_minimum < tol) && verbose
                            println("Block: ",n_block," - Solved using ",string(SS_optimizer)," and initial guess; maximum residual = ",maximum(abs,ss_solve_blocks(parameters_and_solved_vars,transformer.(sol_values))))
                        end
                    end
                end
            end

        end
    end

    return sol_values, sol_minimum
end


# implicit = ImplicitFunction(block_solver, ss_solve_blocks)

function block_solver(parameters_and_solved_vars::Vector{ℱ.Dual{Z,S,N}}, 
    n_block::Int, 
    ss_solve_blocks::Function, 
    # SS_optimizer, 
    f::OptimizationFunction, 
    guess::Vector{Float64}, 
    lbs::Vector{Float64}, 
    ubs::Vector{Float64};
    tol = eps(Float64),
    maxtime = 120,
    starting_points = [.9, 1, 1.1, .75, 1.5, -.5, 2, .25],
    fail_fast_solvers_only = true,
    verbose = false) where {Z,S,N}

    # unpack: AoS -> SoA
    inp = ℱ.value.(parameters_and_solved_vars)

    # you can play with the dimension here, sometimes it makes sense to transpose
    ps = mapreduce(ℱ.partials, hcat, parameters_and_solved_vars)'

    if verbose println("Solution for derivatives.") end
    # get f(vs)
    val, min = block_solver(inp, 
                        n_block, 
                        ss_solve_blocks, 
                        # SS_optimizer, 
                        f, 
                        guess, 
                        lbs, 
                        ubs;
                        tol = tol,
                        maxtime = maxtime,
                        starting_points = starting_points,
                        fail_fast_solvers_only = fail_fast_solvers_only,
                        verbose = verbose)

    if min > tol
        jvp = fill(0,length(val),length(inp)) * ps
    else
        # get J(f, vs) * ps (cheating). Write your custom rule here
        B = ℱ.jacobian(x -> ss_solve_blocks(x,transformer.(val)), inp)
        A = ℱ.jacobian(x -> ss_solve_blocks(inp,transformer.(x)), val)

        jvp = (-A \ B) * ps
    end

    # pack: SoA -> AoS
    return reshape(map(val, eachrow(jvp)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(val)), min
end






function solve!(𝓂::ℳ; 
    parameters = nothing, 
    dynamics::Bool = false, 
    algorithm::Symbol = :riccati, 
    symbolic_SS::Bool = false,
    verbose = false)

    @assert algorithm ∈ [:linear_time_iteration, :riccati, :first_order, :second_order, :third_order]

    if dynamics
        𝓂.solution.outdated_algorithms = union(intersect(𝓂.solution.algorithms,[algorithm]),𝓂.solution.outdated_algorithms)
        𝓂.solution.algorithms = union(𝓂.solution.algorithms,[algorithm])
    end

    if !𝓂.solution.functions_written 
        # consolidate bounds info
        double_info = intersect(𝓂.bounds⁺,𝓂.bounded_vars)
        𝓂.lower_bounds[indexin(double_info,𝓂.bounded_vars)] = max.(eps(Float32),𝓂.lower_bounds[indexin(double_info,𝓂.bounded_vars)])

        new_info = setdiff(𝓂.bounds⁺,𝓂.bounded_vars)
        𝓂.bounded_vars = vcat(𝓂.bounded_vars,new_info)
        𝓂.lower_bounds = vcat(𝓂.lower_bounds,fill(eps(Float32),length(new_info)))
        𝓂.upper_bounds = vcat(𝓂.upper_bounds,fill(1e12,length(new_info)))


        symbolics = create_symbols_eqs!(𝓂)
        remove_redundant_SS_vars!(𝓂,symbolics)
        solve_steady_state!(𝓂, symbolic_SS, symbolics, verbose = verbose)
        write_functions_mapping!(𝓂, symbolics)
        𝓂.solution.functions_written = true
    end

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    if dynamics
        if any([:riccati, :first_order, :second_order, :third_order] .∈ ([algorithm],)) && any([:riccati, :first_order] .∈ (𝓂.solution.outdated_algorithms,))
            SS_and_pars, solution_error = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, verbose)

            ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)
            
            sol_mat = calculate_first_order_solution(∇₁; T = 𝓂.timings)
            
            state_update₁ = function(state::Vector{Float64}, shock::Vector{Float64}) sol_mat * [state[𝓂.timings.past_not_future_and_mixed_idx]; shock] end
            
            𝓂.solution.perturbation.first_order = perturbation_solution(sol_mat, state_update₁)
            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:riccati, :first_order])

            𝓂.solution.non_stochastic_steady_state = SS_and_pars
            𝓂.solution.outdated_NSSS = false

        end
        
        if any([:second_order, :third_order] .∈ ([algorithm],)) && :second_order ∈ 𝓂.solution.outdated_algorithms
            SS_and_pars, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, verbose) : (𝓂.solution.non_stochastic_steady_state, eps())

            if !any([:riccati, :first_order] .∈ (𝓂.solution.outdated_algorithms,))
                ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)
            end

            ∇₂ = calculate_hessian(𝓂.parameter_values,SS_and_pars,𝓂)
            𝐒₂ = calculate_second_order_solution(∇₁, 
                                                ∇₂, 
                                                𝓂.solution.perturbation.first_order.solution_matrix; 
                                                T = 𝓂.timings)

            𝐒₁ = [𝓂.solution.perturbation.first_order.solution_matrix[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝓂.solution.perturbation.first_order.solution_matrix[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]
            
            state_update₂ = function(state::Vector{Float64}, shock::Vector{Float64})
                aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                            1
                            shock]
                return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
            end

            # Calculate stochastic SS
            state = zeros(𝓂.timings.nVars)
            shock = zeros(𝓂.timings.nExo)

            delta = 1

            while delta > eps(Float64)
                state_tmp =  state_update₂(state,shock)
                delta = sum(abs,state_tmp - state)
                state = state_tmp
            end

            all_variables = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

            all_variables[indexin(𝓂.aux,all_variables)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
            
            NSSS_labels = [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]
            
            all_SS = [SS_and_pars[indexin([s],NSSS_labels)...] for s in all_variables]
            # we need all variables for the stochastic steady state because even laads and lags have different SSS then the non-lead-lag ones (contrary to the no stochastic steady state) and we cannot recover them otherwise

            stochastic_steady_state = all_SS + state

            𝓂.solution.perturbation.second_order = higher_order_perturbation_solution(𝐒₂,stochastic_steady_state,state_update₂)

            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:second_order])
            
        end
        
        if :third_order == algorithm && :third_order ∈ 𝓂.solution.outdated_algorithms
            SS_and_pars, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, verbose) : (𝓂.solution.non_stochastic_steady_state, eps())

            if !any([:riccati, :first_order] .∈ (𝓂.solution.outdated_algorithms,))
                ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)
                𝐒₁ = [𝓂.solution.perturbation.first_order.solution_matrix[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝓂.solution.perturbation.first_order.solution_matrix[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]
            end

            if :second_order ∉ 𝓂.solution.outdated_algorithms
                ∇₂ = calculate_hessian(𝓂.parameter_values,SS_and_pars,𝓂)
                𝐒₂ = 𝓂.solution.perturbation.second_order.solution_matrix
            end
            
            ∇₃ = calculate_third_order_derivatives(𝓂.parameter_values,SS_and_pars,𝓂)
            
            𝐒₃ = calculate_third_order_solution(∇₁, 
                                                ∇₂, 
                                                ∇₃, 
                                                𝓂.solution.perturbation.first_order.solution_matrix, 
                                                𝓂.solution.perturbation.second_order.solution_matrix; 
                                                T = 𝓂.timings)

            state_update₃ = function(state::Vector{Float64}, shock::Vector{Float64})
                aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                                1
                                shock]
                return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
            end

            # Calculate stochastic SS
            state = zeros(𝓂.timings.nVars)
            shock = zeros(𝓂.timings.nExo)

            delta = 1

            while delta > eps(Float64)
                state_tmp =  state_update₃(state,shock)
                delta = sum(abs,state_tmp - state)
                state = state_tmp
            end

            all_variables = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

            all_variables[indexin(𝓂.aux,all_variables)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
            
            NSSS_labels = [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]
            
            all_SS = [SS_and_pars[indexin([s],NSSS_labels)...] for s in all_variables]
            # we need all variables for the stochastic steady state because even laads and lags have different SSS then the non-lead-lag ones (contrary to the no stochastic steady state) and we cannot recover them otherwise
            
            stochastic_steady_state = all_SS + state

            𝓂.solution.perturbation.third_order = higher_order_perturbation_solution(𝐒₃,stochastic_steady_state,state_update₃)

            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:third_order])
            
        end
        
        if :linear_time_iteration == algorithm && :linear_time_iteration ∈ 𝓂.solution.outdated_algorithms
            SS_and_pars, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, verbose) : (𝓂.solution.non_stochastic_steady_state, eps())

            ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)
            
            sol_mat = calculate_linear_time_iteration_solution(∇₁; T = 𝓂.timings)
            
            state_update₁ₜ = function(state::Vector{Float64}, shock::Vector{Float64}) sol_mat * [state[𝓂.timings.past_not_future_and_mixed_idx]; shock] end
            
            𝓂.solution.perturbation.linear_time_iteration = perturbation_solution(sol_mat, state_update₁ₜ)
            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:linear_time_iteration])

            𝓂.solution.non_stochastic_steady_state = SS_and_pars
            𝓂.solution.outdated_NSSS = false
            
        end
    end
    return nothing
end





function write_functions_mapping!(𝓂::ℳ, symbolics::symbolics)
    present_varss = map(x->Symbol(string(x) * "₍₀₎"),sort(setdiff(union(𝓂.var_present,𝓂.aux_present,𝓂.exo_present), 𝓂.nonnegativity_auxilliary_vars)))
    future_varss  = map(x->Symbol(string(x) * "₍₁₎"),sort(setdiff(union(𝓂.var_future,𝓂.aux_future,𝓂.exo_future), 𝓂.nonnegativity_auxilliary_vars)))
    past_varss    = map(x->Symbol(string(x) * "₍₋₁₎"),sort(setdiff(union(𝓂.var_past,𝓂.aux_past,𝓂.exo_past), 𝓂.nonnegativity_auxilliary_vars)))
    shock_varss   = map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.exo)
    ss_varss      = map(x->Symbol(string(x) * "₍ₛₛ₎"),𝓂.var)

    steady_state = []
    for (i, var) in enumerate(ss_varss)
        push!(steady_state,:($var = X̄[$i]))
        # ii += 1
    end

    ii = 1

    alll = []
    for var in future_varss
        push!(alll,:($var = X[$ii]))
        ii += 1
    end

    for var in present_varss
        push!(alll,:($var = X[$ii]))
        ii += 1
    end

    for var in past_varss
        push!(alll,:($var = X[$ii]))
        ii += 1
    end

    for var in shock_varss
        push!(alll,:($var = X[$ii]))
        ii += 1
    end


    # paras = []
    # push!(paras,:((;$(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)...)) = params))

    paras = []
    for (i, parss) in enumerate(vcat(𝓂.parameters,𝓂.calibration_equations_parameters))
        push!(paras,:($parss = params[$i]))
    end

    # watch out with naming of parameters in model and functions
    mod_func2 = :(function model_function_uni_redux(X::Vector{Real}, params::Vector{Real}, X̄::Vector{Real})
        $(alll...)
        $(paras...)
		$(𝓂.calibration_equations_no_var...)
        $(steady_state...)
        [$(𝓂.dyn_equations...)]
    end)


    𝓂.model_function = @RuntimeGeneratedFunction(mod_func2)


    dyn_future_list = collect(reduce(union, symbolics.dyn_future_list))
    dyn_present_list = collect(reduce(union, symbolics.dyn_present_list))
    dyn_past_list = collect(reduce(union, symbolics.dyn_past_list))
    dyn_exo_list = collect(reduce(union,symbolics.dyn_exo_list))

    future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_future_list))
    present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_present_list))
    past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_past_list))
    exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))
    
    vars = [dyn_future_list[indexin(sort(future),future)]...,
            dyn_present_list[indexin(sort(present),present)]...,
            dyn_past_list[indexin(sort(past),past)]...,
            dyn_exo_list[indexin(sort(exo),exo)]...]
    
    eqs = symbolics.dyn_equations

    first_order = []
    second_order = []
    third_order = []
    row1 = Int[]
    row2 = Int[]
    row3 = Int[]
    column1 = Int[]
    column2 = Int[]
    column3 = Int[]
    i1 = 1
    i2 = 1
    i3 = 1
    
    for (c1,var1) in enumerate(vars)
        for (r,eq) in enumerate(eqs)
            if var1 ∈ free_symbols(eq)
                deriv_first = diff(eq,var1)
                if deriv_first != 0 
                    deriv_expr = Meta.parse(string(deriv_first.subs(SymPy.PI,SymPy.N(SymPy.PI))))
                    push!(first_order, :($(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, deriv_expr))))
                    push!(row1,r)
                    push!(column1,c1)
                    i1 += 1
                    for (c2,var2) in enumerate(vars)
                        if var2 ∈ free_symbols(deriv_first)
                            deriv_second = diff(deriv_first,var2)
                            if deriv_second != 0 
                                deriv_expr = Meta.parse(string(deriv_second.subs(SymPy.PI,SymPy.N(SymPy.PI))))
                                push!(second_order, :($(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, deriv_expr))))
                                push!(row2,r)
                                push!(column2,(c1 - 1) * length(vars) + c2)
                                i2 += 1
                                for (c3,var3) in enumerate(vars)
                                    if var3 ∈ free_symbols(deriv_second)
                                        deriv_third = diff(deriv_second,var3)
                                        if deriv_third != 0 
                                            deriv_expr = Meta.parse(string(deriv_third.subs(SymPy.PI,SymPy.N(SymPy.PI))))
                                            push!(third_order, :($(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, deriv_expr))))
                                            push!(row3,r)
                                            push!(column3,(c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3)
                                            i3 += 1
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    mod_func3 = :(function model_jacobian(X::Vector{Number}, params::Vector{Number}, X̄::Vector{Number})
        $(alll...)
        $(paras...)
        $(𝓂.calibration_equations_no_var...)
        $(steady_state...)
        sparse([$(row1...)], [$(column1...)], [$(first_order...)], $(length(eqs)), $(length(vars)))
    end)

    𝓂.model_jacobian = @RuntimeGeneratedFunction(mod_func3)


    if length(row2) == 0 
        out = :(spzeros($(length(eqs)), $(length(vars)^2)))
    else 
        out = :(sparse([$(row2...)], [$(column2...)], [$(second_order...)], $(length(eqs)), $(length(vars)^2)))
    end

    mod_func4 = :(function model_hessian(X::Vector{Number}, params::Vector{Number}, X̄::Vector{Number})
        $(alll...)
        $(paras...)
        $(𝓂.calibration_equations_no_var...)
        $(steady_state...)
        $out
    end)

    𝓂.model_hessian = @RuntimeGeneratedFunction(mod_func4)


    if length(row3) == 0 
        out = :(spzeros($(length(eqs)), $(length(vars)^3)))
    else 
        out = :(sparse([$(row3...)], [$(column3...)], [$(third_order...)], $(length(eqs)), $(length(vars)^3)))
    end

    mod_func5 = :(function model_hessian(X::Vector{Number}, params::Vector{Number}, X̄::Vector{Number})
        $(alll...)
        $(paras...)
        $(𝓂.calibration_equations_no_var...)
        $(steady_state...)
        $out
    end)

    𝓂.model_third_order_derivatives = @RuntimeGeneratedFunction(mod_func5)


    calib_eqs = []
    for (i, eqs) in enumerate(𝓂.solved_vals) 
        varss = 𝓂.solved_vars[i]
        push!(calib_eqs,:($varss = $eqs))
    end

    for varss in 𝓂.exo
        push!(calib_eqs,:($varss = 0))
    end

    calib_pars = []
    for (i, parss) in enumerate(𝓂.parameters)
        push!(calib_pars,:($parss = parameters[$i]))
    end

    var_out = []
    ii =  1
    for var in 𝓂.var
        push!(var_out,:($var = SS[$ii]))
        ii += 1
    end

    par_out = []
    for cal in 𝓂.calibration_equations_parameters
        push!(par_out,:($cal = SS[$ii]))
        ii += 1
    end

    calib_pars = []
    for (i, parss) in enumerate(𝓂.parameters)
        push!(calib_pars,:($parss = parameters[$i]))
    end

    test_func = :(function test_SS(parameters::Vector{Float64}, SS::Vector{Float64})
        $(calib_pars...) 
        $(var_out...)
        $(par_out...)
        [$(𝓂.ss_equations...),$(𝓂.calibration_equations...)]
    end)

    𝓂.solution.valid_steady_state_solution = @RuntimeGeneratedFunction(test_func)

    𝓂.solution.outdated_algorithms = Set([:linear_time_iteration, :riccati, :first_order, :second_order, :third_order])
    return nothing
end



write_parameters_input!(𝓂::ℳ, parameters::Nothing; verbose = true) = return parameters
write_parameters_input!(𝓂::ℳ, parameters::Pair{Symbol,<: Number}; verbose = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{Symbol,<: Number},Vararg{Pair{Symbol,<: Number}}}; verbose = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Vector{Pair{Symbol, Float64}}; verbose = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters), verbose = verbose)



function write_parameters_input!(𝓂::ℳ, parameters::Dict{Symbol,<: Number}; verbose = true)
    if length(setdiff(collect(keys(parameters)),𝓂.parameters))>0
        println("Parameters not part of the model: ",setdiff(collect(keys(parameters)),𝓂.parameters))
        for kk in setdiff(collect(keys(parameters)),𝓂.parameters)
            delete!(parameters,kk)
        end
    end

    bounds_broken = false

    for i in 1:length(parameters)
        bnd_idx = findfirst(x->x==collect(keys(parameters))[i],𝓂.bounded_vars)
        if !isnothing(bnd_idx)
            if collect(values(parameters))[i] > 𝓂.upper_bounds[bnd_idx]
                # println("Calibration is out of bounds for ",collect(keys(parameters))[i],":\t",collect(values(parameters))[i]," > ",𝓂.upper_bounds[bnd_idx] + eps())
                println("Bounds error for",collect(keys(parameters))[i]," < ",𝓂.upper_bounds[bnd_idx] + eps(),"\tparameter value: ",collect(values(parameters))[i])
                bounds_broken = true
                continue
            end
            if collect(values(parameters))[i] < 𝓂.lower_bounds[bnd_idx]
                # println("Calibration is out of bounds for ",collect(keys(parameters))[i],":\t",collect(values(parameters))[i]," < ",𝓂.lower_bounds[bnd_idx] - eps())
                println("Bounds error for",collect(keys(parameters))[i]," > ",𝓂.lower_bounds[bnd_idx] + eps(),"\tparameter value: ",collect(values(parameters))[i])
                bounds_broken = true
                continue
            end
        end
    end

    if bounds_broken
        println("Parameters unchanged.")
    else
        ntrsct_idx = map(x-> getindex(1:length(𝓂.parameter_values),𝓂.parameters .== x)[1],collect(keys(parameters)))
        

        
        if !all(𝓂.parameter_values[ntrsct_idx] .== collect(values(parameters)))
            if verbose println("Parameter changes: ") end
            𝓂.solution.outdated_algorithms = Set([:linear_time_iteration, :riccati, :first_order, :second_order, :third_order])
        end
            
        for i in 1:length(parameters)
            if 𝓂.parameter_values[ntrsct_idx[i]] != collect(values(parameters))[i]
                if collect(keys(parameters))[i] ∈ 𝓂.SS_dependencies[end][2] && 𝓂.solution.outdated_NSSS == false
                    𝓂.solution.outdated_NSSS = true
                end
                
                if verbose println("\t",𝓂.parameters[ntrsct_idx[i]],"\tfrom ",𝓂.parameter_values[ntrsct_idx[i]],"\tto ",collect(values(parameters))[i]) end

                𝓂.parameter_values[ntrsct_idx[i]] = collect(values(parameters))[i]
            end
        end
    end

    if 𝓂.solution.outdated_NSSS == true && verbose println("New parameters changed the steady state.") end
end


write_parameters_input!(𝓂::ℳ, parameters::Tuple{<: Number,Vararg{<: Number}}; verbose = true) = write_parameters_input!(𝓂::ℳ, vec(collect(parameters)), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Matrix{<: Number}; verbose = true) = write_parameters_input!(𝓂::ℳ, vec(collect(parameters)), verbose = verbose)


function write_parameters_input!(𝓂::ℳ, parameters::Vector{<: Number}; verbose = true)
    if length(parameters) > length(𝓂.parameter_values)
        println("Model has "*string(length(𝓂.parameter_values))*" parameters. "*string(length(parameters))*" were provided. The following will be ignored: "*string(parameters[length(𝓂.parameter_values)+1:end]...))

        parameters = parameters[1:length(𝓂.parameter_values)]
    end

    bounds_broken = false

    for i in 1:length(parameters)
        bnd_idx = findfirst(x -> x == 𝓂.parameters[i], 𝓂.bounded_vars)
        if !isnothing(bnd_idx)
            if collect(values(parameters))[i] > 𝓂.upper_bounds[bnd_idx]
                println("Bounds error for",𝓂.parameters[i]," < ",𝓂.upper_bounds[bnd_idx] + eps(),"\tparameter value: ",𝓂.parameter_values[i])
                bounds_broken = true
                continue
            end
            if collect(values(parameters))[i] < 𝓂.lower_bounds[bnd_idx]
                println("Bounds error for",𝓂.parameters[i]," > ",𝓂.lower_bounds[bnd_idx] + eps(),"\tparameter value: ",𝓂.parameter_values[i])
                bounds_broken = true
                continue
            end
        end
    end

    if bounds_broken
        println("Parameters unchanged.")
    else
        if !all(parameters .== 𝓂.parameter_values[1:length(parameters)])
            𝓂.solution.outdated_algorithms = Set([:linear_time_iteration, :riccati, :first_order, :second_order, :third_order])

            match_idx = []
            for (i, v) in enumerate(parameters)
                if v != 𝓂.parameter_values[i]
                     push!(match_idx,i)
                end
            end
            
            changed_vals = parameters[match_idx]
            changes_pars = 𝓂.parameters[match_idx]

            for p in changes_pars
                if p ∈ 𝓂.SS_dependencies[end][2] && 𝓂.solution.outdated_NSSS == false
                    𝓂.solution.outdated_NSSS = true
                    # println("SS outdated.")
                end
            end

            if verbose 
                println("Parameter changes: ")
                for (i,m) in enumerate(match_idx)
                    println("\t",changes_pars[i],"\tfrom ",𝓂.parameter_values[m],"\tto ",changed_vals[i])
                end
            end

            𝓂.parameter_values[match_idx] = parameters[match_idx]
        end
    end
    if 𝓂.solution.outdated_NSSS == true && verbose println("New parameters changed the steady state.") end
end



function SS_parameter_derivatives(parameters::Vector{<: Number}, parameters_idx, 𝓂::ℳ; verbose = false)
    𝓂.parameter_values[parameters_idx] = parameters
    𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, verbose)
end


function SS_parameter_derivatives(parameters::Number, parameters_idx::Int, 𝓂::ℳ; verbose = false)
    𝓂.parameter_values[parameters_idx] = parameters
    𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, verbose)
end


function covariance_parameter_derivatives(parameters::Vector{<: Number}, parameters_idx, 𝓂::ℳ; verbose = false)
    𝓂.parameter_values[parameters_idx] = parameters
    convert(Vector{Number},max.(ℒ.diag(calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)[1]),eps(Float64)))
end


function covariance_parameter_derivatives(parameters::Number, parameters_idx::Int, 𝓂::ℳ; verbose = false)
    𝓂.parameter_values[parameters_idx] = parameters
    convert(Vector{Number},max.(ℒ.diag(calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)[1]),eps(Float64)))
end



function calculate_jacobian(parameters::Vector{<: Number}, SS_and_pars::AbstractArray{<: Number}, 𝓂::ℳ)
    var_past = @ignore_derivatives setdiff(𝓂.var_past,𝓂.nonnegativity_auxilliary_vars)
    var_present = @ignore_derivatives setdiff(𝓂.var_present,𝓂.nonnegativity_auxilliary_vars)
    var_future = @ignore_derivatives setdiff(𝓂.var_future,𝓂.nonnegativity_auxilliary_vars)

    SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
    calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]
    # par = ComponentVector(vcat(parameters,calibrated_parameters),Axis(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)))
    par = vcat(parameters,calibrated_parameters)

    past_idx = @ignore_derivatives [indexin(sort([var_past; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_past,𝓂.exo_past))]), sort(union(𝓂.var,𝓂.exo_present)))...]
    SS_past =       length(past_idx) > 0 ? SS[past_idx] : zeros(0) #; zeros(length(𝓂.exo_past))...]
    
    present_idx = @ignore_derivatives [indexin(sort([var_present; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_present,𝓂.exo_present))]), sort(union(𝓂.var,𝓂.exo_present)))...]
    SS_present =    length(present_idx) > 0 ? SS[present_idx] : zeros(0)#; zeros(length(𝓂.exo_present))...]
    
    future_idx = @ignore_derivatives [indexin(sort([var_future; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_future,𝓂.exo_future))]), sort(union(𝓂.var,𝓂.exo_present)))...]
    SS_future =     length(future_idx) > 0 ? SS[future_idx] : zeros(0)#; zeros(length(𝓂.exo_future))...]

    shocks_ss = zeros(length(𝓂.exo))

    # return ℱ.jacobian(x -> 𝓂.model_function(x, par, SS), [SS_future; SS_present; SS_past; shocks_ss])#, SS_and_pars
    return collect(𝓂.model_jacobian([SS_future; SS_present; SS_past; shocks_ss], par, SS))
end



function calculate_hessian(parameters::Vector{<: Number}, SS_and_pars::AbstractArray{<: Number}, 𝓂::ℳ)
    var_past = setdiff(𝓂.var_past,𝓂.nonnegativity_auxilliary_vars)
    var_present = setdiff(𝓂.var_present,𝓂.nonnegativity_auxilliary_vars)
    var_future = setdiff(𝓂.var_future,𝓂.nonnegativity_auxilliary_vars)

    SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
    calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]
	par = ComponentVector( vcat(parameters,calibrated_parameters),Axis(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)))
    
    past_idx = [indexin(sort([var_past; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_past,𝓂.exo_past))]), sort(union(𝓂.var,𝓂.exo_present)))...]
    SS_past =       length(past_idx) > 0 ? SS[past_idx] : zeros(0) #; zeros(length(𝓂.exo_past))...]
    
    present_idx = [indexin(sort([var_present; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_present,𝓂.exo_present))]), sort(union(𝓂.var,𝓂.exo_present)))...]
    SS_present =    length(present_idx) > 0 ? SS[present_idx] : zeros(0)#; zeros(length(𝓂.exo_present))...]
    
    future_idx = [indexin(sort([var_future; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_future,𝓂.exo_future))]), sort(union(𝓂.var,𝓂.exo_present)))...]
    SS_future =     length(future_idx) > 0 ? SS[future_idx] : zeros(0)#; zeros(length(𝓂.exo_future))...]

    shocks_ss = zeros(length(𝓂.exo))

    nk = 𝓂.timings.nPast_not_future_and_mixed + 𝓂.timings.nVars + 𝓂.timings.nFuture_not_past_and_mixed + length(𝓂.exo)
        
    # return sparse(reshape(ℱ.jacobian(x -> ℱ.jacobian(x -> (𝓂.model_function(x, par, SS)), x), [SS_future; SS_present; SS_past; shocks_ss] ), 𝓂.timings.nVars, nk^2))#, SS_and_pars
    return 𝓂.model_hessian([SS_future; SS_present; SS_past; shocks_ss], par, SS)
end



function calculate_third_order_derivatives(parameters::Vector{<: Number}, SS_and_pars::AbstractArray{<: Number}, 𝓂::ℳ)
    var_past = setdiff(𝓂.var_past,𝓂.nonnegativity_auxilliary_vars)
    var_present = setdiff(𝓂.var_present,𝓂.nonnegativity_auxilliary_vars)
    var_future = setdiff(𝓂.var_future,𝓂.nonnegativity_auxilliary_vars)

    SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
    calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]
	par = ComponentVector( vcat(parameters,calibrated_parameters),Axis(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)))
    
    past_idx = [indexin(sort([var_past; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_past,𝓂.exo_past))]), sort(union(𝓂.var,𝓂.exo_present)))...]
    SS_past =       length(past_idx) > 0 ? SS[past_idx] : zeros(0) #; zeros(length(𝓂.exo_past))...]
    
    present_idx = [indexin(sort([var_present; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_present,𝓂.exo_present))]), sort(union(𝓂.var,𝓂.exo_present)))...]
    SS_present =    length(present_idx) > 0 ? SS[present_idx] : zeros(0)#; zeros(length(𝓂.exo_present))...]
    
    future_idx = [indexin(sort([var_future; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_future,𝓂.exo_future))]), sort(union(𝓂.var,𝓂.exo_present)))...]
    SS_future =     length(future_idx) > 0 ? SS[future_idx] : zeros(0)#; zeros(length(𝓂.exo_future))...]

    shocks_ss = zeros(length(𝓂.exo))

    nk = 𝓂.timings.nPast_not_future_and_mixed + 𝓂.timings.nVars + 𝓂.timings.nFuture_not_past_and_mixed + length(𝓂.exo)
      
    # return sparse(reshape(ℱ.jacobian(x -> ℱ.jacobian(x -> ℱ.jacobian(x -> 𝓂.model_function(x, par, SS), x), x), [SS_future; SS_present; SS_past; shocks_ss] ), 𝓂.timings.nVars, nk^3))#, SS_and_pars
    return 𝓂.model_third_order_derivatives([SS_future; SS_present; SS_past; shocks_ss], par, SS)
end



function calculate_linear_time_iteration_solution(∇₁::AbstractMatrix{Float64}; T::timings)
    expand = @views [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
              ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    ∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    ∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    ∇₋ = @views ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]
    ∇ₑ = @views ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]
  
    maxiter = 1000
    tol = eps(Float32)

    F = zero(∇₋)
    S = zero(∇₋)
    # F = randn(size(∇₋))
    # S = randn(size(∇₋))
    
    error = one(tol) + tol
    iter = 0

    while error > tol && iter <= maxiter
        F̂ = -(∇₊ * F + ∇₀) \ ∇₋
        Ŝ = -(∇₋ * S + ∇₀) \ ∇₊
        
        error = maximum(∇₊ * F̂ * F̂ + ∇₀ * F̂ + ∇₋)
        
        F = F̂
        S = Ŝ
        
        iter += 1
    end

    if iter == maxiter
        outmessage = "Convergence Failed. Max Iterations Reached. Error: $error"
    elseif maximum(abs,ℒ.eigen(F).values) > 1.0
        outmessage = "No Stable Solution Exists!"
    elseif maximum(abs,ℒ.eigen(S).values) > 1.0
        outmessage = "Multiple Solutions Exist!"
    end

    Q = -(∇₊ * F + ∇₀) \ ∇ₑ

    @views hcat(F[:,T.past_not_future_and_mixed_idx],Q)
end



function riccati_forward(∇₁::AbstractMatrix{<: Number}; T::timings, explosive::Bool = false)#::AbstractMatrix{Real}
    ∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
    ∇₀ = @view ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]
    ∇₋ = @view ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]

    Q    = ℒ.qr(collect(∇₀[:,T.present_only_idx]))
    Qinv = Q.Q'

    A₊ = Qinv * ∇₊
    A₀ = Qinv * ∇₀
    A₋ = Qinv * ∇₋

    dynIndex = T.nPresent_only+1:T.nVars

    Ã₊  = @view A₊[dynIndex,:]
    Ã₋  = @view A₋[dynIndex,:]
    Ã₀₊ = @view A₀[dynIndex, T.future_not_past_and_mixed_idx]
    Ã₀₋ = @views A₀[dynIndex, T.past_not_future_idx] * ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.not_mixed_in_past_idx,:]
    
    Z₊ = zeros(T.nMixed,T.nFuture_not_past_and_mixed)
    I₊ = @view ℒ.diagm(ones(T.nFuture_not_past_and_mixed))[T.mixed_in_future_idx,:]

    Z₋ = zeros(T.nMixed,T.nPast_not_future_and_mixed)
    I₋ = @view ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.mixed_in_past_idx,:]

    D = vcat(hcat(Ã₀₋, Ã₊), hcat(I₋, Z₊))
    E = vcat(hcat(-Ã₋,-Ã₀₊), hcat(Z₋, I₊))
    # this is the companion form and by itself the linearisation of the matrix polynomial used in the linear time iteration method. see: https://opus4.kobv.de/opus4-matheon/files/209/240.pdf
    schdcmp = ℒ.schur(D,E)

    if explosive # returns false for NaN gen. eigenvalue which is correct here bc they are > 1
        eigenselect = abs.(schdcmp.β ./ schdcmp.α) .>= 1

        ℒ.ordschur!(schdcmp, eigenselect)

        Z₂₁ = @view schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
        Z₁₁ = @view schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

        S₁₁    = @view schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
        T₁₁    = @view schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

        Z₁₁inv = ℒ.pinv(Z₁₁)
    else
        eigenselect = abs.(schdcmp.β ./ schdcmp.α) .< 1

        ℒ.ordschur!(schdcmp, eigenselect)

        Z₂₁ = @view schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
        Z₁₁ = @view schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

        S₁₁    = @view schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
        T₁₁    = @view schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

        Z₁₁inv = inv(Z₁₁)
    end
    
    D      = Z₂₁ * Z₁₁inv
    L      = Z₁₁ * (S₁₁ \ T₁₁) * Z₁₁inv

    sol = @views vcat(L[T.not_mixed_in_past_idx,:], D)

    Ā₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_only_idx]
    A₊ᵤ  = @view A₊[1:T.nPresent_only,:]
    Ã₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_but_not_only_idx]
    A₋ᵤ  = @view A₋[1:T.nPresent_only,:]

    A    = @views vcat(- Ā₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ), sol)
    
    @view A[T.reorder,:]
end


function riccati_conditions(∇₁::AbstractMatrix{<: Number}, sol_d::AbstractMatrix{<: Number}; T::timings, explosive::Bool = false) #::AbstractMatrix{Real},
    expand = @views [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
              ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    A = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    B = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    C = @views ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]

    sol_buf = sol_d * expand[2]

    err1 = A * sol_buf * sol_buf + B * sol_buf + C

    @view err1[:,T.past_not_future_and_mixed_idx]
end



function riccati_forward(∇₁::AbstractMatrix{ℱ.Dual{Z,S,N}}; T::timings = T, explosive::Bool = false) where {Z,S,N}
    # unpack: AoS -> SoA
    ∇̂₁ = ℱ.value.(∇₁)
    # you can play with the dimension here, sometimes it makes sense to transpose
    ps = mapreduce(ℱ.partials, hcat, ∇₁)'

    # get f(vs)
    val = riccati_forward(∇̂₁;T = T, explosive = explosive)

    # get J(f, vs) * ps (cheating). Write your custom rule here
    B = ℱ.jacobian(x -> riccati_conditions(x, val; T = T), ∇̂₁)
    A = ℱ.jacobian(x -> riccati_conditions(∇̂₁, x; T = T), val)

    jvp = (-A \ B) * ps

    # pack: SoA -> AoS
    return reshape(map(val, eachrow(jvp)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end,size(val))
end

# riccati_AD = ImplicitFunction(riccati_forward, riccati_conditions)

riccati_(∇₁;T, explosive) = ImplicitFunction(∇₁ -> riccati_forward(∇₁, T=T, explosive=explosive), (x,y)->riccati_conditions(x,y,T=T,explosive=explosive))

function calculate_first_order_solution(∇₁::AbstractMatrix{<: Number}; T::timings, explosive::Bool = false)
    # A = riccati_AD(∇₁, T = T, explosive = explosive)
    riccati = riccati_(∇₁, T = T, explosive = explosive)
    A = riccati(∇₁)
    # A = riccati_forward(∇₁, T = T, explosive = explosive)

    Jm = @view(ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:])
    
    ∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:]
    ∇₀ = @view ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    ∇ₑ = @view ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

    B = -((∇₊ * A * Jm + ∇₀) \ ∇ₑ)

    return hcat(A, B)
end



function  calculate_second_order_solution(∇₁::AbstractMatrix{Float64}, #first order derivatives
                                            ∇₂::AbstractMatrix{Float64}, #second order derivatives
                                            𝑺₁::AbstractMatrix{Float64};  #first order solution
                                            T::timings)
    # inspired by Levintal
    tol = eps(Float32)

    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx;
    i₋ = T.past_not_future_and_mixed_idx;

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo;
    n  = T.nVars
    nₑ₋ = n₋ + 1 + nₑ


    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] |> sparse
    droptol!(𝐒₁,tol)

    # set up vector to capture volatility effect
    redu = sparsevec(nₑ₋ - nₑ + 1:nₑ₋, 1)
    redu_idxs = findnz(ℒ.kron(redu, redu))[1]
    𝛔 = @views sparse(redu_idxs[Int.(range(1,nₑ^2,nₑ))], fill(n₋ * (nₑ₋ + 1) + 1, nₑ), 1, nₑ₋^2, nₑ₋^2)

    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];
    
    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)];


    # setup compression matrices
    colls2 = [nₑ₋ * (i-1) + k for i in 1:nₑ₋ for k in 1:i]
    𝐂₂ = sparse(colls2, 1:length(colls2) , 1)
    𝐔₂ = 𝐂₂' * sparse([i <= k ? (k - 1) * nₑ₋ + i : (i - 1) * nₑ₋ + k for k in 1:nₑ₋ for i in 1:nₑ₋], 1:nₑ₋^2, 1)


    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]

    ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = -∇₂ * (ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * 𝛔) * 𝐂₂ 

    X = sparse(∇₁₊𝐒₁➕∇₁₀ \ ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹)
    droptol!(X,tol)


    ∇₁₊ = @views sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])

    B = sparse(∇₁₊𝐒₁➕∇₁₀ \ ∇₁₊)
    droptol!(B,tol)


    C = (𝐔₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐔₂ * 𝛔) * 𝐂₂
    droptol!(C,tol)

    A = spdiagm(ones(n))

    lm = LinearMap{Float64}(x -> A * reshape(x,size(X)) - B * reshape(x,size(X)) * C, size(X)[1] * size(X)[2])

    𝐒₂ = sparse(reshape(ℐ.bicgstabl(lm, vec(-X)), size(X))) * 𝐔₂ # fastest
    droptol!(𝐒₂,tol)

    return 𝐒₂
end



function  calculate_third_order_solution(∇₁::AbstractMatrix{Float64}, #first order derivatives
                                            ∇₂::AbstractMatrix{Float64}, #second order derivatives
                                            ∇₃::AbstractMatrix{Float64}, #third order derivatives
                                            𝑺₁::AbstractMatrix{Float64}, #first order solution
                                            𝐒₂::AbstractMatrix{Float64}; #second order solution
                                            T::timings)
    # inspired by Levintal
    tol = eps(Float32)

    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx;
    i₋ = T.past_not_future_and_mixed_idx;

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo;
    n = T.nVars
    n̄ = n₋ + n + n₊ + nₑ
    nₑ₋ = n₋ + 1 + nₑ


    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] |> sparse
    droptol!(𝐒₁,tol)

    # set up vector to capture volatility effect
    redu = sparsevec(nₑ₋ - nₑ + 1:nₑ₋, 1)
    redu_idxs = findnz(ℒ.kron(redu, redu))[1]
    𝛔 = @views sparse(redu_idxs[Int.(range(1,nₑ^2,nₑ))], fill(n₋ * (nₑ₋ + 1) + 1, nₑ), 1, nₑ₋^2, nₑ₋^2)


    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)];

    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]


    ∇₁₊ = @views sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])

    B = sparse(∇₁₊𝐒₁➕∇₁₀ \ ∇₁₊)
    droptol!(B,tol)
    
    # compression matrices for third order
    colls3 = [nₑ₋^2 * (i-1) + nₑ₋ * (k-1) + l for i in 1:nₑ₋ for k in 1:i for l in 1:k]
    𝐂₃ = sparse(colls3, 1:length(colls3) , 1)
    
    idxs = []
    for k in 1:nₑ₋
        for j in 1:nₑ₋
            for i in 1:nₑ₋
                sorted_ids = sort([k,j,i])
                push!(idxs, (sorted_ids[3] - 1) * nₑ₋ ^ 2 + (sorted_ids[2] - 1) * nₑ₋ + sorted_ids[1])
            end
        end
    end
    
    𝐔₃ = 𝐂₃' * sparse(idxs,1:nₑ₋ ^ 3, 1)
    
    
    # permutation matrices
    M = reshape(1:nₑ₋^3,1,nₑ₋,nₑ₋,nₑ₋)
    𝐏 = @views sparse(reshape(spdiagm(ones(nₑ₋^3))[:,PermutedDimsArray(M,[1, 4, 2, 3])],nₑ₋^3,nₑ₋^3)
                           + reshape(spdiagm(ones(nₑ₋^3))[:,PermutedDimsArray(M,[1, 2, 4, 3])],nₑ₋^3,nₑ₋^3)
                           + reshape(spdiagm(ones(nₑ₋^3))[:,PermutedDimsArray(M,[1, 2, 3, 4])],nₑ₋^3,nₑ₋^3))
    

    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [(𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
            𝐒₂
            zeros(n₋ + nₑ, nₑ₋^2)];
        
    𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:] 
             zeros(n₋ + n + nₑ, nₑ₋^2)];
    
    𝐗₃ = -∇₃ * ℒ.kron(ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋), ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋)
    
    𝐏₁ₗ  = @views sparse(spdiagm(ones(n̄^3))[vec(permutedims(reshape(1:n̄^3,n̄,n̄,n̄),(1,3,2))),:])
    𝐏₁ᵣ  = @views sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(1,3,2)))])
    𝐏₂ₗ  = @views sparse(spdiagm(ones(n̄^3))[vec(permutedims(reshape(1:n̄^3,n̄,n̄,n̄),(3,1,2))),:])
    𝐏₂ᵣ  = @views sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(3,1,2)))])

    tmpkron = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * 𝛔)
    out = - ∇₃ * tmpkron - ∇₃ * 𝐏₁ₗ * tmpkron * 𝐏₁ᵣ - ∇₃ * 𝐏₂ₗ * tmpkron * 𝐏₂ᵣ
    𝐗₃ += out
    
    
    
    tmp𝐗₃ = -∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎) 
    
    𝐏₁ₗ = sparse(spdiagm(ones(nₑ₋^3))[vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(2,1,3))),:])
    𝐏₁ᵣ = sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(2,1,3)))])

    tmpkron1 = -∇₂ *  ℒ.kron(𝐒₁₊╱𝟎,𝐒₂₊╱𝟎)
    tmpkron2 = ℒ.kron(𝛔,𝐒₁₋╱𝟏ₑ)
    out2 = tmpkron1 * tmpkron2 +  tmpkron1 * 𝐏₁ₗ * tmpkron2 * 𝐏₁ᵣ
    
    𝐗₃ += (tmp𝐗₃ + out2 + -∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₂₊╱𝟎 * 𝛔)) * 𝐏# |> findnz
    
    𝐗₃ += @views -∇₁₊ * 𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, [𝐒₂[i₋,:] ; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]) * 𝐏
    droptol!(𝐗₃,tol)
    
    
    X = sparse(∇₁₊𝐒₁➕∇₁₀ \ 𝐗₃ * 𝐂₃)
    droptol!(X,tol)
    
    
    𝐏₁ₗ = @views sparse(spdiagm(ones(nₑ₋^3))[vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(1,3,2))),:])
    𝐏₁ᵣ = @views sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(1,3,2)))])
    𝐏₂ₗ = @views sparse(spdiagm(ones(nₑ₋^3))[vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(3,1,2))),:])
    𝐏₂ᵣ = @views sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(3,1,2)))])

    tmpkron = ℒ.kron(𝐒₁₋╱𝟏ₑ,𝛔)
    
    C = 𝐔₃ * tmpkron + 𝐔₃ * 𝐏₁ₗ * tmpkron * 𝐏₁ᵣ + 𝐔₃ * 𝐏₂ₗ * tmpkron * 𝐏₂ᵣ
    C += 𝐔₃ * ℒ.kron(𝐒₁₋╱𝟏ₑ,ℒ.kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ))
    C *= 𝐂₃
    droptol!(C,tol)
    
    
    A = spdiagm(ones(n))
    lm = LinearMap{Float64}(x -> A * reshape(x,size(X)) - B * reshape(x,size(X)) * C, size(X)[1] * size(X)[2])
    
    𝐒₃ = sparse(reshape(ℐ.bicgstabl(lm, vec(-X)),size(X))) * 𝐔₃ # fastest
    droptol!(𝐒₃,tol)
    
    
    return 𝐒₃
end





function irf(state_update::Function, initial_state::Vector{Float64}, T::timings; 
    periods::Int = 40, 
    shocks::Union{Symbol_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Symbol_input = :all, 
    negative_shock::Bool = false)

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == T.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        periods += size(shocks)[2]

        shock_history = zeros(T.nExo, periods)

        shock_history[:,1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks isa KeyedArray{Float64}
        shock_input = axiskeys(shocks)[1]

        periods += size(shocks)[2]

        @assert length(setdiff(shock_input, T.exo)) == 0 "Provided shocks which are not part of the model."

        shock_history = zeros(T.nExo, periods)

        shock_history[indexin(shock_input,T.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,T)
    end

    var_idx = parse_variables_input_to_index(variables, T)

    if shocks == :simulate
        shock_history = randn(T.nExo,periods)

        Y = zeros(T.nVars,periods,1)
        Y[:,1,1] = state_update(initial_state,shock_history[:,1])

        for t in 1:periods-1
            Y[:,t+1,1] = state_update(Y[:,t,1],shock_history[:,t+1])
        end

        return KeyedArray(Y[var_idx,:,:];  Variables = T.var[var_idx], Periods = 1:periods, Shocks = [:simulate])
    elseif shocks == :none
        Y = zeros(T.nVars,periods,1)

        shck = T.nExo == 0 ? Vector{Float64}(undef, 0) : zeros(T.nExo)
        
        Y[:,1,1] = state_update(initial_state,shck)

        for t in 1:periods-1
            Y[:,t+1,1] = state_update(Y[:,t,1],shck)
        end

        return KeyedArray(Y[var_idx,:,:];  Variables = T.var[var_idx], Periods = 1:periods, Shocks = [:none])
    else
        Y = zeros(T.nVars,periods,length(shock_idx))

        for ii in shock_idx
            if shocks != :simulate && shocks isa Symbol_input
                shock_history = zeros(T.nExo,periods)
                shock_history[ii,1] = negative_shock ? -1 : 1
            end

            Y[:,1,ii] = state_update(initial_state,shock_history[:,1])

            for t in 1:periods-1
                Y[:,t+1,ii] = state_update(Y[:,t,ii],shock_history[:,t+1])
            end
        end

        return KeyedArray(Y[var_idx,:,:];  Variables = T.var[var_idx], Periods = 1:periods, Shocks = shocks isa Symbol_input ? [T.exo[shock_idx]...] : [:Shock_matrix])
    end
end



function girf(state_update::Function, T::timings; 
    periods::Int = 40, 
    shocks::Union{Symbol_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Symbol_input = :all, 
    negative_shock::Bool = false, 
    warmup_periods::Int = 100, 
    draws::Int = 50, 
    iterations_to_steady_state::Int = 500)

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == T.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        periods += size(shocks)[2]

        shock_history = zeros(T.nExo, periods)

        shock_history[:,1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks isa KeyedArray{Float64}
        shock_input = axiskeys(shocks)[1]

        periods += size(shocks)[2]

        @assert length(setdiff(shock_input, T.exo)) == 0 "Provided shocks which are not part of the model."

        shock_history = zeros(T.nExo, periods)

        shock_history[indexin(shock_input,T.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,T)
    end

    var_idx = parse_variables_input_to_index(variables, T)

    Y = zeros(T.nVars,periods,length(shock_idx))

    initial_state = zeros(T.nVars)

    for warm in 1:iterations_to_steady_state
        initial_state = state_update(initial_state, zeros(T.nExo))
    end

    for ii in shock_idx
        for draw in 1:draws
            for i in 1:warmup_periods
                initial_state = state_update(initial_state, randn(T.nExo))
            end

            Y1 = zeros(T.nVars, periods)
            Y2 = zeros(T.nVars, periods)

            baseline_noise = randn(T.nExo)

            if shocks != :simulate && shocks isa Symbol_input
                shock_history = zeros(T.nExo,periods)
                shock_history[ii,1] = negative_shock ? -1 : 1
            end

            Y1[:,1] = state_update(initial_state, baseline_noise)
            Y2[:,1] = state_update(initial_state, baseline_noise + shock_history[:,1])

            for t in 1:periods-1
                baseline_noise = randn(T.nExo)

                Y1[:,t+1] = state_update(Y1[:,t],baseline_noise)
                Y2[:,t+1] = state_update(Y2[:,t],baseline_noise + shock_history[:,t])
            end

            Y[:,:,ii] += Y2 - Y1
        end
        Y[:,:,ii] /= draws
    end
    
    return KeyedArray(Y[var_idx,:,:];  Variables = T.var[var_idx], Periods = 1:periods, Shocks = shocks isa Symbol_input ? [T.exo[shock_idx]...] : [:Shock_matrix])
end


function parse_variables_input_to_index(variables::Symbol_input, T::timings)
    if variables == :all
        return indexin(setdiff(T.var,T.aux),sort(union(T.var,T.aux,T.exo_present)))
        # return indexin(setdiff(setdiff(T.var,T.exo_present),T.aux),sort(union(T.var,T.aux,T.exo_present)))
    elseif variables isa Matrix{Symbol}
        if !issubset(variables,T.var)
            return @warn "Following variables are not part of the model: " * string.(setdiff(variables,T.var))
        end
        return getindex(1:length(T.var),convert(Vector{Bool},vec(sum(variables .== T.var,dims= 2))))
    elseif variables isa Vector{Symbol}
        if !issubset(variables,T.var)
            return @warn "Following variables are not part of the model: " * string.(setdiff(variables,T.var))
        end
        return getindex(1:length(T.var),convert(Vector{Bool},vec(sum(reshape(variables,1,length(variables)) .== T.var,dims= 2))))
    elseif variables isa Tuple{Symbol,Vararg{Symbol}}
        if !issubset(variables,T.var)
            return @warn "Following variables are not part of the model: " * string.(setdiff(variables,T.var))
        end
        return getindex(1:length(T.var),convert(Vector{Bool},vec(sum(reshape(collect(variables),1,length(variables)) .== T.var,dims= 2))))
    elseif variables isa Symbol
        if !issubset([variables],T.var)
            return @warn "Following variable is not part of the model: " * string(setdiff([variables],T.var)[1])
        end
        return getindex(1:length(T.var),variables .== T.var)
    else
        return @warn "Invalid argument in variables"
    end
end


function parse_shocks_input_to_index(shocks::Symbol_input, T::timings)
    if shocks == :all
        shock_idx = 1:T.nExo
    elseif shocks == :none
        shock_idx = 1
    elseif shocks == :simulate
        shock_idx = 1
    elseif shocks isa Matrix{Symbol}
        if !issubset(shocks,T.exo)
            return @warn "Following shocks are not part of the model: " * string.(setdiff(shocks,T.exo))
        end
        shock_idx = getindex(1:T.nExo,convert(Vector{Bool},vec(sum(shocks .== T.exo,dims= 2))))
    elseif shocks isa Vector{Symbol}
        if !issubset(shocks,T.exo)
            return @warn "Following shocks are not part of the model: " * string.(setdiff(shocks,T.exo))
        end
        shock_idx = getindex(1:T.nExo,convert(Vector{Bool},vec(sum(reshape(shocks,1,length(shocks)) .== T.exo, dims= 2))))
    elseif shocks isa Tuple{Symbol, Vararg{Symbol}}
        if !issubset(shocks,T.exo)
            return @warn "Following shocks are not part of the model: " * string.(setdiff(shocks,T.exo))
        end
        shock_idx = getindex(1:T.nExo,convert(Vector{Bool},vec(sum(reshape(collect(shocks),1,length(shocks)) .== T.exo,dims= 2))))
    elseif shocks isa Symbol
        if !issubset([shocks],T.exo)
            return @warn "Following shock is not part of the model: " * string(setdiff([shocks],T.exo)[1])
        end
        shock_idx = getindex(1:T.nExo,shocks .== T.exo)
    else
        return @warn "Invalid argument in shocks"
    end
end






function parse_algorithm_to_state_update(algorithm::Symbol, 𝓂::ℳ)
    if :linear_time_iteration == algorithm
        state_update = 𝓂.solution.perturbation.linear_time_iteration.state_update

    elseif algorithm ∈ [:riccati, :first_order]
        state_update = 𝓂.solution.perturbation.first_order.state_update

    elseif :second_order == algorithm
        state_update = 𝓂.solution.perturbation.second_order.state_update

    elseif :third_order == algorithm
        state_update = 𝓂.solution.perturbation.third_order.state_update
    end

    return state_update
end


function calculate_covariance(parameters::Vector{<: Number}, 𝓂::ℳ; verbose = false)
    SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, false, verbose)
    
	∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)

    sol = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    covar_raw = calculate_covariance_forward(sol,T = 𝓂.timings, subset_indices = collect(1:𝓂.timings.nVars))

    return covar_raw, sol , ∇₁, SS_and_pars
end

function calculate_covariance_forward(𝑺₁::AbstractMatrix{<: Number}; T::timings, subset_indices::Vector{Int64})
    A = @views 𝑺₁[subset_indices,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(subset_indices)))[indexin(T.past_not_future_and_mixed_idx,subset_indices),:]
    C = @views 𝑺₁[subset_indices,T.nPast_not_future_and_mixed+1:end]
    
    CC = C * C'

    lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))
    
    reshape(ℐ.bicgstabl(lm, vec(-CC)), size(CC))
end


function calculate_covariance_forward(𝑺₁::AbstractMatrix{ℱ.Dual{Z,S,N}}; T::timings = T, subset_indices::Vector{Int64} = subset_indices) where {Z,S,N}
    # unpack: AoS -> SoA
    𝑺₁̂ = ℱ.value.(𝑺₁)
    # you can play with the dimension here, sometimes it makes sense to transpose
    ps = mapreduce(ℱ.partials, hcat, 𝑺₁)'

    # get f(vs)
    val = calculate_covariance_forward(𝑺₁̂, T = T, subset_indices = subset_indices)

    # get J(f, vs) * ps (cheating). Write your custom rule here
    B = ℱ.jacobian(x -> calculate_covariance_conditions(x, val, T = T, subset_indices = subset_indices), 𝑺₁̂)
    A = ℱ.jacobian(x -> calculate_covariance_conditions(𝑺₁̂, x, T = T, subset_indices = subset_indices), val)

    jvp = (-A \ B) * ps

    # pack: SoA -> AoS
    return reshape(map(val, eachrow(jvp)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end,size(val))
end


function calculate_covariance_conditions(𝑺₁::AbstractMatrix{<: Number}, covar::AbstractMatrix{<: Number}; T::timings, subset_indices::Vector{Int64})
    A = @views 𝑺₁[subset_indices,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(subset_indices)))[@ignore_derivatives(indexin(T.past_not_future_and_mixed_idx,subset_indices)),:]
    C = @views 𝑺₁[subset_indices,T.nPast_not_future_and_mixed+1:end]
    
    A * covar * A' + C * C' - covar
end


calculate_covariance_AD(sol; T, subset_indices) = ImplicitFunction(sol->calculate_covariance_forward(sol, T=T, subset_indices = subset_indices), (x,y)->calculate_covariance_conditions(x,y,T=T, subset_indices = subset_indices))
# calculate_covariance_AD(sol, T = 𝓂.timings, subset_indices = Int64[observables_and_states...])

function calculate_kalman_filter_loglikelihood(𝓂::ℳ, data::AbstractArray{Float64}, observables::Vector{Symbol}; parameters = nothing, verbose = false, tol = eps())
    @assert length(observables) == size(data)[1] "Data columns and number of observables are not identical. Make sure the data contains only the selected observables."
    @assert length(observables) <= 𝓂.timings.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."

    @ignore_derivatives sort!(observables)

    solve!(𝓂, verbose = verbose)

    # data = data(observables,:) .- collect(𝓂.SS_solve_func(𝓂.parameter_values, 𝓂.SS_init_guess,𝓂)[observables])

    SS_and_pars, solution_error = 𝓂.SS_solve_func(isnothing(parameters) ? 𝓂.parameter_values : parameters, 𝓂, true, verbose)
    
    if solution_error > tol
        return -1e6
    end

    NSSS_labels = @ignore_derivatives [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]

    obs_indices = @ignore_derivatives indexin(observables,NSSS_labels)

    data_in_deviations = collect(data(observables)) .- SS_and_pars[obs_indices]

    # 𝓂.solution.non_stochastic_steady_state = ℱ.value.(SS_and_pars)

	∇₁ = calculate_jacobian(isnothing(parameters) ? 𝓂.parameter_values : parameters, SS_and_pars, 𝓂)

    sol = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    observables_and_states = @ignore_derivatives sort(union(𝓂.timings.past_not_future_and_mixed_idx,indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))))

    A = @views sol[observables_and_states,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(𝓂.timings.past_not_future_and_mixed_idx,observables_and_states)),:]
    B = @views sol[observables_and_states,𝓂.timings.nPast_not_future_and_mixed+1:end]

    C = @views ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))),observables_and_states)),:]

    𝐁 = B * B'

    # Gaussian Prior

    calculate_covariance_ = calculate_covariance_AD(sol, T = 𝓂.timings, subset_indices = Int64[observables_and_states...])

    P = calculate_covariance_(sol)
    # P = reshape((ℒ.I - ℒ.kron(A, A)) \ reshape(𝐁, prod(size(A)), 1), size(A))
    u = zeros(length(observables_and_states))
    # u = SS_and_pars[sort(union(𝓂.timings.past_not_future_and_mixed,observables))] |> collect
    z = C * u
    
    loglik = 0.0

    for t in 1:size(data)[2]
        v = data_in_deviations[:,t] - z

        F = C * P * C'

        # F = (F + F') / 2

        # loglik += log(max(eps(),ℒ.det(F))) + v' * ℒ.pinv(F) * v
        # K = P * C' * ℒ.pinv(F)

        # loglik += log(max(eps(),ℒ.det(F))) + v' / F  * v
        loglik += log(ℒ.det(F)) + v' / F  * v
        K = P * C' / F

        P = A * (P - K * C * P) * A' + 𝐁

        u = A * (u + K * v)
        
        z = C * u 
    end

    return -(loglik + length(data) * log(2 * 3.141592653589793)) / 2 # otherwise conflicts with model parameters assignment
end



end