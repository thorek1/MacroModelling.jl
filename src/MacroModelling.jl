module MacroModelling


import DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF, TYPEDSIGNATURES, TYPEDFIELDS
using StatsFuns, SpecialFunctions
import SymPy: @vars, solve, subs, Sym
import ForwardDiff as ℱ 
import SparseArrays: SparseMatrixCSC, sparse, spzeros, droptol!, sparsevec, spdiagm, findnz#, sparse!
import LinearAlgebra as ℒ
using Optimization, OptimizationNLopt
import BlockTriangularForm
import Subscripts: super, sub
using IterativeSolvers, LinearMaps
using ComponentArrays
# using NamedArrays
using AxisKeys

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

# Type definitions
Symbol_input = Union{Symbol,Vector{Symbol},Matrix{Symbol},Tuple{Symbol,Vararg{Symbol}}}

# Imports
include("structures.jl")
include("macros.jl")
include("get_functions.jl")
include("plotting.jl")


export @model, @parameters, solve!
export plot_irfs, plot_irf, plot_IRF, plot, plot_simulations
export get_irfs, get_irf, get_IRF, simulate
export get_solution, get_first_order_solution, get_perturbation_solution
export get_steady_state, get_SS, get_non_stochastic_steady_state, get_stochastic_steady_state
export get_moments
export calculate_jacobian, calculate_hessian, calculate_third_order_derivatives
export calculate_first_order_solution, calculate_second_order_solution, calculate_third_order_solution#, calculate_jacobian_manual, calculate_jacobian_sparse, calculate_jacobian_threaded
export calculate_kalman_filter_loglikelihood


# Internal
export irf, girf
# export riccati_forward, block_solver, remove_redundant_SS_vars!, write_parameters_input!


# StatsFuns
norminvcdf(p::Number) = -erfcinv(2*p) * sqrt2
norminv(p::Number) = norminvcdf(p)
pnorm(p::Number) = normcdf(p)
dnorm(p::Number) = normpdf(p)
qnorm(p::Number) = norminvcdf(p)





Base.show(io::IO, 𝓂::ℳ) = println(io, 
                "Model: ",𝓂.model_name, 
                "\nVariables: ",length(𝓂.var),
                "\nShocks: ",length(𝓂.exo),
                "\nParameters: ",length(𝓂.par),
                "\nAuxiliary variables: ",length(𝓂.exo_present) + length(𝓂.aux),
                # "\nCalibration equations: ",length(𝓂.calibration_equations),
                # "\nVariable bounds (upper,lower,any): ",sum(𝓂.upper_bounds .< Inf),", ",sum(𝓂.lower_bounds .> -Inf),", ",length(𝓂.bounds),
                # "\nNon-stochastic-steady-state found: ",!𝓂.solution.NSSS_outdated
                )


function create_symbols_eqs!(𝓂::ℳ)
    # create symbols in module scope
    symbols_in_equation = union(𝓂.var,𝓂.par,𝓂.parameters,𝓂.exo,𝓂.dynamic_variables,𝓂.nonnegativity_auxilliary_vars)#,𝓂.dynamic_variables_future)
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
                # map(x->eval(:($x)),𝓂.dyn_equations),
                # map(x->eval(:($x)),𝓂.dyn_equations_future),

                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift_var_present_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift_var_past_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift_var_future_list),

                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift2_var_past_list),

                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_var_present_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_var_past_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_var_future_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_ss_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_exo_list),

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
            
            if length(soll) == 0 || soll == Sym[0] # take out variable if it is redundant from that euation only
                push!(symbolics.var_redundant_list[i],var_to_solve)
                ss_equations[i] = ss_equations[i].subs(var_to_solve,1)
            end

        end
    end

end




function solve_steady_state!(𝓂::ℳ,symbolic_SS, symbolics::symbolics)
    unknowns = union(symbolics.var,symbolics.nonnegativity_auxilliary_vars,symbolics.calibration_equations_parameters)

    if length(unknowns) > length(symbolics.ss_equations) + length(symbolics.calibration_equations)
        println("Unable to solve steady state. More unknowns than equations.")
    end

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

    ss_equations = vcat(symbolics.ss_equations,symbolics.calibration_equations) .|> Sym
    # println(ss_equations)

    SS_solve_func = []

    atoms_in_equations = Set()
    atoms_in_equations_list = []
    relevant_pars_across = []

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
                push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
                push!(atoms_in_equations_list,[])
            else

                push!(𝓂.solved_vars,Symbol(var_to_solve))
                push!(𝓂.solved_vals,Meta.parse(string(soll[1])))
                
                # atoms = reduce(union,soll[1].atoms())
                [push!(atoms_in_equations, a) for a in soll[1].atoms()]
                push!(atoms_in_equations_list, Set(Symbol.(soll[1].atoms())))
                # println(atoms_in_equations)
                # push!(atoms_in_equations, soll[1].atoms())

                push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))

            end

            # push!(single_eqs,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
            # solve symbolically
        else

            vars_to_solve = collect(unknowns)[vars[:,vars[2,:] .== n][1,:]]

            # println(var_to_solve)
            # println(Sym(ss_equations[eqs[:,eqs[2,:] .== n][1,:]]))
            eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

            numerical_sol = false
            
            if symbolic_SS
                soll = try solve(Sym(eqs_to_solve),vars_to_solve)
                # soll = try solve(Sym(eqs_to_solve),var_order)#,check=false,force = true,manual=true)
                catch
                end

                # println(soll)
                if isnothing(soll)
                    println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
                    numerical_sol = true
                    # continue
                elseif length(soll) == 0
                    println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
                    numerical_sol = true
                    # continue
                elseif length(intersect(vars_to_solve,reduce(union,map(x->x.atoms(),collect(soll[1]))))) > 0
                    println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
                    numerical_sol = true
                    # println("Could not solve for: ",intersect(var_list,reduce(union,map(x->x.atoms(),solll)))...)
                    # break_ind = true
                    # break
                else
                    println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " symbolically.")

                    # relevant_pars = reduce(union,vcat(𝓂.par_list,𝓂.par_calib_list)[eqs[:,eqs[2,:] .== n][1,:]])
                    # relevant_pars = reduce(union,map(x->x.atoms(),collect(soll[1])))
                    atoms = reduce(union,map(x->x.atoms(),collect(soll[1])))
                    # println(atoms)
                    [push!(atoms_in_equations, a) for a in atoms]
                    
                    for k in 1:length(vars_to_solve)
                        push!(𝓂.solved_vars,Symbol(vars_to_solve[k]))
                        push!(𝓂.solved_vals,Meta.parse(string(soll[1][k]))) #using convert(Expr,x) leads to ugly expressions

                        push!(atoms_in_equations_list, Set(Symbol.(soll[1][k].atoms())))
                        push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
                    end
                end


            end
                
            # try symbolically and use numerical if it does not work
            if numerical_sol || !symbolic_SS
                if !symbolic_SS
                    println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " numerically.")
                end
                push!(𝓂.solved_vars,Symbol.(collect(unknowns)[vars[:,vars[2,:] .== n][1,:]]))
                push!(𝓂.solved_vals,Meta.parse.(string.(ss_equations[eqs[:,eqs[2,:] .== n][1,:]])))
                
                syms_in_eqs = Set(Symbol.(Sym(ss_equations[eqs[:,eqs[2,:] .== n][1,:]]).atoms()))
                push!(atoms_in_equations_list,setdiff(syms_in_eqs, 𝓂.solved_vars[end]))

                calib_pars = []
                calib_pars_input = []
                relevant_pars = reduce(union,vcat(𝓂.par_list_aux_SS,𝓂.par_calib_list)[eqs[:,eqs[2,:] .== n][1,:]])
                relevant_pars_across = union(relevant_pars_across,relevant_pars)
                # println(relevant_pars)
                iii = 1
                for i in 1:length(𝓂.parameters) 
                    valss   = 𝓂.parameter_values[i]
                    parss = 𝓂.parameters[i]
                    # println(:($parss))
                    if :($parss) ∈ relevant_pars
                        push!(calib_pars,:($parss = inputs[$iii]))
                        push!(calib_pars_input,:($parss))
                        iii += 1
                        # push!(calib_pars,:($parss = $valss))
                    end
                end


                guess = []
                result = []
                sorted_vars = sort(setdiff(𝓂.solved_vars[end],𝓂.nonnegativity_auxilliary_vars))
                for i in 1:length(sorted_vars) 
                    parss = sorted_vars[i]
                    push!(guess,:($parss = guess[$i]))
                    push!(result,:($parss = sol[$i]))
                end

                other_vars = []
                other_vars_input = []
                other_vars_inverse = []
                other_vrs = intersect(setdiff(union(𝓂.var,𝓂.calibration_equations_parameters),sort(𝓂.solved_vars[end])),syms_in_eqs)
                # println(other_vrs)
                for k in 1:length(other_vrs)
                    # var_idx = [[findfirst(x->x==y,𝓂.var) for y in other_vrs[k]]]
                    var_idx = findfirst(x -> x == other_vrs[k], union(𝓂.var,𝓂.calibration_equations_parameters))
                    push!(other_vars,:($(other_vrs[k]) = inputs[$iii]))
                    push!(other_vars_input,:($(other_vrs[k])))
                    iii += 1
                    push!(other_vars_inverse,:(𝓂.SS_init_guess[$var_idx] = $(other_vrs[k])))
                end
                
                # separate out auxilliary variables (nonnegativity)
                nnaux = []
                nnaux_error = []
                push!(nnaux_error, :(aux_error = 0))
                solved_vals = []
                
                for val in 𝓂.solved_vals[end]
                    if (val.args[1] == :+ && val.args[3] ∈ 𝓂.nonnegativity_auxilliary_vars) 
                        push!(nnaux,:($(val.args[3]) = max(eps(),-$(val.args[2]))))
                        push!(nnaux_error, :(aux_error += min(0.0,-$(val.args[2]))))
                    elseif (val.args[1] == :- && val.args[2] ∈ 𝓂.nonnegativity_auxilliary_vars) 
                        push!(nnaux,:($(val.args[2]) = max(eps(),$(val.args[3]))))
                        push!(nnaux_error, :(aux_error += min(0.0,$(val.args[3]))))
                    else
                        push!(solved_vals,val)
                    end
                end

                # sort nnaux vars so that they enter in right order. avoid using a variable before it is declared
                if length(nnaux) > 1
                    function get_symbols(ex)
                        list = Set()
                        postwalk(x -> x isa Symbol ? push!(list, x) : x, ex)
                        return list
                    end

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
                end


                funcs = :(function block(guess::Vector{Float64},inputs::Vector{Float64})
                        $(guess...) 
                        $(calib_pars...) # add those variables which were previously solved and are used in the equations
                        $(other_vars...) # take only those that appear in equations - DONE
                        
                        $(nnaux...)
                        return [$(solved_vals...)]
                    end)

                push!(solved_vals,:(aux_error))

                funcs_optim = :(function block(guess::Vector{Float64},inputs::Vector{Float64})
                    $(guess...) 
                    $(calib_pars...) # add those variables which were previously solved and are used in the equations
                    $(other_vars...) # take only those that appear in equations - DONE

                    $(nnaux...)
                    $(nnaux_error...)
                    return sum(abs2,[$(solved_vals...)])
                end)
            
                𝓂.SS_init_guess = [fill(1,length(𝓂.var)); fill(.5, length(𝓂.calibration_equations_parameters))]
                
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
                push!(SS_solve_func,:(𝓂.SS_init_guess = initial_guess))
                # push!(SS_solve_func,:($(other_vars_inverse...)))
                push!(SS_solve_func,:(f = OptimizationFunction(𝓂.ss_solve_blocks_optim[$(n_block)], Optimization.AutoForwardDiff())))
                # push!(SS_solve_func,:(prob = OptimizationProblem(f, 𝓂.SS_init_guess, 𝓂, lb = [$(lbs...)], ub = [$(ubs...)])))
                push!(SS_solve_func,:(inits = max.(lbs,min.(ubs,𝓂.SS_init_guess[$([findfirst(x->x==y,union(𝓂.var,𝓂.calibration_equations_parameters)) for y in sorted_vars])]))))
                
                
                # push!(SS_solve_func,:(𝓂.SS_init_guess[$([findfirst(x -> x==y, union(𝓂.var,𝓂.calibration_equations_parameters)) for y in sorted_vars])] = sol = block_solver([$(calib_pars_input...),$(other_vars_input...)], 
                push!(SS_solve_func,:(sol = block_solver([$(calib_pars_input...),$(other_vars_input...)], 
                        $(n_block), 
                        𝓂.ss_solve_blocks[$(n_block)], 
                        𝓂.SS_optimizer, 
                        f, 
                        inits, 
                        lbs, 
                        ubs)))
                        
                push!(SS_solve_func,:($(result...)))            

                push!(𝓂.ss_solve_blocks,@RuntimeGeneratedFunction(funcs))
                push!(𝓂.ss_solve_blocks_optim,@RuntimeGeneratedFunction(funcs_optim))
                
                n_block += 1
            end
        end
        n -= 1
    end

    unknwns = Symbol.(collect(unknowns))

    parameters_in_equations = []

    for i in 1:length(𝓂.parameters) 
        # valss   = 𝓂.parameter_values[i]
        parss = 𝓂.parameters[i]
        # println(:($parss))
        if parss ∈ union(Symbol.(atoms_in_equations),relevant_pars_across)
            push!(parameters_in_equations,:($parss = parameters[$i]))
            # push!(calib_pars,:($parss = $valss))
        end
    end
    
    dependencies = []
    for i in 1:length(atoms_in_equations_list)
        push!(dependencies,𝓂.solved_vars[i] => intersect(atoms_in_equations_list[i],union(𝓂.var,𝓂.parameters)))
    end
    push!(dependencies,:SS_relevant_calibration_parameters => intersect(reduce(union,atoms_in_equations_list),𝓂.parameters))
    # print(dependencies)
    𝓂.SS_dependencies = dependencies
    # push!(SS_solve_func,:(return [$(unknwns...)]))

    
    dyn_exos = []
    for dex in union(𝓂.exo_past,𝓂.exo_future)
        push!(dyn_exos,:($dex = 0))
    end

    push!(SS_solve_func,:($(dyn_exos...)))

    push!(SS_solve_func,:(SS_init_guess = ([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)])))
    # push!(SS_solve_func,:(print(typeof(SS_init_guess))))
    push!(SS_solve_func,:(𝓂.SS_init_guess = typeof(SS_init_guess) == Vector{Float64} ? SS_init_guess : ℱ.value.(SS_init_guess)))
    

    # push!(SS_solve_func,:(return ComponentArray(non_stochastic_steady_state = [$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...)], calibrated_parameters = length([$(𝓂.calibration_equations_parameters...)]) > 0 ? [$(𝓂.calibration_equations_parameters...)] : 0.0)))
    push!(SS_solve_func,:(return ComponentVector([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]))))

    solve_exp = :(function solve_SS(parameters::Vector{Real}, initial_guess::Vector{Real}, 𝓂::ℳ)
                            $(parameters_in_equations...)
                            $(SS_solve_func...)
                            end)

    # println(solve_exp)

    𝓂.SS_solve_func = @RuntimeGeneratedFunction(solve_exp)

    return nothing
end


function block_solver(inputs::Vector{Float64}, 
                        n_block::Int, 
                        ss_solve_blocks::Function, 
                        SS_optimizer, 
                        f::OptimizationFunction, 
                        guess::Vector{Float64}, 
                        lbs::Vector{Float64}, 
                        ubs::Vector{Float64})
    
    prob = OptimizationProblem(f, guess, inputs, lb = lbs, ub = ubs)
    sol = solve(prob, SS_optimizer(), local_maxiters=10000)
    
    if (sol.minimum > eps()) | (sum(abs2,ss_solve_blocks(sol,inputs)) > eps())
                    println("Block: ",n_block," - Solution not found. Trying optimizer: LN_BOBYQA.")
                    sol = solve(prob, NLopt.LN_BOBYQA(), local_maxiters=10000)
                end
    
    if (sol.minimum > eps()) | (sum(abs2,ss_solve_blocks(sol,inputs)) > eps())
        println("Block: ",n_block," - Local solution not found. Trying global solution.")
        sol = solve(prob, NLopt.GD_MLSL_LDS(), local_method = NLopt.LD_LBFGS(), local_maxiters=10000, population = length(ubs))
    end
    
    if (sol.minimum > eps()) | (sum(abs2,ss_solve_blocks(sol,inputs)) > eps())
        println("Block: ",n_block," - No solution found. Trying with positive domain.")
        
        inits = max.(max.(lbs,eps()),min.(ubs,guess))
        prob = OptimizationProblem(f, guess, inits, lb = max.(lbs,eps()), ub = ubs)
        sol = solve(prob, SS_optimizer(), local_maxiters=10000)
    end
    
    if (sol.minimum > eps()) | (sum(abs2,ss_solve_blocks(sol,inputs)) > eps())
        println("Block: ",n_block," - Solution not found. Trying optimizer: L-BFGS.")
        sol = solve(prob, NLopt.LN_BOBYQA() , local_maxiters=10000)
    end
    
    if (sol.minimum > eps()) | (sum(abs2,ss_solve_blocks(sol,inputs)) > eps())
        println("Block: ",n_block," - Local solution not found. Trying global solution.")
        sol = solve(prob, NLopt.GD_MLSL_LDS(), local_method = NLopt.LD_LBFGS(), local_maxiters=10000, population = length(ubs))
    end
    
    if (sol.minimum > eps()) | (sum(abs2,ss_solve_blocks(sol,inputs)) > eps())
        error("Block: ",n_block," - No solution found. Consider changing bounds.")
    end

    return sol.u
end




function block_solver(inputs::Vector{ℱ.Dual{Z,S,N}}, 
    n_block::Int, 
    ss_solve_blocks::Function, 
    SS_optimizer, 
    f::OptimizationFunction, 
    guess::Vector{Float64}, 
    lbs::Vector{Float64}, 
    ubs::Vector{Float64}) where {Z,S,N}

    # unpack: AoS -> SoA
    inp = ℱ.value.(inputs)

    # you can play with the dimension here, sometimes it makes sense to transpose
    ps = mapreduce(ℱ.partials, hcat, inputs)'

    # get f(vs)
    val = block_solver(inp, 
                        n_block, 
                        ss_solve_blocks, 
                        SS_optimizer, 
                        f, 
                        guess, 
                        lbs, 
                        ubs)

    # get J(f, vs) * ps (cheating). Write your custom rule here
    B = ℱ.jacobian(x -> ss_solve_blocks(val, x), inp)
    A = ℱ.jacobian(x -> ss_solve_blocks(x, inp), val)

    jvp = (-A \ B) * ps

    # pack: SoA -> AoS
    return reshape(map(val, eachrow(jvp)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end,size(val))
end






function solve!(𝓂::ℳ; 
    parameters = nothing, 
    dynamics::Bool = false, 
    algorithm::Symbol = :first_order, 
    symbolic_SS::Bool = false)

    if algorithm == :all_predefined
        algorithm = 𝓂.solution.algorithm
    elseif algorithm ∈ [:linear_time_iteration, :dynare, :first_order, :riccati]
        𝓂.solution.algorithm = union(𝓂.solution.algorithm,[algorithm])
    end


    if !𝓂.solution.functions_written 
        # consolidate bounds info
        double_info = intersect(𝓂.bounds⁺,𝓂.bounded_vars)
        𝓂.lower_bounds[indexin(double_info,𝓂.bounded_vars)] = max.(eps(),𝓂.lower_bounds[indexin(double_info,𝓂.bounded_vars)])

        new_info = setdiff(𝓂.bounds⁺,𝓂.bounded_vars)
        𝓂.bounded_vars = vcat(𝓂.bounded_vars,new_info)
        𝓂.lower_bounds = vcat(𝓂.lower_bounds,fill(eps(),length(new_info)))
        𝓂.upper_bounds = vcat(𝓂.upper_bounds,fill(Inf,length(new_info)))


        symbolics = create_symbols_eqs!(𝓂)
        remove_redundant_SS_vars!(𝓂,symbolics)
        solve_steady_state!(𝓂,symbolic_SS,symbolics)
        write_functions_mapping!(𝓂)
        𝓂.solution.functions_written = true
    end

    if !isnothing(parameters)
        write_parameters_input!(𝓂,parameters)
    end

    if dynamics
        if 𝓂.solution.outdated
            if  algorithm ∈ [:dynare, :riccati, :first_order]
                SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
								jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)
								
								sol_mat = calculate_first_order_solution(jacobian; T = 𝓂.timings)
                state_update = function(state::Vector{Float64}, shock::Vector{Float64}) sol_mat * [state[𝓂.timings.past_not_future_and_mixed_idx]; shock] end
                
                𝓂.solution.perturbation.first_order = perturbation_solution(sol_mat, state_update)

                𝓂.solution.non_stochastic_steady_state = SS_and_pars
                𝓂.solution.NSSS_outdated = false
            end
            if :second_order == algorithm #∈ 𝓂.solution.algorithm
                # calculate_second_order_solution!(𝓂)
                if length(𝓂.solution.perturbation.first_order.solution_matrix) == 0
                    SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
										jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)

                    sol_mat = calculate_first_order_solution(jacobian; T = 𝓂.timings)

                    state_update = function(state::Vector{Float64}, shock::Vector{Float64}) sol_mat * [state[𝓂.timings.past_not_future_and_mixed_idx]; shock] end
                            
                    𝓂.solution.perturbation.first_order = perturbation_solution(sol_mat, state_update)
                    
                    𝓂.solution.non_stochastic_steady_state = SS_and_pars
                    𝓂.solution.NSSS_outdated = false
                end
                SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
								jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)
								
								hessian = calculate_hessian(𝓂.parameter_values,SS_and_pars,𝓂)
                𝐒₂ = calculate_second_order_solution(jacobian, 
                                                hessian, 
                                                𝓂.solution.perturbation.first_order.solution_matrix; 
                                                T = 𝓂.timings)

                𝐒₁ = [𝓂.solution.perturbation.first_order.solution_matrix[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝓂.solution.perturbation.first_order.solution_matrix[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]
                
                𝓂.solution.perturbation.second_order.solution_matrix = 𝐒₂
                𝓂.solution.perturbation.second_order.state_update = function(state::Vector{Float64}, shock::Vector{Float64})
                    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                                1
                                shock]
                    return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
                end

                
                state = zeros(𝓂.timings.nVars)
                shock = zeros(𝓂.timings.nExo)

                delta = 1

                while delta > eps(Float64)
                    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx,:]
                                1
                                shock]
                    state_tmp =  𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
                    delta = sum(abs,state_tmp - state)
                    state = state_tmp
                end

                𝓂.solution.perturbation.second_order.stochastic_steady_state = SS_and_pars[1:end - length(𝓂.calibration_equations)] + vec(state)
                𝓂.solution.outdated = false
            end
            if :third_order == algorithm #∈ 𝓂.solution.algorithm
                # calculate_third_order_solution(𝓂)
                # make sure 1st order solution is available
                if length(𝓂.solution.perturbation.first_order.solution_matrix) == 0
                    SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
										jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)

                    sol_mat = calculate_first_order_solution(jacobian; T = 𝓂.timings)

                    state_update = function(state::Vector{Float64}, shock::Vector{Float64}) sol_mat * [state[𝓂.timings.past_not_future_and_mixed_idx]; shock] end
                    
                    𝓂.solution.perturbation.first_order = perturbation_solution(sol_mat, state_update)

                    𝓂.solution.outdated = false
                    
                    𝓂.solution.non_stochastic_steady_state = SS_and_pars
                    𝓂.solution.NSSS_outdated = false
                end

                # make sure 2nd order solution is available
                if length(𝓂.solution.perturbation.second_order.solution_matrix) == 0
                    # calculate_second_order_solution(𝓂)
                    
                    SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
										jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)

										hessian = calculate_hessian(𝓂.parameter_values,SS_and_pars,𝓂)

                    𝐒₂ = calculate_second_order_solution(jacobian, 
                                                        hessian, 
                                                        𝓂.solution.perturbation.first_order.solution_matrix; 
                                                        T = 𝓂.timings)

                    𝐒₁ = [𝓂.solution.perturbation.first_order.solution_matrix[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝓂.solution.perturbation.first_order.solution_matrix[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]
                    
                    𝓂.solution.perturbation.second_order.solution_matrix = 𝐒₂
                    𝓂.solution.perturbation.second_order.state_update = function(state::Vector{Float64}, shock::Vector{Float64})
                        aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                                    1
                                    shock]
                        return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
                    end


                    state = zeros(𝓂.timings.nVars)
                    shock = zeros(𝓂.timings.nExo)

                    delta = 1

                    while delta > eps(Float64)
                        aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx,:]
                                    1
                                    shock]
                        state_tmp =  𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
                        delta = sum(abs,state_tmp - state)
                        state = state_tmp
                    end

                    𝓂.solution.perturbation.second_order.stochastic_steady_state = SS_and_pars[1:end - length(𝓂.calibration_equations)] + vec(state)
                    𝓂.solution.outdated = false

                end

                SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
								jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)

								hessian = calculate_hessian(𝓂.parameter_values,SS_and_pars,𝓂)
                ∇₃ = calculate_third_order_derivatives(𝓂.parameter_values,SS_and_pars,𝓂)

                𝐒₃ = calculate_third_order_solution(jacobian, 
                                                        hessian, 
                                                        ∇₃, 
                                                        𝓂.solution.perturbation.first_order.solution_matrix, 
                                                        𝓂.solution.perturbation.second_order.solution_matrix; 
                                                        T = 𝓂.timings)


                𝐒₂ = 𝓂.solution.perturbation.second_order.solution_matrix
                
                𝐒₁ = [𝓂.solution.perturbation.first_order.solution_matrix[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝓂.solution.perturbation.first_order.solution_matrix[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]
                
                𝓂.solution.perturbation.third_order.solution_matrix = 𝐒₃
                𝓂.solution.perturbation.third_order.state_update = function(state::Vector{Float64}, shock::Vector{Float64})
                    
                    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                                 1
                                 shock]

                    return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
                end

                state = zeros(𝓂.timings.nVars)
                shock = zeros(𝓂.timings.nExo)

                delta = 1

                while delta > eps(Float64)
                    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx,:]
                                 1
                                 shock]

                    state_tmp =  𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
                    delta = sum(abs,state_tmp - state)
                    state = state_tmp
                end

                𝓂.solution.perturbation.third_order.stochastic_steady_state = SS_and_pars[1:end - length(𝓂.calibration_equations)] + vec(state)
                𝓂.solution.outdated = false

            end
            if :linear_time_iteration == algorithm #∈ 𝓂.solution.algorithm
                SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
								jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)
								sol_mat = calculate_linear_time_iteration_solution(jacobian; T = 𝓂.timings)
                state_update = function(state::Vector{Float64}, shock::Vector{Float64}) sol_mat * [state[𝓂.timings.past_not_future_and_mixed_idx]; shock] end
                
                𝓂.solution.perturbation.linear_time_iteration = perturbation_solution(sol_mat, state_update)
                
                𝓂.solution.non_stochastic_steady_state = SS_and_pars
                𝓂.solution.NSSS_outdated = false
            end
            if length(intersect(𝓂.solution.algorithm,[:linear_time_iteration, :dynare, :riccati, :first_order])) == 0
                println("Algorithm to solve dynamics not found.")
            end
        end
        if length(𝓂.solution.perturbation.linear_time_iteration.solution_matrix) == 0 && :linear_time_iteration == algorithm #∈ 𝓂.solution.algorithm
                SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
								jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)
								sol_mat = calculate_linear_time_iteration_solution(jacobian; T = 𝓂.timings)
                state_update = function(state::Vector{Float64}, shock::Vector{Float64}) sol_mat * [state[𝓂.timings.past_not_future_and_mixed_idx]; shock] end
                
                𝓂.solution.perturbation.linear_time_iteration = perturbation_solution(sol_mat, state_update)
                
                𝓂.solution.non_stochastic_steady_state = SS_and_pars
                𝓂.solution.NSSS_outdated = false
        end
        if length(𝓂.solution.perturbation.first_order.solution_matrix) == 0 && algorithm ∈ [:dynare, :riccati, :first_order]
                SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
								jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)

                sol_mat = calculate_first_order_solution(jacobian; T = 𝓂.timings)

                state_update = function(state::Vector{Float64}, shock::Vector{Float64}) sol_mat * [state[𝓂.timings.past_not_future_and_mixed_idx]; shock] end
                
                𝓂.solution.perturbation.first_order = perturbation_solution(sol_mat, state_update)
                
                𝓂.solution.non_stochastic_steady_state = SS_and_pars
                𝓂.solution.NSSS_outdated = false
        end
        if length(𝓂.solution.perturbation.second_order.solution_matrix) == 0 && :second_order == algorithm #∈ 𝓂.solution.algorithm
                # calculate_second_order_solution!(𝓂)
                if length(𝓂.solution.perturbation.first_order.solution_matrix) == 0  
                    SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
										∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)
										sol_mat = calculate_first_order_solution(∇₁; T = 𝓂.timings)

                    state_update = function(state::Vector{Float64}, shock::Vector{Float64}) sol_mat * [state[𝓂.timings.past_not_future_and_mixed_idx]; shock] end
                            
                    𝓂.solution.perturbation.first_order = perturbation_solution(sol_mat, state_update)
                    
                    𝓂.solution.non_stochastic_steady_state = SS_and_pars
                    𝓂.solution.NSSS_outdated = false
                end
								SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
								jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)
										
                hessian = calculate_hessian(𝓂.parameter_values,SS_and_pars,𝓂)

                𝐒₂ = calculate_second_order_solution(jacobian, 
                                                        hessian, 
                                                𝓂.solution.perturbation.first_order.solution_matrix; 
                                                T = 𝓂.timings)

                𝐒₁ = [𝓂.solution.perturbation.first_order.solution_matrix[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝓂.solution.perturbation.first_order.solution_matrix[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]
                
                𝓂.solution.perturbation.second_order.solution_matrix = 𝐒₂
                𝓂.solution.perturbation.second_order.state_update = function(state::Vector{Float64}, shock::Vector{Float64})
                    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                                1
                                shock]
                    return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
                end


                state = zeros(𝓂.timings.nVars)
                shock = zeros(𝓂.timings.nExo)

                delta = 1

                while delta > eps(Float64)
                    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx,:]
                                1
                                shock]
                    state_tmp =  𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
                    delta = sum(abs,state_tmp - state)
                    state = state_tmp
                end

                𝓂.solution.perturbation.second_order.stochastic_steady_state = SS_and_pars[1:end - length(𝓂.calibration_equations)] + vec(state)
                𝓂.solution.outdated = false

        end
        if length(𝓂.solution.perturbation.third_order.solution_matrix) == 0 && :third_order == algorithm #∈ 𝓂.solution.algorithm
                # calculate_third_order_solution(𝓂)

                # make sure 1st order solution is available
                if length(𝓂.solution.perturbation.first_order.solution_matrix) == 0
                    SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
										jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)
										
                    sol_mat = calculate_first_order_solution(jacobian; T = 𝓂.timings)
                    
                    state_update = function(state::Vector{Float64}, shock::Vector{Float64}) sol_mat * [state[𝓂.timings.past_not_future_and_mixed_idx]; shock] end
                    
                    𝓂.solution.perturbation.first_order = perturbation_solution(sol_mat, state_update)

                    𝓂.solution.outdated = false
                    
                    𝓂.solution.non_stochastic_steady_state = SS_and_pars
                    𝓂.solution.NSSS_outdated = false
                end

                # make sure 2nd order solution is available
                if length(𝓂.solution.perturbation.second_order.solution_matrix) == 0
                    # calculate_second_order_solution(𝓂)
                    
                    SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
										jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)
										hessian = calculate_hessian(𝓂.parameter_values,SS_and_pars,𝓂)

                    𝐒₂ = calculate_second_order_solution(jacobian, 
                                                        hessian, 
                                                        𝓂.solution.perturbation.first_order.solution_matrix; 
                                                        T = 𝓂.timings)

                    𝐒₁ = [𝓂.solution.perturbation.first_order.solution_matrix[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝓂.solution.perturbation.first_order.solution_matrix[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]
                    
                    𝓂.solution.perturbation.second_order.solution_matrix = 𝐒₂
                    𝓂.solution.perturbation.second_order.state_update = function(state::Vector{Float64}, shock::Vector{Float64})
                        aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                                    1
                                    shock]
                        return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
                    end


                    state = zeros(𝓂.timings.nVars)
                    shock = zeros(𝓂.timings.nExo)

                    delta = 1

                    while delta > eps(Float64)
                        aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx,:]
                                    1
                                    shock]
                        state_tmp =  𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
                        delta = sum(abs,state_tmp - state)
                        state = state_tmp
                    end

                    𝓂.solution.perturbation.second_order.stochastic_steady_state = SS_and_pars[1:end - length(𝓂.calibration_equations)] + vec(state)
                    𝓂.solution.outdated = false

                end

                SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
								jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)
								hessian = calculate_hessian(𝓂.parameter_values,SS_and_pars,𝓂)
                ∇₃ = calculate_third_order_derivatives(𝓂.parameter_values,SS_and_pars,𝓂)

                𝐒₃ = calculate_third_order_solution(jacobian, 
                                                        hessian, 
                                                        ∇₃, 
                                                        𝓂.solution.perturbation.first_order.solution_matrix, 
                                                        𝓂.solution.perturbation.second_order.solution_matrix; 
                                                        T = 𝓂.timings)


                𝐒₂ = 𝓂.solution.perturbation.second_order.solution_matrix
                
                𝐒₁ = [𝓂.solution.perturbation.first_order.solution_matrix[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝓂.solution.perturbation.first_order.solution_matrix[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]
                
                𝓂.solution.perturbation.third_order.solution_matrix = 𝐒₃
                𝓂.solution.perturbation.third_order.state_update = function(state::Vector{Float64}, shock::Vector{Float64})
                    
                    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                                 1
                                 shock]

                    return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
                end

                state = zeros(𝓂.timings.nVars)
                shock = zeros(𝓂.timings.nExo)

                delta = 1

                while delta > eps(Float64)
                    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx,:]
                                 1
                                 shock]

                    state_tmp =  𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
                    delta = sum(abs,state_tmp - state)
                    state = state_tmp
                end

                𝓂.solution.perturbation.third_order.stochastic_steady_state = SS_and_pars[1:end - length(𝓂.calibration_equations)] + vec(state)
                𝓂.solution.outdated = false
        end
    end
end





function write_functions_mapping!(𝓂::ℳ)
    present_varss = map(x->Symbol(string(x) * "₍₀₎"),sort(setdiff(union(𝓂.var_present,𝓂.aux_present,𝓂.exo_present), 𝓂.nonnegativity_auxilliary_vars)))
    future_varss  = map(x->Symbol(string(x) * "₍₁₎"),sort(setdiff(union(𝓂.var_future,𝓂.aux_future,𝓂.exo_future), 𝓂.nonnegativity_auxilliary_vars)))
    past_varss    = map(x->Symbol(string(x) * "₍₋₁₎"),sort(setdiff(union(𝓂.var_past,𝓂.aux_past,𝓂.exo_past), 𝓂.nonnegativity_auxilliary_vars)))
    shock_varss   = map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.exo)
    ss_varss      = map(x->Symbol(string(x) * "₍ₛₛ₎"),𝓂.var)

    steady_state = []
    for ii in 1:length(ss_varss)
        push!(steady_state,:($(ss_varss[ii]) = X̄[$ii]))
        # ii += 1
    end

    ii = 1

    alll = []
    for j in 1:length(future_varss)
        push!(alll,:($(future_varss[j]) = X[$ii]))
        ii += 1
    end

    for i in 1:length(present_varss)
        push!(alll,:($(present_varss[i]) = X[$ii]))
        ii += 1
    end

    for l in 1:length(past_varss)
        push!(alll,:($(past_varss[l]) = X[$ii]))
        ii += 1
    end

    for k in 1:length(shock_varss)
        push!(alll,:($(shock_varss[k]) = X[$ii]))
        ii += 1
    end


    paras = []
    push!(paras,:((;$(𝓂.par...)) = params))

    # watch out with naming of parameters in model and functions
    mod_func2 = :(function model_function_uni_redux(X::Vector{Real}, params::Vector{Real}, X̄::Vector{Real})
        $(alll...)
        $(paras...)
        $(steady_state...)
        [$(𝓂.dyn_equations...)]
    end)


    𝓂.model_function = @RuntimeGeneratedFunction(mod_func2)
    # model_function = 𝓂.model_function

    calib_eqs = []
    for i in 1:length(𝓂.solved_vals) 
        eqs   = 𝓂.solved_vals[i]
        varss = 𝓂.solved_vars[i]
        push!(calib_eqs,:($varss = $eqs))
    end

    for i in 1:length(𝓂.exo)
        varss = 𝓂.exo[i]
        push!(calib_eqs,:($varss = 0))
    end

    calib_pars = []
    for i in 1:length(𝓂.parameters)
        parss = 𝓂.parameters[i]
        push!(calib_pars,:($parss = parameters[$i]))
    end

    var_out = []
    ii =  1
    for i in 1:length(𝓂.var) 
        push!(var_out,:($(𝓂.var[i]) = SS[$ii]))
        ii += 1
    end

    par_out = []
    for i in 1:length(𝓂.calibration_equations_parameters) 
        push!(par_out,:($(𝓂.calibration_equations_parameters[i]) = SS[$ii]))
        ii += 1
    end

    calib_pars = []
    for i in 1:length(𝓂.parameters)
        parss = 𝓂.parameters[i]
        push!(calib_pars,:($parss = parameters[$i]))
    end

    test_func = :(function test_SS(parameters::Vector{Float64}, SS::Vector{Float64})
        $(calib_pars...) 
        # var, par = get_non_stochastic_steady_state(𝓂)
        $(var_out...)
        $(par_out...)
        [$(𝓂.ss_equations...),$(𝓂.calibration_equations...)]
        # all(isapprox.([$(𝓂.ss_equations...)],0,atol = eps(Float32)))
    end)
    # println(test_func)

    𝓂.solution.valid_steady_state_solution = @RuntimeGeneratedFunction(test_func)

    𝓂.solution.outdated = true
    return nothing
end



write_parameters_input!(𝓂::ℳ, parameters::Pair{Symbol,<: Number}) = write_parameters_input!(𝓂::ℳ, Dict(parameters))
write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{Symbol,<: Number},Vararg{Pair{Symbol,<: Number}}}) = write_parameters_input!(𝓂::ℳ, Dict(parameters))



function write_parameters_input!(𝓂::ℳ, parameters::Dict{Symbol,<: Number})
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
            println("Parameter changes: ")
            𝓂.solution.outdated = true
        end
            
        for i in 1:length(parameters)
            if 𝓂.parameter_values[ntrsct_idx[i]] != collect(values(parameters))[i]
                if collect(keys(parameters))[i] ∈ 𝓂.SS_dependencies[end][2] && 𝓂.solution.NSSS_outdated == false
                    𝓂.solution.NSSS_outdated = true
                    # println("SS outdated.")
                end
                
                println("\t",𝓂.parameters[ntrsct_idx[i]],"\tfrom ",𝓂.parameter_values[ntrsct_idx[i]],"\tto ",collect(values(parameters))[i])

                𝓂.parameter_values[ntrsct_idx[i]] = collect(values(parameters))[i]
            end
        end
    end

    if 𝓂.solution.NSSS_outdated == true println("New parameters changed the steady state.") end
end


write_parameters_input!(𝓂::ℳ, parameters::Tuple{<: Number,Vararg{<: Number}}) = write_parameters_input!(𝓂::ℳ, vec(collect(parameters)))
write_parameters_input!(𝓂::ℳ, parameters::Matrix{<: Number}) = write_parameters_input!(𝓂::ℳ, vec(collect(parameters)))


function write_parameters_input!(𝓂::ℳ, parameters::Vector{<: Number})
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
            𝓂.solution.outdated = true

            match_idx = parameters .!= 𝓂.parameter_values[1:length(parameters)]

            changed_vals = parameters[match_idx]
            changes_pars = 𝓂.parameters[1:length(parameters)][match_idx]

            for p in changes_pars
                if p ∈ 𝓂.SS_dependencies[end][2] && 𝓂.solution.NSSS_outdated == false
                    𝓂.solution.NSSS_outdated = true
                    # println("SS outdated.")
                end
            end

            println("Parameter changes: ")
            for ii in 1:sum(match_idx)
                println("\t",changes_pars[ii],"\tfrom ",𝓂.parameter_values[1:length(parameters)][match_idx][ii],"\tto ",changed_vals[ii])
            end

            𝓂.parameter_values[1:length(parameters)][match_idx] = parameters[match_idx]
        end
    end
    if 𝓂.solution.NSSS_outdated == true println("New parameters changed the steady state.") end
end



function SS_parameter_derivatives(parameters::Vector{<: Number}, parameters_idx, 𝓂::ℳ)
    𝓂.parameter_values[parameters_idx] = parameters
    𝓂.SS_solve_func(𝓂.parameter_values, 𝓂.SS_init_guess, 𝓂)
end


function SS_parameter_derivatives(parameters::Number, parameters_idx::Int, 𝓂::ℳ)
    𝓂.parameter_values[parameters_idx] = parameters
    𝓂.SS_solve_func(𝓂.parameter_values, 𝓂.SS_init_guess, 𝓂)
end


function covariance_parameter_derivatives(parameters::Vector{<: Number}, parameters_idx, 𝓂::ℳ)
    𝓂.parameter_values[parameters_idx] = parameters
    convert(Vector{Number},max.(ℒ.diag(calculate_covariance(𝓂.parameter_values, 𝓂)),eps(Float64)))
end


function covariance_parameter_derivatives(parameters::Number, parameters_idx::Int, 𝓂::ℳ)
    𝓂.parameter_values[parameters_idx] = parameters
    convert(Vector{Number},max.(ℒ.diag(calculate_covariance(𝓂.parameter_values, 𝓂)),eps(Float64)))
end



function get_non_stochastic_steady_state_internal(𝓂::ℳ; parameters = nothing)
    solve!(𝓂;dynamics = false,parameters = parameters)

    return 𝓂.solution.NSSS_outdated ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂.SS_init_guess, 𝓂) : 𝓂.solution.non_stochastic_steady_state
end



function calculate_jacobian(parameters::Vector{<: Number}, SS_and_pars::AbstractArray{<: Number}, 𝓂::ℳ)
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

    return ℱ.jacobian(x -> 𝓂.model_function(x, par, SS), [SS_future; SS_present; SS_past; shocks_ss]), SS_and_pars
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
        
    return sparse(reshape(ℱ.jacobian(x -> ℱ.jacobian(x -> (𝓂.model_function(x, par, SS)), x), [SS_future; SS_present; SS_past; shocks_ss] ), 𝓂.timings.nVars, nk^2))#, SS_and_pars
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
      
    return sparse(reshape(ℱ.jacobian(x -> ℱ.jacobian(x -> ℱ.jacobian(x -> 𝓂.model_function(x, par, SS), x), x), [SS_future; SS_present; SS_past; shocks_ss] ), 𝓂.timings.nVars, nk^3))#, SS_and_pars
 end



# function calculate_linear_time_iteration_solution(𝓂::ℳ)
function calculate_linear_time_iteration_solution(jacobian::AbstractMatrix{Float64}; T::timings)
    expand = [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
              ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    ∇₊ = jacobian[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    ∇₀ = jacobian[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    ∇₋ = jacobian[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]
    ∇ₑ = jacobian[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]
  
    maxiter = 1000
    tol = eps(Float32)

    # @btime begin
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
    # else
    #     outmessage = "Convergence Successful!"
    end

    Q = -(∇₊ * F + ∇₀) \ ∇ₑ
    # F = sparse(F)

    sol_mat = hcat(F[:,T.past_not_future_and_mixed_idx],Q)
    
    return sol_mat
end



function riccati_forward(jacobian::AbstractMatrix{<: Number}; T::timings, explosive::Bool = false)#::AbstractMatrix{Real}
    ∇₊ = @view jacobian[:,1:T.nFuture_not_past_and_mixed]
    ∇₀ = @view jacobian[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]
    ∇₋ = @view jacobian[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]

    Q    = ℒ.qr(collect(∇₀[:,T.present_only_idx]))
    Qinv = Q.Q'

    A₊ = Qinv * ∇₊
    A₀ = Qinv * ∇₀
    A₋ = Qinv * ∇₋

    dynIndex = T.nPresent_only+1:T.nVars

    Ã₊  = @view A₊[dynIndex,:]
    Ã₋  = @view A₋[dynIndex,:]
    Ã₀₊ = @view A₀[dynIndex, T.future_not_past_and_mixed_idx]
    Ã₀₋ = A₀[dynIndex, T.past_not_future_idx]* ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.not_mixed_in_past_idx,:]
    
    Z₊ = zeros(T.nMixed,T.nFuture_not_past_and_mixed)
    I₊ = ℒ.diagm(ones(T.nFuture_not_past_and_mixed))[T.mixed_in_future_idx,:]

    Z₋ = zeros(T.nMixed,T.nPast_not_future_and_mixed)
    I₋ = ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.mixed_in_past_idx,:]

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

    sol = vcat(L[T.not_mixed_in_past_idx,:], D)

    Ā₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_only_idx]
    A₊ᵤ  = @view A₊[1:T.nPresent_only,:]
    Ã₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_but_not_only_idx]
    A₋ᵤ  = @view A₋[1:T.nPresent_only,:]

    A    = vcat(- Ā₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ), sol)
    
    A[T.reorder,:]
end


# function riccati_forward(jacobian::AbstractMatrix{Float64}; T::timings, explosive::Bool = false)
function riccati_conditions(jacc::AbstractMatrix{<: Number}, sol_d::AbstractMatrix{<: Number}; T::timings) #::AbstractMatrix{Real},
    expand = [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
              ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    A = jacc[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    B = jacc[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    C = jacc[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]

    sol_buf = sol_d * expand[2]

    err1 = A * sol_buf * sol_buf + B * sol_buf + C
    return err1[:,T.past_not_future_and_mixed_idx]
end



function riccati_forward(jacc::AbstractMatrix{ℱ.Dual{Z,S,N}}; T::timings = T, explosive::Bool = false) where {Z,S,N}
    # unpack: AoS -> SoA
    jacobian = ℱ.value.(jacc)
    # you can play with the dimension here, sometimes it makes sense to transpose
    ps = mapreduce(ℱ.partials, hcat, jacc)'

    # get f(vs)
    val = riccati_forward(jacobian;T = T, explosive = explosive)

    # get J(f, vs) * ps (cheating). Write your custom rule here
    B = ℱ.jacobian(x -> riccati_conditions(x, val; T = T), jacobian)
    A = ℱ.jacobian(x -> riccati_conditions(jacobian, x; T = T), val)

    jvp = (-A \ B) * ps

    # pack: SoA -> AoS
    return reshape(map(val, eachrow(jvp)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end,size(val))
end


function calculate_first_order_solution(jacobian::AbstractMatrix{<: Number}; T::timings, explosive::Bool = false)
    A = riccati_forward(jacobian, T = T, explosive = explosive)

    Jm = ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]
    
    ∇₊ = jacobian[:,1:T.nFuture_not_past_and_mixed] * ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:]
    ∇₀ = jacobian[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    ∇ₑ = jacobian[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

    B = -((∇₊ * A * Jm + ∇₀) \ ∇ₑ)

    return hcat(A, B)
end



# function calculate_second_order_solution!(𝓂::ℳ)
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
    𝐒₁ = [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] |> sparse
    droptol!(𝐒₁,tol)

    # set up vector to capture volatility effect
    redu = sparsevec(nₑ₋ - nₑ + 1:nₑ₋, 1)
    redu_idxs = findnz(ℒ.kron(redu, redu))[1]
    𝛔 = sparse(redu_idxs[Int.(range(1,nₑ^2,nₑ))], fill(n₋ * (nₑ₋ + 1) + 1, nₑ), 1, nₑ₋^2, nₑ₋^2)

    𝐒₁₋╱𝟏ₑ = [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];
    
    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
            𝐒₁
            spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];

    𝐒₁₊╱𝟎 = [𝐒₁[i₊,:]
            zeros(n₋ + n + nₑ, nₑ₋)];


    # setup compression matrices
    colls2 = [nₑ₋ * (i-1) + k for i in 1:nₑ₋ for k in 1:i]
    𝐂₂ = sparse(colls2, 1:length(colls2) , 1)
    𝐔₂ = 𝐂₂' * sparse([i <= k ? (k - 1) * nₑ₋ + i : (i - 1) * nₑ₋ + k for k in 1:nₑ₋ for i in 1:nₑ₋], 1:nₑ₋^2, 1)


    ∇₁₊𝐒₁➕∇₁₀ = -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]

    ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = -∇₂ * (ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * 𝛔) * 𝐂₂ 

    X = sparse(∇₁₊𝐒₁➕∇₁₀ \ ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹)
    droptol!(X,tol)


    ∇₁₊ = sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])

    B = sparse(∇₁₊𝐒₁➕∇₁₀ \ ∇₁₊)
    droptol!(B,tol)


    C = (𝐔₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐔₂ * 𝛔) * 𝐂₂
    droptol!(C,tol)

    A = spdiagm(ones(n))

    lm = LinearMap{Float64}(x -> A * reshape(x,size(X)) - B * reshape(x,size(X)) * C, size(X)[1] * size(X)[2])

    𝐒₂ = sparse(reshape(bicgstabl(lm, vec(-X)), size(X))) * 𝐔₂ # fastest
    droptol!(𝐒₂,tol)

    return 𝐒₂
end






# function calculate_third_order_solution!(𝓂::ℳ)
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
    𝐒₁ = [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] |> sparse
    droptol!(𝐒₁,tol)

    # set up vector to capture volatility effect
    redu = sparsevec(nₑ₋ - nₑ + 1:nₑ₋, 1)
    redu_idxs = findnz(ℒ.kron(redu, redu))[1]
    𝛔 = sparse(redu_idxs[Int.(range(1,nₑ^2,nₑ))], fill(n₋ * (nₑ₋ + 1) + 1, nₑ), 1, nₑ₋^2, nₑ₋^2)


    𝐒₁₋╱𝟏ₑ = [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
            𝐒₁
            spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];

    𝐒₁₊╱𝟎 = [𝐒₁[i₊,:]
            zeros(n₋ + n + nₑ, nₑ₋)];

    ∇₁₊𝐒₁➕∇₁₀ = -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]


    ∇₁₊ = sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])

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
    𝐏 = sparse(reshape(spdiagm(ones(nₑ₋^3))[:,PermutedDimsArray(M,[1, 4, 2, 3])],nₑ₋^3,nₑ₋^3)
                           + reshape(spdiagm(ones(nₑ₋^3))[:,PermutedDimsArray(M,[1, 2, 4, 3])],nₑ₋^3,nₑ₋^3)
                           + reshape(spdiagm(ones(nₑ₋^3))[:,PermutedDimsArray(M,[1, 2, 3, 4])],nₑ₋^3,nₑ₋^3))
    

    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 =[(𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
            𝐒₂
            zeros(n₋ + nₑ, nₑ₋^2)];
        
    𝐒₂₊╱𝟎 = [𝐒₂[i₊,:] 
             zeros(n₋ + n + nₑ, nₑ₋^2)];
             𝐒₁₊╱𝟎
    
    𝐗₃ = -∇₃ * ℒ.kron(ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋), ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋)
    
    𝐏₁ₗ = sparse(spdiagm(ones(n̄^3))[vec(permutedims(reshape(1:n̄^3,n̄,n̄,n̄),(1,3,2))),:])
    𝐏₁ᵣ = sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(1,3,2)))])
    𝐏₂ₗ = sparse(spdiagm(ones(n̄^3))[vec(permutedims(reshape(1:n̄^3,n̄,n̄,n̄),(3,1,2))),:])
    𝐏₂ᵣ = sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(3,1,2)))])

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
    
    𝐗₃ += -∇₁₊ * 𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, [𝐒₂[i₋,:] ; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]) * 𝐏
    droptol!(𝐗₃,tol)
    
    
    X = sparse(∇₁₊𝐒₁➕∇₁₀ \ 𝐗₃ * 𝐂₃)
    droptol!(X,tol)
    
    
    𝐏₁ₗ = sparse(spdiagm(ones(nₑ₋^3))[vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(1,3,2))),:])
    𝐏₁ᵣ = sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(1,3,2)))])
    𝐏₂ₗ = sparse(spdiagm(ones(nₑ₋^3))[vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(3,1,2))),:])
    𝐏₂ᵣ = sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(3,1,2)))])

    tmpkron = ℒ.kron(𝐒₁₋╱𝟏ₑ,𝛔)
    
    C = 𝐔₃ * tmpkron + 𝐔₃ * 𝐏₁ₗ * tmpkron * 𝐏₁ᵣ + 𝐔₃ * 𝐏₂ₗ * tmpkron * 𝐏₂ᵣ
    C += 𝐔₃ * ℒ.kron(𝐒₁₋╱𝟏ₑ,ℒ.kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ))
    C *= 𝐂₃
    droptol!(C,tol)
    
    
    A = spdiagm(ones(n))
    lm = LinearMap{Float64}(x -> A * reshape(x,size(X)) - B * reshape(x,size(X)) * C, size(X)[1] * size(X)[2])
    
    𝐒₃ = sparse(reshape(bicgstabl(lm, vec(-X)),size(X))) * 𝐔₃ # fastest
    droptol!(𝐒₃,tol)
    
    
    return 𝐒₃
end





function irf(state_update::Function, initial_state::Vector{Float64}, T::timings; 
    periods::Int = 40, 
    shocks::Symbol_input = :all, 
    variables::Symbol_input = :all, 
    negative_shock::Bool = false)

    if shocks == :all
        shock_idx = 1:T.nExo
    elseif shocks == :simulate
        shock_idx = 1:T.nExo
    elseif shocks == :none
        shock_idx = 1:T.nExo
    elseif shocks isa Matrix{Symbol}
        if !issubset(shocks,T.exo)
            return @warn "Following shocks are not part of the model: " * string.(setdiff(shocks,T.exo))
        end
        shock_idx = getindex(1:T.nExo,convert(Vector{Bool},vec(sum(shocks .== T.exo,dims= 2))))
    elseif shocks isa Vector{Symbol}
        if !issubset(shocks,𝓂.timings.exo)
            return @warn "Following shocks are not part of the model: " * string.(setdiff(shocks,𝓂.timings.exo))
        end
        shock_idx = getindex(1:𝓂.timings.nExo,convert(Vector{Bool},vec(sum(reshape(shocks,1,length(variables)) .== 𝓂.timings.exo,dims= 2))))
    elseif shocks isa Tuple
        if !issubset(shocks,𝓂.timings.exo)
            return @warn "Following shocks are not part of the model: " * string.(setdiff(shocks,𝓂.timings.exo))
        end
        shock_idx = getindex(1:𝓂.timings.nExo,convert(Vector{Bool},vec(sum(reshape(collect(shocks),1,length(variables)) .== 𝓂.timings.exo,dims= 2))))
    elseif shocks isa Symbol
        if !issubset([shocks],T.exo)
            return @warn "Following shock is not part of the model: " * string(setdiff([shocks],T.exo)[1])
        end
        shock_idx = getindex(1:T.nExo,shocks .== T.exo)
    else
        return @warn "Invalid argument in shocks"
    end

    var_idx = parse_variables_input_to_index(variables, T)

    if shocks == :simulate
        ET = randn(T.nExo,periods)

        Y = zeros(T.nVars,periods,1)
        Y[:,1,1] = state_update(initial_state,ET[:,1])

        for t in 1:periods-1
            Y[:,t+1,1] = state_update(Y[:,t,1],ET[:,t+1])
        end

        return KeyedArray(Y[var_idx,:,:];  Variables = T.var[var_idx], Period = 1:periods, Shock = [:simulate])
    elseif shocks == :none
        Y = zeros(T.nVars,periods,1)
        shck = T.nExo == 0 ? Vector{Float64}(undef, 0) : [0.0]
        Y[:,1,1] = state_update(initial_state,shck)

        for t in 1:periods-1
            Y[:,t+1,1] = state_update(Y[:,t,1],shck)
        end

        return KeyedArray(Y[var_idx,:,:];  Variables = T.var[var_idx], Period = 1:periods, Shock = [:none])
    else
        Y = zeros(T.nVars,periods,T.nExo)

        for ii in shock_idx
            if shocks != :simulate
                ET = zeros(T.nExo,periods)
                ET[ii,1] = negative_shock ? -1 : 1
            end

            Y[:,1,ii] = state_update(initial_state,ET[:,1])

            for t in 1:periods-1
                Y[:,t+1,ii] = state_update(Y[:,t,ii],ET[:,t+1])
            end
        end

        return KeyedArray(Y[var_idx,:,shock_idx];  Variables = T.var[var_idx], Period = 1:periods, Shock = T.exo[shock_idx])
    end

    # return Y[var_idx,:,shock_idx]
end



function girf(state_update::Function, T::timings; 
    periods::Int = 40, 
    shocks::Symbol_input = :all, 
    variables::Symbol_input = :all, 
    negative_shock::Bool = false, 
    warmup_periods::Int = 100, 
    draws::Int = 50, 
    iterations_to_steady_state::Int = 500)

    shock_idx = parse_shocks_input_to_index(shocks,T)

    var_idx = parse_variables_input_to_index(variables, T)

    Y = zeros(T.nVars,periods,T.nExo)

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

            shock = zeros(T.nExo)

            shock[ii] = negative_shock ? -1 : 1

            shock += baseline_noise

            Y1[:,1] = state_update(initial_state, baseline_noise)
            Y2[:,1] = state_update(initial_state, shock)

            for t in 1:periods-1
                baseline_noise = randn(T.nExo)

                Y1[:,t+1] = state_update(Y1[:,t],baseline_noise)
                Y2[:,t+1] = state_update(Y2[:,t],baseline_noise)
            end

            Y[:,:,ii] += Y2 - Y1
        end
        Y[:,:,ii] /= draws
    end
    
    # return Y[var_idx,:,shock_idx]
    return KeyedArray(Y[var_idx,:,shock_idx];  Variables = T.var[var_idx], Period = 1:periods, Shock = T.exo[shock_idx])

end

function parse_variables_input_to_index(variables::Symbol_input, T::timings)
    if variables == :all
        return indexin(T.var,sort(union(T.var,T.aux,T.exo_present)))
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
        shock_idx = getindex(1:T.nExo,convert(Vector{Bool},vec(sum(reshape(shocks,1,length(variables)) .== T.exo,dims= 2))))
    elseif shocks isa Tuple{Symbol, Vararg{Symbol}}
        if !issubset(shocks,T.exo)
            return @warn "Following shocks are not part of the model: " * string.(setdiff(shocks,T.exo))
        end
        shock_idx = getindex(1:T.nExo,convert(Vector{Bool},vec(sum(reshape(collect(shocks),1,length(variables)) .== T.exo,dims= 2))))
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

    elseif algorithm ∈ [:dynare, :riccati, :first_order]
        state_update = 𝓂.solution.perturbation.first_order.state_update

    elseif :second_order == algorithm
        state_update = 𝓂.solution.perturbation.second_order.state_update

    elseif :third_order == algorithm
        state_update = 𝓂.solution.perturbation.third_order.state_update
    end

    return state_update
end


function calculate_covariance(parameters::Vector{<: Number}, 𝓂::ℳ)
    SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
		jacc = calculate_jacobian(parameters, SS_and_pars, 𝓂)

    sol = calculate_first_order_solution(jacc; T = 𝓂.timings)

    A = sol[:,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]
    C = sol[:,𝓂.timings.nPast_not_future_and_mixed+1:end]

    covar_dcmp = sparse(ℒ.triu(reshape((ℒ.I - ℒ.kron(A, conj(A))) \ reshape(C * C', prod(size(A)), 1), size(A))))

    droptol!(covar_dcmp,eps(Float64))

    return covar_dcmp
end




function calculate_kalman_filter_loglikelihood(𝓂::ℳ, data::AbstractArray{Float64}, observables::Vector{Symbol}; parameters = nothing)
    if length(observables) != size(data)[1]
        @error "Data columns and number of observables are not identical. Make sure the data contains only the selected observables."
    end
    
    if length(observables) > 𝓂.timings.nExo
        @error "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."
    end
    
    SS_and_pars = 𝓂.SS_solve_func(parameters, 𝓂.SS_init_guess, 𝓂)
    
		jacc = calculate_jacobian(parameters, SS_and_pars, 𝓂)

    sol = calculate_first_order_solution(jacc; T = 𝓂.timings)

    observables_and_states = sort(union(𝓂.timings.past_not_future_and_mixed_idx,indexin(observables,𝓂.var)))

    A = sol[observables_and_states,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(observables_and_states)))[indexin(𝓂.timings.past_not_future_and_mixed_idx,observables_and_states)
    ,:]
    B = sol[observables_and_states,𝓂.timings.nPast_not_future_and_mixed+1:end]

    C = ℒ.diagm(ones(length(observables_and_states)))[indexin(sort(indexin(observables,𝓂.var)),observables_and_states),:]

    𝐁 = B * B'

    T = size(data)[2]

    # Gaussian Prior
    P = reshape((ℒ.I - ℒ.kron(A, A)) \ reshape(B * B', prod(size(A)), 1), size(A))
    u = zeros(length(observables_and_states))
    z = C * u
    
    loglik = 0.0

    for t in 1:T
        v = data[:,t] - z

        F = C * P * C'

        loglik += log(ℒ.det(F)) + v' / F * v

        K = P * C' / F

        P = A * (P - K * C * P) * A' + 𝐁

        u = A * (u + K * v)
        
        z = C * u 
    end
    return -(loglik + length(observables) * log(2 * 3.141592653589793) * T) / 2 # otherwise conflicts with model parameters assignment
end



end