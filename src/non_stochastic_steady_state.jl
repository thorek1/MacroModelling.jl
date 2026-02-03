
function write_steady_state_solver_function!(𝓂::ℳ, symbolic_SS, Symbolics::symbolics; verbose::Bool = false, avoid_solve::Bool = false)
    unknowns = union(Symbolics.calibration_equations_parameters, Symbolics.vars_in_ss_equations)

    @assert length(unknowns) <= length(Symbolics.ss_equations) + length(Symbolics.calibration_equations) "Unable to solve steady state. More unknowns than equations."

    eq_list = vcat(union.(setdiff.(union.(Symbolics.var_list_aux_SS,
                                        Symbolics.ss_list_aux_SS),
                                    Symbolics.var_redundant_list),
                            Symbolics.par_list_aux_SS),
                    union.(Symbolics.ss_calib_list,
                            Symbolics.par_calib_list))

    vars, eqs, n_blocks, incidence_matrix = compute_block_triangularization(unknowns, eq_list)
    
    n = n_blocks

    ss_equations = vcat(Symbolics.ss_equations,Symbolics.calibration_equations)# .|> SPyPyC.Sym
    # println(ss_equations)

    SS_solve_func = []

    atoms_in_equations = Set{Symbol}()
    atoms_in_equations_list = []
    relevant_pars_across = Symbol[]
    NSSS_solver_cache_init_tmp = []

    solved_vars = []
    solved_vals = []

    min_max_errors = []

    unique_➕_eqs = Dict{Union{Expr,Symbol},Symbol}()

    while n > 0 
        if length(eqs[:,eqs[2,:] .== n]) == 2
            var_to_solve_for = unknowns[vars[:,vars[2,:] .== n][1]]

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
                push!(SS_solve_func, :(if solution_error > tol.NSSS_acceptance_tol if verbose println("Failed for min max terms in equations with error $solution_error") end; scale = scale * .3 + solved_scale * .7; continue end))
                eq_to_solve = eval(minmax_fixed_eqs)
            end
            
            if avoid_solve || count_ops(Meta.parse(string(eq_to_solve))) > 15
                soll = nothing
            else
                soll = solve_symbolically(eq_to_solve,var_to_solve_for)
            end

            if isnothing(soll) || isempty(soll)
                println("Failed finding solution symbolically for: ",var_to_solve_for," in: ",eq_to_solve)
                
                eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]

                write_block_solution!(𝓂, SS_solve_func, [var_to_solve_for], [eq_to_solve], relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, solved_vars, solved_vals)
                # write_domain_safe_block_solution!(𝓂, SS_solve_func, [var_to_solve_for], [eq_to_solve], relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, unique_➕_eqs)  
            elseif soll[1].is_number == true
                ss_equations = [replace_symbolic(eq, var_to_solve_for, soll[1]) for eq in ss_equations]
                
                push!(solved_vars,Symbol(var_to_solve_for))
                push!(solved_vals,Meta.parse(string(soll[1])))

                if (solved_vars[end] ∈ 𝓂.constants.post_model_macro.➕_vars) 
                    push!(SS_solve_func,:($(solved_vars[end]) = max(eps(),$(solved_vals[end]))))
                else
                    push!(SS_solve_func,:($(solved_vars[end]) = $(solved_vals[end])))
                end

                push!(atoms_in_equations_list,[])
            else
                push!(solved_vars,Symbol(var_to_solve_for))
                push!(solved_vals,Meta.parse(string(soll[1])))
                
                [push!(atoms_in_equations, Symbol(a)) for a in soll[1].atoms()]
                push!(atoms_in_equations_list, Set(union(setdiff(get_symbols(parsed_eq_to_solve_for), get_symbols(minmax_fixed_eqs)),Symbol.(soll[1].atoms()))))

                if (solved_vars[end] ∈ 𝓂.constants.post_model_macro.➕_vars)
                    push!(SS_solve_func,:($(solved_vars[end]) = begin
                        _bounds = get($(𝓂.constants.post_parameters_macro.bounds), $(QuoteNode(solved_vars[end])), (eps(), 1e12))
                        min(max(_bounds[1], $(solved_vals[end])), _bounds[2])
                    end))
                    push!(SS_solve_func,:(solution_error += $(Expr(:call,:abs, Expr(:call, :-, solved_vars[end], solved_vals[end])))))
                    push!(SS_solve_func, :(if solution_error > tol.NSSS_acceptance_tol if verbose println("Failed for analytical aux variables with error $solution_error") end; scale = scale * .3 + solved_scale * .7; continue end))
                    
                    unique_➕_eqs[solved_vals[end]] = solved_vars[end]
                else
                    vars_to_exclude = [vcat(Symbol.(var_to_solve_for), 𝓂.constants.post_model_macro.➕_vars), Symbol[]]
                    
                    rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep = make_equation_robust_to_domain_errors([solved_vals[end]], vars_to_exclude, 𝓂.constants.post_parameters_macro.bounds, 𝓂.constants.post_model_macro.➕_vars, unique_➕_eqs)
    
                    if length(vcat(ss_and_aux_equations_error, ss_and_aux_equations_error_dep)) > 0
                        push!(SS_solve_func,vcat(ss_and_aux_equations, ss_and_aux_equations_dep)...)
                        push!(SS_solve_func,:(solution_error += $(Expr(:call, :+, vcat(ss_and_aux_equations_error, ss_and_aux_equations_error_dep)...))))
                        push!(SS_solve_func, :(if solution_error > tol.NSSS_acceptance_tol if verbose println("Failed for analytical variables with error $solution_error") end; scale = scale * .3 + solved_scale * .7; continue end))
                    end
                    
                    push!(SS_solve_func,:($(solved_vars[end]) = $(rewritten_eqs[1])))
                end

                if haskey(𝓂.constants.post_parameters_macro.bounds, solved_vars[end]) && solved_vars[end] ∉ 𝓂.constants.post_model_macro.➕_vars
                    push!(SS_solve_func,:(solution_error += abs(min(max($(𝓂.constants.post_parameters_macro.bounds[solved_vars[end]][1]), $(solved_vars[end])), $(𝓂.constants.post_parameters_macro.bounds[solved_vars[end]][2])) - $(solved_vars[end]))))
                    push!(SS_solve_func, :(if solution_error > tol.NSSS_acceptance_tol if verbose println("Failed for bounded variables with error $solution_error") end; scale = scale * .3 + solved_scale * .7; continue end))
                end
            end
        else
            vars_to_solve = unknowns[vars[:,vars[2,:] .== n][1,:]]

            eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

            numerical_sol = false
            
            if symbolic_SS
                if avoid_solve || count_ops(Meta.parse(string(eqs_to_solve))) > 15
                    soll = nothing
                else
                    soll = solve_symbolically(eqs_to_solve,vars_to_solve)
                end

                if isnothing(soll) || isempty(soll) || length(intersect((union(SPyPyC.free_symbols.(collect(values(soll)))...) .|> SPyPyC.:↓),(vars_to_solve .|> SPyPyC.:↓))) > 0
                    if verbose println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.") end

                    numerical_sol = true
                else
                    if verbose println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " symbolically.") end
                    
                    atoms = reduce(union,map(x->x.atoms(),collect(values(soll))))

                    for a in atoms push!(atoms_in_equations, Symbol(a)) end
                    
                    for vars in vars_to_solve
                        push!(solved_vars,Symbol(vars))
                        push!(solved_vals,Meta.parse(string(soll[vars]))) #using convert(Expr,x) leads to ugly expressions

                        push!(atoms_in_equations_list, Set(Symbol.(soll[vars].atoms())))
                        push!(SS_solve_func,:($(solved_vars[end]) = $(solved_vals[end])))
                    end
                end
            end

            eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]

            incidence_matrix_subset = incidence_matrix[vars[:,vars[2,:] .== n][1,:], eq_idx_in_block_to_solve]

            # try symbolically and use numerical if it does not work
            if numerical_sol || !symbolic_SS
                pv = sortperm(vars_to_solve, by = Symbol)
                pe = sortperm(eqs_to_solve, by = string)

                if length(pe) > 5
                    write_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, solved_vars, solved_vals)
                    # write_domain_safe_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, unique_➕_eqs)
                else
                    solved_system = partial_solve(eqs_to_solve[pe], vars_to_solve[pv], incidence_matrix_subset[pv,pe], avoid_solve = avoid_solve)
                    
                    # if !isnothing(solved_system) && !any(contains.(string.(vcat(solved_system[3],solved_system[4])), "LambertW")) && !any(contains.(string.(vcat(solved_system[3],solved_system[4])), "Heaviside")) 
                    #     write_reduced_block_solution!(𝓂, SS_solve_func, solved_system, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, 
                    #     𝓂.constants.post_model_macro.➕_vars, unique_➕_eqs)  
                    # else
                        write_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, solved_vars, solved_vals)  
                        # write_domain_safe_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, unique_➕_eqs)  
                    # end
                end

                if !symbolic_SS && verbose
                    println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " numerically.")
                end
            end
        end
        n -= 1
    end

    push!(NSSS_solver_cache_init_tmp, fill(Inf, length(𝓂.constants.post_complete_parameters.parameters)))
    push!(𝓂.caches.solver_cache, NSSS_solver_cache_init_tmp)

    unknwns = Symbol.(unknowns)

    collect_calibration_no_var_parameters!(atoms_in_equations, 𝓂)
    
    parameters_in_equations = build_parameters_in_equations(𝓂, atoms_in_equations, relevant_pars_across)
    
    build_dependencies!(𝓂, atoms_in_equations_list, solved_vars)
    
    dyn_exos = build_dyn_exos_expressions(𝓂)

    push!(SS_solve_func,:($(dyn_exos...)))
    
    push!(SS_solve_func, min_max_errors...)
    # push!(SS_solve_func,:(push!(NSSS_solver_cache_tmp, params_scaled_flt)))
    
    push!(SS_solve_func,:(if length(NSSS_solver_cache_tmp) == 0 NSSS_solver_cache_tmp = [copy(params_flt)] else NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., copy(params_flt)] end))
    

    # push!(SS_solve_func,:(for pars in 𝓂.caches.solver_cache
    #                             latest = sqrt(sum(abs2,pars[end] - params_flt))# / max(sum(abs2,pars[end]), sum(abs,params_flt))
    #                             if latest <= current_best
    #                                 current_best = latest
    #                             end
    #                         end))
        push!(SS_solve_func,:(if (current_best > 1e-8) && (solution_error < tol.NSSS_acceptance_tol) && (scale == 1)
                                    reverse_diff_friendly_push!(𝓂.caches.solver_cache, NSSS_solver_cache_tmp)
                            end))
    # push!(SS_solve_func,:(if length(𝓂.caches.solver_cache) > 100 popfirst!(𝓂.caches.solver_cache) end))
    
    # push!(SS_solve_func,:(SS_init_guess = ([$(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_past,𝓂.constants.post_model_macro.exo_future))...), $(𝓂.calibration_equations_parameters...)])))

    # push!(SS_solve_func,:(𝓂.SS_init_guess = typeof(SS_init_guess) == Vector{Float64} ? SS_init_guess : ℱ.value.(SS_init_guess)))

    # push!(SS_solve_func,:(return ComponentVector([$(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_past,𝓂.constants.post_model_macro.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.constants.post_model_macro.exo_present,𝓂.constants.post_model_macro.var))...,𝓂.calibration_equations_parameters...]))))


    # fix parameter bounds
    par_bounds = build_parameter_bounds_expressions(𝓂, atoms_in_equations, relevant_pars_across)

    solve_exp = :(function solve_SS(initial_parameters::Vector{Real}, 
                                    𝓂::ℳ,
                                    # fail_fast_solvers_only::Bool, 
                                    tol::Tolerances,
                                    verbose::Bool, 
                                    cold_start::Bool,
                                    solver_parameters::Vector{solver_parameters})
                    initial_parameters = typeof(initial_parameters) == Vector{Float64} ? initial_parameters : ℱ.value.(initial_parameters)

                    initial_parameters_tmp = copy(initial_parameters)

                    parameters = copy(initial_parameters)
                    params_flt = copy(initial_parameters)
                    
                    current_best = sum(abs2,𝓂.caches.solver_cache[end][end] - initial_parameters)
                    closest_solution_init = 𝓂.caches.solver_cache[end]
                    
                    for pars in 𝓂.caches.solver_cache
                        copy!(initial_parameters_tmp, pars[end])

                        ℒ.axpy!(-1,initial_parameters,initial_parameters_tmp)

                        latest = sum(abs2,initial_parameters_tmp)
                        if latest <= current_best
                            current_best = latest
                            closest_solution_init = pars
                        end
                    end

                    # closest_solution = copy(closest_solution_init)
                    # solution_error = 1.0
                    # iters = 0
                    range_iters = 0
                    solution_error = 1.0
                    solved_scale = 0
                    # range_length = [ 1, 2, 4, 8,16,32,64,128,1024]
                    scale = 1.0

                    NSSS_solver_cache_scale = CircularBuffer{Vector{Vector{Float64}}}(500)
                    push!(NSSS_solver_cache_scale, closest_solution_init)
                    # fail_fast_solvers_only = true
                    while range_iters <= (cold_start ? 1 : 500) && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)
                        range_iters += 1
                        fail_fast_solvers_only = range_iters > 1 ? true : false

                        if abs(solved_scale - scale) < 1e-2
                            # println(NSSS_solver_cache_scale[end])
                            break 
                        end

                        # println("i: $range_iters - scale: $scale - solved_scale: $solved_scale")
                        # println(closest_solution[end])
                    # for range_ in range_length
                        # rangee = range(0,1,range_+1)
                        # for scale in rangee[2:end]
                            # scale = 6*scale^5 - 15*scale^4 + 10*scale^3 # smootherstep

                            # if scale <= solved_scale continue end

                            
                            current_best = sum(abs2,NSSS_solver_cache_scale[end][end] - initial_parameters)
                            closest_solution = NSSS_solver_cache_scale[end]

                            for pars in NSSS_solver_cache_scale
                                copy!(initial_parameters_tmp, pars[end])
                                
                                ℒ.axpy!(-1,initial_parameters,initial_parameters_tmp)

                                latest = sum(abs2,initial_parameters_tmp)

                                if latest <= current_best
                                    current_best = latest
                                    closest_solution = pars
                                end
                            end

                            # println(closest_solution)

                            if all(isfinite,closest_solution[end]) && initial_parameters != closest_solution_init[end]
                                parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
                            else
                                parameters = copy(initial_parameters)
                            end
                            params_flt = parameters
                            
                            # println(parameters)

                            $(parameters_in_equations...)
                            $(par_bounds...)
                            $(𝓂.equations.calibration_no_var...)
                            NSSS_solver_cache_tmp = []
                            solution_error = 0.0
                            iters = 0
                            $(SS_solve_func...)

                            if solution_error < tol.NSSS_acceptance_tol
                                # println("solved for $scale; $range_iters")
                                solved_scale = scale
                                if scale == 1
                                    # return ComponentVector([$(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_past,𝓂.constants.post_model_macro.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.constants.post_model_macro.exo_present,𝓂.constants.post_model_macro.var))...,𝓂.calibration_equations_parameters...])), solution_error
                                    # NSSS_solution = [$(Symbol.(replace.(string.(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_past,𝓂.constants.post_model_macro.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)]
                                    # NSSS_solution[abs.(NSSS_solution) .< 1e-12] .= 0 # doesn't work with Zygote
                                    return [$(Symbol.(replace.(string.(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_past,𝓂.constants.post_model_macro.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.equations.calibration_parameters...)], (solution_error, iters)
                                else
                                    reverse_diff_friendly_push!(NSSS_solver_cache_scale, NSSS_solver_cache_tmp)
                                end

                                if scale > .95
                                    scale = 1
                                else
                                    # scale = (scale + 1) / 2
                                    scale = scale * .4 + .6
                                end
                            # else
                            #     println("no sol")
                            #     scale  = (scale + solved_scale) / 2
                            #     println("scale $scale")
                            # elseif scale == 1 && range_ == range_length[end]
                            #     return [$(Symbol.(replace.(string.(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_past,𝓂.constants.post_model_macro.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)], (solution_error, iters)
                            end
                    #     end
                    end
                    return zeros($(length(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_past,𝓂.constants.post_model_macro.exo_future)) + length(𝓂.equations.calibration_parameters))), (1, 0)
                end)


    𝓂.functions.NSSS_solve = @RuntimeGeneratedFunction(solve_exp)
    # 𝓂.functions.NSSS_solve = eval(solve_exp)

    return nothing
end



function write_steady_state_solver_function!(𝓂::ℳ;
                            cse = true,
                            skipzeros = true,
                            density_threshold::Float64 = .1,
                            nnz_parallel_threshold::Int = 1000000,
                            min_length::Int = 1000,
                            verbose::Bool = false)
    unknowns = union(𝓂.constants.post_model_macro.vars_in_ss_equations, 𝓂.equations.calibration_parameters)

    @assert length(unknowns) <= length(𝓂.equations.steady_state_aux) + length(𝓂.equations.calibration) "Unable to solve steady state. More unknowns than equations."

    eq_list = vcat(union.(union.(𝓂.constants.post_model_macro.var_list_aux_SS,
                    𝓂.constants.post_model_macro.ss_list_aux_SS),
                𝓂.constants.post_model_macro.par_list_aux_SS),
            union.(𝓂.constants.post_parameters_macro.ss_calib_list,
                𝓂.constants.post_parameters_macro.par_calib_list))

    vars, eqs, n_blocks, incidence_matrix = compute_block_triangularization(unknowns, eq_list)
    
    n = n_blocks

    ss_equations = vcat(𝓂.equations.steady_state_aux,𝓂.equations.calibration)

    SS_solve_func = []

    atoms_in_equations = Set{Symbol}()
    atoms_in_equations_list = []
    relevant_pars_across = []
    NSSS_solver_cache_init_tmp = []

    solved_vars = []
    solved_vals = []

    n_block = 1

    while n > 0
        vars_to_solve = unknowns[vars[:,vars[2,:] .== n][1,:]]

        eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

        # try symbolically and use numerical if it does not work
        if verbose
            println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " numerically.")
        end
        
        push!(solved_vars,Symbol.(vars_to_solve))
        push!(solved_vals,Meta.parse.(string.(eqs_to_solve)))

        syms_in_eqs = Set()

        for i in eqs_to_solve
            push!(syms_in_eqs, get_symbols(i)...)
        end

        # println(syms_in_eqs)
        push!(atoms_in_equations_list,setdiff(syms_in_eqs, solved_vars[end]))

        # calib_pars = []
        calib_pars_input = []
        relevant_pars = reduce(union,vcat(𝓂.constants.post_model_macro.par_list_aux_SS,𝓂.constants.post_parameters_macro.par_calib_list)[eqs[:,eqs[2,:] .== n][1,:]])
        relevant_pars_across = union(relevant_pars_across,relevant_pars)
        
        iii = 1
        for parss in union(𝓂.constants.post_complete_parameters.parameters,𝓂.constants.post_parameters_macro.parameters_as_function_of_parameters)
            # valss   = 𝓂.parameter_values[i]
            if :($parss) ∈ relevant_pars
                # push!(calib_pars,:($parss = parameters_and_solved_vars[$iii]))
                push!(calib_pars_input,:($parss))
                iii += 1
            end
        end


        # guess = Expr[]
        # untransformed_guess = Expr[]
        result = Expr[]
        sorted_vars = sort(solved_vars[end])
        # sorted_vars = sort(setdiff(solved_vars[end],𝓂.constants.post_model_macro.➕_vars))
        for (i, parss) in enumerate(sorted_vars) 
            # push!(guess,:($parss = guess[$i]))
            # push!(untransformed_guess,:($parss = undo_transform(guess[$i],transformation_level)))
            push!(result,:($parss = sol[$i]))
        end

        
        # separate out auxiliary variables (nonnegativity)
        nnaux = []
        # nnaux_linear = []
        # nnaux_error = []
        # push!(nnaux_error, :(aux_error = 0))
        solved_vals_local = Expr[]
        # solved_vals_in_place = Expr[]
        
        eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]


        other_vrs_eliminated_by_sympy = Set()

        for (i,val) in enumerate(solved_vals[end])
            if typeof(val) ∈ [Symbol,Float64,Int]
                push!(solved_vals_local,val)
                # push!(solved_vals_in_place, :(ℰ[$i] = $val))
            else
                if eq_idx_in_block_to_solve[i] ∈ 𝓂.constants.post_model_macro.ss_equations_with_aux_variables
                    val = vcat(𝓂.equations.steady_state_aux,𝓂.equations.calibration)[eq_idx_in_block_to_solve[i]]
                    push!(nnaux,:($(val.args[2]) = max(eps(),$(val.args[3]))))
                    push!(other_vrs_eliminated_by_sympy, val.args[2])
                    # push!(nnaux_linear,:($val))
                    push!(solved_vals_local,:($val))
                    # push!(solved_vals_in_place,:(ℰ[$i] = $val))
                    # push!(nnaux_error, :(aux_error += min(eps(),$(val.args[3]))))
                else
                    push!(solved_vals_local,postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))
                    # push!(solved_vals_in_place, :(ℰ[$i] = $(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))))
                end
            end
        end

        # println(other_vrs_eliminated_by_sympy)
        # sort nnaux vars so that they enter in right order. avoid using a variable before it is declared
        # println(nnaux)
        if length(nnaux) > 1
            all_symbols = map(x->x.args[1],nnaux) #relevant symbols come first in respective equations

            nn_symbols = map(x->intersect(all_symbols,x), get_symbols.(nnaux))
            
            inc_matrix = fill(0,length(all_symbols),length(all_symbols))

            for i in 1:length(all_symbols)
                for k in 1:length(nn_symbols)
                    inc_matrix[i,k] = collect(all_symbols)[i] ∈ collect(nn_symbols)[k]
                end
            end

            QQ, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(inc_matrix))

            nnaux = nnaux[QQ]
            # nnaux_linear = nnaux_linear[QQ]
        end


        # other_vars = []
        other_vars_input = []
        # other_vars_inverse = []
        other_vrs = intersect( setdiff( union(𝓂.constants.post_model_macro.var, 𝓂.equations.calibration_parameters, 𝓂.constants.post_model_macro.➕_vars),
                                            sort(solved_vars[end]) ),
                                union(syms_in_eqs, other_vrs_eliminated_by_sympy, setdiff(reduce(union, get_symbols.(nnaux), init = []), map(x->x.args[1],nnaux)) ) )

        for var in other_vrs
            # var_idx = findfirst(x -> x == var, union(𝓂.constants.post_model_macro.var,𝓂.calibration_equations_parameters))
            # push!(other_vars,:($(var) = parameters_and_solved_vars[$iii]))
            push!(other_vars_input,:($(var)))
            iii += 1
            # push!(other_vars_inverse,:(𝓂.SS_init_guess[$var_idx] = $(var)))
        end

        parameters_and_solved_vars = vcat(calib_pars_input, other_vrs)

        ng = length(sorted_vars)
        np = length(parameters_and_solved_vars)
        nd = 0
        nx = iii - 1
    

        Symbolics.@variables 𝔊[1:ng] 𝔓[1:np]


        parameter_dict = Dict{Symbol, Symbol}()
        back_to_array_dict = Dict{Symbolics.Num, Symbolics.Num}()
        # aux_vars = Symbol[]
        # aux_expr = []
    
    
        for (i,v) in enumerate(sorted_vars)
            push!(parameter_dict, v => :($(Symbol("𝔊_$i"))))
            push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔊_$i"))), @__MODULE__) => 𝔊[i])
        end
    
        for (i,v) in enumerate(parameters_and_solved_vars)
            push!(parameter_dict, v => :($(Symbol("𝔓_$i"))))
            push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔓_$i"))), @__MODULE__) => 𝔓[i])
        end
    
        # for (i,v) in enumerate(ss_and_aux_equations_dep)
        #     push!(aux_vars, v.args[1])
        #     push!(aux_expr, v.args[2])
        # end
    
        # aux_replacements = Dict(aux_vars .=> aux_expr)
    
        replaced_solved_vals = solved_vals_local |> 
            # x -> replace_symbols.(x, Ref(aux_replacements)) |> 
            x -> replace_symbols.(x, Ref(parameter_dict)) |> 
            x -> Symbolics.parse_expr_to_symbolic.(x, Ref(@__MODULE__)) |>
            x -> Symbolics.substitute.(x, Ref(back_to_array_dict))
    
        lennz = length(replaced_solved_vals)
    
        if lennz > nnz_parallel_threshold
            parallel = Symbolics.ShardedForm(1500,4)
        else
            parallel = Symbolics.SerialForm()
        end
    
        _, calc_block! = Symbolics.build_function(replaced_solved_vals, 𝔊, 𝔓,
                                                    cse = cse, 
                                                    skipzeros = skipzeros, 
                                                    # nanmath = false,
                                                    parallel = parallel,
                                                    expression_module = @__MODULE__,
                                                    expression = Val(false))::Tuple{<:Function, <:Function}
    
        # 𝐷 = zeros(Symbolics.Num, nd)
    
        # ϵᵃ = zeros(nd)
    
        # calc_block_aux!(𝐷, 𝔊, 𝔓)
    
        ϵˢ = zeros(Symbolics.Num, ng)
    
        ϵ = zeros(ng)
    
        # calc_block!(ϵˢ, 𝔊, 𝔓, 𝐷)
    
        ∂block_∂parameters_and_solved_vars = Symbolics.sparsejacobian(replaced_solved_vals, 𝔊) # nϵ x nx
    
        lennz = nnz(∂block_∂parameters_and_solved_vars)
    
        if (lennz / length(∂block_∂parameters_and_solved_vars) > density_threshold) || (length(∂block_∂parameters_and_solved_vars) < min_length)
            derivatives_mat = convert(Matrix, ∂block_∂parameters_and_solved_vars)
            buffer = zeros(Float64, size(∂block_∂parameters_and_solved_vars))
        else
            derivatives_mat = ∂block_∂parameters_and_solved_vars
            buffer = similar(∂block_∂parameters_and_solved_vars, Float64)
            buffer.nzval .= 1
        end
    
        chol_buff = buffer * buffer'

        chol_buff += ℒ.I

        prob = 𝒮.LinearProblem(chol_buff, ϵ, 𝒮.CholeskyFactorization())

        chol_buffer = 𝒮.init(prob, 𝒮.CholeskyFactorization(), verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)

        prob = 𝒮.LinearProblem(buffer, ϵ, 𝒮.LUFactorization())

        lu_buffer = 𝒮.init(prob, 𝒮.LUFactorization(), verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)

        if lennz > nnz_parallel_threshold
            parallel = Symbolics.ShardedForm(1500,4)
        else
            parallel = Symbolics.SerialForm()
        end
        
        _, func_exprs = Symbolics.build_function(derivatives_mat, 𝔊, 𝔓,
                                                    cse = cse, 
                                                    skipzeros = skipzeros, 
                                                    # nanmath = false,
                                                    parallel = parallel,
                                                    expression_module = @__MODULE__,
                                                    expression = Val(false))::Tuple{<:Function, <:Function}
    
    
        Symbolics.@variables 𝔊[1:ng+nx]
    
        ext_diff = Symbolics.Num[]
        for i in 1:nx
            push!(ext_diff, 𝔓[i] - 𝔊[ng + i])
        end
        replaced_solved_vals_ext = vcat(replaced_solved_vals, ext_diff)
    
        _, calc_ext_block! = Symbolics.build_function(replaced_solved_vals_ext, 𝔊, 𝔓,
                                                    cse = cse, 
                                                    skipzeros = skipzeros, 
                                                    # nanmath = false,
                                                    parallel = parallel,
                                                    expression_module = @__MODULE__,
                                                    expression = Val(false))::Tuple{<:Function, <:Function}
    
        ϵᵉ = zeros(ng + nx)
        
        # ϵˢᵉ = zeros(Symbolics.Num, ng + nx)
    
        # calc_block_aux!(𝐷, 𝔊, 𝔓)
    
        # Evaluate the function symbolically
        # calc_ext_block!(ϵˢᵉ, 𝔊, 𝔓, 𝐷)
    
        ∂ext_block_∂parameters_and_solved_vars = Symbolics.sparsejacobian(replaced_solved_vals_ext, 𝔊) # nϵ x nx
    
        lennz = nnz(∂ext_block_∂parameters_and_solved_vars)
    
        if (lennz / length(∂ext_block_∂parameters_and_solved_vars) > density_threshold) || (length(∂ext_block_∂parameters_and_solved_vars) < min_length)
            derivatives_mat_ext = convert(Matrix, ∂ext_block_∂parameters_and_solved_vars)
            ext_buffer = zeros(Float64, size(∂ext_block_∂parameters_and_solved_vars))
        else
            derivatives_mat_ext = ∂ext_block_∂parameters_and_solved_vars
            ext_buffer = similar(∂ext_block_∂parameters_and_solved_vars, Float64)
            ext_buffer.nzval .= 1
        end
    
        ext_chol_buff = ext_buffer * ext_buffer'

        ext_chol_buff += ℒ.I

        prob = 𝒮.LinearProblem(ext_chol_buff, ϵᵉ, 𝒮.CholeskyFactorization())

        ext_chol_buffer = 𝒮.init(prob, 𝒮.CholeskyFactorization(), verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)

        prob = 𝒮.LinearProblem(ext_buffer, ϵᵉ, 𝒮.LUFactorization())

        ext_lu_buffer = 𝒮.init(prob, 𝒮.LUFactorization(), verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)

        if lennz > nnz_parallel_threshold
            parallel = Symbolics.ShardedForm(1500,4)
        else
            parallel = Symbolics.SerialForm()
        end
        
        _, ext_func_exprs = Symbolics.build_function(derivatives_mat_ext, 𝔊, 𝔓,
                                                    cse = cse, 
                                                    skipzeros = skipzeros, 
                                                    # nanmath = false,
                                                    parallel = parallel,
                                                    expression_module = @__MODULE__,
                                                    expression = Val(false))::Tuple{<:Function, <:Function}
    


        push!(NSSS_solver_cache_init_tmp,fill(1.205996189998029, length(sorted_vars)))
        push!(NSSS_solver_cache_init_tmp,[Inf])

        # WARNING: infinite bounds are transformed to 1e12
        lbs = []
        ubs = []
        
        limit_boundaries = 1e12

        for i in vcat(sorted_vars, calib_pars_input, other_vars_input)
            if haskey(𝓂.constants.post_parameters_macro.bounds, i)
                push!(lbs,𝓂.constants.post_parameters_macro.bounds[i][1] == -Inf ? -limit_boundaries+rand() : 𝓂.constants.post_parameters_macro.bounds[i][1])
                push!(ubs,𝓂.constants.post_parameters_macro.bounds[i][2] ==  Inf ?  limit_boundaries-rand() : 𝓂.constants.post_parameters_macro.bounds[i][2])
            else
                push!(lbs,-limit_boundaries+rand())
                push!(ubs,limit_boundaries+rand())
            end
        end

        push!(SS_solve_func,:(params_and_solved_vars = [$(calib_pars_input...),$(other_vars_input...)]))

        push!(SS_solve_func,:(lbs = [$(lbs...)]))
        push!(SS_solve_func,:(ubs = [$(ubs...)]))
        
        push!(SS_solve_func,:(inits = [max.(lbs[1:length(closest_solution[$(2*(n_block-1)+1)])], min.(ubs[1:length(closest_solution[$(2*(n_block-1)+1)])], closest_solution[$(2*(n_block-1)+1)])), closest_solution[$(2*n_block)]]))

        push!(SS_solve_func,:(solution = block_solver(length(params_and_solved_vars) == 0 ? [0.0] : params_and_solved_vars,
                                                                $(n_block), 
                                                                𝓂.NSSS.solve_blocks_in_place[$(n_block)], 
                                                                # 𝓂.ss_solve_blocks[$(n_block)], 
                                                                # 𝓂.ss_solve_blocks_no_transform[$(n_block)], 
                                                                # f, 
                                                                inits,
                                                                lbs, 
                                                                ubs,
                                                                solver_parameters,
                                                                fail_fast_solvers_only,
                                                                cold_start,
                                                                verbose)))
        
        # push!(SS_solve_func,:(solution = block_solver_RD(length([$(calib_pars_input...),$(other_vars_input...)]) == 0 ? [0.0] : [$(calib_pars_input...),$(other_vars_input...)])))#, 
        
        push!(SS_solve_func,:(iters += solution[2][2])) 
        push!(SS_solve_func,:(solution_error += solution[2][1])) 
        push!(SS_solve_func,:(sol = solution[1]))

        # push!(SS_solve_func,:(solution = block_solver_RD(length([$(calib_pars_input...),$(other_vars_input...)]) == 0 ? [0.0] : [$(calib_pars_input...),$(other_vars_input...)])))#, 
        
        # push!(SS_solve_func,:(solution_error += sum(abs2,𝓂.ss_solve_blocks[$(n_block)](length([$(calib_pars_input...),$(other_vars_input...)]) == 0 ? [0.0] : [$(calib_pars_input...),$(other_vars_input...)],solution))))

        push!(SS_solve_func,:($(result...)))   
        
        push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol)]))
        push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(params_and_solved_vars) == Vector{Float64} ? params_and_solved_vars : ℱ.value.(params_and_solved_vars)]))

        # Create nonlinear solver workspaces for regular and extended problems
        workspace = Nonlinear_solver_workspace(ϵ, buffer, chol_buffer, lu_buffer)
        ext_workspace = Nonlinear_solver_workspace(ϵᵉ, ext_buffer, ext_chol_buffer, ext_lu_buffer)

        push!(𝓂.NSSS.solve_blocks_in_place, 
            ss_solve_block(
                function_and_jacobian(calc_block!::Function, func_exprs::Function, workspace),
                function_and_jacobian(calc_ext_block!::Function, ext_func_exprs::Function, ext_workspace)
            )
        )

        n_block += 1
        
        n -= 1
    end

    push!(NSSS_solver_cache_init_tmp,[Inf])
    push!(NSSS_solver_cache_init_tmp,fill(Inf,length(𝓂.constants.post_complete_parameters.parameters)))
    push!(𝓂.caches.solver_cache,NSSS_solver_cache_init_tmp)

    unknwns = Symbol.(unknowns)

    collect_calibration_no_var_parameters!(atoms_in_equations, 𝓂)
    
    parameters_in_equations = build_parameters_in_equations(𝓂, atoms_in_equations, relevant_pars_across)
    
    build_dependencies!(𝓂, atoms_in_equations_list, solved_vars)

    dyn_exos = build_dyn_exos_expressions(𝓂)

    push!(SS_solve_func,:($(dyn_exos...)))

    # push!(SS_solve_func,:(push!(NSSS_solver_cache_tmp, params_scaled_flt)))
    push!(SS_solve_func,:(if length(NSSS_solver_cache_tmp) == 0 NSSS_solver_cache_tmp = [copy(params_flt)] else NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., copy(params_flt)] end))
    
    push!(SS_solve_func,:(current_best = sqrt(sum(abs2,𝓂.caches.solver_cache[end][end] - params_flt))))# / max(sum(abs2,𝓂.caches.solver_cache[end][end]), sum(abs2,params_flt))))

    push!(SS_solve_func,:(for pars in 𝓂.caches.solver_cache
                                latest = sqrt(sum(abs2,pars[end] - params_flt))# / max(sum(abs2,pars[end]), sum(abs,params_flt))
                                if latest <= current_best
                                    current_best = latest
                                end
                            end))

    push!(SS_solve_func,:(if (current_best > 1e-8) && (solution_error < tol.NSSS_acceptance_tol)
                                    reverse_diff_friendly_push!(𝓂.caches.solver_cache, NSSS_solver_cache_tmp)
                                # solved_scale = scale
                            end))

    # fix parameter bounds
    par_bounds = build_parameter_bounds_expressions(𝓂, atoms_in_equations, relevant_pars_across)

    solve_exp = :(function solve_SS(initial_parameters::Vector{Real}, 
                                    𝓂::ℳ, 
                                    tol::Tolerances,
                                    # fail_fast_solvers_only::Bool, 
                                    verbose::Bool, 
                                    cold_start::Bool,
                                    solver_parameters::Vector{solver_parameters})
                    initial_parameters = typeof(initial_parameters) == Vector{Float64} ? initial_parameters : ℱ.value.(initial_parameters)

                    parameters = copy(initial_parameters)
                    params_flt = copy(initial_parameters)
                    
                    current_best = sum(abs2,𝓂.caches.solver_cache[end][end] - initial_parameters)
                    closest_solution_init = 𝓂.caches.solver_cache[end]
                    
                    for pars in 𝓂.caches.solver_cache
                        latest = sum(abs2,pars[end] - initial_parameters)
                        if latest <= current_best
                            current_best = latest
                            closest_solution_init = pars
                        end
                    end

                    # closest_solution = closest_solution_init
                    # solution_error = 1.0
                    # iters = 0
                    range_iters = 0
                    solution_error = 1.0
                    solved_scale = 0
                    # range_length = [ 1, 2, 4, 8,16,32,64,128,1024]
                    scale = 1.0

                    while range_iters <= 500 && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)
                        range_iters += 1
                        fail_fast_solvers_only = range_iters > 1 ? true : false

                    # for range_ in range_length
                        # rangee = range(0,1,range_+1)
                        # for scale in rangee[2:end]
                            # scale = 6*scale^5 - 15*scale^4 + 10*scale^3 # smootherstep

                            # if scale <= solved_scale continue end

                            current_best = sum(abs2,𝓂.caches.solver_cache[end][end] - initial_parameters)
                            closest_solution = 𝓂.caches.solver_cache[end]

                            for pars in 𝓂.caches.solver_cache
                                latest = sum(abs2,pars[end] - initial_parameters)
                                if latest <= current_best
                                    current_best = latest
                                    closest_solution = pars
                                end
                            end

                            # Zero initial value if starting without guess
                            if !isfinite(sum(abs,closest_solution[2]))
                                closest_solution = copy(closest_solution)
                                for i in 1:2:length(closest_solution)
                                    closest_solution[i] = zeros(length(closest_solution[i]))
                                end
                            end

                            # println(closest_solution)

                            if all(isfinite,closest_solution[end]) && initial_parameters != closest_solution_init[end]
                                parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
                            else
                                parameters = copy(initial_parameters)
                            end
                            params_flt = parameters
                            
                            # println(parameters)

                            $(parameters_in_equations...)
                            $(par_bounds...)
                            $(𝓂.equations.calibration_no_var...)
                            NSSS_solver_cache_tmp = []
                            solution_error = 0.0
                            iters = 0
                            $(SS_solve_func...)

                            if solution_error < tol.NSSS_acceptance_tol
                                # println("solved for $scale; $range_iters")
                                solved_scale = scale
                                if scale == 1
                                    # return ComponentVector([$(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_past,𝓂.constants.post_model_macro.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.constants.post_model_macro.exo_present,𝓂.constants.post_model_macro.var))...,𝓂.calibration_equations_parameters...])), solution_error
                                    # NSSS_solution = [$(Symbol.(replace.(string.(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_past,𝓂.constants.post_model_macro.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)]
                                    # NSSS_solution[abs.(NSSS_solution) .< 1e-12] .= 0 # doesn't work with Zygote
                                    return [$(Symbol.(replace.(string.(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_past,𝓂.constants.post_model_macro.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.equations.calibration_parameters...)], (solution_error, iters)
                                else
                                    reverse_diff_friendly_push!(NSSS_solver_cache_scale, NSSS_solver_cache_tmp)
                                end

                                if scale > .95
                                    scale = 1
                                else
                                    # scale = (scale + 1) / 2
                                    scale = scale * .4 + .6
                                end
                            # else
                            #     println("no sol")
                            #     scale  = (scale + solved_scale) / 2
                            #     println("scale $scale")
                            # elseif scale == 1 && range_ == range_length[end]
                            #     return [$(Symbol.(replace.(string.(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_past,𝓂.constants.post_model_macro.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)], (solution_error, iters)
                            end
                    #     end
                    end
                    return zeros($(length(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_past,𝓂.constants.post_model_macro.exo_future)) + length(𝓂.equations.calibration_parameters))), (1, 0)
                end)

    𝓂.functions.NSSS_solve = @RuntimeGeneratedFunction(solve_exp)
    # 𝓂.functions.NSSS_solve = eval(solve_exp)

    return nothing
end
