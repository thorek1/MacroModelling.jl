# Shared helper functions for steady-state solver generation
# These functions eliminate code duplication between the two versions of
# write_steady_state_solver_function! (symbolic and precompiled/numerical)

"""
    compute_block_triangularization(unknowns, eq_list)

Compute block triangular form ordering for the steady-state equation system.
Uses BlockTriangularForm.order to decompose the system into ordered blocks.

# Arguments
- `unknowns`: Vector of unknown variables/parameters to solve for
- `eq_list`: Vector of sets, each containing symbols that appear in that equation

# Returns
- `vars`: Variable ordering with block assignments (2×n matrix)
- `eqs`: Equation ordering with block assignments (2×n matrix)
- `n_blocks`: Number of blocks
- `incidence_matrix`: Sparse incidence matrix
"""
function compute_block_triangularization(unknowns, eq_list)
    incidence_matrix = spzeros(Int, length(unknowns), length(unknowns))

    for (i, u) in enumerate(unknowns)
        for (k, e) in enumerate(eq_list)
            incidence_matrix[i, k] = u ∈ e
        end
    end

    Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(incidence_matrix)
    R̂ = Int[]
    for i in 1:n_blocks
        [push!(R̂, n_blocks - i + 1) for ii in R[i]:R[i+1]-1]
    end
    push!(R̂, 1)

    vars = hcat(P, R̂)'
    eqs = hcat(Q, R̂)'
    
    @assert all(eqs[1,:] .> 0) "Could not solve system of steady state and calibration equations. Number of redundant equations: " * repr(sum(eqs[1,:] .< 0)) * ". Try defining some steady state values as parameters (e.g. r[ss] -> r̄). Nonstationary variables are not supported as of now."
    
    return vars, eqs, n_blocks, incidence_matrix
end


"""
    build_parameters_in_equations(𝓂, atoms_in_equations, relevant_pars_across)

Build the parameter assignment expressions for use in the generated solve_SS function.

# Returns
Vector of expressions like `:(param = parameters[i])` for parameters that appear
in the steady-state equations.
"""
function build_parameters_in_equations(𝓂, atoms_in_equations, relevant_pars_across)
    parameters_in_equations = []

    for (i, parss) in enumerate(𝓂.constants.post_complete_parameters.parameters) 
        if parss ∈ union(atoms_in_equations, relevant_pars_across)
            push!(parameters_in_equations, :($parss = parameters[$i]))
        end
    end
    
    return parameters_in_equations
end


"""
    build_dependencies!(𝓂, atoms_in_equations_list, solved_vars)

Build and store the dependency tracking information in 𝓂.NSSS.dependencies.
Tracks which parameters and variables each solved variable depends on.

# Returns
The dependencies vector.
"""
function build_dependencies!(𝓂, atoms_in_equations_list, solved_vars)
    dependencies = []
    for (i, a) in enumerate(atoms_in_equations_list)
        push!(dependencies, solved_vars[i] => intersect(a, union(𝓂.constants.post_model_macro.var, 𝓂.constants.post_complete_parameters.parameters)))
    end

    push!(dependencies, :SS_relevant_calibration_parameters => intersect(reduce(union, atoms_in_equations_list), 𝓂.constants.post_complete_parameters.parameters))

    𝓂.NSSS.dependencies = dependencies
    
    return dependencies
end


"""
    build_dyn_exos_expressions(𝓂)

Build expressions that set dynamic exogenous variables (past/future) to zero for steady state.

# Returns
Vector of expressions like `:(exo_var = 0)`.
"""
function build_dyn_exos_expressions(𝓂)
    dyn_exos = []
    for dex in union(𝓂.constants.post_model_macro.exo_past, 𝓂.constants.post_model_macro.exo_future)
        push!(dyn_exos, :($dex = 0))
    end
    return dyn_exos
end


"""
    build_parameter_bounds_expressions(𝓂, atoms_in_equations, relevant_pars_across)

Build expressions that clamp parameters to their defined bounds.

# Returns
Vector of bound-clamping expressions like `:(param = min(max(param, lb), ub))`.
"""
function build_parameter_bounds_expressions(𝓂, atoms_in_equations, relevant_pars_across)
    par_bounds = []
    
    for varpar in intersect(𝓂.constants.post_complete_parameters.parameters, union(atoms_in_equations, relevant_pars_across))
        if haskey(𝓂.constants.post_parameters_macro.bounds, varpar)
            push!(par_bounds, :($varpar = min(max($varpar,$(𝓂.constants.post_parameters_macro.bounds[varpar][1])),$(𝓂.constants.post_parameters_macro.bounds[varpar][2]))))
        end
    end
    
    return par_bounds
end


"""
    collect_calibration_no_var_parameters!(atoms_in_equations::Set{Symbol}, 𝓂)

Add parameters from calibration_no_var equations to the atoms_in_equations set.
Also returns a set of parameters_only_in_par_defs for internal use.
"""
function collect_calibration_no_var_parameters!(atoms_in_equations::Set{Symbol}, 𝓂)
    parameters_only_in_par_defs = Set()
    if length(𝓂.equations.calibration_no_var) > 0
        atoms = reduce(union, get_symbols.(𝓂.equations.calibration_no_var))
        [push!(atoms_in_equations, a) for a in atoms]
        [push!(parameters_only_in_par_defs, a) for a in atoms]
    end
    return parameters_only_in_par_defs
end


"""
    build_return_variables(𝓂)

Build the return variable list for solve_SS function.

# Returns
- Symbols for variables in return statement
- Length of return vector
"""
function build_return_variables(𝓂)
    vars_expr = Symbol.(replace.(string.(sort(union(
        𝓂.constants.post_model_macro.var,
        𝓂.constants.post_model_macro.exo_past,
        𝓂.constants.post_model_macro.exo_future
    ))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
    
    return_length = length(union(
        𝓂.constants.post_model_macro.var,
        𝓂.constants.post_model_macro.exo_past,
        𝓂.constants.post_model_macro.exo_future
    )) + length(𝓂.equations.calibration_parameters)
    
    return vars_expr, return_length
end


"""
    build_model_specific_solver_core_function(𝓂, parameters_in_equations, par_bounds, SS_solve_func)

Build a RuntimeGeneratedFunction that contains all model-specific steady-state solving logic.
This function takes a parameter vector and returns solved variables and calibration parameters.

# Arguments
- `𝓂`: The model struct
- `parameters_in_equations`: Parameter assignment expressions
- `par_bounds`: Parameter bounds expressions
- `SS_solve_func`: Block solving expressions

# Returns
A RuntimeGeneratedFunction that processes parameters and solves for steady state.
"""
function build_model_specific_solver_core_function(𝓂, parameters_in_equations, par_bounds, SS_solve_func)
    vars_expr, _ = build_return_variables(𝓂)
    
    core_func_exp = quote
        function solve_SS_core(parameters::Vector{<:Real}, 𝓂::ℳ, closest_solution,
                              fail_fast_solvers_only::Bool, cold_start::Bool,
                              solver_parameters::Vector{solver_parameters},
                              verbose::Bool, tol::Tolerances)
            params_flt = parameters
            $(parameters_in_equations...)
            $(par_bounds...)
            $(𝓂.equations.calibration_no_var...)
            NSSS_solver_cache_tmp = []
            solution_error = 0.0
            iters = 0
            $(SS_solve_func...)
            
            return [$(vars_expr...), $(𝓂.equations.calibration_parameters...)], NSSS_solver_cache_tmp, solution_error, iters
        end
    end
    
    return @RuntimeGeneratedFunction(core_func_exp)
end


"""
    build_solve_SS_expression(𝓂, parameters_in_equations, par_bounds, SS_solve_func)

Build the solve_SS function that separates generic solver logic from model-specific computations.
This is the REFACTORED version where:
- Model-specific parts (parameter processing, variable computation) are in a RuntimeGeneratedFunction
- Generic solver logic (iteration, scaling, caching) is in a normal function wrapper

# Arguments
- `𝓂`: The model struct
- `parameters_in_equations`: Parameter assignment expressions
- `par_bounds`: Parameter bounds expressions  
- `SS_solve_func`: Block solving expressions
- `precompiled::Bool`: Whether this is the precompiled version (V2) with multi-element cache structure

# Returns
An Expr representing the solve_SS function.
"""
function build_solve_SS_expression(𝓂, parameters_in_equations, par_bounds, SS_solve_func; precompiled::Bool = false)
    vars_expr, return_length = build_return_variables(𝓂)
    
    # Build the model-specific RGF that handles all parameter/variable operations
    solver_core = build_model_specific_solver_core_function(𝓂, parameters_in_equations, par_bounds, SS_solve_func)
    
    # Store this RGF in the model for use by the normal wrapper function
    𝓂.functions.NSSS_solver_core = solver_core
    
    # Zero initial value check only for precompiled version (V2) which has multi-element cache entries
    zero_init_check = precompiled ? quote
        # Zero initial value if starting without guess
        if !isfinite(sum(abs,closest_solution[2]))
            closest_solution = copy(closest_solution)
            for i in 1:2:length(closest_solution)
                closest_solution[i] = zeros(length(closest_solution[i]))
            end
        end
    end : :()
    
    # This is now a normal function (not an RGF) that orchestrates the solving process
    # It calls the model-specific RGF (solver_core) for all parameter/variable operations
    solve_exp = :(function solve_SS(initial_parameters::Vector{Real}, 
                                    𝓂::ℳ,
                                    tol::Tolerances,
                                    verbose::Bool, 
                                    cold_start::Bool,
                                    solver_parameters::Vector{solver_parameters})
                    initial_parameters = typeof(initial_parameters) == Vector{Float64} ? initial_parameters : ℱ.value.(initial_parameters)

                    initial_parameters_tmp = copy(initial_parameters)

                    parameters = copy(initial_parameters)
                    
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

                    range_iters = 0
                    solution_error = 1.0
                    solved_scale = 0
                    scale = 1.0

                    NSSS_solver_cache_scale = CircularBuffer{Vector{Vector{Float64}}}(500)
                    push!(NSSS_solver_cache_scale, closest_solution_init)

                    while range_iters <= (cold_start ? 1 : 500) && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)
                        range_iters += 1
                        fail_fast_solvers_only = range_iters > 1 ? true : false

                        if abs(solved_scale - scale) < 1e-2
                            break 
                        end

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

                        $zero_init_check

                        if all(isfinite,closest_solution[end]) && initial_parameters != closest_solution_init[end]
                            parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
                        else
                            parameters = copy(initial_parameters)
                        end

                        # Call the model-specific RGF to do all parameter processing and variable solving
                        output, NSSS_solver_cache_tmp, solution_error, iters = 𝓂.functions.NSSS_solver_core(
                            parameters, 𝓂, closest_solution, fail_fast_solvers_only, 
                            cold_start, solver_parameters, verbose, tol)

                        if solution_error < tol.NSSS_acceptance_tol
                            solved_scale = scale
                            if scale == 1
                                return output, (solution_error, iters)
                            else
                                reverse_diff_friendly_push!(NSSS_solver_cache_scale, NSSS_solver_cache_tmp)
                            end

                            if scale > .95
                                scale = 1
                            else
                                scale = scale * .4 + .6
                            end
                        end
                    end
                    return zeros($return_length), (1, 0)
                end)
    
    return solve_exp
end
