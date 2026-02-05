# General steady state solver orchestration
# This file contains the general logic for solving steady state equations.
# Model-specific calculations are delegated to RuntimeGeneratedFunctions stored in 𝓂.NSSS.

"""
    solve_steady_state_precompiled(
        initial_parameters::Vector{Real},
        𝓂::ℳ,
        tol::Tolerances,
        verbose::Bool,
        cold_start::Bool,
        solver_parameters::Vector{solver_parameters}
    )

General orchestration function for solving steady state equations with precompiled blocks.
This function is a normal Julia function (not a RuntimeGeneratedFunction) that:
- Manages parameter scaling and iterative refinement
- Handles cache management and closest solution tracking
- Iterates over blocks, calling block_solver for each
- Delegates model-specific operations to RTGFs in 𝓂.NSSS

The model-specific RTGFs handle:
- Parameter vector → named parameters (setup_parameters_and_bounds)
- Named vars → block input vectors (block_metadata[i].get_block_inputs)
- Block solutions → updated named vars (block_metadata[i].update_solution)
- Setting dynamic exogenous variables (set_dynamic_exogenous)
- Named vars → solution vector (extract_solution_vector)
"""
function solve_steady_state_precompiled(
    initial_parameters::Vector{Real},
    𝓂::ℳ,
    tol::Tolerances,
    verbose::Bool,
    cold_start::Bool,
    solver_parameters::Vector{solver_parameters}
)
    # Type conversion
    initial_parameters = typeof(initial_parameters) == Vector{Float64} ?
        initial_parameters : ℱ.value.(initial_parameters)

    initial_parameters_tmp = copy(initial_parameters)
    parameters = copy(initial_parameters)
    params_flt = copy(initial_parameters)

    # Find closest cached solution
    current_best = sum(abs2, 𝓂.caches.solver_cache[end][end] - initial_parameters)
    closest_solution_init = 𝓂.caches.solver_cache[end]

    for pars in 𝓂.caches.solver_cache
        copy!(initial_parameters_tmp, pars[end])
        ℒ.axpy!(-1, initial_parameters, initial_parameters_tmp)
        latest = sum(abs2, initial_parameters_tmp)
        if latest <= current_best
            current_best = latest
            closest_solution_init = pars
        end
    end

    # Initialize iteration variables
    range_iters = 0
    solution_error = 1.0
    solved_scale = 0
    scale = 1.0

    NSSS_solver_cache_scale = CircularBuffer{Vector{Vector{Float64}}}(500)
    push!(NSSS_solver_cache_scale, closest_solution_init)

    # Main iteration loop: try different scales to converge
    while range_iters <= (cold_start ? 1 : 500) &&
          !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)

        range_iters += 1
        fail_fast_solvers_only = range_iters > 1

        if abs(solved_scale - scale) < 1e-2
            break
        end

        # Find closest solution in scale-specific cache
        current_best = sum(abs2, NSSS_solver_cache_scale[end][end] - initial_parameters)
        closest_solution = NSSS_solver_cache_scale[end]

        for pars in NSSS_solver_cache_scale
            copy!(initial_parameters_tmp, pars[end])
            ℒ.axpy!(-1, initial_parameters, initial_parameters_tmp)
            latest = sum(abs2, initial_parameters_tmp)
            if latest <= current_best
                current_best = latest
                closest_solution = pars
            end
        end

        # Zero initial value if starting without guess
        if !isfinite(sum(abs, closest_solution[2]))
            closest_solution = copy(closest_solution)
            for i in 1:2:length(closest_solution)
                closest_solution[i] = zeros(length(closest_solution[i]))
            end
        end

        # Scale parameters between current and cached
        if all(isfinite, closest_solution[end]) && initial_parameters != closest_solution_init[end]
            parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
        else
            parameters = copy(initial_parameters)
        end
        params_flt = parameters

        # Setup parameters and apply bounds (model-specific RTGF)
        named_vars = 𝓂.NSSS.setup_parameters_and_bounds(parameters)

        # Initialize block iteration
        NSSS_solver_cache_tmp = []
        solution_error = 0.0
        iters = 0

        # Iterate over blocks
        n_blocks = length(𝓂.NSSS.solve_blocks_in_place)
        for n_block in 1:n_blocks
            # Extract block inputs from named variables (model-specific RTGF)
            block_inputs = 𝓂.NSSS.block_metadata[n_block].get_block_inputs(named_vars)

            # Get bounds and initial guess for this block
            lbs = 𝓂.NSSS.block_metadata[n_block].lower_bounds
            ubs = 𝓂.NSSS.block_metadata[n_block].upper_bounds

            inits = [
                max.(lbs[1:length(closest_solution[2*(n_block-1)+1])],
                     min.(ubs[1:length(closest_solution[2*(n_block-1)+1])],
                          closest_solution[2*(n_block-1)+1])),
                closest_solution[2*n_block]
            ]

            # Solve block using general block_solver
            solution = block_solver(
                length(block_inputs) == 0 ? [0.0] : block_inputs,
                n_block,
                𝓂.NSSS.solve_blocks_in_place[n_block],
                inits,
                lbs,
                ubs,
                solver_parameters,
                fail_fast_solvers_only,
                cold_start,
                verbose
            )

            iters += solution[2][2]
            solution_error += solution[2][1]
            sol = solution[1]

            # Update named variables with block solution (model-specific RTGF)
            named_vars = 𝓂.NSSS.block_metadata[n_block].update_solution(named_vars, sol)

            # Cache solution
            NSSS_solver_cache_tmp = [
                NSSS_solver_cache_tmp...,
                typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol)
            ]
            NSSS_solver_cache_tmp = [
                NSSS_solver_cache_tmp...,
                typeof(block_inputs) == Vector{Float64} ?
                    block_inputs : ℱ.value.(block_inputs)
            ]
        end

        # Set dynamic exogenous variables to zero (model-specific RTGF)
        named_vars = 𝓂.NSSS.set_dynamic_exogenous(named_vars)

        # Update cache with parameter vector
        if length(NSSS_solver_cache_tmp) == 0
            NSSS_solver_cache_tmp = [copy(params_flt)]
        else
            NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., copy(params_flt)]
        end

        # Check for convergence
        current_best = sqrt(sum(abs2, 𝓂.caches.solver_cache[end][end] - params_flt))

        for pars in 𝓂.caches.solver_cache
            latest = sqrt(sum(abs2, pars[end] - params_flt))
            if latest <= current_best
                current_best = latest
            end
        end

        # Update global cache if we found a new good solution
        if (current_best > 1e-8) && (solution_error < tol.NSSS_acceptance_tol)
            reverse_diff_friendly_push!(𝓂.caches.solver_cache, NSSS_solver_cache_tmp)
        end

        # Check convergence and update scale
        if solution_error < tol.NSSS_acceptance_tol
            solved_scale = scale
            if scale == 1
                # Extract solution vector (model-specific RTGF)
                return 𝓂.NSSS.extract_solution_vector(named_vars), (solution_error, iters)
            else
                reverse_diff_friendly_push!(NSSS_solver_cache_scale, NSSS_solver_cache_tmp)
            end

            if scale > 0.95
                scale = 1
            else
                scale = scale * 0.4 + 0.6
            end
        end
    end

    # Failed to converge
    return zeros(𝓂.NSSS.solution_vector_length), (1, 0)
end
