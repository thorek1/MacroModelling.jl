# Non-stochastic steady state (NSSS) solver
# 
# This file contains the normal Julia function wrapper for NSSS solving.
# The wrapper handles cache management and continuation method, while delegating
# model-specific equation solving to the RTGF.

using DataStructures: CircularBuffer
import LinearAlgebra as ℒ
import ChainRulesCore: @ignore_derivatives

"""
    solve_nsss_wrapper(
        parameter_values::Vector{<:Real},
        𝓂::ℳ,
        tol::Tolerances,
        verbose::Bool,
        cold_start::Bool,
        solver_params::Vector{solver_parameters}
    )::Tuple{Vector, Tuple{Real, Int}}

Normal Julia function wrapper for NSSS solving.

This function handles the cache management and continuation method for solving
the non-stochastic steady state. It delegates model-specific equation solving
to the RTGF `𝓂.functions.NSSS_solve`.

The continuation method gradually transitions from a cached solution to the
target parameters using a scaling approach, which improves convergence.

# Arguments
- `parameter_values`: Parameter values to solve at
- `𝓂`: Model structure
- `tol`: Tolerance settings
- `verbose`: Whether to print verbose output
- `cold_start`: Whether this is a cold start (limits iterations to 1)
- `solver_params`: Solver configuration

# Returns
- Tuple of (solution_vector, (solution_error, iterations))
"""
function solve_nsss_wrapper(
    parameter_values::Vector{<:Real},
    𝓂::ℳ,
    tol::Tolerances,
    verbose::Bool,
    cold_start::Bool,
    solver_params::Vector{solver_parameters}
)::Tuple{Vector, Tuple{Real, Int}}
    
    # Type conversion for AD compatibility
    initial_parameters = typeof(parameter_values) == Vector{Float64} ? 
                        parameter_values : 
                        ℱ.value.(parameter_values)
    
    # Find closest cached solution as starting point
    current_best = sum(abs2, 𝓂.caches.solver_cache[end][end] - initial_parameters)
    closest_solution_init = 𝓂.caches.solver_cache[end]
    
    for pars in 𝓂.caches.solver_cache
        latest = sum(abs2, pars[end] - initial_parameters)
        if latest <= current_best
            current_best = latest
            closest_solution_init = pars
        end
    end
    
    # Initialize continuation method variables
    range_iters = 0
    solution_error = 1.0
    solved_scale = 0.0
    scale = 1.0
    
    # Continuation method: iterate with scaling to gradually approach target
    max_iters = cold_start ? 1 : 500
    
    while range_iters <= max_iters && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)
        range_iters += 1
        fail_fast_solvers_only = range_iters > 1
        
        # Find closest solution in cache for this iteration
        current_best = sum(abs2, 𝓂.caches.solver_cache[end][end] - initial_parameters)
        closest_solution = 𝓂.caches.solver_cache[end]
        
        for pars in 𝓂.caches.solver_cache
            latest = sum(abs2, pars[end] - initial_parameters)
            if latest <= current_best
                current_best = latest
                closest_solution = pars
            end
        end
        
        # Zero initial value if starting without valid guess
        # Only applies to non-CircularBuffer version with solution cache structure
        if length(closest_solution) > 1 && !isfinite(sum(abs, closest_solution[2]))
            closest_solution = copy(closest_solution)
            for i in 1:2:length(closest_solution)
                closest_solution[i] = zeros(length(closest_solution[i]))
            end
        end
        
        # Interpolate parameters between current and cached solution
        if all(isfinite, closest_solution[end]) && initial_parameters != closest_solution_init[end]
            parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
        else
            parameters = copy(initial_parameters)
        end
        
        # Call model-specific RTGF to solve equations at scaled parameters
        SS_and_pars, (solution_error, iters), NSSS_solver_cache_tmp = 𝓂.functions.NSSS_solve(
            parameters,
            𝓂,
            tol,
            verbose,
            fail_fast_solvers_only,
            solver_params
        )
        
        # Check convergence and update scaling
        if solution_error < tol.NSSS_acceptance_tol
            solved_scale = scale
            
            if scale == 1
                # Fully converged at target parameters
                return SS_and_pars, (solution_error, iters)
            end
            
            # Update scale for next iteration
            if scale > 0.95
                scale = 1.0
            else
                scale = scale * 0.4 + 0.6
            end
        end
    end
    
    # Failed to converge - return zeros
    n_vars = length(union(
        𝓂.constants.post_model_macro.var,
        𝓂.constants.post_model_macro.exo_past,
        𝓂.constants.post_model_macro.exo_future
    ))
    n_params = length(𝓂.equations.calibration_parameters)
    
    return zeros(n_vars + n_params), (1.0, 0)
end
