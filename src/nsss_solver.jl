# Non-stochastic steady state (NSSS) solver
# 
# This file contains:
# 1. Step execution functions for individual NSSS solve steps
# 2. The solve_nsss_steps orchestrator that iterates over steps
# 3. The solve_nsss_wrapper that handles cache management and continuation method

using DataStructures: CircularBuffer
import LinearAlgebra as ℒ
import ChainRulesCore: @ignore_derivatives

# Temporary debug counter
const _nsss_debug_counter = Ref(0)


# ============================================================================
# Step execution functions
# ============================================================================

"""
    execute_step!(step::AnalyticalNSSSStep, sol_vec, params_vec, args...)

Execute an analytical NSSS solve step. Evaluates the compiled symbolic function
to compute one or more unknowns and writes them to the solution vector.

Returns: (error, iterations, cache_entries)
"""
function execute_step!(step::AnalyticalNSSSStep, sol_vec::Vector{Float64}, 
                       params_vec::Vector{Float64},
                       closest_solution, 𝓂, tol, fail_fast_solvers_only,
                       cold_start, solver_parameters, verbose)
    error = 0.0
    
    # Phase 1: Compute auxiliary variables (domain-safety ➕_vars)
    if step.aux_func! !== nothing
        step.aux_func!(step.aux_buffer, sol_vec, params_vec)
        for (i, idx) in enumerate(step.aux_write_indices)
            sol_vec[idx] = step.aux_buffer[i]
        end
        
        # Domain safety error check
        if step.error_func! !== nothing
            step.error_func!(step.error_buffer, sol_vec, params_vec)
            error += sum(abs, step.error_buffer)
        end
    end
    
    # Phase 2: Compute target variable(s)
    if !isempty(step.write_indices)
        step.eval_func!(step.buffer, sol_vec, params_vec)
        
        # Apply bounds: ➕_vars get clamped, user-bounded vars only get error
        for (i, idx) in enumerate(step.write_indices)
            raw = step.buffer[i]
            if step.has_bounds[i]
                clamped = clamp(raw, step.lower_bounds[i], step.upper_bounds[i])
                error += abs(clamped - raw)
                if step.clamp_to_bounds[i]
                    # ➕_var: store clamped value (matches main branch behavior)
                    sol_vec[idx] = clamped
                else
                    # User bounds: store raw value, only accumulate error (matches main branch)
                    sol_vec[idx] = raw
                end
            else
                sol_vec[idx] = raw
            end
        end
    end
    
    return error, 0, Vector{Float64}[]
end


"""
    execute_step!(step::NumericalNSSSStep, sol_vec, params_vec, args...)

Execute a numerical NSSS solve step. Gathers parameters and solved variables,
then calls `block_solver` to numerically solve for the unknowns.

Returns: (error, iterations, cache_entries)
"""
function execute_step!(step::NumericalNSSSStep, sol_vec::Vector{Float64}, 
                       params_vec::Vector{Float64},
                       closest_solution, 𝓂, tol, fail_fast_solvers_only,
                       cold_start, solver_parameters, verbose)
    error = 0.0
    
    # Phase 1: Compute auxiliary variables (domain-safety, if any)
    if step.aux_func! !== nothing
        step.aux_func!(step.aux_buffer, sol_vec, params_vec)
        for (i, idx) in enumerate(step.aux_write_indices)
            sol_vec[idx] = step.aux_buffer[i]
        end
        
        # Domain safety error check
        if step.aux_error_func! !== nothing
            step.aux_error_func!(step.aux_error_buffer, sol_vec, params_vec)
            error += sum(abs, step.aux_error_buffer)
            if error > tol.NSSS_acceptance_tol
                if verbose
                    println("Failed for aux variables with error $error")
                end
                return error, 0, Vector{Float64}[]
            end
        end
    end
    
    # Gather params_and_solved_vars from the solution and parameter vectors
    n_params = length(step.param_gather_indices)
    n_vars = length(step.var_gather_indices)
    params_and_solved_vars = Vector{Float64}(undef, n_params + n_vars)
    for (i, idx) in enumerate(step.param_gather_indices)
        params_and_solved_vars[i] = params_vec[idx]
    end
    for (i, idx) in enumerate(step.var_gather_indices)
        params_and_solved_vars[n_params + i] = sol_vec[idx]
    end
    
    # Build initial guesses from closest cached solution
    n = step.block_index
    cache_sol = closest_solution[2*(n-1)+1]
    cache_par = closest_solution[2*n]
    inits = [
        max.(step.lbs[1:length(cache_sol)], min.(step.ubs[1:length(cache_sol)], cache_sol)),
        cache_par
    ]
    
    # Call block solver
    solution = block_solver(
        params_and_solved_vars,
        n,
        𝓂.NSSS.solve_blocks_in_place[n],
        inits,
        step.lbs,
        step.ubs,
        solver_parameters,
        fail_fast_solvers_only,
        cold_start,
        verbose
    )
    
    # Accumulate error and iterations
    error += solution[2][1]
    iters = solution[2][2]
    
    # Write results to solution vector
    sol = solution[1]
    for (i, idx) in enumerate(step.write_indices)
        sol_vec[idx] = sol[i]
    end
    
    # Build cache entries for this block
    cache_entries = [
        typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol),
        typeof(params_and_solved_vars) == Vector{Float64} ? params_and_solved_vars : ℱ.value.(params_and_solved_vars)
    ]
    
    return error, iters, cache_entries
end


# ============================================================================
# Orchestrator: solve_nsss_steps
# ============================================================================

"""
    solve_nsss_steps(parameters, 𝓂, tol, verbose, fail_fast_solvers_only,
                     closest_solution, cold_start, solver_params)

Solve the NSSS by executing pre-compiled solve steps in a single pass.

Each step is either an `AnalyticalNSSSStep` (compiled symbolic evaluation)
or a `NumericalNSSSStep` (calls block_solver). Steps are executed in order,
filling the solution vector progressively.

Returns: (SS_and_pars, (solution_error, iters), NSSS_solver_cache_tmp)
"""
function solve_nsss_steps(
    parameters::Vector{Float64},
    𝓂::ℳ,
    tol::Tolerances,
    verbose::Bool,
    fail_fast_solvers_only::Bool,
    closest_solution,
    cold_start::Bool,
    solver_params::Vector{solver_parameters}
)
    nsss = 𝓂.NSSS
    
    # Prepare extended parameter vector (raw params → bounded + calibration_no_var)
    params_vec = Vector{Float64}(undef, nsss.n_ext_params)
    nsss.param_prep!(params_vec, parameters)
    
    # Initialize solution vector
    sol_vec = zeros(Float64, nsss.n_sol)
    
    # Single pass through all steps
    NSSS_solver_cache_tmp = Vector{Float64}[]
    solution_error = 0.0
    iters = 0
    
    for step in nsss.solve_steps
        step_error, step_iters, step_cache = execute_step!(
            step, sol_vec, params_vec, closest_solution, 𝓂, tol,
            fail_fast_solvers_only, cold_start, solver_params, verbose
        )
        
        # Accumulate errors from ALL step types (analytical and numerical).
        # On the main branch, every error check triggers scale adjustment + continue
        # when solution_error exceeds tolerance. We replicate this by breaking out
        # of the step loop and letting the wrapper handle scale adjustment.
        solution_error += step_error
        
        if step isa NumericalNSSSStep
            iters += step_iters
            append!(NSSS_solver_cache_tmp, step_cache)
        end
        
        if solution_error > tol.NSSS_acceptance_tol
            if verbose || _nsss_debug_counter[] <= 3
                desc = step isa NumericalNSSSStep ? step.description : (step isa AnalyticalNSSSStep ? step.description : "unknown")
                println("  [NSSS] Break at step '$desc': step_err=$step_error, cum_err=$solution_error")
                _nsss_debug_counter[] += 1
            end
            break
        end
    end
    
    # Build SS_and_pars from solution vector using output indices
    SS_and_pars = sol_vec[nsss.output_indices]
    
    # If failed to converge, return zeros
    if solution_error >= tol.NSSS_acceptance_tol
        SS_and_pars = zeros(Float64, length(nsss.output_indices))
    end
    
    # Append parameters to cache 
    if isempty(NSSS_solver_cache_tmp)
        NSSS_solver_cache_tmp = [copy(parameters)]
    else
        push!(NSSS_solver_cache_tmp, copy(parameters))
    end
    
    return SS_and_pars, (solution_error, iters), NSSS_solver_cache_tmp
end


# ============================================================================
# Wrapper: solve_nsss_wrapper (handles cache + continuation method)
# ============================================================================

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
    SS_and_pars = Float64[]
    
    # Local intermediate cache for warm starts at intermediate scales
    NSSS_solver_cache_scale = CircularBuffer{Vector{Vector{Float64}}}(500)
    push!(NSSS_solver_cache_scale, closest_solution_init)
    
    # Continuation method: iterate with scaling to gradually approach target
    max_iters = cold_start ? 1 : 500
    
    while range_iters <= max_iters && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)
        range_iters += 1
        fail_fast_solvers_only = range_iters > 1
        
        # Stall detection: if scale and solved_scale are very close, break
        # Matches main branch logic at line ~5088
        if abs(solved_scale - scale) < 1e-2
            break
        end
        
        # Find closest solution from LOCAL intermediate cache
        current_best = sum(abs2, NSSS_solver_cache_scale[end][end] - initial_parameters)
        closest_solution = NSSS_solver_cache_scale[end]
        
        for pars in NSSS_solver_cache_scale
            latest = sum(abs2, pars[end] - initial_parameters)
            if latest <= current_best
                current_best = latest
                closest_solution = pars
            end
        end
        
        # Interpolate parameters between target and cached solution
        if all(isfinite, closest_solution[end]) && initial_parameters != closest_solution_init[end]
            parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
        else
            parameters = copy(initial_parameters)
        end
        
        # Call step-based solver
        SS_and_pars, (solution_error, iters), NSSS_solver_cache_tmp = solve_nsss_steps(
            parameters,
            𝓂,
            tol,
            verbose,
            fail_fast_solvers_only,
            closest_solution,
            cold_start,
            solver_params
        )
        
        # Check convergence and update scaling
        if solution_error < tol.NSSS_acceptance_tol
            solved_scale = scale
            
            if scale == 1
                # Fully converged at target parameters — update global cache and return
                current_best_global = sqrt(sum(abs2, 𝓂.caches.solver_cache[end][end] - initial_parameters))
                for pars in 𝓂.caches.solver_cache
                    latest = sqrt(sum(abs2, pars[end] - initial_parameters))
                    if latest <= current_best_global
                        current_best_global = latest
                    end
                end
                if current_best_global > 1e-8
                    reverse_diff_friendly_push!(𝓂.caches.solver_cache, NSSS_solver_cache_tmp)
                end
                
                return SS_and_pars, (solution_error, iters)
            end
            
            # Cache intermediate result for warm starts
            push!(NSSS_solver_cache_scale, NSSS_solver_cache_tmp)
            
            # Advance scale toward 1.0
            if scale > 0.95
                scale = 1.0
            else
                scale = scale * 0.4 + 0.6
            end
        end
        # Note: Main branch has NO explicit failure-case scale adjustment.
        # When solution_error >= tol, the loop simply continues with the same scale,
        # relying on stall detection (abs(solved_scale - scale) < 1e-2) to break.
    end
    
    # Failed to converge - return zeros with matching output length
    n_output = length(𝓂.NSSS.output_indices)
    
    return zeros(n_output), (1.0, 0)
end
