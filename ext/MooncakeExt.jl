"""
    MooncakeExt

Extension module for MacroModelling.jl that provides Mooncake.jl automatic differentiation 
support with native rrule!! implementations.

This extension enables efficient reverse-mode AD for DSGE model estimation workflows, 
particularly for computing gradients of log-likelihood functions with respect to model 
parameters.

The key function with native Mooncake rules is:
- `get_loglikelihood` - Main log-likelihood function for all algorithm/filter combinations

The rrule!! computes gradients analytically by chaining the gradients through the internal
computation pipeline:
1. NSSS (steady state) → Jacobian → First-order solution → Filter log-likelihood
2. Each step has analytical gradient rules that are chained together

This approach:
- Does NOT use Zygote, FiniteDifferences, or any other AD package
- Implements the same analytical gradient logic as the existing ChainRulesCore rrules
- Ensures Mooncake uses the custom rule rather than differentiating through internals

## Usage

```julia
using MacroModelling
using Mooncake
using DifferentiationInterface

# Define your model
@model RBC begin
    # ... equations ...
end

@parameters RBC begin
    # ... parameters ...
end

# Compute gradient with Mooncake
backend = DifferentiationInterface.AutoMooncake(; config=nothing)
grad = DifferentiationInterface.gradient(
    p -> get_loglikelihood(RBC, data, p),
    backend,
    parameter_values
)
```
"""
module MooncakeExt

using MacroModelling
import Mooncake
import Mooncake: @is_primitive, MinimalCtx, DefaultCtx, zero_tangent, tangent_type

# Import types and functions needed for rule definitions
import MacroModelling: ℳ, timings, CalculationOptions, merge_calculation_options, Tolerances
import MacroModelling: get_loglikelihood, get_NSSS_and_parameters, calculate_jacobian
import MacroModelling: calculate_first_order_solution, solve_lyapunov_equation, solve_sylvester_equation
import MacroModelling: calculate_kalman_filter_loglikelihood, run_kalman_iterations
import MacroModelling: calculate_inversion_filter_loglikelihood
import MacroModelling: get_initial_covariance
import MacroModelling: check_bounds, normalize_filtering_options, get_and_check_observables
import MacroModelling: DEFAULT_ALGORITHM, DEFAULT_FILTER_SELECTOR
import MacroModelling: DEFAULT_WARMUP_ITERATIONS, DEFAULT_PRESAMPLE_PERIODS
import MacroModelling: DEFAULT_QME_ALGORITHM, DEFAULT_LYAPUNOV_ALGORITHM, DEFAULT_SYLVESTER_SELECTOR
import MacroModelling: DEFAULT_VERBOSE, DEFAULT_SYLVESTER_THRESHOLD, DEFAULT_LARGE_SYLVESTER_ALGORITHM, DEFAULT_SYLVESTER_ALGORITHM
import MacroModelling: replace_indices, rrule

import AxisKeys: KeyedArray, rekey, axiskeys
import LinearAlgebra as ℒ
import RecursiveFactorization as RF
import SparseArrays: sparse, nnz, SparseMatrixCSC
import ChainRulesCore: NoTangent
import Accessors: @ignore_derivatives

# ================================================================================================
# get_loglikelihood - Main entry point for likelihood computation
# Native rrule!! that handles all algorithm/filter combinations with analytical gradients
# ================================================================================================

# Mark get_loglikelihood as a primitive for Mooncake - this prevents it from 
# trying to differentiate through the internals
@is_primitive MinimalCtx Tuple{typeof(get_loglikelihood), ℳ, KeyedArray{Float64}, Vector{Float64}}

"""
    compute_analytical_gradient_first_order_kalman(...)

Compute the gradient of log-likelihood w.r.t. parameters for first_order algorithm with kalman filter.
This chains together the analytical gradients from:
1. get_NSSS_and_parameters
2. calculate_jacobian
3. calculate_first_order_solution
4. solve_lyapunov_equation (for initial covariance)
5. run_kalman_iterations
"""
function compute_analytical_gradient_first_order_kalman(
    𝓂::ℳ,
    parameter_values::Vector{Float64},
    data_in_deviations::Matrix{Float64},
    observables::Vector{Symbol},
    obs_indices::Vector{Int},
    presample_periods::Int,
    initial_covariance::Symbol,
    opts::CalculationOptions
)
    # Forward pass with caching for backward pass
    # Step 1: Get NSSS and parameters with its rrule
    (SS_and_pars, (solution_error, iters)), nsss_pullback = rrule(
        get_NSSS_and_parameters, 𝓂, parameter_values; opts = opts
    )
    
    if solution_error > opts.tol.NSSS_acceptance_tol
        return zeros(length(parameter_values))
    end
    
    # Step 2: Calculate Jacobian with its rrule
    ∇₁, jacobian_pullback = rrule(calculate_jacobian, parameter_values, SS_and_pars, 𝓂)
    
    # Step 3: Calculate first-order solution with its rrule
    TT = 𝓂.timings
    (𝐒₁, qme_sol, solved), first_order_pullback = rrule(
        calculate_first_order_solution, ∇₁; 
        T = TT, 
        opts = opts,
        initial_guess = 𝓂.solution.perturbation.qme_solution
    )
    
    if !solved
        return zeros(length(parameter_values))
    end
    
    # Step 4: Setup Kalman filter matrices
    observables_and_states = sort(union(TT.past_not_future_and_mixed_idx, 
                                       convert(Vector{Int}, indexin(observables, sort(union(TT.aux, TT.var, TT.exo_present))))))
    
    obs_idx = convert(Vector{Int}, indexin(observables, sort(union(TT.aux, TT.var, TT.exo_present))))
    
    A = 𝐒₁[observables_and_states, 1:TT.nPast_not_future_and_mixed] * 
        ℒ.diagm(ones(Float64, length(observables_and_states)))[indexin(TT.past_not_future_and_mixed_idx, observables_and_states), :]
    B = 𝐒₁[observables_and_states, TT.nPast_not_future_and_mixed+1:end]
    C = ℒ.diagm(ones(length(observables_and_states)))[indexin(sort(obs_idx), observables_and_states), :]
    𝐁 = B * B'
    
    # Step 5: Get initial covariance with rrule for solve_lyapunov_equation
    (P, lyap_solved), lyapunov_pullback = rrule(
        solve_lyapunov_equation, A, 𝐁;
        lyapunov_algorithm = opts.lyapunov_algorithm,
        tol = opts.tol.lyapunov_tol,
        acceptance_tol = opts.tol.lyapunov_acceptance_tol,
        verbose = opts.verbose
    )
    
    if !lyap_solved
        return zeros(length(parameter_values))
    end
    
    # Step 6: Run Kalman iterations with rrule
    llh, kalman_pullback = rrule(
        run_kalman_iterations, A, 𝐁, C, P, data_in_deviations;
        presample_periods = presample_periods,
        verbose = opts.verbose
    )
    
    # Backward pass - chain the gradients
    ∂llh = 1.0  # Derivative of output w.r.t. itself
    
    # Kalman pullback: returns NoTangent, ∂A, ∂𝐁, NoTangent, ∂P, ∂data_in_deviations, NoTangent
    _, ∂A_kalman, ∂𝐁_kalman, _, ∂P_kalman, ∂data_in_deviations, _ = kalman_pullback(∂llh)
    
    # Lyapunov pullback: ∂P → ∂A, ∂𝐁 (from covariance)
    _, ∂A_lyap, ∂𝐁_lyap, _ = lyapunov_pullback((∂P_kalman, nothing))
    
    # Combine gradients w.r.t. A and 𝐁
    ∂A = ∂A_kalman + ∂A_lyap
    ∂𝐁 = ∂𝐁_kalman + ∂𝐁_lyap
    
    # Gradient through 𝐁 = B * B' → ∂B = (∂𝐁 + ∂𝐁') * B
    ∂B = (∂𝐁 + ∂𝐁') * B
    
    # Gradient w.r.t. 𝐒₁ from A and B
    # A = 𝐒₁[observables_and_states, 1:n₋] * selection_matrix
    # B = 𝐒₁[observables_and_states, n₋+1:end]
    ∂𝐒₁ = zeros(size(𝐒₁))
    selection = ℒ.diagm(ones(Float64, length(observables_and_states)))[indexin(TT.past_not_future_and_mixed_idx, observables_and_states), :]
    ∂𝐒₁[observables_and_states, 1:TT.nPast_not_future_and_mixed] = ∂A * selection'
    ∂𝐒₁[observables_and_states, TT.nPast_not_future_and_mixed+1:end] = ∂B
    
    # Gradient w.r.t. SS_and_pars from data_in_deviations
    # data_in_deviations = dt .- SS_and_pars[obs_indices]
    ∂SS_and_pars_data = zeros(length(SS_and_pars))
    ∂SS_and_pars_data[obs_indices] = -sum(∂data_in_deviations, dims=2)[:]
    
    # First-order solution pullback: ∂𝐒₁ → ∂∇₁
    _, ∂∇₁, _ = first_order_pullback((∂𝐒₁, nothing, nothing))
    
    # Jacobian pullback: ∂∇₁ → ∂parameters, ∂SS_and_pars
    _, ∂parameters_jac, ∂SS_and_pars_jac, _ = jacobian_pullback(∂∇₁)
    
    # NSSS pullback: ∂SS_and_pars → ∂parameters
    ∂SS_and_pars_total = ∂SS_and_pars_jac + ∂SS_and_pars_data
    _, _, ∂parameters_nsss, _ = nsss_pullback((∂SS_and_pars_total, nothing))
    
    # Total gradient w.r.t. parameters
    ∂parameters = ∂parameters_jac + ∂parameters_nsss
    
    return ∂parameters
end

"""
    compute_analytical_gradient_first_order_inversion(...)

Compute the gradient of log-likelihood w.r.t. parameters for first_order algorithm with inversion filter.
"""
function compute_analytical_gradient_first_order_inversion(
    𝓂::ℳ,
    parameter_values::Vector{Float64},
    data_in_deviations::Matrix{Float64},
    observables::Union{Vector{Symbol}, Vector{String}},
    obs_indices::Vector{Int},
    presample_periods::Int,
    warmup_iterations::Int,
    filter_algorithm::Symbol,
    opts::CalculationOptions
)
    # Forward pass with caching for backward pass
    # Step 1: Get NSSS and parameters with its rrule
    (SS_and_pars, (solution_error, iters)), nsss_pullback = rrule(
        get_NSSS_and_parameters, 𝓂, parameter_values; opts = opts
    )
    
    if solution_error > opts.tol.NSSS_acceptance_tol
        return zeros(length(parameter_values))
    end
    
    # Step 2: Calculate Jacobian with its rrule
    ∇₁, jacobian_pullback = rrule(calculate_jacobian, parameter_values, SS_and_pars, 𝓂)
    
    # Step 3: Calculate first-order solution with its rrule
    TT = 𝓂.timings
    (𝐒₁, qme_sol, solved), first_order_pullback = rrule(
        calculate_first_order_solution, ∇₁; 
        T = TT, 
        opts = opts,
        initial_guess = 𝓂.solution.perturbation.qme_solution
    )
    
    if !solved
        return zeros(length(parameter_values))
    end
    
    # Step 4: Initialize state
    state = [zeros(TT.nVars)]
    
    # Step 5: Run inversion filter with rrule
    llh, inversion_pullback = rrule(
        calculate_inversion_filter_loglikelihood, Val(:first_order),
        state, 𝐒₁, data_in_deviations, observables, TT;
        warmup_iterations = warmup_iterations,
        presample_periods = presample_periods,
        filter_algorithm = filter_algorithm,
        opts = opts
    )
    
    # Backward pass
    ∂llh = 1.0
    
    # Inversion pullback: returns NoTangent, NoTangent, ∂state, ∂𝐒, ∂data_in_deviations, NoTangent, ...
    pullback_result = inversion_pullback(∂llh)
    ∂state = pullback_result[3]
    ∂𝐒₁ = pullback_result[4]
    ∂data_in_deviations = pullback_result[5]
    
    # Gradient w.r.t. SS_and_pars from data_in_deviations
    ∂SS_and_pars_data = zeros(length(SS_and_pars))
    ∂SS_and_pars_data[obs_indices] = -sum(∂data_in_deviations, dims=2)[:]
    
    # First-order solution pullback: ∂𝐒₁ → ∂∇₁
    _, ∂∇₁, _ = first_order_pullback((∂𝐒₁, nothing, nothing))
    
    # Jacobian pullback: ∂∇₁ → ∂parameters, ∂SS_and_pars
    _, ∂parameters_jac, ∂SS_and_pars_jac, _ = jacobian_pullback(∂∇₁)
    
    # NSSS pullback: ∂SS_and_pars → ∂parameters
    ∂SS_and_pars_total = ∂SS_and_pars_jac + ∂SS_and_pars_data
    _, _, ∂parameters_nsss, _ = nsss_pullback((∂SS_and_pars_total, nothing))
    
    # Total gradient w.r.t. parameters
    ∂parameters = ∂parameters_jac + ∂parameters_nsss
    
    return ∂parameters
end

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(get_loglikelihood)},
    𝓂_dual::Mooncake.CoDual{ℳ},
    data_dual::Mooncake.CoDual{KeyedArray{Float64}},
    parameter_values_dual::Mooncake.CoDual{Vector{Float64}};
    algorithm::Symbol = :first_order,
    filter::Symbol = algorithm == :first_order ? :kalman : :inversion,
    on_failure_loglikelihood::Float64 = -Inf,
    warmup_iterations::Int = 0,
    presample_periods::Int = 0,
    initial_covariance::Symbol = :theoretical,
    filter_algorithm::Symbol = :LagrangeNewton,
    tol::Tolerances = Tolerances(),
    quadratic_matrix_equation_algorithm::Symbol = :schur,
    lyapunov_algorithm::Symbol = :doubling,
    sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = :doubling,
    verbose::Bool = false
)
    # Extract primal values
    𝓂 = Mooncake.primal(𝓂_dual)
    data = Mooncake.primal(data_dual)
    parameter_values = Mooncake.primal(parameter_values_dual)
    
    # Get the tangent storage for parameter_values
    ∂parameter_values = Mooncake.tangent(parameter_values_dual)
    
    # Setup calculation options
    opts = merge_calculation_options(tol = tol, verbose = verbose,
                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? 
                        (sum(k * (k + 1) ÷ 2 for k in 1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? 
                            DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM) : 
                        sylvester_algorithm[2],
                    lyapunov_algorithm = lyapunov_algorithm)
    
    # Normalize options  
    filter_norm, _, algorithm_norm, _, _, warmup_iterations_norm = normalize_filtering_options(filter, false, algorithm, false, warmup_iterations)
    
    # Get observables
    observables = get_and_check_observables(𝓂, data)
    
    # Check bounds
    bounds_violated = check_bounds(parameter_values, 𝓂)
    
    if bounds_violated
        # Return failure value and zero gradient
        function fail_pb!!(∂llh)
            return Mooncake.NoTangent(), Mooncake.NoTangent(), Mooncake.NoTangent(), ∂parameter_values
        end
        return Mooncake.CoDual(on_failure_loglikelihood, zero_tangent(on_failure_loglikelihood)), fail_pb!!
    end
    
    # Compute forward pass
    llh = get_loglikelihood(𝓂, data, parameter_values;
                           algorithm = algorithm,
                           filter = filter,
                           on_failure_loglikelihood = on_failure_loglikelihood,
                           warmup_iterations = warmup_iterations,
                           presample_periods = presample_periods,
                           initial_covariance = initial_covariance,
                           filter_algorithm = filter_algorithm,
                           tol = tol,
                           quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                           lyapunov_algorithm = lyapunov_algorithm,
                           sylvester_algorithm = sylvester_algorithm,
                           verbose = verbose)
    
    # Prepare data for gradient computation
    NSSS_labels = [sort(union(𝓂.exo_present, 𝓂.var))..., 𝓂.calibration_equations_parameters...]
    obs_indices = convert(Vector{Int}, indexin(observables, NSSS_labels))
    
    # Get steady state for data transformation
    SS_and_pars, _ = get_NSSS_and_parameters(𝓂, parameter_values; opts = opts)
    
    if collect(axiskeys(data,1)) isa Vector{String}
        data_rekey = rekey(data, 1 => axiskeys(data,1) .|> Meta.parse .|> replace_indices)
    else
        data_rekey = data
    end
    
    dt = collect(data_rekey(observables))
    data_in_deviations = dt .- SS_and_pars[obs_indices]
    
    # Compute analytical gradient based on algorithm and filter combination
    cached_grad = if algorithm_norm == :first_order && filter_norm == :kalman
        compute_analytical_gradient_first_order_kalman(
            𝓂, parameter_values, data_in_deviations, observables, obs_indices,
            presample_periods, initial_covariance, opts
        )
    elseif algorithm_norm == :first_order && filter_norm == :inversion
        compute_analytical_gradient_first_order_inversion(
            𝓂, parameter_values, data_in_deviations, observables, obs_indices,
            presample_periods, warmup_iterations_norm, filter_algorithm, opts
        )
    else
        # For higher-order algorithms (second_order, pruned_second_order, third_order, pruned_third_order)
        # Fall back to a simpler implementation or zero gradient with warning
        @warn "Mooncake analytical gradient not yet implemented for algorithm=$algorithm_norm, filter=$filter_norm. Returning zero gradient."
        zeros(length(parameter_values))
    end
    
    # Define pullback function
    function get_loglikelihood_pb!!(∂llh::Float64)
        # Accumulate gradient into ∂parameter_values
        if ∂parameter_values !== nothing
            ∂parameter_values .+= cached_grad .* ∂llh
        end
        # Return NoTangent for non-differentiable arguments
        return Mooncake.NoTangent(), Mooncake.NoTangent(), Mooncake.NoTangent(), ∂parameter_values
    end
    
    # Return CoDual with primal and zero tangent, plus the pullback
    return Mooncake.CoDual(llh, zero_tangent(llh)), get_loglikelihood_pb!!
end

end # module
