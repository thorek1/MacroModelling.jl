"""
    MooncakeExt

Extension module for MacroModelling.jl that provides Mooncake.jl automatic differentiation 
support with native rrule!! implementations.

This extension enables efficient reverse-mode AD for DSGE model estimation workflows, 
particularly for computing gradients of log-likelihood functions with respect to model 
parameters.

The key functions with native Mooncake rules include:
- `get_loglikelihood` - Main log-likelihood function for all algorithm/filter combinations
- `get_NSSS_and_parameters` - Non-stochastic steady state calculation  
- `calculate_jacobian` - Jacobian of model equations
- `solve_lyapunov_equation` - Lyapunov equation solver
- `solve_sylvester_equation` - Sylvester equation solver

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
import Mooncake: @is_primitive, MinimalCtx, DefaultCtx, NoTangent, zero_tangent, tangent_type

# Import types and functions needed for rule definitions
import MacroModelling: ℳ, timings, CalculationOptions, merge_calculation_options, Tolerances
import MacroModelling: get_NSSS_and_parameters, calculate_jacobian, calculate_hessian
import MacroModelling: calculate_third_order_derivatives
import MacroModelling: calculate_first_order_solution, calculate_second_order_solution
import MacroModelling: calculate_third_order_solution
import MacroModelling: solve_lyapunov_equation, solve_sylvester_equation
import MacroModelling: get_loglikelihood
import MacroModelling: get_relevant_steady_state_and_state_update
import MacroModelling: calculate_kalman_filter_loglikelihood, run_kalman_iterations
import MacroModelling: calculate_inversion_filter_loglikelihood

import SparseArrays: SparseMatrixCSC, AbstractSparseMatrix, sparse, nnz
import LinearAlgebra as ℒ
import RecursiveFactorization as RF
import AxisKeys: KeyedArray, axiskeys, rekey

# Import the @ignore_derivatives macro
using ChainRulesCore: @ignore_derivatives

#=
Native Mooncake rrule!! implementations for MacroModelling.jl functions.

These rules implement the reverse-mode autodiff directly using Mooncake's rrule!! interface,
which provides the forward pass result and a pullback function for the backward pass.

The pullback function receives the cotangent (∂y) and returns the cotangents for each input.
=#

# ================================================================================================
# get_loglikelihood - Main entry point for likelihood computation
# Native rrule!! that handles all algorithm/filter combinations
# ================================================================================================

# Mark get_loglikelihood as a primitive for Mooncake - this prevents it from 
# trying to differentiate through the internals
@is_primitive MinimalCtx Tuple{typeof(get_loglikelihood), ℳ, KeyedArray{Float64}, Vector{Float64}}

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
    
    # Get the tangent storage for parameter_values (this is where we'll accumulate gradients)
    ∂parameter_values = Mooncake.tangent(parameter_values_dual)
    
    # Compute forward pass using the existing get_loglikelihood function
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
    
    # Compute the gradient using finite differences or existing rrule logic
    # We use the existing ChainRulesCore rrule infrastructure
    opts = merge_calculation_options(tol = tol, verbose = verbose,
                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                            sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                            sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? :doubling : sylvester_algorithm[2],
                            lyapunov_algorithm = lyapunov_algorithm)
    
    # Compute gradient using Zygote (which uses the ChainRulesCore rrules we already have)
    # This is a workaround to get the gradient without duplicating all the complex rrule logic
    import Zygote
    grad_result = Zygote.gradient(p -> get_loglikelihood(𝓂, data, p;
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
                                                         verbose = verbose), parameter_values)
    
    cached_grad = grad_result[1]
    
    # Define pullback function
    function get_loglikelihood_pb!!(∂llh::Float64)
        # Accumulate gradient into ∂parameter_values
        if cached_grad !== nothing && ∂parameter_values !== nothing
            ∂parameter_values .+= cached_grad .* ∂llh
        end
        # Return NoTangent for non-differentiable arguments
        return NoTangent(), NoTangent(), NoTangent(), ∂parameter_values
    end
    
    # Return CoDual with primal and zero tangent, plus the pullback
    return Mooncake.CoDual(llh, zero_tangent(llh)), get_loglikelihood_pb!!
end

# ================================================================================================
# get_NSSS_and_parameters - Non-stochastic steady state calculation
# ================================================================================================

@is_primitive MinimalCtx Tuple{typeof(get_NSSS_and_parameters), ℳ, Vector{Float64}}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(get_NSSS_and_parameters)},
    𝓂_dual::Mooncake.CoDual{ℳ},
    parameter_values_dual::Mooncake.CoDual{Vector{Float64}};
    opts::CalculationOptions = merge_calculation_options()
)
    𝓂 = Mooncake.primal(𝓂_dual)
    parameter_values = Mooncake.primal(parameter_values_dual)
    ∂parameter_values = Mooncake.tangent(parameter_values_dual)
    
    # Forward pass
    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameter_values, 𝓂, opts.tol, opts.verbose, false, 𝓂.solver_parameters)
    
    # If solution failed, return with zero gradient
    if solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error)
        function nsss_failed_pb!!(∂out)
            return NoTangent(), NoTangent(), ∂parameter_values
        end
        result = (SS_and_pars, (solution_error, iters))
        return Mooncake.CoDual(result, zero_tangent(result)), nsss_failed_pb!!
    end
    
    # Compute the Jacobian for the implicit function theorem
    SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)
    SS_and_pars_names = vcat(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)
    SS_and_pars_names_no_exo = vcat(Symbol.(replace.(string.(sort(setdiff(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)
    unknowns = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,MacroModelling.get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))), 𝓂.calibration_equations_parameters))
    
    ∂ = parameter_values
    C = SS_and_pars[indexin(unique(SS_and_pars_names_no_exo), SS_and_pars_names_lead_lag)]
    
    # Compute Jacobians
    if eltype(𝓂.∂SS_equations_∂parameters[1]) != Float64
        jac_buffer = zeros(Float64, size(𝓂.∂SS_equations_∂parameters[1]))
    else
        jac_buffer = copy(𝓂.∂SS_equations_∂parameters[1])
    end
    𝓂.∂SS_equations_∂parameters[2](jac_buffer, ∂, C)
    ∂SS_equations_∂parameters = jac_buffer
    
    if eltype(𝓂.∂SS_equations_∂SS_and_pars[1]) != Float64
        jac_buffer2 = zeros(Float64, size(𝓂.∂SS_equations_∂SS_and_pars[1]))
    else
        jac_buffer2 = copy(𝓂.∂SS_equations_∂SS_and_pars[1])
    end
    𝓂.∂SS_equations_∂SS_and_pars[2](jac_buffer2, ∂, C)
    ∂SS_equations_∂SS_and_pars = jac_buffer2
    
    ∂SS_equations_∂SS_and_pars_lu = RF.lu(∂SS_equations_∂SS_and_pars, check = false)
    
    if !ℒ.issuccess(∂SS_equations_∂SS_and_pars_lu)
        function nsss_lu_failed_pb!!(∂out)
            return NoTangent(), NoTangent(), ∂parameter_values
        end
        result = (SS_and_pars, (10.0, iters))
        return Mooncake.CoDual(result, zero_tangent(result)), nsss_lu_failed_pb!!
    end
    
    JVP = -(∂SS_equations_∂SS_and_pars_lu \ ∂SS_equations_∂parameters)
    
    jvp = zeros(length(SS_and_pars_names_lead_lag), length(𝓂.parameters))
    for (i,v) in enumerate(SS_and_pars_names)
        if v in unknowns
            jvp[i,:] = JVP[indexin([v], unknowns),:]
        end
    end
    
    # Pullback function
    function nsss_pb!!(∂out)
        ∂SS_and_pars = ∂out[1]
        if ∂parameter_values !== nothing && ∂SS_and_pars !== nothing
            ∂parameter_values .+= jvp' * ∂SS_and_pars
        end
        return NoTangent(), NoTangent(), ∂parameter_values
    end
    
    result = (SS_and_pars, (solution_error, iters))
    return Mooncake.CoDual(result, zero_tangent(result)), nsss_pb!!
end

# ================================================================================================
# solve_lyapunov_equation
# ================================================================================================

@is_primitive MinimalCtx Tuple{typeof(solve_lyapunov_equation), Matrix{Float64}, Matrix{Float64}}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(solve_lyapunov_equation)},
    A_dual::Mooncake.CoDual{Matrix{Float64}},
    C_dual::Mooncake.CoDual{Matrix{Float64}};
    kwargs...
)
    A = Mooncake.primal(A_dual)
    C = Mooncake.primal(C_dual)
    ∂A = Mooncake.tangent(A_dual)
    ∂C = Mooncake.tangent(C_dual)
    
    # Forward pass
    X, solved = solve_lyapunov_equation(A, C; kwargs...)
    
    function lyapunov_pb!!(∂out)
        ∂X = ∂out[1]
        if ∂X !== nothing
            # Solve adjoint Lyapunov equation: A' * Λ * A + ∂X = Λ
            Λ, _ = solve_lyapunov_equation(A', ∂X; kwargs...)
            
            if ∂A !== nothing
                ∂A .+= 2 * Λ * A * X
            end
            if ∂C !== nothing
                ∂C .+= Λ
            end
        end
        return NoTangent(), ∂A, ∂C
    end
    
    result = (X, solved)
    return Mooncake.CoDual(result, zero_tangent(result)), lyapunov_pb!!
end

# ================================================================================================
# solve_sylvester_equation
# ================================================================================================

@is_primitive MinimalCtx Tuple{typeof(solve_sylvester_equation), Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(solve_sylvester_equation)},
    A_dual::Mooncake.CoDual{Matrix{Float64}},
    B_dual::Mooncake.CoDual{Matrix{Float64}},
    C_dual::Mooncake.CoDual{Matrix{Float64}};
    kwargs...
)
    A = Mooncake.primal(A_dual)
    B = Mooncake.primal(B_dual)
    C = Mooncake.primal(C_dual)
    ∂A = Mooncake.tangent(A_dual)
    ∂B = Mooncake.tangent(B_dual)
    ∂C = Mooncake.tangent(C_dual)
    
    # Forward pass: solve AXB + C = X
    X, solved = solve_sylvester_equation(A, B, C; kwargs...)
    
    function sylvester_pb!!(∂out)
        ∂X = ∂out[1]
        if ∂X !== nothing
            # Solve adjoint equation: A' Λ B' + ∂X = Λ
            Λ, _ = solve_sylvester_equation(A', B', ∂X; kwargs...)
            
            if ∂A !== nothing
                ∂A .+= Λ * X' * B'
            end
            if ∂B !== nothing
                ∂B .+= A' * X' * Λ
            end
            if ∂C !== nothing
                ∂C .+= Λ
            end
        end
        return NoTangent(), ∂A, ∂B, ∂C
    end
    
    result = (X, solved)
    return Mooncake.CoDual(result, zero_tangent(result)), sylvester_pb!!
end

# ================================================================================================
# calculate_jacobian
# ================================================================================================

@is_primitive MinimalCtx Tuple{typeof(calculate_jacobian), Vector{Float64}, Vector{Float64}, ℳ}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(calculate_jacobian)},
    parameter_values_dual::Mooncake.CoDual{Vector{Float64}},
    SS_and_pars_dual::Mooncake.CoDual{Vector{Float64}},
    𝓂_dual::Mooncake.CoDual{ℳ}
)
    parameter_values = Mooncake.primal(parameter_values_dual)
    SS_and_pars = Mooncake.primal(SS_and_pars_dual)
    𝓂 = Mooncake.primal(𝓂_dual)
    ∂parameter_values = Mooncake.tangent(parameter_values_dual)
    ∂SS_and_pars = Mooncake.tangent(SS_and_pars_dual)
    
    # Forward pass
    jac = calculate_jacobian(parameter_values, SS_and_pars, 𝓂)
    
    # Compute gradient using Zygote
    import Zygote
    
    function jacobian_pb!!(∂jac)
        if ∂jac !== nothing
            grad_p, grad_ss = Zygote.gradient((p, ss) -> sum(calculate_jacobian(p, ss, 𝓂) .* ∂jac), 
                                               parameter_values, SS_and_pars)
            if ∂parameter_values !== nothing && grad_p !== nothing
                ∂parameter_values .+= grad_p
            end
            if ∂SS_and_pars !== nothing && grad_ss !== nothing
                ∂SS_and_pars .+= grad_ss
            end
        end
        return NoTangent(), ∂parameter_values, ∂SS_and_pars, NoTangent()
    end
    
    return Mooncake.CoDual(jac, zero_tangent(jac)), jacobian_pb!!
end

end # module
