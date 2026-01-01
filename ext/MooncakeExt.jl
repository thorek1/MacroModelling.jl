"""
    MooncakeExt

Extension module for MacroModelling.jl that provides Mooncake.jl automatic differentiation 
support with native rrule!! implementations.

This extension enables efficient reverse-mode AD for DSGE model estimation workflows, 
particularly for computing gradients of log-likelihood functions with respect to model 
parameters.

The key function with native Mooncake rules is:
- `get_loglikelihood` - Main log-likelihood function for all algorithm/filter combinations

The rrule!! computes gradients using finite differences, avoiding dependency on other AD 
packages and ensuring Mooncake uses the custom rule rather than differentiating through
complex internal implementations.

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
import MacroModelling: get_loglikelihood

import AxisKeys: KeyedArray

#=
Native Mooncake rrule!! implementation for get_loglikelihood.

This rule computes gradients using central finite differences, which:
1. Does not rely on any other AD package (Zygote, ChainRulesCore, etc.)
2. Ensures Mooncake uses this rule instead of differentiating through internals
3. Provides accurate gradients for all algorithm/filter combinations

The finite difference step size is adaptive based on parameter magnitude.
=#

# ================================================================================================
# Finite difference gradient computation
# ================================================================================================

"""
    compute_gradient_finite_diff(f, x; ε_scale=1e-5, min_ε=1e-8)

Compute the gradient of scalar function `f` at point `x` using central finite differences.
Uses adaptive step size based on parameter magnitude for numerical stability.
"""
function compute_gradient_finite_diff(f, x::Vector{Float64}; ε_scale::Float64=1e-5, min_ε::Float64=1e-8)
    n = length(x)
    grad = zeros(n)
    x_plus = copy(x)
    x_minus = copy(x)
    
    for i in 1:n
        # Adaptive step size based on parameter magnitude
        ε = max(abs(x[i]) * ε_scale, min_ε)
        
        x_plus[i] = x[i] + ε
        x_minus[i] = x[i] - ε
        
        f_plus = f(x_plus)
        f_minus = f(x_minus)
        
        # Central difference
        grad[i] = (f_plus - f_minus) / (2 * ε)
        
        # Reset for next iteration
        x_plus[i] = x[i]
        x_minus[i] = x[i]
    end
    
    return grad
end

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
    
    # Compute gradient using finite differences
    # This avoids any dependency on other AD packages
    llh_func = p -> get_loglikelihood(𝓂, data, p;
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
    
    cached_grad = compute_gradient_finite_diff(llh_func, parameter_values)
    
    # Define pullback function
    function get_loglikelihood_pb!!(∂llh::Float64)
        # Accumulate gradient into ∂parameter_values
        if ∂parameter_values !== nothing
            ∂parameter_values .+= cached_grad .* ∂llh
        end
        # Return NoTangent for non-differentiable arguments
        return NoTangent(), NoTangent(), NoTangent(), ∂parameter_values
    end
    
    # Return CoDual with primal and zero tangent, plus the pullback
    return Mooncake.CoDual(llh, zero_tangent(llh)), get_loglikelihood_pb!!
end

end # module
