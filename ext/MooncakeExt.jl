"""
    MooncakeExt

Extension module for MacroModelling.jl that provides Mooncake.jl automatic differentiation 
support by wrapping existing ChainRulesCore rrules.

This extension enables efficient reverse-mode AD for DSGE model estimation workflows, 
particularly for computing gradients of log-likelihood functions with respect to model 
parameters.

The key functions wrapped include:
- `get_NSSS_and_parameters` - Non-stochastic steady state calculation
- `calculate_jacobian` - Jacobian of model equations
- `calculate_hessian` - Hessian of model equations  
- `calculate_third_order_derivatives` - Third order derivatives
- `calculate_first_order_solution` - First order perturbation solution
- `calculate_second_order_solution` - Second order perturbation solution
- `calculate_third_order_solution` - Third order perturbation solution
- `solve_lyapunov_equation` - Lyapunov equation solver
- `solve_sylvester_equation` - Sylvester equation solver

## Usage

When both MacroModelling and Mooncake are loaded, this extension automatically makes
Mooncake aware of the custom differentiation rules defined in MacroModelling.jl via
ChainRulesCore. This allows Mooncake to efficiently compute gradients without
differentiating through the complex internal implementations.

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

# Compute gradients of log-likelihood with Mooncake
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
import Mooncake: @from_rrule

# Import types and functions that are needed for rule definitions
import MacroModelling: ℳ, timings, CalculationOptions, merge_calculation_options
import MacroModelling: get_NSSS_and_parameters, calculate_jacobian, calculate_hessian
import MacroModelling: calculate_third_order_derivatives
import MacroModelling: calculate_first_order_solution, calculate_second_order_solution
import MacroModelling: calculate_third_order_solution
import MacroModelling: solve_lyapunov_equation, solve_sylvester_equation
import MacroModelling: calculate_second_order_stochastic_steady_state
import MacroModelling: calculate_third_order_stochastic_steady_state
import MacroModelling: run_kalman_iterations
import MacroModelling: sparse_preallocated!

import SparseArrays: SparseMatrixCSC, AbstractSparseMatrix

# Import ChainRulesCore to ensure the rrules are available
import ChainRulesCore

#=
The @from_rrule macro in Mooncake wraps ChainRulesCore rrules to be usable by Mooncake's AD.

For functions with existing ChainRulesCore rrules defined in MacroModelling.jl, we use 
@from_rrule to make them available to Mooncake without duplicating the differentiation logic.

The syntax requires: @from_rrule ctx Tuple{typeof(func), ArgType1, ArgType2, ...} 
or: @from_rrule ctx Tuple{typeof(func), T1, T2, ...} where {T1, T2, ...}

Note: Mooncake will use these rules instead of trying to differentiate through the function
implementation, which can significantly reduce compile times and improve performance.
=#

# ================================================================================================
# Non-Stochastic Steady State (NSSS) calculation
# ================================================================================================

# Wrap the NSSS calculation rule - this is the foundation for all other derivatives
# The ChainRulesCore rrule computes the implicit derivative via the implicit function theorem
@from_rrule Mooncake.DefaultCtx Tuple{typeof(get_NSSS_and_parameters), ℳ, Vector{S}} where {S<:Real}

# ================================================================================================
# Jacobian, Hessian, and Third Order Derivatives
# ================================================================================================

# These derivatives are used in the perturbation solution algorithms
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_jacobian), Vector{T}, Vector{S}, ℳ} where {T<:Real, S<:Real}
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_hessian), Vector{T}, Vector{S}, ℳ} where {T<:Real, S<:Real}
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_third_order_derivatives), Vector{T}, Vector{S}, ℳ} where {T<:Real, S<:Real}

# ================================================================================================
# Perturbation Solutions
# ================================================================================================

# First order solution - wrap the ChainRulesCore rrule
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_first_order_solution), Matrix{R}} where {R<:AbstractFloat}

# Second order solution
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_second_order_solution), AbstractMatrix{S}, SparseMatrixCSC{S}, AbstractMatrix{S}} where {S<:Real}

# Third order solution  
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_third_order_solution), AbstractMatrix{S}, SparseMatrixCSC{S}, SparseMatrixCSC{S}, AbstractMatrix{S}} where {S<:Real}

# ================================================================================================
# Matrix Equation Solvers (Lyapunov and Sylvester)
# ================================================================================================

# These are used in covariance computations and solution algorithms
@from_rrule Mooncake.DefaultCtx Tuple{typeof(solve_lyapunov_equation), AbstractMatrix{Float64}, AbstractMatrix{Float64}}
@from_rrule Mooncake.DefaultCtx Tuple{typeof(solve_sylvester_equation), M, N, O} where {M<:AbstractMatrix, N<:AbstractMatrix, O<:AbstractMatrix}

# ================================================================================================
# Stochastic Steady State
# ================================================================================================

# Second order stochastic steady state calculation
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_second_order_stochastic_steady_state), Val{:newton}, Matrix{Float64}, AbstractSparseMatrix{Float64}, timings, Int}

# Third order stochastic steady state calculation
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_third_order_stochastic_steady_state), Val{:newton}, Matrix{Float64}, AbstractSparseMatrix{Float64}, AbstractSparseMatrix{Float64}, timings, Int}

# ================================================================================================
# Kalman Filter
# ================================================================================================

# Kalman filter iterations for likelihood computation
@from_rrule Mooncake.DefaultCtx Tuple{typeof(run_kalman_iterations), AbstractMatrix{Float64}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, Int, AbstractMatrix{Float64}}

# ================================================================================================
# Utility Functions
# ================================================================================================

# Sparse matrix preallocation (used in higher order solutions)  
@from_rrule Mooncake.DefaultCtx Tuple{typeof(sparse_preallocated!), Matrix{T}} where {T<:Real}

end # module
