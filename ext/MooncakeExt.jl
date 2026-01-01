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

## Performance Notes

Mooncake.jl has an inherent "time to first gradient" compilation cost. The first call to 
`prepare_gradient` will be slow (typically 30-90 seconds depending on the complexity of the 
function being differentiated), but subsequent gradient evaluations using the prepared 
result are very fast (~0.01-0.1 seconds). This is a fundamental aspect of Mooncake's design.

**Best practices for optimal performance:**
1. Call `prepare_gradient` once and reuse the preparation object for multiple gradient evaluations
2. The prep time is a one-time cost per function signature
3. For estimation workflows, the prep cost is amortized over many gradient evaluations

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

# Prepare gradient computation once (slow, ~30-90 seconds)
backend = DifferentiationInterface.AutoMooncake(; config=nothing)
prep = DifferentiationInterface.prepare_gradient(
    p -> get_loglikelihood(RBC, data, p),
    backend,
    parameter_values
)

# Compute gradients many times (fast, ~0.01-0.1 seconds each)
for i in 1:1000
    grad = DifferentiationInterface.gradient(
        p -> get_loglikelihood(RBC, data, p),
        prep,
        backend,
        parameter_values
    )
end
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

We use concrete Float64 types where possible instead of generic type parameters to reduce
compilation overhead. The @from_rrule macro tells Mooncake to use ChainRulesCore's rrules
instead of differentiating through the function implementation.

Note: Some functions have keyword arguments with default values. These use `has_kwargs=true`
to signal that kwargs are present but their derivatives are treated as zero.
=#

# ================================================================================================
# Non-Stochastic Steady State (NSSS) calculation  
# ================================================================================================

# Wrap the NSSS calculation rule - this is the foundation for all other derivatives
# The ChainRulesCore rrule computes the implicit derivative via the implicit function theorem
# This function has keyword arguments (opts), so we specify has_kwargs=true
@from_rrule Mooncake.DefaultCtx Tuple{typeof(get_NSSS_and_parameters), ℳ, Vector{Float64}} true

# ================================================================================================
# Jacobian, Hessian, and Third Order Derivatives
# ================================================================================================

# These derivatives are used in the perturbation solution algorithms
# Using concrete Float64 types to reduce compilation overhead
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_jacobian), Vector{Float64}, Vector{Float64}, ℳ}
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_hessian), Vector{Float64}, Vector{Float64}, ℳ}
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_third_order_derivatives), Vector{Float64}, Vector{Float64}, ℳ}

# ================================================================================================
# Perturbation Solutions
# ================================================================================================

# First order solution - wrap the ChainRulesCore rrule  
# This function has keyword arguments (T, opts, initial_guess), so we specify has_kwargs=true
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_first_order_solution), Matrix{Float64}} true

# Second order solution - has keyword arguments
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_second_order_solution), Matrix{Float64}, SparseMatrixCSC{Float64, Int}, Matrix{Float64}} true

# Third order solution - has keyword arguments
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_third_order_solution), Matrix{Float64}, SparseMatrixCSC{Float64, Int}, SparseMatrixCSC{Float64, Int}, Matrix{Float64}} true

# ================================================================================================
# Matrix Equation Solvers (Lyapunov and Sylvester)
# ================================================================================================

# These are used in covariance computations and solution algorithms
@from_rrule Mooncake.DefaultCtx Tuple{typeof(solve_lyapunov_equation), Matrix{Float64}, Matrix{Float64}}
@from_rrule Mooncake.DefaultCtx Tuple{typeof(solve_sylvester_equation), Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}

# ================================================================================================
# Stochastic Steady State
# ================================================================================================

# Second order stochastic steady state calculation
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_second_order_stochastic_steady_state), Val{:newton}, Matrix{Float64}, SparseMatrixCSC{Float64, Int}, timings, Int}

# Third order stochastic steady state calculation
@from_rrule Mooncake.DefaultCtx Tuple{typeof(calculate_third_order_stochastic_steady_state), Val{:newton}, Matrix{Float64}, SparseMatrixCSC{Float64, Int}, SparseMatrixCSC{Float64, Int}, timings, Int}

# ================================================================================================
# Kalman Filter
# ================================================================================================

# Kalman filter iterations for likelihood computation
@from_rrule Mooncake.DefaultCtx Tuple{typeof(run_kalman_iterations), Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Int, Matrix{Float64}}

# ================================================================================================
# Utility Functions
# ================================================================================================

# Sparse matrix preallocation (used in higher order solutions)  
@from_rrule Mooncake.DefaultCtx Tuple{typeof(sparse_preallocated!), Matrix{Float64}} true

end # module
