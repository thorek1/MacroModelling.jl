# Steady State

This guide explains how the steady state is handled in MacroModelling.jl, including delayed parameter declarations and how to override the internal algorithms with a custom steady state function.

## Overview

The steady state of a DSGE model is the equilibrium point where all variables remain constant over time (in the absence of shocks). Computing the steady state is a crucial first step before solving the model using perturbation methods.

MacroModelling.jl provides:
1. **Delayed parameter declaration**: Parameters can be defined implicitly through calibration equations
2. **Automatic steady state solver**: A sophisticated internal algorithm that solves for the steady state
3. **Custom steady state functions**: The ability to override the internal solver with your own function

## Delayed Parameter Declaration

In many DSGE models, some parameters are calibrated to match steady state values rather than being set directly. MacroModelling.jl supports this through calibration equations in the `@parameters` macro.

For example, instead of directly specifying the discount factor β, you might want to calibrate it to match a target steady state interest rate:

```julia
@model Example begin
    # ... model equations ...
    1 = β * R[0]  # Euler equation
end

@parameters Example begin
    R[ss] = 1.01  # Target steady state interest rate
    # β is implicitly defined to satisfy R[ss] = 1.01
end
```

The `[ss]` notation indicates a steady state target. The solver will find the parameter values that make these targets hold in equilibrium.

## Internal Steady State Algorithm

MacroModelling.jl uses a sophisticated algorithm to solve for the non-stochastic steady state (NSSS). The algorithm proceeds through several steps designed to maximize efficiency and robustness:

### Step 1: Eliminate Redundant Variables

The algorithm first identifies and eliminates redundant variables from the system. Variables that are simple transformations of other variables (e.g., `y = x + z`) are substituted out, reducing the dimensionality of the problem.

### Step 2: Partition into Independent Blocks

The reduced system is partitioned into independent blocks that can be solved separately. This block decomposition exploits the sparsity structure of the model equations, allowing smaller subproblems to be solved in sequence rather than tackling the full system at once.

### Step 3: Attempt Symbolic Solution

For each block, the algorithm attempts a full or partial symbolic solution using computer algebra (via SymPyPythonCall.jl). When possible, closed-form solutions are obtained, which:
- Provide exact solutions without numerical error
- Enable faster computation
- Allow for analytical derivatives

### Step 4: Create Auxiliary Variables for Domain-Constrained Terms

For terms with domain constraints (e.g., `log(x+y)`, `x^y`), auxiliary variables are created to handle these constraints explicitly. This transformation helps ensure that numerical solutions respect domain requirements.

### Step 5: Custom Nonlinear Equations Solver

For blocks that cannot be solved symbolically, a custom system of nonlinear equations solver is employed. This solver is a Levenberg-Marquardt type algorithm with line-search that includes:

- **Box constraints**: Variables can be constrained to lie within specified bounds
- **Domain transformation**: A hyperbolic sine transformation is applied to handle unbounded variables while maintaining numerical stability
- **Adaptive failure recovery**: Upon failure, the solver optimizes over solver parameters and starting points to find a fast solution

### Step 6: Select Optimal Solver Parameters

The algorithm selects solver parameters and starting points that maximise speed for the specific model structure. This adaptive approach ensures efficient computation across diverse model specifications.

## Custom Steady State Functions

For complex models where the internal solver may struggle, or when you have analytical solutions available, you can provide a custom steady state function. There are three ways to specify this:

### Method 1: Via the `@parameters` Macro

```julia
# Define your steady state function
function my_ss(parameters::Vector{T}, m) where T
    # parameters is ordered as: m.parameters (e.g., [:α, :β, :δ, :ρ, :std_z])
    α, β, δ, ρ, std_z = parameters
    
    # Compute steady state values
    k_ss = ((1/β - 1 + δ) / α)^(1/(α-1))
    q_ss = k_ss^α
    c_ss = q_ss - δ*k_ss
    z_ss = zero(T)
    
    # Return values in variable order: m.var (e.g., [:c, :k, :q, :z])
    return [c_ss, k_ss, q_ss, z_ss]
end

@parameters RBC steady_state_function = my_ss begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end
```

### Method 2: Via Function Arguments

All functions that accept a `parameters` argument also accept a `steady_state_function` argument:

```julia
# Pass the steady state function to specific function calls
get_irf(RBC, steady_state_function = my_ss)
get_steady_state(RBC, steady_state_function = my_ss)
simulate(RBC, steady_state_function = my_ss)
```

### Method 3: Via `set_steady_state!`

You can set a persistent custom steady state function that will be used for all subsequent computations:

```julia
# Set the custom function
set_steady_state!(RBC, my_ss)

# Now all functions use the custom solver
get_steady_state(RBC)
get_irf(RBC)

# Clear to revert to the internal solver
clear_steady_state!(RBC)
```

### Writing a Custom Steady State Function

Your custom function must have the following signature:

```julia
function my_steady_state(parameters::Vector{T}, m) where T
    # Input:
    # - parameters: Vector of parameter values in the order of m.parameters
    # - m: The model object (used to access model structure)
    
    # Your computation here...
    
    # Output:
    # - Vector of steady state values in the order of m.var
    return steady_state_values
end
```

**Important considerations:**
- Use the type parameter `T` to ensure compatibility with automatic differentiation
- Access `m.parameters` to see the parameter ordering
- Access `m.var` to see the required variable ordering for the output
- The function should work with both `Float64` and dual number types for gradient computation

### For Models with Calibration Equations

If your model has calibrated parameters (parameters determined by steady state conditions), you also need to provide a `calibrated_parameters_function`:

```julia
function my_calib_params(parameters::Vector{T}, m) where T
    # Compute calibrated parameter values from free parameters
    # Return in the order of m.calibrated_parameters
    return calibrated_values
end

@parameters Model steady_state_function = my_ss calibrated_parameters_function = my_calib_params begin
    # parameter definitions
end
```

## Example: RBC Model with Custom Steady State

Here's a complete example using the basic RBC model:

```julia
using MacroModelling

@model RBC begin
    1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α-1) + (1-δ))
    c[0] + k[0] = (1-δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

# Custom steady state function with analytical solution
function rbc_steady_state(params::Vector{T}, m) where T
    α, β, δ, ρ, std_z = params
    
    # Analytical steady state
    z_ss = zero(T)
    k_ss = ((1/β - 1 + δ) / α)^(1/(α-1))
    q_ss = k_ss^α
    c_ss = q_ss - δ * k_ss
    
    # Return in order: [:c, :k, :q, :z]
    return [c_ss, k_ss, q_ss, z_ss]
end

@parameters RBC steady_state_function = rbc_steady_state begin
    α = 0.5
    β = 0.95
    δ = 0.02
    ρ = 0.2
    std_z = 0.01
end

# Verify the steady state
ss = get_steady_state(RBC)
```

## When to Use Custom Steady State Functions

Consider using a custom steady state function when:

1. **You have an analytical solution**: Analytical solutions are more accurate and faster than numerical solutions
2. **The internal solver struggles**: Complex models may have multiple equilibria or convergence issues
3. **Performance is critical**: For estimation with many likelihood evaluations, custom functions can speed up computation
4. **Debugging**: To verify that your model equations are correct by comparing against known solutions

The internal solver is robust and works well for most models, so start with the automatic solver and only switch to a custom function if needed.
