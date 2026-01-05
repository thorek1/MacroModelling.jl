# Balanced Growth Path Handling

Many DSGE models feature non-stationary variables that grow along a balanced growth path (BGP). Common examples include output, consumption, capital, and wages in models with technological progress. `MacroModelling.jl` provides automatic handling of such models through the `deflator` and `trend_var` options.

## Overview

In a model with balanced growth, certain variables grow at constant rates over time. For example, with labor-augmenting technological progress at rate `γ`, output, consumption, and capital all grow at rate `γ` in the long run.

To solve such models using perturbation methods, we need to transform the non-stationary variables into stationary ones by "detrending" - dividing by the level of technology (or another appropriate trend variable).

`MacroModelling.jl` automates this process:
1. Specify which variables should be detrended using `deflator` in `@model`
2. Optionally specify trend variable growth factors using `trend_var` in `@parameters`
3. Write your model equations in terms of detrended variables
4. The package automatically handles the transformation

## Basic Usage

### Specifying Deflators

Use the `deflator` option in `@model` to specify which variables are non-stationary and what their deflator (trend variable) is:

```julia
@model RBC_growth deflator = Dict(:y => :A, :c => :A, :k => :A) begin
    # A is the trend variable (e.g., technology level)
    A[0] = γ * A[-1]
    
    # Write equations in terms of detrended variables (y/A, c/A, k/A)
    # The package transforms y[0] → y[0] * A[0] internally
    y[0] = k[-1]^α
    c[0] + k[0] = y[0] + (1-δ)*k[-1]
    1/c[0] = β * (1/c[1]) * (α * y[1]/k[0] + 1-δ)
end
```

The `deflator` option takes a `Dict{Symbol, Symbol}` where:
- Keys are the non-stationary variables to be detrended
- Values are the trend variables to use as deflators

### Specifying Trend Variable Growth Factors

Use the `trend_var` option in `@parameters` to document the growth factors of trend variables:

```julia
@parameters RBC_growth trend_var = Dict(:A => :γ) begin
    γ = 1.02    # 2% growth rate
    α = 0.33
    β = 0.99
    δ = 0.025
end
```

## How It Works

When you specify a deflator for a variable, `MacroModelling.jl` automatically transforms the equations. For each variable `v` with deflator `d`:

- `v[0]` is transformed to `(v[0] * d[0])`
- `v[-1]` is transformed to `(v[-1] * d[-1])`
- `v[1]` is transformed to `(v[1] * d[1])`
- `v[ss]` is transformed to `(v[ss] * d[ss])`

This means you write your model in terms of detrended variables, and the package automatically "re-trends" them to recover the original (level) equations.

## Checking Balanced Growth Configuration

You can inspect the balanced growth path configuration using these functions:

```julia
# Check if a model has balanced growth handling enabled
has_balanced_growth(model)

# Get detailed information about the balanced growth configuration
info = get_balanced_growth_path_info(model)
# Returns: (has_balanced_growth, trend_vars, deflators, detrended_vars)
```

## Example: RBC Model with Technological Progress

Here's a complete example of an RBC model with exogenous labor-augmenting technological progress:

```julia
using MacroModelling

# Define the model with deflators
@model RBC_BGP deflator = Dict(:c => :A, :k => :A) begin
    # Technology grows at rate γ
    A[0] = γ * A[-1]
    
    # Euler equation (in detrended terms)
    1/c[0] = β * (1/c[1]) * (α * A[1]^(1-α) * k[0]^(α-1) + 1-δ)
    
    # Resource constraint (in detrended terms)
    c[0] + k[0] = A[0]^(1-α) * k[-1]^α + (1-δ)*k[-1]
    
    # Productivity shock (stationary)
    z[0] = ρ * z[-1] + σ * eps[x]
end

# Define parameters with trend variable information
@parameters RBC_BGP trend_var = Dict(:A => :γ) begin
    γ = 1.005   # 0.5% quarterly growth
    α = 0.33
    β = 0.99
    δ = 0.025
    ρ = 0.9
    σ = 0.01
end

# The model can now be solved and analyzed
get_steady_state(RBC_BGP)
```

## Notes and Best Practices

1. **Trend Variable Equation**: Always include an equation defining the evolution of the trend variable (e.g., `A[0] = γ * A[-1]`).

2. **Consistent Detrending**: Make sure all variables that grow at the same rate use the same deflator.

3. **Steady State**: The detrended variables should have well-defined steady states. If steady state solving fails, consider providing initial guesses.

4. **Comparison with Other Packages**: This feature is similar to Dynare's `trend_var` and `deflator` options, though the syntax differs slightly.

## See Also

- [`@model`](@ref) - Main model definition macro
- [`@parameters`](@ref) - Parameter definition macro
- [`get_balanced_growth_path_info`](@ref) - Get balanced growth configuration
- [`has_balanced_growth`](@ref) - Check if balanced growth is enabled
