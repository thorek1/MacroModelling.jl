# Ramsey Optimal Policy

This guide explains how to derive and analyze Ramsey optimal policy using MacroModelling.jl. Ramsey optimal policy is a framework where a benevolent social planner maximizes welfare subject to the constraints imposed by private agents' behavior (the model equations).

## Overview

In the Ramsey problem, a planner chooses policy instruments (like the interest rate or tax rates) to maximize discounted welfare:

```math
\max_{instruments} E_0 \sum_{t=0}^{\infty} \beta^t U(y_t)
```

subject to the model's equilibrium conditions:

```math
f_i(y_{t+1}, y_t, y_{t-1}, \varepsilon_t) = 0, \quad \forall i
```

MacroModelling.jl can automatically derive the first-order conditions (FOCs) for this problem using the `@ramsey` macro.

## Basic Usage

### Step 1: Define Your Model

First, define a standard DSGE model:

```julia
using MacroModelling

@model RBC begin
    1/c[0] = β * (1/c[1]) * (α * exp(z[1]) * k[0]^(α-1) + (1-δ))
    c[0] + k[0] = (1-δ) * k[-1] + q[0]  
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end
```

### Step 2: Create the Ramsey Model

Use the `@ramsey` macro to derive the optimal policy FOCs:

```julia
result = @ramsey RBC begin
    objective = log(c[0])     # Utility function
    instruments = [q]         # Policy instrument(s)
    discount = β              # Discount factor
end
```

This returns a NamedTuple with:
- `equations`: All equations in the Ramsey system (constraints + FOCs)
- `focs`: Just the first-order conditions
- `multipliers`: The Lagrange multiplier symbols
- `variables`: All variables including multipliers
- `objective`, `instruments`, `discount`: The configuration

### Step 3: View the Results

```julia
println("Original equations: ", length(RBC.original_equations))
println("Total Ramsey equations: ", length(result.equations))
println("Lagrange multipliers: ", result.multipliers)

println("\nFirst-Order Conditions:")
for (i, foc) in enumerate(result.focs)
    println("  FOC $i: ", foc)
end
```

## Example Output

For the RBC model above, the `@ramsey` macro generates:

```
Creating Ramsey model: RBC_ramsey
  Original equations: 4
  Ramsey equations (total): 8
  Lagrange multipliers: [:Lagr_mult_1, :Lagr_mult_2, :Lagr_mult_3, :Lagr_mult_4]
  Instruments: [:q]
```

## Multiple Instruments

You can specify multiple policy instruments:

```julia
result = @ramsey NK begin
    objective = log(C[0]) - N[0]^(1+φ)/(1+φ)
    instruments = [R, τ]    # Interest rate and tax rate
    discount = β
end
```

## Understanding the Output

### Original Equations (Constraints)
These are the model's equilibrium conditions that constrain the planner's choices.

### Lagrange Multipliers
Each constraint gets a multiplier (`Lagr_mult_1`, `Lagr_mult_2`, ...). These represent the shadow value of relaxing each constraint.

### First-Order Conditions
The FOCs determine optimal policy. For each variable `y`:

```math
\frac{\partial U}{\partial y_{j,t}} - \sum_i \left[ \lambda_{i,t} \frac{\partial f_i}{\partial y_{j,t}} + \beta \lambda_{i,t+1} \frac{\partial f_i}{\partial y_{j,t+1}} + \frac{\lambda_{i,t-1}}{\beta} \frac{\partial f_i}{\partial y_{j,t-1}} \right] = 0
```

## Building the Full Ramsey Model

To create a solvable model, you need to include the generated equations in a new `@model` block. The returned `result.equations` contains both the original model equations and the new FOCs.

```julia
# Get the Ramsey equations
result = @ramsey RBC begin
    objective = log(c[0])
    instruments = [q]
    discount = β
end

# The equations are available for building a new model
println("All equations: ", result.equations)
println("New variables (including multipliers): ", result.variables)
```

## Mathematical Details

The Ramsey Lagrangian is:

```math
\mathcal{L} = E_0 \sum_{t=0}^{\infty} \beta^t \left[ U(y_t) - \sum_i \lambda_{i,t} f_i(y_{t+1}, y_t, y_{t-1}, \varepsilon_t) \right]
```

Taking FOCs with respect to each variable `y_j`:

```math
\frac{\partial \mathcal{L}}{\partial y_{j,t}} = \beta^t \left[ \frac{\partial U}{\partial y_{j,t}} - \sum_i \lambda_{i,t} \frac{\partial f_i}{\partial y_{j,t}} - \beta \lambda_{i,t+1} \frac{\partial f_i}{\partial y_{j,t+1}} - \frac{\lambda_{i,t-1}}{\beta} \frac{\partial f_i}{\partial y_{j,t-1}} \right] = 0
```

## API Reference

```@docs
@ramsey
parse_ramsey_block
transform_equations_for_ramsey
```
