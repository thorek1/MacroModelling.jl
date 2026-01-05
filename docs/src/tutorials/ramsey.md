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

MacroModelling.jl can automatically derive the first-order conditions (FOCs) for this problem by:
1. Introducing Lagrange multipliers for each constraint
2. Computing partial derivatives of the Lagrangian
3. Generating the optimality conditions

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

### Step 2: Derive Ramsey Equations

Use `get_ramsey_equations` to derive the optimal policy FOCs:

```julia
# Define objective function and policy instrument
objective = :(log(c[0]))  # Utility function
instruments = [:q]        # Policy instrument(s)

# Get the Ramsey system
equations, multipliers, variables = get_ramsey_equations(
    RBC,
    objective,
    instruments,
    discount = :β,
    verbose = true
)
```

This returns:
- `equations`: All equations in the Ramsey system (constraints + FOCs)
- `multipliers`: The Lagrange multiplier symbols (e.g., `[:λ₁, :λ₂, :λ₃, :λ₄]`)
- `variables`: All variables including multipliers

### Step 3: Analyze the Results

Use `ramsey_summary` for a structured view:

```julia
summary = ramsey_summary(RBC, :(log(c[0])), [:q], discount = :β)

println("Multipliers: ", summary.multipliers)
println("Number of constraints: ", summary.n_constraints)
println("Number of FOCs: ", summary.n_focs)
```

Use `print_ramsey_equations` for formatted output:

```julia
print_ramsey_equations(summary.equations, summary.multipliers, summary.n_constraints)
```

## Understanding the Output

### Original Equations (Constraints)
These are the model's equilibrium conditions that constrain the planner's choices.

### Lagrange Multipliers
Each constraint gets a multiplier (λ₁, λ₂, ...). These represent the shadow value of relaxing each constraint.

### First-Order Conditions
The FOCs determine optimal policy. For each variable `y`:

```math
\frac{\partial U}{\partial y_{j,t}} - \sum_i \left[ \lambda_{i,t} \frac{\partial f_i}{\partial y_{j,t}} + \beta \lambda_{i,t+1} \frac{\partial f_i}{\partial y_{j,t+1}} + \frac{\lambda_{i,t-1}}{\beta} \frac{\partial f_i}{\partial y_{j,t-1}} \right] = 0
```

## Example: New Keynesian Optimal Monetary Policy

Here's a more realistic example with a New Keynesian model:

```julia
using MacroModelling

# Load a New Keynesian model
include("models/RBC_CME.jl")

# Define welfare objective (negative of squared deviations)
# Standard central bank loss function: π² + λ_y * y²
objective = :(-0.5 * Pi[0]^2)

# Interest rate is the policy instrument
instruments = [:R]

# Get Ramsey equations
eqs, mults, vars = get_ramsey_equations(
    m,  # The NK model
    objective,
    instruments,
    discount = :beta,
    verbose = true
)

# View the system
print_ramsey_equations(eqs, mults, length(m.original_equations))
```

## Tips and Best Practices

1. **Choose appropriate instruments**: Policy instruments should be variables that the planner can directly control (e.g., interest rates, tax rates).

2. **Objective function**: The objective should be the per-period utility or loss function. For welfare analysis, use utility; for policy rules, use negative of loss.

3. **Discount factor**: Make sure to specify the correct discount factor symbol that matches your model parameters.

4. **Interpretation**: The multipliers in the FOCs represent intertemporal trade-offs in optimal policy.

## Mathematical Details

The Ramsey Lagrangian is:

```math
\mathcal{L} = E_0 \sum_{t=0}^{\infty} \beta^t \left[ U(y_t) - \sum_i \lambda_{i,t} f_i(y_{t+1}, y_t, y_{t-1}, \varepsilon_t) \right]
```

Taking FOCs with respect to each variable `y_j`:

```math
\frac{\partial \mathcal{L}}{\partial y_{j,t}} = \beta^t \left[ \frac{\partial U}{\partial y_{j,t}} - \sum_i \lambda_{i,t} \frac{\partial f_i}{\partial y_{j,t}} - \beta \lambda_{i,t+1} \frac{\partial f_i}{\partial y_{j,t+1}} - \frac{\lambda_{i,t-1}}{\beta} \frac{\partial f_i}{\partial y_{j,t-1}} \right] = 0
```

Simplifying by multiplying by `β^{-t}` gives the FOC shown above.

## API Reference

```@docs
get_ramsey_equations
print_ramsey_equations
ramsey_summary
```
