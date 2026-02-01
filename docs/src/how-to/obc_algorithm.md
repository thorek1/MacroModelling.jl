# Algorithm for Occasionally Binding Constraints

This page explains the algorithm used by `MacroModelling.jl` to handle occasionally binding constraints (OBC) in DSGE models. For examples of how to use OBC features in practice, see the [Occasionally Binding Constraints guide](obc.md).

## Overview

Occasionally binding constraints are nonlinear features that cannot be captured by standard perturbation methods, which provide local linear (or higher-order polynomial) approximations around a steady state. `MacroModelling.jl` implements an algorithm that combines the computational efficiency of perturbation solutions with the ability to enforce occasionally binding constraints dynamically.

The key insight is to augment the model with **anticipated shocks** that are chosen optimally at each point in time to ensure constraint equations are satisfied over a finite forecast horizon.

## Mathematical Formulation

### Constraint Specification

Users specify occasionally binding constraints using `max` or `min` operators in model equations. For example:

```julia
R[0] = max(R̄, 1/β * Pi[0]^ϕᵖⁱ * (Y[0]/Y[ss])^ϕʸ * exp(nu[0]))
```

This ensures that the interest rate `R[0]` never falls below the effective lower bound `R̄`.

### Model Augmentation

When the parser encounters a `max(a, b)` or `min(a, b)` operator, the model is automatically augmented as follows:

1. **Auxiliary variables** are introduced:
   - For `max(a, b)`: create variables `χᵒᵇᶜ⁺ˡ` (left argument), `χᵒᵇᶜ⁺ʳ` (right argument), and `Χᵒᵇᶜ⁺` (the constraint itself)
   - For `min(a, b)`: similar variables with superscript `⁻` instead of `⁺`

2. **Anticipated shocks** are added:
   - A sequence of anticipated shocks `ϵᵒᵇᶜ⁽ⁱ⁾` for `i = 0, 1, ..., H` where `H` is the forecast horizon (default: 40 periods)
   - These shocks are added to the equation containing the constraint

3. **Constraint equation** is transformed:
   ```
   Original: x = max(a, b)
   Transformed: x = Χᵒᵇᶜ - ϵᵒᵇᶜ
   where: Χᵒᵇᶜ = max(a, b)
   ```

4. **Shock propagation equations** are added:
   ```
   ϵᵒᵇᶜ[0] = ϵᵒᵇᶜᴸ⁽⁻ᴴ⁾
   ϵᵒᵇᶜᴸ⁽⁻⁰⁾[0] = activeᵒᵇᶜshocks * ϵᵒᵇᶜ⁽ᴴ⁾[x]
   ϵᵒᵇᶜᴸ⁽⁻ⁱ⁾[0] = ϵᵒᵇᶜᴸ⁽⁻⁽ⁱ⁻¹⁾⁾[-1] + activeᵒᵇᶜshocks * ϵᵒᵇᶜ⁽ᴴ⁻ⁱ⁾[x]
   ```
   
   where:
   - `activeᵒᵇᶜshocks` is a parameter that enables (=1) or disables (=0) OBC enforcement
   - `[x]` denotes exogenous shocks (shocks not determined by model dynamics)
   - `ϵᵒᵇᶜ⁽ⁱ⁾` are the anticipated shocks at different horizons
   
   These equations implement a telescoping sum that allows the algorithm to inject anticipated shocks at different horizons to enforce the constraint.

## Optimization Algorithm

At each time period during simulation or impulse response computation, the algorithm must determine values for the anticipated shocks that enforce the occasionally binding constraints. This is formulated as a **constrained optimization problem**.

### Objective Function

The algorithm seeks to **minimize the magnitude** of the anticipated shocks:

```
minimize: Σᵢ (ϵᵒᵇᶜ⁽ⁱ⁾)²
```

This ensures that constraints are enforced with minimal intervention to the model dynamics.

### Constraint Violation Function

For each occasionally binding constraint in the model, a **violation function** is constructed that measures whether the constraint would be satisfied over the unconditional forecast horizon (default 40 periods) given a particular set of anticipated shocks.

The violation function computes:
1. The unconditional forecast starting from the current state
2. The value of the constraint inequality at each period in the forecast
3. Violations occur when:
   - For `max(a, b)`: the computed value falls below `max(a, b)`
   - For `min(a, b)`: the computed value exceeds `min(a, b)`

### Optimization Problem

The complete optimization problem is:

```
minimize:    Σᵢ (ϵᵒᵇᶜ⁽ⁱ⁾)²
subject to:  constraint_violation(ϵᵒᵇᶜ⁽⁰⁾, ..., ϵᵒᵇᶜ⁽ᴴ⁾) ≤ 0
```

where the constraint function ensures that the occasionally binding constraint is satisfied at each period of the forecast horizon.

### Numerical Solver

The optimization is solved using the **NLopt** library with the **SLSQP** (Sequential Least Squares Quadratic Programming) algorithm:

- **Algorithm**: `LD_SLSQP` (Local derivative-based SLSQP)
- **Tolerance**: `eps(Float32)` for both absolute x and function tolerances
- **Maximum evaluations**: 500
- **Initial guess**: Zero vector (no anticipated shocks)

The algorithm computes gradients using automatic differentiation to efficiently solve the constrained optimization problem.

## Workflow

The complete workflow for enforcing OBC during simulation/IRF computation is:

1. **Check constraint status**: Evaluate whether any constraints would be violated in the unconditional forecast from the current state with zero anticipated shocks

2. **If violated**: Set up and solve the optimization problem to find minimal anticipated shocks that enforce all constraints

3. **Update state**: Use the first-order (or higher-order) perturbation solution with the computed shocks (both structural and anticipated) to update the state

4. **Proceed to next period**: Repeat the process for each time period in the simulation

5. **If optimization fails**: If the algorithm cannot find shocks that satisfy the constraints (which can happen if the forecast horizon is too short or the constraint is incompatible with model dynamics), a warning is issued

## Key Parameters

- `max_obc_horizon`: The forecast horizon over which constraints are enforced (default: 40). This can be set in the model definition:
  ```julia
  @model MyModel max_obc_horizon = 60 begin
      # ... equations ...
  end
  ```
  
  Increasing this parameter can help when the algorithm struggles to find feasible shocks, but it increases computational cost as it adds more decision variables to the optimization problem.

- `ignore_obc`: When calling simulation or IRF functions, setting `ignore_obc = true` bypasses the OBC enforcement and uses only the perturbation solution. This is useful for comparison purposes.

## Computational Considerations

### Performance

- The algorithm solves a constrained optimization problem at each time period, which is more expensive than evaluating a perturbation solution directly
- Computational cost scales with:
  - Number of constraints: Each `max`/`min` operator adds anticipated shocks
  - Forecast horizon: Longer horizons require more decision variables
  - Model size: Larger models have more expensive forecast computations

### Numerical Stability

- The algorithm works best when constraints are "occasionally" binding rather than always binding
- The non-stochastic steady state (NSSS) must be computed at a point where constraints are **not** binding
- If a constraint would be binding at the NSSS, add bounds on variables in the `@parameters` block to find an alternative NSSS where constraints are slack

### Limitations

- Only finite-horizon enforcement: Constraints are guaranteed to hold over the forecast horizon, not indefinitely
- Local nature: The approach relies on a local perturbation approximation, which may be less accurate far from the steady state
- No theoretical moments: Theoretical moments cannot be computed for models with OBC; use simulations instead

## References

The implementation is based on the approach described in:

- [@citet cuba2019likelihood]: "Likelihood evaluation of models with occasionally binding constraints" - This paper describes methods for handling OBC in DSGE models and provides the theoretical foundation for the anticipated shocks approach.

For additional context on perturbation methods and pruning (used for higher-order approximations with OBC):

- [@citet andreasen2018pruning]: "The Pruned State-Space System for Non-Linear DSGE Models: Theory and Empirical Applications"

## Comparison with Alternative Methods

Other approaches to handling occasionally binding constraints include:

1. **Piecewise linear methods**: Solve the model in different "regimes" depending on which constraints bind
2. **Global solution methods**: Use projection or value function iteration to capture nonlinearities globally
3. **Extended path methods**: Solve a sequence of perfect foresight problems with terminal conditions

The anticipated shocks approach used by `MacroModelling.jl` offers a middle ground:
- Faster than global methods or extended path
- Captures constraint binding endogenously (unlike fixed regime switching)
- Works seamlessly with existing perturbation solution infrastructure
- Well-suited for models where constraints bind occasionally but not always
