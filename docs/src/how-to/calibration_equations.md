# Calibration equations

Calibration equations let a parameter be determined implicitly by a steady-state target rather than set to a fixed value. Instead of choosing a number for a parameter, a condition on the model's steady state is specified and the solver finds the parameter value that satisfies it.

## When to use calibration equations

Calibration equations are useful when an empirical target is easier to observe than the structural parameter itself. Common cases include:

- Pinning steady-state labour supply to a data average (e.g. one-third of available time)
- Matching a capital-output or investment-output ratio
- Fixing the steady-state gross interest rate to imply a specific discount factor
- Ensuring a government-spending-to-output ratio matches national accounts data

## Syntax

A calibration equation links one free parameter to one steady-state condition. Two equivalent forms are supported in the `@parameters` block:

```julia
# Form 1: parameter on the left
param | steady_state_equation = target

# Form 2: parameter on the right
steady_state_equation = target | param
```

Both forms tell the solver: "find the value of `param` such that `steady_state_equation = target` holds in the non-stochastic steady state."

Calibration equations are mixed freely with ordinary parameter assignments:

```julia
@parameters model_name begin
    σ = 1                          # fixed value
    ψ | l[ss] = 1/3               # calibrated: ψ adjusts so l_ss = 1/3
    δ = 0.025                      # fixed value
    β | R[ss] = 1.0035             # calibrated: β adjusts so R_ss = 1.0035
end
```

## Examples from included models

### Targeting steady-state labour supply

In `RBC_baseline.jl` the disutility-of-labour parameter `ψ` is pinned so that steady-state hours equal one-third of the time endowment:

```julia
ψ | l[ss] = 1/3
```

The same pattern appears in `Ascari_Sbordone_2014.jl` (`d_n | N[ss] = 1/3`) and `Caldara_et_al_2012.jl` (`l[ss] = 1/3 | ν`).

### Targeting a steady-state ratio

In `RBC_baseline.jl` the steady-state government spending level is set to match a spending-to-output ratio:

```julia
g_y = 0.2038
ḡ | ḡ = g_y * y[ss]
```

A more involved ratio target appears in `JQ_2012_RBC.jl`, where the debt-to-output ratio pins `ξ̄`:

```julia
b[ss] / (y[ss] * (1 + r[ss])) = BY_ratio | ξ̄
```

### Targeting a steady-state price or rate

In `Smets_Wouters_2003.jl` two auxiliary parameters ensure that steady-state inflation equals its target:

```julia
calibr_pi_obj | 1 = pi_obj[ss]
calibr_pi | pi[ss] = pi_obj[ss]
```

In `Backus_Kehoe_Kydland_1992.jl` the discount factor `beta` is determined by a target level for the steady-state capital stock, using the alternative syntax:

```julia
K_ss = 11
K[ss] = K_ss | beta
```

### Multiple calibration equations in one block

The test model `RBC_CME_calibration_equations_and_parameter_definitions.jl` shows several calibration equations alongside ordinary definitions:

```julia
@parameters m begin
    alpha | k[ss] / (4 * y[ss]) = cap_share
    cap_share = 1.66

    beta | R[ss] = R_ss
    R_ss = 1.0035

    delta = .0226

    Pibar | Pi[ss] = Pi_ss
    Pi_ss = R_ss - Pi_real
    Pi_real = 1/1000

    phi_pi = 1.5
    rhoz = 9 / 10
    std_eps = .0068
    rho_z_delta = rhoz
    std_z_delta = .005
end
```

Here `alpha`, `beta`, and `Pibar` are all calibrated while `delta`, `phi_pi`, and the remaining parameters are set directly. Targets such as `cap_share` and `R_ss` are themselves defined as parameters, keeping the block self-documenting.

## Common pitfalls

**No solution exists.** If the target is inconsistent with the model structure the steady-state solver will fail. For example, requesting a capital-output ratio that implies a negative depreciation rate has no valid solution. Review the target value and the model equations when this happens.

**Solver convergence.** The nonlinear solver needs a reasonable starting region. Providing bounds and initial guesses for calibrated parameters can help:

```julia
@parameters model_name guess = Dict(:α => 0.3) begin
    α | k[ss] / (4 * y[ss]) = 1.5
    0 < α < 1
end
```

Bounds must be written as standalone comparison statements (e.g. `0 < α < 1` or `α > 0` on separate lines), not appended to the calibration equation. The `guess` keyword provides an initial value for the solver.

**One equation per parameter.** Each calibration equation pins exactly one parameter. Adding a second calibration equation for the same parameter, or using one equation for two parameters, will cause an error.

## Verifying calibrated values

After defining the model, call [`get_steady_state`](@ref) to inspect the solved steady state and confirm that the targets are met:

```julia
get_steady_state(model_name)
```

The output table shows all steady-state variable values and calibrated parameter values. Check that the targeted variables match the specified values.
