# Balanced-Growth-Path Implementation

For a level model, a balanced-growth path is represented as

```math
x_t = \bar{x} + g_x t.
```

Therefore,

```math
x[0] = \bar{x}, \qquad
x[-1] = \bar{x} - g_x, \qquad
x[1] = \bar{x} + g_x.
```

The steady-state equations are evaluated at two time origins. This produces
equations for both the level `\bar{x}` and the growth rate `g_x`.

For example, the equation

```math
x_t = x_{t-1} + g
```

implies

```math
\bar{x} = (\bar{x} - g_x) + g
```

and therefore `g_x = g`.

`SS` exposes these results as separate `Steady_state` and `Growth_rate`
columns.

## Expectations models

For `RBCexpect`, the parser cannot identify the trend from the collapsed
steady-state equations alone. After `@parameters`, dynamic models are also
checked for calibrated persistence parameters `rho` with `abs(rho) >= 1`.
Such models are treated as BGP-aware and receive a growth-rate column.

For the equation

```math
g_t = (1-\rho_g)\bar{g} + \rho_g g_{t-1},
```

an additive trend `g_t = \bar{g}_g + \gamma_g t` requires

```math
\gamma_g = \rho_g \gamma_g.
```

With `rho_g = 1.01`, the only additive-growth solution is
`\gamma_g = 0`. Strictly speaking, `1.01` is explosive rather than an exact
unit root.

## IRFs

The BGP is

```math
B_t = \bar{x} + g_x t.
```

With `levels = true`, the IRF includes the deterministic drift `g_x t`.
With `levels = false`, the result is the deviation from the BGP:

```math
\operatorname{IRF}_t = x_t - B_t.
```

For `RBCexpect`, the additive growth rate is zero, so its BGP coincides with
the fixed steady-state level.

## Finite BGP moments

The covariance of a trending level is not finite. For a trending variable,
the implementation instead uses its stationary first difference:

```math
\Delta x_t = x_t - x_{t-1}.
```

If the first-order solution is

```math
y_t = S s_{t-1} + E\varepsilon_t,
```

the BGP-difference transformation subtracts the unit-root state contribution
from the corresponding rows of `S`. The finite covariance is then computed as

```math
\Sigma_y^\Delta =
D\Sigma_sD^\prime + EE^\prime,
```

where `D` is the transformed solution matrix. Trending variables are labelled
`Delta_x` in moment outputs.
