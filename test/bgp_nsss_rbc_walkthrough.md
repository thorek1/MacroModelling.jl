# Exact BGP NSSS construction: a simple RBC example

This document traces the default BGP non-stochastic steady-state (NSSS)
path from a level model to the residual system that is passed to the existing
NSSS solver. The example is deliberately small, but includes a lag, a lead,
capital accumulation, production, and a forward Euler equation.

The important distinction is:

- the raw two-point construction below is used only to obtain the BGP NSSS;
- the symbolic stationarized model is retained for perturbation, moments, and
  expectations after the NSSS has been found.

There is no separate BGP nonlinear solver. The raw BGP residual system is
processed into a temporary ordinary model and then sent through the existing
NSSS block solver, numerical block solver, continuation, cache, and residual
checking machinery.

## 1. A trend-growth RBC model

Consider the following deterministic RBC model, written in levels:

```julia
@model TrendRBC begin
    a[0] = gA * a[-1]
    y[0] = a[0] * k[-1]^alpha
    c[0] + k[0] = y[0] + (1 - delta) * k[-1]
    1 / c[0] = beta / c[1] * (alpha * y[1] / k[0] + 1 - delta)
end

@parameters TrendRBC begin
    gA = 1.02
    alpha = 0.33
    beta = 0.99
    delta = 0.025
end
```

`a` is a technology level growing at the gross factor `gA`. The other
variables are capital, output, and consumption. Since technology grows and
the production function is Cobb--Douglas, capital, output, and consumption
also grow on the BGP. A conventional fixed point with
(a_t=a_{t-1}) would incorrectly force `gA = 1`.

For each endogenous variable (x), introduce a normalized level (x) at
the reference date and a gross BGP factor (G_x). The notation below uses

```math
a^*,k^*,y^*,c^*,qquad
G_a,G_k,G_y,G_c.
```

The stars are omitted in the code-generated expressions: the symbols `a`,
`k`, `y`, and `c` denote the normalized NSSS levels, while `aᴳ`, `kᴳ`,
`yᴳ`, and `cᴳ` denote the hidden gross-growth unknowns.

## 2. The two-point substitution

The direct BGP NSSS construction evaluates every original equation at two
time origins, (s=0) and (s=1). A timed endogenous reference is replaced
according to

```math
 x_{t+j}\quad\longmapsto\quad x^*G_x^{j+s}.
```

Thus, for a fixed shift (s),

```math
\begin{aligned}
x[0]  &\mapsto x^*G_x^s,\\
x[-1] &\mapsto x^*G_x^{s-1},\\
x[1]  &\mapsto x^*G_x^{s+1}.
\end{aligned}
```

In particular, at the first point (`s = 0`):

```math
x[0]\mapsto x^*,
\qquad
x[-1]\mapsto \frac{x^*}{G_x},
\qquad
x[1]\mapsto x^*G_x.
```

At the second point (`s = 1`):

```math
x[0]\mapsto x^*G_x,
\qquad
x[-1]\mapsto x^*,
\qquad
x[1]\mapsto x^*(G_x)^2.
```

Parameters are unchanged. Exogenous shock references are set to their
deterministic NSSS values. The current level of each independent trend
driver is anchored to one. Here that anchor is

```math
a^*=1.
```

The shifted copy of the driver equation is omitted. Once (a^*=1) is
imposed, the two copies of the technology law contain the same information:
the first copy identifies (G_a), and the second is its multiplication by
the already imposed one-step growth identity.

## 3. The transformed RBC equations

The ordinary NSSS solver works with residuals, so each equality below means
“left-hand side minus right-hand side equals zero”. The equations are shown
as equalities first because that makes the timing transformation transparent.

### 3.1 Shift (s=0)

The technology law becomes

```math
a = g_A\frac{a}{G_a}.
```

The production equation becomes

```math
y = a\left(\frac{k}{G_k}\right)^\alpha.
```

The resource constraint becomes

```math
c+k = y+(1-\delta)\frac{k}{G_k}.
```

The Euler equation contains the lead values (c[1]) and (y[1]), so it
becomes

```math
\frac{1}{c}
=
\frac{\beta}{cG_c}
\left(\frac{\alpha yG_y}{k}+1-\delta\right).
```

### 3.2 Shift (s=1)

The shifted technology law is omitted. The remaining equations become

```math
yG_y = aG_a k^\alpha,
```

```math
cG_c+kG_k = yG_y+(1-\delta)k,
```

and

```math
\frac{1}{cG_c}
=
\frac{\beta}{cG_c^2}
\left(\frac{\alpha yG_y^2}{kG_k}+1-\delta\right).
```

Finally, the independent trend level anchor is appended:

```math
a=1.
```

The complete direct NSSS residual system is therefore

```math
R(u;\theta)=0,
```

with unknown vector

```math
u=
\begin{bmatrix}
a & k & y & c & G_a & G_k & G_y & G_c
\end{bmatrix}^{\prime}
```

and parameter vector

```math
\theta=\begin{bmatrix}g_A & \alpha & \beta & \delta\end{bmatrix}^{\prime}.
```

There are eight transformed equations for eight NSSS unknowns:

```math
\begin{aligned}
R_1&=a-g_Aa/G_a,\\
R_2&=y-a(k/G_k)^\alpha,\\
R_3&=c+k-y-(1-\delta)k/G_k,\\
R_4&=1/c-\beta(cG_c)^{-1}(\alpha yG_y/k+1-\delta),\\
R_5&=yG_y-aG_ak^\alpha,\\
R_6&=cG_c+kG_k-yG_y-(1-\delta)k,\\
R_7&=(cG_c)^{-1}-\beta(cG_c^2)^{-1}
       (\alpha yG_y^2/(kG_k)+1-\delta),\\
R_8&=a-1.
\end{aligned}
```

This is the exact multiplicative two-point problem. No additive increment
such as (x^G) is introduced, and no post-solution differencing is needed.

## 4. What solution does this system represent?

For this model, the resource constraint implies that capital, output, and
consumption share a common BGP factor (G). Production then implies

```math
G=G_aG^\alpha,
\qquad\Longrightarrow\qquad
G=G_a^{1/(1-\alpha)}.
```

The technology law gives

```math
G_a=g_A.
```

With the parameter values above,

```math
G_a=1.02,
\qquad
G_k=G_y=G_c\approx1.0299972786.
```

A numerical solution of the full residual system is approximately

```text
a  = 1.0000000000
k  = 11.5340484448
y  =  2.2193105567
c  =  1.6034436129

aᴳ = 1.0200000000
kᴳ = 1.0299972786
yᴳ = 1.0299972786
cᴳ = 1.0299972786
```

The level `a = 1` is a normalization. Multiplying the initial technology
level by another positive constant changes the corresponding level path but
does not change the BGP growth factors or the stationary dynamics.

## 5. How this enters the existing NSSS solver

The implementation has two model representations for an active BGP:

1. the processed symbolic stationarized model, used by perturbation and
   moments;
2. a cached temporary raw-equation model, used only for direct BGP NSSS.

The call path for the default `Float64` NSSS request is:

```text
get_NSSS_and_parameters(active_model, parameters)
    |
    | active stationarization and no custom NSSS function
    v
direct_bgp_nsss_and_parameters(active_model, parameters)
    |
    v
ensure_direct_bgp_steady_state_model!
    |
    | deepcopy active model and restore original equations
    | build shift-0 and shift-1 raw residual equations
    | add driver-level anchors
    | process them as an ordinary temporary model
    v
get_NSSS_and_parameters(raw_bgp_model, parameters)
    |
    | raw_bgp_model has stationarization = nothing
    | therefore this call takes the ordinary NSSS path
    v
write_steady_state_solver_function!
    |
    v
solve_nsss_wrapper -> solve_nsss_steps -> block solvers
```

The recursive-looking call is intentional. The temporary raw BGP model has
already had its equations transformed into an ordinary stationary residual
system, so its second `get_NSSS_and_parameters` call does not dispatch back
to the BGP path.

### 5.1 Building the temporary model

The implementation in
[`src/perturbation/direct_bgp.jl`](../src/perturbation/direct_bgp.jl)
performs the following operations:

1. Copy the original equations and the active trend-driver list.
2. For `shift in (0, 1)`, replace every timed endogenous reference by
   `level * growth_factor^(timing + shift)`.
3. Skip only the shifted copy of each driver-growth law.
4. Append `driver[0] = 1` for each independent trend driver.
5. Pass the transformed equations through `process_model_equations`.
6. Set the temporary model's stationarization metadata to `nothing`.
7. Call `set_up_steady_state_solver!` on this temporary model.

For the RBC example, the temporary processed model therefore has the
ordinary NSSS variables

```julia
[:a, :c, :k, :y, :aᴳ, :cᴳ, :kᴳ, :yᴳ]
```

up to the package's deterministic variable ordering. Its generated equation
functions evaluate the eight residuals (R_1,ldots,R_8) above.

### 5.2 Building the NSSS solve structure

`write_steady_state_solver_function!` receives the temporary model and
constructs the same structures used for an ordinary stationary model:

- the set of unknowns in the steady-state equations;
- the equation-variable incidence matrix;
- the ordered analytical and numerical solution blocks;
- parameter-preparation functions;
- residual-check functions;
- reusable solution, parameter, guess, and error buffers;
- `nsss_sol_names`, the full internal solution ordering;
- `nsss_output_indices`, the public output selection.

For this example, the internal unknown vector contains all four levels and
all four gross factors. There are no hidden “trend paths” or infinite
sequences in the nonlinear solver: only the normalized reference levels and
one factor per timed endogenous variable are solved.

The incidence matrix is used to find which equations can be solved for which
unknowns. Analytical blocks are evaluated directly when available. Numerical
blocks call the configured nonlinear block solver with the current parameter
vector, bounds, guesses, and solver parameters. The existing continuation
wrapper can interpolate from a nearby cached solution when the requested
parameter vector is not already solved.

### 5.3 Executing and checking the solve

`solve_nsss_steps` then:

1. prepares the extended parameter vector;
2. clears or initializes the full solution buffer;
3. executes each analytical or numerical solution block in order;
4. writes each block's solution back into the full solution vector;
5. accumulates block errors and iteration counts;
6. evaluates the generated NSSS residual check;
7. returns the public `SS_and_pars` slice and the `(solution_error, iterations)`
   pair.

For a successful solution, the raw temporary model's full workspace buffer
contains the eight values shown above. The direct BGP wrapper copies those
values into the active model by matching `nsss_sol_names`. It then uses the
active model's `nsss_output_indices` to return only public normalized levels
and calibrated parameters. The hidden `ᴳ` values remain in the active
workspace because perturbation needs them.

## 6. What happens after NSSS?

The raw two-point representation is not used to construct a second
perturbation algorithm. Once the direct NSSS has supplied the levels and
growth factors, the active processed stationary model is used normally:

```text
raw direct BGP NSSS
    |
    v
full internal vector (levels, hidden growth factors, calibration parameters)
    |
    v
ordinary stationary Jacobian/Hessian/third-order derivative functions
    |
    v
ordinary perturbation, expectations, moments, filtering, and IRFs
```

The public steady-state object hides the growth-factor variables, but
`internal_steady_state_and_parameters` reconstructs them from the active NSSS
workspace before derivative evaluation. This is why the ordinary derivative
functions correctly differentiate terms such as

```math
d(\widehat x_{t+1}G_{t+1})
=G^*d\widehat x_{t+1}+\widehat x^*dG_{t+1}.
```

## 7. Code anchors

The main implementation points are:

- [`get_NSSS_and_parameters`](../src/MacroModelling.jl) — dispatches active
  BGP `Float64` requests to the direct path;
- [`direct_bgp.jl`](../src/perturbation/direct_bgp.jl) — constructs and caches
  the raw two-point model and maps its full solution back to the active model;
- [`nsss_solver.jl`](../src/steady_state/nsss_solver.jl) — builds the ordinary
  NSSS blocks, executes them, applies continuation, and checks residuals;
- [`moments.jl`](../src/moments.jl) — reconstructs the full internal BGP
  steady-state vector before derivative and moment calculations.

The central design choice is therefore small: BGP-specific logic changes the
NSSS residual representation and the public/internal solution mapping, while
the nonlinear NSSS solver and the downstream stationary perturbation
infrastructure remain the package's existing machinery.
