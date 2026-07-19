# Automatic BGP Preprocessing and Steady-State Solution

This document explains what remains automatic in the BGP implementation,
which parts of the numerical machinery are reused, how trend candidates are
detected, and how this relates to the warning in Canova and
Sæterhagen Paulsen that unit roots cannot generally be identified
automatically.

## 1. Short answer

The original automatic steady-state solver is still used. Active BGP models
first receive structural growth metadata and a symbolic stationary
representation for perturbation and moments. The default NSSS call then uses
a cached raw-equation BGP shadow: it evaluates the original equations at two
consecutive BGP points, assigns a gross factor to each timed endogenous
variable, adds driver-level anchors, and sends that residual system to the
ordinary solver. The processed stationary model is then passed to the
ordinary derivative and perturbation infrastructure.

The pipeline is:

```text
raw equations
    |
    v
structural trend-candidate detection
    |
    v
symbolic growth restrictions
    |
    v
multiplicative stationarization
    |
    v
processed stationary model + raw BGP NSSS shadow
    |                              |
    |                              v
    |                    automatic gross two-point NSSS solve
    v
derivatives and perturbation solution
    |
    v
IRFs, moments, covariance, filtering, likelihood
```

The numerical steady-state and perturbation algorithms are reused after the
selected BGP residual system has been built, but BGP-specific changes are
required for hidden growth variables, free trend levels, internal indexing,
public output, IRFs, moments, and representation switching.

## 2. Standard steady-state problem

For a stationary model,

```math
F(x_{t-k},\ldots,x_t,\ldots,x_{t+\ell},\theta,\varepsilon_t)=0,
```

the deterministic steady state sets shocks to their deterministic values and
imposes

```math
x_{t-k}=\cdots=x_t=\cdots=x_{t+\ell}=x^*.
```

The numerical problem is therefore

```math
F(x^*,\ldots,x^*,\theta,0)=0.
```

If calibration parameters \(p_c\) are also unknown, the solver works with

```math
u=(x^*,p_c),
\qquad
R(u;\theta)=0.
```

The standard NSSS problem assumes that the level vector is finite and that
the equations determine the relevant levels locally. The numerical solver
can then use the model's analytical blocks, nonlinear block solver, bounds,
initial guesses, and continuation.

## 3. BGP steady-state problem

For a trending variable, write

```math
X_{i,t}=H_{i,t}\widehat X_{i,t},
```

where \(H_{i,t}\) is the trend and \(\widehat X_{i,t}\) is stationary. A BGP
solves for

```math
\widehat X_{i,t}=\widehat X_i^*,
\qquad
G_{i,t}=\frac{H_{i,t}}{H_{i,t-1}}=G_i^*.
```

The correct deterministic problem is therefore

```math
\widetilde F(\widehat X^*,G^*,\theta,0)=0.
```

For example,

```julia
x[0] = g[0] * x[-1]
```

implies

```math
x_t=g_t x_{t-1}.
```

The BGP does not solve for a finite constant level \(x^*\). It solves for a
normalized level and a gross growth factor:

```math
\widehat x^*=1,
\qquad
G_x^*=g^*.
```

The level path is reconstructed from an initial normalization:

```math
x_t=x_0\prod_{s=1}^{t}G_{x,s}.
```

The normalization `x[0] = 1` is not an economic claim that the original level
equals one. It removes the arbitrary absolute level of the trend.

## 4. Multiplicative stationarization

The implementation identifies independent trend drivers
\(A_1,\ldots,A_m\), then derives coefficients \(B_{ij}(\theta)\) such that

```math
\gamma_i=\sum_{j=1}^{m}B_{ij}(\theta)\gamma_{A_j}.
```

Equivalently,

```math
H_{i,t}=\prod_jH_{A_j,t}^{B_{ij}(\theta)},
\qquad
G_{i,t}=\prod_j\left(G_{A_j,t}\right)^{B_{ij}(\theta)}.
```

The generated equations include the appropriate trend factors:

```math
\begin{aligned}
X_{i,t} &\mapsto \widehat X_{i,t},\\
X_{i,t+k} &\mapsto
\widehat X_{i,t+k}
\prod_{s=1}^{k}G_{i,t+s},
&&k>0,\\
X_{i,t+k} &\mapsto
\frac{\widehat X_{i,t+k}}
{\prod_{s=k+1}^{0}G_{i,t+s}},
&&k<0.
\end{aligned}
```

For a driver equation,

```julia
a[0] = f[0] * a[-1]
```

the generated stationary system contains

```julia
aᴳ[0] = f[0]
a[0] = 1
```

The first equation solves the gross growth factor and the second fixes the
arbitrary current trend level. A ratio equation such as

```julia
a[0] / a[-1] = f[0]
```

generates the same type of growth equation.

The raw two-point shadow applies the same logic without using the stationary
equation AST as its NSSS input. For every original equation it creates a
shift-0 and shift-1 residual. A timed variable is replaced by

```math
x_{t+k}\mapsto x^*G_x^{k+s},\qquad s\in\{0,1\},
```

so lags divide by (G_x) and leads multiply by it. The shift-1 copy of an
independent driver law is redundant once the driver level is anchored at one
and is omitted. The remaining shifted equations solve the gross factors
implied by production, aggregation, market-clearing, and expectation
relations.

Expectations are transformed before derivative generation. For example,

```math
x_t=\beta E_t x_{t+1}
```

becomes

```math
\widehat x_t
=\beta E_t\left[\widehat x_{t+1}G_{x,t+1}\right].
```

Thus the stationary model retains the correct future trend adjustment.

## 5. What the automatic NSSS solver solves

The internal BGP NSSS unknown vector contains

```math
u_{\mathrm{BGP}}
=
\left(
\widehat x_{\mathrm{ss}},
G_{A_1,\mathrm{ss}},\ldots,G_{A_m,\mathrm{ss}},
p_c,
\text{domain auxiliaries}
\right).
```

The hidden growth symbols use the `ᴳ` suffix. They are ordinary internal
unknowns, not post-processing values. They remain in the internal system
because they:

- determine the BGP;
- enter transformed leads and lags;
- can affect the perturbation state;
- are needed for level IRF reconstruction;
- can be needed by higher-order moments.

The BGP system is sent through the existing NSSS setup:

1. unknowns are collected from processed steady-state and calibration
   equations;
2. an incidence matrix is formed;
3. block-triangular ordering partitions the equations;
4. numerical blocks are solved using the existing block solvers;
5. bounds, guesses, and solver parameters are applied;
6. residuals and finite-value checks are evaluated;
7. nearby cached solutions and continuation are used as warm starts.

Consequently, there is no separate user-facing BGP root solver. The
implementation supplies a stationary BGP system to the ordinary automatic
NSSS machinery.

## 6. Free trend levels and rank deficiency

An independent trend's absolute level is often a free initial condition. This
can make the steady-state Jacobian rank deficient even though the growth
factors and normalized equilibrium are well defined.

For active BGP models, the NSSS setup:

1. evaluates the NSSS Jacobian with growth unknowns included;
2. applies a rank-revealing column selection;
3. retains growth-factor columns;
4. identifies redundant level columns;
5. removes redundant level rows from the numerical incidence structure;
6. lets the existing indeterminate-variable path assign a default level,
   normally zero, or use an explicit `x[ss]` anchor.

This treats a free trend normalization as a normalization rather than as a
missing economic equation. In linear models, changing this normalization
changes the level origin but not the growth dynamics.

## 7. What remains unchanged and what is adapted

### Reused in principle

The following machinery still operates on the generated stationary model:

- steady-state block decomposition;
- numerical block solvers;
- parameter bounds and initial guesses;
- continuation and warm-start caches;
- residual checks;
- Jacobian, Hessian, and higher-order derivatives;
- perturbation solution algorithms;
- Lyapunov and Sylvester calculations;
- filtering and likelihood calculations.

### Adapted for BGP support

The implementation adds or changes:

- generated hidden `ᴳ` growth variables;
- trend-driver level normalizations;
- BGP-specific steady-state and state mappings;
- rank-based free-level handling;
- exclusion of hidden growth variables from public axes;
- the `Growth_rate` steady-state column;
- BGP-relative IRFs;
- level IRF reconstruction from accumulated growth factors;
- internal growth-variable reconstruction for moments;
- parameter-dependent BGP metadata refresh;
- representation switching when a parameter draw changes the BGP mode.

Therefore, the core numerical algorithms are reused, but the complete code
path is not literally untouched.

## 8. How trend detection is performed

Trend detection occurs before the NSSS solve. It does not infer trends by
examining the numerical steady-state result or by computing eigenvalues after
solving.

### Ratio candidates

The parser recognizes a direct ratio:

```julia
x[0] / x[-1] = g[0]
```

This identifies `x` as a candidate trend driver and `g[0]` as its gross
growth factor.

### Multiplicative candidates

The parser also recognizes:

```julia
x[0] = f[0] * x[-1]
```

It checks that the right-hand side contains exactly one lagged `x`, that
other factors do not contain another lagged `x`, and that they do not contain
the current `x`. The remaining product is treated as the candidate growth
factor \(f[0]\).

### Active-mode classification

For a factor depending only on parameters, the current dispatch rule is:

```math
|f|\geq 1
\quad\Longrightarrow\quad
\text{active BGP representation}.
```

For \(|f|<1\), the model stays on the stationary fast path.

Ratio candidates and factors containing timed endogenous variables are
treated as active because their classification cannot be determined from one
constant numeric value. Unresolved or non-finite factors are also treated as
active so that the model fails explicitly during the subsequent checks.

The classification rule is a representation-selection heuristic. It is not a
proof that a BGP exists, is unique, or is economically intended.

### Additive candidates

The parser recognizes certain exact additive unit-root forms, such as:

```julia
x[0] = x[-1] + u[0]
```

These are rejected as unsupported multiplicative BGPs rather than silently
being converted into gross growth factors. Mixed additive and multiplicative
trend structures are rejected consistently.

### Parameter-trigger caching

The detector records which parameters can affect candidate factors. If a
factor contains a timed endogenous variable, the parser recursively follows
the equations defining that variable and collects their parameter
dependencies.

These dependencies become cached integer `trigger_indices`. During estimation,
only those parameter entries are compared against cached values. If none
changed, the representation is retained. If they changed, candidates are
reclassified and the model is rebuilt only if the mode or active driver set
changes.

## 9. Is the steady-state solve still automatic?

Yes, for supported models. The user does not need to provide a separate BGP
steady-state function. A model written with recognizable multiplicative
trend equations is automatically:

1. detected;
2. assigned symbolic growth restrictions and normalized timing rules;
3. converted into a cached gross two-point raw-equation BGP NSSS residual system;
4. passed to the existing automatic NSSS solver;
5. solved jointly for normalized levels and growth factors.

The internal `get_NSSS_and_parameters` entry point uses the direct raw
two-point route automatically for active BGP models. There is no separate
NSSS-method switch: the generated stationary equations are used for
perturbation, while the cached raw-equation representation supplies the NSSS.

For ordinary stationary models, the existing fast path remains active and no
BGP stationarization is performed.

The automatic solver was therefore not removed. It was generalized by adding
an automatic preprocessing and dispatch layer, a raw BGP residual shadow, and
BGP-specific handling in the NSSS setup and public result mapping.

## 10. Relation to Canova and Sæterhagen Paulsen

The paper's statement that unit roots cannot generally be determined
automatically is not contradicted by this implementation.

For an arbitrary nonlinear model, the equations may not uniquely reveal:

- which variables are economically intended to carry trends;
- which unit roots are independent;
- which variables are cointegrated;
- which deflator or trend representation should be used;
- whether a persistent relation represents a structural trend or ordinary
  dynamics.

The current implementation does not solve that general identification
problem. It uses a restricted syntactic shortcut:

1. identify obvious multiplicative trend candidates;
2. treat those candidates as possible independent drivers;
3. derive growth restrictions conditional on that driver set;
4. solve the normalized BGP NSSS residual system using the direct raw
   two-point shadow.

The growth-restriction matrix does not discover the trend drivers by itself.
The drivers are selected first by structural pattern matching. The matrix then
determines how other variables' growth relates to those drivers.

The division of responsibility is therefore:

```text
trend-driver identification   partly heuristic and syntax-based
growth-restriction solution   symbolic and systematic
normalized NSSS solution      numerical and automatic
```

This is automatic under a restricted model class, not universal automatic
unit-root identification.

## 11. What the implementation cannot infer reliably

The detector may fail or require a different formulation when:

- a unit root is hidden inside a general multivariate transition matrix;
- the unit-root eigenvalue is not represented by an explicit multiplicative
  law;
- trends are additive rather than multiplicative;
- several economically different trend decompositions are possible;
- the trend relation is written in an algebraically equivalent but
  unsupported form;
- the growth restriction matrix is rank deficient;
- the normalized BGP does not exist or is not unique.

For example, a system

```math
x_t=A x_{t-1}+B y_{t-1}
```

may have a unit eigenvalue without containing a directly recognizable
equation of the form

```math
x_t=f_t x_{t-1}.
```

The current detector is not a general eigenvalue-based unit-root detector.

Successful structural detection also does not guarantee a successful
steady-state solve. Failure can still result from:

- no economically valid normalized root;
- multiple roots and a poor initial guess;
- domain violations;
- non-positive or non-finite growth factors;
- solver bounds or tolerances;
- an ill-conditioned local Jacobian;
- an unstable perturbation solution.

The correct behavior in these cases is an explicit diagnostic or failed
model evaluation, not a plausible-looking conventional fixed point.

## 12. Summary

The implementation preserves the automatic steady-state solver, but adds a
restricted automatic preprocessing layer:

```text
recognize supported multiplicative trend laws
    -> derive symbolic growth restrictions
    -> generate a stationary BGP system
    -> solve normalized levels and growth factors automatically
    -> reuse the stationary numerical machinery
```

The automatic part begins after a structural trend representation has been
recognized. It does not claim to solve the general economic problem of
identifying unit roots from arbitrary nonlinear equations, which is the
limitation emphasized by Canova and Sæterhagen Paulsen.
