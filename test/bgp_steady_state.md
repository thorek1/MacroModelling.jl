# Steady-State Handling with Balanced Growth Paths

This document describes how `MacroModelling.jl` handles a non-stochastic
steady state (NSSS) when the model has a balanced growth path (BGP). It
distinguishes the economic problem from the numerical problem, explains the
implemented solver path, and compares the approach with NBToolbox, Dynare,
IRIS, and RISE.

## 1. The standard steady-state problem

For a stationary model written as

```math
F(x_{t-k},\ldots,x_t,\ldots,x_{t+\ell},\theta,\varepsilon_t)=0,
```

the deterministic non-stochastic steady state sets shocks to their
deterministic values, usually zero, and sets every time-indexed endogenous
variable to the same constant:

```math
x_{t-k}=\cdots=x_t=\cdots=x_{t+\ell}=x^{*}.
```

The solver therefore finds

```math
F(x^{*},\ldots,x^{*},\theta,0)=0.
```

Calibration equations can add unknown parameters to this system. If
\(p_c\) denotes calibration parameters, the numerical unknown vector is
effectively

```math
u=(x^{*},p_c),
```

and the residual system is

```math
R(u;\theta)=0.
```

The standard NSSS problem assumes that the level vector \(x^{*}\) is finite
and that the equations determine the relevant levels locally. A nonlinear
root finder, block solver, or analytical steady-state block can then be
used. The Jacobian

```math
R_u(u;\theta)
```

should have sufficient rank for the unknowns being solved.

## 2. Why this problem is different for a BGP

For a trending model, a finite level fixed point generally does not exist.
Instead, levels move along a deterministic growth path. Write each variable
as

```math
x_{i,t}=H_{i,t}\widehat{x}_{i,t},
```

where \(H_{i,t}\) is the trend component and \(\widehat{x}_{i,t}\) is
stationary. A BGP is characterized by:

```math
\widehat{x}_{i,t}=\widehat{x}_i^{*},
\qquad
\frac{H_{i,t}}{H_{i,t-1}}=G_{i}^{*}.
```

The level \(x_{i,t}\) is not constant unless \(G_i^{*}=1\). The correct
deterministic problem is therefore

```math
\widetilde F(\widehat{x}^{*},G^{*},\theta,0)=0,
```

where the transformed equations include the growth factors needed for leads
and lags.

For example, if

```julia
x[0] = g[0] * x[-1]
```

then the level path satisfies

```math
x_t=g_t x_{t-1}.
```

The BGP does not solve for a finite constant \(x^{*}\). It solves for a
normalized level and a gross growth factor:

```math
\widehat{x}^{*}=1,
\qquad
G_x^{*}=g^{*}.
```

The absolute level of \(x_t\) is an initial-condition normalization. Its
future level is reconstructed as

```math
x_t=x_0\prod_{s=1}^{t}G_{x,s}.
```

This distinction matters because directly imposing \(x_t=x_{t-1}=x^{*}\)
would force \(G_x=1\) and would solve the wrong economic problem.

## 3. Multiplicative stationarization

The implementation identifies independent trend drivers
\(A_1,\ldots,A_m\), and derives coefficients \(b_{ij}(\theta)\) such that

```math
\gamma_i=\sum_{j=1}^{m}b_{ij}(\theta)\gamma_{A_j}.
```

Equivalently,

```math
H_{i,t}=\prod_{j=1}^{m}H_{A_j,t}^{b_{ij}(\theta)},
\qquad
G_{i,t}=\prod_{j=1}^{m}G_{A_j,t}^{b_{ij}(\theta)}.
```

The raw model is transformed before the steady-state equations, derivative
functions, perturbation solution, and moment calculations are generated.
The timing rules are:

```math
\begin{aligned}
x_{i,t} &\mapsto \widehat{x}_{i,t},\\
x_{i,t+k} &\mapsto
\widehat{x}_{i,t+k}
\prod_{s=1}^{k}G_{i,t+s},
&&k>0,\\
x_{i,t+k} &\mapsto
\frac{\widehat{x}_{i,t+k}}
{\prod_{s=k+1}^{0}G_{i,t+s}},
&&k<0.
\end{aligned}
```

For a driver equation

```julia
a[0] = f[0] * a[-1]
```

the generated stationary system contains

```julia
aᴳ[0] = f[0]
a[0] = 1
```

The first equation solves the gross growth factor; the second fixes the
arbitrary current trend level. A ratio law such as

```julia
a[0] / a[-1] = f[0]
```

generates the same type of growth equation.

Forward expectations receive future growth factors. For example,

```math
x_t=\beta E_t x_{t+1}
```

becomes

```math
\widehat{x}_t
=\beta E_t\left[\widehat{x}_{t+1}G_{x,t+1}\right].
```

Consequently, the BGP problem is stationary even when the original model
contains expectations of trending levels.

## 4. Growth restrictions and the BGP unknowns

The parser derives growth restrictions using:

```math
\begin{aligned}
\gamma(xy)&=\gamma(x)+\gamma(y),\\
\gamma(x/y)&=\gamma(x)-\gamma(y),\\
\gamma(x^p)&=p\gamma(x).
\end{aligned}
```

Terms in an additive expression must have equal growth. An equation
\(L_t=R_t\) contributes

```math
\gamma(L_t)-\gamma(R_t)=0.
```

After adding driver normalizations, the restrictions are assembled as

```math
M(\theta)\gamma=r.
```

For each independent driver \(A_j\), the implementation solves

```math
M(\theta)b^{(j)}=e_j,
```

and stores the coefficient vectors \(b^{(j)}\). The coefficients retain
parameter expressions rather than being frozen numerically at the initial
calibration. Numeric coefficient values are refreshed when estimation
parameters change without reconstructing the stationary model.

The internal BGP NSSS unknown vector contains:

```math
u_{\mathrm{BGP}}
=
\left(
\widehat{x}_{\mathrm{ss}},
G_{A_1,\mathrm{ss}},\ldots,G_{A_m,\mathrm{ss}},
p_c,
\text{domain auxiliaries}
\right).
```

The hidden growth symbols use the `ᴳ` suffix, for example `aᴳ`. They are
ordinary internal NSSS unknowns. They are not included in the public
variable axis, but they must remain in the internal solution because:

- growth equations determine the BGP;
- lead and lag transformations use them;
- perturbation states can depend on them;
- level IRFs accumulate them;
- higher-order moments may need them.

## 5. How the implemented NSSS solver handles a BGP

### 5.1 Build the stationary model first

At model setup, the stationarization pass:

1. preserves the raw equations in `equations.original`;
2. identifies multiplicative trend candidates;
3. derives the symbolic growth restriction matrix;
4. solves for \(b_{ij}(\theta)\);
5. creates growth-factor equations;
6. rewrites current, lead, and lag references;
7. normalizes trend-driver levels;
8. sends the generated equations through the ordinary model-processing
   pipeline.

The resulting processed model has the same downstream NSSS architecture as a
stationary model, but with additional hidden growth variables and growth
factors in its equations.

### 5.2 Construct the NSSS dependency structure

`write_steady_state_solver_function!` identifies unknowns from the processed
steady-state equations and calibration equations. In a BGP, this set includes
the hidden `ᴳ` variables.

The solver builds an incidence matrix describing which equations contain
which unknowns. A block-triangular ordering is then used to partition the
problem into solvable blocks. Numerical blocks are solved with the existing
block-solver machinery, bounds, guesses, and solver parameter sets.

This means the BGP does not use a wholly separate nonlinear solver. It uses
the existing NSSS solver after the equations have been made stationary.

### 5.3 Anchor free trend levels

The normalized stationary system can still contain redundant level
directions. In the original level model, the absolute level of an independent
stochastic trend is a free initial condition. It must not be mistaken for a
missing economic equation.

For BGP systems, the NSSS setup:

1. evaluates the NSSS Jacobian with growth unknowns included;
2. applies a rank-revealing column selection;
3. keeps growth columns as determined unknowns;
4. identifies redundant level columns;
5. removes their incidence rows from the numerical solve;
6. lets the existing indeterminate-variable path assign a default level,
   normally zero, unless a user supplied an explicit `x[ss]` anchor.

This creates a valid particular normalization of the free trend level while
preserving the BGP growth factors. For a linear model, changing this
normalization changes the origin of the level path, not the growth dynamics.

An explicit steady-state anchor has priority over the automatic default. It
can be used when a particular level normalization is needed for reporting or
for nonlinear equations whose domain depends on the chosen level.

### 5.4 Execute numerical blocks and check residuals

The numerical NSSS path:

1. prepares the extended parameter vector;
2. initializes a reusable solution buffer;
3. executes the ordered steady-state blocks;
4. writes levels, growth variables, calibration parameters, and auxiliaries
   into the internal solution vector;
5. checks finite values and equation residuals;
6. returns a failure when the residual exceeds the configured tolerance.

The solver wrapper uses cached nearby solutions and a continuation method. For
a new parameter vector, it can solve an interpolated problem between a
nearby cached calibration and the target calibration before attempting the
full target. Successful intermediate solutions are retained as warm starts.

The resulting `SS_and_pars` vector is internal and can contain growth
variables. Public APIs filter those variables and construct a separate
`Growth_rate` column.

### 5.5 Obtain the reference state used by dynamics

For stationary models, the reference steady state and non-stochastic steady
state are obtained directly from public steady-state values.

For BGP models, the public values omit hidden growth variables, so the
implementation takes:

```text
public normalized variables  <- public steady-state output
hidden growth variables      <- internal NSSS solution buffer
```

This produces the complete internal reference vector needed by state updates,
IRFs, and perturbation routines. The stationary deviation used by dynamics is
then computed in the normalized system, not by subtracting a fixed level from
an ever-growing raw variable.

## 6. Difference from the standard NSSS problem

| Feature | Standard stationary NSSS | BGP NSSS |
| --- | --- | --- |
| Economic object | Finite fixed point \(x^{*}\) | Normalized fixed point plus growth factors \((\widehat{x}^{*},G^{*})\) |
| Level timing | All leads/lags equal the same level | Leads/lags include future or past growth factors |
| Trend level | Determined by equations or anchors | Arbitrary for independent trends; normalized and possibly anchored |
| Growth variables | Usually absent | Hidden `ᴳ` unknowns are solved jointly |
| Jacobian | Usually square after block reduction | May contain free level directions and requires rank-based anchoring |
| Public output | Steady-state levels | Normalized levels plus `Growth_rate` |
| Covariance | Covariance of stationary levels | Covariance of normalized stationary variables |
| IRF interpretation | Deviations from the fixed point | Deviations from the BGP when `levels = false`; accumulated levels when `levels = true` |
| Failure meaning | No finite fixed point or numerical root | Inconsistent growth restrictions, unsupported trend form, rank failure, or no normalized root |

The crucial difference is not only an extra column in the output. The BGP
changes the equations whose residuals are solved and the variables with
respect to which derivatives are taken.

## 7. How other packages handle the problem

### NBToolbox

NBToolbox is the closest conceptual comparison. It follows the symbolic
stationarization approach from Canova and Sæterhagen Paulsen:

1. identify nonstationary variables, normally through an explicit
   `unitrootvars` declaration;
2. derive restrictions on growth rates;
3. construct growth functions;
4. rewrite the model into stationary form;
5. solve the stationary model and its BGP.

The main difference is the interface. NBToolbox asks the user to declare the
unit-root variables, while `MacroModelling.jl` attempts to infer supported
multiplicative trend drivers from the equations. NBToolbox consequently gives
more explicit control over what is treated as a trend; `MacroModelling.jl`
requires less model bookkeeping but can reject or misclassify structures
outside its recognized form.

### Dynare

Dynare detrends models internally after the user declares the trend
structure. Typical declarations specify:

```dynare
trend_var(growth_factor=gA) A;
var(deflator=A) Y;
```

Dynare also supports log trends and log deflators. The user supplies the
growth factor and the deflator mapping; Dynare performs the algebraic
detrending and then solves the resulting stationary model.

The difference is therefore where automatic inference stops:

- Dynare automates the rewrite after trend metadata is declared.
- `MacroModelling.jl` tries to infer the trend metadata from compatible
  equations before performing the rewrite.

Dynare gives the researcher stronger direct control over deflators and
alternative trend specifications. `MacroModelling.jl` is more concise for
models whose multiplicative laws expose the trend structure directly.

### IRIS

IRIS has a growth-aware steady-state workflow. The user provides growth
status, trend equations, or related model metadata, and `steady` can solve
for levels together with steady-state changes or growth rates. IRIS can work
with nonlinear nonstationary models without requiring the user to manually
write every detrended equation.

The division of labor differs:

- IRIS uses a numerical growth-augmented steady-state and model-solution
  workflow.
- `MacroModelling.jl` derives symbolic growth restrictions and constructs a
  stationary equation system before numerical steady-state solution.

IRIS is flexible when trend status or steady-state growth must be supplied
through a numerical workflow. The symbolic approach makes the inferred
growth identities and timing factors inspectable before the root solver runs.

### RISE

RISE commonly handles a nonstationary steady state through a user-supplied
steady-state function or file. For a nonstationary model, the returned
steady-state object contains level and growth-rate information for each
endogenous variable. RISE can complete unspecified values numerically and
supports regime-specific steady-state logic.

RISE therefore puts more responsibility in the steady-state callback:

- the researcher specifies the BGP levels and growth rates;
- RISE solves or completes the remaining numerical problem.

`MacroModelling.jl` instead derives the BGP representation from the model
equations and passes the resulting stationary system to its generic NSSS
solver. RISE is more flexible for bespoke regimes and manually derived BGPs;
`MacroModelling.jl` is more automatic for the supported algebraic class.

### Comparison summary

| Package | Trend information supplied by user | Main BGP mechanism | Main strength |
| --- | --- | --- | --- |
| `MacroModelling.jl` | Multiplicative equations and parameters | Infer restrictions, stationarize, solve normalized NSSS plus hidden growth factors | Automatic inference and one integrated solver path |
| NBToolbox | Explicit nonstationary/unit-root variables | Symbolic stationarization | Closest paper-based symbolic workflow with explicit control |
| Dynare | Trend variables, growth factors, deflators | Declared detrending then stationary solution | Explicit and mature trend/deflator interface |
| IRIS | Growth status, trend metadata, numerical starting values | Growth-aware numerical steady state | Flexible nonlinear and numerical workflow |
| RISE | Steady-state/BGP callback or file | User-supplied level/growth object plus numerical completion | Flexible custom and regime-dependent steady states |

## 8. Advantages of the current approach

### Correct object for a trending model

The solver finds a normalized BGP rather than pretending that an ever-growing
level is a finite fixed point. Growth factors are part of the solved
equilibrium object.

### One stationary numerical pipeline

After transformation, the ordinary NSSS block solver, Jacobian generation,
perturbation solver, Lyapunov routines, and higher-order moment machinery can
be reused. This avoids maintaining a separate numerical theory for every
downstream API.

### Expectations receive the correct timing factors

Lead variables inside expectations are multiplied by future growth factors.
This keeps forward-looking equations stationary without dropping the
intertemporal trend adjustment.

### Finite covariance in the correct coordinates

The covariance is calculated for normalized stationary variables. A unit-root
level variable therefore does not automatically imply an infinite covariance
in the object being reported.

### Suitable for repeated estimation draws

The structural BGP profile identifies trigger parameters. If an estimation
draw changes ordinary parameters but not the BGP mode or active driver set,
the generated stationary representation is retained. Parameter-dependent
growth coefficients are evaluated again without rerunning the parser.

### Visible structural failures

Additive unit roots, unsupported mixed trend structures, rank-deficient
growth restrictions, and invalid growth factors are surfaced rather than
silently treated as stationary.

## 9. Costs and limitations

### The transformation is not universal

The current implementation is designed for multiplicative stationarization.
It intentionally rejects a pure additive law such as

```julia
x[0] = x[-1] + u[0]
```

and mixed models containing both unsupported additive and multiplicative
trend structures.

### Structural recognition is pattern-based

A model must expose a trend carrier through a recognized ratio or
multiplicative law. A mathematically equivalent model written with an
unsupported rearrangement, implicit division, or unusual function may not be
recognized automatically.

### Growth restrictions must be identifiable

The restriction matrix must have sufficient rank, and the restrictions must
be numerically consistent at the current parameter values. A rank-deficient
system can arise from redundant trends, missing equations, or parameter
values that make previously independent restrictions collapse.

### The normalized nonlinear root can still fail

Successful stationarization does not prove that a BGP exists or is unique.
The generated normalized NSSS can fail because of:

- no economically valid root;
- multiple roots and a poor initial guess;
- domain violations in logs, powers, or ratios;
- non-positive or non-finite gross growth factors;
- solver bounds or tolerances;
- unstable or ill-conditioned local Jacobians.

### Near-boundary mode changes can be sensitive

The early detector classifies a constant multiplicative candidate using the
current rule \(|f|\geq 1\). Draws near the boundary can switch between the
stationary and active representations. A mode switch is handled explicitly,
but it requires rebuilding the processed representation and resetting solver
state.

### Normalized levels require interpretation

The public steady-state level of a trend carrier is a normalization, not
necessarily an observed level. The `Growth_rate` column must be used together
with the normalized level when interpreting a BGP.

### Current implementation does not have every possible cache optimization

The current estimation optimization avoids repeated stationarization when the
representation is unchanged and refreshes numeric growth metadata in place.
It does not yet maintain two fully compiled raw/BGP representation snapshots
or a separate dedicated BGP workspace. Switching representations therefore
reprocesses the retained raw or stationary equations.

## 10. Robustness assessment

The approach is robust when all of the following hold:

1. the model expresses trends through positive multiplicative gross factors;
2. the trend-driver equations are structurally recognizable;
3. growth restrictions are full rank and consistent;
4. parameter-dependent exponents can be evaluated at every draw;
5. the normalized NSSS has a finite root within the configured domain;
6. the resulting stationary perturbation solution is valid.

It is not robust to arbitrary nonstationarity by design. The correct failure
mode for an unsupported model is an explicit diagnostic or failed NSSS
solution, not a plausible-looking fixed point obtained by silently dropping
trend factors.

For estimation, a failed NSSS or invalid BGP draw is propagated as a failed
model evaluation according to the likelihood's configured failure value. A
previous draw's growth factors are not used as a silent substitute.

## 11. Implementation anchors

The main implementation points are:

- `src/parser/stationarization.jl` — trend detection, symbolic restrictions,
  coefficient solving, AST rewriting, and mode switching.
- `src/structures.jl` — BGP modes, detection metadata, and model structures.
- `src/steady_state/nsss_solver.jl` — block construction, BGP free-level
  anchoring, numerical NSSS execution, residual checks, and continuation.
- `src/parser/model_setup.jl` — construction of reference and non-stochastic
  steady states used by IRFs and state updates.
- `src/get_functions.jl` — public steady-state growth output, normalized IRFs,
  and level reconstruction.
- `src/moments.jl` — internal growth-variable reconstruction for moment
  calculations.
- `test/test_balanced_growth_path.jl` — focused BGP steady-state, expectation,
  IRF, moment, and mode-switching regressions.

Further package comparisons and source links are available in
[`bgp_implementation_comparison.md`](bgp_implementation_comparison.md).

## 12. Sources

- Canova and Sæterhagen Paulsen, *Symbolic Stationarization of Dynamic
  Equilibrium Models*, Norges Bank Working Paper 18/2021:
  [paper PDF](https://www.econstor.eu/bitstream/10419/264937/1/178583083X.pdf).
- NBToolbox:
  [github.com/Coksp1/NBTOOLBOX](https://github.com/Coksp1/NBTOOLBOX).
- IRIS structural modeling and growth-aware steady states:
  [`@Model`](https://iris-solutions-team.github.io/iris-reference/StructuralModeling/%40Model/index.html)
  and
  [`steady`](https://iris-solutions-team.github.io/iris-reference/StructuralModeling/%40Model/steady.html).
- Dynare model-file trend declarations:
  [Dynare manual](https://www.dynare.org/manual/the-model-file.html).
- RISE steady state and BGP interface:
  [RISE documentation](https://jmaih.github.io/rise-modern-docs/ModelShapes/DSGE/Steady%20state%20and%20balanced-growth%20path.html).
