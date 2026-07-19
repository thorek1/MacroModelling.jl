# Comparison with IRIS, NBToolbox, Dynare, and RISE

This document explains how the symbolic balanced-growth implementation in
`MacroModelling.jl` compares with four related frameworks. The short answer
is:

- **NBToolbox** is the closest conceptual relative because it implements the
  symbolic stationarization algorithm in Canova and Sæterhagen Paulsen
  (2021/2023).
- **Dynare** also transforms nonstationary equations internally, but the user
  must declare the trend variables, growth factors, and deflators.
- **IRIS** accepts nonlinear nonstationary models and solves a
  growth-augmented steady state, but its stationarization is numerical and
  requires the model's growth status and trend information.
- **RISE** represents the BGP through a user-supplied steady-state file whose
  output contains both levels and growth rates. It can complete a partial
  solution numerically, but it does not infer the full symbolic
  stationarization from arbitrary raw equations.

`MacroModelling.jl` removes the separate trend declaration for the supported
class of models: it infers multiplicative trend carriers and their growth
relations from the equations, creates the stationary system internally, and
then uses the ordinary steady-state and perturbation solvers.

## Summary table

| Framework | What the user supplies | What is automated | How the BGP is solved |
| --- | --- | --- | --- |
| `MacroModelling.jl` | Equations in a compatible multiplicative level form and parameter values | Trend-carrier detection, growth restrictions, growth equations, equation rewrite, direct raw BGP NSSS, perturbation | Solve normalized levels and gross growth factors in a cached two-point raw system, then reuse stationary perturbation |
| NBToolbox | Nonstationary equations plus an explicit `unitrootvars` block | Symbolic growth restrictions, identification of nonstationary variables, `G` functions, stationarization | Generate a stationary model, then solve its BGP and stationary system |
| IRIS | Nonstationary model, growth status, trend equations/metadata, and numerical starting values as needed | Numerical growth-augmented steady state and nonlinear model solution | Solve levels together with steady-state changes or growth rates using `steady(..., Growth=true)` |
| Dynare | `trend_var`/`log_trend_var`, growth factors, and `deflator`/`log_deflator` declarations | Equation detrending after the user supplies trend metadata | Detrend internally, then solve the resulting stationary model |
| RISE | A steady-state file or `@steady_state_model`; for a nonstationary model, levels and growth-rate columns | Numerical completion of missing parts and regime handling | Solve the user-specified level/growth BGP object, with numerical fallback for incomplete solutions |

The table distinguishes two meanings of “automatic”:

1. **Transformation automaticity:** whether the software rewrites equations
   after the user has supplied trend metadata.
2. **Trend-identification automaticity:** whether the software discovers the
   trend variables, trend functions, and growth factors from the raw equations.

Dynare is automatic in the first sense, but not the second. The current
`MacroModelling.jl` implementation aims to be automatic in both senses for
its supported multiplicative representation.

## What problem is being solved?

Suppose the original model is

```math
\mathbb E_t F(X_{t+1},X_t,X_{t-1},A_{t+1},A_t,A_{t-1},U_t;\theta)=0,
```

where \(X_t\) contains endogenous variables, \(U_t\) contains stationary
shocks, and \(A_t\) contains nonstationary trend carriers. A conventional
steady state does not exist for the level variables if they grow forever.

For each trending variable, write

```math
X_t^i = H_t^i(A_t)\widehat X_t^i,
\qquad
G_t^i = \frac{H_t^i(A_t)}{H_{t-1}^i(A_{t-1})}.
```

The BGP is a fixed point in the normalized variables \(\widehat X_t^i\)
together with constant gross growth factors \(G_t^i\). The original levels
are not solved as finite steady-state values; their arbitrary trend levels
are normalized away.

For a trend carrier itself, the normalization is

```math
A_t \mapsto 1,\qquad
A_{t+1}\mapsto G_{t+1}^A,\qquad
A_{t-1}\mapsto \frac{1}{G_t^A}.
```

For an ordinary trending variable, the timing substitutions are

```math
\begin{aligned}
X_t^i     &\mapsto \widehat X_t^i,\\
X_{t+1}^i &\mapsto \widehat X_{t+1}^iG_{t+1}^i,\\
X_{t-1}^i &\mapsto \frac{\widehat X_{t-1}^i}{G_t^i}.
\end{aligned}
```

After these substitutions, the model has a stationary representation:

```math
\mathbb E_t\widetilde F(\widehat X_{t+1},\widehat X_t,\widehat X_{t-1},
G_{t+1},G_t,U_t;\theta)=0,
```

plus equations describing the growth factors. Standard NSSS, perturbation,
Lyapunov, and higher-order moment algorithms can then be applied.

## How `MacroModelling.jl` solves the trending steady state

### 1. Infer growth restrictions

The parser assigns a symbolic growth rate \(\gamma_i\) to each endogenous
variable. It propagates growth through expressions:

```math
\gamma(xy)=\gamma(x)+\gamma(y),\qquad
\gamma(x/y)=\gamma(x)-\gamma(y),\qquad
\gamma(x^p)=p\gamma(x).
```

Additive terms must have equal growth. Thus

```math
Y=C+I
```

generates

```math
\gamma_Y=\gamma_C=\gamma_I,
```

while

```math
Y=A K^\alpha L^{1-\alpha}
```

generates

```math
\gamma_Y=\gamma_A+\alpha\gamma_K+(1-\alpha)\gamma_L.
```

Collecting all such restrictions gives a linear system for the unknown
growth rates. The implementation adds normalizations for independent trend
carriers and solves for coefficients \(b_{ij}\):

```math
\gamma_i=\sum_j b_{ij}\gamma_{A_j}.
```

This is the dual, growth-rate problem. It is much easier to solve than
guessing every \(H_i(A)\) directly.

### 2. Create growth-factor equations

For every identified trend carrier, the implementation creates a gross
growth variable \(G_t^i\). For example,

```julia
x[0] = x[-1] * g[0]
```

becomes the growth-factor equation

```math
G_t^x=g_t.
```

If a variable's growth depends on several carriers, the solved coefficients
give

```math
G_t^i=\prod_j (G_t^{A_j})^{b_{ij}}.
```

The trend-carrier equations and these generated equations remain inside the
stationary model. Growth factors can therefore be stochastic and can depend
on endogenous variables.

### 3. Normalize trend levels and rewrite equations

The current normalized level of each trend carrier is set to one. All
nonstationary level variables are replaced by normalized variables and the
appropriate current, lead, or lag growth factor.

The transformation is performed before the NSSS Jacobian, perturbation
derivatives, expectations system, and moment equations are generated. This
is the central difference from an output-only correction.

### 4. Solve the direct BGP NSSS

The generated NSSS unknown vector contains:

```math
(\widehat X_{\mathrm{ss}},G_{\mathrm{ss}},
\text{calibration parameters}).
```

The default direct route evaluates the original equations at two consecutive
points and solves a fixed point of the normalized levels and gross growth
factors. For example, with

```julia
a[0] = (1 - ρ) * γ + ρ * a[-1] + σ * e[x]
x[0] = a[0] * x[-1]
```

the deterministic BGP is

```math
\widehat x_{\mathrm{ss}}=1,\qquad
G^x_{\mathrm{ss}}=a_{\mathrm{ss}}=\gamma.
```

The public steady-state result reports the normalized value \(1\) for \(x\)
and reports \(\log(\gamma)\) in its `Growth_rate` column. It does not try to
return the infinite sequence \(x_0\gamma^t\) as a conventional steady state.
The generated stationary equations are retained for all downstream
derivative and perturbation calculations.

### 5. Solve perturbations and reconstruct levels

Perturbation is performed around the stationary fixed point. With
`levels = false`, an IRF is a response of \(\widehat X_t\), hence a response
relative to the BGP.

With `levels = true`, the simulated gross growth factors are accumulated:

```math
X_t^i =
\widehat X_t^i
\prod_{s=1}^{t}G_s^i.
```

For stochastic growth, this product is path-dependent. A fixed additive
drift is not equivalent because a growth shock changes all subsequent level
scales.

## Comparison with NBToolbox

NBToolbox is the closest implementation because it follows the same
four-stage logic:

1. find restrictions on BGP growth rates;
2. identify nonstationary variables;
3. compute their \(G_i\) functions;
4. rewrite the model in stationary form.

The important interface difference is that NBToolbox asks the user to
identify the nonstationary sources explicitly in a `unitrootvars` block. The
processes for those variables are then written in the nonstationary model
equations. The `stationarize` command uses symbolic operations to construct a
stationary model, and the generated model can be inspected or written to a
file.

`MacroModelling.jl` adopts the same mathematical idea but does not expose a
`unitrootvars` block. It detects candidates structurally from multiplicative
laws and ratios, constructs the growth restriction system, and keeps the
generated equations internally. This is more convenient for the common case,
but gives the user less explicit control over alternative trend assumptions.

Both approaches:

- support deterministic and stochastic trends;
- can handle exogenous and endogenous trend sources;
- use symbolic growth arithmetic rather than a numerical approximation to the
  growth restrictions;
- require a BGP to exist and the restrictions to have sufficient rank.

Neither method proves that every model has a unique BGP merely by running the
symbolic restriction calculation. A successful transformation supplies a
consistent stationary representation; existence and uniqueness still depend on
the model and the subsequent nonlinear solve.

## Comparison with IRIS

IRIS supports nonlinear nonstationary structural models and has an explicit
growth-aware steady-state workflow. Its `steady` command can be called with
`Growth=true`, in which case it computes both steady-state levels and
steady-state changes or growth rates. The IRIS documentation also exposes
growth status, log-variable status, deterministic trend equations, and
options for fixing levels or growth rates.

The main difference is where the hard work occurs:

- IRIS uses a numerical growth-augmented steady-state and model-solution
  workflow.
- `MacroModelling.jl` first derives symbolic growth restrictions and rewrites
  the equations into a stationary model.

The paper's comparison describes the IRIS stationarization step as numerical,
which means the transformed solution can inherit numerical approximation
error. The symbolic approach here keeps the growth identities and generated
stationary equations inspectable before numerical solution.

IRIS is therefore not “manual” in the sense of requiring the user to rewrite
every equation by hand, but it is not the same kind of automatic inference as
this implementation. The user still needs to mark or specify the model's
growth structure and choose whether growth is active when calculating the
steady state.

## Comparison with Dynare

Dynare can accept a model written in nonstationary form and detrend it
internally, but it requires explicit trend metadata. The current manual
provides:

```dynare
trend_var(growth_factor=gA) A;
var(deflator=A) Y;
```

For additive trends in logs, Dynare provides `log_trend_var` and
`log_deflator`. The deflator and growth factor expressions must be supplied
by the user and declared in the required order.

Thus Dynare automates the **rewrite after declaration**:

- it knows which endogenous variables use which deflators;
- it knows the growth factor for each trend variable;
- it constructs the detrended model for the solution routines.

It does not generally infer from

```julia
x[0] = x[-1] * g[0]
```

that \(x\) is the trend carrier, that \(G_t^x=g_t\), and that all equations
using \(x\) must receive the corresponding lead and lag factors. Those facts
are the information that `MacroModelling.jl` derives automatically.

The two approaches are otherwise close in spirit: both solve a stationary
representation rather than attempting a conventional finite steady state for
an ever-growing level. Dynare offers more explicit user control over
deflators; `MacroModelling.jl` offers more automatic inference for the
supported equation class.

## Comparison with RISE

RISE represents a nonstationary steady state/BGP through a steady-state file.
Its documented interface expects:

- a vector of levels for a stationary model; or
- an `endo_nbr`-by-2 object for a nonstationary model, containing the level and
  growth rate of each endogenous variable.

The user-written function returns those values and can also return parameters
implied by steady-state conditions. If the analytical or user-supplied part
is incomplete, RISE can numerically solve the remaining variables. RISE also
supports steady-state loops, bounds, and multiple regimes.

This is a different division of responsibility:

- RISE gives the user a flexible BGP callback and solves or completes the
  requested BGP values.
- `MacroModelling.jl` derives the BGP growth relations and stationary
  equations from the model syntax before calling its generic NSSS solver.

RISE is consequently flexible when the researcher already knows the BGP or
needs regime-specific custom logic. `MacroModelling.jl` is more automatic
when the model follows the recognized multiplicative structure, because the
user does not have to write a separate BGP function.

## Is the implementation fully automatic?

No—not in the sense of handling every possible nonstationary equation
without assumptions. It is automatic at the model-interface level for a
well-defined class:

### Automatic parts

`MacroModelling.jl` automatically:

1. scans the raw equations for structural multiplicative trend carriers and
   level ratios;
2. propagates growth through products, ratios, powers, sums, and supported
   functions;
3. assembles and solves the balanced-growth restriction system;
4. creates gross-growth equations;
5. rewrites current, lead, and lag references;
6. builds the direct raw BGP NSSS system and the stationary perturbation system;
7. reports public normalized steady states and logarithmic growth;
8. reconstructs level IRFs from simulated growth-factor paths.

### Required assumptions and user responsibilities

The user must still:

- write trends multiplicatively, with positive gross growth factors;
- provide parameter values that allow growth exponents to be resolved;
- use supported expression forms;
- specify any steady-state level anchors required by the model;
- ensure the restrictions are consistent and rank sufficient;
- inspect failures when the model has multiple possible trend
  classifications.

The implementation intentionally rejects pure additive random walks such as

```julia
x[0] = x[-1] + g
```

and does not silently reinterpret an explosive additive level process as a
positive growth factor. For example, the original additive/explosive
`RBCexpect` process must be rewritten so that the variable entering the level
law is a compatible multiplicative gross-growth factor.

The method also does not prove BGP existence or uniqueness for arbitrary
nonlinear models. It derives necessary growth consistency conditions under
the supported algebra, then lets the stationary nonlinear solver determine
whether the normalized equilibrium can actually be solved.

## Sources

- Canova and Sæterhagen Paulsen, *Symbolic Stationarization of Dynamic
  Equilibrium Models*, Norges Bank Working Paper 18/2021, revised 2023:
  [paper PDF](https://www.econstor.eu/bitstream/10419/264937/1/178583083X.pdf).
- NBToolbox source and examples:
  [github.com/Coksp1/NBTOOLBOX](https://github.com/Coksp1/NBTOOLBOX).
- IRIS structural-model overview:
  [IRIS reference](https://iris-solutions-team.github.io/iris-reference/StructuralModeling/%40Model/index.html).
- IRIS growth-aware steady state:
  [`steady`](https://iris-solutions-team.github.io/iris-reference/StructuralModeling/%40Model/steady.html)
  and
  [`changeGrowthStatus`](https://iris-solutions-team.github.io/iris-reference/StructuralModeling/%40Model/changeGrowthStatus.html).
- Dynare model-file declarations:
  [Dynare manual, variable and trend declarations](https://www.dynare.org/manual/the-model-file.html).
- Dynare nonstationary example:
  [Dynare examples](https://www.dynare.org/manual/examples.html).
- RISE steady state and BGP interface:
  [RISE documentation](https://jmaih.github.io/rise-modern-docs/ModelShapes/DSGE/Steady%20state%20and%20balanced-growth%20path.html).
