# The Mathematics of Symbolic Balanced-Growth Stationarization

This file explains the transformation used by `MacroModelling.jl`. The
implementation follows the multiplicative stationarization idea in Canova and
Sæterhagen Paulsen, Norges Bank Working Paper 18/2021, but infers trend
relations from the model equations instead of requiring a separate
user-declared trend block.

## 1. Levels, trends, and stationary variables

Let \(X_t^i\) be a variable written in levels. A balanced-growth path (BGP)
assumes that its deterministic trend can be written as a positive function
\(H_t^i\) of one or more trend carriers \(A_t\):

```math
X_t^i = H_t^i(A_t)\widehat X_t^i.
```

The transformed variable is

```math
\widehat X_t^i = \frac{X_t^i}{H_t^i(A_t)},
```

and its gross trend factor is

```math
G_t^i = \frac{H_t^i(A_t)}{H_{t-1}^i(A_{t-1})}.
```

On a BGP, \(\widehat X_t^i\) is stationary. The perturbation solver therefore
works with \(\widehat X_t^i\), not with the non-stationary level \(X_t^i\).
The public `Steady_state` value is the stationary normalized level, while
`Growth_rate` is

```math
\texttt{Growth\_rate}_i = \log(G^i).
```

This is logarithmic gross growth: a gross factor of \(1.02\) is reported as
\(\log(1.02)\), not as \(0.02\).

## 2. Deriving balanced-growth restrictions

The parser assigns each endogenous variable a symbolic growth rate
\(\gamma_i\). Parameters, constants, and shocks have zero deterministic
growth. Growth is propagated through expressions using

```math
\begin{aligned}
\gamma(xy) &= \gamma(x)+\gamma(y),\\
\gamma(x/y) &= \gamma(x)-\gamma(y),\\
\gamma(x^p) &= p\gamma(x).
\end{aligned}
```

For an additive expression, all non-zero terms must have the same growth.
For example,

```math
Y_t = C_t + I_t
```

implies

```math
\gamma_Y = \gamma_C = \gamma_I.
```

For a production function,

```math
Y_t = A_t K_t^\alpha L_t^{1-\alpha},
```

the restriction is

```math
\gamma_Y = \gamma_A+\alpha\gamma_K+(1-\alpha)\gamma_L.
```

Every equation produces linear restrictions of the form

```math
B\gamma = 0.
```

The implementation identifies a variable that carries a trend from a
structural multiplicative law such as

```julia
x[0] = x[-1] * g[0]
```

or a ratio such as `x[0] / x[-1]`. The selected trend carriers are normalized
to independent unit growth directions. Solving the augmented rank-aware
system gives coefficients \(b_{ij}\) such that

```math
\gamma_i = \sum_j b_{ij}\gamma_{A_j}.
```

Equivalently, the trend function is represented as

```math
H_t^i = \prod_j H_t^{A_j\,b_{ij}},
\qquad
G_t^i = \prod_j (G_t^{A_j})^{b_{ij}}.
```

Rank-deficient or inconsistent restrictions are errors. The implementation
also rejects pure additive unit-root laws such as

```julia
x[0] = x[-1] + g
```

because they do not define a positive multiplicative gross-growth factor
without changing the model.

## 3. Rewriting dynamic equations

Substitute

```math
X_t^i = \widehat X_t^i H_t^i
```

into every equation and divide out the common trend. The timing rules used by
the transformed equations are

```math
\begin{aligned}
X_t^i     &\mapsto \widehat X_t^i,\\
X_{t+1}^i &\mapsto \widehat X_{t+1}^iG_{t+1}^i,\\
X_{t-1}^i &\mapsto \frac{\widehat X_{t-1}^i}{G_t^i}.
\end{aligned}
```

For a trend carrier itself, its normalized current level is set to one:

```math
A_t\mapsto 1,\qquad
A_{t+1}\mapsto G_{t+1}^A,\qquad
A_{t-1}\mapsto \frac{1}{G_t^A}.
```

This is a normalization of the arbitrary level of the trend carrier, not a
claim that its original level is economically equal to one.

### Example: a stochastic growth factor

Consider

```julia
a[0] = (1 - ρ) * γ + ρ * a[-1] + σ * e[x]
x[0] = a[0] * x[-1]
```

The level \(x_t\) is the trend-carrying variable and \(a_t\) is its gross
growth factor. The transformation introduces \(G_t^x\) and produces

```math
G_t^x = a_t,\qquad \widehat x_t = 1.
```

The process for \(a_t\) remains a stationary equation. On the deterministic
path, \(a=\gamma\), so

```math
\log G^x = \log(\gamma).
```

If \(a_t\) is hit by a shock, \(G_t^x\) changes with that shock and the
original level response accumulates the changing factor.

## 4. Forward expectations

Expectations do not require a separate method. A lead receives the future
growth factor. For example, if \(x_t\) trends,

```math
x_t = b x_{t-1} + \beta E_t x_{t+1},
```

becomes, after dividing by the current trend,

```math
\widehat x_t =
b\frac{\widehat x_{t-1}}{G_t^x}
\beta E_t\left[\widehat x_{t+1}G_{t+1}^x\right].
```

Thus the transformed expectation is stationary while retaining the correct
future trend. Omitting \(G_{t+1}^x\) would incorrectly compare a future level
with a current normalized level.

## 5. Steady states and IRFs

The stationary equations, including the generated growth-factor equations,
are passed to the ordinary perturbation and moment solvers. The default NSSS
call instead uses a cached raw-equation BGP shadow with the same growth
identities. Growth-factor variables are kept internally because they are
needed by the dynamics, but are removed from public variable axes.

For `levels = false`, an IRF is the response of the normalized variable
\(\widehat X_t^i\). It is therefore a response relative to the stationary
BGP, rather than a response relative to a fixed level steady state.

For `levels = true`, the simulated growth factors are accumulated. If
\(\widehat X_t^i\) is the stationary response, the reconstructed level path is

```math
X_t^i =
\widehat X_t^i
\prod_{s=1}^{t}G_s^i
```

up to the chosen initial level normalization. With several trend carriers,
the accumulated factor is

```math
\prod_j\prod_{s=1}^{t}(G_s^{A_j})^{b_{ij}}.
```

This is why level IRFs correctly handle stochastic growth shocks; adding a
fixed deterministic drift would not.

## 6. Covariances and higher-order moments

The first-order perturbation solution of the stationary system has the form

```math
s_t = A s_{t-1} + C\varepsilon_t.
```

Its covariance solves the discrete Lyapunov equation

```math
\Sigma_s = A\Sigma_s A^\prime + CC^\prime.
```

Because the trend has been divided out, \(s_t\) is stationary and
\(\Sigma_s\) is finite whenever the transformed solution is stable. The
reported covariance is therefore the covariance of the stationary normalized
variables; no post-hoc first-difference transformation or `Delta_x`
renaming is needed.

Second- and third-order moments use the same stationary perturbation system.
The hidden growth-factor entries are reconstructed in the internal variable
ordering before the nonlinear mean and covariance formulas are evaluated.
They are filtered only when results are returned through the public API.

### Native perturbation path at higher order

The direct BGP method changes the NSSS input, not the perturbation derivative
engine. After the raw two-point solve, the complete internal BGP vector is
reconstructed, including hidden growth-factor variables. The ordinary
stationary Jacobian, Hessian, and third-order derivative functions then
evaluate the already stationarized equations on that vector.

This keeps the QME, Sylvester, stochastic-steady-state, moment, filtering,
and estimation machinery unchanged. It also gives exactly the same tensors
as an ordinary stationary model evaluated on the same full internal vector;
the direct-vs-stationary equality is tested for forward expectations and
three independent trend drivers. Public steady-state outputs continue to
omit hidden growth variables, so only the internal/public mapping and its
analytical pullback are BGP-specific.

## 7. Difference from the former additive implementation

The former two-point approach kept the original non-stationary level
equations and added additive unknowns such as

```math
x_t = x_{t-1} + x^G.
```

It then tried to repair IRFs and covariances after solving. That approach:

1. did not make the equations stationary before differentiation;
2. evaluated a lag as \(x^*-\Delta x\), which is only a first-order/additive
   approximation to the exact multiplicative lag \(x^*/G_x\);
3. could produce infinite level covariances for unit-root variables;
4. could not reconstruct paths with stochastic multiplicative growth;
5. handled forward-looking terms without the paper's explicit lead growth
   factor.

The current implementation uses exact gross factors in the raw two-point NSSS
cache and the symbolic stationary representation for derivatives,
perturbation, moments, and expectations. Original equations are retained for
rebuilding the cache, while the default NSSS path solves both normalized
levels and gross factors directly.

## 8. Implementation representation

The implementation keeps two related representations of a model:

1. `equations.original` stores the equations entered by the user.
2. `equations.stationarization` stores the metadata and generated equations
   used when an active multiplicative BGP is present.

The stationarization metadata contains:

```julia
stationarization_metadata(
    trend_drivers,
    trending_variables,
    growth_variables,
    growth_exponents,
    growth_exponent_expressions,
    original_equations,
    stationary_equations,
)
```

Here `growth_exponents[name]` is the current numeric vector
\((b_{i1},\ldots,b_{im})\), while
`growth_exponent_expressions[name]` retains the symbolic parameter-dependent
form of that vector. The growth variables are hidden symbols with the suffix
`ᴳ`, for example `zᴳ`. They are ordinary endogenous variables in the internal
stationary model, not a separate post-processing object.

The parser also stores a structural dispatch profile:

```julia
bgp_detection_metadata(
    candidate_drivers,
    active_drivers,
    candidate_kinds,
    candidate_factors,
    candidate_has_timed_variables,
    additive_candidates,
    trigger_parameters,
    trigger_indices,
    trigger_values,
    parameter_indices,
    mode,
)
```

The mode is one of:

```julia
BGP_STATIONARY_MODE
BGP_ACTIVE_MODE
BGP_UNSUPPORTED_MODE
```

The stationary mode leaves `equations.stationarization === nothing`, so
ordinary models continue through the existing non-BGP numerical path. The
active mode replaces the processed equation system with the stationary
equations and generated growth variables. The unsupported mode is used for
additive unit-root structures and raises an explicit error instead of
silently applying a multiplicative transformation.

## 9. Structural BGP detection

Detection is performed once from the raw equations and the initial complete
parameter vector. A variable is a candidate trend driver when the parser
finds either:

```julia
x[0] / x[-1] = g[0]
x[0] = f[0] * x[-1]
```

The first form is a ratio candidate. The second form is a multiplicative
candidate, where `f[0]` may itself contain stationary variables or parameters.
The parser separately detects exact additive unit-root structures such as

```julia
x[0] = x[-1] + u[0]
```

and rejects them because an additive increment does not define a positive
gross factor \(G_t^x\).

The structural profile also retains the parsed candidate factors and the
additive candidates. Trigger-changing parameter updates can therefore
reclassify the cached factors directly; they do not rescan the raw equations.
Only a mode or active-driver change rebuilds the stationarized representation.

For a numeric multiplicative factor \(f\), the current dispatch rule treats
the candidate as active when

```math
|f| \geq 1.
```

Ratio candidates and factors containing timed variables are always treated as
active because their growth cannot be decided from a single constant value.
An unresolved or non-finite factor is also treated as active. This is an early
representation-selection rule; it is not a substitute for the stability
checks performed by the solution algorithms.

The trigger set is smaller than the full parameter set. Direct parameter
symbols in a candidate factor are collected first. If the factor contains a
timed endogenous variable, the parser follows the equation defining that
variable and recursively collects the parameters and timed variables it
depends on. These parameter names are converted once to integer
`trigger_indices`. Later parameter updates compare only those entries with
the cached `trigger_values` before reclassifying candidates.

## 10. Symbolic growth-restriction construction

The parser represents the growth form of an expression as a sparse mapping
from variable names to exponents. For a variable \(x_i\), write this mapping
as \(\gamma(x_i)\). The operations implemented by the parser are:

```math
\begin{aligned}
\gamma(c) &= 0,\\
\gamma(x_i x_j) &= \gamma(x_i)+\gamma(x_j),\\
\gamma(x_i/x_j) &= \gamma(x_i)-\gamma(x_j),\\
\gamma(x_i^p) &= p\gamma(x_i).
\end{aligned}
```

For addition, all terms must have the same growth:

```math
\gamma(x+y):\qquad \gamma(x)-\gamma(y)=0.
```

An equation \(L_t=R_t\) therefore contributes the restrictions

```math
\gamma(L_t)-\gamma(R_t)=0,
```

along with the restrictions generated recursively by the expressions on both
sides. Functions such as `exp` and `log` require their argument to have zero
growth. A non-stationary exponent in a power is rejected.

Let the endogenous variables be ordered as
\(\mathcal V=(v_1,\ldots,v_n)\), and let the independent drivers be
\(\mathcal A=(A_1,\ldots,A_m)\). The restrictions become a matrix equation

```math
M(\theta)\gamma = r.
```

The implementation adds rows \(\gamma_i=0\) for variables not mentioned by
any restriction and rows \(\gamma_{A_j}=1\) to define one unit direction for
each driver. For each driver \(A_j\), it solves

```math
M(\theta)b^{(j)}=e_j,
```

where \(e_j\) selects the corresponding driver normalization. The complete
coefficient matrix is

```math
B(\theta)=
\begin{bmatrix}
b^{(1)} & \cdots & b^{(m)}
\end{bmatrix},
\qquad
\gamma_i=\sum_{j=1}^{m}B_{ij}(\theta)\gamma_{A_j}.
```

The rank and residual checks are evaluated numerically at the current
parameter values. The solve itself is performed symbolically using
parameter expressions, with pivot selection based on their current numeric
values. Consequently, a coefficient such as

```math
B_{ij}(\theta)=\frac{1-\alpha}{1+\beta}
```

is retained as an expression instead of being frozen at its initial
calibration.

## 11. Equation rewriting in the generated AST

For every timed reference, the generated equation contains the factor
implied by \(B(\theta)\). If \(k>0\), the factor for variable \(v_i\) is

```math
\prod_{s=1}^{k}\prod_{j=1}^{m}
\left(G_{t+s}^{A_j}\right)^{B_{ij}(\theta)}.
```

If \(k<0\), it is

```math
\left[
\prod_{s=k+1}^{0}\prod_{j=1}^{m}
\left(G_{t+s}^{A_j}\right)^{B_{ij}(\theta)}
\right]^{-1}.
```

Thus the generated AST applies:

```math
\begin{aligned}
v_{i,t} &\mapsto \widehat v_{i,t},\\
v_{i,t+k} &\mapsto
\widehat v_{i,t+k}
\prod_{s=1}^{k}\prod_j(G_{t+s}^{A_j})^{B_{ij}},
&&k>0,\\
v_{i,t+k} &\mapsto
\widehat v_{i,t+k}
\left(\prod_{s=k+1}^{0}\prod_j(G_{t+s}^{A_j})^{B_{ij}}\right)^{-1},
&&k<0.
\end{aligned}
```

For a trend driver, the factor is not reconstructed from \(B\). The driver
equation is converted directly into an equation for its hidden growth
variable:

```julia
x[0] = f[0] * x[-1]  # becomes xᴳ[0] = f[0]
```

The driver level is then normalized in the stationary system:

```julia
x[0] = 1
```

This removes the arbitrary level of the trend while retaining its growth
dynamics. The original equations are copied before this rewrite and remain
available in `equations.original`.

## 12. Estimation-time dispatch

Parameter updates enter through `write_parameters_input!`. After validation
and assignment of the new values, the update path is:

```text
update parameter_values
        |
        v
compare trigger_indices with cached trigger_values
        |
        +-- unchanged --> keep current representation
        |
        +-- changed --> reclassify candidate drivers
                              |
                              +-- same mode and drivers
                              |       refresh numeric BGP metadata only
                              |
                              +-- stationary -> active
                              |       rebuild stationary representation
                              |
                              +-- active -> stationary
                                      restore raw representation
```

The active representation is not rebuilt merely because a parameter changed.
When the mode and active driver set remain unchanged, the symbolic equations,
processed model structure, generated functions, and model dimensions remain
valid. The numeric growth exponent arrays are refreshed in place by evaluating
the retained `growth_exponent_expressions` against the new parameter vector.

A mode switch can change the number and ordering of internal variables, so it
reprocesses the appropriate raw or stationary equations, replaces the model
workspaces, and resets solver state. The current implementation rebuilds from
the retained raw equations when switching modes; it does not yet maintain two
fully compiled representation snapshots or a separate dedicated BGP
workspace.

Likelihood calls pass their candidate parameter vector into `solve!` before
steady-state, perturbation, or filtering calculations. The same forwarding is
used by the analytical likelihood pullbacks. This prevents an estimation draw
from being evaluated with stale model parameters.

The estimation observation map removes hidden `ᴳ` variables before matching
observables to `SS_and_pars`. Growth variables remain available to the
internal solver and state transition system but never become observable
columns by accident.

## 13. Steady-state growth output

The public steady-state output filters hidden growth variables from the
variable axis. For an active BGP, it inserts a separate `:Growth_rate` column
between `:Steady_state` and the parameter columns.

If the solved driver growth variable has gross factor
\(G^{A_j}>0\), its reported log growth is

```math
g_{A_j}=\log(G^{A_j}).
```

For a variable \(v_i\), the reported growth rate is the exponent-weighted
sum

```math
g_i=\sum_{j=1}^{m}B_{ij}(\theta)\log(G^{A_j}).
```

Therefore the reported value is a logarithmic per-period gross growth rate,
not the level of the hidden `v_iᴳ` symbol and not \(G_i-1\). Calibration
parameters and variables with zero growth receive a zero entry in the growth
column.

## 14. IRFs, covariances, and moments

For an active BGP, `levels = false` keeps the response in stationary
coordinates:

```math
\widehat v_{i,t}-\widehat v_{i,\mathrm{BGP}}.
```

It does not subtract the fixed, untrended level steady state from a trending
variable. Hidden growth variables are removed from the returned variable
axis, but remain in the internal state transition when needed.

For `levels = true`, the implementation first computes the stationary
response and then accumulates the simulated driver gross factors. With
\(C_{j,t}=\prod_{s=1}^{t}G_{s}^{A_j}\), the level reconstruction is

```math
v_{i,t}
=\widehat v_{i,t}
\prod_{j=1}^{m}C_{j,t}^{B_{ij}(\theta)}.
```

This uses the simulated growth factors rather than a fixed deterministic
drift, so shocks to a stochastic growth process affect both the current
normalized response and the accumulated level path.

After stationarization, the first-order state system has the usual form

```math
s_t=A s_{t-1}+C\varepsilon_t.
```

Its finite covariance is obtained from

```math
\Sigma=A\Sigma A^\prime+CC^\prime.
```

The covariance is finite because the perturbation system is written in
stationary coordinates. The internal moment routines reconstruct the hidden
growth-factor entries in solver order when required by higher-order formulas;
public result axes continue to omit those entries.

### Direct BGP perturbation

The direct method changes only the NSSS representation. After the raw
two-point solve, the complete internal BGP vector is available:

```math
u^*=(\widehat X^*,G^*,p_c).
```

The ordinary derivative functions are evaluated on this vector exactly as for
a stationary model. Hidden growth variables are included in the internal
vector even though they are omitted from public steady-state axes. Since the
stationary equations already contain the lead and lag growth factors, their
ordinary symbolic derivatives generate terms such as

```math
d(\widehat X_{t+1}G_{t+1})
=G^*d\widehat X_{t+1}+\widehat X^*dG_{t+1}.
```

The existing QME, Sylvester, stochastic-steady-state, moment, filtering, and
estimation routines are therefore reused. BGP-specific code only reconstructs
the hidden growth entries before derivative evaluation and maps internal
cotangents back to public steady-state outputs.

## 15. End-to-end mathematical summary

The complete transformation can be summarized as:

```math
\begin{aligned}
\text{raw model:}\quad&
F\left(X_{t-\ell},\ldots,X_{t+k},\theta,\varepsilon_t\right)=0,\\
\text{trend representation:}\quad&
X_{i,t}=H_{i,t}(\theta)\widehat X_{i,t},\\
\text{growth representation:}\quad&
H_{i,t}=\prod_j H_{A_j,t}^{B_{ij}(\theta)},\\
\text{driver dynamics:}\quad&
G_{A_j,t}=\frac{H_{A_j,t}}{H_{A_j,t-1}},\\
\text{stationary model:}\quad&
\widehat F\left(\widehat X_{t-\ell:t+k},
G_{A,t-\ell+1:t+k},\theta,\varepsilon_t\right)=0,\\
\text{reported growth:}\quad&
g_i=\sum_jB_{ij}(\theta)\log G_{A_j}.
\end{aligned}
```

The key implementation decision is that \(B_{ij}(\theta)\) is represented
symbolically in the generated stationary equations and evaluated numerically
for each estimation draw. The parser and model dimensions are therefore
reused across ordinary draws, while representation changes are reserved for
draws that change the structural BGP mode or active driver set.
