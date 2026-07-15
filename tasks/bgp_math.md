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
are passed to the ordinary NSSS and perturbation solvers. Growth-factor
variables are kept internally because they are needed by the dynamics, but
are removed from public variable axes.

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

## 7. Difference from the former additive implementation

The former approach kept the original non-stationary level equations and
added additive unknowns such as

```math
x_t = x_{t-1} + x^G.
```

It then tried to repair IRFs and covariances after solving. That approach:

1. did not make the equations stationary before differentiation;
2. represented growth as an additive level change rather than a positive
   gross factor;
3. could produce infinite level covariances for unit-root variables;
4. could not reconstruct paths with stochastic multiplicative growth;
5. handled forward-looking terms without the paper's explicit lead growth
   factor.

The current approach transforms the equations before steady-state solution,
derivatives, perturbation, moments, and expectations are computed. Original
equations are retained for inspection, while all numerical work uses the
stationary representation.
