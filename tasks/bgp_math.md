# Symbolic Stationarization for Balanced Growth

The level representation follows Canova and Sæterhagen Paulsen,
Norges Bank Working Paper 18/2021. For each variable,

```math
\widehat X_t^i = X_t^i / H^i(A_t), \qquad
G_t^i = H^i(A_t) / H^i(A_{t-1}).
```

The parser seeds each timed variable with a symbolic log-growth term and
propagates growth through the expression tree:

```math
\begin{aligned}
g(xy) &= g(x) + g(y),\\
g(x/y) &= g(x) - g(y),\\
g(x^p) &= p\,g(x).
\end{aligned}
```

Additive terms must have equal growth. These restrictions form a linear
system in variable and trend-driver growth rates. Automatically selected
drivers receive an independent normalization, and the solved coefficients
define each variable's trend function.

The dynamic equations are then rewritten before perturbation:

```math
X_t^i \mapsto \widehat X_t^i,\qquad
X_{t+1}^i \mapsto \widehat X_{t+1}^i G_{t+1}^i,\qquad
X_{t-1}^i \mapsto \widehat X_{t-1}^i/G_t^i.
```

Trend-driver laws are represented directly as gross-growth equations, which
avoids evaluating reciprocal growth factors at a zero solver probe. `SS`
reports `log(G)` in the `Growth_rate` column. `levels = false` returns the
stationary transformed response; `levels = true` accumulates the simulated
gross-growth path to reconstruct the original level.

This differs from the former additive implementation, which solved a
nonstationary level system and applied first-difference transformations only
to outputs and moments. Covariances now come directly from the stationary
perturbation system and therefore remain finite without relabeling variables
as `Delta_x`.

Pure additive random walks are intentionally rejected. They cannot be
represented by a positive multiplicative trend without changing the model's
economic meaning; the user must supply a multiplicative growth-factor law.
