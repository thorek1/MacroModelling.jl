# Filters

Every likelihood in `MacroModelling.jl` answers the same question: given the model, the parameters, and the data, how plausible is what we observed? They differ in *how they deal with the states we cannot see*.

The model is a state-space system. The solution gives a transition,

```math
x_t = g(x_{t-1}, \varepsilon_t), \qquad \varepsilon_t \sim N(0, I),
```

where ``g`` is linear at first order and nonlinear at higher orders, and an observation equation that picks the observed variables out of the state and (optionally) adds measurement error,

```math
y_t = x_t[\text{observables}] + \eta_t, \qquad \eta_t \sim N(0, H).
```

The likelihood we want is ``p(y_{1:T}) = \prod_t p(y_t \mid y_{1:t-1})``. Each filter is a different way of tracking the distribution of ``x_t`` given the data so far, which is what turns that product into something computable.

Select a filter with the `filter` keyword:

```julia
get_loglikelihood(model, data, parameters; filter = :kalman)
```

Two inputs cut across all of them and are covered separately below: `measurement_error`, the noise on the observation, and `initial_covariance`, the prior on the state at the start of the sample. There is also a [filter-free likelihood](@ref "The filter-free likelihood") that does not integrate the shocks out at all, but treats them as parameters to be sampled.

## Choosing a filter

| filter | models | likelihood | differentiable | measurement error | smoothing | relative cost |
|---|---|---|---|---|---|---|
| `:kalman` | linear (`:first_order`) | exact | yes | optional (incl. correlated) | yes (Durbin–Koopman) | 1× |
| `:inversion` | linear and nonlinear | exact given the shocks | yes | not available | n/a (filtered = smoothed) | ~1–10× |
| `:bootstrap_particle` | linear and nonlinear | stochastic, unbiased | no | required (incl. correlated) | yes (genealogy) | ~10³× |
| `:auxiliary_particle` | linear and nonlinear | stochastic, unbiased | no | required (incl. correlated) | yes (genealogy) | ~2× bootstrap |
| `:tempered_particle` | linear and nonlinear | stochastic, unbiased | no | required (incl. correlated) | yes (genealogy) | ~5–10× bootstrap |

A short decision rule:

- **Linear model?** Use `:kalman`. It is exact, fast and differentiable, so gradient-based samplers (NUTS/HMC) work. There is no reason to use anything else.
- **Nonlinear model, at least as many shocks as observables, no measurement error?** Use `:inversion` (the default at higher order). It is exact and differentiable.
- **Nonlinear model with measurement error, or fewer shocks than observables?** Use a particle filter. Start with `:tempered_particle` if the observation is informative (small measurement error, many observables), otherwise `:bootstrap_particle`.
- Particle-filter likelihoods are noisy and non-differentiable: pair them with gradient-free samplers such as slice sampling (Pigeons.jl) or nested sampling.

By default the package picks `:kalman` for `:first_order` and `:inversion` for the nonlinear algorithms.

## What you get by default

Every knob discussed on this page has a default, and the defaults are not neutral — they are what determines the number you get from `get_loglikelihood(model, data, parameters)` with no keywords.

| setting | default | consequence |
|---|---|---|
| `filter` | `:kalman` at `:first_order`, `:inversion` at every higher order | nonlinear models are filtered *exactly given the shocks*, with no measurement error |
| `measurement_error` | `:auto` | **none** for Kalman and inversion; ``(0.1 s_i)^2`` per observable for the particle filters |
| `initial_covariance` | `:theoretical` | the ergodic covariance — *not* the inversion filter's implicit ``BB'``, which is why Kalman and inversion likelihoods differ by default |
| `smooth` | `true` for the Kalman filter, `false` otherwise | the particle filters **do** support smoothing but do not use it unless asked |
| `presample_periods` | `0` | the initial-condition transient is included in the likelihood |
| `warmup_iterations` | `0` | — |
| `on_failure_loglikelihood` | `-Inf`; `-1e6` for the particle filters | a stochastic failure rejects one proposal instead of killing a sampler's chain |
| `n_particles` | `10_000` | |
| `particle_resampling` | `:systematic`, threshold `0.5` | resample only when the effective sample size halves |
| `particle_initial_state_scaling` | `1.0` | the initial cloud has exactly the ergodic spread |
| tempering | ratio `2.0`, 1 MH step, ≤100 stages, scale `0.3` | only used by `:tempered_particle` |

Three consequences are worth internalising, because they surprise people:

1. **Kalman and inversion likelihoods are not comparable out of the box**, even on a first-order model with as many shocks as observables. They differ by the initial covariance alone (see below), and on Smets-Wouters that is worth hundreds of log points. Match `initial_covariance` before comparing.
2. **Likelihoods computed under different measurement error are not comparable at all** — ``H`` shifts the level of every period. Since `:auto` is data-driven, that includes two particle-filter runs on different samples.
3. **Switching perturbation order silently switches filter**, from Kalman to inversion, and with it the assumption about measurement error and the initial state. If you want a like-for-like comparison across orders, set `filter` explicitly.

## The Kalman filter

For a linear model with Gaussian shocks the filtering distribution stays Gaussian forever, so tracking it only requires tracking a mean and a covariance. The recursion alternates prediction and update:

```math
\begin{aligned}
v_t &= y_t - C u_t, & F_t &= C P_t C' + H,\\
u_{t+1} &= A (u_t + K_t v_t), & P_{t+1} &= A (P_t - K_t C P_t) A' + BB',
\end{aligned}
```

with the Kalman gain ``K_t = P_t C' F_t^{-1}``. Each period contributes

```math
\log p(y_t \mid y_{1:t-1}) = -\tfrac{1}{2}\left( d\log 2\pi + \log\det F_t + v_t' F_t^{-1} v_t \right).
```

This is exact — no approximation beyond the linearity of the model itself — and every step is a smooth function of the parameters, which is why the Kalman likelihood is differentiable and works with NUTS.

`initial_covariance` sets ``P_1``: `:theoretical` solves the Lyapunov equation ``P = APA' + BB'`` for the ergodic covariance, `:diagonal` starts diffuse (10 on the diagonal), or supply your own matrix. Missing observations are handled by shrinking the update to the observed rows in that period; a fully unobserved period becomes a pure prediction step.

The Kalman filter also supports **smoothing** (`smooth = true`), i.e. estimates of ``x_t`` given the *whole* sample rather than only the past. `get_model_estimates`, `get_estimated_shocks` and the estimate plots use the Durbin–Koopman smoother. The particle filters support smoothing too (see below); the inversion filter does not.

**References:** Kalman (1960); Durbin & Koopman (2012), *Time Series Analysis by State Space Methods*.

## The inversion filter

The inversion filter takes a different route: instead of integrating the shocks out, it asks *which shocks would have produced exactly this data?* Given the state ``x_{t-1}`` it solves

```math
y_t = g(x_{t-1}, \varepsilon_t)[\text{observables}]
```

for ``\varepsilon_t``. At first order this is a linear solve; at higher order it is a small Newton problem per period (`filter_algorithm = :LagrangeNewton`). The recovered shocks are then scored under their own standard normal prior, with a Jacobian term for the change of variables:

```math
\log p(y_{1:T}) = -\tfrac{1}{2}\sum_t \left( \varepsilon_t'\varepsilon_t + d\log 2\pi \right) - \sum_t \log\left|\det J_t\right|.
```

Because it inverts the observation equation exactly, the inversion filter:

- needs **at least as many shocks as observables** (otherwise the system is not invertible), and
- admits **no measurement error** — measurement error would make the mapping stochastic, and there would be nothing left to invert uniquely.

In exchange it is exact for nonlinear models, deterministic, and differentiable, which makes it the default at higher order. It provides no smoothed estimates.

**References:** Fair & Taylor (1983); Cuba-Borda, Guerrieri, Iacoviello & Zhong (2019).

### More shocks than observables

The inversion filter still runs when there are more shocks than observables: the per-period system ``y_t = CAx_{t-1} + Z\varepsilon_t`` (with ``Z = CB``) is then under-determined, and it returns the **minimum-norm** solution ``\varepsilon_t = Z^{+}v_t``, ``Z^{+} = Z'(ZZ')^{-1}``. Two things are worth being explicit about, because neither is visible from the output.

First, the good news: minimum norm is not an arbitrary tie-break. For Gaussian shocks ``Z^{+}v = E[\varepsilon \mid v]`` is the conditional mean, and ``\|Z^{+}v\|^2 = v'(ZZ')^{-1}v``, so the score remains a proper Gaussian density ``\log N(v_t; 0, ZZ')`` — exactly the same expression as in the square case.

Second, the **implicit assumption**. That expression is the Kalman contribution with the posterior state covariance *clamped to zero*. The filter propagates a single point ``\hat x_t`` and, at the next period, treats it as if it were known exactly. That is self-consistent only when the observation actually pins the state down, i.e. when

```math
P_{t|t} = P - PC'(CPC')^{-1}CP = 0 \quad\Longleftrightarrow\quad \mathrm{rank}(CB) = n_\varepsilon,
```

which requires **at least as many observables as shocks**. With more shocks than observables the rank condition fails, ``P_{t|t} > 0`` necessarily, and the assumption is simply false. The consequence is that

```math
F^{\text{kal}}_t = ZZ' + CA\,P_{t|t}\,A'C' \;\supsetneq\; ZZ' = F^{\text{inv}},
```

so the inversion filter **understates the innovation covariance**: it treats innovations as more surprising than they are, because it is pretending to know a state it cannot know. Its likelihood is a certainty-equivalent approximation, not ``p(y_{1:T})``.

How wrong it is depends on ``\|CA P_{t|t} A'C'\|`` relative to ``\|ZZ'\|`` — how much of the *unidentified* subspace propagates into the next period's observables — rather than on the shock/observable counts as such. That ratio is computable from a cheap first-order Kalman recursion even when the model is being filtered at third order, and is the right thing to look at before trusting an under-identified inversion likelihood. On the package's small RBC example with one observable and two shocks it is ``\approx 0.008``, which is why the two filters nearly agree there despite the state being genuinely unidentified; in a model whose unidentified directions propagate strongly the gap would be large.

There is a second, separate approximation: the minimum-norm choice is made **greedily**, period by period, minimising ``\|\varepsilon_t\|`` given ``\hat x_{t-1}`` without regard for the fact that the null-space component moves ``x_t`` and hence the cost of matching later observations. The Kalman disturbance smoother solves the same minimum-norm problem *globally* over the whole path. The two coincide only when there is no null space to redistribute over, i.e. ``n_y = n_\varepsilon``.

If the ratio above is large, the options are to rebalance the model so that ``n_y = n_\varepsilon`` (what most applied work does — Smets-Wouters has seven of each), or to use a particle filter, which represents the whole posterior instead of a point and handles the under-identified case natively.


## Particle filters

When the model is nonlinear *and* there is measurement error, the filtering distribution is no longer Gaussian and no longer invertible. Particle filters represent it by a cloud of ``N`` weighted draws ("particles") and update the cloud each period. They are the general-purpose fallback: they work for any transition, any number of shocks, and any measurement error.

All variants share the same skeleton:

1. **Predict** — push every particle through the model's transition with a freshly drawn shock.
2. **Weight** — score each particle by ``p(y_t \mid x_t)``. The weighted average of those scores estimates the period's likelihood contribution:
   ```math
   \widehat{p}(y_t \mid y_{1:t-1}) = \sum_p W_{t-1}^p \, p(y_t \mid x_t^p).
   ```
3. **Resample** — when the weights get too uneven, replace the weighted cloud by an equally weighted one, so particles are spent where the probability mass is.

The crucial property is that ``\widehat{p}(y_{1:T})`` is an **unbiased** estimator of the true likelihood, for any ``N``. That is what makes particle filters usable inside a sampler (pseudo-marginal MCMC targets the exact posterior despite the noise). Note the consequence for the *log* likelihood: by Jensen's inequality ``E[\log \widehat{p}] < \log p``, with a downward bias of roughly ``\mathrm{Var}(\log\widehat p)/2``. So a particle log-likelihood is systematically a little *below* the Kalman value on a linear model, and the gap shrinks as ``N`` grows — this is the expected behaviour, not a bug.

Because the estimate is random, a repeated evaluation at the same parameters gives a different number unless you fix the stream: pass a seeded generator via `particle_rng` (e.g. `particle_rng = Random.Xoshiro(1)`). Inside a sampler, reuse the same seed across parameter draws only if you deliberately want common random numbers; otherwise let the sampler see fresh noise, which is what pseudo-marginal correctness assumes. Cost scales roughly linearly in `n_particles` and in the sample length, and the number of particles needed grows quickly with the number of observables.

### Why measurement error is required

Without measurement error the observation equation is a deterministic function of the state. A particle would have to reproduce ``y_t`` *exactly* to get non-zero weight, which happens with probability zero — every weight collapses to zero and the filter dies. Measurement error smears the observation density and gives particles something to score against.

`measurement_error = :auto` (the default) therefore resolves, for the particle filters, to a variance of ``(0.1 s_i)^2`` per observable, ``s_i`` being that observable's sample standard deviation — and to *no* measurement error for the Kalman and inversion filters. For serious work set it explicitly or estimate it: the level of the likelihood depends on it, so likelihoods computed under different measurement errors are not comparable. See [Measurement error and the initial covariance](@ref) for what ``H`` is and the other jobs it does.

### Bootstrap (`:bootstrap_particle`)

The plain sequential-importance-resampling filter: propose from the model's own transition (the "prior"), weight by the observation density. Simple, robust, and the cheapest per particle.

Its weakness is that it proposes blindly — it draws shocks without looking at ``y_t``, so when the observation is very informative most particles land in implausible places and are discarded. This gets worse as the number of observables grows (the weights concentrate exponentially in the observation dimension), which is why a 7-observable model needs far more particles than a 2-observable one at the same measurement error.

**References:** Gordon, Salmond & Smith (1993); Fernández-Villaverde & Rubio-Ramírez (2007) for the DSGE application.

### Auxiliary (`:auxiliary_particle`)

Peeks at the observation *before* deciding which particles to propagate. Each ancestor gets a cheap preview of how plausible its children will look (the measurement density at its zero-shock transition mean, inflated by the shock-induced predictive variance), and ancestors are resampled in proportion to weight × preview. Only then are shocks drawn.

Selecting on a preview biases the cloud, so the second stage divides the preview back out — the child's weight is the true density divided by the preview used to pick its parent. The preview cancels exactly, so the estimator stays unbiased regardless of how good the preview is; a poor preview costs efficiency, never correctness.

Helps most when the signal is informative *and* the one-step-ahead state is well predicted by its mean. It costs roughly one extra transition evaluation per particle per period.

**Reference:** Pitt & Shephard (1999).

### Tempered (`:tempered_particle`)

Instead of confronting the particles with the full observation in one step, the tempered filter introduces the information gradually. Within each period it walks a bridging sequence ``0 = \phi_0 < \phi_1 < \dots < \phi_N = 1``, at each stage using an inflated measurement covariance ``H/\phi``: early stages are nearly uninformative and easy to match, later stages sharpen towards the true density. At every stage the particles are reweighted by the incremental density, resampled, and then **mutated** by a few random-walk Metropolis steps on their shocks that target the stage's tempered posterior. The stage contributions telescope back to the period's likelihood.

The mutation is what makes this powerful: it *moves* particles towards the data rather than merely reweighting the ones that happen to be well placed, so the cloud does not degenerate even when the observation is sharp. The bridging schedule is chosen adaptively to hit a target inefficiency ratio (`tempering_target_ratio`), so hard periods automatically get more stages than easy ones.

In practice this buys a large variance reduction per particle — several times lower standard deviation than the bootstrap filter at the same ``N`` — at several times the cost per particle. It is the right default when the bootstrap filter degenerates.

**Reference:** Herbst & Schorfheide (2019), *Tempered Particle Filtering*; see also Herbst & Schorfheide (2015), *Bayesian Estimation of DSGE Models*.

### Smoothing

`smooth = true` returns ``E[x_t \mid y_{1:T}]`` rather than ``E[x_t \mid y_{1:t}]``, i.e. estimates that use the *whole* sample. For the particle filters this is done by **fixed-interval smoothing along the filter's genealogy**: every particle surviving at ``T`` carries the ancestral line that produced it, and those lines are draws from the joint smoothing distribution ``p(x_{1:T} \mid y_{1:T})``, so averaging them with the terminal weights gives the smoothed moments directly.

Why not forward-filtering backward-smoothing (FFBS)? FFBS — and backward simulation generally — reweights the time-``t`` particles by the backward transition density ``p(\tilde x_{t+1} \mid x_t^i)``, so that any ancestor can be re-paired with any successor and the genealogy's degeneracy disappears. In a DSGE that density does not exist. The transition is ``x_{t+1} = g(x_t, \varepsilon_{t+1})`` with fewer shocks than states, so it maps ``x_t`` onto a lower-dimensional manifold and ``p(x_{t+1} \mid x_t)`` is a Dirac on it. Re-pairing a stored ``\tilde x_{t+1}`` with a *different* ancestor ``x_t^i`` would require an ``\varepsilon`` solving ``g(x_t^i, \varepsilon) = \tilde x_{t+1}`` — an overdetermined system with no solution for almost every ``i``. Every ancestor weight is zero and the backward draw is undefined. (This is the same identification arithmetic that makes the *inversion* filter work when shocks and observables balance, and fail otherwise.)

Recovering FFBS therefore requires an approximation — a kernel-regularised transition, with a bandwidth that trades bias against the degeneracy it removes — rather than a drop-in replacement. The genealogy smoother is exact for the model as written, so it is the default. The practical lever against degeneracy is `filter = :tempered_particle`, whose within-period Metropolis rejuvenation keeps many more distinct support points alive at the same `n_particles`, which is what the backward pass is short of.

#### Shock decomposition

A shock decomposition needs a shock path, and the particle filters supply one — filtered with `smooth = false`, smoothed with `smooth = true` — so they decompose either way. At **first order** the contributions are additive and the split is exact — each shock's contribution is propagated through the linear transition and the columns sum to the total. At **pruned second and third order** the contributions are *not* additive, which is exactly what the Aumann–Shapley (marginal contribution) attribution is for; both attributions are available, exactly as for the inversion filter: `marginal_contribution = false` gives the sequential split with an explicit interaction column, and `marginal_contribution = true` gives the Aumann–Shapley split that distributes the interaction across the shocks. Non-pruned `:second_order` / `:third_order` have no decomposition at any filter.

One subtlety specific to a Monte-Carlo filter: the smoothed *mean* path is not itself a model trajectory, because averaging does not commute with a nonlinear transition (``E[g(x,\varepsilon)] \neq g(E[x],E[\varepsilon])``). The pruned decomposition therefore attributes the trajectory implied by the smoothed shocks — the same object the inversion filter decomposes — so that the contributions close exactly.

The known limitation is **path degeneracy**: ancestral lines coalesce as one goes back in time, so the earliest periods rest on fewer distinct trajectories than the particle count suggests. More particles push the coalescence point further back, and `:tempered_particle` slows the coalescence itself. This is also why the standard deviations reported by `get_estimated_variable_standard_deviations` understate uncertainty early in the sample when `smooth = true`. Smoothing also stores the whole cloud, so its memory cost is about ``n_{vars} \times N \times T \times 8`` bytes — worth keeping in mind before raising `n_particles` for a long sample.

### Resampling schemes

`particle_resampling` selects how survivors are drawn. All schemes are unbiased, so the choice affects only the extra Monte-Carlo noise that resampling itself injects, ordered here from least to most:

- `:systematic` (default) — one uniform draw, then ``N`` equally spaced points through the cumulative weights. Lowest variance and cheapest.
- `:stratified` — one independent uniform per stratum of width ``1/N``. Nearly as good, with independent draws.
- `:residual` — assign ``\lfloor N W_i \rfloor`` copies deterministically, draw only the remainder.
- `:multinomial` — ``N`` independent draws. The textbook scheme, and the noisiest.

Resampling only happens when the effective sample size ``1/\sum_i W_i^2`` falls below `particle_resampling_threshold * n_particles` (default 0.5), which avoids paying the noise in periods that do not need it.

**References:** Kitagawa (1996); Douc & Cappé (2005).

## Measurement error and the initial covariance

Two inputs are not part of the model's economics but change every likelihood, and they are easy to conflate because both are called "uncertainty". They are about different objects and act on different timescales.

| | `measurement_error` ``H`` | `initial_covariance` ``P_1`` |
|---|---|---|
| uncertainty about | the **observation** ``y_t`` | the latent **state** ``x_1`` |
| acts | every period, forever | once, at ``t = 1`` |
| enters | ``F_t = CP_tC' + H`` | seeds the Riccati recursion |
| over time | **permanent** | **decays**, at the filter's own error-dynamics rate |
| encodes | distrust of the data; misspecification; a singularity fix | ignorance about where the economy started |
| changes the model? | yes — adds a noise term to the observation equation | no — it is a prior on initial conditions |

### Measurement error

``H`` is the covariance of ``\eta_t``, **never a standard deviation**: a scalar is the common variance of every observable, a vector the per-observable variances, and a matrix the full covariance. It does three quite different jobs, which are worth keeping separate:

1. **Genuine measurement error** — hours from the establishment survey is not the model's ``n_t``; GDP gets revised for years. The literal reading, and the least common reason it is used.
2. **Stochastic singularity** — a model with ``n_\varepsilon`` shocks generates observables on an ``n_\varepsilon``-dimensional manifold. Observe more series than that and the model implies exact deterministic relationships among them; ``CP_tC'`` is rank-deficient and the likelihood is not small but *undefined*. ``H > 0`` is the standard fix, and the alternative — adding shocks — is a genuine modelling choice, not a technicality (see below).
3. **Misspecification you would rather quarantine** — giving a series the model cannot match a noise term so it does not dominate the likelihood.

Mechanically it is the denominator of the Kalman gain ``K_t = P_tC'(CP_tC' + H)^{-1}``: large ``H`` means a small gain and a filter that trusts its own prediction; ``H \to 0`` means a filter that takes the data at face value. So ``H`` is a dial between *trust the model* and *trust the data*, and it is the same dial that decides whether a surprise in the data becomes an inferred structural shock or is written off as noise.

The distinction between a shock and measurement error is worth stating plainly, because both add a dimension of randomness and both cure a singularity: **a shock is variation the model transmits; measurement error is variation the model refuses to transmit.** A technology shock moves consumption, investment and hours through the model's propagation; a measurement error in output moves output's *observation* and nothing else.

For the particle filters there is a fourth, purely computational reason: without ``H > 0`` the observation density is a Dirac, no particle ever reproduces ``y_t`` exactly, every weight is zero and the filter dies. That has nothing to do with whether you believe in measurement error — it is why `:auto` picks a *small* data-driven value rather than an economically motivated one.

### The initial covariance

``P_1`` is the prior on the state at the start of the sample: where the economy was before the first observation. It seeds

```math
P_{t+1} = A(P_t - K_tCP_t)A' + BB',
```

which contracts to a fixed point that does **not** depend on ``P_1``. So the choice eventually stops mattering — but the *rate* is the filter's own error dynamics, which can be slow. On Smets-Wouters (2007) the ergodic prior and ``P_1 = BB'`` still differ by nearly 500 log points over 184 observations. This is what `presample_periods` is for: discard the periods in which ``P_1`` is still being felt.

The options are `:theoretical` (the ergodic covariance solving ``\Sigma = A\Sigma A' + BB'`` — right if the sample is a draw from the stationary distribution), `:diagonal` (``10I``, deliberately over-dispersed), or an explicit matrix.

!!! warning "Timing convention differs between filters"
    The particle filters' `initial_covariance` is ``\mathrm{Var}(x_0)`` — the cloud is drawn around the initial state and *then* propagated — whereas the Kalman filter's is ``P_1 = \mathrm{Var}(x_1)``, the first *predicted* state. They correspond as ``P_1 = A\,\mathrm{Var}(x_0)\,A' + BB'``. This is invisible at the `:theoretical` default, because the ergodic covariance is the fixed point of exactly that map and so is carried to itself — which is why passing `:theoretical` to both lines them up. It matters as soon as you supply a matrix: to reproduce a Kalman run with ``P_1 = BB'`` you must pass a **zero** matrix to the particle filter, not ``BB'``.

### How the two interact

They are not independent. With ``H > 0`` the gain shrinks, so ``P`` decays to its fixed point more slowly *and* that fixed point is strictly positive even when the state would otherwise be exactly identified:

| ``H`` | ``\lVert P_\infty^{\text{post}} \rVert`` |
|---|---|
| ``0`` | ``0`` exactly |
| ``10^{-8}`` | ``5.4\times10^{-7}`` |
| ``10^{-6}`` | ``4.6\times10^{-5}`` |
| ``10^{-4}`` | ``7.3\times10^{-4}`` |

Measurement error means you can never learn the state exactly. That is the real reason the inversion filter does not accept it: the inversion filter's ``P \equiv 0`` and ``H > 0`` are **contradictory assumptions**, not merely a missing feature.


## How the filters relate

- **Particle → Kalman.** On a *linear* model with Gaussian shocks, the particle filters estimate exactly the quantity the Kalman filter computes in closed form. As ``N \to \infty`` the particle log-likelihood converges to the Kalman log-likelihood (from below, by the Jensen bias above). This is the sharpest correctness check available and is exactly what the package's tests do, on both a small RBC model and Smets-Wouters (2007).
- **Kalman → particle.** The Kalman filter is the special case where the transition and observation are linear and the noise Gaussian, so the "cloud" is fully described by its first two moments.
- **Inversion → particle.** Both handle nonlinear models, but they make opposite trades. The inversion filter assumes measurement error is *zero* and recovers the shocks exactly; the particle filter assumes measurement error is *positive* and integrates the shocks out. As the measurement error goes to zero the particle filter degenerates towards the inversion filter's problem — and this is precisely where it needs the most particles.
- **Inversion → Kalman.** The relationship is exact and worth stating precisely. Writing ``Z = CB``, the inversion filter's per-period score is ``\log N(v_t; 0, ZZ')`` with ``v_t = y_t - CA\hat x_{t-1}`` — in *both* the square case (``Z^{-1}``) and the under-determined case (minimum norm, ``Z^{+} = Z'(ZZ')^{-1}``), since ``\|Z^{+}v\|^2 = v'(ZZ')^{-1}v``. The minimum-norm choice is not an arbitrary tie-break: for Gaussian shocks ``Z^{+}v = E[\varepsilon \mid v]``, the conditional mean. That expression is exactly the Kalman contribution with the posterior state covariance **clamped to zero** (``P_{t|t-1} = BB'``), and correspondingly the gains coincide: the inversion filter's is ``BZ^{+}``, the Kalman's is ``P_tC'F_t^{-1}``, equal iff ``P_{t|t-1} = BB'``. So: *the inversion filter is the Kalman filter that assumes the state is known exactly.* Whether that is legitimate is precisely whether ``P_{t|t} = P - PC'(CPC')^{-1}CP`` really vanishes, which needs ``\mathrm{rank}(CB) = n_\varepsilon`` — **at least as many observables as shocks**. With *more observables than shocks* the system is stochastically singular and only the Kalman filter (with measurement error) is defined. With *more shocks than observables* the clamp is simply false: ``P_{t|t} > 0`` necessarily, so ``F_t^{\text{kal}} = ZZ' + CAP_{t|t}A'C' \supsetneq ZZ' = F^{\text{inv}}`` and the inversion filter understates the innovation covariance — it treats innovations as more surprising than they are, because it pretends to know a state it cannot know. The size of the discrepancy is governed not by the shock/observable counts as such but by how much of the *unidentified* subspace propagates into the next period's observables, ``\|CAP_{t|t}A'C'\|`` relative to ``\|ZZ'\|``; when the unidentified directions barely propagate the two nearly agree anyway. There is a second, related gap: the inversion filter minimises ``\|\varepsilon_t\|`` *greedily*, period by period, ignoring that the null-space component moves ``x_t`` and hence the cost of matching later observations, whereas the Kalman disturbance smoother solves the same minimum-norm problem globally over the whole path. Finally, even when ``n_y = n_\varepsilon`` the agreement is only asymptotic: the state-estimate error obeys ``\delta_t = (I - BZ^{-1}C)A\,\delta_{t-1}``, whose spectral radius is the invertibility (fundamentalness) condition, so a near-unit-root inverse system takes many periods to forget the initial condition. Add measurement error and the inversion filter is not defined at all. At higher order its per-period Newton solve has no Kalman counterpart, which is why it — not the Kalman filter — is the default for nonlinear algorithms.
- **Correlated measurement error.** All filters that admit measurement error accept an arbitrary covariance: pass `measurement_error` a matrix instead of a vector of variances. The Kalman filter adds it to ``F_t`` directly; the particle filters factorise ``H`` once per missing-data pattern and score against the resulting triangular solve. The diagonal case is detected and takes a faster elementwise path, so there is no cost to the common case. A third option is to write the correlation into the model itself as measurement-error processes in the observation equations, which moves it into the state transition and makes ``H`` diagonal again — worth doing when the measurement errors are persistent rather than merely contemporaneously correlated.

### When are they the same filter?

The differences above are not vague family resemblances — on a linear model the three filters coincide exactly, under stated conditions, and the conditions are all about ``H`` and ``P_1``.

| pair | equivalent when | how exact |
|---|---|---|
| inversion ``\equiv`` Kalman | ``n_y \ge n_\varepsilon``, ``H = 0``, and the Kalman filter is started at ``P_1 = BB'`` | **exact, period by period** |
| inversion ``\approx`` Kalman | same but with the ergodic ``P_1`` | only asymptotically, at the rate of the inverse-system dynamics |
| particle ``\to`` Kalman | linear model, same ``H``, matching initial covariance, ``N \to \infty`` | up to Monte-Carlo error, from below (Jensen) |

The first row is the sharp statement, and it is the one to remember: **the inversion filter *is* the Kalman filter that assumes the state is known exactly.** It fixes ``x_0`` at the steady state, so the only uncertainty about ``x_1`` is that period's shocks, ``\mathrm{Var}(x_1) = BB'``; given ``n_y \ge n_\varepsilon`` the update then drives the posterior covariance to exactly zero and it stays there. Hand the Kalman filter that same prior and the two agree to machine precision — verified in the test suite on a small RBC and on Smets-Wouters (2007) with seven shocks, seven observables and 184 periods.

The second row is why the two filters normally *disagree* even on a square system: the default ergodic prior is a genuinely different starting point, and the error decays as ``\delta_t = (I - BZ^{-1}C)A\,\delta_{t-1}``. The spectral radius of that matrix is the **invertibility** (fundamentalness) condition — the "poor man's invertibility condition" of Fernández-Villaverde, Rubio-Ramírez, Sargent & Watson. If it exceeds one the inversion filter's state estimate never converges and its likelihood is wrong at any sample length; if it is close to one (0.98 in the RBC example) convergence is real but slow.

### Equivalences above first order

Above first order there is no Kalman filter to check against, but two exact references remain.

**A linear model filtered at a nonlinear order.** If the model's higher-order solution terms vanish, every perturbation order describes the same system, so a particle filter run at `:pruned_second_order` or `:third_order` must still reproduce the *Kalman* likelihood. Smets-Wouters (2007) in its log-linearised form is exactly such a model — the inversion filter returns an identical value at all five orders — which makes it a rare thing: complex enough to be a real test (40 variables, 7 shocks, 184 periods) with a known exact answer, yet exercising the pruned and non-pruned second- and third-order transitions and the pruned particle layout. Deviations from the Kalman value at ``H = 2s_i``, averaged over seeds:

| order | deviation |
|---|---|
| `first_order` | ``-2.7`` |
| `pruned_second_order` | ``-0.3`` |
| `second_order` | ``-0.3`` |

All within Monte-Carlo error and on the expected (downward) side of it. Pruned and non-pruned agree *exactly*, as they must when there is nothing to prune.

Third order on forty variables costs minutes per evaluation, so the package tests it the same way but on a small linear model, where the whole sweep — all five orders against the Kalman value — runs in seconds. The logic is identical; only the model is cheaper.

**The zero-measurement-error limit.** On a genuinely nonlinear model the reference is the inversion filter. As ``H \to 0`` the measurement density collapses onto the change of variables ``y \mapsto \varepsilon``,

```math
p(y_t \mid x_{t-1}) \;\longrightarrow\; N(\hat\varepsilon_t; 0, I)\,/\,|\det Z(x_{t-1})|,
```

which is exactly the inversion filter's per-period contribution. Give the particle filter a degenerate initial cloud (`initial_covariance = 0`, matching the inversion filter's assumption that ``x_0`` is known) and the two must agree in that limit, at any order. On the package's RBC example at `:pruned_second_order`, against an inversion value of ``386.6``:

| measurement-error variance | deviation |
|---|---|
| ``10^{-4}`` | ``-35.9`` |
| ``10^{-5}`` | ``-1.6`` |
| ``10^{-6}`` | ``-3.2`` |

The non-monotonicity is the interesting part and is not a defect: shrinking ``H`` is precisely what makes the importance weights degenerate, so the approach to the inversion filter stalls at a floor set by particle noise rather than continuing to zero. This is the same tension noted earlier — as measurement error vanishes the particle filter degenerates towards the inversion filter's problem, and that is exactly where it needs the most particles.

### Does the equivalence carry to the states?

Yes, and it is a sharper check than the likelihood: a likelihood is one scalar, whereas the states and shocks pin the whole path.

The reason is immediate once the gain is written out. With ``P_{t|t-1} = BB'`` the Kalman gain is

```math
K_t = P_{t|t-1}C'F_t^{-1} = BB'C'(CBB'C')^{-1} = BZ'(ZZ')^{-1} = BZ^{+},
```

so ``\hat x_{t|t} = A\hat x_{t-1|t-1} + BZ^{+}v_t`` — literally the inversion filter's recursion. The estimated **shocks** are then the same object as well, since the inversion filter's ``\hat\varepsilon_t = Z^{+}v_t`` is exactly the Kalman disturbance estimate.

Measured on Smets-Wouters (2007), relative maximum deviation across all variables and all 184 periods:

| comparison | deviation |
|---|---|
| inversion states vs Kalman **smoothed** states (``P_1 = BB'``) | ``7\times10^{-11}`` |
| inversion shocks vs Kalman smoothed shocks (``P_1 = BB'``) | ``9\times10^{-11}`` |
| inversion shocks vs Kalman **filtered** shocks (``P_1 = BB'``) | ``7\times10^{-11}`` |
| inversion states vs Kalman smoothed states (**ergodic** ``P_1``) | ``0.99`` |

One wrinkle worth knowing. The states match the **smoothed** Kalman estimates, not the filtered ones. That is not a contradiction of "the inversion filter's filtered and smoothed estimates coincide" — it reflects what is being conditioned on. The estimates are reported for all model variables, and a single period's seven observations do not pin all forty of them contemporaneously; the *full sample* does, through the model's own restrictions. Directly checkable: under ``P_1 = BB'`` the Kalman **smoothed** dispersion collapses to ``\approx 0`` (max standard deviation ``3.6\times10^{-5}``) while the filtered dispersion does not. Since exact identification of the state is precisely the inversion filter's assumption, the smoothed estimates are the ones it reproduces. The *shocks* match under both, because a period's shocks are pinned by that period's observations alone.

!!! note "`initial_covariance` is expressed in different bases"
    On `get_loglikelihood` the matrix is over `union(past states, observables)`; on the estimate functions (`get_estimated_variables` and friends) it is over *all* model variables. Build ``BB'`` from the same rows you intend to filter over — this is the one place where passing the matrix from the wrong path silently fails with a dimension mismatch rather than a wrong answer.

The third row is the particle filters' correctness check, and is exactly what the package's tests do: on a linear model the particle log-likelihood must approach the Kalman value as ``N`` grows, approaching it *from below* because of the Jensen bias.

## The quadratic Kalman filter

`filter = :quadratic_kalman`, available only for `algorithm = :pruned_second_order`.

### The idea

A pruned second-order solution is *exactly linear* in an augmented state. Writing the
package's own recursion,

```math
\begin{aligned}
\mathrm{aug}_1 &= [x_{1,t-1}[\text{past}];\ 1;\ \varepsilon_t], \\
x_{1,t} &= \mathbf{S}_1\,\mathrm{aug}_1, \\
x_{2,t} &= \mathbf{S}_1\,[x_{2,t-1}[\text{past}];0;0] + \tfrac12\mathbf{S}_2(\mathrm{aug}_1\otimes\mathrm{aug}_1),
\end{aligned}
```

the quadratic term uses only the *first-order* piece ``x_1``. That is what pruning buys:
stacking

```math
z_t = [\,x_{1,t};\ x_{2,t};\ x_{1,t}[\text{past}]\otimes x_{1,t}[\text{past}]\,]
```

makes every block affine in ``z_{t-1}``, because ``\mathrm{aug}_1\otimes\mathrm{aug}_1``
expands into terms that are quadratic in ``x_{1,t-1}[\text{past}]`` (carried by the third
block), linear in it, or constant. The observation ``y_t = (x_1+x_2)[\text{observables}]``
is a plain selection, so the system is linear and a Kalman filter applies. This is
Kollmann (2015).

Without pruning there is no such representation: ``x_t`` is quadratic in ``x_{t-1}``, so
``x_t\otimes x_t`` is quartic, needing ``x^{\otimes4}``, then ``x^{\otimes8}`` — the
hierarchy never closes. Pruning truncates it at exactly one rung.

### Side by side with the linear Kalman filter

It is the *same* recursion. Both filters run predict → innovate → update → accumulate, and
both score the innovation with the identical Gaussian formula. Setting
``\mathbf{S}_2 = 0`` collapses the quadratic filter onto the linear one exactly (this is a
test in the suite). The differences are entirely in what is being propagated.

```
linear Kalman (src/filter/kalman.jl)      quadratic Kalman (src/filter/quadratic_kalman.jl)
─────────────────────────────────────     ────────────────────────────────────────────────
                                          G   = reshape(g₀ + Λ(P_z z))    ← state-dependent
P̂  = A P A' + 𝐁                           P̂   = 𝒜 P 𝒜' + G G' + Q_H
û  = A u                                  ẑ   = 𝒜 z + c                  ← non-zero drift
v  = yₜ − C û                             v   = yₜ − 𝒞 ẑ
F  = C P̂ C' + H                           F   = 𝒞 P̂ 𝒞' + H
ll += log|F| + v'F⁻¹v                     ll -= ½(v'F⁻¹v + log|F| + n log 2π)
K  = P̂ C' F⁻¹                             K   = P̂ 𝒞' F⁻¹
u  = û + K v                              z   = ẑ + K v
P  = P̂ − K C P̂                            P   = P̂ − K 𝒞 P̂
```

| | linear Kalman | quadratic Kalman |
|---|---|---|
| state carried | ``x_t`` | ``z_t = [x_1;\ x_2;\ \mathrm{vech}(x_{1,p}\otimes x_{1,p})]`` |
| dimension (SW07) | 34 | 446 |
| transition | ``x' = Ax + B\varepsilon`` | ``z' = \mathcal{A}z + c + w(z,\varepsilon)`` |
| drift ``c`` | zero — certainty equivalence empties ``\mathbf{S}_1``'s constant column | non-zero — carries the risk correction |
| noise covariance | ``\mathbf{B} = BB'``, **constant** | ``G(z)G(z)' + Q_H``, **depends on the state** |
| innovation | ``B\varepsilon`` — Gaussian | ``G\varepsilon + H(\varepsilon\otimes\varepsilon - \mathrm{vec}\,I)`` — **not** Gaussian |
| observation | ``y = Cx``, general ``C`` | ``y = (x_1+x_2)[\text{obs}]`` — a selection of two blocks |
| solve per period | LU of ``F`` (``n_{obs}^3``) | Cholesky of ``F`` (``n_{obs}^3``) |
| dominant cost | ``2n^3`` | ``2n_z^3`` — about ``2250\times`` more at SW07 sizes |
| exact? | yes, for a linear Gaussian model | no — a moment-matching approximation |

Three of these carry real consequences.

**The noise covariance moved inside the loop.** In the linear filter ``\mathbf{B} = BB'`` is
built once and added every period. In the quadratic filter the innovation loading ``G``
is affine in ``z``, so ``Q`` must be rebuilt from the current state estimate at each ``t``.
That is precisely the conditional heteroskedasticity a second-order solution adds — the
model's shock impact depends on where the state is — and it is why the filter is not merely
a linear filter on a bigger vector.

**The innovation is no longer Gaussian.** ``\varepsilon\otimes\varepsilon`` is a
``\chi^2``-type object; matching only its first two moments discards every higher cumulant.
The linear filter has nothing to discard, which is why it is exact and this one is not; the
next section works through what survives the approximation.

**The cost is cubic in a squared dimension.** ``n_z`` grows like ``n_{past}^2/2``, so the
``O(n_z^3)`` covariance propagation grows like ``n_{past}^6``. This is the single fact that
governs when the filter is usable.

### What is exact, and what is not

The transition is exactly linear and the conditional first two moments are closed form.
Writing ``\mathrm{aug}_1 = \bar a + S\varepsilon``, every block of the innovation is

```math
w = G\varepsilon + H(\varepsilon\otimes\varepsilon - \mathrm{vec}\,I),
```

linear plus centred-quadratic in ``\varepsilon``. Gaussian third moments vanish, so the two
parts are uncorrelated and, using
``E[(\varepsilon\otimes\varepsilon)(\varepsilon\otimes\varepsilon)'] = \mathrm{vec}(I)\mathrm{vec}(I)' + I + K``,

```math
\mathrm{Var}(w) = GG' + H(I+K)H',
```

with ``K`` the commutation matrix. ``H`` is constant; ``G`` depends on the state and is
evaluated at the filtered mean.

What is approximated is the conditional *distribution*. ``\varepsilon\otimes\varepsilon`` is a
squared Gaussian — skewed, not Gaussian — so the recursion delivers the best **linear**
projection rather than the exact conditional mean.

### The bias, and where it comes from

Given ``z_{t-1}``, the true next state is determined by ``\varepsilon``, so the exact
conditional distribution lives on a curved ``n_\varepsilon``-dimensional surface. A Kalman
filter can only carry a Gaussian ellipsoid, and fitting one to that surface needs more
directions than the surface has:

```math
\mathrm{rank}(Q) = n_\varepsilon + \tfrac{n_\varepsilon(n_\varepsilon+1)}{2}.
```

| model | ``n_\varepsilon`` | true dimension | rank(Q) | excess |
|---|---|---|---|---|
| small RBC | 2 | 2 | 5 | 3 |
| Smets-Wouters (2007) | 7 | 7 | 33 | 26 |

The excess directions are **fictitious uncertainty**, an artefact of the Gaussian
approximation. Two things make them permanent. First, they do not come from ``\mathbf{S}_2``
— zeroing its ``\varepsilon\otimes\varepsilon`` block leaves the rank unchanged. They come from
``\mathrm{kron}(V,V)`` in the *first-order* solution: since
``x_1[\text{past}] = (\text{deterministic}) + V\varepsilon``, the state
``q = x_1[\text{past}]\otimes x_1[\text{past}]`` inherits ``V\varepsilon\otimes V\varepsilon``
whatever ``\mathbf{S}_2`` is. The quadratic noise is intrinsic to carrying a Kronecker term
as a state. Second, the observation has **zero loading on** ``q`` — the data never sees that
block directly, so it can never shrink the fictitious uncertainty. It persists and leaks into
the predicted observables, which is why the likelihood error does *not* vanish as the
measurement error goes to zero.

Measured against the inversion filter, which is the **exact** likelihood here (as many shocks
as observables, no measurement error, so it is a deterministic change of variables):

| measurement-error variance | quadratic Kalman | exact | gap |
|---|---|---|---|
| ``10^{-5}`` | 211.2451 | 213.6137 | ``-2.37`` |
| ``10^{-6}`` | 212.0289 | 213.6137 | ``-1.58`` |
| ``10^{-8}`` | 212.0937 | 213.6137 | ``-1.52`` |
| ``10^{-10}`` | 212.0943 | 213.6137 | ``-1.52`` |

It converges to a persistent gap rather than to the truth. The size is governed by
``n_\varepsilon(n_\varepsilon+1)/(2\,n_{obs})`` — 1.5 for the RBC, 4.0 for Smets-Wouters — and by
persistence: raising ``\rho`` from 0.4/0.6 to 0.98 on the same model multiplies the error
per period by 13. It is *not* governed by the size of the second-order terms, which is a
natural but wrong guess.

### What that means for usability

The bias falls almost entirely on the **level** of the likelihood, not on its shape. Profiling
against the exact likelihood, the gap varies by only about 0.2 log points across a parameter
grid, and the mode is unchanged:

| parameter | truth | argmax, exact | argmax, quadratic Kalman |
|---|---|---|---|
| shock std | 0.02 | 0.021 | 0.021 |
| persistence | 0.4 | 0.39 | 0.39 |

And the filter does the job it was designed for. Latent-state accuracy, as a fraction of each
state's own standard deviation: ``1.4\times10^{-5}`` and ``7\times10^{-7}`` for the two observed
variables, 2.8% for capital, and about 11% for the two unobserved shock processes.

**Use it for**: latent state and shock estimates at pruned second order — that is what it is
for, it is deterministic, and it is far faster than a particle filter. Point estimation, where
the mode is essentially unaffected.

**Do not use it for**: model comparison, marginal likelihoods or Bayes factors — the level
error differs across models, since it scales with the shock-to-observable ratio. Reported
standard errors without checking curvature first, since the bias is not exactly constant.
Models with many shocks per observable or near-unit persistence, where the error per period
grows sharply.

**Alternatives when the likelihood level matters**: the inversion filter is exact when shocks
and observables balance and there is no measurement error; a particle filter is consistent at
any order; and a sigma-point filter on the *unpruned* solution (Andreasen, 2013) avoids the
augmented state altogether.

!!! note "This is not the quadratic Kalman filter of Monfort, Renne & Roussellet"
    That method targets a *linear* Gaussian transition with a **quadratic measurement**
    equation, where the data loads directly on the Kronecker block and therefore shrinks its
    uncertainty every period — which is why the original paper reports large gains over the
    extended and unscented filters. A pruned DSGE is the mirror image: the quadratic terms are
    in the transition and the observation is a plain selection with zero loading on the
    Kronecker block. The machinery is shared, the regime is not.

### Cost

The augmented dimension is ``2n_r + n_{past}(n_{past}+1)/2``, where ``n_r`` counts the retained
rows (past states plus observables) and the Kronecker block is carried compressed as a
``\mathrm{vech}``. On Smets-Wouters that is 446. The covariance recursion is ``O(n_z^3)`` and
dominates everything else — per period, measured:

| operation | cost | ms | share |
|---|---|---|---|
| ``\mathcal{A}P_c`` | ``n_z^3`` (88.7M flops) | 1.09 | 42% |
| ``(\mathcal{A}P_c)\mathcal{A}'`` | ``n_z^3`` (88.7M flops) | 1.26 | 49% |
| symmetrisation ×2 | ``n_z^2`` | 0.10 | 4% |
| ``P_p - K\,CP`` | ``n_z^2 n_{obs}`` | 0.03 | 1% |
| ``GG'`` | ``n_z^2 n_\varepsilon`` | 0.03 | 1% |
| ``\mathcal{C}P_p`` | ``n_{obs}n_z^2`` | 0.06 | 2% |
| build ``G`` | ``n_z n_\varepsilon n_{past}`` | 0.02 | 1% |

The two matrix triple-products are 91% of the loop. By contrast the inversion filter solves an
``n_\varepsilon \times n_\varepsilon`` system per period — ``7^3`` against ``446^3``, a factor of
about ``2.6\times10^5`` in flops on the dominant term. That gap is structural: it is the price
of propagating a covariance over the Kronecker-augmented state, and no amount of tuning removes
it. Sparsity does not help either — ``\mathcal{A}`` is about 50% dense, and a sparse
representation measures 10× *slower* than the dense one.

**References:** Kollmann (2015), *Computational Economics* 45, 239–260 — the filter implemented
here. Andreasen, Fernández-Villaverde & Rubio-Ramírez (2018) — the pruned state-space
representation. Monfort, Renne & Roussellet (2015), *Journal of Econometrics* 187, 43–56 — the
quadratic Kalman filter for quadratic measurement equations. Andreasen (2013), *Journal of
Applied Econometrics* 28, 929–955 — the central difference Kalman filter, the unpruned
alternative.

## The cubic Kalman filter

`filter = :cubic_kalman`, available only for `algorithm = :pruned_third_order`.

The same construction one order up. Pruning truncates the Kronecker hierarchy at a fixed
rung at *every* order, so the pruned third-order solution is again exactly linear — in a
larger augmented state,

```math
z_t = [\,x_1;\ x_2;\ x_3;\ a\otimes a;\ a\otimes b;\ a\otimes a\otimes a\,],
\qquad a = x_1[\text{past}],\ b = x_2[\text{past}].
```

Writing ``a_n = Ma + v`` with ``v`` state-independent and ``u = Ma``, the new blocks close
back onto the existing ones:

```math
\begin{aligned}
q_{11}' &= (M\otimes M)q_{11} + u\otimes v + v\otimes u + v\otimes v,\\
q_{12}' &= (M\otimes M)q_{12} + (M\otimes W_q)q_{111} + (M\otimes W_l)q_{11} + u\otimes w_c + v\otimes b_n,\\
q_{111}' &= (M\otimes M\otimes M)q_{111} + \text{3 perms of }((M\otimes M)q_{11})\otimes v + \text{3 perms of } u\otimes v\otimes v + v^{\otimes3}.
\end{aligned}
```

No fourth-order block appears, because ``a_n`` carries no ``q_{11}`` term — that is why the
system closes. The closure is the whole filter: recomputing the new blocks as
``\mathrm{kron}(a_n,a_n)`` would be quadratic in ``z`` and silently destroy the linearity
everything rests on. What is approximate is exactly what is approximate at second order —
the innovation is not Gaussian and only its first two moments are matched.

Validated on an RBC model (2 shocks, 3 past states) against a converged bootstrap particle
filter: **181.78 against 181.43 over 60 periods, a gap of 0.006 per period** — smaller than
the quadratic filter's 0.025 on the comparable model.

``q_{11}`` and ``q_{111}`` are symmetric, so both are carried compressed — one entry per
sorted multi-index, the same ``\mathrm{vech}`` idea the quadratic filter uses, applied by
indexing rather than through duplication and elimination matrices. That takes the augmented
dimension from ``3n_r + 2n_{past}^2 + n_{past}^3`` down to

```math
n_z = 3n_r + \tfrac{n_{past}(n_{past}+1)}{2} + n_{past}^2 + \tfrac{n_{past}(n_{past}+1)(n_{past}+2)}{6},
```

which is roughly a sixth of the ``n_{past}^3`` block and, since the recursion is
``O(n_z^3)``, worth two orders of magnitude in flops on a mid-sized model.

!!! warning "It still only fits small models"
    Cost grows as ``n_{past}^9`` regardless of the constant factor.

    | ``n_{past}`` | ``n_z`` (compressed) | was | est. ms/period | verdict |
    |---|---|---|---|---|
    | 3 | 40 | 60 | <0.1 | fine |
    | 8 | 256 | 676 | 0.3 | fine |
    | 10 | 420 | 1245 | 1.5 | fine |
    | 12 | 640 | 2070 | 5 | usable |
    | 15 | 1091 | 3891 | 26 | usable |
    | 20 | 2231 | 8881 | 222 | marginal |
    | 27 (Smets-Wouters) | 4863 | 21243 | 2300 | no — 190 MB per matrix |

    `build_cubic_kalman_system_from_constants` refuses above
    `CUBIC_KALMAN_MAX_DIMENSION` (2500) rather than appearing to hang. For anything
    larger use the inversion filter or a particle filter.

The step function is allocation-free, the recursion runs on preallocated buffers with
in-place BLAS, the observation is applied by indexing its three selected rows rather than by
a gemm, and the quadrature contracts its nodes with a single gemm — all as in the quadratic
filter. Two things are not carried over: the transition is recovered by Gauss-Hermite
quadrature (exact, since the integrands are degree six) rather than assembled analytically,
and there is no hand-written `rrule`, so the filter is not differentiable in reverse mode.

## The filter-free likelihood

There is a fourth option that is not a `filter` value at all, because it does not filter: instead of integrating the shocks out, it treats them as **parameters** and asks you to supply them.

```julia
get_loglikelihood(model, data, parameters, shocks, measurement_error_std;
                  algorithm = :pruned_second_order)
```

Given a full path of structural shocks it forward-simulates the model, compares the implied observable path to the data under a Gaussian measurement-error model, and returns the *measurement* part of the joint log-likelihood. The priors on the shocks (typically standard normal) and on the measurement-error scale are yours to declare in the probabilistic-programming model — this function is the building block, not the whole posterior.

Why bother, when a filter integrates the shocks out for you? Because the resulting object is **smooth and differentiable at every perturbation order**, with no resampling and no per-period nonlinear solve. That makes gradient-based samplers (NUTS/HMC) usable on genuinely nonlinear models, at the cost of a much larger parameter space — ``T \times n_\varepsilon`` extra latent variables. This is the approach of Childers, Fernández-Villaverde, Perla, Rackauckas & Wu (2025).

Two things to note, both of which follow from there being no filtering distribution to track:

- **There is no initial covariance.** `initial_state` is a fixed input, not a distribution; the sampler explores the shocks, not a state posterior.
- **Measurement error carries all the noise, and is mandatory.** Given the shocks, the model path is deterministic, so without measurement error the density is degenerate. This is the opposite extreme from the inversion filter, where measurement error must be *zero*.

One naming wrinkle worth flagging: on this signature the argument is `measurement_error_std` and is a **standard deviation** (a matrix means per-period standard deviations, ``n_{obs} \times T``), whereas the filter-based `measurement_error` is a variance/covariance (a matrix means a full covariance, ``n_{obs} \times n_{obs}``). The names differ deliberately, because a matrix means different things in the two.
