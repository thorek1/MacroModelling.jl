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

`measurement_error` is the covariance ``H`` of ``\eta_t``, never a standard deviation: a scalar is the common variance of every observable, a vector the per-observable variances, and a matrix the full covariance. `measurement_error = :auto` (the default) resolves to a variance of ``(0.1 s_i)^2`` per observable, where ``s_i`` is that observable's sample standard deviation, for the particle filters — and to *no* measurement error for the Kalman and inversion filters. For serious work set it explicitly or estimate it: the level of the likelihood depends on it, so likelihoods computed under different measurement errors are not comparable.

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

## How the filters relate

- **Particle → Kalman.** On a *linear* model with Gaussian shocks, the particle filters estimate exactly the quantity the Kalman filter computes in closed form. As ``N \to \infty`` the particle log-likelihood converges to the Kalman log-likelihood (from below, by the Jensen bias above). This is the sharpest correctness check available and is exactly what the package's tests do, on both a small RBC model and Smets-Wouters (2007).
- **Kalman → particle.** The Kalman filter is the special case where the transition and observation are linear and the noise Gaussian, so the "cloud" is fully described by its first two moments.
- **Inversion → particle.** Both handle nonlinear models, but they make opposite trades. The inversion filter assumes measurement error is *zero* and recovers the shocks exactly; the particle filter assumes measurement error is *positive* and integrates the shocks out. As the measurement error goes to zero the particle filter degenerates towards the inversion filter's problem — and this is precisely where it needs the most particles.
- **Inversion → Kalman.** The relationship is exact and worth stating precisely. Writing ``Z = CB``, the inversion filter's per-period score is ``\log N(v_t; 0, ZZ')`` with ``v_t = y_t - CA\hat x_{t-1}`` — in *both* the square case (``Z^{-1}``) and the under-determined case (minimum norm, ``Z^{+} = Z'(ZZ')^{-1}``), since ``\|Z^{+}v\|^2 = v'(ZZ')^{-1}v``. The minimum-norm choice is not an arbitrary tie-break: for Gaussian shocks ``Z^{+}v = E[\varepsilon \mid v]``, the conditional mean. That expression is exactly the Kalman contribution with the posterior state covariance **clamped to zero** (``P_{t|t-1} = BB'``), and correspondingly the gains coincide: the inversion filter's is ``BZ^{+}``, the Kalman's is ``P_tC'F_t^{-1}``, equal iff ``P_{t|t-1} = BB'``. So: *the inversion filter is the Kalman filter that assumes the state is known exactly.* Whether that is legitimate is precisely whether ``P_{t|t} = P - PC'(CPC')^{-1}CP`` really vanishes, which needs ``\mathrm{rank}(CB) = n_\varepsilon`` — **at least as many observables as shocks**. With *more observables than shocks* the system is stochastically singular and only the Kalman filter (with measurement error) is defined. With *more shocks than observables* the clamp is simply false: ``P_{t|t} > 0`` necessarily, so ``F_t^{\text{kal}} = ZZ' + CAP_{t|t}A'C' \supsetneq ZZ' = F^{\text{inv}}`` and the inversion filter understates the innovation covariance — it treats innovations as more surprising than they are, because it pretends to know a state it cannot know. The size of the discrepancy is governed not by the shock/observable counts as such but by how much of the *unidentified* subspace propagates into the next period's observables, ``\|CAP_{t|t}A'C'\|`` relative to ``\|ZZ'\|``; when the unidentified directions barely propagate the two nearly agree anyway. There is a second, related gap: the inversion filter minimises ``\|\varepsilon_t\|`` *greedily*, period by period, ignoring that the null-space component moves ``x_t`` and hence the cost of matching later observations, whereas the Kalman disturbance smoother solves the same minimum-norm problem globally over the whole path. Finally, even when ``n_y = n_\varepsilon`` the agreement is only asymptotic: the state-estimate error obeys ``\delta_t = (I - BZ^{-1}C)A\,\delta_{t-1}``, whose spectral radius is the invertibility (fundamentalness) condition, so a near-unit-root inverse system takes many periods to forget the initial condition. Add measurement error and the inversion filter is not defined at all. At higher order its per-period Newton solve has no Kalman counterpart, which is why it — not the Kalman filter — is the default for nonlinear algorithms.
- **Correlated measurement error.** All filters that admit measurement error accept an arbitrary covariance: pass `measurement_error` a matrix instead of a vector of variances. The Kalman filter adds it to ``F_t`` directly; the particle filters factorise ``H`` once per missing-data pattern and score against the resulting triangular solve. The diagonal case is detected and takes a faster elementwise path, so there is no cost to the common case. A third option is to write the correlation into the model itself as measurement-error processes in the observation equations, which moves it into the state transition and makes ``H`` diagonal again — worth doing when the measurement errors are persistent rather than merely contemporaneously correlated.

