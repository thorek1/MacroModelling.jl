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
| `:guided_particle` (`:particle`) | linear and nonlinear | stochastic, unbiased | no | required (incl. correlated) | yes (genealogy) | ~2× bootstrap |
| `:bootstrap_particle` | linear and nonlinear | stochastic, unbiased | no | required (incl. correlated) | yes (genealogy) | ~10³× |
| `:auxiliary_particle` | linear and nonlinear | stochastic, unbiased | no | required (incl. correlated) | yes (genealogy) | ~2× bootstrap |
| `:tempered_particle` | linear and nonlinear | stochastic, unbiased | no | required (incl. correlated) | yes (genealogy) | ~5–10× bootstrap |

A short decision rule:

- **Linear model?** Use `:kalman`. It is exact, fast and differentiable, so gradient-based samplers (NUTS/HMC) work. There is no reason to use anything else.
- **Nonlinear model, at least as many shocks as observables, no measurement error?** Use `:inversion` (the default at higher order). It is exact and differentiable.
- **Nonlinear model with measurement error, or fewer shocks than observables?** Use a particle filter. `filter = :particle` gives you `:guided_particle`, which is the right default: it draws the shock from its own conditional rather than blindly, and measures both cheaper and more accurate than `:tempered_particle` on every comparison in [Guided (`:guided_particle`, which `:particle` selects)](@ref). It buys that with one assumption — that the observation is close to linear in the shock, which is what the proposal is built from. Fall back to `:tempered_particle`, which anneals from the prior and assumes nothing, when that assumption fails badly enough that the guided filter's own bridge cannot repair it. You do not have to guess when: the filter warns if the post-bridge effective sample size averages under 5 % of `n_particles`. Expect to pay roughly ten times the compute. `:bootstrap_particle` and `:auxiliary_particle` are baselines, not recommendations.
- Particle-filter likelihoods are noisy and non-differentiable: pair them with gradient-free samplers such as slice sampling (Pigeons.jl) or nested sampling.
- **Want *estimates* (`get_estimated_shocks`, `get_estimated_variables`, the estimate plots) rather than a likelihood?** The choice of filter matters more here than it does for the likelihood — see [Estimates versus likelihoods](@ref). Stay with `:guided_particle`: what estimates need is a filter that *moves* particles onto the observation, and of the two that do, it is the more accurate and the cheaper. Use `:bootstrap_particle` and `:auxiliary_particle` as diagnostics only.

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
| `particle_mh_steps` | `2` for `:guided_particle`, `4` otherwise | the guided filter bridges from a proposal already close to the target and needs far less mutation; the tempered filter bridges from the prior and needs it badly (0.221 against 0.106 at one step versus four) |
| tempering, other | ratio `1.5`, ≤100 stages, starting scale `1.0` | the ratio is set more aggressively than Herbst & Schorfheide's own value because for the tempered filter the mutation, not the particle count, limits accuracy; the scale adapts during the run |

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

The whole swarm is propagated at once — the perturbation transition becomes a handful of `gemm` calls per period rather than one matrix-vector product per particle — and those calls are split across Julia's threads, so starting Julia with `-t auto` is worth roughly a factor of seven on a many-core machine. The split is by fixed column blocks and every random draw is made outside it, so the parallelism introduces no *statistical* dependence on the thread count.

It does not deliver bitwise determinism across machines, though, and the reason is worth knowing if you are comparing runs. Within one process a seed reproduces exactly. Across processes or thread counts the block partitioning and the vectorised reductions reassociate the same sums differently, which perturbs results at the 1e-15 level — and a particle filter can amplify that, because one flipped accept/reject or resampling boundary sends the cloud down a different (equally valid) path. **Compare seeds within a single session**, and treat a number computed on a different machine as a different draw rather than a bug.

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

The mutation is what makes this powerful: it *moves* particles towards the data rather than merely reweighting the ones that happen to be well placed, so the cloud does not degenerate even when the observation is sharp. The bridging schedule is chosen adaptively to hit a target inefficiency ratio (`particle_target_ratio`), so hard periods automatically get more stages than easy ones.

The mutation only earns that if its steps are the right size, and the right size is neither known in advance nor constant. The stage-``\phi`` target on the shocks is

```math
\pi_\phi(\varepsilon) \;\propto\; N(\varepsilon; 0, I)\,\exp\!\left(-\tfrac{\phi}{2}\, e(\varepsilon)' H^{-1} e(\varepsilon)\right),
```

which contracts as ``\phi`` rises and is strongly anisotropic — the observables pin some shocks far more tightly than others. Two things keep the sweep well scaled. First, the proposal is **preconditioned**: linearising ``e(\varepsilon) \approx e(0) - B_o \varepsilon`` with ``B_o`` the first-order impact of the shocks on the observables makes ``\pi_\phi`` Gaussian with covariance ``(I + \phi\, B_o' H^{-1} B_o)^{-1}``, and the step is drawn from that shape (an ``n_\varepsilon \times n_\varepsilon`` Cholesky per stage — negligible next to a transition evaluation). Second, the overall step **scale adapts** during the run towards a 25 % acceptance rate. `particle_mh_scale` is therefore only the starting point, expressed in units of the stage's own posterior scale, so a value near one is right whatever the model — on Smets–Wouters it settles at ``\approx 0.92``, essentially the textbook ``2.38/\sqrt{n_\varepsilon}``.

In practice this buys a large variance reduction per particle — several times lower standard deviation than the bootstrap filter at the same ``N`` — at several times the cost per particle. It is the right default when the bootstrap filter degenerates.

**Reference:** Herbst & Schorfheide (2019), *Tempered Particle Filtering*; see also Herbst & Schorfheide (2015), *Bayesian Estimation of DSGE Models*.

### Estimates versus likelihoods

The unbiasedness result above is about the *likelihood*. It says nothing about the quality of the **estimates** — ``E[x_t \mid y_{1:t}]`` and ``E[\varepsilon_t \mid y_{1:t}]``, which is what `get_estimated_variables`, `get_estimated_shocks`, `get_model_estimates` and the estimate plots report. Those are weighted averages over the cloud, and a weighted average is only as good as the number of particles actually carrying weight.

That is where the bootstrap filter's blind proposal bites hardest. With as many observables as shocks and a small ``H`` — the standard DSGE setup — the weights concentrate on a handful of particles, so the effective sample size collapses. The likelihood estimate survives this (it is still unbiased, just noisy, and the noise averages out over a sampler's iterations); a single reported shock path does not. Re-run it with a different `particle_rng` and the numbers move, sometimes enough to flip the sign of the shock the period is being attributed to.

Mutation fixes this at the source. Both `:guided_particle` and `:tempered_particle` run within-period Metropolis sweeps that *move* particles onto the observation instead of discarding the ones that missed, so the cloud the moments are taken over has many more distinct support points. On the nonlinear Smets–Wouters model at pruned second order (seven observables, seven shocks, the default ``(0.1 s_i)^2`` measurement error), that is the difference between shock estimates that agree across seeds and shock estimates that are essentially noise.

Between the two, the guided filter is the better default for estimates just as it is for likelihoods. It bridges from the conditional rather than from the prior, so it needs far fewer stages to reach the same place: on that same problem its across-seed spread of the shock estimates is 0.078 against the tempered filter's 0.093, in an eighth of the time (the table under [Guided (`:guided_particle`, which `:particle` selects)](@ref)). Reach for `:tempered_particle` when the guided proposal itself is the problem, which it tells you about.

If you do use one of the non-mutating variants for estimates, the filter tells you when the cloud has degenerated: it warns when the average effective sample size falls below 5 % of `n_particles`. Take that warning literally — raising `n_particles` shifts the threshold but not the underlying problem, which is the proposal.

#### What actually limits the precision

Even with the tempered filter, a period's estimate is not governed by `n_particles` directly, but by how many **distinct ancestors** survive that period. The measurement equation is informative about ``x_{t-1}`` as well as about ``\varepsilon_t``, so the tempering has to concentrate the ancestor cloud, and each of its stages resamples. On the Smets–Wouters problem above that leaves roughly 2 % of the swarm as distinct ancestors — a few hundred out of ten thousand — and the reported moments are averages over those. (Deferring the within-period resampling until the weights degenerate, the usual adaptive-SMC remedy, was tried and measured there: the surviving ancestors rose only from 2.2 % to 2.6 % while the stage count rose from 9.0 to 11.5, so per unit of work it was slightly *worse*. Resampling every stage, as Herbst & Schorfheide do, is kept.)

More particles push that back, but on the Smets–Wouters problem they stop helping surprisingly early: the across-seed spread of the last periods' shock estimates falls from 0.23 (in units of one shock standard deviation) at ``N = 2\,500`` to 0.16 at ``N = 10\,000``, and then **stops** — ``N = 40\,000`` gives 0.16 as well.

That plateau is *not* a statement about what the data can identify. It is the mutation running out of mixing: once the cloud is only being nudged rather than genuinely rejuvenated, adding particles adds copies of the same few trajectories. The two controls that fix it are the ones that govern rejuvenation, and both keep paying long after `n_particles` has stopped:

| `target_ratio` | `mh_steps` | seed sd of the shock estimates | sd of the log likelihood | cost |
|---|---|---|---|---|
| 2.0 | 2 | 0.199 | 147.8 | 1.0× |
| 2.0 | 4 | 0.144 | 85.9 | 1.7× |
| 1.5 | 2 | 0.161 | 67.6 | 1.3× |
| **1.5** | **4** | **0.106** | **39.1** | 2.3× |
| 2.0 | 8 | 0.088 | — | 3.1× |

(Measured at ``N = 4\,000`` over ten seeds. A lower ratio makes each bridging step gentler, so fewer ancestors are lost at its resampling; more MH steps rejuvenate harder within each step. The two compound, and both improve the likelihood as well as the estimates — which is why both defaults are set above Herbst & Schorfheide's values of 2.0 and 1. Per unit of compute both beat raising `n_particles`, and for the estimates `n_particles` stops helping altogether past a few thousand.)

The investment-specific shock `eqs` makes the point sharply. At the defaults its seed spread is 0.26 against an estimate of 0.36 — it looks unidentified, and quadrupling the particle count barely moves it. At `particle_mh_steps = 8` the spread falls to 0.068. Nothing about the data changed; the cloud simply started mixing. **A shock that looks unidentified under a particle filter should be retested with harder rejuvenation before that is believed.**

The practical reading: past `n_particles` of a few thousand, spend the next unit of compute on `particle_mh_steps` (or a lower `particle_target_ratio`), not on more particles. `particle_mh_steps = 8` is worth trying whenever a shock looks unstable.

**The direct check, and the one worth running:** call the same estimate under two or three different `particle_rng` seeds and compare. That takes seconds now and tells you exactly which of your shock estimates you can lean on. `tasks/particle_filter_diagnostics.jl` in the repository does this and prints the per-shock table above.

### Guided (`:guided_particle`, which `:particle` selects)

This is the filter `filter = :particle` gives you, and the one to reach for first.

**The idea.** The other three variants all draw the shock without looking at ``y_t`` and then repair the damage — the bootstrap filter by discarding whatever missed, the auxiliary filter by preselecting ancestors, the tempered filter by running an MCMC inside every period. This one uses the observation to draw the shock in the first place.

It can do that because of a structural feature most DSGEs share: about as many structural shocks as observables, and a measurement error that is small next to the data. Given the ancestor ``x_{t-1}``, the observation then very nearly *determines* ``\varepsilon_t``. Linearising the observed transition in the shock, ``C\,g(x_{t-1},\varepsilon) \approx m_p + B_o\varepsilon`` with ``B_o`` the first-order impact of the shocks on the observables, makes that conditional exactly Gaussian:

```math
p(\varepsilon_t \mid x_{t-1}, y_t) = N(\mu_p, M^{-1}),
\qquad M = I + B_o' H^{-1} B_o,
\qquad \mu_p = M^{-1} B_o' H^{-1} r_p ,
```

where ``r_p = y_t - m_p`` is the residual left by the zero-shock prediction. Note what ``M`` does *not* depend on: the particle. It is a property of the model and of ``H`` alone, so it is factorised once per missing-data pattern — once for the whole sample when the data has no holes — and each particle's conditional mean is then one small matrix product away from its own residual.

**One period, step by step.**

1. Push the whole cloud forward with ``\varepsilon = 0`` and read off each particle's residual ``r_p``. One batched transition.
2. Form ``\mu_p = M^{-1}B_o'H^{-1}r_p``, then refine it with two Gauss–Newton steps, ``\varepsilon \leftarrow \varepsilon + M^{-1}(B_o'H^{-1}r(\varepsilon) - \varepsilon)``, which use the *true* residual rather than the linearisation. Two more transitions.
3. Draw ``\varepsilon_j = \mu_j + U^{-1}z_j`` with ``z_j`` standard normal and ``U'U = M``, push the cloud forward again, and weight.
4. Bridge from the proposal towards the exact conditional if the weights call for it (below), resample if the effective sample size has fallen, and move on.

**Why the weights nearly vanish.** Writing out the importance weight of step 3,

```math
\log w_j = \log Z - \tfrac{1}{2}\left(\|\varepsilon_j\|^2 + r_j'H^{-1}r_j - \|z_j\|^2\right),
```

every ``\varepsilon``-dependent term cancels when the transition is *linear* in the shock. The weights are then identically constant: the filter has no conditional weight variance at all, and the only Monte-Carlo error left is the irreducible one across ancestors. That is the classical optimal-importance-function result of Doucet, Godsill & Andrieu (2000), and it is why the filter is so much stronger at first order. At higher order the cancellation is no longer exact, and what survives is precisely the curvature the linearisation misses — which the perturbation itself treats as small.

**Why the mode is refined.** Step 2 is not decoration. ``\mu_p = M^{-1}B_o'H^{-1}r_p`` is the mode only when the observed transition is linear in the shock; at pruned second order it is not, and a mis-centred proposal in seven dimensions against a target this tight is expensive. The Gauss–Newton steps lift the effective sample size of the importance weights from 0.17 to 0.48 of the cloud. Two steps is measured to be the right number: cutting it does not even save time, because a worse centre makes the bridge below take more stages and a stage costs the same transition a Newton step does. (Solving for the shock that explains the observation and then sampling around it is the implicit particle filter of Chorin, Morzfeld & Tu, 2010, from geophysical data assimilation.)

**What it costs and buys.** Six or seven batched transition evaluations per period, against the tempered filter's tens. Measured on Smets & Wouters (2007) with seven euro-area observables over 215 quarters, at the shipped defaults, over 16 paired seeds:

| | | `:tempered_particle` | `:guided_particle` |
|---|---|---|---|
| first order | seed sd of the shock estimates | 0.147 (47.7 s) | **0.075** (4.1 s) |
| pruned second order | seed sd of the shock estimates | 0.093 (90.9 s) | **0.078** (11.2 s) |
| first order | sd of the log likelihood | 76.5 (15.1 s) | **69.4** (1.1 s) |
| pruned second order | sd of the log likelihood | 72.3 (43.4 s) | **55.1** (3.8 s) |

That is between eight and twelve times less compute for the same accuracy or better, on every one of the four measurements. On a linear model, where the proposal is exact, the gap is wider still: the log-likelihood's standard deviation is a few thousandths against the tempered filter's 0.079 and the bootstrap filter's 0.185, and the mean sits on the Kalman value.

**Reaching the truth gradually.** The proposal is only as good as its linearisation, and in a period the model can barely explain — a crisis observation ten or more measurement-error units out — it is centred somewhere the conditional's mass is not. Weighting in a single step then gives heavy-tailed weights: measured on the pruned second-order problem, the worst period's effective sample size was *two to four particles whatever `n_particles` was*, and widening the proposal did not help (the mass is not where the Gaussian is, at any width).

So the filter does not do it in one step. It bridges

```math
\gamma_\beta(\varepsilon) \;\propto\; q(\varepsilon)^{1-\beta}\,\tilde\pi(\varepsilon)^{\beta}, \qquad \beta: 0 \to 1,
```

reweighting, resampling and mutating along the way — annealed importance sampling (Neal, 2001) started from the Laplace approximation rather than from the prior. With ``L(\varepsilon) = \log\tilde\pi - \log q`` the incremental weight is ``\exp((\beta'-\beta)L)``, so the same inefficiency-targeting schedule the tempered filter uses picks the steps, and the Metropolis acceptance interpolates between targeting ``q`` exactly at ``\beta = 0`` and the tempered filter's own acceptance at ``\beta = 1``.

The point is that the schedule is adaptive, so this is nearly free: on the euro-area problem it takes **1.2 stages per period on average** (at most 6–8, in exactly the periods that need them), and where the proposal is already good it jumps straight to ``\beta = 1`` and reduces to the one-step filter. What it buys is the failure mode: the worst period's effective sample size goes from 0.00025 of the cloud to **0.50**, and the average from 0.24 to 0.92.

The filter still warns when the average effective sample size falls below 5 % of `n_particles`. If that fires, the model is nonlinear enough in the shock that `:tempered_particle`, which assumes nothing, is worth its extra cost.

#### Does it survive a crisis? COVID on the euro-area data

The obvious worry about a filter built on a linearisation is what it does when the data stops behaving. The euro-area sample contains the sharpest test available: 2020Q2 sits about 170 measurement-error units away from the sample mean across the seven observables, and 2020Q3 about 149. Four things were checked.

**It runs.** Every shock estimate is finite in every period under every seed. No failure penalty is triggered.

**The bridge spends where it is needed.** This is the design working as intended — the schedule is adaptive, so it buys extra stages exactly in the periods that are hard and nowhere else:

| | stages per period | effective sample size (mean) | (worst) |
|---|---|---|---|
| the 207 ordinary periods | 1.39 | 0.96 | 0.50 |
| the 8 COVID quarters | 2.88 | 0.90 | **0.78** |

2020Q2, the hardest quarter in the sample, takes six stages and comes out with an effective sample size of 1.00. The *worst* COVID quarter is better defended than the worst ordinary one.

**The estimates agree with the filter that assumes nothing.** Through the COVID window the guided and tempered filters give the same picture — for 2020Q2, a wage-markup shock of ``4.48 \pm 0.47`` against the tempered filter's ``4.67``, a risk-premium shock of ``-2.53 \pm 0.26`` against ``-2.80``. The inversion filter differs sharply there (``5.78`` for the technology shock where the particle filters say ``0.33``), which is not a disagreement about the filter but about the model: with no measurement error the inversion filter must explain the entire COVID observation with structural shocks, so it produces enormous ones.

**It does not contaminate what comes after.** The concern with a cloud that has been through an extreme observation is that it collapses and never recovers. Measuring the across-seed dispersion era by era within one sample says otherwise:

| era | periods | dispersion |
|---|---|---|
| pre-2000 | 117 | 0.167 |
| 2008–09 financial crisis | 8 | 0.108 |
| 2010–19 calm | 40 | 0.100 |
| 2020–21 COVID | 8 | 0.229 |
| **2022 onwards** | 10 | **0.108** |

The COVID quarters themselves are harder, as they should be. The periods after them return to 0.108 — indistinguishable from the calm decade before and from the financial crisis. (The elevated pre-2000 figure is the filter still forgetting its diffuse initial cloud, not a crisis effect.)

#### Tuning it: the settings barely matter, and here is the evidence

Every option was swept on the pruned second-order euro-area problem at ``N = 4\,000`` over 32 *paired* seeds — the same seed set for each configuration, because at 10–16 seeds the same setting measured 0.058 and 0.091 in two runs and nothing can be concluded from that. `sd·√t` is the cost-normalised figure, since a Monte-Carlo error falls like one over the square root of the work.

| option | values tried | dispersion | cost | best `sd·√t` at |
|---|---|---|---|---|
| `particle_mh_steps` | 0, 1, 2, **4**, 8 | 0.097 – 0.091 | 2.4 – 9.2 s | 1 |
| `particle_target_ratio` | 1.2, **1.5**, 2, 3, 10 | 0.085 – 0.102 | 5.2 – 7.2 s | 1.5 / 10 |
| `particle_resampling_threshold` | 0.25, **0.5**, 0.75 | 0.084 – 0.091 | 5.7 – 5.9 s | 0.25 |
| `particle_resampling` | **`:systematic`**, `:stratified` | 0.091, 0.102 | 5.8 – 5.9 s | `:systematic` |
| Gauss–Newton steps (internal) | 0, 1, **2**, 3 | 0.107 – 0.092 | 6.0 – 8.0 s | 2 |

Every dispersion in that table lies between 0.084 and 0.107, against a measurement standard error of about 13 %. **No option changes the accuracy by a detectable amount**; they change the cost by a factor of four. The defaults are what they are for reasons that survive that:

- **Gauss–Newton steps = 2** is a genuine optimum rather than a tie. Cutting it does not even save time: a worse-centred proposal makes the bridge take more stages, and a stage costs the same transition a Newton step does (0 steps → 2.07 stages and 8.0 s; 2 steps → 1.30 stages and 6.0 s).
- **`particle_mh_steps` is resolved per filter** — 2 for `:guided_particle`, 4 otherwise — because the two filters are not remotely equally sensitive to it. The guided filter's estimates are flat in this knob (any value from 0 to 8 lands inside the measurement noise) and only its likelihood discriminates, putting the optimum at 2; `:tempered_particle`, which bridges from the prior, halves its dispersion going from one step to four (0.221 against 0.106). Setting `particle_mh_steps = 1` explicitly saves the guided filter roughly 40 % of its runtime at no measured cost to the estimates.
- The nominally-best `particle_resampling_threshold = 0.25` and `particle_target_ratio = 1.2` beat the defaults by 8 % and 7 %, comfortably inside the noise. Chasing those would be fitting to one sample of one problem.

#### What does not work: buying accuracy with particles

The one thing worth knowing before spending anything is that on this problem **more particles do not help**. Quadrupling ``N`` from 4 000 to 16 000 leaves the dispersion where it was, at every mutation setting:

| `particle_mh_steps` | ``N = 4\,000`` | ``N = 16\,000`` | ratio (2.0 would be textbook) |
|---|---|---|---|
| 0 | 0.099 | 0.108 | 0.91 |
| 1 | 0.094 | 0.104 | 0.90 |
| 4 | 0.090 | 0.099 | 0.92 |

The importance weights are healthy throughout (effective sample size ~0.9 of the cloud), so this is not the weight degeneracy the bridge fixed. It is the *other* degeneracy: the mutation refreshes ``\varepsilon_t`` but never the ancestor states, so over a long sample the cloud settles onto a trajectory that more particles do not change. Replication does average that away — see above — which is why it, and not `n_particles`, is the lever.

#### Spend compute on replication, not on particles

The failure above has an unusual and very useful consequence. The heavy-tailed periods are redrawn afresh from every RNG seed, so the error they cause is *independent across runs* even though it does not shrink within one. Averaging ``K`` independent runs therefore cuts the dispersion by ``\sqrt{K}`` exactly, while raising `n_particles` does nothing — and because the filter is an order of magnitude cheaper than the alternatives, ``K`` can be large. Measured on the pruned second-order euro-area problem, guided at ``N = 4\,000``:

| ``K`` | across-run sd | ``1/\sqrt{K}`` prediction | cost |
|---|---|---|---|
| 1 | 0.089 | 0.089 | 2.3 s |
| 4 | 0.042 | 0.045 | 9.0 s |
| 8 | 0.030 | 0.032 | 18.0 s |
| 16 | **0.017** | 0.022 | 36.1 s |

The tempered filter reaches 0.089 in 82 s on the same problem, so sixteen guided replicates are **five times tighter at less than half the cost**. The average is also in the right place: over 48 replicates it differs from the tempered filter's own path by an RMS of 0.054, comfortably inside the 0.085 the tempered reference disagrees with *itself* by across two halves of its seeds, at a correlation of 0.994. So this is variance reduction, not a stable wrong answer.

```julia
using Statistics
shocks = mean(collect(get_estimated_shocks(model, data;
                          filter = :guided_particle, algorithm = :pruned_second_order,
                          n_particles = 4_000, particle_rng = Random.Xoshiro(s)))
              for s in 1:16)
```

The runs are independent, so this parallelises trivially. It is also the honest way to *report* the uncertainty: the spread across those runs is the Monte-Carlo error of the estimate, and it costs nothing extra to look at.

One detail worth knowing about the reported shocks. With `particle_mh_steps = 0` the filter reports the conditional mean ``\mu_p`` rather than the shock it drew. Both are consistent for ``E[\varepsilon_t \mid y_{1:t}] = E[\mu(x_{t-1}) \mid y_{1:t}]``, but the conditional mean has already integrated the draw out and so carries none of its variance — a Rao-Blackwellisation, exact to the order the linearisation is. With rejuvenation switched on the particles are draws from the exact conditional instead, so the drawn shock is reported.

**References:** the conditionally optimal importance function is Doucet, Godsill & Andrieu (2000); building it from a local Gaussian approximation is the "unscented"/optimised particle filter family (van der Merwe, Doucet, de Freitas & Wan, 2000; Andreasen, 2013, for DSGE); solving for the shock that explains the observation before sampling around it is the implicit particle filter of Chorin, Morzfeld & Tu (2010) from geophysical data assimilation. Full adaptation in the sense of Pitt & Shephard (1999) was tried and deliberately *not* kept — see the source comment in `src/filter/particle.jl` for the measurement that rules it out here.

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
