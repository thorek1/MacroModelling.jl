# Agent Progress

## Current task

Make the particle filters fast, allocation-free and seed-reliable on the Smets & Wouters
(2007) euro-area filtering problem in `pf_debug_repl.jl`. See the second half of
`tasks/todo.md`; the compressed higher-order work below is the previous task.

## Completed

- Established the Julia environment with `Pkg.instantiate()` after the initial package load reported the missing `DocStringExtensions` dependency.
- Added allocation-free compressed pair/triple Kronecker-vector kernels, matrix overloads, compressed index maps, and analytical VJP helpers.
- Added explicit `compressed_kron²_power`/`compressed_kron³_power` kernels and matching analytical power VJPs; legacy `*_same` names remain compatibility aliases.
- Precomputed triangular pair-coordinate maps and cubic shock-state row maps in model constants, then threaded those maps through inversion and higher-order warmup pullbacks.
- Kept ordinary, pruned, OBC, IRF, filter-free, particle, inversion, shock-decomposition, Aumann–Shapley, and higher-order reverse-mode transition paths in compressed coordinates.
- Updated stochastic-steady-state Newton residual/Jacobian assembly to use compressed state/constant columns and triangular/cubic scratch buffers.
- Preserved compressed matrices at internal state-update and stochastic-steady-state interfaces; public `get_solution` still expands to its documented full tensor output.
- Added focused kernel/reference-equivalence tests and a static transition audit.

## Verification

- `test/test_compressed_kron.jl`: 86/86 vector checks, 4/4 power edge-case checks, 6/6 analytical VJP checks, 3/3 state-update equivalence checks, 2/2 cached-row-map checks, and 67/67 static-audit checks.
- The same focused kernel/audit test passed again after threading cached maps through all third-order reverse-mode warmup callers; Julia rebuilt the package successfully.
- Third-order cache setup smoke on the RBC_CME model produced 20 shock-state-state and 12 shock-shock-state indices with matching cached row maps.
- `test/test_inversion_filter_likelihood.jl`: 7/7 focused likelihood-level checks (5m46.2s; includes the intentional 701-point profile).
- Inversion filter likelihood smoke: 7/7.
- Third-order and pruned-third-order transition/IRF smoke: 4/4.
- Particle-filter compressed transition smoke: 2/2.
- Filter-free likelihood smoke across second, pruned-second, third, and pruned-third order: 4/4.
- All edited Julia files parse cleanly and `git diff --check` is clean.

## Intentional full-space conversions

- Public `get_solution` expands with `𝐔₂`/`𝐔₃` for compatibility with its documented duplicated-axis tensor output.
- Existing moment formulas and their pullbacks use full pair/triple tensor coordinates where those formulas explicitly require them.
- Third-order Sylvester/RHS construction expands the second-order solution internally; this remains isolated to solver assembly and is not used by state updates.
- The compression matrices `𝐔₂`/`𝐔₃` remain as representation/solver constants, but no active ordinary/pruned transition, simulation, IRF, filtering, particle-filter, OBC, or filter-free transition path expands with them.

The full test suite was intentionally not run, per the task instructions. The large third-order Gali OBC probe was stopped during its expensive solve after the pruned-second-order compressed transition assertion passed; smaller OBC and higher-order probes covered the active compressed code paths.

A redundant `--compiled-modules=no` source-load probe was interrupted after 8.5 minutes while compiling the dependency graph; the normal package rebuild and focused tests completed successfully.

## Performance benchmark follow-up

- Created a detached `HEAD` baseline worktree and ran the same benchmark harness against the pre-compressed full-Kronecker implementation.
- Added the reproducible harnesses [tasks/benchmark_compressed_vs_full.jl](tasks/benchmark_compressed_vs_full.jl) and [tasks/benchmark_particle_parallel_prototype.jl](tasks/benchmark_particle_parallel_prototype.jl).
- On a 5-variable RBC model (3 past states, 2 shocks, 32 periods, pruned third order), cached filter-free likelihood was 0.216 ms compressed versus 1.371 ms full, about 6.3x faster, with 165 kB versus 255 kB allocated.
- The same small-model 2048-particle bootstrap filter was 153 ms compressed versus 111 ms full. This is a crossover case: resampling, random draws, and measurement scoring dominate, and dense BLAS/full Kronecker work can be competitive at small augmented dimension.
- Preallocated direct transition sweeps (4,000 steps) measured the following compressed/full medians: augmented dimension 6, 1.91/2.76 ms; 13, 7.39/6.17 ms; 21, 82.1/995 ms; 33, 451/4,596 ms. The compressed loop allocated zero bytes during the sweep; the full reference allocated about 1.47 MB, 6.08 MB, 326 MB, and 1.20 GB respectively.
- Added explicit same-vector power kernels and allocation-free power VJPs. Against a local pre-specialization permutation reference, cubic power kernels were 2.6–4.7x faster for augmented sizes 6–33; pair kernels were within measurement noise because their arithmetic is already minimal.
- A four-thread particle propagation/scoring prototype with thread-local scratch achieved 1.88x speedup on 32,768 particles × 12 steps at augmented dimension 21, with identical output and 31 kB scheduler allocation. Production parallelism is not enabled yet.

## Particle parallelism assessment

The predict/score loops in bootstrap, auxiliary, and tempered filters are particle-independent within a period and can be threaded. Resampling, weight normalization/reduction, tempering level selection, and MH acceptance remain sequential synchronization points. A safe production design must pre-draw shocks sequentially (preserving `particle_rng` order), use a typed chunk function with one scratch/shock/measurement buffer per chunk, and either restrict the fast path to diagonal measurement error or give dense measurement-error state one cache per chunk. An opt-in threshold is preferable because the small RBC end-to-end benchmark did not benefit.

## CI repair follow-up

- Added `Statistics` to the test target so the particle-filter test environment can load the stdlib explicitly.
- Made particle-pool cross-representation copy methods explicit for JET's nested-vector union split, and narrowed validated particle measurement error before the keyword dispatch boundary.
- Added the missing cached shock–shock–state row bindings in ordinary and pruned third-order inversion paths; pure shock-coordinate selectors now use global compressed maps without state offsets.
- Reproduced the Windows inversion `BoundsError` with the affected selector shape and verified second-order RBC calibration and inversion calls return finite results after the global-map fix. The focused missing-data likelihood and shock-estimation probes also pass.
- The old CI run's later SSS finite-difference and Turing initialisation failures were not treated as solved by masking them; they require separate numerical reproductions if they recur after these direct dispatch and index fixes.

LoopVectorization is not applicable to the branchy triangular cubic kernel; a branchless candidate was slower. Existing preallocation, function barriers, `mul!`, `@inbounds`, and dense/sparse selection are already used where applicable.

## Review follow-up: invariant compressed index maps

- Added lazy constant caches for all pair maps used by inversion, `find_shocks`, and reverse-mode warmup paths, including the distinct no-volatility shock×state selector.
- Added lazy constant caches for the four third-order selector maps used by those paths.
- Removed every runtime `compressed_pair_indices`/`compressed_triple_indices` call from transition, filtering, inversion, and rrule code; the only remaining calls are one-time cache initialization and the helper definitions.
- Extended the static audit to reject runtime pair/triple map construction.
- Package-load/constructor smoke, the compressed-kernel suite (83/83 static-audit checks), and a pruned-third-order RBC model-level map-equivalence smoke all pass.

## Review follow-up: symmetric tangents, SSS filtering, and scratch reuse

- Corrected Aumann–Shapley higher-order tangents: the symmetric compressed pair derivative uses coefficient `1` after the outer `1/2`, the symmetric cubic derivative uses coefficient `1/2` after the outer `1/6`, and mixed pair derivatives retain both compressed terms with coefficient `1`.
- Added regression checks against full `𝐔₂`/`𝐔₃` directional derivatives, including the repeated-input multiplicities.
- Added allocation-free `compressed_triple_state_to_pair!` and `compressed_triple_state_pair_to_shock!` kernels and reused existing third-order inversion workspace matrices in repeated forward and reverse-mode call sites. The bang kernels measure only 32 bytes of call overhead after warm-up and do not allocate their output matrices.
- The selector matrices are structurally sparse (the state-to-pair example has 110 nonzeros out of 1100 entries), but their current consumers accept dense workspace matrices and the sparse alternative would require a cached CSC pattern. The current change removes repeated matrix allocation without changing matrix representation; sparse CSC caching remains a separate optimization candidate.
- Stochastic-steady-state Newton now receives only past/mixed rows and the state/constant prefix of compressed policy matrices from the calculating interface. The full compressed matrices remain available for returned interfaces and final state evaluation; the public Newton wrapper retains its full-matrix-compatible default behavior for rrules.

Verification for this follow-up:

- `test/test_compressed_kron.jl`: all test groups passed (86 vector, 4 power-edge, 6 VJP, 3 state-equivalence, 3 directional-derivative, 6 cached-helper, and 83 static-audit assertions).
- `test/test_inversion_filter_likelihood.jl`: 7/7 checks.
- User-facing RBC smoke: second-, third-, and pruned-third-order stochastic steady states all returned finite values.

## ForwardDiff gradient repair

- Added the `filtered` keyword to the ForwardDiff higher-order stochastic-steady-state Newton overloads and kept their residual/Jacobian contractions compressed.
- Replaced the remaining full `kron(aug₁, aug₁)` construction in the pruned second-order filter-free forward rrule with the compressed power kernel.
- Fixed pruned third-order reverse pullbacks so the compressed cross-term VJP is accumulated into the existing linear `aug₂` cotangent instead of overwriting it; the visible and warmup paths now match the full-coordinate accumulation semantics.

Verification:

- `test/test_filter_free_gradients.jl`: 185/185 passed in 5m03.4s in an isolated environment containing the optional ForwardDiff, Zygote, FiniteDifferences, DifferentiationInterface, ADTypes, and Mooncake test dependencies.
- The focused boundary probe passed all five algorithms, with analytical shock cotangents agreeing with fifth-order central finite differences to approximately 1e-12.
- The full test suite remains intentionally unrun.


## Particle filters: speed, allocations and seed reliability

Problem: `get_estimated_shocks` with a particle filter on Smets & Wouters (2007), pruned
second order, 7 euro-area observables, 215 quarters (`pf_debug_repl.jl`).

### What was wrong

- The reported hot loop (`particle.jl` 2083-2101) was a symptom. The cause was that the
  whole estimates path in `filter_data_with_model` never received the typed treatment the
  likelihood path has: the transition was a closure over branch-assigned variables, so
  every particle-period went through dynamic dispatch, and the solution matrices were left
  sparse instead of being densified.
- The tempered filter's Metropolis mutation used a fixed isotropic step. The stage-phi
  target contracts as phi rises and is strongly anisotropic across shocks, so a single
  fixed scale cannot suit it and the cloud was barely rejuvenated.

### What changed

- Rewrote the filters around a batched particle cloud (`nVars x n_particles` matrices, one
  per pruned state component). A period is now a handful of `gemm` calls instead of
  `n_particles` `gemv` calls, and the estimates path shares that machinery instead of
  duplicating it. `src/filter/particle.jl` is about 200 lines shorter.
- Split the propagation across Julia threads by column block, one scratch slot per task.
  Every random draw happens outside the parallel region, so the result is bit-identical
  whatever the thread count.
- Preconditioned the Metropolis proposal with the stage's own Gaussian covariance
  `(I + phi B'H^-1 B)^-1` and adapted its scale towards 25% acceptance.
- Added a warning when the cloud degenerates (average ESS below 5% of `n_particles`).
- Workspace: replaced the six fixed `nVars x n_particles` buffers with a pool handed out in
  groups of one per pruned component (`ensure_particle_pools!`).

### Measured (SW07 pruned second order, 184-period US sample, n_particles = 2000)

| | before | after |
|---|---|---|
| `get_estimated_shocks`, bootstrap | 12.06 s, 4900 MiB, 319M allocs | 0.35 s, 19.4 MiB, 36k allocs |
| `get_estimated_shocks`, tempered | 92.75 s, 6042 MiB, 382M allocs | 8.4 s, 85 MiB, 720k allocs |
| `get_loglikelihood`, bootstrap | 2.08 s, 8.7 MiB | 0.28 s, 14.8 MiB |
| `get_loglikelihood`, tempered | 24.80 s | 8.4 s |

Batched propagation alone: 2.57 us/particle serial, 0.33 us/particle on 32 threads (7.8x),
with the parallel and serial results asserted bit-identical.

### Reliability, on the euro-area problem

- Bootstrap and auxiliary filters are unusable for *estimates* here: average effective
  sample size 0.02% of `n_particles`. This now warns.
- Tempered filter, adaptive mutation: acceptance sits at 0.250 from the first period and
  the scale settles at ~0.92, close to the textbook 2.38/sqrt(7) — evidence the
  preconditioner captures the target's shape.
- The across-seed spread of the last periods' shock estimates plateaus at ~0.16 (units of
  one shock sd) by `n_particles = 10_000`; 40 000 gives 0.163. The existing default is
  therefore already at the achievable floor.
- The floor is one shock. Per shock, seed sd over the estimate's own size: ea 0.18,
  eb 0.20, eg 0.16, em 0.29, epinf 0.20, ew 0.19, **eqs 0.91**. The investment-specific
  shock is not identified at the `(0.1 s)^2` auto measurement error on investment growth,
  which is the noisiest observable. This is a property of the data, not of the filter.
- Deferring the within-period resampling until the weights degenerate (adaptive SMC) was
  implemented, measured and reverted: distinct surviving ancestors rose only from 2.24% to
  2.63% while stages per period rose from 9.0 to 11.5. The finding is recorded in the
  source comment so it is not re-attempted.

### Verification

- `tasks/pf_verify.jl`: batched compressed-Kronecker kernels against the reference vector
  kernels; the batched transition against the model's own `state_update` at all five
  perturbation orders; Kalman-likelihood equivalence on a linear model for every order and
  every variant (bootstrap 191.514, auxiliary 191.558, tempered 191.538 against a Kalman
  191.534, tempered lowest variance); filtered estimates within 0.014 of the Kalman
  filter's; missing data; smoothing; both shock decompositions.
- `test/test_particle_filter.jl`: 142/142. Its plotting testset was excluded because this
  container has no `libgobject` and `Plots` cannot precompile; nothing else was skipped.
- The full test suite was intentionally not run.

### Harnesses (in `tasks/`)

`particle_filter_diagnostics.jl` (per-shock seed stability — the one to run),
`pf_verify.jl`, `pf_alloc_probe.jl`, `pf_block_bench.jl`, `pf_tempering_trace.jl`,
`pf_ea_convergence.jl`, `pf_ea_pershock.jl`, `pf_profile.jl`.

### Option sweep follow-up

Every particle-filter option screened one at a time on the euro-area problem
(`n_particles` 4000, 10 seeds, across-seed sd of the last 8 periods' shock
estimates). Two options dominate, both acting on how much particle diversity
survives a period, and both were previously left at Herbst & Schorfheide's values:

  target_ratio  mh_steps    estimates sd    log-likelihood sd    cost
      2.0          2           0.199              147.8          1.0x
      2.0          4           0.144               85.9          1.7x
      1.5          2           0.161               67.6          1.3x
      1.5          4           0.106               39.1          2.3x
      2.0          8           0.088                 -           3.1x

They compound, they improve the likelihood as well as the estimates, and both are
better per unit of compute than raising `n_particles` — which for the estimates
stops helping past a few thousand particles. Defaults changed to
`tempering_target_ratio = 1.5`, `tempering_mh_steps = 4`.

Confirmed at `n_particles` 10 000: `tempering_mh_steps = 8` halves the across-seed
spread of every shock (rms over all seven 0.152 -> 0.077).

Options that did not improve on the default: `particle_resampling` (stratified and
multinomial edge out systematic on the identified shocks but are worse on `eqs`,
with no decisive aggregate gain), `initial_covariance = 0` (better on the
identified shocks, much worse on `eqs`), `tempering_mh_scale` (no material effect,
which is the intended behaviour since it adapts), and the `bootstrap`/`auxiliary`
variants (3x worse than tempered for estimates).

### A correction to the previous section

The earlier conclusion that the investment-specific shock `eqs` "is not identified
at the auto measurement error" was **wrong**. It was a mutation mixing failure: at
`tempering_mh_steps = 8` its across-seed spread falls from 0.26 to 0.068 with no
change to the data or the measurement error. A shock that looks unidentified under
a particle filter should be retested with harder rejuvenation before that is
believed. The documentation has been corrected.

Also corrected: the claim that results are bit-identical across thread counts. They
are exact within a process (and across workspace resizes and configuration changes),
but block partitioning and the vectorised reductions reassociate sums, so results
differ at ~1e-15 across processes and ~1e-12 across thread counts. Seeds should be
compared within one session.

### Verification after the default change

- `tasks/pf_verify.jl`: all groups pass. The tempered filter tightened further on
  the linear-model reference — likelihood sd 0.079 against 0.097 at the old
  defaults and 0.185 for the bootstrap filter — and its missing-data likelihood
  moved from 268.18 to 267.88 against a Kalman 267.94.
- `test/test_particle_filter.jl`: 142/142 (plotting testset excluded as before).

## Theory follow-up: a guided (conditionally optimal proposal) particle filter

The option sweep established that mutation *mixing* was the binding constraint, i.e. the
filter was spending its whole budget recovering information the bootstrap proposal had
discarded. So the next step was to fix the proposal rather than the mutation.

### The idea

A DSGE usually has about as many structural shocks as observables and a measurement error
small next to the data, so given the ancestor the observation very nearly *determines* the
shock. Linearising the observed transition in the shock, `C g(x,eps) ~ m + Bo eps`, makes
the conditional Gaussian with a covariance that does not depend on the particle:

    p(eps_t | x_{t-1}, y_t) = N(mu_p, M^-1),  M = I + Bo' H^-1 Bo,  mu_p = M^-1 Bo' H^-1 r_p

and the importance weight collapses to

    log w = logZ - (1/2)(|eps|^2 + r'H^-1 r - |z|^2),

in which every eps-dependent term cancels when the transition is linear in the shock — so
the weights are then identically constant and the only variance left is the irreducible one
across ancestors. `M` is factorised once per missing-data pattern. Added as
`filter = :guided_particle`.

Two refinements matter:

- **Gauss-Newton mode finding** against the *true* residual rather than the linearisation
  (the implicit particle filter of Chorin, Morzfeld & Tu). This is the decisive step at
  higher order: it lifted the importance weights' effective sample size from 0.17 to 0.48
  of the cloud on the euro-area problem.
- **Rao-Blackwellised shock estimate**: report `E[eps|x,y]` rather than the draw, since
  `E[eps_t|y_1:t] = E[mu(x_{t-1})|y_1:t]`. Used when no rejuvenation is requested.

`tempering_mh_steps` is reused as the number of Metropolis rejuvenation steps at phi = 1,
which repairs the cloud in periods the proposal misses.

### Tried and rejected

Full adaptation in the sense of Pitt & Shephard — preselecting ancestors on the closed-form
predictive density — is *actively harmful* here. The Laplace lambda overstates how well a
tail ancestor explains the observation; that ancestor wins the first-stage resampling and
its correction weight then turns out to be negligible. Because a larger cloud reaches
further into the tail, the failure got worse with `n_particles` (seed dispersion 0.065 at
4 000 particles, 0.136 at 40 000, with the worst period's effective sample size stuck at
two to four particles whatever N was). Resampling once on the *combined* weight makes
lambda cancel algebraically and removes the mechanism. Recorded in the source comment.

### Measured (EA data, SW07, 7 observables, 215 quarters, 10 seeds)

                              tempered            guided
  first order   est. sd       0.105 (40.6 s)      0.050 (2.1 s)
                loglik sd     109.5 (17.8 s)       15.9 (0.7 s)
  pruned 2nd    est. sd       0.089 (82.0 s)      0.125 (5.8 s)
                loglik sd      60.0 (43.7 s)       27.0 (2.0 s)

Linear-model reference (`tasks/pf_verify.jl`, Kalman = 191.534): likelihood sd 0.005 for
guided against 0.079 tempered and 0.185 bootstrap, mean 191.536 — 16x tighter than the
tempered filter. Its missing-data likelihood is also the closest of the four to the Kalman
value (267.964 against 267.937).

Verdict: guided wins outright for **likelihoods** at both orders — 2-7x tighter standard
deviation at 20-25x less cost, which is the case that matters because a sampler wants
thousands of them — and for **estimates** at first order. At pruned second order its
estimates are about 1.4x less accurate than the tempered filter's while being an order of
magnitude cheaper. Defaults were left unchanged and `:tempered_particle` remains the
recommendation when accuracy on a strongly nonlinear model is what matters.

### Known limitation

The guided filter's estimates do not improve with `n_particles` on the pruned second-order
problem (0.097 at 4 000, 0.125 at 10 000, 0.136 at 40 000). In the few periods the model can
barely explain, the importance weights are heavy-tailed enough that the worst period's
effective sample size sits at two to four particles whatever N is. Metropolis rejuvenation
repairs the cloud but not the weights. The average-ESS warning surfaces the regime.

### Identified next step

Annealed importance sampling *from* the Laplace approximation: bridge along
`q^(1-beta) pi^beta` rather than prior to posterior. The incremental weight is
`exp(dbeta * log omega)` with `log omega` already computed, so the existing adaptive
schedule would take one step where the proposal is good (cheap, the common case) and many
where it is not (robust, the crisis periods) — precisely the missing robustness. Also
unexplored: per-particle Jacobians, which would correct the proposal's *covariance* rather
than only its centre, and randomised quasi-Monte-Carlo point sets for the shock draws
(Gerber & Chopin's SQMC).

### Verification

- `tasks/pf_verify.jl`: all groups pass with the new variant included. On the linear
  reference the guided filter's likelihood sd is 0.005 against 0.079 (tempered) and 0.185
  (bootstrap); its filtered estimates match the Kalman filter's to 0.0144 like the others;
  its missing-data likelihood is the closest of the four to the Kalman value.
- `test/test_particle_filter.jl`: 142/142 with `:guided_particle` registered (plotting
  testset excluded as before — this container has no `libgobject`).

## Roadmap item 1: the annealed bridge (kept)

The guided proposal's remaining defect was that a single importance-weighting step has
heavy-tailed weights in the periods the model can barely explain — the worst period's
effective sample size was two to four particles whatever `n_particles` was, and
over-dispersing the proposal was measured and does not help (the conditional's mass is
not where the Gaussian is, at any width).

So the filter now reaches the conditional gradually, bridging
`gamma_beta = q^(1-beta) * pitilde^beta` from the proposal to the truth. With
`L = log pitilde - log q` the incremental weight is `exp(dbeta * L)`, so the tempered
filter's existing inefficiency-targeting schedule picks the steps, and the Metropolis
acceptance interpolates between targeting the proposal exactly at beta = 0 and the
tempered filter's own acceptance at beta = 1. This is annealed importance sampling
(Neal, 2001) started from a Laplace approximation instead of from the prior. The
one-step case is the plain guided filter, so it is a strict generalisation.

Measured on the euro-area problem (pruned second order, 12 seeds):

  guided annealed N= 4000   rms_all 0.0861   1.2 stages/period   ESS mean 0.916 min 0.508    6.2 s
  guided annealed N=10000   rms_all 0.0579   1.3 stages/period   ESS mean 0.916 min 0.500   15.3 s
  guided annealed N=40000   rms_all 0.0957   1.4 stages/period   ESS mean 0.966 min 0.510   73.3 s
  tempered         N=10000  rms_all 0.0872                                                  80.8 s

The failure mode is gone: worst-period effective sample size 0.00025 -> 0.50, average
0.24 -> 0.92. It costs almost nothing because the schedule averages 1.2 stages per
period and only spends more where the proposal is poor. At `n_particles` 10 000 this is
1.5x tighter than the previous best at 5.3x less cost; at 4 000 it matches it at 13x
less cost. On a linear model the bridge is correctly a no-op and the likelihood is the
Kalman value to four decimals (191.53457 against 191.53411, sd 0.005 — the best of the
four variants).

This makes roadmap item 3 (Pareto-smoothed importance sampling) obsolete — it existed to
tame exactly the heavy tails that are now gone — and much reduces items 2 and 4. Items 5
(sequential quasi-Monte Carlo) and 6 (quadratic Kalman filter) remain open.

Honest caveat: the seed-dispersion metric is noisy at 10-16 seeds (the same
configuration measured 0.0579 and 0.0905 in two runs), so the N-scaling is not cleanly
resolved. The mechanism numbers (effective sample size, stages per period) are not
subject to that noise.

Verification: `tasks/pf_verify.jl` all groups pass with guided best in every one;
`test/test_particle_filter.jl` 142/142.

## Guided-filter defaults and the `:particle` alias

Tuned the guided filter on its own terms and gave it its own defaults.

Every option was swept at 32 *paired* seeds (the same seed set per configuration; at
10-16 seeds the same setting had measured 0.058 and 0.091 in two runs). Accuracy is
flat across every option — all dispersions fall in 0.086-0.107 against a ~13% standard
error — while cost varies by a factor of four. So cost decides, and the one
well-supported change is the mutation count.

`tempering_mh_steps` is now **1 for `:guided_particle`** and stays 4 for the others,
via `DEFAULT_TEMPERING_MH_STEPS_SELECTOR(filter)` following the package's existing
`DEFAULT_SMOOTH_SELECTOR` pattern. The guided filter bridges from a proposal already
close to the target so it needs little mutation; the tempered filter bridges from the
prior and needs it badly (0.221 against 0.106 at one step versus four). Verified on the
likelihood as well as the estimates, since the knob governs both paths: EA
log-likelihood sd 22.5 / 19.6 / 27.5 / 23.6 at mh 1 / 2 / 4 / 8, flat, with mh=1
cost-normalised best.

`:particle` now aliases `:guided_particle` rather than `:bootstrap_particle`. This is a
deliberate behaviour change: the alias pointed at the worst of the four variants, by
roughly a factor of six in dispersion at several times the cost per unit of accuracy.
`test/test_particle_filter.jl`'s alias assertion, the `FILTER®` and
`TEMPERING_MH_STEPS®` docstrings, and `docs/src/filters.md` were updated to match.

r* = 1.5, resampling threshold 0.5, `:systematic` and two Gauss-Newton steps were all
kept: their nominal challengers sit inside the noise, and r* = 10 would loosen the
bridge's safety net for no measured gain.

Honest note: on the tiny linear test model the guided likelihood's dispersion rises
from 0.003 at mh=4 to 0.017 at mh=1. Both remain far below the tempered filter's 0.079
and the bootstrap filter's 0.185, and the effect is absent on the realistic problem.

Verification: `tasks/pf_verify.jl` all groups pass with guided best in every one;
`test/test_particle_filter.jl` 142/142; and a direct wiring check confirming the alias
resolves to guided, the guided default equals an explicit `tempering_mh_steps = 1`, and
the tempered default still equals an explicit 4.

## Guided default corrected, COVID robustness, and the filters.md explanation

**Correction.** `DEFAULT_GUIDED_MH_STEPS` is 2, not the 1 set previously. That choice was
made on the estimates metric, where every value of mh measures the same inside the noise
so the cheapest won — but the same knob governs the likelihood, which is not flat in mh.
Over 32 paired seeds the log-likelihood dispersion is 31.4 / 24.2 / 27.3 at mh 1 / 2 / 4
at first order and 34.4 / 24.9 / 35.0 at pruned second order: two steps is the minimum at
both orders and per unit of compute, and the estimates are tied between 1 and 2. A second
correction: an earlier claim of mine that the first-order likelihood was very sensitive to
mh (15.9 against 69.4) came from unpaired seeds and is withdrawn — that statistic is noisy
enough that two unpaired runs of one configuration gave 22.5 and 55.1.

**Head-to-head at the shipped defaults**, 16 paired seeds, euro-area data: guided against
tempered gives estimate dispersions of 0.075 (4.1 s) against 0.147 (47.7 s) at first order
and 0.078 (11.2 s) against 0.093 (90.9 s) at pruned second order, and log-likelihood
dispersions of 69.4 (1.1 s) against 76.5 (15.1 s) and 55.1 (3.8 s) against 72.3 (43.4 s).
Eight to twelve times less compute for equal accuracy or better on all four.

**COVID.** The euro-area sample's 2020Q2 sits about 170 measurement-error units from the
sample mean. The guided filter runs clean through it: every estimate finite under every
seed; the adaptive bridge spends 2.88 stages per period in the COVID window against 1.39
elsewhere and holds the effective sample size *higher* there (worst COVID quarter 0.78
against a worst ordinary period of 0.50, with 2020Q2 itself at 1.00 over six stages); the
estimates agree with the tempered filter; and the periods afterwards return to calm-period
dispersion (0.108, against 0.100 for 2010-19 and 0.229 during COVID itself), so the cloud
recovers rather than staying degraded.

**Documentation.** The guided section of `docs/src/filters.md` was rewritten as an
explanation — the idea, the algorithm step by step, why the importance weights cancel, why
the mode is refined, cost and benefit, the COVID evidence, and the tuning evidence — and
the alias, filter table, decision rule and defaults table updated.

Verification: `tasks/pf_verify.jl` all groups pass with guided best in every one;
`test/test_particle_filter.jl` 142/142.
