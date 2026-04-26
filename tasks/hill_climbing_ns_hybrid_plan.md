# Implementation Plan: Hill-Climbing / Co-Area Nested Sampling Hybrid

## Overview

A hybrid algorithm that combines nested sampling's unbiased log-evidence identity with
hill-climbing (Newton/L-BFGS gradient ascent) to replace the expensive deep-shell
constrained-prior replenishment step. The key idea is the co-area formula: a hill-climbing
trajectory from a base-contour seed to a target likelihood level sweeps a shell whose
prior volume is captured exactly by the surface Jacobian accumulated along the path.

## Mathematical Foundation

The algorithm rests on the layer-cake identity

    Z = ∫ L(x) π(x) dx = ∫₀^L_max X(λ) dλ,   X(λ) = Pr_π(L > λ)

and the co-area formula

    dX/dλ = -∫_{L=λ} π(x) / ‖∇L(x)‖ dσ(x)

Combined they give

    Z = ∫₀^L_max ∫_{L=λ} π(x)/‖∇L(x)‖ dσ(x) dλ

A hill-climbing flow φ_t with ṡ = v(x), ∇L·v > 0 parameterises the level sets:
time along the trajectory maps to likelihood level, and the surface Jacobian J_Σ
(integrated via a scalar ODE alongside the climb) measures how much prior volume
the trajectory swept per unit likelihood.

## Architecture

### Three-Stage Pipeline

```
Stage 1: NS warm-up           Stage 2: Co-area quadrature        Stage 3: Laplace caps
─────────────────────    →    ──────────────────────────────  →   ──────────────────────
Standard NS until              K trajectories per basin,            Per-mode Gaussian
L_worst > λ★                  accumulate shell volumes              approximation above
Gives X̂(λ★), K seeds          via surface Jacobian ODE              λ_Lap
on Σ_{λ★}
```

**Total estimator:**

    Ẑ = Ẑ_NS(λ < λ★) + Σᵢ L̄ᵢ V̂ᵢ + Σⱼ p̂ⱼ X̂(λ_Lap) Laplace(mⱼ)

---

## Implementation Phases

### Phase 0 — Infrastructure (prerequisite)

- [ ] **0.1** Add `gradient_and_logjoint(θ)` interface returning `(log_joint, ∇log_joint)`.
  Backends: ForwardDiff, Mooncake, user-supplied. Must be a single function call to
  avoid double evaluation (gradient already evaluates the function).
- [ ] **0.2** Add `hessian_vector_product(θ, v)` interface for the surface-Jacobian ODE.
  Forward-over-reverse or finite-difference fallback. Cost must be O(d) not O(d²).
- [ ] **0.3** Implement a thin NS warm-up wrapper around an existing backend (e.g.
  NestedSamplers.jl or a hand-rolled slice-NS). Must output: `log_X_star`, live-point
  set `{x₀⁽ᵏ⁾}` at level `λ★`, and accumulated `Ẑ_NS` for shells below `λ★`.
- [ ] **0.4** Define `HybridNSResult` struct: `log_Z`, `log_Z_err`, `posterior_samples`,
  `n_likelihood_evals`, `n_gradient_evals`, `mode_locations`, `stage_contributions`.

---

### Phase 1 — Hill-Climbing Trajectory Engine

Goal: given seed `x₀` and target level `λ_target`, produce a trajectory and accumulated
shell-volume estimate.

- [ ] **1.1** Implement `ascent_step(x, ∇logL, H_approx; step_type)` with options:
  - `:gradient` — normalised gradient step (no Hessian needed)
  - `:newton_lm` — Levenberg–Marquardt damped Newton step
  - `:lbfgs` — L-BFGS two-loop recursion (memory `m`, default 20)
  The step must be **deterministic** given `(x, step_type)` to keep the trajectory
  invertible (required for Jacobian computation).

- [ ] **1.2** Implement line search satisfying **strong Wolfe conditions** (Hager–Zhang or
  More–Thuente). Deterministic bracketing only; no stochastic line search. This keeps
  the trajectory diffeomorphic.

- [ ] **1.3** Implement **surface Jacobian ODE** alongside the climb.
  Along trajectory φ_t, integrate scalar:

      d/dt log J_Σ = ∇·v(φ_t) − n̂ᵀ (∇v) n̂,   n̂ = ∇L / ‖∇L‖

  where the tangential divergence term uses one Hessian-vector product per step:
  `hvp(φ_t, n̂)` to get `(∇v)n̂`. Accumulate `log_J_Σ` as a running scalar.

- [ ] **1.4** Implement `climb_trajectory(x₀, log_joint_and_grad, hvp; λ_target,
  λ_stop_laplace, step_type, max_steps, log_level_grid)`:
  - Steps until `L(x) ≥ λ_target` or `max_steps` reached.
  - Records `(x, L(x), log_J_Σ, π(x)/‖∇L(x)‖)` at each level-grid crossing.
  - Returns trajectory struct: endpoint, shell-volume bins `V̂[i]`, log Jacobian at end,
    convergence flag, n_evals.

- [ ] **1.5** Implement **saddle detection**: monitor `min_eigenvalue(∇²logL)` cheaply
  via a Lanczos step (one HVP) at each trajectory point. If sign flips and the trajectory
  is continuing toward a saddle, flag and split into two branches (one per descending
  eigendirection). This prevents negative-volume accumulation near saddles.

- [ ] **1.6** Unit tests:
  - 2D Gaussian: analytic shell volumes, check trajectory J_Σ gives exact answer.
  - 2D banana: verify trajectory does not miss the ridge.
  - Saddle function: verify split-at-saddle triggers correctly.

---

### Phase 2 — Seed Distribution on Base Contour

Goal: given NS live points at level `λ★`, produce seeds that are approximately
i.i.d. from the prior conditioned on `L = λ★`.

- [ ] **2.1** Implement **importance-weight projection**: live points `{x⁽ᵏ⁾}` with
  `L(x⁽ᵏ⁾) ≈ λ★` are reweighted by `1/‖∇L(x⁽ᵏ⁾)‖` to approximate surface measure.
  Normalise weights; resample `K` seeds with replacement (systematic resampling for
  low variance).

- [ ] **2.2** Implement **short reverse-flow projection**: for live points above `λ★`,
  run a few reverse-gradient steps to land on `Σ_{λ★}`. This is more accurate when
  live points are well above the contour.

- [ ] **2.3** Implement **effective-sample-size (ESS) check**: compute ESS of seed
  weights; if ESS < K/4, warn and fall back to plain resampling (biased but robust).

- [ ] **2.4** Unit test: bivariate Gaussian, compare seed distribution on `Σ_{λ★}` to
  analytic surface measure.

---

### Phase 3 — Basin Clustering and Mode Discovery

Goal: cluster trajectory endpoints into modes; estimate per-basin prior mass.

- [ ] **3.1** After all `K` trajectories reach `λ_stop_laplace`, collect endpoints
  `{m⁽ᵏ⁾}`. Run **single-linkage clustering** with distance threshold `ε_cluster`
  (user-settable, default = 0.01 × prior diameter).

- [ ] **3.2** For each cluster, compute the mode precisely by running a few extra Newton
  steps from the cluster centroid. Cache `m_j`, `∇²logL(m_j)`, `log L(m_j)`.

- [ ] **3.3** Compute basin attraction fractions `p̂_j` = (number of trajectories ending
  in basin `j`) / K, weighted by seed importance weights from Phase 2.

- [ ] **3.4** Implement **basin-consistency check** across two independent batches of
  `K/2` trajectories each. If any `p̂_j` differs by more than `2σ` between batches,
  increase `K` adaptively (double and rerun the unstable basins).

- [ ] **3.5** Unit test: 5-mode mixture of Gaussians in 10D, verify all 5 modes found
  and `p̂_j` match analytic weights to within Monte Carlo error.

---

### Phase 4 — Shell Volume Estimator

Goal: combine trajectory integrands into shell-volume estimates `V̂_i`.

- [ ] **4.1** Define a **likelihood level grid** `λ★ = λ_0 < λ_1 < ... < λ_M = λ_Lap`.
  Options:
  - Fixed equal `Δ(logL)` spacing.
  - Adaptive: spacing chosen so estimated variance per bin is equal (requires a pilot
    run of `K_pilot ≈ 10` trajectories).

- [ ] **4.2** For each shell `i` and trajectory `k`, estimate the shell-volume contribution:

      V̂ᵢ⁽ᵏ⁾ = ∫_{λᵢ₋₁}^{λᵢ} [π(φ_s(x₀⁽ᵏ⁾)) / ‖∇L(φ_s(x₀⁽ᵏ⁾))‖] J_Σ(x₀⁽ᵏ⁾, s) ds

  Approximated by trapezoid rule along the trajectory's recorded grid crossings.

- [ ] **4.3** Average over trajectories:

      V̂ᵢ = (X̂(λ★) / K) Σₖ wₖ V̂ᵢ⁽ᵏ⁾

  where `wₖ` are the seed importance weights.

- [ ] **4.4** Compute Monte Carlo variance estimate for each `V̂ᵢ` as the sample variance
  of `{wₖ V̂ᵢ⁽ᵏ⁾}`. This feeds into the total `log_Z_err` in `HybridNSResult`.

- [ ] **4.5** Unit test: analytic Gaussian in `d=10`, compare `Σᵢ λ̄ᵢ V̂ᵢ` to analytic `Z`.

---

### Phase 5 — Laplace Cap

Goal: integrate the top of the posterior (near modes) analytically.

- [ ] **5.1** For each mode `m_j`, compute the Laplace approximation:

      Ẑ_j^Lap = L(m_j) (2π)^{d/2} |−∇²logL(m_j)|^{-1/2} π(m_j)

  Use the Hessian cached in Phase 3. For `d > 100`, use log-determinant via L-BFGS
  implicit Hessian rather than the full matrix.

- [ ] **5.2** Implement **sloppy-mode correction**: if `κ(∇²logL(m_j)) > 10⁶` (sloppy
  mode / near-flat direction), flag and offer a higher-order Bartlett correction or
  bridge-sampling refinement using the trajectory points near the mode.

- [ ] **5.3** Assemble cap contribution:

      Ẑ_cap = Σⱼ p̂_j X̂(λ_Lap) Ẑ_j^Lap / [L(m_j) (2π)^{d/2} |...|^{-1/2} π(m_j)]

  (The `p̂_j` handles multimodal weighting.)

- [ ] **5.4** Unit test: compare Ẑ_cap to known `Z` for a well-conditioned Gaussian
  at several `d`.

---

### Phase 6 — Full Estimator Assembly and Error Propagation

- [ ] **6.1** Implement `assemble_log_Z(ns_result, shell_volumes, cap_result)`:
  combine the three stage contributions, propagate errors via delta method, output
  `log_Z ± log_Z_err`.

- [ ] **6.2** Implement **consistency diagnostic**: compare `Ẑ_NS + Ẑ_shells + Ẑ_cap`
  to a NS-only estimate run to convergence. If they differ by more than `3 log_Z_err`,
  emit a warning listing likely cause (missed mode, saddle, sloppy Laplace).

- [ ] **6.3** Implement **posterior sample output**:
  - NS stage: use standard NS dead-point weights.
  - Shell stage: push trajectory grid points back to original coordinates, weight by
    `L(x) π(x) V̂ᵢ⁽ᵏ⁾ / Ẑ_shells`.
  - Laplace cap: draw from per-mode Gaussians, importance-correct by `L(x)π(x)` vs
    Gaussian density, merge.

- [ ] **6.4** Integration test: 20D mixture of 3 Gaussians with known `Z`, verify
  `|log Ẑ − log Z_true| < 0.1` and posterior sample coverage matches analytic marginals.

---

### Phase 7 — Adaptive Tuning and Robustness

- [ ] **7.1** **Automatic `λ★` selection**: run a pilot NS for `500 N_live` iterations,
  monitor `dC_step/dλ` (replenishment cost vs level); switch to hybrid when the marginal
  cost of one NS shell exceeds the cost of one trajectory.

- [ ] **7.2** **Adaptive `K`**: start with `K = max(50, N_live)`. After Phase 3 basin
  check, adaptively add trajectories until basin ESS > 100 per mode and shell-volume
  variance falls below target.

- [ ] **7.3** **Fallback path**: if any of the following occur, revert to vanilla NS for
  the remaining shells and skip the Laplace cap:
  - `‖∇L(x)‖ < ε_grad` on > 20% of trajectory points (non-smooth likelihood).
  - Saddle splits produce > 3× the expected number of basins (rugged landscape).
  - Basin ESS < 20 after doubling (poor mode coverage).

- [ ] **7.4** **Parallelism**: trajectories are independent given seeds; parallelise over
  `K` with `Threads.@threads`. NS warm-up is serial (standard NS constraint).

---

### Phase 8 — Benchmarking and Validation

Benchmark targets:

| Test problem | d | Modes | Expected hybrid speed-up |
|---|---|---|---|
| Multivariate Gaussian | 50 | 1 | 20–100× |
| Rosenbrock (banana) | 20 | 1 | 5–20× |
| Mixture of Gaussians | 50 | 5 | 5–30× |
| DSGE log-joint (FS2000) | ~17 | 1–2 | 10–50× |
| DSGE log-joint (SW07) | ~41 | 1–2 | 10–50× |
| Many-mode mixture | 30 | 30 | 1–5× (baseline check) |

- [ ] **8.1** Implement benchmark harness: run both vanilla NS (PolyChord-style slice)
  and hybrid on each test problem; report `n_evals`, `log_Z ± err`, wall time.
- [ ] **8.2** Run all benchmarks; write results to `tasks/hybrid_ns_bench_results.json`.
- [ ] **8.3** Profile Phase 1 trajectory engine; confirm surface-Jacobian ODE is < 20%
  of total trajectory cost.
- [ ] **8.4** Verify fallback path triggers correctly on a non-smooth test likelihood.

---

## File Layout

```
src/
  inference/
    hybrid_ns/
      types.jl            # HybridNSResult, TrajectoryResult, BasinCluster
      gradient_interface.jl  # gradient_and_logjoint, hvp
      trajectory.jl       # climb_trajectory, ascent_step, line_search, surface_jacobian_ode
      seeds.jl            # seed_from_live_points, importance_project
      clustering.jl       # basin_cluster, mode_refine, attraction_fractions
      shell_volumes.jl    # level_grid, shell_volume_estimator, variance_estimator
      laplace_cap.jl      # laplace_approximation, sloppy_mode_correction
      assembler.jl        # assemble_log_Z, posterior_samples
      adaptive.jl         # auto_lambda_star, adaptive_K, fallback_logic
      hybrid_ns.jl        # top-level entry point, calls all stages
test/
  hybrid_ns/
    test_trajectory.jl
    test_seeds.jl
    test_clustering.jl
    test_shell_volumes.jl
    test_laplace_cap.jl
    test_integration.jl   # end-to-end on known analytic posteriors
tasks/
  hybrid_ns_bench_results.json   (generated)
  hill_climbing_ns_hybrid_plan.md  (this file)
```

---

## Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Saddle-induced negative volume | Phase 1.5 saddle detector + trajectory split |
| Missed modes in basin clustering | Phase 3.4 consistency check + adaptive K |
| Surface-Jacobian blowup near mode | Stop at λ_Lap before gradient vanishes |
| Non-smooth likelihood destroys trajectories | Phase 7.3 fallback to vanilla NS |
| Weight degeneracy in seed resampling | Phase 2.3 ESS check; warn and fallback |
| Sloppy Laplace cap | Phase 5.2 higher-order or bridge-sampling correction |
| L-BFGS log-det inaccuracy for large d | Cross-check vs finite-difference log-det on small d |

---

## Acceptance Criteria

- End-to-end integration test passes: 20D 3-mode mixture, `|log Ẑ − log Z_true| < 0.1`.
- Benchmark on FS2000 DSGE achieves ≥ 10× speed-up over vanilla NS at same accuracy.
- Fallback path triggers on non-smooth test case; NS-only result is returned without error.
- All unit tests pass.
- No unbiasedness violation: 1000 independent runs on 10D Gaussian give `log Ẑ` within
  2 standard deviations of the analytic value on > 95% of runs.
