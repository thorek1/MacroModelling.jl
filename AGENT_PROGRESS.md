# Agent progress

## Current task

Merge the particle-filter parent changes, keep pruned higher-order Kalman recursions compressed, and verify Kollmann, Ivashchenko, SW07 accuracy, and likelihood gradients.

## Status

- Parent branch `origin/particle-filter` was merged and its particle registry/defaults and documentation were retained alongside the Ivashchenko changes.
- Quadratic and cubic Kollmann-style conditional covariance corrections remain implemented entirely in compressed symmetric pair/triple coordinates.
- A separate `:ivashchenko_kalman` filter now evaluates raw second-/third-order polynomial
  solution maps and closes Gaussian moments through fourth/sixth order, respectively.
- The Ivashchenko filter has coupled theoretical mean/covariance initialization, optional
  measurement error, partial and fully missing observation support, an RTS smoother, and
  analytical reverse-mode rules for both raw second- and third-order solutions.
- CI now isolates the Pigeons/DynamicPPL resolver branch and has a dedicated Ivashchenko test row.

## Verification

- The scalar Gaussian moment reproduction passes.
- `test/test_ivashchenko_kalman.jl` passes 35/35, including Monte-Carlo moments, both orders,
  initialization modes, partial/fully missing observations, RTS smoothing, standard deviations,
  and reverse-vs-forward gradient checks.
- The isolated cubic tensor reverse check matches ForwardDiff to `8.9e-16`; the public third-order
  likelihood reverse check differs from ForwardDiff by `3.9e-9` with theoretical initialization
  and `2.8e-7` with the diagonal prior at the largest parameter gradient.
- `test/test_quadratic_kalman.jl` passes 33/33 and `test/test_cubic_kalman.jl` passes 30/30.
- CI YAML parsing and both non-Pigeons and Pigeons resolver probes pass; the direct root
  `Pkg.test()` remains intentionally unsatisfiable because it includes incompatible optional
  targets together, which the workflow-pruning steps resolve.
- `git diff --check` passes; module loading succeeds in the isolated test environment.

## Implementation decision

Ivashchenko's non-pruned Gaussian QKF is a separate algorithm rather than a switch on the
pruned augmented-state recursion. Its fourth-moment closure and unpruned state-product
dynamics require separate initialization and moment contractions. The cubic implementation
is an explicit extension of that idea; it is not attributed to Ivashchenko's second-order paper.

## Final verification

- Compressed-kernel and transition audits pass; focused tests pass: quadratic 33/33,
  cubic 30/30, Ivashchenko 35/35.
- SW07 EA compressed-space benchmark includes 138 quarters and the COVID period. Inversion
  LLH is -1062.10141543824 and pruned quadratic-Kalman LLH is -1119.47663867078; their
  difference is -57.37522323254, as expected for distinct deterministic approximations.
- On SW07 EA data (138 quarters, including COVID), the deterministic comparison is:
  inversion `-1062.10141543824` in `0.0136 s`, pruned quadratic-Kalman
  `-1119.47663867078` in `0.5691 s`, and Ivashchenko second-order
  `-1098.49135415507` in `0.6615 s`. The levels differ because these are distinct
  approximations; inversion is exact conditional on the shocks, while both Gaussian
  filters integrate a different approximate filtering distribution.
- With the guided particle filter, 2,000 particles, two MH steps, theoretical initial
  covariance, and three independent seeds, the LLHs were `-1113.59805208216`,
  `-1103.20716926924`, and `-1105.96203817403`; mean `-1107.58908650848`, SD `5.3831`,
  and median runtime `1.6329 s`. The particle LLH is stochastic and uses the package's
  `measurement_error = :auto`, so it is a benchmark rather than an exactly level-matched
  likelihood comparison. The diagonal diffuse prior produced `-Inf` for both difficult
  Gaussian filters/particle runs; theoretical initialization is used for the reported
  comparison.
- The higher-order Kalman likelihood profile on SW07 is `~550 ms` and `61.44 MiB`; the
  two dense `n_z×n_z` covariance products account for about 91% of the forward loop.
  `n_z = 446`, so sparse matrices would not help; the state transition is about 50% dense.
- The quadratic-Kalman pullback now reuses forward tape and reverse-sweep workspaces and
  uses Cholesky `ldiv!`/`rdiv!` plus in-place matrix products. It allocates `663,419,296`
  bytes (`632.69 MiB`) and has a median reverse time of `2.636 s` on the profile run,
  down from `5,629,214,896` bytes and about `3.02 s` before optimization. The remaining
  cost is dominated by the recorded dense covariance tape and the model-solution/Sylvester
  adjoint, not a generic linear solve that could be replaced cheaply.
- Warmed gradient checks: inversion ForwardDiff/reverse times 2.4796/0.0585 seconds with
  maximum gap 1.83e-10; quadratic-Kalman 438.2315/3.8058 seconds with maximum gap 7.64e-10.
- `q12 = a⊗b` is not symmetric: `q12[i,j] = a_i b_j`, while swapping indices gives
  `a_j b_i`, which is generally different because `a = x₁` and `b = x₂` differ. The
  pruned higher-order implementation therefore compresses only `q11` and `q111`; keeping
  the `nPast²` mixed block is required for correctness while remaining compressed everywhere
  permutation symmetry exists.
- Ivashchenko SW07 profile after the compressed second-order workspace/shortcut pass:
  likelihood `-1098.4913541550661`, warmed median `306.567 ms`, `33,260,032` bytes
  (`31.72 MiB`), and `130,925` allocations. The physical filtered covariance is only
  `27×27`, but the closure uses `d = nPast+nExo = 34` random inputs and
  `595 = 34·35/2` compressed random pairs. The second-order covariance now uses the exact
  identity `Cov(yᵣ,yₛ) = 1/2 tr(Tᵣ Σ Tₛ Σ)` for symmetric Hessians, avoiding the dense
  `595×595` pair-covariance matrix and its tape copies.
- Ivashchenko reverse mode now uses the matching analytical quadratic pullback and reusable
  moment workspaces: warmed median `1.131 s`, `272,275,456` bytes (`260.68 MiB`), and
  `211,822` allocations. This is down from `3.983 s` and `13,676,536,800` bytes
  (`12.73 GiB`). The remaining profile is concentrated in the 67-output quadratic moment
  products, coupled theoretical stationary initialization, and recorded model/supergradient
  work; the physical 27×27 measurement update is not the bottleneck. The likelihood is
  unchanged to floating-point roundoff. The pruned Kalman forward/reverse profiles remain
  `~552 ms/2.636 s` with `61.44 MiB/632.69 MiB` allocation.
- Direct root `Pkg.test()` remains intentionally unsatisfiable because incompatible optional
  targets are resolved together; focused tests use `tasks/isolated_test_env`.
