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

## Allocation/BLAS optimization pass (2026-08-03)

- The four higher-order variants remain structurally distinct and compressed as requested:
  pruned quadratic/cubic Kollmann recursions and non-pruned second-/third-order Ivashchenko
  moment closures. No dense augmented state was introduced.
- The pruned quadratic path now exploits its selector structure directly when forming the
  lagged covariance terms, avoiding selector GEMMs. Its analytical pullback uses BLAS `ger!`
  for large rank-one updates. A wider batched GEMM was benchmarked on SW07 and rejected because
  it was slower than the existing seven block GEMMs.
- The cubic pullback reuses forward products, adjoint products, and stationary-adjoint buffers.
  On the RBC four-variant profile this reduced pruned-cubic reverse allocation from roughly
  58 MiB to 21.6 MiB while preserving the analytical gradient.
- Ivashchenko moment contractions now use reusable third-order workspaces and `mul!`; the
  filtering pass reuses measurement/update buffers and has a full-observation factorization
  path. On SW07 this reduced forward bytes to 24,017,472 (22.90 MiB) and reverse bytes to
  271,724,768 (259.10 MiB). Allocation counts increased to 316,162 and 398,439 respectively,
  so both bytes and counts are reported; the byte reduction did not produce a material runtime
  improvement on this run.
- A ping-pong stationary-initialization rewrite was measured and reverted because it increased
  the Ivashchenko SW07 path to about 68 MiB and 3.4 million allocations. The retained changes
  are only measured workspace/contraction improvements.
- Focused verification after the pass: `test/test_quadratic_kalman.jl` 33/33,
  `test/test_cubic_kalman.jl` 30/30, and `test/test_ivashchenko_kalman.jl` 35/35.
- Current deterministic SW07 EA benchmark (138 quarters, seven observables, COVID period,
  warmed isolated environment): inversion LLH `-1062.1014154382399` in `0.013596875 s`,
  pruned quadratic Kalman LLH `-1119.47663867078` in `0.5736505 s`, and Ivashchenko
  second-order LLH `-1098.4913541550648` in `0.322061875 s`.
- Direct SW07 profile samples report pruned quadratic forward `64,428,496` bytes and reverse
  `668,828,080` bytes; Ivashchenko forward `24,017,472` bytes and reverse `271,724,768` bytes.
  The direct timed samples were variable, so the deterministic benchmark above is the runtime
  comparison and the profile is used for allocation/cost attribution.
- `git diff --check` passes. The full test suite was not run because the repository's optional
  resolver targets are intentionally incompatible; the isolated focused suites are the required
  verification for this pass.

## Repository-wide allocation pattern pass (2026-08-03)

- Audited Kalman, Ivashchenko, particle, inversion, cubic, quadratic, Sylvester, and reverse-mode
  code for repeated inverses, temporary solves, `repeat`-based scaling, and avoidable matrix
  products. Only the following measured, local changes were retained.
- The non-Float64 Kalman branch now reuses its solve vector with `copyto!`/`ldiv!` and updates the
  gain with `rdiv!` instead of forming `inv(LU)`. The isolated microbenchmark reduced the generic
  solve from 16 to 8 allocations and 6.53 KiB to 2.34 KiB; numerical error was below `4e-16` on
  the Float64 equivalence check and the ForwardDiff Kalman reproduction passed.
- The legacy Kalman smoother/decomposition no longer creates `repeat(shock', n_state)` for each
  period. Direct column scaling was about 3.4× faster in the microbenchmark and reduced the
  temporary allocation from 3.28 KiB/4 allocations to 1.64 KiB/2 allocations.
- Ivashchenko RTS smoothing now uses Cholesky-backed `rdiv!` solves instead of explicit covariance
  inverses. The isolated smoother check matched to `2.3e-15` and reduced temporary bytes by about
  42%; the full Ivashchenko suite still passes 35/35.
- The particle tempering proposal now computes `U⁻¹` through an in-place triangular solve. The
  isolated check matches the inverse reference exactly; the microbenchmark was slightly faster
  and reduced temporary bytes, though allocation count was unchanged-to-slightly higher, so this
  is recorded as a memory/robustness cleanup rather than a major runtime claim.
- `test/test_particle_filter.jl` could not run in the isolated environment because it imports the
  unavailable `StatsPlots` package. The proposal-factor equivalence check ran independently.

## LinearSolve workspace audit (2026-08-03)

- Preallocated `LinearSolve.CholeskyFactorization` and `FastLUFactorization` caches were compared
  with direct in-place LAPACK on the filter-relevant dimensions. The cache solves were numerically
  exact and allocation-free when given a dedicated factor buffer; Cholesky medians were
  `0.333/2.208/528.625 μs` at dimensions `7/27/446`, versus `0.292/2.000/475.584 μs` for direct
  `cholesky!` plus `ldiv!`. The current FastLapack LU workspace remained about `0.95 ms` at `446`.
- The real gain solve is a `446×7` matrix divided on the right by a `7×7` innovation factor.
  Reusing the LinearSolve cache's factor object was exact and allocation-free but measured
  `4.334 μs` versus `4.125 μs` for direct `cholesky!` plus `rdiv!`. The cache's normal matrix-RHS
  initialization does not provide a usable multi-RHS `u` buffer in the current LinearSolve
  version, so using its internal factor field would add coupling without improving the hot path.
- LinearSolve factorization backends can overwrite their cached `A`; a separate preallocated
  factor buffer must be refilled before every solve. Reusing the source matrix without restoring it
  gives invalid repeated factorizations and was excluded from the measurements.
- The higher-order filter covariance buffers now use direct in-place `cholesky!`/`lu!` where the
  unfactored covariance is dead. This removes factor-object allocations while preserving the
  existing multi-RHS `rdiv!`/inverse solves. LinearSolve remains appropriate for the existing
  single-RHS SSS caches, but no LinearSolve path was added to the higher-order filters.
- Focused verification after this pass: quadratic Kalman 33/33, cubic Kalman 30/30, Ivashchenko
  35/35, and the repository allocation-pattern check passed. SW07 EA compressed forward LLHs
  remained `-1119.47663867078` (pruned quadratic) and `-1098.4913541550648` (Ivashchenko).
