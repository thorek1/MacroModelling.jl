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
- Per the follow-up backend requirement, the Float64 higher-order filter paths now factor with
  `LinearSolve.FastLUFactorization()` (the repository's FastLapack extension) and use the
  existing FastLapack `solve_lu_left!`/`solve_lu_right!` wrappers for matrix RHS solves. Generic
  non-Float64 paths retain the Cholesky/Julia fallback. The requested LinearSolve/FastLapack
  combination is allocation-free in the period loop; its 446×7 gain microbenchmark is about
  3.8% slower than calling the standalone FastLapack workspace directly, but it is the selected
  backend and preserves the established repository abstraction.
- Focused verification after this pass: quadratic Kalman 33/33, cubic Kalman 30/30, Ivashchenko
  35/35, and the repository allocation-pattern check passed. SW07 EA compressed forward LLHs
  remained `-1119.47663867078` (pruned quadratic) and `-1098.4913541550648` (Ivashchenko).

## Compressed third-order Hermite/contraction pass (2026-08-03)

- The pruned cubic Kollmann recursion now tapes its forward products in preallocated buffers and
  reuses the corresponding reverse products, rank-one updates, and stationary-adjoint scratch.
  The generic non-Float64 gain solve also uses `copyto!` plus in-place `rdiv!` instead of forming
  a transposed solve result. The analytical pullback remains the source of the gradient.
- The non-pruned Ivashchenko third-order covariance no longer constructs the dense
  `n_triple×n_triple` sixth-moment matrix. Its six fully cross-connected Gaussian pairings are
  evaluated by sequential compressed mode contractions through `d×n_pair` and `n_pair×d`
  workspaces; the other nine pairings are the covariance of the cubic Hermite linear component.
  The pullback reverses those same contractions analytically and recomputes small intermediates
  rather than taping a dense sixth-moment object.
- The direct sixth-moment identity test compares the factorized compressed result with all 15
  Gaussian pairings for the third-order RBC system; the focused Ivashchenko suite now passes
  36/36. The final focused suites are quadratic 33/33, cubic 30/30, and Ivashchenko 36/36.
- Representative four-variant RBC profile (40 periods) after this pass:
  pruned quadratic `0.363 ms/1.240 ms` forward/reverse with `0.14/0.83 MiB` allocated;
  pruned cubic `3.124 ms/14.547 ms` with `1.04/5.28 MiB`; Ivashchenko second order
  `1.194 ms/4.944 ms` with `0.70/3.08 MiB`; and Ivashchenko third order
  `8.928 ms/34.615 ms` with `3.93/17.11 MiB`. Relative to the immediately preceding
  third-order profile, the Ivashchenko third-order path is about 55% faster forward, 50% faster
  in reverse, and uses about 44% fewer reverse bytes. The second-order Ivashchenko and pruned
  quadratic timings are noisy across runs, so no additional wall-time gain is claimed for them.
- On the SW07 EA sample (138 quarters, including COVID), second-order Ivashchenko retains the
  exact LLH `-1098.4913541550648`; the measured forward profile is `330.310 ms` and `23.93 MiB`,
  while reverse is about `1.008 s` and `260.97 MiB`. The profile identifies stationary
  initialization, quadratic moment products, and model/supergradient work as the dominant
  costs; the 27×27 measurement update is not the bottleneck. A third-order SW07 attempt at the
  supplied setup returns `-Inf`, so it is recorded as a stationary/finite-likelihood failure,
  not as a valid accuracy benchmark.
- A common filter-level pullback-buffer rewrite was benchmarked and reverted: it did not improve
  wall time and increased the reverse allocation count to about 1.82 million. Runtime-first
  profiling therefore retained only the compressed third-order factorization and measured local
  workspaces. Full `Pkg.test()` remains intentionally unrun; the focused isolated suites above
  are the verification evidence.

## SW07 EA per-algorithm profile refresh (2026-08-03)

- Current warmed likelihood timings on the 138-quarter EA sample are: inversion `14.1117 ms`,
  pruned quadratic Kalman `545.432 ms`, Ivashchenko second order `321.584 ms`, and guided
  particle with 2,000 particles `3.512–3.550 s` per seed. The corresponding LLHs are
  `-1062.1014154382399`, `-1119.4766386707768`, `-1098.4913541550648`, and guided-PF draws
  `-1099.2006420496748`, `-1102.9177271673316`, `-1102.076127449579` (mean `-1101.3981655555283`,
  SD `1.9490779077078477`).
- Flat likelihood profiles identify the inversion bottleneck as the per-observation
  Lagrange-Newton shock solve: repeated compressed quadratic/Jacobian assembly and LU of the
  Newton system in `find_shocks`. The quadratic Kalman bottleneck is the dense 446×446 augmented
  covariance propagation/noise covariance and measurement update, with BLAS SYRK/GEMM work
  dominating. Ivashchenko is dominated by its 67-output quadratic moment products and coupled
  theoretical stationary initialization; the 27×27 observation update is comparatively small.
- Guided PF is dominated by repeated `propagate_cloud!` calls inside the per-period Newton proposal
  and eight-step annealed MH mutation, including compressed quadratic propagation for all 2,000
  particles. Cloud gathering/copying and Mahalanobis evaluation are secondary. The profile used
  50 inversion repetitions, three deterministic-filter repetitions, and one guided-PF run; output
  is in `tasks/sw07_ea_breakdown.log`.

## Pruned Ivashchenko Gaussian closure (2026-08-03)

- Implemented `:ivashchenko_kalman` for `:pruned_second_order` and
  `:pruned_third_order`. The implementation composes the pruned stage recursion into a
  compressed polynomial map over the flattened stage states, then uses the existing analytical
  Gaussian moment closure. It keeps only state/observable rows for likelihood evaluation and
  does not construct Kollmann's dense pair/triple augmented state.
- The effective coefficient map agrees with direct pruned recursion on the RBC probe to maximum
  errors `0.0` (second order) and `4.44e-16` (third order). The focused pruned test now passes
  `22/22`, including both estimate paths, smoothing, a second-order Zygote/ForwardDiff gradient
  check (maximum absolute gap `2.2e-6`), and a third-order reverse/central-difference check
  (`-1292.162631` versus `-1292.162688` for the alpha coordinate). The raw Ivashchenko suite
  passes `36/36`.
- Matched RBC benchmark (40 periods, theoretical initial covariance, no measurement error):

  | filter | order | LLH | median time | allocated |
  | --- | ---: | ---: | ---: | ---: |
  | inversion | 2 | `142.1725907560` | `0.132 ms` | `136,416 B` |
  | Kollmann quadratic | 2 | `140.4410776334` | `0.182 ms` | `151,888 B` |
  | pruned Ivashchenko | 2 | `140.3689995857` | `2.939 ms` | `1,502,832 B` |
  | inversion | 3 | `142.1687319621` | `0.225 ms` | `156,160 B` |
  | Kollmann cubic | 3 | `140.4525552888` | `3.117 ms` | `1,086,784 B` |
  | pruned Ivashchenko | 3 | `140.3619727848` | `230.225 ms` | `43,067,072 B` |

  The close second/third-order Kollmann/Ivashchenko LLHs on this small matched case are an
  approximation comparison, not an identity claim: the filters propagate different Gaussian
  objects after starting from the same pruned polynomial recursion.
- Pre-staged-initialization SW07 EA benchmark (138 quarters, seven observables including COVID;
  warmed isolated environment; theoretical initial covariance and no measurement error):

  | filter | order | LLH | median time | allocated |
  | --- | ---: | ---: | ---: | ---: |
  | inversion | 2 | `-1062.1014154382` | `3.414 ms` | `3.21 MB` |
  | Kollmann quadratic | 2 | `-1119.4766386708` | `558.483 ms` | `64.39 MB` |
  | pruned Ivashchenko | 2 | `-1091.9397819781` | `1.654 s` | `82.26 MB` |
  | inversion | 3 | `-1063.5772989044` | `28.381 ms` | `17.06 MB` |
  | raw Ivashchenko | 2 | `-1098.4913541551` | `~0.32 s` | `~24 MB` |

  The historical pre-shortcut pruned Ivashchenko second-order likelihood is finite and sits
  between inversion and Kollmann; its generic stationary initialization was the main extra cost.
  The pruned third-order SW07
  builder rejects before allocation at an estimated `1118.1 MiB` compressed workspace, while
  Kollmann cubic independently exceeds its augmented-dimension limit (`4863 > 2500`). Thus no
  third-order SW07 LLH is reported for either higher-order Kalman approximation.
- The pre-shortcut EA pruned-Ivashchenko profile was dominated by coupled stationary
  initialization and quadratic Gaussian moment contractions (compressed pair kernels and
  their transpose/BLAS products). The 27×27 measurement update is not the bottleneck.

## Staged pruned stationary initialization (2026-08-03)

- Audited the existing block-triangular Lyapunov implementation in `src/moments.jl` and reused
  its Sylvester cross-block plus smaller Lyapunov solves. The new forward Float64 path in
  `src/filter/ivashchenko_kalman.jl` solves pruned second- and third-order stationary means and
  covariances stage by stage, entirely in compressed physical-state coordinates. It falls back
  to the original fixed-point iteration for AD element types, missing workspaces, and failed
  structural/numerical checks. The taped reverse path intentionally remains on the generic
  iteration because its existing analytical pullback records those iteration tapes.
- The compressed staged mean matches Kollmann's augmented mean projection to `2.1e-17` on the
  RBC check. The staged covariance matches the previous generic Ivashchenko fixed point to
  `1.7e-18` on RBC and `4.8e-14` on SW07. The projected Kollmann physical covariance differs by
  `2.60` max norm on SW07, confirming a closure difference rather than an implementation error.
- The SW07 EA benchmark was rerun with both filters on `:pruned_second_order`, theoretical
  initialization, no measurement error, 138 quarters including COVID, and seven observables:

  | filter | LLH | median runtime | allocated |
  | --- | ---: | ---: | ---: |
  | Kollmann quadratic | `-1119.4766386708` | `540.995 ms` | `64,389,504 B` (`61.4 MiB`) |
  | pruned Ivashchenko | `-1091.9397819857` | `531.030 ms` | `30,777,440 B` (`29.4 MiB`) |

- The stationary initializer alone is `10.9 ms` and `7,560,576 B` on SW07, versus `1.12 s` and
  `56,331,488 B` for the old generic iteration. The remaining Ivashchenko cost is the repeated
  67-output compressed Gaussian moment contraction; the physical 27×27 measurement update is
  still not the bottleneck.
- Focused verification after this pass: pruned Ivashchenko `26/26`, raw Ivashchenko `36/36`,
  and the small pruned third-order stationary residuals are `3.2e-20` for the mean and `8.7e-19`
  for the covariance. Full `Pkg.test()` was not run because the repository's optional resolver
  targets remain intentionally incompatible.

## Alternative Gaussian covariance closures (2026-08-03)

- The Gaussianity assumptions are now stated explicitly. Ivashchenko assumes the physical
  state/shock input is jointly Gaussian, contracts the polynomial map's moments exactly under
  that input law (fourth moments at second order and sixth moments at third order), and then
  Gaussianises the transformed physical output before the Kalman update. Kollmann instead
  lifts products into an augmented monomial state and Gaussianises that lifted vector inside a
  linear Kalman recursion. Product coordinates are not jointly Gaussian in general, so the two
  closures need not have the same physical covariance or likelihood.
- Added the differentiable `ivashchenko_gaussian_closure` keyword to `get_loglikelihood` and
  the inner Ivashchenko pass. `:exact` remains the default. `:linearized` keeps the exact
  nonlinear Gaussian mean and the effective first-order/Hermite Jacobian covariance while
  dropping nonlinear variance. `:diagonal` is available for second order and keeps exact
  curvature variances but sets curvature cross-covariances to zero. Both modes have analytical
  pullbacks; the cubic `:linearized` path reverses `EΣE'` and the cubic Hermite Jacobian without
  materialising nonlinear covariance contractions.
- SW07 EA compressed filter benchmark (138 quarters, seven observables, including COVID,
  theoretical initialization, no measurement error) gives:

  | closure | LLH | median runtime | allocated |
  | --- | ---: | ---: | ---: |
  | `:exact` | `-1091.9397819857` | `539.8 ms` | `20,362,464 B` (`19.42 MiB`) |
  | `:linearized` | `-1062.2036107071` | `323.0 ms` | `14,128,160 B` (`13.47 MiB`) |
  | `:diagonal` | `-1194.8536608614` | `491.0 ms` | `17,183,840 B` (`16.39 MiB`) |

  The linearized mode is about 40% faster and 31% lower in forward bytes, but its LLH is an
  intentionally different approximation. The diagonal mode saves less time because it still
  computes one Hessian-times-covariance product per output.
- At the SW07 stationary point, the discarded exact curvature covariance has trace `479.13`;
  its leading eigenvalue contains `87.43%`, the leading four `99.86%`, and the leading eight
  `99.99996%` of positive trace. A fixed low-rank output basis is therefore promising, but it
  was not exposed: the optimal basis depends on parameter-dependent Hessians/covariances, and
  treating it as fixed would make the analytical gradient silently inconsistent. A safe
  low-rank version would need differentiated eigensensitivity or a parameter-independent
  feature basis.
- Dropping the second-order mean correction was also prototyped. It saves only about `7.5%`
  of the isolated moment-kernel time beyond `:linearized` while changing the SW07 output mean
  by `8.46` in max norm, so it was not retained as a public closure.
- Focused verification after the closure pass: raw Ivashchenko `45/45`, pruned Ivashchenko
  `32/32`, and quadratic Kalman `33/33`. The full test suite remains intentionally unrun.

## Exact structural and batched contraction pass (2026-08-04)

- Raw Ivashchenko likelihood systems now support an exact compact output map. Likelihood and
  missing-data calls retain only the state rows and requested observables (`nout=34` on raw SW07,
  versus `67` for the estimate/smoothing map); the full-row default remains unchanged for the
  estimate API. The raw compact and full systems produce the same SW07 EA LLH to roundoff.
- The input covariance is block diagonal, `Σ = diag(P, Iₑ)`. Forward and analytical reverse
  products against `Σ` now use BLAS for the state block and direct scaling for the shock block,
  avoiding dense multiplies through the zero state–shock blocks. The isolated exact contraction
  moved from `153.2 μs` to `131.9 μs` on the SW07 compressed shape with zero per-call allocations.
- The effective quadratic/cubic Jacobians are now formed in batches from analytical compressed
  pair/triple derivative matrices, followed by one matrix RHS GEMM. The exact second-order
  Jacobian kernel moved from `148.5 μs` to `59.5 μs`; its reverse contribution moved from
  `226.9 μs` and `68` allocations to `46.4 μs` and zero allocations. The cubic reverse matrix-RHS
  batch matches `gS₃` exactly and the VJP to `2.3e-14`, moving from `49.3 μs` to `34.8 μs`.
- On the raw SW07 EA direct filter path, full rows benchmark at `182.2 ms` and `18,637,216 B`
  (`17.77 MiB`), while the exact compact likelihood path is `112.4 ms` and `17,938,304 B`
  (`17.11 MiB`), both with `207,121` allocations. Both LLHs are
  `-1098.4913541550602`; the public path reports `-1098.4913541550643`.
- The final public `get_loglikelihood` benchmark on the same EA data is `116.2 ms`,
  `21,423,168 B` (`20.43 MiB`), and `270,293` allocations, with LLH
  `-1098.4913541550643`.
- The raw SW07 reverse likelihood benchmark is `388.2 ms`, `223,698,064 B` (`191.05 MiB`),
  and `324,949` allocations. Forward/reverse gradients agree to `1.04e-10` at the tested
  parameter vector. The compact-output regression includes a single-observable raw likelihood
  and passes its reverse/forward gradient comparison.
- Structural sparsity is not an exact win for SW07: the raw quadratic Hessian is `86.08%` dense
  (`66,674` nonzeros out of the full storage; `31,078` in the compact map). An upper-triangular
  scalar covariance contraction reproduces the BLAS result to `2.3e-13` but takes `675 μs`
  versus `132 μs`, so the dense GEMM is retained. Parameter-dependent low-rank/eigenbasis
  truncation remains excluded because it changes the closure and requires differentiated basis
  sensitivities.
- Focused verification after this pass: raw Ivashchenko `48/48`, pruned Ivashchenko `32/32`,
  quadratic Kalman `33/33`, and cubic Kalman `30/30`. Full `Pkg.test()` remains intentionally
  unrun because the optional resolver targets are incompatible together.

## Exact Kollmann noise-contraction batch pass (2026-08-04)

- Exact sufficient-row compression was already present in both Kollmann builders: they retain
  only past-state and observable rows. The Ivashchenko shortcut `Σ = diag(P, Iₑ)` is not
  transferable because Kollmann's augmented covariance is full; its analogous exact structure
  is in the innovation-noise contractions.
- Quadratic Kollmann now batches `Lⱼ P` for all shock blocks as one matrix-RHS product, then
  accumulates `Q += (LⱼP)Lⱼ'`. The internal forward, stationary, and taped paths use a
  preallocated `LPaAll` workspace; the old blockwise path remains available for compatibility.
  On SW07's `446×446` shape the isolated contraction moved from `769.6 μs` to `626.8 μs`, with
  maximum difference `7.3e-12`.
- Cubic Kollmann now batches `Λnoise * Pnoise[:, first:last]` in chunks of at most 16 columns
  on the primal, stationary, taped, and analytical reverse paths. The chunk bound avoids a
  large temporary near the cubic dimension limit. On the representative RBC cubic shape the
  forward contraction moved from `25.7 μs` and 12 allocations to `19.9 μs` and zero allocations;
  the reverse batch is exact and covered by the cubic gradient tests.
- SW07 EA pruned quadratic likelihood remains `-1119.4766386707768`; the current public path
  benchmarks at `529.0 ms` and `62.72 MiB` (`112,196` allocations). The dense `446×446`
  covariance propagation still dominates, so the end-to-end gain is smaller than the isolated
  noise-kernel gain. Cubic SW07 remains outside the existing augmented-dimension guard.
- Focused verification after this pass: quadratic Kalman `34/34` and cubic Kalman `31/31`.
  The earlier raw/pruned Ivashchenko suites remain `48/48` and `32/32`; full `Pkg.test()` stays
  intentionally unrun.
