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
- At 2,000 tempered-particle draws, three LLHs were -302423.09, -475188.95, and
  -298818.53, demonstrating severe particle degeneracy on this sample rather than
  likelihood consistency.
- Warmed gradient checks: inversion ForwardDiff/reverse times 2.4796/0.0585 seconds with
  maximum gap 1.83e-10; quadratic-Kalman 438.2315/3.8058 seconds with maximum gap 7.64e-10.
- Direct root `Pkg.test()` remains intentionally unsatisfiable because incompatible optional
  targets are resolved together; focused tests use `tasks/isolated_test_env`.
