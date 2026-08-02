# Agent progress

## Current task

Implement Ivashchenko's unpruned Gaussian moment-closure filter as a separate filter for raw second- and third-order solutions, and resolve the CI dependency isolation issue.

## Status

- Repository progress file was absent at task start; this file records the current task.
- Quadratic and cubic Kollmann-style conditional covariance corrections remain implemented.
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
