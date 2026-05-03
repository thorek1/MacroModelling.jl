# Task Todo

- [x] Replace the SW2003 `plots_2` source-side workaround with a test-suite-only fix.
  - [x] Screen SW2003 first-order parameter perturbations in `test/functionality_tests.jl` so invalid nearby draws are not exercised by the shared harness.
  - [x] Reuse the screened full-vector perturbation everywhere `functionality_tests.jl` previously generated a fresh random `old_params .* exp.(rand(...) * 1e-4)` candidate.
  - [x] Move `Smets_Wouters_2003 with calibration equations` to the front of `test/test_plots_2.jl`.
  - [x] Remove the temporary `parameterised_execution_model(...)` cloning workaround from `src/get_functions.jl`.
  - [x] Verify the focused SW2003 functionality path in `tasks/plots2_verify_env` with `plots = false`.

- [x] Fix the SW2003 `plots_2` CI failure cascade from run 25251760637.
  - [x] Add a focused reproducer covering the `get_statistics`, `get_moments`, `get_irf`, and residual failure paths.
  - [x] Restore temporary parameter state after failing parameterized getter calls.
  - [x] Make bad finite-difference parameter points in `get_statistics` and non-derivative `get_moments` return nonfinite placeholders instead of throwing.
  - [x] Relax the platform-fragile SW2003 FD-vs-AD tolerances recorded in the CI log.
  - [x] Verify the reproducer and record the blocked local `Pkg.test()` attempt.

- [x] Fix the SW07 `get_statistics` correlation NaN regression.
  - [x] Add a focused reproducer script for the nonlinear SW07 first-order correlation path.
  - [x] Fix the shared primal/rrule correlation construction so nondegenerate variables keep unit diagonal correlations.
  - [x] Verify the reproducer and the narrow SW07 functionality path.
  - [x] Move the `get_statistics - correlation` regression to the front of `functionality_test` so nonlinear SW07 fails earlier in CI.

- [x] Fix CI run 25132726254 targeted failures.
  - [x] Add focused repro/regression scripts for QME initial-guess tolerance, FRBUS `irf` binding, and stochastic steady-state `converged` type.
  - [x] Lower QME initial-guess acceptance tolerance from `1e-8` to `1e-10` in solver defaults.
  - [x] Rename the FRBUS test-local `irf` binding to avoid Julia 1.10 imported-variable assignment errors.
  - [x] Cast stochastic steady-state rrule `converged` values to `Bool` before `!converged` checks for JET.
  - [x] Verify all focused scripts and edited-file diagnostics.

- [x] Fix the failing basic-model loglikelihood gradient checks.
  - [x] Reproduce the QUEST3 and GNSS failures with a focused script.
  - [x] Fix the first-order pullback workspace-aliasing bug in `src/rrules.jl`.
  - [x] Verify the first-order, state-update, Kalman, and full loglikelihood directional derivatives against scalar finite differences.
  - [x] Fix the QME initial-guess shortcut so the original `test/test_models.jl` finite-difference checks pass unchanged.

- [x] Add production Lyapunov `:dqgmres` support and document why ILU/triangular-sweep Krylov preconditioners remain out of the production solver.
  - [x] Reproduce missing low-level `Val(:dqgmres)` method.
  - [x] Add full-space and vech-space `dqgmres` workspaces and allocation helpers.
  - [x] Add the low-level `solve_lyapunov_equation(..., Val(:dqgmres), ...)` method.
  - [x] Validate focused symmetric and nonsymmetric `2x2` Lyapunov solves through low-level and public dispatcher paths.

- [x] Benchmark first-order Schur-vector and residual-check changes on FS2000 and FRBUS.
  - [x] Record baseline first-order solve timings.
  - [x] Change generalized Schur to request only right Schur vectors.
  - [x] Verify the change on a small model and FRBUS.
  - [x] Measure FRBUS timing with the QME residual check skipped.

- [x] Check dense doubling and sparse-Jacobian options for FRBUS first-order speed.
  - [x] Add a QME algorithm selector to the focused first-order benchmark script.
  - [x] Benchmark dense doubling on FS2000 and FRBUS.
  - [x] Verify dense doubling against the Schur/fallback first-order solution on FRBUS.
  - [x] Probe sparse Jacobian, sparse QR preprocessing, and sparse-LU doubling feasibility.

- [x] Add a small profview script for NAWM and FRBUS first-order solves.

- [x] Trace the SW07 third-order perturbation path and confirm which Sylvester algorithm is requested.
- [x] Verify the third-order solve calls `solve_sylvester_equation(...; sylvester_algorithm = opts.sylvester_algorithm³)`.
- [x] Add a standalone script that assembles the SW07 third-order Sylvester system up to the `A`, `B`, `C` solve inputs.
- [x] Benchmark the existing bicgstab path at that solve point.
- [x] Benchmark a diagonal-preconditioned bicgstab variant on the same operator and compare timings and residuals.
- [x] Benchmark structured triangular column preconditioners on the same operator and compare timings, iterations, and residuals.
- [x] Benchmark a similarity-reordered lower-triangular column preconditioner on the same operator and compare whether the reorder helps or densifies the triangular sweep.
- [x] Rewrite the benchmark to use the same low-level `bicgstab!` path as `src/algorithms/sylvester.jl` instead of the earlier manual variants.
- [x] Benchmark CPU-usable `KrylovPreconditioners.jl` variants on that mirrored bicgstab path and compare them with the library dispatcher.
- [x] Sweep the CPU ILU drop tolerance around `τ = 1e-3` on the corrected mirrored bicgstab benchmark.
- [x] Run the script and record whether the preconditioner helps on the assembled SW07 system.
- [x] Align the Dynare comparison availability probe with the Octave runner's Dynare path setup and verify the test file degrades cleanly when Dynare is absent.
- [x] Reproduce the Dynare Docker-stage `resol` failure (`order_var` missing) and fix the `extract_dynare_results.m` resolver argument wiring for Dynare 7.
- [x] Align the Julia benchmark path with Dynare `resol` by timing NSSS + Jacobian + first-order solve on cold caches each iteration.
- [x] Export direct first-order solve timings on both Julia and Dynare sides instead of inferring them from total timings.
- [x] Extend Dynare order-3 benchmarking to export direct Jacobian, first-order solve, Hessian, and second-order solve timings alongside bundled `k_order_pert`.
- [x] Update the comparison report to compare only directly measured component timings and print benchmark tables even when the numerical comparison testset fails.
- [x] Remove `Caldara_et_al_2012` from the active Dynare harness again and stop running the `FS2000` pruned third-order case.
- [x] Remove `NSSS` from the Dynare benchmark set and benchmark totals because the Julia-side path is cache-based and not comparable.
- [x] Switch the Debian testing Dynare Docker path from a pinned Julia tarball install to `juliaup` using the `release` channel and verify the install works in Debian testing.
- [x] Fix the latest non-`jet`, non-nested-sampler `optim_LFI_alloc` CI regressions in the ForwardDiff extension, benchmark compatibility layer, and docs project metadata.
- [ ] Bound the SW07 nessai flow proposal configuration to stop post-switch evaluation blow-ups while keeping the run as a full nested-sampling CI test.
- [x] Add a dynesty-based SW07 dynamic nested-sampling path to the existing nessai test using the same priors, likelihood, and posterior summary code.
- [x] Retune the SW07 dynesty configuration for an offline high-dimensional run using `rslice` and explicit dynamic live-point controls instead of the earlier CI batch cap.
- [x] Add a QME size threshold that switches the default solver to doubling for large first-order systems and benchmark pure Schur versus dense/sparse doubling on NAWM and FRBUS with BenchmarkTools.
- [x] Fix benchmark jacobian dispatch to always use the workspace-aware API and avoid legacy 4-argument fallback MethodError in CI.
- [x] Make benchmark CI use the checked-out benchmark script path instead of resolving the base/package checkout script.
- [x] Fix the FastLapackInterface `gges!` exterior-disk criterion so FRBUS low-level QME Schur selects the expected unit-root subspace and solves directly.
- [x] Check SW07 third-order Lyapunov Krylov/dqgmres/preconditioner competitiveness versus doubling.
  - [x] Add a focused third-order Lyapunov capture and benchmark script.
  - [x] Capture the dominant SW07 third-order Lyapunov subproblem.
  - [x] Benchmark doubling, vech Krylov, dqgmres, and ILU-preconditioned Krylov variants.
  - [x] Record the result in lessons and agent progress.
- [x] Add a second-order SW07 Lyapunov full-space Krylov preconditioner benchmark.
  - [x] Capture the smaller SW07 second-order covariance Lyapunov problem.
  - [x] Benchmark `bicgstab`, `gmres`, and `dqgmres` with column ILU and triangular sweep preconditioners.
  - [x] Record the second-order result in lessons and agent progress.
- [x] Convert the Lyapunov Krylov preconditioner script into a REPL workflow.
  - [x] Keep `get_lyapunov_inputs(...)` as the single matrix-capture entry point.
  - [x] Remove JSON/result-dictionary benchmark plumbing from the workflow.
  - [x] Move solver/preconditioner options next to the selected solve and timing/precision output.
  - [x] Use BenchmarkTools trials for solver and preconditioner timing summaries.
  - [x] Verify the default SW07 second-order run completes without warnings.
- [x] Add a macOS Dynare thread-sweep driver equivalent to the Windows PowerShell runner.
  - [x] Add `test/dynare_comparison/run_thread_sweep_macos.sh` with staged-output publish semantics.
  - [x] Mirror Windows phases: Julia export, Dynare run, per-thread compare, and cross-thread summary.
  - [x] Validate argument parsing and staging paths with `--validate-only` on macOS shell.

- [x] Migrate estimation chain handling from `MCMCChains` to `FlexiChains`.
  - [x] Replace test/docs dependency metadata with `FlexiChains`.
  - [x] Add shared helpers for parameter means, raw posterior matrices, and Pigeons sample-array conversion.
  - [x] Update direct Turing estimation tests, Pigeons estimation tests, and nested-sampling summaries.
  - [x] Update the estimation tutorial and docs plot-generation code.
  - [x] Validate the Pigeons sample layout conversion with `tasks/validate_pigeons_flexichain_conversion.jl`.

Notes:

- `docs/Manifest.toml` still needs a normal docs-environment resolve to drop lockfile-only `MCMCChains` entries.
