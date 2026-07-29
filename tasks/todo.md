# Compressed higher-order state updates

- [x] Add compressed pair/triple vector kernels and focused tests.
- [x] Migrate core state-update and stochastic-steady-state paths.
- [x] Migrate filtering, inversion, OBC, IRF, and filter-free paths.
- [x] Migrate higher-order rrules and analytical VJPs.
- [x] Audit remaining full conversions and run focused verification.
- [x] Add explicit power kernels/VJPs for repeated-input contractions.
- [x] Precompute compressed pair and cubic row maps used by inversion pullbacks.
- [x] Verify power kernels, cached row maps, static hot-path audit, and inversion likelihood levels.
- [x] Repair CI regressions in particle-filter dispatch, third-order inversion row bindings, and global compressed selector maps.
- [x] Cache every invariant pair/triple compressed selector used by inversion, `find_shocks`, and higher-order rrules; remove runtime map construction.
- [ ] Re-run the targeted GitHub CI jobs after pushing the repair commit.

The full test suite was not run. A large third-order Gali OBC probe was stopped during its expensive solve after the second-order compressed transition assertion passed; see `AGENT_PROGRESS.md` for the focused evidence.

## Performance follow-up

- [x] Benchmark compressed versus pre-compressed full-Kronecker user-facing and hot transition paths.
- [x] Apply same-vector cubic and analytical VJP allocation optimizations.
- [x] Prototype and assess thread-local particle propagation/scoring.
- [ ] Add opt-in production particle parallelism after settling RNG, dense-measurement-error, and API semantics.

## Review follow-up

- [x] Restore symmetric Aumann–Shapley derivative multiplicities in compressed coordinates and cover them with full-reference tests.
- [x] Reuse preallocated compressed state-to-pair and state-pair-to-shock matrices in inversion and reverse-mode paths.
- [x] Restrict stochastic-steady-state Newton inputs to the required past/mixed rows and state/constant compressed columns.
- [x] Verify focused kernels, inversion likelihoods, and user-facing stochastic steady states.
- [ ] Add cached sparse CSC selector patterns only if a follow-up benchmark shows they beat dense workspace reuse in the full filter workload.
