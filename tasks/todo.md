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
- [ ] Re-run the targeted GitHub CI jobs after pushing the repair commit.

The full test suite was not run. A large third-order Gali OBC probe was stopped during its expensive solve after the second-order compressed transition assertion passed; see `AGENT_PROGRESS.md` for the focused evidence.

## Performance follow-up

- [x] Benchmark compressed versus pre-compressed full-Kronecker user-facing and hot transition paths.
- [x] Apply same-vector cubic and analytical VJP allocation optimizations.
- [x] Prototype and assess thread-local particle propagation/scoring.
- [ ] Add opt-in production particle parallelism after settling RNG, dense-measurement-error, and API semantics.
