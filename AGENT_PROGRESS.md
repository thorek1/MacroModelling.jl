# AGENT_PROGRESS

## Current task
- Applied linked-list row index to mat_mult_kron forward-pass functions (solution.jl)

## Completed
- Replaced `A[row,:]` CSC row scan in 4-arg `mat_mult_kron` forward pass with linked-list row index — 1.43x speedup on SW07
- Replaced `A[row,:]` + `rv |> unique` in 3-arg `mat_mult_kron` forward pass with linked-list row index — 2.35x speedup on SW07
- Both produce numerically identical results (max error = 0.0)
- E2E verified: get_irf(SW07, algorithm=:third_order) succeeds
- Replaced `sparse(A')` transpose in `sparse_ABAt()` (moments.jl) with linked-list row index — 1.71x speedup on SW07
- Replaced `Dict{Int,Vector{Int}}` row grouping in `mat_mult_kron` pullback (rrules.jl) with linked-list — 1.20x speedup on SW07
- Both changes produce numerically identical results (max error < 1e-10)
- Integration tests pass: get_irf(:third_order), get_moments(:pruned_third_order), sparse_ABAt vs dense

### Previous: ILU preconditioner refactor
- Removed KrylovPreconditioners.jl dependency; inlined only needed ILU subset into codebase
- Moved preconditioner code to src/algorithms/preconditioner.jl (self-contained)
- Simplified: removed dead ILUSortedSet type, folded ILULinkedLists into ILURowReader, removed unused fields
- Replaced all direct CSC field access with SparseArrays API (rowvals, nonzeros, getcolptr, nzrange)
- Consolidated code: eliminated ILURowReader struct + 9 micro-functions, replaced ILUAccumulator with minimal SparseAccum (no methods). Reduced from 3 structs + 18 functions (291 lines) to 2 structs + 8 functions (264 lines). All logic inlined directly in ilu() with plain array operations.
- Added rowvals, nzrange, getcolptr to SparseArrays imports in MacroModelling.jl
- All unit tests pass (tasks/test_preconditioner.jl): ILU factorization, ldiv! accuracy, build_ilu_preconditioner
- Integration test passes: Smets_Wouters_2007 model IRFs computed correctly

### Previous: Lyapunov caching
- Added `covariance_first_order` and `covariance_second_order` cache fields to `valid_for_caches` and `caches` structs (structures.jl)
- Updated caches constructor in parser/macros.jl
- Added cache invalidation in MacroModelling.jl (reset on model reparse)
- Added `CACHE_VALIDITY_FIELDS` entries for new cache fields
- Implemented cache check + store in `calculate_covariance` (moments.jl): exact cache hit on parameter match, warm start via initial_guess on miss
- Implemented cache check + store in `calculate_second_order_moments_with_covariance` (moments.jl): same pattern for 2nd-order Lyapunov
- Wired up initial_guess from cache in rrule for `calculate_covariance` (rrules.jl) + stores result after solve
- Wired up initial_guess from cache in rrule for `calculate_second_order_moments_with_covariance` (rrules.jl) + stores result after solve
- All tests pass (tasks/test_lyapunov_cache.jl): cache hit, invalidation, numerical correctness

## Previous work
- Confirmed the benchmark script only imported solve_lyapunov_equation when Lyapunov_workspace existed, which breaks older benchmarked tags.
- Patched the benchmark script to import solve_lyapunov_equation unconditionally and keep Lyapunov_workspace conditional.
- Patched both benchmark workflows to pass --add="MatrixEquations" to benchpkg so the temp benchmark environment includes the weak dependency needed by the MatrixEquations extension.
- Confirmed the benchmark Jacobian shim could fall through to a removed 3-argument calculate_jacobian overload in newer package revisions.
- Patched the benchmark Jacobian and first-order dispatch shims to use concrete struct-field compatibility checks instead of fragile hasproperty/applicable gating.
- Verified file diagnostics are clean.
- Verified solve_lyapunov_equation is importable in a focused Julia check.
- Verified both workflow files contain the MatrixEquations benchpkg add flag.

## Remaining
- Re-run the benchmark workflow in CI to confirm the Jacobian dispatch fix resolves the pull-request benchmark job end-to-end.
