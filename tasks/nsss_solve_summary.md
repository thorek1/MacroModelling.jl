**Main Branch NSSS_solve Logic**
- RTGF `solve_SS` converts `initial_parameters` to `Vector{Float64}`, finds the closest cached parameter vector by squared distance, and uses that cache entry as the warm start.
- Continuation scaling loop (`scale` in $[0,1]$) runs up to 500 iterations (or 1 for cold start). On success it advances `scale` toward 1.0; there is no explicit failure-branch scale update, so failure leaves `scale` unchanged and the loop continues until the cap.
- Before solving, parameter bounds are enforced by injecting clamped assignments for bounded parameters; calibration-no-var equations are evaluated after that.
- Per block, `SS_solve_func` builds `params_and_solved_vars`, constructs `lbs`/`ubs` (infinite bounds are mapped to +/-1e12 with random jitter), clamps cache-based initial guesses into bounds, and calls `block_solver`.
- Error accumulation is `solution_error += solution[2][1]` across blocks; total iterations accumulate from `solution[2][2]`. A min/max validation expression list (if any) is appended to `SS_solve_func` and contributes to `solution_error`.
- Caches: `NSSS_solver_cache_tmp` stores `(sol, params_and_solved_vars)` per block, then appends the final parameter vector. If `solution_error < tol` and `current_best > 1e-8` (and `scale == 1`), the cache is pushed to the global cache.
- If the selected cache entry has `Inf` in the block-parameter slot, the solver zeros the cached guesses before solving (cold-start fallback).

**Current Branch Logic**
- The RTGF `NSSS_solve` is replaced; the solver path is `solve_nsss_wrapper` plus `solve_nsss_steps`, operating on precompiled step objects.
- `solve_nsss_steps` builds `params_vec` via `param_prep!` (bounded parameters + calibration-no-var), initializes `sol_vec`, and executes each step in order.
- Analytical steps compute optional aux variables (domain safety), then evaluate target expressions. Bounds behavior splits:
  - Plus vars: clamp to bounds, add `abs(clamped - raw)` to error, and store the clamped value.
  - User-bounded vars: add the same error term but store the raw value.
- Numerical steps compute aux variables and optional aux-error checks (early exit on tolerance breach), gather `params_and_solved_vars`, clamp cached guesses into bounds, and call `block_solver`.
- Error accumulation: `solution_error` sums all step errors, while iterations and cache entries are recorded only for numerical steps. If `solution_error > tol`, step execution breaks and the solver returns zeros.
- Caches: per numerical block, `NSSS_solver_cache_tmp` stores `(sol, params_and_solved_vars)` and always appends the parameter vector at the end.
- Continuation: `solve_nsss_wrapper` maintains a local CircularBuffer of intermediate caches, interpolates parameters from `closest_solution_init`, and updates `scale`. On failure it explicitly backs off `scale` with `scale = scale * 0.3 + solved_scale * 0.7`. On success, it pushes intermediates to the local cache and updates the global cache only when `scale == 1` and the nearest-cache distance exceeds `1e-8`.

**Key Differences**
- RTGF vs step-based execution: main branch uses a monolithic RTGF with inline block solves; the branch uses explicit step objects (`AnalyticalNSSSStep`, `NumericalNSSSStep`) orchestrated in Julia.
- Failure handling in continuation: main branch has no explicit failure-branch scale update; the branch explicitly reduces `scale` toward `solved_scale` on failure.
- Error sources: main branch aggregates numerical block errors (plus any min/max validation expressions); the branch includes analytical-step bound and domain-safety errors in the total and can early-exit on aux errors.
- Bounds application: main branch clamps bounded parameters via injected assignments and clamps initial guesses via `lbs/ubs`; the branch pushes parameter bounds into `param_prep!` and handles per-variable bounds inside analytical steps (with different clamp/write behavior for plus vs user bounds).
- Cache selection and warm starts: main branch can zero cached guesses when the stored cache is infinite; the branch does not zero caches but instead clamps cached guesses to bounds and uses a local scale cache for intermediate solutions.
- Output assembly: main branch returns a fixed ordered list of variables and calibration parameters; the branch builds `SS_and_pars` via `output_indices` from the solution vector.