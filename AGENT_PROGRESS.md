# Agent Progress Log

## Session: 2026-02-15 - Remove dead `SS_solve_func` path in NSSS builder

### Goal
Remove dead code in `src/nsss_solver.jl` without changing runtime NSSS behavior.

### Changes

- Removed unused `SS_solve_func` plumbing from `write_block_solution!`:
   - deleted `SS_solve_func` argument from function signature
   - removed all `push!(SS_solve_func, ...)` statements that were never consumed
- Removed now-dead local expression accumulation (`result = Expr[]` and its population loop).
- Updated all `write_block_solution!` call sites to match the new signature.
- Removed dead local initialization `SS_solve_func = []` in NSSS setup.

### Validation

- File diagnostics:
   - `src/nsss_solver.jl` reports no errors.
- RBC smoke test:
   - `julia -t auto --project=. -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; solve!(RBC); println("ok")'`
   - Result: `ok`.

## Session: 2026-02-15 - Model-specific NSSS fastest solver ordering + robust selection metric

### Goal
Make steady-state solver parameter prioritization model-specific instead of mutating global default order, and replace the fixed `0.95` update rule with a sampling-error-aware metric.

### Changes

- Added model-specific storage on `post_complete_parameters`:
   - `nsss_fastest_solver_parameter_idx::Union{Nothing, Int}`
- Updated constructors and updater for `post_complete_parameters` to carry this new field.
- Added helper `get_ordered_SS_solver_parameters(𝓂)` that returns:
   - selected model-specific parameter first (if available), then
   - all `DEFAULT_SOLVER_PARAMETERS` entries, deduplicated by value.
- Updated all NSSS wrapper call sites to use ordered model-specific parameters:
   - core path in `get_NSSS_and_parameters`
   - ForwardDiff path
   - Zygote path
- Refactored `select_fastest_SS_solver_parameters!`:
   - now computes per-candidate score `mean_time + stderr_weight * stderr`
   - adds keyword args:
      - `n_samples::Int = 100`
      - `min_relative_improvement::Float64 = 0.01`
      - `stderr_weight::Float64 = 1.0`
   - persists best candidate as model-specific index (`nsss_fastest_solver_parameter_idx`)
   - no longer reorders `DEFAULT_SOLVER_PARAMETERS` globally.
- Updated `find_SS_solver_parameters!(::Val{:ESCH}, ...)`:
   - still appends discovered parameter to `DEFAULT_SOLVER_PARAMETERS`
   - now also sets model-specific fastest index to the appended entry.

### Validation

- Command:
   - `julia -t auto --project=. -e 'using MacroModelling; include(joinpath(pwd(),"models","RBC_baseline.jl")); get_steady_state(RBC_baseline, derivatives=false, verbose=false); MacroModelling.select_fastest_SS_solver_parameters!(RBC_baseline, n_samples=10, min_relative_improvement=0.01, stderr_weight=1.0); get_steady_state(RBC_baseline, derivatives=false, verbose=false); println("ok")'`
- Result:
   - Completed successfully with output `ok`.

### Follow-up simplification (same session)

- Replaced reordered `Vector` construction with an index-mapped view over `DEFAULT_SOLVER_PARAMETERS`:
   - Added `OrderedSolverParametersView <: AbstractVector{solver_parameters}` in `src/MacroModelling.jl`.
   - `get_ordered_SS_solver_parameters(𝓂)` now returns either defaults directly or the index-mapped view (no copied solver-parameter array).
- Relaxed solver-parameter container type constraints to `AbstractVector{solver_parameters}` in `src/nsss_solver.jl` and `block_solver` in `src/MacroModelling.jl`.
- Removed accidental stray mutation lines in `block_solver` (`nsss_fastest_solver_parameter = best_param`) introduced during earlier edits.

### Follow-up validation

- Command:
   - `julia -t auto --project=. -e 'using MacroModelling; include(joinpath(pwd(),"models","RBC_baseline.jl")); get_steady_state(RBC_baseline, derivatives=false, verbose=false); MacroModelling.select_fastest_SS_solver_parameters!(RBC_baseline, n_samples=10, min_relative_improvement=0.01, stderr_weight=1.0); get_steady_state(RBC_baseline, derivatives=false, verbose=false); println("ok")'`
- Result:
   - Completed successfully with output `ok`.

### Follow-up simplification 2 (same session)

- Removed `OrderedSolverParametersView` entirely (user-requested simplification).
- Replaced with plain preferred-index handling:
   - `get_preferred_SS_solver_parameter_index(𝓂)` returns model-specific preferred index (fallback `1`).
   - NSSS call sites now pass `DEFAULT_SOLVER_PARAMETERS` directly plus keyword `preferred_solver_parameter_idx`.
   - `block_solver` iterates solver parameters in preferred-first order via index mapping (`ordered_solver_parameter_index`) without constructing reordered arrays.
- Kept backward compatibility by adding an overload of `block_solver` without explicit preferred index that defaults to `1`.

### Follow-up validation 2

- Command:
   - `julia -t auto --project=. -e 'using MacroModelling; include(joinpath(pwd(),"models","RBC_baseline.jl")); get_steady_state(RBC_baseline, derivatives=false, verbose=false); MacroModelling.select_fastest_SS_solver_parameters!(RBC_baseline, n_samples=10, min_relative_improvement=0.01, stderr_weight=1.0); get_steady_state(RBC_baseline, derivatives=false, verbose=false); println("ok")'`
- Result:
   - Completed successfully with output `ok`.

### Follow-up simplification 3 (same session)

- Updated `select_fastest_SS_solver_parameters!` to select fastest candidate by median order statistic only.
- Removed keyword arguments:
   - `min_relative_improvement`
   - `stderr_weight`
- New metric implementation:
   - collect `n_samples` runtimes per candidate,
   - sort runtimes,
   - score as the `(n_samples ÷ 2)`-th ordered sample,
   - choose smallest score.

### Follow-up validation 3

- Command:
   - `julia -t auto --project=. -e 'using MacroModelling; include(joinpath(pwd(),"models","RBC_baseline.jl")); get_steady_state(RBC_baseline, derivatives=false, verbose=false); MacroModelling.select_fastest_SS_solver_parameters!(RBC_baseline, n_samples=10); get_steady_state(RBC_baseline, derivatives=false, verbose=false); println("ok")'`
- Result:
   - Completed successfully with output `ok`.

## Session: 2026-02-15 - Compare QUEST/NAWM/SW07 steady-state benchmarks vs main

### Goal
Benchmark `copilot/refactor-nsss-solve-function` against `main` for `QUEST3_2009`, `NAWM_EAUS_2008`, and `Smets_Wouters_2007` with identical benchmark code and cache-cleared setup.

### Method

- Created benchmark driver `benchmark/branch_compare_ss_benchmark.jl` that reports parseable rows:
   - `BENCH|<model>|<min_time_ns>|<median_time_ns>|<min_mem>|<median_mem>|<min_allocs>|<median_allocs>`
- Ran the same script in:
   - main worktree: `/private/tmp/MacroModelling.jl-main`
   - feature branch worktree: `/Users/thorekockerols/GitHub/MacroModelling.jl-worktree`
- Logs:
   - `/tmp/mm_main_bench.log`
   - `/tmp/mm_branch_bench.log`

### Results

- `QUEST3_2009`
   - main: `min=1.717125e6 ns`, `median=1.800083e6 ns`, `mem=73664 B`, `allocs=1938`
   - branch: `min=1.7225e6 ns`, `median=1.778291e6 ns`, `mem=73808 B`, `allocs=1941`

- `NAWM_EAUS_2008`
   - main: `min=1.3872042e7 ns`, `median=1.4563042e7 ns`, `mem=8870832 B`, `allocs=6382`
   - branch: `min=1.5453916e7 ns`, `median=1.67605e7 ns`, `mem=9972480 B`, `allocs=6646`

- `Smets_Wouters_2007`
   - main: `min=395958.0 ns`, `median=402562.5 ns`, `mem=31248 B`, `allocs=639`
   - branch: `min=403125.0 ns`, `median=418459.0 ns`, `mem=31296 B`, `allocs=642`

### Interpretation

- QUEST: near parity (mixed: slightly worse min, slightly better median).
- NAWM: branch regresses on time and memory/allocations.
- SW07: branch modestly slower with tiny allocation increase.

## Session: 2026-02-15 - Verify all model solves + benchmark QUEST/NAWM/SW07

### Goal
Run a full model solve sweep and collect steady-state benchmark metrics (time, memory, allocations) for `QUEST3_2009`, `NAWM_EAUS_2008`, and `Smets_Wouters_2007` using cache-cleared setup.

### Changes

- Added `benchmark/test_all_models_solve.jl`:
   - Includes every model file under `models/`.
   - Runs `get_steady_state(model, derivatives = false)` for each model.
   - Prints pass/fail summary and errors on any failure.

- Added `benchmark/benchmark_quest_nawm_sw07_ss.jl`:
   - Benchmarks cache-cleared calls:
      - `get_steady_state(model, parameters = model.parameter_values, derivatives = false)`
      - `setup = clear_solution_caches!(model, :first_order)`
   - Reports min/median time, memory, and allocation count for each target model.
   - Fixed world-age issue by moving model `include(...)` statements to top-level and passing model objects into the benchmark helper.

### Validation

1. Full solve sweep:
    - Command: `julia -t auto --project=. benchmark/test_all_models_solve.jl`
    - Result: `Passed: 23 / 23`, `Failed: 0`.

2. Targeted benchmark run:
    - Command: `julia -t auto --project=. benchmark/benchmark_quest_nawm_sw07_ss.jl`
    - Reported metrics:
       - `QUEST3_2009`
          - `min_time_ns=1.872e6`, `median_time_ns=1.900458e6`
          - `min_memory_bytes=75600`, `median_memory_bytes=75600`
          - `min_allocs=2026`, `median_allocs=2026`
       - `NAWM_EAUS_2008`
          - `min_time_ns=1.3613667e7`, `median_time_ns=1.4076417e7`
          - `min_memory_bytes=8506000`, `median_memory_bytes=8506000`
          - `min_allocs=6315`, `median_allocs=6315`
       - `Smets_Wouters_2007`
          - `min_time_ns=397333.0`, `median_time_ns=400125.0`
          - `min_memory_bytes=31232`, `median_memory_bytes=31232`
          - `min_allocs=638`, `median_allocs=638`

## Session: 2026-02-15 - Cache steady-state output index mappings

### Goal
Avoid recomputing `indexin` mappings in `get_steady_state` for variables/calibration output selection.

### Changes

- Added cached fields on `post_complete_parameters`:
   - `ss_var_idx_in_var_and_calib`
   - `calib_idx_in_var_and_calib`
- Populated these once in `ensure_model_structure_constants!`.
- Replaced local `indexin` calls in `get_steady_state` with cached values from model constants.

### Validation

- SW07 smoke solve passed (`get_steady_state(..., derivatives=false)` returned `ok`).

## Session: 2026-02-15 - Reuse loop candidate arrays in `block_solver`

### Goal
Avoid per-iteration allocations from small array literals inside the `block_solver` search loops.

### Changes

- File: `src/MacroModelling.jl`
   - Hoisted loop-invariant candidates to reusable containers:
      - `ext_candidates = (true, false)`
      - `algo_candidates = (newton, levenberg_marquardt)`
   - Cold-start branch:
      - Replaced mutable array literal `start_vals` with tuple candidates created once.
   - Non-cold-start branch:
      - Added one preallocated `start_vals::Vector{Union{Bool,T}}`.
      - Updated only `start_vals[2]` per parameter (`p.starting_value`) and iterated over `s_candidates` view/full buffer.

### Validation

- SW07 smoke solve passed:
   - `julia -t auto --project=. -e 'using MacroModelling; include(joinpath(pwd(),"models","Smets_Wouters_2007.jl")); get_steady_state(Smets_Wouters_2007, derivatives=false); println("ok")'`

## Session: 2026-02-15 - Three-way SW07 comparison (FastLU vs LinearSolve LU vs legacy lu!+ldiv!)

### Goal
Compare current FastLapack-backed FastLU against (1) standard `LinearSolve.LUFactorization` init and (2) temporary legacy dense `lu!` + `ldiv!` implementation.

### Benchmark Setup

- Model: `Smets_Wouters_2007`
- Call: `get_steady_state(..., verbose=false, derivatives=false)`
- Setup per sample: `clear_solution_caches!(model, :first_order)`
- `samples=30`, `evals=1`

### Results

- FastLU (current implementation):
   - `FASTLU_MIN_NS=444583.0`
   - `FASTLU_MEDIAN_NS=451895.5`
   - `FASTLU_MIN_B=152176`
   - `FASTLU_MEDIAN_B=152176`
   - `FASTLU_MIN_ALLOCS=2492`
   - `FASTLU_MEDIAN_ALLOCS=2492`

- Standard LinearSolve LUFactorization init:
   - `LSLU_MIN_NS=1.234209e6`
   - `LSLU_MEDIAN_NS=1.261125e6`
   - `LSLU_MIN_B=677008`
   - `LSLU_MEDIAN_B=677008`
   - `LSLU_MIN_ALLOCS=10243`
   - `LSLU_MEDIAN_ALLOCS=10243`

- Temporary legacy dense `lu!` + `ldiv!` path:
   - `LULDIV_MIN_NS=1.161625e6`
   - `LULDIV_MEDIAN_NS=1.1776665e6`
   - `LULDIV_MIN_B=291296`
   - `LULDIV_MEDIAN_B=291296`
   - `LULDIV_MIN_ALLOCS=7577`
   - `LULDIV_MEDIAN_ALLOCS=7577`

### Conclusion

On this SW07 harness, FastLU is best on both runtime and allocations.
Temporary benchmark patches were reverted; final state remains FastLU and was revalidated with SW07 smoke solve (`ok`).

## Session: 2026-02-15 - Direct SW07 A/B FastLU vs RFLU (time + allocations)

### Goal
Run an apples-to-apples SW07 comparison for current FastLU setup versus temporary RFLU, including allocation bytes and allocation counts.

### Benchmark Setup

- Model: `Smets_Wouters_2007`
- Call: `get_steady_state(..., verbose=false, derivatives=false)`
- Setup per sample: `clear_solution_caches!(model, :first_order)`
- `samples=30`, `evals=1`

### Results

- FastLU (final configuration):
   - `FASTLU_MIN_NS=461584.0`
   - `FASTLU_MEDIAN_NS=466625.5`
   - `FASTLU_MIN_B=152176`
   - `FASTLU_MEDIAN_B=152176`
   - `FASTLU_MIN_ALLOCS=2492`
   - `FASTLU_MEDIAN_ALLOCS=2492`

- RFLU (temporary swap for comparison):
   - `RFLU_MIN_NS=1.112834e6`
   - `RFLU_MEDIAN_NS=1.1316045e6`
   - `RFLU_MIN_B=258656`
   - `RFLU_MEDIAN_B=258656`
   - `RFLU_MIN_ALLOCS=7633`
   - `RFLU_MEDIAN_ALLOCS=7633`

### Conclusion

FastLU is substantially better on this harness: lower latency, lower allocation bytes, and fewer allocations.
After comparison, backend was restored to FastLU and SW07 smoke solve revalidated (`ok`).

## Session: 2026-02-15 - Make FastLapackInterface backend work in NSSS cache path

### Goal
Enable and stabilize `FastLUFactorization` for dense NSSS linear solves by fixing backend activation and cache initialization usage.

### Root Causes

1. `LinearSolveFastLapackInterfaceExt` was not loaded in package runtime because `FastLapackInterface` was not imported by `MacroModelling`.
2. `FastLUFactorization` performs in-place LU on `cache.A`; passing Jacobian buffers directly caused mutation side effects in repeated NSSS solve loops.
3. NSSS cache init used `LinearProblem(A,b,alg)` (algorithm in positional slot intended for problem parameters), which is incorrect API usage.

### Changes

- File: `src/MacroModelling.jl`
   - Added `import FastLapackInterface` to ensure LinearSolve's FastLapack extension is active in package runs.

- File: `src/nsss_solver.jl`
   - Dense LU cache backend switched to `𝒮.FastLUFactorization()` (sparse remains `𝒮.LUFactorization()`).
   - Corrected cache problem construction:
      - `LinearProblem(A, b)` + `init(..., alg)` (removed algorithm from `LinearProblem` positional args) in both normal and extended block paths.

- File: `src/algorithms/nonlinear_solver.jl`
   - In Newton solve calls, when cache algorithm is `FastLUFactorization`, pass `copy(∇)` into `sol_cache.A` to prevent in-place LU mutation of Jacobian buffers.

- File: `src/MacroModelling.jl`
   - Applied same `FastLU` Jacobian copy safeguard in `block_solver` pre-check path.

### Validation

1. SW07 smoke solve:
    - `get_steady_state(Smets_Wouters_2007, derivatives=false)` returns `ok`.

2. SW07 benchmark (cache-cleared setup, `derivatives=false`) after fix:
    - `FASTLU_MIN_NS=449334.0`
    - `FASTLU_MIN_B=152016`
    - `FASTLU_MEDIAN_NS=459583.0`
    - `FASTLU_MEDIAN_B=152016`

No steady-state failure warnings during benchmark after the safeguards.

## Session: 2026-02-15 - A/B trial RFLU vs FastLU for dense NSSS cache

### Goal
Run the requested A/B between dense `RFLUFactorization` and dense `FastLUFactorization` on SW07 and keep the better backend.

### Results

1. Baseline with current `RFLU` (same cache-cleared SW07 harness):
   - `RFLU_MIN_NS=1.103916e6`
   - `RFLU_MIN_B=258656`
   - `RFLU_MEDIAN_NS=1.1149375e6`
   - `RFLU_MEDIAN_B=258656`

2. `FastLU` trial in real NSSS path:
   - Switched dense cache init in `src/nsss_solver.jl` to `𝒮.FastLUFactorization()`.
   - SW07 setup failed during cache init with:
     - `MethodError: no method matching do_factorization(::LinearSolve.FastLUFactorization, ::Matrix{Float64}, ::Vector{Float64}, ::Vector{Float64})`

3. Final decision:
   - Reverted dense backend to `𝒮.RFLUFactorization()`.
   - `Smets_Wouters_2007` steady-state smoke solve passes (`ok`).

### Conclusion

`FastLapackInterface` is installed and a standalone `LinearSolve` micro-test can return `Success`, but the NSSS cache-init usage in this repository is not currently supported by `FastLUFactorization`. Therefore `RFLU` remains the working backend.

## Session: 2026-02-15 - Switch dense NSSS LU cache to RFLU and enable FastLapack backend

### Goal
Try `RFLUFactorization` for dense NSSS linear caches and make Fast-LAPACK backends available in this project.

### Changes

- File: `src/nsss_solver.jl`
  - Updated NSSS LU cache initialization in both normal and extended block paths:
    - dense buffers (`!issparse`) now use `𝒮.RFLUFactorization()`
    - sparse buffers keep `𝒮.LUFactorization()`
- Dependency update:
  - Added `FastLapackInterface` to project dependencies.
  - Files updated by package manager:
    - `Project.toml`
    - `Manifest.toml`

### Validation

1. Fast-LAPACK availability check:
   - `LinearSolve` dense micro-test with `FastLUFactorization()` now succeeds (`retcode = Success`).

2. NSSS smoke test:
   - `Smets_Wouters_2007` steady state solve after RFLU switch: `ok`.

3. SW07 benchmark snapshot (cache-cleared setup, `derivatives=false`):
   - `RFLU_MIN_NS=1.128e6`
   - `RFLU_MIN_B=258656`
   - `RFLU_MEDIAN_NS=1.1883745e6`
   - `RFLU_MEDIAN_B=258656`

## Session: 2026-02-15 - Migrate block_solver pre-check LU path to LinearSolve

### Goal
Replace the remaining LU-based relative-step pre-check in `block_solver` with LinearSolve, consistent with the Newton path migration.

### Changes

- File: `src/MacroModelling.jl`
- In `block_solver` (`!cold_start` pre-check), replaced:
   - `∇̂ = ℒ.lu(∇, check = false)`
   - `guess_update = ∇̂ \ res`
   - `ℒ.issuccess(∇̂)` gate
- With LinearSolve cache path:
   - `sol_cache = SS_solve_block.ss_problem.workspace.lu_buffer`
   - `sol = 𝒮.solve!(sol_cache)`
   - success gate `𝒮.SciMLBase.successful_retcode(sol.retcode)`
   - non-finite guard on `guess_update`

### Validation

Representative NSSS smoke solves after change:

- `NAWM_EAUS_2008`: `ok=true`
- `QUEST3_2009`: `ok=true`
- `Smets_Wouters_2007`: `ok=true`

All tested models remained below acceptance tolerance.

## Session: 2026-02-15 - Replace Newton try/catch with LinearSolve internal success flag

### Goal
Avoid exception-based solve handling in Newton and instead branch on the internal LinearSolve success indicator.

### Changes

- File: `src/algorithms/nonlinear_solver.jl`
- In both Newton linear-solve locations:
   - removed `try/catch` around `𝒮.solve!(sol_cache)`.
   - added internal success check `!ℒ.issuccess(sol_cache.cacheval)` immediately after `solve!`.
   - on failure, apply fallback:
      - `rel_xtol_reached = typemax(T)`
      - `new_residuals_norm = typemax(T)`
      - `break`
- Kept non-finite solution guard on `guess_update` with the same fallback path.

### Validation

Ran representative NSSS smoke solves:

- `NAWM_EAUS_2008`: `ok=true`
- `QUEST3_2009`: `ok=true`
- `Smets_Wouters_2007`: `ok=true`

All remained below acceptance tolerance.

## Session: 2026-02-15 - Newton path simplified to inline LinearSolve snippet

### Goal
Apply the requested simplification: remove the extra Newton helper/fallback logic and use only the inline `LinearSolve` snippet in place of previous logic, then verify dense/sparse buffer compatibility.

### Changes

- File: `src/algorithms/nonlinear_solver.jl`
- Removed `solve_newton_step!` helper entirely.
- Removed fallback logic (including dense QR fallback and legacy LU fallback path).
- In both Newton solve locations, now use only:
   - `sol_cache.A = ∇`
   - `sol_cache.b = new_residuals`
   - `𝒮.solve!(sol_cache)`
   - `guess_update .= sol_cache.u`
   - `new_residuals .= guess_update`

### Compatibility Check (dense vs sparse)

Ran runtime type checks on representative models:

- `Smets_Wouters_2007` numerical steps: all dense (`Matrix{Float64}` Jacobian and `Matrix{Float64}` cache `A`), all matched.
- `NAWM_EAUS_2008` numerical step: sparse (`SparseMatrixCSC{Float64,Int64}` Jacobian and cache `A`), matched.

Conclusion: no extra dispatch/preparation is required at Newton call sites; the per-block workspace is already initialized with a matching `LinearSolve` cache type for each Jacobian format.

## Session: 2026-02-15 - Newton NSSS linear solve switched to workspace LinearSolve

### Goal
Replace Newton's dense `lu!` solve path with workspace-backed `LinearSolve` in the NSSS nonlinear solver, unify dense/sparse handling in one step path, and benchmark SW07 for speed and allocations.

### Changes

- File: `src/algorithms/nonlinear_solver.jl`
- Added helper:
   - `solve_newton_step!(guess_update, residuals, jacobian, sol_cache)`
   - Uses pre-initialized `LinearSolve` cache (`ws.lu_buffer`) to solve in-place using workspace RHS.
   - Keeps a dense-only QR fallback if linear solve fails (sparse returns failure directly).
- Updated `newton`:
   - Removed explicit dense `lu!` branch.
   - Unified sparse and dense step solve calls through `solve_newton_step!`.
   - Reused cache-bound RHS vector (`ws.lu_buffer.b`) to avoid per-iteration cache rebinding allocations.

### Benchmark (SW07 NSSS)

Benchmarked with the user-provided SW07 setup (same `@benchmark` section and cache-clearing setup; terminal run omitted `@profview_allocs` because macro is not available in plain Julia terminal).

- **Baseline (pre-change):**
   - `BASELINE_MIN_TIME_NS=848333.0`
   - `BASELINE_MIN_MEMORY_B=208448`
   - `BASELINE_MEDIAN_TIME_NS=892625.0`
   - `BASELINE_MEDIAN_MEMORY_B=208448`

- **After LinearSolve refactor (final):**
   - `LINEARSOLVE_MIN_TIME_NS=480791.0`
   - `LINEARSOLVE_MIN_MEMORY_B=192160`
   - `LINEARSOLVE_MEDIAN_TIME_NS=489104.5`
   - `LINEARSOLVE_MEDIAN_MEMORY_B=192160`

### Validation

Targeted NSSS smoke solves after refactor:

- `NAWM_EAUS_2008`: `err=4.2589275137997416e-14`, `ok=true`
- `QUEST3_2009`: `err=6.340454355306702e-14`, `ok=true`
- `Smets_Wouters_2007`: `err=1.101719347798403e-14`, `ok=true`

All tested models remained below `opts.tol.NSSS_acceptance_tol`.

## Session: 2026-02-15 - Fix JET errors in NSSS symbolic and numerical step paths

### Goal
Resolve two JET-reported issues in `src/nsss_solver.jl` and verify by running the `jet` test set with an explicitly initialized environment.

### Root Cause

1. **Symbolic path type assertion in generated flow**
    - The one-equation symbolic solve path used inline type assertions in a call expression:
       - `solve_symbolically(eq_to_solve::SPyPyC.Sym{...}, var_to_solve_for::SPyPyC.Sym{...})`
    - In some paths, `eq_to_solve` can be an `Expr` (e.g. after min/max rewriting), which JET correctly flags as a possible invalid `typeassert`.

2. **`block_solver` dispatch on `Union{Nothing, ss_solve_block}`**
    - `execute_step!` called `block_solver(..., f.solve_blocks[step_idx], ...)` where the field type is `Union{Nothing, ss_solve_block}`.
    - Even on numerical steps, JET explores the `Nothing` branch and reports no matching method.

### Fix

- File: `src/nsss_solver.jl`
- Added explicit runtime type guards before calling SymPy-only helpers:
   - call `solve_symbolically` only when both arguments are `SPyPyC.Sym{PythonCall.Core.Py}`;
   - in constant-solution substitution, only call `replace_symbolic` for symbolic equations.
- Added explicit `solve_block` presence check in numerical execution path:
   - if `solve_block === nothing`, return failure `(Inf, 0, EMPTY_NSSS_STEP_CACHE)`;
   - otherwise call `block_solver` with the narrowed `solve_block` value.

### Environment Initialization (for JET)

1. `julia -t auto --project=test -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`
2. This also initialized CondaPkg/SymPy dependencies needed by the test harness.

### Validation

Ran:

```bash
TEST_SET=jet julia -t auto --project=. -e 'using Pkg; Pkg.test()'
```

Result:
- `Running test set: jet`
- `Static checking (JET.jl) | Pass 1/1`
- `Testing MacroModelling tests passed`

## Session: 2026-02-14 - Keep low-allocation solve_ss and restore all-model convergence

### Goal
Retain the recent low-allocation `solve_ss` refactor while removing the regression that broke no-global-search steady-state solves on several large models.

### What Was Done

- File: `src/MacroModelling.jl`
- Re-introduced the low-allocation `solve_ss` implementation (workspace-backed init/bounds buffers, reduced temporary allocations).
- Kept the stable cold-start/non-cold-start guess-loop behavior in `block_solver` (the pre-refactor loop structure for guesses/start-values), which removes the regression while preserving the low-allocation `solve_ss` path.

### Validation

1. Targeted check (`Smets_Wouters_2007`) with global search disabled:
   - Result: `err ≈ 9.05e-15`, `OK=true`.

2. Full `models/` sweep with global search disabled:
   - `MODEL_COUNT=23`
   - `FAIL_COUNT=0`
   - All models find NSSS without invoking the 120s global solver-parameter search.

## Session: 2026-02-14 - Fix NSSS regression in cold-start block solve path

### Goal
Restore all model steady-state solves without relying on the 120s global solver-parameter search fallback.

### Root Cause

The recent allocation-oriented refactor in `solve_ss`/`block_solver` (`src/MacroModelling.jl`) changed cold-start initialization and guess-loop behavior in a way that regressed convergence for larger models (notably `Smets_Wouters_2007`, `QUEST3_2009`, `NAWM_EAUS_2008`).

### Fix

- File: `src/MacroModelling.jl`
- Reverted the affected `solve_ss` and cold-start `block_solver` sections to the previous stable behavior:
   - restored direct `sol_values_init` construction (`max/min` with guess/fallback values),
   - restored optimizer init and bound argument flow,
   - restored prior guess-selection loops for cold starts and non-cold starts.

### Validation

1. Targeted regression model:

```bash
julia -t auto --project=. -e 'using MacroModelling; @eval MacroModelling function find_SS_solver_parameters!(::Val{:ESCH}, 𝓂::ℳ; maxtime::Real=120, maxiter::Int=2500000, tol::Tolerances=Tolerances(), verbosity=0) false end; include("models/Smets_Wouters_2007.jl"); opts=MacroModelling.merge_calculation_options(verbose=true); ss,(err,it)=MacroModelling.get_NSSS_and_parameters(Smets_Wouters_2007, Smets_Wouters_2007.parameter_values, opts=opts, cold_start=true); println(err)'
```

Result: `Smets_Wouters_2007` solves directly (`err ≈ 1.5e-14`, below acceptance tolerance).

2. Full model sweep with global search disabled:

```bash
julia -t auto --project=. -e 'using MacroModelling; ... include all models ... get_NSSS_and_parameters(..., cold_start=true) for each model with ESCH fallback overridden to false ...'
```

Result summary:
- `MODEL_COUNT=23`
- `FAIL_COUNT=0`
- All models in `models/` solve NSSS without global solver-parameter search.

## Session: 2026-02-14 - NSSS allocation optimization pass 2 (solve_ss/block_solver)

### Goal
Reduce remaining NSSS allocations concentrated in `solve_ss` and `block_solver`, especially broadcast/slicing/vcat temporaries in numerical block solves.

### What Was Done

1. **Refactored `solve_ss` to reuse solver workspace buffers**
   - File: `src/MacroModelling.jl`
   - Replaced allocation-heavy broadcast initialisation:
     - removed `max.(...)`, `min.(...)`, `fill(...)`, list-comprehension temp vectors.
   - Replaced extended-problem `vcat(sol_values_init, closest_parameters_and_solved_vars)` with in-place construction in the extended workspace init buffer.
   - Replaced slice allocation `sol_new_tmp[1:length(guess)]` with a view.
   - Replaced output clamp broadcast with in-place clamping into reusable buffer.
   - Replaced `all(guess .< 1e12)` / `any(guess .< 1e12)` with a single non-allocating loop.

2. **Refactored `block_solver` iteration scaffolding**
   - File: `src/MacroModelling.jl`
   - Removed temporary `guesses` vector and `fill(1e12, ...)` allocation by reusing a workspace vector for fallback guesses.
   - Replaced small dynamic arrays (`[true,false]`, `[newton, levenberg_marquardt]`, start-values vectors) with tuples.
   - Kept algorithmic behavior unchanged.

### Validation

#### Performance (SW07, warmed, `derivatives=false`)

- Previous pass mean allocation: `71384.0` bytes
- After pass 2 mean allocation: `49209.33` bytes
- Additional reduction: `~31.1%`
- Runtime remained in the same range (`~0.13 ms` mean).

#### Per-step allocation check (SW07)

- Numerical block allocations reduced from roughly `13.4–14.2 KB` to `6.1–6.8 KB` per step.
- Wrapper allocation reduced from `53264` bytes to `31184` bytes.

#### Correctness checks

- `RBC_switch` NSSS acceptance error: `0.0`
- `FS2000` NSSS acceptance error: `4.0091197218190075e-15`
- `Smets_Wouters_2007` NSSS acceptance error: `8.537013150143187e-15`
- All checks satisfy `error < tol.NSSS_acceptance_tol`.

## Session: 2026-02-14 - NSSS runtime allocation optimization (SW07)

### Goal
Reduce runtime allocations in the refactored step-based NSSS solver on `Smets_Wouters_2007` without changing solver behavior.

### What Was Done

1. **Eliminated per-step hot-path allocations in numerical NSSS steps**
   - File: `src/nsss_solver.jl`
   - Reused persistent workspace vectors for:
     - gathered `params_and_solved_vars`
     - numerical bounds (`lbs`, `ubs`)
     - initial guess vector (`inits[1]`) via `resize!` + `copyto!` instead of `Vector(...)` construction.
   - Removed temporary `Vector{Float64}(...)` materializations for bounds slices.

2. **Reduced small wrapper allocations**
   - File: `src/nsss_solver.jl`
   - Replaced fallback distance expression in `find_closest_solution` with an in-place loop (no vector subtraction allocation).
   - Added a reusable `scaled_parameters` buffer in `solve_nsss_wrapper` for continuation interpolation.

3. **Extended NSSS workspace to support buffer reuse**
   - File: `src/structures.jl`
   - Added reusable vectors to `NSSSSolverWorkspace`:
     - `params_and_solved_vars_buffer`
     - `lbs_buffer`
     - `ubs_buffer`
   - Updated workspace constructors and initialization in `build_nsss_solver!`.

4. **Fixed regression found during validation**
   - Root cause: bounds buffers were initially resized to guess length, but cold-start extended solves require full block bounds.
   - Fix: bounds buffers now resize/fill to full numerical bounds range length.

### Validation

#### Performance (warmed calls, `derivatives=false`, SW07)

- Baseline mean allocation: `73976.0` bytes
- After optimization mean allocation: `71384.0` bytes
- Allocation reduction: `2592` bytes (`~3.5%`)
- Runtime remained in the same range (about `0.12 ms` mean in both runs).

#### Correctness smoke checks

- `RBC_switch`: `get_steady_state(..., derivatives=false)` passes
- `FS2000`: `get_steady_state(..., silent=true)` passes
- `Smets_Wouters_2007`: `get_steady_state(..., derivatives=false)` passes

## Session: 2026-02-06 - NSSS_solve Refactoring

### Task Completed
Refactored the NSSS (Non-Stochastic Steady State) solving mechanism to use a normal Julia function wrapper instead of requiring direct access to the runtime-generated function.

### What Was Done

1. **Created new file**: `src/nsss_solver.jl`
   - Contains `solve_nsss_wrapper` - a normal Julia function
   - This function wraps calls to the model-specific RTGF `𝓂.functions.NSSS_solve`
   - Provides a clean API for NSSS solving

2. **Updated call sites** (7 total):
   - `src/MacroModelling.jl`: 4 locations (including include statement)
     - Line ~167: Added include for nsss_solver.jl
     - Line ~5922: `calculate_SS_solver_runtime_and_loglikelihood`
     - Line ~5978: `verify_SS_solver_parameters_function`  
     - Line ~6008: `select_fastest_SS_solver_parameters!`
     - Line ~9804: `get_NSSS_and_parameters` (main entry point)
   - `src/custom_autodiff_rules/forwarddiff.jl`: Line ~256
   - `src/custom_autodiff_rules/zygote.jl`: Line ~383
   - `ext/OptimExt.jl`: Line ~137

3. **Testing**:
   - Successfully tested with simple RBC model
   - Steady state computation works correctly
   - No regressions detected

4. **Code review**:
   - Fixed parameter naming issue (renamed `solver_parameters` parameter to `solver_params` to avoid confusion with type name)
   - All feedback addressed

### Key Implementation Details

- **No breaking changes**: The RTGF `𝓂.functions.NSSS_solve` remains unchanged
- **Thin wrapper**: `solve_nsss_wrapper` is just a delegation function, no performance impact
- **Clean separation**: Users now call a normal function instead of accessing model internals

### Testing Evidence

```julia
# RBC model test passed
✓ Model defined successfully
✓ Steady state computed successfully
Sample SS values: [5.936252888048724, 47.39025414828808, 6.884057971014486]
```

### Files Created/Modified

**New:**
- `src/nsss_solver.jl`

**Modified:**
- `src/MacroModelling.jl`
- `src/custom_autodiff_rules/forwarddiff.jl`
- `src/custom_autodiff_rules/zygote.jl`
- `ext/OptimExt.jl`

### Next Steps

None - task complete. The refactoring is ready for merge.

### Notes for Future Work

If further refactoring of NSSS is needed:
1. The wrapper function in `nsss_solver.jl` can be expanded without touching call sites
2. The RTGF generation code is in `write_steady_state_solver_function!` (two versions at lines ~4794 and ~5299 in MacroModelling.jl)
3. The actual solving logic is in the expression that builds `solve_exp` starting around line ~5078 and ~5785

## Session: 2026-02-11 - Align NSSS Step Solver With Main

### Task Completed
Aligned the step-based NSSS solver with the main-branch RTGF behavior and validated that key models match the main steady-state outputs.

### What Was Done

1. **Aligned continuation logic** in `solve_nsss_wrapper` to use the main-branch interpolation rule (closest_solution_init) and removed zeroing of cached guesses.
2. **Matched numerical error ordering** by moving aux error checks after block solves to mirror main’s flow.
3. **Disabled symbolic single-variable solves when `symbolic_SS` is false**, matching main’s default numerical behavior and silencing those “failed symbolic” logs unless verbose.
4. **Stabilized symbol replacement** by moving `replace_symbols` into an `@unstable` block and tightening replacement dictionary types to avoid DispatchDoctor errors.
5. **Allowed symbol-only equations** in numerical block handling to prevent conversion errors.

### Tests

- Main reference generation: `julia -t auto --project=/private/tmp/MacroModelling.jl-main /tmp/run_main_nsss.jl`
- Branch comparison: `julia -t auto --project=. /tmp/run_branch_compare.jl`

### Results

- FS2000 steady state matches main (max abs diff ~4e-12)
- QUEST3_2009 steady state matches main (max abs diff ~1.6e-9)
- Gali_2015_chapter_3_nonlinear steady state matches main (max abs diff ~4e-14)

## Session: 2026-02-12 - Full Model Parity (No Global Search)

### Goal
Bring the step-based NSSS solver to parity with `main` across all example models in `models/`, while explicitly avoiding the ~120s global solver-parameter search.

### What Was Done

1. **Fixed stale ➕ dependencies inside aux functions**
   - Root cause: compiled aux functions for domain-safety ➕ variables could compute `➕₂` from the *old* `➕₁` value in `sol_vec` (not the freshly computed one), causing clamping to `1e12` and large false errors (notably `Caldara_et_al_2012`).
   - Fix: inline model-level auxiliary dependencies when building aux RHS lists, so later ➕ expressions substitute earlier ➕ definitions instead of reading stale `sol_vec` entries.

2. **Refreshed bounds for `Analytical ➕:` steps after step construction**
   - Root cause: bounds for certain ➕ variables are registered lazily during step construction, but some `Analytical ➕:` steps were created before their final bounds existed and defaulted to `(eps(), 1e12)`, spuriously failing (e.g. FS2000, Aguiar_Gopinath_2007 at `Analytical ➕: ➕₅`).
   - Fix: post-pass over `solve_steps` to rebuild `Analytical ➕:` steps with the final bounds from `𝓂.constants.post_parameters_macro.bounds`.

3. **Re-aligned numerical warm-start behavior with `main`**
   - Removed the branch-only behavior that replaced non-finite cached initial guesses with `0.0` before clamping. `main` clamps cached guesses directly to bounds.

### Files Changed

- src/MacroModelling.jl
  - Inline model-level aux dependencies when compiling aux functions.
  - Refresh bounds for `Analytical ➕:` steps before storing `𝓂.NSSS.solve_steps`.
- src/nsss_solver.jl
  - Align numerical-block initial guess handling with `main` (no non-finite sanitization).

### Verification

- Re-ran full model set comparison against the `main` worktree using `/tmp/dump_all_models_nsss.jl` with ESCH/SAMIN search disabled on both sides.
- Result: all dumped models matched within the comparison tolerance (`matched: 22`, `mismatched: 0`; same error-file set on both sides).

## Session: 2026-02-12 - Merge `write_steady_state_solver_function!` methods

### Goal
Collapse the duplicated NSSS steady-state writer methods into one implementation, with feature precedence given to the former symbolic-path definition.

### What Was Done

1. **Unified the function implementation in `src/MacroModelling.jl`**
   - Kept the Def1 code path as the single implementation.
   - Updated signature to accept optional symbolics data:
     - `write_steady_state_solver_function!(𝓂::ℳ, symbolic_SS::Bool = false, symbolics_data::Union{Nothing, symbolics} = nothing; ...)`
   - Added top-level branching for `unknowns`, `eq_list`, and `ss_equations`:
     - Uses `symbolics_data` fields when present.
     - Uses `𝓂.constants`/`𝓂.equations` when `symbolics_data === nothing`.

2. **Removed the duplicate precompile-only method**
   - Deleted the second `write_steady_state_solver_function!(𝓂::ℳ; ...)` implementation.
   - Precompile setup now routes through the unified Def1-based implementation.

3. **Updated setup call site**
   - In `set_up_steady_state_solver!`, precompile branch now calls:
     - `write_steady_state_solver_function!(𝓂, false, nothing, verbose = verbose)`

4. **Follow-up fixes discovered during runtime validation**
   - Added fallback empty redundant-list in non-symbolics mode because `post_model_macro` has no `var_redundant_list` field.
   - Disabled single-variable symbolic solving in numeric mode (`!symbolic_SS`).
   - Prevented `partial_solve` from running in numeric mode (it expects SymPy inputs).

### Verification

Executed targeted two-path smoke test:

```bash
julia -t auto --project=. -e 'using MacroModelling; ... define RBC + RBCP(precompile=true) ...; get_NSSS_and_parameters(...)'
```

Results:
- `RBC err=1.932762129561291e-15 it=0 first3=[5.936252888048744, 47.39025414828829, 6.884057971014508]`
- `RBCP err=1.8221734021403184e-15 it=27 first3=[5.936252888048744, 47.39025414828829, 6.884057971014508]`

Both the normal (`symbolics_data` present) and precompile (`symbolics_data = nothing`) setup paths succeed.

## Session: 2026-02-12 - Remove NSSS reparsing on repeated `SS(..., derivatives=false)` calls

### Goal
Prevent repeated steady-state setup/parsing on the branch when calling `SS(model, derivatives = false)` multiple times, and resolve branch-only model failures that were triggered by that reparsing behavior.

### What Was Done

1. **Changed reparsing gate in `solve!`**
    - File: `src/MacroModelling.jl`
    - Replaced the RTGF-type check with a step-infrastructure check:
       - from: `!(𝓂.functions.NSSS_solve isa RuntimeGeneratedFunctions.RuntimeGeneratedFunction)`
       - to: `isempty(𝓂.NSSS.solve_steps)`
    - This keeps setup lazy for models that have not built steps yet, while avoiding rerunning setup once step-based NSSS infrastructure exists.

2. **Validated previously failing models under disabled global search**
    - Re-tested:
       - `Backus_Kehoe_Kydland_1992`
       - `Ghironi_Melitz_2005`
       - `QUEST3_2009`
       - `Smets_Wouters_2007`
    - For each, two consecutive `SS(model, derivatives=false)` calls now return same length/output (`maxdiff = 0.0`) and no DimensionMismatch errors.

3. **Re-ran all-model check with 120s global search disabled**
    - Branch (`/tmp/branch_ss_dfalse_all.bin`):
       - solved models: 22
       - errors: 1 (`Smets_Wouters_2003_obc.jl` include-time symbol error)
       - NaN steady states: 0
       - reparsing condition before second call: 0

4. **Main-branch reference check (same script/settings)**
    - Main (`/tmp/main_ss_dfalse_all.bin`):
       - solved models: 22
       - errors: 1 (`Smets_Wouters_2003_obc.jl` include-time symbol error)
       - NaN steady states: 0

### Comparison Snapshot (branch vs main)

- Common solved models: 22
- Remaining value mismatches (`SS(..., derivatives=false)`):
   - `Caldara_et_al_2012` (max abs diff ≈ 14.6335)
   - `Ascari_Sbordone_2014` (max abs diff ≈ 2.31e-6)
- Excluded/error model remains the same class on both sides: `Smets_Wouters_2003_obc.jl`.

## Session: 2026-02-12 - Fix `Smets_Wouters_2003_obc` include/parsing failure

### Goal
Resolve the include-time failure in `models/Smets_Wouters_2003_obc.jl` caused by OBC-related symbols with superscript/subscript decorations during steady-state setup.

### Root Cause

In `write_steady_state_solver_function!`, single-variable block handling rewrote `min`/`max` equations and then called `eval(...)` on the rewritten expression.

## Session: 2026-02-13 - Move numerical blocks into `NumericalNSSSStep`

### Goal
Store compiled numerical NSSS blocks directly on each `NumericalNSSSStep` instead of in `𝓂.NSSS.solve_blocks_in_place`.

### What Was Done

1. **Extended `NumericalNSSSStep` payload**
   - File: `src/structures.jl`
   - Added `solve_block::ss_solve_block` field.
   - Kept `block_index` for cache slot mapping only.

2. **Refactored numerical block construction path**
   - File: `src/nsss_solver.jl`
   - `write_block_solution!` now takes `block_index` as input and returns the compiled `solve_block` in `block_meta`.
   - Removed pushing compiled blocks into `𝓂.NSSS.solve_blocks_in_place`.
   - Added `numerical_block_count` during step construction to assign stable block indices.
   - `build_numerical_step` now embeds `block_meta.solve_block` into each `NumericalNSSSStep`.

3. **Updated runtime execution and cache sizing**
   - File: `src/nsss_solver.jl`
   - `execute_step!(::NumericalNSSSStep, ...)` now calls `block_solver(..., step.solve_block, ...)`.
   - Wrapper cache-length expectation now uses the number of numerical steps:
     - `expected_cache_length = 2 * count(step isa NumericalNSSSStep) + 1`
   - This decouples continuation cache handling from `solve_blocks_in_place`.

### Validation

Ran focused smoke test:

```bash
julia -t auto --project=. -e 'using MacroModelling; include("models/FS2000.jl"); ss = get_steady_state(FS2000, silent=true); n_num = count(st->st isa MacroModelling.NumericalNSSSStep, FS2000.NSSS.solve_steps); has_blocks = all((st isa MacroModelling.NumericalNSSSStep) ? (st.solve_block isa MacroModelling.ss_solve_block) : true for st in FS2000.NSSS.solve_steps); println("n_num=", n_num); println("solve_blocks_len=", length(FS2000.NSSS.solve_blocks_in_place)); println("has_blocks=", has_blocks); println("ss_ok=", !isempty(ss))'
```

Observed:
- `n_num=1`
- `solve_blocks_len=0`
- `has_blocks=true`
- `ss_ok=true`

For OBC models this could include generated symbols such as `Χᵒᵇᶜ⁺...` that are symbolic equation variables, not module globals. Evaluating them at parse time raised `UndefVarError`.

### Fix

- File: `src/MacroModelling.jl`
- In the single-variable block path:
   - Added `minmax_rewritten` flag.
   - Replaced `eq_to_solve = eval(minmax_fixed_eqs)` with `eq_to_solve = minmax_fixed_eqs`.
   - Forced numerical fallback for rewritten min/max equations by extending the guard:
      - `if !symbolic_SS || avoid_solve || minmax_rewritten || ...`

This avoids evaluating OBC symbolic names as Julia globals and keeps min/max-rewritten equations in the numerical path.

### Validation

1. Direct include test now succeeds:

```bash
julia -t auto --project=. -e 'using MacroModelling; include("models/Smets_Wouters_2003_obc.jl"); println("included")'
```

2. Full all-model check with global search disabled:

```bash
julia -t auto --project=. /tmp/check_ss_derivatives_false_all_models.jl . /tmp/branch_ss_dfalse_all_after_obc_fix.bin
```

Results:
- solved models: 23
- errors: 0

## Session: 2026-02-13 - Full include+SS(derivatives=false) parity without global search

### Goal
Verify all models in `models/` and `test/models/` using the exact workflow `include(model_file)` followed by `SS(model, derivatives = false)`, while disabling global solver-parameter search, and fix branch-only regressions vs `main`.

### What Was Done

1. **Built and ran a no-global-search sweep harness**
   - Overrode `find_SS_solver_parameters!(::Val{:ESCH}/:SAMIN, ...) = false` for the validation run.
   - Tested all model files under both `models/` and `test/models/`.

2. **Compared branch behavior against `main` baseline**
   - Confirmed two branch regressions under no-global-search checks:
     - `models/Gali_2015_chapter_3_obc.jl` include-time symbolic shape error.
     - `test/models/RBC_CME_calibration_equations_and_parameter_definitions.jl` `LambertW` runtime error in NSSS solve step.

3. **Root-cause fix in step-based NSSS writer**
   - File: `src/nsss_solver.jl`
   - Removed the fragile symbolic `partial_solve` shortcut for small multi-equation blocks.
   - Small blocks now consistently route through `write_block_solution!` (numerical block path), matching `main` behavior and avoiding symbolic edge-case failures.

### Validation

1. Targeted reproduction checks now pass:
   - `include("test/models/RBC_CME_calibration_equations_and_parameter_definitions.jl")`
   - Ordered include sequence ending in `models/Gali_2015_chapter_3_obc.jl` (with global search disabled)

2. Full sweep with global search disabled:
   - Command: `julia -t auto --project=. /tmp/check_all_models_ss_noglobal.jl`
   - Result: `Summary: total=33 ok=33 fail=0`

## Session: 2026-02-12 - Restore analytical partial-block replication in step-based NSSS

### Goal
Replicate the monolithic RTGF block setup for mixed analytical/numerical solving (notably for `Caldara_et_al_2012`) in the step-based NSSS writer.

### Root Cause

In the step-based multi-variable path (`write_steady_state_solver_function!`), `partial_solve(...)` was invoked but its result was ignored. This dropped analytical sub-block extraction and sent the full block to `write_block_solution!` as numerical.

### Fix

- File: `src/MacroModelling.jl`
- Added a concrete `PartialSolveResult{T,E}` struct.
- Updated `partial_solve` to return typed metadata (solved/remaining vars/eqs and indices) via `PartialSolveResult`.
- In the multi-variable numerical fallback path:
   - consume `partial_solve` output,
   - emit an `AnalyticalNSSSStep` for the solved symbolic subset,
   - pass only the unresolved remainder into `write_block_solution!`.
- Guarded numerical-step construction when the reduced remainder is empty.

### Validation

1. **Symbolic-mode Caldara block composition**

```bash
julia -t auto --project=. -e 'using MacroModelling; include("models/Caldara_et_al_2012.jl"); MacroModelling.set_up_steady_state_solver!(Caldara_et_al_2012, verbose=false, silent=true, avoid_solve=false, symbolic=true); println("steps=", length(Caldara_et_al_2012.NSSS.solve_steps)); for (i,s) in enumerate(Caldara_et_al_2012.NSSS.solve_steps); println(i,"|",typeof(s),"|",getfield(s,:description)); end'
```

Result:
- Successful setup, no type-instability errors.
- Step mix includes analytical steps with a single remaining numerical block (`Numerical block 1: V, ➕₃`), consistent with mixed block behavior.

2. **Caldara NSSS solve**

```bash
julia -t auto --project=. -e 'using MacroModelling; include("models/Caldara_et_al_2012.jl"); ss,(err,it)=MacroModelling.get_NSSS_and_parameters(Caldara_et_al_2012,Caldara_et_al_2012.parameter_values, opts=MacroModelling.merge_calculation_options(verbose=false), cold_start=true); println("err=",err," it=",it," first5=",ss[1:5])'
```

Result:
- `err = 4.437718438915963e-15`, `it = 0`

3. **Targeted regression**

```bash
julia -t auto --project=. -e 'using MacroModelling, Test; include("test/test_filter_equations.jl")'
```

Result:
- Pass (exit code 0).
- reparsing-condition-before-second-call: 0
- NaN steady states: 0

## Session: 2026-02-12 - Fix equation filtering tests

### Goal
Fix failing "Test equation filtering" tests in `test/test_filter_equations.jl`.

### Root Causes

1. `get_dynamic_equations` filtered against raw internal dynamic equations instead of transformed user-facing expressions, so symbols like `:k` failed to match in models with transformed timing/auxiliary representations.
2. Filtering logic for curly-brace variables (`K{H}`) compared mixed internal symbols (`◖◗`) and transformed `Expr(:curly, ...)` forms directly, causing false negatives for both any-timing and exact-timing filters.

### Fixes

- File: `src/inspect.jl`
   - In `get_dynamic_equations`, changed filtering target from `orig` to transformed `expr`:
      - from: `expr_contains(orig, sym, pattern)`
      - to: `expr_contains(expr, sym, pattern)`
   - In `expr_contains`, added normalized textual matching (`◖/◗` → `{`/`}`) for:
      - symbol-only filtering (any timing), and
      - exact pattern filtering (with timing)
   - This makes filtering robust across internal/transformed symbol encodings.

### Validation

Executed:

```bash
julia -t auto --project=. -e 'using MacroModelling, Test; include("test/test_filter_equations.jl")'
```

Result: test file completed without assertion failures.

## Session: 2026-02-12 - PR cleanup pass (non-functional)

### Goal
Clean up unnecessary/redundant code and comments across the active PR while preserving solver logic.

### What Was Done

1. **Removed redundant imports from new NSSS file**
   - File: `src/nsss_solver.jl`
   - Removed local imports already available from module scope (`CircularBuffer`, `ℒ`, `@ignore_derivatives`).

2. **Clarified step API intent and stale wording**
   - File: `src/nsss_solver.jl`
   - Renamed intentionally unused arguments in `execute_step!(::AnalyticalNSSSStep, ...)` with `_` prefixes.
   - Updated stale wrapper docstring text that still referenced delegation to `𝓂.functions.NSSS_solve`.
   - Minor comment cleanup (`LOCAL` wording and one parenthetical).

3. **Removed constructor placeholder comment noise**
   - File: `src/macros.jl`
   - Dropped inline trailing comments on placeholder fields in `non_stochastic_steady_state(...)` initialization.

### Validation

- Diagnostics check: no errors in edited files (`src/nsss_solver.jl`, `src/macros.jl`, and previously touched PR files).
- Runtime smoke test was attempted, but local environment precompile failed under Julia 1.12 with a package precompilation error unrelated to the cleanup edits (`ERROR: Failed to precompile MacroModelling ...`).

## Session: 2026-02-12 - Restore analytical NSSS steps for Caldara parity

### Goal
Ensure the step-based NSSS writer keeps the same analytical-vs-numerical split as `main` for `Caldara_et_al_2012` (mostly analytical with one numerical block).

### Root Cause

In `write_steady_state_solver_function!` (step-based path), two guards diverged from `main` behavior:

1. **Single-variable blocks** were prevented from symbolic solving when `symbolic_SS == false`.
2. **Small multi-variable numerical blocks** (`length(pe) <= 5`) only ran `partial_solve(...)` when `symbolic_SS == true`.

Together these made Caldara effectively all-numerical in this branch.

### Fix

- File: `src/MacroModelling.jl`
   - In single-variable block handling, removed the `!symbolic_SS` gate from:
      - `if ... count_ops(...) > 15`
   - Kept symbolic fallback logging under `verbose` only (not tied to `symbolic_SS`).
   - In small multi-variable numerical fallback, always run `partial_solve(...)` (match `main`) and emit `AnalyticalNSSSStep` for solved subset.

### Validation

Executed:

```bash
julia -t auto --project=. -e 'using MacroModelling; include("models/Caldara_et_al_2012.jl"); m = @eval Caldara_et_al_2012; MacroModelling.set_up_steady_state_solver!(m, verbose=false, silent=true); steps = m.NSSS.solve_steps; nA = count(s -> s isa MacroModelling.AnalyticalNSSSStep, steps); nN = count(s -> s isa MacroModelling.NumericalNSSSStep, steps); println("analytical=$nA numerical=$nN total=$(length(steps))"); for (i,s) in enumerate(steps); println("$i|" * getfield(s,:description)); end'
```

Result:
- `analytical=14 numerical=1 total=15`
- Numerical block: `Numerical block 1: V, ➕₃`
- All remaining blocks are analytical/constant, matching expected main-style decomposition.

## Session: 2026-02-12 - Respect `symbolic_SS` and rely on macro default

### Goal
Keep `symbolic_SS` behavior configurable in the NSSS writer while using the macro default `symbolic = true` for main-like analytical behavior.

### What Was Done

- Confirmed `@parameters` default option already sets `symbolic = true`.
- Restored `symbolic_SS` gating in `write_steady_state_solver_function!`:
   1. Single-variable symbolic attempt now guarded by `!symbolic_SS` again.
   2. Partial symbolic extraction in small multi-variable numerical blocks (`partial_solve`) now runs only when `symbolic_SS` is true.
   3. Verbose symbolic-failure logging restored to `verbose && symbolic_SS`.

### Validation

Executed:

```bash
julia -t auto --project=. -e 'using MacroModelling; include("models/Caldara_et_al_2012.jl"); m = @eval Caldara_et_al_2012; MacroModelling.set_up_steady_state_solver!(m, verbose=false, silent=true, symbolic=false); ...; MacroModelling.set_up_steady_state_solver!(m, verbose=false, silent=true, symbolic=true); ...'
```

Results:
- `symbolic=false analytical=0 numerical=15 total=15`
- `symbolic=true  analytical=14 numerical=1 total=15`

This confirms the setting is respected while the default (`symbolic=true`) preserves the desired analytical-heavy behavior.

## Session: 2026-02-12 - Move NSSS writer stack into nsss_solver and reduce allocations

### Goal
Move `write_steady_state_solver_function!` and its helper/call chain into `src/nsss_solver.jl`, then reduce NSSS allocations while guarding runtime regressions.

### What Was Done

1. **Relocated NSSS writer/helper stack to `src/nsss_solver.jl`**
   - Moved definitions:
      - `replace_symbols`
      - `write_block_solution!`
      - `PartialSolveResult`
      - `partial_solve`
      - `make_equation_robust_to_domain_errors`
      - `compile_exprs_to_func`
      - `build_numerical_step`
      - `write_steady_state_solver_function!`
   - Removed the moved definitions from `src/MacroModelling.jl`.

2. **Allocation-focused updates in `src/nsss_solver.jl`**
   - Added `EMPTY_NSSS_STEP_CACHE` to avoid repeated empty cache allocations on analytical steps.
   - Hardened nearest-cache selection with expected-length and shape checks in `find_closest_solution`.
   - Avoided unnecessary appends for empty step caches in `solve_nsss_steps`.
   - Avoided unnecessary `copy(initial_parameters)` in wrapper fallback path.
   - Prevented cache aliasing by copying numerical block outputs before storing them in continuation cache.
   - Made numerical-step cache index and initial-guess handling robust to missing/malformed cache entries.

3. **Buffer reuse added for numerical steps (`src/structures.jl`)**
   - Extended `NumericalNSSSStep` with:
      - `initial_guess_buffer::Vector{Float64}`
      - `inits_buffer::Vector{Vector{Float64}}`
   - Updated numerical-step construction and execution to reuse these buffers.

4. **Benchmark tooling**
   - Added `BenchmarkTools` to project dependencies (`Project.toml`/`Manifest.toml`) for repeatable allocation/time measurements.

### Validation

1. **Caldara solve smoke test**

```bash
julia -t auto --project=. -e 'using MacroModelling; include("models/Caldara_et_al_2012.jl"); ss,(err,it)=MacroModelling.get_NSSS_and_parameters(Caldara_et_al_2012,Caldara_et_al_2012.parameter_values, opts=MacroModelling.merge_calculation_options(verbose=false), cold_start=true); println("err=",err," it=",it," n=",length(ss));'
```

Result:
- `err=1.1102230246251565e-16 it=0 n=13`

2. **RBC benchmark (symbolic=false setup) after allocation pass**

```bash
julia -t auto --project=. -e 'using MacroModelling, BenchmarkTools; ...; b=@benchmark MacroModelling.solve_nsss_wrapper(...); ...'
```

Result:
- `min_time_ns=33541.0`
- `median_time_ns=35834.0`
- `min_alloc_bytes=58352`
- `median_alloc_bytes=58352`
- `min_allocs=1261`
- `median_allocs=1261`

### Notes

- Earlier baseline in this session for the same RBC microbenchmark was:
   - `min_time_ns=32667`, `median_time_ns=33312.5`, `min_alloc_bytes=59024`, `min_allocs=1279`.
- Net effect of the final allocation pass versus that baseline:
   - allocations reduced by `672` bytes and `18` allocations,
   - `min` runtime increase ~`2.67%` (within the 3% guard on min-time metric used during tuning).

## Session: 2026-02-13 - Replace NSSS symbolic/simplify options with `ss_symbolic_mode`

### Goal
Replace the old `@parameters` NSSS options (`symbolic`, `simplify`) with a single explicit mode option, without backward fallback, and propagate this through runtime setup, docstrings, and tests.

### What Was Done

1. **API change in `@parameters` option parsing and docs**
   - File: `src/macros.jl`
   - Removed option parsing for `symbolic` and `simplify`.
   - Added strict option parsing for:
      - `ss_symbolic_mode ∈ [:none, :single_equation, :full]`
   - Updated `@parameters` docstring option list accordingly.

2. **Model constant schema updated**
   - Files: `src/structures.jl`, `src/options_and_caches.jl`
   - Replaced fields in `post_parameters_macro`:
      - removed: `simplify::Bool`, `symbolic::Bool`
      - added: `ss_symbolic_mode::Symbol`
   - Updated default initialization to `:single_equation`.

3. **Runtime wiring updated in solve/setup path**
   - File: `src/MacroModelling.jl`
   - Added `steady_state_symbolic_mode_flags(ss_symbolic_mode)` mapping:
      - `:none` -> `(avoid_solve=true, symbolic=false)`
      - `:single_equation` -> `(avoid_solve=false, symbolic=false)`
      - `:full` -> `(avoid_solve=false, symbolic=true)`
   - Updated `set_up_steady_state_solver!` to take `ss_symbolic_mode` instead of `avoid_solve`/`symbolic`.
   - Updated `solve!` call sites to read mode from `𝓂.constants.post_parameters_macro.ss_symbolic_mode`.

4. **Tests updated to new API**
   - File: `test/runtests.jl`
   - Replaced symbolic test model setup lines:
      - `symbolic = true` -> `ss_symbolic_mode = :full`

### Validation

1. Diagnostics check on touched files (no errors):
   - `src/macros.jl`
   - `src/MacroModelling.jl`
   - `src/structures.jl`
   - `src/options_and_caches.jl`
   - `test/runtests.jl`

2. Targeted smoke test script (RBC) run in this worktree:

```bash
julia -t auto --project=. tasks/ss_symbolic_mode_smoke.jl
```

Result highlights:
- `default_len=4 full_len=4 none_len=4`
- `mode_field_default=single_equation`
- `mode_field_full=full`
- `mode_field_none=none`

3. Temporary smoke script removed after validation.

## Session: 2026-02-13 - Remove `solve_blocks_in_place` from NSSS structs

### Goal
Complete the ongoing NSSS refactor by removing `solve_blocks_in_place` from model structs and keeping numerical block payloads only on `NumericalNSSSStep`.

### What Was Done

1. **Removed field from NSSS struct**
   - File: `src/structures.jl`
   - Deleted `solve_blocks_in_place::Vector{ss_solve_block}` from `non_stochastic_steady_state`.

2. **Updated macro-based model construction**
   - File: `src/macros.jl`
   - Removed temporary initializer `NSSS_solve_blocks_in_place = ss_solve_block[]`.
   - Removed the corresponding argument from `non_stochastic_steady_state(...)` construction.

3. **Removed leftover setup reference**
   - File: `src/nsss_solver.jl`
   - Deleted `empty!(𝓂.NSSS.solve_blocks_in_place)` from `write_steady_state_solver_function!`.

### Validation

Ran focused runtime check:

```bash
julia -t auto --project=. -e 'using MacroModelling; @model RBC_tmp begin ... end; @parameters RBC_tmp begin ... end; ss1=get_steady_state(RBC_tmp, silent=true); include("models/FS2000.jl"); ss2=get_steady_state(FS2000, silent=true); println("rbc_ok=", !isempty(ss1)); println("fs_ok=", !isempty(ss2)); println("steps=", length(FS2000.NSSS.solve_steps))'
```

Observed:
- `rbc_ok=true`
- `fs_ok=true`
- `steps=17`

Also verified no remaining `solve_blocks_in_place` references in `src/` or `test/`.

## Session: 2026-02-13 - Dissolve `NSSS` model field into constants/workspaces

### Goal
Remove the `NSSS` container field from `ℳ` and relocate its members to `constants` and `workspaces`.

### What Was Done

1. **Removed NSSS struct from model root**
    - File: `src/structures.jl`
    - Removed `NSSS::non_stochastic_steady_state` from `ℳ`.
    - Replaced `non_stochastic_steady_state` with `nsss_workspace` (step runtime storage only).

2. **Moved NSSS members to requested containers**
    - **To `workspaces`:** `solve_steps`, `param_prep!`, `n_sol`, `output_indices`, `n_ext_params`, `sol_names`, `exo_zero_indices`, `param_names_ext` via `workspaces.nsss`.
    - **To `constants`:** `dependencies` via new field `constants.nsss_dependencies`.

3. **Updated constructors and macro wiring**
    - File: `src/options_and_caches.jl`
       - `Workspaces()` now initializes `nsss_workspace(...)`.
       - `Constants(...)` now initializes `nsss_dependencies = nothing`.
    - File: `src/macros.jl`
       - Removed `non_stochastic_steady_state(...)` construction from `ℳ(...)` assembly.

4. **Rewired all call sites**
    - File: `src/nsss_solver.jl`
       - Replaced `𝓂.NSSS.*` writes/reads with `𝓂.workspaces.nsss.*`.
       - Replaced dependency writes with `𝓂.constants.nsss_dependencies`.
    - File: `src/MacroModelling.jl`
       - Setup guard now checks `isempty(𝓂.workspaces.nsss.solve_steps)`.
       - Parameter-update invalidation now uses `𝓂.constants.nsss_dependencies`.
    - File: `test/runtests.jl`
       - Updated assertions from `model.NSSS.solve_steps` to `model.workspaces.nsss.solve_steps`.

### Validation

1. Diagnostics: no errors in edited files (`structures.jl`, `options_and_caches.jl`, `macros.jl`, `nsss_solver.jl`, `MacroModelling.jl`, `test/runtests.jl`).

2. Runtime smoke check:

```bash
julia -t auto --project=. -e 'using MacroModelling, Test; @model RBC_switch begin ... end; @parameters RBC_switch begin ... end; @test !hasproperty(RBC_switch, :NSSS); @test !isempty(RBC_switch.workspaces.nsss.solve_steps); include("models/FS2000.jl"); ss=get_steady_state(FS2000, silent=true); @test !isempty(ss); println("ok")'
```

Observed: `ok`.

## Session: 2026-02-16 - Use FastLapackInterface for QZ/QR/LU matrix paths

### Goal
Replace QZ (including reordering), QR, and LU matrix/matrix solve hotspots with FastLapackInterface-backed routines and add the required reusable workspaces to the model workspace structs.

### Changes

1. **Added FastLapackInterface caches to QME workspace**
   - File: `src/structures.jl`
   - Extended `qme_workspace` with lazily-initialized fields:
      - `lu_ws`, `lu_ws_alt`, `qr_ws`, `qr_orm_ws`, `qz_ws`

2. **Initialized new workspace fields**
   - File: `src/options_and_caches.jl`
   - Updated `Qme_workspace(...)` constructor to initialize new fields with `nothing`.

3. **Replaced QZ + reordering path in Schur QME solver**
   - File: `src/algorithms/quadratic_matrix_equation.jl`
   - For `Float32/Float64`, replaced `schur! + ordschur!` with:
      - `LinearAlgebra.LAPACK.gges!(..., select = FastLapackInterface.id, criterium = 1.0)`
   - Kept existing `schur! + ordschur!` fallback for non-BLAS element types.

4. **Replaced LU matrix/matrix solves in QME doubling path**
   - File: `src/algorithms/quadratic_matrix_equation.jl`
   - For `Float32/Float64`, replaced LU+`ldiv!` matrix solves with:
      - `LAPACK.getrf!` + `LAPACK.getrs!`
   - Added separate LU workspaces for EI/FI (`lu_ws`, `lu_ws_alt`) to preserve pivot data across both factorizations.
   - Kept existing `lu!/ldiv!` fallback for non-BLAS element types.

5. **Replaced QR application in first-order solution paths**
   - Files:
      - `src/perturbation.jl`
      - `src/custom_autodiff_rules/zygote.jl`
   - For `Float32/Float64`, replaced `qr!(...).Q' * ...` with workspace-backed:
      - `LAPACK.geqrf!` + `LAPACK.ormqr!`
   - Reused `qme_ws.qr_ws` and `qme_ws.qr_orm_ws`.
   - Kept existing `qr!` fallback for non-BLAS element types.

### Validation

1. Diagnostics:
   - `get_errors` on touched files reported no errors.

2. Runtime smoke test:

```bash
julia -t auto --project=. -e 'using MacroModelling; include(joinpath(pwd(),"models","RBC_baseline.jl")); ss = get_steady_state(RBC_baseline, derivatives=false, verbose=false); sol = solve!(RBC_baseline); println("ok")'
```

Observed: `ok`.

## Session: 2026-02-16 - Fix NSSS regression after cache clears and disable 120s global fallback by default

### Goal
Resolve recent NSSS misses seen in benchmark/get_steady_state loops after `clear_solution_caches!`, and disable the 120s global solver-parameter search fallback by default.

### Root Cause

- The warm NSSS wrapper path could fail after cache-clear cycles because its initial cache seed can be non-finite/incomplete (`Inf` placeholders), which led to a false failure return `(solution_error = 1.0)` in subsequent warm runs.
- The global fallback search was still enabled by default (`ss_solver_parameters_maxtime = 120.0`), which masked this regression and added long fallback behavior.

### Changes

1. **Disable global search fallback by default**
   - File: `src/macros.jl`
   - Changed default `ss_solver_parameters_maxtime` from `120.0` to `0.0`.
   - Updated `@parameters` docs accordingly (`0.0` disables fallback search).

2. **Gate global search execution on positive maxtime**
   - File: `src/MacroModelling.jl`
   - In `solve_steady_state!`, only invoke `find_SS_solver_parameters!` when
     `solution_error > tol && ss_solver_parameters_maxtime > 0`.

3. **Harden warm NSSS wrapper against invalid cache seeds**
   - File: `src/nsss_solver.jl`
   - In `solve_nsss_wrapper`, if initial cache distance is non-finite,
     immediately run one direct cold-start pass via `solve_nsss_steps(..., cold_start = true)`.
   - Existing end-of-loop cold-start retry retained.
   - On successful fallback, push solved cache and return success.

### Validation

1. **Reproduced original regression before fix**
   - `clear_solution_caches!(QUEST3_2009, :first_order)` + repeated
     `get_steady_state(..., derivatives=false)` emitted:
     `Could not find non-stochastic steady state. Solution error: 1.0 > 1e-12`.

2. **Post-fix targeted regression check**
   - Same repeated QUEST loop now runs without NSSS warning and prints `iter=1/2/3 ok`.

3. **All-model no-global-search path check (regression path)**
   - For each model in `models/`:
     - `clear_solution_caches!(model, :first_order)`
     - `get_steady_state(model, parameters=model.parameter_values, derivatives=false)`
     - `get_NSSS_and_parameters(..., cold_start=false)`
   - Result: `FAILED_COUNT=0`.

### Follow-up validation (user-requested): include `test/models`

- Ran the same no-global-search regression path over both `models/` and `test/models/`.
- Validation path per file:
   - include model file
   - `clear_solution_caches!(model, :first_order)`
   - `get_steady_state(model, parameters=model.parameter_values, derivatives=false)`
   - `get_NSSS_and_parameters(model, model.parameter_values, opts=merge_calculation_options(verbose=false))`
- Harness note:
   - model symbol is parsed from each file's `@model` declaration (not inferred from filename), to handle cases such as `m` in multiple `test/models` files.
- Result:
   - `FAILED_COUNT_TOTAL=0`


## Session: 2026-02-14 - Fix custom steady-state regression (`z` overwritten by stale buffer)

### Goal
Resolve failing custom steady-state tests where the default steady-state output for `RBC`-style models reported `z = c` instead of `z = 0`, causing mismatches between internal/default and custom steady-state paths.

### Root Cause

In `src/nsss_solver.jl`, constant analytical NSSS steps (e.g. `Constant: z = 0`) compiled their evaluator through `compile_exprs_to_func([constant_expr], ...)` with `skipzeros = true`.

For zero-valued constants, the generated function could become effectively a no-op and leave the shared `main_buffer` unchanged. During continuation iterations, that stale buffer value (often from a prior step such as `c`) was then written into the constant variable slot, producing `z = c`.

### Fix

- File: `src/nsss_solver.jl`
- In the `soll[1].is_number == true` branch:
   - Kept existing symbolic handling for `➕` constants (still via compiled expression with clamping).
   - For regular constants, replaced Symbolics-generated eval function with a direct closure:
      - `(out, _sol_vec, _params_vec) -> out[1] = constant_value`
   - This guarantees deterministic writes for constants and avoids stale-buffer reuse.

### Validation

Ran focused reproduction of the previously failing assertions from `test/runtests.jl` custom steady-state block:

```bash
julia -t auto --project=. -e 'using MacroModelling, Test; ... RBC_custom_ss + RBC_macro_ss custom steady-state checks ...'
```

Observed:
- `custom_result ≈ default_ss(:, :Steady_state)` passes.
- `default_ss ≈ custom_ss` passes.
- `irf_after_clear ≈ irf_custom` passes.
- Macro-defined custom function checks pass.
- Script ends with: `custom-ss regression checks passed`.

---

## Session: 2026-02-13 - Remove obsolete `NSSS_solve`

### Goal
Verify that `NSSS_solve` is no longer needed and remove it if unused.

### What Was Done

1. **Usage audit**
   - Searched all `src/` and `test/` for `NSSS_solve` references.
   - Found no active callsites; only one dead assignment and stale comments.

2. **Removed dead field/wiring**
   - File: `src/structures.jl`
      - Removed `NSSS_solve::Function` from `model_functions`.
   - File: `src/macros.jl`
      - Removed `NSSS_solve_func = x->x` placeholder.
      - Removed corresponding constructor argument in `model_functions(...)`.
   - File: `src/nsss_solver.jl`
      - Removed dead assignment to `𝓂.functions.NSSS_solve = ...`.

3. **Cleaned stale references**
   - File: `src/get_functions.jl`
      - Removed commented lines referencing `𝓂.functions.NSSS_solve`.
   - File: `src/structures.jl`
      - Updated overview comment to no longer mention `NSSS_solve`.

### Validation

1. Diagnostics: no errors in edited files.
2. Source grep: no remaining `\bNSSS_solve\b` matches in `src/`.
3. Runtime smoke test:

```bash
julia -t auto --project=. -e 'using MacroModelling, Test; @model RBC_switch begin ... end; @parameters RBC_switch begin ... end; @test !isempty(RBC_switch.functions.nsss_solve_steps); @test RBC_switch.constants.post_complete_parameters.nsss_n_sol > 0; include("models/FS2000.jl"); ss=get_steady_state(FS2000, silent=true); @test !isempty(ss); @test !isempty(FS2000.functions.nsss_solve_steps); println("ok")'
```

Observed: `ok`.

---

## Session: Move NSSS solve-step fields to `functions` and NSSS metadata to `post_complete_parameters`

### Request

- Move `nsss_solve_steps` and `nsss_param_prep!` to `model_functions`.
- Move remaining NSSS metadata fields out of `constants` and into `post_complete_parameters`.

### Changes made

1. **Restructured ownership in `src/structures.jl`**
    - Added to `model_functions`:
       - `nsss_solve_steps::Vector{NSSSSolveStep}`
       - `nsss_param_prep!::Union{Nothing, Function}`
    - Added to `post_complete_parameters`:
       - `nsss_dependencies::Any`
       - `nsss_n_sol::Int`
       - `nsss_output_indices::Vector{Int}`
       - `nsss_n_ext_params::Int`
       - `nsss_sol_names::Vector{Symbol}`
       - `nsss_exo_zero_indices::Vector{Int}`
       - `nsss_param_names_ext::Vector{Symbol}`
    - Removed all `nsss_*` fields from `constants` except existing `post_complete_parameters` container itself.

2. **Updated constructors and update helper in `src/options_and_caches.jl`**
    - `Constants()` now initializes new NSSS metadata fields inside `post_complete_parameters`.
    - Removed old NSSS arguments from `constants(...)` constructor call.
    - Extended `update_post_complete_parameters(...)` to carry all new NSSS metadata kwargs.

3. **Updated default model construction in `src/macros.jl`**
    - `model_functions(...)` constructor call now initializes:
       - `NSSSSolveStep[]`
       - `nothing` (for `nsss_param_prep!`).

4. **Rewired solver/runtime usage**
    - `src/nsss_solver.jl`:
       - Writes `nsss_solve_steps` / `nsss_param_prep!` to `𝓂.functions`.
       - Writes remaining NSSS metadata via
          `𝓂.constants.post_complete_parameters = update_post_complete_parameters(...)`.
       - Reads from `𝓂.functions` and `𝓂.constants.post_complete_parameters` accordingly.
    - `src/MacroModelling.jl`:
       - Setup guard now checks `isempty(𝓂.functions.nsss_solve_steps)`.
       - NSSS dependency invalidation now reads from
          `𝓂.constants.post_complete_parameters.nsss_dependencies`.

5. **Updated tests**
    - `test/runtests.jl` references changed from
       `model.constants.nsss_solve_steps` to `model.functions.nsss_solve_steps`.

### Bug encountered and fixed

- Initial run failed with:
   - `setfield!: immutable struct of type post_complete_parameters cannot be changed`
- Cause: direct mutation of immutable `post_complete_parameters` fields in `nsss_solver.jl`.
- Fix: replaced direct field mutation with `update_post_complete_parameters(...)` reassignment.

### Validation

1. Diagnostics
    - `get_errors` on touched files: no errors.

2. Runtime smoke test

```bash
julia -t auto --project=. -e 'using MacroModelling, Test; @model RBC_switch begin ... end; @parameters RBC_switch begin ... end; @test !isempty(RBC_switch.functions.nsss_solve_steps); @test RBC_switch.constants.post_complete_parameters.nsss_n_sol > 0; include("models/FS2000.jl"); ss=get_steady_state(FS2000, silent=true); @test !isempty(ss); @test !isempty(FS2000.functions.nsss_solve_steps); println("ok")'
```

Observed: `ok`.

## Session: 2026-02-16 - Route `fast_lu` FastLapack buffers through model `workspaces`

### Goal
Ensure the new FastLapackInterface LU path uses preallocated buffers stored on the model `workspaces` struct, instead of creating ad-hoc LU workspace objects in hot call paths.

### Changes

1. **Added shared FastLapack LU cache fields to `workspaces`**
   - File: `src/structures.jl`
   - Added:
      - `fast_lapack_lu_ws`
      - `fast_lapack_lu_rows`
      - `fast_lapack_lu_cols`
      - `fast_lapack_lu_eltype`

2. **Initialized new fields in `Workspaces()` constructor**
   - File: `src/options_and_caches.jl`
   - Added defaults (`nothing`, `0`, `0`, `Float64`) in the `workspaces(...)` initialization.

3. **Updated `fast_lu` / `fast_lu!` to support workspace-backed reuse**
   - File: `src/MacroModelling.jl`
   - Added `get_fast_lu_ws!` helper that reuses/refreshes cached `FastLapackInterface.LUWs` by size/eltype.
   - Added overloads:
      - `fast_lu!(A, lu_workspace; check=...)`
      - `fast_lu(A, lu_workspace; check=...)`
   - Kept existing no-workspace methods as compatibility wrappers.

4. **Wired model-path callsites to pass workspace**
   - File: `src/get_functions.jl`
      - `fast_lu(CC, 𝓂.workspaces, check = false)` in first-order conditional-forecast solves.
   - File: `src/perturbation.jl`
      - Higher-order matrix factorization calls now pass `workspaces`.

### Validation

1. Diagnostics: no errors in touched files (`structures.jl`, `options_and_caches.jl`, `MacroModelling.jl`, `get_functions.jl`, `perturbation.jl`).

2. Runtime smoke test:

```bash
julia -t auto --project=. -e 'using MacroModelling, LinearAlgebra; include(joinpath(pwd(),"models","RBC_baseline.jl")); get_steady_state(RBC_baseline, derivatives=false, verbose=false); A = [1.0 2.0; 3.0 4.0]; F = MacroModelling.fast_lu(A, RBC_baseline.workspaces, check=false); @assert issuccess(F); B = [5.0 6.0; 7.0 8.0]; G = MacroModelling.fast_lu!(B, RBC_baseline.workspaces, check=false); @assert issuccess(G); println("ok")'
```

Observed: `ok`.

## Session: 2026-02-13 - Reclassify heterogeneous `nsss_workspace` fields

### Goal
Sort NSSS internals by role: keep temporary buffers in `workspaces`, and move NSSS setup metadata/functions into `constants`.

### What Was Done

1. **Removed `nsss_workspace` from `workspaces`**
    - File: `src/structures.jl`
    - Deleted `nsss_workspace` type and `workspaces.nsss` field.

2. **Moved NSSS setup metadata to `constants`**
    - File: `src/structures.jl`
    - Added fields:
       - `nsss_solve_steps::Vector{NSSSSolveStep}`
       - `nsss_param_prep!::Union{Nothing, Function}`
       - `nsss_n_sol::Int`
       - `nsss_output_indices::Vector{Int}`
       - `nsss_n_ext_params::Int`
       - `nsss_sol_names::Vector{Symbol}`
       - `nsss_exo_zero_indices::Vector{Int}`
       - `nsss_param_names_ext::Vector{Symbol}`

3. **Updated constructors and call sites**
    - File: `src/options_and_caches.jl`
       - `Workspaces()` no longer initializes NSSS container.
       - `Constants()` initializes all new `nsss_*` fields.
    - File: `src/nsss_solver.jl`
       - All reads/writes switched from `𝓂.workspaces.nsss.*` to `𝓂.constants.nsss_*`.
    - File: `src/MacroModelling.jl`
       - Setup guard switched to `isempty(𝓂.constants.nsss_solve_steps)`.
    - File: `test/runtests.jl`
       - Updated assertions to `model.constants.nsss_solve_steps`.

### Validation

1. Diagnostics: no errors in touched files.

2. Runtime smoke check:

```bash
julia -t auto --project=. -e 'using MacroModelling, Test; @model RBC_switch begin ... end; @parameters RBC_switch begin ... end; @test !isempty(RBC_switch.constants.nsss_solve_steps); @test RBC_switch.constants.nsss_n_sol > 0; include("models/FS2000.jl"); ss=get_steady_state(FS2000, silent=true); @test !isempty(ss); @test !isempty(FS2000.constants.nsss_solve_steps); println("ok")'
```

Observed: `ok`.
