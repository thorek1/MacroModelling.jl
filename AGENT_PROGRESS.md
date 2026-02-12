# Agent Progress Log

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
