# Agent Progress

Current task: CI fixes for PR #262 (optim_LFI_alloc branch) — completed.

### CI Fixes for PR #262

**Problem:** 18+ CI checks failing due to missing imports, API changes, and Julia 1.10 Mooncake incompatibility.

**5 Error Categories Fixed:**

1. **Missing `using Test` + imports** in 7 estimation test files — added `using Test`, `import FiniteDifferences`, `import LinearAlgebra as ℒ`
2. **Missing `clear_solution_caches!` import** in 7 higher_order/plots test files — added `import MacroModelling: clear_solution_caches!`
3. **VarNamedTuple iterate error** in 2 files — changed `mode_estimateLBFGS.params |> collect` → `collect(mode_estimateLBFGS.params.all_params)`
4. **Mooncake compilation failure on Julia 1.10** — wrapped `AutoMooncake` gradient calls in `if VERSION >= v"1.11"` guards in test_standalone_function.jl
5. **JET unsatisfiable on Julia 1.13** — external dependency issue, cannot fix in repo

**Files modified:**
- test/test_estimation.jl, test/test_sw07_estimation.jl, test/test_1st_order_inversion_filter_estimation.jl, test/test_2nd_order_estimation.jl, test/test_3rd_order_estimation.jl, test/test_pruned_2nd_order_estimation.jl, test/test_pruned_3rd_order_estimation.jl (Fix 1)
- test/test_higher_order_1.jl, test/test_higher_order_2.jl, test/test_higher_order_3.jl, test/test_plots_1.jl, test/test_plots_2.jl, test/test_plots_3.jl, test/test_plots_4.jl (Fix 2)
- test/test_3rd_order_estimation.jl, test/test_pruned_3rd_order_estimation.jl (Fix 3)
- test/test_standalone_function.jl (Fix 4)

## Completed

### OBC code extraction to src/obc.jl

**Task:** Move all OBC-related functions from MacroModelling.jl into a separate src/obc.jl file.

**Changes:**
- Created `src/obc.jl` (~650 lines) with 16 functions:
  - Parsing: `check_for_minmax`, `transform_obc`, `parse_occasionally_binding_constraints`, `write_obc_violation_equations`
  - OBC flag: `process_ignore_obc_flag`
  - Violation setup: `set_up_obc_violation_function!`
  - NLopt callbacks: `obc_objective_optim_fun`, `obc_constraint_optim_fun`
  - Analytical Jacobian: `compute_obc_analytical_jacobian!`, `_obc_dYdx_first_order!`, `_obc_dYdx_nonpruned_higher!`, `_obc_dYdx_pruned!`, `_fill_obc_constraint_jacobian!`
  - Solution: `calculate_first_order_obc_solution!`
  - State update: `obc_state_update` (standalone with explicit `𝓂, algorithm` args)
- Removed all moved functions from `src/MacroModelling.jl` (~650 lines removed)
- Added `include("obc.jl")` after `nsss_solver.jl`, before `macros.jl`
- Replaced 40-line `obc_state_update` closure in `compute_irf_responses` with 1-line lambda delegating to standalone function

**Verified:**
- Analytical Jacobian still matches finite differences to 1.03e-11
- OBC IRFs with binding ZLB (5 periods) compute correctly on Galí 2015 model
- Model parsing, constraint detection, and ignore_obc mode all work

### OBC Analytical Jacobian (replaces central finite differences)

**Problem:** The OBC constraint Jacobian for NLopt's `LD_SLSQP` was computed via central finite differences (2n function evaluations). User requested analytical or Symbolics-based Jacobian.

**Solution:** Analytical Jacobian derived from the perturbation solution structure:
- First-order: Y is linear in x → dY/dx propagated through Ŝ₁ matrix
- Second/third order (non-pruned): JVP through Kronecker product derivatives
- Pruned second/third order: component-wise JVP with separate y₁, y₂, y₃ tracking

**Changes:**
- `src/structures.jl` — added `obc_constraint_info::Vector{Tuple{Int, Int, Float64}}` field to `model_functions`
- `src/macros.jl` — added empty init to constructor
- `src/MacroModelling.jl`:
  - Replaced `obc_constraint_optim_fun` finite-diff block with call to `compute_obc_analytical_jacobian!`
  - Added helper functions: `_obc_dYdx_first_order!`, `_obc_dYdx_nonpruned_higher!`, `_obc_dYdx_pruned!`, `_fill_obc_constraint_jacobian!`
  - Extended `set_up_obc_violation_function!` to extract and store (left_row, right_row, sign) constraint metadata from χᵒᵇᶜ variable pairing

**Verified:**
- Analytical Jacobian matches finite differences to 8.66e-12 (machine precision) on Galí ZLB model
- OBC IRFs with binding ZLB constraint compute correctly

### ForwardDiff → Extension migration (all 5 phases done)

**Phase 1 – `primal()` helper + cache stamps:**
- Added `primal(x::Real) = x` helper to `src/MacroModelling.jl` (~L495).
- Replaced 6 `ℱ.value`/`ℱ.Dual` cache-stamp sites across `src/perturbation.jl` (L224, L406, L672), `src/nsss_solver.jl` (L1816-1817, L1984), and `src/MacroModelling.jl` (L9006) with `Float64.(primal.(...))`.

**Phase 2 – OBC finite-difference Jacobian:**
- Replaced `ℱ.jacobian` call in `obc_constraint_optim_fun` (src/MacroModelling.jl L869-891) with central finite-difference Jacobian (`h = cbrt(eps(S))`).

**Phase 3 – Create `ext/ForwardDiffExt.jl`:**
- Moved all 10 Dual-number method overloads from `src/custom_autodiff_rules/forwarddiff.jl` into new extension module `ext/ForwardDiffExt.jl`.
- Extension imports ~35 symbols from MacroModelling (types, functions, constants).
- Added `MacroModelling.primal(x::ℱ.Dual) = ℱ.value(x)` in extension.

**Phase 4 – Project.toml + core import removal:**
- Moved ForwardDiff from `[deps]` to `[weakdeps]` in Project.toml.
- Added `ForwardDiffExt = "ForwardDiff"` to `[extensions]`.
- Added ForwardDiff to `[extras]` and `[targets]` test.
- Commented out `import ForwardDiff as ℱ` and `include("./custom_autodiff_rules/forwarddiff.jl")` in `src/MacroModelling.jl`.

**Phase 5 – Verification:**
- Core loads without ForwardDiff; extension triggers when ForwardDiff is loaded.
- `primal(1.0) = 1.0` (core) and `primal(Dual(3.0,1.0)) = 3.0` (extension) both work.
- All 5 key Dual-method specializations registered (solve_sylvester_equation, solve_lyapunov_equation, get_NSSS_and_parameters, calculate_first_order_solution, calculate_loglikelihood).
- `ForwardDiff.gradient` through `get_NSSS_and_parameters` returns correct finite gradient.
- Core model operations (steady state, moments, IRFs) work without ForwardDiff loaded.

## Files modified

- `src/MacroModelling.jl` — added `primal()`, replaced cache stamps, replaced OBC Jacobian, removed ForwardDiff imports
- `src/perturbation.jl` — 3 cache stamp replacements
- `src/nsss_solver.jl` — 2 cache stamp replacements
- `ext/ForwardDiffExt.jl` — NEW: all 10 Dual-number method overloads
- `Project.toml` — ForwardDiff moved from [deps] to [weakdeps], extension registered

## Notes

- `src/custom_autodiff_rules/forwarddiff.jl` is no longer included but still exists on disk (dead code, can be deleted).
- **WARNING:** Do NOT use `Pkg.rm("ForwardDiff")` — it removes ForwardDiff from ALL sections including [weakdeps], [extras], [targets].
- rrules in `src/custom_autodiff_rules/rrules.jl` remain in core (ChainRulesCore doesn't depend on ForwardDiff).
