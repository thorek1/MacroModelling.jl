## Summary
- Reworked first-order index cache to avoid `Union{Nothing,...}` and added cached identity-slice matrices for `expand` usage.
- Switched `calculate_first_order_solution` to positional `𝓂` wrappers (Real and Dual) and updated call sites to stop passing caches explicitly.
- Added a runnable script for RBC baseline pruned third-order standard deviations and updated the standalone cache test to use the new initializer and positional wrapper.
- Fixed pruned mean calculation to use state-only kron indices and added a runnable script for RBC baseline pruned second-order mean.
- Fixed `get_steady_state` axis labels for models with calibration equations and added a runnable script to exercise calibration equations.
- Added a moments cache for kron selectors, `I_plus_s_s`, and `e⁴`/`e⁶`, plus substate/dependency caches, and switched `moments.jl` to use cached values.
- Extended model structure caching with steady-state selector matrices, full NSSS display names, and cached variable indices; updated steady-state and variable parsing code to use cached values.
- Added a targeted cache validation script for steady-state expansion, custom steady state mapping, and cached variable indices.
- Moved cache creation helpers into `src/options_and_caches.jl` (name display, computational constants, model structure, first-order indices, selector matrix, moments) and removed duplicate definitions from `src/MacroModelling.jl` and `src/moments.jl`.
- Refactored cache usage to call ensure helpers once per high-level call and use `𝓂.caches` in `src/moments.jl`, `src/perturbation.jl`, `src/get_functions.jl`, and filter paths; updated steady-state/model-structure callers to use cached selectors.

## Tests
- `TEST_SET=basic julia --project -e 'using Pkg; Pkg.test()'` failed: unsatisfiable requirements during Pkg resolve (DynamicPPL/Pigeons/Turing/JET constraints).
- `julia --project -e 'using Test; include("test/test_standalone_function.jl")'` failed: `FiniteDifferences` not found in current environment.
- `julia --project run_rbc_pruned_third_order_std.jl` succeeded (prints `get_std` output for `RBC_baseline`, algorithm `:pruned_third_order`).
- Re-ran `julia --project run_rbc_pruned_second_order_mean.jl`: succeeded (prints `get_mean` output for `RBC_baseline`, algorithm `:pruned_second_order`).
- Re-ran `julia --project run_rbc_pruned_third_order_std.jl`: succeeded (prints `get_std` output for `RBC_baseline`, algorithm `:pruned_third_order`).
- Re-ran `julia --project run_rbc_pruned_third_order_std.jl`: succeeded (prints `get_std` output for `RBC_baseline`, algorithm `:pruned_third_order`).
- `julia --project run_rbc_pruned_second_order_mean.jl` succeeded (prints `get_mean` output for `RBC_baseline`, algorithm `:pruned_second_order`).
- `julia --project run_rbc_pruned_third_order_std.jl` succeeded (prints `get_std` output for `RBC_baseline`, algorithm `:pruned_third_order`).
- `julia --project run_cache_steady_state_checks.jl` failed: `get_relevant_steady_states` not exported (used unqualified call).
- `julia --project run_cache_steady_state_checks.jl` succeeded after qualifying `MacroModelling.get_relevant_steady_states`.
- `julia --project run_rbc_calibration_equations_ss.jl` succeeded (prints calibration equations, calibrated parameter values, and steady state for `RBC_CME_calibration_equations`).
- `julia --project -e 'using MacroModelling; include("models/RBC_baseline.jl"); MacroModelling.calculate_second_order_moments(RBC_baseline.parameter_values, RBC_baseline);'` failed: `calculate_hessian` method mismatch (hessian functions not initialized for second-order).
- `julia --project -e 'using MacroModelling; include("models/RBC_baseline.jl"); solve!(RBC_baseline); MacroModelling.calculate_second_order_moments(RBC_baseline.parameter_values, RBC_baseline);'` failed: same `calculate_hessian` method mismatch.
- `julia --project -e 'using MacroModelling; include("models/RBC_baseline.jl"); solve!(RBC_baseline; algorithm = :second_order); MacroModelling.calculate_second_order_moments(RBC_baseline.parameter_values, RBC_baseline);'` succeeded.
- `julia --project run_rbc_pruned_second_order_mean.jl` succeeded (prints `get_mean` output for `RBC_baseline`, algorithm `:pruned_second_order`).
- `julia --project run_rbc_pruned_second_order_mean.jl` succeeded (prints `get_mean` output for `RBC_baseline`, algorithm `:pruned_second_order`).
- `julia --project run_rbc_pruned_third_order_std.jl` succeeded (prints `get_std` output for `RBC_baseline`, algorithm `:pruned_third_order`).
- `julia --project run_cache_steady_state_checks.jl` succeeded (prints steady-state cache checks for `RBC_baseline` and `RBC_switch`).

## Remaining
- Resolve test environment dependencies and rerun tests.
