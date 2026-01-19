# Agent Progress Log

## 2026-01-19

### In Progress (2026-01-19)

- Migrating parameter/calibration constants to `post_parameters_macro` across the codebase (partial; compatibility accessors added).

### Completed (2026-01-19)

- Removed `Base.getproperty`/`Base.setproperty!` overrides on `ℳ` and updated call sites to use `post_parameters_macro` fields directly.
- Removed `Base.getproperty`/`Base.setproperty!` overrides on `constants`; inlined cache fields into `post_parameters_macro` and moved computational/auxiliary caches to `constants`.
- Updated `options_and_caches`, `get_functions`, `inspect`, and `dynare` to use `post_parameters_macro` fields.
- Updated `MacroModelling.jl` in several key sections (model display, symbol creation, SS check, parameter input handling) to use `post_parameters_macro`.
- Updated steady-state docs to reference `post_parameters_macro.calibration_equations_parameters`.
- Updated `options_and_caches`, `perturbation`, `moments`, `filter`, `get_functions`, and tests to use inlined cache fields and constants-level caches.
- Replaced prefixed cache fields with unprefixed fields in `post_complete_parameters` and moved completion-time caches out of `post_parameters_macro`.
- Rebuilt `@parameters` macro output to recreate `post_parameters_macro` and `post_complete_parameters` immutably, and fixed `guess_dict` initialization order.
- Renamed auxiliary matrix caches to `second_order`/`third_order`, moved higher-order cache fields into those structs, dissolved `moments_cache`/`conditional_forecast_index_cache`/`computational_constants_cache`, and added parametric axes plus `diag_nVars` to `post_complete_parameters`.

### Tests (2026-01-19)

- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; ss = get_steady_state(RBC); @assert length(ss) > 0; println("ok")'`
- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; ss = get_steady_state(RBC); @assert length(ss) > 0; println("ok")'`
- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; ss = get_steady_state(RBC); @assert length(ss) > 0; println("ok")'`
- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; ss = get_steady_state(RBC); @assert length(ss) > 0; println("ok")'`
- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; ss = get_steady_state(RBC); @assert length(ss) > 0; println("ok")'`
- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; ss = get_steady_state(RBC); @assert length(ss) > 0; println("ok")'`

## 2026-01-18

### Completed (2026-01-18)

- Renamed the `model` struct to `post_model_macro` and updated all direct type/constructor references.
- Verified all call sites reference `post_model_macro` and no `model` type references remain.
- Moved model-macro constant lists (`vars_in_ss_equations`, `dyn_var_*`, `dyn_*`) into `post_model_macro` and removed redundant fields from `ℳ`.
- Updated caches and call sites to read constants from `post_model_macro` (including tests).
- Restored `vars_in_ss_equations` field in `model_structure_cache` and aligned cache initialization with `post_model_macro`.
- Fixed standalone test imports and aligned `post_model_macro` usage in standalone tests.
- Adjusted steady-state indexing to exclude `➕_vars` in `get_steady_state`.
- Fixed dynare export to use the passed model argument.
- Updated FS2000 custom steady state to return full `vars_in_ss_equations` (including `➕` auxiliaries) in correct order.
- Added no-aux steady-state variable list to `post_model_macro` and caches; custom steady state functions now use no-aux list.
- Updated FS2000 custom steady state to exclude auxiliary `➕` variables.

### Tests (2026-01-18)

- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; ss = get_steady_state(RBC); @assert length(RBC.constants.post_model_macro.vars_in_ss_equations) > 0; @assert length(RBC.constants.post_model_macro.dyn_var_present_list) > 0; println("ok")'`
- `julia -t auto --project=test test/test_standalone_function.jl`
- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; write_mod_file(RBC); println("dynare export ok")'`
- `julia -t auto --project=test -e 'using MacroModelling; include("models/FS2000.jl"); get_steady_state(FS2000, steady_state_function = FS2000_custom_steady_state_function!); println("custom ss ok")'`
- `julia -t auto --project=test -e 'using MacroModelling; include("models/FS2000.jl"); get_steady_state(FS2000, steady_state_function = FS2000_custom_steady_state_function!); println("custom ss ok")'`

### Notes

- No prior AGENT_PROGRESS.md existed in the workspace; created this log.
