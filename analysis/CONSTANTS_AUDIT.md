# Constants audit

This file answers two questions (heuristically):
1. Which `constants` fields/subfields are not needed by any exported/user-facing API entrypoint?
2. Which functions appear to mutate `.constants.*` after construction (potential further-processing of constants)?

Entrypoints:
- Exported functions: 42
- Exported macros: 2 (not included in call graph)
- get_/plot_ subset: 40

## Unused by exported entrypoints (candidates)

- `post_model_macro.present_only` (no references found)
- `post_model_macro.future_not_past` (no references found)
- `post_model_macro.past_not_future` (no references found)
- `post_model_macro.present_but_not_only` (no references found)
- `post_model_macro.mixed_in_past` (no references found)
- `post_model_macro.not_mixed_in_past` (no references found)
- `post_model_macro.mixed_in_future` (no references found)
- `post_model_macro.aux_past` (no references found)
- `post_model_macro.nPresent_only` (no references found)
- `post_model_macro.nMixed` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_model_macro.nPresent_but_not_only` (no references found)
- `post_model_macro.present_only_idx` (no references found)
- `post_model_macro.present_but_not_only_idx` (no references found)
- `post_model_macro.future_not_past_and_mixed_idx` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_model_macro.not_mixed_in_past_idx` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_model_macro.mixed_in_past_idx` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_model_macro.mixed_in_future_idx` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_model_macro.past_not_future_idx` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_model_macro.reorder` (no references found)
- `post_model_macro.dynamic_order` (no references found)
- `post_model_macro.dyn_var_future_list` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_model_macro.dyn_var_present_list` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_model_macro.dyn_var_past_list` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_model_macro.dyn_ss_list` (no references found)
- `post_model_macro.dyn_exo_list` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_model_macro.dyn_future_list` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_model_macro.dyn_present_list` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_model_macro.dyn_past_list` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_parameters_macro.parameters_as_function_of_parameters` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_parameters_macro.precompile` (no references found)
- `post_parameters_macro.simplify` (no references found)
- `post_parameters_macro.guess` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_parameters_macro.ss_calib_list` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_parameters_macro.par_calib_list` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_parameters_macro.ss_no_var_calib_list` (no references found)
- `post_parameters_macro.par_no_var_calib_list` (no references found)
- `post_parameters_macro.calibration_equations_no_var` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_parameters_macro.bounds` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_complete_parameters.dyn_var_future_idx` (no references found)
- `post_complete_parameters.dyn_var_present_idx` (no references found)
- `post_complete_parameters.dyn_var_past_idx` (no references found)
- `post_complete_parameters.dyn_ss_idx` (no references found)
- `post_complete_parameters.shocks_ss` (no references found)
- `post_complete_parameters.calib_axis` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_complete_parameters.exo_axis_plain` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_complete_parameters.var_has_curly` (no references found)
- `post_complete_parameters.exo_has_curly` (no references found)
- `post_complete_parameters.all_variables` (no references found)
- `post_complete_parameters.NSSS_labels` (no references found)
- `post_complete_parameters.aux_indices` (no references found)
- `post_complete_parameters.processed_all_variables` (no references found)
- `post_complete_parameters.full_NSSS_display` (no references found)
- `post_complete_parameters.steady_state_expand_matrix` (referenced, but no exported entrypoint reaches the referencing functions)
- `post_complete_parameters.custom_ss_expand_matrix` (no references found)
- `post_complete_parameters.vars_in_ss_equations` (no references found)
- `post_complete_parameters.vars_in_ss_equations_with_aux` (no references found)
- `post_complete_parameters.SS_and_pars_names_lead_lag` (no references found)
- `post_complete_parameters.SS_and_pars_names_no_exo` (no references found)
- `post_complete_parameters.SS_and_pars_no_exo_idx` (no references found)
- `post_complete_parameters.vars_idx_excluding_aux_obc` (no references found)
- `post_complete_parameters.vars_idx_excluding_obc` (no references found)
- `second_order.kron_s_e` (no references found)
- `second_order.shock_idxs2` (no references found)
- `second_order.kron_states` (no references found)
- `second_order.I_plus_s_s` (no references found)
- `second_order.e4` (no references found)
- `third_order.shock_idxs2` (no references found)
- `third_order.shock_idxs3` (referenced, but no exported entrypoint reaches the referencing functions)
- `third_order.shockvar1_idxs` (referenced, but no exported entrypoint reaches the referencing functions)
- `third_order.shockvar2_idxs` (referenced, but no exported entrypoint reaches the referencing functions)
- `third_order.shockvar3_idxs` (referenced, but no exported entrypoint reaches the referencing functions)
- `third_order.e6` (referenced, but no exported entrypoint reaches the referencing functions)
- `third_order.kron_e_v` (referenced, but no exported entrypoint reaches the referencing functions)
- `third_order.substate_cache` (referenced, but no exported entrypoint reaches the referencing functions)
- `third_order.dependency_kron_cache` (referenced, but no exported entrypoint reaches the referencing functions)

## Used by exported entrypoints

| Field | Used by exported? | Used by get_/plot_? | #Readers | #Mutators |
|---|---:|---:|---:|---:|
| `post_model_macro` | yes | yes | 39 | 0 |
| `post_model_macro.present_only` | no | no | 0 | 0 |
| `post_model_macro.future_not_past` | no | no | 0 | 0 |
| `post_model_macro.past_not_future` | no | no | 0 | 0 |
| `post_model_macro.mixed` | yes | yes | 1 | 0 |
| `post_model_macro.future_not_past_and_mixed` | yes | yes | 1 | 0 |
| `post_model_macro.past_not_future_and_mixed` | yes | yes | 3 | 0 |
| `post_model_macro.present_but_not_only` | no | no | 0 | 0 |
| `post_model_macro.mixed_in_past` | no | no | 0 | 0 |
| `post_model_macro.not_mixed_in_past` | no | no | 0 | 0 |
| `post_model_macro.mixed_in_future` | no | no | 0 | 0 |
| `post_model_macro.var` | yes | yes | 26 | 2 |
| `post_model_macro.exo` | yes | yes | 9 | 0 |
| `post_model_macro.exo_past` | yes | yes | 2 | 0 |
| `post_model_macro.exo_present` | yes | yes | 6 | 0 |
| `post_model_macro.exo_future` | yes | yes | 2 | 0 |
| `post_model_macro.aux` | yes | yes | 5 | 0 |
| `post_model_macro.aux_present` | yes | yes | 1 | 0 |
| `post_model_macro.aux_future` | yes | yes | 1 | 0 |
| `post_model_macro.aux_past` | no | no | 0 | 0 |
| `post_model_macro.nPresent_only` | no | no | 0 | 0 |
| `post_model_macro.nMixed` | no | no | 1 | 0 |
| `post_model_macro.nFuture_not_past_and_mixed` | yes | yes | 3 | 0 |
| `post_model_macro.nPast_not_future_and_mixed` | yes | yes | 14 | 0 |
| `post_model_macro.nPresent_but_not_only` | no | no | 0 | 0 |
| `post_model_macro.nVars` | yes | yes | 7 | 0 |
| `post_model_macro.nExo` | yes | yes | 13 | 1 |
| `post_model_macro.present_only_idx` | no | no | 0 | 0 |
| `post_model_macro.present_but_not_only_idx` | no | no | 0 | 0 |
| `post_model_macro.future_not_past_and_mixed_idx` | no | no | 1 | 0 |
| `post_model_macro.not_mixed_in_past_idx` | no | no | 1 | 0 |
| `post_model_macro.past_not_future_and_mixed_idx` | yes | yes | 7 | 0 |
| `post_model_macro.mixed_in_past_idx` | no | no | 1 | 0 |
| `post_model_macro.mixed_in_future_idx` | no | no | 1 | 0 |
| `post_model_macro.past_not_future_idx` | no | no | 1 | 0 |
| `post_model_macro.reorder` | no | no | 0 | 0 |
| `post_model_macro.dynamic_order` | no | no | 0 | 0 |
| `post_model_macro.vars_in_ss_equations` | yes | yes | 3 | 0 |
| `post_model_macro.vars_in_ss_equations_no_aux` | yes | yes | 2 | 0 |
| `post_model_macro.dyn_var_future_list` | no | no | 1 | 0 |
| `post_model_macro.dyn_var_present_list` | no | no | 1 | 0 |
| `post_model_macro.dyn_var_past_list` | no | no | 1 | 0 |
| `post_model_macro.dyn_ss_list` | no | no | 0 | 0 |
| `post_model_macro.dyn_exo_list` | no | no | 1 | 0 |
| `post_model_macro.dyn_future_list` | no | no | 1 | 0 |
| `post_model_macro.dyn_present_list` | no | no | 1 | 0 |
| `post_model_macro.dyn_past_list` | no | no | 1 | 0 |
| `post_parameters_macro` | yes | yes | 8 | 0 |
| `post_parameters_macro.parameters_as_function_of_parameters` | no | no | 1 | 0 |
| `post_parameters_macro.precompile` | no | no | 0 | 0 |
| `post_parameters_macro.simplify` | no | no | 0 | 0 |
| `post_parameters_macro.guess` | no | no | 1 | 0 |
| `post_parameters_macro.ss_calib_list` | no | no | 1 | 0 |
| `post_parameters_macro.par_calib_list` | no | no | 1 | 0 |
| `post_parameters_macro.ss_no_var_calib_list` | no | no | 0 | 0 |
| `post_parameters_macro.par_no_var_calib_list` | no | no | 0 | 0 |
| `post_parameters_macro.calibration_equations_no_var` | no | no | 1 | 0 |
| `post_parameters_macro.calibration_equations` | yes | yes | 2 | 1 |
| `post_parameters_macro.calibration_equations_parameters` | yes | yes | 6 | 0 |
| `post_parameters_macro.bounds` | no | yes | 2 | 0 |
| `post_complete_parameters` | yes | yes | 24 | 0 |
| `post_complete_parameters.parameters` | yes | yes | 16 | 0 |
| `post_complete_parameters.missing_parameters` | yes | yes | 1 | 0 |
| `post_complete_parameters.dyn_var_future_idx` | no | no | 0 | 0 |
| `post_complete_parameters.dyn_var_present_idx` | no | no | 0 | 0 |
| `post_complete_parameters.dyn_var_past_idx` | no | no | 0 | 0 |
| `post_complete_parameters.dyn_ss_idx` | no | no | 0 | 0 |
| `post_complete_parameters.shocks_ss` | no | no | 0 | 0 |
| `post_complete_parameters.diag_nVars` | yes | yes | 2 | 0 |
| `post_complete_parameters.var_axis` | yes | yes | 5 | 0 |
| `post_complete_parameters.calib_axis` | no | no | 1 | 0 |
| `post_complete_parameters.exo_axis_plain` | no | no | 1 | 0 |
| `post_complete_parameters.exo_axis_with_subscript` | yes | yes | 1 | 0 |
| `post_complete_parameters.var_has_curly` | no | no | 0 | 0 |
| `post_complete_parameters.exo_has_curly` | no | no | 0 | 0 |
| `post_complete_parameters.SS_and_pars_names` | yes | yes | 1 | 0 |
| `post_complete_parameters.all_variables` | no | no | 0 | 0 |
| `post_complete_parameters.NSSS_labels` | no | no | 0 | 0 |
| `post_complete_parameters.aux_indices` | no | no | 0 | 0 |
| `post_complete_parameters.processed_all_variables` | no | no | 0 | 0 |
| `post_complete_parameters.full_NSSS_display` | no | no | 0 | 0 |
| `post_complete_parameters.steady_state_expand_matrix` | no | no | 1 | 0 |
| `post_complete_parameters.custom_ss_expand_matrix` | no | no | 0 | 0 |
| `post_complete_parameters.vars_in_ss_equations` | no | no | 0 | 0 |
| `post_complete_parameters.vars_in_ss_equations_with_aux` | no | no | 0 | 0 |
| `post_complete_parameters.SS_and_pars_names_lead_lag` | no | no | 0 | 0 |
| `post_complete_parameters.SS_and_pars_names_no_exo` | no | no | 0 | 0 |
| `post_complete_parameters.SS_and_pars_no_exo_idx` | no | no | 0 | 0 |
| `post_complete_parameters.vars_idx_excluding_aux_obc` | no | no | 0 | 0 |
| `post_complete_parameters.vars_idx_excluding_obc` | no | no | 0 | 0 |
| `post_complete_parameters.initialized` | yes | yes | 1 | 0 |
| `post_complete_parameters.dyn_index` | yes | yes | 1 | 0 |
| `post_complete_parameters.reverse_dynamic_order` | yes | yes | 1 | 0 |
| `post_complete_parameters.comb` | yes | yes | 1 | 0 |
| `post_complete_parameters.future_not_past_and_mixed_in_comb` | yes | yes | 1 | 0 |
| `post_complete_parameters.past_not_future_and_mixed_in_comb` | yes | yes | 1 | 0 |
| `post_complete_parameters.Ir` | yes | yes | 1 | 0 |
| `post_complete_parameters.nabla_zero_cols` | yes | yes | 1 | 0 |
| `post_complete_parameters.nabla_minus_cols` | yes | yes | 1 | 0 |
| `post_complete_parameters.nabla_e_start` | yes | yes | 1 | 0 |
| `post_complete_parameters.expand_future` | yes | yes | 1 | 0 |
| `post_complete_parameters.expand_past` | yes | yes | 1 | 0 |
| `second_order` | yes | yes | 5 | 0 |
| `second_order.s_in_s` | yes | yes | 2 | 1 |
| `second_order.kron_s_s` | yes | yes | 1 | 1 |
| `second_order.kron_e_e` | yes | yes | 1 | 1 |
| `second_order.kron_v_v` | yes | yes | 1 | 1 |
| `second_order.kron_s_e` | no | no | 0 | 0 |
| `second_order.kron_e_s` | yes | yes | 1 | 1 |
| `second_order.shockvar_idxs` | yes | yes | 1 | 1 |
| `second_order.shock_idxs` | yes | yes | 2 | 1 |
| `second_order.shock_idxs2` | no | no | 0 | 0 |
| `second_order.kron_states` | no | no | 0 | 0 |
| `second_order.I_plus_s_s` | no | no | 0 | 0 |
| `second_order.e4` | no | no | 0 | 0 |
| `third_order` | yes | yes | 6 | 0 |
| `third_order.shock_idxs2` | no | no | 0 | 0 |
| `third_order.shock_idxs3` | no | no | 2 | 1 |
| `third_order.shockvar1_idxs` | no | no | 2 | 1 |
| `third_order.shockvar2_idxs` | no | no | 2 | 1 |
| `third_order.shockvar3_idxs` | no | no | 2 | 1 |
| `third_order.e6` | no | no | 1 | 1 |
| `third_order.kron_e_v` | no | no | 1 | 1 |
| `third_order.substate_cache` | no | no | 1 | 0 |
| `third_order.dependency_kron_cache` | no | no | 1 | 0 |

## Post-construction mutations of constants (review)

These are places where code assigns into `.constants.* = ...` inside a function body.
This can indicate that constants hold intermediate data that can be further processed without new inputs, or that lazy caches are populated.

| Function | Location |
|---|---:|
| `rrule` | src/MacroModelling.jl:362 |
| `rrule` | src/MacroModelling.jl:362 |
| `rrule` | src/MacroModelling.jl:621 |
| `rrule` | src/MacroModelling.jl:621 |
| `solve_ss` | src/MacroModelling.jl:2381 |
| `calculate_second_order_stochastic_steady_state` | src/MacroModelling.jl:2638 |
| `rrule` | src/filter/inversion.jl:207 |
| `ensure_computational_constants_cache!` | src/options_and_caches.jl:370 |
| `ensure_computational_constants_cache!` | src/options_and_caches.jl:375 |
| `ensure_computational_constants_cache!` | src/options_and_caches.jl:376 |
| `ensure_computational_constants_cache!` | src/options_and_caches.jl:377 |
| `ensure_computational_constants_cache!` | src/options_and_caches.jl:378 |
| `ensure_computational_constants_cache!` | src/options_and_caches.jl:379 |
| `ensure_computational_constants_cache!` | src/options_and_caches.jl:380 |
| `ensure_computational_constants_cache!` | src/options_and_caches.jl:419 |
| `ensure_computational_constants_cache!` | src/options_and_caches.jl:424 |
| `ensure_computational_constants_cache!` | src/options_and_caches.jl:425 |
| `ensure_computational_constants_cache!` | src/options_and_caches.jl:426 |
| `ensure_computational_constants_cache!` | src/options_and_caches.jl:427 |
| `ensure_computational_constants_cache!` | src/options_and_caches.jl:428 |
| `ensure_computational_constants_cache!` | src/options_and_caches.jl:429 |
| `ensure_conditional_forecast_index_cache!` | src/options_and_caches.jl:475 |
| `ensure_conditional_forecast_index_cache!` | src/options_and_caches.jl:477 |
| `ensure_conditional_forecast_index_cache!` | src/options_and_caches.jl:478 |
| `ensure_conditional_forecast_index_cache!` | src/options_and_caches.jl:479 |
| `ensure_conditional_forecast_index_cache!` | src/options_and_caches.jl:524 |
| `ensure_conditional_forecast_index_cache!` | src/options_and_caches.jl:526 |
| `ensure_conditional_forecast_index_cache!` | src/options_and_caches.jl:527 |
| `ensure_conditional_forecast_index_cache!` | src/options_and_caches.jl:528 |
| `ensure_moments_cache!` | src/options_and_caches.jl:774 |
| `ensure_moments_cache!` | src/options_and_caches.jl:777 |

## Notes

- This audit is static and heuristic; macros and dynamic dispatch can hide true reachability.
- If a field is only used during model construction via macros, it may appear as unused-by-exported here.
