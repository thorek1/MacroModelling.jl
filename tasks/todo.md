# Task TODO

- [x] Sweep changed branch areas for docstring/comment drift vs current logic.
- [x] Patch stale comments in filter code paths.
- [x] Run focused validation (`using MacroModelling`).
- [x] Record session progress and lessons.
- [x] Second pass on `docs/src` for cache-validity wording (`valid_for`) consistency.
- [x] Add standalone `rrule(::typeof(calculate_kalman_filter_loglikelihood), ...)` that inlines Kalman forward/reverse logic instead of relying on `run_kalman_iterations` AD rule.
- [x] Validate package load and new rrule method registration (`using MacroModelling`; `ChainRulesCore.rrule` method scan).
- [x] Remove `rrule(::typeof(run_kalman_iterations), ...)` after introducing parent Kalman loglikelihood rule.
- [x] Add ForwardDiff specialization for `calculate_kalman_filter_loglikelihood`.
- [x] Validate SW07 Kalman loglikelihood gradients with `ForwardDiff` vs `Zygote`.
- [x] Refactor `get_loglikelihood` to compute `obs_indices` once and pass indices to Kalman/Inversion loglikelihood paths.
- [x] Re-run focused parity/smoke check after index-plumbing and workspace-signature refactor.
- [x] Rename `ensure_kalman_buffers!` to `ensure_kalman_workspaces!` and route Kalman workspace allocation through root `workspaces`.
- [x] Remove filter if/else wrapper and use unified `calculate_loglikelihood(Val(filter), Val(algorithm), ...)` dispatch with aligned Kalman/Inversion AD signatures.
- [x] Validate AD paths for both filters (`:kalman`, `:inversion`) with both `ForwardDiff` and `Zygote` on `RBC_baseline`.
- [x] Fix inversion first-order AD dispatch/type constraints for ForwardDiff dual parameters.
- [x] Fix inversion first-order Zygote pullback tangent ordering/arity after unified call-signature migration.
- [x] Re-run SW07 ForwardDiff-vs-Zygote Kalman parity check after inversion AD fixes.
- [x] Add and run estimation-like LLH harness that triggers primal/AD calls without running full estimation loops.
- [x] Fix higher-order inversion Zygote pullback tangent ordering for `:second_order`, `:pruned_second_order`, `:third_order`, and `:pruned_third_order`.
- [x] Re-run estimation-like harness and confirm all targeted cases pass.
- [x] Add ForwardDiff specializations for `get_relevant_steady_state_and_state_update` across `:first_order`, `:second_order`, `:pruned_second_order`, `:third_order`, and `:pruned_third_order`.
- [x] Remove temporary Zygote `rrule` definitions for `get_relevant_steady_state_and_state_update` that differentiated via `ForwardDiff.gradient`.
- [x] Implement `rrule(::typeof(get_relevant_steady_state_and_state_update), ...)` without calling `ForwardDiff` inside pullbacks (first-order fully chained; higher-order variants delegate to stochastic steady-state pullbacks when available).
- [x] Implement wrapper `rrule`s for `calculate_second_order_stochastic_steady_state(parameters, 𝓂; ...)` and `calculate_third_order_stochastic_steady_state(parameters, 𝓂; ...)` by composing existing pullbacks.
- [x] Validate pullback parameter cotangents against ForwardDiff gradients on FS2000 for all five variants.
- [x] Make `tasks/compare_ss_and_pars_jacobian_caldara.jl` runnable in Zygote-only mode (env toggles for FD/FWD/ZYG and run Zygote first).
- [x] Fix higher-order Zygote pullbacks to handle `NoTangent` safely before matrix slice assignments (`third_order`/`pruned_third_order`).

## Performance Optimization (Items 1-6)

- [x] Item 1: Eliminate double forward in higher-order `get_relevant_steady_state_and_state_update` rrules — call inner rrule in forward pass, capture `ss_pb` for pullback.
- [x] Item 2: Fix `Tolerances` struct field types from `AbstractFloat` to `Float64` for type stability.
- [x] Item 3: Rewrite `first_order_solution_pullback` to use `mul!` with workspace buffers from `sylvester_workspace`.
- [x] Item 4: Cache structural kron index sets — replace inline `kron` index computations in 4 inversion filter rrules with reads from `ensure_conditional_forecast_constants!`.
- [x] Item 5: Eliminate `vcat(x,1)` allocations in Newton loops — pre-allocate `x_aug` in SSS Newton solvers (MacroModelling.jl and zygote.jl SSS rrules). Also replace `copy()*0` with `zero()`.
- [ ] Item 6: Pre-allocate pullback gradient accumulators — move `zero()` allocations from inside pullback closures to forward pass scope.

## Compressed Third-Order Solution (see tasks/plan.md)

- [x] Step 1: `compressed_kron_sigma` — B_σ in compressed b₃ space
- [x] Step 2: `compressed_kron³_vec` — compressed cubic Kronecker product for vectors
- [x] Step 3: B matrix in `calculate_third_order_solution` uses compressed functions
- [x] Step 4: `∇₃_kron_sigma_compressed!` — ∇₃ contribution in compressed space
- [x] Step 5: Precompute `𝐏𝐂₃ = 𝐏 * 𝐂₃` and use in C assembly
- [x] Step 6: Apply sparsity micro-optimizations (BitSet mask checks in `compressed_kron³`, CSC `colptr` rowmask extraction in primal third-order path)
- [x] Step 6b: Cache `triple_lookup_∇` in `higher_order_workspace` and reuse it in primal + AD third-order paths (validated parity + benchmark rerun)
- [ ] Step 7: Update downstream `𝐒₃ * 𝐔₃` decompression sites (14+ sites in MacroModelling.jl, get_functions.jl, moments.jl, zygote.jl, filter code)
- [ ] Step 8: Remove cubic-size auxiliary matrices from constants struct
- [x] Step 9: Update Zygote rrules for third-order forward-pass changes (compressed adjoint functions validated on Caldara model)
- [ ] Step 10: Clean up workspace fields (tmpkron0, tmpkron22 no longer needed)
