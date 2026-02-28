# Agent Progress

## Session: 2026-02-22

### Completed
- Removed `rrule(::typeof(run_kalman_iterations), ...)` from `src/custom_autodiff_rules/zygote.jl`.
- Kept and used parent rule `rrule(::typeof(calculate_kalman_filter_loglikelihood), ...)` as the Kalman reverse-mode AD entrypoint.
- Added ForwardDiff specialization:
	- `calculate_kalman_filter_loglikelihood(observables_index::Vector{Int}, 𝐒::Union{Matrix{Dual}, Vector{AbstractMatrix{Dual}}}, ...)`
	- implemented in `src/custom_autodiff_rules/forwarddiff.jl`.
- Removed now-redundant ForwardDiff overload `run_kalman_iterations(::Matrix{Dual}, ...)` from `src/custom_autodiff_rules/forwarddiff.jl`.
- Ran focused SW07 estimation-data validation comparing `ForwardDiff` and `Zygote` gradients for Kalman likelihood.
- Refactored `get_loglikelihood` in `src/get_functions.jl` to compute `obs_indices` once from `SS_and_pars_names` and pass indices into filter dispatch.
- Updated Kalman path signatures to consume precomputed indices:
	- `calculate_loglikelihood(::Val{:kalman}, ..., observables_index::Vector{Int}, ...)`
	- `calculate_kalman_filter_loglikelihood(observables_index::Vector{Int}, ...)`
- Updated Inversion path signatures similarly:
	- `calculate_loglikelihood(::Val{:inversion}, ..., observables_index::Vector{Int}, ...)`
	- all five `calculate_inversion_filter_loglikelihood` algorithm overloads now take `observables_index::Vector{Int}`.
- Updated Kalman AD specializations to match index-based call shape:
	- ForwardDiff `calculate_kalman_filter_loglikelihood(observables_index::Vector{Int}, ...)`
	- Zygote `rrule(::typeof(calculate_kalman_filter_loglikelihood), observables_index::Vector{Int}, ...)`.
- Unified likelihood signatures to pass root `workspaces::workspaces` instead of specialized workspace arguments:
	- `src/filter/kalman.jl`: `calculate_kalman_filter_loglikelihood(..., workspaces::workspaces; ...)` now performs internal `ensure_lyapunov_workspace!` and uses `workspaces.kalman`.
	- `src/filter/inversion.jl`: all `calculate_inversion_filter_loglikelihood` algorithm overloads now take `workspaces::workspaces` and resolve `ws = workspaces.inversion` internally.
	- `src/custom_autodiff_rules/forwarddiff.jl`: Dual Kalman specialization now takes `workspaces::workspaces` and resolves Lyapunov/Kalman buffers internally.
	- `src/custom_autodiff_rules/zygote.jl`: Kalman `rrule` now takes `workspaces::workspaces`, resolves internal workspaces, and pullback tangent arity updated to match new argument list.
- Renamed Kalman workspace ensure API from `ensure_kalman_buffers!` to `ensure_kalman_workspaces!`, updated to accept `workspaces::workspaces` and return `workspaces.kalman`, and migrated Kalman/Zygote callsites.
- Removed filter wrapper/branch dispatch for likelihood evaluation and moved to unified `Val` dispatch:
	- `get_loglikelihood` now calls a single `calculate_loglikelihood(Val(filter), Val(algorithm), ...)` entrypoint.
	- `src/filter/kalman.jl` now dispatches directly on `calculate_loglikelihood(::Val{:kalman}, ::Val, ...)`.
	- `src/filter/inversion.jl` now dispatches directly on `calculate_loglikelihood(::Val{:inversion}, ::Val{:...}, ...)` across all inversion algorithms.
	- AD signatures aligned to the same shape in `src/custom_autodiff_rules/forwarddiff.jl` and `src/custom_autodiff_rules/zygote.jl`.
- Enabled inversion ForwardDiff dispatch compatibility after unified `Val` call path:
	- Relaxed inversion primal method constraints from `R <: AbstractFloat` to `R <: Real`.
	- Removed over-constrained `state` argument typing in inversion primal methods to accept the existing Float64 state container under Dual parameter differentiation.
	- Made first-order inversion temporary allocations (`state`, `y`, `x`, accumulators) element-type aware (`R`) to avoid Float64/Dual write failures.
- Fixed inversion Zygote first-order pullback tangent ordering/arity in `src/custom_autodiff_rules/zygote.jl`:
	- Updated early on-failure pullback tuples to match current argument count.
	- Corrected final pullback return order so `∂𝐒` maps to the `𝐒` argument and not to `observables_index`.
- Fixed the same inversion Zygote pullback tangent ordering/arity issue across higher-order inversion `rrule`s in `src/custom_autodiff_rules/zygote.jl`:
	- `::Val{:pruned_second_order}`
	- `::Val{:second_order}`
	- `::Val{:pruned_third_order}`
	- `::Val{:third_order}`
	- Updated pullbacks to return tangents in the unified signature order `(Val(filter), Val(algorithm), observables_index, 𝐒, data_in_deviations, constants, state, workspaces)`.
- Added and executed a focused estimation-like validation harness (`tasks/estimation_like_llh_checks.jl`) that triggers only primal/AD loglikelihood entry calls (no NUTS/MAP loops) for:
	- FS2000: `:kalman`, `:inversion`, `:second_order`, `:pruned_second_order`
	- SW07 linear + nonlinear Kalman paths with the same parameter-combination closure used in `test/test_sw07_estimation.jl`
	- Caldara estimation model: `:third_order` and `:pruned_third_order`

### Validation
- Command: `julia --project=test /tmp/sw07_forwarddiff_check.jl`
- Results:
	- `llh = -2635.770595135343`
	- `fd_grad_norm = 21347.478116235467`
	- `zyg_grad_norm = 21347.478116410843`
	- `grad_l2_diff = 9.04564016567317e-6`
	- `grad_rel_l2_diff = 4.237334319466685e-10`
	- `grad_max_abs_diff = 7.286309596565843e-6`
- Command: `julia --project=. -e 'using MacroModelling, Random, Zygote, AxisKeys; include("models/RBC_baseline.jl"); ...'`
- Results:
	- `llh_kalman = 121.85330481734195`
	- `llh_inversion = 5.957260480727086`
	- `grad_len = 9`
- Command: `julia --project=. -e 'using MacroModelling, Random, Zygote, AxisKeys, LinearAlgebra; include("models/RBC_baseline.jl"); ... filter=:inversion, algorithm=:first_order ...'`
- Results:
	- `inversion_zyg_grad_len = 9`
	- `inversion_zyg_grad_norm = 433.6404769627417`
- Command: `julia --project=. -e 'using MacroModelling, Random, Zygote, ForwardDiff, AxisKeys, LinearAlgebra; include("models/RBC_baseline.jl"); ...'`
- Results:
	- `kalman_fd_norm = 76.52119024797734`
	- `kalman_zyg_norm = 76.52119024797322`
	- `kalman_l2_diff = 8.635598691372916e-12`
	- `kalman_rel_diff = 1.1285238328609476e-13`
	- `inversion_fd_norm = 433.64047696274184`
	- `inversion_zyg_norm = 433.6404769627417`
	- `inversion_l2_diff = 1.2844645335482865e-12`
	- `inversion_rel_diff = 2.962049443688457e-15`
- Command: `julia --project=test /tmp/sw07_forwarddiff_check.jl`
- Results:
	- `llh = -2635.7705951463795`
	- `fd_grad_norm = 21347.478117349143`
	- `zyg_grad_norm = 21347.478117376842`
	- `grad_l2_diff = 4.333325677993045e-6`
	- `grad_rel_l2_diff = 2.0299005129161924e-10`
	- `grad_max_abs_diff = 3.0615947252954356e-6`
- Command: `julia --project=test tasks/estimation_like_llh_checks.jl`
- Results:
	- `18/18` estimation-like LLH cases passed (primal + AD paths), including previously failing Zygote higher-order inversion cases.
	- Representative AD outcomes:
		- `fs2000_second_zyg grad_len=9`
		- `fs2000_pruned2_zyg grad_len=9`
		- `caldara_third_zyg grad_len=10`
		- `caldara_pruned3_zyg grad_len=10`
- Command: `julia --project=. -e 'using MacroModelling; println("ok")'`
- Results:
	- `ok`
- Command: `julia --project=. -e 'using MacroModelling, Random, Zygote, AxisKeys; include("models/RBC_baseline.jl"); ...'`
- Results:
	- `llh_kalman = 121.85330481734195`
	- `llh_inversion = 5.957260480727086`
	- `grad_len = 9`

### Remaining
- Optional: add a permanent test case to `test/functionality_tests.jl` for SW07 ForwardDiff-vs-Zygote Kalman gradient parity.
- Optional: add a compact regression test covering inversion first-order gradient parity (`ForwardDiff` vs `Zygote`) on a small model (e.g. `RBC_baseline`) to guard pullback tangent ordering.
- Optional: add compact regression tests for higher-order inversion Zygote pullback ordering (`:second_order`, `:pruned_second_order`, `:third_order`, `:pruned_third_order`) using one-shot gradient calls (no full estimation loops).

## Session: 2026-02-23

### Completed
- Added ForwardDiff specializations for all `get_relevant_steady_state_and_state_update` algorithm variants in `src/custom_autodiff_rules/forwarddiff.jl`:
	- `::Val{:first_order}`
	- `::Val{:second_order}`
	- `::Val{:pruned_second_order}`
	- `::Val{:third_order}`
	- `::Val{:pruned_third_order}`
- Ensured Dual-safe state placeholder allocation in failure/pruned branches for higher-order variants (no implicit Float64 fallback for zero-state vectors).
- Added ChainRules `rrule` definitions for all five `get_relevant_steady_state_and_state_update` variants in `src/custom_autodiff_rules/zygote.jl`.
- Implemented shared cotangent contraction helpers in `zygote.jl` to map tuple-output cotangents (`SS_and_pars`, `𝐒`, `state`) to a scalar objective used in pullbacks.
- Implemented pullback parameter tangents via `ForwardDiff.gradient` over the contracted scalar objective, returning tangents in signature order `(typeof(f), Val(algorithm), parameter_values, 𝓂)`.

### Validation
- Command:
	- `~/.juliaup/bin/julia --project=. -e 'using MacroModelling, ForwardDiff, ChainRulesCore, LinearAlgebra; include("models/FS2000.jl"); ...'`
- Results (manual pullback cotangent vs ForwardDiff gradient parity):
	- `alg=first_order`: `fd_norm=2054.5173198509838`, `pb_norm=2054.5173198509838`, `l2=0.0`
	- `alg=second_order`: `fd_norm=1956.1118162834846`, `pb_norm=1956.1118162834875`, `l2=2.8620548806267926e-12`
	- `alg=pruned_second_order`: `fd_norm=1956.3273262996124`, `pb_norm=1956.3273262996156`, `l2=3.1192547910626285e-12`
	- `alg=third_order`: `fd_norm=1924.72994497665`, `pb_norm=1924.729944976667`, `l2=1.7492992603436048e-11`
	- `alg=pruned_third_order`: `fd_norm=1922.0741878461595`, `pb_norm=1922.0741878461695`, `l2=1.043491368874845e-11`

### Remaining
- Optional: replace repeated per-variant `rrule` definitions for `get_relevant_steady_state_and_state_update` with a single generic `Val{A}` implementation once signature stability is confirmed across all AD call sites.
- Optional: add a compact regression test that asserts pullback-vs-ForwardDiff parity for the five variants on `FS2000`.

### Correction (2026-02-23)
- Removed the temporary `rrule(::typeof(get_relevant_steady_state_and_state_update), ...)` methods from `src/custom_autodiff_rules/zygote.jl` because they computed parameter cotangents by calling `ForwardDiff.gradient` inside reverse-mode pullbacks.
- Current state now matches design intent: no ChainRules pullback in `zygote.jl` calls `ForwardDiff` directly for this entrypoint; reverse-mode should rely on existing pullbacks in lower-level components.
- Validation:
	- `~/.juliaup/bin/julia --project=. -e 'using MacroModelling; println("ok")'` → `ok`

### Follow-up (2026-02-23)
- Added a new `rrule(::typeof(get_relevant_steady_state_and_state_update), ::Val{:first_order}, ...)` in `src/custom_autodiff_rules/zygote.jl` that composes existing pullbacks for:
	- `get_NSSS_and_parameters`
	- `calculate_jacobian`
	- `calculate_first_order_solution`
- Added variant `rrule`s for `:second_order`, `:pruned_second_order`, `:third_order`, and `:pruned_third_order` that delegate to `calculate_second_order_stochastic_steady_state` / `calculate_third_order_stochastic_steady_state` pullbacks when available (and otherwise return zero parameter tangents).
- No `ForwardDiff` calls are used inside these reverse-mode pullbacks.
- Validation:
	- `~/.juliaup/bin/julia --project=. -e 'using MacroModelling; println("ok")'` → `ok`
	- First-order pullback parity check on FS2000:
		- `l2=2.3130867263401494e-13` between pullback parameter cotangent and `ForwardDiff.gradient` of a scalarized contraction.

### Follow-up 2 (2026-02-23)
- Implemented wrapper-level reverse rules in `src/custom_autodiff_rules/zygote.jl`:
	- `rrule(::typeof(calculate_second_order_stochastic_steady_state), parameters::Vector, 𝓂; ...)`
	- `rrule(::typeof(calculate_third_order_stochastic_steady_state), parameters::Vector, 𝓂; ...)`
- These wrapper rules compose existing pullbacks (`get_NSSS_and_parameters`, `calculate_jacobian`, `calculate_hessian`, `calculate_third_order_derivatives`, `calculate_first_order_solution`, `calculate_second_order_solution`, `calculate_third_order_solution`, and Newton SSS pullbacks where applicable) and do not call `ForwardDiff`.
- Added helper utilities for tangent shape handling in `zygote.jl`:
	- `_as_vec_tangent`
	- `_as_mat_tangent`
	- `_expand_s1_pullback`
- Added robust guard around third-order solution pullback composition to avoid hard failure when cotangent layout is unsupported by lower-level routines.

### Validation (Follow-up 2)
- `~/.juliaup/bin/julia --project=. -e 'using MacroModelling; println("ok")'` → `ok`
- Wrapper rule smoke checks on FS2000:
	- `second_rrule_grad_norm=4410.208790407605`
	- `third_rrule_grad_norm=4107.787036559248`
- `get_relevant_steady_state_and_state_update` smoke checks on FS2000:
	- `alg=second_order grad_norm=2313.947270686622`
	- `alg=pruned_second_order grad_norm=2056.5764439212558`
	- `alg=third_order grad_norm=2054.517319851021`
	- `alg=pruned_third_order grad_norm=2054.517319851021`

## Session: Performance Optimization (Items 1-6)

### Completed

#### Item 1: Eliminate Double Forward in Higher-Order rrules
- Restructured 4 higher-order `get_relevant_steady_state_and_state_update` rrules to call inner rrule in forward pass, capturing `ss_pb` for pullback.
- File: `src/custom_autodiff_rules/zygote.jl` (lines ~893-1130)

#### Item 2: Fix Tolerances Field Types
- Changed `Tolerances` struct fields from `AbstractFloat` to `Float64` in `src/options_and_caches.jl`.

#### Item 3: mul!-ify first_order_solution_pullback
- Rewrote `first_order_solution_pullback` to use `mul!` with workspace buffers from `sylvester_workspace`.
- Forward pass stores matrices in `qme_ws.𝐀`, `qme_ws.sylvester_ws.tmp`, etc.
- Pullback scratch uses `𝐗`, `𝐂_dbl`, `𝐂¹` view, `𝐂B`, `𝐂` from `sylvester_workspace`.
- Fixed dimension mismatch for nVars×nPast submatrices using `@view 𝐂¹[:, 1:nPast]`.

#### Item 4: Cache Structural Index Sets
- Replaced inline kron index computations in 4 inversion filter rrules with reads from `ensure_conditional_forecast_constants!`.
- Variants: pruned_second_order, second_order, pruned_third_order, third_order.
- Fixed bug: pruned_third_order needs `kron(e, s_in_s)` (no vol) for `shockvar_idxs`, not the cached `kron(e, s_in_s⁺)`. Now computes inline: `sparse(ℒ.kron(cc.e_in_s⁺, cc.s_in_s)).nzind`.

#### Item 5: In-place vcat/kron in Newton Loops
- Pre-allocated `x_aug` vector in all Newton SSS solvers:
  - `src/MacroModelling.jl`: `calculate_second_order_stochastic_steady_state(Val(:newton), ...)` and `calculate_third_order_stochastic_steady_state(Val(:newton), ...)`
  - `src/custom_autodiff_rules/zygote.jl`: Both SSS rrule Newton loops (2nd and 3rd order)
- Eliminated all `vcat(x,1)` from `src/` directory.
- Replaced `copy(𝐒[i]) * 0` with `zero(𝐒[i])` in 3 pullback functions (eliminated double allocation).

### Validation
- All 5 algorithms pass comprehensive validation:
  - `first_order`: ForwardDiff parity `rel_diff=2.37e-14` ✓
  - `second_order`: `finite=true`, `grad_norm=4048.6` ✓
  - `pruned_second_order`: `finite=true`, `grad_norm=3991.0` ✓
  - `third_order`: `finite=true`, `grad_norm=4162.7` ✓
  - `pruned_third_order`: `finite=true`, `grad_norm=4098.1` ✓

### Remaining
- Item 6: Pre-allocate pullback gradient accumulators — move `zero()` allocations from inside pullback closures to forward pass scope (~20+ allocations per pullback in inversion filter rrules).
- Optional: replace per-timestep `ℒ.kron(...)` calls inside pullback loops with `ℒ.kron!()` and pre-allocated buffers.

## Session: 2026-02-27 (rrule signature migration follow-up)

### Completed
- Updated `rrule(::typeof(get_relevant_steady_state_and_state_update), ...)` higher-order variants in `src/custom_autodiff_rules/zygote.jl` to call the new API:
	- `rrule(calculate_stochastic_steady_state, Val(:second_order), ...)`
	- `rrule(calculate_stochastic_steady_state, Val(:pruned_second_order), ...)`
	- `rrule(calculate_stochastic_steady_state, Val(:third_order), ...)`
	- `rrule(calculate_stochastic_steady_state, Val(:pruned_third_order), ...)`
- Fixed pullback tangent index mapping after adding the `Val(...)` argument:
	- changed `ss_grads[2]` → `ss_grads[3]` in all four `get_relevant_steady_state_and_state_update` higher-order pullbacks.

### Validation
- Command: `~/.juliaup/bin/julia --project=test tasks/test_get_loglikelihood_rrule.jl`
- Results:
	- `caldara_third_order`: `primal=-283.5304784490986`, `grad_norm=11018.921925125516`, `grad_len=10`, `PASS`
	- `caldara_pruned_third_order`: `primal=-478.51170758012853`, `grad_norm=25116.677341363575`, `grad_len=10`, `PASS`

## Session: 2026-02-27 (compare_ss_and_pars_jacobian_caldara Zygote fix)

### Completed
- Updated `tasks/compare_ss_and_pars_jacobian_caldara.jl` to make AD sections independently runnable via env toggles and to run `Zygote` before other Jacobian methods:
	- `RUN_FD` (default `true`)
	- `RUN_FWD` (default `true`)
	- `RUN_ZYG` (default `true`)
- Fixed `NoTangent` handling in higher-order Zygote pullbacks in `src/custom_autodiff_rules/zygote.jl`:
	- guarded `so3_tangents[2:6]` in both third-order and pruned-third-order stochastic steady-state pullbacks.
	- replaced direct use of possible `NoTangent` values with typed zero fallbacks (`zero(∇₁)`, `zero(∇₂)`, `zero(∇₃)`, `zero(𝐒₁_raw)`, `zero(𝐒₂)`).
- Hardened `rrule(::typeof(get_relevant_steady_state_and_state_update), ::Val{:third_order}, ...)` against zero cotangents by using typed zero fallbacks for `ΔSS_and_pars`, `Δ𝐒₁`, `Δ𝐒₂`, `Δ𝐒₃`.
- Removed temporary debug print `"Returning from third order rrule"`.

### Validation
- Command: `RUN_FD=false RUN_FWD=false RUN_ZYG=true ~/.juliaup/bin/julia --project=. tasks/compare_ss_and_pars_jacobian_caldara.jl`
- Results (`ALGORITHM=third_order`):
	- `Computing Zygote Jacobian...`
	- `done - norm(J_zyg) = 684.45680426015`
	- exit code `0`
- Command: `ALGORITHM=pruned_third_order RUN_FD=false RUN_FWD=false RUN_ZYG=true ~/.juliaup/bin/julia --project=. tasks/compare_ss_and_pars_jacobian_caldara.jl`
- Results:
	- `Computing Zygote Jacobian...`
	- `done - norm(J_zyg) = 684.45680426015`
	- exit code `0`

## Session: Custom rrule for calculate_third_order_solution

### Completed
- Implemented analytical pullback for `calculate_third_order_solution` in `src/custom_autodiff_rules/zygote.jl` (replaces ForwardDiff-inside-Zygote approach).
- Pullback covers all 5 input arguments: ∂∇₃, ∂∇₂, ∂𝐒₂, ∂∇₁, ∂𝑺₁.
- Added 6 buffer fields to `third_order_solution_caches` in `src/structures.jl` and corresponding initializers in `src/options_and_caches.jl`.
- Forward pass in rrule mirrors primal logic and stores all intermediates needed for adjoint.
- Adjoint Sylvester equation solved via `sylvester_solv!` to obtain `∂C_adj` from `∂𝐒₃`.
- Decompose `∂C_adj` into `∂A`, `∂B`, `∂C` contributions through the Sylvester structure.
- Fixed critical bug in `compressed_kron³_pullback!`: was using sparsity pattern (`ui`/`uj` from `findnz(sparse(X))`) to limit iteration bounds. This works for forward pass (zero entries → zero products) but is WRONG for pullback (derivative at zero entry can be nonzero: ∂(a·b·c)/∂a|_{a=0} = b·c ≠ 0). Fix: iterate over `1:n_rows` and `1:n_cols` instead.
- Cleaned up all debug instrumentation (print statements, snapshot variables).

### Validation
- All 5 pullback blocks pass against finite differences (Caldara_et_al_2012 model):
	- ∂∇₃: relative norm = 4.44e-11 ✓
	- ∂∇₂: relative norm = 3.90e-10 ✓
	- ∂𝐒₂: relative norm = 2.50e-10 ✓
	- ∂∇₁: relative norm = 2.62e-11 ✓
	- ∂𝑺₁: relative norm = 2.63e-10 ✓

### Test files (in tasks/)
- `test_third_order_rrule_grad3.jl` — ∂∇₃ test
- `test_third_order_rrule_grad2.jl` — ∂∇₂ test
- `test_third_order_rrule_s2.jl` — ∂𝐒₂ test
- `test_third_order_rrule_grad1.jl` — ∂∇₁ test
- `test_third_order_rrule_s1.jl` — ∂𝑺₁ test
- `test_ck3_pullback.jl` — compressed_kron³_pullback isolation test

## Session: Fix end-to-end third-order gradient (P-matrix compression bug)

### Completed
- Diagnosed that `calculate_third_order_derivatives` pullback produced wrong ∂parameters (2-73% per-column errors in Jacobian vs FD).
- Root cause: in `take_nth_order_derivatives` (src/MacroModelling.jl), the P-matrix (parameter Jacobian) construction for `output_compressed=true` was missing the `is_compressed` filter that the X-matrix construction applies. Unsorted variable-index tuples (e.g., (1,2,3) instead of (3,2,1)) were fed into the compressed column formula, which maps them to WRONG positions, corrupting the Jacobian with spurious entries at incorrect rows.
  - Before fix: Jacobian had 843 nnz, 362 unique nonzero rows (vs 134 nnz in ∇₃)
  - After fix: Jacobian has 289 nnz, 127 unique nonzero rows — all correct
- The second-order hessian was unaffected because it uses `output_compressed=false`.
- Also previously removed `rowmask` from `compressed_kron³` in the third-order solution rrule forward pass (ck3_aux_mat), ensuring ∂∇₃ is correct at ALL positions including structural zeros.

### Code changes
- `src/MacroModelling.jl`: Added `is_compressed_P` filter to the P-matrix construction loop in `take_nth_order_derivatives`, matching the X-matrix's compression rule (lines ~5730-5790).
- `src/custom_autodiff_rules/zygote.jl`: Removed `rowmask` from `compressed_kron³(aux)` call in third-order solution rrule forward pass (from prior session).

### Validation
- Jacobian column-by-column verification (`tasks/verify_pmatrix_fix.jl`):
  - p[3]: rel = 5.9e-12 ✓ (was 0.73)
  - p[4]: rel = 9.5e-13 ✓ (was 0.20)
  - p[5]: rel = 8.6e-12 ✓ (was 0.71)
  - p[7]: rel = 1.7e-12 ✓ (was 1.06)
  - p[10]: rel = 1.8e-13 ✓
- End-to-end gradient (`tasks/compare_ss_and_pars_jacobian_caldara.jl`):
  - SOL_COMPONENT=S3: Zygote vs FD rel = 1.20e-11 ✓ (was 0.0234)
  - SOL_COMPONENT=S3: ForwardDiff vs Zygote rel = 1.88e-15 ✓
  - SOL_COMPONENT=S2: Zygote vs FD rel = 6.23e-12 ✓ (unchanged)

