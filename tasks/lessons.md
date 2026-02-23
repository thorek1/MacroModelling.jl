# Lessons

- Once a parent-level AD rule exists for Kalman loglikelihood, keeping a nested `run_kalman_iterations` reverse rule is usually redundant and can be removed to reduce maintenance surface.
- For ForwardDiff paths using `initial_covariance = :diagonal`, promote constant covariance matrices to Dual-valued arrays explicitly so downstream Dual-typed Kalman recursion remains type-stable.
- A direct SW07 data-based gradient comparison (`ForwardDiff` vs `Zygote`) is a reliable regression check after Kalman AD refactors.
- If `get_loglikelihood` already computes observable positions against `SS_and_pars_names`, pass those indices through filter dispatch (`calculate_loglikelihood`) instead of remapping names in each Kalman/Inversion backend.
- Keep likelihood and AD entrypoint signatures aligned on `workspaces::workspaces`; resolve specialized buffers (`ensure_lyapunov_workspace!`, `workspaces.kalman`, `workspaces.inversion`) inside the concrete likelihood functions to reduce dispatch drift and argument-order bugs.
- Prefer workspace-root ensure APIs for shared subsystems (e.g. Kalman) so callsites return the concrete sub-workspace from one canonical entrypoint and avoid mixed direct/sub-workspace initialization patterns.
- For unified `Val` dispatch (`calculate_loglikelihood(Val(filter), Val(algorithm), ...)`), keep positional argument order and keyword sets identical across primal, ForwardDiff, and Zygote `rrule` methods; even one missing keyword or tangent slot causes runtime AD failures.
- After changing `rrule` positional signatures, re-check pullback return tuple ordering/length against `ChainRulesCore` conventions: one missing `NoTangent()` can silently shift tangents onto wrong arguments (e.g. `∂data` routed into `∂𝐒`) and only surface later as matrix-dimension errors upstream.
- In quick Julia validation harnesses that use `do` blocks, helper signatures must accept function arguments first (or call without `do` syntax); otherwise failures can be masked as unrelated runtime errors in the harness itself.
- For tuple-valued wrappers (e.g. `get_relevant_steady_state_and_state_update`), a robust generic pullback pattern is to contract output cotangents to a scalar and differentiate that scalar with `ForwardDiff.gradient`; this avoids manually deriving and maintaining large case-by-case VJPs.
- When adding ForwardDiff support for higher-order variants, ensure failure-path placeholder arrays use the active numeric element type (e.g. Dual) rather than default `Float64` to prevent silent type-instability and dropped derivatives.
- Do not call `ForwardDiff.gradient` from inside ChainRules reverse-mode pullbacks (`rrule` methods in `zygote.jl`); reverse-mode rules should compose existing pullbacks in the call graph instead of nesting forward-mode AD inside VJPs.
- When caching kron-based index sets (e.g. `shockvar_idxs` in `ensure_computational_constants!`), verify that the basis vectors match each consumer. The pruned variants use `s_in_s` (states without volatility) while the cache computes `kron(e_in_s⁺, s_in_s⁺)` (with volatility). Always compare the original inline computation against the cached version before substituting.
- `replace_string_in_file` cannot match strings containing multibyte Unicode characters (Δ, ∂, ℒ, x̂, B̂, Ĉ, etc.). Use Python `open(file, 'r', encoding='utf-8')` with line-index-based replacement for reliable editing of Julia files with mathematical Unicode identifiers.
- `copy(X) * 0` is doubly wasteful — it allocates the copy then multiplies by zero. Use `zero(X)` instead for a single allocation.

