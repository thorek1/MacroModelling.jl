# Plan: Custom rrule for `get_statistics`

## Goal
Add a `ChainRulesCore.rrule` for `get_statistics` that avoids Zygote tracing the function body. The rrule calls existing sub-rrules for heavy computations and applies analytical pullbacks for lightweight post-processing.

Only `parameter_values` has a nonzero cotangent; all other arguments/kwargs are non-differentiable.

## File location
All code goes in `src/custom_autodiff_rules/zygote.jl`, appended after existing rrules (before EOF).

---

## Step 1: Function signature and non-differentiable setup

Define the rrule matching the exact signature of `get_statistics` (line 3290 of `src/get_functions.jl`), including all keyword arguments. Replicate the `@ignore_derivatives`-annotated setup:
- `parse_variables_input_to_index` calls for all 6 index arrays
- `covar_groups` parsing
- `opts` construction
- `solve!()` call

These are all non-differentiable and identical to the forward pass.

## Step 2: Parameter assembly with analytical scatter pullback

Replicate:
```julia
all_parameters = vcat(other_parameter_values, parameter_values)[sort_idx]
```

Precompute the reverse-scatter index:
```julia
n_other = length(other_parameter_values)
pv_positions_in_all = sort_idx  # positions where parameter_values land after sort
# For pullback: ∂parameter_values[i] = ∂all_parameters[sort_idx[n_other + i]]
pv_gather_idx = [sort_idx[n_other + i] for i in 1:length(parameter_values)]
```

Wait — simpler: `invperm(sort_idx)` gives positions in `vcat(other, pv)` for each slot in `all_parameters`. The parameter_values occupy indices `(n_other+1):(n_other+n_pv)` in the pre-sort vector. So:
```julia
inv_sort = invperm(sort_idx)
pv_positions_in_all = inv_sort[(n_other+1):end]
```
is wrong direction. Let me think again.

`all_parameters[j] = concat[sort_idx[j]]` where `concat = vcat(other, pv)`.

For the pullback: `∂concat[sort_idx[j]] += ∂all_parameters[j]`, i.e., `∂concat = ∂all_parameters[invperm(sort_idx)]`.

Then `∂pv = ∂concat[(n_other+1):end]`.

Combined: `∂parameter_values = ∂all_parameters[invperm(sort_idx)][(n_other+1):end]`

This is a constant-index gather — no AD needed.

## Step 3: Forward pass — dispatch to sub-rrules

Based on `algorithm` and which statistics are requested, call one of 6 existing sub-rrules, storing both result and pullback closure:

| Path | Condition | Sub-rrule call | Stored pullback |
|------|-----------|----------------|-----------------|
| A | NSSS-only | `rrule(get_NSSS_and_parameters, 𝓂, all_parameters; opts)` | `nsss_pb` |
| B | `:first_order` | `rrule(calculate_covariance, all_parameters, 𝓂; opts)` | `cov_pb` |
| C | `:pruned_second_order`, mean-only | `rrule(calculate_second_order_moments, all_parameters, 𝓂; opts)` | `som_pb` |
| D | `:pruned_second_order`, with moments | `rrule(calculate_second_order_moments_with_covariance, all_parameters, 𝓂; opts)` | `somc_pb` |
| E | `:pruned_third_order`, no autocorr | `rrule(calculate_third_order_moments, all_parameters, observables, 𝓂; covariance, opts)` | `tom_pb` |
| F | `:pruned_third_order`, with autocorr | `rrule(calculate_third_order_moments_with_autocorrelation, all_parameters, observables, 𝓂; autocorrelation_periods, covariance, opts)` | `toma_pb` |

## Step 4: Post-processing for variance and standard deviation (analytical pullback)

**Forward:**
```julia
d = diag(covar_dcmp)
varrs = max.(d, eps())                     # used if variance requested
st_dev = sqrt.(abs.(max.(d, eps())))       # used if std_dev requested
```

**Pullback** (given `∂varrs_sel` and `∂st_dev_sel` at selected indices):
1. Expand to full variable size
2. `∂d_from_var[i] = ∂varrs_full[i] * (d[i] > eps() ? 1.0 : 0.0)` (max gate)
3. `∂d_from_std[i] = ∂st_dev_full[i] / (2 * st_dev_full[i]) * (d[i] > eps() ? 1.0 : 0.0)` (chain rule through sqrt∘abs∘max)
4. `∂d_total = ∂d_from_var + ∂d_from_std`
5. `∂covar_dcmp += Diagonal(∂d_total)` (diag pullback)

## Step 5: Post-processing for covariance (analytical pullback)

**Forward:** `covar_dcmp_sp = triu(covar_dcmp)`, then:
- **Non-grouped:** `result = covar_dcmp_sp[covar_var_idx, covar_var_idx]`
- **Grouped:** scatter specific `(i,j)` entries

**Pullback** (given `∂covar_out`):
1. **Non-grouped:** `∂covar_dcmp_sp = zeros; ∂covar_dcmp_sp[covar_var_idx, covar_var_idx] = ∂covar_out`
2. **Grouped:** scatter cotangents back
3. `triu` pullback: `∂covar_dcmp[i,j] += ∂covar_dcmp_sp[i,j]` only for `i ≤ j`

## Step 6: Post-processing for autocorrelation — 1st order (analytical pullback)

**Forward:**
```julia
A = sol[:, 1:nPast] * P       # P = I[past_idx, :]
d_inv = 1 ./ max.(diag(covar_dcmp), eps())
# Store R_i = A^i * covar_dcmp for each period
autocorr[:, i] = diag(A^i * covar_dcmp) .* d_inv
```

**Pullback** (given `∂autocorr_out` at selected indices):
1. Expand to full: `∂autocorr_full[autocorr_var_idx, :] = ∂autocorr_out`
2. Zero out rows where `diag(covar_dcmp) < tol`
3. Store `R_0 = covar_dcmp, R_i = A * R_{i-1}` during forward
4. Reverse loop: for each period `i`:
   - `∂diag_Ri = ∂autocorr_full[:, i] .* d_inv`
   - `∂R_i += Diagonal(∂diag_Ri)`
   - `∂d_inv -= ∂autocorr_full[:, i] .* diag(R_i)` → `∂d += ... * (-d_inv²)`
5. Reverse through `R_i = A * R_{i-1}`:
   - `∂A += ∂R_i * R_{i-1}'`
   - `∂R_{i-1} += A' * ∂R_i`
6. After loop: `∂covar_dcmp += ∂R_0 + Diagonal(∂d .* mask)`
7. `∂sol[:, 1:nPast] += ∂A * P'`

## Step 7: Post-processing for autocorrelation — 2nd order (analytical pullback)

**Forward:**
```julia
P_i = I
for i in autocorrelation_periods:
    autocorr[:, i] = diag(ŝ_to_y₂ * P_i * autocorr_tmp) ./ d
    P_i = P_i * ŝ_to_ŝ₂
```

**Pullback:** Reverse through the power iteration:
- Store all `P_i` during forward
- For each period (reverse):
  - `∂M_i = Diagonal(∂autocorr_full[:, i] ./ d)` where `M_i = ŝ_to_y₂ * P_i * autocorr_tmp`
  - `∂ŝ_to_y₂ += ∂M_i * (P_i * autocorr_tmp)'`
  - `∂P_i += ŝ_to_y₂' * ∂M_i * autocorr_tmp'`  (note: P_i contribution, not accumulated)
  - `∂autocorr_tmp += (ŝ_to_y₂ * P_i)' * ∂M_i`
  - `∂d -= ∂autocorr_full[:, i] .* diag(M_i) ./ d²`
- Reverse through `P_i = P_{i-1} * ŝ_to_ŝ₂`:
  - `∂P_{i-1} += ∂P_i * ŝ_to_ŝ₂'`
  - `∂ŝ_to_ŝ₂ += P_{i-1}' * ∂P_i`

For **pruned_third_order**: the sub-rrule already returns autocorrelation and handles its own pullback — no additional post-processing beyond indexing.

## Step 8: Compose the full pullback and return

The pullback function:
1. **Extract per-statistic cotangents** from the incoming `∂ret` (Dict-shaped)
2. **Run post-processing pullbacks** (Steps 4–7) to get `∂covar_dcmp`, `∂sol`/`∂ŝ_to_ŝ₂`/`∂ŝ_to_y₂`/`∂autocorr_tmp`, `∂state_μ`, `∂SS_and_pars`
3. **Pack the sub-rrule cotangent tuple** matching the sub-rrule's return structure
4. **Call the stored sub-pullback** to get `∂all_parameters`
5. **Scatter** `∂all_parameters` → `∂parameter_values` via the precomputed gather index
6. **Return** `(NoTangent(), ∂parameter_values)` (kwargs are non-differentiable)

Handle `!solved` early returns by returning `zeros(T, np)`.

---

## Verification

- Run existing test block in `test/functionality_tests.jl` (lines ~2490-2700)
- Write focused validation script `tasks/test_get_statistics_rrule.jl` comparing Zygote vs FiniteDifferences for each statistic × each algorithm on RBC_baseline
