# Aumann–Shapley shock decomposition: implementation + benchmark

## Implementation

`src/filter/inversion.jl` ships two new drivers alongside the existing
polynomial-coefficient path:

- `aumann_shapley_shock_decomposition_pruned_2nd_order!`
- `aumann_shapley_shock_decomposition_pruned_3rd_order!`

Both maintain, per Gauss–Legendre node `s_k`:

- one primal pruned-state trajectory under shocks scaled by `s_k`
  (`ŝ₁(s_k), ŝ₂(s_k)[, ŝ₃(s_k)]`); plus a separate `s = 0` trajectory
  for `V(∅)`;
- one tangent trajectory per shock direction `i`, computed by chain rule
  through the same recursion using `eps_dir = ε[i]·eᵢ` as the shock
  perturbation.

Per-period Shapley shares are accumulated as

```
φᵢ(v, t) ≈ Σ_k w_k · (v_{k,i,t} + w_{k,i,t} [ + u_{k,i,t} ])
```

A benchmark-only switch `MacroModelling.SHOCK_DECOMP_MC_METHOD[]` toggles
between `:polynomial` (default, exact, production) and `:aumann_shapley`.
Polynomial remains the default; nothing in the public API is changed.

## Benchmarks

### RBC_CME (nE = 2, T = 30)

| algorithm           | rel diff   | poly time | AS time | AS/poly |
|---------------------|------------|-----------|---------|---------|
| pruned_second_order | 4.07e-16   | 0.27 ms   | 0.36 ms | 1.34×   |
| pruned_third_order  | 3.29e-07   | 0.70 ms   | 1.24 ms | 1.76×   |

### SW07 (nE = 7, nVars = 66, T = 30)

| algorithm           | rel diff   | poly time | AS time   | AS/poly |
|---------------------|------------|-----------|-----------|---------|
| pruned_second_order | 2.70e-15   | 36.9 ms   | 22.4 ms   | 0.61×   |
| pruned_third_order  | 6.94e-09   | 3279.4 ms | 1264.7 ms | 0.39×   |

## Findings

**Second order (k = 2): AS exactly matches the polynomial Shapley value
(machine precision) on every model tested.** AS scales as `O(n_nodes · nE)`
vector recursions per period, polynomial as a single gemm over a bundle of
`O(C(nE + 2, 2))` coefficient columns. For very small `nE` (RBC_CME, nE = 2)
the BLAS bundle wins; for moderate `nE` (SW07, nE = 7) AS becomes ~1.65×
faster.

**Third order (k = 3): AS does *not* exactly match the polynomial Shapley
value, but the discrepancy is small in practice** (3e-7 on RBC_CME, 7e-9 on
SW07). The reason is theoretical: the pruned recursion at scaled shocks
`x[i]·ε[i]` produces a polynomial in `x` of degree up to `k`, whereas the
multilinear extension required by the AS theorem enforces `x[i]² = x[i]`.
For `k = 2` a fortuitous identity (`∫₀¹ 2s ds = 1 = ∫₀¹ ds`) makes the two
integrals coincide. For `k = 3` they differ on diagonal-times-distinct terms
like `c · x[i]² · x[j]`: AS over the raw recursion attributes `2c/3` to
shock `i` and `c/3` to shock `j`, whereas the multilinear-extension AS
(matching the polynomial Shapley value) attributes `c/2` each. **The sum
is identical (Shapley efficiency holds for both)**, only the per-shock split
differs. The magnitude depends on the third-derivative tensor `𝐒[3]`'s
diagonal entries; it is small in well-behaved models.

For 3rd order on SW07, AS is ~2.6× faster than the polynomial path because
polynomial scales as `C(nE + 3, 3) = 120` coefficient columns at nE = 7,
while AS scales as `n_nodes · nE = 21` tangent recursions per period.

## Conclusion

| model size  | k = 2 winner       | k = 3 winner                          |
|-------------|--------------------|---------------------------------------|
| small (nE≈2)| polynomial (exact, faster) | polynomial (exact, faster)    |
| medium (nE≈7)| AS (exact, ~1.7× faster) | AS (~2.6× faster, ~1e-8 split error) |

The polynomial-coefficient driver remains the **default** because it gives
the exact Shapley value at any nE. The AS driver is shipped as an opt-in
path (`SHOCK_DECOMP_MC_METHOD[] = :aumann_shapley`) for users who want to
benchmark or who need the speed advantage on larger models and can tolerate
~1e-8 perturbations of the per-shock split at 3rd order.
