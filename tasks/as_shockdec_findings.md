# Aumann–Shapley shock decomposition: implementation + benchmark

## Implementation

`src/filter/inversion.jl` now ships two new drivers alongside the existing
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
between `:polynomial` (default, exact, production) and `:aumann_shapley`
(this driver). The polynomial driver remains the default; nothing in the
public API is changed.

## Benchmark — RBC_CME (nE = 2, T = 30)

| algorithm           | rel diff   | poly time | AS time | AS/poly |
|---------------------|------------|-----------|---------|---------|
| pruned_second_order | 4.07e-16   | 0.27 ms   | 0.36 ms | 1.34×   |
| pruned_third_order  | 3.29e-07   | 0.70 ms   | 1.24 ms | 1.76×   |

## Findings

**Second order (k = 2): AS matches the polynomial Shapley value to
machine precision, but is slower.** The polynomial path stores at most
`1 + nE + C(nE, 2)` Möbius coefficients and propagates them via a single
BLAS gemm per period; AS runs `n_nodes·(1 + nE)` separate vector-only
recursions. For `nE = 2` the BLAS bundle wins. The crossover should
occur at much larger `nE`, but in any case AS for k = 2 reproduces the
exact answer.

**Third order (k = 3): AS does NOT match the polynomial Shapley value.**
The discrepancy is theoretical, not a bug. The pruned recursion at
scaled shocks `x[i]·ε[i]` does not encode the multilinear extension of
`V(S)`; it encodes a polynomial of degree `> 1` in each `x[i]`. The two
agree on the boundary `x ∈ {0, 1}^nE` but differ on the interior because
the multilinear extension enforces `x[i]² = x[i]`, while the raw
recursion produces genuine `x[i]³` terms via the `kron(kron(aug₁, aug₁),
aug₁)` block.

For 2nd order the discrepancy collapses under the AS integral by a
fortuitous identity (`∫₀¹ 2s ds = 1 = ∫₀¹ ds`), so naive AS still gives
the exact Shapley value. For 3rd order the identity breaks: a
diagonal-times-distinct term `c · x[i]² · x[j]` integrated under the
naive AS gives `2c/3` for shock `i`, whereas the multilinear extension
collapses it to `c · x[i] · x[j]` and AS gives `c/2` for shock `i`. The
two splits differ.

**Conclusion.** The polynomial-coefficient driver remains the right
choice for shock decomposition under pruned 2nd/3rd order:

- It is the exact Shapley value (no quadrature error and no MLE
  approximation issue at 3rd order).
- It is faster on the tested model size.
- It is BLAS-friendly (single gemm per period over a coefficient bundle).

AS remains the right driver for the **variance decomposition**
characteristic function (where each evaluation is a Lyapunov solve, so
trading 2^nE solves for `O(nE^2)` solves is a clear win) but offers no
analogous advantage when the characteristic function is a vector
recursion as in shock decomposition.

The new AS drivers are kept in the source as benchmark tooling and as a
worked example of a forward-mode tangent through pruned recursions.
