# Shapley decompositions in pruned higher-order perturbation

This page explains, intuitively and in detail, the two Shapley algorithms
implemented in MacroModelling.jl for pruned higher-order solutions:

- **`get_variance_decomposition(..., marginal_contribution = true)`** — Aumann–Shapley
  path-integral driver in `src/aumann_shapley.jl` and `src/moments.jl`.
- **`get_shock_decomposition(..., marginal_contribution = true)`** — direct
  polynomial-coefficient state propagation in `src/polynomial_coalition.jl` and
  `src/filter/inversion.jl`.

Both produce the (same) Shapley value but exploit different structural facts.

---

## 1. The shared cooperative-game picture

You have `nᵉ` shocks (the "players"). For any subset `S ⊆ {1,…,nᵉ}` you can
compute a number `V(S)` — the value if only players in `S` cooperate. Two
natural problems:

- **Variance decomposition:** `V(S)` is the unconditional variance of variable
  `v` in the model when shocks outside `S` are switched off.
- **Shock decomposition:** `V_t(S)` is the value of variable `v` at period `t`
  along a fixed historical path, when shocks outside `S` are switched off.

The Shapley value `φᵢ(v)` is the unique allocation rule satisfying efficiency
(`Σᵢ φᵢ = V(N) − V(∅)`), symmetry, dummy and additivity. Its definitional
formula is a weighted average over all `nᵉ!` orderings of the shocks:

```
φᵢ = (1/nᵉ!) · Σ_{orderings π} [V(predecessors(i, π) ∪ {i}) − V(predecessors(i, π))]
```

**Intuition.** Line up the shocks in a random order and credit each shock with
the marginal jump in `V` it causes when it joins. Shapley = the average
marginal contribution over all orderings.

That definition is intractable: it requires `2ⁿᵉ` evaluations of `V`. Both
algorithms below use a structural fact about pruned perturbation to bypass
enumeration entirely.

---

## 2. The structural fact: `V` is a low-degree polynomial in coalition indicators

In pruned `k`-th-order perturbation, `V(S)` is a polynomial of total degree
`≤ k` in the binary indicators `1_S`:

```
V(S) = c_∅
     + Σ_i        c_{i}    · 1_{i∈S}
     + Σ_{i<j}    c_{ij}   · 1_{i∈S} · 1_{j∈S}
     + Σ_{i<j<l}  c_{ijl}  · 1_{i∈S} · 1_{j∈S} · 1_{l∈S}
     + …
```
with `c_T = 0` for `|T| > k`.

**Why?** Each component of the augmented shock vector `ê` in pruned
perturbation is a Kronecker product of base shocks (`εᵢ`, `εⱼεₖ`, `εᵢεⱼεₖ`, …).
Restricting to coalition `S` zeroes every Kronecker piece that mentions a shock
outside `S`. So a coalition acts on `ê` as multiplication by a monomial product
`∏_{j ∈ piece} 1_{j∈S}` of degree at most `k`.

When `V` is multilinear like this, the Shapley value has a clean closed form
(Owen, 1972):
```
φᵢ = Σ_{T ∋ i} c_T / |T|
```
**Intuition.** Each monomial coefficient is split equally among the shocks it
mentions: `c_{ij}` came from shocks `i` and `j` cooperating, so half the
credit goes to each; `c_{ijl}` is split three ways; and so on.

So if you have the polynomial coefficients `c_T`, you have the Shapley value —
no orderings, no marginal contributions to enumerate. The two algorithms differ
only in how they obtain the `c_T` (or their integrated equivalent), because in
one setting the coefficients are easy to read off and in the other they are
hidden behind a Lyapunov solve.

---

## 3. Variance decomposition: Aumann–Shapley path integral

### Why the polynomial coefficients are hidden here

Computing `V(S)` requires solving a discrete Lyapunov equation
```
Σ(S) = A · Σ(S) · Aᵀ + C(S)
```
where `A` is the pruned-state transition (the same for every `S` — this
matters) and `C(S)` is the masked shock-cumulant block. Then
`V(S) = diag(ŝ_to_y · Σ(S) · ŝ_to_yᵀ + boundary terms)`.

The Lyapunov solve is linear in `C`, but `V` is the diagonal of a
sum-of-squares object, so the polynomial coefficients in `1_S` only emerge
*after* you solve. Reading off one coefficient `c_T` requires evaluating `V`
at enough subsets to invert Möbius — i.e. one Lyapunov solve per monomial.

For `nᵉ` shocks at order `k` that costs `Σ_{j ≤ k} C(nᵉ, j)` solves. SW07:
98 solves at second order, 126 at third. Doable but expensive.

### The Aumann–Shapley trick

Instead of recovering each `c_T` and then summing `c_T / |T|`, integrate the
directional derivative along a path. Aumann & Shapley (1974): for any smooth
extension `Ṽ` of `V` from the cube vertices `{0,1}ⁿᵉ` to the unit cube
`[0,1]ⁿᵉ`,
```
φᵢ = ∫₀¹ ∂Ṽ(t · 𝟙)/∂xᵢ  dt
```

**Intuition.** Walk in a straight line from "no shocks active" `(0,…,0)` to
"all shocks active" `(1,…,1)`. At each point on the path, ask: *if I nudge
shock i alone, how much does V change?* — that is the directional derivative
`∂Ṽ/∂xᵢ`. Average that nudge response over the whole walk. That average IS
shock i's Shapley share.

Picture it as a hike from valley to summit:

- The path is a straight diagonal line in shock-activity-space.
- At every point you stop and measure the slope in each shock's direction.
- Each shock's Shapley value = the average slope along the hike in that
  shock's direction.

This is just the integral form of the equal-share splitting from §2.

### Why this beats enumeration

Use the multilinear extension: replace each `1_{i∈S}` with `xᵢ ∈ [0,1]`. Then

- `Ṽ(t · 𝟙)` is a univariate polynomial in `t` of degree `≤ k`.
- `∂Ṽ(t · 𝟙)/∂xᵢ` is a polynomial in `t` of degree `≤ k − 1`.
- Gauss–Legendre quadrature with `⌈k/2⌉` nodes integrates a polynomial of
  degree `≤ k − 1` *exactly*.

So you replace `Σ_{j ≤ k} C(nᵉ, j)` Lyapunov solves with **`⌈k/2⌉ · nᵉ`
Lyapunov solves**, with *no quadrature error*. SW07: 14 (k=2) vs 98; 21
(k=3) vs 126. Empirical wall-clock on SW07: 5.03× speedup at second order,
~10× at third (max abs diff vs polynomial driver: 3.9e-14 at second order).

### What the code does, step by step

For each Gauss–Legendre node `tₖ ∈ (0,1)` and each shock direction `eᵢ`:

1. **Continuous coalition mask `m(x) ∈ ℝᴺ` at `x = tₖ · 𝟙`.** Each entry of
   `ê`'s mask becomes `∏_{j ∈ unique-shock-indices} xⱼ` instead of the boolean
   indicator. Implemented in
   `continuous_coalition_mask_{second,third}_order`.

2. **Directional derivative `ṁ = ∂m/∂xᵢ` at `x = tₖ · 𝟙`.** Sparse: only
   entries whose monomial mentions shock `i` are nonzero. Implemented in
   `mask_directional_derivative_{second,third}_order`.

3. **Directional derivative of `C`:**
   ```
   Ċ = ê·diag(ṁ)·Γ·êᵀ + ê·diag(m)·Γ·diag(ṁ)·êᵀ + (Eᴸᶻ cross term at 3rd order)
   ```

4. **One Lyapunov solve:**
   ```
   Σ̇(tₖ, eᵢ) = A · Σ̇(tₖ, eᵢ) · Aᵀ + Ċ
   ```
   with the same `A` as the unmasked problem (so a Schur factorisation is in
   principle reusable across all nodes/directions).

5. **Convert to variable space and accumulate with the Gauss–Legendre weight:**
   ```
   φᵢ(v)  +=  wₖ · diag(ŝ_to_y · Σ̇ · ŝ_to_yᵀ + boundary terms)[v]
   ```

After all `n_nodes · nᵉ` solves you have `φᵢ(v)`. Normalise by
`total_var(v)` to get shares (rows of the output sum to 1).

### How to interpret the output

Rows are variables, columns are shocks. There is no `:Cross_shock_interaction`
column when `marginal_contribution = true`. Entry `[v, i]` is the fraction of
variable `v`'s pruned-higher-order variance attributable to shock `i`, with
all cross-shock interactions allocated by the Shapley rule.

Cross-shock interactions like `c_{ij}` (the variance contribution that emerges
only when shocks `i` and `j` cooperate, e.g. through products of shocks in the
policy function) are split 50/50 between `i` and `j`. Triples are split three
ways. So a shock's share of the variance includes its own "own-shock"
contribution plus its fair share of every interaction pot it is a member of.

Negative shares or shares > 1 are possible (`V` need not be monotone in `S` at
higher order), but they remain the unique fair allocation in the Shapley
axiomatic sense.

---

## 4. Shock decomposition: direct polynomial-coefficient propagation

### Why the polynomial coefficients are visible here

Per period `t`, given a fixed historical shock sequence `{ε₁, …, ε_T}`, the
pruned-state recursion is deterministic:
```
ŝ_{t+1} = 𝐒₁ · ŝ_t  +  𝐒₂ · (ŝ_t ⊗ ŝ_t) / 2  +  𝐒₃ · (ŝ_t ⊗ ŝ_t ⊗ ŝ_t) / 6  +  …
```
A coalition `S` corresponds to keeping each `εⱼ` if `j ∈ S` and zeroing it
otherwise. So `V_t(S)` equals `ŝ_t(S)` evaluated under coalition `S`.

The polynomial-in-`1_S` structure comes directly from the recursion:

- `𝐒₁ · ŝ_t` is linear — the `εⱼ`-component carries indicator `1_{j∈S}` (degree 1).
- `𝐒₂ · (ŝ_t ⊗ ŝ_t)` is quadratic — a term involving `εⱼ εₖ` carries
  `1_{j∈S} · 1_{k∈S}` (degree 2).
- `𝐒₃ · (ŝ_t ⊗ ŝ_t ⊗ ŝ_t)` adds degree 3.

After pruning at order `k`, the highest-degree monomial in `1_S` that can
appear in `ŝ_t` is `k`. **The Kronecker structure of the recursion is the
polynomial structure**: the coefficients `c_T` are not hidden behind a solve,
they are literally being computed as part of the time recursion.

### The trick: stop tracking states, track polynomials of states

Instead of running the recursion `2ⁿᵉ` times (one per coalition), or even once
per Möbius monomial, carry **one polynomial-valued state vector** through time.
Each "state" `ŝ_t` becomes a polynomial in `1_S` whose coefficients are
state-shaped vectors.

The data structure is `PolyState`:

- `coefs::Matrix` — one column per Möbius monomial `T`, each column a
  state-sized vector.
- A shared `MonomialIndex(nᵉ, k)` that enumerates the at most
  `Σ_{j ≤ k} C(nᵉ, j)` monomials of size `≤ k`. SW07 at `k = 3`: 64 monomials.

Two operations:

1. **`poly_apply!(out, S, p)`** — apply a constant matrix to a polynomial.
   By linearity,
   `(S · p)(x) = Σ_T (S · p.coefs[:, T]) · ∏_{i∈T} xᵢ`
   so this is `out.coefs += S * p.coefs`: one BLAS `gemm` of size
   `(state-dim × #monomials)`.

2. **`poly_kron!(out, p, q; truncate_to = k)`** — Kronecker product of two
   polynomials.
   `(p ⊗ q)(x) = Σ_T Σ_{T'} (p.coefs[:, T] ⊗ q.coefs[:, T']) · ∏_{i ∈ T ∪ T'} xᵢ`
   The trick: any product term whose combined monomial has degree `> k` cannot
   contribute to `V` at any binary `1_S` (`V` is degree `≤ k` by pruning), so
   it is dropped. This caps the per-step work at `O(#monomials²)`.

### What the per-period algorithm looks like (second order)

```julia
poly_kron!(kron_aug₁, aug₁, aug₁; truncate_to = 2)         # (ε⊗ε) polynomial
poly_apply!(new₁, 𝐒[1], aug₁)                              # linear part propagates
poly_apply!(new₂, 𝐒[1], aug₂)                              # 2nd-order linear-in-state piece
poly_apply!(new₂, 𝐒[2], kron_aug₁; α = 0.5, β = 1.0)       # quadratic piece
shapley_from_poly!(decomposition[:, 1:nE, t], path_poly)   # equal-share aggregation
```

Each line is one matrix-multiply of `(state-dim × #monomials)`. Compared to:

- Naive coalition enumeration: `2ⁿᵉ` independent state-recursion runs per
  period.
- Möbius-coefficient enumeration via separate runs: `#monomials` separate
  recursion runs per period.

…polynomial propagation does the same work as one recursion *per matrix-
multiply size*, just on a wider matrix. There are zero solves anywhere, and
you get every Möbius coefficient *exactly* — no quadrature, no Lyapunov
tolerance, just BLAS.

### Picturing it

A standard shock decomposition carries a state vector through time, doing
matrix-multiplies and (for higher orders) Kroneckered products at each step.
The polynomial-coefficient version carries a **bundle of state vectors** —
one per monomial — through time, doing the same matrix-multiplies on the
bundle, with the Kronecker step combining bundles-of-bundles while throwing
away the high-degree pieces that pruning guarantees cannot matter.

At the end of period `t`, look at the bundle. The column for monomial
`T = {1, 3}` is the part of the state that exists *only* because shocks 1
and 3 cooperated (it contains, say, products `ε₁ · ε₃` from `𝐒₂`, or higher
products that involve both). Multiply out by `ŝ_to_y` to obtain the same
column for the observable.

### Final aggregation: `shapley_from_poly!`

For each variable `v` and each monomial `T` in the bundle:
```
φᵢ(v) += coefs[v, T] / |T|     for every i ∈ T
```
This is exactly the Owen formula from §2, applied directly to the
coefficients the recursion just computed.

### How to interpret the output

The output has dimensions `(variables × shocks × periods)`. Entry `[v, i, t]`
is the contribution of shock `i` to the value of variable `v` at period `t`,
with all cross-shock interactions in the per-period polynomial allocated by
the Shapley rule.

Concretely, at period `t`:

- **Linear parts** (contributions like `α · εᵢ_τ` for some past τ ≤ t) are
  fully credited to shock `i`.
- **Quadratic parts** like `β · εᵢ_τ · εⱼ_σ` are split 50/50 between `i` and
  `j`.
- **Cubic parts** with three distinct shocks are split three ways.
- **Squared-of-same-shock terms** `γ · εᵢ_τ²` go entirely to shock `i` (the
  monomial `{i}` already contains only one player).

Row sums (sum over shocks at fixed `t`) reproduce the model variable's value
at `t` exactly, matching the per-shock contributions of a linear shock
decomposition — Shapley efficiency.

---

## 5. Side-by-side

|  | Variance decomposition | Shock decomposition |
|---|---|---|
| What `V(S)` is | Unconditional variance with shocks-outside-`S` = 0 | Per-period state value with shocks-outside-`S` = 0 |
| Cost of one `V` evaluation | One Lyapunov solve | Effectively one matrix-multiply per period |
| Are `c_T` directly readable? | No — emerge only after solve | **Yes — they ARE the Kronecker structure** |
| Therefore Shapley via | **Aumann–Shapley path integral** + Gauss–Legendre quadrature | **Direct polynomial propagation** + equal-share aggregation |
| Solves needed | `n_nodes · nᵉ` Lyapunov solves with shared `A` | **Zero solves** |
| Quadrature error | Zero (integrand is polynomial of degree `≤ k − 1`) | N/A |
| Speedup vs naive enumeration (SW07) | ~5× (k=2), ~6× (k=3) | ~5× (k=2), ~10× (k=3) — and exact |

Both are the right algorithm for their setting. AS pays for indirect access to
`V` through Lyapunov solves by integrating cleverly; polynomial propagation
exploits the fact that the Möbius coefficients are *literally being computed*
by the recursion already, so it only needs to remember them.

The unifying mental model:

> *In pruned k-th-order perturbation, `V` is a low-degree polynomial in
> coalition indicators. Shapley = equal-share splitting of the polynomial's
> coefficients. The two algorithms are the two clean ways of extracting that
> splitting given the cost structure of `V` in their domain.*

---

## References

- Aumann, R. J. and Shapley, L. S. (1974). *Values of Non-Atomic Games*.
  Princeton University Press.
- Owen, G. (1972). *Multilinear Extensions of Games*. Management Science
  18(5): 64–79.
