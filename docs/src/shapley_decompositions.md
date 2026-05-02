# Shapley decompositions in pruned higher-order perturbation

This page explains, intuitively and in detail, the two Shapley algorithms
implemented in MacroModelling.jl for pruned higher-order solutions:

- **`get_variance_decomposition(..., marginal_contribution = true)`** — Aumann–Shapley
  path-integral driver in `src/aumann_shapley.jl` and `src/moments.jl`.
- **`get_shock_decomposition(..., marginal_contribution = true)`** — Aumann–Shapley
  forward-tangent driver in `src/filter/inversion.jl`.

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

## 4. Shock decomposition: Aumann–Shapley forward tangents

### Why AS is also the right tool here

Per period `t`, given a fixed historical shock sequence `{ε₁, …, ε_T}`, the
pruned-state recursion is deterministic:
```
ŝ_{t+1} = 𝐒₁ · ŝ_t  +  𝐒₂ · (ŝ_t ⊗ ŝ_t) / 2  +  𝐒₃ · (ŝ_t ⊗ ŝ_t ⊗ ŝ_t) / 6  +  …
```
A coalition `S` corresponds to keeping each `εⱼ` if `j ∈ S` and zeroing it
otherwise. So `V_t(S)` equals `ŝ_t(S)` evaluated under coalition `S`, and is
a polynomial of degree `≤ k` in `1_S`.

We could either (a) propagate every Möbius coefficient `c_T` directly through
the recursion via wide Kronecker bundles, or (b) compute the AS path integral
along the diagonal `x = s · 𝟙` using forward-mode tangents through the
ordinary pruned recursion. The Kronecker-bundle approach is BLAS-friendly but
its bundle width grows as `Σ_{j ≤ k} C(nᵉ, j)` — at SW07 (`nᵉ = 7, k = 3`)
that is 120 columns. The forward-tangent AS approach instead carries
`n_nodes · nᵉ` plain vector recursions per period — at SW07 that is 21
vectors. For models beyond very small `nᵉ` the AS path is faster.

### The trick: scaled-shock primal + per-direction tangents

The same identity from §3 applies:
```
φᵢ(v, t) = ∫₀¹ ∂Ṽ_t(s · 𝟙)/∂xᵢ ds  ≈  Σ_k w_k · ∂Ṽ_t(s_k · 𝟙)/∂xᵢ
```
For each Gauss–Legendre node `s_k` we run two recursions:

1. A **primal** trajectory under shocks scaled by `s_k`: `ŝ_t(s_k)` evolves
   with effective shock vector `s_k · ε_t`.
2. For each shock direction `i = 1, …, nᵉ`, a **tangent** trajectory
   `v_t = ∂ŝ_t / ∂xᵢ` evaluated at `x = s_k · 𝟙`. The tangent recursion is
   the chain-rule lift of the primal:
```
ȧ₁_t = [v_t[past_idx]; 0; εᵢ_t · eᵢ]
v_{t+1} = 𝐒₁ · ȧ₁_t  +  𝐒₂ · (ȧ₁_t ⊗ a₁_t  +  a₁_t ⊗ ȧ₁_t) / 2  +  …
```
(the only non-trivial step is recognising that `∂(xᵢ · εᵢ_t)/∂xᵢ = εᵢ_t`,
so the tangent's shock slot is `εᵢ_t · eᵢ`, *not* `s_k · εᵢ_t · eᵢ`).

A separate `s = 0` primal trajectory provides `V_t(∅)` for the residual
column.

### Per-period Shapley accumulation

```
φᵢ(v, t) ≈ Σ_k w_k · v_{k,i,t}[v]   (plus higher-order tangents at k = 3)
```

Two Gauss–Legendre nodes suffice at second order (the integrand is degree
`≤ 1` in `s`); three nodes at third order (degree `≤ 2`).

### Cost picture

| Per period           | Polynomial bundle (alternative) | Forward-tangent AS (this driver) |
|----------------------|---------------------------------|----------------------------------|
| Primal work          | One gemm of width `C(nᵉ+k, k)`  | `n_nodes` plain recursions       |
| Tangent work         | folded into the bundle          | `n_nodes · nᵉ` plain recursions  |
| Memory               | `state × C(nᵉ+k, k)` bundle     | `state × n_nodes · (nᵉ + 1)`     |

For SW07 (`nᵉ = 7, k = 3`) the forward-tangent driver runs ~2.6× faster than
the polynomial bundle.

### Exactness caveat at third order

At second order, AS via the scaled-shock primal produces the *exact* Shapley
value (matches the multilinear-extension answer to machine precision; verified
to `≤ 3·10⁻¹⁵` on RBC_CME and SW07). The reason is a fortuitous integral
identity: `∫₀¹ 2s ds = 1 = ∫₀¹ 1 ds`, so the diagonal Kronecker term
`x[i]² · ε[i]²` (from the recursion) and the multilinear-extension term
`x[i] · ε[i]²` integrate to the same value under `∫ ∂/∂x[i]`.

At third order this identity breaks for diagonal-times-distinct mixed terms
like `c · x[i]² · x[j]`: AS via the scaled-shock primal attributes `2c/3` to
shock `i` and `c/3` to shock `j`, while the multilinear-extension Shapley
value would attribute `c/2` each. **Shapley efficiency holds exactly for both
splits** (the totals match), only the per-shock allocation differs by an
amount controlled by the magnitude of `𝐒₃`'s diagonal entries — empirically
≈ 10⁻⁸ relative on SW07.

### How to interpret the output

The output has dimensions `(variables × shocks × periods)`. Entry `[v, i, t]`
is the contribution of shock `i` to the value of variable `v` at period `t`,
with all cross-shock interactions in the per-period polynomial allocated by
the Shapley rule.

Concretely, at period `t`:

- **Linear parts** (contributions like `α · εᵢ_τ` for past `τ ≤ t`) are
  fully credited to shock `i`.
- **Quadratic cross-shock parts** like `β · εᵢ_τ · εⱼ_σ` are split 50/50 between
  `i` and `j`.
- **Cubic three-distinct-shock parts** are split three ways.
- **Squared-of-same-shock terms** `γ · εᵢ_τ²` go entirely to shock `i`.

Row sums (sum over shocks at fixed `t`) reproduce the model variable's value
at `t` — Shapley efficiency.

---

## 5. Side-by-side

|  | Variance decomposition | Shock decomposition |
|---|---|---|
| What `V(S)` is | Unconditional variance with shocks-outside-`S` = 0 | Per-period state value with shocks-outside-`S` = 0 |
| Cost of one `V` evaluation | One Lyapunov solve | One vector recursion per period |
| Driver | **Aumann–Shapley path integral** + Lyapunov solves at each Gauss–Legendre node | **Aumann–Shapley path integral** + forward tangents at each Gauss–Legendre node |
| Solves needed | `n_nodes · nᵉ` Lyapunov solves with shared `A` | Zero solves (just `(n_nodes · (nᵉ + 1)) × T` plain matrix-vector products) |
| Quadrature error | Zero (integrand is polynomial of degree `≤ k − 1` in the diagonal-path parameter) | Zero at `k = 2`; ≈10⁻⁸ split-only perturbation at `k = 3` from the V-vs-MLE diagonal-kron mismatch |

The unifying mental model:

> *In pruned k-th-order perturbation, `V` is a low-degree polynomial in
> coalition indicators. The Aumann–Shapley path integral collapses the
> 2ⁿᵉ-coalition definition of Shapley to a handful of derivative
> evaluations along the diagonal. Variance decomposition uses the integral
> via Lyapunov-solve characteristic functions; shock decomposition uses it
> via forward-mode tangent recursions through the pruned state.*

---

## References

- Aumann, R. J. and Shapley, L. S. (1974). *Values of Non-Atomic Games*.
  Princeton University Press.
- Owen, G. (1972). *Multilinear Extensions of Games*. Management Science
  18(5): 64–79.
