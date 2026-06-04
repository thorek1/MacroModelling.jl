# Shapley decompositions for pruned higher-order perturbation

Pruned second- and third-order perturbation solutions involve cross-shock
interactions: part of the unconditional variance, and part of the value of a
filtered series at a given period, cannot be attributed to a single shock by
inspection. Two routines optionally allocate this interaction term across the
individual shocks using a marginal-contribution (Shapley value) rule:

- [`get_variance_decomposition`](@ref) with `marginal_contribution = true`
  allocates the cross-shock interaction term of the **unconditional variance**.
- [`get_shock_decomposition`](@ref) with `marginal_contribution = true`
  allocates the cross-shock interaction term of the **fitted historical
  series** returned by the inversion filter.

Both routines compute the same allocation rule. They differ only in what is
being decomposed (a Lyapunov-equation solution versus a forward simulation of
the pruned state recursion) and therefore in the numerical method used.

The remainder of this page describes the underlying construction and how
the result is reported.

---

## Coalition value and Shapley allocation

Let `nᵉ` denote the number of structural shocks and write
`N = {1, …, nᵉ}`. For a subset `S ⊆ N` of "active" shocks define a coalition
value `V(S)` whose meaning depends on the routine:

- For [`get_variance_decomposition`](@ref): the variance of a given variable
  under the pruned solution when shocks outside `S` are silenced.
- For [`get_shock_decomposition`](@ref): the value of a given variable at a
  given period along the fitted shock path, with shocks outside `S` set to
  zero.

The Shapley value `φᵢ` of shock `i ∈ N` is the unique additive allocation
of `V(N) − V(∅)` satisfying efficiency, symmetry, the dummy property, and
additivity across games:

```
φᵢ  =  (1 / nᵉ!) · Σ_{π ∈ Π}  [ V(prev(i,π) ∪ {i}) − V(prev(i,π)) ]
```

where the sum runs over all `nᵉ!` orderings `π` of `N`. Direct evaluation
requires `2ⁿᵉ` evaluations of `V` and is impractical beyond very small `nᵉ`.

Under a pruned `k`-th-order perturbation solution, `V(S)` is a polynomial of
degree at most `k` in the indicator variables `1_{i ∈ S}` (Möbius
decomposition):

```
V(S)  =  c_∅
       + Σ_i        c_i      · 1_{i ∈ S}
       + Σ_{i < j}  c_{ij}   · 1_{i ∈ S} · 1_{j ∈ S}
       + …                                                   (terms of degree ≤ k)
```

This holds because each entry of the augmented shock vector `ê` of the pruned
recursion is a Kronecker product of base shocks of length at most `k`, and
silencing a shock outside `S` zeroes every entry that contains it. The
Shapley value of a polynomial in indicators admits the closed form
([Owen, 1972](@cite owen1972multilinear)):

```
φᵢ  =  Σ_{T ∋ i}  c_T / |T|
```

Reading off the coefficients `c_T` directly still requires up to
`Σ_{j ≤ k} C(nᵉ, j)` evaluations of `V`. The implementation avoids this
enumeration by evaluating an equivalent path integral on the multilinear
extension of `V` instead.

---

## Aumann–Shapley path integral on the multilinear extension

Extend `V` to a continuous function `Ṽ` on the unit cube `[0,1]ⁿᵉ` by
replacing every indicator `1_{i ∈ S}` in the polynomial above by `xᵢ`
(multilinear extension). The Aumann–Shapley identity
([Aumann and Shapley, 1974](@cite aumannshapley1974)) gives the Shapley value
as a path integral along the diagonal of the cube:

```
φᵢ  =  ∫₀¹  ∂Ṽ(t · 𝟙) / ∂xᵢ   dt
```

The integrand `∂Ṽ(t · 𝟙) / ∂xᵢ` is a univariate polynomial in `t` of degree
at most `k − 1`. Gauss–Legendre quadrature with `⌈k/2⌉` nodes integrates
such a polynomial exactly: two nodes at second order, three at third order.

This collapses the cost from `2ⁿᵉ` evaluations of `V` to at most
`⌈k/2⌉ · nᵉ` evaluations of `∂Ṽ / ∂xᵢ` along the diagonal. The quadrature
nodes and weights on `(0, 1)` are obtained from a Gauss–Legendre rule
rescaled to the unit interval. The remainder of the construction differs by
routine.

---

## Variance decomposition

### Coalition value

For a given variable `v`, `V(S)` is the `v`-th diagonal entry of the
unconditional covariance produced by the pruned-state Lyapunov equation when
shocks outside `S` are silenced. With pruned state transition `A` and
cumulant block `C(S)`, the inner Lyapunov system is

```
Σ(S)  =  A · Σ(S) · Aᵀ  +  C(S)
```

and the variable-space covariance equals `ŝ_to_y · Σ(S) · ŝ_to_yᵀ` plus the
fixed boundary terms of the pruned second- and third-order moment
construction. Silencing shocks
outside `S` is implemented as a binary mask on the augmented shock vector
`ê`: at third order an `ê`-component is retained only when all its
exogenous-shock indices belong to `S` (see the docstring of
[`get_variance_decomposition`](@ref)).

### Continuous mask and its directional derivative

The multilinear extension corresponds to replacing the binary mask by a
continuous one: the mask entry attached to an `ê`-component is the product
of the activation levels `xⱼ` over the unique base shocks the component
mentions. Its derivative `ṁ = ∂m / ∂xᵢ` is sparse — only mask entries that
mention shock `i` are non-zero.

### Per node and per direction

For each Gauss–Legendre node `tₖ` on `(0, 1)` and each shock direction
`i ∈ {1, …, nᵉ}`, the construction proceeds as follows:

1. Evaluate `m` and `ṁ` at `x = tₖ · 𝟙`.
2. Apply the product rule to the cumulant block. With
   `C = ê · diag(m) · Γ · diag(m) · êᵀ`,
   ```
   Ċ  =  ê · diag(ṁ) · Γ · diag(m) · êᵀ
       + ê · diag(m)  · Γ · diag(ṁ) · êᵀ
       (+ cross terms involving the autocorrelation block Eᴸᶻ at third order)
   ```
   Sparsity of `Γ` and `ṁ` is preserved throughout; the right-hand side
   `Ċ` is kept sparse for the tangent Lyapunov solve.
3. Solve the tangent Lyapunov system with the same transition `A` (so the
   Schur factorisation of `A` is reusable across all nodes and directions):
   ```
   Σ̇(tₖ, i)  =  A · Σ̇(tₖ, i) · Aᵀ  +  Ċ
   ```
4. Accumulate the Gauss–Legendre contribution to the per-variable Shapley
   value
   ```
   φᵢ(v)  +=  wₖ · diag( ŝ_to_y · Σ̇(tₖ, i) · ŝ_to_yᵀ  +  boundary terms )[v]
   ```

The per-variable totals are divided by the unconditional variance from the
standard moment routines to obtain shares.

### Cost

| Order | Coefficient enumeration | Aumann–Shapley path integral |
|-------|--------------------------|------------------------------|
| 2     | `C(nᵉ, 1) + C(nᵉ, 2)` Lyapunov solves | `2 · nᵉ` Lyapunov solves |
| 3     | up to `Σ_{j ≤ 3} C(nᵉ, j)` Lyapunov solves | `3 · nᵉ` Lyapunov solves |

For `Smets_Wouters_2007` (`nᵉ = 7`) the second-order count is `14` versus
`28`, and the third-order count is `21` versus up to `126`.

### Convergence safeguard

The variance decomposition starts from the low-order Gauss–Legendre rule
and reruns with one additional node up to a maximum of seven whenever the
relative Shapley-efficiency closure error exceeds `1e-3`. The closure error
is the maximum relative deviation of the row sums from one (efficiency:
`Σᵢ φᵢ(v) = V(N)(v) − V(∅)(v)`).

### Output

With `marginal_contribution = true` the returned `KeyedArray` is
`nVars × nᵉ` and omits the `:Cross_shock_interaction` column. Row sums equal
one up to numerical noise; rows whose unconditional variance is below `eps()`
are reported as zero. Because the polynomial `V(S)` need not be monotone in
`S`, individual entries may fall outside `[0, 1]`.

---

## Shock decomposition

### Coalition value

For a fixed shock path `{ε₁, …, ε_T}` produced by the inversion filter and
for a variable `v`, `V_t(S)` is the value of `v` at period `t` along the
pruned state trajectory when shocks outside `S` are zeroed. The pruned
second-order recursion reads

```
ŝ_{t+1}  =  𝐒₁ · âₜ  +  ½ · 𝐒₂ · (âₜ ⊗ âₜ)
```

with the augmented vector `âₜ = [ŝₜ[past_idx]; 1; εₜ]`; the third-order
recursion adds a cubic Kronecker block. The mapping from active shocks to
`ŝₜ` is a polynomial of degree at most `k` in the indicators
`1_{i ∈ S}`, so the multilinear extension and the path-integral identity
apply unchanged.

### Forward-mode tangents

Materialising the polynomial coefficients explicitly would require carrying a
Kronecker bundle of width `C(nᵉ + k, k)` through every period. The shock
decomposition evaluates the path integral instead by running, at each
Gauss–Legendre node `sₖ`, one primal pruned trajectory with shocks scaled
by `sₖ` together with `nᵉ` forward-mode directional-derivative trajectories.

For each shock direction `i`, the tangent trajectory `v_t = ∂ŝₜ / ∂xᵢ`
satisfies the chain-rule lift of the recursion. At second order,

```
ȧ₁,ₜ      =  [ v_t[past_idx];  0;  εᵢ,ₜ · eᵢ ]
v_{t+1}   =  𝐒₁ · ȧ₁,ₜ  +  ½ · 𝐒₂ · (ȧ₁,ₜ ⊗ a₁,ₜ + a₁,ₜ ⊗ ȧ₁,ₜ)
```

where `eᵢ` is the `i`-th unit vector. The shock slot of the tangent
contains the *un-scaled* `εᵢ,ₜ` because `∂(xᵢ · εᵢ,ₜ) / ∂xᵢ = εᵢ,ₜ`.
The third-order driver adds the analogous cubic Kronecker tangents.

An additional `s = 0` primal trajectory supplies `V_t(∅)` and is stored as
the `:Initial_values` column.

### Per-period accumulation

```
φᵢ(v, t)  =  Σ_k  wₖ · v_{k, i, t}[v]    (plus higher-order tangents at k = 3)
```

Two Gauss–Legendre nodes suffice for the second-order recursion (the
integrand is degree at most one in `s`); three suffice for the third-order
recursion. Both drivers wrap the per-node routine in the same closure-error
refinement loop used by the variance decomposition, up to seven nodes.

### Cost

| Quantity per period | Polynomial bundle | Forward-tangent path integral |
|---------------------|-------------------|-------------------------------|
| Primal recursions   | one matrix-matrix product of width `C(nᵉ + k, k)` | `n_nodes` plain recursions |
| Tangent recursions  | folded into the bundle | `n_nodes · nᵉ` plain recursions |
| Memory              | `nState × C(nᵉ + k, k)` | `nState × n_nodes · (nᵉ + 1)` |

### Exactness

At second order the recursion-based path integral reproduces the Shapley
value of the multilinear extension exactly: the diagonal Kronecker term
`xᵢ² · εᵢ²` from the recursion and the multilinear-extension term
`xᵢ · εᵢ²` integrate to the same value under `∫₀¹ ∂/∂xᵢ`. Empirically the
closure error on `RBC_CME` and `Smets_Wouters_2007` is at machine precision.

At third order the same identity does not hold for mixed terms of the form
`c · xᵢ² · xⱼ`: the scaled-shock primal attributes `2c/3` to shock `i` and
`c/3` to shock `j`, whereas the strict multilinear-extension Shapley value
splits it `c/2`–`c/2`. Efficiency (row sums) is preserved exactly; only the
per-shock split differs by an amount controlled by the diagonal entries of
`𝐒₃`. The discrepancy is empirically of order `1e-8` relative on
`Smets_Wouters_2007`.

### Output

With `marginal_contribution = true` the returned `KeyedArray` has shape
`nVars × (nᵉ + 1) × T` and contains the `nᵉ` per-shock columns plus
`:Initial_values`. The `:Nonlinearities` column produced under
`marginal_contribution = false` is absorbed into the per-shock columns.
Slice sums over the shock axis at fixed `(v, t)` reproduce the fitted value
of variable `v` at period `t` minus the `:Initial_values` contribution
(Shapley efficiency).

---

## Summary

| | Variance decomposition | Shock decomposition |
|---|---|---|
| Routine | [`get_variance_decomposition`](@ref) | [`get_shock_decomposition`](@ref) |
| Coalition value `V(S)` | Unconditional variance under active set `S` | Fitted value at period `t` under active set `S` |
| Cost per `V` evaluation | One Lyapunov solve | One vector recursion per period |
| Quadrature work | `n_nodes · nᵉ` Lyapunov solves with shared `A` | `n_nodes · (nᵉ + 1) · T` plain matrix-vector products |
| Quadrature error | Zero (integrand of degree ≤ `k − 1`) | Zero at `k = 2`; `O(1e-8)` split-only perturbation at `k = 3` |

---

## References

- Aumann, R. J. and Shapley, L. S. (1974). *Values of Non-Atomic Games*.
  Princeton University Press.
- Owen, G. (1972). Multilinear Extensions of Games. *Management Science*
  18(5), 64–79.
- Shapley, L. S. (1953). A Value for n-Person Games. In *Contributions to
  the Theory of Games II*, 307–317.
