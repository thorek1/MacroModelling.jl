# Analysis: Third-Order Moment Calculation — Compressed Space Speedup

## Executive Summary

The third-order moment calculation (`calculate_third_order_moments` in `src/moments.jl:692–917`) can benefit significantly from working in compressed space. The main opportunity lies in **compressing the symmetric Kronecker product components of the augmented state vector** used in the Lyapunov equation, rather than just avoiding the `𝐒₃` expansion step.

**Key finding**: For a model with nˢ=10 dependent state variables, the Lyapunov system dimension would shrink from **1230 to 405** (3.0× reduction), yielding an estimated **9× speedup** in the Lyapunov solve (doubling algorithm), or **28× for dense Bartels-Stewart**. Memory usage drops from ~12 MB to ~1.3 MB for the covariance matrix alone.

## Current Implementation

### Pipeline

```
calculate_third_order_moments (moments.jl:692)
  ├── calculate_second_order_moments_with_covariance → Σᶻ₂, 𝐒₁, 𝐒₂_raw, ...
  ├── 𝐒₂ = 𝐒₂_raw * 𝐔₂           # expand 2nd-order compressed → full (line 706)
  ├── calculate_third_order_solution → 𝐒₃ (compressed: nVars × m₃)
  ├── 𝐒₃ *= 𝐔₃                     # expand 3rd-order compressed → full (line 727)
  ├── determine_efficient_order      # dependency analysis on expanded 𝐒₃
  └── for ords in orders             # Lyapunov loop (lines 760–914)
       ├── Extract sub-blocks from 𝐒₁, 𝐒₂, 𝐒₃ by column indexing
       ├── Build ŝ_to_ŝ₃ (augmented state transition matrix)
       ├── Build ê_to_ŝ₃ (shock-to-state matrix)
       ├── Build Γ₃, Eᴸᶻ (shock covariance, cross-covariance)
       ├── Solve Lyapunov: ŝ_to_ŝ₃ · Σᶻ₃ · ŝ_to_ŝ₃' + C = Σᶻ₃
       └── Compute Σʸ₃ = ŝ_to_y₃ · Σᶻ₃ · ŝ_to_y₃' + ...
```

### Augmented State Vector

The third-order pruned state vector has 6 components:

| Block | Component   | Dimension     | Symmetric? |
|-------|------------|---------------|------------|
| 1     | ŝₜ         | nˢ            | N/A        |
| 2     | ŝₜ^δ₂      | nˢ            | N/A        |
| 3     | ŝₜ⊗ŝₜ      | **nˢ²**       | **Yes**    |
| 4     | ŝₜ^δ₃      | nˢ            | N/A        |
| 5     | ŝₜ⊗ŝₜ^δ₂   | nˢ²           | No         |
| 6     | ŝₜ⊗ŝₜ⊗ŝₜ  | **nˢ³**       | **Yes**    |

**Total current**: 3nˢ + 2nˢ² + nˢ³

Components 3 and 6 are symmetric because they are Kronecker self-products of the same vector. This means:
- `ŝ⊗ŝ`: element (i,j) = element (j,i), so only nˢ(nˢ+1)/2 unique values
- `ŝ⊗ŝ⊗ŝ`: element (i,j,k) = any permutation, so only nˢ(nˢ+1)(nˢ+2)/6 unique values

Component 5 (`ŝ⊗ŝδ₂`) is NOT symmetric since ŝ and ŝδ₂ are different variables.

## Optimization Opportunities

### Opportunity 1: Skip `𝐒₃` Expansion (Quick Win)

**Current** (line 727): `𝐒₃ *= 𝓂.constants.third_order.𝐔₃` expands from compressed `(nVars × m₃)` to full `(nVars × n³)`.

**Alternative**: Map the column selectors (e.g., `ℒ.kron(kron_s_s, s_in_s⁺)`) directly to compressed column indices. Since `𝐔₃` has exactly one nonzero per column, each full-space column index maps to a unique compressed column index. This mapping can be precomputed.

**Savings**: Avoids allocating the expanded `nVars × n³` sparse matrix. However, the extracted sub-blocks (e.g., `s_s_s_to_y₃` of shape `nObs × nˢ³`) remain the same size because each compressed column must be expanded to represent all permutations (up to 6) of the corresponding triple in the full-space representation that the Lyapunov loop expects.

**Impact**: Low — memory savings only, no compute reduction for the Lyapunov solve.

### Opportunity 2: Compress ŝ⊗ŝ⊗ŝ in the Lyapunov State (Major Win) ⭐

Replace the nˢ³-dimensional component with nˢ(nˢ+1)(nˢ+2)/6 unique sorted triples.

**State vector change**:
```
Before: [ŝ, ŝδ₂, ŝ⊗ŝ, ŝδ₃, ŝ⊗ŝδ₂, ŝ⊗ŝ⊗ŝ]     dim = 3nˢ + 2nˢ² + nˢ³
After:  [ŝ, ŝδ₂, ŝ⊗ŝ, ŝδ₃, ŝ⊗ŝδ₂, vech₃(ŝ⊗ŝ⊗ŝ)]  dim = 3nˢ + 2nˢ² + m₃ˢ
```

where `m₃ˢ = nˢ(nˢ+1)(nˢ+2)/6` and `vech₃` selects unique sorted triples.

**Transition matrix block changes** (in `ŝ_to_ŝ₃`):
- Block (6,6): `kron(s₁, s₁⊗s₁)` (nˢ³×nˢ³) → `U₃ˢ · kron(s₁, s₁⊗s₁) · C₃ˢ` (m₃ˢ×m₃ˢ)
  - This is exactly what `compressed_kron³(s_to_s₁)` already computes!
- Block (4,6): `s_s_s_to_s₃/6` (nˢ×nˢ³) → `s_s_s_to_s₃/6 · C₃ˢ` (nˢ×m₃ˢ)
- Block (5,6): `kron(s₁, s₂/2)` (nˢ²×nˢ³) → `kron(s₁, s₂/2) · C₃ˢ` (nˢ²×m₃ˢ)

**Shock matrix changes** (in `ê_to_ŝ₃`):
- Row 6 blocks coupling to ŝ⊗ŝ⊗ŝ need left-multiplication by `U₃ˢ`
- The transformations are straightforward compositions with the duplication matrix

**Dimension savings**:

| nˢ  | N_current | N_compressed | State ratio | Lyap matrix speedup |
|-----|-----------|-------------|-------------|---------------------|
| 3   | 54        | 34          | 1.6×        | 2.5×                |
| 5   | 190       | 90          | 2.1×        | 4.5×                |
| 10  | 1230      | 405         | 3.0×        | 9.2×                |
| 15  | 3870      | 1070        | 3.6×        | 13.1×               |
| 20  | 8860      | 2210        | 4.0×        | 16.1×               |

### Opportunity 3: Also Compress ŝ⊗ŝ (Additional Win)

Replace the nˢ²-dimensional component 3 (ŝ⊗ŝ) with nˢ(nˢ+1)/2.

Note: Component 5 (ŝ⊗ŝδ₂) is NOT symmetric and cannot be compressed.

**Additional savings**: ~45% of the nˢ² component. For nˢ=10, saves 45 dimensions (100→55).

**Complexity**: Moderate — the `I_plus_s_s` symmetrization matrix already in the code (`reshape(kron(vec(I(nˢ)), I(nˢ)), nˢ², nˢ²) + I`, which is `K_{nˢ,nˢ} + I` where K is the commutation matrix) handles the symmetry for vec(A⊗B + B⊗A), though the actual compression would use standard duplication matrices C₂ˢ and U₂ˢ for the substate dimension nˢ.

### Opportunity 4: Compress Symmetric Shock Components (Smaller Win)

The shock vector `ê` has components like `ε⊗ε` (nᵉ²), `ŝ⊗ŝ⊗ε` (nˢ²nᵉ), `ŝ⊗ε⊗ε` (nˢnᵉ²), and `ε⊗ε⊗ε` (nᵉ³), each with symmetric sub-products.

**Impact**: Smaller than state compression since nᵉ is typically small (1–5 shocks), but still worth doing for consistency.

## Implementation Roadmap

### Phase 1: Skip `𝐒₃` Expansion (Low effort, low impact)
1. Precompute mapping from full column indices to compressed column indices
2. Replace `𝐒₃[obs, kron_idx]` with `𝐒₃_raw[obs, compressed_kron_idx]`
3. This requires a column-duplication step since `s_s_s_to_y₃` must still have nˢ³ columns for the Lyapunov loop

### Phase 2: Compress ŝ⊗ŝ⊗ŝ in Lyapunov State (Moderate effort, high impact) ⭐
1. Construct local `C₃ˢ` and `U₃ˢ` matrices for the dependency-substate dimension nˢ
2. Transform block (6,6) of `ŝ_to_ŝ₃` using `compressed_kron³(s_to_s₁)` (already available!)
3. Transform blocks (4,6) and (5,6) by right-multiplying by `C₃ˢ`
4. Transform row 6 of `ê_to_ŝ₃` by left-multiplying by `U₃ˢ`
5. Transform `ŝ_to_y₃` last block and `ê_to_y₃` last block accordingly
6. Transform relevant blocks in `Γ₃` and `Eᴸᶻ`
7. Update `μˢ₃δμˢ₁` computation for compressed `s_s_s_to_s₃`

### Phase 3: Compress ŝ⊗ŝ (Additional effort, moderate impact)
1. Construct local `C₂ˢ` and `U₂ˢ` for nˢ-dimensional second-order compression
2. Transform block (3,3) and related blocks
3. Adjust `I_plus_s_s` usage for compressed representation

## Mathematical Justification

The symmetry exploitation is valid because:

1. **ŝ⊗ŝ symmetry**: `(ŝ⊗ŝ)_{ij} = ŝᵢŝⱼ = ŝⱼŝᵢ = (ŝ⊗ŝ)_{ji}`. Only `nˢ(nˢ+1)/2` unique elements.

2. **ŝ⊗ŝ⊗ŝ symmetry**: `(ŝ⊗ŝ⊗ŝ)_{ijk} = ŝᵢŝⱼŝₖ` is invariant under all 6 permutations of (i,j,k). Only `nˢ(nˢ+1)(nˢ+2)/6` unique elements.

3. **Dynamics preserve symmetry**: If `(ŝ⊗ŝ⊗ŝ)ₜ` is symmetric, then
   `(s₁⊗s₁⊗s₁)(ŝ⊗ŝ⊗ŝ)ₜ` is also symmetric, because it equals `(s₁ŝ)⊗(s₁ŝ)⊗(s₁ŝ)`.

4. **Covariance inherits symmetry**: The Lyapunov solution Σᶻ₃ respects the block structure of the compressed state.

5. **Duplication matrix identity**: `vec(ŝ⊗ŝ⊗ŝ) = C₃ˢ · vech₃(ŝ⊗ŝ⊗ŝ)` and `vech₃(ŝ⊗ŝ⊗ŝ) = U₃ˢ · vec(ŝ⊗ŝ⊗ŝ)` where `U₃ˢ · C₃ˢ = I`.

## Existing Infrastructure

The codebase already has all the building blocks:
- `compressed_kron³(A)` — computes `U₃ · kron(A,A,A) · C₃` directly in compressed form
- `𝐔₃`, `𝐂₃` — duplication/unique matrices for the full augmented state dimension nₑ₋ (new substate-dimension matrices `U₃ˢ` and `C₃ˢ` for nˢ must be constructed using the same algorithm, parameterized by nˢ instead of nₑ₋)
- `I_plus_s_s` — commutation+identity matrix for ŝ⊗ŝ symmetrization (line 792)
- `mat_mult_kron` — Kron-free matrix multiplication

The key transformation `U₃ˢ · kron(s₁, s₁⊗s₁) · C₃ˢ` is exactly what the existing `compressed_kron³` produces for the substate.

## Risk Assessment

- **Correctness**: Low risk — the math is well-defined and the duplication matrix identities are standard
- **AD compatibility**: Medium risk — the rrule for `calculate_third_order_moments` would need corresponding changes to handle compressed Lyapunov blocks
- **Sparsity patterns**: Low risk — compressed representations tend to be denser per element but smaller overall, which favors dense Lyapunov algorithms
