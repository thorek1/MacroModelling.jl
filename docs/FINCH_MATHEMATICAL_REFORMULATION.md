# Mathematical Reformulation for Finch.jl Integration

## Overview

This document describes a fundamental reformulation of the higher-order perturbation solution algorithms using Finch.jl's tensor contraction capabilities at a mathematical level, rather than just wrapping individual operations.

## Current Approach vs. Fundamental Approach

### Current Approach (Wrapper-based)
```julia
# Wraps individual operations
C = mat_mult_kron_finch(∇₂, S₁, S₁, M₂)
B = mat_mult_kron_finch(M₂, S₁, S₁, C₂)
```

### Fundamental Approach (Tensor Contraction-based)
```julia
# Express as multi-index tensor contractions
# C[i,k] = ∇₂[i,j₁,j₂] * S₁[j₁,l₁] * S₁[j₂,l₂] * M₂[l₁,l₂,k]
# Let Finch optimize the entire contraction
```

## Mathematical Structure

### Second Order Solution

The second-order solution involves solving a Sylvester equation:
```
A * X - X * B = C
```

Where:
- **A**: Transformed first-order derivatives
- **B**: Involves Kronecker products of first-order solution matrices
- **C**: Involves second-order derivatives contracted with Kronecker products

#### Matrix B Assembly

**Original formulation:**
```julia
B = M₂.𝐔₂ * kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) * M₂.𝐂₂ + M₂.𝐔₂ * M₂.𝛔 * M₂.𝐂₂
```

**Tensor contraction formulation:**
```
B[i,k] = U₂[i,j] * (S₁[j₁,l₁] * S₁[j₂,l₂] * δ[j, j₁⊗j₂]) * C₂[l₁⊗l₂, k]
         + U₂[i,j] * σ[j,m] * C₂[m,k]
```

Where `j₁⊗j₂` represents the Kronecker product index mapping.

#### Matrix C Assembly

**Original formulation:**
```julia
C = ∇₁₊𝐒₁➕∇₁₀⁻¹ * (∇₂ * kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) * M₂.𝐂₂
    + ∇₂ * kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔 * M₂.𝐂₂)
```

**Tensor contraction formulation:**
```
temp[i,k] = ∇₂[i,j] * S_combined[j₁,l₁] * S_combined[j₂,l₂] * δ[j, j₁⊗j₂] * C₂[l₁⊗l₂, k]
            + ∇₂[i,j] * S₁₊[j₁,l₁] * S₁₊[j₂,l₂] * δ[j, j₁⊗j₂] * σ[l₁⊗l₂,m] * C₂[m,k]
C[i,k] = (∇₁₊𝐒₁➕∇₁₀⁻¹)[i,i'] * temp[i',k]
```

## Finch.jl Implementation Strategy

### 1. Express Kronecker Product as Index Mapping

Instead of materializing `kron(A, B)`, use Finch to handle multi-dimensional indexing:

```julia
# Define a Finch tensor with a custom index structure
# that maps (i, j⊗k) to (i, j, k)
@finch begin
    result[i, l] .= 0
    for i=_, j1=_, j2=_, k1=_, k2=_
        # Kronecker index: j = (j1-1)*n_j2 + j2
        # Kronecker index: k = (k1-1)*n_k2 + k2
        result[i, (k1-1)*n_k2 + k2] += A[i, (j1-1)*n_j2 + j2] * B[j1, k1] * C[j2, k2]
    end
end
```

### 2. Fuse Multiple Operations

Let Finch fuse the entire computation graph:

```julia
@finch begin
    # Compute B matrix directly as tensor contraction
    B .= 0
    for i=_, j1=_, j2=_, k1=_, k2=_, m=_
        j_kron = (j1-1)*n_S1 + j2
        k_kron = (k1-1)*n_S1 + k2
        B[i, m] += U2[i, j_kron] * S1[j1, k1] * S1[j2, k2] * C2[k_kron, m]
    end
    
    # Add the sigma term
    for i=_, j=_, m=_, k=_
        B[i, k] += U2[i, j] * sigma[j, m] * C2[m, k]
    end
end
```

### 3. Use Finch's Sparse Tensor Formats

Leverage Finch's ability to choose optimal sparse formats:

```julia
# Let Finch determine the best format for intermediate results
@finch mode=:fast begin
    # Finch will analyze sparsity patterns and choose
    # whether to use COO, CSR, or other formats
    temp = Tensor(SparseByteMap(SparseList(Element(0.0))))
    # ... computations ...
end
```

## Third Order Solution

Similar principles apply but with 3rd-order tensors:

### Matrix B Assembly (3rd Order)

**Tensor contraction formulation:**
```
B[i,k] = U₃[i,j] * (S₁[j₁,l₁] * S₁[j₂,l₂] * S₁[j₃,l₃] * δ[j, j₁⊗j₂⊗j₃]) * C₃[l₁⊗l₂⊗l₃, k]
```

With symmetry exploitation:
```
# Only compute for j₁ ≤ j₂ ≤ j₃
# Use divisors for repeated indices
```

### Compressed Kronecker Product

Instead of computing and storing the full 3rd Kronecker power, express it as:

```julia
@finch begin
    result .= 0
    for i1=_, i2=_, i3=_, j1=_, j2=_, j3=_
        if i1 >= i2 >= i3 && j1 >= j2 >= j3  # Symmetry
            row = compressed_index(i1, i2, i3)
            col = compressed_index(j1, j2, j3)
            divisor = symmetry_divisor(i1, i2, i3) * symmetry_divisor(j1, j2, j3)
            result[row, col] = (a[i1,j1]*a[i2,j2]*a[i3,j3] + ...) / divisor
        end
    end
end
```

## Benefits of Fundamental Approach

1. **Global Optimization**: Finch can optimize the entire computation graph, not just individual operations
2. **Memory Efficiency**: Intermediate Kronecker products never materialized
3. **Format Selection**: Finch chooses optimal sparse formats for each intermediate tensor
4. **Loop Fusion**: Multiple operations fused into single loops
5. **Type Stability**: Better type inference across the entire computation
6. **Compile-Time Optimization**: Finch generates specialized code for the entire contraction

## Implementation Checklist

- [ ] Express B matrix assembly as tensor contraction
- [ ] Express C matrix assembly as tensor contraction
- [ ] Implement Kronecker index mapping in Finch
- [ ] Handle symmetry in 3rd order contractions
- [ ] Let Finch optimize sparse format selection
- [ ] Benchmark against current implementation
- [ ] Validate numerical accuracy

## Example: Complete Second Order in Finch

```julia
function calculate_second_order_solution_finch_fundamental(∇₁, ∇₂, 𝑺₁, M₂, ℂC; T, opts)
    # Setup dimensions and indices (same as before)
    n₋, n₊, nₑ, n = T.nPast_not_future_and_mixed, T.nFuture_not_past_and_mixed, T.nExo, T.nVars
    
    # Convert all matrices to Finch tensors
    ∇₂_finch = Finch.Tensor(Dense(SparseList(Element(0.0))), ∇₂)
    S1_finch = Finch.Tensor(Dense(Dense(Element(0.0))), 𝐒₁)
    U2_finch = Finch.Tensor(Dense(Dense(Element(0.0))), M₂.𝐔₂)
    C2_finch = Finch.Tensor(Dense(Dense(Element(0.0))), M₂.𝐂₂)
    
    # Express the entire B and C computation as tensor contractions
    @finch begin
        # Compute B as a single fused operation
        B_finch .= 0
        for i=_, j1=_, j2=_, k1=_, k2=_, m=_
            j_idx = (j1-1)*size(S1,1) + j2
            k_idx = (k1-1)*size(S1,2) + k2
            B_finch[i, m] += U2_finch[i, j_idx] * S1_finch[j1, k1] * S1_finch[j2, k2] * C2_finch[k_idx, m]
        end
        
        # Compute C as a single fused operation
        C_finch .= 0
        for i=_, j1=_, j2=_, k1=_, k2=_, m=_
            j_idx = (j1-1)*size(S_combined,1) + j2
            k_idx = (k1-1)*size(S_combined,2) + k2
            C_finch[i, m] += ∇₂_finch[i, j_idx] * S_combined[j1, k1] * S_combined[j2, k2] * C2_finch[k_idx, m]
        end
    end
    
    # Convert back and solve Sylvester
    B = Array(B_finch)
    C = Array(C_finch)
    return solve_sylvester_equation(A, B, C, ...)
end
```

This fundamental approach lets Finch see the entire mathematical structure and optimize globally.
