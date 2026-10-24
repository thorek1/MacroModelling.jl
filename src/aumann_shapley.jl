#=
Aumann–Shapley driver for higher-order variance decomposition.

For a coalition characteristic function `V(S)` defined as the diagonal of the
pruned higher-order variance under shock subset `S`, the Möbius coefficients
of the multilinear extension `Ṽ(x) = Σ_T c_T ∏_{i∈T} x_i` are zero for
`|T| > d` (d = 4 for pruned 2nd order, d = 6 for pruned 3rd order).

By the Aumann–Shapley theorem, the discrete Shapley value coincides with

    φ_i = ∫₀¹ ∂Ṽ(t·𝟙)/∂x_i  dt

For a polynomial of total degree ≤ d in `x`, the integrand is a polynomial in
`t` of degree ≤ d − 1 and is integrated exactly by Gauss–Legendre with
`ceil(d/2)` nodes (2 for 2nd order, 3 for 3rd order). At each node we
need `n_e` forward Lyapunov sensitivities (one per direction `eᵢ`); each
sensitivity is one Lyapunov solve with right-hand side `Ċ(x;eᵢ)`.

Total Lyapunov solves: `n_nodes · n_e`, independent of subset enumeration.
The production drivers default to the low-order rules (2 nodes at second
order, 3 at third order) and incrementally rerun with up to 7 nodes when the
relative Shapley-efficiency closure error exceeds `1e-3`.

This file provides:
* `continuous_coalition_mask_second_order` / `_third_order`
* `mask_directional_derivative_second_order` / `_third_order`
* `gausslegendre_unit_interval`
* The `calculate_aumann_shapley_*_order` drivers live in `moments.jl`.
=#
const AUMANN_SHAPLEY_REFINEMENT_RTOL = 1e-3
const AUMANN_SHAPLEY_REFINEMENT_MAX_NODES = 7

"""
$(SIGNATURES)
Multilinear extension of the second-order coalition indicator mask evaluated
at `x ∈ [0,1]^nᵉ`. Block layout: `[nᵉ, nᵉ², nˢ·nᵉ]`. For each block
entry indexed by tuple `I` of shock indices, `m_I(x) = ∏_{j ∈ unique(I)} x_j`
so that on `x ∈ {0,1}^nᵉ` the result equals the original `BitVector` mask.

In plain math, the three blocks of the augmented shock vector ê at second
order are `(eⱼ)`, `(eⱼ·eₖ)`, `(sₐ·eⱼ)`. The continuous mask m(x) returns,
component by component:
  block 1 (size nᵉ):   m[j]        = xⱼ
  block 2 (size nᵉ²):  m[j,k]      = xⱼ · xₖ      (= xⱼ on the diagonal j=k)
  block 3 (size nˢnᵉ): m[a,j]      = xⱼ           (state index a does not enter)
At x ∈ {0,1}ⁿᵉ this collapses to the indicator of "every shock index in the
component lies in the coalition S", recovering the discrete coalition mask.
"""
function continuous_coalition_mask_second_order(x::AbstractVector{T}, nᵉ::Int, nˢ::Int) where T <: Real
    block_sizes = (nᵉ, nᵉ^2, nˢ * nᵉ)
    N = sum(block_sizes)
    m = zeros(T, N)
    off = 0
    # Block 1: single shock eⱼ → mask value xⱼ.
    @inbounds for j in 1:nᵉ
        m[off + j] = x[j]
    end
    off += block_sizes[1]
    # Block 2: shock-by-shock kron eⱼ·eₖ → product over unique indices.
    @inbounds for j in 1:nᵉ, k in 1:nᵉ
        m[off + (j - 1) * nᵉ + k] = (j == k) ? x[j] : x[j] * x[k]
    end
    off += block_sizes[2]
    # Block 3: state-by-shock kron sₐ·eⱼ → only shock index j matters.
    @inbounds for a in 1:nˢ, j in 1:nᵉ
        m[off + (a - 1) * nᵉ + j] = x[j]
    end
    return m
end

"""
$(SIGNATURES)
Directional derivative `∂m/∂x_i` of `continuous_coalition_mask_second_order`
at point `x`, returned as a dense vector. Sparse in structure (only entries
whose monomial contains shock `i` are nonzero) but materialised dense for
straightforward use in BLAS calls downstream.
"""
function mask_directional_derivative_second_order(x::AbstractVector{T}, i::Int, nᵉ::Int, nˢ::Int) where T <: Real
    block_sizes = (nᵉ, nᵉ^2, nˢ * nᵉ)
    N = sum(block_sizes)
    dm = zeros(T, N)
    mask_directional_derivative_second_order!(dm, x, i, nᵉ, nˢ)
    return dm
end

"""
$(SIGNATURES)
In-place version of `mask_directional_derivative_second_order`. Writes into
pre-allocated output vector `dm` (length `nᵉ + nᵉ² + nˢ·nᵉ`).

In plain math, this is ∂m(x)/∂xᵢ for the second-order mask. Per block:
  block 1:  ∂m[j]/∂xᵢ   = δⱼᵢ
  block 2:  ∂m[j,k]/∂xᵢ = δⱼᵢ·xₖ + δₖᵢ·xⱼ              (= δⱼᵢ on diag j=k)
  block 3:  ∂m[a,j]/∂xᵢ = δⱼᵢ
Note the support is sparse: only components whose index tuple contains shock i
have a nonzero derivative.
"""
function mask_directional_derivative_second_order!(dm::AbstractVector{T}, x::AbstractVector{T}, i::Int, nᵉ::Int, nˢ::Int) where T <: Real
    block_sizes = (nᵉ, nᵉ^2, nˢ * nᵉ)
    fill!(dm, zero(T))
    off = 0
    # Block 1: ∂(xⱼ)/∂xᵢ = δⱼᵢ.
    @inbounds dm[off + i] = one(T)
    off += block_sizes[1]
    # Block 2: ∂(xⱼxₖ)/∂xᵢ = δⱼᵢxₖ + δₖᵢxⱼ; on the diagonal j=k=i it is 1.
    @inbounds for j in 1:nᵉ, k in 1:nᵉ
        if j == k && j == i
            dm[off + (j - 1) * nᵉ + k] = one(T)
        elseif j != k
            if j == i
                dm[off + (j - 1) * nᵉ + k] = x[k]
            elseif k == i
                dm[off + (j - 1) * nᵉ + k] = x[j]
            end
        end
    end
    off += block_sizes[2]
    # Block 3: ∂(xⱼ)/∂xᵢ = δⱼᵢ, independent of the state index a.
    @inbounds for a in 1:nˢ
        dm[off + (a - 1) * nᵉ + i] = one(T)
    end
    return dm
end

"""
$(SIGNATURES)
Multilinear extension of the third-order coalition indicator mask. Block
layout:
`(nᵉ, nᵉ², nˢ·nᵉ, nˢ·nᵉ, nˢ²·nᵉ, nˢ·nᵉ², nᵉ³)`. Each entry's mask value
equals the product of `x[j]` over the unique shock indices appearing in that
entry's index tuple.

In plain math, the seven blocks of the augmented shock vector ê at third
order are `(eⱼ), (eⱼ·eₖ), (sₐ·eⱼ), (sₐ·eⱼ), (sₐ·sᵦ·eⱼ), (sₐ·eⱼ·eₖ), (eⱼ·eₖ·eₗ)`.
The continuous mask m(x) returns, block by block:
  block 1: xⱼ
  block 2: xⱼ·xₖ                     (= xⱼ on the diagonal)
  block 3: xⱼ                         (state indices ignored)
  block 4: xⱼ                         (same shape as block 3)
  block 5: xⱼ                         (two state indices ignored)
  block 6: xⱼ·xₖ                     (state index ignored)
  block 7: xⱼ·xₖ·xₗ                  (product over unique({j,k,l}))
At x ∈ {0,1}ⁿᵉ this matches the discrete coalition indicator.
"""
function continuous_coalition_mask_third_order(x::AbstractVector{T}, nᵉ::Int, nˢ::Int) where T <: Real
    block_sizes = (nᵉ, nᵉ^2, nˢ * nᵉ, nˢ * nᵉ, nˢ^2 * nᵉ, nˢ * nᵉ^2, nᵉ^3)
    N = sum(block_sizes)
    m = zeros(T, N)
    off = 0
    # Block 1 — eⱼ: m = xⱼ
    @inbounds for j in 1:nᵉ
        m[off + j] = x[j]
    end
    off += block_sizes[1]
    # Block 2 — eⱼ·eₖ: m = xⱼ·xₖ (xⱼ on the diagonal)
    @inbounds for j in 1:nᵉ, k in 1:nᵉ
        m[off + (j - 1) * nᵉ + k] = (j == k) ? x[j] : x[j] * x[k]
    end
    off += block_sizes[2]
    # Block 3 — sₐ·eⱼ: m = xⱼ (state index a irrelevant)
    @inbounds for a in 1:nˢ, j in 1:nᵉ
        m[off + (a - 1) * nᵉ + j] = x[j]
    end
    off += block_sizes[3]
    # Block 4 — same shape as block 3
    @inbounds for a in 1:nˢ, j in 1:nᵉ
        m[off + (a - 1) * nᵉ + j] = x[j]
    end
    off += block_sizes[4]
    # Block 5 — sₐ·sᵦ·eⱼ: m = xⱼ (two state indices irrelevant)
    @inbounds for p in 1:nˢ^2, j in 1:nᵉ
        m[off + (p - 1) * nᵉ + j] = x[j]
    end
    off += block_sizes[5]
    # Block 6 — sₐ·eⱼ·eₖ: m = xⱼ·xₖ (state index irrelevant)
    @inbounds for a in 1:nˢ, j in 1:nᵉ, k in 1:nᵉ
        m[off + (a - 1) * nᵉ^2 + (j - 1) * nᵉ + k] = (j == k) ? x[j] : x[j] * x[k]
    end
    off += block_sizes[6]
    # Block 7 — eⱼ·eₖ·eₗ: m = ∏_{q ∈ unique({j,k,l})} x_q
    @inbounds for j in 1:nᵉ, k in 1:nᵉ, l in 1:nᵉ
        if j == k == l
            v = x[j]
        elseif j == k
            v = x[j] * x[l]
        elseif j == l
            v = x[j] * x[k]
        elseif k == l
            v = x[j] * x[k]
        else
            v = x[j] * x[k] * x[l]
        end
        m[off + (j - 1) * nᵉ^2 + (k - 1) * nᵉ + l] = v
    end
    return m
end

"""
$(SIGNATURES)
Directional derivative `∂m/∂x_i` of `continuous_coalition_mask_third_order`
at `x`, returned dense.
"""
function mask_directional_derivative_third_order(x::AbstractVector{T}, i::Int, nᵉ::Int, nˢ::Int) where T <: Real
    block_sizes = (nᵉ, nᵉ^2, nˢ * nᵉ, nˢ * nᵉ, nˢ^2 * nᵉ, nˢ * nᵉ^2, nᵉ^3)
    N = sum(block_sizes)
    dm = zeros(T, N)
    mask_directional_derivative_third_order!(dm, x, i, nᵉ, nˢ)
    return dm
end

"""
$(SIGNATURES)
In-place version of `mask_directional_derivative_third_order`. Writes into
pre-allocated output vector `dm`.

In plain math, this is ∂m(x)/∂xᵢ for the third-order mask. The general rule
is the product rule applied to the multilinear monomial of each component:
∂(∏_{j∈U} xⱼ)/∂xᵢ = ∏_{j∈U\\{i}} xⱼ if i ∈ U, else 0, where U is the set
of unique shock indices in that component.  Block 7 (triple e·e·e) is the
only place where three indices can coincide, so it splits into three sub-cases
(all equal / two equal / all distinct) of the same product rule.
"""
function mask_directional_derivative_third_order!(dm::AbstractVector{T}, x::AbstractVector{T}, i::Int, nᵉ::Int, nˢ::Int) where T <: Real
    block_sizes = (nᵉ, nᵉ^2, nˢ * nᵉ, nˢ * nᵉ, nˢ^2 * nᵉ, nˢ * nᵉ^2, nᵉ^3)
    fill!(dm, zero(T))
    off = 0
    @inbounds dm[off + i] = one(T)                                       # block 1
    off += block_sizes[1]
    @inbounds for j in 1:nᵉ, k in 1:nᵉ                         # block 2
        if j == k && j == i
            dm[off + (j - 1) * nᵉ + k] = one(T)
        elseif j != k
            if j == i
                dm[off + (j - 1) * nᵉ + k] = x[k]
            elseif k == i
                dm[off + (j - 1) * nᵉ + k] = x[j]
            end
        end
    end
    off += block_sizes[2]
    @inbounds for a in 1:nˢ                                    # block 3
        dm[off + (a - 1) * nᵉ + i] = one(T)
    end
    off += block_sizes[3]
    @inbounds for a in 1:nˢ                                    # block 4
        dm[off + (a - 1) * nᵉ + i] = one(T)
    end
    off += block_sizes[4]
    @inbounds for p in 1:nˢ^2                                  # block 5
        dm[off + (p - 1) * nᵉ + i] = one(T)
    end
    off += block_sizes[5]
    @inbounds for a in 1:nˢ, j in 1:nᵉ, k in 1:nᵉ              # block 6
        if j == k && j == i
            dm[off + (a - 1) * nᵉ^2 + (j - 1) * nᵉ + k] = one(T)
        elseif j != k
            if j == i
                dm[off + (a - 1) * nᵉ^2 + (j - 1) * nᵉ + k] = x[k]
            elseif k == i
                dm[off + (a - 1) * nᵉ^2 + (j - 1) * nᵉ + k] = x[j]
            end
        end
    end
    off += block_sizes[6]
    @inbounds for j in 1:nᵉ, k in 1:nᵉ, l in 1:nᵉ              # block 7
        v = zero(T)
        if j == k == l == i
            v = one(T)
        elseif j == k == l
            v = zero(T)
        elseif j == k
            v = (j == i ? x[l] : zero(T)) + (l == i ? x[j] : zero(T))
        elseif j == l
            v = (j == i ? x[k] : zero(T)) + (k == i ? x[j] : zero(T))
        elseif k == l
            v = (j == i ? x[k] : zero(T)) + (k == i ? x[j] : zero(T))
        else
            v = (j == i ? x[k] * x[l] : zero(T)) +
                (k == i ? x[j] * x[l] : zero(T)) +
                (l == i ? x[j] * x[k] : zero(T))
        end
        dm[off + (j - 1) * nᵉ^2 + (k - 1) * nᵉ + l] = v
    end
    return dm
end

"""
$(SIGNATURES)
Hand-coded Gauss–Legendre nodes/weights on the unit interval `[0, 1]` for
`n ∈ {1, 2, 3, 4}`. Avoids a runtime dependency on `FastGaussQuadrature`. The
nodes and weights integrate polynomials of degree ≤ `2n − 1` exactly.

In plain math, this returns `(t_k, w_k)` such that
    ∫₀¹ p(t) dt ≈ Σ_k w_k · p(t_k)
is exact for every polynomial p of degree ≤ 2n−1. We obtain the unit-interval
rule by affinely mapping the standard `[-1,1]` Gauss–Legendre rule:
`t = (ξ + 1)/2` with weights halved accordingly.
"""
function gausslegendre_unit_interval(n::Int)
    if n == 1
        return [0.5], [1.0]
    elseif n == 2
        a = 1 / sqrt(3.0)
        return [0.5 - a/2, 0.5 + a/2], [0.5, 0.5]
    elseif n == 3
        a = sqrt(3/5)
        return [0.5 - a/2, 0.5, 0.5 + a/2], [5/18, 4/9, 5/18]
    elseif n == 4
        c = sqrt(6/5)
        n1 = sqrt(3/7 - 2/7 * c); n2 = sqrt(3/7 + 2/7 * c)
        w1 = (18 + sqrt(30)) / 36; w2 = (18 - sqrt(30)) / 36
        nodes_pm1   = [-n2, -n1, n1, n2]
        weights_pm1 = [w2, w1, w1, w2]
        return (nodes_pm1 .+ 1) ./ 2, weights_pm1 ./ 2
    elseif n == 5
        nodes_pm1 = [-0.9061798459386640, -0.5384693101056831, 0.0,
                     0.5384693101056831, 0.9061798459386640]
        weights_pm1 = [0.2369268850561891, 0.4786286704993665,
                       0.5688888888888889, 0.4786286704993665,
                       0.2369268850561891]
        return (nodes_pm1 .+ 1) ./ 2, weights_pm1 ./ 2
    elseif n == 6
        nodes_pm1 = [-0.9324695142031521, -0.6612093864662645,
                     -0.2386191860831969, 0.2386191860831969,
                     0.6612093864662645, 0.9324695142031521]
        weights_pm1 = [0.1713244923791704, 0.3607615730481386,
                       0.4679139345726910, 0.4679139345726910,
                       0.3607615730481386, 0.1713244923791704]
        return (nodes_pm1 .+ 1) ./ 2, weights_pm1 ./ 2
    elseif n == 7
        nodes_pm1 = [-0.9491079123427585, -0.7415311855993945,
                     -0.4058451513773972, 0.0,
                     0.4058451513773972, 0.7415311855993945,
                     0.9491079123427585]
        weights_pm1 = [0.1294849661688697, 0.2797053914892766,
                       0.3818300505051189, 0.4179591836734694,
                       0.3818300505051189, 0.2797053914892766,
                       0.1294849661688697]
        return (nodes_pm1 .+ 1) ./ 2, weights_pm1 ./ 2
    else
        error("Gauss–Legendre on [0,1] hand-coded only for n ∈ {1,2,3,4,5,6,7}; got n=$n")
    end
end
