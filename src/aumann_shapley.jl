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
For `n_e = 7` this is 14 (2nd order) and 21 (3rd order).

This file provides:
* `continuous_coalition_mask_second_order` / `_third_order`
* `mask_directional_derivative_second_order` / `_third_order`
* `gausslegendre_unit_interval`
* The `calculate_aumann_shapley_*_order` drivers live in `moments.jl`.
=#

"""
$(SIGNATURES)
Multilinear extension of the second-order coalition indicator mask evaluated
at `x ∈ [0,1]^nᵉ`. Block layout: `[nᵉ, nᵉ², nˢ·nᵉ]`. For each block
entry indexed by tuple `I` of shock indices, `m_I(x) = ∏_{j ∈ unique(I)} x_j`
so that on `x ∈ {0,1}^nᵉ` the result equals the original `BitVector` mask.
"""
function continuous_coalition_mask_second_order(x::AbstractVector{T}, nᵉ::Int, nˢ::Int) where T <: Real
    block_sizes = (nᵉ, nᵉ^2, nˢ * nᵉ)
    N = sum(block_sizes)
    m = zeros(T, N)
    off = 0
    @inbounds for j in 1:nᵉ
        m[off + j] = x[j]
    end
    off += block_sizes[1]
    @inbounds for j in 1:nᵉ, k in 1:nᵉ
        m[off + (j - 1) * nᵉ + k] = (j == k) ? x[j] : x[j] * x[k]
    end
    off += block_sizes[2]
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
"""
function mask_directional_derivative_second_order!(dm::AbstractVector{T}, x::AbstractVector{T}, i::Int, nᵉ::Int, nˢ::Int) where T <: Real
    block_sizes = (nᵉ, nᵉ^2, nˢ * nᵉ)
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
    return dm
end

"""
$(SIGNATURES)
Multilinear extension of the third-order coalition indicator mask. Block
layout:
`(nᵉ, nᵉ², nˢ·nᵉ, nˢ·nᵉ, nˢ²·nᵉ, nˢ·nᵉ², nᵉ³)`. Each entry's mask value
equals the product of `x[j]` over the unique shock indices appearing in that
entry's index tuple.
"""
function continuous_coalition_mask_third_order(x::AbstractVector{T}, nᵉ::Int, nˢ::Int) where T <: Real
    block_sizes = (nᵉ, nᵉ^2, nˢ * nᵉ, nˢ * nᵉ, nˢ^2 * nᵉ, nˢ * nᵉ^2, nᵉ^3)
    N = sum(block_sizes)
    m = zeros(T, N)
    off = 0
    @inbounds for j in 1:nᵉ
        m[off + j] = x[j]
    end
    off += block_sizes[1]
    @inbounds for j in 1:nᵉ, k in 1:nᵉ
        m[off + (j - 1) * nᵉ + k] = (j == k) ? x[j] : x[j] * x[k]
    end
    off += block_sizes[2]
    @inbounds for a in 1:nˢ, j in 1:nᵉ
        m[off + (a - 1) * nᵉ + j] = x[j]
    end
    off += block_sizes[3]
    @inbounds for a in 1:nˢ, j in 1:nᵉ
        m[off + (a - 1) * nᵉ + j] = x[j]
    end
    off += block_sizes[4]
    @inbounds for p in 1:nˢ^2, j in 1:nᵉ
        m[off + (p - 1) * nᵉ + j] = x[j]
    end
    off += block_sizes[5]
    @inbounds for a in 1:nˢ, j in 1:nᵉ, k in 1:nᵉ
        m[off + (a - 1) * nᵉ^2 + (j - 1) * nᵉ + k] = (j == k) ? x[j] : x[j] * x[k]
    end
    off += block_sizes[6]
    @inbounds for j in 1:nᵉ, k in 1:nᵉ, l in 1:nᵉ
        # multilinear extension over unique({j,k,l})
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
`n ∈ {2, 3, 4}`. Avoids a runtime dependency on `FastGaussQuadrature`. The
nodes and weights integrate polynomials of degree ≤ `2n − 1` exactly.
"""
function gausslegendre_unit_interval(n::Int)
    if n == 2
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
    else
        error("Gauss–Legendre on [0,1] hand-coded only for n ∈ {2,3,4}; got n=$n")
    end
end
