"""
Polynomial-coalition representation for the inversion-filter Shapley shock
decomposition under pruned higher-order solutions.

For pruned k-order perturbation, the value V(S) of a model variable as a
function of the active shock coalition S ⊆ {1..nᵉ} is a polynomial of total
degree ≤ k in the binary indicators `1_S`. Such a polynomial has only
`Σ_{j=0..k} C(nᵉ, j) ≈ nᵉᵏ/k!` Möbius coefficients — far fewer than the
2^nᵉ coalition values an exhaustive enumeration would compute.

The per-period state recursion in `src/filter/inversion.jl`
(`shapley_shock_decomposition_pruned_{2,3}_order!`) propagates these
coefficients directly via `PolyState` + Kronecker products, then aggregates
to per-shock Shapley shares via `shapley_from_poly!` (each monomial of
size `m` contributes its coefficient equally to its `m` member shocks).
This is the exact Shapley value (no quadrature error) at the minimum
number of state propagations.

The unconditional higher-order variance decomposition uses the
Aumann–Shapley path-integral driver in `src/aumann_shapley.jl` instead,
which is cheaper for that setting because the characteristic function
there requires a Lyapunov solve per evaluation.
"""

"""
    MonomialIndex(nᵉ, k)

Enumerate every subset `T ⊆ {1..nᵉ}` with `|T| ≤ k`. Each subset is encoded
as a `UInt64` bitmask (bit `i-1` set iff shock `i ∈ T`); ids are assigned
in order of increasing size, then lexicographic mask. Requires `nᵉ ≤ 64`.

Fields: `masks[id]`, `id[mask]`, `sizes[id]`, `members[id]` (sorted shock
indices), `n_monomials`.
"""
struct MonomialIndex
    nᵉ::Int
    k::Int
    masks::Vector{UInt64}
    id::Dict{UInt64, Int}
    sizes::Vector{Int}
    members::Vector{Vector{Int}}
    n_monomials::Int
end

function MonomialIndex(nᵉ::Int, k::Int)
    @assert nᵉ ≤ 64 "MonomialIndex requires nᵉ ≤ 64 (UInt64 bitmask)."
    @assert k ≥ 0 "Monomial degree k must be non-negative."

    masks   = UInt64[UInt64(0)]
    members = Vector{Vector{Int}}([Int[]])
    sizes   = Int[0]

    for sz in 1:min(k, nᵉ)
        for combo in combinations(1:nᵉ, sz)
            mask = UInt64(0)
            for i in combo
                mask |= UInt64(1) << (i - 1)
            end
            push!(masks, mask)
            push!(members, copy(combo))
            push!(sizes, sz)
        end
    end

    id = Dict{UInt64, Int}(m => j for (j, m) in enumerate(masks))

    return MonomialIndex(nᵉ, k, masks, id, sizes, members, length(masks))
end

"""
    PolyState{T}

Polynomial in the indicator vector `1_S` of total degree ≤ `idx.k`, stored
as an `(n_state × n_monomials)` coefficient matrix. The polynomial value
at coalition `S` is `p(S)[v] = Σ_j coefs[v, j] · 1_{T_j ⊆ S}`.
"""
struct PolyState{T <: Real}
    coefs::Matrix{T}
    idx::MonomialIndex
end

PolyState(n_state::Int, idx::MonomialIndex) =
    PolyState{Float64}(zeros(Float64, n_state, idx.n_monomials), idx)

"""
    poly_constant!(p, v)

Set `p` to the degree-0 polynomial whose constant term equals `v`.
"""
function poly_constant!(p::PolyState, v::AbstractVector)
    @assert length(v) == size(p.coefs, 1) "constant vector length mismatch"
    fill!(p.coefs, 0)
    @inbounds for i in eachindex(v)
        p.coefs[i, 1] = v[i]   # column 1 is the empty-subset (constant) monomial
    end
    return p
end

"""
    poly_aug₁!(out, past, past_idx, include_constant, sck)

Build the augmented degree-1 polynomial state used by pruned higher-order
recursions:

    aug[1:n_past]              = past[past_idx]
    aug[n_past + 1]            = include_constant ? 1 : 0
    aug[n_past + 1 + i]        = sck[i] · 1_S(i)

`out` must have `size(out.coefs, 1) == length(past_idx) + 1 + nᵉ` and share
`out.idx` with `past`.
"""
function poly_aug₁!(out::PolyState, past::PolyState,
                    past_idx::AbstractVector{Int},
                    include_constant::Bool,
                    sck::AbstractVector)
    idx = out.idx
    nᵉ = idx.nᵉ
    n_past = length(past_idx)
    @assert size(out.coefs, 1) == n_past + 1 + nᵉ "aug₁ size mismatch"
    @assert length(sck) == nᵉ "shock vector length must equal nᵉ"
    @assert past.idx === idx "MonomialIndex mismatch"

    fill!(out.coefs, 0)

    @inbounds for j in 1:idx.n_monomials, (row, src) in enumerate(past_idx)
        out.coefs[row, j] = past.coefs[src, j]
    end

    @inbounds out.coefs[n_past + 1, 1] = include_constant ? 1.0 : 0.0

    @inbounds for i in 1:nᵉ
        col = idx.id[UInt64(1) << (i - 1)]
        out.coefs[n_past + 1 + i, col] = sck[i]
    end

    return out
end

"""
    poly_kron!(out, a, b; truncate_to = a.idx.k)

Polynomial Kronecker product `out(S) = kron(a(S), b(S))`, truncated to
total degree `truncate_to`. Pairs whose support union exceeds the cap are
discarded.
"""
function poly_kron!(out::PolyState, a::PolyState, b::PolyState;
                    truncate_to::Int = a.idx.k)
    idx = out.idx
    @assert idx === a.idx === b.idx "MonomialIndex mismatch"
    na = size(a.coefs, 1)
    nb = size(b.coefs, 1)
    @assert size(out.coefs, 1) == na * nb "kron output size mismatch"

    fill!(out.coefs, 0)

    @inbounds for ja in 1:idx.n_monomials
        idx.sizes[ja] > truncate_to && continue
        ma = idx.masks[ja]
        any_a = false
        for α in 1:na
            if a.coefs[α, ja] != 0
                any_a = true; break
            end
        end
        any_a || continue

        for jb in 1:idx.n_monomials
            idx.sizes[jb] > truncate_to && continue
            mu = ma | idx.masks[jb]
            count_ones(mu) > truncate_to && continue
            jt = idx.id[mu]

            for α in 1:na
                aval = a.coefs[α, ja]
                aval == 0 && continue
                base = (α - 1) * nb
                for β in 1:nb
                    bval = b.coefs[β, jb]
                    bval == 0 && continue
                    out.coefs[base + β, jt] += aval * bval
                end
            end
        end
    end

    return out
end

"""
    poly_apply!(out, M, p; α = 1.0, β = 0.0)

Compute `out.coefs .= α * M * p.coefs .+ β * out.coefs` via one BLAS gemm
(the linear map `M` acts on each monomial column independently).
"""
function poly_apply!(out::PolyState, M::AbstractMatrix, p::PolyState;
                     α::Real = 1.0, β::Real = 0.0)
    @assert out.idx === p.idx "MonomialIndex mismatch"
    @assert size(M, 2) == size(p.coefs, 1) "operator/state size mismatch"
    @assert size(M, 1) == size(out.coefs, 1) "operator/output size mismatch"
    if β == 0
        ℒ.mul!(out.coefs, M, p.coefs)
        if α != 1
            out.coefs .*= α
        end
    else
        ℒ.mul!(out.coefs, M, p.coefs, α, β)
    end
    return out
end

"""Value `p(∅)` at the empty coalition (no shocks active)."""
@inline poly_value_at_empty(p::PolyState) = @view p.coefs[:, 1]

"""Value `p(N)` at the grand coalition: row-sum of the coefficient matrix."""
poly_value_at_full(p::PolyState) = vec(sum(p.coefs, dims = 2))

"""
    shapley_from_poly!(φ, p::PolyState)
    shapley_from_poly!(φ, coefs::AbstractMatrix, idx::MonomialIndex)

Aggregate per-shock Shapley values from polynomial coefficients. Each
monomial of size `m ≥ 1` contributes its full coefficient divided equally
among its `m` member shocks (Shapley symmetry on monomial games).
`φ` must be `(size(coefs, 1), idx.nᵉ)` and is overwritten.
"""
function shapley_from_poly!(φ::AbstractMatrix, coefs::AbstractMatrix,
                            idx::MonomialIndex)
    n = size(coefs, 1)
    @assert size(φ) == (n, idx.nᵉ) "shapley output size mismatch"
    @assert size(coefs, 2) == idx.n_monomials "coefs/monomial-count mismatch"
    fill!(φ, 0)
    @inbounds for j in 2:idx.n_monomials
        m = idx.sizes[j]
        invm = 1.0 / m
        members = idx.members[j]
        for v in 1:n
            cv = coefs[v, j]
            cv == 0 && continue
            contrib = cv * invm
            for i in members
                φ[v, i] += contrib
            end
        end
    end
    return φ
end

shapley_from_poly!(φ::AbstractMatrix, p::PolyState) =
    shapley_from_poly!(φ, p.coefs, p.idx)

