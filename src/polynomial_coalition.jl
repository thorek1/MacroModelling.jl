"""
Polynomial-coalition representation for marginal-contribution (Shapley)
attribution under pruned higher-order solutions.

For pruned k-order perturbation, the value V(S) of a model variable as a
function of the active shock coalition S ⊆ {1..nᵉ} is a polynomial of total
degree ≤ k in the binary indicators `1_S`. Such a polynomial is fully
described by `Σ_{j=0..k} C(nᵉ, j) ≈ nᵉᵏ/k!` coefficients — far fewer than
the 2^nᵉ coalition values an exhaustive enumeration would compute.

This file provides

  - `MonomialIndex(nᵉ, k)`            : enumerate the subsets of size ≤ k
                                        (bitmask + Dict lookup)
  - `PolyState{T}`                    : (n_state × n_monomials) coefficient
                                        matrix backing one pruned-state
                                        component
  - `poly_constant!`, `poly_indicator!`: build degree-0/degree-1 columns
  - `poly_aug₁!`                      : assemble the augmented degree-1
                                        state `[past; const; sck·1_S]`
  - `poly_kron!`                      : polynomial Kronecker product,
                                        union of supporting subsets via
                                        bitmask OR (truncated at k)
  - `poly_apply!`                     : matrix multiply (one gemm over the
                                        coefficient matrix)
  - `poly_value_at_empty`,
    `poly_value_at_full`              : V(∅) and V(N) extraction
  - `shapley_from_poly!`              : equal-share Shapley aggregation
                                        over monomials

The whole module is self-contained and has no MacroModelling-specific
dependencies, so it is reused by both the inversion-filter shock
decomposition and (planned) the higher-order variance decomposition.
"""

"""
    MonomialIndex(nᵉ, k)

Enumerate every subset `T ⊆ {1..nᵉ}` with `|T| ≤ k`. Each subset is encoded
as a `UInt64` bitmask (bit `i-1` set iff shock `i ∈ T`); a stable integer
id is assigned in order of increasing size, then lexicographic mask.

Fields:
- `nᵉ`              number of shocks (must be ≤ 64)
- `k`               maximum monomial degree
- `masks`           `Vector{UInt64}` — `masks[id]` is the bitmask of
                    monomial `id`
- `id`              `Dict{UInt64, Int}` — `id[mask]` returns the id (or
                    raises if the mask exceeds degree `k`)
- `sizes`           `Vector{Int}` — `sizes[id] == count_ones(masks[id])`
- `members`         `Vector{Vector{Int}}` — sorted shock indices in each
                    monomial; `members[1] == Int[]`
- `n_monomials`     total number of monomials = `Σ_{j=0..k} C(nᵉ, j)`
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

    masks = UInt64[]
    members = Vector{Vector{Int}}()
    sizes = Int[]

    push!(masks, UInt64(0))
    push!(members, Int[])
    push!(sizes, 0)

    for sz in 1:min(k, nᵉ)
        # `Combinatorics.combinations(1:nᵉ, sz)` enumerates size-`sz`
        # subsets in lex order; copy to own the vectors and to give
        # each monomial a stable id
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

    id = Dict{UInt64, Int}()
    for (j, m) in enumerate(masks)
        id[m] = j
    end

    return MonomialIndex(nᵉ, k, masks, id, sizes, members, length(masks))
end

# Lex-order enumeration of size-`sz` subsets of {1..nᵉ}.
# (No longer used — replaced by `Combinatorics.combinations`. Retained
# only to keep symbol-name churn small in tests; safe to delete.)

"""
    full_mask(idx)

Bitmask of the grand coalition `N = {1..nᵉ}`.
"""
@inline full_mask(idx::MonomialIndex) = idx.nᵉ == 0 ? UInt64(0) : (UInt64(1) << idx.nᵉ) - UInt64(1)

"""
    PolyState{T}

A polynomial state in the indicator vector `1_S`, of total degree ≤ `idx.k`.

`coefs[v, j]` is the coefficient of monomial `j` for state-vector entry `v`.
Concretely the polynomial value at coalition `S` is

    p(S)[v] = Σ_j coefs[v, j] · 1_{T_j ⊆ S}

i.e. the sum of all coefficient columns whose subset is contained in `S`.
"""
struct PolyState{T <: Real}
    coefs::Matrix{T}
    idx::MonomialIndex
end

PolyState(::Type{T}, n_state::Int, idx::MonomialIndex) where {T <: Real} =
    PolyState{T}(zeros(T, n_state, idx.n_monomials), idx)

PolyState(n_state::Int, idx::MonomialIndex) = PolyState(Float64, n_state, idx)

@inline n_state(p::PolyState) = size(p.coefs, 1)

"""
    poly_zero!(p)

Reset all coefficients to zero (in place).
"""
function poly_zero!(p::PolyState)
    fill!(p.coefs, 0)
    return p
end

"""
    poly_constant!(p, v)

Set `p` to the degree-0 polynomial whose constant term equals the vector
`v` (length `n_state`). All other monomial columns are set to zero.
"""
function poly_constant!(p::PolyState, v::AbstractVector)
    @assert length(v) == n_state(p) "constant vector length mismatch"
    poly_zero!(p)
    @inbounds for i in eachindex(v)
        p.coefs[i, 1] = v[i]   # column 1 is monomial id of ∅
    end
    return p
end

"""
    poly_aug₁!(out, past, past_idx, include_constant, sck)

Build the augmented degree-1 polynomial state used by pruned higher-order
recursions:

    aug[1:n_past]              = past[past_idx]              (degree ≤ 1)
    aug[n_past + 1]            = include_constant ? 1 : 0    (degree 0)
    aug[n_past + 1 + i]        = sck[i] · 1_S(i)             (degree 1)

`past` is a `PolyState` (degree ≤ 1 in 1_S), `past_idx` selects the
state indices to carry forward, `sck` is the shock realisation vector
(length nᵉ), and `include_constant` controls the middle "1" used by the
linear `aug₁` and the "0" used by `aug₁̂` in the 3rd-order recursion.

`out` must have `n_state == length(past_idx) + 1 + nᵉ` and share
`out.idx` with `past`.
"""
function poly_aug₁!(out::PolyState, past::PolyState,
                    past_idx::AbstractVector{Int},
                    include_constant::Bool,
                    sck::AbstractVector)
    idx = out.idx
    nᵉ = idx.nᵉ
    n_past = length(past_idx)
    @assert n_state(out) == n_past + 1 + nᵉ "aug₁ size mismatch"
    @assert length(sck) == nᵉ "shock vector length must equal nᵉ"
    @assert past.idx === idx "MonomialIndex mismatch"

    poly_zero!(out)

    # carry past entries (any degree present in `past`)
    @inbounds for j in 1:idx.n_monomials, (row, src) in enumerate(past_idx)
        out.coefs[row, j] = past.coefs[src, j]
    end

    # constant block
    @inbounds out.coefs[n_past + 1, 1] = include_constant ? 1.0 : 0.0

    # shock block: each entry is sck[i] · 1_S(i), so it lives in the
    # column for the singleton monomial {i}
    @inbounds for i in 1:nᵉ
        col = idx.id[UInt64(1) << (i - 1)]
        out.coefs[n_past + 1 + i, col] = sck[i]
    end

    return out
end

"""
    poly_kron!(out, a, b; truncate_to = a.idx.k)

Polynomial Kronecker product. Computes the product polynomial
`p(S) = kron(a(S), b(S))` term-by-term:

    out[(α-1)·n_b + β, j_T] = Σ_{(j_a, j_b): T_a ∪ T_b = T} a[α, j_a] · b[β, j_b]

`out.idx` must equal `a.idx` (and `b.idx`); pairs whose union exceeds
`truncate_to` are discarded. The output `out` must have
`n_state(out) == n_state(a) * n_state(b)` and is overwritten.
"""
function poly_kron!(out::PolyState, a::PolyState, b::PolyState;
                    truncate_to::Int = a.idx.k)
    idx = out.idx
    @assert idx === a.idx === b.idx "MonomialIndex mismatch"
    na = n_state(a)
    nb = n_state(b)
    @assert n_state(out) == na * nb "kron output size mismatch"

    poly_zero!(out)

    @inbounds for ja in 1:idx.n_monomials
        size_a = idx.sizes[ja]
        size_a > truncate_to && continue
        ma = idx.masks[ja]
        # quick check: any nonzero in column ja of a?
        any_a = false
        for α in 1:na
            if a.coefs[α, ja] != 0
                any_a = true; break
            end
        end
        any_a || continue

        for jb in 1:idx.n_monomials
            size_b = idx.sizes[jb]
            size_b > truncate_to && continue
            mb = idx.masks[jb]
            mu = ma | mb
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

Compute `out.coefs .= α * M * p.coefs .+ β * out.coefs`. One BLAS gemm
applied to the whole coefficient block — the linear map `M` acts on
each monomial's coefficient column independently.
"""
function poly_apply!(out::PolyState, M::AbstractMatrix, p::PolyState;
                     α::Real = 1.0, β::Real = 0.0)
    @assert out.idx === p.idx "MonomialIndex mismatch"
    @assert size(M, 2) == n_state(p) "operator/state size mismatch"
    @assert size(M, 1) == n_state(out) "operator/output size mismatch"
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

"""
    poly_axpy!(out, α, p)

In-place `out.coefs .+= α .* p.coefs`. Used to accumulate the multiple
contributions to the higher-order pruning components (e.g.
`state₂ = 𝐒[1]·aug₂ + 𝐒[2]·kron(aug₁, aug₁)/2`).
"""
function poly_axpy!(out::PolyState, α::Real, p::PolyState)
    @assert out.idx === p.idx "MonomialIndex mismatch"
    @assert size(out.coefs) == size(p.coefs) "size mismatch"
    @inbounds @simd for i in eachindex(out.coefs)
        out.coefs[i] += α * p.coefs[i]
    end
    return out
end

"""
    poly_value_at_empty(p)

Return `p(∅)`, the value at the empty coalition (no shocks active).
This is just the constant column of `p.coefs`.
"""
@inline poly_value_at_empty(p::PolyState) = @view p.coefs[:, 1]

"""
    poly_value_at_full(p)

Return `p(N)`, the value at the grand coalition (all shocks active).
Since `1_T(N) = 1` for every `T ⊆ N`, this is the row-sum of the
coefficient matrix.
"""
poly_value_at_full(p::PolyState) = vec(sum(p.coefs, dims = 2))

"""
    shapley_from_poly!(φ, p::PolyState)
    shapley_from_poly!(φ, coefs::AbstractMatrix, idx::MonomialIndex)

Aggregate per-shock Shapley values from polynomial coefficients.

For a polynomial coalitional game with monomials `T ⊆ {1..nᵉ}`, the
Shapley value satisfies the equal-share property: each monomial of size
`m ≥ 1` contributes its full coefficient divided equally among its `m`
member shocks (this follows from Shapley symmetry on monomial games).

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
        m == 0 && continue
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

"""
    poly_coefs_from_subset_values!(poly_coefs, V_at_T, idx)

Given `V_at_T[:, j] == V(T_j)`, the value of a polynomial coalitional game
at the subset corresponding to monomial id `j`, fill `poly_coefs[:, j]`
with the unanimity-basis (i.e. polynomial) coefficient at monomial `T_j`,
using the recursive identity

    V(T) = Σ_{T' ⊆ T} poly_coefs[T']
  ⇒ poly_coefs[T] = V(T) − Σ_{T' ⊊ T} poly_coefs[T']

Monomials are processed in increasing-size order (already guaranteed by
`MonomialIndex` construction), so every proper subset's coefficient is
already available when its parent is processed.

`poly_coefs` and `V_at_T` must be `(n_state, idx.n_monomials)`.
"""
function poly_coefs_from_subset_values!(poly_coefs::AbstractMatrix,
                                        V_at_T::AbstractMatrix,
                                        idx::MonomialIndex)
    @assert size(poly_coefs) == size(V_at_T) "size mismatch"
    @assert size(poly_coefs, 2) == idx.n_monomials "monomial count mismatch"
    fill!(poly_coefs, 0)
    @views poly_coefs[:, 1] .= V_at_T[:, 1]
    @inbounds for j in 2:idx.n_monomials
        T_mask = idx.masks[j]
        @views poly_coefs[:, j] .= V_at_T[:, j]
        # iterate strict proper subsets of T_mask via Knuth's subset-of-mask trick
        sub = (T_mask - UInt64(1)) & T_mask
        while sub > 0
            Tp_id = idx.id[sub]
            @views poly_coefs[:, j] .-= poly_coefs[:, Tp_id]
            sub = (sub - UInt64(1)) & T_mask
        end
        # also subtract the constant (∅) coefficient
        @views poly_coefs[:, j] .-= poly_coefs[:, 1]
    end
    return poly_coefs
end

"""
    column_supports_second_order(nᵉ, nˢ) -> Vector{UInt64}

For each column of the augmented shock vector `ê` used by the
second-order pruned variance decomposition, return a `UInt64` bitmask of
the shock indices that column depends on. Block layout matches
`build_coalition_mask_second_order`:

  block 1: nᵉ        cols, support = {j}
  block 2: nᵉ²       cols, support = {j, k}
  block 3: nˢ · nᵉ   cols, support = {j}

A column `c` is included in coalition `S` iff its support mask is a
subset of `S`'s mask, exactly mirroring the masking the existing
exhaustive code performs.
"""
function column_supports_second_order(nᵉ::Int, nˢ::Int)
    @assert nᵉ ≤ 64 "column_supports_second_order requires nᵉ ≤ 64"
    block_sizes = (nᵉ, nᵉ^2, nˢ * nᵉ)
    N = sum(block_sizes)
    sup = zeros(UInt64, N)
    bit(i) = UInt64(1) << (i - 1)

    off = 0
    for j in 1:nᵉ
        sup[off + j] = bit(j)
    end
    off += nᵉ

    for j in 1:nᵉ, k in 1:nᵉ
        sup[off + (j - 1) * nᵉ + k] = bit(j) | bit(k)
    end
    off += nᵉ^2

    for a in 1:nˢ, j in 1:nᵉ
        sup[off + (a - 1) * nᵉ + j] = bit(j)
    end

    return sup
end

"""
    column_supports_third_order(nᵉ, nˢ) -> Vector{UInt64}

Per-column shock-index support masks for the third-order augmented
shock vector `ê`. Block layout matches `build_coalition_mask_third_order`:

  block 1: nᵉ          cols, support = {j}
  block 2: nᵉ²         cols, support = {j, k}
  block 3: nˢ · nᵉ     cols, support = {j}
  block 4: nˢ · nᵉ     cols, support = {j}
  block 5: nˢ² · nᵉ    cols, support = {j}
  block 6: nˢ · nᵉ²    cols, support = {j, k}
  block 7: nᵉ³         cols, support = {j, k, l}
"""
function column_supports_third_order(nᵉ::Int, nˢ::Int)
    @assert nᵉ ≤ 64 "column_supports_third_order requires nᵉ ≤ 64"
    block_sizes = (nᵉ, nᵉ^2, nˢ * nᵉ, nˢ * nᵉ, nˢ^2 * nᵉ, nˢ * nᵉ^2, nᵉ^3)
    N = sum(block_sizes)
    sup = zeros(UInt64, N)
    bit(i) = UInt64(1) << (i - 1)

    off = 0
    for j in 1:nᵉ
        sup[off + j] = bit(j)
    end
    off += nᵉ

    for j in 1:nᵉ, k in 1:nᵉ
        sup[off + (j - 1) * nᵉ + k] = bit(j) | bit(k)
    end
    off += nᵉ^2

    for a in 1:nˢ, j in 1:nᵉ
        sup[off + (a - 1) * nᵉ + j] = bit(j)
    end
    off += nˢ * nᵉ

    for a in 1:nˢ, j in 1:nᵉ
        sup[off + (a - 1) * nᵉ + j] = bit(j)
    end
    off += nˢ * nᵉ

    for p in 1:nˢ^2, j in 1:nᵉ
        sup[off + (p - 1) * nᵉ + j] = bit(j)
    end
    off += nˢ^2 * nᵉ

    for a in 1:nˢ, j in 1:nᵉ, k in 1:nᵉ
        sup[off + (a - 1) * nᵉ^2 + (j - 1) * nᵉ + k] = bit(j) | bit(k)
    end
    off += nˢ * nᵉ^2

    for j in 1:nᵉ, k in 1:nᵉ, l in 1:nᵉ
        sup[off + (j - 1) * nᵉ^2 + (k - 1) * nᵉ + l] = bit(j) | bit(k) | bit(l)
    end

    return sup
end

"""
    monomial_column_index(supports, idx) -> Vector{Int}

Map each column of the augmented shock vector to the id of the monomial
its shock-support equals. The result lets per-monomial code build its
column mask in `O(N)` rather than `O(N · nᵉ)` lookups.

Columns whose support exceeds `idx.k` (which would lie outside `idx`) are
flagged with `0` and must be filtered out by the caller.
"""
function monomial_column_index(supports::Vector{UInt64}, idx::MonomialIndex)
    out = zeros(Int, length(supports))
    @inbounds for (c, m) in enumerate(supports)
        out[c] = get(idx.id, m, 0)
    end
    return out
end

