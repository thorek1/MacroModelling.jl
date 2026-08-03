@stable default_mode = "disable" begin

# Cubic Kalman filter for the pruned third-order solution — the third-order
# analogue of `./quadratic_kalman.jl`, built on the same idea and validated the
# same way.
#
# The idea. Pruning truncates the Kronecker hierarchy at a fixed rung, so a
# pruned solution of order n is *exactly linear* in an augmented state. At second
# order that state is [x₁; x₂; a⊗a] with a = x₁[past]. At third order the
# recursion is
#
#   aug₁ = [a; 1; ε],  aug₁ʰ = [a; 0; ε],  aug₂ = [b; 0; 0],  aug₃ = [p; 0; 0]
#   x₁ₜ  = 𝐒₁ aug₁
#   x₂ₜ  = 𝐒₁ aug₂ + ½ 𝐒₂ (aug₁ ⊗ aug₁)
#   x₃ₜ  = 𝐒₁ aug₃ +   𝐒₂ (aug₁ʰ ⊗ aug₂) + ⅙ 𝐒₃ (aug₁ ⊗ aug₁ ⊗ aug₁)
#
# (a, b, p are the past rows of x₁, x₂, x₃), so the state must additionally carry
#
#   q₁₁ = a⊗a,   q₁₂ = a⊗b,   q₁₁₁ = a⊗a⊗a
#
# and the system closes: writing aₙ = M a + v with v = mc + Vε state-independent,
# and bₙ = M b + W_q q₁₁ + W_l a + w_c,
#
#   q₁₁'  = (M⊗M) q₁₁ + u⊗v + v⊗u + v⊗v
#   q₁₂'  = (M⊗M) q₁₂ + (M⊗W_q) q₁₁₁ + (M⊗W_l) q₁₁ + u⊗w_c + v⊗bₙ
#   q₁₁₁' = (M⊗M⊗M) q₁₁₁ + [3 permutations of ((M⊗M)q₁₁)⊗v]
#                         + [3 permutations of u⊗v⊗v] + v⊗v⊗v
#
# with u = M a. No fourth-order block appears, because aₙ carries no q₁₁ term.
# The closure is what makes this work; recomputing the new blocks as kron(aₙ,aₙ)
# would be quadratic in z and silently break the linearity the filter rests on.
#
# What is exact and what is not. The transition is exactly linear, and both
# conditional moments are closed forms — f is a degree-3 polynomial in ε with
# z-affine coefficients, so recovering those coefficients once gives E[f] = C(z)m
# and Var(f) = C(z)ΨC(z)' exactly, with no quadrature. What is approximate is the
# same thing as at second order: the innovation is not Gaussian, and the filter
# matches only its first two moments. See `docs/src/filters.md`.
#
# Derivatives. Reverse mode has a hand-written adjoint (step, build and recursion,
# each verified against ForwardDiff); forward mode works too via the promoted
# element type. Note that `eltype(sys.S1)` rather than a stored type field is what
# the hot paths branch on — a `DataType`-typed field infers as `DataType`, not
# `Type{Float64}`, which costs specialisation and measured 6.5x on the primal.
#
# Cost. q₁₁ and q₁₁₁ are symmetric and carried compressed, giving an augmented
# dimension of 3n_r + n_past(n_past+1)/2 + n_past² + n_past(n_past+1)(n_past+2)/6.
# The O(n_z³) covariance recursion still scales as n_past⁹, so the filter stays
# confined to small models — see `CUBIC_KALMAN_MAX_DIMENSION` below.

# The covariance recursion is two n_z×n_z triple products per period. Beyond this
# dimension a single period costs seconds and a single matrix hundreds of MB, so
# refuse with a message that names the cause instead of appearing to hang.
const CUBIC_KALMAN_MAX_DIMENSION = 2500

cubic_pair_index(i::Int, j::Int, n::Int) = begin
    i, j = max(i, j), min(i, j)
    (i - 1) * i ÷ 2 + j
end

cubic_triple_index(i::Int, j::Int, k::Int, n::Int) = begin
    i, j, k = sort((i, j, k), rev = true)
    (i - 1) * i * (i + 1) ÷ 6 + (j - 1) * j ÷ 2 + k
end

# Several intermediates below are computed as a matrix whose *row-major* flatten
# is the Kronecker vector wanted — "the rowvec of R" in the comments, meaning the
# vector v with v[(i-1)*size(R,2)+r] == R[i,r]. The step writes those entries out
# by indexing rather than materialising the flatten.

# Index maps for the symmetric Kronecker blocks. `a⊗a` is symmetric and `a⊗a⊗a`
# fully symmetric, so the state carries one entry per sorted multi-index — the
# same vech idea the quadratic filter uses, but applied by indexing rather than
# by multiplying with duplication and elimination matrices, which would cost more
# than the compression saves. `expand` maps a full Kronecker position onto its
# compressed slot; `canonical` maps a slot back onto one representative position,
# which is exact precisely because the compressed blocks are symmetric.
function symmetric_pair_maps(n::Int)
    slot = Dict{NTuple{2,Int},Int}()
    m = 0
    for i in 1:n, j in i:n
        m += 1
        slot[(i, j)] = m
    end
    expand = Vector{Int}(undef, n * n)
    @inbounds for i in 1:n, j in 1:n
        expand[(i-1)*n+j] = slot[minmax(i, j)]
    end
    canonical = Vector{Int}(undef, m)
    @inbounds for i in 1:n, j in i:n
        canonical[slot[(i, j)]] = (i-1)*n + j
    end
    return expand, canonical
end

function symmetric_triple_maps(n::Int)
    slot = Dict{NTuple{3,Int},Int}()
    m = 0
    for i in 1:n, j in i:n, k in j:n
        m += 1
        slot[(i, j, k)] = m
    end
    expand = Vector{Int}(undef, n^3)
    @inbounds for i in 1:n, j in 1:n, k in 1:n
        s = sort!([i, j, k])
        expand[((i-1)*n + (j-1))*n + k] = slot[(s[1], s[2], s[3])]
    end
    canonical = Vector{Int}(undef, m)
    @inbounds for i in 1:n, j in i:n, k in j:n
        canonical[slot[(i, j, k)]] = ((i-1)*n + (j-1))*n + k
    end
    return expand, canonical
end

function compressed_pair_power_matrix(matrix)
    n = size(matrix, 1)
    pairs = [(i, j) for i in 1:n for j in 1:i]
    out = zeros(eltype(matrix), length(pairs), length(pairs))
    @inbounds for (row, (i, j)) in enumerate(pairs), (column, (p, q)) in enumerate(pairs)
        if p == q
            out[row, column] = matrix[i, p] * matrix[j, p]
        else
            out[row, column] = matrix[i, p] * matrix[j, q] + matrix[i, q] * matrix[j, p]
        end
    end
    return out
end

function compressed_triple_power_matrix(matrix)
    n = size(matrix, 1)
    triples = [(i, j, k) for i in 1:n for j in 1:i for k in 1:j]
    out = zeros(eltype(matrix), length(triples), length(triples))
    @inbounds for (row, (i, j, k)) in enumerate(triples), (column, (p, q, r)) in enumerate(triples)
        if p == q == r
            out[row, column] = matrix[i, p] * matrix[j, p] * matrix[k, p]
        elseif p == q
            out[row, column] = matrix[i, p] * matrix[j, p] * matrix[k, r] +
                               matrix[i, p] * matrix[j, r] * matrix[k, p] +
                               matrix[i, r] * matrix[j, p] * matrix[k, p]
        elseif q == r
            out[row, column] = matrix[i, p] * matrix[j, q] * matrix[k, q] +
                               matrix[i, q] * matrix[j, p] * matrix[k, q] +
                               matrix[i, q] * matrix[j, q] * matrix[k, p]
        else
            out[row, column] = matrix[i, p] * matrix[j, q] * matrix[k, r] +
                               matrix[i, p] * matrix[j, r] * matrix[k, q] +
                               matrix[i, q] * matrix[j, p] * matrix[k, r] +
                               matrix[i, q] * matrix[j, r] * matrix[k, p] +
                               matrix[i, r] * matrix[j, p] * matrix[k, q] +
                               matrix[i, r] * matrix[j, q] * matrix[k, p]
        end
    end
    return out
end

function compressed_pair_power_matrix_pullback!(matrix_bar, matrix, cotangent)
    n = size(matrix, 1)
    pairs = [(i, j) for i in 1:n for j in 1:i]
    @inbounds for (row, (i, j)) in enumerate(pairs), (column, (p, q)) in enumerate(pairs)
        value = cotangent[row, column]
        if p == q
            matrix_bar[i, p] += value * matrix[j, p]
            matrix_bar[j, p] += value * matrix[i, p]
        else
            matrix_bar[i, p] += value * matrix[j, q]
            matrix_bar[j, q] += value * matrix[i, p]
            matrix_bar[i, q] += value * matrix[j, p]
            matrix_bar[j, p] += value * matrix[i, q]
        end
    end
    return matrix_bar
end

@inline function accumulate_triple_term!(matrix_bar, matrix, value, i, j, k, p, q, r)
    matrix_bar[i, p] += value * matrix[j, q] * matrix[k, r]
    matrix_bar[j, q] += value * matrix[i, p] * matrix[k, r]
    matrix_bar[k, r] += value * matrix[i, p] * matrix[j, q]
    return matrix_bar
end

function compressed_triple_power_matrix_pullback!(matrix_bar, matrix, cotangent)
    n = size(matrix, 1)
    triples = [(i, j, k) for i in 1:n for j in 1:i for k in 1:j]
    @inbounds for (row, (i, j, k)) in enumerate(triples), (column, (p, q, r)) in enumerate(triples)
        value = cotangent[row, column]
        if p == q == r
            accumulate_triple_term!(matrix_bar, matrix, value, i, j, k, p, p, p)
        elseif p == q
            accumulate_triple_term!(matrix_bar, matrix, value, i, j, k, p, p, r)
            accumulate_triple_term!(matrix_bar, matrix, value, i, j, k, p, r, p)
            accumulate_triple_term!(matrix_bar, matrix, value, i, j, k, r, p, p)
        elseif q == r
            accumulate_triple_term!(matrix_bar, matrix, value, i, j, k, p, q, q)
            accumulate_triple_term!(matrix_bar, matrix, value, i, j, k, q, p, q)
            accumulate_triple_term!(matrix_bar, matrix, value, i, j, k, q, q, p)
        else
            accumulate_triple_term!(matrix_bar, matrix, value, i, j, k, p, q, r)
            accumulate_triple_term!(matrix_bar, matrix, value, i, j, k, p, r, q)
            accumulate_triple_term!(matrix_bar, matrix, value, i, j, k, q, p, r)
            accumulate_triple_term!(matrix_bar, matrix, value, i, j, k, q, r, p)
            accumulate_triple_term!(matrix_bar, matrix, value, i, j, k, r, p, q)
            accumulate_triple_term!(matrix_bar, matrix, value, i, j, k, r, q, p)
        end
    end
    return matrix_bar
end

# ── analytic assembly ────────────────────────────────────────────────────────
#
# `f(z, ·)` is a polynomial of degree ≤ 3 in ε whose coefficients are affine in z:
#
#   f(z, ε) = Σ_α c_α(z) ε^α ,   |α| ≤ 3.
#
# Recovering the coefficient matrix C(z) = [c_α(z)]_α once therefore gives both
# moments in closed form, with no quadrature anywhere:
#
#   E[f]   = C(z) m,        m_α  = E[ε^α]
#   Var(f) = C(z) Ψ C(z)',  Ψ_αβ = E[ε^{α+β}] − E[ε^α] E[ε^β]
#
# and because ε is a vector of *independent* standard normals, E[ε^α] factorises
# into double factorials — no Isserlis pairings needed. This replaces a tensor
# Gauss-Hermite rule whose node count grew as `npt^nExo`: the coefficient basis
# has only C(nExo+3, 3) elements (10 for two shocks, 120 for seven).

# E[ε^α] = ∏ᵢ (αᵢ−1)!! when every αᵢ is even, and 0 otherwise.
double_factorial(n::Int) = n <= 0 ? 1.0 : Float64(prod(n:-2:1))
gaussian_moment(α) = all(iseven, α) ? prod(double_factorial(a - 1) for a in α) : 0.0

# Exponent vectors α with |α| ≤ maxdeg, in n variables.
function monomial_exponents(n::Int, maxdeg::Int = 3)
    out = Vector{Vector{Int}}()
    cur = zeros(Int, n)
    function rec(pos, rem)
        if pos > n
            push!(out, copy(cur))
            return
        end
        for d in 0:rem
            cur[pos] = d
            rec(pos + 1, rem - d)
        end
        cur[pos] = 0
        return
    end
    rec(1, maxdeg)
    return out
end

"""
Interpolation data for recovering a degree-≤3 polynomial in `nExo` variables from
its values: the exponent set, a unisolvent set of evaluation points, the inverse
Vandermonde, and the Gaussian moment vector and covariance of the monomials.
"""
function cubic_noise_basis(nExo::Int; seed::Int = 42)
    exps = monomial_exponents(nExo, 3)
    N = length(exps)
    # Any N points with an invertible Vandermonde will do; Gaussian draws are
    # unisolvent with probability one, and the conditioning is checked rather
    # than assumed.
    rng = Random.Xoshiro(seed)
    pts = [randn(rng, nExo) for _ in 1:N]
    V = [prod(pts[p][k]^exps[m][k] for k in 1:nExo) for p in 1:N, m in 1:N]
    if !isfinite(ℒ.cond(V)) || ℒ.cond(V) > 1e10
        error("The cubic Kalman filter could not build a well-conditioned polynomial " *
              "basis for $nExo shocks (condition number $(ℒ.cond(V))). This is a bug; " *
              "please report it.")
    end
    # f(z, ε_p) = C(z) V[p,:]'  ⇒  F = C V'  ⇒  C = F (V')⁻¹
    W = Matrix(transpose(inv(V)))
    m = [gaussian_moment(a) for a in exps]
    Ψ = [gaussian_moment(exps[i] .+ exps[j]) - m[i] * m[j] for i in 1:N, j in 1:N]
    Ψ = (Ψ + Ψ') / 2
    return (; exps, pts, W, m, Ψ, N)
end

# Probabilists' Gauss-Hermite nodes and weights via Golub-Welsch. Retained so the
# tests can cross-check the analytic assembly against quadrature.
function gauss_hermite_nodes(n::Int)
    J = ℒ.SymTridiagonal(zeros(n), sqrt.(1:n-1))
    E = ℒ.eigen(J)
    return E.values, (E.vectors[1, :]) .^ 2
end

# Tensor product rule over `nExo` independent standard normals. `npt` points per
# dimension integrate polynomials of degree 2·npt−1 exactly; the integrands here
# reach degree six, so npt = 4 suffices and is the default.
function gauss_hermite_tensor(nExo::Int, npt::Int)
    x, w = gauss_hermite_nodes(npt)
    nodes = Vector{Vector{Float64}}()
    wts = Float64[]
    for I in Iterators.product(ntuple(_ -> 1:npt, nExo)...)
        push!(nodes, [x[I[k]] for k in 1:nExo])
        push!(wts, prod(w[I[k]] for k in 1:nExo))
    end
    return nodes, wts
end

"""
Everything the step needs that is derived from the solution matrices, in one
place so the forward pass and its adjoint cannot drift apart:

  aₙ = M a + mc + V ε,   bₙ = M b + B2·K₂

with K₂ split into its (a,a), (a,tail)+(tail,a) and (tail,tail) parts — the
coefficients `Wq`, `Wl_t` and `Bc` — so each can be routed onto the right state
block.
"""
function cubic_derived_matrices(S1, S2, Pm, nPast::Int, nExo::Int, na::Int)
    Tv = promote_type(eltype(S1), eltype(S2))
    A1 = Pm * S1
    M = A1[:, 1:nPast]
    mc = A1[:, nPast+1]
    V = A1[:, nPast+2:na]

    B2 = Pm * S2 / 2
    ntail = 1 + nExo
    n_pair = na * (na + 1) ÷ 2
    pair_index(i, j) = max(i, j) * (max(i, j) - 1) ÷ 2 + min(i, j)
    Wq = zeros(Tv, nPast, nPast * (nPast + 1) ÷ 2)
    for i in 1:nPast, j in 1:i
        Wq[:, pair_index(i, j)] = (i == j ? 1 : 2) .* B2[:, pair_index(i, j)]
    end
    Wl_t = [zeros(Tv, nPast, nPast) for _ in 1:ntail]
    for t in 1:ntail, k in 1:nPast
        Wl_t[t][:, k] = 2 .* B2[:, pair_index(k, nPast + t)]
    end
    n_tail_pair = ntail * (ntail + 1) ÷ 2
    Bc = zeros(Tv, nPast, n_tail_pair)
    tail_pair = 0
    for i in 1:ntail, j in 1:i
        tail_pair += 1
        Bc[:, tail_pair] .= B2[:, pair_index(nPast + i, nPast + j)]
    end
    M2 = compressed_pair_power_matrix(M)
    M3 = compressed_triple_power_matrix(M)
    return M, mc, V, B2, Wq, Wl_t, Bc, M2, M3
end

"""
Fold every intermediate cotangent back onto `S1`, `S2` and `S3`: the live-column
accumulators the step adjoint writes, and the derived blocks of
`cubic_derived_matrices`. The compressed pair/triple power maps are resolved onto
`M` first, so their cotangents must be accumulated before this is called.
"""
function cubic_derived_pullback!(∂, sys)
    (; M, Pm, nPast, nExo, na) = sys
    ∂S1, ∂S2 = ∂.S1, ∂.S2
    ∂M, ∂mc, ∂V = ∂.M, ∂.mc, ∂.V
    ∂Wq, ∂Wl_t, ∂Bc, ∂M2, ∂M3 = ∂.Wq, ∂.Wl_t, ∂.Bc, ∂.M2, ∂.M3

    # live-column cotangents accumulated by the step adjoint
    @inbounds for (r, j) in enumerate(sys.k2cols)
        @views ∂S2[:, j] .+= ∂.S2k2[:, r]
    end
    @inbounds for (r, j) in enumerate(sys.k12cols)
        @views ∂S2[:, j] .+= ∂.S2k12[:, r]
    end
    @inbounds for (r, j) in enumerate(sys.k3cols)
        @views ∂.S3[:, j] .+= ∂.S3k3[:, r]
    end

    compressed_pair_power_matrix_pullback!(∂M, M, ∂M2)
    compressed_triple_power_matrix_pullback!(∂M, M, ∂M3)

    # A1 = Pm S1 ; M, mc, V are its column blocks
    ∂A1 = hcat(∂M, reshape(∂mc, nPast, 1), ∂V)
    ∂S1 .+= Pm' * ∂A1

    # the K₂ splits, all linear scatters out of compressed B2
    n_pair = na * (na + 1) ÷ 2
    ∂B2 = zeros(eltype(∂S2), nPast, n_pair)
    pair_index(i, j) = max(i, j) * (max(i, j) - 1) ÷ 2 + min(i, j)
    for i in 1:nPast, j in 1:i
        factor = i == j ? 1 : 2
        @views ∂B2[:, pair_index(i, j)] .+= factor .* ∂Wq[:, pair_index(i, j)]
    end
    for t in 1:(1+nExo), k in 1:nPast
        @views ∂B2[:, pair_index(k, nPast+t)] .+= 2 .* ∂Wl_t[t][:, k]
    end
    tail_pair = 0
    for i in 1:(1+nExo), j in 1:i
        tail_pair += 1
        @views ∂B2[:, pair_index(nPast+i, nPast+j)] .+= ∂Bc[:, tail_pair]
    end
    ∂S2 .+= Pm' * ∂B2 ./ 2
    return ∂S1, ∂S2
end

"""
Constant structure of the cubic augmented system:
`z = [x₁; x₂; x₃; q₁₁; q₁₂; q₁₁₁]` over the retained rows (past states plus
observables), together with the coefficient blocks that keep the step affine.
"""
function build_cubic_kalman_system_from_constants(cons, 𝐒₁, 𝐒₂, 𝐒₃, observables_index::Vector{Int})
    T = cons.post_model_macro
    nPast, nExo = T.nPast_not_future_and_mixed, T.nExo
    past = T.past_not_future_and_mixed_idx

    # As in the quadratic filter, carry only the rows the recursion actually
    # reads: past states for the transition, observables for the measurement.
    oas = sort(union(past, observables_index))
    nr = length(oas)
    pos = Dict(v => i for (i, v) in enumerate(oas))

    na = nPast + 1 + nExo
    # q₁₁ and q₁₁₁ are carried compressed; q₁₂ = a⊗b has no symmetry to exploit.
    exp2, can2 = symmetric_pair_maps(nPast)
    exp3, can3 = symmetric_triple_maps(nPast)
    pair_indices = [(i, j) for i in 1:na for j in 1:i]
    triple_indices = [(i, j, k) for i in 1:na for j in 1:i for k in 1:j]
    nq11, nq12, nq111 = length(can2), nPast^2, length(can3)
    nz = 3nr + nq11 + nq12 + nq111

    if nz > CUBIC_KALMAN_MAX_DIMENSION
        error("The cubic Kalman filter needs an augmented state of dimension " *
              "$nz (= 3·$nr + $nq11 + $nq12 + $nq111) for this model, and its " *
              "covariance recursion is O(n_z³) per period. The limit is " *
              "$CUBIC_KALMAN_MAX_DIMENSION (`CUBIC_KALMAN_MAX_DIMENSION`). " *
              "Use `filter = :inversion` or a particle filter instead.")
    end

    # Carry the solution matrices' element type so ForwardDiff duals flow through
    # the whole assembly; the 0/1 selection matrices stay Float64 and promote on
    # contact.
    Tv = promote_type(eltype(𝐒₁), eltype(𝐒₂), eltype(𝐒₃))
    S1 = Matrix{Tv}(Matrix(𝐒₁)[oas, :])
    S2 = Matrix{Tv}(Matrix(𝐒₂)[oas, :])
    S3 = Matrix{Tv}(Matrix(𝐒₃)[oas, :])

    Pm = zeros(nPast, nr)
    for (i, j) in enumerate(past)
        Pm[i, pos[j]] = 1.0
    end

    r1, r2, r3 = 1:nr, nr+1:2nr, 2nr+1:3nr
    i11 = 3nr+1:3nr+nq11
    i12 = 3nr+nq11+1:3nr+nq11+nq12
    i111 = 3nr+nq11+nq12+1:nz

    # yₜ = (x₁ + x₂ + x₃)[observables]
    C = zeros(length(observables_index), nz)
    for (i, j) in enumerate(observables_index)
        C[i, pos[j]] = 1.0
        C[i, nr+pos[j]] = 1.0
        C[i, 2nr+pos[j]] = 1.0
    end

    ntail = 1 + nExo
    M, mc, V, B2, Wq, Wl_t, Bc, M2, M3 = cubic_derived_matrices(S1, S2, Pm, nPast, nExo, na)

    # The Kronecker inputs are contracted against 𝐒₂ and 𝐒₃, whose columns are
    # largely structurally zero — a third-order solution has no cross-derivative
    # for most index triples. Keeping only the live columns shrinks both the
    # vector that has to be built and the (memory-bound, very wide) product that
    # consumes it; on a four-shock model that is 536 of 1331 columns for 𝐒₃, and
    # `S3 * K3` alone was half the cost of a step.
    # Liveness is taken from the *structural* pattern of the sparse solution
    # matrices, not from numerical zeros of a densified copy: a column that is
    # merely zero at this parameter draw may be nonzero at the next one, and
    # dropping it would silently zero a real derivative. Reading the stored
    # pattern is also what the rest of the package assumes about 𝐒.
    S2sp = 𝐒₂[oas, :]
    S3sp = 𝐒₃[oas, :]
    live(A, cols) = A isa SparseArrays.AbstractSparseMatrix ?
        [j for j in cols if A.colptr[j+1] > A.colptr[j]] :
        [j for j in cols if any(!iszero, view(A, :, j))]

    k2cols = live(S2sp, 1:length(pair_indices))
    k2_ij = [pair_indices[j] for j in k2cols]
    S2k2 = S2[:, k2cols]

    k12cols = live(S2sp, 1:length(pair_indices))
    k12_ij = [pair_indices[j] for j in k12cols]
    S2k12 = S2[:, k12cols]

    k3cols = live(S3sp, 1:length(triple_indices))
    k3_ijk = [triple_indices[j] for j in k3cols]
    S3k3 = S3[:, k3cols]

    # Observation rows are a selection of the x₁, x₂ and x₃ blocks; carrying the
    # three positions lets the recursion index instead of running a gemm with a
    # 0/1 matrix, as the quadratic filter does with its two.
    op1 = [pos[j] for j in observables_index]
    op2 = op1 .+ nr
    op3 = op1 .+ 2nr

    # Decoded multi-indices of the canonical entries, so the step can write the
    # compressed blocks directly instead of materialising the full nPast³ vector.
    can2_ij = [(fld(r - 1, nPast) + 1, mod(r - 1, nPast) + 1) for r in can2]
    can3_ijk = [(fld(r - 1, nPast * nPast) + 1,
                 mod(fld(r - 1, nPast), nPast) + 1,
                 mod(r - 1, nPast) + 1) for r in can3]

    # Only these augmented-state blocks can enter the affine noise loading:
    # past x₁ rows, past x₂ rows, and q₁₁.  The larger covariance is still needed
    # by the Kalman recursion, but the additional Kollmann correction only reads
    # this structural support.
    past_positions = [findfirst(!iszero, view(Pm, i, :)) for i in 1:nPast]
    noise_state_indices = vcat(first(r1) .+ past_positions .- 1,
                               first(r2) .+ past_positions .- 1,
                               collect(i11))

    return (; nr, nPast, nExo, na, nz, oas, S1, S2, S3, Pm, C, op1, op2, op3,
            r1, r2, r3, i11, i12, i111, nq11, nq12, nq111,
            exp2, can2, exp3, can3, can2_ij, can3_ijk, pair_indices, triple_indices,
            M, mc, V, B2, Wq, Wl_t, Bc, M2, M3, ntail, noise_state_indices,
            k2cols, k2_ij, S2k2, k12cols, k12_ij, S2k12, k3cols, k3_ijk, S3k3)
end

"""
Preallocated buffers for `cubic_kalman_step!`. The step is called
`(n_z + 1) · n_nodes` times to build the transition and `n_nodes` times per
period, and it is entirely allocation-bound — it does a few hundred flops but
allocated ~13 kB per call before these buffers existed.
"""
function cubic_kalman_workspace(sys, Tv = eltype(sys.S1))
    (; nr, nPast, na, ntail) = sys
    nP2 = nPast * nPast
    nq11, nq111 = sys.nq11, sys.nq111
    ntail_pair = ntail * (ntail + 1) ÷ 2
    zeros(n...) = Base.zeros(Tv, n...)
    return (; a = zeros(nPast), b = zeros(nPast), p = zeros(nPast),
            q11 = zeros(nq11), q12 = zeros(nP2), q111 = zeros(nq111),
            tail = zeros(ntail), tt = zeros(ntail_pair),
            aug1 = zeros(na), aug1h = zeros(na), aug2 = zeros(na), aug3 = zeros(na),
            K2 = zeros(length(sys.k2cols)), K12 = zeros(length(sys.k12cols)),
            K3 = zeros(length(sys.k3cols)),
            x1n = zeros(nr), x2n = zeros(nr), x3n = zeros(nr),
            u = zeros(nPast), v = zeros(nPast), bn = zeros(nPast), wc = zeros(nPast),
            Q11 = zeros(nPast, nPast), Q12 = zeros(nPast, nPast), Q111 = zeros(nPast, nq11),
            R2 = zeros(nPast, nPast), Tmp = zeros(nPast, nPast),
            MQ111 = zeros(nPast, nq11), R3 = zeros(nq111),
            Wl = zeros(nPast, nPast), t2 = zeros(nq11), vv = zeros(nq11),
            q2_aug = zeros(length(sys.pair_indices)),
            q2_cross = zeros(length(sys.pair_indices)),
            q3_aug = zeros(length(sys.triple_indices)), q3_uuv = zeros(nq111), q3_uvv = zeros(nq111),
            scratch_out = zeros(sys.nz),
            pull_u = zeros(nPast), pull_v = zeros(nPast), pull_t2 = zeros(nq11),
            pull_wc = zeros(nPast), pull_bn = zeros(nPast), pull_R2 = zeros(nPast, nPast),
            pull_R3 = zeros(nq111), pull_R2t = zeros(nPast, nPast),
            pull_MQ11 = zeros(nPast, nPast), pull_MQ12 = zeros(nPast, nPast),
            pull_MQ111 = zeros(nPast, nq11), pull_M = zeros(nPast, nPast),
            pull_Wq = zeros(nPast, nq11), pull_Wl = zeros(nPast, nPast),
            pull_grad_MQ11 = zeros(nPast, nPast), pull_grad_MQ12 = zeros(nPast, nPast),
            pull_grad_MQ111 = zeros(nPast, nq11),
            pull_x2n = zeros(nr))
end

"""
One step of the augmented map, `z ↦ f(z, ε)`. Affine in `z` by construction:
every product of two `z`-dependent quantities is read off an existing block
rather than recomputed.
"""
function cubic_kalman_step!(out::AbstractVector, sys, z::AbstractVector, ε::AbstractVector, ws)
    (; nr, nPast, nExo, na, S1, S2, S3, Pm, r1, r2, r3, i11, i12, i111,
       exp2, exp3, can2_ij, can3_ijk, pair_indices, triple_indices,
       M, mc, V, Wq, Wl_t, Bc, M2, M3, ntail) = sys
    (; a, b, p, q11, q12, q111, tail, tt, aug1, aug1h, aug2, aug3, K2, K12, K3,
       x1n, x2n, x3n, u, v, bn, wc, Q11, Q12, Q111, R2, Tmp, MQ111, R3, Wl, t2, vv,
       q2_aug, q2_cross, q3_aug) = ws
    nP = nPast
    nP2 = nP * nP

    ℒ.mul!(a, Pm, view(z, r1))
    ℒ.mul!(b, Pm, view(z, r2))
    ℒ.mul!(p, Pm, view(z, r3))

    # The symmetric blocks are stored as unique raw products.
    o11 = first(i11) - 1
    o111 = first(i111) - 1
    r = 0
    @inbounds for i in 1:nP, j in 1:i
        r += 1
        q11[r] = z[o11 + exp2[(i-1)*nP+j]]
    end
    r = 0
    @inbounds for i in 1:nP, j in 1:i, k in 1:j
        r += 1
        q111[r] = z[o111 + exp3[((i-1)*nP + (j-1))*nP+k]]
    end
    @inbounds for (r, k) in enumerate(i12)
        q12[r] = z[k]
    end

    tail[1] = one(eltype(tail))
    @inbounds for i in 1:nExo
        tail[1+i] = ε[i]
    end
    @inbounds for i in 1:nP
        aug1[i] = a[i]; aug1h[i] = a[i]; aug2[i] = b[i]; aug3[i] = p[i]
    end
    aug1[nP+1] = 1.0; aug1h[nP+1] = 0.0; aug2[nP+1] = 0.0; aug3[nP+1] = 0.0
    @inbounds for i in 1:nExo
        aug1[nP+1+i] = ε[i]; aug1h[nP+1+i] = ε[i]; aug2[nP+1+i] = 0.0; aug3[nP+1+i] = 0.0
    end

    # Contract the solution tensors in the compressed augmented basis.  The
    # state-state products must be read from q₁₁/q₁₁₁; recomputing them from
    # `a` would make this map quadratic/cubic in z and destroy the linear
    # state-space representation.
    @inbounds for (r, (i, j)) in enumerate(pair_indices)
        if i <= nP
            q2_aug[r] = i == j ? q11[cubic_pair_index(i, j, nP)] :
                2 * q11[cubic_pair_index(i, j, nP)]
            q2_cross[r] = i == j ? q12[(i-1)*nP+j] :
                q12[(i-1)*nP+j] + q12[(j-1)*nP+i]
        elseif j <= nP
            q2_aug[r] = 2 * aug1[i] * a[j]
            q2_cross[r] = aug1h[i] * b[j]
        else
            q2_aug[r] = i == j ? aug1[i] * aug1[j] : 2 * aug1[i] * aug1[j]
            q2_cross[r] = zero(eltype(q2_cross))
        end
    end
    @inbounds for (r, (i, j, k)) in enumerate(triple_indices)
        if i <= nP
            factor = i == j == k ? 1 : i == j ? 3 : j == k ? 3 : 6
            q3_aug[r] = factor * q111[cubic_triple_index(i, j, k, nP)]
        elseif j <= nP
            factor = j == k ? 3 : 6
            q3_aug[r] = factor * aug1[i] * q11[cubic_pair_index(j, k, nP)]
        elseif k <= nP
            factor = i == j ? 3 : 6
            q3_aug[r] = factor * aug1[i] * aug1[j] * a[k]
        else
            q3_aug[r] = (i == j == k ? 1 : i == j ? 3 : j == k ? 3 : 6) *
                aug1[i] * aug1[j] * aug1[k]
        end
    end
    @inbounds for r in eachindex(K2)
        K2[r] = q2_aug[sys.k2cols[r]]
        K12[r] = q2_cross[sys.k12cols[r]]
    end
    @inbounds for r in eachindex(K3)
        K3[r] = q3_aug[sys.k3cols[r]]
    end

    ℒ.mul!(x1n, S1, aug1)
    ℒ.mul!(x2n, S1, aug2); ℒ.mul!(x2n, sys.S2k2, K2, 0.5, 1.0)
    ℒ.mul!(x3n, S1, aug3); ℒ.mul!(x3n, sys.S2k12, K12, 1.0, 1.0)
    ℒ.mul!(x3n, sys.S3k3, K3, 1/6, 1.0)

    # New Kronecker blocks, kept affine in z.
    ℒ.mul!(u, M, a)                    # z-dependent, linear in a
    copyto!(v, mc); ℒ.mul!(v, V, ε, 1.0, 1.0)   # z-independent
    ℒ.mul!(bn, Pm, x2n)                # affine in z

    @inbounds for j in 1:nP, s in 1:nP
        Q11[j, s] = q11[cubic_pair_index(j, s, nP)]
        Q12[j, s] = q12[(j-1)*nP+s]
    end
    @inbounds for i in 1:nP, r in 1:i
        s = cubic_pair_index(i, r, nP)
        for j in 1:nP
            Q111[j, s] = q111[cubic_triple_index(j, i, r, nP)]
        end
    end

    ℒ.mul!(Tmp, M, Q11); ℒ.mul!(R2, Tmp, M')     # M Q₁₁ M' — its rowvec is u⊗u
    @inbounds for (s, (i, r)) in enumerate(can2_ij)
        t2[cubic_pair_index(i, r, nP)] = R2[i, r]
    end
    @inbounds for (s, (i, j)) in enumerate(can2_ij)
        vv[s] = v[i] * v[j]
    end

    fill!(Wl, 0.0)
    @inbounds for t in 1:ntail
        ℒ.axpy!(tail[t], Wl_t[t], Wl)
    end
    tail_pair = 0
    @inbounds for i in 1:ntail, j in 1:i
        tail_pair += 1
        tt[tail_pair] = i == j ? tail[i] * tail[j] : 2 * tail[i] * tail[j]
    end
    ℒ.mul!(wc, Bc, tt)

    # R2 ← M Q₁₂ M' + (M Q₁₁₁) Wq' + (M Q₁₁) Wl'; its rowvec is the z-dependent
    # part of q₁₂', and Tmp still holds M Q₁₁ from above.
    ℒ.mul!(MQ111, M, Q111)
    ℒ.mul!(R2, Tmp, Wl')
    ℒ.mul!(R2, MQ111, Wq', 1.0, 1.0)
    ℒ.mul!(Tmp, M, Q12)
    ℒ.mul!(R2, Tmp, M', 1.0, 1.0)

    ℒ.mul!(R3, M3, q111)               # compressed raw u⊗u⊗u

    @inbounds for i in 1:nr
        out[i] = x1n[i]; out[nr+i] = x2n[i]; out[2nr+i] = x3n[i]
    end
    # q₁₁' and q₁₁₁' are symmetric, so only the canonical entries are formed.
    @inbounds for (s, (i, j)) in enumerate(can2_ij)
        out[o11+s] = t2[cubic_pair_index(i, j, nP)] + u[i]*v[j] + v[i]*u[j] + v[i]*v[j]
    end
    @inbounds for i in 1:nP, j in 1:nP
        out[first(i12)-1 + (i-1)*nP+j] = R2[i, j] + u[i]*wc[j] + v[i]*bn[j]
    end
    @inbounds for (s, (i, j, k)) in enumerate(can3_ijk)
        out[o111+s] = R3[cubic_triple_index(i, j, k, nP)] +
                      t2[cubic_pair_index(i, j, nP)] * v[k] +
                      t2[cubic_pair_index(i, k, nP)] * v[j] +
                      t2[cubic_pair_index(j, k, nP)] * v[i] +
                      u[i] * v[j] * v[k] + v[i] * u[j] * v[k] + v[i] * v[j] * u[k] +
                      v[i] * v[j] * v[k]
    end
    return out
end

"""
Adjoint of `cubic_kalman_step!` with respect to the solution matrices, at a fixed
`(z, ε)`. Everything built from `z` and `ε` alone — `aug*`, `K₂`, `K₁₂`, `K₃`,
`Q₁₁`, `Q₁₂`, `Q₁₁₁`, `tail`, `tt` — is constant here, so only the paths through
`S1, S2, S3` and the derived blocks carry cotangents.

Accumulates into `∂` in place; call `cubic_derived_pullback!` afterwards to fold
the derived-block cotangents onto `S1` and `S2`.
"""
function cubic_rank_one_add!(matrix, left, right)
    @inbounds for j in axes(matrix, 2), i in axes(matrix, 1)
        matrix[i, j] += left[i] * right[j]
    end
    return matrix
end

function cubic_kalman_step_pullback!(∂, sys, z::AbstractVector, ε::AbstractVector,
                                     ∂out::AbstractVector, ws)
    (; nr, nPast, nExo, na, S1, S2, Pm, r1, r2, r3, i11, i12, i111,
       exp2, exp3, can2_ij, can3_ijk, pair_indices, triple_indices,
       M, mc, V, Wq, Wl_t, Bc, M2, M3, ntail) = sys
    nP = nPast
    nP2 = nP * nP

    # ── recompute the forward intermediates the adjoint needs ────────────────
    cubic_kalman_step!(ws.scratch_out, sys, z, ε, ws)
    (; a, b, p, q11, q12, q111, tail, tt, aug1, aug2, aug3, K2, K12, K3,
       x2n, u, v, bn, wc, Q11, Q12, Q111, Wl, t2, vv) = ws
    MQ11, MQ12, MQ111 = ws.pull_MQ11, ws.pull_MQ12, ws.pull_MQ111
    ℒ.mul!(MQ11, M, Q11)
    ℒ.mul!(MQ12, M, Q12)
    ℒ.mul!(MQ111, M, Q111)

    o11 = first(i11) - 1
    o12 = first(i12) - 1
    o111 = first(i111) - 1

    ∂u, ∂v, ∂t2 = ws.pull_u, ws.pull_v, ws.pull_t2
    ∂wc, ∂bn = ws.pull_wc, ws.pull_bn
    ∂R2, ∂R3 = ws.pull_R2, ws.pull_R3
    fill!(∂u, zero(eltype(∂u))); fill!(∂v, zero(eltype(∂v)))
    fill!(∂t2, zero(eltype(∂t2))); fill!(∂wc, zero(eltype(∂wc)))
    fill!(∂bn, zero(eltype(∂bn))); fill!(∂R2, zero(eltype(∂R2)))
    fill!(∂R3, zero(eltype(∂R3)))

    # ── seed from the output blocks ──────────────────────────────────────────
    @inbounds for (s, (i, j)) in enumerate(can2_ij)
        g = ∂out[o11+s]
        ∂t2[cubic_pair_index(i, j, nP)] += g
        ∂u[i] += g * v[j]; ∂v[j] += g * u[i]
        ∂v[i] += g * u[j]; ∂u[j] += g * v[i]
        ∂v[i] += g * v[j]; ∂v[j] += g * v[i]
    end
    @inbounds for i in 1:nP, j in 1:nP
        g = ∂out[o12+(i-1)*nP+j]
        ∂R2[i, j] += g
        ∂u[i] += g * wc[j]; ∂wc[j] += g * u[i]
        ∂v[i] += g * bn[j]; ∂bn[j] += g * v[i]
    end
    @inbounds for (s, (i, j, k)) in enumerate(can3_ijk)
        g = ∂out[o111+s]
        ∂R3[cubic_triple_index(i, j, k, nP)] += g
        ∂t2[cubic_pair_index(i, j, nP)] += g * v[k]
        ∂v[k] += g * t2[cubic_pair_index(i, j, nP)]
        ∂t2[cubic_pair_index(i, k, nP)] += g * v[j]
        ∂v[j] += g * t2[cubic_pair_index(i, k, nP)]
        ∂t2[cubic_pair_index(j, k, nP)] += g * v[i]
        ∂v[i] += g * t2[cubic_pair_index(j, k, nP)]
        ∂u[i] += g * v[j] * v[k]; ∂v[j] += g * u[i] * v[k]
        ∂v[k] += g * u[i] * v[j]
        ∂v[i] += g * u[j] * v[k]; ∂u[j] += g * v[i] * v[k]
        ∂v[k] += g * v[i] * u[j]
        ∂v[i] += g * v[j] * u[k]; ∂v[j] += g * v[i] * u[k]
        ∂u[k] += g * v[i] * v[j]
        ∂v[i] += g * v[j] * v[k]; ∂v[j] += g * v[i] * v[k]
        ∂v[k] += g * v[i] * v[j]
    end

    # R3 = M₃ q₁₁₁ in the compressed raw-product basis.
    ℒ.mul!(∂.M3, ∂R3, q111', one(eltype(∂.M3)), one(eltype(∂.M3)))

    # R2 = MQ11 Wl' + MQ111 Wq' + MQ12 M'
    ∂MQ11, ∂MQ111, ∂MQ12 = ws.pull_grad_MQ11, ws.pull_grad_MQ111, ws.pull_grad_MQ12
    ∂Wq, ∂Wl, ∂M = ws.pull_Wq, ws.pull_Wl, ws.pull_M
    ℒ.mul!(∂MQ11, ∂R2, Wl)
    ℒ.mul!(∂Wl, ∂R2', MQ11)
    ℒ.mul!(∂MQ111, ∂R2, Wq)
    ℒ.mul!(∂Wq, ∂R2', MQ111)
    ∂.Wq .+= ∂Wq
    ℒ.mul!(∂MQ12, ∂R2, M)
    ℒ.mul!(∂M, ∂R2', MQ12)

    # t2 contains the upper-triangular raw entries of MQ11M'.
    ∂R2t = ws.pull_R2t
    fill!(∂R2t, zero(eltype(∂R2t)))
    @inbounds for (i, j) in can2_ij
        ∂R2t[i, j] += ∂t2[cubic_pair_index(i, j, nP)]
    end
    ℒ.mul!(∂MQ11, ∂R2t, M, one(eltype(∂MQ11)), one(eltype(∂MQ11)))
    ℒ.mul!(∂M, ∂R2t', MQ11, one(eltype(∂M)), one(eltype(∂M)))

    # MQ11 = M Q11, MQ12 = M Q12, MQ111 = M Q111.
    ℒ.mul!(∂M, ∂MQ11, Q11', one(eltype(∂M)), one(eltype(∂M)))
    ℒ.mul!(∂M, ∂MQ12, Q12', one(eltype(∂M)), one(eltype(∂M)))
    ℒ.mul!(∂M, ∂MQ111, Q111', one(eltype(∂M)), one(eltype(∂M)))

    # wc = Bc tt ; Wl = Σ_t tail[t] Wl_t[t]
    ℒ.mul!(∂.Bc, ∂wc, tt', one(eltype(∂.Bc)), one(eltype(∂.Bc)))
    @inbounds for t in 1:ntail
        ∂.Wl_t[t] .+= tail[t] .* ∂Wl
    end

    # u = M a ; v = mc + V ε
    cubic_rank_one_add!(∂M, ∂u, a)
    ∂.mc .+= ∂v
    ℒ.mul!(∂.V, ∂v, ε', one(eltype(∂.V)), one(eltype(∂.V)))
    ∂.M .+= ∂M

    # bn = Pm x2n, and x2n is itself an output block
    ∂x1n = view(∂out, 1:nr)
    ∂x2n = ws.pull_x2n
    copyto!(∂x2n, view(∂out, nr+1:2nr))
    ℒ.mul!(∂x2n, Pm', ∂bn, one(eltype(∂x2n)), one(eltype(∂x2n)))
    ∂x3n = view(∂out, 2nr+1:3nr)

    # x1n = S1 aug1 ; x2n = S1 aug2 + ½ S2 K2 ; x3n = S1 aug3 + S2 K12 + ⅙ S3 K3.
    # The 𝐒₂/𝐒₃ cotangents accumulate on the live columns and are scattered back
    # once, in `cubic_derived_pullback!`.
    cubic_rank_one_add!(∂.S1, ∂x1n, aug1)
    cubic_rank_one_add!(∂.S1, ∂x2n, aug2)
    cubic_rank_one_add!(∂.S1, ∂x3n, aug3)
    ℒ.mul!(∂.S2k2, ∂x2n, K2', 0.5, one(eltype(∂.S2k2)))
    ℒ.mul!(∂.S2k12, ∂x3n, K12', one(eltype(∂.S2k12)), one(eltype(∂.S2k12)))
    ℒ.mul!(∂.S3k3, ∂x3n, K3', 1 / 6, one(eltype(∂.S3k3)))

    return ∂
end

# Zeroed cotangent accumulators matching the system's blocks.
function cubic_kalman_cotangents(sys)
    T = eltype(sys.S1)
    (; S1 = zeros(T, size(sys.S1)), S2 = zeros(T, size(sys.S2)), S3 = zeros(T, size(sys.S3)),
       M = zeros(T, size(sys.M)), mc = zeros(T, length(sys.mc)), V = zeros(T, size(sys.V)),
       Wq = zeros(T, size(sys.Wq)), Wl_t = [zeros(T, size(w)) for w in sys.Wl_t],
       Bc = zeros(T, size(sys.Bc)), M2 = zeros(T, size(sys.M2)), M3 = zeros(T, size(sys.M3)),
       S2k2 = zeros(T, size(sys.S2k2)), S2k12 = zeros(T, size(sys.S2k12)),
       S3k3 = zeros(T, size(sys.S3k3)))
end

# Allocating convenience wrapper, used by the tests.
function cubic_kalman_step(sys, z::AbstractVector, ε::AbstractVector, ws = cubic_kalman_workspace(sys))
    return cubic_kalman_step!(Vector{eltype(sys.S1)}(undef, sys.nz), sys, z, ε, ws)
end

# E[f(z,·)] and Var(f(z,·)) under ε ~ N(0,I), exactly.
#
# The node evaluations are stacked into one matrix and contracted with a single
# gemm rather than accumulated as `nnodes` rank-one updates: same arithmetic, but
# it runs at BLAS-3 rather than BLAS-2 speed. `buf` may be supplied to reuse the
# stacking buffer across periods.
function cubic_kalman_moments(sys, z, nodes, wts; buf = nothing, ws = cubic_kalman_workspace(sys))
    nz = sys.nz
    Fm = buf === nothing ? Matrix{eltype(sys.S1)}(undef, nz, length(nodes)) : buf
    @inbounds for (n, ε) in enumerate(nodes)
        cubic_kalman_step!(view(Fm, :, n), sys, z, ε, ws)
    end
    m = Fm * wts
    S = (Fm .* wts') * Fm'
    ℒ.mul!(S, m, m', -one(eltype(S)), one(eltype(S)))
    return m, (S + S') / 2
end

# The step is affine, so the transition matrix and drift are recovered exactly
# from evaluations at the origin and at each basis vector.
function build_cubic_kalman_transition(sys, nodes, wts; ws = cubic_kalman_workspace(sys))
    # Only the mean is needed here, so skip the variance the moment routine would
    # otherwise form — an nz×nz gemm per basis vector, nz+1 of them.
    Fm = Matrix{eltype(sys.S1)}(undef, sys.nz, length(nodes))
    mean_at = function (z)
        @inbounds for (n, ε) in enumerate(nodes)
            cubic_kalman_step!(view(Fm, :, n), sys, z, ε, ws)
        end
        return Fm * wts
    end
    c = mean_at(zeros(sys.nz))
    𝒜 = zeros(sys.nz, sys.nz)
    e = zeros(sys.nz)
    for j in 1:sys.nz
        fill!(e, 0.0)
        e[j] = 1.0
        𝒜[:, j] = mean_at(e) - c
    end
    return 𝒜, c
end

"""
Assemble the whole system analytically: the transition `𝒜, c` and the affine
noise factor `vec(C(z)) = c₀ + Λz` from which `Q(z) = C(z) Ψ C(z)'`.

Costs `(n_z + 1) · N` evaluations of the step, where `N = C(nExo+3, 3)` — the
same shape of work the quadrature build did, but with a node count that grows
polynomially in the number of shocks instead of exponentially. Afterwards no
quadrature is needed at all, per period or otherwise.
"""
function build_cubic_kalman_system(sys, basis; ws = cubic_kalman_workspace(sys))
    nz, N, Tv = sys.nz, basis.N, eltype(sys.S1)
    Fm = Matrix{Tv}(undef, nz, N)
    coefficients_at = function (z)
        @inbounds for (p, ε) in enumerate(basis.pts)
            cubic_kalman_step!(view(Fm, :, p), sys, z, ε, ws)
        end
        return Fm * basis.W          # nz × N
    end

    C0 = coefficients_at(zeros(Tv, nz))
    c₀ = vec(C0)
    Λ = Matrix{Tv}(undef, nz * N, nz)
    𝒜 = Matrix{Tv}(undef, nz, nz)
    c = C0 * basis.m
    e = zeros(Tv, nz)
    for j in 1:nz
        fill!(e, zero(Tv))
        e[j] = one(Tv)
        ΔC = coefficients_at(e)
        ΔC .-= C0
        Λ[:, j] = vec(ΔC)
        # E[f] = C(z) m is affine in z, so column j of 𝒜 is ΔC·m.
        ℒ.mul!(view(𝒜, :, j), ΔC, basis.m)
    end
    return 𝒜, c, c₀, Λ
end

# Fill the conditional covariance of a cubic innovation.  With
# C(z) = C̄ + Σᵢ zᵢDᵢ and shock-moment covariance Ψ,
#
#   E[C(Z)ΨC(Z)'] = C̄ΨC̄' + Σᵢⱼ Cov(Zᵢ,Zⱼ) DᵢΨDⱼ'.
#
# `Λnoise` contains only the columns Dᵢ that can be nonzero structurally.  This
# avoids forming or multiplying by the full augmented covariance for the extra
# term while retaining the full matrix Q required by the Kalman update.
function cubic_kalman_noise_covariance!(Q, C, Λnoise, Ψ, Pc, noise_state_indices,
                                        Pnoise, mixvec, mixΨ, CΨ)
    nI = length(noise_state_indices)
    nz, N = size(C)
    ℒ.mul!(CΨ, C, Ψ)
    ℒ.mul!(Q, CΨ, C')
    @inbounds for j in 1:nI, i in 1:nI
        Pnoise[i, j] = Pc[noise_state_indices[i], noise_state_indices[j]]
    end
    @inbounds for i in 1:nI
        ℒ.mul!(mixvec, Λnoise, view(Pnoise, :, i))
        mix = reshape(mixvec, nz, N)
        ℒ.mul!(mixΨ, mix, Ψ)
        Di = reshape(view(Λnoise, :, i), nz, N)
        ℒ.mul!(Q, mixΨ, Di', one(eltype(Q)), one(eltype(Q)))
    end
    @inbounds for j in 1:nz, i in 1:j
        m = (Q[i, j] + Q[j, i]) / 2
        Q[i, j] = m; Q[j, i] = m
    end
    return Q
end

# Solve the cubic stationary covariance fixed point
#
#   Σ = 𝒜Σ𝒜' + C̄ΨC̄' + K(Σ),
#
# where K is the state-covariance correction above.  Unlike the quadratic case,
# q₁₁ is itself in the loading support, so the correction is coupled to the
# augmented covariance.  The Float64 path stops on convergence; the dual path
# runs a fixed number of iterations so ForwardDiff sees a smooth computation.
function cubic_kalman_initial_covariance(𝒜, Cbar, Λnoise, Ψ, noise_state_indices;
                                         workspaces = nothing,
                                         lyapunov_algorithm::Symbol = :doubling,
                                         max_iterations::Int = 100,
                                         tolerance::Real = 1e-12)
    nz, N = size(Cbar)
    Tv = promote_type(eltype(𝒜), eltype(Cbar), eltype(Λnoise), eltype(Ψ))
    Qbase = Matrix{Tv}(undef, nz, nz)
    CΨ = Matrix{Tv}(undef, nz, N)
    ℒ.mul!(CΨ, Cbar, Ψ)
    ℒ.mul!(Qbase, CΨ, Cbar')
    Qbase = (Qbase + Qbase') / 2

    Pnoise = Matrix{Tv}(undef, length(noise_state_indices), length(noise_state_indices))
    mixvec = Vector{Tv}(undef, nz * N)
    mixΨ = Matrix{Tv}(undef, nz, N)
    Q = Matrix{Tv}(undef, nz, nz)
    Σ = qkf_lyapunov(𝒜, Qbase; workspaces = workspaces,
                     lyapunov_algorithm = lyapunov_algorithm)
    float_path = eltype(𝒜) <: AbstractFloat
    converged = false
    for iteration in 1:max_iterations
        copyto!(Q, Qbase)
        @inbounds for j in 1:length(noise_state_indices), i in 1:length(noise_state_indices)
            Pnoise[i, j] = Σ[noise_state_indices[i], noise_state_indices[j]]
        end
        @inbounds for i in 1:length(noise_state_indices)
            ℒ.mul!(mixvec, Λnoise, view(Pnoise, :, i))
            mix = reshape(mixvec, nz, N)
            ℒ.mul!(mixΨ, mix, Ψ)
            Di = reshape(view(Λnoise, :, i), nz, N)
            ℒ.mul!(Q, mixΨ, Di', one(Tv), one(Tv))
        end
        @inbounds for j in 1:nz, i in 1:j
            m = (Q[i, j] + Q[j, i]) / 2
            Q[i, j] = m; Q[j, i] = m
        end
        Σnew = qkf_lyapunov(𝒜, Q; workspaces = workspaces,
                            lyapunov_algorithm = lyapunov_algorithm)
        if float_path
            difference = maximum(abs, Σnew - Σ)
            scale = max(1.0, maximum(abs, Σnew))
            if difference <= tolerance * scale
                Σ = Σnew
                converged = true
                break
            end
        end
        Σ = Σnew
    end
    float_path && !converged && error("The cubic Kalman stationary covariance fixed point did not converge " *
                                     "within $max_iterations iterations.")
    return Σ
end

# Adjoint of the coupled stationary covariance equation.  This is the transpose
# fixed point X = 𝒜'X𝒜 + K*(X) + Σ̄, solved with a Lyapunov-preconditioned
# iteration because K* is inexpensive on the restricted loading support.  It
# supplies the exact implicit pullback of the converged Float64 covariance
# iteration without differentiating through it.
function cubic_kalman_stationary_adjoint(𝒜, Λnoise, Ψ, noise_state_indices, Σ̄;
                                         workspaces = nothing,
                                         lyapunov_algorithm::Symbol = :doubling,
                                         max_iterations::Int = 100,
                                         tolerance::Real = 1e-12)
    nz = size(Σ̄, 1)
    coefficient_count = size(Λnoise, 1) ÷ nz
    nI = length(noise_state_indices)
    X = (Σ̄ + Σ̄') / 2
    P̄noise = zeros(eltype(X), nI, nI)
    E = zeros(eltype(X), nz, coefficient_count)
    EΨ = zeros(eltype(X), nz, coefficient_count)
    Xold = similar(X)
    rhs = copy(Σ̄)
    X = qkf_lyapunov(Matrix(𝒜'), rhs; workspaces = workspaces,
                     lyapunov_algorithm = lyapunov_algorithm)
    for iteration in 1:max_iterations
        fill!(P̄noise, zero(eltype(P̄noise)))
        @inbounds for i in 1:nI
            Di = reshape(view(Λnoise, :, i), nz, coefficient_count)
            ℒ.mul!(E, X, Di)
            ℒ.mul!(EΨ, E, Ψ)
            for j in 1:nI
                Dj = reshape(view(Λnoise, :, j), nz, coefficient_count)
                P̄noise[i, j] = ℒ.dot(EΨ, Dj)
            end
        end
        copyto!(rhs, Σ̄)
        @inbounds for j in 1:nI, i in 1:nI
            rhs[noise_state_indices[i], noise_state_indices[j]] += P̄noise[i, j]
        end
        copyto!(Xold, X)
        Xnew = qkf_lyapunov(Matrix(𝒜'), rhs; workspaces = workspaces,
                            lyapunov_algorithm = lyapunov_algorithm)
        @inbounds for j in 1:nz, i in 1:j-1
            value = (Xnew[i, j] + Xnew[j, i]) / 2
            Xnew[i, j] = value
            Xnew[j, i] = value
        end
        difference = zero(eltype(Xnew))
        scale = zero(eltype(Xnew))
        @inbounds for j in 1:nz, i in 1:nz
            difference = max(difference, abs(Xnew[i, j] - Xold[i, j]))
            scale = max(scale, abs(Xnew[i, j]))
        end
        X = Xnew
        difference <= tolerance * max(one(eltype(Xnew)), scale) && return X
    end
    error("The cubic Kalman stationary covariance adjoint did not converge " *
          "within $max_iterations iterations.")
end

"""
Adjoint of `build_cubic_kalman_system`. The build is linear in the collected step
evaluations — `C(z) = F(z) W`, then `c = C(0)m`, `c₀ = vec C(0)`,
`𝒜[:,j] = ΔC_j m`, `Λ[:,j] = vec ΔC_j` with `ΔC_j = C(e_j) − C(0)` — so those
maps transpose directly, and the only real work is replaying the step adjoint at
the same `(n_z + 1)·N` points the forward pass visited.
"""
function build_cubic_kalman_system_pullback!(∂, sys, basis, ∂𝒜, ∂c, ∂c₀, ∂Λ;
                                             ws = cubic_kalman_workspace(sys))
    nz, N = sys.nz, basis.N
    m, W = basis.m, basis.W

    # C(0) is hit by c, by c₀, and negatively by every ΔC_j. Both of those sums
    # collapse — Σⱼ ∂𝒜[:,j] m' = (Σⱼ ∂𝒜[:,j]) m' and reshape is linear — so this
    # is one pass rather than n_z, each of which allocated an n_z×N array.
    ones_nz = ones(eltype(∂𝒜), nz)
    ∂C0 = ∂c * m' .+ reshape(∂c₀, nz, N) .-
          (∂𝒜 * ones_nz) * m' .- reshape(∂Λ * ones_nz, nz, N)

    # ∂F = ∂ΔC W' with ∂ΔC = ∂𝒜[:,j] m' + reshape(∂Λ[:,j]); the first term is
    # rank one and contracts to an outer product with W m, so nothing per-`j`
    # needs to be materialised.
    Wm = W * m
    ∂F = Matrix{eltype(∂𝒜)}(undef, nz, N)
    zj = zeros(nz)

    ℒ.mul!(∂F, ∂C0, W')
    @inbounds for (p, ε) in enumerate(basis.pts)
        cubic_kalman_step_pullback!(∂, sys, zj, ε, view(∂F, :, p), ws)
    end

    @inbounds for j in 1:nz
        ℒ.mul!(∂F, reshape(view(∂Λ, :, j), nz, N), W')
        ℒ.mul!(∂F, view(∂𝒜, :, j), Wm', one(eltype(∂F)), one(eltype(∂F)))
        fill!(zj, 0.0); zj[j] = 1.0
        for (p, ε) in enumerate(basis.pts)
            cubic_kalman_step_pullback!(∂, sys, zj, ε, view(∂F, :, p), ws)
        end
    end

    cubic_derived_pullback!(∂, sys)
    return ∂
end

"""
Kalman recursion on the cubic augmented state. Mirrors `run_quadratic_kalman`:
the noise covariance is rebuilt from the current state estimate every period,
because it depends on the state exactly as `G(z)G(z)'` does at second order —
here as `C(z) Ψ C(z)'`, with `C` affine in `z`, so a period costs one matvec and
two gemms rather than a quadrature sweep.
"""
function run_cubic_kalman(sys, data_in_deviations::AbstractMatrix{<:Real};
                          measurement_error::Union{Nothing,AbstractVector{<:Real},AbstractMatrix{<:Real}} = nothing,
                          presample_periods::Int = 0,
                          on_failure_loglikelihood::Real = -Inf,
                          workspaces = nothing,
                          lyapunov_algorithm::Symbol = :doubling)
    nz = sys.nz
    n_obs, nT = size(data_in_deviations)
    presample_periods = normalize_presample_periods(presample_periods, nT)

    # Promote over every differentiable input. The preallocated buffers below fix
    # the element type, so missing one makes forward-mode AD fail with respect to
    # exactly that argument.
    Tv = promote_type(eltype(sys.S1), eltype(data_in_deviations),
                      measurement_error === nothing ? Float64 : eltype(measurement_error))

    Hm = if measurement_error === nothing
        zeros(Tv, n_obs, n_obs)
    elseif measurement_error isa AbstractMatrix
        Matrix{Tv}(measurement_error)
    else
        Matrix{Tv}(ℒ.Diagonal(collect(measurement_error)))
    end

    ws = cubic_kalman_workspace(sys, Tv)
    basis = cubic_noise_basis(sys.nExo)
    𝒜, c, c₀, Λ = build_cubic_kalman_system(sys, basis; ws = ws)
    N, Ψ = basis.N, basis.Ψ
    noise_state_indices = sys.noise_state_indices
    Λnoise = Matrix(Λ[:, noise_state_indices])

    cvec = Vector{Tv}(undef, nz * N)
    CΨ = Matrix{Tv}(undef, nz, N)
    Q = Matrix{Tv}(undef, nz, nz)
    Pnoise = Matrix{Tv}(undef, length(noise_state_indices), length(noise_state_indices))
    mixvec = Vector{Tv}(undef, nz * N)
    mixΨ = Matrix{Tv}(undef, nz, N)
    # Q(z) = C(z) Ψ C(z)' with vec(C) = c₀ + Λz.
    noise_covariance! = function (Q, z, Pc)
        copyto!(cvec, c₀)
        ℒ.mul!(cvec, Λ, z, one(Tv), one(Tv))
        C = reshape(cvec, nz, N)
        cubic_kalman_noise_covariance!(Q, C, Λnoise, Ψ, Pc, noise_state_indices,
                                       Pnoise, mixvec, mixΨ, CΨ)
    end

    z = (Matrix{Tv}(ℒ.I(nz)) - 𝒜) \ c
    Cbar = reshape(c₀ + Λ * z, nz, N)
    Σ = cubic_kalman_initial_covariance(𝒜, Cbar, Λnoise, Ψ, noise_state_indices;
                                        workspaces = workspaces,
                                        lyapunov_algorithm = lyapunov_algorithm)

    # Preallocate the recursion's working matrices once, as the quadratic filter
    # does: the covariance propagation is the whole cost, and allocating an nz×nz
    # temporary per period competes with it directly.
    op1, op2, op3 = sys.op1, sys.op2, sys.op3
    Pp = Matrix{Tv}(undef, nz, nz)
    Tm = Matrix{Tv}(undef, nz, nz)
    Pc = Matrix{Tv}(undef, nz, nz); copyto!(Pc, Σ)
    zp = Vector{Tv}(undef, nz)
    CP = Matrix{Tv}(undef, n_obs, nz)
    F = Matrix{Tv}(undef, n_obs, n_obs)
    Kg = Matrix{Tv}(undef, nz, n_obs)
    v = Vector{Tv}(undef, n_obs)
    Fv = Vector{Tv}(undef, n_obs)

    ll = zero(Tv)
    log2pi = log(2π)
    @inbounds for t in 1:nT
        noise_covariance!(Q, z, Pc)

        # Pp = 𝒜 Pc 𝒜' + Q
        ℒ.mul!(Tm, 𝒜, Pc)
        ℒ.mul!(Pp, Tm, 𝒜')
        Pp .+= Q
        for j in 1:nz, i in 1:j
            m = (Pp[i, j] + Pp[j, i]) / 2
            Pp[i, j] = m; Pp[j, i] = m
        end

        ℒ.mul!(zp, 𝒜, z); zp .+= c

        # yₜ = (x₁ + x₂ + x₃)[observables] — three selected rows, so index rather
        # than multiply by C.
        for i in 1:n_obs
            v[i] = data_in_deviations[i, t] - (zp[op1[i]] + zp[op2[i]] + zp[op3[i]])
            for k in 1:nz
                CP[i, k] = Pp[op1[i], k] + Pp[op2[i], k] + Pp[op3[i], k]
            end
        end
        for i in 1:n_obs, j in 1:n_obs
            F[i, j] = CP[i, op1[j]] + CP[i, op2[j]] + CP[i, op3[j]] + Hm[i, j]
        end
        for i in 1:n_obs, j in 1:i-1
            m = (F[i, j] + F[j, i]) / 2
            F[i, j] = m; F[j, i] = m
        end

        # Fc = chol(F), overwriting F because the covariance is dead after factorization.
        Fc = ℒ.cholesky!(ℒ.Symmetric(F), check = false)
        ℒ.issuccess(Fc) || return on_failure_loglikelihood

        if t > presample_periods
            copyto!(Fv, v); ℒ.ldiv!(Fc, Fv)
            ll -= 0.5 * (ℒ.dot(v, Fv) + ℒ.logdet(Fc) + n_obs * log2pi)
            isfinite(ll) || return on_failure_loglikelihood
        end

        # K = CP' F⁻¹ ; z = zp + K v ; Pc = Pp − K CP.  `rdiv!` has no Cholesky
        # method for dual element types, so the AD path solves and transposes.
        if Tv === Float64
            copyto!(Kg, CP'); ℒ.rdiv!(Kg, Fc)
        else
            Kg = Matrix((Fc \ CP)')
        end
        copyto!(z, zp); ℒ.mul!(z, Kg, v, one(Tv), one(Tv))
        copyto!(Pc, Pp); ℒ.mul!(Pc, Kg, CP, -one(Tv), one(Tv))
        for j in 1:nz, i in 1:j
            m = (Pc[i, j] + Pc[j, i]) / 2
            Pc[i, j] = m; Pc[j, i] = m
        end
    end
    return ll
end


"""
Taped forward pass plus adjoint for the cubic Kalman recursion. Mirrors
`quadratic_kalman_recursion`'s verified adjoint; the one structural difference is
the noise term, `Q = C(\bar z)ΨC(\bar z)' + Q_state` with `vec(C) = c₀ + Λz`, in place of
`G(\bar z)G(\bar z)' + Q_H + Q_state`.

`Q̄` is symmetric here because `P̄p` is symmetrised before use, so the cotangent of
`C` is `2 Q̄ C Ψ` rather than `(Q̄ + Q̄')CΨ`.
"""
function cubic_kalman_recursion_taped(𝒜, c, c₀, Λ, Ψ, Hm, Y, 𝒞, z0, Σ0, nz, N,
                                      presample_periods, on_failure_loglikelihood,
                                      noise_state_indices)
    n_obs, nT = size(Y)
    z = copy(z0); Pc = copy(Σ0)
    zs = Vector{Vector{Float64}}(); Ps = Vector{Matrix{Float64}}()
    Cs = Vector{Matrix{Float64}}(); Pas = Vector{Matrix{Float64}}()
    vs = Vector{Vector{Float64}}()
    CPs = Vector{Matrix{Float64}}(); Fis = Vector{Matrix{Float64}}()
    Ks = Vector{Matrix{Float64}}()
    ll = 0.0; log2pi = log(2π)
    Λnoise = Matrix(Λ[:, noise_state_indices])
    Pnoise = zeros(length(noise_state_indices), length(noise_state_indices))
    mixvec = zeros(nz * N); mixΨ = zeros(nz, N); CΨ = zeros(nz, N); Q = zeros(nz, nz)

    for t in 1:nT
        push!(zs, copy(z)); push!(Ps, copy(Pc))
        C = reshape(c₀ + Λ * z, nz, N)
        cubic_kalman_noise_covariance!(Q, C, Λnoise, Ψ, Pc, noise_state_indices,
                                       Pnoise, mixvec, mixΨ, CΨ)
        push!(Pas, copy(Pnoise))
        zp = 𝒜 * z + c
        Pp = 𝒜 * Pc * 𝒜' + Q; Pp = (Pp + Pp') / 2
        v = Y[:, t] - 𝒞 * zp
        CP = 𝒞 * Pp
        F = CP * 𝒞' + Hm; F = (F + F') / 2
        # Fc = chol(F), overwriting F because the covariance is dead after factorization.
        Fc = ℒ.cholesky!(ℒ.Symmetric(F), check = false)
        ℒ.issuccess(Fc) || return on_failure_loglikelihood, nothing
        Fi = inv(Fc)
        if t > presample_periods
            ll -= 0.5 * (ℒ.dot(v, Fi * v) + ℒ.logdet(Fc) + n_obs * log2pi)
        end
        K = CP' * Fi
        z = zp + K * v
        Pc = Pp - K * CP; Pc = (Pc + Pc') / 2
        push!(Cs, C); push!(vs, v); push!(CPs, CP); push!(Fis, Fi); push!(Ks, K)
    end
    return ll, (; zs, Ps, Cs, Pas, vs, CPs, Fis, Ks)
end

function cubic_kalman_recursion_pullback(tape, 𝒜, c, c₀, Λ, Ψ, 𝒞, nz, N, n_obs, nT,
                                         presample_periods, ∂ll, noise_state_indices)
    (; zs, Ps, Cs, Pas, vs, CPs, Fis, Ks) = tape
    𝒜̄ = zeros(nz, nz); c̄ = zeros(nz); c̄₀ = zeros(length(c₀)); Λ̄ = zeros(size(Λ))
    H̄m = zeros(n_obs, n_obs); Ȳ = zeros(n_obs, nT)
    z̄ = zeros(nz); P̄ = zeros(nz, nz)
    nI = length(noise_state_indices)
    Λnoise = view(Λ, :, noise_state_indices)
    P̄noise = zeros(nI, nI)
    E = zeros(nz, N)
    EΨ = zeros(nz, N)
    mixvec = zeros(nz * N)
    Dmix = reshape(mixvec, nz, N)
    D̄ = zeros(nz, N)

    for t in nT:-1:1
        z_, P_, C, v, CP, Fi, K = zs[t], Ps[t], Cs[t], vs[t], CPs[t], Fis[t], Ks[t]
        P̄p = copy(P̄)
        K̄ = -P̄ * CP'
        C̄P = -K' * P̄
        z̄p = copy(z̄)
        K̄ .+= z̄ * v'
        v̄ = K' * z̄
        C̄P .+= Fi * K̄'
        F̄ = -Fi * (CP * K̄) * Fi
        if t > presample_periods
            v̄ .+= -∂ll * (Fi * v)
            F̄ .+= ∂ll * 0.5 * (Fi * v * v' * Fi - Fi)
        end
        F̄ = (F̄ + F̄') / 2
        C̄P .+= F̄ * 𝒞
        H̄m .+= F̄
        P̄p .+= 𝒞' * C̄P
        z̄p .+= -𝒞' * v̄
        Ȳ[:, t] .+= v̄
        P̄p = (P̄p + P̄p') / 2
        𝒜̄ .+= 2 .* (P̄p * 𝒜 * P_)
        P̄ = 𝒜' * P̄p * 𝒜
        Q̄ = P̄p
        C̄ = 2 .* (Q̄ * C * Ψ)
        vC̄ = vec(C̄)
        c̄₀ .+= vC̄
        Λ̄ .+= vC̄ * z_'
        fill!(P̄noise, zero(eltype(P̄noise)))
        @inbounds for i in 1:nI
            Di = reshape(view(Λnoise, :, i), nz, N)
            ℒ.mul!(E, Q̄, Di)
            ℒ.mul!(EΨ, E, Ψ)
            for j in 1:nI
                Dj = reshape(view(Λnoise, :, j), nz, N)
                P̄noise[i, j] = ℒ.dot(EΨ, Dj)
            end
            ℒ.mul!(mixvec, Λnoise, view(Pas[t], :, i))
            ℒ.mul!(D̄, Q̄, Dmix)
            ℒ.mul!(E, D̄, Ψ)
            column = noise_state_indices[i]
            @inbounds for k in eachindex(E)
                Λ̄[k, column] += 2 * E[k]
            end
        end
        @inbounds for j in 1:nI, i in 1:nI
            P̄[noise_state_indices[i], noise_state_indices[j]] += P̄noise[i, j]
        end
        𝒜̄ .+= z̄p * z_'
        c̄ .+= z̄p
        z̄ = 𝒜' * z̄p + Λ' * vC̄
    end
    return 𝒜̄, c̄, c̄₀, Λ̄, H̄m, Ȳ, z̄, P̄
end


# ── standard filter interface ────────────────────────────────────────────────
function calculate_loglikelihood(::Val{:cubic_kalman},
                                 ::Val{:pruned_third_order},
                                 observables_index::Vector{Int},
                                 𝐒,
                                 data_in_deviations::AbstractMatrix,
                                 constants,
                                 state,
                                 workspaces;
                                 warmup_iterations::Int = 0,
                                 presample_periods::Int = 0,
                                 initial_covariance = :theoretical,
                                 filter_algorithm::Symbol = :LagrangeNewton,
                                 lyapunov_algorithm::Symbol = :doubling,
                                 on_failure_loglikelihood = -Inf,
                                 measurement_error = nothing,
                                 opts::CalculationOptions = merge_calculation_options())
    sys = build_cubic_kalman_system_from_constants(constants, 𝐒[1], 𝐒[2], 𝐒[3], observables_index)
    return run_cubic_kalman(sys, data_in_deviations;
                            measurement_error = measurement_error,
                            presample_periods = presample_periods,
                            on_failure_loglikelihood = on_failure_loglikelihood,
                            workspaces = workspaces,
                            lyapunov_algorithm = lyapunov_algorithm)
end

function rrule(::typeof(calculate_loglikelihood),
               ::Val{:cubic_kalman},
               ::Val{:pruned_third_order},
               observables_index::Vector{Int},
               𝐒,
               data_in_deviations::AbstractMatrix,
               constants,
               state,
               workspaces;
               warmup_iterations::Int = 0,
               presample_periods::Int = 0,
               initial_covariance = :theoretical,
               filter_algorithm::Symbol = :LagrangeNewton,
               lyapunov_algorithm::Symbol = :doubling,
               on_failure_loglikelihood = -Inf,
               measurement_error = nothing,
               opts::CalculationOptions = merge_calculation_options())
    sys = build_cubic_kalman_system_from_constants(constants, 𝐒[1], 𝐒[2], 𝐒[3], observables_index)
    nz = sys.nz
    n_obs, nT = size(data_in_deviations)
    presample = normalize_presample_periods(presample_periods, nT)

    Hm = measurement_error === nothing ? zeros(n_obs, n_obs) :
         measurement_error isa AbstractMatrix ? Matrix{Float64}(measurement_error) :
         Matrix{Float64}(ℒ.Diagonal(collect(measurement_error)))

    ws = cubic_kalman_workspace(sys)
    basis = cubic_noise_basis(sys.nExo)
    𝒜, c, c₀, Λ = build_cubic_kalman_system(sys, basis; ws = ws)
    N, Ψ, 𝒞 = basis.N, basis.Ψ, sys.C
    noise_state_indices = sys.noise_state_indices
    Λnoise = Matrix(Λ[:, noise_state_indices])

    z₀ = (Matrix{Float64}(ℒ.I(nz)) - 𝒜) \ c
    C₀ = reshape(c₀ + Λ * z₀, nz, N)
    Σ₀ = cubic_kalman_initial_covariance(𝒜, C₀, Λnoise, Ψ, noise_state_indices;
                                          workspaces = workspaces,
                                          lyapunov_algorithm = lyapunov_algorithm)

    llh, tape = cubic_kalman_recursion_taped(𝒜, c, c₀, Λ, Ψ, Hm, Matrix(data_in_deviations),
                                             𝒞, z₀, Σ₀, nz, N, presample, on_failure_loglikelihood,
                                             noise_state_indices)

    nine(x...) = (NoTangent(), NoTangent(), NoTangent(), NoTangent(), x[1], x[2],
                  NoTangent(), x[3], NoTangent())

    if !isfinite(llh) || tape === nothing
        return llh, _ -> nine(NoTangent(), NoTangent(), NoTangent())
    end

    function cubic_kalman_loglikelihood_pullback(∂llh_bar)
        ∂llh = unthunk(∂llh_bar)
        𝒜̄, c̄, c̄₀, Λ̄, _, Ȳ, z̄₀, Σ̄₀ =
            cubic_kalman_recursion_pullback(tape, 𝒜, c, c₀, Λ, Ψ, 𝒞, nz, N, n_obs, nT,
                                            presample, ∂llh, noise_state_indices)

        # Σ₀ = 𝒜Σ₀𝒜' + C̄ΨC̄' + K(Σ₀).  The adjoint is the corresponding
        # coupled fixed point, so this remains the exact pullback of the
        # state-dependent stationary covariance.
        X = cubic_kalman_stationary_adjoint(𝒜, Λnoise, Ψ, noise_state_indices,
                                            Σ̄₀; workspaces = workspaces,
                                            lyapunov_algorithm = lyapunov_algorithm)
        𝒜̄ .+= 2 .* (X * 𝒜 * Σ₀)
        Q̄₀ = (X + X') / 2
        C̄₀ = 2 .* (Q̄₀ * C₀ * Ψ)
        vC̄₀ = vec(C̄₀)
        c̄₀ = c̄₀ .+ vC̄₀
        Λ̄ = Λ̄ .+ vC̄₀ * z₀'
        z̄₀ = z̄₀ .+ Λ' * vC̄₀
        P₀ = Σ₀[noise_state_indices, noise_state_indices]
        @inbounds for i in 1:length(noise_state_indices)
            Dmix = reshape(Λnoise * view(P₀, :, i), nz, N)
            D̄ = 2 .* (Q̄₀ * Dmix * Ψ)
            Λ̄[:, noise_state_indices[i]] .+= vec(D̄)
        end

        # z₀ = (I − 𝒜)⁻¹ c
        y = (Matrix{Float64}(ℒ.I(nz)) - 𝒜)' \ z̄₀
        c̄ = c̄ .+ y
        𝒜̄ .+= y * z₀'

        ∂ = cubic_kalman_cotangents(sys)
        build_cubic_kalman_system_pullback!(∂, sys, basis, 𝒜̄, c̄, c̄₀, Λ̄; ws = ws)

        # scatter the retained rows back onto the full solution matrices
        ∂𝐒1 = zeros(size(𝐒[1])); ∂𝐒2 = zeros(size(𝐒[2])); ∂𝐒3 = zeros(size(𝐒[3]))
        ∂𝐒1[sys.oas, :] = ∂.S1
        ∂𝐒2[sys.oas, :] = ∂.S2
        ∂𝐒3[sys.oas, :] = ∂.S3
        ∂state = [zeros(length(s)) for s in state]
        return nine([∂𝐒1, ∂𝐒2, ∂𝐒3], Ȳ, ∂state)
    end

    return llh, cubic_kalman_loglikelihood_pullback
end

end # @stable
