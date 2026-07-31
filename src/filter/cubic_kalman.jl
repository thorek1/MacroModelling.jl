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
# What is exact and what is not. The transition is exactly linear, and the
# conditional mean and variance are computed by Gauss-Hermite quadrature that is
# exact for the polynomials involved (f is cubic in ε, so f f' is degree six).
# What is approximate is the same thing as at second order: the innovation is not
# Gaussian, and the filter matches only its first two moments. See
# `docs/src/filters.md`.
#
# Cost. q₁₁ and q₁₁₁ are symmetric and carried compressed, giving an augmented
# dimension of 3n_r + n_past(n_past+1)/2 + n_past² + n_past(n_past+1)(n_past+2)/6.
# The O(n_z³) covariance recursion still scales as n_past⁹, so the filter stays
# confined to small models — see `CUBIC_KALMAN_MAX_DIMENSION` below.

# The covariance recursion is two n_z×n_z triple products per period. Beyond this
# dimension a single period costs seconds and a single matrix hundreds of MB, so
# refuse with a message that names the cause instead of appearing to hang.
const CUBIC_KALMAN_MAX_DIMENSION = 2500

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
    nq11, nq12, nq111 = length(can2), nPast^2, length(can3)
    nz = 3nr + nq11 + nq12 + nq111

    if nz > CUBIC_KALMAN_MAX_DIMENSION
        error("The cubic Kalman filter needs an augmented state of dimension " *
              "$nz (= 3·$nr + $nq11 + $nq12 + $nq111) for this model, and its " *
              "covariance recursion is O(n_z³) per period. The limit is " *
              "$CUBIC_KALMAN_MAX_DIMENSION (`CUBIC_KALMAN_MAX_DIMENSION`). " *
              "Use `filter = :inversion` or a particle filter instead.")
    end

    S1 = Matrix(𝐒₁)[oas, :]
    S2 = Matrix(𝐒₂)[oas, :]
    S3 = Matrix(𝐒₃)[oas, :]

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

    # aₙ = M a + mc + V ε
    A1 = Pm * S1
    M = A1[:, 1:nPast]
    mc = A1[:, nPast+1]
    V = A1[:, nPast+2:na]

    # bₙ = M b + B2·K₂, with K₂ split into its (a,a), (a,tail)+(tail,a) and
    # (tail,tail) parts so each can be routed onto the right state block.
    B2 = Pm * S2 / 2
    ntail = 1 + nExo
    Wq = zeros(nPast, nPast * nPast)
    for i in 1:nPast, j in 1:nPast
        Wq[:, (i-1)*nPast+j] = B2[:, (i-1)*na+j]
    end
    Wl_t = [zeros(nPast, nPast) for _ in 1:ntail]
    for t in 1:ntail, k in 1:nPast
        Wl_t[t][:, k] = B2[:, (k-1)*na+nPast+t] + B2[:, (nPast+t-1)*na+k]
    end
    Bc = B2[:, [(i-1)*na + j for i in nPast+1:na for j in nPast+1:na]]
    MM = ℒ.kron(M, M)

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

    return (; nr, nPast, nExo, na, nz, oas, S1, S2, S3, Pm, C, op1, op2, op3,
            r1, r2, r3, i11, i12, i111, nq11, nq12, nq111,
            exp2, can2, exp3, can3, can2_ij, can3_ijk,
            M, mc, V, B2, Wq, Wl_t, Bc, MM, ntail)
end

"""
Preallocated buffers for `cubic_kalman_step!`. The step is called
`(n_z + 1) · n_nodes` times to build the transition and `n_nodes` times per
period, and it is entirely allocation-bound — it does a few hundred flops but
allocated ~13 kB per call before these buffers existed.
"""
function cubic_kalman_workspace(sys)
    (; nr, nPast, na, ntail) = sys
    nP2 = nPast * nPast
    return (; a = zeros(nPast), b = zeros(nPast), p = zeros(nPast),
            q11 = zeros(nP2), q12 = zeros(nP2), q111 = zeros(nPast^3),
            tail = zeros(ntail), tt = zeros(ntail * ntail),
            aug1 = zeros(na), aug1h = zeros(na), aug2 = zeros(na), aug3 = zeros(na),
            K2 = zeros(na * na), K12 = zeros(na * na), K3 = zeros(na * na * na),
            x1n = zeros(nr), x2n = zeros(nr), x3n = zeros(nr),
            u = zeros(nPast), v = zeros(nPast), bn = zeros(nPast), wc = zeros(nPast),
            Q11 = zeros(nPast, nPast), Q12 = zeros(nPast, nPast), Q111 = zeros(nPast, nP2),
            R2 = zeros(nPast, nPast), Tmp = zeros(nPast, nPast),
            MQ111 = zeros(nPast, nP2), R3 = zeros(nPast, nP2),
            Wl = zeros(nPast, nPast), t2 = zeros(nP2), vv = zeros(nP2))
end

"""
One step of the augmented map, `z ↦ f(z, ε)`. Affine in `z` by construction:
every product of two `z`-dependent quantities is read off an existing block
rather than recomputed.
"""
function cubic_kalman_step!(out::AbstractVector, sys, z::AbstractVector, ε::AbstractVector, ws)
    (; nr, nPast, nExo, na, S1, S2, S3, Pm, r1, r2, r3, i11, i12, i111,
       exp2, can2_ij, can3_ijk, M, mc, V, Wq, Wl_t, Bc, MM, ntail) = sys
    (; a, b, p, q11, q12, q111, tail, tt, aug1, aug1h, aug2, aug3, K2, K12, K3,
       x1n, x2n, x3n, u, v, bn, wc, Q11, Q12, Q111, R2, Tmp, MQ111, R3, Wl, t2, vv) = ws
    nP = nPast
    nP2 = nP * nP

    ℒ.mul!(a, Pm, view(z, r1))
    ℒ.mul!(b, Pm, view(z, r2))
    ℒ.mul!(p, Pm, view(z, r3))

    # The symmetric blocks arrive compressed; expand them so the algebra below is
    # written on plain Kronecker products.
    o11 = first(i11) - 1
    o111 = first(i111) - 1
    @inbounds for r in eachindex(q11)
        q11[r] = z[o11+exp2[r]]
    end
    @inbounds for r in eachindex(q111)
        q111[r] = z[o111+sys.exp3[r]]
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

    # Kronecker inputs, with the all-past blocks read from the state.
    @inbounds for i in 1:na, j in 1:na
        K2[(i-1)*na+j] = (i <= nP && j <= nP) ? q11[(i-1)*nP+j] : aug1[i] * aug1[j]
    end
    fill!(K12, 0.0)
    @inbounds for i in 1:na, j in 1:nP
        K12[(i-1)*na+j] = (i <= nP) ? q12[(i-1)*nP+j] : aug1h[i] * aug2[j]
    end
    @inbounds for i in 1:na, j in 1:na, k in 1:na
        r = ((i-1)*na + (j-1)) * na + k
        ci = i <= nP; cj = j <= nP; ck = k <= nP
        n = ci + cj + ck
        K3[r] = if n == 3
            q111[((i-1)*nP + (j-1))*nP + k]
        elseif n == 2
            if ci && cj
                q11[(i-1)*nP+j] * aug1[k]
            elseif ci && ck
                q11[(i-1)*nP+k] * aug1[j]
            else
                q11[(j-1)*nP+k] * aug1[i]
            end
        else
            aug1[i] * aug1[j] * aug1[k]
        end
    end

    ℒ.mul!(x1n, S1, aug1)
    ℒ.mul!(x2n, S1, aug2); ℒ.mul!(x2n, S2, K2, 0.5, 1.0)
    ℒ.mul!(x3n, S1, aug3); ℒ.mul!(x3n, S2, K12, 1.0, 1.0); ℒ.mul!(x3n, S3, K3, 1/6, 1.0)

    # New Kronecker blocks, kept affine in z.
    ℒ.mul!(u, M, a)                    # z-dependent, linear in a
    copyto!(v, mc); ℒ.mul!(v, V, ε, 1.0, 1.0)   # z-independent
    ℒ.mul!(bn, Pm, x2n)                # affine in z

    @inbounds for j in 1:nP, s in 1:nP
        Q11[j, s] = q11[(j-1)*nP+s]
        Q12[j, s] = q12[(j-1)*nP+s]
    end
    @inbounds for j in 1:nP, s in 1:nP2
        Q111[j, s] = q111[(j-1)*nP2+s]
    end

    ℒ.mul!(Tmp, M, Q11); ℒ.mul!(R2, Tmp, M')     # M Q₁₁ M' — its rowvec is u⊗u
    @inbounds for i in 1:nP, r in 1:nP
        t2[(i-1)*nP+r] = R2[i, r]
    end
    @inbounds for i in 1:nP, j in 1:nP
        vv[(i-1)*nP+j] = v[i] * v[j]
    end

    fill!(Wl, 0.0)
    @inbounds for t in 1:ntail
        ℒ.axpy!(tail[t], Wl_t[t], Wl)
    end
    @inbounds for i in 1:ntail, j in 1:ntail
        tt[(i-1)*ntail+j] = tail[i] * tail[j]
    end
    ℒ.mul!(wc, Bc, tt)

    # R2 ← M Q₁₂ M' + (M Q₁₁₁) Wq' + (M Q₁₁) Wl'; its rowvec is the z-dependent
    # part of q₁₂', and Tmp still holds M Q₁₁ from above.
    ℒ.mul!(MQ111, M, Q111)
    ℒ.mul!(R2, Tmp, Wl')
    ℒ.mul!(R2, MQ111, Wq', 1.0, 1.0)
    ℒ.mul!(Tmp, M, Q12)
    ℒ.mul!(R2, Tmp, M', 1.0, 1.0)

    ℒ.mul!(R3, MQ111, MM')             # u⊗u⊗u = (M⊗M⊗M) q₁₁₁

    @inbounds for i in 1:nr
        out[i] = x1n[i]; out[nr+i] = x2n[i]; out[2nr+i] = x3n[i]
    end
    # q₁₁' and q₁₁₁' are symmetric, so only the canonical entries are formed.
    @inbounds for (s, (i, j)) in enumerate(can2_ij)
        out[o11+s] = t2[(i-1)*nP+j] + u[i]*v[j] + v[i]*u[j] + v[i]*v[j]
    end
    @inbounds for i in 1:nP, j in 1:nP
        out[first(i12)-1 + (i-1)*nP+j] = R2[i, j] + u[i]*wc[j] + v[i]*bn[j]
    end
    @inbounds for (s, (i, j, k)) in enumerate(can3_ijk)
        out[o111+s] = R3[i, (j-1)*nP+k] +          # u⊗u⊗u
                      t2[(i-1)*nP+j] * v[k] +      # u⊗u⊗v
                      v[i] * t2[(j-1)*nP+k] +      # v⊗u⊗u
                      t2[(i-1)*nP+k] * v[j] +      # u⊗v⊗u
                      u[i] * vv[(j-1)*nP+k] +      # u⊗v⊗v
                      vv[(i-1)*nP+j] * u[k] +      # v⊗v⊗u
                      v[i] * u[j] * v[k] +         # v⊗u⊗v
                      vv[(i-1)*nP+j] * v[k]        # v⊗v⊗v
    end
    return out
end

# Allocating convenience wrapper, used by the tests.
function cubic_kalman_step(sys, z::AbstractVector, ε::AbstractVector, ws = cubic_kalman_workspace(sys))
    return cubic_kalman_step!(Vector{Float64}(undef, sys.nz), sys, z, ε, ws)
end

# E[f(z,·)] and Var(f(z,·)) under ε ~ N(0,I), exactly.
#
# The node evaluations are stacked into one matrix and contracted with a single
# gemm rather than accumulated as `nnodes` rank-one updates: same arithmetic, but
# it runs at BLAS-3 rather than BLAS-2 speed. `buf` may be supplied to reuse the
# stacking buffer across periods.
function cubic_kalman_moments(sys, z, nodes, wts; buf = nothing, ws = cubic_kalman_workspace(sys))
    nz = sys.nz
    Fm = buf === nothing ? Matrix{Float64}(undef, nz, length(nodes)) : buf
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
    Fm = Matrix{Float64}(undef, sys.nz, length(nodes))
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
    nz, N = sys.nz, basis.N
    Fm = Matrix{Float64}(undef, nz, N)
    coefficients_at = function (z)
        @inbounds for (p, ε) in enumerate(basis.pts)
            cubic_kalman_step!(view(Fm, :, p), sys, z, ε, ws)
        end
        return Fm * basis.W          # nz × N
    end

    C0 = coefficients_at(zeros(nz))
    c₀ = vec(C0)
    Λ = Matrix{Float64}(undef, nz * N, nz)
    𝒜 = Matrix{Float64}(undef, nz, nz)
    c = C0 * basis.m
    e = zeros(nz)
    for j in 1:nz
        fill!(e, 0.0)
        e[j] = 1.0
        ΔC = coefficients_at(e)
        ΔC .-= C0
        Λ[:, j] = vec(ΔC)
        # E[f] = C(z) m is affine in z, so column j of 𝒜 is ΔC·m.
        ℒ.mul!(view(𝒜, :, j), ΔC, basis.m)
    end
    return 𝒜, c, c₀, Λ
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

    Hm = if measurement_error === nothing
        zeros(n_obs, n_obs)
    elseif measurement_error isa AbstractMatrix
        Matrix{Float64}(measurement_error)
    else
        Matrix{Float64}(ℒ.Diagonal(collect(measurement_error)))
    end

    ws = cubic_kalman_workspace(sys)
    basis = cubic_noise_basis(sys.nExo)
    𝒜, c, c₀, Λ = build_cubic_kalman_system(sys, basis; ws = ws)
    N, Ψ = basis.N, basis.Ψ

    cvec = Vector{Float64}(undef, nz * N)
    CΨ = Matrix{Float64}(undef, nz, N)
    Q = Matrix{Float64}(undef, nz, nz)
    # Q(z) = C(z) Ψ C(z)' with vec(C) = c₀ + Λz.
    noise_covariance! = function (Q, z)
        copyto!(cvec, c₀)
        ℒ.mul!(cvec, Λ, z, 1.0, 1.0)
        C = reshape(cvec, nz, N)
        ℒ.mul!(CΨ, C, Ψ)
        ℒ.mul!(Q, CΨ, C')
        for j in 1:nz, i in 1:j
            m = (Q[i, j] + Q[j, i]) / 2
            Q[i, j] = m; Q[j, i] = m
        end
        return Q
    end

    z = (Matrix{Float64}(ℒ.I(nz)) - 𝒜) \ c
    Σ = qkf_lyapunov(𝒜, noise_covariance!(Q, z); workspaces = workspaces,
                     lyapunov_algorithm = lyapunov_algorithm)

    # Preallocate the recursion's working matrices once, as the quadratic filter
    # does: the covariance propagation is the whole cost, and allocating an nz×nz
    # temporary per period competes with it directly.
    op1, op2, op3 = sys.op1, sys.op2, sys.op3
    Pp = Matrix{Float64}(undef, nz, nz)
    Tm = Matrix{Float64}(undef, nz, nz)
    Pc = Matrix{Float64}(undef, nz, nz); copyto!(Pc, Σ)
    zp = Vector{Float64}(undef, nz)
    CP = Matrix{Float64}(undef, n_obs, nz)
    F = Matrix{Float64}(undef, n_obs, n_obs)
    Kg = Matrix{Float64}(undef, nz, n_obs)
    v = Vector{Float64}(undef, n_obs)
    Fv = Vector{Float64}(undef, n_obs)

    ll = 0.0
    log2pi = log(2π)
    @inbounds for t in 1:nT
        noise_covariance!(Q, z)

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

        Fc = ℒ.cholesky(F, check = false)
        ℒ.issuccess(Fc) || return on_failure_loglikelihood

        if t > presample_periods
            copyto!(Fv, v); ℒ.ldiv!(Fc, Fv)
            ll -= 0.5 * (ℒ.dot(v, Fv) + ℒ.logdet(Fc) + n_obs * log2pi)
            isfinite(ll) || return on_failure_loglikelihood
        end

        # K = CP' F⁻¹ ; z = zp + K v ; Pc = Pp − K CP
        copyto!(Kg, CP'); ℒ.rdiv!(Kg, Fc)
        copyto!(z, zp); ℒ.mul!(z, Kg, v, 1.0, 1.0)
        copyto!(Pc, Pp); ℒ.mul!(Pc, Kg, CP, -1.0, 1.0)
        for j in 1:nz, i in 1:j
            m = (Pc[i, j] + Pc[j, i]) / 2
            Pc[i, j] = m; Pc[j, i] = m
        end
    end
    return ll
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

end # @stable
