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
# Cost. The augmented dimension is 3n_r + 2n_past² + n_past³, so the O(n_z³)
# covariance recursion scales as n_past⁹. That confines the filter to small
# models — see `CUBIC_KALMAN_MAX_DIMENSION` below.

# The covariance recursion is two n_z×n_z triple products per period. Beyond this
# dimension a single period costs seconds and a single matrix hundreds of MB, so
# refuse with a message that names the cause instead of appearing to hang.
const CUBIC_KALMAN_MAX_DIMENSION = 2500

# row-major flatten: rowvec(R)[(i-1)*size(R,2)+r] == R[i,r]
rowvec(R) = vec(permutedims(R))

# Probabilists' Gauss-Hermite nodes and weights via Golub-Welsch.
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
    nq11, nq12, nq111 = nPast^2, nPast^2, nPast^3
    nz = 3nr + nq11 + nq12 + nq111

    if nz > CUBIC_KALMAN_MAX_DIMENSION
        error("The cubic Kalman filter needs an augmented state of dimension " *
              "$nz (= 3·$nr + 2·$(nPast)² + $(nPast)³) for this model, and its " *
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

    return (; nr, nPast, nExo, na, nz, oas, S1, S2, S3, Pm, C,
            r1, r2, r3, i11, i12, i111, nq11, nq12, nq111,
            M, mc, V, B2, Wq, Wl_t, Bc, MM, ntail)
end

"""
One step of the augmented map, `z ↦ f(z, ε)`. Affine in `z` by construction:
every product of two `z`-dependent quantities is read off an existing block
rather than recomputed.
"""
function cubic_kalman_step(sys, z::AbstractVector, ε::AbstractVector)
    (; nPast, nExo, na, S1, S2, S3, Pm, r1, r2, r3, i11, i12, i111,
       M, mc, V, Wq, Wl_t, Bc, MM, ntail) = sys
    a = Pm * view(z, r1)
    b = Pm * view(z, r2)
    p = Pm * view(z, r3)
    q11 = collect(view(z, i11))
    q12 = collect(view(z, i12))
    q111 = collect(view(z, i111))

    tail = vcat(one(eltype(ε)), ε)
    aug1 = vcat(a, 1.0, ε)
    aug1h = vcat(a, 0.0, ε)
    aug2 = vcat(b, 0.0, zeros(nExo))
    aug3 = vcat(p, 0.0, zeros(nExo))

    # Kronecker inputs, with the all-past blocks read from the state.
    K2 = Vector{Float64}(undef, na * na)
    @inbounds for i in 1:na, j in 1:na
        K2[(i-1)*na+j] = (i <= nPast && j <= nPast) ? q11[(i-1)*nPast+j] : aug1[i] * aug1[j]
    end
    K12 = zeros(na * na)
    @inbounds for i in 1:na, j in 1:nPast
        K12[(i-1)*na+j] = (i <= nPast) ? q12[(i-1)*nPast+j] : aug1h[i] * aug2[j]
    end
    K3 = Vector{Float64}(undef, na * na * na)
    @inbounds for i in 1:na, j in 1:na, k in 1:na
        r = ((i-1)*na + (j-1)) * na + k
        ci = i <= nPast; cj = j <= nPast; ck = k <= nPast
        n = ci + cj + ck
        K3[r] = if n == 3
            q111[((i-1)*nPast + (j-1))*nPast + k]
        elseif n == 2
            if ci && cj
                q11[(i-1)*nPast+j] * aug1[k]
            elseif ci && ck
                q11[(i-1)*nPast+k] * aug1[j]
            else
                q11[(j-1)*nPast+k] * aug1[i]
            end
        else
            aug1[i] * aug1[j] * aug1[k]
        end
    end

    x1n = S1 * aug1
    x2n = S1 * aug2 + S2 * K2 / 2
    x3n = S1 * aug3 + S2 * K12 + S3 * K3 / 6

    # New Kronecker blocks, kept affine in z.
    u = M * a                      # z-dependent, linear in a
    v = mc + V * ε                 # z-independent
    bn = Pm * x2n                  # affine in z
    Q11 = Matrix(reshape(q11, nPast, nPast)')
    Q12 = Matrix(reshape(q12, nPast, nPast)')
    Q111 = Matrix(reshape(q111, nPast * nPast, nPast)')

    t2 = rowvec(M * Q11 * M')      # = u⊗u = (M⊗M) q₁₁
    q11n = t2 + ℒ.kron(u, v) + ℒ.kron(v, u) + ℒ.kron(v, v)

    Wl = zeros(nPast, nPast)
    for t in 1:ntail
        Wl .+= tail[t] .* Wl_t[t]
    end
    wc = Bc * ℒ.kron(tail, tail)
    q12n = rowvec(M * Q12 * M') +      # u⊗(M b)    = (M⊗M) q₁₂
           rowvec(M * Q111 * Wq') +    # u⊗(Wq q₁₁) = (M⊗Wq) q₁₁₁
           rowvec(M * Q11 * Wl') +     # u⊗(Wl a)   = (M⊗Wl) q₁₁
           ℒ.kron(u, wc) + ℒ.kron(v, bn)

    vv = ℒ.kron(v, v)
    q111n = rowvec(M * Q111 * MM') +          # u⊗u⊗u
            ℒ.kron(t2, v) + ℒ.kron(v, t2) +   # u⊗u⊗v , v⊗u⊗u
            ℒ.kron(u, vv) + ℒ.kron(vv, u) +   # u⊗v⊗v , v⊗v⊗u
            ℒ.kron(vv, v)                     # v⊗v⊗v
    @inbounds for i in 1:nPast, j in 1:nPast, k in 1:nPast
        r = ((i-1)*nPast + (j-1)) * nPast + k
        q111n[r] += t2[(i-1)*nPast+k] * v[j]  # u⊗v⊗u
        q111n[r] += v[i] * u[j] * v[k]        # v⊗u⊗v
    end

    return vcat(x1n, x2n, x3n, q11n, q12n, q111n)
end

# E[f(z,·)] and Var(f(z,·)) under ε ~ N(0,I), exactly.
function cubic_kalman_moments(sys, z, nodes, wts)
    m = zeros(sys.nz)
    S = zeros(sys.nz, sys.nz)
    for (ε, w) in zip(nodes, wts)
        fz = cubic_kalman_step(sys, z, ε)
        m .+= w .* fz
        ℒ.mul!(S, fz, fz', w, one(w))
    end
    S .-= m * m'
    return m, (S + S') / 2
end

# The step is affine, so the transition matrix and drift are recovered exactly
# from evaluations at the origin and at each basis vector.
function build_cubic_kalman_transition(sys, nodes, wts)
    c, _ = cubic_kalman_moments(sys, zeros(sys.nz), nodes, wts)
    𝒜 = zeros(sys.nz, sys.nz)
    e = zeros(sys.nz)
    for j in 1:sys.nz
        fill!(e, 0.0)
        e[j] = 1.0
        mj, _ = cubic_kalman_moments(sys, e, nodes, wts)
        𝒜[:, j] = mj - c
    end
    return 𝒜, c
end

"""
Kalman recursion on the cubic augmented state. Mirrors `run_quadratic_kalman`:
the noise covariance is rebuilt from the current state estimate every period,
because it depends on the state exactly as `G(z)G(z)'` does at second order.
"""
function run_cubic_kalman(sys, data_in_deviations::AbstractMatrix{<:Real};
                          measurement_error::Union{Nothing,AbstractVector{<:Real},AbstractMatrix{<:Real}} = nothing,
                          presample_periods::Int = 0,
                          on_failure_loglikelihood::Real = -Inf,
                          quadrature_points::Int = 4,
                          workspaces = nothing,
                          lyapunov_algorithm::Symbol = :doubling)
    nz, C = sys.nz, sys.C
    n_obs, nT = size(data_in_deviations)
    presample_periods = normalize_presample_periods(presample_periods, nT)

    Hm = if measurement_error === nothing
        zeros(n_obs, n_obs)
    elseif measurement_error isa AbstractMatrix
        Matrix{Float64}(measurement_error)
    else
        Matrix{Float64}(ℒ.Diagonal(collect(measurement_error)))
    end

    nodes, wts = gauss_hermite_tensor(sys.nExo, quadrature_points)
    𝒜, c = build_cubic_kalman_transition(sys, nodes, wts)

    z = (Matrix{Float64}(ℒ.I(nz)) - 𝒜) \ c
    _, Q̄ = cubic_kalman_moments(sys, z, nodes, wts)
    Σ = qkf_lyapunov(𝒜, Q̄; workspaces = workspaces, lyapunov_algorithm = lyapunov_algorithm)

    ll = 0.0
    for t in 1:nT
        _, Q = cubic_kalman_moments(sys, z, nodes, wts)
        zp = 𝒜 * z + c
        Pp = 𝒜 * Σ * 𝒜' + Q
        Pp = (Pp + Pp') / 2

        v = data_in_deviations[:, t] - C * zp
        F = C * Pp * C' + Hm
        F = (F + F') / 2

        Fc = ℒ.cholesky(F, check = false)
        ℒ.issuccess(Fc) || return on_failure_loglikelihood

        if t > presample_periods
            ll -= 0.5 * (ℒ.dot(v, Fc \ v) + ℒ.logdet(Fc) + n_obs * log(2π))
            isfinite(ll) || return on_failure_loglikelihood
        end

        Kg = Pp * C' / Fc
        z = zp + Kg * v
        Σ = Pp - Kg * C * Pp
        Σ = (Σ + Σ') / 2
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
