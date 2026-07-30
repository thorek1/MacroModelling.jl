@stable default_mode = "disable" begin

# Quadratic Kalman filter (Monfort, Renne & Roussellet, 2015) for the pruned
# second-order solution.
#
# The idea. A pruned second-order solution is *linear* in an augmented state —
# this is the pruned state-space representation of Andreasen, Fernández-Villaverde
# & Rubio-Ramírez (2018). Writing the package's own recursion,
#
#   aug₁ = [x₁ₜ₋₁[past]; 1; εₜ]
#   x₁ₜ  = 𝐒₁ aug₁
#   x₂ₜ  = 𝐒₁ [x₂ₜ₋₁[past]; 0; 0] + ½ 𝐒₂ (aug₁ ⊗ aug₁)
#
# and stacking
#
#   z ₜ = [ x₁ₜ ; x₂ₜ ; x₁ₜ[past] ⊗ x₁ₜ[past] ]
#
# every block above becomes affine in zₜ₋₁, because aug₁ ⊗ aug₁ expands into terms
# that are quadratic in x₁ₜ₋₁[past] (carried by the third block), linear in it, or
# constant. The observation is a plain selection, yₜ = (x₁ₜ + x₂ₜ)[observables],
# so the whole system is linear and a Kalman filter applies.
#
# What is exact and what is not. The transition is *exactly* linear in z — no
# approximation — and the conditional first and second moments of the innovation
# are computed in closed form (below). What the filter approximates is the
# conditional *distribution*: the innovation is quadratic in εₜ and therefore not
# Gaussian, so the Kalman recursion delivers the best **linear** projection rather
# than the exact conditional mean. It also treats the third block as a free state
# rather than enforcing that it equals the Kronecker square of the first, which is
# what makes the filter linear in the first place. On a linear model (𝐒₂ = 0) both
# approximations vanish and the filter reproduces the Kalman likelihood exactly —
# that is the correctness test in `test/test_quadratic_kalman.jl`.
#
# The innovation. With aug₁ = ā + Sε, where ā = [x₁ₜ₋₁[past]; 1; 0] collects the
# predictable part and S selects the shocks, every block of the innovation has the
# form
#
#   w = G ε + H (ε⊗ε − vec(I)),
#
# linear plus centred-quadratic in ε. Because the Gaussian third moment vanishes
# the two parts are uncorrelated, so
#
#   Var(w) = G G' + H (I + K) H',      K the commutation matrix,
#
# using E[(ε⊗ε)(ε⊗ε)'] = vec(I)vec(I)' + I + K. `H` is constant; `G` depends on the
# state, and is evaluated at the filtered mean each period — the defining choice of
# the quadratic Kalman filter.
#
# Cost. The augmented state has dimension 2·nVars + nPast², which is 808 for
# Smets-Wouters (2007). The covariance recursion is therefore O(nz³) per period and
# dominates everything else; expect seconds rather than milliseconds per likelihood.


# Duplication/elimination for the Kronecker block. q = x₁ₚ ⊗ x₁ₚ = vec(x₁ₚx₁ₚ') is
# symmetric, so only nPast(nPast+1)/2 of its nPast² entries are distinct. Carrying
# vech(x₁ₚx₁ₚ') instead of vec cuts the augmented dimension — on Smets-Wouters from
# 808 to 483 — and the covariance recursion is O(nz³), so that is roughly a 4.7×
# saving. `D` maps vech ↦ vec and `L` vec ↦ vech, with L*D = I.
function duplication_elimination(n::Int)
    ns = n * (n + 1) ÷ 2
    D = spzeros(n * n, ns)
    L = spzeros(ns, n * n)
    k = 0
    @inbounds for j in 1:n, i in j:n          # column-major lower triangle
        k += 1
        D[(j - 1) * n + i, k] = 1.0
        D[(i - 1) * n + j, k] = 1.0           # symmetric partner (same entry if i==j)
        L[k, (j - 1) * n + i] = 1.0
    end
    return D, L
end

# Commutation matrix K with K vec(A) = vec(A'), for A of size n×n.
function commutation_matrix(n::Int)
    K = spzeros(n * n, n * n)
    @inbounds for i in 1:n, j in 1:n
        K[(i - 1) * n + j, (j - 1) * n + i] = 1.0
    end
    return K
end

"""
Build the augmented linear state-space representation of the pruned second-order
solution, together with the pieces needed for the state-dependent innovation
covariance. `𝐒₁`/`𝐒₂` are the expanded solution matrices as returned by
`get_relevant_steady_state_and_state_update(Val(:pruned_second_order), …)`.
"""
build_quadratic_kalman_system(𝓂::ℳ, 𝐒₁, 𝐒₂, oi::Vector{Int}) =
    build_quadratic_kalman_system_from_constants(𝓂.constants, 𝐒₁, 𝐒₂, oi)

function build_quadratic_kalman_system_from_constants(cons, 𝐒₁, 𝐒₂, observables_index::Vector{Int})
    T = cons.post_model_macro
    nVars, nPast, nExo = T.nVars, T.nPast_not_future_and_mixed, T.nExo
    past = T.past_not_future_and_mixed_idx

    na = nPast + 1 + nExo                  # length of aug₁
    nq = nPast * (nPast + 1) ÷ 2           # compressed Kronecker block (vech, not vec)
    nz = 0                                 # set below, once nr is known
    Dp, Lp = duplication_elimination(nPast)

    # Keep the element type of the solution matrices so ForwardDiff duals flow
    # through: the selection matrices below stay Float64 and promote on contact.
    # Only the past states (needed by the transition) and the observables (needed
    # by the measurement) are ever read out of the x₁/x₂ blocks, so carry just
    # those rows instead of all nVars. On Smets-Wouters that is 34 rows rather
    # than 67, and the covariance recursion is cubic in the total dimension.
    oas = sort(union(past, observables_index))
    nr  = length(oas)
    pos = Dict(v => i for (i, v) in enumerate(oas))

    nz  = 2nr + nq                         # augmented state dimension
    S1 = Matrix(𝐒₁)[oas, :]
    S2 = Matrix(𝐒₂)[oas, :]
    Tv = promote_type(eltype(S1), eltype(S2))

    # past-state selection (within the retained rows), shock selection, and ā = Ea·[x₁ₚ; 1]
    P = zeros(nPast, nr)
    @inbounds for (i, j) in enumerate(past); P[i, pos[j]] = 1.0; end
    S = zeros(na, nExo); S[nPast+2:end, :] = ℒ.I(nExo)
    Ea = zeros(na, nPast + 1); Ea[1:nPast, 1:nPast] = ℒ.I(nPast); Ea[nPast+1, nPast+1] = 1.0

    # ā ⊗ ā = Eaa · [q; x₁ₚ; 1] — the structural identity that closes the system
    Eaa = spzeros(na * na, nPast^2 + nPast + 1)
    @inbounds for i in 1:na, j in 1:na
        r = (i - 1) * na + j
        if i <= nPast && j <= nPast
            Eaa[r, (i - 1) * nPast + j] = 1.0
        elseif i <= nPast && j == nPast + 1
            Eaa[r, nPast^2 + i] = 1.0
        elseif i == nPast + 1 && j <= nPast
            Eaa[r, nPast^2 + j] = 1.0
        elseif i == nPast + 1 && j == nPast + 1
            Eaa[r, nPast^2 + nPast + 1] = 1.0
        end
    end
    Eq = Eaa[:, 1:nPast^2] * Dp                # consume vech instead of vec
    Ep = Eaa[:, nPast^2+1:nPast^2+nPast]
    E1 = Eaa[:, nPast^2+nPast+1]

    PS1  = P * S1
    V    = PS1 * S
    SS   = ℒ.kron(S, S)
    vecI = vec(Matrix{Float64}(ℒ.I(nExo)))
    PP   = ℒ.kron(PS1, PS1)
    EaP  = Ea[:, 1:nPast] * P
    A1   = S1 * EaP

    r1, r2, rq = 1:nr, nr+1:2nr, 2nr+1:nz

    𝒜 = zeros(Tv, nz, nz)
    c = zeros(Tv, nz)
    𝒜[r1, r1] = A1;                     c[r1] = S1 * Ea[:, nPast+1]
    𝒜[r2, r2] = A1
    𝒜[r2, rq] = S2 * Eq / 2
    𝒜[r2, r1] = S2 * Ep * P / 2;        c[r2] = S2 * (E1 + SS * vecI) / 2
    𝒜[rq, rq] = Lp * (PP * Eq)
    𝒜[rq, r1] = Lp * (PP * Ep * P);     c[rq] = Lp * (PP * E1 + ℒ.kron(V, V) * vecI)

    𝒞 = zeros(length(observables_index), nz)
    @inbounds for (i, j) in enumerate(observables_index)
        𝒞[i, pos[j]] = 1.0            # x₁ block
        𝒞[i, nr + pos[j]] = 1.0       # x₂ block
    end

    # constant (state-independent) part of the innovation covariance
    Hq = [zeros(Tv, nr, nExo^2); S2 * SS / 2; Lp * ℒ.kron(V, V)]
    IK = Matrix{Float64}(ℒ.I(nExo^2)) + Matrix(commutation_matrix(nExo))
    QH = Hq * IK * Hq'
    QH = (QH + QH') / 2

    # The constant blocks are returned as well: the reverse-mode rule needs them
    # to push cotangents from (𝒜, c, QH, g₀, Λ) back onto 𝐒₁ and 𝐒₂.
    return (; nVars, nr, oas, nPast, nExo, na, nq, nz, past, P, S, Ea, S1, S2, PS1, V, Dp, Lp,
              𝒜, c, 𝒞, QH, G1 = S1 * S, r1,
              r2 = nr+1:2nr, rq = 2nr+1:nz,
              Eq, Ep, E1, SS, vecI, IK, EaP, Ea1 = Ea[:, nPast+1], EpP = Ep * P,
              PP, KVV = ℒ.kron(V, V), Hq)
end

# State-dependent loading of the linear-in-ε part of the innovation, at state z.
function quadratic_kalman_G(sys, z::AbstractVector{<:Real})
    ā = sys.Ea * vcat(sys.P * view(z, sys.r1), one(eltype(z)))
    ū = sys.PS1 * ā
    G2 = sys.S2 * (ℒ.kron(ā, sys.S) + ℒ.kron(sys.S, ā)) / 2
    Gq = sys.Lp * (ℒ.kron(ū, sys.V) + ℒ.kron(sys.V, ū))
    return vcat(sys.G1, G2, Gq)
end


# G(z) is *affine* in z: both the 𝐒₂ block and the Kronecker block are linear in
# ā, which is affine in z. So vec(G(z)) = g₀ + Λ·(P z₍ₓ₁₎), and the whole
# state-dependence of the innovation covariance collapses to one matrix. Building
# Λ column by column from the affine map is exact (not a finite difference) and
# avoids hand-deriving Kronecker adjoints in the reverse pass.
function quadratic_kalman_affine_G(sys)
    nz, nExo, nPast = sys.nz, sys.nExo, sys.nPast
    z0 = zeros(eltype(sys.𝒜), nz)
    g0 = vec(quadratic_kalman_G(sys, z0))
    Λ = similar(g0, length(g0), nPast)
    e = zeros(eltype(sys.𝒜), nz)
    @inbounds for i in 1:nPast
        fill!(e, zero(eltype(e)))
        # P selects past rows out of the x₁ block, so the i-th past coordinate is
        # the row of P with a one in it
        j = findfirst(!iszero, view(sys.P, i, :))
        e[j] = one(eltype(e))
        Λ[:, i] = vec(quadratic_kalman_G(sys, e)) - g0
    end
    return g0, Λ
end

"""
The quadratic Kalman recursion, given the augmented system in the form the
reverse-mode rule needs. Split out from `run_quadratic_kalman` so that the part
that scales with the sample length — and dominates the cost at O(T·nz³) — carries
a hand-written adjoint, while the one-off construction of the system matrices is
left to ordinary AD.
"""
function quadratic_kalman_recursion(𝒜, c, QH, g0, Λ, Hm, Y, 𝒞, Pz, z0, Σ0, nz::Int, nExo::Int,
                                    presample_periods::Int, on_failure_loglikelihood::Real)
    n_obs, nT = size(Y)
    # Promote over every differentiable input, not just a few: the preallocated
    # buffers below fix the element type, so missing one makes forward-mode AD
    # fail with respect to exactly that argument.
    Tv = promote_type(eltype(𝒜), eltype(c), eltype(QH), eltype(g0), eltype(Λ),
                      eltype(Hm), eltype(Y), eltype(z0), eltype(Σ0))

    # 𝒞 is a selection: row i picks the x₁ and x₂ entries of observable i. Doing
    # that by indexing rather than by three gemms with a 0/1 matrix removes the
    # only dense products that scale with n_obs·nz².
    p1 = [findfirst(!iszero, view(𝒞, i, :)) for i in 1:n_obs]
    p2 = [findlast(!iszero,  view(𝒞, i, :)) for i in 1:n_obs]

    # Preallocate once per call. The naive version allocated ~18 MB per period,
    # which cost more than the covariance propagation it was feeding.
    Pc = Matrix{Tv}(undef, nz, nz); copyto!(Pc, Σ0)
    Pp = Matrix{Tv}(undef, nz, nz)
    Tm = Matrix{Tv}(undef, nz, nz)
    z  = Vector{Tv}(undef, nz); copyto!(z, z0)
    zp = Vector{Tv}(undef, nz)
    gv = Vector{Tv}(undef, nz * nExo)
    x1p = Vector{Tv}(undef, size(Pz, 1))
    CP = Matrix{Tv}(undef, n_obs, nz)
    F  = Matrix{Tv}(undef, n_obs, n_obs)
    Kg = Matrix{Tv}(undef, nz, n_obs)
    v  = Vector{Tv}(undef, n_obs)
    Fv = Vector{Tv}(undef, n_obs)

    ll = zero(Tv); log2pi = log(2π)

    @inbounds for t in 1:nT
        # G(z) = reshape(g₀ + Λ(Pz z))
        ℒ.mul!(x1p, Pz, z)
        copyto!(gv, g0); ℒ.mul!(gv, Λ, x1p, one(Tv), one(Tv))
        G = reshape(gv, nz, nExo)

        # Pp = 𝒜 Pc 𝒜' + G G' + QH   (the rank-nExo term as one gemm update)
        ℒ.mul!(Tm, 𝒜, Pc)
        ℒ.mul!(Pp, Tm, 𝒜')
        Pp .+= QH
        ℒ.mul!(Pp, G, G', one(Tv), one(Tv))
        for j in 1:nz, i in 1:j
            m = (Pp[i, j] + Pp[j, i]) / 2; Pp[i, j] = m; Pp[j, i] = m
        end

        ℒ.mul!(zp, 𝒜, z); zp .+= c

        for i in 1:n_obs
            v[i] = Y[i, t] - (zp[p1[i]] + zp[p2[i]])
            for k in 1:nz
                CP[i, k] = Pp[p1[i], k] + Pp[p2[i], k]
            end
        end
        for i in 1:n_obs, j in 1:n_obs
            F[i, j] = CP[i, p1[j]] + CP[i, p2[j]] + Hm[i, j]
        end
        for i in 1:n_obs, j in 1:i-1
            m = (F[i, j] + F[j, i]) / 2; F[i, j] = m; F[j, i] = m
        end

        Fc = ℒ.cholesky(F, check = false)
        ℒ.issuccess(Fc) || return Tv(on_failure_loglikelihood)

        if t > presample_periods
            copyto!(Fv, v); ℒ.ldiv!(Fc, Fv)
            ll -= 0.5 * (ℒ.dot(v, Fv) + ℒ.logdet(Fc) + n_obs * log2pi)
            isfinite(ll) || return Tv(on_failure_loglikelihood)
        end

        # K = CP' F⁻¹ ; z = zp + K v ; Pc = Pp − K CP
        copyto!(Kg, CP'); ℒ.rdiv!(Kg, Fc)
        copyto!(z, zp); ℒ.mul!(z, Kg, v, one(Tv), one(Tv))
        copyto!(Pc, Pp); ℒ.mul!(Pc, Kg, CP, -one(Tv), one(Tv))
        for j in 1:nz, i in 1:j
            m = (Pc[i, j] + Pc[j, i]) / 2; Pc[i, j] = m; Pc[j, i] = m
        end
    end

    return ll
end

# Hand-written reverse mode. Every cotangent is verified against ForwardDiff to
# machine precision in test/test_quadratic_kalman.jl. Note the forward pass
# symmetrises F, so the cotangent reaching CP and Hm is (F̄+F̄')/2 — omitting that
# leaves d/dHm wrong by ~2% while every other derivative still looks exact.
function rrule(::typeof(quadratic_kalman_recursion), 𝒜, c, QH, g0, Λ, Hm, Y, 𝒞, Pz, z0, Σ0,
               nz::Int, nExo::Int, presample_periods::Int, on_failure_loglikelihood::Real)
    n_obs, nT = size(Y)
    zs = Vector{Vector{Float64}}(undef, nT); Ps = Vector{Matrix{Float64}}(undef, nT)
    Gs = Vector{Matrix{Float64}}(undef, nT); vs = Vector{Vector{Float64}}(undef, nT)
    CPs = Vector{Matrix{Float64}}(undef, nT); Fis = Vector{Matrix{Float64}}(undef, nT)
    Ks = Vector{Matrix{Float64}}(undef, nT)
    z = copy(z0); Pc = copy(Σ0); ll = 0.0; log2pi = log(2π); failed = false
    for t in 1:nT
        zs[t] = copy(z); Ps[t] = copy(Pc)
        G = reshape(g0 + Λ * (Pz * z), nz, nExo); Gs[t] = G
        Q = G * G' + QH
        zp = 𝒜 * z + c
        Pp = 𝒜 * Pc * 𝒜' + Q; Pp = (Pp + Pp') / 2
        v = Y[:, t] - 𝒞 * zp; vs[t] = v
        CP = 𝒞 * Pp; CPs[t] = CP
        F = CP * 𝒞' + Hm; F = (F + F') / 2
        Fc = ℒ.cholesky(F, check = false)
        if !ℒ.issuccess(Fc); failed = true; break; end
        Fi = inv(Fc); Fis[t] = Fi
        t > presample_periods && (ll -= 0.5 * (ℒ.dot(v, Fi * v) + ℒ.logdet(Fc) + n_obs * log2pi))
        K = CP' * Fi; Ks[t] = K
        z = zp + K * v; Pc = Pp - K * CP; Pc = (Pc + Pc') / 2
    end

    if failed || !isfinite(ll)
        nt = ntuple(_ -> NoTangent(), 15)
        return Float64(on_failure_loglikelihood), _ -> nt
    end

    function quadratic_kalman_recursion_pullback(∂ll_bar)
        ∂ll = unthunk(∂ll_bar)
        𝒜̄ = zeros(nz, nz); c̄ = zeros(nz); Q̄H = zeros(nz, nz)
        ḡ0 = zeros(length(g0)); Λ̄ = zeros(size(Λ)); H̄m = zeros(n_obs, n_obs)
        Ȳ = zeros(size(Y)); z̄ = zeros(nz); P̄ = zeros(nz, nz)
        for t in nT:-1:1
            z_, P_, G, v, CP, Fi, K = zs[t], Ps[t], Gs[t], vs[t], CPs[t], Fis[t], Ks[t]
            P̄p = copy(P̄)
            K̄  = -P̄ * CP'
            C̄P = -K' * P̄
            z̄p = copy(z̄)
            K̄ .+= z̄ * v'
            v̄  = K' * z̄
            C̄P .+= Fi * K̄'
            F̄  = -Fi * (CP * K̄) * Fi
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
            P̄  = 𝒜' * P̄p * 𝒜
            Q̄  = P̄p
            Ḡ  = 2 .* (Q̄ * G)
            Q̄H .+= Q̄
            vḠ = vec(Ḡ)
            ḡ0 .+= vḠ
            Λ̄  .+= vḠ * (Pz * z_)'
            𝒜̄ .+= z̄p * z_'
            c̄  .+= z̄p
            z̄   = 𝒜' * z̄p + Pz' * (Λ' * vḠ)
        end
        return (NoTangent(), 𝒜̄, c̄, Q̄H, ḡ0, Λ̄, H̄m, Ȳ, NoTangent(), NoTangent(),
                z̄, P̄, NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    return ll, quadratic_kalman_recursion_pullback
end

"""
Run the quadratic Kalman filter and return the loglikelihood. `data_in_deviations`
holds the observables as deviations from the non-stochastic steady state (rows in
the same order as `observables_index` used to build `sys`).

The filter is initialised at the ergodic mean and covariance of the augmented
system. The mean solves `(I − 𝒜)z̄ = c` directly and the covariance
`Σ = 𝒜Σ𝒜' + Q̄` by doubling — iterating either would need thousands of steps on a
model with roots near unity.
"""
function run_quadratic_kalman(sys,
                              data_in_deviations::AbstractMatrix{<:Real};
                              measurement_error::Union{Nothing,AbstractVector{<:Real},AbstractMatrix{<:Real}} = nothing,
                              presample_periods::Int = 0,
                              on_failure_loglikelihood::Real = -Inf)
    nz = sys.nz
    𝒜, c, 𝒞, QH = sys.𝒜, sys.c, sys.𝒞, sys.QH
    n_obs, nT = size(data_in_deviations)
    presample_periods = normalize_presample_periods(presample_periods, nT)

    Tv = promote_type(eltype(𝒜), eltype(data_in_deviations),
                      measurement_error === nothing ? Float64 : eltype(measurement_error))

    Hm = if measurement_error === nothing
        zeros(Tv, n_obs, n_obs)
    elseif measurement_error isa AbstractMatrix
        Matrix{Tv}(measurement_error)
    else
        Matrix{Tv}(ℒ.Diagonal(collect(measurement_error)))
    end

    z̄ = (Matrix{Tv}(ℒ.I(nz)) - 𝒜) \ c
    Gbar = quadratic_kalman_G(sys, z̄)
    Q̄ = Gbar * Gbar' + QH
    Q̄ = (Q̄ + Q̄') / 2

    Σ = copy(Q̄)
    Ak = copy(𝒜)
    for _ in 1:60
        Σn = Ak * Σ * Ak' + Σ
        Σn = (Σn + Σn') / 2
        if maximum(abs, Σn - Σ) < 1e-12 * max(1.0, maximum(abs, Σn))
            Σ = Σn
            break
        end
        Σ = Σn
        Ak = Ak * Ak
        maximum(abs, Ak) < 1e-14 && break
    end

    # Hand off to the taped recursion, which carries the hand-written adjoint.
    g0, Λ = quadratic_kalman_affine_G(sys)
    Pz = sys.P * [Matrix{Tv}(ℒ.I(sys.nr)) zeros(Tv, sys.nr, nz - sys.nr)]

    return quadratic_kalman_recursion(𝒜, c, QH, g0, Λ, Hm, Matrix(data_in_deviations),
                                      𝒞, Pz, z̄, Σ, nz, sys.nExo,
                                      presample_periods, on_failure_loglikelihood)
end



# Adjoints of kron(A,B) with respect to each factor.
function kron_adjoint_A(M, B, m, n, p, q)
    A = zeros(eltype(M), m, n)
    @inbounds for i in 1:m, j in 1:n
        A[i, j] = sum(view(M, (i-1)*p+1:i*p, (j-1)*q+1:j*q) .* B)
    end
    return A
end
function kron_adjoint_B(M, A, m, n, p, q)
    B = zeros(eltype(M), p, q)
    @inbounds for i in 1:m, j in 1:n
        @views B .+= A[i, j] .* M[(i-1)*p+1:i*p, (j-1)*q+1:j*q]
    end
    return B
end

# Discrete Lyapunov X = A X A' + Q by doubling.
function qkf_lyapunov(A, Q; iters::Int = 80)
    X = copy(Q); Ak = copy(A)
    for _ in 1:iters
        Xn = Ak * X * Ak' + X; Xn = (Xn + Xn') / 2
        if maximum(abs, Xn - X) < 1e-15 * max(1.0, maximum(abs, Xn)); X = Xn; break; end
        X = Xn; Ak = Ak * Ak
        maximum(abs, Ak) < 1e-16 && break
    end
    return (X + X') / 2
end

qkf_Pz(sys) = sys.P * [Matrix{Float64}(ℒ.I(sys.nr)) zeros(sys.nr, sys.nz - sys.nr)]

"""
Push the cotangents of the augmented system back onto the solution matrices.
Covers the build, the ergodic initialisation (including the Lyapunov adjoint) and
the recursion. Verified against ForwardDiff to ~1e-15 in the test suite.
"""
function quadratic_kalman_pullback(sys, data_in_deviations, Hm, presample_periods, ∂ll)
    nr, nP, nE, na, nz = sys.nr, sys.nPast, sys.nExo, sys.na, sys.nz
    r1, r2, rq = sys.r1, sys.r2, sys.rq
    Lp = Matrix(sys.Lp); Eq = Matrix(sys.Eq); Ep = Matrix(sys.Ep); E1 = Vector(sys.E1)
    g0, Λ = quadratic_kalman_affine_G(sys)
    Pz = qkf_Pz(sys)

    z0 = (Matrix{Float64}(ℒ.I(nz)) - sys.𝒜) \ sys.c
    G0 = reshape(g0 + Λ * (Pz * z0), nz, nE)
    Q0 = G0 * G0' + sys.QH; Q0 = (Q0 + Q0') / 2
    Σ0 = qkf_lyapunov(sys.𝒜, Q0)

    R = last(rrule(quadratic_kalman_recursion, sys.𝒜, sys.c, sys.QH, g0, Λ, Hm,
                   Matrix(data_in_deviations), sys.𝒞, Pz, z0, Σ0, nz, nE,
                   presample_periods, -Inf))(∂ll)
    𝒜̄ = copy(R[2]); c̄ = copy(R[3]); Q̄H = copy(R[4])
    ḡ0 = copy(R[5]); Λ̄ = copy(R[6]); Ȳ = copy(R[8])
    z̄0 = copy(R[11]); Σ̄0 = copy(R[12])

    # Σ0 = 𝒜Σ0𝒜' + Q0  ⇒  X solves X = 𝒜'X𝒜 + Σ̄0
    X = qkf_lyapunov(Matrix(sys.𝒜'), (Σ̄0 + Σ̄0') / 2)
    𝒜̄ .+= 2 .* (X * sys.𝒜 * Σ0)
    Q̄0 = (X + X') / 2
    Ḡ0 = 2 .* (Q̄0 * G0); Q̄H .+= Q̄0
    vG0 = vec(Ḡ0); ḡ0 .+= vG0; Λ̄ .+= vG0 * (Pz * z0)'
    z̄0 .+= Pz' * (Λ' * vG0)
    λ = (Matrix{Float64}(ℒ.I(nz)) - sys.𝒜)' \ z̄0
    c̄ .+= λ; 𝒜̄ .+= λ * z0'

    S̄1 = zeros(size(sys.S1)); S̄2 = zeros(size(sys.S2))
    P̄S1 = zeros(size(sys.PS1)); V̄ = zeros(size(sys.V))
    P̄P = zeros(size(sys.PP)); K̄VV = zeros(size(sys.KVV))

    Ā1 = 𝒜̄[r1, r1] + 𝒜̄[r2, r2]
    S̄1 .+= Ā1 * sys.EaP'
    S̄2 .+= 𝒜̄[r2, rq] * Eq' / 2 + 𝒜̄[r2, r1] * sys.EpP' / 2
    P̄P .+= Lp' * 𝒜̄[rq, rq] * Eq' + Lp' * 𝒜̄[rq, r1] * sys.EpP'
    S̄1 .+= c̄[r1] * sys.Ea1'
    S̄2 .+= c̄[r2] * (E1 + sys.SS * sys.vecI)' / 2
    lc = Lp' * c̄[rq]
    P̄P .+= lc * E1'; K̄VV .+= lc * sys.vecI'

    Rq = (Q̄H + Q̄H') / 2
    H̄q = 2 .* (Rq * sys.Hq * sys.IK)
    S̄2 .+= H̄q[nr+1:2nr, :] * (sys.SS / 2)'
    K̄VV .+= Lp' * H̄q[2nr+1:end, :]

    function absorb_G!(Ḡ, z)
        ā = sys.Ea * vcat(sys.P * view(z, r1), 1.0)
        ū = sys.PS1 * ā
        Ma = ℒ.kron(ā, sys.S) + ℒ.kron(sys.S, ā)
        S̄1 .+= Ḡ[r1, :] * sys.S'
        S̄2 .+= Ḡ[r2, :] * Ma' / 2
        Gq = Lp' * Ḡ[rq, :]
        ū̄ = vec(kron_adjoint_A(Gq, sys.V, nP, 1, nP, nE)) .+
             vec(kron_adjoint_B(Gq, sys.V, nP, nE, nP, 1))
        V̄ .+= kron_adjoint_B(Gq, reshape(ū, nP, 1), nP, 1, nP, nE) .+
               kron_adjoint_A(Gq, reshape(ū, nP, 1), nP, nE, nP, 1)
        P̄S1 .+= ū̄ * ā'
    end
    absorb_G!(reshape(ḡ0 .- vec(sum(Λ̄, dims = 2)), nz, nE), zeros(nz))
    @inbounds for i in 1:nP
        e = zeros(nz); e[findfirst(!iszero, view(sys.P, i, :))] = 1.0
        absorb_G!(reshape(Λ̄[:, i], nz, nE), e)
    end

    P̄S1 .+= kron_adjoint_A(P̄P, sys.PS1, nP, na, nP, na) .+
             kron_adjoint_B(P̄P, sys.PS1, nP, na, nP, na)
    V̄  .+= kron_adjoint_A(K̄VV, sys.V, nP, nE, nP, nE) .+
             kron_adjoint_B(K̄VV, sys.V, nP, nE, nP, nE)
    P̄S1 .+= V̄ * sys.S'
    S̄1  .+= sys.P' * P̄S1

    return S̄1, S̄2, Ȳ
end




# ── standard filter interface ────────────────────────────────────────────────
# Routing through `calculate_loglikelihood` (rather than a special branch in
# `get_loglikelihood`) is what lets the existing reverse-mode machinery reach the
# filter: the top-level rrule looks for `rrule(calculate_loglikelihood, Val(filter), …)`
# and falls back to a zero gradient when none exists.
function calculate_loglikelihood(::Val{:quadratic_kalman},
                                 ::Val{:pruned_second_order},
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
    sys = build_quadratic_kalman_system_from_constants(constants, 𝐒[1], 𝐒[2], observables_index)
    return run_quadratic_kalman(sys, data_in_deviations;
                                measurement_error = measurement_error,
                                presample_periods = presample_periods,
                                on_failure_loglikelihood = on_failure_loglikelihood)
end

function rrule(::typeof(calculate_loglikelihood),
               ::Val{:quadratic_kalman},
               ::Val{:pruned_second_order},
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
    sys = build_quadratic_kalman_system_from_constants(constants, 𝐒[1], 𝐒[2], observables_index)
    n_obs = size(data_in_deviations, 1)
    Hm = measurement_error === nothing ? zeros(n_obs, n_obs) :
         measurement_error isa AbstractMatrix ? Matrix{Float64}(measurement_error) :
         Matrix{Float64}(ℒ.Diagonal(collect(measurement_error)))
    llh = run_quadratic_kalman(sys, data_in_deviations;
                               measurement_error = measurement_error,
                               presample_periods = presample_periods,
                               on_failure_loglikelihood = on_failure_loglikelihood)

    nine(x...) = (NoTangent(), NoTangent(), NoTangent(), NoTangent(), x[1], x[2],
                  NoTangent(), x[3], NoTangent())

    if !isfinite(llh)
        return llh, _ -> nine(NoTangent(), NoTangent(), NoTangent())
    end

    function quadratic_kalman_loglikelihood_pullback(∂llh_bar)
        ∂llh = unthunk(∂llh_bar)
        S̄1r, S̄2r, Ȳ = quadratic_kalman_pullback(sys, data_in_deviations, Hm,
                                                 presample_periods, ∂llh)
        # scatter the retained rows back onto the full solution matrices
        ∂𝐒1 = zeros(size(𝐒[1])); ∂𝐒2 = zeros(size(𝐒[2]))
        ∂𝐒1[sys.oas, :] = S̄1r
        ∂𝐒2[sys.oas, :] = S̄2r
        ∂state = [zeros(length(s)) for s in state]
        return nine([∂𝐒1, ∂𝐒2], Ȳ, ∂state)
    end

    return llh, quadratic_kalman_loglikelihood_pullback
end


end # @stable
