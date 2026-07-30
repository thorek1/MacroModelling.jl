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
function build_quadratic_kalman_system(𝓂::ℳ, 𝐒₁, 𝐒₂, observables_index::Vector{Int})
    T = 𝓂.constants.post_model_macro
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
    A1   = S1 * Ea[:, 1:nPast] * P

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

    return (; nVars, nr, oas, nPast, nExo, na, nq, nz, past, P, S, Ea, S1, S2, PS1, V, Dp, Lp,
              𝒜, c, 𝒞, QH, G1 = S1 * S, r1)
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
    Tv = promote_type(eltype(𝒜), eltype(Y), eltype(g0))
    z = copy(z0); Pc = copy(Σ0)
    ll = zero(Tv); log2pi = log(2π)
    for t in 1:nT
        G = reshape(g0 + Λ * (Pz * z), nz, nExo)
        Q = G * G' + QH
        zp = 𝒜 * z + c
        Pp = 𝒜 * Pc * 𝒜' + Q; Pp = (Pp + Pp') / 2
        v  = view(Y, :, t) - 𝒞 * zp
        CP = 𝒞 * Pp
        F  = CP * 𝒞' + Hm; F = (F + F') / 2
        Fc = ℒ.cholesky(F, check = false)
        ℒ.issuccess(Fc) || return Tv(on_failure_loglikelihood)
        if t > presample_periods
            ll -= 0.5 * (ℒ.dot(v, Fc \ v) + ℒ.logdet(Fc) + n_obs * log2pi)
            isfinite(ll) || return Tv(on_failure_loglikelihood)
        end
        K = CP' / Fc
        z = zp + K * v
        Pc = Pp - K * CP; Pc = (Pc + Pc') / 2
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

end # @stable
