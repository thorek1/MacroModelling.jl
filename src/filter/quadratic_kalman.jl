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

    na = nPast + 1 + nExo          # length of aug₁
    nq = nPast^2                   # length of the Kronecker block
    nz = 2nVars + nq               # augmented state dimension

    S1 = Matrix{Float64}(𝐒₁)
    S2 = Matrix{Float64}(𝐒₂)

    # past-state selection, shock selection, and the embedding ā = Ea·[x₁ₚ; 1]
    P = zeros(nPast, nVars)
    @inbounds for (i, j) in enumerate(past); P[i, j] = 1.0; end
    S = zeros(na, nExo); S[nPast+2:end, :] = ℒ.I(nExo)
    Ea = zeros(na, nPast + 1); Ea[1:nPast, 1:nPast] = ℒ.I(nPast); Ea[nPast+1, nPast+1] = 1.0

    # ā ⊗ ā = Eaa · [q; x₁ₚ; 1] — the structural identity that closes the system
    Eaa = spzeros(na * na, nq + nPast + 1)
    @inbounds for i in 1:na, j in 1:na
        r = (i - 1) * na + j
        if i <= nPast && j <= nPast
            Eaa[r, (i - 1) * nPast + j] = 1.0
        elseif i <= nPast && j == nPast + 1
            Eaa[r, nq + i] = 1.0
        elseif i == nPast + 1 && j <= nPast
            Eaa[r, nq + j] = 1.0
        elseif i == nPast + 1 && j == nPast + 1
            Eaa[r, nq + nPast + 1] = 1.0
        end
    end
    Eq = Eaa[:, 1:nq]
    Ep = Eaa[:, nq+1:nq+nPast]
    E1 = Eaa[:, nq+nPast+1]

    PS1  = P * S1
    V    = PS1 * S
    SS   = ℒ.kron(S, S)
    vecI = vec(Matrix{Float64}(ℒ.I(nExo)))
    PP   = ℒ.kron(PS1, PS1)
    A1   = S1 * Ea[:, 1:nPast] * P

    r1, r2, rq = 1:nVars, nVars+1:2nVars, 2nVars+1:nz

    𝒜 = zeros(nz, nz)
    c = zeros(nz)
    𝒜[r1, r1] = A1;                     c[r1] = S1 * Ea[:, nPast+1]
    𝒜[r2, r2] = A1
    𝒜[r2, rq] = S2 * Eq / 2
    𝒜[r2, r1] = S2 * Ep * P / 2;        c[r2] = S2 * (E1 + SS * vecI) / 2
    𝒜[rq, rq] = PP * Eq
    𝒜[rq, r1] = PP * Ep * P;            c[rq] = PP * E1 + ℒ.kron(V, V) * vecI

    𝒞 = zeros(length(observables_index), nz)
    @inbounds for (i, j) in enumerate(observables_index)
        𝒞[i, j] = 1.0            # x₁ block
        𝒞[i, nVars + j] = 1.0    # x₂ block
    end

    # constant (state-independent) part of the innovation covariance
    Hq = [zeros(nVars, nExo^2); S2 * SS / 2; ℒ.kron(V, V)]
    IK = Matrix{Float64}(ℒ.I(nExo^2)) + Matrix(commutation_matrix(nExo))
    QH = Hq * IK * Hq'
    QH = (QH + QH') / 2

    return (; nVars, nPast, nExo, na, nq, nz, past, P, S, Ea, S1, S2, PS1, V,
              𝒜, c, 𝒞, QH, G1 = S1 * S, r1)
end

# State-dependent loading of the linear-in-ε part of the innovation, at state z.
function quadratic_kalman_G(sys, z::AbstractVector{Float64})
    ā = sys.Ea * vcat(sys.P * view(z, sys.r1), 1.0)
    ū = sys.PS1 * ā
    G2 = sys.S2 * (ℒ.kron(ā, sys.S) + ℒ.kron(sys.S, ā)) / 2
    Gq = ℒ.kron(ū, sys.V) + ℒ.kron(sys.V, ū)
    return vcat(sys.G1, G2, Gq)
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
                              data_in_deviations::AbstractMatrix{Float64};
                              measurement_error::Union{Nothing,AbstractVector{<:Real},AbstractMatrix{<:Real}} = nothing,
                              presample_periods::Int = 0,
                              on_failure_loglikelihood::Real = -Inf)
    nz = sys.nz
    𝒜, c, 𝒞, QH = sys.𝒜, sys.c, sys.𝒞, sys.QH
    n_obs, nT = size(data_in_deviations)
    presample_periods = normalize_presample_periods(presample_periods, nT)

    Hm = if measurement_error === nothing
        zeros(n_obs, n_obs)
    elseif measurement_error isa AbstractMatrix
        Matrix{Float64}(measurement_error)
    else
        Matrix{Float64}(ℒ.Diagonal(collect(float.(measurement_error))))
    end

    z̄ = (Matrix{Float64}(ℒ.I(nz)) - 𝒜) \ c
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

    z = copy(z̄)
    Pc = copy(Σ)
    loglik = 0.0
    log2pi = log(2π)

    for t in 1:nT
        G = quadratic_kalman_G(sys, z)
        Q = G * G' + QH

        zp = 𝒜 * z + c
        Pp = 𝒜 * Pc * 𝒜' + Q
        Pp = (Pp + Pp') / 2

        v  = view(data_in_deviations, :, t) - 𝒞 * zp
        CP = 𝒞 * Pp
        F  = CP * 𝒞' + Hm
        F  = (F + F') / 2

        Fc = ℒ.cholesky(F, check = false)
        ℒ.issuccess(Fc) || return Float64(on_failure_loglikelihood)

        if t > presample_periods
            loglik -= 0.5 * (ℒ.dot(v, Fc \ v) + ℒ.logdet(Fc) + n_obs * log2pi)
            isfinite(loglik) || return Float64(on_failure_loglikelihood)
        end

        K = CP' / Fc
        z = zp + K * v
        Pc = Pp - K * CP
        Pc = (Pc + Pc') / 2
    end

    return loglik
end

end # @stable
