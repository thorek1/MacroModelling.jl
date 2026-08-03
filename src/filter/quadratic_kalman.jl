@stable default_mode = "disable" begin

# Kollmann-style quadratic Kalman filter for the pruned second-order solution.
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
#   zₜ = [x₁ₜ; x₂ₜ; vech(x₁ₜ[past]x₁ₜ[past]')]
#
# every block above becomes affine in zₜ₋₁, because the symmetric products in
# aug₁⊗aug₁ expand into terms that are quadratic in x₁ₜ₋₁[past] (carried by the
# third block), linear in it, or constant. The observation is a plain selection,
# yₜ = (x₁ₜ + x₂ₜ)[observables],
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
#   w = G ε + H (vech(εε') − E[vech(εε')]),
#
# linear plus centred-quadratic in ε. Because the Gaussian third moment vanishes
# the two parts are uncorrelated, so
#
#   Var(w) = G G' + H Var(vech(εε')) H',
#
# `H` is constant; `G` depends on the state. The plug-in term uses the filtered
# mean, while the recursion also adds the exact covariance of this affine loading
# under the filtered state covariance.
#
# The symmetric pair block is stored directly in its unique lower-triangular
# ordering; no duplication/elimination or dense Kronecker matrix is constructed.
# The retained augmented dimension is 2·n_r + nPast(nPast+1)/2 (446 for SW07),
# and its covariance recursion remains O(n_z³) per period.

quadratic_pair_indices(n::Int) = [(i, j) for i in 1:n for j in 1:i]

function quadratic_vech_index(i::Int, j::Int, n::Int)
    i, j = max(i, j), min(i, j)
    return (j - 1) * (2n - j + 2) ÷ 2 + i - j + 1
end

function quadratic_pair_reorder(n::Int)
    nq0 = n * (n + 1) ÷ 2
    out = zeros(Float64, nq0, nq0)
    @inbounds for (column, (i, j)) in enumerate(quadratic_pair_indices(n))
        out[quadratic_vech_index(i, j, n), column] = 1.0
    end
    return out
end

"""Map raw unique pair products through a rectangular linear map."""
function raw_pair_power_matrix(A)
    nout, nin = size(A)
    out_pairs = quadratic_pair_indices(nout)
    in_pairs = quadratic_pair_indices(nin)
    out = zeros(eltype(A), length(out_pairs), length(in_pairs))
    @inbounds for (row, (i, j)) in enumerate(out_pairs), (column, (p, q)) in enumerate(in_pairs)
        if p == q
            out[row, column] = A[i, p] * A[j, p]
        else
            out[row, column] = A[i, p] * A[j, q] + A[i, q] * A[j, p]
        end
    end
    return out
end

function raw_pair_power_matrix_pullback!(Abar, A, cotangent)
    nout, nin = size(A)
    @inbounds for (row, (i, j)) in enumerate(quadratic_pair_indices(nout)),
                      (column, (p, q)) in enumerate(quadratic_pair_indices(nin))
        value = cotangent[row, column]
        if p == q
            Abar[i, p] += value * A[j, p]
            Abar[j, p] += value * A[i, p]
        else
            Abar[i, p] += value * A[j, q]
            Abar[j, q] += value * A[i, p]
            Abar[i, q] += value * A[j, p]
            Abar[j, p] += value * A[i, q]
        end
    end
    return Abar
end

function compressed_pair_power_matrix_rect(A)
    nout, nin = size(A)
    out_pairs = quadratic_pair_indices(nout)
    in_pairs = quadratic_pair_indices(nin)
    out = zeros(eltype(A), length(out_pairs), length(in_pairs))
    @inbounds for (row, (i, j)) in enumerate(out_pairs), (column, (p, q)) in enumerate(in_pairs)
        factor = i == j ? 1 : 2
        if p == q
            out[row, column] = factor * A[i, p] * A[j, p]
        else
            out[row, column] = factor * (A[i, p] * A[j, q] + A[i, q] * A[j, p])
        end
    end
    return out
end

"""Compressed pair products of a vector with every column of a matrix."""
function compressed_pair_mixed_matrix(a, B)
    n = length(a)
    size(B, 1) == n || throw(DimensionMismatch("compressed pair inputs must have equal row count"))
    out = zeros(promote_type(eltype(a), eltype(B)), n * (n + 1) ÷ 2, size(B, 2))
    @inbounds for (row, (i, j)) in enumerate(quadratic_pair_indices(n))
        if i == j
            out[row, :] .= a[i] .* B[j, :]
        else
            out[row, :] .= a[i] .* B[j, :] .+ a[j] .* B[i, :]
        end
    end
    return out
end

function raw_pair_mixed_matrix(a, B)
    n = length(a)
    size(B, 1) == n || throw(DimensionMismatch("raw pair inputs must have equal row count"))
    out = zeros(promote_type(eltype(a), eltype(B)), n * (n + 1) ÷ 2, size(B, 2))
    @inbounds for (row, (i, j)) in enumerate(quadratic_pair_indices(n))
        if i == j
            out[row, :] .= 2 .* a[i] .* B[j, :]
        else
            out[row, :] .= a[i] .* B[j, :] .+ a[j] .* B[i, :]
        end
    end
    return out
end

function raw_pair_mixed_pullback!(abar, Bbar, a, B, cotangent)
    @inbounds for (row, (i, j)) in enumerate(quadratic_pair_indices(length(a)))
        value = view(cotangent, row, :)
        if i == j
            abar[i] += 2 * ℒ.dot(value, view(B, j, :))
            Bbar[j, :] .+= 2 * a[i] .* value
        else
            abar[i] += ℒ.dot(value, view(B, j, :))
            abar[j] += ℒ.dot(value, view(B, i, :))
            Bbar[j, :] .+= a[i] .* value
            Bbar[i, :] .+= a[j] .* value
        end
    end
    return abar, Bbar
end

"""Raw pair products of independent standard-normal loadings."""
function raw_pair_noise_mean(B)
    n = size(B, 1)
    out = zeros(eltype(B), n * (n + 1) ÷ 2)
    @inbounds for (row, (i, j)) in enumerate(quadratic_pair_indices(n))
        out[row] = ℒ.dot(view(B, i, :), view(B, j, :))
    end
    return out
end

function raw_pair_noise_pullback!(Bbar, B, cotangent)
    @inbounds for (row, (i, j)) in enumerate(quadratic_pair_indices(size(B, 1)))
        value = cotangent[row]
        if i == j
            Bbar[i, :] .+= 2 * value .* B[i, :]
        else
            Bbar[i, :] .+= value .* B[j, :]
            Bbar[j, :] .+= value .* B[i, :]
        end
    end
    return Bbar
end

function compressed_pair_noise_mean(B)
    n = size(B, 1)
    out = zeros(eltype(B), n * (n + 1) ÷ 2)
    @inbounds for (row, (i, j)) in enumerate(quadratic_pair_indices(n))
        out[row] = (i == j ? 1 : 2) * ℒ.dot(view(B, i, :), view(B, j, :))
    end
    return out
end

quadratic_noise_covariance(nExo::Int) = ℒ.Diagonal([i == j ? 2.0 : 1.0
                                                  for (i, j) in quadratic_pair_indices(nExo)])

"""Lift `[q_raw; x; 1]` to the compressed square of `[x; 1]`."""
function quadratic_pair_lift(nPast::Int, nExo::Int = 0)
    na0 = nPast + 1 + nExo
    out = zeros(Float64, na0 * (na0 + 1) ÷ 2, nPast * (nPast + 1) ÷ 2 + nPast + 1)
    nq0 = nPast * (nPast + 1) ÷ 2
    @inbounds for (row, (i, j)) in enumerate(quadratic_pair_indices(na0))
        if i <= nPast
            if j <= nPast
                out[row, quadratic_vech_index(i, j, nPast)] = i == j ? 1.0 : 2.0
            else
                out[row, nq0 + i] = 2.0
            end
        elseif i == nPast + 1
            if j <= nPast
                out[row, nq0 + j] = 2.0
            else
                out[row, nq0 + nPast + 1] = 1.0
            end
        end
    end
    return out
end

"""Lift `[q_raw; x; 1]` to the raw square of `[x; 1]`."""
function quadratic_raw_pair_lift(nPast::Int, nExo::Int = 0)
    na0 = nPast + 1 + nExo
    out = zeros(Float64, na0 * (na0 + 1) ÷ 2, nPast * (nPast + 1) ÷ 2 + nPast + 1)
    nq0 = nPast * (nPast + 1) ÷ 2
    @inbounds for (row, (i, j)) in enumerate(quadratic_pair_indices(na0))
        if i <= nPast
            if j <= nPast
                out[row, quadratic_vech_index(i, j, nPast)] = 1.0
            else
                out[row, nq0 + i] = 1.0
            end
        elseif i == nPast + 1
            if j <= nPast
                out[row, nq0 + j] = 1.0
            else
                out[row, nq0 + nPast + 1] = 1.0
            end
        end
    end
    return out
end

"""
Build the augmented linear state-space representation of the pruned second-order
solution, together with the pieces needed for the state-dependent innovation
covariance. `𝐒₁`/`𝐒₂` are the solution matrices in the compressed symmetric
quadratic basis as returned by
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

    PS1  = P * S1
    V    = PS1 * S
    EaP  = Ea[:, 1:nPast] * P
    A1   = S1 * EaP

    # The solution uses the unique compressed Hessian columns.  Keep the
    # retained q block raw (one entry per distinct state product), and build
    # the two pair lifts explicitly: `pair_lift` supplies the multiplicities
    # required by the compressed S₂ basis, while `raw_lift` closes the q
    # recursion without doubling off-diagonal state products.
    pair_lift = quadratic_pair_lift(nPast, nExo)
    raw_lift = quadratic_raw_pair_lift(nPast, nExo)
    pair_reorder = quadratic_pair_reorder(nPast)
    pair_transition = pair_reorder * raw_pair_power_matrix(PS1)
    pair_noise = pair_reorder * raw_pair_noise_mean(V)
    compressed_shock_noise = compressed_pair_noise_mean(S)
    pair_solution = S2 * pair_lift
    pair_state = pair_transition * raw_lift
    shock_pair_solution = compressed_pair_power_matrix_rect(S)
    n_pair_exo = nExo * (nExo + 1) ÷ 2
    shock_pair_covariance = Matrix(quadratic_noise_covariance(nExo))

    r1, r2, rq = 1:nr, nr+1:2nr, 2nr+1:nz

    𝒜 = zeros(Tv, nz, nz)
    c = zeros(Tv, nz)
    𝒜[r1, r1] = A1;                     c[r1] = S1 * Ea[:, nPast+1]
    𝒜[r2, r2] = A1
    𝒜[r2, rq] = pair_solution[:, 1:nq] / 2
    𝒜[r2, r1] = pair_solution[:, nq+1:nq+nPast] * P / 2
    c[r2] = S2 * (pair_lift[:, end] + compressed_shock_noise) / 2
    𝒜[rq, rq] = pair_state[:, 1:nq]
    𝒜[rq, r1] = pair_state[:, nq+1:nq+nPast] * P
    c[rq] = pair_state[:, end] + pair_noise

    𝒞 = zeros(length(observables_index), nz)
    @inbounds for (i, j) in enumerate(observables_index)
        𝒞[i, pos[j]] = 1.0            # x₁ block
        𝒞[i, nr + pos[j]] = 1.0       # x₂ block
    end

    # constant (state-independent) part of the innovation covariance
    Hq = [zeros(Tv, nr, n_pair_exo); S2 * shock_pair_solution / 2;
          pair_reorder * raw_pair_power_matrix(V)]
    IK = shock_pair_covariance
    QH = Hq * IK * Hq'
    QH = (QH + QH') / 2

    # The constant blocks are returned as well: the reverse-mode rule needs them
    # to push cotangents from (𝒜, c, QH, g₀, Λ) back onto 𝐒₁ and 𝐒₂.
    return (; nVars, nr, oas, nPast, nExo, na, nq, nz, past, P, S, Ea, S1, S2, PS1, V,
              𝒜, c, 𝒞, QH, G1 = S1 * S, r1,
              r2 = nr+1:2nr, rq = 2nr+1:nz,
              Eq = pair_lift, Ep = pair_lift[:, nq+1:nq+nPast],
              E1 = pair_lift[:, end], SS = compressed_pair_power_matrix_rect(S),
              vecI = raw_pair_noise_mean(Matrix{Float64}(ℒ.I(nExo))), IK,
              EaP, Ea1 = Ea[:, nPast+1], EpP = pair_lift[:, nq+1:nq+nPast] * P,
              PP = pair_transition, KVV = pair_reorder * raw_pair_power_matrix(V), Hq,
              pair_reorder,
              pair_lift, raw_lift, pair_transition, pair_noise, compressed_shock_noise,
              shock_pair_solution, shock_pair_covariance)
end

# State-dependent loading of the linear-in-ε part of the innovation, at state z.
function quadratic_kalman_G(sys, z::AbstractVector{<:Real})
    ā = sys.Ea * vcat(sys.P * view(z, sys.r1), one(eltype(z)))
    ū = sys.PS1 * ā
    G2 = sys.S2 * compressed_pair_mixed_matrix(ā, sys.S) / 2
    Gq = sys.pair_reorder * raw_pair_mixed_matrix(ū, sys.V)
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

# Conditional covariance of the quadratic innovation.  If G(z) = Ḡ + Σᵢ zᵢGᵢ
# and z has covariance P, the state-dependent part contributes
#
#     Σᵢⱼ Pᵢⱼ Gᵢ Gⱼ'
#
# in addition to the plug-in term ḠḠ'.  Pz selects the past first-order state
# from z, so only that small covariance is needed here; the full augmented
# covariance is still required by the Kalman prediction/update itself.
function quadratic_kalman_noise_covariance!(Q, G, QH, Λ, Pz, Pc, PzPc, Pa, LPa)
    ℒ.mul!(PzPc, Pz, Pc)
    ℒ.mul!(Pa, PzPc, Pz')
    copyto!(Q, QH)
    ℒ.mul!(Q, G, G', one(eltype(Q)), one(eltype(Q)))
    nz, nExo = size(G)
    @inbounds for j in 1:nExo
        rows = (j - 1) * nz + 1:j * nz
        ℒ.mul!(LPa, view(Λ, rows, :), Pa)
        ℒ.mul!(Q, LPa, view(Λ, rows, :)', one(eltype(Q)), one(eltype(Q)))
    end
    @inbounds for j in 1:nz, i in 1:j
        m = (Q[i, j] + Q[j, i]) / 2
        Q[i, j] = m; Q[j, i] = m
    end
    return Q
end

# The first-order block is autonomous in the pruned system.  Its stationary
# covariance can therefore be solved separately and used to evaluate the
# state-dependent innovation covariance at the ergodic initialization without
# introducing a nonlinear covariance fixed point.
function quadratic_kalman_initial_covariance(sys, z0, g0, Λ, Pz;
                                             workspaces = nothing,
                                             lyapunov_algorithm::Symbol = :doubling,
                                             initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0, 0))
    A1 = Matrix(view(sys.𝒜, sys.r1, sys.r1))
    Q1 = sys.G1 * sys.G1'
    Σ1 = qkf_lyapunov(A1, Q1; workspaces = workspaces,
                      lyapunov_algorithm = lyapunov_algorithm)
    G0 = reshape(g0 + Λ * (Pz * z0), sys.nz, sys.nExo)
    Tv = promote_type(eltype(G0), eltype(sys.QH), eltype(Σ1))
    Q0 = Matrix{Tv}(undef, sys.nz, sys.nz)
    PzPc = Matrix{Tv}(undef, sys.nPast, sys.nr)
    Pa = Matrix{Tv}(undef, sys.nPast, sys.nPast)
    LPa = Matrix{Tv}(undef, sys.nz, sys.nPast)
    quadratic_kalman_noise_covariance!(Q0, G0, sys.QH, Λ, sys.P, Σ1,
                                       PzPc, Pa, LPa)
    Σ0 = qkf_lyapunov(sys.𝒜, Q0; workspaces = workspaces,
                      lyapunov_algorithm = lyapunov_algorithm,
                      initial_guess = initial_guess)
    return Σ0, Σ1
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
    Q  = Matrix{Tv}(undef, nz, nz)
    z  = Vector{Tv}(undef, nz); copyto!(z, z0)
    zp = Vector{Tv}(undef, nz)
    gv = Vector{Tv}(undef, nz * nExo)
    x1p = Vector{Tv}(undef, size(Pz, 1))
    PzPc = Matrix{Tv}(undef, size(Pz, 1), nz)
    Pa = Matrix{Tv}(undef, size(Pz, 1), size(Pz, 1))
    LPa = Matrix{Tv}(undef, nz, size(Pz, 1))
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

        # Pp = 𝒜 Pc 𝒜' + E[Var(w | z)]
        ℒ.mul!(Tm, 𝒜, Pc)
        ℒ.mul!(Pp, Tm, 𝒜')
        quadratic_kalman_noise_covariance!(Q, G, QH, Λ, Pz, Pc, PzPc, Pa, LPa)
        Pp .+= Q
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
    Gs = Vector{Matrix{Float64}}(undef, nT); Pas = Vector{Matrix{Float64}}(undef, nT)
    vs = Vector{Vector{Float64}}(undef, nT)
    CPs = Vector{Matrix{Float64}}(undef, nT); Fis = Vector{Matrix{Float64}}(undef, nT)
    Ks = Vector{Matrix{Float64}}(undef, nT)
    @inbounds for t in 1:nT
        zs[t] = zeros(nz); Ps[t] = zeros(nz, nz)
        Gs[t] = zeros(nz, nExo); Pas[t] = zeros(size(Pz, 1), size(Pz, 1))
        vs[t] = zeros(n_obs); CPs[t] = zeros(n_obs, nz)
        Fis[t] = zeros(n_obs, n_obs); Ks[t] = zeros(nz, n_obs)
    end
    z = copy(z0); Pc = copy(Σ0); Pp = zeros(nz, nz); Tm = zeros(nz, nz)
    Q = zeros(nz, nz); zp = zeros(nz); gv = zeros(nz * nExo)
    x1p = zeros(size(Pz, 1)); PzPc = zeros(size(Pz, 1), nz)
    Pa = zeros(size(Pz, 1), size(Pz, 1)); LPa = zeros(nz, size(Pz, 1))
    CP = zeros(n_obs, nz); F = zeros(n_obs, n_obs); Fv = zeros(n_obs)
    identity_obs = Matrix{Float64}(ℒ.I(n_obs))
    ll = 0.0; log2pi = log(2π); failed = false
    for t in 1:nT
        copyto!(zs[t], z); copyto!(Ps[t], Pc)
        ℒ.mul!(x1p, Pz, z)
        copyto!(gv, g0); ℒ.mul!(gv, Λ, x1p, 1.0, 1.0)
        G = reshape(gv, nz, nExo)
        quadratic_kalman_noise_covariance!(Q, G, QH, Λ, Pz, Pc, PzPc, Pa, LPa)
        copyto!(Gs[t], G); copyto!(Pas[t], Pa)
        ℒ.mul!(zp, 𝒜, z); zp .+= c
        ℒ.mul!(Tm, 𝒜, Pc); ℒ.mul!(Pp, Tm, 𝒜')
        Pp .+= Q
        @inbounds for j in 1:nz, i in 1:j-1
            value = (Pp[i, j] + Pp[j, i]) / 2
            Pp[i, j] = value; Pp[j, i] = value
        end
        copyto!(vs[t], view(Y, :, t)); ℒ.mul!(vs[t], 𝒞, zp, -1.0, 1.0)
        ℒ.mul!(CP, 𝒞, Pp); copyto!(CPs[t], CP)
        ℒ.mul!(F, CP, 𝒞'); F .+= Hm
        @inbounds for j in 1:n_obs, i in 1:j-1
            value = (F[i, j] + F[j, i]) / 2
            F[i, j] = value; F[j, i] = value
        end
        Fc = ℒ.cholesky(F, check = false)
        if !ℒ.issuccess(Fc); failed = true; break; end
        copyto!(Fis[t], identity_obs); ℒ.ldiv!(Fc, Fis[t])
        if t > presample_periods
            copyto!(Fv, vs[t]); ℒ.mul!(Fv, Fis[t], vs[t])
            ll -= 0.5 * (ℒ.dot(vs[t], Fv) + ℒ.logdet(Fc) + n_obs * log2pi)
        end
        copyto!(Ks[t], CP'); ℒ.rdiv!(Ks[t], Fc)
        copyto!(z, zp); ℒ.mul!(z, Ks[t], vs[t], 1.0, 1.0)
        copyto!(Pc, Pp); ℒ.mul!(Pc, Ks[t], CP, -1.0, 1.0)
        @inbounds for j in 1:nz, i in 1:j-1
            value = (Pc[i, j] + Pc[j, i]) / 2
            Pc[i, j] = value; Pc[j, i] = value
        end
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

        # Reuse reverse-sweep buffers. The old implementation constructed several
        # large temporaries inside every period; on SW07 that dominated the
        # pullback's allocation profile even though the algebra itself is closed.
        P̄p = zeros(nz, nz); K̄ = zeros(nz, n_obs); C̄P = zeros(n_obs, nz)
        z̄p = zeros(nz); v̄ = zeros(n_obs); F̄ = zeros(n_obs, n_obs)
        Fv = zeros(n_obs); Ḡ = zeros(nz, nExo)
        P̄a = zeros(size(Pz, 1), size(Pz, 1)); L̄ = zeros(nz, size(Pz, 1))
        Pz_z = zeros(size(Pz, 1)); tmp_nz = zeros(nz); tmp_past = zeros(size(Pz, 1))
        tmp_nzn = zeros(nz, nz); tmp_nzn₂ = zeros(nz, nz)
        tmp_nzobs = zeros(nz, n_obs); tmp_obsnz = zeros(n_obs, nz)
        tmp_obsobs = zeros(n_obs, n_obs); tmp_obsobs₂ = zeros(n_obs, n_obs)
        tmp_nzpast = zeros(nz, size(Pz, 1)); tmp_pastnz = zeros(size(Pz, 1), nz)
        tmp_pastpast = zeros(size(Pz, 1), size(Pz, 1))

        function rank_one_add!(matrix, left, right, scale)
            @inbounds for j in axes(matrix, 2), i in axes(matrix, 1)
                matrix[i, j] += scale * left[i] * right[j]
            end
            return matrix
        end

        function add_scaled!(matrix, source, scale)
            @inbounds for j in axes(matrix, 2), i in axes(matrix, 1)
                matrix[i, j] += scale * source[i, j]
            end
            return matrix
        end

        function symmetrize!(matrix)
            @inbounds for j in axes(matrix, 2), i in 1:j-1
                value = (matrix[i, j] + matrix[j, i]) / 2
                matrix[i, j] = value
                matrix[j, i] = value
            end
            return matrix
        end

        for t in nT:-1:1
            z_, P_, G, v, CP, Fi, K = zs[t], Ps[t], Gs[t], vs[t], CPs[t], Fis[t], Ks[t]
            copyto!(P̄p, P̄)
            ℒ.mul!(K̄, P̄, CP')
            K̄ .*= -1
            ℒ.mul!(C̄P, K', P̄)
            C̄P .*= -1
            copyto!(z̄p, z̄)
            rank_one_add!(K̄, z̄, v, 1)
            ℒ.mul!(v̄, K', z̄)
            ℒ.mul!(tmp_obsnz, Fi, K̄')
            C̄P .+= tmp_obsnz
            ℒ.mul!(tmp_obsobs, CP, K̄)
            ℒ.mul!(F̄, Fi, tmp_obsobs)
            ℒ.mul!(tmp_obsobs₂, F̄, Fi)
            F̄ .= -tmp_obsobs₂
            if t > presample_periods
                ℒ.mul!(Fv, Fi, v)
                v̄ .-= ∂ll .* Fv
                rank_one_add!(F̄, Fv, Fv, ∂ll / 2)
                add_scaled!(F̄, Fi, -∂ll / 2)
            end
            symmetrize!(F̄)
            ℒ.mul!(tmp_obsnz, F̄, 𝒞)
            C̄P .+= tmp_obsnz
            H̄m .+= F̄
            ℒ.mul!(tmp_nzn, 𝒞', C̄P)
            P̄p .+= tmp_nzn
            ℒ.mul!(tmp_nz, 𝒞', v̄)
            z̄p .-= tmp_nz
            @views Ȳ[:, t] .+= v̄
            symmetrize!(P̄p)
            ℒ.mul!(tmp_nzn, P̄p, 𝒜)
            ℒ.mul!(tmp_nzn₂, tmp_nzn, P_)
            add_scaled!(𝒜̄, tmp_nzn₂, 2)
            ℒ.mul!(tmp_nzn, 𝒜', P̄p)
            ℒ.mul!(P̄, tmp_nzn, 𝒜)
            ℒ.mul!(Ḡ, P̄p, G, 2, 0)
            Q̄H .+= P̄p
            vḠ = vec(Ḡ)
            ḡ0 .+= vḠ
            ℒ.mul!(Pz_z, Pz, z_)
            rank_one_add!(Λ̄, vḠ, Pz_z, 1)
            fill!(P̄a, 0)
            @inbounds for j in 1:nExo
                rows = (j - 1) * nz + 1:j * nz
                L = view(Λ, rows, :)
                ℒ.mul!(tmp_nzpast, P̄p, L)
                ℒ.mul!(L̄, tmp_nzpast, Pas[t], 2, 0)
                Λ̄[rows, :] .+= L̄
                ℒ.mul!(tmp_pastnz, L', P̄p)
                ℒ.mul!(tmp_pastpast, tmp_pastnz, L)
                P̄a .+= tmp_pastpast
            end
            symmetrize!(P̄a)
            ℒ.mul!(tmp_pastnz, P̄a, Pz)
            ℒ.mul!(tmp_nzn, Pz', tmp_pastnz)
            P̄ .+= tmp_nzn
            rank_one_add!(𝒜̄, z̄p, z_, 1)
            c̄  .+= z̄p
            ℒ.mul!(tmp_past, Λ', vḠ)
            ℒ.mul!(tmp_nz, Pz', tmp_past)
            ℒ.mul!(z̄, 𝒜', z̄p)
            z̄ .+= tmp_nz
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
                              on_failure_loglikelihood::Real = -Inf,
                              workspaces = nothing,
                              lyapunov_algorithm::Symbol = :doubling,
                              initial_covariance_out::Union{Nothing,Base.RefValue} = nothing)
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

    # The first-order block supplies the only covariance needed by the
    # state-dependent loading at the stationary initialization.
    g0, Λ = quadratic_kalman_affine_G(sys)
    Pz = sys.P * [Matrix{Tv}(ℒ.I(sys.nr)) zeros(Tv, sys.nr, nz - sys.nr)]
    Σ, Σ1 = quadratic_kalman_initial_covariance(sys, z̄, g0, Λ, Pz;
                                                workspaces = workspaces,
                                                lyapunov_algorithm = lyapunov_algorithm)

    # The reverse pass needs this exact matrix again. Handing it back lets the
    # pullback skip a second identical Lyapunov solve (a residual check on an
    # exact guess instead of a full doubling run).
    initial_covariance_out === nothing || (initial_covariance_out[] = Σ)

    return quadratic_kalman_recursion(𝒜, c, QH, g0, Λ, Hm, Matrix(data_in_deviations),
                                      𝒞, Pz, z̄, Σ, nz, sys.nExo,
                                      presample_periods, on_failure_loglikelihood)
end
# Discrete Lyapunov X = A X A' + Q.
#
# Float64 problems go through the package's workspace-backed doubling solver, which
# reuses its buffers instead of allocating a fresh nz×nz triple product per
# iteration. AD element types fall back to the self-contained loop below, since
# `solve_lyapunov_equation` is restricted to `Float64`.
#
# `initial_guess` pays off only when the guess is *exact*: the solver checks its
# residual (two nz³ products) and returns it, ~15× faster than a full solve on
# SW07. Under even a 1e-6 relative parameter move the check fails and the solve
# runs anyway, making it a net ~6% loss — so this is worth threading from the
# forward pass into the reverse pass, which re-solves the identical equation, but
# *not* worth caching across sampler draws.
function qkf_lyapunov(A, Q;
                      workspaces = nothing,
                      initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0, 0),
                      lyapunov_algorithm::Symbol = :doubling,
                      iters::Int = 80)
    if workspaces !== nothing && eltype(A) === Float64 && eltype(Q) === Float64
        ws = ensure_lyapunov_workspace!(workspaces, size(A, 1), :second_order)
        X, converged = solve_lyapunov_equation(Matrix(A), Matrix(Q), ws;
                                               initial_guess = initial_guess,
                                               lyapunov_algorithm = lyapunov_algorithm,
                                               verbose = false)
        # The solver may hand back one of its own buffers, so symmetrising into a
        # fresh matrix here doubles as taking ownership of the result.
        converged && return (X + X') / 2
    end

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
function quadratic_kalman_pullback(sys, data_in_deviations, Hm, presample_periods, ∂ll;
                                   workspaces = nothing,
                                   lyapunov_algorithm::Symbol = :doubling,
                                   initial_covariance::AbstractMatrix{<:AbstractFloat} = zeros(0, 0))
    nr, nP, nE, na, nq, nz = sys.nr, sys.nPast, sys.nExo, sys.na, sys.nq, sys.nz
    r1, r2, rq = sys.r1, sys.r2, sys.rq
    g0, Λ = quadratic_kalman_affine_G(sys)
    Pz = qkf_Pz(sys)

    z0 = (Matrix{Float64}(ℒ.I(nz)) - sys.𝒜) \ sys.c
    Σ0, Σ1 = quadratic_kalman_initial_covariance(sys, z0, g0, Λ, Pz;
                                                  workspaces = workspaces,
                                                  lyapunov_algorithm = lyapunov_algorithm,
                                                  initial_guess = initial_covariance)
    G0 = reshape(g0 + Λ * (Pz * z0), nz, nE)

    R = last(rrule(quadratic_kalman_recursion, sys.𝒜, sys.c, sys.QH, g0, Λ, Hm,
                   Matrix(data_in_deviations), sys.𝒞, Pz, z0, Σ0, nz, nE,
                   presample_periods, -Inf))(∂ll)
    𝒜̄ = copy(R[2]); c̄ = copy(R[3]); Q̄H = copy(R[4])
    ḡ0 = copy(R[5]); Λ̄ = copy(R[6]); Ȳ = copy(R[8])
    z̄0 = copy(R[11]); Σ̄0 = copy(R[12])

    # Σ0 = 𝒜Σ0𝒜' + Q0  ⇒  X solves X = 𝒜'X𝒜 + Σ̄0
    X = qkf_lyapunov(Matrix(sys.𝒜'), (Σ̄0 + Σ̄0') / 2; workspaces = workspaces,
                     lyapunov_algorithm = lyapunov_algorithm)
    𝒜̄ .+= 2 .* (X * sys.𝒜 * Σ0)
    Q̄0 = (X + X') / 2
    Ḡ0 = 2 .* (Q̄0 * G0); Q̄H .+= Q̄0
    vG0 = vec(Ḡ0); ḡ0 .+= vG0; Λ̄ .+= vG0 * (Pz * z0)'
    z̄0 .+= Pz' * (Λ' * vG0)

    # The stationary first-order covariance enters Q0 through the same
    # state-dependent loading correction as the recursion.  Differentiate its
    # Lyapunov equation separately; this keeps the pullback analytical and
    # avoids differentiating through a nonlinear covariance iteration.
    Pa0 = sys.P * Σ1 * sys.P'
    P̄a0 = zeros(nP, nP)
    @inbounds for j in 1:nE
        rows = (j - 1) * nz + 1:j * nz
        L = view(Λ, rows, :)
        Λ̄[rows, :] .+= 2 .* (Q̄0 * L * Pa0)
        P̄a0 .+= L' * Q̄0 * L
    end
    P̄a0 = (P̄a0 + P̄a0') / 2
    Σ̄1 = sys.P' * P̄a0 * sys.P
    A1 = Matrix(view(sys.𝒜, r1, r1))
    X1 = qkf_lyapunov(A1', Σ̄1; workspaces = workspaces,
                      lyapunov_algorithm = lyapunov_algorithm)
    Ā1_initial = 2 .* (X1 * A1 * Σ1)
    Ḡ1_initial = 2 .* ((X1 + X1') / 2 * sys.G1)
    λ = (Matrix{Float64}(ℒ.I(nz)) - sys.𝒜)' \ z̄0
    c̄ .+= λ; 𝒜̄ .+= λ * z0'

    S̄1 = zeros(size(sys.S1)); S̄2 = zeros(size(sys.S2))
    P̄S1 = zeros(size(sys.PS1)); V̄ = zeros(size(sys.V))

    Ā1 = 𝒜̄[r1, r1] + 𝒜̄[r2, r2]
    S̄1 .+= (Ā1 + Ā1_initial) * sys.EaP'
    S̄1 .+= Ḡ1_initial * sys.S'
    S̄1 .+= c̄[r1] * sys.Ea1'

    # x₂ transition and its constant shock-pair loading use the compressed S₂
    # basis directly.
    S̄2 .+= 𝒜̄[r2, rq] * sys.pair_lift[:, 1:nq]' / 2
    S̄2 .+= 𝒜̄[r2, r1] * (sys.pair_lift[:, nq+1:nq+nP] * sys.P)' / 2
    S̄2 .+= c̄[r2] * (sys.pair_lift[:, end] + sys.compressed_shock_noise)' / 2

    # q' = pair_transition * raw_lift * [q; x; 1] + pair_noise.  The
    # transition is carried in raw unique pair coordinates and reordered only
    # at the q-state boundary.
    pair_statē = zeros(size(sys.pair_transition, 1), size(sys.raw_lift, 2))
    pair_statē[:, 1:nq] .+= 𝒜̄[rq, rq]
    pair_statē[:, nq+1:nq+nP] .+= 𝒜̄[rq, r1] * sys.P'
    pair_statē[:, end] .+= c̄[rq]
    pair_transition̄ = pair_statē * sys.raw_lift'
    raw_transition̄ = sys.pair_reorder' * pair_transition̄
    raw_pair_power_matrix_pullback!(P̄S1, sys.PS1, raw_transition̄)
    raw_noisē = sys.pair_reorder' * c̄[rq]
    raw_pair_noise_pullback!(V̄, sys.V, raw_noisē)

    # QH = Hq * IK * Hq'.
    Rq = (Q̄H + Q̄H') / 2
    H̄q = 2 .* (Rq * sys.Hq * sys.IK)
    S̄2 .+= H̄q[r2, :] * sys.shock_pair_solution' / 2
    raw_pair_power_matrix_pullback!(V̄, sys.V,
                                    sys.pair_reorder' * H̄q[rq, :])

    function absorb_G!(Ḡ, z)
        ā = sys.Ea * vcat(sys.P * view(z, r1), 1.0)
        ū = sys.PS1 * ā
        S̄1 .+= Ḡ[r1, :] * sys.S'
        S̄2 .+= Ḡ[r2, :] * compressed_pair_mixed_matrix(ā, sys.S)' / 2
        ū̄ = zeros(nP); V̄local = zeros(size(sys.V))
        raw_pair_mixed_pullback!(ū̄, V̄local, ū, sys.V,
                                 sys.pair_reorder' * Ḡ[rq, :])
        V̄ .+= V̄local
        P̄S1 .+= ū̄ * ā'
    end
    absorb_G!(reshape(ḡ0 .- vec(sum(Λ̄, dims = 2)), nz, nE), zeros(nz))
    @inbounds for i in 1:nP
        e = zeros(nz); e[findfirst(!iszero, view(sys.P, i, :))] = 1.0
        absorb_G!(reshape(Λ̄[:, i], nz, nE), e)
    end

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
                                on_failure_loglikelihood = on_failure_loglikelihood,
                                workspaces = workspaces,
                                lyapunov_algorithm = lyapunov_algorithm)
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
    Σ₀ref = Ref{Matrix{Float64}}()
    llh = run_quadratic_kalman(sys, data_in_deviations;
                               measurement_error = measurement_error,
                               presample_periods = presample_periods,
                               on_failure_loglikelihood = on_failure_loglikelihood,
                               workspaces = workspaces,
                               lyapunov_algorithm = lyapunov_algorithm,
                               initial_covariance_out = Σ₀ref)

    nine(x...) = (NoTangent(), NoTangent(), NoTangent(), NoTangent(), x[1], x[2],
                  NoTangent(), x[3], NoTangent())

    if !isfinite(llh)
        return llh, _ -> nine(NoTangent(), NoTangent(), NoTangent())
    end

    function quadratic_kalman_loglikelihood_pullback(∂llh_bar)
        ∂llh = unthunk(∂llh_bar)
        S̄1r, S̄2r, Ȳ = quadratic_kalman_pullback(sys, data_in_deviations, Hm,
                                                 presample_periods, ∂llh;
                                                 workspaces = workspaces,
                                                 lyapunov_algorithm = lyapunov_algorithm,
                                                 initial_covariance = isassigned(Σ₀ref) ?
                                                                      Σ₀ref[] : zeros(0, 0))
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
