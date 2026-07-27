@stable default_mode = "disable" begin

# Particle filters for the (possibly nonlinear) DSGE state-space representation.
#
# The measurement equation is  yₜ = full_stateₜ[observables] + ηₜ,  ηₜ ~ N(0, H).
#
# H is a *covariance*, not a standard deviation: a vector argument is read as the
# per-observable variances, a matrix as the full covariance. Both are supported.
# The filters only ever need H⁻¹ and log det H, so the diagonal case is an
# elementwise loop and the correlated case a triangular solve against a Cholesky
# factor of H restricted to the rows observed in the period, cached per
# missing-data pattern (see `DenseMeasurementError` below). Diagonal is the
# default and the fast path, and is also the more natural modelling choice: a
# persistent, correlated measurement error is usually better written into the
# model itself as measurement-error processes in the observation equations, which
# moves the correlation into the state transition and leaves H diagonal again.
#
# Why the *Kalman* filter needs H at all — it has no degeneracy problem the way a
# particle filter does, and works fine with H = 0. Three reasons it is offered:
#   (1) Stochastic singularity. With more observables than structural shocks, the
#       model-implied observables lie on a lower-dimensional manifold, C P C' is
#       singular, and the likelihood is undefined. H > 0 is the standard fix, and
#       what lets a 7-observable model be estimated with fewer than 7 shocks.
#   (2) Model misspecification / data revisions. Smets-Wouters-style estimations
#       routinely put measurement error on a subset of observables so that series
#       the model cannot hope to match exactly do not dominate the likelihood.
#   (3) It is what makes the particle filters checkable. A particle filter needs
#       H > 0, so validating one against the exact likelihood on a linear model
#       requires the Kalman filter to score the *same* H. That comparison is the
#       sharpest correctness test available and is what `test_particle_filter.jl`
#       and `test_particle_filter_sw07.jl` do.
#
# The structural shocks are i.i.d. standard normal (their standard deviations are
# baked into the solution matrices 𝐒), and the state transition is the
# perturbation solution's `state_update` (first order through pruned third order).
#
# Three variants are provided, each selected by its own `filter` value:
#   :bootstrap_particle — sequential importance resampling, i.e. the bootstrap
#                         filter of Gordon, Salmond & Smith (1993), applied to
#                         DSGE models by Fernández-Villaverde & Rubio-Ramírez
#                         (2007)
#   :auxiliary_particle — auxiliary particle filter of Pitt & Shephard (1999)
#   :tempered_particle  — tempered particle filter of Herbst & Schorfheide (2019)
#
# The particle filter is a stochastic likelihood estimator and is **not**
# differentiable (resampling is discontinuous); it is intended for use with
# gradient-free samplers (e.g. Pigeons slice sampling, nested sampling).


# ── Resampling schemes ───────────────────────────────────────────────────────
#
# Why resample at all? Reweighting alone degenerates: after a few periods almost
# all of the weight sits on one particle and the cloud carries no information
# about the state distribution. Resampling replaces the weighted cloud with an
# equally weighted one — duplicating heavy particles, dropping light ones — so
# the computational effort follows the probability mass.
#
# Every scheme below is unbiased (E[times particle i is picked] = N·Wᵢ), so the
# likelihood estimate stays unbiased whichever one is used. They differ only in
# the *variance* of the counts, i.e. how much extra Monte-Carlo noise resampling
# itself injects. Ranked from lowest to highest added variance:
#
#   :systematic  — one uniform draw, then N equally spaced points through the
#                  cumulative weights. Lowest variance and the cheapest (a single
#                  random number, one pass); the sensible default. Its one caveat
#                  is that the N draws are perfectly correlated, which can matter
#                  for some theoretical guarantees but not for the likelihood.
#   :stratified  — one independent uniform per stratum of width 1/N. Almost as
#                  low variance as systematic but with independent draws, which
#                  restores those guarantees; a safe, slightly noisier default.
#   :residual    — deterministically assign ⌊N·Wᵢ⌋ copies, then draw only the
#                  remainder multinomially. Removes the integer part of the noise
#                  entirely; useful when a few particles dominate the weights.
#   :multinomial — N independent draws from the weights. The textbook scheme and
#                  the easiest to reason about, but the noisiest; kept mainly as
#                  a reference implementation.
#
# Each returns a length-N vector of ancestor indices drawn from the normalised
# weights `W` (which must sum to one).

# Effective sample size, 1 / Σ Wᵢ². Equals N when the weights are uniform and 1
# when a single particle holds all the mass, so it measures how many particles
# are "really" contributing. The filters resample once it drops below a fraction
# of N, which avoids paying the resampling noise in periods that do not need it.
effective_sample_size(W::AbstractVector{<:Real}) = 1.0 / sum(abs2, W)

# Walk N equally spaced points u₀, u₀+1/N, … through the cumulative weights, with
# a single random offset u₀ ∈ [0, 1/N). A particle of weight Wᵢ spans Wᵢ·N spacings
# so it is picked either ⌊N·Wᵢ⌋ or ⌈N·Wᵢ⌉ times — never far from its expectation.
function systematic_resample_indices(rng::Random.AbstractRNG, W::AbstractVector{<:Real})
    N = length(W)
    idxs = Vector{Int}(undef, N)
    u0 = rand(rng) / N
    c = W[1]
    i = 1
    @inbounds for j in 1:N
        u = u0 + (j - 1) / N
        while u > c && i < N
            i += 1
            c += W[i]
        end
        idxs[j] = i
    end
    return idxs
end

# Split [0,1) into N strata of width 1/N and draw one independent uniform inside
# each. Guarantees at most one draw per stratum (so counts stay close to N·Wᵢ)
# while keeping the draws independent, unlike the systematic scheme.
function stratified_resample_indices(rng::Random.AbstractRNG, W::AbstractVector{<:Real})
    N = length(W)
    idxs = Vector{Int}(undef, N)
    c = W[1]
    i = 1
    @inbounds for j in 1:N
        u = (j - 1 + rand(rng)) / N
        while u > c && i < N
            i += 1
            c += W[i]
        end
        idxs[j] = i
    end
    return idxs
end

# N independent draws from the categorical distribution defined by W, via binary
# search on the cumulative weights. Simplest and noisiest: nothing prevents a
# particle with weight 1/N from being drawn three times or not at all.
function multinomial_resample_indices(rng::Random.AbstractRNG, W::AbstractVector{<:Real})
    N = length(W)
    c = cumsum(W)
    c[end] = one(eltype(c))   # guard against round-off so rand() ≤ c[end]
    idxs = Vector{Int}(undef, N)
    @inbounds for j in 1:N
        idxs[j] = searchsortedfirst(c, rand(rng))
    end
    return idxs
end

# Deterministic part first: particle i gets ⌊N·Wᵢ⌋ guaranteed copies, which carry
# no randomness at all. Only the leftover R = N - Σ⌊N·Wᵢ⌋ slots are drawn, from
# the renormalised fractional weights. Cuts the variance of the integer part to
# zero, which helps most when a handful of particles dominate.
function residual_resample_indices(rng::Random.AbstractRNG, W::AbstractVector{<:Real})
    N = length(W)
    idxs = Vector{Int}(undef, N)
    counts = floor.(Int, N .* W)
    k = 0
    @inbounds for i in 1:N
        for _ in 1:counts[i]
            k += 1
            idxs[k] = i
        end
    end
    R = N - k
    if R > 0
        resid = N .* W .- counts
        s = sum(resid)
        if s <= 0            # numerical degeneracy: fall back to multinomial
            c = cumsum(W); c[end] = one(eltype(c))
            @inbounds for _ in 1:R
                k += 1
                idxs[k] = searchsortedfirst(c, rand(rng))
            end
        else
            resid ./= s
            c = cumsum(resid); c[end] = one(eltype(c))
            @inbounds for _ in 1:R
                k += 1
                idxs[k] = searchsortedfirst(c, rand(rng))
            end
        end
    end
    return idxs
end

function particle_resample_indices(rng::Random.AbstractRNG, W::AbstractVector{<:Real}, scheme::Symbol)
    if scheme == :systematic
        return systematic_resample_indices(rng, W)
    elseif scheme == :stratified
        return stratified_resample_indices(rng, W)
    elseif scheme == :multinomial
        return multinomial_resample_indices(rng, W)
    elseif scheme == :residual
        return residual_resample_indices(rng, W)
    else
        error("Unknown resampling scheme `:$scheme`. Choose from `:systematic`, `:stratified`, `:multinomial`, `:residual`.")
    end
end


# ── Shared setup ─────────────────────────────────────────────────────────────

# Covariance used to spread the initial particle cloud over the full state.
# `:theoretical` (default) uses the first-order ergodic (unconditional) state
# covariance Σ solving the discrete Lyapunov equation Σ = A Σ A' + B B' (built
# from the cached first-order solution, the usual choice for a stationary model);
# `:diagonal` uses 10·I; an nVars×nVars matrix is used directly.
#
# Note the timing convention, which differs from the Kalman filter's argument of
# the same name. Here Σ is the covariance of x₀, the cloud the filter starts
# from *before* the first transition; the Kalman filter's `initial_covariance` is
# P₁ = Var(x₁), the covariance of the first *predicted* state. The two therefore
# correspond as P₁ = A Σ A' + BB'. This is invisible at the `:theoretical`
# default — the ergodic covariance is the fixed point of that very map, so it is
# carried to itself — which is why passing `:theoretical` to both filters lines
# them up. It matters as soon as an explicit matrix is supplied: to reproduce a
# Kalman run with P₁ = BB' (i.e. the inversion filter), pass a zero matrix here,
# not BB'.
#
# Σ is deliberately the *first-order* covariance at every perturbation order, and
# does not follow `algorithm`. Elsewhere in the package `:theoretical` is only
# ever reached from the Kalman filter (`get_initial_covariance` in kalman.jl),
# which is first-order-only, so there is no precedent either way — the choice is
# specific to this file. Order-consistent ergodic covariances do exist
# (`calculate_second_order_moments_with_covariance` and the third-order routines
# in moments.jl), but they are expensive, live in the augmented pruned-state basis
# rather than the nVars basis the cloud needs, and buy very little: Σ only seeds
# period 1, and the filter forgets it within a handful of periods (which is what
# `presample_periods` is for). Higher-order terms also perturb the *mean* of the
# ergodic distribution, not just its spread, and that shift is already carried by
# `state` from `get_relevant_steady_state_and_state_update`. Pass an explicit
# matrix, or widen the cloud with `particle_initial_state_scaling`, if the
# first-order spread is too tight for a strongly nonlinear model.
function particle_initial_state_covariance(𝓂::ℳ, T, opts::CalculationOptions,
                                           initial_covariance::Union{Symbol,AbstractMatrix{<:Real}})
    nVars = T.nVars

    if initial_covariance isa AbstractMatrix && size(initial_covariance) == (nVars, nVars)
        return Matrix{Float64}(initial_covariance), true
    elseif initial_covariance === :diagonal
        return Matrix{Float64}(10.0 * ℒ.I(nVars)), true
    end

    nPast = T.nPast_not_future_and_mixed
    past_idx = T.past_not_future_and_mixed_idx

    S₁ = 𝓂.caches.first_order_solution_matrix

    A_full = zeros(Float64, nVars, nVars)
    @views A_full[:, past_idx] .= S₁[:, 1:nPast]
    B_full = @views Matrix{Float64}(S₁[:, nPast+1:end])
    𝐁 = B_full * B_full'

    lyap_ws = ensure_lyapunov_workspace!(𝓂.workspaces, nVars, :first_order)
    Σ, solved = solve_lyapunov_equation(A_full, 𝐁, lyap_ws,
                                        lyapunov_algorithm = opts.lyapunov_algorithm,
                                        tol = opts.tol.first_order.lyapunov,
                                        verbose = opts.verbose)

    return solved ? Matrix{Float64}(Σ) : Matrix{Float64}(ℒ.I(nVars)), solved
end

# Lower-triangular factor L (L Lᵀ ≈ scaling·Σ) for sampling the initial cloud.
# Falls back to a diagonal factor if Σ is not numerically positive definite.
function particle_initial_cloud_factor(Σ::Matrix{Float64}, scaling::Float64)
    nVars = size(Σ, 1)
    Σs = ℒ.Symmetric(scaling .* Σ)
    jitter = 1e-12 * (ℒ.tr(Σs) / max(nVars, 1) + 1.0)
    chol = ℒ.cholesky(Σs + jitter * ℒ.I(nVars), check = false)
    if ℒ.issuccess(chol)
        return Matrix{Float64}(chol.L)
    else
        return ℒ.diagm(sqrt.(max.(scaling .* ℒ.diag(Σ), 0.0)))
    end
end

# Build the initial particle cloud. Each particle carries the same representation
# `state_update` consumes: a flat `Vector` for non-pruned orders, or a
# `Vector{Vector}` (first-order + higher-order components) for pruned orders. The
# first-order part is randomised around the initial mean with covariance
# `scaling·Σ`; higher-order pruned components are initialised deterministically.
function initialise_particles(rng::Random.AbstractRNG, state, pruning::Bool,
                              L::Matrix{Float64}, n_particles::Int, nVars::Int)
    if pruning
        mean1 = Vector{Float64}(state[1])
        rest  = [Vector{Float64}(state[c]) for c in 2:length(state)]
        return [vcat([mean1 .+ L * randn(rng, nVars)], [copy(r) for r in rest]) for _ in 1:n_particles]
    else
        mean1 = state isa AbstractVector{<:AbstractVector} ? Vector{Float64}(state[1]) : Vector{Float64}(state)
        return [mean1 .+ L * randn(rng, nVars) for _ in 1:n_particles]
    end
end

# Full (summed) model state a particle reports to the measurement equation.
@inline particle_full_state(p, pruning::Bool) = pruning ? sum(p) : p

# Log Gaussian measurement density of the observed rows for one particle's
# predicted observables, with diagonal measurement-error variances `me_var`.
# `rows` indexes the observed observables at the current period; returns -Inf on
# a non-finite prediction.
@inline function particle_log_measurement_density(full::AbstractVector, data_col, observables_index,
                                                  me_var, rows, log2pi::Float64)
    q = 0.0
    @inbounds for r in rows
        f = full[observables_index[r]]
        isfinite(f) || return -Inf
        v = data_col[r] - f
        q += v * v / me_var[r] + log2pi + log(me_var[r])
    end
    return -0.5 * q
end


# ── Non-diagonal measurement error ───────────────────────────────────────────
# Everything above takes `me_var`, the diagonal of H, and reads it elementwise —
# the fast path, and the default. A correlated H needs the same two quantities,
# vᵀH⁻¹v and log det H, but restricted to the rows observed in the period. Both
# come from one Cholesky factor of H[rows, rows], so we factorise once per
# missing-data pattern and cache it; with complete data that is a single
# factorisation for the whole sample. `DenseMeasurementError` is then accepted
# anywhere `me_var` is, by dispatch, leaving the diagonal path untouched.
mutable struct DenseMeasurementError
    H::Matrix{Float64}
    # rows pattern → (lower Cholesky factor of H[rows, rows], log det H[rows, rows]).
    # Kept as a plain matrix rather than a `Cholesky`: the solve below is a hand
    # written forward substitution, which for the handful of observables a DSGE
    # has is both faster and allocation-free (a triangular `ldiv!` on a view of
    # the scratch buffer heap-allocates the view, once per particle per period).
    factors::Dict{Vector{Int},Tuple{Matrix{Float64},Float64}}
    buf::Vector{Float64}   # innovation scratch, sized to the current pattern
    # The row pattern is constant across a period's inner loop, so the dictionary
    # lookup (which needs a freshly allocated key vector) is guarded by a
    # one-entry memo on the last pattern seen.
    last_rows::Vector{Int}
    last_L::Matrix{Float64}
    last_logdet::Float64
end

function DenseMeasurementError(H::AbstractMatrix{<:Real})
    Hf = Matrix{Float64}(H)
    return DenseMeasurementError(Hf,
                                 Dict{Vector{Int},Tuple{Matrix{Float64},Float64}}(),
                                 Float64[],
                                 Int[],
                                 zeros(Float64, 0, 0),
                                 0.0)
end

# Allocation-free comparison of the memoised pattern against the current `rows`
# (which may be a Vector, a range, or any iterable of indices).
@inline function same_rows(last::Vector{Int}, rows)
    length(last) == length(rows) || return false
    @inbounds for (k, r) in enumerate(rows)
        last[k] == r || return false
    end
    return true
end

# Point the memo at the factorisation for `rows`, computing it on first sight.
@inline function me_sync!(me::DenseMeasurementError, rows)
    same_rows(me.last_rows, rows) && return nothing

    key = collect(Int, rows)
    cached = get(me.factors, key, nothing)
    if cached === nothing
        F = ℒ.cholesky(ℒ.Symmetric(me.H[key, key]))
        cached = (Matrix{Float64}(F.L), 2 * sum(log, ℒ.diag(F.U)))
        me.factors[key] = cached
    end
    me.last_rows   = key
    me.last_L      = cached[1]
    me.last_logdet = cached[2]
    resize!(me.buf, length(key))
    return nothing
end

# vᵀH⁻¹v over the observed rows. `Inf` on a non-finite prediction, matching the
# diagonal version's contract (an impossible particle gets zero weight).
# vᵀH⁻¹v = ‖L⁻¹v‖², so one forward substitution gives both the solve and the norm.
@inline function particle_quadratic_form(full::AbstractVector, data_col, observables_index,
                                         me::DenseMeasurementError, rows)
    me_sync!(me, rows)
    v = me.buf
    @inbounds for (k, r) in enumerate(rows)
        f = full[observables_index[r]]
        isfinite(f) || return Inf
        v[k] = data_col[r] - f
    end
    return dense_me_quadform!(v, me.last_L)
end

# In-place forward substitution L y = v, returning ‖y‖².
@inline function dense_me_quadform!(v::Vector{Float64}, L::Matrix{Float64})
    q = 0.0
    @inbounds for i in eachindex(v)
        acc = v[i]
        for j in 1:i-1
            acc -= L[i, j] * v[j]
        end
        y = acc / L[i, i]
        v[i] = y
        q += y * y
    end
    return q
end

@inline function particle_measurement_logZ(me::DenseMeasurementError, rows, log2pi::Float64)
    me_sync!(me, rows)
    return -0.5 * (length(rows) * log2pi + me.last_logdet)
end

@inline function particle_log_measurement_density(full::AbstractVector, data_col, observables_index,
                                                  me::DenseMeasurementError, rows, log2pi::Float64)
    q = particle_quadratic_form(full, data_col, observables_index, me, rows)
    isfinite(q) || return -Inf
    return particle_measurement_logZ(me, rows, log2pi) - 0.5 * q
end

# The diagonal of H: what the auxiliary filter's first-stage preview needs (it
# only has to be a rough predictive variance — the second-stage reweighting is
# exact whatever the preview used).
me_diagonal(me_var::AbstractVector) = me_var
me_diagonal(me::DenseMeasurementError) = ℒ.diag(me.H)

# Elementwise reciprocals for the batched first-order path; the dense filter
# keeps its factorisation instead and is passed through unchanged.
me_inverse_diagonal(me_var::AbstractVector) = 1.0 ./ me_var
me_inverse_diagonal(me::DenseMeasurementError) = me

# Build the measurement-error representation the kernels take from whatever
# `resolve_measurement_error` produced.
build_particle_measurement_error(me::AbstractVector{<:Real}) = Float64.(me)
build_particle_measurement_error(me::AbstractMatrix{<:Real}) = ℒ.isdiag(me) ? collect(Float64, ℒ.diag(me)) : DenseMeasurementError(me)

# Positivity check shared by all kernels.
function assert_positive_measurement_error(me::AbstractVector)
    @assert all(x -> x > 0, me) "The particle filters require strictly positive measurement-error variances for every observable."
    return nothing
end
function assert_positive_measurement_error(me::DenseMeasurementError)
    @assert ℒ.isposdef(ℒ.Symmetric(me.H)) "The particle filters require a positive definite measurement-error covariance."
    return nothing
end


# ── Allocation-free higher-order transitions ─────────────────────────────────
# In-place transitions for the nonlinear orders, mirroring the closures built by
# `parse_algorithm_to_state_update` but writing into a preallocated `out` with
# preallocated `aug`/kron scratch and BLAS `mul!`/`kron!` (no per-call heap
# allocation). The pruned orders reuse `pruned_state_update_{2nd,3rd}_order!`.
# `𝐒[1]` here is the augmented first-order matrix (constant column inserted).

# Gather aug = [state[past]; const; shock] into the preallocated `aug` (explicit
# loops avoid the SubArray allocation of `state[past_idx]`).
@inline function fill_aug!(aug, state, past_idx, shock, const_val)
    n_past = length(past_idx)
    @inbounds for i in 1:n_past
        aug[i] = state[past_idx[i]]
    end
    @inbounds aug[n_past + 1] = const_val
    @inbounds for e in eachindex(shock)
        aug[n_past + 1 + e] = shock[e]
    end
    return aug
end

# out = 𝐒₁·aug + ½ 𝐒₂·(aug⊗aug),  aug = [state[past]; 1; shock].
function nonpruned_state_update_2nd_order!(out, state, past_idx, shock, aug, kk, 𝐒)
    fill_aug!(aug, state, past_idx, shock, 1.0)
    ℒ.kron!(kk, aug, aug)
    ℒ.mul!(out, 𝐒[1], aug)
    ℒ.mul!(out, 𝐒[2], kk, 0.5, 1.0)
    return out
end

# out = 𝐒₁·aug + ½ 𝐒₂·(aug⊗aug) + ⅙ 𝐒₃·(aug⊗aug⊗aug).
function nonpruned_state_update_3rd_order!(out, state, past_idx, shock, aug, kk, kkk, 𝐒)
    fill_aug!(aug, state, past_idx, shock, 1.0)
    ℒ.kron!(kk, aug, aug)
    ℒ.kron!(kkk, kk, aug)
    ℒ.mul!(out, 𝐒[1], aug)
    ℒ.mul!(out, 𝐒[2], kk, 0.5, 1.0)
    ℒ.mul!(out, 𝐒[3], kkk, 1 / 6, 1.0)
    return out
end

# Allocation-free pruned updates (explicit past-gather; components zero the
# constant slot and, for the higher-order parts, the shock slots).
function pf_pruned_2nd!(new_s1, new_s2, s1, s2, past_idx, shock, aug1, aug2, kk, 𝐒)
    n_past = length(past_idx)
    fill_aug!(aug1, s1, past_idx, shock, 1.0)
    @inbounds for i in 1:n_past
        aug2[i] = s2[past_idx[i]]
    end
    @inbounds aug2[n_past + 1] = 0.0
    @inbounds for e in eachindex(shock)
        aug2[n_past + 1 + e] = 0.0
    end
    ℒ.kron!(kk, aug1, aug1)
    ℒ.mul!(new_s1, 𝐒[1], aug1)
    ℒ.mul!(new_s2, 𝐒[1], aug2)
    ℒ.mul!(new_s2, 𝐒[2], kk, 0.5, 1.0)
    return nothing
end

function pf_pruned_3rd!(new_s1, new_s2, new_s3, s1, s2, s3, past_idx, shock,
                        aug1, aug1̂, aug2, aug3, k11, k12̂, k111, 𝐒)
    n_past = length(past_idx)
    fill_aug!(aug1,  s1, past_idx, shock, 1.0)
    fill_aug!(aug1̂, s1, past_idx, shock, 0.0)
    @inbounds for i in 1:n_past
        aug2[i] = s2[past_idx[i]]
        aug3[i] = s3[past_idx[i]]
    end
    @inbounds aug2[n_past + 1] = 0.0
    @inbounds aug3[n_past + 1] = 0.0
    @inbounds for e in eachindex(shock)
        aug2[n_past + 1 + e] = 0.0
        aug3[n_past + 1 + e] = 0.0
    end
    ℒ.kron!(k11,  aug1, aug1)
    ℒ.kron!(k12̂,  aug1̂, aug2)
    ℒ.kron!(k111, k11,  aug1)
    ℒ.mul!(new_s1, 𝐒[1], aug1)
    ℒ.mul!(new_s2, 𝐒[1], aug2)
    ℒ.mul!(new_s2, 𝐒[2], k11, 0.5, 1.0)
    ℒ.mul!(new_s3, 𝐒[1], aug3)
    ℒ.mul!(new_s3, 𝐒[2], k12̂, 1.0, 1.0)
    ℒ.mul!(new_s3, 𝐒[3], k111, 1 / 6, 1.0)
    return nothing
end

# Preallocated kron/aug scratch for one particle, sized per algorithm.
function build_higher_scratch(::Val{:second_order}, nPast::Int, nExo::Int)
    naug = nPast + 1 + nExo
    (aug = Vector{Float64}(undef, naug), kk = Vector{Float64}(undef, naug^2))
end
function build_higher_scratch(::Val{:third_order}, nPast::Int, nExo::Int)
    naug = nPast + 1 + nExo
    (aug = Vector{Float64}(undef, naug), kk = Vector{Float64}(undef, naug^2), kkk = Vector{Float64}(undef, naug^3))
end
function build_higher_scratch(::Val{:pruned_second_order}, nPast::Int, nExo::Int)
    naug = nPast + 1 + nExo
    (aug1 = Vector{Float64}(undef, naug), aug2 = Vector{Float64}(undef, naug),
     kk = Vector{Float64}(undef, naug^2), zero_shock = zeros(Float64, nExo))
end
function build_higher_scratch(::Val{:pruned_third_order}, nPast::Int, nExo::Int)
    naug = nPast + 1 + nExo
    (aug1 = Vector{Float64}(undef, naug), aug1̂ = Vector{Float64}(undef, naug),
     aug2 = Vector{Float64}(undef, naug), aug3 = Vector{Float64}(undef, naug),
     k11 = Vector{Float64}(undef, naug^2), k12̂ = Vector{Float64}(undef, naug^2),
     k111 = Vector{Float64}(undef, naug^3), zero_shock = zeros(Float64, nExo))
end

# In-place propagation dispatch: writes the next state into `out`.
@inline higher_propagate!(::Val{:second_order}, out, state, shock, past_idx, 𝐒, scr) =
    nonpruned_state_update_2nd_order!(out, state, past_idx, shock, scr.aug, scr.kk, 𝐒)
@inline higher_propagate!(::Val{:third_order}, out, state, shock, past_idx, 𝐒, scr) =
    nonpruned_state_update_3rd_order!(out, state, past_idx, shock, scr.aug, scr.kk, scr.kkk, 𝐒)
@inline higher_propagate!(::Val{:pruned_second_order}, out, state, shock, past_idx, 𝐒, scr) =
    pf_pruned_2nd!(out[1], out[2], state[1], state[2], past_idx, shock, scr.aug1, scr.aug2, scr.kk, 𝐒)
@inline higher_propagate!(::Val{:pruned_third_order}, out, state, shock, past_idx, 𝐒, scr) =
    pf_pruned_3rd!(out[1], out[2], out[3], state[1], state[2], state[3], past_idx, shock, scr.aug1, scr.aug1̂, scr.aug2, scr.aug3, scr.k11, scr.k12̂, scr.k111, 𝐒)

# Deep-copy a particle (flat vector or vector-of-components) into a preallocated slot.
@inline copy_particle!(dst::AbstractVector{Float64}, src::AbstractVector{Float64}) = copyto!(dst, src)
@inline function copy_particle!(dst::AbstractVector{<:AbstractVector}, src::AbstractVector{<:AbstractVector})
    @inbounds for c in eachindex(dst)
        copyto!(dst[c], src[c])
    end
    return dst
end

# A zeroed particle with the same shape as `template` (for the second pool).
zeros_like_particle(template::AbstractVector{Float64}) = zeros(Float64, length(template))
zeros_like_particle(template::AbstractVector{<:AbstractVector}) = [zeros(Float64, length(c)) for c in template]

# Concretely-typed initial particle cloud (avoids the type-unstable `Union` that
# `initialise_particles` returns because its `pruning` branch is a runtime Bool).
# Dispatching on `Val{algo}` fixes the element type per specialization so the hot
# loop is allocation-free. The RNG draw order matches `initialise_particles`.
function init_higher_particles(::Union{Val{:second_order},Val{:third_order}}, rng, state, L, n_particles, nVars)
    mean0 = Vector{Float64}(state)
    return Vector{Float64}[mean0 .+ L * randn(rng, nVars) for _ in 1:n_particles]
end
function init_higher_particles(::Val{:pruned_second_order}, rng, state, L, n_particles, nVars)
    m1 = Vector{Float64}(state[1]); m2 = Vector{Float64}(state[2])
    return Vector{Vector{Float64}}[[m1 .+ L * randn(rng, nVars), copy(m2)] for _ in 1:n_particles]
end
function init_higher_particles(::Val{:pruned_third_order}, rng, state, L, n_particles, nVars)
    m1 = Vector{Float64}(state[1]); m2 = Vector{Float64}(state[2]); m3 = Vector{Float64}(state[3])
    return Vector{Vector{Float64}}[[m1 .+ L * randn(rng, nVars), copy(m2), copy(m3)] for _ in 1:n_particles]
end

# Concrete pool type per algorithm, used to type-assert the initial cloud so the
# large kwarg method body doesn't lose the element type (which would send the
# function-barrier call through dynamic dispatch).
particle_pool_type(::Union{Val{:second_order},Val{:third_order}}) = Vector{Vector{Float64}}
particle_pool_type(::Union{Val{:pruned_second_order},Val{:pruned_third_order}}) = Vector{Vector{Vector{Float64}}}

# Full model state a particle reports to the measurement, without allocation:
# the state itself for non-pruned orders, the sum of components (into `full_buf`)
# for pruned orders. Dispatches on the particle representation (type-stable).
@inline measurement_full(p::AbstractVector{Float64}, full_buf) = p
@inline function measurement_full(p::AbstractVector{<:AbstractVector}, full_buf)
    fill!(full_buf, 0.0)
    @inbounds for c in eachindex(p)
        pc = p[c]
        for i in eachindex(full_buf)
            full_buf[i] += pc[i]
        end
    end
    return full_buf
end


# ── Bootstrap (sequential importance resampling) particle filter ─────────────

function run_particle_filter(::Val{algo},
                             ::Val{:bootstrap},
                             observables_index::Vector{Int},
                             𝐒,
                             data_in_deviations::AbstractMatrix,
                             constants::constants,
                             state,
                             𝓂::ℳ,
                             measurement_error::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
                             obs_idx_per_t::Vector{Vector{Int}},
                             has_missing::Bool;
                             n_particles::Int = DEFAULT_N_PARTICLES,
                             particle_resampling::Symbol = DEFAULT_PARTICLE_RESAMPLING,
                             particle_resampling_threshold::Real = DEFAULT_PARTICLE_RESAMPLING_THRESHOLD,
                             particle_initial_state_scaling::Real = DEFAULT_PARTICLE_INITIAL_STATE_SCALING,
                             particle_rng::Random.AbstractRNG = Random.default_rng(),
                             presample_periods::Int = 0,
                             initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
                             on_failure_loglikelihood::Real = -Inf,
                             tempering_target_ratio::Real = DEFAULT_TEMPERING_TARGET_RATIO,
                             tempering_mh_steps::Int = DEFAULT_TEMPERING_MH_STEPS,
                             tempering_max_stages::Int = DEFAULT_TEMPERING_MAX_STAGES,
                             tempering_mh_scale::Real = DEFAULT_TEMPERING_MH_SCALE,
                             opts::CalculationOptions = merge_calculation_options())::Float64 where {algo}
    T = constants.post_model_macro
    rng = particle_rng
    resampling = particle_resampling
    resampling_threshold = particle_resampling_threshold
    initial_state_prior_scaling_factor = particle_initial_state_scaling
    nVars = T.nVars
    nExo  = T.nExo
    nT    = size(data_in_deviations, 2)
    presample_periods = normalize_presample_periods(presample_periods, nT)
    log2pi = log(2π)

    me_var = build_particle_measurement_error(measurement_error)
    assert_positive_measurement_error(me_var)

    past_idx = T.past_not_future_and_mixed_idx
    𝐒f = [Matrix{Float64}(S) for S in 𝐒]
    scr = build_higher_scratch(Val(algo), T.nPast_not_future_and_mixed, nExo)

    Σ, _ = particle_initial_state_covariance(𝓂, T, opts, initial_covariance)
    L = particle_initial_cloud_factor(Σ, Float64(initial_state_prior_scaling_factor))
    particles  = init_higher_particles(Val(algo), rng, state, L, n_particles, nVars)::particle_pool_type(Val(algo))
    particles2 = [zeros_like_particle(particles[1]) for _ in 1:n_particles]

    return bootstrap_higher_loop(Val(algo), particles, particles2, 𝐒f, scr, past_idx,
                                 nVars, nExo, nT, presample_periods, observables_index,
                                 data_in_deviations, obs_idx_per_t, has_missing, me_var,
                                 resampling, resampling_threshold, rng, on_failure_loglikelihood, log2pi)
end

# The bootstrap recursion, period by period. One iteration of the loop below is
# the textbook predict / weight / resample cycle:
#
#   1. PREDICT  draw a fresh shock for every particle and push it through the
#               model's state transition. The cloud now represents p(xₜ | y₁..ₜ₋₁).
#   2. WEIGHT   score each particle by how well it explains today's observation,
#               p(yₜ | xₜ). Averaging those scores over the (weighted) cloud is an
#               unbiased estimate of the period's likelihood contribution, which
#               is what gets accumulated into `loglik`.
#   3. RESAMPLE if the weights have become too uneven, replace the weighted cloud
#               by an equally weighted one so the next predict step spends its
#               particles where the probability mass actually is.
#
# Arguments (the ones that are not self-evident):
#   particles / particles2  two pools of the same shape, used as a double buffer:
#                           we always write the propagated cloud into the spare
#                           pool and then swap, which avoids allocating per period.
#   𝐒f                      perturbation solution matrices, already densified.
#   scr                     preallocated kron/augmented-state scratch for the
#                           nonlinear transition (see `build_higher_scratch`).
#   past_idx                positions of the predetermined states inside a state
#                           vector — what the transition actually reads.
#   me_var                  per-observable measurement-error variances.
#   rows                    which observables are actually observed this period
#                           (all of them unless the data has holes).
#
# Function barrier: `particles`/`particles2` arrive with a concrete element type,
# so the hot loop specialises and runs allocation-free (the enclosing kwarg method
# body is too large for inference to keep the pool types).
function bootstrap_higher_loop(::Val{algo}, particles, particles2, 𝐒f, scr, past_idx,
                               nVars, nExo, nT, presample_periods, observables_index,
                               data_in_deviations, obs_idx_per_t, has_missing, me_var,
                               resampling, resampling_threshold, rng, on_failure_loglikelihood, log2pi) where {algo}
    n_particles = length(particles)
    shock    = Vector{Float64}(undef, nExo)     # one draw of structural shocks, reused
    full_buf = Vector{Float64}(undef, nVars)    # summed state of a pruned particle
    W        = fill(1.0 / n_particles, n_particles)  # normalised importance weights
    logdens  = Vector{Float64}(undef, n_particles)   # log p(yₜ | xₜ) per particle
    idx      = Vector{Int}(undef, n_particles)       # ancestor indices from resampling
    bins     = Vector{Float64}(undef, n_particles)   # cumulative-weight scratch
    loglik   = 0.0

    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view data_in_deviations[:, t]

        if isempty(rows)
            # No observation this period: propagate only, weights unchanged.
            @inbounds for p in 1:n_particles
                Random.randn!(rng, shock)
                higher_propagate!(Val(algo), particles2[p], particles[p], shock, past_idx, 𝐒f, scr)
            end
            particles, particles2 = particles2, particles
            continue
        end

        # 1. PREDICT + 2. SCORE, fused into one pass over the cloud: draw this
        #    particle's shocks, push it one period forward, and evaluate how well
        #    the resulting state explains today's observation.
        @inbounds for p in 1:n_particles
            Random.randn!(rng, shock)
            higher_propagate!(Val(algo), particles2[p], particles[p], shock, past_idx, 𝐒f, scr)
            full = measurement_full(particles2[p], full_buf)
            logdens[p] = particle_log_measurement_density(full, data_col, observables_index, me_var, rows, log2pi)
        end
        particles, particles2 = particles2, particles   # propagated cloud becomes current

        # The period's likelihood contribution is log Σₚ Wₚ·p(yₜ|xₜᵖ). Factor out
        # the largest log-density first (log-sum-exp) so the exponentials cannot
        # underflow to zero when every particle fits the data poorly.
        m = maximum(logdens)
        if !isfinite(m)
            # every particle is impossible (or the model blew up): give up cleanly
            return Float64(on_failure_loglikelihood)
        end

        s = 0.0
        @inbounds for p in 1:n_particles
            s += W[p] * exp(logdens[p] - m)
        end
        if s <= 0 || !isfinite(s)
            return Float64(on_failure_loglikelihood)
        end

        ll_t = m + log(s)
        if t > presample_periods            # presample periods only warm the cloud up
            loglik += ll_t
        end

        # Bayes update of the weights: Wₚ ∝ Wₚ · p(yₜ|xₜᵖ), normalised by `s`.
        @inbounds for p in 1:n_particles
            W[p] = W[p] * exp(logdens[p] - m) / s
        end

        # 3. RESAMPLE, but only once the cloud has actually degenerated. Doing it
        #    every period would add resampling noise for nothing; doing it never
        #    would leave all the weight on a single particle within a few periods.
        if effective_sample_size(W) < resampling_threshold * n_particles
            particle_resample_indices!(idx, bins, rng, W, resampling)
            @inbounds for j in 1:n_particles
                copy_particle!(particles2[j], particles[idx[j]])
            end
            particles, particles2 = particles2, particles
            fill!(W, 1.0 / n_particles)     # survivors are equally likely again
        end
    end

    return isfinite(loglik) ? loglik : Float64(on_failure_loglikelihood)
end


# ── Shared measurement helpers for the auxiliary and tempered filters ────────

# Quadratic form eᵀH⁻¹e over the observed rows, with diagonal H (variances
# `me_var`). Returns Inf on a non-finite prediction (an impossible particle).
@inline function particle_quadratic_form(full::AbstractVector, data_col, observables_index, me_var, rows)
    q = 0.0
    @inbounds for r in rows
        f = full[observables_index[r]]
        isfinite(f) || return Inf
        v = data_col[r] - f
        q += v * v / me_var[r]
    end
    return q
end

# Log normalising constant of the Gaussian measurement density over the observed
# rows: -½(dₒ·log2π + Σ log me_var[r]).
@inline function particle_measurement_logZ(me_var, rows, log2pi::Float64)
    z = 0.0
    @inbounds for r in rows
        z += log2pi + log(me_var[r])
    end
    return -0.5 * z
end


# ── Auxiliary particle filter (Pitt & Shephard, 1999) ────────────────────────
#
# The problem it fixes. The bootstrap filter propagates every particle blindly —
# it draws shocks from the prior, *then* looks at the observation. Particles that
# were already heading somewhere the data rules out are propagated anyway and
# then killed by a near-zero weight, so a large part of the cloud is wasted. The
# more informative the observation (small measurement error, many observables),
# the more wasteful this is.
#
# The idea. Peek at the observation *before* choosing which ancestors to
# propagate. For each ancestor compute a cheap preview of how plausible it is
# going to look next period — here the measurement density at its transition
# mean (the zero-shock prediction), inflated by the shock-induced predictive
# variance so the preview is not artificially sharp. Resample ancestors in
# proportion to weight × preview, so parents likely to produce good children get
# more offspring, and only then draw shocks and propagate.
#
# Keeping it honest. Selecting ancestors with a preview biases the cloud, so the
# second stage divides it back out: each child's weight is the true measurement
# density divided by the preview used to pick its parent. The preview cancels
# exactly, leaving an unbiased likelihood estimate whatever preview is used — a
# bad preview costs efficiency, never correctness.
#
# When it helps. Most when the observation is informative but the one-step-ahead
# state is well predicted by its mean. It costs roughly one extra transition
# evaluation per particle per period, so with a weak signal (large measurement
# error) the plain bootstrap filter is the better trade.

function run_particle_filter(::Val{algo},
                             ::Val{:auxiliary},
                             observables_index::Vector{Int},
                             𝐒,
                             data_in_deviations::AbstractMatrix,
                             constants::constants,
                             state,
                             𝓂::ℳ,
                             measurement_error::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
                             obs_idx_per_t::Vector{Vector{Int}},
                             has_missing::Bool;
                             n_particles::Int = DEFAULT_N_PARTICLES,
                             particle_resampling::Symbol = DEFAULT_PARTICLE_RESAMPLING,
                             particle_resampling_threshold::Real = DEFAULT_PARTICLE_RESAMPLING_THRESHOLD,
                             particle_initial_state_scaling::Real = DEFAULT_PARTICLE_INITIAL_STATE_SCALING,
                             particle_rng::Random.AbstractRNG = Random.default_rng(),
                             presample_periods::Int = 0,
                             initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
                             on_failure_loglikelihood::Real = -Inf,
                             tempering_target_ratio::Real = DEFAULT_TEMPERING_TARGET_RATIO,
                             tempering_mh_steps::Int = DEFAULT_TEMPERING_MH_STEPS,
                             tempering_max_stages::Int = DEFAULT_TEMPERING_MAX_STAGES,
                             tempering_mh_scale::Real = DEFAULT_TEMPERING_MH_SCALE,
                             opts::CalculationOptions = merge_calculation_options())::Float64 where {algo}
    T = constants.post_model_macro
    rng = particle_rng
    resampling = particle_resampling
    resampling_threshold = particle_resampling_threshold
    initial_state_prior_scaling_factor = particle_initial_state_scaling
    nVars = T.nVars
    nExo  = T.nExo
    nT    = size(data_in_deviations, 2)
    presample_periods = normalize_presample_periods(presample_periods, nT)
    log2pi = log(2π)

    me_var = build_particle_measurement_error(measurement_error)
    assert_positive_measurement_error(me_var)

    past_idx = T.past_not_future_and_mixed_idx
    𝐒f = [Matrix{Float64}(S) for S in 𝐒]
    scr = build_higher_scratch(Val(algo), T.nPast_not_future_and_mixed, nExo)

    # One-step-ahead predictive variance of each observable due to the structural
    # shocks (diagonal of Cₒ BBᵀ Cₒᵀ from the first-order shock loading), used to
    # scale the auxiliary first-stage weights. Evaluating the predictive density
    # at the transition mean alone would be near-degenerate when the observable is
    # shock-driven; inflating by the shock spread keeps the proxy well-conditioned.
    nPast = T.nPast_not_future_and_mixed
    S₁cache = 𝓂.caches.first_order_solution_matrix
    pred_var = Float64[sum(abs2, @view S₁cache[observables_index[i], nPast+1:end]) for i in eachindex(observables_index)] .+ me_diagonal(me_var)

    Σ, _ = particle_initial_state_covariance(𝓂, T, opts, initial_covariance)
    L = particle_initial_cloud_factor(Σ, Float64(initial_state_prior_scaling_factor))
    particles  = init_higher_particles(Val(algo), rng, state, L, n_particles, nVars)::particle_pool_type(Val(algo))
    particles2 = [zeros_like_particle(particles[1]) for _ in 1:n_particles]
    mu_particle = zeros_like_particle(particles[1])

    return auxiliary_higher_loop(Val(algo), particles, particles2, mu_particle, 𝐒f, scr, past_idx,
                                 nVars, nExo, nT, presample_periods, observables_index,
                                 data_in_deviations, obs_idx_per_t, has_missing, me_var, pred_var,
                                 resampling, rng, on_failure_loglikelihood, log2pi)
end

# Function barrier for the auxiliary loop (see `bootstrap_higher_loop`).
function auxiliary_higher_loop(::Val{algo}, particles, particles2, mu_particle, 𝐒f, scr, past_idx,
                               nVars, nExo, nT, presample_periods, observables_index,
                               data_in_deviations, obs_idx_per_t, has_missing, me_var, pred_var,
                               resampling, rng, on_failure_loglikelihood, log2pi) where {algo}
    n_particles = length(particles)
    zero_shock = zeros(Float64, nExo)
    shock = Vector{Float64}(undef, nExo)
    full_buf = Vector{Float64}(undef, nVars)
    W = fill(1.0 / n_particles, n_particles)
    logg̃ = Vector{Float64}(undef, n_particles)   # first-stage predictive log-density
    logw = Vector{Float64}(undef, n_particles)    # second-stage log-weight
    λ    = Vector{Float64}(undef, n_particles)
    idx  = Vector{Int}(undef, n_particles)
    bins = Vector{Float64}(undef, n_particles)
    loglik = 0.0

    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view data_in_deviations[:, t]

        # First stage: predictive density at the transition mean (zero shock),
        # spread by the shock-induced predictive variance `pred_var`.
        @inbounds for p in 1:n_particles
            higher_propagate!(Val(algo), mu_particle, particles[p], zero_shock, past_idx, 𝐒f, scr)
            μ = measurement_full(mu_particle, full_buf)
            logg̃[p] = particle_log_measurement_density(μ, data_col, observables_index, pred_var, rows, log2pi)
        end

        # First-stage (auxiliary) weights λ ∝ W · g̃, and κ = Σ W·g̃.
        mλ = -Inf
        @inbounds for p in 1:n_particles
            lλ = log(W[p]) + logg̃[p]
            mλ = lλ > mλ ? lλ : mλ
        end
        if !isfinite(mλ)
            return Float64(on_failure_loglikelihood)
        end
        sλ = 0.0
        @inbounds for p in 1:n_particles
            sλ += exp(log(W[p]) + logg̃[p] - mλ)
        end
        logκ = mλ + log(sλ)
        @inbounds for p in 1:n_particles
            λ[p] = exp(log(W[p]) + logg̃[p] - logκ)
        end

        # Resample ancestors ∝ λ, propagate with fresh shocks, second-stage weight
        # w = g(yₜ|xₜ) / g̃(ancestor).
        particle_resample_indices!(idx, bins, rng, λ, resampling)
        @inbounds for j in 1:n_particles
            a = idx[j]
            Random.randn!(rng, shock)
            higher_propagate!(Val(algo), particles2[j], particles[a], shock, past_idx, 𝐒f, scr)
            full = measurement_full(particles2[j], full_buf)
            logw[j] = particle_log_measurement_density(full, data_col, observables_index, me_var, rows, log2pi) - logg̃[a]
        end
        particles, particles2 = particles2, particles

        mw = maximum(logw)
        if !isfinite(mw)
            return Float64(on_failure_loglikelihood)
        end
        sw = 0.0
        @inbounds for j in 1:n_particles
            sw += exp(logw[j] - mw)
        end
        if sw <= 0 || !isfinite(sw)
            return Float64(on_failure_loglikelihood)
        end

        ll_t = logκ + (mw + log(sw) - log(n_particles))
        if t > presample_periods
            loglik += ll_t
        end

        logsw = mw + log(sw)
        @inbounds for j in 1:n_particles
            W[j] = exp(logw[j] - logsw)
        end
    end

    return isfinite(loglik) ? loglik : Float64(on_failure_loglikelihood)
end


# ── Tempered particle filter (Herbst & Schorfheide, 2019) ────────────────────
# Within each period the measurement information is introduced gradually through
# a bridging sequence 0 = φ₀ < φ₁ < … < φ_N = 1 (the measurement covariance is
# inflated to H/φ). Each stage reweights by the tempered density increment,
# resamples, and mutates the particles' shocks with a random-walk Metropolis step
# targeting the stage-φ posterior. This dramatically lowers the variance of the
# likelihood estimate relative to the bootstrap filter at equal particle count.

# Inefficiency ratio N·Σ(wᵖ)² / (Σwᵖ)² for the incremental weights
# wᵖ = exp(-(φ-φ_old)/2 · dᵖ). Increasing in φ, equal to 1 at φ = φ_old.
function tempered_inefficiency(φ::Float64, φ_old::Float64, d::Vector{Float64}, n_particles::Int)
    Δ = (φ - φ_old) / 2
    maxla = -Inf
    @inbounds for p in 1:n_particles
        la = isfinite(d[p]) ? -Δ * d[p] : -Inf
        maxla = la > maxla ? la : maxla
    end
    isfinite(maxla) || return Inf
    S1 = 0.0
    S2 = 0.0
    @inbounds for p in 1:n_particles
        if isfinite(d[p])
            e = exp(-Δ * d[p] - maxla)
            S1 += e
            S2 += e * e
        end
    end
    return S1 > 0 ? n_particles * S2 / (S1 * S1) : Inf
end

# Next tempering level in (φ_old, 1] targeting inefficiency `r_star` by bisection.
function tempered_next_phi(φ_old::Float64, d::Vector{Float64}, r_star::Float64, n_particles::Int)
    if tempered_inefficiency(1.0, φ_old, d, n_particles) <= r_star
        return 1.0
    end
    lo = φ_old
    hi = 1.0
    for _ in 1:100
        mid = 0.5 * (lo + hi)
        if tempered_inefficiency(mid, φ_old, d, n_particles) < r_star
            lo = mid
        else
            hi = mid
        end
        hi - lo < 1e-8 && break
    end
    return 0.5 * (lo + hi)
end

function run_particle_filter(::Val{algo},
                             ::Val{:tempered},
                             observables_index::Vector{Int},
                             𝐒,
                             data_in_deviations::AbstractMatrix,
                             constants::constants,
                             state,
                             𝓂::ℳ,
                             measurement_error::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
                             obs_idx_per_t::Vector{Vector{Int}},
                             has_missing::Bool;
                             n_particles::Int = DEFAULT_N_PARTICLES,
                             particle_resampling::Symbol = DEFAULT_PARTICLE_RESAMPLING,
                             particle_resampling_threshold::Real = DEFAULT_PARTICLE_RESAMPLING_THRESHOLD,
                             particle_initial_state_scaling::Real = DEFAULT_PARTICLE_INITIAL_STATE_SCALING,
                             particle_rng::Random.AbstractRNG = Random.default_rng(),
                             presample_periods::Int = 0,
                             initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
                             on_failure_loglikelihood::Real = -Inf,
                             tempering_target_ratio::Real = DEFAULT_TEMPERING_TARGET_RATIO,
                             tempering_mh_steps::Int = DEFAULT_TEMPERING_MH_STEPS,
                             tempering_max_stages::Int = DEFAULT_TEMPERING_MAX_STAGES,
                             tempering_mh_scale::Real = DEFAULT_TEMPERING_MH_SCALE,
                             opts::CalculationOptions = merge_calculation_options())::Float64 where {algo}
    T = constants.post_model_macro
    rng = particle_rng
    resampling = particle_resampling
    resampling_threshold = particle_resampling_threshold
    initial_state_prior_scaling_factor = particle_initial_state_scaling
    nVars = T.nVars
    nExo  = T.nExo
    nT    = size(data_in_deviations, 2)
    presample_periods = normalize_presample_periods(presample_periods, nT)
    log2pi = log(2π)

    me_var = build_particle_measurement_error(measurement_error)
    assert_positive_measurement_error(me_var)

    r_star = Float64(tempering_target_ratio)
    c = Float64(tempering_mh_scale)
    n_mh = tempering_mh_steps
    max_stages = tempering_max_stages

    past_idx = T.past_not_future_and_mixed_idx
    𝐒f = [Matrix{Float64}(S) for S in 𝐒]
    scr = build_higher_scratch(Val(algo), T.nPast_not_future_and_mixed, nExo)

    Σ, _ = particle_initial_state_covariance(𝓂, T, opts, initial_covariance)
    L = particle_initial_cloud_factor(Σ, Float64(initial_state_prior_scaling_factor))

    # Double-buffered particle pools (ancestors and states + resample scratch).
    anc  = init_higher_particles(Val(algo), rng, state, L, n_particles, nVars)::particle_pool_type(Val(algo))
    return tempered_higher_loop(Val(algo), anc, 𝐒f, scr, past_idx, nVars, nExo, nT,
                                presample_periods, observables_index, data_in_deviations,
                                obs_idx_per_t, has_missing, me_var, resampling, r_star, c,
                                n_mh, max_stages, rng, on_failure_loglikelihood, log2pi)
end

# Function barrier for the tempered loop (see `bootstrap_higher_loop`).
function tempered_higher_loop(::Val{algo}, anc0, 𝐒f, scr, past_idx, nVars, nExo, nT,
                              presample_periods, observables_index, data_in_deviations,
                              obs_idx_per_t, has_missing, me_var, resampling, r_star, c,
                              n_mh, max_stages, rng, on_failure_loglikelihood, log2pi) where {algo}
    # `anc0` is captured (read-only) by the pool comprehensions below; the pools
    # that get swapped are separate locals, so nothing captured is ever reassigned
    # (which would force Julia to `Core.Box` it and make the loop type-unstable).
    anc0 = anc0::particle_pool_type(Val(algo))
    n_particles = length(anc0)
    anc  = [zeros_like_particle(anc0[1]) for _ in 1:n_particles]
    st   = [zeros_like_particle(anc0[1]) for _ in 1:n_particles]
    anc2 = [zeros_like_particle(anc0[1]) for _ in 1:n_particles]
    st2  = [zeros_like_particle(anc0[1]) for _ in 1:n_particles]
    @inbounds for p in 1:n_particles
        copy_particle!(anc[p], anc0[p])
    end
    sh   = [Vector{Float64}(undef, nExo) for _ in 1:n_particles]
    sh2  = [Vector{Float64}(undef, nExo) for _ in 1:n_particles]
    dv   = Vector{Float64}(undef, n_particles)
    dv2  = Vector{Float64}(undef, n_particles)

    sprop    = zeros_like_particle(anc0[1])
    eprop    = Vector{Float64}(undef, nExo)
    full_buf = Vector{Float64}(undef, nVars)
    logw = Vector{Float64}(undef, n_particles)
    Wn   = Vector{Float64}(undef, n_particles)
    idx  = Vector{Int}(undef, n_particles)
    bins = Vector{Float64}(undef, n_particles)
    loglik = 0.0

    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view data_in_deviations[:, t]
        d_obs = length(rows)

        # Bootstrap proposal: propagate every ancestor (`anc`) with a fresh shock.
        @inbounds for p in 1:n_particles
            Random.randn!(rng, sh[p])
        end
        @inbounds for p in 1:n_particles
            higher_propagate!(Val(algo), st[p], anc[p], sh[p], past_idx, 𝐒f, scr)
            dv[p] = particle_quadratic_form(measurement_full(st[p], full_buf), data_col, observables_index, me_var, rows)
        end
        if all(!isfinite, dv)
            return Float64(on_failure_loglikelihood)
        end

        period_ll = 0.0
        φ_old = 0.0
        stage = 0
        while φ_old < 1.0 - 1e-12 && stage < max_stages
            stage += 1
            φ_new = tempered_next_phi(φ_old, dv, r_star, n_particles)

            if φ_old == 0.0
                logZ = particle_measurement_logZ(me_var, rows, log2pi)
                @inbounds for p in 1:n_particles
                    logw[p] = logZ + 0.5 * d_obs * log(φ_new) - 0.5 * φ_new * dv[p]
                end
            else
                lr = 0.5 * d_obs * (log(φ_new) - log(φ_old))
                @inbounds for p in 1:n_particles
                    logw[p] = lr - 0.5 * (φ_new - φ_old) * dv[p]
                end
            end

            m = maximum(logw)
            if !isfinite(m)
                return Float64(on_failure_loglikelihood)
            end
            s = 0.0
            @inbounds for p in 1:n_particles
                s += exp(logw[p] - m)
            end
            if s <= 0 || !isfinite(s)
                return Float64(on_failure_loglikelihood)
            end
            period_ll += m + log(s) - log(n_particles)

            logsw = m + log(s)
            @inbounds for p in 1:n_particles
                Wn[p] = exp(logw[p] - logsw)
            end
            particle_resample_indices!(idx, bins, rng, Wn, resampling)
            @inbounds for j in 1:n_particles
                a = idx[j]
                copy_particle!(anc2[j], anc[a]); copyto!(sh2[j], sh[a]); copy_particle!(st2[j], st[a]); dv2[j] = dv[a]
            end
            anc, anc2 = anc2, anc
            sh,  sh2  = sh2,  sh
            st,  st2  = st2,  st
            dv,  dv2  = dv2,  dv

            # Mutation: random-walk Metropolis on the shocks, targeting the
            # stage-φ posterior  π(ε) ∝ N(ε;0,I) · exp(-φ/2 · e(ε)ᵀH⁻¹e(ε)).
            @inbounds for p in 1:n_particles
                shp = sh[p]
                for _ in 1:n_mh
                    Random.randn!(rng, eprop)
                    esq_old = 0.0
                    esq_new = 0.0
                    for e in 1:nExo
                        ep = shp[e] + c * eprop[e]
                        eprop[e] = ep
                        esq_new += ep * ep
                        esq_old += shp[e] * shp[e]
                    end
                    higher_propagate!(Val(algo), sprop, anc[p], eprop, past_idx, 𝐒f, scr)
                    dprop = particle_quadratic_form(measurement_full(sprop, full_buf), data_col, observables_index, me_var, rows)
                    logα = -0.5 * ((esq_new - esq_old) + φ_new * (dprop - dv[p]))
                    if log(rand(rng)) < logα
                        copyto!(shp, eprop)
                        copy_particle!(st[p], sprop)
                        dv[p] = dprop
                    end
                end
            end

            φ_old = φ_new
        end

        if t > presample_periods
            loglik += period_ll
        end
        # Carry the filtered states forward as next period's ancestors (copy so the
        # `anc`/`st` pool identities stay stable for inference).
        @inbounds for p in 1:n_particles
            copy_particle!(anc[p], st[p])
        end
    end

    return isfinite(loglik) ? loglik : Float64(on_failure_loglikelihood)
end


# ── Optimised first-order (linear) fast paths ────────────────────────────────
# For the linear state space the transition is  xₜ = A·xₜ₋₁[past] + B·εₜ. These
# `::Val{:first_order}` methods replace the type-unstable, per-call-allocating
# `state_update` closure with a typed, BLAS-backed (`mul!`), fully preallocated
# implementation. Particle pools are double-buffered so resampling and the
# tempered Metropolis mutation run in place, with no heap allocation in the hot
# loop. (Higher orders use the generic methods above.) Buffer-reuse for the
# resampling index/cumulative arrays follows LowLevelParticleFilters.jl.

# In-place resampling: ancestor indices are written into `idx`; `bins` is a
# cumulative-weight scratch used by the multinomial/residual schemes.
function systematic_resample_indices!(idx::Vector{Int}, rng::Random.AbstractRNG, W::AbstractVector{<:Real})
    N = length(W)
    u0 = rand(rng) / N
    c = W[1]; i = 1
    @inbounds for j in 1:N
        u = u0 + (j - 1) / N
        while u > c && i < N; i += 1; c += W[i]; end
        idx[j] = i
    end
    return idx
end

function stratified_resample_indices!(idx::Vector{Int}, rng::Random.AbstractRNG, W::AbstractVector{<:Real})
    N = length(W)
    c = W[1]; i = 1
    @inbounds for j in 1:N
        u = (j - 1 + rand(rng)) / N
        while u > c && i < N; i += 1; c += W[i]; end
        idx[j] = i
    end
    return idx
end

function multinomial_resample_indices!(idx::Vector{Int}, bins::Vector{Float64}, rng::Random.AbstractRNG, W::AbstractVector{<:Real})
    N = length(W)
    cumsum!(bins, W); bins[N] = one(eltype(bins))
    @inbounds for j in 1:N
        idx[j] = searchsortedfirst(bins, rand(rng))
    end
    return idx
end

function residual_resample_indices!(idx::Vector{Int}, bins::Vector{Float64}, rng::Random.AbstractRNG, W::AbstractVector{<:Real})
    N = length(W)
    k = 0
    @inbounds for i in 1:N
        ni = floor(Int, N * W[i])
        for _ in 1:ni; k += 1; idx[k] = i; end
    end
    R = N - k
    if R > 0
        s = 0.0
        @inbounds for i in 1:N
            bins[i] = N * W[i] - floor(N * W[i]); s += bins[i]
        end
        if s <= 0
            cumsum!(bins, W)
        else
            @inbounds for i in 1:N; bins[i] /= s; end
            cumsum!(bins, bins)
        end
        bins[N] = one(eltype(bins))
        @inbounds for _ in 1:R
            k += 1; idx[k] = searchsortedfirst(bins, rand(rng))
        end
    end
    return idx
end

@inline function particle_resample_indices!(idx::Vector{Int}, bins::Vector{Float64}, rng::Random.AbstractRNG, W::AbstractVector{<:Real}, scheme::Symbol)
    if scheme == :systematic
        systematic_resample_indices!(idx, rng, W)
    elseif scheme == :stratified
        stratified_resample_indices!(idx, rng, W)
    elseif scheme == :multinomial
        multinomial_resample_indices!(idx, bins, rng, W)
    elseif scheme == :residual
        residual_resample_indices!(idx, bins, rng, W)
    else
        error("Unknown resampling scheme `:$scheme`. Choose from `:systematic`, `:stratified`, `:multinomial`, `:residual`.")
    end
    return idx
end

# Typed, preallocated first-order transition xₜ = A·xₜ₋₁[past] + B·εₜ.
# Particles are stored as the columns of an nVars × N matrix so that the whole
# swarm is propagated with two BLAS gemm calls (Xₜ = A·Xₜ₋₁ + B·Eₜ) instead of
# N small gemv calls. `A` is the full nVars × nVars one-step transition (zero
# outside the predetermined-state columns), `B` the nVars × nExo shock loading.
struct LinearParticleTransition
    A::Matrix{Float64}
    B::Matrix{Float64}
end

function build_linear_particle_transition(𝐒::AbstractMatrix, T)
    nVars = T.nVars
    nPast = T.nPast_not_future_and_mixed
    S₁ = Matrix{Float64}(𝐒)
    A = zeros(Float64, nVars, nVars)
    @views A[:, T.past_not_future_and_mixed_idx] .= S₁[:, 1:nPast]
    B = Matrix{Float64}(@view S₁[:, nPast+1:end])
    return LinearParticleTransition(A, B)
end

# X₂ = A·X + B·E for the whole swarm at once (two gemm). Columns are particles.
@inline function propagate_batch!(X2::Matrix{Float64}, tr::LinearParticleTransition, X::Matrix{Float64}, E::Matrix{Float64})
    ℒ.mul!(X2, tr.A, X)
    ℒ.mul!(X2, tr.B, E, 1.0, 1.0)
    return X2
end

# Base = A·Anc (shock-independent part of the transition, one gemm; reused across
# Metropolis proposals which only vary the shock B·E term).
@inline function base_batch!(Base::Matrix{Float64}, tr::LinearParticleTransition, Anc::Matrix{Float64})
    ℒ.mul!(Base, tr.A, Anc)
    return Base
end

# Quadratic form eᵀH⁻¹e over the observed rows for particle column `p`.
@inline function linear_quadform_col(X::Matrix{Float64}, p::Int, data_col, observables_index, inv_me_var, rows)
    q = 0.0
    @inbounds for k in eachindex(rows)
        r = rows[k]
        f = X[observables_index[r], p]
        isfinite(f) || return Inf
        v = data_col[r] - f
        q += v * v * inv_me_var[r]
    end
    return q
end

# Same, for a correlated H: gather the innovation, then one triangular solve.
@inline function linear_quadform_col(X::Matrix{Float64}, p::Int, data_col, observables_index,
                                     me::DenseMeasurementError, rows)
    me_sync!(me, rows)
    v = me.buf
    @inbounds for k in eachindex(rows)
        r = rows[k]
        f = X[observables_index[r], p]
        isfinite(f) || return Inf
        v[k] = data_col[r] - f
    end
    return dense_me_quadform!(v, me.last_L)
end

# Copy column `src` of `X` into column `dst` of `Y` (contiguous, allocation-free).
@inline function copy_col!(Y::Matrix{Float64}, dst::Int, X::Matrix{Float64}, src::Int)
    @inbounds for i in axes(X, 1)
        Y[i, dst] = X[i, src]
    end
    return Y
end

# Draw the initial cloud into the columns of X (nVars × N): X = mean0 .+ L·Z.
function init_linear_particles!(X::Matrix{Float64}, rng::Random.AbstractRNG,
                                mean0::AbstractVector{Float64}, L::Matrix{Float64}, Z::Matrix{Float64})
    Random.randn!(rng, Z)
    ℒ.mul!(X, L, Z)
    @inbounds for p in axes(X, 2), i in axes(X, 1)
        X[i, p] += mean0[i]
    end
    return X
end

function run_particle_filter(::Val{:first_order},
                             ::Val{:bootstrap},
                             observables_index::Vector{Int},
                             𝐒,
                             data_in_deviations::AbstractMatrix,
                             constants::constants,
                             state,
                             𝓂::ℳ,
                             measurement_error::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
                             obs_idx_per_t::Vector{Vector{Int}},
                             has_missing::Bool;
                             n_particles::Int = DEFAULT_N_PARTICLES,
                             particle_resampling::Symbol = DEFAULT_PARTICLE_RESAMPLING,
                             particle_resampling_threshold::Real = DEFAULT_PARTICLE_RESAMPLING_THRESHOLD,
                             particle_initial_state_scaling::Real = DEFAULT_PARTICLE_INITIAL_STATE_SCALING,
                             particle_rng::Random.AbstractRNG = Random.default_rng(),
                             presample_periods::Int = 0,
                             initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
                             on_failure_loglikelihood::Real = -Inf,
                             tempering_target_ratio::Real = DEFAULT_TEMPERING_TARGET_RATIO,
                             tempering_mh_steps::Int = DEFAULT_TEMPERING_MH_STEPS,
                             tempering_max_stages::Int = DEFAULT_TEMPERING_MAX_STAGES,
                             tempering_mh_scale::Real = DEFAULT_TEMPERING_MH_SCALE,
                             opts::CalculationOptions = merge_calculation_options())::Float64
    T = constants.post_model_macro
    rng = particle_rng
    resampling = particle_resampling
    resampling_threshold = particle_resampling_threshold
    initial_state_prior_scaling_factor = particle_initial_state_scaling
    nVars = T.nVars
    nExo  = T.nExo
    nT    = size(data_in_deviations, 2)
    presample_periods = normalize_presample_periods(presample_periods, nT)
    log2pi = log(2π)

    me_var = build_particle_measurement_error(measurement_error)
    assert_positive_measurement_error(me_var)
    inv_me_var = me_inverse_diagonal(me_var)

    tr = build_linear_particle_transition(𝐒, T)
    Σ, _ = particle_initial_state_covariance(𝓂, T, opts, initial_covariance)
    L = particle_initial_cloud_factor(Σ, Float64(initial_state_prior_scaling_factor))

    # Cloud and scratch come from the model's workspace, so a sampler that calls
    # this thousands of times pays for them once (see `ensure_particle_workspace!`).
    pws = ensure_particle_workspace!(𝓂.workspaces, nVars, nExo, n_particles)
    X, X2, Z = pws.X, pws.X2, pws.Anc
    E = pws.E
    W, logdens, idx, bins = pws.W, pws.logdens, pws.idx, pws.bins
    fill!(W, 1.0 / n_particles)

    mean0 = state isa AbstractVector{<:AbstractVector} ? Vector{Float64}(state[1]) : Vector{Float64}(state)
    init_linear_particles!(X, rng, mean0, L, Z)

    loglik = 0.0
    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view data_in_deviations[:, t]

        Random.randn!(rng, E)
        propagate_batch!(X2, tr, X, E)
        X, X2 = X2, X

        isempty(rows) && continue

        logZ = particle_measurement_logZ(me_var, rows, log2pi)
        @inbounds for p in 1:n_particles
            logdens[p] = logZ - 0.5 * linear_quadform_col(X, p, data_col, observables_index, inv_me_var, rows)
        end

        m = maximum(logdens)
        if !isfinite(m)
            return Float64(on_failure_loglikelihood)
        end
        s = 0.0
        @inbounds for p in 1:n_particles
            s += W[p] * exp(logdens[p] - m)
        end
        if s <= 0 || !isfinite(s)
            return Float64(on_failure_loglikelihood)
        end

        ll_t = m + log(s)
        if t > presample_periods
            loglik += ll_t
        end

        @inbounds for p in 1:n_particles
            W[p] = W[p] * exp(logdens[p] - m) / s
        end

        if effective_sample_size(W) < resampling_threshold * n_particles
            particle_resample_indices!(idx, bins, rng, W, resampling)
            @inbounds for j in 1:n_particles
                copy_col!(X2, j, X, idx[j])
            end
            X, X2 = X2, X
            fill!(W, 1.0 / n_particles)
        end
    end

    return isfinite(loglik) ? loglik : Float64(on_failure_loglikelihood)
end


function run_particle_filter(::Val{:first_order},
                             ::Val{:auxiliary},
                             observables_index::Vector{Int},
                             𝐒,
                             data_in_deviations::AbstractMatrix,
                             constants::constants,
                             state,
                             𝓂::ℳ,
                             measurement_error::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
                             obs_idx_per_t::Vector{Vector{Int}},
                             has_missing::Bool;
                             n_particles::Int = DEFAULT_N_PARTICLES,
                             particle_resampling::Symbol = DEFAULT_PARTICLE_RESAMPLING,
                             particle_resampling_threshold::Real = DEFAULT_PARTICLE_RESAMPLING_THRESHOLD,
                             particle_initial_state_scaling::Real = DEFAULT_PARTICLE_INITIAL_STATE_SCALING,
                             particle_rng::Random.AbstractRNG = Random.default_rng(),
                             presample_periods::Int = 0,
                             initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
                             on_failure_loglikelihood::Real = -Inf,
                             tempering_target_ratio::Real = DEFAULT_TEMPERING_TARGET_RATIO,
                             tempering_mh_steps::Int = DEFAULT_TEMPERING_MH_STEPS,
                             tempering_max_stages::Int = DEFAULT_TEMPERING_MAX_STAGES,
                             tempering_mh_scale::Real = DEFAULT_TEMPERING_MH_SCALE,
                             opts::CalculationOptions = merge_calculation_options())::Float64
    T = constants.post_model_macro
    rng = particle_rng
    resampling = particle_resampling
    resampling_threshold = particle_resampling_threshold
    initial_state_prior_scaling_factor = particle_initial_state_scaling
    nVars = T.nVars
    nExo  = T.nExo
    nT    = size(data_in_deviations, 2)
    presample_periods = normalize_presample_periods(presample_periods, nT)
    log2pi = log(2π)

    me_var = build_particle_measurement_error(measurement_error)
    assert_positive_measurement_error(me_var)
    inv_me_var = me_inverse_diagonal(me_var)

    tr = build_linear_particle_transition(𝐒, T)

    # Predictive variance of each observable (shock spread + measurement error).
    me_diag = me_diagonal(me_var)
    pred_var = Vector{Float64}(undef, length(observables_index))
    @inbounds for i in eachindex(observables_index)
        pred_var[i] = sum(abs2, @view tr.B[observables_index[i], :]) + me_diag[i]
    end
    inv_pred_var = 1.0 ./ pred_var

    Σ, _ = particle_initial_state_covariance(𝓂, T, opts, initial_covariance)
    L = particle_initial_cloud_factor(Σ, Float64(initial_state_prior_scaling_factor))

    pws = ensure_particle_workspace!(𝓂.workspaces, nVars, nExo, n_particles)
    X, X2, AncX = pws.X, pws.X2, pws.Anc
    E = pws.E
    W, logg̃, logw = pws.W, pws.logdens, pws.logw
    lam, idx, bins = pws.lam, pws.idx, pws.bins
    fill!(W, 1.0 / n_particles)

    mean0 = state isa AbstractVector{<:AbstractVector} ? Vector{Float64}(state[1]) : Vector{Float64}(state)
    init_linear_particles!(X, rng, mean0, L, AncX)

    loglik = 0.0
    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view data_in_deviations[:, t]
        logZ  = particle_measurement_logZ(me_var, rows, log2pi)
        logZp = particle_measurement_logZ(pred_var, rows, log2pi)

        # First stage: predictive density at the transition mean μ = A·X (one gemm).
        base_batch!(X2, tr, X)
        @inbounds for p in 1:n_particles
            logg̃[p] = logZp - 0.5 * linear_quadform_col(X2, p, data_col, observables_index, inv_pred_var, rows)
        end

        mλ = -Inf
        @inbounds for p in 1:n_particles
            lλ = log(W[p]) + logg̃[p]
            mλ = lλ > mλ ? lλ : mλ
        end
        if !isfinite(mλ)
            return Float64(on_failure_loglikelihood)
        end
        sλ = 0.0
        @inbounds for p in 1:n_particles
            sλ += exp(log(W[p]) + logg̃[p] - mλ)
        end
        logκ = mλ + log(sλ)
        @inbounds for p in 1:n_particles
            lam[p] = exp(log(W[p]) + logg̃[p] - logκ)
        end

        # Resample ancestors ∝ λ, gather them, and propagate with fresh shocks.
        particle_resample_indices!(idx, bins, rng, lam, resampling)
        @inbounds for j in 1:n_particles
            copy_col!(AncX, j, X, idx[j])
        end
        Random.randn!(rng, E)
        propagate_batch!(X2, tr, AncX, E)
        @inbounds for j in 1:n_particles
            logw[j] = (logZ - 0.5 * linear_quadform_col(X2, j, data_col, observables_index, inv_me_var, rows)) - logg̃[idx[j]]
        end
        X, X2 = X2, X

        mw = maximum(logw)
        if !isfinite(mw)
            return Float64(on_failure_loglikelihood)
        end
        sw = 0.0
        @inbounds for j in 1:n_particles
            sw += exp(logw[j] - mw)
        end
        if sw <= 0 || !isfinite(sw)
            return Float64(on_failure_loglikelihood)
        end

        ll_t = logκ + (mw + log(sw) - log(n_particles))
        if t > presample_periods
            loglik += ll_t
        end

        logsw = mw + log(sw)
        @inbounds for j in 1:n_particles
            W[j] = exp(logw[j] - logsw)
        end
    end

    return isfinite(loglik) ? loglik : Float64(on_failure_loglikelihood)
end


function run_particle_filter(::Val{:first_order},
                             ::Val{:tempered},
                             observables_index::Vector{Int},
                             𝐒,
                             data_in_deviations::AbstractMatrix,
                             constants::constants,
                             state,
                             𝓂::ℳ,
                             measurement_error::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
                             obs_idx_per_t::Vector{Vector{Int}},
                             has_missing::Bool;
                             n_particles::Int = DEFAULT_N_PARTICLES,
                             particle_resampling::Symbol = DEFAULT_PARTICLE_RESAMPLING,
                             particle_resampling_threshold::Real = DEFAULT_PARTICLE_RESAMPLING_THRESHOLD,
                             particle_initial_state_scaling::Real = DEFAULT_PARTICLE_INITIAL_STATE_SCALING,
                             particle_rng::Random.AbstractRNG = Random.default_rng(),
                             presample_periods::Int = 0,
                             initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
                             on_failure_loglikelihood::Real = -Inf,
                             tempering_target_ratio::Real = DEFAULT_TEMPERING_TARGET_RATIO,
                             tempering_mh_steps::Int = DEFAULT_TEMPERING_MH_STEPS,
                             tempering_max_stages::Int = DEFAULT_TEMPERING_MAX_STAGES,
                             tempering_mh_scale::Real = DEFAULT_TEMPERING_MH_SCALE,
                             opts::CalculationOptions = merge_calculation_options())::Float64
    T = constants.post_model_macro
    rng = particle_rng
    resampling = particle_resampling
    resampling_threshold = particle_resampling_threshold
    initial_state_prior_scaling_factor = particle_initial_state_scaling
    nVars = T.nVars
    nExo  = T.nExo
    nT    = size(data_in_deviations, 2)
    presample_periods = normalize_presample_periods(presample_periods, nT)
    log2pi = log(2π)

    me_var = build_particle_measurement_error(measurement_error)
    assert_positive_measurement_error(me_var)
    inv_me_var = me_inverse_diagonal(me_var)

    r_star = Float64(tempering_target_ratio)
    c = Float64(tempering_mh_scale)
    n_mh = tempering_mh_steps
    max_stages = tempering_max_stages

    tr = build_linear_particle_transition(𝐒, T)
    Σ, _ = particle_initial_state_covariance(𝓂, T, opts, initial_covariance)
    L = particle_initial_cloud_factor(Σ, Float64(initial_state_prior_scaling_factor))

    # Double-buffered particle pools (columns are particles), taken from the
    # model's workspace. The locals below are swapped in place of copying, which
    # leaves the workspace fields pointing at whichever buffer ends up where —
    # harmless, since every buffer is written before it is read.
    pws = ensure_particle_workspace!(𝓂.workspaces, nVars, nExo, n_particles)
    Anc, Anc2 = pws.Anc, pws.Anc2
    Sh,  Sh2  = pws.E,   pws.E2
    St,  St2  = pws.St,  pws.St2
    dv,  dv2  = pws.dv,  pws.dv2

    Base  = pws.X
    Sprop = pws.X2
    Eprop = pws.Eprop
    logw  = pws.logw
    Wn    = pws.Wn
    idx   = pws.idx
    bins  = pws.bins

    mean0 = state isa AbstractVector{<:AbstractVector} ? Vector{Float64}(state[1]) : Vector{Float64}(state)
    init_linear_particles!(St, rng, mean0, L, Anc2)

    loglik = 0.0
    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view data_in_deviations[:, t]
        d_obs = length(rows)

        # Ancestors = previous filtered states; propagate the whole swarm (2 gemm).
        copyto!(Anc, St)
        Random.randn!(rng, Sh)
        propagate_batch!(St, tr, Anc, Sh)
        @inbounds for p in 1:n_particles
            dv[p] = linear_quadform_col(St, p, data_col, observables_index, inv_me_var, rows)
        end
        if all(!isfinite, dv)
            return Float64(on_failure_loglikelihood)
        end

        logZ = particle_measurement_logZ(me_var, rows, log2pi)
        period_ll = 0.0
        φ_old = 0.0
        stage = 0
        while φ_old < 1.0 - 1e-12 && stage < max_stages
            stage += 1
            φ_new = tempered_next_phi(φ_old, dv, r_star, n_particles)

            if φ_old == 0.0
                @inbounds for p in 1:n_particles
                    logw[p] = logZ + 0.5 * d_obs * log(φ_new) - 0.5 * φ_new * dv[p]
                end
            else
                lr = 0.5 * d_obs * (log(φ_new) - log(φ_old))
                @inbounds for p in 1:n_particles
                    logw[p] = lr - 0.5 * (φ_new - φ_old) * dv[p]
                end
            end

            m = maximum(logw)
            if !isfinite(m)
                return Float64(on_failure_loglikelihood)
            end
            s = 0.0
            @inbounds for p in 1:n_particles
                s += exp(logw[p] - m)
            end
            if s <= 0 || !isfinite(s)
                return Float64(on_failure_loglikelihood)
            end
            period_ll += m + log(s) - log(n_particles)

            logsw = m + log(s)
            @inbounds for p in 1:n_particles
                Wn[p] = exp(logw[p] - logsw)
            end

            particle_resample_indices!(idx, bins, rng, Wn, resampling)
            @inbounds for j in 1:n_particles
                a = idx[j]
                copy_col!(Anc2, j, Anc, a); copy_col!(Sh2, j, Sh, a); copy_col!(St2, j, St, a); dv2[j] = dv[a]
            end
            Anc, Anc2 = Anc2, Anc
            Sh,  Sh2  = Sh2,  Sh
            St,  St2  = St2,  St
            dv,  dv2  = dv2,  dv

            # Metropolis mutation on the shocks. Base = A·Anc is shock-independent
            # (one gemm); each proposal only recomputes the batched shock term
            # B·Eprop before per-particle accept/reject.
            base_batch!(Base, tr, Anc)
            for _ in 1:n_mh
                Random.randn!(rng, Eprop)
                @inbounds for k in eachindex(Eprop)
                    Eprop[k] = Sh[k] + c * Eprop[k]
                end
                ℒ.mul!(Sprop, tr.B, Eprop)
                Sprop .+= Base
                @inbounds for p in 1:n_particles
                    dprop = linear_quadform_col(Sprop, p, data_col, observables_index, inv_me_var, rows)
                    esq_old = 0.0
                    esq_new = 0.0
                    for e in 1:nExo
                        so = Sh[e, p]; sn = Eprop[e, p]
                        esq_old += so * so
                        esq_new += sn * sn
                    end
                    logα = -0.5 * ((esq_new - esq_old) + φ_new * (dprop - dv[p]))
                    if log(rand(rng)) < logα
                        copy_col!(Sh, p, Eprop, p)
                        copy_col!(St, p, Sprop, p)
                        dv[p] = dprop
                    end
                end
            end

            φ_old = φ_new
        end

        if t > presample_periods
            loglik += period_ll
        end
    end

    return isfinite(loglik) ? loglik : Float64(on_failure_loglikelihood)
end


# ── Filtered estimates from a particle filter ────────────────────────────────
#
# `filter_data_with_model` is the entry point behind `get_model_estimates`,
# `get_estimated_variables`, `get_estimated_shocks` and the estimate plots. For
# the particle filters it returns the *filtered* moments of the particle cloud:
#
#   variables            mean of the cloud       (states)
#   standard_deviations  spread of the cloud     (states)
#   shocks               mean of the drawn shocks
#
# With `smooth = false` these condition on the past only, E[xₜ | y₁..ₜ]. With
# `smooth = true` they condition on the whole sample, E[xₜ | y₁..T], obtained by
# the genealogy smoother in `smooth_particle_trajectories!` below.
#
# All three particle variants target the same filtering distribution p(xₜ|y₁..ₜ)
# — they differ only in how they get there — so the recursion below is shared.
# `:bootstrap_particle` and `:auxiliary_particle` use the plain
# predict/weight/resample step: the auxiliary filter's look-ahead proposal changes
# the *variance* of the likelihood estimate, not the cloud it leaves behind, so
# there is nothing extra to do for the moments. `:tempered_particle` does change
# the cloud: within each period it bridges from the prior to the full measurement
# density in stages, resampling and rejuvenating the shocks by random-walk
# Metropolis at each one. That leaves a cloud with far more distinct support
# points at the same `n_particles`, which is exactly what the moments (and the
# smoother, which walks the genealogy) benefit from — so the tempering controls
# act here too.
#
# `decomposition` is the shock decomposition of whichever shock path was
# produced — filtered when `smooth = false`, smoothed when `smooth = true`.
@unstable function filter_data_with_model(𝓂::ℳ,
    data_in_deviations::KeyedArray{Float64},
    ::Val{algo},
    ::Val{pf};
    warmup_iterations::Int = 0,
    opts::CalculationOptions = merge_calculation_options(),
    smooth::Bool = true,
    tempering_target_ratio::Real = DEFAULT_TEMPERING_TARGET_RATIO,
    tempering_mh_steps::Int = DEFAULT_TEMPERING_MH_STEPS,
    tempering_max_stages::Int = DEFAULT_TEMPERING_MAX_STAGES,
    tempering_mh_scale::Real = DEFAULT_TEMPERING_MH_SCALE,
    measurement_error::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
    n_particles::Int = DEFAULT_N_PARTICLES,
    particle_resampling::Symbol = DEFAULT_PARTICLE_RESAMPLING,
    particle_resampling_threshold::Real = DEFAULT_PARTICLE_RESAMPLING_THRESHOLD,
    particle_initial_state_scaling::Real = DEFAULT_PARTICLE_INITIAL_STATE_SCALING,
    particle_rng::Random.AbstractRNG = Random.default_rng(),
    marginal_contribution::Bool = false,
    initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical) where {algo, pf}

    @assert pf ∈ PARTICLE_FILTERS "`filter_data_with_model` was dispatched to the particle path with `filter = :$(pf)`."

    obs_axis = collect(axiskeys(data_in_deviations, 1))
    observables = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    constants = 𝓂.constants
    T = constants.post_model_macro
    nVars = T.nVars
    nExo  = T.nExo
    past_idx = T.past_not_future_and_mixed_idx

    ss_names = constants.post_complete_parameters.SS_and_pars_names
    observables_index = convert(Vector{Int}, indexin(observables, ss_names))

    dat = missing_data_to_nan(collect(data_in_deviations))
    obs_idx_per_t, has_missing = build_obs_index(dat)
    nT = size(dat, 2)

    # `measurement_error` arrives already resolved (a variance vector or a
    # covariance matrix) — `:auto` is handled by the user-facing entry points.
    me_var = build_particle_measurement_error(measurement_error)
    assert_positive_measurement_error(me_var)

    # solution matrices and the initial state, exactly as the likelihood path builds them
    _, _, 𝐒, state, solved = get_relevant_steady_state_and_state_update(Val(algo), 𝓂.parameter_values, 𝓂, opts = opts)
    @assert solved "Could not solve the model for `algorithm = $(algo)`; cannot run the particle filter."

    Σ, _ = particle_initial_state_covariance(𝓂, T, opts, initial_covariance)
    L = particle_initial_cloud_factor(Σ, Float64(particle_initial_state_scaling))

    log2pi = log(2π)
    rng = particle_rng

    # storage for the filtered moments
    variables  = zeros(nVars, nT)
    stds       = zeros(nVars, nT)
    shocks_out = zeros(nExo, nT)

    W       = fill(1.0 / n_particles, n_particles)
    logdens = Vector{Float64}(undef, n_particles)
    idx     = Vector{Int}(undef, n_particles)
    bins    = Vector{Float64}(undef, n_particles)
    shocks  = [Vector{Float64}(undef, nExo) for _ in 1:n_particles]
    full_buf = Vector{Float64}(undef, nVars)

    if algo == :first_order
        tr = build_linear_particle_transition(𝐒, T)
        mean0 = state isa AbstractVector{<:AbstractVector} ? Vector{Float64}(state[1]) : Vector{Float64}(state)
        parts  = [mean0 .+ L * randn(rng, nVars) for _ in 1:n_particles]
        parts2 = [zeros(Float64, nVars) for _ in 1:n_particles]
        propagate! = (out, prev, sh) -> linear_propagate_estimates!(out, tr, prev, sh)
    else
        𝐒f = [Matrix{Float64}(S) for S in 𝐒]
        scr = build_higher_scratch(Val(algo), T.nPast_not_future_and_mixed, nExo)
        parts  = init_higher_particles(Val(algo), rng, state, L, n_particles, nVars)
        parts2 = [zeros_like_particle(parts[1]) for _ in 1:n_particles]
        propagate! = (out, prev, sh) -> higher_propagate!(Val(algo), out, prev, sh, past_idx, 𝐒f, scr)
    end

    # Smoothing storage (see the backward pass below). `hist_*` keep the cloud and
    # the shocks of every period; `parent[t]` is the resampling map applied at the
    # end of period t (empty ⇒ no resampling ⇒ identity), which is the genealogy
    # the backward pass walks.
    hist_states = smooth ? [Matrix{Float64}(undef, nVars, n_particles) for _ in 1:nT] : Matrix{Float64}[]
    hist_shocks = smooth ? [Matrix{Float64}(undef, nExo,  n_particles) for _ in 1:nT] : Matrix{Float64}[]
    parent      = smooth ? [Int[] for _ in 1:nT] : Vector{Int}[]
    # `within[t]` is the composition of the tempering stages' resampling maps for
    # period t: post-stage slot ↦ pre-stage slot. Empty for the non-tempered
    # variants, which do no within-period resampling.
    within      = smooth ? [Int[] for _ in 1:nT] : Vector{Int}[]
    terminal_weights = smooth ? fill(1.0 / n_particles, n_particles) : Float64[]

    tempered = pf == :tempered_particle
    r_star     = Float64(tempering_target_ratio)
    mh_scale   = Float64(tempering_mh_scale)
    n_mh       = tempering_mh_steps
    max_stages = tempering_max_stages
    # scratch used only by the tempering stages
    anc_pool  = tempered ? [zeros_like_particle(parts[1]) for _ in 1:n_particles] : typeof(parts)()
    anc_pool2 = tempered ? [zeros_like_particle(parts[1]) for _ in 1:n_particles] : typeof(parts)()
    shocks2   = tempered ? [Vector{Float64}(undef, nExo) for _ in 1:n_particles] : Vector{Float64}[]
    sprop     = tempered ? zeros_like_particle(parts[1]) : parts[1]
    eprop     = tempered ? Vector{Float64}(undef, nExo) : Float64[]
    dv        = tempered ? Vector{Float64}(undef, n_particles) : Float64[]
    dv2       = tempered ? Vector{Float64}(undef, n_particles) : Float64[]
    comp      = tempered ? collect(1:n_particles) : Int[]
    comp2     = tempered ? collect(1:n_particles) : Int[]

    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view dat[:, t]

        if tempered
            @inbounds for p in 1:n_particles
                copy_particle!(anc_pool[p], parts[p])
            end
        end

        @inbounds for p in 1:n_particles
            Random.randn!(rng, shocks[p])
            propagate!(parts2[p], parts[p], shocks[p])
        end
        parts, parts2 = parts2, parts

        if isempty(rows)
            # nothing observed: the weights are unchanged, the cloud just predicts
            fill!(logdens, 0.0)
        elseif tempered
            # Bridge from the prior to the full measurement density in stages,
            # resampling and rejuvenating at each one. The likelihood contribution
            # is not needed here (that is `run_particle_filter`'s job) — only the
            # cloud that comes out, which ends up equally weighted.
            @inbounds for p in 1:n_particles
                dv[p] = particle_quadratic_form(measurement_full(parts[p], full_buf), data_col, observables_index, me_var, rows)
            end
            if any(isfinite, dv)
                @inbounds for p in 1:n_particles
                    comp[p] = p
                end
                d_obs = length(rows)
                φ_old = 0.0
                stage = 0
                while φ_old < 1.0 - 1e-12 && stage < max_stages
                    stage += 1
                    φ_new = tempered_next_phi(φ_old, dv, r_star, n_particles)

                    lr = 0.5 * d_obs * (φ_old == 0.0 ? log(φ_new) : log(φ_new) - log(φ_old))
                    @inbounds for p in 1:n_particles
                        logdens[p] = lr - 0.5 * (φ_new - φ_old) * dv[p]
                    end
                    m = maximum(logdens)
                    isfinite(m) || break
                    sw = 0.0
                    @inbounds for p in 1:n_particles
                        sw += exp(logdens[p] - m)
                    end
                    (sw > 0 && isfinite(sw)) || break
                    logsw = m + log(sw)
                    @inbounds for p in 1:n_particles
                        W[p] = exp(logdens[p] - logsw)
                    end

                    particle_resample_indices!(idx, bins, rng, W, particle_resampling)
                    @inbounds for j in 1:n_particles
                        a = idx[j]
                        copy_particle!(anc_pool2[j], anc_pool[a])
                        copyto!(shocks2[j], shocks[a])
                        copy_particle!(parts2[j], parts[a])
                        dv2[j]   = dv[a]
                        comp2[j] = comp[a]
                    end
                    anc_pool, anc_pool2 = anc_pool2, anc_pool
                    shocks,   shocks2   = shocks2,   shocks
                    parts,    parts2    = parts2,    parts
                    dv,       dv2       = dv2,       dv
                    comp,     comp2     = comp2,     comp

                    # Rejuvenate: random-walk Metropolis on the shocks, targeting
                    # π(ε) ∝ N(ε;0,I)·exp(-φ/2·e(ε)ᵀH⁻¹e(ε)).
                    @inbounds for p in 1:n_particles
                        shp = shocks[p]
                        for _ in 1:n_mh
                            Random.randn!(rng, eprop)
                            esq_old = 0.0; esq_new = 0.0
                            for e in 1:nExo
                                ep = shp[e] + mh_scale * eprop[e]
                                eprop[e] = ep
                                esq_new += ep * ep
                                esq_old += shp[e] * shp[e]
                            end
                            propagate!(sprop, anc_pool[p], eprop)
                            dprop = particle_quadratic_form(measurement_full(sprop, full_buf), data_col, observables_index, me_var, rows)
                            if log(rand(rng)) < -0.5 * ((esq_new - esq_old) + φ_new * (dprop - dv[p]))
                                copyto!(shp, eprop)
                                copy_particle!(parts[p], sprop)
                                dv[p] = dprop
                            end
                        end
                    end

                    φ_old = φ_new
                end
                # the tempering stages leave an equally weighted cloud
                fill!(W, 1.0 / n_particles)
                if smooth; within[t] = copy(comp); end
            end
        else
            @inbounds for p in 1:n_particles
                full = measurement_full(parts[p], full_buf)
                logdens[p] = particle_log_measurement_density(full, data_col, observables_index, me_var, rows, log2pi)
            end
            m = maximum(logdens)
            if isfinite(m)
                s = 0.0
                @inbounds for p in 1:n_particles
                    s += W[p] * exp(logdens[p] - m)
                end
                if s > 0 && isfinite(s)
                    @inbounds for p in 1:n_particles
                        W[p] = W[p] * exp(logdens[p] - m) / s
                    end
                end
            end
        end

        if smooth
            # The backward pass starts from the cloud as stored here, i.e. *before*
            # any resampling at the end of this period, so it needs the weights in
            # that same indexing. Resampling would overwrite `W` with uniform ones.
            copyto!(terminal_weights, W)
            # keep the whole cloud so the backward pass can walk the genealogy
            Hs = hist_states[t]; Hh = hist_shocks[t]
            @inbounds for p in 1:n_particles
                full = measurement_full(parts[p], full_buf)
                for i in 1:nVars; Hs[i, p] = full[i]; end
                sp = shocks[p]
                for e in 1:nExo; Hh[e, p] = sp[e]; end
            end
        else
            # filtered moments of the (weighted) cloud
            @inbounds for p in 1:n_particles
                full = measurement_full(parts[p], full_buf)
                w = W[p]
                for i in 1:nVars
                    variables[i, t] += w * full[i]
                end
                for e in 1:nExo
                    shocks_out[e, t] += w * shocks[p][e]
                end
            end
            @inbounds for p in 1:n_particles
                full = measurement_full(parts[p], full_buf)
                w = W[p]
                for i in 1:nVars
                    stds[i, t] += w * (full[i] - variables[i, t])^2
                end
            end
            @inbounds for i in 1:nVars
                stds[i, t] = sqrt(max(stds[i, t], 0.0))
            end
        end

        if effective_sample_size(W) < particle_resampling_threshold * n_particles
            particle_resample_indices!(idx, bins, rng, W, particle_resampling)
            @inbounds for j in 1:n_particles
                copy_particle!(parts2[j], parts[idx[j]])
            end
            parts, parts2 = parts2, parts
            fill!(W, 1.0 / n_particles)
            if smooth; parent[t] = copy(idx); end
        end
    end

    if smooth
        smooth_particle_trajectories!(variables, stds, shocks_out, hist_states, hist_shocks, parent, within, terminal_weights)
    end

    # ── Shock decomposition ──────────────────────────────────────────────────
    # A decomposition needs a shock path; the particle filter supplies one (the
    # smoothed shocks above), so the same attribution the inversion filter uses
    # applies here. At first order contributions are additive and the split is
    # exact. At pruned higher order they are not additive, which is precisely
    # what the Aumann-Shapley (marginal contribution) attribution is for, so the
    # pruned decomposition reuses the routines in `inversion.jl`. Non-pruned
    # `:second_order` / `:third_order` have no decomposition at all (the caller
    # already turns `shock_decomposition` off for them).
    # Column layout follows the inversion filter: with the Aumann-Shapley
    # attribution (and at first order) it is [contributions…, baseline, total] =
    # nExo+2; the sequential pruned attribution adds an explicit interaction and
    # residual column, [contributions…, interaction, residual, total] = nExo+3.
    sequential_pruned = algo ∈ (:pruned_second_order, :pruned_third_order) && !marginal_contribution
    decomposition = zeros(nVars, sequential_pruned ? nExo + 3 : nExo + 2, nT)
    decomposition[:, end, :] .= variables

    if algo == :first_order
        𝐒₁ = 𝐒 isa AbstractMatrix ? 𝐒 : 𝐒[1]
        init_vec = state isa AbstractVector{<:AbstractVector} ? state[1] : state
        sck = zeros(nExo)
        @inbounds for i in 1:nExo
            fill!(sck, 0.0); sck[i] = shocks_out[i, 1]
            decomposition[:, i, 1] .= 𝐒₁ * vcat(init_vec[past_idx], sck)
        end
        decomposition[:, end - 1, 1] .= decomposition[:, end, 1] - sum(decomposition[:, 1:end-2, 1], dims = 2)
        for t in 2:nT
            @inbounds for i in 1:nExo
                fill!(sck, 0.0); sck[i] = shocks_out[i, t]
                decomposition[:, i, t] .= 𝐒₁ * vcat(decomposition[past_idx, i, t-1], sck)
            end
            decomposition[:, end - 1, t] .= decomposition[:, end, t] - sum(decomposition[:, 1:end-2, t], dims = 2)
        end
    elseif algo ∈ (:pruned_second_order, :pruned_third_order) && marginal_contribution
        # The Aumann-Shapley attribution requires `variables` and `shocks` to lie
        # on the *same* model trajectory — it checks that the contributions plus
        # the zero-shock baseline reproduce `variables`. A smoothed mean is not a
        # model path (averaging does not commute with the nonlinear transition:
        # E[g(x,ε)] ≠ g(E[x],E[ε])), so feeding it in directly leaves a closure
        # error that the routine tries to remove by refining its quadrature.
        # Decompose the trajectory implied by the smoothed shocks instead, which
        # is the same object the inversion filter decomposes and closes exactly.
        traj = zeros(nVars, nT)
        cur  = deepcopy(state)
        nxt  = zeros_like_particle(cur)
        buf  = Vector{Float64}(undef, nVars)
        shk  = Vector{Float64}(undef, nExo)
        for t in 1:nT
            @inbounds for e in 1:nExo; shk[e] = shocks_out[e, t]; end
            higher_propagate!(Val(algo), nxt, cur, shk, past_idx, 𝐒f, scr)
            full = measurement_full(nxt, buf)
            @inbounds for i in 1:nVars; traj[i, t] = full[i]; end
            cur, nxt = nxt, cur
        end
        decomposition[:, end, :] .= traj

        if algo == :pruned_second_order
            aumann_shapley_shock_decomposition_pruned_2nd_order!(decomposition, traj, shocks_out,
                                                                 state, 𝐒, T, nExo; verbose = opts.verbose)
        else
            aumann_shapley_shock_decomposition_pruned_3rd_order!(decomposition, traj, shocks_out,
                                                                 state, 𝐒, T, nExo; verbose = opts.verbose)
        end
    elseif sequential_pruned
        # Sequential attribution: run one trajectory per shock with only that
        # shock switched on, plus one with all of them. Each single-shock path is
        # that shock's contribution; the all-shock path minus the sum of the
        # single-shock paths is the interaction the nonlinearity creates (this is
        # the term the Aumann-Shapley variant instead distributes across shocks),
        # and whatever is still left over goes into the residual column.
        states_dec = [deepcopy(state) for _ in 1:nExo + 1]
        nxt   = zeros_like_particle(state)
        dbuf  = Vector{Float64}(undef, nVars)
        single = zeros(nExo)
        allsh  = Vector{Float64}(undef, nExo)

        for t in 1:nT
            @inbounds for ii in 1:nExo
                fill!(single, 0.0); single[ii] = shocks_out[ii, t]
                higher_propagate!(Val(algo), nxt, states_dec[ii], single, past_idx, 𝐒f, scr)
                copy_particle!(states_dec[ii], nxt)
                full = measurement_full(states_dec[ii], dbuf)
                for v in 1:nVars; decomposition[v, ii, t] = full[v]; end
            end

            @inbounds for e in 1:nExo; allsh[e] = shocks_out[e, t]; end
            higher_propagate!(Val(algo), nxt, states_dec[end], allsh, past_idx, 𝐒f, scr)
            copy_particle!(states_dec[end], nxt)
            full = measurement_full(states_dec[end], dbuf)

            # interaction = all-shock path − Σ single-shock paths
            @inbounds for v in 1:nVars; decomposition[v, end - 2, t] = full[v]; end
            decomposition[:, end - 2, t] .-= sum(decomposition[:, 1:end-3, t], dims = 2)
            # residual = reported estimate − everything attributed so far
            decomposition[:, end - 1, t] .= variables[:, t]
            decomposition[:, end - 1, t] .-= sum(decomposition[:, 1:end-2, t], dims = 2)
        end
    else
        @info "Shock decomposition is not available for $(algo) solutions (use a pruned solution); returning zeros." maxlog = 1
    end

    return variables, shocks_out, stds, decomposition
end

# Backward pass of the particle smoother (fixed-interval smoothing by genealogy,
# a.k.a. forward-filtering backward-sampling on the filter's ancestral lines).
#
# Why this and not the textbook backward kernel? The usual particle smoother
# reweights particles at t by the backward transition density p(xₜ₊₁ | xₜ). For a
# DSGE that density is *singular*: with fewer shocks than states the transition
# maps xₜ onto a lower-dimensional manifold, so p(xₜ₊₁ | xₜ) is a Dirac on that
# manifold and the reweighting is undefined. What is well defined is the filter's
# own genealogy: every surviving particle at T carries the ancestral line that
# produced it, and those lines are draws from p(x₁..T | y₁..T). Averaging the
# lines with the final weights therefore gives the smoothed moments directly.
#
# `parent[t]` is the resampling map applied at the end of period t (empty means no
# resampling happened, i.e. the identity). Walking it backwards from T turns each
# final particle index into the index it occupied at every earlier period.
#
# Caveat worth knowing: ancestral lines coalesce as one goes back in time (path
# degeneracy), so the smoothed estimate for the earliest periods rests on fewer
# distinct trajectories than the particle count suggests. More particles push the
# coalescence point further back.
function smooth_particle_trajectories!(variables::Matrix{Float64},
                                       stds::Matrix{Float64},
                                       shocks_out::Matrix{Float64},
                                       hist_states::Vector{Matrix{Float64}},
                                       hist_shocks::Vector{Matrix{Float64}},
                                       parent::Vector{Vector{Int}},
                                       within::Vector{Vector{Int}},
                                       W::Vector{Float64})
    nT = length(hist_states)
    nT == 0 && return nothing
    nVars = size(variables, 1)
    nExo  = size(shocks_out, 1)
    n_particles = length(W)

    # lineage: where each final particle sat at the period currently being visited
    lineage = collect(1:n_particles)

    for t in nT:-1:1
        Hs = hist_states[t]; Hh = hist_shocks[t]

        @inbounds for p in 1:n_particles
            a = lineage[p]; w = W[p]
            for i in 1:nVars
                variables[i, t] += w * Hs[i, a]
            end
            for e in 1:nExo
                shocks_out[e, t] += w * Hh[e, a]
            end
        end
        @inbounds for p in 1:n_particles
            a = lineage[p]; w = W[p]
            for i in 1:nVars
                stds[i, t] += w * (Hs[i, a] - variables[i, t])^2
            end
        end
        @inbounds for i in 1:nVars
            stds[i, t] = sqrt(max(stds[i, t], 0.0))
        end

        # Step the lineage back one period. Two maps may apply, innermost first:
        # the tempered filter's within-period resampling (post-stage slot ↦
        # pre-stage slot at t, which is the slot the ancestor occupied after the
        # end-of-(t-1) resampling), then the end-of-(t-1) resampling itself
        # (↦ the slot indexing `hist_states[t-1]`). Either may be empty.
        if t > 1
            wit = within[t]
            if !isempty(wit)
                @inbounds for p in 1:n_particles
                    lineage[p] = wit[lineage[p]]
                end
            end
            par = parent[t-1]
            if !isempty(par)
                @inbounds for p in 1:n_particles
                    lineage[p] = par[lineage[p]]
                end
            end
        end
    end

    return nothing
end

# out = A·prev[past] + B·shock for a single particle (estimates path; the
# likelihood path propagates the whole swarm with one gemm instead).
@inline function linear_propagate_estimates!(out::Vector{Float64}, tr::LinearParticleTransition,
                                             prev::Vector{Float64}, shock::Vector{Float64})
    ℒ.mul!(out, tr.A, prev)
    ℒ.mul!(out, tr.B, shock, 1.0, 1.0)
    return out
end

end # @stable
