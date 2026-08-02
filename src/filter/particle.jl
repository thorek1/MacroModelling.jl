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
# perturbation solution (first order through pruned third order), evaluated for
# the whole swarm at once — see "Batched transitions" below.
#
# Three variants are provided, each selected by its own `filter` value:
#   :bootstrap_particle — sequential importance resampling, i.e. the bootstrap
#                         filter of Gordon, Salmond & Smith (1993), applied to
#                         DSGE models by Fernández-Villaverde & Rubio-Ramírez
#                         (2007)
#   :auxiliary_particle — auxiliary particle filter of Pitt & Shephard (1999)
#   :tempered_particle  — tempered particle filter of Herbst & Schorfheide (2019)
#
# Which one to use is not a matter of taste once the observation is informative.
# With as many observables as shocks and a small H — the usual DSGE setting — the
# bootstrap proposal draws shocks from the prior and only then looks at the data,
# so almost all of its weight lands on a handful of particles. Its likelihood
# estimate is still unbiased, but its *filtered moments* (what
# `get_estimated_shocks` and friends report) are then averages over an effective
# sample of a few particles and vary wildly from seed to seed. The tempered
# filter exists precisely to fix that and is the variant to reach for whenever
# estimates, rather than a likelihood, are what is wanted.
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

# In-place resampling: ancestor indices are written into `idx`; `bins` is a
# cumulative-weight scratch used by the multinomial/residual schemes. Buffer
# reuse for the index/cumulative arrays follows LowLevelParticleFilters.jl.

# Walk N equally spaced points u₀, u₀+1/N, … through the cumulative weights, with
# a single random offset u₀ ∈ [0, 1/N). A particle of weight Wᵢ spans Wᵢ·N spacings
# so it is picked either ⌊N·Wᵢ⌋ or ⌈N·Wᵢ⌉ times — never far from its expectation.
function systematic_resample_indices!(idx::Vector{Int}, rng::Random.AbstractRNG, W::AbstractVector{<:Real})
    N = length(W)
    u0 = rand(rng) / N
    c = W[1]
    i = 1
    @inbounds for j in 1:N
        u = u0 + (j - 1) / N
        while u > c && i < N
            i += 1
            c += W[i]
        end
        idx[j] = i
    end
    return idx
end

# Split [0,1) into N strata of width 1/N and draw one independent uniform inside
# each. Guarantees at most one draw per stratum (so counts stay close to N·Wᵢ)
# while keeping the draws independent, unlike the systematic scheme.
function stratified_resample_indices!(idx::Vector{Int}, rng::Random.AbstractRNG, W::AbstractVector{<:Real})
    N = length(W)
    c = W[1]
    i = 1
    @inbounds for j in 1:N
        u = (j - 1 + rand(rng)) / N
        while u > c && i < N
            i += 1
            c += W[i]
        end
        idx[j] = i
    end
    return idx
end

# N independent draws from the categorical distribution defined by W, via binary
# search on the cumulative weights. Simplest and noisiest: nothing prevents a
# particle with weight 1/N from being drawn three times or not at all.
function multinomial_resample_indices!(idx::Vector{Int}, bins::Vector{Float64}, rng::Random.AbstractRNG, W::AbstractVector{<:Real})
    N = length(W)
    cumsum!(bins, W)
    bins[N] = one(eltype(bins))   # guard against round-off
    @inbounds for j in 1:N
        idx[j] = searchsortedfirst(bins, rand(rng))
    end
    return idx
end

# Deterministic part first: particle i gets ⌊N·Wᵢ⌋ guaranteed copies, which carry
# no randomness at all. Only the leftover R = N - Σ⌊N·Wᵢ⌋ slots are drawn, from
# the renormalised fractional weights. Cuts the variance of the integer part to
# zero, which helps most when a handful of particles dominate.
function residual_resample_indices!(idx::Vector{Int}, bins::Vector{Float64}, rng::Random.AbstractRNG, W::AbstractVector{<:Real})
    N = length(W)
    k = 0
    @inbounds for i in 1:N
        ni = floor(Int, N * W[i])
        for _ in 1:ni
            k += 1
            idx[k] = i
        end
    end
    R = N - k
    if R > 0
        s = 0.0
        @inbounds for i in 1:N
            bins[i] = N * W[i] - floor(N * W[i])
            s += bins[i]
        end
        if s <= 0            # numerical degeneracy: fall back to multinomial
            cumsum!(bins, W)
        else
            @inbounds for i in 1:N
                bins[i] /= s
            end
            cumsum!(bins, bins)
        end
        bins[N] = one(eltype(bins))
        @inbounds for _ in 1:R
            k += 1
            idx[k] = searchsortedfirst(bins, rand(rng))
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

# Allocating convenience form (used by the tests and by anything that just wants
# a set of ancestor indices without keeping scratch around).
function particle_resample_indices(rng::Random.AbstractRNG, W::AbstractVector{<:Real}, scheme::Symbol)
    N = length(W)
    return particle_resample_indices!(Vector{Int}(undef, N), Vector{Float64}(undef, N), rng, W, scheme)
end

# Normalise exp(logw) into `W` and return log Σ exp(logw). Returns `-Inf` (and
# leaves `W` untouched) when every entry is impossible or the sum is not finite,
# which is how the filters detect a period no particle can explain.
@inline function normalise_log_weights!(W::Vector{Float64}, logw::Vector{Float64})
    m = -Inf
    @inbounds for p in eachindex(logw)
        lp = logw[p]
        m = lp > m ? lp : m
    end
    isfinite(m) || return -Inf
    s = 0.0
    @inbounds for p in eachindex(logw)
        s += exp(logw[p] - m)
    end
    (s > 0 && isfinite(s)) || return -Inf
    ls = m + log(s)
    @inbounds for p in eachindex(logw)
        W[p] = exp(logw[p] - ls)
    end
    return ls
end

# Same, but for weights that carry a prior `W` (the bootstrap/auxiliary update
# Wₚ ∝ Wₚ·p(yₜ|xₜᵖ)): returns log Σₚ Wₚ·exp(logdensₚ) and renormalises `W`.
@inline function reweight_log_weights!(W::Vector{Float64}, logdens::Vector{Float64})
    m = -Inf
    @inbounds for p in eachindex(logdens)
        lp = logdens[p]
        m = lp > m ? lp : m
    end
    isfinite(m) || return -Inf
    s = 0.0
    @inbounds for p in eachindex(logdens)
        s += W[p] * exp(logdens[p] - m)
    end
    (s > 0 && isfinite(s)) || return -Inf
    @inbounds for p in eachindex(logdens)
        W[p] = W[p] * exp(logdens[p] - m) / s
    end
    return m + log(s)
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


# ── Non-diagonal measurement error ───────────────────────────────────────────
# Everything below takes `me_var`, the diagonal of H, and reads it elementwise —
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

# The diagonal of H: what the auxiliary filter's first-stage preview needs (it
# only has to be a rough predictive variance — the second-stage reweighting is
# exact whatever the preview used).
me_diagonal(me_var::AbstractVector) = me_var
me_diagonal(me::DenseMeasurementError) = ℒ.diag(me.H)

# Elementwise reciprocals for the batched scoring kernels; the dense filter keeps
# its factorisation instead and is passed through unchanged.
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


# ── Measurement scoring on a batched cloud ───────────────────────────────────
# The cloud is stored column-wise (see "Batched transitions"), so every kernel
# below reads particle `p` out of column `p` of the full-state matrix `F`.

# Quadratic form eᵀH⁻¹e over the observed rows for particle column `p`, with
# diagonal H (`inv_me_var` holds the reciprocal variances). Returns `Inf` on a
# non-finite prediction, i.e. an impossible particle.
@inline function particle_quadform_col(F::AbstractMatrix{Float64}, p::Int, data_col, observables_index, inv_me_var, rows)
    q = 0.0
    @inbounds for k in eachindex(rows)
        r = rows[k]
        f = F[observables_index[r], p]
        isfinite(f) || return Inf
        v = data_col[r] - f
        q += v * v * inv_me_var[r]
    end
    return q
end

# Same, for a correlated H: gather the innovation, then one triangular solve.
# vᵀH⁻¹v = ‖L⁻¹v‖², so the forward substitution gives both the solve and the norm.
@inline function particle_quadform_col(F::AbstractMatrix{Float64}, p::Int, data_col, observables_index,
                                       me::DenseMeasurementError, rows)
    me_sync!(me, rows)
    v = me.buf
    @inbounds for k in eachindex(rows)
        r = rows[k]
        f = F[observables_index[r], p]
        isfinite(f) || return Inf
        v[k] = data_col[r] - f
    end
    return dense_me_quadform!(v, me.last_L)
end

# Log normalising constant of the Gaussian measurement density over the observed
# rows: -½(dₒ·log2π + log det H[rows, rows]).
@inline function particle_measurement_logZ(me_var::AbstractVector, rows, log2pi::Float64)
    z = 0.0
    @inbounds for r in rows
        z += log2pi + log(me_var[r])
    end
    return -0.5 * z
end

@inline function particle_measurement_logZ(me::DenseMeasurementError, rows, log2pi::Float64)
    me_sync!(me, rows)
    return -0.5 * (length(rows) * log2pi + me.last_logdet)
end

# log p(yₜ | xₜᵖ) for every column of `F`, written into `logdens`. Kept serial:
# with a correlated H these share the one innovation buffer inside
# `DenseMeasurementError`, and at a handful of observables per particle they are
# nowhere near the cost of a transition anyway.
function score_cloud!(logdens::Vector{Float64}, F::Matrix{Float64}, data_col, observables_index,
                      me_var, inv_me_var, rows, log2pi::Float64)
    logZ = particle_measurement_logZ(me_var, rows, log2pi)
    @inbounds for p in eachindex(logdens)
        q = particle_quadform_col(F, p, data_col, observables_index, inv_me_var, rows)
        logdens[p] = isfinite(q) ? logZ - 0.5 * q : -Inf
    end
    return logdens
end

# eᵀH⁻¹e for every column of `F`, written into `dv` (what the tempering schedule
# and the Metropolis mutation work with).
function quadform_cloud!(dv::Vector{Float64}, F::Matrix{Float64}, data_col, observables_index,
                         inv_me_var, rows)
    @inbounds for p in eachindex(dv)
        dv[p] = particle_quadform_col(F, p, data_col, observables_index, inv_me_var, rows)
    end
    return dv
end


# ── Batched transitions ──────────────────────────────────────────────────────
# A particle is a state vector; a *cloud* is an `nVars × N` matrix whose columns
# are particles. Pruned solutions carry the state in several components, so a
# cloud is an `NTuple{K, Matrix{Float64}}` — K = 1 at first, second and third
# order, 2 at pruned second order, 3 at pruned third order.
#
# Storing the swarm this way is what makes the filters fast. The perturbation
# transition is, for every order, a set of matrix-vector products against the
# augmented state [xₜ₋₁[past]; 1; εₜ] and its compressed Kronecker powers. Laid
# out column-wise those become matrix-*matrix* products over the whole swarm, so
# one period costs a handful of `gemm` calls instead of N `gemv` calls: the same
# arithmetic at BLAS-3 efficiency, with the augmented and Kronecker scratch
# allocated once.
#
# The scratch is sized to a fixed column block rather than to N, so its memory
# does not grow with the particle count (the compressed cube of the augmented
# state has n(n+1)(n+2)/6 rows, which for a medium DSGE is already thousands).

# Number of pruned state components per algorithm, as a `Val` so `ntuple` stays
# type-stable at the call site.
pruned_components(::Val{:first_order})         = Val(1)
pruned_components(::Val{:second_order})        = Val(1)
pruned_components(::Val{:third_order})         = Val(1)
pruned_components(::Val{:pruned_second_order}) = Val(2)
pruned_components(::Val{:pruned_third_order})  = Val(3)

n_components(::Val{K}) where {K} = K::Int

# Block sizing, from `default_options.jl`:
#   DEFAULT_PARTICLE_SCRATCH_BYTES       upper bound on the memory the
#     augmented/Kronecker scratch may occupy in total across all threads. It is
#     what caps the column-block size at third order, where the compressed cube of
#     the augmented state already has thousands of rows.
#   DEFAULT_PARTICLE_MIN_BLOCK           below this the `gemm` calls stop
#     amortising their own overhead.
#   DEFAULT_PARTICLE_PARALLEL_MIN_WORK   below this much arithmetic per sweep
#     (scratch rows × particles) the task overhead outweighs the parallelism and
#     the swarm is propagated on the calling thread, so small models with few
#     particles behave exactly as if none of this existed.

# The perturbation solution in the form the batched kernels want: `𝐒₁` is always
# the *augmented* first-order matrix (nVars × (nPast+1+nExo), constant column
# included — zero at first order, where the solution has no constant term), and
# `𝐒₂`/`𝐒₃` are the compressed second/third-order matrices, densified because
# they are more than half full for a typical DSGE and are multiplied by tall
# dense blocks.
struct ParticleTransition
    𝐒₁::Matrix{Float64}
    𝐒₂::Matrix{Float64}
    𝐒₃::Matrix{Float64}
    past_idx::Vector{Int}
    nVars::Int
    nPast::Int
    nExo::Int
    naug::Int
end

# First order is the only case that has to build its own augmented `𝐒₁`: the
# solution has no constant term, so the constant column is inserted as zeros.
function build_particle_transition(::Val{:first_order}, 𝐒, T)
    nVars = T.nVars
    nPast = T.nPast_not_future_and_mixed
    nExo  = T.nExo
    naug  = nPast + 1 + nExo
    empty = Matrix{Float64}(undef, nVars, 0)

    S = Matrix{Float64}(𝐒 isa AbstractMatrix ? 𝐒 : 𝐒[1])
    𝐒₁ = zeros(Float64, nVars, naug)
    @views 𝐒₁[:, 1:nPast]      .= S[:, 1:nPast]
    @views 𝐒₁[:, nPast+2:naug] .= S[:, nPast+1:end]   # constant column stays zero
    return ParticleTransition(𝐒₁, empty, empty, T.past_not_future_and_mixed_idx, nVars, nPast, nExo, naug)
end

# Why 𝐒₂/𝐒₃ are densified, when the solution stores them sparse.
#
# In the *compressed* basis they are not sparse. Compression merges each set of
# symmetric duplicate columns into one, so a compressed column is zero only when
# every uncompressed column it stands for was, and the surviving matrix is mostly
# full. Measured on the solutions themselves:
#
#           𝐒₂ shape       density    𝐒₃ shape        density
#   FS2000  18 x   28       0.48      18 x    84       0.28
#   Gali    23 x   36       0.78      23 x   120       0.73
#   SW03    54 x  435       0.87      54 x  4495       0.84
#   SW07    66 x  595       0.62      66 x  7140       0.51
#
# and dense `gemm` beats `SparseMatrixCSC` multiplication over the whole of that
# range. Against a swarm block of 512 / 4096 columns, sparse costs 3.2-7.9x more
# with BLAS on one thread and 2.6-26x more on four. The crossover — measured by
# thinning a 66 x 7140 matrix — sits near 1 % density: at 5 % sparse is still
# 1.35x slower single-threaded, and only at 1 % does it win (0.32x), which no
# model here comes close to. The shape is what does it: the operand is short and
# very wide, so the dense kernel is entirely cache-blocked BLAS while the CSC
# kernel is single-threaded scattered accumulation.
#
# If a model ever does produce a genuinely sparse compressed 𝐒₃, this is the one
# place to branch on `nnz`; the propagation code below needs no change, since
# `mul!` dispatches on the type it is handed.
function build_particle_transition(::Val{algo}, 𝐒, T) where {algo}
    nVars = T.nVars
    nPast = T.nPast_not_future_and_mixed
    nExo  = T.nExo
    naug  = nPast + 1 + nExo
    empty = Matrix{Float64}(undef, nVars, 0)

    𝐒₁ = Matrix{Float64}(𝐒[1])
    𝐒₂ = length(𝐒) >= 2 ? Matrix{Float64}(𝐒[2]) : empty
    𝐒₃ = length(𝐒) >= 3 ? Matrix{Float64}(𝐒[3]) : empty
    return ParticleTransition(𝐒₁, 𝐒₂, 𝐒₃, T.past_not_future_and_mixed_idx, nVars, nPast, nExo, naug)
end

# Blocked augmented/Kronecker scratch for one worker task. Buffers an algorithm
# does not use are left empty rather than absent, so the type is the same at
# every order and the filters never dispatch on the scratch.
#   aug1  augmented first-order state [x¹[past]; 1; ε]
#   aug2  same for the second pruned component (constant and shock slots zeroed)
#   aug3  same for the third pruned component
#   augĥ  aug1 with the constant slot zeroed (pruned third-order cross term)
#   kk    compressed aug1 ⊗ aug1
#   kk2   compressed augĥ ⊗ aug2
#   kkk   compressed aug1 ⊗ aug1 ⊗ aug1
struct ScratchSlot
    aug1::Matrix{Float64}
    aug2::Matrix{Float64}
    aug3::Matrix{Float64}
    augĥ::Matrix{Float64}
    kk::Matrix{Float64}
    kk2::Matrix{Float64}
    kkk::Matrix{Float64}
end

no_buffer() = Matrix{Float64}(undef, 0, 0)

# Rows of a compressed square/cube of an `n`-vector.
@inline n_pair_rows(n::Int)   = n * (n + 1) ÷ 2
@inline n_triple_rows(n::Int) = n * (n + 1) * (n + 2) ÷ 6

@inline scratch_buffer(n::Int, blk::Int) = Matrix{Float64}(undef, n, blk)

function build_scratch_slot(::Val{:first_order}, naug::Int, blk::Int)
    return ScratchSlot(scratch_buffer(naug, blk), no_buffer(), no_buffer(), no_buffer(),
                       no_buffer(), no_buffer(), no_buffer())
end

function build_scratch_slot(::Val{:second_order}, naug::Int, blk::Int)
    return ScratchSlot(scratch_buffer(naug, blk), no_buffer(), no_buffer(), no_buffer(),
                       scratch_buffer(n_pair_rows(naug), blk), no_buffer(), no_buffer())
end

function build_scratch_slot(::Val{:third_order}, naug::Int, blk::Int)
    return ScratchSlot(scratch_buffer(naug, blk), no_buffer(), no_buffer(), no_buffer(),
                       scratch_buffer(n_pair_rows(naug), blk), no_buffer(),
                       scratch_buffer(n_triple_rows(naug), blk))
end

function build_scratch_slot(::Val{:pruned_second_order}, naug::Int, blk::Int)
    return ScratchSlot(scratch_buffer(naug, blk), scratch_buffer(naug, blk), no_buffer(), no_buffer(),
                       scratch_buffer(n_pair_rows(naug), blk), no_buffer(), no_buffer())
end

function build_scratch_slot(::Val{:pruned_third_order}, naug::Int, blk::Int)
    return ScratchSlot(scratch_buffer(naug, blk), scratch_buffer(naug, blk),
                       scratch_buffer(naug, blk), scratch_buffer(naug, blk),
                       scratch_buffer(n_pair_rows(naug), blk), scratch_buffer(n_pair_rows(naug), blk),
                       scratch_buffer(n_triple_rows(naug), blk))
end

# Scratch rows the algorithm needs per column, which is what sets the memory cost
# of a block.
scratch_rows(::Val{:first_order}, naug::Int)         = naug
scratch_rows(::Val{:second_order}, naug::Int)        = naug + n_pair_rows(naug)
scratch_rows(::Val{:third_order}, naug::Int)         = naug + n_pair_rows(naug) + n_triple_rows(naug)
scratch_rows(::Val{:pruned_second_order}, naug::Int) = 2 * naug + n_pair_rows(naug)
scratch_rows(::Val{:pruned_third_order}, naug::Int)  = 4 * naug + 2 * n_pair_rows(naug) + n_triple_rows(naug)

# The transition scratch, one slot per worker task. Column blocks read disjoint
# columns of the current cloud and write disjoint columns of the next one, so the
# swarm can be propagated in parallel and the result is bit-identical whatever
# the thread count — every random draw happens outside this loop. The block size
# is chosen to give each task roughly one block while keeping the total scratch
# inside `DEFAULT_PARTICLE_SCRATCH_BYTES`.
struct BatchScratch
    slots::Vector{ScratchSlot}
    blk::Int
end

function build_batch_scratch(::Val{algo}, naug::Int, n_particles::Int)::BatchScratch where {algo}
    rows = scratch_rows(Val(algo), naug)
    nt = rows * n_particles >= DEFAULT_PARTICLE_PARALLEL_MIN_WORK ? max(Threads.nthreads(), 1) : 1
    blk_mem = max(DEFAULT_PARTICLE_MIN_BLOCK, DEFAULT_PARTICLE_SCRATCH_BYTES ÷ (8 * rows * nt))
    blk = clamp(cld(n_particles, nt), DEFAULT_PARTICLE_MIN_BLOCK, blk_mem)
    blk = min(blk, n_particles)
    n_slots = max(1, min(nt, cld(n_particles, blk)))
    slots = ScratchSlot[build_scratch_slot(Val(algo), naug, blk) for _ in 1:n_slots]
    return BatchScratch(slots, blk)
end

# aug[:, j] = [X[past_idx, cols[j]]; const_val; with_shocks ? E[:, cols[j]] : 0].
@inline function fill_aug_block!(aug::AbstractMatrix{Float64}, X::Matrix{Float64}, past_idx::Vector{Int},
                                 E::Matrix{Float64}, cols::UnitRange{Int}, const_val::Float64, with_shocks::Bool)
    nPast = length(past_idx)
    nExo  = size(E, 1)
    @inbounds for (j, p) in enumerate(cols)
        for i in 1:nPast
            aug[i, j] = X[past_idx[i], p]
        end
        aug[nPast + 1, j] = const_val
        if with_shocks
            for e in 1:nExo
                aug[nPast + 1 + e, j] = E[e, p]
            end
        else
            for e in 1:nExo
                aug[nPast + 1 + e, j] = 0.0
            end
        end
    end
    return aug
end

# One column block of the transition, written into `Xn`. `X` is the current
# cloud, `E` the shocks. Each method mirrors the corresponding closure built by
# `parse_algorithm_to_state_update`, with `aug = [x[past]; 1; ε]`.
#
# Every operand is a `view` into a preallocated buffer. Slicing whole columns of
# a `Matrix` gives a contiguous `SubArray`, which is a `StridedMatrix`, so `mul!`
# reaches BLAS `gemm` on it directly — no copy of the block, and no allocation:
# `@allocated` over a statically dispatched `propagate_block!` and
# `propagate_cloud!` is 0 bytes at every order, at any block size.
#
# Method bodies:
#
#   :second_order         x⁺ = 𝐒₁·aug + ½ 𝐒₂·(aug⊗aug)
#   :third_order          x⁺ = 𝐒₁·aug + ½ 𝐒₂·(aug⊗aug) + ⅙ 𝐒₃·(aug⊗aug⊗aug)
#   :pruned_second_order  x¹⁺ = 𝐒₁·aug¹,  x²⁺ = 𝐒₁·aug² + ½ 𝐒₂·(aug¹⊗aug¹)
#   :pruned_third_order   adds x³⁺ = 𝐒₁·aug³ + 𝐒₂·(aug¹̂⊗aug²) + ⅙ 𝐒₃·(aug¹⊗aug¹⊗aug¹)
#
# where the higher pruned components zero the constant and shock slots of their
# augmented vector, and aug¹̂ is aug¹ with the constant slot zeroed.
function propagate_block!(::Val{:first_order}, tr::ParticleTransition, scr,
                          Xn::NTuple{K,Matrix{Float64}}, X::NTuple{K,Matrix{Float64}},
                          E::Matrix{Float64}, cols::UnitRange{Int}) where {K}
    b = length(cols)
    a1 = view(scr.aug1, :, 1:b)
    o1 = view(Xn[1], :, cols)

    fill_aug_block!(a1, X[1], tr.past_idx, E, cols, 1.0, true)
    ℒ.mul!(o1, tr.𝐒₁, a1)                           # x⁺ = 𝐒₁·aug
    return Xn
end

function propagate_block!(::Val{:second_order}, tr::ParticleTransition, scr,
                          Xn::NTuple{K,Matrix{Float64}}, X::NTuple{K,Matrix{Float64}},
                          E::Matrix{Float64}, cols::UnitRange{Int}) where {K}
    b = length(cols)
    a1 = view(scr.aug1, :, 1:b)
    kk = view(scr.kk, :, 1:b)
    o1 = view(Xn[1], :, cols)

    fill_aug_block!(a1, X[1], tr.past_idx, E, cols, 1.0, true)
    compressed_kron²_power_columns!(kk, a1)
    ℒ.mul!(o1, tr.𝐒₁, a1)                           # x⁺ = 𝐒₁·aug
    ℒ.mul!(o1, tr.𝐒₂, kk, 0.5, 1.0)                 # x⁺ += ½ 𝐒₂·(aug⊗aug)
    return Xn
end

function propagate_block!(::Val{:third_order}, tr::ParticleTransition, scr,
                          Xn::NTuple{K,Matrix{Float64}}, X::NTuple{K,Matrix{Float64}},
                          E::Matrix{Float64}, cols::UnitRange{Int}) where {K}
    b = length(cols)
    a1  = view(scr.aug1, :, 1:b)
    kk  = view(scr.kk, :, 1:b)
    kkk = view(scr.kkk, :, 1:b)
    o1  = view(Xn[1], :, cols)

    fill_aug_block!(a1, X[1], tr.past_idx, E, cols, 1.0, true)
    compressed_kron²_power_columns!(kk, a1)
    compressed_kron³_power_columns!(kkk, a1)
    ℒ.mul!(o1, tr.𝐒₁, a1)                           # x⁺ = 𝐒₁·aug
    ℒ.mul!(o1, tr.𝐒₂, kk, 0.5, 1.0)                 # x⁺ += ½ 𝐒₂·(aug⊗aug)
    ℒ.mul!(o1, tr.𝐒₃, kkk, 1 / 6, 1.0)              # x⁺ += ⅙ 𝐒₃·(aug⊗aug⊗aug)
    return Xn
end

function propagate_block!(::Val{:pruned_second_order}, tr::ParticleTransition, scr,
                          Xn::NTuple{K,Matrix{Float64}}, X::NTuple{K,Matrix{Float64}},
                          E::Matrix{Float64}, cols::UnitRange{Int}) where {K}
    b = length(cols)
    past_idx = tr.past_idx
    a1 = view(scr.aug1, :, 1:b)
    a2 = view(scr.aug2, :, 1:b)
    kk = view(scr.kk, :, 1:b)
    o1 = view(Xn[1], :, cols)
    o2 = view(Xn[2], :, cols)

    fill_aug_block!(a1, X[1], past_idx, E, cols, 1.0, true)
    fill_aug_block!(a2, X[2], past_idx, E, cols, 0.0, false)
    compressed_kron²_power_columns!(kk, a1)
    ℒ.mul!(o1, tr.𝐒₁, a1)                           # x¹⁺ = 𝐒₁·aug¹
    ℒ.mul!(o2, tr.𝐒₁, a2)                           # x²⁺ = 𝐒₁·aug²
    ℒ.mul!(o2, tr.𝐒₂, kk, 0.5, 1.0)                 # x²⁺ += ½ 𝐒₂·(aug¹⊗aug¹)
    return Xn
end

function propagate_block!(::Val{:pruned_third_order}, tr::ParticleTransition, scr,
                          Xn::NTuple{K,Matrix{Float64}}, X::NTuple{K,Matrix{Float64}},
                          E::Matrix{Float64}, cols::UnitRange{Int}) where {K}
    b = length(cols)
    past_idx = tr.past_idx
    a1 = view(scr.aug1, :, 1:b)
    a2 = view(scr.aug2, :, 1:b)
    a3 = view(scr.aug3, :, 1:b)
    aĥ = view(scr.augĥ, :, 1:b)
    kk  = view(scr.kk, :, 1:b)
    kk2 = view(scr.kk2, :, 1:b)
    kkk = view(scr.kkk, :, 1:b)
    o1 = view(Xn[1], :, cols)
    o2 = view(Xn[2], :, cols)
    o3 = view(Xn[3], :, cols)

    fill_aug_block!(a1, X[1], past_idx, E, cols, 1.0, true)
    fill_aug_block!(aĥ, X[1], past_idx, E, cols, 0.0, true)
    fill_aug_block!(a2, X[2], past_idx, E, cols, 0.0, false)
    fill_aug_block!(a3, X[3], past_idx, E, cols, 0.0, false)
    compressed_kron²_power_columns!(kk, a1)
    compressed_kron²_columns!(kk2, aĥ, a2)
    compressed_kron³_power_columns!(kkk, a1)
    ℒ.mul!(o1, tr.𝐒₁, a1)                           # x¹⁺ = 𝐒₁·aug¹
    ℒ.mul!(o2, tr.𝐒₁, a2)                           # x²⁺ = 𝐒₁·aug²
    ℒ.mul!(o2, tr.𝐒₂, kk, 0.5, 1.0)                 # x²⁺ += ½ 𝐒₂·(aug¹⊗aug¹)
    ℒ.mul!(o3, tr.𝐒₁, a3)                           # x³⁺ = 𝐒₁·aug³
    ℒ.mul!(o3, tr.𝐒₂, kk2, 1.0, 1.0)                # x³⁺ += 𝐒₂·(aug¹̂⊗aug²)
    ℒ.mul!(o3, tr.𝐒₃, kkk, 1 / 6, 1.0)              # x³⁺ += ⅙ 𝐒₃·(aug¹⊗aug¹⊗aug¹)
    return Xn
end

# Push the whole swarm one period forward, block by block, one task per scratch
# slot. Blocks are handed out in contiguous chunks so the assignment — and hence
# the result — does not depend on how the scheduler interleaves them.
#
# Why this is threaded at all, when most of a block is `mul!`. `𝐒₂`/`𝐒₃` are
# dense (see `build_particle_transition` for why that is the right call, and for
# where the sparse crossover lies), so those are `gemm` calls and BLAS is already
# threading them. What BLAS cannot touch is the other part of a
# block: `fill_aug_block!` and the compressed Kronecker kernels are plain
# sequential Julia loops, and they are not a rounding error — measured per block
# at 10 000 particles they are ~26 % of the time at pruned second order and ~37 %
# at pruned third, where the cube of the augmented state is the dominant buffer.
#
# So the two layers overlap, and how much this buys depends entirely on what BLAS
# is doing. On a 4-thread machine, spawning over blocks against a 4-thread BLAS is
# worth nothing at first order (the guard below keeps it single-tasked anyway),
# ~1.1x at pruned second order and ~1.2-1.4x at pruned third. Against a
# single-threaded BLAS the same code is worth 2.6-2.9x.
#
# That second number is the one that matters, because a particle filter's usual
# job is supplying a likelihood to a sampler, and samplers are run with BLAS
# pinned to one thread so the chains do not fight over cores. Keeping the
# block-level parallelism means the filter still scales in exactly that setting,
# and costs nothing measurable in the setting where BLAS is doing the work
# instead.
function propagate_cloud!(::Val{algo}, tr::ParticleTransition, bs::BatchScratch,
                          Xn::NTuple{K,Matrix{Float64}}, X::NTuple{K,Matrix{Float64}},
                          E::Matrix{Float64}) where {algo, K}
    N = size(E, 2)
    blk = bs.blk
    n_blocks = cld(N, blk)
    n_tasks = min(length(bs.slots), n_blocks)

    if n_tasks <= 1
        for b in 1:n_blocks
            propagate_block!(Val(algo), tr, bs.slots[1], Xn, X, E, block_cols(b, blk, N))
        end
        return Xn
    end

    per_task = cld(n_blocks, n_tasks)
    @sync for i in 1:n_tasks
        first_block = (i - 1) * per_task + 1
        first_block > n_blocks && break
        last_block = min(i * per_task, n_blocks)
        slot = bs.slots[i]
        Threads.@spawn for b in first_block:last_block
            propagate_block!(Val(algo), tr, slot, Xn, X, E, block_cols(b, blk, N))
        end
    end
    return Xn
end

@inline block_cols(b::Int, blk::Int, N::Int) = ((b - 1) * blk + 1):min(b * blk, N)

# `DEFAULT_PARTICLE_COPY_CHUNK` and `DEFAULT_PARTICLE_COPY_MAX_TASKS` set the
# columns per task for the pure-copy passes (the resampling gather and the
# Metropolis accept). Those are memory-bound rather than arithmetic-bound, so
# they saturate after a handful of threads and want far coarser chunks than the
# transition does — splitting them finely just buys task overhead.
#
# Run `f` over a fixed partition of `1:N` into contiguous column ranges, in
# parallel once there are enough of them to be worth it. The partition does not
# depend on the scheduler, and neither does the result: every use below writes
# into columns its own range owns.
@inline function foreach_column_chunk(f::F, N::Int) where {F}
    nt = min(max(Threads.nthreads(), 1), DEFAULT_PARTICLE_COPY_MAX_TASKS, N ÷ DEFAULT_PARTICLE_COPY_CHUNK)
    if nt <= 1
        f(1:N)
        return nothing
    end
    per = cld(N, nt)
    @sync for i in 1:nt
        lo = (i - 1) * per + 1
        lo > N && break
        hi = min(i * per, N)
        Threads.@spawn f(lo:hi)
    end
    return nothing
end

# The full model state each particle reports to the measurement equation: the
# single component at non-pruned orders (no copy), the sum of the components at
# pruned orders (written into `F`).
@inline full_states!(F::Matrix{Float64}, X::NTuple{1,Matrix{Float64}}) = X[1]
@inline function full_states!(F::Matrix{Float64}, X::NTuple{K,Matrix{Float64}}) where {K}
    copyto!(F, X[1])
    @inbounds for k in 2:K
        Xk = X[k]
        for i in eachindex(F)
            F[i] += Xk[i]
        end
    end
    return F
end

# Take `n_groups` clouds of `K` components each out of the workspace pools.
function cloud_group(pools::Vector{Matrix{Float64}}, group::Int, ::Val{K}) where {K}
    off = (group - 1) * K
    return ntuple(k -> pools[off + k], Val(K))
end

# Y[:, j] = X[:, idx[j]] for every component (the resampling gather). This moves
# as many bytes per stage as the transition does arithmetic, so it is chunked
# across threads the same way.
function gather_cloud!(Y::NTuple{K,Matrix{Float64}}, X::NTuple{K,Matrix{Float64}}, idx::Vector{Int}) where {K}
    foreach_column_chunk(length(idx)) do cols
        @inbounds for k in 1:K
            Yk = Y[k]
            Xk = X[k]
            nrow = size(Xk, 1)
            for j in cols
                a = idx[j]
                for i in 1:nrow
                    Yk[i, j] = Xk[i, a]
                end
            end
        end
    end
    return Y
end

@inline function copy_cloud!(Y::NTuple{K,Matrix{Float64}}, X::NTuple{K,Matrix{Float64}}) where {K}
    @inbounds for k in 1:K
        copyto!(Y[k], X[k])
    end
    return Y
end

# Copy column `src` of `X` into column `dst` of `Y` (contiguous, allocation-free).
@inline function copy_col!(Y::AbstractMatrix{Float64}, dst::Int, X::AbstractMatrix{Float64}, src::Int)
    @inbounds for i in axes(X, 1)
        Y[i, dst] = X[i, src]
    end
    return Y
end

@inline function copy_cloud_col!(Y::NTuple{K,Matrix{Float64}}, dst::Int, X::NTuple{K,Matrix{Float64}}, src::Int) where {K}
    @inbounds for k in 1:K
        copy_col!(Y[k], dst, X[k], src)
    end
    return Y
end

# The initial state as a plain vector per pruned component.
state_components(state) = state isa AbstractVector{<:AbstractVector} ?
    [Vector{Float64}(c) for c in state] : [Vector{Float64}(state)]

# Put the deterministic initial state into every column of the cloud.
function fill_cloud_from_state!(X::NTuple{K,Matrix{Float64}}, state) where {K}
    comps = state_components(state)
    @inbounds for k in 1:K
        Xk = X[k]
        mk = k <= length(comps) ? comps[k] : nothing
        for p in axes(Xk, 2), i in axes(Xk, 1)
            Xk[i, p] = mk === nothing ? 0.0 : mk[i]
        end
    end
    return X
end

# Seed the cloud: the first-order component is drawn around the initial mean with
# covariance L Lᵀ, the higher pruned components are set deterministically (they
# have no unconditional spread of their own at the start of the sample).
function init_cloud!(X::NTuple{K,Matrix{Float64}}, rng::Random.AbstractRNG, state,
                     L::Matrix{Float64}, Z::Matrix{Float64}) where {K}
    fill_cloud_from_state!(X, state)
    Random.randn!(rng, Z)
    ℒ.mul!(X[1], L, Z, 1.0, 1.0)                    # x¹ = mean + L·z
    return X
end

# Allocate a cloud outside the workspace (used by the shock decomposition, which
# runs a handful of deterministic trajectories rather than a particle swarm).
alloc_cloud(::Val{K}, nVars::Int, N::Int) where {K} = ntuple(_ -> zeros(Float64, nVars, N), Val(K))

# Common setup every `run_particle_filter` method needs.
function particle_filter_setup(::Val{algo}, 𝐒, T, 𝓂, measurement_error, n_particles,
                               initial_state_prior_scaling_factor, initial_covariance, opts) where {algo}
    me_var = build_particle_measurement_error(measurement_error)
    assert_positive_measurement_error(me_var)

    tr  = build_particle_transition(Val(algo), 𝐒, T)
    scr = build_batch_scratch(Val(algo), tr.naug, n_particles)

    Σ, _ = particle_initial_state_covariance(𝓂, T, opts, initial_covariance)
    L = particle_initial_cloud_factor(Σ, Float64(initial_state_prior_scaling_factor))

    pws = ensure_particle_workspace!(𝓂.workspaces, T.nVars, T.nExo, n_particles)

    return me_var, me_inverse_diagonal(me_var), tr, scr, L, pws
end


# ── Bootstrap (sequential importance resampling) particle filter ─────────────
#
# One iteration of the loop is the textbook predict / weight / resample cycle:
#
#   1. PREDICT  draw a fresh shock for every particle and push the swarm through
#               the model's state transition. The cloud now represents
#               p(xₜ | y₁..ₜ₋₁).
#   2. WEIGHT   score each particle by how well it explains today's observation,
#               p(yₜ | xₜ). Averaging those scores over the (weighted) cloud is an
#               unbiased estimate of the period's likelihood contribution, which
#               is what gets accumulated into `loglik`.
#   3. RESAMPLE if the weights have become too uneven, replace the weighted cloud
#               by an equally weighted one so the next predict step spends its
#               particles where the probability mass actually is.

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
                             particle_target_ratio::Real = DEFAULT_PARTICLE_TARGET_RATIO,
                             particle_mh_steps::Int = DEFAULT_TEMPERED_MH_STEPS,
                             particle_max_stages::Int = DEFAULT_PARTICLE_MAX_STAGES,
                             particle_mh_scale::Real = DEFAULT_PARTICLE_MH_SCALE,
                             opts::CalculationOptions = merge_calculation_options())::Float64 where {algo}
    T = constants.post_model_macro
    nT = size(data_in_deviations, 2)

    me_var, inv_me_var, tr, scr, L, pws =
        particle_filter_setup(Val(algo), 𝐒, T, 𝓂, measurement_error, n_particles,
                              particle_initial_state_scaling, initial_covariance, opts)

    K = pruned_components(Val(algo))
    nK = n_components(K)
    pools = ensure_particle_pools!(pws, 2 * nK + 1)
    X  = cloud_group(pools, 1, K)
    X_scratch = cloud_group(pools, 2, K)
    Fbuf = pools[2 * nK + 1]

    init_cloud!(X, particle_rng, state, L, Fbuf)

    return bootstrap_loop!(Val(algo), tr, scr, X, X_scratch, Fbuf, pws.E, pws.W, pws.logdens,
                           pws.idx, pws.bins, nT, normalize_presample_periods(presample_periods, nT),
                           observables_index, data_in_deviations, obs_idx_per_t, has_missing,
                           me_var, inv_me_var, particle_resampling, particle_resampling_threshold,
                           particle_rng, Float64(on_failure_loglikelihood), log(2π))
end

# Function barrier: the enclosing kwarg method body is too large for inference to
# keep the cloud types, so the hot loop lives here and runs allocation-free.
function bootstrap_loop!(::Val{algo}, tr, scr, X::NTuple{K,Matrix{Float64}}, X_scratch::NTuple{K,Matrix{Float64}},
                         Fbuf, E, W, logdens, idx, bins, nT, presample_periods, observables_index,
                         data_in_deviations, obs_idx_per_t, has_missing, me_var, inv_me_var,
                         resampling, resampling_threshold, rng, on_failure_loglikelihood, log2pi) where {algo, K}
    n_particles = size(E, 2)
    fill!(W, 1.0 / n_particles)
    loglik = 0.0

    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view data_in_deviations[:, t]

        # 1. PREDICT
        Random.randn!(rng, E)
        propagate_cloud!(Val(algo), tr, scr, X_scratch, X, E)
        X, X_scratch = X_scratch, X

        isempty(rows) && continue      # nothing observed: weights unchanged

        # 2. WEIGHT. The period's likelihood contribution is log Σₚ Wₚ·p(yₜ|xₜᵖ);
        #    `reweight_log_weights!` factors out the largest log-density first
        #    (log-sum-exp) so the exponentials cannot underflow to zero when every
        #    particle fits the data poorly.
        F = full_states!(Fbuf, X)
        score_cloud!(logdens, F, data_col, observables_index, me_var, inv_me_var, rows, log2pi)
        ll_t = reweight_log_weights!(W, logdens)
        isfinite(ll_t) || return on_failure_loglikelihood

        if t > presample_periods       # presample periods only warm the cloud up
            loglik += ll_t
        end

        # 3. RESAMPLE, but only once the cloud has actually degenerated. Doing it
        #    every period would add resampling noise for nothing; doing it never
        #    would leave all the weight on a single particle within a few periods.
        if effective_sample_size(W) < resampling_threshold * n_particles
            particle_resample_indices!(idx, bins, rng, W, resampling)
            gather_cloud!(X_scratch, X, idx)
            X, X_scratch = X_scratch, X
            fill!(W, 1.0 / n_particles)   # survivors are equally likely again
        end
    end

    return isfinite(loglik) ? loglik : on_failure_loglikelihood
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

# How far each observable can plausibly move in one period, used to spread the
# preview density above.
#
# The preview scores an ancestor at its zero-shock prediction. Judging it with the
# measurement-error variance alone would be far too strict: next period's shock
# will move the observable too, and an ancestor should not be discarded for
# missing the data by an amount a normal shock could easily cover. So the spread
# used here is "how much next period's shock moves this observable" plus "how
# noisily it is measured" — the first term being the row's own shock loading, the
# variance of the observable under a unit-normal shock draw.
#
# Getting this wrong is a matter of efficiency rather than correctness (the second
# stage divides the preview back out either way), but too tight a spread makes the
# preview reject almost every ancestor and collapses the cloud.
function auxiliary_predictive_variance(𝓂::ℳ, T, observables_index, me_var)
    nPast = T.nPast_not_future_and_mixed
    S₁ = 𝓂.caches.first_order_solution_matrix
    return Float64[sum(abs2, @view S₁[observables_index[i], nPast+1:end]) for i in eachindex(observables_index)] .+ me_diagonal(me_var)
end

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
                             particle_target_ratio::Real = DEFAULT_PARTICLE_TARGET_RATIO,
                             particle_mh_steps::Int = DEFAULT_TEMPERED_MH_STEPS,
                             particle_max_stages::Int = DEFAULT_PARTICLE_MAX_STAGES,
                             particle_mh_scale::Real = DEFAULT_PARTICLE_MH_SCALE,
                             opts::CalculationOptions = merge_calculation_options())::Float64 where {algo}
    T = constants.post_model_macro
    nT = size(data_in_deviations, 2)

    me_var, inv_me_var, tr, scr, L, pws =
        particle_filter_setup(Val(algo), 𝐒, T, 𝓂, measurement_error, n_particles,
                              particle_initial_state_scaling, initial_covariance, opts)

    pred_var = auxiliary_predictive_variance(𝓂, T, observables_index, me_var)

    K = pruned_components(Val(algo))
    nK = n_components(K)
    pools = ensure_particle_pools!(pws, 3 * nK + 1)
    X   = cloud_group(pools, 1, K)
    X_scratch  = cloud_group(pools, 2, K)
    anc = cloud_group(pools, 3, K)
    Fbuf = pools[3 * nK + 1]

    init_cloud!(X, particle_rng, state, L, Fbuf)

    return auxiliary_loop!(Val(algo), tr, scr, X, X_scratch, anc, Fbuf, pws.E, pws.W, pws.logdens,
                           pws.logw, pws.lam, pws.idx, pws.bins, nT,
                           normalize_presample_periods(presample_periods, nT),
                           observables_index, data_in_deviations, obs_idx_per_t, has_missing,
                           me_var, inv_me_var, pred_var, 1.0 ./ pred_var, particle_resampling,
                           particle_rng, Float64(on_failure_loglikelihood), log(2π))
end

function auxiliary_loop!(::Val{algo}, tr, scr, X::NTuple{K,Matrix{Float64}}, X_scratch::NTuple{K,Matrix{Float64}},
                         anc::NTuple{K,Matrix{Float64}}, Fbuf, E, W, logg̃, logw, lam, idx, bins,
                         nT, presample_periods, observables_index, data_in_deviations, obs_idx_per_t,
                         has_missing, me_var, inv_me_var, pred_var, inv_pred_var, resampling, rng,
                         on_failure_loglikelihood, log2pi) where {algo, K}
    n_particles = size(E, 2)
    fill!(W, 1.0 / n_particles)
    loglik = 0.0

    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view data_in_deviations[:, t]

        # First stage: predictive density at the transition mean (zero shock),
        # spread by the shock-induced predictive variance `pred_var`.
        fill!(E, 0.0)
        propagate_cloud!(Val(algo), tr, scr, X_scratch, X, E)
        Fμ = full_states!(Fbuf, X_scratch)
        score_cloud!(logg̃, Fμ, data_col, observables_index, pred_var, inv_pred_var, rows, log2pi)

        # First-stage (auxiliary) weights λ ∝ W·g̃, with κ = Σ W·g̃.
        copyto!(lam, W)
        logκ = reweight_log_weights!(lam, logg̃)
        isfinite(logκ) || return on_failure_loglikelihood

        # Resample ancestors ∝ λ, propagate with fresh shocks, second-stage weight
        # w = g(yₜ|xₜ) / g̃(ancestor).
        particle_resample_indices!(idx, bins, rng, lam, resampling)
        gather_cloud!(anc, X, idx)
        Random.randn!(rng, E)
        propagate_cloud!(Val(algo), tr, scr, X_scratch, anc, E)
        F = full_states!(Fbuf, X_scratch)
        score_cloud!(logw, F, data_col, observables_index, me_var, inv_me_var, rows, log2pi)
        @inbounds for j in 1:n_particles
            logw[j] -= logg̃[idx[j]]
        end
        X, X_scratch = X_scratch, X

        logsw = normalise_log_weights!(W, logw)
        isfinite(logsw) || return on_failure_loglikelihood

        ll_t = logκ + logsw - log(n_particles)
        if t > presample_periods
            loglik += ll_t
        end
    end

    return isfinite(loglik) ? loglik : on_failure_loglikelihood
end


# ── Guided (conditionally optimal proposal) particle filter ──────────────────
#
# The observation of this filter is that a DSGE usually has as many structural
# shocks as observables, and a measurement error that is small next to the data.
# Given the ancestor xₜ₋₁, the observation then very nearly *determines* εₜ, and
# the conditional
#
#     p(εₜ | xₜ₋₁, yₜ) ∝ N(εₜ; 0, I) · N(yₜ; C·g(xₜ₋₁, εₜ), H)
#
# is available in closed form. Linearising the observed transition in the shock,
# C·g(xₜ₋₁, ε) ≈ mₚ + Bₒ ε with Bₒ the first-order shock loading on the observed
# rows and mₚ the zero-shock prediction, gives
#
#     p(εₜ | xₜ₋₁, yₜ) = N(μₚ, M⁻¹),   M = I + BₒᵀH⁻¹Bₒ,   μₚ = M⁻¹BₒᵀH⁻¹rₚ,
#     p(yₜ | xₜ₋₁)     = N(yₜ; mₚ, H + BₒBₒᵀ),
#
# with rₚ = yₜ - mₚ. Two things are worth noticing. M does not depend on the
# particle — only on the model and H — so it is factorised once per missing-data
# pattern, and μₚ is one small matrix product away from the residual. And the
# predictive density needed to *choose* ancestors is closed form too, so the
# filter can be fully adapted in the sense of Pitt & Shephard: pick ancestors by
# how well they explain yₜ before drawing any shock, then draw the shock from its
# own conditional.
#
# Why the ancestors are *not* preselected. Being able to score ancestors before
# drawing a shock is tempting: kill the hopeless ones early and spend the whole
# cloud on the promising ones, which is what "full adaptation" in the sense of
# Pitt & Shephard means. That was implemented and measured, and it made things
# worse, for a reason worth stating.
#
# The score λ is not the real predictive density, only the Gaussian approximation
# to it. Where the model fits, the two agree. Where it does not — a crisis period,
# an ancestor far out in the tail — the approximation is optimistic: it claims the
# ancestor can explain the observation much better than it actually can. Selecting
# on λ then hands most of the cloud to exactly the particles whose score is least
# trustworthy. The correction weight that follows does mark them down, but by then
# the selection has happened and there is nothing left to correct: the cloud is
# already made of copies of one bad ancestor.
#
# What makes this pathological rather than merely inefficient is that it does not
# improve with `n_particles`. A larger cloud reaches further into the tail, so it
# finds more of the ancestors the approximation flatters — measured on a euro-area
# model, a tenfold increase in particles left the worst period's effective sample
# size at a handful of particles and made the run-to-run spread of the estimates
# visibly *worse*. The usual remedy for a noisy particle filter does not apply.
#
# Resampling once on the combined weight removes the failure mode outright, and it
# is worth seeing why it is free rather than a trade-off. Draw the shock first and
# form a single weight afterwards:
#
#     wⱼ = N(εⱼ;0,I)·p(yₜ|g(xₜ₋₁,εⱼ)) / q(εⱼ)
#         = exp(logZ - ½(‖εⱼ‖² + rⱼᵀH⁻¹rⱼ - ‖zⱼ‖²)).
#
# The proposal q contains λ, so λ appears in the numerator and the denominator and
# divides out exactly before anything is selected. An over-confident λ therefore
# costs nothing — it never gets a vote on which particles survive. It is kept only
# as a diagnostic.
#
# What this buys. The bootstrap proposal ignores yₜ when drawing εₜ, and the
# tempered filter recovers the lost information by running a within-period MCMC —
# tens of transition evaluations per period. Here the same information is used
# directly: two transition evaluations per period, one at ε = 0 to get the
# residual and one at the drawn shock. And the importance weight
#
#     ωⱼ = N(εⱼ;0,I)·p(yₜ|g(xₜ₋₁,εⱼ)) / (λ^{a(j)}·q(εⱼ))
#
# is *identically one* when the transition is linear in the shock: every term
# involving εⱼ cancels. At pruned second order it is one up to the curvature the
# linearisation misses, which is exactly the residual the perturbation itself
# treats as small. So the weights barely vary, the cloud barely degenerates, and
# there is only one resampling per period rather than one per tempering stage.
#
# References. The conditionally optimal importance function is Doucet, Godsill &
# Andrieu (2000); full adaptation is Pitt & Shephard (1999). Building the proposal
# from a local Gaussian approximation is the "unscented"/"optimised" particle
# filter family (van der Merwe, Doucet, de Freitas & Wan, 2000; Andreasen, 2013,
# for DSGE), and solving for the shock that explains the observation before
# sampling around it is the implicit particle filter of Chorin, Morzfeld & Tu
# (2010) from geophysical data assimilation.

# Everything the proposal needs, precomputed per missing-data pattern: the
# observed shock loading, the Cholesky factor of M (`U`, upper, M = UᵀU), its
# inverse (so `Uinv·z` has covariance M⁻¹), the map `K` from residual to
# conditional mean, and the log normalisation of p(yₜ|xₜ₋₁).
#
# Every buffer is allocated once at the full observable count and refilled in
# place by `rebuild_guided_proposal!` whenever the missing-data pattern changes.
# `d` is how many observables are live; `Bo`, `HinvBo` and `K` use only their
# first `d` rows (columns for `K`), the rest are stale.
#
# Everything here is sized by model dimensions — one row per shock, one per
# observable — so a rebuild was never expensive. The preallocation is worth having
# because with ragged data the pattern can change every period, and it keeps that
# case down to a constant handful of bytes per rebuild (the factorization
# wrapper) instead of a fresh set of matrices; it is not expected to save
# measurable time in the common case of one rebuild per run.
mutable struct GuidedProposal
    const U::Matrix{Float64}       # nExo × nExo, upper Cholesky factor of M = I + BₒᵀH⁻¹Bₒ
    const Uinv::Matrix{Float64}    # nExo × nExo, U⁻¹; Uinv·Uinvᵀ = M⁻¹
    const Minv::Matrix{Float64}    # nExo × nExo, M⁻¹, for the Newton refinement below
    const K::Matrix{Float64}       # nExo × nObs, M⁻¹BₒᵀH⁻¹ in columns 1:d, so μ = K·r
    const Bo::Matrix{Float64}      # nObs × nExo, observed shock loading in rows 1:d
    const HinvBo::Matrix{Float64}  # nObs × nExo, H⁻¹Bₒ in rows 1:d
    const M::Matrix{Float64}       # nExo × nExo, overwritten by its own Cholesky factor
    logZ::Float64                  # -½(d·log2π + log|H| + log|M|)
    d::Int                         # live observables
end

function GuidedProposal(nObs::Int, nExo::Int)
    return GuidedProposal(Matrix{Float64}(undef, nExo, nExo), Matrix{Float64}(undef, nExo, nExo),
                          Matrix{Float64}(undef, nExo, nExo), Matrix{Float64}(undef, nExo, nObs),
                          Matrix{Float64}(undef, nObs, nExo), Matrix{Float64}(undef, nObs, nExo),
                          Matrix{Float64}(undef, nExo, nExo), 0.0, 0)
end

# The live blocks. `K` is what the filters multiply the residual by, so it is the
# one that has to be sliced at every use.
@inline proposal_K(gp::GuidedProposal) = view(gp.K, :, 1:gp.d)

# How many Newton steps refine the proposal's centre
# (`DEFAULT_GUIDED_NEWTON_STEPS`).
#
# `μ = K·r(0)` is the mode only when the observed transition is linear in the
# shock. At pruned second order it is not, and a mis-centred proposal in seven
# dimensions against a target this tight is expensive. Each Newton step
# re-evaluates the *true* residual at the current centre and moves towards the mode
# of the exact conditional,
#     ε ← ε + M⁻¹(BₒᵀH⁻¹r(ε) - ε),
# which is the Gauss-Newton iteration on -½‖ε‖² - ½r(ε)ᵀH⁻¹r(ε). Finding the mode
# and sampling around it with the Laplace covariance is the implicit particle
# filter of Chorin, Morzfeld & Tu (2010).
#
# Two is measured to be the right number on mid-sized models, and the reason it is
# not a speed/accuracy trade-off is worth knowing: a Newton step costs one batched
# transition, but so does a bridging stage, and a badly centred proposal needs more
# bridging stages. Skipping the refinement therefore buys no time — it just moves
# the same work somewhere less useful. Going from none to two improves the
# estimates and gets slightly *faster*; past two the centre has stopped moving and
# the extra transitions are wasted.

# Width of the proposal, as a multiple of the Laplace scale
# (`DEFAULT_GUIDED_PROPOSAL_SCALE`). Kept at one, and the measurement that says so
# is worth recording.
#
# `M⁻¹` is the curvature of a *linearised* problem, so it is a guess at the true
# conditional's spread. Importance sampling is not symmetric in that error — a
# proposal wider than the target has bounded weights, one narrower can have infinite
# weight variance — so deliberately over-dispersing looks like cheap insurance
# against the crisis periods where this filter degenerates (Hesterberg's defensive
# importance sampling, 1995). Measured on the euro-area problem it is not: at scales
# 1.0 / 1.5 / 2.5 the seed dispersion went 0.091 / 0.090 / 0.138 and the mean weight
# ESS fell 0.235 / 0.174 / 0.015, while the *worst* period's effective sample size
# stayed at one particle in four thousand at every scale.
#
# That last number is the informative one: widening the proposal does not find the
# missing mass at all. The crisis-period failure is that the conditional's mode is
# somewhere the Gaussian approximation does not put it, not that the Gaussian is too
# narrow — so it wants a remedy that *relocates* (annealing from this proposal to the
# true conditional, or a block move over several periods), not one that inflates.

# Observed rows of the first-order shock loading, into the proposal's own buffer.
function observed_shock_loading!(Bo::AbstractMatrix{Float64}, S₁::AbstractMatrix,
                                 observables_index::Vector{Int}, nPast::Int, rows)
    @inbounds for (k, r) in enumerate(rows)
        Bo[k, :] .= @view S₁[observables_index[r], nPast+1:end]
    end
    return Bo
end

# H⁻¹Bₒ, for either measurement-error form, written into `HinvBo`. Returns log|H|.
function guided_hinv_loading!(HinvBo::AbstractMatrix{Float64}, Bo::AbstractMatrix{Float64},
                              me_var::AbstractVector, rows)
    @inbounds for (k, r) in enumerate(rows)
        HinvBo[k, :] .= @view(Bo[k, :]) ./ me_var[r]
    end
    return sum(log(me_var[r]) for r in rows)
end

function guided_hinv_loading!(HinvBo::AbstractMatrix{Float64}, Bo::AbstractMatrix{Float64},
                              me::DenseMeasurementError, rows)
    me_sync!(me, rows)
    Lo = ℒ.LowerTriangular(me.last_L)
    copyto!(HinvBo, Bo)
    ℒ.ldiv!(Lo, HinvBo)               # L⁻¹Bₒ
    ℒ.ldiv!(Lo', HinvBo)              # H⁻¹Bₒ = L⁻ᵀL⁻¹Bₒ
    return me.last_logdet
end

# Refill `gp` for the missing-data pattern `rows`, reusing its buffers throughout.
#
# The solve is a Cholesky of M = I + BₒᵀH⁻¹Bₒ, which is nExo × nExo — one row and
# column per structural shock. `cholesky!`/`ldiv!` are the direct LAPACK calls at
# that size; routing them through a `LinearSolve` cache the way the
# stochastic-steady-state Newton solves do would add the cache indirection without
# reaching a different kernel, so it is deliberately not done here.
function rebuild_guided_proposal!(gp::GuidedProposal, S₁::AbstractMatrix,
                                  observables_index::Vector{Int}, nPast::Int,
                                  me_var, rows, log2pi::Float64)
    d = length(rows)
    gp.d = d
    Bo     = view(gp.Bo, 1:d, :)
    HinvBo = view(gp.HinvBo, 1:d, :)

    observed_shock_loading!(Bo, S₁, observables_index, nPast, rows)
    logdetH = guided_hinv_loading!(HinvBo, Bo, me_var, rows)

    copyto!(gp.M, ℒ.I)
    ℒ.mul!(gp.M, Bo', HinvBo, 1.0, 1.0)                # M = I + BₒᵀH⁻¹Bₒ
    F = ℒ.cholesky!(ℒ.Symmetric(gp.M))                 # overwrites gp.M with its factor

    copyto!(gp.U, F.U)
    logdetM = 2 * sum(log, ℒ.diag(gp.U))

    copyto!(gp.Uinv, ℒ.I)
    ℒ.ldiv!(ℒ.UpperTriangular(gp.U), gp.Uinv)          # U⁻¹, so Uinv·Uinvᵀ = M⁻¹
    ℒ.mul!(gp.Minv, gp.Uinv, gp.Uinv')

    K = view(gp.K, :, 1:d)
    ℒ.transpose!(K, HinvBo)
    ℒ.ldiv!(F, K)                                      # K = M⁻¹BₒᵀH⁻¹

    gp.logZ = -0.5 * (d * log2pi + logdetH + logdetM)
    return gp
end

# Annealing from the proposal to the truth.
#
# The guided proposal is a Gaussian approximation, and in the periods this model
# can barely explain it is the wrong Gaussian — not too narrow (widening it was
# measured and does not help) but centred somewhere the true conditional's mass is
# not. A single importance-weighting step then has heavy-tailed weights and the
# cloud collapses onto a handful of particles however many were started with.
#
# The fix is to reach the truth gradually instead of in one step. Bridge
#
#     γ_β(ε) ∝ q(ε)^(1-β) · π̃(ε)^β ,   β: 0 → 1,   π̃(ε) = N(ε;0,I)·N(yₜ; ŷ(ε), H),
#
# reweighting, resampling and mutating along the way. At β = 0 the particles are
# exact draws from q, at β = 1 they target the conditional. Writing
#
#     L(ε) = log π̃(ε) - log q(ε) = logZ - ½‖ε‖² - ½·e(ε)ᵀH⁻¹e(ε) + ½(ε-μ)ᵀM(ε-μ),
#
# the incremental weight between two levels is exp((β' - β)·L), so the same
# inefficiency-targeting schedule the tempered filter uses picks the steps — with
# `-L` in place of its quadratic form. The one-step case is exactly the plain
# guided filter, so this is a strict generalisation: where the proposal is good the
# schedule jumps straight to β = 1 and costs nothing extra, and only the awkward
# periods pay for more stages.
#
# This is annealed importance sampling (Neal, 2001) started from a Laplace
# approximation rather than from the prior, which is how SMC samplers are usually
# initialised; the difference from the tempered filter is only *what* it bridges
# from, and that is what makes it cheap.

# ‖U(ε - μ)‖² = (ε-μ)ᵀM(ε-μ) for column `p`, with `U` the upper Cholesky factor of M.
@inline function mahalanobis_M(E::Matrix{Float64}, Mu::Matrix{Float64}, U::Matrix{Float64}, p::Int)
    n = size(E, 1)
    acc = 0.0
    @inbounds for i in 1:n
        v = 0.0
        for j in i:n
            v += U[i, j] * (E[j, p] - Mu[j, p])
        end
        acc += v * v
    end
    return acc
end

# L(ε) for every particle, given the measurement quadratic forms `dv`.
function guided_bridge_gap!(Lvec::Vector{Float64}, E::Matrix{Float64}, Mu::Matrix{Float64},
                            dv::Vector{Float64}, gp::GuidedProposal)
    nExo = size(E, 1)
    @inbounds for p in eachindex(Lvec)
        d = dv[p]
        if !isfinite(d)
            Lvec[p] = -Inf
            continue
        end
        esq = 0.0
        for e in 1:nExo
            esq += E[e, p] * E[e, p]
        end
        Lvec[p] = gp.logZ - 0.5 * esq - 0.5 * d + 0.5 * mahalanobis_M(E, Mu, gp.U, p)
    end
    return Lvec
end

# One Metropolis sweep against γ_β, preconditioned by M⁻¹ (the same factor the
# proposal uses, which is the right shape at both ends of the bridge). Returns the
# acceptance rate.
function guided_anneal_mutate!(::Val{algo}, tr, scr, gp, St::NTuple{K,Matrix{Float64}},
                               parts_proposed::NTuple{K,Matrix{Float64}}, anc::NTuple{K,Matrix{Float64}},
                               Fbuf, E, Eprop, Z, R, Mu, dv, dprop, accept,
                               c::Float64, β::Float64, data_col, observables_index,
                               inv_me_var, rows, rng) where {algo, K}
    n_particles = size(E, 2)
    nExo = size(E, 1)

    Random.randn!(rng, Z)
    ℒ.mul!(Eprop, gp.Uinv, Z, c, 0.0)                 # ε' - ε = c·U⁻¹z
    @inbounds for i in eachindex(Eprop)
        Eprop[i] += E[i]
    end

    propagate_cloud!(Val(algo), tr, scr, parts_proposed, anc, Eprop)
    residual_cloud!(R, full_states!(Fbuf, parts_proposed), data_col, observables_index, rows)

    accepted = 0
    @inbounds for p in 1:n_particles
        # log γ_β = (1-β)·log q + β·log π̃, so the ratio needs both ends of the
        # bridge at both shocks: the proposal's Mahalanobis form under M for q,
        # and ‖ε‖² together with the measurement quadratic form for π̃.
        measurement_proposed = residual_quadform(R, p, inv_me_var, rows)
        dprop[p] = measurement_proposed
        isfinite(measurement_proposed) || continue
        shock_norm²_current  = 0.0
        shock_norm²_proposed = 0.0
        for e in 1:nExo
            current  = E[e, p]
            proposed = Eprop[e, p]
            shock_norm²_current  += current * current
            shock_norm²_proposed += proposed * proposed
        end
        proposal_form_current  = mahalanobis_M(E, Mu, gp.U, p)
        proposal_form_proposed = mahalanobis_M(Eprop, Mu, gp.U, p)
        logα = -0.5 * (1 - β) * (proposal_form_proposed - proposal_form_current) -
               0.5 * β * (shock_norm²_proposed - shock_norm²_current) -
               0.5 * β * (measurement_proposed - dv[p])
        acc = log(rand(rng)) < logα
        accept[p] = acc
        if acc
            dv[p] = measurement_proposed
            accepted += 1
        else
            accept[p] = false
        end
    end

    @inbounds for p in 1:n_particles
        if accept[p]
            copy_col!(E, p, Eprop, p)
            copy_cloud_col!(St, p, parts_proposed, p)
        end
    end

    return accepted / n_particles
end

# yₜ - (predicted observables) for every column of `F`, into `R` (d × N).
function residual_cloud!(R::Matrix{Float64}, F::Matrix{Float64}, data_col, observables_index, rows)
    @inbounds for p in axes(R, 2)
        for k in eachindex(rows)
            r = rows[k]
            f = F[observables_index[r], p]
            R[k, p] = isfinite(f) ? data_col[r] - f : NaN
        end
    end
    return R
end

# rᵀH⁻¹r for column `p` of a residual block.
@inline function residual_quadform(R::Matrix{Float64}, p::Int, inv_me_var::AbstractVector, rows)
    q = 0.0
    @inbounds for k in eachindex(rows)
        v = R[k, p]
        isfinite(v) || return Inf
        q += v * v * inv_me_var[rows[k]]
    end
    return q
end

@inline function residual_quadform(R::Matrix{Float64}, p::Int, me::DenseMeasurementError, rows)
    me_sync!(me, rows)
    v = me.buf
    @inbounds for k in eachindex(rows)
        x = R[k, p]
        isfinite(x) || return Inf
        v[k] = x
    end
    return dense_me_quadform!(v, me.last_L)
end

# One Gauss-Newton step towards the mode: μ ← μ + M⁻¹(BₒᵀH⁻¹r(μ) - μ), batched.
# `R` holds the residuals at the current `Mu`; `Tmp` is nExo × N scratch.
#
# This does not need as many observables as shocks, or the reverse. The step is
# taken in shock space throughout: `K` is nExo × d and `M` is nExo × nExo however
# many observables `d` are live. Nor can it break down when d < nExo — M is
# I + BₒᵀH⁻¹Bₒ, a positive-definite matrix plus the identity, so it is invertible
# even when Bₒ has a large null space and the observation pins down only some
# directions of the shock. The unidentified directions simply keep their prior:
# where Bₒ says nothing, M is the identity there and the step leaves μ at zero.
# That is exactly the case the inversion filter cannot handle at all, and it is
# the reason the particle filters accept fewer shocks than observables as well as
# more.
function guided_newton_step!(Mu::Matrix{Float64}, R::Matrix{Float64}, Tmp::Matrix{Float64},
                             gp::GuidedProposal)
    ℒ.mul!(Tmp, proposal_K(gp), view(R, 1:gp.d, :))   # M⁻¹BₒᵀH⁻¹r(μ)
    ℒ.mul!(Tmp, gp.Minv, Mu, -1.0, 1.0)         # ... - M⁻¹μ
    @inbounds for i in eachindex(Mu)
        Mu[i] += Tmp[i]
    end
    return Mu
end

# The first stage's predictive density, by Laplace approximation at the mode:
#     p(yₜ|xₜ₋₁) ≈ (2π)^(-d/2)|H|^(-1/2)|M|^(-1/2)·exp(-½‖μ‖² - ½r(μ)ᵀH⁻¹r(μ)).
# `R` must hold the residuals evaluated at `Mu`. When the transition is linear in
# the shock this reduces exactly to N(yₜ; mₚ, H + BₒBₒᵀ).
#
# Where the Laplace approximation sits in the algorithm, and where it does not.
# It does two jobs, and the reported likelihood is not one of them:
#
#   1. It shapes the proposal. Expanding log p(εₜ|xₜ₋₁,yₜ) to second order about
#      its mode gives a Gaussian with mean μ (the mode `guided_newton_step!`
#      refines towards) and covariance M⁻¹. That Gaussian is q, the distribution
#      the shocks are actually drawn from. A wrong Laplace approximation makes q
#      a poor fit — more bridging stages, a lower ESS — but never a wrong answer.
#
#   2. It is the λ this function returns, kept only as a diagnostic. λ appears in
#      both the numerator and the denominator of the importance weight and
#      divides out exactly (see the derivation above `GuidedProposal`), which is
#      what stops an over-confident approximation from steering the resampling.
#
# The likelihood the filter reports is the ordinary particle-filter estimate:
# log of the weighted average of the exact p(yₜ|xₜᵖ) over the cloud, accumulated
# across the bridging stages. Nothing in it is Gaussian by assumption. So the
# approximation controls the filter's *efficiency*, and its errors show up as
# variance rather than as bias.
function guided_lambda!(logλ::Vector{Float64}, Mu::Matrix{Float64}, R::Matrix{Float64},
                        gp::GuidedProposal, inv_me_var, rows)
    nExo = size(Mu, 1)
    @inbounds for p in eachindex(logλ)
        qμ = residual_quadform(R, p, inv_me_var, rows)
        if !isfinite(qμ)
            logλ[p] = -Inf
            continue
        end
        msq = 0.0
        for e in 1:nExo
            msq += Mu[e, p] * Mu[e, p]
        end
        logλ[p] = gp.logZ - 0.5 * msq - 0.5 * qμ
    end
    return logλ
end

function run_particle_filter(::Val{algo},
                             ::Val{:guided},
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
                             particle_target_ratio::Real = DEFAULT_PARTICLE_TARGET_RATIO,
                             particle_mh_steps::Int = DEFAULT_GUIDED_MH_STEPS,
                             particle_max_stages::Int = DEFAULT_PARTICLE_MAX_STAGES,
                             particle_mh_scale::Real = DEFAULT_PARTICLE_MH_SCALE,
                             opts::CalculationOptions = merge_calculation_options())::Float64 where {algo}
    T = constants.post_model_macro
    nT = size(data_in_deviations, 2)

    me_var, inv_me_var, tr, scr, L, pws =
        particle_filter_setup(Val(algo), 𝐒, T, 𝓂, measurement_error, n_particles,
                              particle_initial_state_scaling, initial_covariance, opts)

    K = pruned_components(Val(algo))
    nK = n_components(K)
    pools = ensure_particle_pools!(pws, 4 * nK + 1)
    X    = cloud_group(pools, 1, K)
    X_scratch   = cloud_group(pools, 2, K)
    anc  = cloud_group(pools, 3, K)
    anc_scratch = cloud_group(pools, 4, K)
    Fbuf = pools[4 * nK + 1]

    init_cloud!(X, particle_rng, state, L, Fbuf)

    return guided_loop!(Val(algo), tr, scr, X, X_scratch, anc, anc_scratch, Fbuf, pws.E, pws.E2, pws.Eprop,
                        pws.W, pws.logdens, pws.logw, pws.lam, pws.idx, pws.bins, nT,
                        normalize_presample_periods(presample_periods, nT),
                        observables_index, data_in_deviations, obs_idx_per_t, has_missing,
                        me_var, inv_me_var,
                        𝓂.caches.first_order_solution_matrix, T.nPast_not_future_and_mixed,
                        particle_resampling, Float64(particle_resampling_threshold),
                        particle_mh_steps, Float64(particle_mh_scale),
                        Float64(particle_target_ratio), particle_max_stages, particle_rng,
                        Float64(on_failure_loglikelihood), log(2π))
end

function guided_loop!(::Val{algo}, tr, scr, X::NTuple{K,Matrix{Float64}}, X_scratch::NTuple{K,Matrix{Float64}},
                      anc::NTuple{K,Matrix{Float64}}, anc_scratch::NTuple{K,Matrix{Float64}},
                      Fbuf, E, Mu, Tmp, W, logλ, logw, lam,
                      idx, bins, nT, presample_periods, observables_index, data_in_deviations,
                      obs_idx_per_t, has_missing, me_var, inv_me_var, S₁, nPast, resampling,
                      resampling_threshold, n_mh, mh_scale, r_star, max_stages,
                      rng, on_failure_loglikelihood, log2pi) where {algo, K}
    n_particles = size(E, 2)
    nExo = size(E, 1)
    R = Matrix{Float64}(undef, length(observables_index), n_particles)
    Z = Matrix{Float64}(undef, nExo, n_particles)
    Eprop = Matrix{Float64}(undef, nExo, n_particles)
    mu_scratch  = Matrix{Float64}(undef, nExo, n_particles)
    dv     = Vector{Float64}(undef, n_particles)
    dprop  = Vector{Float64}(undef, n_particles)
    Lvec   = Vector{Float64}(undef, n_particles)
    negL   = Vector{Float64}(undef, n_particles)
    accept = Vector{Bool}(undef, n_particles)
    c = mh_scale
    fill!(W, 1.0 / n_particles)
    loglik = 0.0

    gp_rows = Int[]
    gp = GuidedProposal(length(observables_index), nExo)
    rebuild_guided_proposal!(gp, S₁, observables_index, nPast, me_var, eachindex(observables_index), log2pi)

    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view data_in_deviations[:, t]

        if isempty(rows)
            Random.randn!(rng, E)
            propagate_cloud!(Val(algo), tr, scr, X_scratch, X, E)
            X, X_scratch = X_scratch, X
            continue
        end
        if !same_rows(gp_rows, rows)
            rebuild_guided_proposal!(gp, S₁, observables_index, nPast, me_var, rows, log2pi)
            gp_rows = collect(Int, rows)
        end

        # Solve for the shock that best explains yₜ from each ancestor: one
        # transition at ε = 0 for the residual, then Gauss-Newton steps that use the
        # true residual rather than the linearisation.
        fill!(E, 0.0)
        propagate_cloud!(Val(algo), tr, scr, X_scratch, X, E)
        residual_cloud!(R, full_states!(Fbuf, X_scratch), data_col, observables_index, rows)
        ℒ.mul!(Mu, proposal_K(gp), view(R, 1:length(rows), :))
        for _ in 1:DEFAULT_GUIDED_NEWTON_STEPS
            propagate_cloud!(Val(algo), tr, scr, X_scratch, X, Mu)
            residual_cloud!(R, full_states!(Fbuf, X_scratch), data_col, observables_index, rows)
            guided_newton_step!(Mu, R, Tmp, gp)
        end

        # Draw from the conditional, εⱼ = μⱼ + U⁻¹zⱼ, and weight.
        Random.randn!(rng, Z)
        ℒ.mul!(E, gp.Uinv, Z, DEFAULT_GUIDED_PROPOSAL_SCALE, 0.0)
        @inbounds for i in eachindex(E)
            E[i] += Mu[i]
        end
        copy_cloud!(anc, X)
        propagate_cloud!(Val(algo), tr, scr, X_scratch, anc, E)
        copy_cloud!(X, X_scratch)
        residual_cloud!(R, full_states!(Fbuf, X), data_col, observables_index, rows)

        @inbounds for j in 1:n_particles
            dv[j] = residual_quadform(R, j, inv_me_var, rows)
        end
        guided_bridge_gap!(Lvec, E, Mu, dv, gp)

        # Walk from the proposal to the conditional; one step where the proposal is
        # good, more only where it is not.
        ll_t = 0.0
        β_old = 0.0
        stage = 0
        @inbounds for j in 1:n_particles
            negL[j] = -Lvec[j]
        end
        while β_old < 1.0 - 1e-12 && stage < max_stages
            stage += 1
            β_new = any(isfinite, negL) ? tempered_next_phi(β_old, negL, r_star, n_particles) : 1.0
            @inbounds for j in 1:n_particles
                logw[j] = (β_new - β_old) * Lvec[j]
            end
            inc = reweight_log_weights!(W, logw)
            isfinite(inc) || return on_failure_loglikelihood
            ll_t += inc

            if effective_sample_size(W) < resampling_threshold * n_particles
                particle_resample_indices!(idx, bins, rng, W, resampling)
                gather_cloud!(X_scratch, X, idx)
                copy_cloud!(X, X_scratch)
                gather_cloud!(anc_scratch, anc, idx)
                copy_cloud!(anc, anc_scratch)
                @inbounds for j in 1:n_particles
                    a = idx[j]
                    copy_col!(Eprop, j, E, a)
                    copy_col!(mu_scratch, j, Mu, a)
                    dprop[j] = dv[a]
                end
                copyto!(E, Eprop)
                copyto!(Mu, mu_scratch)
                copyto!(dv, dprop)
                fill!(W, 1.0 / n_particles)
            end

            for _ in 1:n_mh
                acc = guided_anneal_mutate!(Val(algo), tr, scr, gp, X, X_scratch, anc, Fbuf,
                                            E, Eprop, Z, R, Mu, dv, dprop, accept, c, β_new,
                                            data_col, observables_index, inv_me_var, rows, rng)
                c = adapt_mh_scale(c, acc)
            end
            guided_bridge_gap!(Lvec, E, Mu, dv, gp)
            @inbounds for j in 1:n_particles
                negL[j] = -Lvec[j]
            end
            β_old = β_new
        end
        @debug "guided period" t stages = stage adaptation_ess = effective_sample_size(W) / n_particles

        if t > presample_periods
            loglik += ll_t
        end
    end

    return isfinite(loglik) ? loglik : on_failure_loglikelihood
end


# ── Tempered particle filter (Herbst & Schorfheide, 2019) ────────────────────
# Within each period the measurement information is introduced gradually through
# a bridging sequence 0 = φ₀ < φ₁ < … < φ_N = 1 (the measurement covariance is
# inflated to H/φ). Each stage reweights by the tempered density increment,
# resamples, and mutates the particles' shocks with a random-walk Metropolis step
# targeting the stage-φ posterior. This dramatically lowers the variance of both
# the likelihood estimate and the filtered moments relative to the bootstrap
# filter at equal particle count.

# Inefficiency ratio N·Σ(wᵖ)² / (Σwᵖ)² for the incremental weights
# wᵖ = exp(-(φ-φ_old)/2 · dᵖ). Increasing in φ, equal to 1 at φ = φ_old.
#
# The largest weight sits at the smallest dᵖ whatever the level, so the
# log-sum-exp shift is known before the loop and the whole ratio takes a single
# vectorised pass. Particles with dᵖ = Inf (an impossible prediction) fall out on
# their own: exp(-Δ·Inf) = 0 for any Δ > 0. This is called ~20 times per
# tempering stage by the bisection below, so the pass is worth having tight.
function tempered_inefficiency(Δ::Float64, dmin::Float64, d::Vector{Float64})
    n = length(d)
    S1 = 0.0
    S2 = 0.0
    @turbo for p in 1:n
        e = exp(-Δ * (d[p] - dmin))
        S1 += e
        S2 += e * e
    end
    return S1 > 0 && isfinite(S2) ? n * S2 / (S1 * S1) : Inf
end

# Next tempering level in (φ_old, 1] targeting inefficiency `r_star` by bisection.
# The bracket is only ever used to pick a step size, so a 1e-6 tolerance is well
# beyond what the schedule can notice and saves a third of the iterations.
function tempered_next_phi(φ_old::Float64, d::Vector{Float64}, r_star::Float64, n_particles::Int)
    dmin = Inf
    @inbounds for p in 1:n_particles
        dp = d[p]
        dmin = (dp < dmin && isfinite(dp)) ? dp : dmin
    end
    isfinite(dmin) || return 1.0

    if tempered_inefficiency((1.0 - φ_old) / 2, dmin, d) <= r_star
        return 1.0
    end
    lo = φ_old
    hi = 1.0
    for _ in 1:60
        mid = 0.5 * (lo + hi)
        if tempered_inefficiency((mid - φ_old) / 2, dmin, d) < r_star
            lo = mid
        else
            hi = mid
        end
        hi - lo < 1e-6 && break
    end
    return 0.5 * (lo + hi)
end

# ── Mutation: preconditioned, adaptive random-walk Metropolis ────────────────
#
# The stage-φ target on the period's shocks is
#     π_φ(ε) ∝ N(ε; 0, I) · exp(-φ/2 · e(ε)ᵀ H⁻¹ e(ε)),
# with e(ε) the measurement residual the shock produces. An isotropic random walk
# with a fixed step is a poor way to explore it: the observation constrains the
# shocks very unevenly (a monetary shock and a price-markup shock move the
# observables by wildly different amounts), and the whole target contracts as φ
# rises. A step small enough to be accepted at φ = 1 then barely moves the
# particle at φ ≈ 0, and the cloud is not rejuvenated at all — which is exactly
# what leaves the filtered estimates at the mercy of the seed.
#
# Two fixes, both cheap:
#
#   * Preconditioning. Linearising e(ε) ≈ e(0) - Bₒ ε with Bₒ the first-order
#     impact of the shocks on the observables makes π_φ Gaussian with covariance
#     (I + φ G)⁻¹, G = Bₒᵀ H⁻¹ Bₒ. Proposing ε' = ε + c·L_φ z with L_φ L_φᵀ =
#     (I + φ G)⁻¹ therefore steps along exactly the directions and by the
#     magnitudes the target allows, at nExo × nExo cost per stage. On a linear
#     model this makes the proposal shape exact; on a nonlinear one it is a good
#     preconditioner because the curvature that matters here is the measurement
#     equation's, not the model's.
#   * Adaptation. `c` is scaled after every mutation step towards a 25 %
#     acceptance rate (the standard random-walk target), so it finds the right
#     magnitude within the first few periods instead of being guessed. The
#     schedule `φ` is already chosen adaptively from the particle system, so this
#     adds no new kind of dependence; each individual Metropolis kernel is still
#     exactly π_φ-invariant.
#
# How this differs from the guided filter's mutation (`guided_anneal_mutate!`).
# The machinery is deliberately the same — bridge in stages, reweight, resample,
# mutate, with the step size chosen by the same inefficiency target. What differs
# is the two endpoints, and everything else follows from that:
#
#   * Where the bridge starts. Here at the prior N(ε;0,I), which knows nothing
#     about yₜ, so the schedule usually needs several stages to arrive (~9 per
#     period on the euro-area problem). The guided filter starts at its Laplace
#     proposal, which already accounts for yₜ, and typically reaches β = 1 in one
#     step — its bridge exists only for the periods the proposal fits badly.
#   * What the mutation is preconditioned by. The preconditioner here has to
#     follow the target as it contracts, hence the φ-dependent (I + φ G)⁻¹ and a
#     fresh Cholesky per stage. The guided kernel reuses M⁻¹, the proposal's own
#     covariance, at every β: it is the right shape at both ends because the
#     bridge only interpolates between two distributions that already share it.
#   * What β multiplies. Here φ scales the measurement term against a fixed
#     prior. There the exponent moves weight from q to π̃, so the Metropolis ratio
#     carries the proposal's Mahalanobis form as well as ‖ε‖².
#
# Both cost one batched transition per mutation step, so the difference in price
# is entirely the difference in stage count.

# `DEFAULT_PARTICLE_LOW_ESS_FRACTION` is the average effective sample size, as a
# fraction of `n_particles`, below which the reported estimates are flagged. At
# that point the weighted moments are the average of a handful of distinct
# particles, so they move materially from seed to seed however many particles are
# nominally in the cloud — the fix is a better proposal, not a longer run.

# Why every tempering stage resamples, following Herbst & Schorfheide.
#
# Each resampling discards the ancestors that lost, and with ~9 stages in a period
# the compounding is severe: on a Smets-Wouters-sized problem only about 2 % of
# the cloud survives a period as *distinct* ancestors (a few hundred out of ten
# thousand). Every filtered moment is an average over those, which is what sets
# the seed-to-seed spread of the reported estimates — not `n_particles` directly.
#
# Deferring the resampling until the weights degenerate (standard adaptive SMC,
# resample at ESS < N/4) was measured on that problem and is *not* worth it: the
# surviving ancestors rise only from 2.24 % to 2.63 % while the stage count rises
# from 9.0 to 11.5, because the schedule then has to take smaller steps. Per unit
# of work it is slightly worse, and it makes the step criterion inconsistent with
# the carried weights. Resampling every stage is both simpler and better here.
#
# What does move the number is the particle count, roughly linearly — which is
# why `n_particles` is the lever to reach for when the estimates need to be
# steadier, and why making the swarm cheap to propagate was worth doing.

# G = Bₒᵀ H⁻¹ Bₒ over the observed rows, from the first-order shock loading.
function shock_information_matrix(S₁::AbstractMatrix, observables_index::Vector{Int},
                                  nPast::Int, me_var::AbstractVector, rows)
    nExo = size(S₁, 2) - nPast
    G = zeros(Float64, nExo, nExo)
    @inbounds for r in rows
        w = 1.0 / me_var[r]
        row = @view S₁[observables_index[r], nPast+1:end]
        for i in 1:nExo, j in 1:nExo
            G[i, j] += w * row[i] * row[j]
        end
    end
    return G
end

function shock_information_matrix(S₁::AbstractMatrix, observables_index::Vector{Int},
                                  nPast::Int, me::DenseMeasurementError, rows)
    me_sync!(me, rows)
    Bo = Matrix{Float64}(undef, length(rows), size(S₁, 2) - nPast)
    @inbounds for (k, r) in enumerate(rows)
        Bo[k, :] .= @view S₁[observables_index[r], nPast+1:end]
    end
    Y = ℒ.LowerTriangular(me.last_L) \ Bo          # Y = L⁻¹Bₒ, so YᵀY = BₒᵀH⁻¹Bₒ
    return Y' * Y
end

# L_φ with L_φ L_φᵀ = (I + φ G)⁻¹. `cholesky(M).U` is the upper factor R with
# M = RᵀR, so R⁻¹ (R⁻¹)ᵀ = M⁻¹ and R⁻¹ can serve as L_φ directly.
function tempering_proposal_factor(G::Matrix{Float64}, φ::Float64)
    n = size(G, 1)
    M = Matrix{Float64}(ℒ.I, n, n)
    @inbounds for i in 1:n, j in 1:n
        M[i, j] += φ * G[i, j]
    end
    F = ℒ.cholesky(ℒ.Symmetric(M), check = false)
    ℒ.issuccess(F) || return Matrix{Float64}(ℒ.I, n, n)
    return Matrix{Float64}(inv(F.U))
end

# One Metropolis sweep over the whole swarm at level `φ`. Proposals are formed
# for every particle at once (one `gemm` for the preconditioned step, one batched
# transition), scored, and then accepted or rejected. Returns the acceptance rate.
#
# The accept/reject decision has to consume the RNG in particle order, so it is
# taken in one cheap serial pass over the proposal scores and only the resulting
# column copies — which move far more bytes than the decision costs — are chunked
# across threads. The RNG is drawn exactly once per particle, in order, either
# way, so the outcome does not depend on the thread count.
function tempered_mutate!(::Val{algo}, tr, scr, St::NTuple{K,Matrix{Float64}},
                          parts_proposed::NTuple{K,Matrix{Float64}}, anc::NTuple{K,Matrix{Float64}},
                          E, Eprop, Z, Fbuf, dv, dprop, accept, Lφ, c::Float64, φ::Float64,
                          data_col, observables_index, inv_me_var, rows, rng) where {algo, K}
    n_particles = size(E, 2)
    nExo = size(E, 1)

    Random.randn!(rng, Z)
    ℒ.mul!(Eprop, Lφ, Z, c, 0.0)                   # ε' - ε = c·L_φ·z
    @inbounds for i in eachindex(Eprop)
        Eprop[i] += E[i]
    end

    propagate_cloud!(Val(algo), tr, scr, parts_proposed, anc, Eprop)
    F = full_states!(Fbuf, parts_proposed)
    quadform_cloud!(dprop, F, data_col, observables_index, inv_me_var, rows)

    accepted = 0
    @inbounds for p in 1:n_particles
        # ‖ε‖² at the current and the proposed shock: the prior part of the
        # Metropolis ratio, the measurement part being `dv`/`dprop`.
        shock_norm²_current  = 0.0
        shock_norm²_proposed = 0.0
        for e in 1:nExo
            current  = E[e, p]
            proposed = Eprop[e, p]
            shock_norm²_current  += current * current
            shock_norm²_proposed += proposed * proposed
        end
        logα = -0.5 * ((shock_norm²_proposed - shock_norm²_current) + φ * (dprop[p] - dv[p]))
        acc = log(rand(rng)) < logα
        accept[p] = acc
        if acc
            dv[p] = dprop[p]
            accepted += 1
        end
    end

    foreach_column_chunk(n_particles) do cols
        @inbounds for p in cols
            if accept[p]
                copy_col!(E, p, Eprop, p)
                copy_cloud_col!(St, p, parts_proposed, p)
            end
        end
    end

    return accepted / n_particles
end

# Multiplicative step towards the target acceptance rate, bounded so a pathological
# period cannot drive the scale to zero or blow it up.
@inline function adapt_mh_scale(c::Float64, acceptance::Float64)
    lo, hi = DEFAULT_PARTICLE_MH_SCALE_BOUNDS
    return clamp(c * exp(DEFAULT_PARTICLE_MH_ADAPTATION_GAIN * (acceptance - DEFAULT_PARTICLE_MH_TARGET_ACCEPTANCE)), lo, hi)
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
                             particle_target_ratio::Real = DEFAULT_PARTICLE_TARGET_RATIO,
                             particle_mh_steps::Int = DEFAULT_TEMPERED_MH_STEPS,
                             particle_max_stages::Int = DEFAULT_PARTICLE_MAX_STAGES,
                             particle_mh_scale::Real = DEFAULT_PARTICLE_MH_SCALE,
                             opts::CalculationOptions = merge_calculation_options())::Float64 where {algo}
    T = constants.post_model_macro
    nT = size(data_in_deviations, 2)

    me_var, inv_me_var, tr, scr, L, pws =
        particle_filter_setup(Val(algo), 𝐒, T, 𝓂, measurement_error, n_particles,
                              particle_initial_state_scaling, initial_covariance, opts)

    K = pruned_components(Val(algo))
    nK = n_components(K)
    pools = ensure_particle_pools!(pws, 5 * nK + 1)
    anc   = cloud_group(pools, 1, K)
    anc_scratch  = cloud_group(pools, 2, K)
    St    = cloud_group(pools, 3, K)
    St_scratch   = cloud_group(pools, 4, K)
    parts_proposed = cloud_group(pools, 5, K)
    Fbuf  = pools[5 * nK + 1]

    init_cloud!(anc, particle_rng, state, L, Fbuf)

    return tempered_loop!(Val(algo), tr, scr, anc, anc_scratch, St, St_scratch, parts_proposed, Fbuf,
                          pws.E, pws.E2, pws.Eprop, pws.logw, pws.Wn, pws.dv, pws.dv2,
                          pws.idx, pws.bins, nT, normalize_presample_periods(presample_periods, nT),
                          observables_index, data_in_deviations, obs_idx_per_t, has_missing,
                          me_var, inv_me_var,
                          𝓂.caches.first_order_solution_matrix, T.nPast_not_future_and_mixed,
                          particle_resampling, Float64(particle_target_ratio),
                          Float64(particle_mh_scale), particle_mh_steps, particle_max_stages,
                          particle_rng, Float64(on_failure_loglikelihood), log(2π))
end

function tempered_loop!(::Val{algo}, tr, scr, anc::NTuple{K,Matrix{Float64}}, anc_scratch::NTuple{K,Matrix{Float64}},
                        St::NTuple{K,Matrix{Float64}}, St_scratch::NTuple{K,Matrix{Float64}},
                        parts_proposed::NTuple{K,Matrix{Float64}}, Fbuf, E, E_scratch, Eprop, logw, Wn, dv, dv_scratch,
                        idx, bins, nT, presample_periods, observables_index, data_in_deviations,
                        obs_idx_per_t, has_missing, me_var, inv_me_var, S₁, nPast, resampling,
                        r_star, mh_scale, n_mh, max_stages, rng, on_failure_loglikelihood, log2pi) where {algo, K}
    n_particles = size(E, 2)
    Z      = Matrix{Float64}(undef, size(E, 1), n_particles)
    dprop  = Vector{Float64}(undef, n_particles)
    accept = Vector{Bool}(undef, n_particles)
    c = mh_scale
    loglik = 0.0

    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view data_in_deviations[:, t]
        d_obs = length(rows)

        # Bootstrap proposal: propagate every ancestor with a fresh shock.
        Random.randn!(rng, E)
        propagate_cloud!(Val(algo), tr, scr, St, anc, E)
        F = full_states!(Fbuf, St)
        quadform_cloud!(dv, F, data_col, observables_index, inv_me_var, rows)

        if isempty(rows)
            anc, St = St, anc
            continue
        end
        if all(!isfinite, dv)
            return on_failure_loglikelihood
        end

        G = shock_information_matrix(S₁, observables_index, nPast, me_var, rows)
        logZ = particle_measurement_logZ(me_var, rows, log2pi)

        period_ll = 0.0
        φ_old = 0.0
        stage = 0
        while φ_old < 1.0 - 1e-12 && stage < max_stages
            stage += 1
            φ_new = tempered_next_phi(φ_old, dv, r_star, n_particles)

            # Incremental weights. The tempered density is
            #   p_φ(y|x) = (2π)^(-d/2) φ^(d/2) |H|^(-1/2) exp(-φ/2·d),
            # so stage one carries the full normalisation and later stages only
            # the ratio p_{φ_new}/p_{φ_old}.
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

            logsw = normalise_log_weights!(Wn, logw)
            isfinite(logsw) || return on_failure_loglikelihood
            period_ll += logsw - log(n_particles)

            particle_resample_indices!(idx, bins, rng, Wn, resampling)
            gather_cloud!(anc_scratch, anc, idx)
            gather_cloud!(St_scratch, St, idx)
            @inbounds for j in 1:n_particles
                copy_col!(E_scratch, j, E, idx[j])
                dv_scratch[j] = dv[idx[j]]
            end
            anc, anc_scratch = anc_scratch, anc
            St,  St_scratch  = St_scratch,  St
            E,   E_scratch   = E_scratch,   E
            dv,  dv_scratch  = dv_scratch,  dv

            Lφ = tempering_proposal_factor(G, φ_new)
            acc_sum = 0.0
            for _ in 1:n_mh
                acc = tempered_mutate!(Val(algo), tr, scr, St, parts_proposed, anc, E, Eprop, Z, Fbuf,
                                       dv, dprop, accept, Lφ, c, φ_new, data_col,
                                       observables_index, inv_me_var, rows, rng)
                c = adapt_mh_scale(c, acc)
                acc_sum += acc
            end
            @debug "tempered stage" t stage φ_new acceptance = acc_sum / max(n_mh, 1) mh_scale = c

            φ_old = φ_new
        end

        if t > presample_periods
            loglik += period_ll
        end
        # The filtered cloud becomes next period's ancestors. A swap rather than a
        # copy: next period's propagation overwrites every column of `St` before
        # reading it, so whatever `anc` held is free to be scribbled over.
        anc, St = St, anc
    end

    return isfinite(loglik) ? loglik : on_failure_loglikelihood
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
# All four particle variants target the same filtering distribution p(xₜ|y₁..ₜ)
# — they differ only in how they get there — so the recursion below is shared.
# `:bootstrap_particle` and `:auxiliary_particle` use the plain
# predict/weight/resample step: the auxiliary filter's look-ahead proposal changes
# the *variance* of the likelihood estimate, not the cloud it leaves behind, so
# there is nothing extra to do for the moments.
#
# The other two do change the cloud, and that is what makes them the ones to use
# when the estimates themselves, rather than a likelihood, are the output. Both
# bridge to the period's target in stages and rejuvenate the shocks by Metropolis
# at each one, which leaves far more distinct support points at the same
# `n_particles` — exactly what the weighted moments, and the smoother that walks
# the genealogy, depend on. So the bridging controls act here too.
#
# Between the two, `:guided_particle` is the better default: it bridges from a
# proposal that already accounts for the observation rather than from the prior,
# and measures both more accurate and much cheaper (see the estimates section of
# `docs/src/filters.md`). `:tempered_particle` is the fallback for the case the
# guided proposal is built on and can get wrong — an observation far from linear
# in the shock — since it assumes nothing about that.
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
    particle_target_ratio::Real = DEFAULT_PARTICLE_TARGET_RATIO,
    particle_mh_steps::Int = DEFAULT_TEMPERED_MH_STEPS,
    particle_max_stages::Int = DEFAULT_PARTICLE_MAX_STAGES,
    particle_mh_scale::Real = DEFAULT_PARTICLE_MH_SCALE,
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

    ss_names = constants.post_complete_parameters.SS_and_pars_names
    observables_index = convert(Vector{Int}, indexin(observables, ss_names))

    dat = missing_data_to_nan(collect(data_in_deviations))
    obs_idx_per_t, has_missing = build_obs_index(dat)
    nT = size(dat, 2)

    # solution matrices and the initial state, exactly as the likelihood path builds them
    _, _, 𝐒, state, solved = get_relevant_steady_state_and_state_update(Val(algo), 𝓂.parameter_values, 𝓂, opts = opts)
    @assert solved "Could not solve the model for `algorithm = $(algo)`; cannot run the particle filter."

    me_var, inv_me_var, tr, scr, L, pws =
        particle_filter_setup(Val(algo), 𝐒, T, 𝓂, measurement_error, n_particles,
                              particle_initial_state_scaling, initial_covariance, opts)

    # storage for the filtered moments
    variables  = zeros(nVars, nT)
    stds       = zeros(nVars, nT)
    shocks_out = zeros(nExo, nT)

    K = pruned_components(Val(algo))
    nK = n_components(K)
    pools = ensure_particle_pools!(pws, 6 * nK + 1)
    parts  = cloud_group(pools, 1, K)
    parts2 = cloud_group(pools, 2, K)
    anc    = cloud_group(pools, 3, K)
    anc2   = cloud_group(pools, 4, K)
    parts_proposed  = cloud_group(pools, 5, K)
    sprop2 = cloud_group(pools, 6, K)
    Fbuf   = pools[6 * nK + 1]

    init_cloud!(parts, particle_rng, state, L, Fbuf)

    if pf == :guided_particle
        guided_estimates_loop!(Val(algo), tr, scr, parts, parts2, anc, parts_proposed, sprop2, Fbuf, pws,
                               variables, stds, shocks_out, nT, observables_index, dat,
                               obs_idx_per_t, has_missing, me_var, inv_me_var,
                               𝓂.caches.first_order_solution_matrix, T.nPast_not_future_and_mixed,
                               particle_resampling, Float64(particle_resampling_threshold),
                               particle_mh_steps, Float64(particle_mh_scale),
                               Float64(particle_target_ratio), particle_max_stages,
                               particle_rng, smooth, log(2π))
    else
        particle_estimates_loop!(Val(algo), Val(pf == :tempered_particle), tr, scr,
                                 parts, parts2, anc, anc2, parts_proposed, Fbuf, pws,
                                 variables, stds, shocks_out, nT, observables_index, dat,
                                 obs_idx_per_t, has_missing, me_var, inv_me_var,
                                 𝓂.caches.first_order_solution_matrix, T.nPast_not_future_and_mixed,
                                 particle_resampling, Float64(particle_resampling_threshold),
                                 Float64(particle_target_ratio), Float64(particle_mh_scale),
                                 particle_mh_steps, particle_max_stages, particle_rng, smooth, log(2π))
    end

    # ── Shock decomposition ──────────────────────────────────────────────────
    # A decomposition needs a shock path; the particle filter supplies one (the
    # filtered or smoothed shocks above), so the same attribution the inversion
    # filter uses applies here. At first order contributions are additive and the
    # split is exact. At pruned higher order they are not additive, which is
    # precisely what the Aumann-Shapley (marginal contribution) attribution is
    # for, so the pruned decomposition reuses the routines in `inversion.jl`.
    # Non-pruned `:second_order` / `:third_order` have no decomposition at all
    # (the caller already turns `shock_decomposition` off for them).
    # Column layout follows the inversion filter: with the Aumann-Shapley
    # attribution (and at first order) it is [contributions…, baseline, total] =
    # nExo+2; the sequential pruned attribution adds an explicit interaction and
    # residual column, [contributions…, interaction, residual, total] = nExo+3.
    sequential_pruned = algo ∈ (:pruned_second_order, :pruned_third_order) && !marginal_contribution
    decomposition = zeros(nVars, sequential_pruned ? nExo + 3 : nExo + 2, nT)
    decomposition[:, end, :] .= variables

    past_idx = T.past_not_future_and_mixed_idx

    if algo == :first_order
        𝐒₁ = 𝐒 isa AbstractMatrix ? 𝐒 : 𝐒[1]
        init_vec = state isa AbstractVector{<:AbstractVector} ? state[1] : state
        sck = zeros(nExo)
        @inbounds for i in 1:nExo
            fill!(sck, 0.0)
            sck[i] = shocks_out[i, 1]
            decomposition[:, i, 1] .= 𝐒₁ * vcat(init_vec[past_idx], sck)
        end
        decomposition[:, end - 1, 1] .= decomposition[:, end, 1] - sum(decomposition[:, 1:end-2, 1], dims = 2)
        for t in 2:nT
            @inbounds for i in 1:nExo
                fill!(sck, 0.0)
                sck[i] = shocks_out[i, t]
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
        traj = shock_path_trajectory(Val(algo), tr, scr, state, shocks_out, nVars, nT)
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
        # and whatever is still left over goes into the residual column. The
        # nExo+1 trajectories are carried as the columns of one cloud, so each
        # period costs a single batched transition.
        sequential_shock_decomposition!(Val(algo), tr, scr, state, shocks_out, variables,
                                        decomposition, nVars, nExo, nT)
    else
        @info "Shock decomposition is not available for $(algo) solutions (use a pruned solution); returning zeros." maxlog = 1
    end

    return variables, shocks_out, stds, decomposition
end

# The forward pass shared by every particle variant. `Val(tempered)` selects the
# within-period tempering stages; the rest of the recursion is identical.
function particle_estimates_loop!(::Val{algo}, ::Val{tempered}, tr, scr,
                                  parts::NTuple{K,Matrix{Float64}}, parts_scratch::NTuple{K,Matrix{Float64}},
                                  anc::NTuple{K,Matrix{Float64}}, anc_scratch::NTuple{K,Matrix{Float64}},
                                  parts_proposed::NTuple{K,Matrix{Float64}}, Fbuf, pws,
                                  variables, stds, shocks_out, nT, observables_index, dat,
                                  obs_idx_per_t, has_missing, me_var, inv_me_var, S₁, nPast,
                                  resampling, resampling_threshold, r_star, mh_scale, n_mh,
                                  max_stages, rng, smooth::Bool, log2pi) where {algo, tempered, K}
    n_particles = size(pws.E, 2)
    nVars = size(variables, 1)
    nExo  = size(shocks_out, 1)

    E, E_scratch, Eprop = pws.E, pws.E2, pws.Eprop
    W, logdens, Wn = pws.W, pws.logdens, pws.Wn
    dv, dv_scratch = pws.dv, pws.dv2
    idx, bins = pws.idx, pws.bins
    Z      = tempered ? Matrix{Float64}(undef, nExo, n_particles) : Matrix{Float64}(undef, 0, 0)
    dprop  = tempered ? Vector{Float64}(undef, n_particles) : Float64[]
    accept = tempered ? Vector{Bool}(undef, n_particles) : Bool[]

    fill!(W, 1.0 / n_particles)
    c = mh_scale
    ess_sum = 0.0      # effective sample size accumulated over scored periods
    n_scored = 0

    # Smoothing storage (see the backward pass below). `hist_*` keep the cloud and
    # the shocks of every period; `parent[t]` is the resampling map applied at the
    # end of period t (empty ⇒ no resampling ⇒ identity), which is the genealogy
    # the backward pass walks. `within[t]` is the composition of the tempering
    # stages' resampling maps for period t: post-stage slot ↦ pre-stage slot.
    hist_states = smooth ? [Matrix{Float64}(undef, nVars, n_particles) for _ in 1:nT] : Matrix{Float64}[]
    hist_shocks = smooth ? [Matrix{Float64}(undef, nExo,  n_particles) for _ in 1:nT] : Matrix{Float64}[]
    parent      = smooth ? [Int[] for _ in 1:nT] : Vector{Int}[]
    within      = smooth ? [Int[] for _ in 1:nT] : Vector{Int}[]
    terminal_weights = smooth ? fill(1.0 / n_particles, n_particles) : Float64[]
    comp  = tempered ? collect(1:n_particles) : Int[]
    comp_scratch = tempered ? collect(1:n_particles) : Int[]

    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view dat[:, t]

        Random.randn!(rng, E)
        if tempered
            # Why this branch keeps last period's cloud when the others discard
            # it. Bootstrap and auxiliary propagate once per period and are then
            # done with xₜ₋₁. Tempering mutates the *shock*: each Metropolis
            # proposal is a new εₜ that has to be pushed through the transition
            # from the same ancestor to see what state it implies, so xₜ₋₁ stays
            # live for the whole period. (The guided filter needs it for the same
            # reason, which is why `guided_estimates_loop!` carries `anc` too.)
            #
            # Hence: park the ancestors in `anc` and propagate into the buffer
            # `anc` was using. A swap, not a copy — the target is fully
            # overwritten before it is read.
            anc, parts = parts, anc
            propagate_cloud!(Val(algo), tr, scr, parts, anc, E)
        else
            propagate_cloud!(Val(algo), tr, scr, parts_scratch, parts, E)
            parts, parts_scratch = parts_scratch, parts
        end
        F = full_states!(Fbuf, parts)

        if isempty(rows)
            # nothing observed: the weights are unchanged, the cloud just predicts
        elseif tempered
            # Bridge from the prior to the full measurement density in stages,
            # resampling and rejuvenating at each one. The likelihood contribution
            # is not needed here (that is `run_particle_filter`'s job) — only the
            # cloud that comes out, which ends up equally weighted.
            quadform_cloud!(dv, F, data_col, observables_index, inv_me_var, rows)
            if any(isfinite, dv)
                @inbounds for p in 1:n_particles
                    comp[p] = p
                end
                G = shock_information_matrix(S₁, observables_index, nPast, me_var, rows)
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
                    isfinite(normalise_log_weights!(Wn, logdens)) || break

                    particle_resample_indices!(idx, bins, rng, Wn, resampling)
                    gather_cloud!(anc_scratch, anc, idx)
                    gather_cloud!(parts_scratch, parts, idx)
                    @inbounds for j in 1:n_particles
                        a = idx[j]
                        copy_col!(E_scratch, j, E, a)
                        dv_scratch[j]   = dv[a]
                        comp_scratch[j] = comp[a]
                    end
                    # Every per-particle quantity carries a `_scratch` double. The
                    # gather above wrote the resampled values into the doubles, so
                    # swapping the two names makes them current — no copy, and the
                    # old buffers become next stage's scratch.
                    anc,   anc_scratch   = anc_scratch,   anc
                    parts, parts_scratch = parts_scratch, parts
                    E,     E_scratch     = E_scratch,     E
                    dv,    dv_scratch    = dv_scratch,    dv
                    comp,  comp_scratch  = comp_scratch,  comp

                    Lφ = tempering_proposal_factor(G, φ_new)
                    acc_sum = 0.0
                    for _ in 1:n_mh
                        acc = tempered_mutate!(Val(algo), tr, scr, parts, parts_proposed, anc, E, Eprop, Z,
                                               Fbuf, dv, dprop, accept, Lφ, c, φ_new, data_col,
                                               observables_index, inv_me_var, rows, rng)
                        c = adapt_mh_scale(c, acc)
                        acc_sum += acc
                    end
                    # `comp` maps a post-stage slot back to the slot it occupied
                    # before the period's tempering began, so the number of
                    # distinct entries is exactly how many of the incoming
                    # ancestors are still represented. That is the quantity the
                    # accuracy of the filtered moments ultimately rests on.
                    @debug "tempered stage" t stage φ_new acceptance = acc_sum / max(n_mh, 1) mh_scale = c distinct_ancestors = length(unique(comp))

                    φ_old = φ_new
                end
                # the tempering stages leave an equally weighted cloud
                fill!(W, 1.0 / n_particles)
                if smooth
                    within[t] = copy(comp)
                end
                F = full_states!(Fbuf, parts)
            end
        else
            score_cloud!(logdens, F, data_col, observables_index, me_var, inv_me_var, rows, log2pi)
            reweight_log_weights!(W, logdens)
            ess_sum += effective_sample_size(W)
            n_scored += 1
        end

        if smooth
            # The backward pass starts from the cloud as stored here, i.e. *before*
            # any resampling at the end of this period, so it needs the weights in
            # that same indexing. Resampling would overwrite `W` with uniform ones.
            copyto!(terminal_weights, W)
            copyto!(hist_states[t], F)
            copyto!(hist_shocks[t], E)
        else
            accumulate_filtered_moments!(variables, stds, shocks_out, t, F, E, W)
        end

        if effective_sample_size(W) < resampling_threshold * n_particles
            particle_resample_indices!(idx, bins, rng, W, resampling)
            gather_cloud!(parts_scratch, parts, idx)
            parts, parts_scratch = parts_scratch, parts
            fill!(W, 1.0 / n_particles)
            if smooth
                parent[t] = copy(idx)
            end
        end
    end

    if smooth
        smooth_particle_trajectories!(variables, stds, shocks_out, hist_states, hist_shocks,
                                      parent, within, terminal_weights)
    end

    # Fail visibly: a cloud this degenerate does not produce estimates worth
    # reading, and the symptom (numbers that change every seed) is easy to
    # mistake for a modelling problem.
    if n_scored > 0
        ess_fraction = ess_sum / (n_scored * n_particles)
        if ess_fraction < DEFAULT_PARTICLE_LOW_ESS_FRACTION
            @warn "The particle cloud carried an effective sample size of only $(round(100 * ess_fraction, digits = 2))% of `n_particles` on average, so these estimates rest on a handful of distinct particles and will change materially from one `particle_rng` seed to the next. Use `filter = :tempered_particle`, which mutates the particles towards the data instead of only reweighting them, or raise `n_particles` / `measurement_error`." maxlog = 1
        end
    end

    return nothing
end


# The reported shock is the *conditional mean* μ(xₜ₋₁), not the shock that was
# drawn. Both are consistent for E[εₜ | y₁..ₜ], because
#     E[εₜ | y₁..ₜ] = E[ E[εₜ | xₜ₋₁, yₜ] | y₁..ₜ ] = E[ μ(xₜ₋₁) | y₁..ₜ ],
# but the conditional mean has already integrated out the draw, so it carries none
# of that draw's variance — a Rao-Blackwellisation. It is exact whenever the
# correction weights are constant, which is precisely when the linearisation
# behind μ is exact, so the residual bias is of the same (second) order as the
# proposal's own approximation error. Switched by `DEFAULT_GUIDED_RAO_BLACKWELL`.

function guided_estimates_loop!(::Val{algo}, tr, scr, parts::NTuple{K,Matrix{Float64}},
                                parts_scratch::NTuple{K,Matrix{Float64}}, anc::NTuple{K,Matrix{Float64}},
                                parts_proposed::NTuple{K,Matrix{Float64}}, sprop2::NTuple{K,Matrix{Float64}},
                                Fbuf, pws, variables, stds, shocks_out, nT, observables_index, dat,
                                obs_idx_per_t, has_missing, me_var, inv_me_var, S₁, nPast,
                                resampling, resampling_threshold, n_mh, mh_scale,
                                r_star, max_stages, rng, smooth::Bool, log2pi) where {algo, K}
    n_particles = size(pws.E, 2)
    nVars = size(variables, 1)
    nExo  = size(shocks_out, 1)
    nObs  = length(observables_index)

    E, Mu, Tmp = pws.E, pws.E2, pws.Eprop
    W, logλ, logw = pws.W, pws.logdens, pws.logw
    idx, bins = pws.idx, pws.bins
    R = Matrix{Float64}(undef, nObs, n_particles)
    Z = Matrix{Float64}(undef, nExo, n_particles)
    Eprop = Matrix{Float64}(undef, nExo, n_particles)
    Tmp2  = Matrix{Float64}(undef, nExo, n_particles)
    dv     = Vector{Float64}(undef, n_particles)
    dprop  = Vector{Float64}(undef, n_particles)
    Lvec   = Vector{Float64}(undef, n_particles)
    negL   = Vector{Float64}(undef, n_particles)
    comp   = Vector{Int}(undef, n_particles)
    accept = Vector{Bool}(undef, n_particles)
    c = mh_scale

    fill!(W, 1.0 / n_particles)

    hist_states = smooth ? [Matrix{Float64}(undef, nVars, n_particles) for _ in 1:nT] : Matrix{Float64}[]
    hist_shocks = smooth ? [Matrix{Float64}(undef, nExo,  n_particles) for _ in 1:nT] : Matrix{Float64}[]
    parent      = smooth ? [Int[] for _ in 1:nT] : Vector{Int}[]
    within      = smooth ? [Int[] for _ in 1:nT] : Vector{Int}[]
    terminal_weights = smooth ? fill(1.0 / n_particles, n_particles) : Float64[]

    gp_rows = Int[]
    gp = GuidedProposal(length(observables_index), nExo)
    rebuild_guided_proposal!(gp, S₁, observables_index, nPast, me_var, eachindex(observables_index), log2pi)
    ess_sum = 0.0
    n_scored = 0

    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view dat[:, t]

        if isempty(rows)
            Random.randn!(rng, E)
            propagate_cloud!(Val(algo), tr, scr, parts_scratch, parts, E)
            parts, parts_scratch = parts_scratch, parts
            F = full_states!(Fbuf, parts)
            if smooth
                copyto!(terminal_weights, W)
                copyto!(hist_states[t], F)
                copyto!(hist_shocks[t], E)
            else
                accumulate_filtered_moments!(variables, stds, shocks_out, t, F, E, W)
            end
            continue
        end
        if !same_rows(gp_rows, rows)
            rebuild_guided_proposal!(gp, S₁, observables_index, nPast, me_var, rows, log2pi)
            gp_rows = collect(Int, rows)
        end

        # Solve for the shock that best explains yₜ from each ancestor.
        fill!(E, 0.0)
        propagate_cloud!(Val(algo), tr, scr, parts_scratch, parts, E)
        residual_cloud!(R, full_states!(Fbuf, parts_scratch), data_col, observables_index, rows)
        ℒ.mul!(Mu, proposal_K(gp), view(R, 1:length(rows), :))
        for _ in 1:DEFAULT_GUIDED_NEWTON_STEPS
            propagate_cloud!(Val(algo), tr, scr, parts_scratch, parts, Mu)
            residual_cloud!(R, full_states!(Fbuf, parts_scratch), data_col, observables_index, rows)
            guided_newton_step!(Mu, R, Tmp, gp)
        end

        # The ancestors are the cloud as it stands; keep a copy so the Metropolis
        # rejuvenation below can re-propagate proposals from them.
        copy_cloud!(anc, parts)

        Random.randn!(rng, Z)
        ℒ.mul!(E, gp.Uinv, Z, DEFAULT_GUIDED_PROPOSAL_SCALE, 0.0)
        @inbounds for i in eachindex(E)
            E[i] += Mu[i]
        end
        propagate_cloud!(Val(algo), tr, scr, parts_scratch, parts, E)
        parts, parts_scratch = parts_scratch, parts
        F = full_states!(Fbuf, parts)
        residual_cloud!(R, F, data_col, observables_index, rows)

        @inbounds for j in 1:n_particles
            dv[j] = residual_quadform(R, j, inv_me_var, rows)
        end
        guided_bridge_gap!(Lvec, E, Mu, dv, gp)

        # Walk from the proposal to the conditional. Where the proposal is good the
        # schedule reaches β = 1 in one step and this is the plain guided filter.
        β_old = 0.0
        stage = 0
        @inbounds for j in 1:n_particles
            negL[j] = -Lvec[j]
        end
        while β_old < 1.0 - 1e-12 && stage < max_stages
            stage += 1
            β_new = any(isfinite, negL) ? tempered_next_phi(β_old, negL, r_star, n_particles) : 1.0
            @inbounds for j in 1:n_particles
                logw[j] = (β_new - β_old) * Lvec[j]
            end
            isfinite(reweight_log_weights!(W, logw)) || (fill!(W, 1.0 / n_particles); break)

            if effective_sample_size(W) < resampling_threshold * n_particles
                particle_resample_indices!(idx, bins, rng, W, resampling)
                gather_cloud!(parts_scratch, parts, idx)
                copy_cloud!(parts, parts_scratch)
                gather_cloud!(parts_proposed, anc, idx)
                copy_cloud!(anc, parts_proposed)
                @inbounds for j in 1:n_particles
                    a = idx[j]
                    copy_col!(Eprop, j, E, a)
                    copy_col!(Tmp2, j, Mu, a)
                    dprop[j] = dv[a]
                end
                copyto!(E, Eprop)
                copyto!(Mu, Tmp2)
                copyto!(dv, dprop)
                fill!(W, 1.0 / n_particles)
                if smooth
                    if isempty(parent[t])
                        parent[t] = copy(idx)
                    else
                        prev = parent[t]
                        @inbounds for j in 1:n_particles
                            comp[j] = prev[idx[j]]
                        end
                        parent[t] = copy(comp)
                    end
                end
            end

            for _ in 1:n_mh
                acc = guided_anneal_mutate!(Val(algo), tr, scr, gp, parts, sprop2, anc, Fbuf,
                                            E, Eprop, Z, R, Mu, dv, dprop, accept, c, β_new,
                                            data_col, observables_index, inv_me_var, rows, rng)
                c = adapt_mh_scale(c, acc)
            end
            guided_bridge_gap!(Lvec, E, Mu, dv, gp)
            @inbounds for j in 1:n_particles
                negL[j] = -Lvec[j]
            end
            β_old = β_new
        end
        F = full_states!(Fbuf, parts)
        ess_sum += effective_sample_size(W)
        n_scored += 1
        @debug "guided period" t stages = stage adaptation_ess = effective_sample_size(W) / n_particles

        # Which shock to report. Without rejuvenation the conditional mean μ is the
        # Rao-Blackwellised choice: E[εₜ|y₁..ₜ] = E[μ(xₜ₋₁)|y₁..ₜ], so averaging μ
        # instead of the draw removes the draw's variance entirely. Once the
        # particles have been mutated they are draws from the exact conditional
        # rather than from the Gaussian μ centres, so μ is no longer their mean and
        # the drawn shock is the consistent estimator.
        reported = (DEFAULT_GUIDED_RAO_BLACKWELL && n_mh == 0 && max_stages <= 1) ? Mu : E
        if smooth
            copyto!(terminal_weights, W)
            copyto!(hist_states[t], F)
            copyto!(hist_shocks[t], reported)
        else
            accumulate_filtered_moments!(variables, stds, shocks_out, t, F, reported, W)
        end

    end

    if smooth
        smooth_particle_trajectories!(variables, stds, shocks_out, hist_states, hist_shocks,
                                      parent, within, terminal_weights)
    end

    if n_scored > 0
        ess_fraction = ess_sum / (n_scored * n_particles)
        if ess_fraction < DEFAULT_PARTICLE_LOW_ESS_FRACTION
            @warn "The guided proposal's importance weights carried an effective sample size of only $(round(100 * ess_fraction, digits = 2))% of `n_particles` on average, which means the observation is far from linear in the shock over this cloud and the closed-form proposal is a poor fit. Use `filter = :tempered_particle`, which makes no such assumption." maxlog = 1
        end
    end

    return nothing
end

# Weighted mean and spread of the cloud in period `t`. Two passes over the
# already-summed full-state matrix `F` — the component sum that produces `F` is
# done once, before this is called, rather than once per pass.
function accumulate_filtered_moments!(variables, stds, shocks_out, t::Int,
                                      F::Matrix{Float64}, E::Matrix{Float64}, W::Vector{Float64})
    nVars = size(variables, 1)
    nExo  = size(shocks_out, 1)
    @inbounds for p in eachindex(W)
        w = W[p]
        for i in 1:nVars
            variables[i, t] += w * F[i, p]
        end
        for e in 1:nExo
            shocks_out[e, t] += w * E[e, p]
        end
    end
    @inbounds for p in eachindex(W)
        w = W[p]
        for i in 1:nVars
            d = F[i, p] - variables[i, t]
            stds[i, t] += w * d * d
        end
    end
    @inbounds for i in 1:nVars
        stds[i, t] = sqrt(max(stds[i, t], 0.0))
    end
    return nothing
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
        Hs = hist_states[t]
        Hh = hist_shocks[t]

        @inbounds for p in 1:n_particles
            a = lineage[p]
            w = W[p]
            for i in 1:nVars
                variables[i, t] += w * Hs[i, a]
            end
            for e in 1:nExo
                shocks_out[e, t] += w * Hh[e, a]
            end
        end
        @inbounds for p in 1:n_particles
            a = lineage[p]
            w = W[p]
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


# ── Deterministic trajectories for the shock decomposition ───────────────────
# Both helpers reuse the batched transition with a handful of "particles": one
# per trajectory. The cloud machinery is exact for N = 1, so there is no separate
# single-state code path to keep in sync.

# The model path implied by a given shock sequence.
function shock_path_trajectory(::Val{algo}, tr::ParticleTransition, scr, state,
                               shocks_out::Matrix{Float64}, nVars::Int, nT::Int) where {algo}
    K = pruned_components(Val(algo))
    cur = alloc_cloud(K, nVars, 1)
    nxt = alloc_cloud(K, nVars, 1)
    fill_cloud_from_state!(cur, state)
    E = Matrix{Float64}(undef, size(shocks_out, 1), 1)
    F = Matrix{Float64}(undef, nVars, 1)
    traj = zeros(nVars, nT)

    for t in 1:nT
        @inbounds for e in axes(shocks_out, 1)
            E[e, 1] = shocks_out[e, t]
        end
        propagate_cloud!(Val(algo), tr, scr, nxt, cur, E)
        full = full_states!(F, nxt)
        @inbounds for i in 1:nVars
            traj[i, t] = full[i, 1]
        end
        cur, nxt = nxt, cur
    end

    return traj
end

# One trajectory per shock with only that shock switched on, plus one with all of
# them, all carried as the columns of a single cloud.
function sequential_shock_decomposition!(::Val{algo}, tr::ParticleTransition, scr, state,
                                         shocks_out::Matrix{Float64}, variables::Matrix{Float64},
                                         decomposition::Array{Float64,3}, nVars::Int, nExo::Int, nT::Int) where {algo}
    K = pruned_components(Val(algo))
    N = nExo + 1
    cur = alloc_cloud(K, nVars, N)
    nxt = alloc_cloud(K, nVars, N)
    fill_cloud_from_state!(cur, state)
    E = zeros(Float64, nExo, N)
    F = Matrix{Float64}(undef, nVars, N)

    for t in 1:nT
        fill!(E, 0.0)
        @inbounds for i in 1:nExo
            E[i, i]   = shocks_out[i, t]     # column i: only shock i active
            E[i, end] = shocks_out[i, t]     # last column: every shock active
        end
        propagate_cloud!(Val(algo), tr, scr, nxt, cur, E)
        full = full_states!(F, nxt)
        cur, nxt = nxt, cur

        @inbounds for i in 1:nExo, v in 1:nVars
            decomposition[v, i, t] = full[v, i]
        end
        # interaction = all-shock path − Σ single-shock paths
        @inbounds for v in 1:nVars
            decomposition[v, end - 2, t] = full[v, N]
        end
        decomposition[:, end - 2, t] .-= sum(decomposition[:, 1:end-3, t], dims = 2)
        # residual = reported estimate − everything attributed so far
        decomposition[:, end - 1, t] .= variables[:, t]
        decomposition[:, end - 1, t] .-= sum(decomposition[:, 1:end-2, t], dims = 2)
    end

    return nothing
end

end # @stable
