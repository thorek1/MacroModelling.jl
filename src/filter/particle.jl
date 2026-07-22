@stable default_mode = "disable" begin

# Particle filters for the (possibly nonlinear) DSGE state-space representation.
#
# The measurement equation is  yₜ = full_stateₜ[observables] + ηₜ,  ηₜ ~ N(0, H),
# with H a diagonal matrix of measurement-error variances. The structural shocks
# are i.i.d. standard normal (their standard deviations are baked into the
# solution matrices 𝐒), and the state transition is the perturbation solution's
# `state_update` (first order through pruned third order).
#
# Three variants are provided, selected by `particle_filter_algorithm`:
#   :bootstrap  — sequential-importance-resampling (as in Dynare 7's
#                 `sequential_importance_particle_filter.m`)
#   :auxiliary  — Pitt & Shephard (1999) auxiliary particle filter
#   :tempered   — Herbst & Schorfheide (2019) tempered particle filter
#
# The particle filter is a stochastic likelihood estimator and is **not**
# differentiable (resampling is discontinuous); it is intended for use with
# gradient-free samplers (e.g. Pigeons slice sampling, nested sampling).


# ── Resampling schemes ───────────────────────────────────────────────────────
# Each returns a length-N vector of ancestor indices drawn from the normalised
# weights `W` (which must sum to one). Systematic/stratified have lower variance
# than multinomial and are the recommended defaults.

# Effective sample size 1 / Σ Wᵢ².
effective_sample_size(W::AbstractVector{<:Real}) = 1.0 / sum(abs2, W)

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
# from the cached first-order solution, as in Dynare); `:diagonal` uses 10·I; an
# nVars×nVars matrix is used directly.
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


# ── Bootstrap (sequential importance resampling) particle filter ─────────────

function run_particle_filter(::Val{algo},
                             ::Val{:bootstrap},
                             observables_index::Vector{Int},
                             𝐒,
                             data_in_deviations::AbstractMatrix,
                             constants::constants,
                             state,
                             𝓂::ℳ,
                             measurement_error_variances::AbstractVector{<:Real},
                             obs_idx_per_t::Vector{Vector{Int}},
                             has_missing::Bool;
                             n_particles::Int = DEFAULT_N_PARTICLES,
                             resampling::Symbol = DEFAULT_RESAMPLING,
                             resampling_threshold::Real = DEFAULT_RESAMPLING_THRESHOLD,
                             initial_state_prior_scaling_factor::Real = DEFAULT_INITIAL_STATE_PRIOR_SCALING_FACTOR,
                             rng::Random.AbstractRNG = Random.default_rng(),
                             presample_periods::Int = 0,
                             initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
                             on_failure_loglikelihood::Real = -Inf,
                             tempering_target_ratio::Real = DEFAULT_TEMPERING_TARGET_RATIO,
                             tempering_mh_steps::Int = DEFAULT_TEMPERING_MH_STEPS,
                             tempering_max_stages::Int = DEFAULT_TEMPERING_MAX_STAGES,
                             tempering_mh_scale::Real = DEFAULT_TEMPERING_MH_SCALE,
                             opts::CalculationOptions = merge_calculation_options())::Float64 where {algo}
    T = constants.post_model_macro
    nVars = T.nVars
    nExo  = T.nExo
    nT    = size(data_in_deviations, 2)
    presample_periods = normalize_presample_periods(presample_periods, nT)
    log2pi = log(2π)

    me_var = Float64.(measurement_error_variances)
    @assert all(x -> x > 0, me_var) "The particle filter requires strictly positive measurement-error variances for every observable."

    state_update, pruning = parse_algorithm_to_state_update(algo, 𝓂, false)

    Σ, _ = particle_initial_state_covariance(𝓂, T, opts, initial_covariance)
    L = particle_initial_cloud_factor(Σ, Float64(initial_state_prior_scaling_factor))
    particles = initialise_particles(rng, state, pruning, L, n_particles, nVars)

    W = fill(1.0 / n_particles, n_particles)
    logdens = Vector{Float64}(undef, n_particles)
    loglik = 0.0

    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view data_in_deviations[:, t]

        if isempty(rows)
            # No observation this period: propagate only, weights unchanged.
            @inbounds for p in 1:n_particles
                particles[p] = state_update(particles[p], randn(rng, nExo))
            end
            continue
        end

        @inbounds for p in 1:n_particles
            particles[p] = state_update(particles[p], randn(rng, nExo))
            full = particle_full_state(particles[p], pruning)
            logdens[p] = particle_log_measurement_density(full, data_col, observables_index, me_var, rows, log2pi)
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
            idxs = particle_resample_indices(rng, W, resampling)
            # `state_update` never mutates its argument in place, so sharing the
            # underlying arrays across duplicated ancestors is safe (each slot is
            # reassigned, never mutated, on the next propagation).
            particles = particles[idxs]
            fill!(W, 1.0 / n_particles)
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
# A look-ahead stage reweights ancestors by the predictive likelihood evaluated
# at the transition mean (zero shock) before propagating, reducing variance when
# the signal is informative. The likelihood estimate remains unbiased.

function run_particle_filter(::Val{algo},
                             ::Val{:auxiliary},
                             observables_index::Vector{Int},
                             𝐒,
                             data_in_deviations::AbstractMatrix,
                             constants::constants,
                             state,
                             𝓂::ℳ,
                             measurement_error_variances::AbstractVector{<:Real},
                             obs_idx_per_t::Vector{Vector{Int}},
                             has_missing::Bool;
                             n_particles::Int = DEFAULT_N_PARTICLES,
                             resampling::Symbol = DEFAULT_RESAMPLING,
                             resampling_threshold::Real = DEFAULT_RESAMPLING_THRESHOLD,
                             initial_state_prior_scaling_factor::Real = DEFAULT_INITIAL_STATE_PRIOR_SCALING_FACTOR,
                             rng::Random.AbstractRNG = Random.default_rng(),
                             presample_periods::Int = 0,
                             initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
                             on_failure_loglikelihood::Real = -Inf,
                             tempering_target_ratio::Real = DEFAULT_TEMPERING_TARGET_RATIO,
                             tempering_mh_steps::Int = DEFAULT_TEMPERING_MH_STEPS,
                             tempering_max_stages::Int = DEFAULT_TEMPERING_MAX_STAGES,
                             tempering_mh_scale::Real = DEFAULT_TEMPERING_MH_SCALE,
                             opts::CalculationOptions = merge_calculation_options())::Float64 where {algo}
    T = constants.post_model_macro
    nVars = T.nVars
    nExo  = T.nExo
    nT    = size(data_in_deviations, 2)
    presample_periods = normalize_presample_periods(presample_periods, nT)
    log2pi = log(2π)

    me_var = Float64.(measurement_error_variances)
    @assert all(x -> x > 0, me_var) "The particle filter requires strictly positive measurement-error variances for every observable."

    state_update, pruning = parse_algorithm_to_state_update(algo, 𝓂, false)

    # One-step-ahead predictive variance of each observable due to the structural
    # shocks (diagonal of Cₒ BBᵀ Cₒᵀ from the first-order shock loading), used to
    # scale the auxiliary first-stage weights. Evaluating the predictive density
    # at the transition mean alone would be near-degenerate when the observable is
    # shock-driven; inflating by the shock spread keeps the proxy well-conditioned.
    nPast = T.nPast_not_future_and_mixed
    S₁cache = 𝓂.caches.first_order_solution_matrix
    pred_var = Float64[sum(abs2, @view S₁cache[observables_index[i], nPast+1:end]) for i in eachindex(observables_index)] .+ me_var

    Σ, _ = particle_initial_state_covariance(𝓂, T, opts, initial_covariance)
    L = particle_initial_cloud_factor(Σ, Float64(initial_state_prior_scaling_factor))
    particles = initialise_particles(rng, state, pruning, L, n_particles, nVars)

    zero_shock = zeros(Float64, nExo)
    W = fill(1.0 / n_particles, n_particles)
    logg̃ = Vector{Float64}(undef, n_particles)   # first-stage predictive log-density
    logw = Vector{Float64}(undef, n_particles)    # second-stage log-weight
    loglik = 0.0

    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view data_in_deviations[:, t]

        # First stage: predictive density at the transition mean (zero shock),
        # spread by the shock-induced predictive variance `pred_var`.
        @inbounds for p in 1:n_particles
            μ = particle_full_state(state_update(particles[p], zero_shock), pruning)
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
        λ = Vector{Float64}(undef, n_particles)
        @inbounds for p in 1:n_particles
            λ[p] = exp(log(W[p]) + logg̃[p] - logκ)
        end

        # Resample ancestors ∝ λ, propagate with fresh shocks, second-stage weight
        # w = g(yₜ|xₜ) / g̃(ancestor).
        idx = particle_resample_indices(rng, λ, resampling)
        newparts = Vector{eltype(particles)}(undef, n_particles)
        @inbounds for j in 1:n_particles
            a = idx[j]
            newparts[j] = state_update(particles[a], randn(rng, nExo))
            full = particle_full_state(newparts[j], pruning)
            logw[j] = particle_log_measurement_density(full, data_col, observables_index, me_var, rows, log2pi) - logg̃[a]
        end

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
        particles = newparts
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
                             measurement_error_variances::AbstractVector{<:Real},
                             obs_idx_per_t::Vector{Vector{Int}},
                             has_missing::Bool;
                             n_particles::Int = DEFAULT_N_PARTICLES,
                             resampling::Symbol = DEFAULT_RESAMPLING,
                             resampling_threshold::Real = DEFAULT_RESAMPLING_THRESHOLD,
                             initial_state_prior_scaling_factor::Real = DEFAULT_INITIAL_STATE_PRIOR_SCALING_FACTOR,
                             rng::Random.AbstractRNG = Random.default_rng(),
                             presample_periods::Int = 0,
                             initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
                             on_failure_loglikelihood::Real = -Inf,
                             tempering_target_ratio::Real = DEFAULT_TEMPERING_TARGET_RATIO,
                             tempering_mh_steps::Int = DEFAULT_TEMPERING_MH_STEPS,
                             tempering_max_stages::Int = DEFAULT_TEMPERING_MAX_STAGES,
                             tempering_mh_scale::Real = DEFAULT_TEMPERING_MH_SCALE,
                             opts::CalculationOptions = merge_calculation_options())::Float64 where {algo}
    T = constants.post_model_macro
    nVars = T.nVars
    nExo  = T.nExo
    nT    = size(data_in_deviations, 2)
    presample_periods = normalize_presample_periods(presample_periods, nT)
    log2pi = log(2π)

    me_var = Float64.(measurement_error_variances)
    @assert all(x -> x > 0, me_var) "The particle filter requires strictly positive measurement-error variances for every observable."

    r_star = Float64(tempering_target_ratio)
    c = Float64(tempering_mh_scale)
    n_mh = tempering_mh_steps
    max_stages = tempering_max_stages

    state_update, pruning = parse_algorithm_to_state_update(algo, 𝓂, false)

    Σ, _ = particle_initial_state_covariance(𝓂, T, opts, initial_covariance)
    L = particle_initial_cloud_factor(Σ, Float64(initial_state_prior_scaling_factor))
    prev_states = initialise_particles(rng, state, pruning, L, n_particles, nVars)

    logw = Vector{Float64}(undef, n_particles)
    loglik = 0.0

    for t in 1:nT
        rows = has_missing ? obs_idx_per_t[t] : eachindex(observables_index)
        data_col = @view data_in_deviations[:, t]
        d_obs = length(rows)

        # Bootstrap proposal: propagate every ancestor with a fresh shock.
        ancestors = prev_states
        shocks = [randn(rng, nExo) for _ in 1:n_particles]
        states = Vector{eltype(prev_states)}(undef, n_particles)
        dvec = Vector{Float64}(undef, n_particles)
        @inbounds for p in 1:n_particles
            states[p] = state_update(ancestors[p], shocks[p])
            dvec[p] = particle_quadratic_form(particle_full_state(states[p], pruning), data_col, observables_index, me_var, rows)
        end

        if all(!isfinite, dvec)
            return Float64(on_failure_loglikelihood)
        end

        period_ll = 0.0
        φ_old = 0.0
        stage = 0
        while φ_old < 1.0 - 1e-12 && stage < max_stages
            stage += 1
            φ_new = tempered_next_phi(φ_old, dvec, r_star, n_particles)

            # Incremental (tempered) log-weights.
            if φ_old == 0.0
                logZ = particle_measurement_logZ(me_var, rows, log2pi)
                @inbounds for p in 1:n_particles
                    logw[p] = logZ + 0.5 * d_obs * log(φ_new) - 0.5 * φ_new * dvec[p]
                end
            else
                lr = 0.5 * d_obs * (log(φ_new) - log(φ_old))
                @inbounds for p in 1:n_particles
                    logw[p] = lr - 0.5 * (φ_new - φ_old) * dvec[p]
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

            # Normalise and resample.
            logsw = m + log(s)
            Wn = Vector{Float64}(undef, n_particles)
            @inbounds for p in 1:n_particles
                Wn[p] = exp(logw[p] - logsw)
            end
            idx = particle_resample_indices(rng, Wn, resampling)
            ancestors = ancestors[idx]
            shocks = shocks[idx]
            states = states[idx]
            dvec = dvec[idx]

            # Mutation: random-walk Metropolis on the shocks, targeting the
            # stage-φ posterior  π(ε) ∝ N(ε;0,I) · exp(-φ/2 · e(ε)ᵀH⁻¹e(ε)).
            @inbounds for p in 1:n_particles
                for _ in 1:n_mh
                    εp = shocks[p]
                    εprop = εp .+ c .* randn(rng, nExo)
                    sprop = state_update(ancestors[p], εprop)
                    dprop = particle_quadratic_form(particle_full_state(sprop, pruning), data_col, observables_index, me_var, rows)
                    logα = -0.5 * ((sum(abs2, εprop) - sum(abs2, εp)) + φ_new * (dprop - dvec[p]))
                    if log(rand(rng)) < logα
                        shocks[p] = εprop
                        states[p] = sprop
                        dvec[p] = dprop
                    end
                end
            end

            φ_old = φ_new
        end

        if t > presample_periods
            loglik += period_ll
        end
        prev_states = states
    end

    return isfinite(loglik) ? loglik : Float64(on_failure_loglikelihood)
end

end # @stable
