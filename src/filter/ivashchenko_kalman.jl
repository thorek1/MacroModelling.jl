@stable default_mode = "disable" begin

# Unpruned Gaussian moment-closure filter in the spirit of Ivashchenko (2014).
#
# The second- and third-order perturbation solutions are treated as the actual
# polynomial transition/measurement map, rather than as a linear map on the
# pruned augmented state.  If u = [xₜ₋₁; εₜ] is Gaussian, the map is expanded
# around E[u] and its moments are closed analytically.  At second order this
# requires fourth Gaussian moments; the cubic extension additionally uses the
# sixth moments through the third Hermite component.

const IVASHCHENKO_STATIONARY_MAXITER = 500
const IVASHCHENKO_STATIONARY_TOLERANCE = 1e-10

polynomial_pair_index(i::Int, j::Int, n::Int) = (i - 1) * n + j
polynomial_triple_index(i::Int, j::Int, k::Int, n::Int) = ((i - 1) * n + j - 1) * n + k

function build_ivashchenko_kalman_system_from_constants(cons, 𝐒, observables_index::Vector{Int}, order::Symbol)
    T = cons.post_model_macro
    nVars, nPast, nExo = T.nVars, T.nPast_not_future_and_mixed, T.nExo
    past = collect(T.past_not_future_and_mixed_idx)
    # Keep all model-variable rows.  The likelihood still selects only the
    # state and observable blocks, while the full row set lets the estimate API
    # return a value for every model variable and makes smoothing well-defined.
    output_rows = collect(1:nVars)
    nout = length(output_rows)
    dv = nPast + 1 + nExo
    d = nPast + nExo

    order ∈ (:second_order, :third_order) ||
        throw(ArgumentError("The Ivashchenko filter supports only second- and third-order solutions."))
    length(𝐒) ≥ (order == :third_order ? 3 : 2) ||
        throw(DimensionMismatch("The $(order) solution must provide S₁, S₂$(order == :third_order ? ", and S₃" : "")."))

    S1 = Matrix(𝐒[1][output_rows, :])
    S2 = Matrix(𝐒[2][output_rows, :])
    size(S1, 2) == dv || throw(DimensionMismatch("S₁ has $(size(S1, 2)) columns; expected $dv."))
    size(S2, 2) == dv^2 || throw(DimensionMismatch("S₂ has $(size(S2, 2)) columns; expected $(dv^2)."))

    scalar_type = order == :third_order ? promote_type(eltype(S1), eltype(S2), eltype(𝐒[3])) : promote_type(eltype(S1), eltype(S2))
    S1 = Matrix{scalar_type}(S1)
    S2 = Matrix{scalar_type}(S2)

    # The raw solution is multiplied by symmetric Kronecker products.  Store
    # the symmetrised derivative tensors; this is algebraically equivalent and
    # makes the Gaussian contractions below use the actual Hessian/third
    # derivative of the polynomial.
    H = zeros(scalar_type, nout, dv, dv)
    @inbounds for a in 1:nout, i in 1:dv, j in 1:dv
        H[a, i, j] = (S2[a, polynomial_pair_index(i, j, dv)] +
                      S2[a, polynomial_pair_index(j, i, dv)]) / 2
    end

    third_derivative = nothing
    if order == :third_order
        S3 = Matrix(𝐒[3][output_rows, :])
        size(S3, 2) == dv^3 || throw(DimensionMismatch("S₃ has $(size(S3, 2)) columns; expected $(dv^3)."))
        S3 = Matrix{scalar_type}(S3)
        third_derivative = zeros(scalar_type, nout, dv, dv, dv)
        @inbounds for a in 1:nout, i in 1:dv, j in 1:dv, k in 1:dv
            third_derivative[a, i, j, k] = (
                S3[a, polynomial_triple_index(i, j, k, dv)] +
                S3[a, polynomial_triple_index(i, k, j, dv)] +
                S3[a, polynomial_triple_index(j, i, k, dv)] +
                S3[a, polynomial_triple_index(j, k, i, dv)] +
                S3[a, polynomial_triple_index(k, i, j, dv)] +
                S3[a, polynomial_triple_index(k, j, i, dv)]) / 6
        end
    end

    row_position = zeros(Int, nVars)
    @inbounds for (i, row) in enumerate(output_rows)
        row_position[row] = i
    end
    state_position = row_position[past]
    observation_position = row_position[observables_index]
    random_indices = vcat(collect(1:nPast), collect(nPast + 2:dv))

    # The linear-limit covariance is a useful starting point for the nonlinear
    # stationary fixed point and also makes the :diagonal option match Kalman's
    # convention exactly.
    state_s1 = S1[state_position, :]
    A = state_s1[:, 1:nPast]
    B = state_s1[:, nPast + 2:dv]

    return (; order, nVars, nPast, nExo, d, dv, nout, output_rows, past,
            S1, H, third_derivative, state_position, observation_position,
            random_indices, A, B)
end

build_ivashchenko_kalman_system(𝓂::ℳ, 𝐒, oi::Vector{Int}, order::Symbol) =
    build_ivashchenko_kalman_system_from_constants(𝓂.constants, 𝐒, oi, order)

function ivashchenko_kalman_workspace(sys, scalar_type::Type)
    nout, d, dv = sys.nout, sys.d, sys.dv
    third = sys.order == :third_order
    return (; covariance_input = zeros(scalar_type, d, d),
            vbar = zeros(scalar_type, dv),
            mean = zeros(scalar_type, nout),
            linear = zeros(scalar_type, nout, d),
            effective_linear = zeros(scalar_type, nout, d),
            hessian = zeros(scalar_type, nout, d, d),
            hessian_covariance = zeros(scalar_type, nout, d, d),
            covariance = zeros(scalar_type, nout, nout),
            third_derivative = third ? zeros(scalar_type, nout, d, d, d) : nothing,
            third_covariance = third ? zeros(scalar_type, nout, d, d, d) : nothing,
            third_scratch = third ? zeros(scalar_type, nout, d, d, d) : nothing)
end

function ivashchenko_transform_third_tensor!(destination, scratch, tensor, covariance)
    nout, d = size(tensor, 1), size(tensor, 2)
    @inbounds for a in 1:nout, p in 1:d, j in 1:d, k in 1:d
        value = zero(eltype(destination))
        for i in 1:d
            value += covariance[p, i] * tensor[a, i, j, k]
        end
        scratch[a, p, j, k] = value
    end
    @inbounds for a in 1:nout, p in 1:d, q in 1:d, k in 1:d
        value = zero(eltype(destination))
        for j in 1:d
            value += covariance[q, j] * scratch[a, p, j, k]
        end
        destination[a, p, q, k] = value
    end
    @inbounds for a in 1:nout, p in 1:d, q in 1:d, r in 1:d
        value = zero(eltype(destination))
        for k in 1:d
            value += covariance[r, k] * destination[a, p, q, k]
        end
        scratch[a, p, q, r] = value
    end
    destination .= scratch
    return nothing
end

function ivashchenko_polynomial_moments!(sys, mean_state, covariance_state, ws)
    nPast, d, dv, nout = sys.nPast, sys.d, sys.dv, sys.nout
    Σ = ws.covariance_input
    fill!(Σ, zero(eltype(Σ)))
    Σ[1:nPast, 1:nPast] .= covariance_state
    @inbounds for i in nPast + 1:d
        Σ[i, i] = one(eltype(Σ))
    end

    fill!(ws.vbar, zero(eltype(ws.vbar)))
    ws.vbar[1:nPast] .= mean_state
    ws.vbar[nPast + 1] = one(eltype(ws.vbar))

    @inbounds for a in 1:nout
        Hfull = view(sys.H, a, :, :)
        third_full = sys.third_derivative === nothing ? nothing : view(sys.third_derivative, a, :, :, :)
        value = ℒ.dot(view(sys.S1, a, :), ws.vbar)
        for i in 1:dv, j in 1:dv
            value += Hfull[i, j] * ws.vbar[i] * ws.vbar[j] / 2
        end
        if third_full !== nothing
            for i in 1:dv, j in 1:dv, k in 1:dv
                value += third_full[i, j, k] * ws.vbar[i] * ws.vbar[j] * ws.vbar[k] / 6
            end
        end

        for r in 1:d
            i = sys.random_indices[r]
            slope = sys.S1[a, i]
            for j in 1:dv
                slope += Hfull[i, j] * ws.vbar[j]
            end
            if third_full !== nothing
                for j in 1:dv, k in 1:dv
                    slope += third_full[i, j, k] * ws.vbar[j] * ws.vbar[k] / 2
                end
            end
            ws.linear[a, r] = slope
        end

        for r in 1:d, s in 1:d
            i, j = sys.random_indices[r], sys.random_indices[s]
            hessian = Hfull[i, j]
            if third_full !== nothing
                for k in 1:dv
                    hessian += third_full[i, j, k] * ws.vbar[k]
                end
            end
            ws.hessian[a, r, s] = hessian
        end

        if third_full !== nothing
            for r in 1:d, s in 1:d, q in 1:d
                ws.third_derivative[a, r, s, q] = third_full[
                    sys.random_indices[r], sys.random_indices[s], sys.random_indices[q]]
            end
        end

        ws.mean[a] = value + sum(ws.hessian[a, r, s] * Σ[r, s] for r in 1:d, s in 1:d) / 2
    end

    ws.effective_linear .= ws.linear
    if sys.order == :third_order
        @inbounds for a in 1:nout, r in 1:d
            correction = zero(eltype(ws.effective_linear))
            for s in 1:d, q in 1:d
                correction += ws.third_derivative[a, r, s, q] * Σ[s, q] / 2
            end
            ws.effective_linear[a, r] += correction
        end
    end

    @inbounds for a in 1:nout
        H = view(ws.hessian, a, :, :)
        ws.hessian_covariance[a, :, :] .= Σ * H * Σ
    end
    ws.covariance .= ws.effective_linear * Σ * ws.effective_linear'
    ws.covariance .+= reshape(ws.hessian, nout, d^2) * reshape(ws.hessian_covariance, nout, d^2)' / 2

    if sys.order == :third_order
        ivashchenko_transform_third_tensor!(ws.third_covariance, ws.third_scratch,
                                             ws.third_derivative, Σ)
        ws.covariance .+= reshape(ws.third_derivative, nout, d^3) *
                          reshape(ws.third_covariance, nout, d^3)' / 6
    end

    @inbounds for j in 1:nout, i in 1:j
        value = (ws.covariance[i, j] + ws.covariance[j, i]) / 2
        ws.covariance[i, j] = value
        ws.covariance[j, i] = value
    end
    return ws.mean, ws.covariance
end

function ivashchenko_stationary_initialization(sys, initial_mean, initial_covariance, ws;
                                               workspaces = nothing,
                                               lyapunov_algorithm::Symbol = :doubling)
    scalar_type = eltype(ws.vbar)
    if initial_covariance isa AbstractMatrix
        size(initial_covariance) == (sys.nPast, sys.nPast) ||
            throw(DimensionMismatch("The Ivashchenko initial covariance must be $(sys.nPast)×$(sys.nPast), got $(size(initial_covariance))."))
        return Vector{scalar_type}(initial_mean), Matrix{scalar_type}(initial_covariance), true
    elseif initial_covariance == :diagonal
        return Vector{scalar_type}(initial_mean), Matrix{scalar_type}(10 .* ℒ.I(sys.nPast)), true
    elseif initial_covariance != :theoretical
        throw(ArgumentError("Unsupported Ivashchenko initial covariance: $(initial_covariance)."))
    end

    # Start from the linear stationary covariance, then solve the coupled
    # nonlinear mean/covariance fixed point implied by the unpruned polynomial.
    P = qkf_lyapunov(sys.A, sys.B * sys.B'; workspaces = workspaces,
                     lyapunov_algorithm = lyapunov_algorithm)
    m = Vector{scalar_type}(initial_mean)
    converged = false
    for _ in 1:IVASHCHENKO_STATIONARY_MAXITER
        ivashchenko_polynomial_moments!(sys, m, P, ws)
        next_m = collect(view(ws.mean, sys.state_position))
        next_P = Matrix(view(ws.covariance, sys.state_position, sys.state_position))
        next_P = (next_P + next_P') / 2
        delta_m = isempty(next_m) ? 0.0 : maximum(abs, primal.(next_m .- m))
        delta_P = isempty(next_P) ? 0.0 : maximum(abs, primal.(next_P .- P))
        if !all(isfinite, primal.(next_m)) || !all(isfinite, primal.(next_P))
            return m, P, false
        end
        m, P = next_m, next_P
        if max(delta_m, delta_P) <= IVASHCHENKO_STATIONARY_TOLERANCE *
                                  max(1.0, maximum(abs, primal.(m)), maximum(abs, primal.(P)))
            converged = true
            break
        end
    end
    return m, P, converged
end

function ivashchenko_measurement_covariance(measurement_error, n_obs, scalar_type)
    if measurement_error === nothing
        return zeros(scalar_type, n_obs, n_obs)
    elseif measurement_error isa AbstractMatrix
        size(measurement_error) == (n_obs, n_obs) ||
            throw(DimensionMismatch("Ivashchenko measurement error must be $n_obs×$n_obs."))
        return Matrix{scalar_type}(measurement_error)
    elseif measurement_error isa AbstractVector
        length(measurement_error) == n_obs ||
            throw(DimensionMismatch("Ivashchenko measurement error must have $n_obs entries."))
        return Matrix{scalar_type}(ℒ.Diagonal(collect(measurement_error)))
    else
        return Matrix{scalar_type}(ℒ.I(n_obs)) .* measurement_error
    end
end

function ivashchenko_subset_measurement_covariance(Hm, idx)
    isempty(idx) && return Matrix{eltype(Hm)}(undef, 0, 0)
    return Matrix(Hm[idx, idx])
end

function ivashchenko_copy_moment_tape(ws)
    return (; covariance_input = copy(ws.covariance_input),
            vbar = copy(ws.vbar), mean = copy(ws.mean), linear = copy(ws.linear),
            effective_linear = copy(ws.effective_linear), hessian = copy(ws.hessian),
            hessian_covariance = copy(ws.hessian_covariance), covariance = copy(ws.covariance),
            third_derivative = ws.third_derivative === nothing ? nothing : copy(ws.third_derivative),
            third_covariance = ws.third_covariance === nothing ? nothing : copy(ws.third_covariance))
end

function ivashchenko_stationary_initialization_taped(sys, initial_mean, initial_covariance, ws;
                                                     workspaces = nothing,
                                                     lyapunov_algorithm::Symbol = :doubling)
    scalar_type = eltype(ws.vbar)
    if initial_covariance isa AbstractMatrix
        return Vector{scalar_type}(initial_mean), Matrix{scalar_type}(initial_covariance), true, nothing
    elseif initial_covariance == :diagonal
        return Vector{scalar_type}(initial_mean), Matrix{scalar_type}(10 .* ℒ.I(sys.nPast)), true, nothing
    elseif initial_covariance != :theoretical
        throw(ArgumentError("Unsupported Ivashchenko initial covariance: $(initial_covariance)."))
    end

    P = qkf_lyapunov(sys.A, sys.B * sys.B'; workspaces = workspaces,
                     lyapunov_algorithm = lyapunov_algorithm)
    linear_covariance = copy(P)
    m = Vector{scalar_type}(initial_mean)
    iterations = NamedTuple[]
    converged = false
    for _ in 1:IVASHCHENKO_STATIONARY_MAXITER
        input_mean = copy(m)
        input_covariance = copy(P)
        ivashchenko_polynomial_moments!(sys, m, P, ws)
        moment_tape = ivashchenko_copy_moment_tape(ws)
        next_m = collect(view(ws.mean, sys.state_position))
        next_P = Matrix(view(ws.covariance, sys.state_position, sys.state_position))
        next_P = (next_P + next_P') / 2
        push!(iterations, (; input_mean, input_covariance, moment_tape))
        delta_m = isempty(next_m) ? 0.0 : maximum(abs, primal.(next_m .- m))
        delta_P = isempty(next_P) ? 0.0 : maximum(abs, primal.(next_P .- P))
        if !all(isfinite, primal.(next_m)) || !all(isfinite, primal.(next_P))
            return m, P, false, (; iterations, linear_covariance)
        end
        m, P = next_m, next_P
        if max(delta_m, delta_P) <= IVASHCHENKO_STATIONARY_TOLERANCE *
                                  max(1.0, maximum(abs, primal.(m)), maximum(abs, primal.(P)))
            converged = true
            break
        end
    end
    return m, P, converged, (; iterations, linear_covariance)
end

function ivashchenko_filter_pass(sys, data_in_deviations::AbstractMatrix{<:Real}, initial_mean;
                                 measurement_error = nothing,
                                 initial_covariance = :theoretical,
                                 presample_periods::Int = 0,
                                 on_failure_loglikelihood::Real = -Inf,
                                 workspaces = nothing,
                                 lyapunov_algorithm::Symbol = :doubling,
                                 record::Bool = false)
    n_obs, nT = size(data_in_deviations)
    presample_periods = normalize_presample_periods(presample_periods, nT)
    scalar_type = promote_type(eltype(sys.S1), eltype(data_in_deviations),
                               measurement_error === nothing ? Float64 :
                               (measurement_error isa AbstractArray ? eltype(measurement_error) : typeof(measurement_error)))
    Hm = ivashchenko_measurement_covariance(measurement_error, n_obs, scalar_type)
    obs_idx_per_t, _ = build_obs_index(data_in_deviations)

    ws = ivashchenko_kalman_workspace(sys, scalar_type)
    if record
        mean_state, covariance_state, initialized, initialization_tape =
            ivashchenko_stationary_initialization_taped(sys, initial_mean, initial_covariance, ws;
                                                        workspaces = workspaces, lyapunov_algorithm = lyapunov_algorithm)
    else
        mean_state, covariance_state, initialized = ivashchenko_stationary_initialization(
            sys, initial_mean, initial_covariance, ws;
            workspaces = workspaces, lyapunov_algorithm = lyapunov_algorithm)
        initialization_tape = nothing
    end
    initialized || return record ? (convert(scalar_type, on_failure_loglikelihood), nothing) :
                                  convert(scalar_type, on_failure_loglikelihood)

    state_position, observation_position = sys.state_position, sys.observation_position
    n_state = length(state_position)
    post_means = record ? Vector{Vector{scalar_type}}(undef, nT) : nothing
    post_covariances = record ? Vector{Matrix{scalar_type}}(undef, nT) : nothing
    input_means = record ? Vector{Vector{scalar_type}}(undef, nT) : nothing
    input_covariances = record ? Vector{Matrix{scalar_type}}(undef, nT) : nothing
    moment_tapes = record ? Vector{Any}(undef, nT) : nothing
    predicted_means = record ? Vector{Vector{scalar_type}}(undef, nT) : nothing
    predicted_covariances = record ? Vector{Matrix{scalar_type}}(undef, nT) : nothing
    output_means = record ? Vector{Vector{scalar_type}}(undef, nT) : nothing
    output_covariances = record ? Vector{Matrix{scalar_type}}(undef, nT) : nothing
    transitions = record ? Vector{Matrix{scalar_type}}(undef, nT) : nothing
    shock_loadings = record ? Vector{Matrix{scalar_type}}(undef, nT) : nothing
    innovations = record ? Vector{Any}(undef, nT) : nothing
    inverse_innovation_covariances = record ? Vector{Any}(undef, nT) : nothing
    gains = record ? Vector{Any}(undef, nT) : nothing
    cross_covariances = record ? Vector{Any}(undef, nT) : nothing
    observed_indices = record ? obs_idx_per_t : nothing
    ll = zero(scalar_type)
    log2pi = log(2π)
    @inbounds for t in 1:nT
        input_mean = record ? copy(mean_state) : nothing
        input_covariance = record ? copy(covariance_state) : nothing
        ivashchenko_polynomial_moments!(sys, mean_state, covariance_state, ws)
        if record
            input_means[t] = input_mean
            input_covariances[t] = input_covariance
            moment_tapes[t] = ivashchenko_copy_moment_tape(ws)
        end
        predicted_mean = collect(view(ws.mean, state_position))
        output_mean = copy(ws.mean)
        predicted_covariance = Matrix(view(ws.covariance, state_position, state_position))
        output_covariance = copy(ws.covariance)
        idx = obs_idx_per_t[t]
        m = length(idx)

        if record
            predicted_means[t] = predicted_mean
            predicted_covariances[t] = predicted_covariance
            output_means[t] = output_mean
            output_covariances[t] = output_covariance
            transitions[t] = Matrix(view(ws.effective_linear, state_position, 1:sys.nPast))
            shock_loadings[t] = Matrix(view(ws.effective_linear, state_position, sys.nPast + 1:sys.d))
        end

        if m == 0
            if record
                innovations[t] = nothing
                inverse_innovation_covariances[t] = nothing
                gains[t] = nothing
                cross_covariances[t] = nothing
            end
            mean_state = predicted_mean
            covariance_state = (predicted_covariance + predicted_covariance') / 2
            if record
                post_means[t] = copy(mean_state)
                post_covariances[t] = copy(covariance_state)
            end
            continue
        end

        observation_mean = collect(view(ws.mean, observation_position[idx]))
        observation_covariance = Matrix(view(ws.covariance, observation_position[idx], observation_position[idx]))
        cross_covariance = Matrix(view(ws.covariance, state_position, observation_position[idx]))
        observations = collect(view(data_in_deviations, idx, t))
        innovation = observations - observation_mean
        F = observation_covariance + ivashchenko_subset_measurement_covariance(Hm, idx)
        F = (F + F') / 2
        factor = ℒ.lu(F, check = false)
        ℒ.issuccess(factor) || return record ? (convert(scalar_type, on_failure_loglikelihood), nothing) :
                                              convert(scalar_type, on_failure_loglikelihood)
        logabsdetF, signF = ℒ.logabsdet(factor)
        (primal(signF) > 0 && isfinite(primal(logabsdetF))) ||
            return record ? (convert(scalar_type, on_failure_loglikelihood), nothing) :
                            convert(scalar_type, on_failure_loglikelihood)
        invF = Matrix(factor \ ℒ.I(m))

        if t > presample_periods
            solved_innovation = invF * innovation
            ll -= (ℒ.dot(innovation, solved_innovation) + logabsdetF + m * log2pi) / 2
            isfinite(primal(ll)) || return record ? (convert(scalar_type, on_failure_loglikelihood), nothing) :
                                                   convert(scalar_type, on_failure_loglikelihood)
        end

        gain = cross_covariance * invF
        if record
            innovations[t] = copy(innovation)
            inverse_innovation_covariances[t] = copy(invF)
            gains[t] = copy(gain)
            cross_covariances[t] = copy(cross_covariance)
        end
        mean_state = predicted_mean + gain * innovation
        covariance_state = predicted_covariance - gain * cross_covariance'
        covariance_state = (covariance_state + covariance_state') / 2
        if record
            post_means[t] = copy(mean_state)
            post_covariances[t] = copy(covariance_state)
        end
    end

    if !record
        return ll
    end
    return ll, (; initialization_tape, input_means, input_covariances, moment_tapes,
                initial_mean = Vector{scalar_type}(initial_mean),
                initial_covariance = copy(input_covariances[1]),
                predicted_means, predicted_covariances, post_means, post_covariances,
                output_means, output_covariances, transitions, shock_loadings,
                innovations, inverse_innovation_covariances, gains, cross_covariances,
                observed_indices, Hm, presample_periods, data = Matrix{scalar_type}(data_in_deviations),
                state_position, observation_position)
end

function run_ivashchenko_kalman(sys, data_in_deviations::AbstractMatrix{<:Real}, initial_mean;
                                measurement_error = nothing,
                                initial_covariance = :theoretical,
                                presample_periods::Int = 0,
                                on_failure_loglikelihood::Real = -Inf,
                                workspaces = nothing,
                                lyapunov_algorithm::Symbol = :doubling)
    return ivashchenko_filter_pass(sys, data_in_deviations, initial_mean;
                                   measurement_error = measurement_error,
                                   initial_covariance = initial_covariance,
                                   presample_periods = presample_periods,
                                   on_failure_loglikelihood = on_failure_loglikelihood,
                                   workspaces = workspaces,
                                   lyapunov_algorithm = lyapunov_algorithm)
end

function ivashchenko_polynomial_moments_pullback(sys, moment_tape, mean_bar, covariance_bar)
    nout, d, dv = sys.nout, sys.d, sys.dv
    Σ = moment_tape.covariance_input
    v = moment_tape.vbar
    L = moment_tape.effective_linear
    K = moment_tape.hessian
    third = moment_tape.third_derivative
    gΣ = zeros(eltype(Σ), d, d)
    gv = zeros(eltype(v), dv)
    gL = zeros(eltype(L), nout, d)
    gK = zeros(eltype(K), nout, d, d)
    gS1 = zeros(eltype(sys.S1), nout, dv)
    gH = zeros(eltype(sys.H), nout, dv, dv)
    gthird = third === nothing ? nothing : zeros(eltype(third), nout, dv, dv, dv)

    # C₁ = L Σ L'.
    @inbounds for a in 1:nout, b in 1:nout, r in 1:d, s in 1:d
        gL[a, r] += (covariance_bar[a, b] + covariance_bar[b, a]) * L[b, s] * Σ[r, s]
        gΣ[r, s] += covariance_bar[a, b] * L[a, r] * L[b, s]
    end

    # C₂ = 1/2 K : (Σ K Σ), written with matrix contractions to keep the cubic
    # reverse pass manageable for the unpruned state dimension.
    hessian_covariance_bar = zeros(eltype(K), nout, d, d)
    @inbounds for b in 1:nout, i in 1:d, j in 1:d, a in 1:nout
        hessian_covariance_bar[b, i, j] += covariance_bar[a, b] * K[a, i, j] / 2
        gK[a, i, j] += covariance_bar[a, b] * moment_tape.hessian_covariance[b, i, j] / 2
    end
    @inbounds for b in 1:nout, i in 1:d, j in 1:d, p in 1:d, q in 1:d
        value = hessian_covariance_bar[b, i, j]
        gK[b, p, q] += value * Σ[i, p] * Σ[q, j]
        gΣ[i, p] += value * K[b, p, q] * Σ[q, j]
        gΣ[q, j] += value * Σ[i, p] * K[b, p, q]
    end

    if third !== nothing
        # C₃ = 1/6 T : (Σ ⊗ Σ ⊗ Σ)T'.
        third_covariance_bar = zeros(eltype(third), nout, d, d, d)
        @inbounds for b in 1:nout, i in 1:d, j in 1:d, k in 1:d, a in 1:nout
            third_covariance_bar[b, i, j, k] += covariance_bar[a, b] * third[a, i, j, k] / 6
            gthird[a, sys.random_indices[i], sys.random_indices[j], sys.random_indices[k]] +=
                covariance_bar[a, b] * moment_tape.third_covariance[b, i, j, k] / 6
        end
        @inbounds for b in 1:nout, i in 1:d, j in 1:d, k in 1:d,
                       p in 1:d, q in 1:d, r in 1:d
            value = third_covariance_bar[b, i, j, k] * third[b, p, q, r]
            gthird[b, sys.random_indices[p], sys.random_indices[q], sys.random_indices[r]] +=
                third_covariance_bar[b, i, j, k] * Σ[i, p] * Σ[j, q] * Σ[k, r]
            gΣ[i, p] += value * Σ[j, q] * Σ[k, r]
            gΣ[j, q] += value * Σ[i, p] * Σ[k, r]
            gΣ[k, r] += value * Σ[i, p] * Σ[j, q]
        end
    end

    # The mean is f(v) + 1/2 K:Σ.
    @inbounds for a in 1:nout, r in 1:d, s in 1:d
        gK[a, r, s] += mean_bar[a] * Σ[r, s] / 2
        gΣ[r, s] += mean_bar[a] * K[a, r, s] / 2
    end

    # L_eff = L₀ + 1/2 T⋅Σ.
    @inbounds for a in 1:nout, r in 1:d
        if third !== nothing
            for s in 1:d, q in 1:d
                i = sys.random_indices[r]
                j = sys.random_indices[s]
                k = sys.random_indices[q]
                gthird[a, i, j, k] += gL[a, r] * Σ[s, q] / 2
                gΣ[s, q] += gL[a, r] * third[a, r, s, q] / 2
            end
        end
    end
    linear_bar = gL

    # K = H[random, random] + T[random, random, :]⋅v.
    @inbounds for a in 1:nout, r in 1:d, s in 1:d
        i, j = sys.random_indices[r], sys.random_indices[s]
        gH[a, i, j] += gK[a, r, s]
        if third !== nothing
            for q in 1:dv
                gthird[a, i, j, q] += gK[a, r, s] * v[q]
                gv[q] += gK[a, r, s] * sys.third_derivative[a, i, j, q]
            end
        end
    end

    # L₀ = S₁[random] + H[random, :]v + 1/2 T[random, :, :]vv.
    @inbounds for a in 1:nout, r in 1:d
        i = sys.random_indices[r]
        gS1[a, i] += linear_bar[a, r]
        for j in 1:dv
            gH[a, i, j] += linear_bar[a, r] * v[j]
            gv[j] += linear_bar[a, r] * sys.H[a, i, j]
        end
        if third !== nothing
            for j in 1:dv, k in 1:dv
                value = linear_bar[a, r] * v[j] * v[k] / 2
                gthird[a, i, j, k] += value
                gv[j] += linear_bar[a, r] * sys.third_derivative[a, i, j, k] * v[k] / 2
                gv[k] += linear_bar[a, r] * sys.third_derivative[a, i, j, k] * v[j] / 2
            end
        end
    end

    # f(v) = S₁v + 1/2 Hv² + 1/6 Tv³.
    @inbounds for a in 1:nout
        coefficient_bar = mean_bar[a]
        for i in 1:dv
            gS1[a, i] += coefficient_bar * v[i]
            gv[i] += coefficient_bar * sys.S1[a, i]
        end
        for i in 1:dv, j in 1:dv
            value = coefficient_bar * v[i] * v[j] / 2
            gH[a, i, j] += value
            gv[i] += coefficient_bar * sys.H[a, i, j] * v[j] / 2
            gv[j] += coefficient_bar * sys.H[a, i, j] * v[i] / 2
        end
        if third !== nothing
            for i in 1:dv, j in 1:dv, k in 1:dv
                value = coefficient_bar * v[i] * v[j] * v[k] / 6
                gthird[a, i, j, k] += value
                gv[i] += coefficient_bar * sys.third_derivative[a, i, j, k] * v[j] * v[k] / 6
                gv[j] += coefficient_bar * sys.third_derivative[a, i, j, k] * v[i] * v[k] / 6
                gv[k] += coefficient_bar * sys.third_derivative[a, i, j, k] * v[i] * v[j] / 6
            end
        end
    end

    gΣ .= (gΣ + gΣ') / 2
    gmean = copy(view(gv, 1:sys.nPast))
    gcovariance = copy(view(gΣ, 1:sys.nPast, 1:sys.nPast))
    return gmean, gcovariance, gS1, gH, gthird
end

function ivashchenko_solution_matrix_pullback(sys, gS1, gH, gthird, solution_matrices)
    gS2 = zeros(eltype(solution_matrices[2]), size(solution_matrices[2]))
    @inbounds for a in 1:sys.nout, i in 1:sys.dv, j in 1:sys.dv
        gS2[a, polynomial_pair_index(i, j, sys.dv)] += (gH[a, i, j] + gH[a, j, i]) / 2
    end
    if gthird === nothing
        return gS1, gS2, nothing
    end
    gS3 = zeros(eltype(solution_matrices[3]), size(solution_matrices[3]))
    @inbounds for a in 1:sys.nout, i in 1:sys.dv, j in 1:sys.dv, k in 1:sys.dv
        gS3[a, polynomial_triple_index(i, j, k, sys.dv)] += (
            gthird[a, i, j, k] + gthird[a, i, k, j] +
            gthird[a, j, i, k] + gthird[a, j, k, i] +
            gthird[a, k, i, j] + gthird[a, k, j, i]) / 6
    end
    return gS1, gS2, gS3
end

function ivashchenko_lyapunov_pullback(A, B, P, covariance_bar; lyapunov_algorithm::Symbol = :doubling)
    adjoint_covariance = qkf_lyapunov(A', covariance_bar;
                                     lyapunov_algorithm = lyapunov_algorithm)
    adjoint_covariance = (adjoint_covariance + adjoint_covariance') / 2
    adjoint_A = adjoint_covariance * A * P' + adjoint_covariance' * A * P
    adjoint_Q = adjoint_covariance
    adjoint_B = (adjoint_Q + adjoint_Q') * B
    return adjoint_A, adjoint_B
end

function ivashchenko_filter_pullback(sys, tape, solution_matrices, scale;
                                     initial_covariance = :theoretical,
                                     lyapunov_algorithm::Symbol = :doubling)
    nT = length(tape.post_means)
    scalar_type = eltype(tape.post_means[1])
    mean_bar = zeros(scalar_type, sys.nPast)
    covariance_bar = zeros(scalar_type, sys.nPast, sys.nPast)
    data_bar = zeros(scalar_type, size(tape.data))
    gS1 = zeros(scalar_type, size(sys.S1))
    gH = zeros(scalar_type, size(sys.H))
    gthird = sys.third_derivative === nothing ? nothing : zeros(scalar_type, size(sys.third_derivative))

    @inbounds for t in nT:-1:1
        idx = tape.observed_indices[t]
        mean_output_bar = zeros(scalar_type, sys.nout)
        covariance_output_bar = zeros(scalar_type, sys.nout, sys.nout)
        predicted_mean_bar = copy(mean_bar)
        predicted_covariance_bar = copy(covariance_bar)
        if isempty(idx)
            mean_output_bar[sys.state_position] .+= predicted_mean_bar
            covariance_output_bar[sys.state_position, sys.state_position] .+= predicted_covariance_bar
        else
            innovation = tape.innovations[t]
            invF = tape.inverse_innovation_covariances[t]
            gain = tape.gains[t]
            cross = tape.cross_covariances[t]
            mean_bar_post = mean_bar
            # The forward covariance update is explicitly symmetrised; only
            # the symmetric part of its cotangent can reach the pre-update
            # covariance and gain.
            covariance_bar_post = (covariance_bar + covariance_bar') / 2

            gain_bar = mean_bar_post * innovation'
            innovation_bar = gain' * mean_bar_post
            gain_bar .-= covariance_bar_post * cross
            cross_bar = -covariance_bar_post' * gain

            inverse_covariance_bar = cross' * gain_bar
            cross_bar .+= gain_bar * invF'
            if t > tape.presample_periods
                innovation_bar .-= scale * (invF * innovation)
            end
            inverse_covariance_bar .+= zero(scalar_type)
            covariance_bar_innovation = -invF' * inverse_covariance_bar * invF'
            if t > tape.presample_periods
                covariance_bar_innovation .-= scale / 2 *
                    (invF' - invF' * innovation * innovation' * invF')
            end
            data_bar[idx, t] .+= innovation_bar
            mean_output_bar[sys.state_position] .+= predicted_mean_bar
            mean_output_bar[sys.observation_position[idx]] .-= innovation_bar
            covariance_output_bar[sys.state_position, sys.state_position] .+= predicted_covariance_bar
            covariance_output_bar[sys.observation_position[idx], sys.observation_position[idx]] .+=
                (covariance_bar_innovation + covariance_bar_innovation') / 2
            # The moment covariance is explicitly symmetric, so split the
            # cross-block cotangent across its two transpose locations.
            covariance_output_bar[sys.state_position, sys.observation_position[idx]] .+= cross_bar / 2
            covariance_output_bar[sys.observation_position[idx], sys.state_position] .+= cross_bar' / 2
        end

        mean_bar, covariance_bar, local_S1, local_H, local_third =
            ivashchenko_polynomial_moments_pullback(sys, tape.moment_tapes[t],
                                                    mean_output_bar, covariance_output_bar)
        gS1 .+= local_S1
        gH .+= local_H
        if gthird !== nothing
            gthird .+= local_third
        end
    end

    if tape.initialization_tape !== nothing
        initialization_mean_bar = copy(mean_bar)
        initialization_covariance_bar = copy(covariance_bar)
        for iteration in reverse(tape.initialization_tape.iterations)
            output_mean_bar = zeros(scalar_type, sys.nout)
            output_covariance_bar = zeros(scalar_type, sys.nout, sys.nout)
            output_mean_bar[sys.state_position] .= initialization_mean_bar
            output_covariance_bar[sys.state_position, sys.state_position] .=
                (initialization_covariance_bar + initialization_covariance_bar') / 2
            initialization_mean_bar, initialization_covariance_bar, local_S1, local_H, local_third =
                ivashchenko_polynomial_moments_pullback(sys, iteration.moment_tape,
                                                        output_mean_bar, output_covariance_bar)
            gS1 .+= local_S1
            gH .+= local_H
            if gthird !== nothing
                gthird .+= local_third
            end
        end
        initial_covariance_bar = (initialization_covariance_bar + initialization_covariance_bar') / 2
        initial_A_bar, initial_B_bar = ivashchenko_lyapunov_pullback(
            sys.A, sys.B, tape.initialization_tape.linear_covariance, initial_covariance_bar;
            lyapunov_algorithm = lyapunov_algorithm)
        gS1[sys.state_position, 1:sys.nPast] .+= initial_A_bar
        gS1[sys.state_position, sys.nPast + 2:sys.dv] .+= initial_B_bar
        mean_bar = initialization_mean_bar
    end

    local_S1, local_S2, local_S3 = ivashchenko_solution_matrix_pullback(
        sys, gS1, gH, gthird, solution_matrices)
    solution_bar = [zeros(scalar_type, size(solution_matrices[1])),
                    zeros(scalar_type, size(solution_matrices[2]))]
    solution_bar[1][sys.output_rows, :] .= local_S1
    solution_bar[2][sys.output_rows, :] .= local_S2
    if local_S3 !== nothing
        solution_bar = vcat(solution_bar, [zeros(scalar_type, size(solution_matrices[3]))])
        solution_bar[3][sys.output_rows, :] .= local_S3
    end
    state_bar = zeros(scalar_type, sys.nVars)
    state_bar[sys.past] .= mean_bar
    return solution_bar, data_bar, state_bar
end

function ivashchenko_smooth_pass(sys, tape)
    nT = length(tape.post_means)
    n_state = length(tape.state_position)
    smoothed_means = [copy(tape.post_means[t]) for t in 1:nT]
    smoothed_covariances = [copy(tape.post_covariances[t]) for t in 1:nT]
    for t in nT - 1:-1:1
        transition = tape.transitions[t + 1]
        cross = transition * tape.post_covariances[t]
        smoother_gain = cross * inv(tape.predicted_covariances[t + 1])
        delta = smoothed_means[t + 1] - tape.predicted_means[t + 1]
        smoothed_means[t] .+= smoother_gain * delta
        smoothed_covariances[t] .= tape.post_covariances[t] +
            smoother_gain * (smoothed_covariances[t + 1] - tape.predicted_covariances[t + 1]) * smoother_gain'
        smoothed_covariances[t] .= (smoothed_covariances[t] + smoothed_covariances[t]') / 2
    end

    variables = zeros(eltype(tape.post_means[1]), sys.nVars, nT)
    standard_deviations = zeros(eltype(tape.post_means[1]), sys.nVars, nT)
    shocks = zeros(eltype(tape.post_means[1]), sys.nExo, nT)
    @inbounds for t in 1:nT
        pred_covariance = tape.predicted_covariances[t]
        state_delta = smoothed_means[t] - tape.predicted_means[t]
        state_regression = tape.output_covariances[t][ :, tape.state_position] * inv(pred_covariance)
        variables[:, t] .= tape.output_means[t] + state_regression * state_delta
        variables[tape.state_position, t] .= smoothed_means[t]
        standard_deviations[:, t] .= sqrt.(abs.(ℒ.diag(tape.output_covariances[t] -
            state_regression * pred_covariance * state_regression')))
        standard_deviations[tape.state_position, t] .= sqrt.(abs.(ℒ.diag(smoothed_covariances[t])))

        shock_regression = tape.shock_loadings[t]' * inv(pred_covariance)
        shocks[:, t] .= shock_regression * state_delta
    end

    decomposition = zeros(eltype(variables), sys.nVars, sys.nExo + 2, nT)
    decomposition[:, end - 1, :] .= variables
    return variables, shocks, standard_deviations, decomposition, smoothed_means, smoothed_covariances
end

function ivashchenko_filter_data_with_model(𝓂::ℳ,
                                            data_in_deviations::KeyedArray{Float64},
                                            order::Symbol;
                                            initial_covariance = :theoretical,
                                            measurement_error = nothing,
                                            smooth::Bool = true,
                                            opts::CalculationOptions = merge_calculation_options())
    constants = initialise_constants!(𝓂)
    T = constants.post_model_macro
    nT = size(data_in_deviations, 2)
    variables = zeros(Float64, T.nVars, nT)
    shocks = zeros(Float64, T.nExo, nT)
    standard_deviations = zeros(Float64, T.nVars, nT)
    decomposition = zeros(Float64, T.nVars, T.nExo + 2, nT)

    result = calculate_stochastic_steady_state(Val(order), 𝓂.parameter_values, 𝓂, opts = opts)
    sss, converged, SS_and_pars, solution_error = result[1:4]
    if !converged || solution_error > opts.tol.nsss.acceptance_tol || !isfinite(solution_error)
        @error "Could not find a stochastic steady state for the Ivashchenko filter."
        return variables, shocks, standard_deviations, decomposition
    end
    𝐒 = order == :second_order ? result[7:8] : result[8:10]
    ensure_model_structure_constants!(constants, 𝓂.equations.calibration_parameters)
    all_SS = expand_steady_state(SS_and_pars, constants.post_complete_parameters)
    state = collect(sss) - all_SS
    observables = get_and_check_observables(T, data_in_deviations)
    observable_indices = convert(Vector{Int}, indexin(observables, constants.post_complete_parameters.SS_and_pars_names))
    sys = build_ivashchenko_kalman_system_from_constants(constants, 𝐒, observable_indices, order)
    data = collect(data_in_deviations)
    pass = ivashchenko_filter_pass(sys, data, state[sys.past];
                                   measurement_error = measurement_error,
                                   initial_covariance = initial_covariance,
                                   presample_periods = 0,
                                   workspaces = 𝓂.workspaces,
                                   lyapunov_algorithm = opts.lyapunov_algorithm,
                                   record = true)
    pass[2] === nothing && return variables, shocks, standard_deviations, decomposition
    if smooth
        return ivashchenko_smooth_pass(sys, pass[2])[1:4]
    end

    tape = pass[2]
    @inbounds for t in 1:nT
        variables[:, t] .= tape.output_means[t]
        standard_deviations[:, t] .= sqrt.(abs.(ℒ.diag(tape.output_covariances[t])))
    end
    decomposition[:, end - 1, :] .= variables
    return variables, shocks, standard_deviations, decomposition
end

@unstable function filter_data_with_model(𝓂::ℳ,
                                          data_in_deviations::KeyedArray{Float64},
                                          ::Val{:second_order},
                                          ::Val{:ivashchenko_kalman};
                                          warmup_iterations::Int = 0,
                                          initial_covariance = :theoretical,
                                          measurement_error = nothing,
                                          smooth::Bool = true,
                                          opts::CalculationOptions = merge_calculation_options())
    return ivashchenko_filter_data_with_model(𝓂, data_in_deviations, :second_order;
                                              initial_covariance = initial_covariance,
                                              measurement_error = measurement_error,
                                              smooth = smooth, opts = opts)
end

@unstable function filter_data_with_model(𝓂::ℳ,
                                          data_in_deviations::KeyedArray{Float64},
                                          ::Val{:third_order},
                                          ::Val{:ivashchenko_kalman};
                                          warmup_iterations::Int = 0,
                                          initial_covariance = :theoretical,
                                          measurement_error = nothing,
                                          smooth::Bool = true,
                                          opts::CalculationOptions = merge_calculation_options())
    return ivashchenko_filter_data_with_model(𝓂, data_in_deviations, :third_order;
                                              initial_covariance = initial_covariance,
                                              measurement_error = measurement_error,
                                              smooth = smooth, opts = opts)
end

function calculate_loglikelihood(::Val{:ivashchenko_kalman}, ::Val{O},
                                 observables_index::Vector{Int}, 𝐒,
                                 data_in_deviations::AbstractMatrix,
                                 constants, state, workspaces;
                                 warmup_iterations::Int = 0,
                                 presample_periods::Int = 0,
                                 initial_covariance = :theoretical,
                                 filter_algorithm::Symbol = :LagrangeNewton,
                                 lyapunov_algorithm::Symbol = :doubling,
                                 on_failure_loglikelihood = -Inf,
                                 measurement_error = nothing,
                                 opts::CalculationOptions = merge_calculation_options()) where {O}
    O ∈ (:second_order, :third_order) ||
        throw(ArgumentError("The Ivashchenko filter requires `algorithm = :second_order` or `:third_order`."))
    sys = build_ivashchenko_kalman_system_from_constants(constants, 𝐒, observables_index, O)
    initial_mean = state[sys.past]
    return run_ivashchenko_kalman(sys, data_in_deviations, initial_mean;
                                  measurement_error = measurement_error,
                                  initial_covariance = initial_covariance,
                                  presample_periods = presample_periods,
                                  on_failure_loglikelihood = on_failure_loglikelihood,
                                  workspaces = workspaces,
                                  lyapunov_algorithm = lyapunov_algorithm)
end

function calculate_loglikelihood_with_missing(::Val{:ivashchenko_kalman}, ::Val{O},
                                              observables_index::Vector{Int}, 𝐒,
                                              data_in_deviations::AbstractMatrix,
                                              constants, state, workspaces,
                                              obs_idx_per_t::Vector{Vector{Int}};
                                              warmup_iterations::Int = 0,
                                              presample_periods::Int = 0,
                                              initial_covariance = :theoretical,
                                              filter_algorithm::Symbol = :LagrangeNewton,
                                              lyapunov_algorithm::Symbol = :doubling,
                                              on_failure_loglikelihood = -Inf,
                                              measurement_error = nothing,
                                              opts::CalculationOptions = merge_calculation_options()) where {O}
    O ∈ (:second_order, :third_order) ||
        throw(ArgumentError("The Ivashchenko filter requires `algorithm = :second_order` or `:third_order`."))
    sys = build_ivashchenko_kalman_system_from_constants(constants, 𝐒, observables_index, O)
    initial_mean = state[sys.past]
    return run_ivashchenko_kalman(sys, data_in_deviations, initial_mean;
                                  measurement_error = measurement_error,
                                  initial_covariance = initial_covariance,
                                  presample_periods = presample_periods,
                                  on_failure_loglikelihood = on_failure_loglikelihood,
                                  workspaces = workspaces,
                                  lyapunov_algorithm = lyapunov_algorithm)
end

end # @stable
