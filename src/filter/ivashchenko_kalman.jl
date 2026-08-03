@stable default_mode = "disable" begin

# Unpruned Gaussian moment-closure filter in the spirit of Ivashchenko (2014).
#
# The second- and third-order perturbation solutions are treated as the actual
# polynomial transition/measurement map, rather than as a linear map on the
# pruned augmented state. If u = [xₜ₋₁; 1; εₜ], its random component is Gaussian;
# the map is expanded around E[u] and its moments are closed analytically. At second order this
# requires fourth Gaussian moments; the cubic extension additionally uses the
# sixth moments through the third Hermite component.

const IVASHCHENKO_STATIONARY_MAXITER = 500
const IVASHCHENKO_STATIONARY_TOLERANCE = 1e-10

compressed_pair_index(i::Int, j::Int, n::Int) = begin
    i, j = max(i, j), min(i, j)
    (i - 1) * i ÷ 2 + j
end

compressed_triple_index(i::Int, j::Int, k::Int, n::Int) = begin
    i, j, k = sort((i, j, k), rev = true)
    (i - 1) * i * (i + 1) ÷ 6 + (j - 1) * j ÷ 2 + k
end

const IVASHCHENKO_SIXTH_PAIRINGS = (
    ((1, 2), (3, 4), (5, 6)), ((1, 2), (3, 5), (4, 6)),
    ((1, 2), (3, 6), (4, 5)), ((1, 3), (2, 4), (5, 6)),
    ((1, 3), (2, 5), (4, 6)), ((1, 3), (2, 6), (4, 5)),
    ((1, 4), (2, 3), (5, 6)), ((1, 4), (2, 5), (3, 6)),
    ((1, 4), (2, 6), (3, 5)), ((1, 5), (2, 3), (4, 6)),
    ((1, 5), (2, 4), (3, 6)), ((1, 5), (2, 6), (3, 4)),
    ((1, 6), (2, 3), (4, 5)), ((1, 6), (2, 4), (3, 5)),
    ((1, 6), (2, 5), (3, 4)))

function ivashchenko_gaussian_sixth(indices::NTuple{6, Int}, covariance)
    value = zero(eltype(covariance))
    @inbounds for pairing in IVASHCHENKO_SIXTH_PAIRINGS
        value += covariance[indices[pairing[1][1]], indices[pairing[1][2]]] *
                 covariance[indices[pairing[2][1]], indices[pairing[2][2]]] *
                 covariance[indices[pairing[3][1]], indices[pairing[3][2]]]
    end
    return value
end

function ivashchenko_gaussian_sixth_pullback!(covariance_bar, weight,
                                              indices::NTuple{6, Int}, covariance)
    @inbounds for pairing in IVASHCHENKO_SIXTH_PAIRINGS
        first_pair, second_pair, third_pair = pairing
        first_value = covariance[indices[first_pair[1]], indices[first_pair[2]]]
        second_value = covariance[indices[second_pair[1]], indices[second_pair[2]]]
        third_value = covariance[indices[third_pair[1]], indices[third_pair[2]]]
        first_weight = weight * second_value * third_value
        second_weight = weight * first_value * third_value
        third_weight = weight * first_value * second_value
        covariance_bar[indices[first_pair[1]], indices[first_pair[2]]] += first_weight
        covariance_bar[indices[second_pair[1]], indices[second_pair[2]]] += second_weight
        covariance_bar[indices[third_pair[1]], indices[third_pair[2]]] += third_weight
    end
    return covariance_bar
end

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
    n_pair = dv * (dv + 1) ÷ 2
    size(S1, 2) == dv || throw(DimensionMismatch("S₁ has $(size(S1, 2)) columns; expected $dv."))
    size(S2, 2) == n_pair || throw(DimensionMismatch("S₂ has $(size(S2, 2)) columns; expected $n_pair compressed quadratic monomials."))

    scalar_type = order == :third_order ? promote_type(eltype(S1), eltype(S2), eltype(𝐒[3])) : promote_type(eltype(S1), eltype(S2))
    S1 = Matrix{scalar_type}(S1)
    S2 = Matrix{scalar_type}(S2)

    S3 = nothing
    if order == :third_order
        n_triple = dv * (dv + 1) * (dv + 2) ÷ 6
        S3 = Matrix(𝐒[3][output_rows, :])
        size(S3, 2) == n_triple || throw(DimensionMismatch("S₃ has $(size(S3, 2)) columns; expected $n_triple compressed cubic monomials."))
        S3 = Matrix{scalar_type}(S3)
    end

    quadratic_hessian = order == :second_order ? zeros(scalar_type, d * d, nout) : nothing

    pair_indices = Tuple{Int, Int}[]
    pair_multiplicities = Int[]
    for i in 1:dv, j in 1:i
        push!(pair_indices, (i, j))
        push!(pair_multiplicities, i == j ? 1 : 2)
    end
    triple_indices = Tuple{Int, Int, Int}[]
    triple_multiplicities = Int[]
    if order == :third_order
        for i in 1:dv, j in 1:i, k in 1:j
            push!(triple_indices, (i, j, k))
            push!(triple_multiplicities, i == j == k ? 1 : (i == j || j == k ? 3 : 6))
        end
    end

    row_position = zeros(Int, nVars)
    @inbounds for (i, row) in enumerate(output_rows)
        row_position[row] = i
    end
    state_position = row_position[past]
    observation_position = row_position[observables_index]
    random_indices = vcat(collect(1:nPast), collect(nPast + 2:dv))
    random_pair_indices = Tuple{Int, Int}[]
    random_pair_columns = Int[]
    random_pair_multiplicities = Int[]
    for r in 1:d, s in 1:r
        push!(random_pair_indices, (r, s))
        i, j = random_indices[r], random_indices[s]
        push!(random_pair_columns, compressed_pair_index(i, j, dv))
        push!(random_pair_multiplicities, r == s ? 1 : 2)
    end
    # Keep the full compressed-pair column layout so it can multiply pair_mean
    # directly; columns involving the deterministic constant remain zero.
    random_hessian = zeros(scalar_type, nout, n_pair)
    @inbounds for p in eachindex(random_pair_indices)
        random_hessian[:, p] .= S2[:, random_pair_columns[p]]
    end
    random_triple_indices = Tuple{Int, Int, Int}[]
    random_triple_columns = Int[]
    random_triple_multiplicities = Int[]
    if order == :third_order
        for r in 1:d, s in 1:r, q in 1:s
            push!(random_triple_indices, (r, s, q))
            i, j, k = random_indices[r], random_indices[s], random_indices[q]
            push!(random_triple_columns, compressed_triple_index(i, j, k, dv))
            push!(random_triple_multiplicities, r == s == q ? 1 : (r == s || s == q ? 3 : 6))
        end
    end

    if quadratic_hessian !== nothing
        @inbounds for p in eachindex(random_pair_indices)
            r, s = random_pair_indices[p]
            column = random_pair_columns[p]
            for output in 1:nout
                value = S2[output, column]
                quadratic_hessian[r + (s - 1) * d, output] = value
                quadratic_hessian[s + (r - 1) * d, output] = value
            end
        end
    end

    # The linear-limit covariance is a useful starting point for the nonlinear
    # stationary fixed point and also makes the :diagonal option match Kalman's
    # convention exactly.
    state_s1 = S1[state_position, :]
    A = state_s1[:, 1:nPast]
    B = state_s1[:, nPast + 2:dv]

    return (; order, nVars, nPast, nExo, d, dv, nout, output_rows, past,
            n_pair, pair_indices, pair_multiplicities, triple_indices,
            triple_multiplicities, S1, S2, S3, state_position,
            observation_position, random_indices, random_pair_indices,
            random_pair_columns, random_pair_multiplicities, random_hessian,
            random_triple_indices, random_triple_columns,
            random_triple_multiplicities, quadratic_hessian, A, B)
end

build_ivashchenko_kalman_system(𝓂::ℳ, 𝐒, oi::Vector{Int}, order::Symbol) =
    build_ivashchenko_kalman_system_from_constants(𝓂.constants, 𝐒, oi, order)

function ivashchenko_kalman_workspace(sys, scalar_type::Type)
    nout, d, n_pair = sys.nout, sys.d, sys.n_pair
    third = sys.order == :third_order
    n_triple = length(sys.triple_indices)
    hessian = third ? copy(sys.random_hessian) : sys.random_hessian
    quadratic_hessian = sys.quadratic_hessian
    return (; covariance_input = zeros(scalar_type, d, d),
            vbar = zeros(scalar_type, sys.dv),
            mean = zeros(scalar_type, nout),
            linear = zeros(scalar_type, nout, d),
            linear_left = zeros(scalar_type, nout, d),
            effective_linear = zeros(scalar_type, nout, d),
            hessian,
            quadratic_hessian,
            quadratic_left = third ? nothing : zeros(scalar_type, d * d, nout),
            quadratic_right = third ? nothing : zeros(scalar_type, d * d, nout),
            pair_mean = zeros(scalar_type, n_pair),
            pair_covariance = third ? zeros(scalar_type, n_pair, n_pair) : nothing,
            pair_product = third ? zeros(scalar_type, nout, n_pair) : nothing,
            covariance = zeros(scalar_type, nout, nout),
            third_derivative = third ? zeros(scalar_type, nout, n_triple) : nothing,
            third_covariance = third ? zeros(scalar_type, n_triple, n_triple) : nothing,
            third_cross = third ? zeros(scalar_type, d, n_triple) : nothing,
            third_linear = third ? zeros(scalar_type, n_triple, d) : nothing,
            third_work = third ? zeros(scalar_type, nout, n_triple) : nothing,
            third_output = third ? zeros(scalar_type, nout, nout) : nothing,
            pair_scratch = zeros(scalar_type, n_pair),
            triple_scratch = third ? zeros(scalar_type, n_triple) : nothing,
            basis = zeros(scalar_type, sys.dv),
            second_basis = zeros(scalar_type, sys.dv))
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

    compressed_kron²_power!(ws.pair_scratch, ws.vbar)
    ℒ.mul!(ws.mean, sys.S1, ws.vbar)
    ℒ.mul!(ws.mean, sys.S2, ws.pair_scratch, 0.5, 1.0)
    if sys.order == :third_order
        compressed_kron³_power!(ws.triple_scratch, ws.vbar)
        ℒ.mul!(ws.mean, sys.S3, ws.triple_scratch, 1 / 6, 1.0)
    end

    fill!(ws.linear, zero(eltype(ws.linear)))
    @inbounds for r in 1:d
        i = sys.random_indices[r]
        fill!(ws.basis, zero(eltype(ws.basis)))
        ws.basis[i] = one(eltype(ws.basis))
        compressed_kron²!(ws.pair_scratch, ws.vbar, ws.basis)
        ℒ.mul!(view(ws.linear, :, r), sys.S2, ws.pair_scratch, 0.5, 0.0)
        @views ws.linear[:, r] .+= sys.S1[:, i]
        if sys.order == :third_order
            compressed_kron³!(ws.triple_scratch, ws.vbar, ws.vbar, ws.basis)
            ℒ.mul!(view(ws.linear, :, r), sys.S3, ws.triple_scratch, 0.5, 1.0)
        end
    end

    if sys.order == :third_order
        @inbounds for p in eachindex(sys.random_pair_indices)
            r, s = sys.random_pair_indices[p]
            ws.hessian[:, p] .= sys.S2[:, sys.random_pair_columns[p]]
            fill!(ws.basis, zero(eltype(ws.basis)))
            ws.basis[sys.random_indices[r]] = one(eltype(ws.basis))
            copyto!(ws.second_basis, ws.basis)
            fill!(ws.basis, zero(eltype(ws.basis)))
            ws.basis[sys.random_indices[s]] = one(eltype(ws.basis))
            compressed_kron³!(ws.triple_scratch, ws.vbar, ws.second_basis, ws.basis)
            ws.hessian[:, p] .+= sys.S3 * ws.triple_scratch
        end
    end

    fill!(ws.pair_mean, zero(eltype(ws.pair_mean)))
    @inbounds for p in eachindex(sys.random_pair_indices)
        r, s = sys.random_pair_indices[p]
        ws.pair_mean[p] = sys.random_pair_multiplicities[p] * Σ[r, s]
    end
    ℒ.mul!(ws.mean, ws.hessian, ws.pair_mean, 0.5, 1.0)
    ws.effective_linear .= ws.linear
    ℒ.mul!(ws.linear_left, ws.linear, Σ)
    ℒ.mul!(ws.covariance, ws.linear_left, transpose(ws.linear), 1.0, 0.0)

    if sys.order == :third_order
        fill!(ws.pair_covariance, zero(eltype(ws.pair_covariance)))
        @inbounds for p in eachindex(sys.random_pair_indices), q in eachindex(sys.random_pair_indices)
            i, j = sys.random_pair_indices[p]
            k, l = sys.random_pair_indices[q]
            ws.pair_covariance[p, q] = sys.random_pair_multiplicities[p] *
                sys.random_pair_multiplicities[q] * (Σ[i, k] * Σ[j, l] + Σ[i, l] * Σ[j, k])
        end
        ℒ.mul!(ws.pair_product, ws.hessian, ws.pair_covariance)
        ℒ.mul!(ws.covariance, ws.pair_product, transpose(ws.hessian), 0.25, 1.0)
        fill!(ws.third_derivative, zero(eltype(ws.third_derivative)))
        @inbounds for p in eachindex(sys.random_triple_columns)
            ws.third_derivative[:, p] .= sys.S3[:, sys.random_triple_columns[p]]
        end
        fill!(ws.third_cross, zero(eltype(ws.third_cross)))
        fill!(ws.third_linear, zero(eltype(ws.third_linear)))
        @inbounds for p in eachindex(sys.random_triple_indices)
            i, j, k = sys.random_triple_indices[p]
            multiplicity = sys.random_triple_multiplicities[p]
            for r in 1:d
                ws.third_cross[r, p] = multiplicity * (Σ[r, i] * Σ[j, k] +
                    Σ[r, j] * Σ[i, k] + Σ[r, k] * Σ[i, j])
                ws.third_linear[p, r] = multiplicity * ((r == i) * Σ[j, k] +
                    (r == j) * Σ[i, k] + (r == k) * Σ[i, j])
            end
        end
        ℒ.mul!(ws.effective_linear, ws.third_derivative, ws.third_linear, 1 / 6, 1.0)
        fill!(ws.third_covariance, zero(eltype(ws.third_covariance)))
        @inbounds for p in eachindex(sys.random_triple_indices), q in eachindex(sys.random_triple_indices)
            i, j, k = sys.random_triple_indices[p]
            l, m, n = sys.random_triple_indices[q]
            indices = (i, j, k, l, m, n)
            ws.third_covariance[p, q] = sys.random_triple_multiplicities[p] *
                sys.random_triple_multiplicities[q] * ivashchenko_gaussian_sixth(indices, Σ)
        end
        ℒ.mul!(ws.third_work, ws.linear, ws.third_cross)
        ℒ.mul!(ws.third_output, ws.third_work, transpose(ws.third_derivative), 1 / 6, 0.0)
        @inbounds for j in 1:nout, i in 1:nout
            ws.covariance[i, j] += ws.third_output[i, j]
            ws.covariance[i, j] += ws.third_output[j, i]
        end
        ℒ.mul!(ws.third_work, ws.third_derivative, ws.third_covariance)
        ℒ.mul!(ws.third_output, ws.third_work, transpose(ws.third_derivative), 1 / 36, 0.0)
        ws.covariance .+= ws.third_output
    else
        @inbounds for output in 1:nout
            hessian = reshape(view(ws.quadratic_hessian, :, output), d, d)
            left = reshape(view(ws.quadratic_left, :, output), d, d)
            right = reshape(view(ws.quadratic_right, :, output), d, d)
            ℒ.mul!(left, hessian, Σ)
            copyto!(right, transpose(left))
        end
        # For symmetric T_r and Σ, the quadratic covariance is
        # 1/2 tr(T_r Σ T_s Σ). This avoids materialising the full pair covariance.
        ℒ.mul!(ws.covariance, transpose(ws.quadratic_left), ws.quadratic_right, 0.5, 1.0)
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
            effective_linear = copy(ws.effective_linear),
            hessian = ws.third_derivative === nothing ? nothing : copy(ws.hessian),
            pair_mean = copy(ws.pair_mean),
            pair_covariance = ws.third_derivative === nothing ? nothing : copy(ws.pair_covariance),
            covariance = copy(ws.covariance),
            third_derivative = ws.third_derivative === nothing ? nothing : copy(ws.third_derivative),
            third_covariance = ws.third_covariance === nothing ? nothing : copy(ws.third_covariance),
            third_cross = ws.third_cross === nothing ? nothing : copy(ws.third_cross),
            third_linear = ws.third_linear === nothing ? nothing : copy(ws.third_linear))
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
    nout = sys.nout
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

    # Reuse the filtering algebra across periods. The tape still receives a
    # copy of every value it needs, but the non-recording likelihood path no
    # longer allocates observation-sized matrices and state updates per period.
    predicted_mean = zeros(scalar_type, n_state)
    predicted_covariance = zeros(scalar_type, n_state, n_state)
    next_mean = zeros(scalar_type, n_state)
    next_covariance = zeros(scalar_type, n_state, n_state)
    output_mean = zeros(scalar_type, nout)
    output_covariance = zeros(scalar_type, nout, nout)
    observation_mean = zeros(scalar_type, n_obs)
    observation_covariance = zeros(scalar_type, n_obs, n_obs)
    cross_covariance = zeros(scalar_type, n_state, n_obs)
    observations = zeros(scalar_type, n_obs)
    innovation = zeros(scalar_type, n_obs)
    innovation_solve = zeros(scalar_type, n_obs)
    innovation_covariance = zeros(scalar_type, n_obs, n_obs)
    inverse_innovation_covariance = zeros(scalar_type, n_obs, n_obs)
    gain = zeros(scalar_type, n_state, n_obs)
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
        @inbounds for i in 1:n_state
            predicted_mean[i] = ws.mean[state_position[i]]
        end
        @inbounds for j in 1:n_state, i in 1:n_state
            predicted_covariance[i, j] = ws.covariance[state_position[i], state_position[j]]
        end
        copyto!(output_mean, ws.mean)
        copyto!(output_covariance, ws.covariance)
        idx = obs_idx_per_t[t]
        m = length(idx)

        if record
            predicted_means[t] = copy(predicted_mean)
            predicted_covariances[t] = copy(predicted_covariance)
            output_means[t] = copy(output_mean)
            output_covariances[t] = copy(output_covariance)
            transition = Matrix{scalar_type}(undef, n_state, sys.nPast)
            shock_loading = Matrix{scalar_type}(undef, n_state, sys.d - sys.nPast)
            @inbounds for j in 1:sys.nPast, i in 1:n_state
                transition[i, j] = ws.effective_linear[state_position[i], j]
            end
            @inbounds for j in 1:(sys.d - sys.nPast), i in 1:n_state
                shock_loading[i, j] = ws.effective_linear[state_position[i], sys.nPast + j]
            end
            transitions[t] = transition
            shock_loadings[t] = shock_loading
        end

        if m == 0
            if record
                innovations[t] = nothing
                inverse_innovation_covariances[t] = nothing
                gains[t] = nothing
                cross_covariances[t] = nothing
            end
            copyto!(next_mean, predicted_mean)
            copyto!(next_covariance, predicted_covariance)
            @inbounds for j in 1:n_state, i in 1:j-1
                value = (next_covariance[i, j] + next_covariance[j, i]) / 2
                next_covariance[i, j] = value
                next_covariance[j, i] = value
            end
            mean_state, next_mean = next_mean, mean_state
            covariance_state, next_covariance = next_covariance, covariance_state
            if record
                post_means[t] = copy(mean_state)
                post_covariances[t] = copy(covariance_state)
            end
            continue
        end

        @inbounds for i in 1:m
            observation_mean[i] = ws.mean[observation_position[idx[i]]]
            observations[i] = data_in_deviations[idx[i], t]
            for k in 1:n_state
                cross_covariance[k, i] = ws.covariance[state_position[k], observation_position[idx[i]]]
            end
        end
        @inbounds for j in 1:m, i in 1:m
            observation_covariance[i, j] = ws.covariance[observation_position[idx[i]], observation_position[idx[j]]]
            observation_covariance[i, j] += Hm[idx[i], idx[j]]
            innovation_covariance[i, j] = observation_covariance[i, j]
        end
        @inbounds for j in 1:m, i in 1:j-1
            value = (innovation_covariance[i, j] + innovation_covariance[j, i]) / 2
            innovation_covariance[i, j] = value
            innovation_covariance[j, i] = value
        end
        @inbounds for i in 1:m
            innovation[i] = observations[i] - observation_mean[i]
        end

        # A full observation period can use the concrete dense buffer directly;
        # LU on a SubArray takes a considerably more allocating generic path.
        factor = m == n_obs ? ℒ.lu(innovation_covariance, check = false) :
                               ℒ.lu(view(innovation_covariance, 1:m, 1:m), check = false)
        ℒ.issuccess(factor) || return record ? (convert(scalar_type, on_failure_loglikelihood), nothing) :
                                              convert(scalar_type, on_failure_loglikelihood)
        logabsdetF, signF = ℒ.logabsdet(factor)
        (primal(signF) > 0 && isfinite(primal(logabsdetF))) ||
            return record ? (convert(scalar_type, on_failure_loglikelihood), nothing) :
                            convert(scalar_type, on_failure_loglikelihood)

        inverse_view = m == n_obs ? inverse_innovation_covariance :
                                   view(inverse_innovation_covariance, 1:m, 1:m)
        fill!(inverse_view, zero(scalar_type))
        @inbounds for i in 1:m
            inverse_view[i, i] = one(scalar_type)
        end
        ℒ.ldiv!(factor, inverse_view)

        if t > presample_periods
            copyto!(view(innovation_solve, 1:m), view(innovation, 1:m))
            ℒ.ldiv!(factor, view(innovation_solve, 1:m))
            ll -= (ℒ.dot(view(innovation, 1:m), view(innovation_solve, 1:m)) + logabsdetF + m * log2pi) / 2
            isfinite(primal(ll)) || return record ? (convert(scalar_type, on_failure_loglikelihood), nothing) :
                                                   convert(scalar_type, on_failure_loglikelihood)
        end

        gain_view = view(gain, :, 1:m)
        ℒ.mul!(gain_view, view(cross_covariance, :, 1:m), inverse_view)
        if record
            innovations[t] = copy(view(innovation, 1:m))
            inverse_innovation_covariances[t] = copy(inverse_view)
            gains[t] = copy(gain_view)
            cross_covariances[t] = copy(view(cross_covariance, :, 1:m))
        end
        copyto!(next_mean, predicted_mean)
        ℒ.mul!(next_mean, gain_view, view(innovation, 1:m), one(scalar_type), one(scalar_type))
        copyto!(next_covariance, predicted_covariance)
        ℒ.mul!(next_covariance, gain_view, view(cross_covariance, :, 1:m)', -one(scalar_type), one(scalar_type))
        @inbounds for j in 1:n_state, i in 1:j-1
            value = (next_covariance[i, j] + next_covariance[j, i]) / 2
            next_covariance[i, j] = value
            next_covariance[j, i] = value
        end
        mean_state, next_mean = next_mean, mean_state
        covariance_state, next_covariance = next_covariance, covariance_state
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

function ivashchenko_quadratic_covariance_pullback!(gΣ, gS2, sys, covariance_bar, Σ, buffers)
    d, nout = sys.d, sys.nout
    T = sys.quadratic_hessian
    mixed = buffers.mixed
    gT = buffers.gT
    left = buffers.left
    right = buffers.right

    # If V_rs = 1/2 tr(T_r Σ T_s Σ), the symmetric covariance cotangent gives
    # dV/dT_r = Σ (Σ_s V̄_rs T_s) Σ and dV/dΣ = Σ_r T_r Σ (Σ_s V̄_rs T_s).
    ℒ.mul!(mixed, T, transpose(covariance_bar))
    @inbounds for output in 1:nout
        Tmix = reshape(view(mixed, :, output), d, d)
        gT_output = reshape(view(gT, :, output), d, d)
        ℒ.mul!(right, Σ, Tmix)
        ℒ.mul!(gT_output, right, Σ)
        T_output = reshape(view(T, :, output), d, d)
        ℒ.mul!(right, T_output, Σ)
        ℒ.mul!(left, right, Tmix)
        gΣ .+= left
    end

    @inbounds for p in eachindex(sys.random_pair_indices)
        r, s = sys.random_pair_indices[p]
        column = sys.random_pair_columns[p]
        if r == s
            for output in 1:nout
                gS2[output, column] += gT[r + (s - 1) * d, output]
            end
        else
            for output in 1:nout
                gS2[output, column] += gT[r + (s - 1) * d, output] +
                                        gT[s + (r - 1) * d, output]
            end
        end
    end
    return gΣ, gS2
end

function ivashchenko_rank_one_add!(matrix, left, right, scale)
    @inbounds for j in axes(matrix, 2), i in axes(matrix, 1)
        matrix[i, j] += scale * left[i] * right[j]
    end
    return matrix
end

function ivashchenko_second_order_moment_pullback!(sys, moment_tape, mean_bar,
                                                   covariance_bar, buffers)
    nout, d = sys.nout, sys.d
    Σ = moment_tape.covariance_input
    v = moment_tape.vbar
    L = moment_tape.linear
    pair_mean = moment_tape.pair_mean
    covariance_bar_symmetric = buffers.covariance_bar_symmetric
    gΣ, gv, gS1, gS2 = buffers.gΣ, buffers.gv, buffers.gS1, buffers.gS2
    gL = buffers.gL

    fill!(gΣ, zero(eltype(gΣ)))
    fill!(gv, zero(eltype(gv)))
    fill!(gS1, zero(eltype(gS1)))
    fill!(gS2, zero(eltype(gS2)))
    @inbounds for j in 1:nout, i in 1:j
        value = (covariance_bar[i, j] + covariance_bar[j, i]) / 2
        covariance_bar_symmetric[i, j] = value
        covariance_bar_symmetric[j, i] = value
    end

    ℒ.mul!(buffers.tmp_noutd, covariance_bar_symmetric, L)
    ℒ.mul!(gL, buffers.tmp_noutd, Σ, 2.0, 0.0)
    ℒ.mul!(buffers.tmp_dnout, transpose(L), covariance_bar_symmetric)
    ℒ.mul!(gΣ, buffers.tmp_dnout, L, 1.0, 0.0)
    ivashchenko_quadratic_covariance_pullback!(gΣ, gS2, sys,
                                               covariance_bar_symmetric, Σ,
                                               buffers)

    compressed_kron²_power!(buffers.pair_value, v)
    ivashchenko_rank_one_add!(gS1, mean_bar, v, 1.0)
    ivashchenko_rank_one_add!(gS2, mean_bar, buffers.pair_value, 0.5)
    ℒ.mul!(gv, transpose(sys.S1), mean_bar)
    ℒ.mul!(buffers.pair_cotangent, transpose(sys.S2), mean_bar, 0.5, 0.0)
    compressed_kron²_power_vjp!(buffers.first_bar, buffers.pair_cotangent, v)
    gv .+= buffers.first_bar

    @inbounds for p in eachindex(sys.random_pair_indices)
        r, s = sys.random_pair_indices[p]
        pair_bar = sys.random_pair_multiplicities[p] * buffers.pair_cotangent[p]
        gΣ[r, s] += pair_bar
        column = sys.random_pair_columns[p]
        for output in 1:nout
            gS2[output, column] += mean_bar[output] * pair_mean[p] / 2
        end
    end

    @inbounds for r in 1:d
        i = sys.random_indices[r]
        fill!(buffers.basis, zero(eltype(buffers.basis)))
        buffers.basis[i] = one(eltype(buffers.basis))
        compressed_kron²!(buffers.pair_value, v, buffers.basis)
        gS1[:, i] .+= view(gL, :, r)
        ivashchenko_rank_one_add!(gS2, view(gL, :, r), buffers.pair_value, 0.5)
        ℒ.mul!(buffers.pair_cotangent, transpose(sys.S2), view(gL, :, r), 0.5, 0.0)
        compressed_kron²_vjp!(buffers.first_bar, buffers.second_bar,
                              buffers.pair_cotangent, v, buffers.basis)
        gv .+= buffers.first_bar
    end

    @inbounds for j in 1:d, i in 1:j-1
        value = (gΣ[i, j] + gΣ[j, i]) / 2
        gΣ[i, j] = value
        gΣ[j, i] = value
    end
    copyto!(buffers.mean_result, view(gv, 1:sys.nPast))
    copyto!(buffers.covariance_result, view(gΣ, 1:sys.nPast, 1:sys.nPast))
    return buffers.mean_result, buffers.covariance_result, gS1, gS2, nothing
end

function ivashchenko_polynomial_moments_pullback(sys, moment_tape, mean_bar, covariance_bar;
                                                 quadratic_buffers = nothing)
    moment_tape.third_derivative === nothing &&
        return ivashchenko_second_order_moment_pullback!(sys, moment_tape,
                                                         mean_bar, covariance_bar,
                                                         quadratic_buffers)
    nout, d, dv = sys.nout, sys.d, sys.dv
    Σ = moment_tape.covariance_input
    v = moment_tape.vbar
    L = moment_tape.linear
    K = moment_tape.hessian === nothing ? sys.random_hessian : moment_tape.hessian
    third = moment_tape.third_derivative
    pair_mean = moment_tape.pair_mean
    gΣ = zeros(eltype(Σ), d, d)
    gv = zeros(eltype(v), dv)
    gS1 = zeros(eltype(sys.S1), nout, dv)
    gS2 = zeros(eltype(sys.S2), nout, sys.n_pair)
    gS3 = third === nothing ? nothing : zeros(eltype(third), nout, length(sys.triple_indices))

    covariance_bar_symmetric = (covariance_bar + covariance_bar') / 2
    gL = 2 * covariance_bar_symmetric * L * Σ
    gΣ .+= L' * covariance_bar_symmetric * L
    gpair_mean = K' * mean_bar / 2

    if third !== nothing
        pair_covariance = moment_tape.pair_covariance
        gK = covariance_bar_symmetric * K * pair_covariance / 2
        gG2 = K' * covariance_bar_symmetric * K / 4
        gK .+= mean_bar * pair_mean' / 2
        third_covariance = moment_tape.third_covariance
        third_cross = moment_tape.third_cross
        gL .+= covariance_bar_symmetric * third * third_cross' / 3
        gT = covariance_bar_symmetric * L * third_cross / 3 +
             covariance_bar_symmetric * third * third_covariance / 18
        gR = L' * covariance_bar_symmetric * third / 3
        gG3 = third' * covariance_bar_symmetric * third / 36

        @inbounds for p in eachindex(sys.random_pair_indices)
            i, j = sys.random_pair_indices[p]
            gΣ[i, j] += sys.random_pair_multiplicities[p] * gpair_mean[p]
        end
        @inbounds for p in eachindex(sys.random_pair_indices), q in eachindex(sys.random_pair_indices)
            i, j = sys.random_pair_indices[p]
            k, l = sys.random_pair_indices[q]
            weight = gG2[p, q] * sys.random_pair_multiplicities[p] *
                     sys.random_pair_multiplicities[q]
            gΣ[i, k] += weight * Σ[j, l]
            gΣ[j, l] += weight * Σ[i, k]
            gΣ[i, l] += weight * Σ[j, k]
            gΣ[j, k] += weight * Σ[i, l]
        end
        @inbounds for p in eachindex(sys.random_triple_indices), r in 1:d
            i, j, k = sys.random_triple_indices[p]
            weight = gR[r, p] * sys.random_triple_multiplicities[p]
            gΣ[r, i] += weight * (Σ[j, k])
            gΣ[j, k] += weight * Σ[r, i]
            gΣ[r, j] += weight * Σ[i, k]
            gΣ[i, k] += weight * Σ[r, j]
            gΣ[r, k] += weight * Σ[i, j]
            gΣ[i, j] += weight * Σ[r, k]
        end
        @inbounds for p in eachindex(sys.random_triple_indices), q in eachindex(sys.random_triple_indices)
            i, j, k = sys.random_triple_indices[p]
            l, m, n = sys.random_triple_indices[q]
            weight = gG3[p, q] * sys.random_triple_multiplicities[p] *
                     sys.random_triple_multiplicities[q]
            ivashchenko_gaussian_sixth_pullback!(gΣ, weight, (i, j, k, l, m, n), Σ)
        end
    else
        @inbounds for p in eachindex(sys.random_pair_indices)
            i, j = sys.random_pair_indices[p]
            gΣ[i, j] += sys.random_pair_multiplicities[p] * gpair_mean[p]
            gS2[:, sys.random_pair_columns[p]] .+= mean_bar * pair_mean[p] / 2
        end
        quadratic_buffers === nothing && (quadratic_buffers = (; mixed = zeros(eltype(Σ), d * d, nout),
                                             gT = zeros(eltype(Σ), d * d, nout),
                                             left = zeros(eltype(Σ), d, d),
                                             right = zeros(eltype(Σ), d, d)))
        ivashchenko_quadratic_covariance_pullback!(gΣ, gS2, sys,
                                                   covariance_bar_symmetric, Σ,
                                                   quadratic_buffers)
    end

    pair_cotangent = zeros(eltype(v), sys.n_pair)
    triple_cotangent = third === nothing ? nothing : zeros(eltype(v), length(sys.triple_indices))
    first_bar = zeros(eltype(v), dv)
    second_bar = zeros(eltype(v), dv)
    third_bar = zeros(eltype(v), dv)
    basis = zeros(eltype(v), dv)
    second_basis = zeros(eltype(v), dv)
    pair_value = zeros(eltype(v), sys.n_pair)
    triple_value = third === nothing ? nothing : zeros(eltype(v), length(sys.triple_indices))

    compressed_kron²_power!(pair_value, v)
    gS1 .+= mean_bar * v'
    gv .+= sys.S1' * mean_bar
    gS2 .+= mean_bar * pair_value' / 2
    pair_cotangent .= sys.S2' * mean_bar / 2
    compressed_kron²_power_vjp!(first_bar, pair_cotangent, v)
    gv .+= first_bar
    if third !== nothing
        compressed_kron³_power!(triple_value, v)
        gS3 .+= mean_bar * triple_value' / 6
        triple_cotangent .= sys.S3' * mean_bar / 6
        compressed_kron³_power_vjp!(first_bar, triple_cotangent, v)
        gv .+= first_bar
    end

    @inbounds for r in 1:d
        i = sys.random_indices[r]
        fill!(basis, zero(eltype(basis)))
        basis[i] = one(eltype(basis))
        compressed_kron²!(pair_value, v, basis)
        gS1[:, i] .+= gL[:, r]
        gS2 .+= gL[:, r] * (pair_value / 2)'
        pair_cotangent .= sys.S2' * gL[:, r] / 2
        compressed_kron²_vjp!(first_bar, second_bar, pair_cotangent, v, basis)
        gv .+= first_bar
        if third !== nothing
            compressed_kron³!(triple_value, v, v, basis)
            gS3 .+= gL[:, r] * (triple_value / 2)'
            triple_cotangent .= sys.S3' * gL[:, r] / 2
            compressed_kron³_vjp!(first_bar, second_bar, third_bar,
                                  triple_cotangent, v, v, basis)
            gv .+= first_bar + second_bar
        end
    end

    if third !== nothing
        @inbounds for p in eachindex(sys.random_pair_indices)
            r, s = sys.random_pair_indices[p]
            global_column = sys.random_pair_columns[p]
            gS2[:, global_column] .+= gK[:, p]
            fill!(basis, zero(eltype(basis)))
            fill!(second_basis, zero(eltype(second_basis)))
            basis[sys.random_indices[r]] = one(eltype(basis))
            second_basis[sys.random_indices[s]] = one(eltype(second_basis))
            compressed_kron³!(triple_value, v, basis, second_basis)
            gS3 .+= gK[:, p] * triple_value'
            triple_cotangent .= sys.S3' * gK[:, p]
            compressed_kron³_vjp!(first_bar, second_bar, third_bar,
                                  triple_cotangent, v, basis, second_basis)
            gv .+= first_bar
        end
    end

    if third !== nothing
        @inbounds for p in eachindex(sys.random_triple_columns)
            gS3[:, sys.random_triple_columns[p]] .+= gT[:, p]
        end
    end

    gΣ .= (gΣ + gΣ') / 2
    gmean = copy(view(gv, 1:sys.nPast))
    gcovariance = copy(view(gΣ, 1:sys.nPast, 1:sys.nPast))
    return gmean, gcovariance, gS1, gS2, gS3
end

function ivashchenko_solution_matrix_pullback(sys, gS1, gS2, gS3, solution_matrices)
    size(solution_matrices[2], 2) == sys.n_pair ||
        throw(DimensionMismatch("Ivashchenko quadratic pullback requires compressed solution columns."))
    if gS3 === nothing
        return gS1, gS2, nothing
    end
    size(solution_matrices[3], 2) == length(sys.triple_indices) ||
        throw(DimensionMismatch("Ivashchenko cubic pullback requires compressed solution columns."))
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
    gS2 = zeros(scalar_type, size(sys.S2))
    gS3 = sys.S3 === nothing ? nothing : zeros(scalar_type, size(sys.S3))
    quadratic_buffers = sys.order == :second_order ?
        (; covariance_bar_symmetric = zeros(scalar_type, sys.nout, sys.nout),
           gΣ = zeros(scalar_type, sys.d, sys.d),
           gv = zeros(scalar_type, sys.dv),
           gS1 = zeros(scalar_type, sys.nout, sys.dv),
           gS2 = zeros(scalar_type, sys.nout, sys.n_pair),
           gL = zeros(scalar_type, sys.nout, sys.d),
           tmp_noutd = zeros(scalar_type, sys.nout, sys.d),
           tmp_dnout = zeros(scalar_type, sys.d, sys.nout),
           pair_cotangent = zeros(scalar_type, sys.n_pair),
           first_bar = zeros(scalar_type, sys.dv),
           second_bar = zeros(scalar_type, sys.dv),
           basis = zeros(scalar_type, sys.dv),
           pair_value = zeros(scalar_type, sys.n_pair),
           mixed = zeros(scalar_type, sys.d * sys.d, sys.nout),
           gT = zeros(scalar_type, sys.d * sys.d, sys.nout),
           left = zeros(scalar_type, sys.d, sys.d),
           right = zeros(scalar_type, sys.d, sys.d),
           mean_result = zeros(scalar_type, sys.nPast),
           covariance_result = zeros(scalar_type, sys.nPast, sys.nPast)) : nothing

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
            # The forward covariance update is explicitly symmetrised; only
            # the symmetric part of its cotangent can reach the pre-update
            # covariance and gain.
            covariance_bar_post = (covariance_bar + covariance_bar') / 2
            gain_bar = mean_bar * innovation'
            innovation_bar = gain' * mean_bar
            gain_bar .-= covariance_bar_post * cross
            cross_bar = -covariance_bar_post' * gain
            inverse_covariance_bar = cross' * gain_bar
            cross_bar .+= gain_bar * invF'
            if t > tape.presample_periods
                innovation_bar .-= scale * (invF * innovation)
            end
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

        mean_bar, covariance_bar, local_S1, local_S2, local_S3 =
            ivashchenko_polynomial_moments_pullback(sys, tape.moment_tapes[t],
                                                    mean_output_bar, covariance_output_bar;
                                                    quadratic_buffers = quadratic_buffers)
        gS1 .+= local_S1
        gS2 .+= local_S2
        if gS3 !== nothing
            gS3 .+= local_S3
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
            initialization_mean_bar, initialization_covariance_bar, local_S1, local_S2, local_S3 =
                ivashchenko_polynomial_moments_pullback(sys, iteration.moment_tape,
                                                        output_mean_bar, output_covariance_bar;
                                                        quadratic_buffers = quadratic_buffers)
            gS1 .+= local_S1
            gS2 .+= local_S2
            if gS3 !== nothing
                gS3 .+= local_S3
            end
        end
        mean_bar = initialization_mean_bar
        covariance_bar = initialization_covariance_bar
        initial_A_bar, initial_B_bar = ivashchenko_lyapunov_pullback(
            sys.A, sys.B, tape.initialization_tape.linear_covariance, covariance_bar;
            lyapunov_algorithm = lyapunov_algorithm)
        gS1[sys.state_position, 1:sys.nPast] .+= initial_A_bar
        gS1[sys.state_position, sys.nPast + 2:sys.dv] .+= initial_B_bar
    end

    local_S1, local_S2, local_S3 = ivashchenko_solution_matrix_pullback(
        sys, gS1, gS2, gS3, solution_matrices)
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
        predicted_factor = ℒ.cholesky(tape.predicted_covariances[t + 1], check = false)
        ℒ.issuccess(predicted_factor) || error(
            "Ivashchenko smoother covariance factorization failed at period $(t + 1).")
        ℒ.rdiv!(cross, predicted_factor)
        smoother_gain = cross
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
        predicted_factor = ℒ.cholesky(pred_covariance, check = false)
        ℒ.issuccess(predicted_factor) || error("Ivashchenko smoother covariance factorization failed at period $t.")
        state_delta = smoothed_means[t] - tape.predicted_means[t]
        state_regression = copy(tape.output_covariances[t][ :, tape.state_position])
        ℒ.rdiv!(state_regression, predicted_factor)
        variables[:, t] .= tape.output_means[t] + state_regression * state_delta
        variables[tape.state_position, t] .= smoothed_means[t]
        standard_deviations[:, t] .= sqrt.(abs.(ℒ.diag(tape.output_covariances[t] -
            state_regression * pred_covariance * state_regression')))
        standard_deviations[tape.state_position, t] .= sqrt.(abs.(ℒ.diag(smoothed_covariances[t])))

        shock_regression = copy(tape.shock_loadings[t]')
        ℒ.rdiv!(shock_regression, predicted_factor)
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
