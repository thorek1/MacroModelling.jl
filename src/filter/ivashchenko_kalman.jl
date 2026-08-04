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
const IVASHCHENKO_GAUSSIAN_CLOSURES = (:exact, :linearized, :diagonal)

function validate_ivashchenko_gaussian_closure(sys, gaussian_closure::Symbol)
    gaussian_closure ∈ IVASHCHENKO_GAUSSIAN_CLOSURES ||
        throw(ArgumentError("Unsupported Ivashchenko Gaussian closure `:$(gaussian_closure)`. Choose :exact, :linearized, or :diagonal."))
    gaussian_closure == :diagonal && sys.order != :second_order &&
        throw(ArgumentError("The :diagonal Ivashchenko Gaussian closure is currently implemented only for second-order systems."))
    return gaussian_closure
end

compressed_pair_index(i::Int, j::Int, n::Int) = begin
    i, j = max(i, j), min(i, j)
    (i - 1) * i ÷ 2 + j
end

@inline function compressed_triple_index(i::Int, j::Int, k::Int, n::Int)
    if i < j
        i, j = j, i
    end
    if i < k
        i, k = k, i
    end
    if j < k
        j, k = k, j
    end
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

function build_ivashchenko_kalman_system_from_constants(cons, 𝐒,
                                                        observables_index::Vector{Int},
                                                        order::Symbol;
                                                        keep_all_rows::Bool = true)
    T = cons.post_model_macro
    nVars, nPast, nExo = T.nVars, T.nPast_not_future_and_mixed, T.nExo
    past = collect(T.past_not_future_and_mixed_idx)
    # The estimate API needs every model-variable row.  A likelihood pass only
    # needs the state recursion and the requested observables, so it can use the
    # exact smaller output map without changing the Gaussian closure.
    output_rows = keep_all_rows ? collect(1:nVars) :
                  sort!(unique(vcat(past, observables_index)))
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

build_ivashchenko_kalman_system(𝓂::ℳ, 𝐒, oi::Vector{Int}, order::Symbol;
                                keep_all_rows::Bool = true) =
    build_ivashchenko_kalman_system_from_constants(𝓂.constants, 𝐒, oi, order;
                                                   keep_all_rows = keep_all_rows)

const IVASHCHENKO_PRUNED_THIRD_MAX_BYTES = 768 * 1024^2

@inline function pruned_ivashchenko_effective_index(stage::Int, i::Int,
                                                     nPast::Int, nStages::Int,
                                                     nExo::Int)
    if i <= nPast
        return (stage - 1) * nPast + i
    elseif i == nPast + 1
        return nStages * nPast + 1
    end
    return nStages * nPast + 1 + (i - nPast - 1)
end

function pruned_ivashchenko_initial_mean(sys, state)
    stages = sys.nStages
    past = sys.original_past
    if state isa AbstractVector{<:AbstractVector}
        length(state) == stages ||
            throw(DimensionMismatch("The pruned Ivashchenko state must contain $stages stage vectors."))
        return vcat((state[stage][past] for stage in 1:stages)...)
    end
    length(state) == sys.original_nVars ||
        throw(DimensionMismatch("The pruned Ivashchenko initial state must have $(sys.original_nVars) entries."))
    result = [zeros(eltype(state), sys.original_nVars) for _ in 1:stages]
    result[1][past] .= state[past]
    return vcat((result[stage][past] for stage in 1:stages)...)
end

function build_pruned_ivashchenko_kalman_system_from_constants(cons, 𝐒,
                                                               observables_index::Vector{Int},
                                                               order::Symbol;
                                                               keep_all_rows::Bool = false)
    order ∈ (:pruned_second_order, :pruned_third_order) ||
        throw(ArgumentError("The pruned Ivashchenko builder requires a pruned second- or third-order algorithm."))
    T = cons.post_model_macro
    original_nVars = T.nVars
    original_nPast = T.nPast_not_future_and_mixed
    nExo = T.nExo
    original_past = collect(T.past_not_future_and_mixed_idx)
    nStages = order == :pruned_second_order ? 2 : 3
    nObs = length(observables_index)
    original_dv = original_nPast + 1 + nExo
    required_order = nStages == 3 ? 3 : 2
    length(𝐒) >= required_order ||
        throw(DimensionMismatch("The $(order) solution must provide the required perturbation matrices."))

    scalar_type = nStages == 3 ?
        promote_type(eltype(𝐒[1]), eltype(𝐒[2]), eltype(𝐒[3])) :
        promote_type(eltype(𝐒[1]), eltype(𝐒[2]))
    original_S1 = Matrix{scalar_type}(𝐒[1])
    original_S2 = Matrix{scalar_type}(𝐒[2])
    size(original_S1) == (original_nVars, original_dv) ||
        throw(DimensionMismatch("The pruned S₁ matrix has the wrong dimensions."))
    size(original_S2, 1) == original_nVars &&
    size(original_S2, 2) == original_dv * (original_dv + 1) ÷ 2 ||
        throw(DimensionMismatch("The pruned S₂ matrix has the wrong compressed dimensions."))
    original_S3 = nothing
    if nStages == 3
        original_S3 = Matrix{scalar_type}(𝐒[3])
        size(original_S3, 1) == original_nVars &&
        size(original_S3, 2) == original_dv * (original_dv + 1) * (original_dv + 2) ÷ 6 ||
            throw(DimensionMismatch("The pruned S₃ matrix has the wrong compressed dimensions."))
    end

    # Only rows needed by the state recursion and the requested observables are
    # retained.  The remaining model rows cannot affect this likelihood pass.
    keep_rows = keep_all_rows ? collect(1:original_nVars) :
                sort!(unique(vcat(original_past, observables_index)))
    nKeep = length(keep_rows)
    past_local = [findfirst(==(row), keep_rows) for row in original_past]
    state_output_positions = Int[]
    for stage in 1:nStages
        append!(state_output_positions, (stage - 1) * nKeep .+ past_local)
    end
    measurement_rows = nStages * nKeep .+ (1:nObs)
    flat_nPast = nStages * original_nPast
    effective_dv = flat_nPast + 1 + nExo
    effective_nout = nStages * nKeep + nObs
    effective_pair_count = effective_dv * (effective_dv + 1) ÷ 2

    if nStages == 3
        effective_triple_count = effective_dv * (effective_dv + 1) * (effective_dv + 2) ÷ 6
        d = effective_dv - 1
        n_pair = d * (d + 1) ÷ 2
        estimated_elements = 5 * effective_nout * effective_triple_count +
                             6 * d * effective_triple_count + n_pair * n_pair +
                             effective_nout * (effective_pair_count + n_pair)
        estimated_bytes = estimated_elements * sizeof(scalar_type)
        estimated_bytes <= IVASHCHENKO_PRUNED_THIRD_MAX_BYTES ||
            throw(ArgumentError("The pruned third-order Ivashchenko closure would require about " *
                                string(round(estimated_bytes / 1024^2, digits = 1)) *
                                " MiB in compressed workspaces for this model. " *
                                "This exceeds the configured " *
                                string(round(IVASHCHENKO_PRUNED_THIRD_MAX_BYTES / 1024^2)) *
                                " MiB safety limit; use Kollmann's cubic filter when its dimension limit permits it, " *
                                "or use inversion/particle filtering for this model."))
    end

    effective_S1 = zeros(scalar_type, effective_nout, effective_dv)
    effective_S2 = zeros(scalar_type, effective_nout, effective_pair_count)
    effective_S3 = nStages == 3 ?
        zeros(scalar_type, effective_nout,
              effective_dv * (effective_dv + 1) * (effective_dv + 2) ÷ 6) : nothing

    for stage in 1:nStages
        output_offset = (stage - 1) * nKeep
        for (row_position_local, row) in enumerate(keep_rows)
            if stage == 1
                @inbounds for i in 1:original_dv
                    effective_S1[output_offset + row_position_local,
                                 pruned_ivashchenko_effective_index(1, i, original_nPast,
                                                                     nStages, nExo)] = original_S1[row, i]
                end
            else
                @inbounds for i in 1:original_nPast
                    effective_S1[output_offset + row_position_local,
                                 pruned_ivashchenko_effective_index(stage, i, original_nPast,
                                                                     nStages, nExo)] = original_S1[row, i]
                end
            end
        end
    end

    @inbounds for i in 1:original_dv, j in 1:i
        original_column = compressed_pair_index(i, j, original_dv)
        effective_column = compressed_pair_index(
            pruned_ivashchenko_effective_index(1, i, original_nPast, nStages, nExo),
            pruned_ivashchenko_effective_index(1, j, original_nPast, nStages, nExo),
            effective_dv)
        for (row_position_local, row) in enumerate(keep_rows)
            effective_S2[nKeep + row_position_local, effective_column] =
                original_S2[row, original_column]
        end
    end

    # The line above writes the second-stage block because it starts at nKeep;
    # keep the explicit offsets below for the mixed third-stage term and to make
    # the two pruned recursions visibly separate.
    if nStages == 3
        @inbounds for i in 1:original_dv, j in 1:i
            original_column = compressed_pair_index(i, j, original_dv)
            if j <= original_nPast
                effective_column = compressed_pair_index(
                    pruned_ivashchenko_effective_index(1, i, original_nPast, nStages, nExo),
                    pruned_ivashchenko_effective_index(2, j, original_nPast, nStages, nExo),
                    effective_dv)
                for (row_position_local, row) in enumerate(keep_rows)
                    effective_S2[2 * nKeep + row_position_local, effective_column] += original_S2[row, original_column]
                end
            end
            if i != j && i <= original_nPast
                effective_column = compressed_pair_index(
                    pruned_ivashchenko_effective_index(1, j, original_nPast, nStages, nExo),
                    pruned_ivashchenko_effective_index(2, i, original_nPast, nStages, nExo),
                    effective_dv)
                for (row_position_local, row) in enumerate(keep_rows)
                    effective_S2[2 * nKeep + row_position_local, effective_column] += original_S2[row, original_column]
                end
            end
        end

        @inbounds for i in 1:original_dv, j in 1:i, k in 1:j
            original_column = compressed_triple_index(i, j, k, original_dv)
            effective_column = compressed_triple_index(
                pruned_ivashchenko_effective_index(1, i, original_nPast, nStages, nExo),
                pruned_ivashchenko_effective_index(1, j, original_nPast, nStages, nExo),
                pruned_ivashchenko_effective_index(1, k, original_nPast, nStages, nExo),
                effective_dv)
            for (row_position_local, row) in enumerate(keep_rows)
                effective_S3[2 * nKeep + row_position_local, effective_column] = original_S3[row, original_column]
            end
        end
    end

    # The observation map is the sum of the pruned stages.  Appending these
    # rows gives the generic Ivashchenko pass the same measurement covariance
    # interface as its unpruned system without adding a physical state block.
    @inbounds for (observation, row) in enumerate(observables_index)
        row_position_local = findfirst(==(row), keep_rows)
        measurement_row = measurement_rows[observation]
        for stage in 1:nStages
            source_row = (stage - 1) * nKeep + row_position_local
            effective_S1[measurement_row, :] .+= effective_S1[source_row, :]
            effective_S2[measurement_row, :] .+= effective_S2[source_row, :]
            if effective_S3 !== nothing
                effective_S3[measurement_row, :] .+= effective_S3[source_row, :]
            end
        end
    end

    fake_constants = (; post_model_macro = (; nVars = effective_nout,
                                             nPast_not_future_and_mixed = flat_nPast,
                                             nExo,
                                             past_not_future_and_mixed_idx = state_output_positions))
    effective_solution = effective_S3 === nothing ?
        [effective_S1, effective_S2] : [effective_S1, effective_S2, effective_S3]
    base = build_ivashchenko_kalman_system_from_constants(
        fake_constants, effective_solution, collect(measurement_rows),
        nStages == 2 ? :second_order : :third_order)
    return merge(base, (; past = collect(1:flat_nPast), pruned = true, nStages,
                        original_nVars, original_nPast,
                        original_past, original_observables = copy(observables_index),
                        keep_rows, nKeep, state_output_positions, measurement_rows,
                        effective_solution, effective_dv))
end

build_pruned_ivashchenko_kalman_system(𝓂::ℳ, 𝐒, oi::Vector{Int}, order::Symbol;
                                       keep_all_rows::Bool = false) =
    build_pruned_ivashchenko_kalman_system_from_constants(𝓂.constants, 𝐒, oi, order;
                                                          keep_all_rows = keep_all_rows)

function pruned_ivashchenko_solution_matrix_pullback(sys, effective_bar, solution_matrices)
    original_bar = [zeros(eltype(effective_bar[1]), size(solution_matrices[1])),
                    zeros(eltype(effective_bar[1]), size(solution_matrices[2]))]
    if length(solution_matrices) == 3
        push!(original_bar, zeros(eltype(effective_bar[1]), size(solution_matrices[3])))
    end
    row_bar_s1 = copy(effective_bar[1])
    row_bar_s2 = copy(effective_bar[2])
    row_bar_s3 = length(effective_bar) == 3 ? copy(effective_bar[3]) : nothing
    @inbounds for (observation, row) in enumerate(sys.original_observables)
        row_position_local = findfirst(==(row), sys.keep_rows)
        measurement_row = sys.measurement_rows[observation]
        for stage in 1:sys.nStages
            source_row = (stage - 1) * sys.nKeep + row_position_local
            row_bar_s1[source_row, :] .+= row_bar_s1[measurement_row, :]
            row_bar_s2[source_row, :] .+= row_bar_s2[measurement_row, :]
            if row_bar_s3 !== nothing
                row_bar_s3[source_row, :] .+= row_bar_s3[measurement_row, :]
            end
        end
    end

    original_dv = sys.original_nPast + 1 + sys.nExo
    @inbounds for stage in 1:sys.nStages
        output_offset = (stage - 1) * sys.nKeep
        for (row_position_local, row) in enumerate(sys.keep_rows)
            source_row = output_offset + row_position_local
            if stage == 1
                for i in 1:original_dv
                    effective_column = pruned_ivashchenko_effective_index(
                        1, i, sys.original_nPast, sys.nStages, sys.nExo)
                    original_bar[1][row, i] += row_bar_s1[source_row, effective_column]
                end
            else
                for i in 1:sys.original_nPast
                    effective_column = pruned_ivashchenko_effective_index(
                        stage, i, sys.original_nPast, sys.nStages, sys.nExo)
                    original_bar[1][row, i] += row_bar_s1[source_row, effective_column]
                end
            end

            if stage == 2
                for i in 1:original_dv, j in 1:i
                    original_column = compressed_pair_index(i, j, original_dv)
                    effective_column = compressed_pair_index(
                        pruned_ivashchenko_effective_index(1, i, sys.original_nPast,
                                                           sys.nStages, sys.nExo),
                        pruned_ivashchenko_effective_index(1, j, sys.original_nPast,
                                                           sys.nStages, sys.nExo),
                        sys.effective_dv)
                    original_bar[2][row, original_column] += row_bar_s2[source_row, effective_column]
                end
            elseif stage == 3
                for i in 1:original_dv, j in 1:i
                    original_column = compressed_pair_index(i, j, original_dv)
                    if j <= sys.original_nPast
                        effective_column = compressed_pair_index(
                            pruned_ivashchenko_effective_index(1, i, sys.original_nPast,
                                                               sys.nStages, sys.nExo),
                            pruned_ivashchenko_effective_index(2, j, sys.original_nPast,
                                                               sys.nStages, sys.nExo),
                            sys.effective_dv)
                        original_bar[2][row, original_column] += row_bar_s2[source_row, effective_column]
                    end
                    if i != j && i <= sys.original_nPast
                        effective_column = compressed_pair_index(
                            pruned_ivashchenko_effective_index(1, j, sys.original_nPast,
                                                               sys.nStages, sys.nExo),
                            pruned_ivashchenko_effective_index(2, i, sys.original_nPast,
                                                               sys.nStages, sys.nExo),
                            sys.effective_dv)
                        original_bar[2][row, original_column] += row_bar_s2[source_row, effective_column]
                    end
                end
                for i in 1:original_dv, j in 1:i, k in 1:j
                    original_column = compressed_triple_index(i, j, k, original_dv)
                    effective_column = compressed_triple_index(
                        pruned_ivashchenko_effective_index(1, i, sys.original_nPast,
                                                           sys.nStages, sys.nExo),
                        pruned_ivashchenko_effective_index(1, j, sys.original_nPast,
                                                           sys.nStages, sys.nExo),
                        pruned_ivashchenko_effective_index(1, k, sys.original_nPast,
                                                           sys.nStages, sys.nExo),
                        sys.effective_dv)
                    original_bar[3][row, original_column] += row_bar_s3[source_row, effective_column]
                end
            end
        end
    end
    return original_bar
end

function pruned_ivashchenko_state_pullback(sys, flat_bar, state)
    if state isa AbstractVector{<:AbstractVector}
        result = [zeros(eltype(flat_bar), size(stage_state)) for stage_state in state]
        @inbounds for stage in 1:sys.nStages
            range = (stage - 1) * sys.original_nPast + 1:stage * sys.original_nPast
            result[stage][sys.original_past] .= flat_bar[range]
        end
        return result
    end
    result = zeros(eltype(flat_bar), length(state))
    result[sys.original_past] .= view(flat_bar, 1:sys.original_nPast)
    return result
end

function ivashchenko_kalman_workspace(sys, scalar_type::Type;
                                      gaussian_closure::Symbol = :exact)
    validate_ivashchenko_gaussian_closure(sys, gaussian_closure)
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
            quadratic_left = third || gaussian_closure == :linearized ? nothing :
                             zeros(scalar_type, d * d, nout),
            quadratic_right = third || gaussian_closure != :exact ? nothing :
                              zeros(scalar_type, d * d, nout),
            pair_mean = zeros(scalar_type, n_pair),
            pair_covariance = third && gaussian_closure == :exact ?
                              zeros(scalar_type, n_pair, n_pair) : nothing,
            pair_product = third && gaussian_closure == :exact ?
                           zeros(scalar_type, nout, n_pair) : nothing,
            covariance = zeros(scalar_type, nout, nout),
            third_derivative = third ? zeros(scalar_type, nout, n_triple) : nothing,
            third_cross = third && gaussian_closure == :exact ?
                          zeros(scalar_type, d, n_triple) : nothing,
            third_linear = third ? zeros(scalar_type, n_triple, d) : nothing,
            third_work = third && gaussian_closure == :exact ?
                         zeros(scalar_type, nout, n_triple) : nothing,
            third_output = third && gaussian_closure == :exact ?
                           zeros(scalar_type, nout, nout) : nothing,
            third_mode_source = third && gaussian_closure == :exact ?
                                zeros(scalar_type, d, n_pair) : nothing,
            third_mode_one = third && gaussian_closure == :exact ?
                             zeros(scalar_type, d, n_pair) : nothing,
            third_mode_two = third && gaussian_closure == :exact ?
                             zeros(scalar_type, n_pair, d) : nothing,
            third_mode_three = third && gaussian_closure == :exact ?
                               zeros(scalar_type, n_pair, d) : nothing,
            third_cross_compressed = third && gaussian_closure == :exact ?
                                     zeros(scalar_type, n_triple, nout) : nothing,
            third_linear_output = third && gaussian_closure == :exact ?
                                  zeros(scalar_type, nout, d) : nothing,
            third_linear_left = third && gaussian_closure == :exact ?
                                zeros(scalar_type, nout, d) : nothing,
            pair_scratch = zeros(scalar_type, n_pair),
            pair_jacobian = zeros(scalar_type, n_pair, d),
            triple_scratch = third ? zeros(scalar_type, n_triple) : nothing,
            triple_jacobian = third ? zeros(scalar_type, n_triple, d) : nothing,
            basis = zeros(scalar_type, sys.dv),
            second_basis = zeros(scalar_type, sys.dv),
            gaussian_closure)
end

function ivashchenko_pair_jacobian!(output, sys, v)
    @inbounds for pair in eachindex(sys.pair_indices)
        i, j = sys.pair_indices[pair]
        for column in 1:sys.d
            random_index = sys.random_indices[column]
            if i == j
                output[pair, column] = i == random_index ? v[i] : zero(eltype(output))
            else
                output[pair, column] = (i == random_index ? v[j] : zero(eltype(output))) +
                                       (j == random_index ? v[i] : zero(eltype(output)))
            end
        end
    end
    return output
end

function ivashchenko_selected_pair_identity_vjp!(gradient, cotangent, sys)
    nPast = sys.nPast
    constant_index = nPast + 1
    @inbounds for (pair, (i, j)) in enumerate(sys.pair_indices)
        if i == j
            if i != constant_index
                column = i <= nPast ? i : i - 1
                gradient[i] += cotangent[pair, column]
            end
        else
            if i != constant_index
                column = i <= nPast ? i : i - 1
                gradient[j] += cotangent[pair, column]
            end
            if j != constant_index
                column = j <= nPast ? j : j - 1
                gradient[i] += cotangent[pair, column]
            end
        end
    end
    return gradient
end

function ivashchenko_triple_jacobian!(output, sys, v, basis)
    @inbounds for column in 1:sys.d
        fill!(basis, zero(eltype(basis)))
        basis[sys.random_indices[column]] = one(eltype(basis))
        compressed_kron³!(view(output, :, column), v, v, basis)
    end
    return output
end

function ivashchenko_right_input_covariance!(destination::AbstractMatrix,
                                             source::AbstractMatrix,
                                             covariance_input::AbstractMatrix,
                                             nPast::Int, scale = 1.0)
    d = size(source, 2)
    if nPast > 0
        # destination[:, 1:nPast] = scale * source[:, 1:nPast] * P.
        ℒ.mul!(view(destination, :, 1:nPast), view(source, :, 1:nPast),
               view(covariance_input, 1:nPast, 1:nPast), scale, 0.0)
    end
    if nPast < d
        # destination[:, nPast+1:d] = scale * source[:, nPast+1:d].
        @inbounds for j in nPast + 1:d, i in axes(source, 1)
            destination[i, j] = scale * source[i, j]
        end
    end
    return destination
end

function ivashchenko_left_input_covariance!(destination::AbstractMatrix,
                                            source::AbstractMatrix,
                                            covariance_input::AbstractMatrix,
                                            nPast::Int, scale = 1.0)
    d = size(source, 1)
    if nPast > 0
        # destination[1:nPast, :] = scale * P * source[1:nPast, :].
        ℒ.mul!(view(destination, 1:nPast, :),
               view(covariance_input, 1:nPast, 1:nPast),
               view(source, 1:nPast, :), scale, 0.0)
    end
    if nPast < d
        # destination[nPast+1:d, :] = scale * source[nPast+1:d, :].
        @inbounds for j in axes(source, 2), i in nPast + 1:d
            destination[i, j] = scale * source[i, j]
        end
    end
    return destination
end

function ivashchenko_third_order_pullback_workspace(sys, scalar_type::Type)
    nout, d, dv = sys.nout, sys.d, sys.dv
    n_pair = sys.n_pair
    n_triple = length(sys.triple_indices)
    return (; covariance_bar_symmetric = zeros(scalar_type, nout, nout),
            gΣ = zeros(scalar_type, d, d),
            gv = zeros(scalar_type, dv),
            gS1 = zeros(scalar_type, nout, dv),
            gS2 = zeros(scalar_type, nout, n_pair),
            gS3 = zeros(scalar_type, nout, n_triple),
            gL = zeros(scalar_type, nout, d),
            gpair_mean = zeros(scalar_type, n_pair),
            gK = zeros(scalar_type, nout, n_pair),
            gG2 = zeros(scalar_type, n_pair, n_pair),
            gT = zeros(scalar_type, nout, n_triple),
            gR = zeros(scalar_type, d, n_triple),
            gR_internal = zeros(scalar_type, n_triple, d),
            gW = zeros(scalar_type, n_triple, nout),
            gU = zeros(scalar_type, nout, d),
            tmp_noutd = zeros(scalar_type, nout, d),
            tmp_dnout = zeros(scalar_type, d, nout),
            tmp_noutpair = zeros(scalar_type, nout, n_pair),
            tmp_pairnout = zeros(scalar_type, n_pair, nout),
            tmp_nouttriple = zeros(scalar_type, nout, n_triple),
            third_mode_source = zeros(scalar_type, d, n_pair),
            third_mode_one = zeros(scalar_type, d, n_pair),
            third_mode_two = zeros(scalar_type, n_pair, d),
            third_mode_three = zeros(scalar_type, n_pair, d),
            third_cross_compressed = zeros(scalar_type, n_triple, nout),
            third_linear_output = zeros(scalar_type, nout, d),
            gmode_source = zeros(scalar_type, d, n_pair),
            gmode_one = zeros(scalar_type, d, n_pair),
            gmode_two = zeros(scalar_type, n_pair, d),
            gmode_three = zeros(scalar_type, n_pair, d),
            pair_jacobian = zeros(scalar_type, n_pair, d),
            pair_cotangent_matrix = zeros(scalar_type, n_pair, d),
            pair_cotangent = zeros(scalar_type, n_pair),
            triple_jacobian = zeros(scalar_type, n_triple, d),
            triple_cotangent_matrix = zeros(scalar_type, n_triple, d),
            triple_cotangent = zeros(scalar_type, n_triple),
            first_bar = zeros(scalar_type, dv),
            second_bar = zeros(scalar_type, dv),
            third_bar = zeros(scalar_type, dv),
            basis = zeros(scalar_type, dv),
            second_basis = zeros(scalar_type, dv),
            pair_value = zeros(scalar_type, n_pair),
            triple_value = zeros(scalar_type, n_triple),
            mean_result = zeros(scalar_type, sys.nPast),
            covariance_result = zeros(scalar_type, sys.nPast, sys.nPast))
end

function ivashchenko_third_order_transform_output!(transformed, output, sys,
                                                   third_derivative, Σ, ws)
    d = sys.d
    pairs = sys.pair_indices
    triples = sys.triple_indices

    # Transform the symmetric cubic coefficient through Σ one mode at a time.
    # The intermediates retain a compressed pair index; no d³ tensor or dense
    # n_triple×n_triple sixth-moment matrix is materialised.
    @inbounds for l in 1:d, pair in eachindex(pairs)
        m, k = pairs[pair]
        ws.third_mode_source[l, pair] =
            third_derivative[output, compressed_triple_index(l, m, k, d)]
    end
    ivashchenko_left_input_covariance!(ws.third_mode_one, ws.third_mode_source,
                                       Σ, sys.nPast)
    @inbounds for pair in eachindex(pairs), k in 1:d
        i, j = pairs[pair]
        value = zero(eltype(ws.third_mode_two))
        for m in 1:d
            value += Σ[j, m] * ws.third_mode_one[i, compressed_pair_index(m, k, d)]
        end
        ws.third_mode_two[pair, k] = value
    end
    @inbounds for pair in eachindex(pairs), k in 1:d
        value = zero(eltype(ws.third_mode_three))
        for n in 1:d
            value += ws.third_mode_two[pair, n] * Σ[k, n]
        end
        ws.third_mode_three[pair, k] = value
    end
    @inbounds for triple in eachindex(triples)
        i, j, k = triples[triple]
        transformed[triple, output] =
            sys.triple_multiplicities[triple] *
            ws.third_mode_three[compressed_pair_index(i, j, d), k]
    end
    return transformed
end

function ivashchenko_third_order_transform!(transformed, sys, third_derivative, Σ, ws)
    @inbounds for output in 1:sys.nout
        ivashchenko_third_order_transform_output!(transformed, output, sys,
                                                  third_derivative, Σ, ws)
    end

    # The six cross pairings of the sixth Gaussian moment.
    return transformed
end

function ivashchenko_third_order_covariance!(covariance, sys, third_derivative, third_linear, Σ, ws)
    ivashchenko_third_order_transform!(ws.third_cross_compressed, sys,
                                       third_derivative, Σ, ws)
    ℒ.mul!(ws.third_output, third_derivative, ws.third_cross_compressed, 1 / 6, 0.0)

    # The remaining nine pairings are the covariance of the cubic Hermite
    # component that is linear in the Gaussian input.
    ℒ.mul!(ws.third_linear_output, third_derivative, third_linear, 1 / 6, 0.0)
    ivashchenko_right_input_covariance!(ws.third_linear_left,
                                        ws.third_linear_output, Σ, sys.nPast)
    ℒ.mul!(ws.third_output, ws.third_linear_left,
           transpose(ws.third_linear_output), 1.0, 1.0)
    covariance .+= ws.third_output
    return covariance
end

function ivashchenko_third_order_covariance_pullback!(gΣ, gT, sys, third_derivative,
                                                      third_linear, covariance_bar,
                                                      Σ, B)
    ivashchenko_third_order_transform!(B.third_cross_compressed, sys,
                                       third_derivative, Σ, B)
    W = B.third_cross_compressed

    # Cross-Hermite term: Vₓ = T·W / 6.
    ℒ.mul!(B.tmp_nouttriple, covariance_bar, transpose(W), 1 / 6, 0.0)
    gT .+= B.tmp_nouttriple
    ℒ.mul!(B.gW, transpose(third_derivative), covariance_bar, 1 / 6, 0.0)

    # Cubic Hermite's linear component: Vᵢ = UΣU', U = T·R / 6.
    ℒ.mul!(B.third_linear_output, third_derivative, third_linear, 1 / 6, 0.0)
    U = B.third_linear_output
    ℒ.mul!(B.tmp_noutd, covariance_bar, U)
    ivashchenko_right_input_covariance!(B.gU, B.tmp_noutd, Σ,
                                        sys.nPast, 2.0)
    ℒ.mul!(B.tmp_dnout, transpose(U), covariance_bar)
    ℒ.mul!(gΣ, B.tmp_dnout, U, 1.0, 1.0)
    ℒ.mul!(gT, B.gU, transpose(third_linear), 1 / 6, 1.0)
    ℒ.mul!(B.gR_internal, transpose(third_derivative), B.gU, 1 / 6, 0.0)

    @inbounds for triple in eachindex(sys.triple_indices)
        i, j, k = sys.triple_indices[triple]
        multiplicity = sys.triple_multiplicities[triple]
        for r in 1:sys.d
            value = multiplicity * B.gR_internal[triple, r]
            r == i && (gΣ[j, k] += value)
            r == j && (gΣ[i, k] += value)
            r == k && (gΣ[i, j] += value)
        end
    end

    # Reverse the three compressed mode products. Recomputing the small mode
    # intermediates avoids recording d³ work for every filtering period.
    @inbounds for output in 1:sys.nout
        ivashchenko_third_order_transform_output!(W, output, sys,
                                                  third_derivative, Σ, B)
        fill!(B.gmode_three, zero(eltype(B.gmode_three)))
        for triple in eachindex(sys.triple_indices)
            i, j, k = sys.triple_indices[triple]
            B.gmode_three[compressed_pair_index(i, j, sys.d), k] +=
                sys.triple_multiplicities[triple] * B.gW[triple, output]
        end
        ℒ.mul!(gΣ, transpose(B.gmode_three), B.third_mode_two, 1.0, 1.0)
        ivashchenko_right_input_covariance!(B.gmode_two, B.gmode_three, Σ,
                                            sys.nPast)
        fill!(B.gmode_one, zero(eltype(B.gmode_one)))
        for pair in eachindex(sys.pair_indices), k in 1:sys.d
            i, j = sys.pair_indices[pair]
            weight = B.gmode_two[pair, k]
            for m in 1:sys.d
                source_pair = compressed_pair_index(m, k, sys.d)
                gΣ[j, m] += weight * B.third_mode_one[i, source_pair]
                B.gmode_one[i, source_pair] += weight * Σ[j, m]
            end
        end
        ℒ.mul!(gΣ, B.gmode_one, transpose(B.third_mode_source), 1.0, 1.0)
        ivashchenko_left_input_covariance!(B.gmode_source, B.gmode_one, Σ,
                                           sys.nPast)
        for l in 1:sys.d, pair in eachindex(sys.pair_indices)
            m, k = sys.pair_indices[pair]
            triple = compressed_triple_index(l, m, k, sys.d)
            gT[output, triple] += B.gmode_source[l, pair]
        end
    end
    return gΣ, gT
end

function ivashchenko_polynomial_moments!(sys, mean_state, covariance_state, ws)
    nPast, d, dv, nout = sys.nPast, sys.d, sys.dv, sys.nout
    gaussian_closure = ws.gaussian_closure
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

    ivashchenko_pair_jacobian!(ws.pair_jacobian, sys, ws.vbar)
    @inbounds for column in 1:d
        copyto!(view(ws.linear, :, column),
                view(sys.S1, :, sys.random_indices[column]))
    end
    # linear = S₁[:, random] + 1/2 * S₂ * D₂(v).
    ℒ.mul!(ws.linear, sys.S2, ws.pair_jacobian, 0.5, 1.0)
    if sys.order == :third_order
        ivashchenko_triple_jacobian!(ws.triple_jacobian, sys, ws.vbar, ws.basis)
        # linear += 1/2 * S₃ * D₃(v).
        ℒ.mul!(ws.linear, sys.S3, ws.triple_jacobian, 0.5, 1.0)
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

    if sys.order == :third_order
        fill!(ws.third_derivative, zero(eltype(ws.third_derivative)))
        @inbounds for p in eachindex(sys.random_triple_columns)
            ws.third_derivative[:, p] .= sys.S3[:, sys.random_triple_columns[p]]
        end
        fill!(ws.third_linear, zero(eltype(ws.third_linear)))
        @inbounds for p in eachindex(sys.random_triple_indices)
            i, j, k = sys.random_triple_indices[p]
            multiplicity = sys.random_triple_multiplicities[p]
            for r in 1:d
                if gaussian_closure == :exact
                    ws.third_cross[r, p] = multiplicity * (Σ[r, i] * Σ[j, k] +
                        Σ[r, j] * Σ[i, k] + Σ[r, k] * Σ[i, j])
                end
                ws.third_linear[p, r] = multiplicity * ((r == i) * Σ[j, k] +
                    (r == j) * Σ[i, k] + (r == k) * Σ[i, j])
            end
        end
        ℒ.mul!(ws.effective_linear, ws.third_derivative, ws.third_linear, 1 / 6, 1.0)
        if gaussian_closure == :linearized
            # Keep the exact Gaussian mean and cubic Hermite Jacobian, but
            # Gaussianise the transformed state with its delta covariance.
            ivashchenko_right_input_covariance!(ws.linear_left,
                                                ws.effective_linear, Σ, nPast)
            ℒ.mul!(ws.covariance, ws.linear_left, transpose(ws.effective_linear), 1.0, 0.0)
        else
            fill!(ws.pair_covariance, zero(eltype(ws.pair_covariance)))
            @inbounds for p in eachindex(sys.random_pair_indices), q in eachindex(sys.random_pair_indices)
                i, j = sys.random_pair_indices[p]
                k, l = sys.random_pair_indices[q]
                ws.pair_covariance[p, q] = sys.random_pair_multiplicities[p] *
                    sys.random_pair_multiplicities[q] * (Σ[i, k] * Σ[j, l] + Σ[i, l] * Σ[j, k])
            end
            ℒ.mul!(ws.pair_product, ws.hessian, ws.pair_covariance)
            ivashchenko_right_input_covariance!(ws.linear_left, ws.linear,
                                                Σ, nPast)
            ℒ.mul!(ws.covariance, ws.linear_left, transpose(ws.linear), 1.0, 0.0)
            ℒ.mul!(ws.covariance, ws.pair_product, transpose(ws.hessian), 0.25, 1.0)
            ivashchenko_third_order_covariance!(ws.covariance, sys, ws.third_derivative,
                                                 ws.third_linear, Σ, ws)
            ℒ.mul!(ws.third_work, ws.linear, ws.third_cross)
            ℒ.mul!(ws.third_output, ws.third_work, transpose(ws.third_derivative), 1 / 6, 0.0)
            @inbounds for j in 1:nout, i in 1:nout
                ws.covariance[i, j] += ws.third_output[i, j]
                ws.covariance[i, j] += ws.third_output[j, i]
            end
        end
    else
        ivashchenko_right_input_covariance!(ws.linear_left, ws.linear, Σ, nPast)
        ℒ.mul!(ws.covariance, ws.linear_left, transpose(ws.linear), 1.0, 0.0)
        if gaussian_closure != :linearized
            @inbounds for output in 1:nout
                hessian = reshape(view(ws.quadratic_hessian, :, output), d, d)
                left = reshape(view(ws.quadratic_left, :, output), d, d)
                ivashchenko_right_input_covariance!(left, hessian, Σ, nPast)
                if gaussian_closure == :exact
                    right = reshape(view(ws.quadratic_right, :, output), d, d)
                    copyto!(right, transpose(left))
                else
                    # Retain each output's exact curvature variance, but
                    # assume curvature shocks are independent across outputs.
                    value = zero(eltype(ws.covariance))
                    for j in 1:d, i in 1:d
                        value += left[i, j] * left[j, i]
                    end
                    ws.covariance[output, output] += value / 2
                end
            end
        end
        if gaussian_closure == :exact
            # For symmetric T_r and Σ, the quadratic covariance is
            # 1/2 tr(T_r Σ T_s Σ). This avoids materialising the full pair covariance.
            ℒ.mul!(ws.covariance, transpose(ws.quadratic_left), ws.quadratic_right, 0.5, 1.0)
        end
    end

    @inbounds for j in 1:nout, i in 1:j
        value = (ws.covariance[i, j] + ws.covariance[j, i]) / 2
        ws.covariance[i, j] = value
        ws.covariance[j, i] = value
    end
    return ws.mean, ws.covariance
end

function ivashchenko_stationary_mean_solve(A, rhs)
    system = Matrix{eltype(A)}(ℒ.I(size(A, 1)))
    system .-= A
    factor = ℒ.lu!(system, check = false)
    ℒ.issuccess(factor) || return nothing
    return factor \ rhs
end

function ivashchenko_stationary_covariance_forcing(sys, state_rows,
                                                   mean_state, covariance_state, ws,
                                                   input_range)
    ivashchenko_polynomial_moments!(sys, mean_state, covariance_state, ws)
    state_covariance = Matrix(view(ws.covariance, state_rows, state_rows))
    state_linear = Matrix(view(ws.effective_linear, state_rows, 1:sys.nPast))
    input_linear = state_linear[:, input_range]
    input_covariance = Matrix(view(covariance_state, input_range, input_range))
    forcing = state_covariance - input_linear * input_covariance * input_linear'
    return state_linear, forcing
end

function ivashchenko_pruned_stationary_initialization(sys, initial_mean, ws;
                                                      workspaces = nothing,
                                                      lyapunov_algorithm::Symbol = :doubling)
    hasproperty(sys, :pruned) && sys.pruned || return nothing
    sys.nStages ∈ (2, 3) || return nothing
    eltype(ws.vbar) === Float64 && workspaces !== nothing || return nothing

    n_stage = sys.original_nPast
    n_state = sys.nPast
    n_stages = sys.nStages
    n_state == n_stages * n_stage || return nothing
    state_rows = sys.state_position
    first_range = 1:n_stage
    second_range = (n_stage + 1):(2n_stage)
    third_range = (2n_stage + 1):(3n_stage)

    # The first pruned stage is linear and the next stage has no first-stage
    # linear term.  These exact zeros are what make the stationary equations
    # block triangular; otherwise the generic fixed-point iteration remains the
    # correct fallback.
    maximum(abs, sys.S2[state_rows[first_range], :]) == 0 || return nothing
    maximum(abs, sys.A[second_range, first_range]) == 0 || return nothing
    if n_stages == 3
        sys.S3 === nothing && return nothing
        maximum(abs, sys.S3[state_rows[second_range], :]) == 0 || return nothing
    end

    A11 = Matrix(sys.A[first_range, first_range])
    first_constant = Vector(sys.S1[state_rows[first_range], sys.nPast + 1])
    first_mean = ivashchenko_stationary_mean_solve(A11, first_constant)
    first_mean === nothing && return nothing
    first_shocks = Matrix(sys.B[first_range, :])
    first_covariance = qkf_lyapunov(A11, first_shocks * first_shocks';
                                    workspaces = workspaces,
                                    lyapunov_algorithm = lyapunov_algorithm)

    mean_state = zeros(Float64, n_state)
    mean_state[first_range] .= first_mean
    covariance_seed = zeros(Float64, n_state, n_state)
    covariance_seed[first_range, first_range] .= first_covariance

    # Stage two's mean is affine in its own state, with all nonlinear forcing
    # determined by stage one.  Solve it after evaluating that forcing once.
    ivashchenko_polynomial_moments!(sys, mean_state, covariance_seed, ws)
    second_rows = state_rows[second_range]
    A22 = Matrix(sys.A[second_range, second_range])
    second_forcing = Vector(ws.mean[second_rows])
    second_mean = ivashchenko_stationary_mean_solve(A22, second_forcing)
    second_mean === nothing && return nothing
    mean_state[second_range] .= second_mean

    # Solve the covariance of the first two stages using the same block
    # Sylvester/Lyapunov pattern used by the higher-order moment code.
    upper_range = 1:(2n_stage)
    first_state_linear, first_forcing = ivashchenko_stationary_covariance_forcing(
        sys, state_rows, mean_state, covariance_seed, ws, first_range)
    opts = merge_calculation_options(lyapunov_algorithm = lyapunov_algorithm)
    upper_covariance, solved = solve_block_triangular_lyapunov(
        first_state_linear[first_range, first_range],
        first_state_linear[second_range, first_range],
        first_state_linear[second_range, second_range],
        first_forcing[second_range, first_range],
        first_forcing[second_range, second_range],
        first_covariance, workspaces, opts)
    solved || return nothing

    if n_stages == 2
        final_covariance = upper_covariance
    else
        # The third-stage mean depends on the already solved joint first/two
        # stage Gaussian law through the mixed quadratic and cubic terms.
        covariance_seed[upper_range, upper_range] .= upper_covariance
        ivashchenko_polynomial_moments!(sys, mean_state, covariance_seed, ws)
        A33 = Matrix(sys.A[third_range, third_range])
        third_rows = state_rows[third_range]
        third_forcing = Vector(ws.mean[third_rows])
        third_mean = ivashchenko_stationary_mean_solve(A33, third_forcing)
        third_mean === nothing && return nothing
        mean_state[third_range] .= third_mean

        final_state_linear, final_forcing = ivashchenko_stationary_covariance_forcing(
            sys, state_rows, mean_state, covariance_seed, ws, upper_range)
        final_covariance, solved = solve_block_triangular_lyapunov(
            final_state_linear[upper_range, upper_range],
            final_state_linear[third_range, upper_range],
            final_state_linear[third_range, third_range],
            final_forcing[third_range, upper_range],
            final_forcing[third_range, third_range],
            upper_covariance, workspaces, opts)
        solved || return nothing
    end

    final_covariance = (final_covariance + final_covariance') / 2
    ivashchenko_polynomial_moments!(sys, mean_state, final_covariance, ws)
    mean_residual = maximum(abs, Vector(ws.mean[state_rows]) - mean_state)
    covariance_residual = maximum(abs,
                                  Matrix(view(ws.covariance, state_rows, state_rows)) -
                                  final_covariance)
    scale = max(1.0, maximum(abs, mean_state), maximum(abs, final_covariance))
    max(mean_residual, covariance_residual) <= 100 *
        IVASHCHENKO_STATIONARY_TOLERANCE * scale || return nothing
    return mean_state, final_covariance, true
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

    pruned_initialization = ivashchenko_pruned_stationary_initialization(
        sys, initial_mean, ws; workspaces = workspaces,
        lyapunov_algorithm = lyapunov_algorithm)
    pruned_initialization !== nothing && return pruned_initialization

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
            pair_covariance = ws.third_derivative === nothing ? nothing :
                              (ws.pair_covariance === nothing ? nothing : copy(ws.pair_covariance)),
            covariance = copy(ws.covariance),
            third_derivative = ws.third_derivative === nothing ? nothing : copy(ws.third_derivative),
            third_cross = ws.third_cross === nothing ? nothing : copy(ws.third_cross),
            third_linear = ws.third_linear === nothing ? nothing : copy(ws.third_linear),
            gaussian_closure = ws.gaussian_closure)
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
                                 gaussian_closure::Symbol = :exact,
                                 record::Bool = false)
    n_obs, nT = size(data_in_deviations)
    presample_periods = normalize_presample_periods(presample_periods, nT)
    scalar_type = promote_type(eltype(sys.S1), eltype(data_in_deviations),
                               measurement_error === nothing ? Float64 :
                               (measurement_error isa AbstractArray ? eltype(measurement_error) : typeof(measurement_error)))
    Hm = ivashchenko_measurement_covariance(measurement_error, n_obs, scalar_type)
    obs_idx_per_t, _ = build_obs_index(data_in_deviations)

    ws = ivashchenko_kalman_workspace(sys, scalar_type;
                                      gaussian_closure = gaussian_closure)
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
    if scalar_type === Float64
        linear_cache = 𝒮.init(𝒮.LinearProblem(innovation_covariance, innovation_solve),
                               𝒮.FastLUFactorization(), verbose = false)
    end
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

        fast_lu = scalar_type === Float64 && m == n_obs
        if fast_lu
            copyto!(innovation_solve, innovation)
            linear_cache.A = innovation_covariance
            linear_cache.b = innovation_solve
            solution = 𝒮.solve!(linear_cache)
            successful = 𝒮.SciMLBase.successful_retcode(solution.retcode) ||
                         solution.retcode == 𝒮.SciMLBase.ReturnCode.Default
            successful || return record ? (convert(scalar_type, on_failure_loglikelihood), nothing) :
                                          convert(scalar_type, on_failure_loglikelihood)
            logabsdetF, signF = ℒ.logabsdet(linear_cache.cacheval.factors)
        else
            # LU on a SubArray takes a considerably more allocating generic path;
            # retain it for partial observations and non-Float64 element types.
            factor = m == n_obs ? ℒ.lu!(innovation_covariance, check = false) :
                                   ℒ.lu(view(innovation_covariance, 1:m, 1:m), check = false)
            ℒ.issuccess(factor) || return record ? (convert(scalar_type, on_failure_loglikelihood), nothing) :
                                                  convert(scalar_type, on_failure_loglikelihood)
            logabsdetF, signF = ℒ.logabsdet(factor)
        end
        (primal(signF) > 0 && isfinite(primal(logabsdetF))) ||
            return record ? (convert(scalar_type, on_failure_loglikelihood), nothing) :
                            convert(scalar_type, on_failure_loglikelihood)

        inverse_view = m == n_obs ? inverse_innovation_covariance :
                                   view(inverse_innovation_covariance, 1:m, 1:m)
        fill!(inverse_view, zero(scalar_type))
        @inbounds for i in 1:m
            inverse_view[i, i] = one(scalar_type)
        end
        if fast_lu
            solve_linear_cache_lu_left!(linear_cache, inverse_view)
        else
            ℒ.ldiv!(factor, inverse_view)
        end

        if t > presample_periods
            if fast_lu
                ll -= (ℒ.dot(view(innovation, 1:m), linear_cache.u) + logabsdetF + m * log2pi) / 2
            else
                copyto!(view(innovation_solve, 1:m), view(innovation, 1:m))
                ℒ.ldiv!(factor, view(innovation_solve, 1:m))
                ll -= (ℒ.dot(view(innovation, 1:m), view(innovation_solve, 1:m)) + logabsdetF + m * log2pi) / 2
            end
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
                                lyapunov_algorithm::Symbol = :doubling,
                                gaussian_closure::Symbol = :exact)
    return ivashchenko_filter_pass(sys, data_in_deviations, initial_mean;
                                   measurement_error = measurement_error,
                                   initial_covariance = initial_covariance,
                                   presample_periods = presample_periods,
                                   on_failure_loglikelihood = on_failure_loglikelihood,
                                   workspaces = workspaces,
                                   lyapunov_algorithm = lyapunov_algorithm,
                                   gaussian_closure = gaussian_closure)
end

function ivashchenko_quadratic_covariance_pullback!(gΣ, gS2, sys, covariance_bar, Σ, buffers;
                                                   gaussian_closure::Symbol = :exact)
    d, nout = sys.d, sys.nout
    T = sys.quadratic_hessian
    mixed = buffers.mixed
    gT = buffers.gT
    left = buffers.left
    right = buffers.right

    if gaussian_closure == :diagonal
        # Only V̄_rr reaches the retained curvature variances. For
        # V_rr = 1/2 tr(T_r Σ T_r Σ), the symmetric adjoints are
        # dV_rr/dT_r = Σ T_r Σ and dV_rr/dΣ = T_r Σ T_r.
        @inbounds for output in 1:nout
            weight = covariance_bar[output, output]
            T_output = reshape(view(T, :, output), d, d)
            gT_output = reshape(view(gT, :, output), d, d)
            ivashchenko_left_input_covariance!(right, T_output, Σ, sys.nPast)
            ivashchenko_right_input_covariance!(gT_output, right, Σ,
                                                sys.nPast, weight)
            ivashchenko_right_input_covariance!(right, T_output, Σ, sys.nPast)
            ℒ.mul!(left, right, T_output, weight, 0.0)
            gΣ .+= left
        end
    else
        # If V_rs = 1/2 tr(T_r Σ T_s Σ), the symmetric covariance cotangent gives
        # dV/dT_r = Σ (Σ_s V̄_rs T_s) Σ and dV/dΣ = Σ_r T_r Σ (Σ_s V̄_rs T_s).
        ℒ.mul!(mixed, T, transpose(covariance_bar))
        @inbounds for output in 1:nout
            Tmix = reshape(view(mixed, :, output), d, d)
            gT_output = reshape(view(gT, :, output), d, d)
            ivashchenko_left_input_covariance!(right, Tmix, Σ, sys.nPast)
            ivashchenko_right_input_covariance!(gT_output, right, Σ, sys.nPast)
            T_output = reshape(view(T, :, output), d, d)
            ivashchenko_right_input_covariance!(right, T_output, Σ, sys.nPast)
            ℒ.mul!(left, right, Tmix)
            gΣ .+= left
        end
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
    if eltype(matrix) === Float64 && eltype(left) === Float64 && eltype(right) === Float64
        # matrix += scale * left * right'; avoid a temporary outer product.
        ℒ.BLAS.ger!(Float64(scale), left, right, matrix)
        return matrix
    end
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
    ivashchenko_right_input_covariance!(gL, buffers.tmp_noutd, Σ,
                                        sys.nPast, 2.0)
    ℒ.mul!(buffers.tmp_dnout, transpose(L), covariance_bar_symmetric)
    ℒ.mul!(gΣ, buffers.tmp_dnout, L, 1.0, 0.0)
    if moment_tape.gaussian_closure != :linearized
        ivashchenko_quadratic_covariance_pullback!(gΣ, gS2, sys,
                                                   covariance_bar_symmetric, Σ,
                                                   buffers;
                                                   gaussian_closure = moment_tape.gaussian_closure)
    end

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

    ivashchenko_pair_jacobian!(buffers.pair_jacobian, sys, v)
    @inbounds for column in 1:d
        solution_column = sys.random_indices[column]
        for output in axes(gS1, 1)
            gS1[output, solution_column] += gL[output, column]
        end
    end
    # gS₂ += 1/2 * gL * D₂(v)'; pair_cotangent = 1/2 * S₂' * gL.
    ℒ.mul!(gS2, gL, transpose(buffers.pair_jacobian), 0.5, 1.0)
    ℒ.mul!(buffers.pair_cotangent_matrix, transpose(sys.S2), gL, 0.5, 0.0)
    ivashchenko_selected_pair_identity_vjp!(gv,
                                            buffers.pair_cotangent_matrix, sys)

    @inbounds for j in 1:d, i in 1:j-1
        value = (gΣ[i, j] + gΣ[j, i]) / 2
        gΣ[i, j] = value
        gΣ[j, i] = value
    end
    copyto!(buffers.mean_result, view(gv, 1:sys.nPast))
    copyto!(buffers.covariance_result, view(gΣ, 1:sys.nPast, 1:sys.nPast))
    return buffers.mean_result, buffers.covariance_result, gS1, gS2, nothing
end

function ivashchenko_third_order_linearized_covariance_pullback!(gΣ, gL, gT, sys,
                                                                  moment_tape,
                                                                  covariance_bar, B)
    Σ = moment_tape.covariance_input
    effective_linear = moment_tape.effective_linear
    third_derivative = moment_tape.third_derivative
    third_linear = moment_tape.third_linear

    # The linearized cubic closure uses V = EΣE', with
    # E = L + T·R/6. Reverse that factorisation before mapping R back to Σ.
    ℒ.mul!(B.tmp_noutd, covariance_bar, effective_linear)
    ivashchenko_right_input_covariance!(gL, B.tmp_noutd, Σ,
                                        sys.nPast, 2.0)
    ℒ.mul!(B.tmp_dnout, transpose(effective_linear), covariance_bar)
    ℒ.mul!(gΣ, B.tmp_dnout, effective_linear, 1.0, 0.0)
    ℒ.mul!(gT, gL, transpose(third_linear), 1 / 6, 0.0)
    ℒ.mul!(B.gR_internal, transpose(third_derivative), gL, 1 / 6, 0.0)

    @inbounds for triple in eachindex(sys.random_triple_indices), r in 1:sys.d
        i, j, k = sys.random_triple_indices[triple]
        weight = sys.triple_multiplicities[triple] * B.gR_internal[triple, r]
        r == i && (gΣ[j, k] += weight)
        r == j && (gΣ[i, k] += weight)
        r == k && (gΣ[i, j] += weight)
    end
    return gΣ, gL, gT
end

function ivashchenko_polynomial_moments_pullback(sys, moment_tape, mean_bar, covariance_bar;
                                                 quadratic_buffers = nothing,
                                                 third_order_buffers = nothing)
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
    B = third_order_buffers === nothing ?
        ivashchenko_third_order_pullback_workspace(sys, eltype(Σ)) : third_order_buffers
    covariance_bar_symmetric = B.covariance_bar_symmetric
    gΣ, gv = B.gΣ, B.gv
    gS1, gS2, gS3 = B.gS1, B.gS2, B.gS3
    gL, gpair_mean = B.gL, B.gpair_mean
    gK, gG2 = B.gK, B.gG2
    gT, gR = B.gT, B.gR
    fill!(gΣ, zero(eltype(gΣ)))
    fill!(gv, zero(eltype(gv)))
    fill!(gS1, zero(eltype(gS1)))
    fill!(gS2, zero(eltype(gS2)))
    fill!(gS3, zero(eltype(gS3)))
    @inbounds for j in 1:nout, i in 1:j
        value = (covariance_bar[i, j] + covariance_bar[j, i]) / 2
        covariance_bar_symmetric[i, j] = value
        covariance_bar_symmetric[j, i] = value
    end

    if third !== nothing && moment_tape.gaussian_closure == :linearized
        ivashchenko_third_order_linearized_covariance_pullback!(
            gΣ, gL, gT, sys, moment_tape, covariance_bar_symmetric, B)
    else
        ℒ.mul!(B.tmp_noutd, covariance_bar_symmetric, L)
        ivashchenko_right_input_covariance!(gL, B.tmp_noutd, Σ,
                                            sys.nPast, 2.0)
        ℒ.mul!(B.tmp_dnout, transpose(L), covariance_bar_symmetric)
        ℒ.mul!(gΣ, B.tmp_dnout, L, 1.0, 0.0)
    end
    ℒ.mul!(gpair_mean, transpose(K), mean_bar, 0.5, 0.0)

    if third !== nothing
        if moment_tape.gaussian_closure == :linearized
            # The linearized covariance helper already reversed the cubic
            # Hermite Jacobian. Only the pair mean remains nonlinear here.
            fill!(gK, zero(eltype(gK)))
            ivashchenko_rank_one_add!(gK, mean_bar, pair_mean, 0.5)
            @inbounds for p in eachindex(sys.random_pair_indices)
                i, j = sys.random_pair_indices[p]
                gΣ[i, j] += sys.random_pair_multiplicities[p] * gpair_mean[p]
            end
        else
            pair_covariance = moment_tape.pair_covariance
            ℒ.mul!(B.tmp_noutpair, covariance_bar_symmetric, K)
            ℒ.mul!(gK, B.tmp_noutpair, pair_covariance, 0.5, 0.0)
            ℒ.mul!(B.tmp_pairnout, transpose(K), covariance_bar_symmetric)
            ℒ.mul!(gG2, B.tmp_pairnout, K, 0.25, 0.0)
            ivashchenko_rank_one_add!(gK, mean_bar, pair_mean, 0.5)
            third_cross = moment_tape.third_cross
            ℒ.mul!(B.tmp_nouttriple, covariance_bar_symmetric, third)
            ℒ.mul!(gL, B.tmp_nouttriple, transpose(third_cross), 1 / 3, 1.0)
            ℒ.mul!(gT, B.tmp_noutd, third_cross, 1 / 3, 0.0)
            ℒ.mul!(gR, B.tmp_dnout, third, 1 / 3, 0.0)
            ivashchenko_third_order_covariance_pullback!(
                gΣ, gT, sys, third, moment_tape.third_linear,
                covariance_bar_symmetric, Σ, B)

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

    pair_cotangent = B.pair_cotangent
    triple_cotangent = B.triple_cotangent
    first_bar = B.first_bar
    second_bar = B.second_bar
    third_bar = B.third_bar
    basis = B.basis
    second_basis = B.second_basis
    pair_value = B.pair_value
    triple_value = B.triple_value

    compressed_kron²_power!(pair_value, v)
    ivashchenko_rank_one_add!(gS1, mean_bar, v, 1.0)
    gv .+= sys.S1' * mean_bar
    ivashchenko_rank_one_add!(gS2, mean_bar, pair_value, 0.5)
    ℒ.mul!(pair_cotangent, transpose(sys.S2), mean_bar, 0.5, 0.0)
    compressed_kron²_power_vjp!(first_bar, pair_cotangent, v)
    gv .+= first_bar
    if third !== nothing
        compressed_kron³_power!(triple_value, v)
        ivashchenko_rank_one_add!(gS3, mean_bar, triple_value, 1 / 6)
        ℒ.mul!(triple_cotangent, transpose(sys.S3), mean_bar, 1 / 6, 0.0)
        compressed_kron³_power_vjp!(first_bar, triple_cotangent, v)
        gv .+= first_bar
    end

    ivashchenko_pair_jacobian!(B.pair_jacobian, sys, v)
    @inbounds for column in 1:d
        solution_column = sys.random_indices[column]
        for output in axes(gS1, 1)
            gS1[output, solution_column] += gL[output, column]
        end
    end
    # gS₂ += 1/2 * gL * D₂(v)'; pair_cotangent = 1/2 * S₂' * gL.
    ℒ.mul!(gS2, gL, transpose(B.pair_jacobian), 0.5, 1.0)
    ℒ.mul!(B.pair_cotangent_matrix, transpose(sys.S2), gL, 0.5, 0.0)
    ivashchenko_selected_pair_identity_vjp!(gv, B.pair_cotangent_matrix, sys)

    if third !== nothing
        ivashchenko_triple_jacobian!(B.triple_jacobian, sys, v, basis)
        # gS₃ += 1/2 * gL * D₃(v)'; triple_cotangent = 1/2 * S₃' * gL.
        ℒ.mul!(gS3, gL, transpose(B.triple_jacobian), 0.5, 1.0)
        ℒ.mul!(B.triple_cotangent_matrix, transpose(sys.S3), gL, 0.5, 0.0)
        @inbounds for r in 1:d
            i = sys.random_indices[r]
            fill!(basis, zero(eltype(basis)))
            basis[i] = one(eltype(basis))
            copyto!(triple_cotangent, view(B.triple_cotangent_matrix, :, r))
            compressed_kron³_vjp!(first_bar, second_bar, third_bar,
                                  triple_cotangent, v, v, basis)
            gv .+= first_bar
            gv .+= second_bar
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
            ivashchenko_rank_one_add!(gS3, view(gK, :, p), triple_value, 1.0)
            ℒ.mul!(triple_cotangent, transpose(sys.S3), view(gK, :, p), 1.0, 0.0)
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

    @inbounds for j in 1:d, i in 1:j-1
        value = (gΣ[i, j] + gΣ[j, i]) / 2
        gΣ[i, j] = value
        gΣ[j, i] = value
    end
    copyto!(B.mean_result, view(gv, 1:sys.nPast))
    copyto!(B.covariance_result, view(gΣ, 1:sys.nPast, 1:sys.nPast))
    return B.mean_result, B.covariance_result, gS1, gS2, gS3
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
           pair_jacobian = zeros(scalar_type, sys.n_pair, sys.d),
           pair_cotangent_matrix = zeros(scalar_type, sys.n_pair, sys.d),
           first_bar = zeros(scalar_type, sys.dv),
           pair_value = zeros(scalar_type, sys.n_pair),
           mixed = zeros(scalar_type, sys.d * sys.d, sys.nout),
           gT = zeros(scalar_type, sys.d * sys.d, sys.nout),
           left = zeros(scalar_type, sys.d, sys.d),
           right = zeros(scalar_type, sys.d, sys.d),
           mean_result = zeros(scalar_type, sys.nPast),
           covariance_result = zeros(scalar_type, sys.nPast, sys.nPast)) : nothing
    third_order_buffers = sys.order == :third_order ?
        ivashchenko_third_order_pullback_workspace(sys, scalar_type) : nothing

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
                                                    quadratic_buffers = quadratic_buffers,
                                                    third_order_buffers = third_order_buffers)
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
                                                        quadratic_buffers = quadratic_buffers,
                                                        third_order_buffers = third_order_buffers)
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
    𝐒 = order ∈ (:second_order, :pruned_second_order) ? result[7:8] : result[8:10]
    ensure_model_structure_constants!(constants, 𝓂.equations.calibration_parameters)
    all_SS = expand_steady_state(SS_and_pars, constants.post_complete_parameters)
    state_deviation = collect(sss) - all_SS
    pruned = order ∈ (:pruned_second_order, :pruned_third_order)
    state = if order == :pruned_second_order
        [zeros(Float64, T.nVars), state_deviation]
    elseif order == :pruned_third_order
        [zeros(Float64, T.nVars), state_deviation, zeros(Float64, T.nVars)]
    else
        state_deviation
    end
    observables = get_and_check_observables(T, data_in_deviations)
    observable_indices = convert(Vector{Int}, indexin(observables, constants.post_complete_parameters.SS_and_pars_names))
    sys = pruned ?
        build_pruned_ivashchenko_kalman_system_from_constants(constants, 𝐒,
                                                              observable_indices, order;
                                                              keep_all_rows = true) :
        build_ivashchenko_kalman_system_from_constants(constants, 𝐒, observable_indices, order)
    data = collect(data_in_deviations)
    initial_mean = pruned ? pruned_ivashchenko_initial_mean(sys, state) : state[sys.past]
    pass = ivashchenko_filter_pass(sys, data, initial_mean;
                                   measurement_error = measurement_error,
                                   initial_covariance = initial_covariance,
                                   presample_periods = 0,
                                   workspaces = 𝓂.workspaces,
                                   lyapunov_algorithm = opts.lyapunov_algorithm,
                                   record = true)
    pass[2] === nothing && return variables, shocks, standard_deviations, decomposition
    if smooth
        if !pruned
            return ivashchenko_smooth_pass(sys, pass[2])[1:4]
        end
        internal_variables, shocks, _, _, _, smoothed_covariances =
            ivashchenko_smooth_pass(sys, pass[2])
        variables = zeros(Float64, T.nVars, nT)
        standard_deviations = zeros(Float64, T.nVars, nT)
        @inbounds for t in 1:nT
            variables[:, t] .= sum((view(internal_variables,
                                         (stage - 1) * sys.nKeep .+ (1:T.nVars), t)
                                     for stage in 1:sys.nStages))
            predicted_covariance = pass[2].predicted_covariances[t]
            predicted_factor = ℒ.cholesky(predicted_covariance, check = false)
            ℒ.issuccess(predicted_factor) || error("Ivashchenko smoother covariance factorization failed at period $t.")
            state_regression = copy(pass[2].output_covariances[t][:, sys.state_position])
            ℒ.rdiv!(state_regression, predicted_factor)
            conditional = pass[2].output_covariances[t] -
                          state_regression * predicted_covariance * state_regression'
            smoothed_output_covariance = conditional +
                state_regression * smoothed_covariances[t] * state_regression'
            for variable in 1:T.nVars
                row_indices = [(stage - 1) * sys.nKeep + variable for stage in 1:sys.nStages]
                standard_deviations[variable, t] = sqrt(abs(sum(smoothed_output_covariance[row_indices, row_indices])))
            end
        end
        decomposition = zeros(Float64, T.nVars, T.nExo + 2, nT)
        decomposition[:, end - 1, :] .= variables
        return variables, shocks, standard_deviations, decomposition
    end

    tape = pass[2]
    @inbounds for t in 1:nT
        if pruned
            for stage in 1:sys.nStages
                variables[:, t] .+= tape.output_means[t][(stage - 1) * sys.nKeep .+ (1:T.nVars)]
            end
            for variable in 1:T.nVars
                output_rows = [(stage - 1) * sys.nKeep + variable for stage in 1:sys.nStages]
                standard_deviations[variable, t] =
                    sqrt(abs(sum(tape.output_covariances[t][output_rows, output_rows])))
            end
        else
            variables[:, t] .= tape.output_means[t]
            standard_deviations[:, t] .= sqrt.(abs.(ℒ.diag(tape.output_covariances[t])))
        end
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

@unstable function filter_data_with_model(𝓂::ℳ,
                                          data_in_deviations::KeyedArray{Float64},
                                          ::Val{:pruned_second_order},
                                          ::Val{:ivashchenko_kalman};
                                          warmup_iterations::Int = 0,
                                          initial_covariance = :theoretical,
                                          measurement_error = nothing,
                                          smooth::Bool = true,
                                          opts::CalculationOptions = merge_calculation_options())
    return ivashchenko_filter_data_with_model(𝓂, data_in_deviations, :pruned_second_order;
                                              initial_covariance = initial_covariance,
                                              measurement_error = measurement_error,
                                              smooth = smooth, opts = opts)
end

@unstable function filter_data_with_model(𝓂::ℳ,
                                          data_in_deviations::KeyedArray{Float64},
                                          ::Val{:pruned_third_order},
                                          ::Val{:ivashchenko_kalman};
                                          warmup_iterations::Int = 0,
                                          initial_covariance = :theoretical,
                                          measurement_error = nothing,
                                          smooth::Bool = true,
                                          opts::CalculationOptions = merge_calculation_options())
    return ivashchenko_filter_data_with_model(𝓂, data_in_deviations, :pruned_third_order;
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
                                 gaussian_closure::Symbol = :exact,
                                 opts::CalculationOptions = merge_calculation_options()) where {O}
    O ∈ (:second_order, :third_order, :pruned_second_order, :pruned_third_order) ||
        throw(ArgumentError("The Ivashchenko filter requires a second- or third-order algorithm."))
    pruned = O ∈ (:pruned_second_order, :pruned_third_order)
    sys = pruned ?
        build_pruned_ivashchenko_kalman_system_from_constants(constants, 𝐒, observables_index, O) :
        build_ivashchenko_kalman_system_from_constants(constants, 𝐒, observables_index, O;
                                                       keep_all_rows = false)
    initial_mean = pruned ? pruned_ivashchenko_initial_mean(sys, state) : state[sys.past]
    return run_ivashchenko_kalman(sys, data_in_deviations, initial_mean;
                                  measurement_error = measurement_error,
                                  initial_covariance = initial_covariance,
                                  presample_periods = presample_periods,
                                  on_failure_loglikelihood = on_failure_loglikelihood,
                                  workspaces = workspaces,
                                  lyapunov_algorithm = lyapunov_algorithm,
                                  gaussian_closure = gaussian_closure)
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
                                              gaussian_closure::Symbol = :exact,
                                              opts::CalculationOptions = merge_calculation_options()) where {O}
    O ∈ (:second_order, :third_order, :pruned_second_order, :pruned_third_order) ||
        throw(ArgumentError("The Ivashchenko filter requires a second- or third-order algorithm."))
    pruned = O ∈ (:pruned_second_order, :pruned_third_order)
    sys = pruned ?
        build_pruned_ivashchenko_kalman_system_from_constants(constants, 𝐒, observables_index, O) :
        build_ivashchenko_kalman_system_from_constants(constants, 𝐒, observables_index, O;
                                                       keep_all_rows = false)
    initial_mean = pruned ? pruned_ivashchenko_initial_mean(sys, state) : state[sys.past]
    return run_ivashchenko_kalman(sys, data_in_deviations, initial_mean;
                                  measurement_error = measurement_error,
                                  initial_covariance = initial_covariance,
                                  presample_periods = presample_periods,
                                  on_failure_loglikelihood = on_failure_loglikelihood,
                                  workspaces = workspaces,
                                  lyapunov_algorithm = lyapunov_algorithm,
                                  gaussian_closure = gaussian_closure)
end

end # @stable
