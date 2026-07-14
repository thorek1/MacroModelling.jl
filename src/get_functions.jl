@stable default_mode = "disable" begin


# ---------------------------------------------------------------------------
# Missing-value support helpers (shared by Kalman and inversion filters)
# ---------------------------------------------------------------------------
# Filters accept data that may contain unobserved entries. The internal
# canonical sentinel is NaN: `missing_data_to_nan` converts `missing` and
# `nothing` to NaN at the public API boundary, and `build_obs_index` then
# classifies any non-finite entry (NaN, Inf, -Inf) as unobserved via
# `isfinite`. A per-period vector of available observable indices is built
# once per call and reused inside the filter loops to slice the (max-sized)
# workspace buffers, so all preallocation tricks are kept.

"""
    missing_data_to_nan(data) -> Matrix{Float64}

Convert a matrix that may contain `missing` or `nothing` values into a
`Matrix{Float64}` where `missing`/`nothing` become `NaN`. `Matrix{Float64}`
inputs are returned unchanged (no copy). Non-finite real values (NaN, Inf,
-Inf) are preserved and are treated as unobserved downstream by
`build_obs_index`.
"""
missing_data_to_nan(data::Matrix{Float64}) = data
function missing_data_to_nan(data::AbstractMatrix{<:Union{Missing,Nothing,Real}})
    out = Matrix{Float64}(undef, size(data, 1), size(data, 2))
    @inbounds for j in axes(data, 2), i in axes(data, 1)
        v = data[i, j]
        out[i, j] = (v === missing || v === nothing) ? NaN : Float64(v)
    end
    return out
end
missing_data_to_nan(data::AbstractMatrix{<:Real}) = convert(Matrix{Float64}, data)
@unstable function missing_data_to_nan(data::KeyedArray)
    raw = missing_data_to_nan(collect(data))::Matrix{Float64}
    names = AxisKeys.NamedDims.dimnames(data)
    return KeyedArray(NamedDimsArray(raw, names); NamedTuple{names}(axiskeys(data))...)
end

"""
    build_obs_index(data) -> (Vector{Vector{Int}}, Bool)

Return `(obs_idx_per_t, has_missing)` where `obs_idx_per_t[t]` is the sorted
vector of row indices that are *observed* (finite) in column `t` of `data`,
and `has_missing` is `true` iff at least one entry is non-finite (NaN, Inf
or -Inf). When `has_missing == false`, callers can take the dense fast path.
"""
function build_obs_index(data::AbstractMatrix{<:Real})
    n, T = size(data)
    obs = Vector{Vector{Int}}(undef, T)
    has_missing = false
    @inbounds for t in 1:T
        m = 0
        for i in 1:n
            if isfinite(data[i, t])
                m += 1
            else
                has_missing = true
            end
        end
        if m == n
            obs[t] = collect(1:n)
        else
            v = Vector{Int}(undef, m)
            k = 0
            for i in 1:n
                if isfinite(data[i, t])
                    k += 1
                    v[k] = i
                end
            end
            obs[t] = v
        end
    end
    return obs, has_missing
end

"""
    informative_period_range(data) -> UnitRange{Int}

Return the smallest contiguous column range that contains every period with at
least one finite observable. Leading and trailing columns with no finite
entries are dropped. If every period is fully unobserved, return `1:0`.
"""
function informative_period_range(obs_idx_per_t::Vector{Vector{Int}})
    first_t = findfirst(idx -> !isempty(idx), obs_idx_per_t)
    first_t === nothing && return 1:0
    last_t = findlast(idx -> !isempty(idx), obs_idx_per_t)
    last_t === nothing && return 1:0
    return first_t:last_t
end

function informative_period_range(data::AbstractMatrix{<:Real})
    obs_idx_per_t, _ = build_obs_index(data)
    return informative_period_range(obs_idx_per_t)
end

informative_period_range(data::KeyedArray) = informative_period_range(collect(data))

function adjust_initial_state(initial_state,
                              algorithm::Symbol,
                              nVars::Int,
                              SSS_delta::AbstractVector{<:Real},
                              reference_steady_state::AbstractVector{<:Real})
    R = promote_type(eltype(SSS_delta), eltype(reference_steady_state))

    if initial_state isa AbstractVector{<:Real}
        if length(initial_state) != nVars
            if algorithm == :pruned_second_order
                return [zeros(R, nVars), zeros(R, nVars) - SSS_delta]
            elseif algorithm == :pruned_third_order
                return [zeros(R, nVars), zeros(R, nVars) - SSS_delta, zeros(R, nVars)]
            else
                return zeros(R, nVars) - SSS_delta
            end
        end

        if algorithm == :pruned_second_order
            return [initial_state - reference_steady_state[1:nVars], zeros(R, nVars) - SSS_delta]
        elseif algorithm == :pruned_third_order
            return [initial_state - reference_steady_state[1:nVars], zeros(R, nVars) - SSS_delta, zeros(R, nVars)]
        else
            return initial_state - (reference_steady_state[1:nVars] + SSS_delta[1:nVars])
        end
    end

    if algorithm ∉ [:pruned_second_order, :pruned_third_order]
        @assert initial_state isa AbstractVector{<:Real} "The solution algorithm has one state vector: initial_state must be a Vector{Float64}."
    end

    return initial_state
end

function adjust_initial_state(initial_state,
                              algorithm::Symbol,
                              𝓂::ℳ,
                              SSS_delta::AbstractVector{<:Real},
                              reference_steady_state::AbstractVector{<:Real})
    return adjust_initial_state(initial_state, algorithm, 𝓂.constants.post_model_macro.nVars, SSS_delta, reference_steady_state)
end

function report_informative_trim(period_range::UnitRange{Int}, n_periods::Int; maxlog::Int = DEFAULT_MAXLOG)
    n_leading = isempty(period_range) ? n_periods : first(period_range) - 1
    n_trailing = isempty(period_range) ? 0 : n_periods - last(period_range)
    if n_leading > 0 || n_trailing > 0
        period_summary = if n_leading > 0 && n_trailing > 0
            "$(n_leading) leading and $(n_trailing) trailing"
        elseif n_leading > 0
            "$(n_leading) leading"
        else
            "$(n_trailing) trailing"
        end
        @warn "The data has $(period_summary) fully unobserved periods. Those periods are disregarded." maxlog = maxlog
    end
    return nothing
end

function trim_informative_sample(data::AbstractMatrix{<:Real};
                                         warn_on_trim::Bool = true,
                                         maxlog::Int = DEFAULT_MAXLOG,
                                         presample_periods::Int = 0,
                                         require_informative_periods::Bool = false)::Tuple{AbstractMatrix{<:Real}, Vector{Vector{Int}}, Bool, UnitRange{Int}}
    obs_idx_per_t, _ = build_obs_index(data)
    period_range = informative_period_range(obs_idx_per_t)
    warn_on_trim && report_informative_trim(period_range, size(data, 2); maxlog = maxlog)
    presample_periods = normalize_presample_periods(presample_periods, length(period_range); maxlog = maxlog)
    if require_informative_periods
        @assert !isempty(period_range) "The data contains no informative periods after removing fully unobserved boundaries."
    end
    trimmed_data = data[:, period_range]
    if isempty(period_range)
        return (trimmed_data, Vector{Vector{Int}}(), false, period_range)
    end
    trimmed_obs_idx = obs_idx_per_t[period_range]
    has_missing = any(length(idx) < size(data, 1) for idx in trimmed_obs_idx)
    return (trimmed_data, trimmed_obs_idx, has_missing, period_range)
end

function trim_informative_sample(data::KeyedArray;
                                         warn_on_trim::Bool = true,
                                         maxlog::Int = DEFAULT_MAXLOG,
                                         presample_periods::Int = 0,
                                         require_informative_periods::Bool = false)::Tuple{KeyedArray, Vector{Vector{Int}}, Bool, UnitRange{Int}}
    raw = collect(data)
    _, trimmed_obs_idx, has_missing, period_range = trim_informative_sample(raw;
                                                                            warn_on_trim = warn_on_trim,
                                                                            maxlog = maxlog,
                                                                            presample_periods = presample_periods,
                                                                            require_informative_periods = require_informative_periods)
    trimmed_data = data[:, period_range]
    return (trimmed_data, trimmed_obs_idx, has_missing, period_range)
end


function prepare_trimmed_data_in_deviations(data::KeyedArray,
                                            𝓂::ℳ,
                                            steady_state::AbstractVector{<:Real};
                                            data_in_levels::Bool = true,
                                            maxlog::Int = DEFAULT_MAXLOG)
    sorted_data = data(sort(axiskeys(data, 1)))
    obs_axis = collect(axiskeys(sorted_data, 1))
    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis
    obs_idx = parse_variables_input_to_index(obs_symbols, 𝓂) |> sort
    raw_data = missing_data_to_nan(sorted_data)
    data_in_deviations = data_in_levels ? raw_data .- steady_state[obs_idx] : raw_data
    trimmed_data, _, _, _ = trim_informative_sample(data_in_deviations; maxlog = maxlog)
    return trimmed_data
end


"""
$(SIGNATURES)
Return the shock decomposition in absolute deviations from the relevant steady state. The non-stochastic steady state (NSSS) is relevant for first order solutions and the stochastic steady state for higher order solutions. The deviations are based on the Kalman smoother or filter (depending on the `smooth` keyword argument) or inversion filter using the provided data and solution of the model. When the defaults are used, the filter is selected automatically—Kalman for first order solutions and inversion otherwise—and smoothing is only enabled when the Kalman filter is active. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

In case of pruned second and pruned third order perturbation algorithms the decomposition additionally contains a term `Nonlinearities`. This term represents the nonlinear interaction between the states in the periods after the shocks arrived and in the case of pruned third order, the interaction between (pruned second order) states and contemporaneous shocks.

Setting `marginal_contribution = true` (only meaningful for `:pruned_second_order` and `:pruned_third_order`) instead replaces the separate `Nonlinearities` column by an Aumann–Shapley allocation of the incremental response above the zero-shock path, using the path-integral identity with Gauss–Legendre quadrature. Equivalently, each shock's column carries its standalone effect plus its marginal-contribution share of the cross-shock interaction, while `Initial_values` remains separate. The implementation starts from the low-order Gauss–Legendre rule (`2 * nᵉ` propagations at second order, `3 * nᵉ` at third order) and incrementally reruns with up to 7 nodes when the relative Shapley-efficiency closure error exceeds `1e-3`. For first-order solutions the option has no effect (silent fallback).

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- $DATA®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $FILTER®
- $ALGORITHM®
- $DATA_IN_LEVELS®
- $SMOOTH®
- `marginal_contribution` [Default: `false`, Type: `Bool`]: if `true` and the algorithm is `:pruned_second_order` or `:pruned_third_order`, replace the separate `Nonlinearities` column by an Aumann–Shapley allocation across shock columns while keeping `Initial_values` separate.
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows, shocks in columns, and periods as the third dimension.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

import Random; Random.seed!(3)

simulation = simulate(RBC)

get_shock_decomposition(RBC,simulation([:c],:,:simulate))
# output
3-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 4-element Vector{Symbol}
→   Shocks ∈ 2-element Vector{Symbol}
◪   Periods ∈ 40-element UnitRange{Int64}
And data, 4×2×40 Array{Float64, 3}:
[showing 3 of 40 slices]
[:, :, 1] ~ (:, :, 1):
        (:eps_z₍ₓ₎)  (:Initial_values)
  (:c)   0.00128797   0.00319151
  (:k)   0.0118536    0.0318
  (:q)   0.0131415    0.00335202
  (:z)   0.00190898   0.000146294

[:, :, 21] ~ (:, :, 21):
        (:eps_z₍ₓ₎)  (:Initial_values)
  (:c)  -0.0428897    0.00132724
  (:k)  -0.425096     0.0132567
  (:q)  -0.0721742    0.00100629
  (:z)  -0.00622294   1.73472e-18

[:, :, 40] ~ (:, :, 40):
        (:eps_z₍ₓ₎)   (:Initial_values)
  (:c)  -0.0116806     0.000573923
  (:k)  -0.116386      0.00573246
  (:q)  -0.012256      0.00043514
  (:z)  -0.000533531   1.0842e-19
```
"""
@unstable function get_shock_decomposition(𝓂::ℳ,
                                data::KeyedArray{D};
                                parameters::ParameterType = nothing,
                                steady_state_function::SteadyStateFunctionType = missing,
                                algorithm::Symbol = DEFAULT_ALGORITHM,
                                filter::Symbol = DEFAULT_FILTER_SELECTOR(algorithm),
                                data_in_levels::Bool = DEFAULT_DATA_IN_LEVELS,
                                warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS,
                                smooth::Bool = DEFAULT_SMOOTH_SELECTOR(filter),
                                marginal_contribution::Bool = false,
                                verbose::Bool = DEFAULT_VERBOSE,
                                tol::Tolerances = Tolerances(),
                                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                                caching::Bool = DEFAULT_CACHING,
                                use_workspaces::Bool = DEFAULT_USE_WORKSPACES)::KeyedArray where {D <: Union{Missing,Nothing,Real}}
    # @nospecialize # reduce compile time

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                                    lyapunov_algorithm = lyapunov_algorithm)

    filter, smooth, algorithm, _, pruning, warmup_iterations = normalize_filtering_options(filter, smooth, algorithm, false, warmup_iterations)

    if marginal_contribution && !pruning
        @info "`marginal_contribution = true` is only meaningful for pruned higher-order solutions (`:pruned_second_order`, `:pruned_third_order`). Setting `marginal_contribution = false` for `algorithm = $(algorithm)`." maxlog = 3
        marginal_contribution = false
    end

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            opts = opts, 
            dynamics = true, 
            algorithm = algorithm)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)

    data_in_deviations = prepare_trimmed_data_in_deviations(data, 𝓂, NSSS; data_in_levels = data_in_levels)

    extra_kw = marginal_contribution ? (; marginal_contribution = true) : NamedTuple()
    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.var_axis
    exo_axis = 𝓂.constants.post_complete_parameters.exo_axis_with_subscript

    if pruning && !marginal_contribution
        axis2 = vcat(exo_axis, :Nonlinearities, :Initial_values)
    else
        axis2 = vcat(exo_axis, :Initial_values)
    end

    if size(data_in_deviations, 2) == 0
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return KeyedArray(zeros(eltype(NSSS), length(axis1), length(axis2), 0); Variables = axis1, Shocks = axis2, Periods = 1:0)
    end

    variables, shocks, standard_deviations, decomposition = filter_data_with_model(𝓂, data_in_deviations, Val(algorithm), Val(filter), 
                                                                                    warmup_iterations = warmup_iterations, 
                                                                                    opts = opts,
                                                                                    smooth = smooth;
                                                                                    extra_kw...)

    if pruning
        if marginal_contribution
            decomposition[:, end - 1, :] .+= SSS_delta
        else
            # decomposition[:,end - 1,:]                  .+= SSS_delta * (size(decomposition,2) - 3)
            decomposition[:,1:(end - 2 - pruning),:]    .+= SSS_delta
            decomposition[:,end - 2,:]                  .-= SSS_delta * (size(decomposition,2) - 4)
        end
    end

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    return KeyedArray(decomposition[:,1:end-1,:];  Variables = axis1, Shocks = axis2, Periods = 1:size(data_in_deviations,2))
end




"""
$(SIGNATURES)
Return the estimated shocks based on the inversion filter (depending on the `filter` keyword argument), or Kalman filter or smoother (depending on the `smooth` keyword argument) using the provided data and (non-)linear solution of the model. By default MacroModelling chooses the Kalman filter for first order solutions and the inversion filter for higher order ones, and only enables smoothing when the Kalman filter is used. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- $DATA®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $FILTER®
- $DATA_IN_LEVELS®
- $SMOOTH®
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with shocks in rows, and periods in columns.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

import Random; Random.seed!(3)

simulation = simulate(RBC)

get_estimated_shocks(RBC,simulation([:c],:,:simulate))
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Shocks ∈ 1-element Vector{Symbol}
→   Periods ∈ 40-element UnitRange{Int64}
And data, 1×40 Matrix{Float64}:
               (1)         (2)        …  (39)         (40)
  (:eps_z₍ₓ₎)    0.190898    1.24786       -0.676457    -0.00870749
```
"""
@unstable function get_estimated_shocks(𝓂::ℳ,
                            data::KeyedArray{D};
                            parameters::ParameterType = nothing,
                            steady_state_function::SteadyStateFunctionType = missing,
                            algorithm::Symbol = DEFAULT_ALGORITHM, 
                            filter::Symbol = DEFAULT_FILTER_SELECTOR(algorithm), 
                            warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS,
                            data_in_levels::Bool = DEFAULT_DATA_IN_LEVELS,
                            smooth::Bool = DEFAULT_SMOOTH_SELECTOR(filter),
                            verbose::Bool = DEFAULT_VERBOSE,
                            tol::Tolerances = Tolerances(),
                            quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                            sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                            lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                            caching::Bool = DEFAULT_CACHING,
                            use_workspaces::Bool = DEFAULT_USE_WORKSPACES)::KeyedArray where {D <: Union{Missing,Nothing,Real}}
    # @nospecialize # reduce compile time

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                            sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                            sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                            lyapunov_algorithm = lyapunov_algorithm)

    filter, smooth, algorithm, _, _, warmup_iterations = normalize_filtering_options(filter, smooth, algorithm, false, warmup_iterations)

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            algorithm = algorithm, 
            opts = opts,
            dynamics = true)
    
    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)

    data_in_deviations = prepare_trimmed_data_in_deviations(data, 𝓂, NSSS; data_in_levels = data_in_levels)

    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.exo_axis_with_subscript

    if size(data_in_deviations, 2) == 0
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return KeyedArray(zeros(eltype(NSSS), length(axis1), 0); Shocks = axis1, Periods = 1:0)
    end

    variables, shocks, standard_deviations, decomposition = filter_data_with_model(𝓂, data_in_deviations, Val(algorithm), Val(filter), 
                                                                                    warmup_iterations = warmup_iterations, 
                                                                                    opts = opts,
                                                                                    smooth = smooth)

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    return KeyedArray(shocks;  Shocks = axis1, Periods = 1:size(data_in_deviations,2))
end






"""
$(SIGNATURES)
Return the estimated variables (in levels by default, see `levels` keyword argument) based on the inversion filter (depending on the `filter` keyword argument), or Kalman filter or smoother (depending on the `smooth` keyword argument) using the provided data and (non-)linear solution of the model. With the default options the Kalman filter is applied to first order solutions, while the inversion filter is used for higher order methods; smoothing is activated automatically only when the Kalman filter is available. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- $DATA®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $FILTER®
- $DATA_IN_LEVELS®
- `levels` [Default: `true`, Type: `Bool`]: $LEVELS®
- $SMOOTH®
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows, and periods in columns.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

import Random; Random.seed!(3)

simulation = simulate(RBC)

get_estimated_variables(RBC,simulation([:c],:,:simulate))
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 4-element Vector{Symbol}
→   Periods ∈ 40-element UnitRange{Int64}
And data, 4×40 Matrix{Float64}:
        (1)           (2)          …  (39)           (40)
  (:c)    5.94073       5.94913          5.9249         5.92515
  (:k)   47.4339       47.5121          47.2781        47.2796
  (:q)    6.90055       6.97596          6.86123        6.87224
  (:z)    0.00205528    0.0128896       -0.00223228    -0.000533531
```
"""
@unstable function get_estimated_variables(𝓂::ℳ,
                                data::KeyedArray{D};
                                parameters::ParameterType = nothing,
                                steady_state_function::SteadyStateFunctionType = missing,
                                algorithm::Symbol = DEFAULT_ALGORITHM, 
                                filter::Symbol = DEFAULT_FILTER_SELECTOR(algorithm), 
                                warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS,
                                data_in_levels::Bool = DEFAULT_DATA_IN_LEVELS,
                                levels::Bool = DEFAULT_LEVELS,
                                smooth::Bool = DEFAULT_SMOOTH_SELECTOR(filter),
                                verbose::Bool = DEFAULT_VERBOSE,
                                tol::Tolerances = Tolerances(),
                                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                                caching::Bool = DEFAULT_CACHING,
                                use_workspaces::Bool = DEFAULT_USE_WORKSPACES)::KeyedArray where {D <: Union{Missing,Nothing,Real}}
    # @nospecialize # reduce compile time                         

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                                lyapunov_algorithm = lyapunov_algorithm)

    filter, smooth, algorithm, _, _, warmup_iterations = normalize_filtering_options(filter, smooth, algorithm, false, warmup_iterations)

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            algorithm = algorithm, 
            opts = opts,
            dynamics = true)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)

    data_in_deviations = prepare_trimmed_data_in_deviations(data, 𝓂, NSSS; data_in_levels = data_in_levels)

    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.var_axis

    if size(data_in_deviations, 2) == 0
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return KeyedArray(zeros(eltype(NSSS), length(axis1), 0); Variables = axis1, Periods = 1:0)
    end

    variables, shocks, standard_deviations, decomposition = filter_data_with_model(𝓂, data_in_deviations, Val(algorithm), Val(filter), 
                                                                                    warmup_iterations = warmup_iterations, 
                                                                                    opts = opts,
                                                                                    smooth = smooth)

    result = KeyedArray(levels ? variables .+ NSSS[1:length(𝓂.constants.post_model_macro.var)] : variables;  Variables = axis1, Periods = 1:size(data_in_deviations,2))

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    return result
end


"""
$(SIGNATURES)
Return the vertical concatenation of `get_estimated_variables` and `get_estimated_shocks`
as a single `KeyedArray` with a common first axis named `Estimates` and the
second axis `Periods`. Variables appear first, followed by shocks.

All keyword arguments are forwarded to the respective functions. See the
docstrings of `get_estimated_variables` and `get_estimated_shocks` for details.

# Arguments
- $MODEL®
- $DATA®

# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $FILTER®
- $DATA_IN_LEVELS®
- `levels` [Default: `true`, Type: `Bool`]: $LEVELS®
- $SMOOTH®
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables followed by shocks in rows, and periods in columns.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

import Random; Random.seed!(3)

simulation = simulate(RBC)

get_model_estimates(RBC,simulation([:c],:,:simulate))
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables_and_shocks ∈ 5-element Vector{Symbol}
→   Periods ∈ 40-element UnitRange{Int64}
And data, 5×40 Matrix{Float64}:
               (1)           (2)          …  (39)           (40)
  (:c)           5.94073       5.94913          5.9249         5.92515
  (:k)          47.4339       47.5121          47.2781        47.2796
  (:q)           6.90055       6.97596          6.86123        6.87224
  (:z)           0.00205528    0.0128896       -0.00223228    -0.000533531
  (:eps_z₍ₓ₎)    0.190898      1.24786    …    -0.676457      -0.00870749
```
"""
@unstable function get_model_estimates(𝓂::ℳ,
                             data::KeyedArray{D};
                             parameters::ParameterType = nothing,
                             steady_state_function::SteadyStateFunctionType = missing,
                             algorithm::Symbol = DEFAULT_ALGORITHM,
                             filter::Symbol = DEFAULT_FILTER_SELECTOR(algorithm),
                             warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS,
                             data_in_levels::Bool = DEFAULT_DATA_IN_LEVELS,
                             levels::Bool = DEFAULT_LEVELS,
                             smooth::Bool = DEFAULT_SMOOTH_SELECTOR(filter),
                             verbose::Bool = DEFAULT_VERBOSE,
                             tol::Tolerances = Tolerances(),
                             quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                             sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                             lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                             caching::Bool = DEFAULT_CACHING,
                             use_workspaces::Bool = DEFAULT_USE_WORKSPACES)::KeyedArray where {D <: Union{Missing,Nothing,Real}}

    vars = get_estimated_variables(𝓂, data;
                                   parameters = parameters,
                                   steady_state_function = steady_state_function,
                                   algorithm = algorithm,
                                   filter = filter,
                                   warmup_iterations = warmup_iterations,
                                   data_in_levels = data_in_levels,
                                   levels = levels,
                                   smooth = smooth,
                                   verbose = verbose,
                                   tol = tol,
                                   quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                   sylvester_algorithm = sylvester_algorithm,
                                   lyapunov_algorithm = lyapunov_algorithm,
                                   caching = caching,
                                   use_workspaces = use_workspaces)

    shks = get_estimated_shocks(𝓂, data;
                                parameters = parameters,
                                steady_state_function = steady_state_function,
                                algorithm = algorithm,
                                filter = filter,
                                warmup_iterations = warmup_iterations,
                                data_in_levels = data_in_levels,
                                smooth = smooth,
                                verbose = verbose,
                                tol = tol,
                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                sylvester_algorithm = sylvester_algorithm,
                                lyapunov_algorithm = lyapunov_algorithm,
                                caching = caching,
                                use_workspaces = use_workspaces)

    # Build unified first axis and concatenate data
    est_labels = vcat(collect(axiskeys(vars, 1)), collect(axiskeys(shks, 1)))
    est_data = vcat(Matrix(vars), Matrix(shks))

    return KeyedArray(est_data; Variables_and_shocks = est_labels, Periods = axiskeys(vars, 2))
end



"""
$(SIGNATURES)
Return the standard deviations of the Kalman smoother or filter (depending on the `smooth` keyword argument) estimates of the model variables based on the provided data and first order solution of the model. For the default settings this function relies on the Kalman filter and therefore keeps smoothing enabled. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- $DATA®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $DATA_IN_LEVELS®
- $SMOOTH®
- $QME®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with standard deviations in rows, and periods in columns.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

import Random; Random.seed!(3)

simulation = simulate(RBC)

get_estimated_variable_standard_deviations(RBC,simulation([:c],:,:simulate))
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Standard_deviations ∈ 4-element Vector{Symbol}
→   Periods ∈ 40-element UnitRange{Int64}
And data, 4×40 Matrix{Float64}:
        (1)           (2)            …  (39)            (40)
  (:c)    1.31709e-9    1.16415e-10        8.23181e-11     0.0
  (:k)    0.00509299    0.000382934        9.31323e-10     1.6131e-9
  (:q)    0.0612887     0.0046082          9.31323e-10     9.31323e-10
  (:z)    0.00961766    0.000723136        0.0             1.64636e-10
```
"""
@unstable function get_estimated_variable_standard_deviations(𝓂::ℳ,
                                                    data::KeyedArray{D};
                                                    parameters::ParameterType = nothing,
                                                    steady_state_function::SteadyStateFunctionType = missing,
                                                    data_in_levels::Bool = DEFAULT_DATA_IN_LEVELS,
                                                    smooth::Bool = DEFAULT_SMOOTH_FLAG,
                                                    verbose::Bool = DEFAULT_VERBOSE,
                                                    tol::Tolerances = Tolerances(),
                                                    quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                                                    lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                                                    caching::Bool = DEFAULT_CACHING,
                                                    use_workspaces::Bool = DEFAULT_USE_WORKSPACES) where {D <: Union{Missing,Nothing,Real}}
    # @nospecialize # reduce compile time                                               

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    lyapunov_algorithm = lyapunov_algorithm)

    algorithm = :first_order

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            opts = opts,
            dynamics = true)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)

    data_in_deviations = prepare_trimmed_data_in_deviations(data, 𝓂, NSSS; data_in_levels = data_in_levels)

    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.var_axis

    if size(data_in_deviations, 2) == 0
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return KeyedArray(zeros(eltype(NSSS), length(axis1), 0); Standard_deviations = axis1, Periods = 1:0)
    end

    variables, shocks, standard_deviations, decomposition = filter_data_with_model(𝓂, data_in_deviations, Val(:first_order), Val(:kalman), 
                                                                                    smooth = smooth, 
                                                                                    opts = opts)

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    return KeyedArray(standard_deviations;  Standard_deviations = axis1, Periods = 1:size(data_in_deviations,2))
end





"""
$(SIGNATURES)
Return the conditional forecast given restrictions on endogenous variables and shocks (optional). By default, the values represent absolute deviations from the relevant steady state (see `levels` for details). The non-stochastic steady state (NSSS) is relevant for first order solutions and the stochastic steady state for higher order solutions. A constrained minimisation problem is solved to find the combination of shocks with the smallest squared magnitude fulfilling the conditions.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- $CONDITIONS®
# Keyword Arguments
- $SHOCK_CONDITIONS®
- $INITIAL_STATE®
- `periods` [Default: `40`, Type: `Int`]: the total number of periods is the sum of the argument provided here and the maximum of periods of the shocks or conditions argument.
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $(VARIABLES®(DEFAULT_VARIABLES_EXCLUDING_OBC))
- $CONDITIONS_IN_LEVELS®
- `levels` [Default: `false`, Type: `Bool`]: $LEVELS®
- $ALGORITHM®
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables  and shocks in rows, and periods in columns.

# Examples
```jldoctest
using MacroModelling
using SparseArrays, AxisKeys

@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME  begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

# c is conditioned to deviate by 0.01 in period 1 and y is conditioned to deviate by 0.02 in period 2
conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,2),Variables = [:c,:y], Periods = 1:2)
conditions[1,1] = .01
conditions[2,2] = .02

# in period 2 second shock (eps_z) is conditioned to take a value of 0.05
shocks = Matrix{Union{Nothing,Float64}}(undef,2,1)
shocks[1,1] = .05

get_conditional_forecast(RBC_CME, conditions, shocks = shocks, conditions_in_levels = false)
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables_and_shocks ∈ 9-element Vector{Symbol}
→   Periods ∈ 42-element UnitRange{Int64}
And data, 9×42 Matrix{Float64}:
                   (1)            …  (41)            (42)
  (:A)               0.0313639          0.000221372     0.000199235
  (:Pi)              0.000780257       -0.000146071    -0.000140137
  (:R)               0.00117156        -0.000219325    -0.000210417
  (:c)               0.01               0.00213278      0.00203751
  (:k)               0.034584     …     0.0397631       0.0380482
  (:y)               0.0446375          0.00129544      0.001222
  (:z_delta)         0.00025            3.69522e-6      3.3257e-6
  (:delta_eps₍ₓ₎)    0.05               0.0             0.0
  (:eps_z₍ₓ₎)        4.61234            0.0             0.0
```

The same can be achieved with the other input formats:
```julia
# conditions = Matrix{Union{Nothing,Float64}}(undef,7,2)
# conditions[4,1] = .01
# conditions[6,2] = .02

# using SparseArrays
# conditions = spzeros(7,2)
# conditions[4,1] = .01
# conditions[6,2] = .02

# shocks = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,1),Variables = [:delta_eps], Periods = [1])
# shocks[1,1] = .05

# using SparseArrays
# shocks = spzeros(2,1)
# shocks[1,1] = .05
```
"""
@unstable function get_conditional_forecast(𝓂::ℳ,
                                conditions::Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}};
                                shocks::Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}, Nothing} = nothing, 
                                initial_state::Union{Vector{Vector{Float64}},Vector{Float64}} = DEFAULT_INITIAL_STATE,
                                periods::Int = DEFAULT_PERIODS, 
                                parameters::ParameterType = nothing,
                                steady_state_function::SteadyStateFunctionType = missing,
                                variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_OBC, 
                                conditions_in_levels::Bool = DEFAULT_CONDITIONS_IN_LEVELS,
                                algorithm::Symbol = DEFAULT_ALGORITHM,
                                levels::Bool = false,
                                verbose::Bool = DEFAULT_VERBOSE,
                                tol::Tolerances = Tolerances(),
                                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                                conditional_forecast_solver::Symbol = :LagrangeNewton,
                                caching::Bool = DEFAULT_CACHING,
                                use_workspaces::Bool = DEFAULT_USE_WORKSPACES)
    # @nospecialize # reduce compile time                        

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                                lyapunov_algorithm = lyapunov_algorithm)

    periods += max(size(conditions,2), shocks isa Nothing ? 1 : size(shocks,2)) # isa Nothing needed otherwise JET tests fail

    if conditions isa SparseMatrixCSC{Float64}
        @assert length(𝓂.constants.post_model_macro.var) == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(length(𝓂.constants.post_model_macro.var)) * " variables (including auxiliary variables): " * repr(𝓂.constants.post_model_macro.var)

        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.constants.post_model_macro.var),periods)
        nzs = findnz(conditions)
        for i in 1:length(nzs[1])
            cond_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
        end
        conditions = cond_tmp
    elseif conditions isa Matrix{Union{Nothing,Float64}}
        @assert length(𝓂.constants.post_model_macro.var) == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(length(𝓂.constants.post_model_macro.var)) * " variables (including auxiliary variables): " * repr(𝓂.constants.post_model_macro.var)

        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.constants.post_model_macro.var),periods)
        cond_tmp[:,axes(conditions,2)] = conditions
        conditions = cond_tmp
    elseif conditions isa KeyedArray{Union{Nothing,Float64}} || conditions isa KeyedArray{Float64}
        conditions_axis = collect(axiskeys(conditions,1))

        conditions_symbols = conditions_axis isa String_input ? conditions_axis .|> Meta.parse .|> replace_indices : conditions_axis

        @assert length(setdiff(conditions_symbols, 𝓂.constants.post_model_macro.var)) == 0 "The following symbols in the first axis of the conditions matrix are not part of the model: " * repr(setdiff(conditions_symbols,𝓂.constants.post_model_macro.var))
        
        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.constants.post_model_macro.var),periods)
        cond_tmp[indexin(sort(conditions_symbols),𝓂.constants.post_model_macro.var),axes(conditions,2)] .= conditions(sort(axiskeys(conditions,1)))
        conditions = cond_tmp
    end
    
    if shocks isa SparseMatrixCSC{Float64}
        @assert length(𝓂.constants.post_model_macro.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.constants.post_model_macro.exo)) * " shocks: " * repr(𝓂.constants.post_model_macro.exo)

        shocks_tmp = Matrix{Union{Nothing,Number}}(nothing,length(𝓂.constants.post_model_macro.exo),periods)
        nzs = findnz(shocks)
        for i in 1:length(nzs[1])
            shocks_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
        end
        shocks = shocks_tmp
    elseif shocks isa Matrix{Union{Nothing,Float64}}
        @assert length(𝓂.constants.post_model_macro.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.constants.post_model_macro.exo)) * " shocks: " * repr(𝓂.constants.post_model_macro.exo)

        shocks_tmp = Matrix{Union{Nothing,Number}}(nothing,length(𝓂.constants.post_model_macro.exo),periods)
        shocks_tmp[:,axes(shocks,2)] = shocks
        shocks = shocks_tmp
    elseif shocks isa KeyedArray{Union{Nothing,Float64}} || shocks isa KeyedArray{Float64}
        shocks_axis = collect(axiskeys(shocks,1))

        shocks_symbols = shocks_axis isa String_input ? shocks_axis .|> Meta.parse .|> replace_indices : shocks_axis

        @assert length(setdiff(shocks_symbols,𝓂.constants.post_model_macro.exo)) == 0 "The following symbols in the first axis of the shocks matrix are not part of the model: " * repr(setdiff(shocks_symbols, 𝓂.constants.post_model_macro.exo))
        
        shocks_tmp = Matrix{Union{Nothing,Number}}(nothing,length(𝓂.constants.post_model_macro.exo),periods)
        shocks_tmp[indexin(sort(shocks_symbols), 𝓂.constants.post_model_macro.exo), axes(shocks,2)] .= shocks(sort(axiskeys(shocks,1)))
        shocks = shocks_tmp
    elseif isnothing(shocks)
        shocks = Matrix{Union{Nothing,Number}}(nothing,length(𝓂.constants.post_model_macro.exo),periods)
    end

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            opts = opts,
            dynamics = true, 
            algorithm = algorithm)

    state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, false)

    reference_steady_state, _, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)
    initial_state = adjust_initial_state(initial_state, algorithm, 𝓂, SSS_delta, reference_steady_state)

    var_idx = parse_variables_input_to_index(variables, 𝓂) |> sort

    Y = zeros(size(𝓂.caches.first_order_solution_matrix,1),periods)

    cond_var_idx = findall(conditions[:,1] .!= nothing)
    
    free_shock_idx = findall(shocks[:,1] .== nothing)

    shocks[free_shock_idx,1] .= 0
    
    if conditions_in_levels
        conditions[cond_var_idx,1] .-= reference_steady_state[cond_var_idx] + SSS_delta[cond_var_idx]
    else
        conditions[cond_var_idx,1] .-= SSS_delta[cond_var_idx]
    end

    @assert length(free_shock_idx) >= length(cond_var_idx) "Exact matching only possible with at least as many free shocks than conditioned variables. Period 1 has " * repr(length(free_shock_idx)) * " free shock(s) and " * repr(length(cond_var_idx)) * " conditioned variable(s)."

    if algorithm ∈ [:second_order, :third_order, :pruned_second_order, :pruned_third_order]
        S₁ = 𝓂.caches.first_order_solution_matrix
        Ŝ₁ = [S₁[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] zeros(𝓂.constants.post_model_macro.nVars) S₁[:,𝓂.constants.post_model_macro.nPast_not_future_and_mixed+1:end]]

        S₂ = nothing
        if size(𝓂.caches.second_order_solution, 2) > 0
            S₂ = 𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂
        end

        S₃ = nothing
        if algorithm ∈ [:third_order, :pruned_third_order] && size(𝓂.caches.third_order_solution, 2) > 0
            S₃ = 𝓂.caches.third_order_solution * 𝓂.constants.third_order.𝐔₃
        end

        ensure_conditional_forecast_constants!(𝓂.constants; third_order = !isnothing(S₃))

        # Use Lagrange-Newton algorithm to find shocks
        x, matched = find_shocks_conditional_forecast(Val(conditional_forecast_solver),
                                                      initial_state,
                                                      Float64[shocks[:,1]...],
                                                      Float64[conditions[cond_var_idx,1]...],
                                                      cond_var_idx,
                                                      free_shock_idx,
                                                      state_update,
                                                      Ŝ₁,
                                                      S₂,
                                                      S₃,
                                                      𝓂.constants,
                                                      𝓂.workspaces.find_shocks;
                                                      verbose = verbose)

        @assert matched "Numerical stabiltiy issues for restrictions in period 1."
    
        shocks[free_shock_idx,1] .= x
                
        initial_state = state_update(initial_state, Float64[shocks[:,1]...])

        Y[:,1] = pruning ? sum(initial_state) : initial_state

        for i in 2:size(conditions,2)
            cond_var_idx = findall(conditions[:,i] .!= nothing)
            
            if conditions_in_levels
                conditions[cond_var_idx,i] .-= reference_steady_state[cond_var_idx] + SSS_delta[cond_var_idx]
            else
                conditions[cond_var_idx,i] .-= SSS_delta[cond_var_idx]
            end
    
            free_shock_idx = findall(shocks[:,i] .== nothing)

            shocks[free_shock_idx,i] .= 0
    
            @assert length(free_shock_idx) >= length(cond_var_idx) "Exact matching only possible with at least as many free shocks than conditioned variables. Period " * repr(i) * " has " * repr(length(free_shock_idx)) * " free shock(s) and " * repr(length(cond_var_idx)) * " conditioned variable(s)."
    
            if length(cond_var_idx) == 0
                # No conditions this period: set free shocks to zero
                shocks[free_shock_idx,i] .= 0
            else
                # Use Lagrange-Newton algorithm to find shocks
                x, matched = find_shocks_conditional_forecast(Val(conditional_forecast_solver),
                                                              pruning ? initial_state : Y[:,i-1],
                                                              Float64[shocks[:,i]...],
                                                              Float64[conditions[cond_var_idx,i]...],
                                                              cond_var_idx,
                                                              free_shock_idx,
                                                              state_update,
                                                              Ŝ₁,
                                                              S₂,
                                                              S₃,
                                                              𝓂.constants,
                                                              𝓂.workspaces.find_shocks;
                                                              verbose = verbose)

                @assert matched "Numerical stabiltiy issues for restrictions in period $i."

                shocks[free_shock_idx,i] .= x
            end

            initial_state = state_update(initial_state, Float64[shocks[:,i]...])

            Y[:,i] = pruning ? sum(initial_state) : initial_state
        end
    elseif algorithm == :first_order
        C = 𝓂.caches.first_order_solution_matrix[:,𝓂.constants.post_model_macro.nPast_not_future_and_mixed+1:end]::Matrix{Float64}
    
        CC = C[cond_var_idx,free_shock_idx]

        if length(cond_var_idx) == 1
            @assert any(CC .!= 0) "Free shocks have no impact on conditioned variable in period 1."
            shocks[free_shock_idx,1] .= 0
            shocks[free_shock_idx,1] = CC \ (conditions[cond_var_idx,1] - state_update(initial_state, Float64[shocks[:,1]...])[cond_var_idx])
        elseif length(free_shock_idx) == length(cond_var_idx)
            CC_lu_ws = FastLapackInterface.LUWs(CC)
            CC_lu_ws, _, ok, CC_lu_handle = factorize_lu!(Val(:FastLapack), CC, CC_lu_ws, size(CC))

            @assert ok "Numerical stabiltiy issues for restrictions in period 1."

            CC_rhs = conditions[cond_var_idx,1] - state_update(initial_state, Float64[shocks[:,1]...])[cond_var_idx]
            solve_lu_left!(CC, CC_rhs, CC_lu_ws, CC_lu_handle)
            shocks[free_shock_idx,1] .= 0
            shocks[free_shock_idx,1] = CC_rhs
        else
            shocks[free_shock_idx,1] .= 0
            shocks[free_shock_idx,1] = CC \ (conditions[cond_var_idx,1] - state_update(initial_state, Float64[shocks[:,1]...])[cond_var_idx])
        end
    
        Y[:,1] = state_update(initial_state, Float64[shocks[:,1]...])

        for i in 2:size(conditions,2)
            cond_var_idx = findall(conditions[:,i] .!= nothing)
            
            if conditions_in_levels
                conditions[cond_var_idx,i] .-= reference_steady_state[cond_var_idx]
            end
    
            free_shock_idx = findall(shocks[:,i] .== nothing)
            shocks[free_shock_idx,i] .= 0
    
            @assert length(free_shock_idx) >= length(cond_var_idx) "Exact matching only possible with more free shocks than conditioned variables. Period " * repr(i) * " has " * repr(length(free_shock_idx)) * " free shock(s) and " * repr(length(cond_var_idx)) * " conditioned variable(s)."
    
            CC = C[cond_var_idx,free_shock_idx]
    
            if length(cond_var_idx) == 1
                @assert any(CC .!= 0) "Free shocks have no impact on conditioned variable in period " * repr(i) * "."
                shocks[free_shock_idx,i] = CC \ (conditions[cond_var_idx,i] - state_update(Y[:,i-1], Float64[shocks[:,i]...])[cond_var_idx])
            elseif length(free_shock_idx) == length(cond_var_idx)
                CC_lu_ws = FastLapackInterface.LUWs(CC)
                CC_lu_ws, _, ok, CC_lu_handle = factorize_lu!(Val(:FastLapack), CC, CC_lu_ws, size(CC))

                @assert ok "Numerical stabiltiy issues for restrictions in period " * repr(i) * "."

                CC_rhs = conditions[cond_var_idx,i] - state_update(Y[:,i-1], Float64[shocks[:,i]...])[cond_var_idx]
                solve_lu_left!(CC, CC_rhs, CC_lu_ws, CC_lu_handle)
                shocks[free_shock_idx,i] = CC_rhs
            else
                shocks[free_shock_idx,i] = CC \ (conditions[cond_var_idx,i] - state_update(Y[:,i-1], Float64[shocks[:,i]...])[cond_var_idx])
            end
    
            Y[:,i] = state_update(Y[:,i-1], Float64[shocks[:,i]...])
        end
    end

    axis1 = [𝓂.constants.post_model_macro.var[var_idx]; 𝓂.constants.post_model_macro.exo]

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        axis1[end-length(𝓂.constants.post_model_macro.exo)+1:end] = axis1[end-length(𝓂.constants.post_model_macro.exo)+1:end] .* "₍ₓ₎"
    else
        axis1 = [𝓂.constants.post_model_macro.var[var_idx]; map(x->Symbol(string(x) * "₍ₓ₎"), 𝓂.constants.post_model_macro.exo)]
    end

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    return KeyedArray([Y[var_idx,:] .+ (levels ? reference_steady_state + SSS_delta : SSS_delta)[var_idx]; convert(Matrix{Float64}, shocks)];  Variables_and_shocks = axis1, Periods = 1:periods)
end


"""
$(SIGNATURES)
Return impulse response functions (IRFs) of the model.
Function to use when differentiating IRFs with respect to parameters.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- $PARAMETER_VALUES®
# Keyword Arguments
- $STEADY_STATE_FUNCTION®
- $PERIODS®
- $(VARIABLES®(DEFAULT_VARIABLES_EXCLUDING_OBC))
- $SHOCKS®
- $NEGATIVE_SHOCK®
- $INITIAL_STATE®1
- `levels` [Default: `false`, Type: `Bool`]: $LEVELS®
- $ALGORITHM®
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Array{<:AbstractFloat, 3}` with variables in rows, periods in columns, and shocks as the third dimension.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_irf(RBC, RBC.parameter_values)
# output
4×40×1 Array{Float64, 3}:
[:, :, 1] =
 0.00674687  0.00729773  0.00715114  0.00687615  …  0.00146962   0.00140619
 0.0620937   0.0718322   0.0712153   0.0686381      0.0146789    0.0140453
 0.0688406   0.0182781   0.00797091  0.0057232      0.00111425   0.00106615
 0.01        0.002       0.0004      8.0e-5         2.74878e-29  5.49756e-30
```
"""

# ── IRF helpers: algorithm-dispatched initial state and forward simulation ──

# Extract/compute initial state for IRF from SSS output
function irf_initial_state(::Val{:first_order}, state, SS_and_pars, initial_state::Vector{Float64}, nVars::Int, ::Type{S}) where S
    initial_state == [0.0] ? zeros(S, nVars) : convert(Vector{S}, initial_state) - SS_and_pars[1:nVars]
end

function irf_initial_state(::Val{:pruned_second_order}, state, SS_and_pars, initial_state::Vector{Float64}, nVars::Int, ::Type{S}) where S
    initial_state == [0.0] ? state : [convert(Vector{S}, initial_state) - SS_and_pars[1:nVars], state[2]]
end

function irf_initial_state(::Val{:pruned_third_order}, state, SS_and_pars, initial_state::Vector{Float64}, nVars::Int, ::Type{S}) where S
    initial_state == [0.0] ? state : [convert(Vector{S}, initial_state) - SS_and_pars[1:nVars], state[2], state[3]]
end

function irf_initial_state(::Val{:second_order}, state, SS_and_pars, initial_state::Vector{Float64}, nVars::Int, ::Type{S}) where S
    initial_state == [0.0] ? (state isa Vector{<:Vector} ? state[1] : state) : convert(Vector{S}, initial_state) - SS_and_pars[1:nVars]
end

function irf_initial_state(::Val{:third_order}, state, SS_and_pars, initial_state::Vector{Float64}, nVars::Int, ::Type{S}) where S
    initial_state == [0.0] ? (state isa Vector{<:Vector} ? state[1] : state) : convert(Vector{S}, initial_state) - SS_and_pars[1:nVars]
end


# Forward simulation storing intermediate states and shocks
function irf_forward_simulate!(::Val{:first_order},
        Y_all::Array{S,3}, states_store, shocks_store,
        init_st, shock_idx, shocks_input, negative_shock, shock_history,
        nExo, periods, past_idx, nVars, 𝐒) where S
    sol_mat = 𝐒
    for (si, ii) in enumerate(shock_idx)
        shock_hist = zeros(nExo, periods)
        if shocks_input isa Union{Symbol_input,String_input}
            shocks_input ≠ :none && (shock_hist[ii, 1] = negative_shock ? -1 : 1)
        else
            shock_hist = shock_history
        end
        states_store[si, 1] = init_st
        for t in 1:periods
            shocks_store[si, t] = shock_hist[:, t]
            prev = states_store[si, t]
            y_t = sol_mat * [prev[past_idx]; shocks_store[si, t]]
            states_store[si, t+1] = y_t
            Y_all[:, t, si] = y_t
        end
    end
end

function irf_forward_simulate!(::Val{:pruned_second_order},
        Y_all::Array{S,3}, states_store, shocks_store,
        init_st, shock_idx, shocks_input, negative_shock, shock_history,
        nExo, periods, past_idx, nVars, 𝐒) where S
    𝐒₁, 𝐒₂ = 𝐒
    for (si, ii) in enumerate(shock_idx)
        shock_hist = zeros(nExo, periods)
        if shocks_input isa Union{Symbol_input,String_input}
            shocks_input ≠ :none && (shock_hist[ii, 1] = negative_shock ? -1 : 1)
        else
            shock_hist = shock_history
        end
        states_store[si, 1] = init_st
        for t in 1:periods
            shocks_store[si, t] = shock_hist[:, t]
            new_st = pruned_second_order_state_update(states_store[si, t], shocks_store[si, t], past_idx, nVars, 𝐒₁, 𝐒₂)
            states_store[si, t+1] = new_st
            Y_all[:, t, si] = sum(new_st)
        end
    end
end

function irf_forward_simulate!(::Val{:pruned_third_order},
        Y_all::Array{S,3}, states_store, shocks_store,
        init_st, shock_idx, shocks_input, negative_shock, shock_history,
        nExo, periods, past_idx, nVars, 𝐒) where S
    𝐒₁, 𝐒₂, 𝐒₃ = 𝐒
    for (si, ii) in enumerate(shock_idx)
        shock_hist = zeros(nExo, periods)
        if shocks_input isa Union{Symbol_input,String_input}
            shocks_input ≠ :none && (shock_hist[ii, 1] = negative_shock ? -1 : 1)
        else
            shock_hist = shock_history
        end
        states_store[si, 1] = init_st
        for t in 1:periods
            shocks_store[si, t] = shock_hist[:, t]
            new_st = pruned_third_order_state_update(states_store[si, t], shocks_store[si, t], past_idx, nVars, 𝐒₁, 𝐒₂, 𝐒₃)
            states_store[si, t+1] = new_st
            Y_all[:, t, si] = sum(new_st)
        end
    end
end

function irf_forward_simulate!(::Val{:second_order},
        Y_all::Array{S,3}, states_store, shocks_store,
        init_st, shock_idx, shocks_input, negative_shock, shock_history,
        nExo, periods, past_idx, nVars, 𝐒) where S
    𝐒₁, 𝐒₂ = 𝐒
    for (si, ii) in enumerate(shock_idx)
        shock_hist = zeros(nExo, periods)
        if shocks_input isa Union{Symbol_input,String_input}
            shocks_input ≠ :none && (shock_hist[ii, 1] = negative_shock ? -1 : 1)
        else
            shock_hist = shock_history
        end
        states_store[si, 1] = init_st
        for t in 1:periods
            shocks_store[si, t] = shock_hist[:, t]
            prev = states_store[si, t]
            aug = [prev[past_idx]; one(S); shocks_store[si, t]]
            y_t = 𝐒₁ * aug + 𝐒₂ * ℒ.kron(aug, aug) / 2
            states_store[si, t+1] = y_t
            Y_all[:, t, si] = y_t
        end
    end
end

function irf_forward_simulate!(::Val{:third_order},
        Y_all::Array{S,3}, states_store, shocks_store,
        init_st, shock_idx, shocks_input, negative_shock, shock_history,
        nExo, periods, past_idx, nVars, 𝐒) where S
    𝐒₁, 𝐒₂, 𝐒₃ = 𝐒
    for (si, ii) in enumerate(shock_idx)
        shock_hist = zeros(nExo, periods)
        if shocks_input isa Union{Symbol_input,String_input}
            shocks_input ≠ :none && (shock_hist[ii, 1] = negative_shock ? -1 : 1)
        else
            shock_hist = shock_history
        end
        states_store[si, 1] = init_st
        for t in 1:periods
            shocks_store[si, t] = shock_hist[:, t]
            prev = states_store[si, t]
            aug = [prev[past_idx]; one(S); shocks_store[si, t]]
            kaug = ℒ.kron(aug, aug)
            y_t = 𝐒₁ * aug + 𝐒₂ * kaug / 2 + 𝐒₃ * ℒ.kron(kaug, aug) / 6
            states_store[si, t+1] = y_t
            Y_all[:, t, si] = y_t
        end
    end
end


function get_irf(𝓂::ℳ,
                    parameters::Vector{S};
                    steady_state_function::SteadyStateFunctionType = missing,
                    periods::Int = DEFAULT_PERIODS,
                    algorithm::Symbol = :first_order,
                    variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_OBC,
                    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = DEFAULT_SHOCK_SELECTION,
                    negative_shock::Bool = DEFAULT_NEGATIVE_SHOCK, 
                    initial_state::Vector{Float64} = DEFAULT_INITIAL_STATE,
                    levels::Bool = false,
                    verbose::Bool = DEFAULT_VERBOSE,
                    tol::Tolerances = Tolerances(),
                    quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                    sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                    lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                    caching::Bool = DEFAULT_CACHING,
                    use_workspaces::Bool = DEFAULT_USE_WORKSPACES)::Array{S,3} where S <: Real

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
        sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
        sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
        lyapunov_algorithm = lyapunov_algorithm)

    estimation = true

    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)

    solve!(𝓂, 
           steady_state_function = steady_state_function,
           opts = opts,
           algorithm = algorithm)

    shocks = 𝓂.constants.post_model_macro.nExo == 0 ? :none : shocks

    @assert shocks != :simulate "Use parameters as a known argument to simulate the model."

    shocks, negative_shock, _, periods, shock_idx, shock_history = process_shocks_input(shocks, negative_shock, 1.0, periods, 𝓂)

    var_idx = parse_variables_input_to_index(variables, 𝓂) |> sort

    nVars = 𝓂.constants.post_model_macro.nVars
    past_idx = 𝓂.constants.post_model_macro.past_not_future_and_mixed_idx
    nPast = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed

    constants_obj, SS_and_pars, 𝐒, state, solved = get_relevant_steady_state_and_state_update(Val(algorithm), parameters, 𝓂, opts = opts, estimation = estimation)

    if !solved
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return fill(S(NaN), length(var_idx), periods, shocks == :none ? 1 : length(shock_idx))
    end

    nExo = 𝓂.constants.post_model_macro.nExo
    nShocks = shocks == :none ? 1 : length(shock_idx)

    # Dispatched initial state and forward simulation
    val_alg = Val(algorithm)
    init_state = irf_initial_state(val_alg, state, SS_and_pars, initial_state, nVars, S)

    Y_all = zeros(S, nVars, periods, nShocks)
    states_store = Array{Any}(undef, nShocks, periods + 1)
    shocks_store = Array{Vector{S}}(undef, nShocks, periods)

    irf_forward_simulate!(val_alg, Y_all, states_store, shocks_store,
        init_state, shock_idx, shocks, negative_shock, shock_history,
        nExo, periods, past_idx, nVars, 𝐒)

    reference_steady_state = SS_and_pars[1:nVars]
    deviations = Y_all[var_idx, :, :]

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    if levels
        return deviations .+ reference_steady_state[var_idx]
    else
        return deviations
    end
end




"""
$(SIGNATURES)
Return impulse response functions (IRFs) of the model. By default, the values represent absolute deviations from the relevant steady state (see `levels` for details). The non-stochastic steady state (NSSS) is relevant for first order solutions and the stochastic steady state for higher order solutions.

If the model contains occasionally binding constraints and `ignore_obc = false` they are enforced using shocks.

# Arguments
- $MODEL®
# Keyword Arguments
- $PERIODS®
- $ALGORITHM®
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $(VARIABLES®(DEFAULT_VARIABLES_EXCLUDING_OBC))
- $SHOCKS®
- $NEGATIVE_SHOCK®
- $GENERALISED_IRF®
- $GENERALISED_IRF_WARMUP_ITERATIONS®
- $GENERALISED_IRF_DRAWS®
- $INITIAL_STATE®
- `levels` [Default: `false`, Type: `Bool`]: $LEVELS®
- $SHOCK_SIZE®
- $IGNORE_OBC®
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows, periods in columns, and shocks as the third dimension.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_irf(RBC)
# output
3-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 4-element Vector{Symbol}
→   Periods ∈ 40-element UnitRange{Int64}
◪   Shocks ∈ 1-element Vector{Symbol}
And data, 4×40×1 Array{Float64, 3}:
[:, :, 1] ~ (:, :, :eps_z):
        (1)           (2)           …  (39)            (40)
  (:c)    0.00674687    0.00729773        0.00146962      0.00140619
  (:k)    0.0620937     0.0718322         0.0146789       0.0140453
  (:q)    0.0688406     0.0182781         0.00111425      0.00106615
  (:z)    0.01          0.002             2.74878e-29     5.49756e-30
```
"""
# Balanced growth path: map each variable to its solved additive per-period growth
# `xᴳ` (the deterministic BGP drift), read from the last non-stochastic steady-state
# solution. Keys are base variable names; trending lead/lag auxiliaries (`xᴸ⁽…⁾`)
# inherit the base variable's growth (resolved at the call site by stripping the
# auxiliary suffix). Returns only nonzero growths; empty for stationary models.
function bgp_growth_by_name(𝓂::ℳ)::Dict{Symbol, Float64}
    growths = Dict{Symbol, Float64}()
    names = 𝓂.constants.post_complete_parameters.nsss_sol_names
    sol   = 𝓂.workspaces.nsss_solver.sol_vec_buffer
    (isempty(names) || length(sol) < length(names)) && return growths
    for (i, n) in enumerate(names)
        s = string(n)
        if endswith(s, "ᴳ") && sol[i] != 0
            growths[Symbol(chop(s, tail = 1))] = sol[i]
        end
    end
    return growths
end

function is_bgp_model(𝓂::ℳ)::Bool
    has_growth_unknowns = any(
        endswith(string(name), "ᴳ")
        for name in 𝓂.constants.post_model_macro.vars_in_ss_equations
    )
    has_growth_unknowns || has_nonstationary_persistence(
        𝓂.constants.post_model_macro,
        𝓂.constants.post_complete_parameters.parameters,
        𝓂.parameter_values,
    )
end

function bgp_growth_rate(𝓂::ℳ, name::Symbol)::Float64
    base = Symbol(replace(string(name), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
    get(bgp_growth_by_name(𝓂), base, 0.0)
end

function bgp_growth_column(𝓂::ℳ, names)::Vector{Float64}
    rates = bgp_growth_by_name(𝓂)
    return [get(rates, Symbol(replace(string(name), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 0.0) for name in names]
end

function bgp_difference_labels(𝓂::ℳ, names)
    rates = bgp_growth_by_name(𝓂)
    return [
        get(rates, Symbol(replace(string(name), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 0.0) == 0.0 ?
        name : Symbol("Delta_" * string(name))
        for name in names
    ]
end

function bgp_difference_covariance(covar::AbstractMatrix{R}, sol::AbstractMatrix{R}, 𝓂::ℳ)::Matrix{R} where R <: Real
    !is_bgp_model(𝓂) && return Matrix(covar)

    T = 𝓂.constants.post_model_macro
    n_state = T.nPast_not_future_and_mixed
    size(sol, 1) == T.nVars && size(sol, 2) == n_state + T.nExo ||
        return Matrix(covar)

    state_idx = T.past_not_future_and_mixed_idx
    state_covariance = Matrix(covar[state_idx, state_idx])
    state_covariance .= ifelse.(isfinite.(state_covariance), state_covariance, zero(R))

    states_to_variables = Matrix(sol[:, 1:n_state])
    shocks_to_variables = Matrix(sol[:, n_state + 1:end])
    states_to_differences = copy(states_to_variables)
    rates = bgp_growth_by_name(𝓂)

    for (row, name) in enumerate(T.var)
        base = Symbol(replace(string(name), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
        get(rates, base, 0.0) == 0.0 && continue
        state_position = findfirst(==(name), T.past_not_future_and_mixed)
        state_position === nothing && continue
        states_to_differences[row, state_position] -= one(R)
    end

    return states_to_differences * state_covariance * states_to_differences' +
           shocks_to_variables * shocks_to_variables'
end

function bgp_difference_covariance_pullback(cotangent::AbstractMatrix{R},
                                            covar::AbstractMatrix{R},
                                            sol::AbstractMatrix{R},
                                            𝓂::ℳ)::Tuple{Matrix{R}, Matrix{R}} where R <: Real
    !is_bgp_model(𝓂) && return Matrix(cotangent), zeros(R, size(sol))

    T = 𝓂.constants.post_model_macro
    n_state = T.nPast_not_future_and_mixed
    size(sol, 1) == T.nVars && size(sol, 2) == n_state + T.nExo ||
        return Matrix(cotangent), zeros(R, size(sol))

    state_idx = T.past_not_future_and_mixed_idx
    state_covariance = Matrix(covar[state_idx, state_idx])
    finite_state_covariance = ifelse.(isfinite.(state_covariance), state_covariance, zero(R))
    states_to_differences = Matrix(sol[:, 1:n_state])
    rates = bgp_growth_by_name(𝓂)

    for (row, name) in enumerate(T.var)
        base = Symbol(replace(string(name), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
        get(rates, base, 0.0) == 0.0 && continue
        state_position = findfirst(==(name), T.past_not_future_and_mixed)
        state_position === nothing && continue
        states_to_differences[row, state_position] -= one(R)
    end

    covariance_cotangent = zeros(R, size(covar))
    covariance_cotangent[state_idx, state_idx] .=
        states_to_differences' * cotangent * states_to_differences

    solution_cotangent = zeros(R, size(sol))
    state_solution_cotangent =
        cotangent * states_to_differences * finite_state_covariance' +
        cotangent' * states_to_differences * finite_state_covariance
    shock_solution_cotangent = (cotangent + cotangent') * sol[:, n_state + 1:end]
    solution_cotangent[:, 1:n_state] .= state_solution_cotangent
    solution_cotangent[:, n_state + 1:end] .= shock_solution_cotangent

    covariance_cotangent[.!isfinite.(covar)] .= zero(R)
    return covariance_cotangent, solution_cotangent
end

function apply_bgp_difference_output!(states_to_variables::AbstractMatrix{R},
                                      output_names,
                                      state_names,
                                      𝓂::ℳ;
                                      state_blocks::Tuple = (1,)) where R <: Real
    !is_bgp_model(𝓂) && return states_to_variables

    rates = bgp_growth_by_name(𝓂)
    n_state = length(state_names)
    for (row, name) in enumerate(output_names)
        base = Symbol(replace(string(name), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
        get(rates, base, 0.0) == 0.0 && continue
        state_position = findfirst(==(name), state_names)
        state_position === nothing && continue
        for block in state_blocks
            column = (block - 1) * n_state + state_position
            column <= size(states_to_variables, 2) && (states_to_variables[row, column] -= one(R))
        end
    end
    return states_to_variables
end

@unstable function get_irf(𝓂::ℳ;
                periods::Int = DEFAULT_PERIODS,
                algorithm::Symbol = DEFAULT_ALGORITHM,
                parameters::ParameterType = nothing,
                steady_state_function::SteadyStateFunctionType = missing,
                variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_OBC, 
                shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = DEFAULT_SHOCKS_EXCLUDING_OBC,
                negative_shock::Bool = DEFAULT_NEGATIVE_SHOCK, 
                generalised_irf::Bool = DEFAULT_GENERALISED_IRF,
                generalised_irf_warmup_iterations::Int = DEFAULT_GENERALISED_IRF_WARMUP,
                generalised_irf_draws::Int = DEFAULT_GENERALISED_IRF_DRAWS,
                initial_state::Union{Vector{Vector{R}},Vector{R}} = DEFAULT_INITIAL_STATE,
                levels::Bool = false,
                shock_size::Real = DEFAULT_SHOCK_SIZE,
                ignore_obc::Bool = DEFAULT_IGNORE_OBC,
                # timer::TimerOutput = TimerOutput(),
                verbose::Bool = DEFAULT_VERBOSE,
                tol::Tolerances = Tolerances(),
                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                caching::Bool = DEFAULT_CACHING,
                use_workspaces::Bool = DEFAULT_USE_WORKSPACES)::KeyedArray where R <: Real
    # @nospecialize # reduce compile time            

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                                lyapunov_algorithm = lyapunov_algorithm)

    # @timeit_debug timer "Wrangling inputs" begin

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks, negative_shock, shock_size, periods, _, _ = process_shocks_input(shocks, negative_shock, shock_size, periods, 𝓂)
    
    ignore_obc, occasionally_binding_constraints, obc_shocks_included = process_ignore_obc_flag(shocks, ignore_obc, 𝓂)

    generalised_irf = adjust_generalised_irf_flag(generalised_irf, generalised_irf_warmup_iterations, generalised_irf_draws, algorithm, occasionally_binding_constraints, shocks)

    # end # timeit_debug
    
    # @timeit_debug timer "Solve model" begin

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            opts = opts,
            dynamics = true, 
            algorithm = algorithm,
            # timer = timer, 
            obc = occasionally_binding_constraints || obc_shocks_included)
    
    # end # timeit_debug

    # @timeit_debug timer "Get relevant steady state" begin

    reference_steady_state, _, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)
    
    # end # timeit_debug

    initial_state = adjust_initial_state(initial_state, algorithm, 𝓂, SSS_delta, reference_steady_state)

    if occasionally_binding_constraints
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    elseif obc_shocks_included
        @assert algorithm ∉ [:pruned_second_order, :second_order, :pruned_third_order, :third_order] "Occasionally binding constraint shocks without enforcing the constraint is only compatible with first order perturbation solutions."

        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    else
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, false)
    end
    
    level = levels ? reference_steady_state + SSS_delta : SSS_delta

    responses = compute_irf_responses(𝓂,
                                        state_update,
                                        initial_state,
                                        level;
                                        periods = periods,
                                        shocks = shocks,
                                        variables = variables,
                                        shock_size = shock_size,
                                        negative_shock = negative_shock,
                                        generalised_irf = generalised_irf,
                                        generalised_irf_warmup_iterations = generalised_irf_warmup_iterations,
                                        generalised_irf_draws = generalised_irf_draws,
                                        enforce_obc = occasionally_binding_constraints,
                                        algorithm = algorithm)

    # Balanced growth path: levels of trending variables follow x_t = anchor + xᴳ·t.
    # The first-order solution works in deviations from the constant anchored steady
    # state, so add the deterministic drift xᴳ·t (period index t) to the level paths.
    # Only the slope xᴳ is meaningful — the anchor is an arbitrary particular solution.
    if levels
        growths = bgp_growth_by_name(𝓂)
        if !isempty(growths)
            P = size(responses, 2)
            for (vi, nm) in enumerate(axiskeys(responses, 1))
                base = Symbol(replace(string(nm), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
                g = get(growths, base, 0.0)
                g == 0.0 && continue
                for t in 1:P
                    @views responses[vi, t, :] .+= g * t
                end
            end
        end
    end

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    return responses

end



"""
See [`get_irf`](@ref)
"""
@unstable get_irfs = get_irf

"""
See [`get_irf`](@ref)
"""
get_IRF = get_irf

# """
# See [`get_irf`](@ref)
# """
# irfs = get_irf

# """
# See [`get_irf`](@ref)
# """
# irf = get_irf

# """
# See [`get_irf`](@ref)
# """
# IRF = get_irf

"""
Wrapper for [`get_irf`](@ref) with `shocks = :simulate`. Function returns values in levels by default.
"""
@unstable simulate(𝓂::ℳ; kwargs...) =  get_irf(𝓂; kwargs..., shocks = :simulate, levels = get(kwargs, :levels, true))#[:,:,1]

"""
Wrapper for [`get_irf`](@ref) with `shocks = :simulate`. Function returns values in levels by default.
"""
@unstable get_simulation(𝓂::ℳ; kwargs...) =  get_irf(𝓂; kwargs..., shocks = :simulate, levels = get(kwargs, :levels, true))#[:,:,1]

"""
Wrapper for [`get_irf`](@ref) with `shocks = :simulate`. Function returns values in levels by default.
"""
@unstable get_simulations(𝓂::ℳ; kwargs...) =  get_irf(𝓂; kwargs..., shocks = :simulate, levels = get(kwargs, :levels, true))#[:,:,1]

"""
Wrapper for [`get_irf`](@ref) with `generalised_irf = true`.
"""
@unstable get_girf(𝓂::ℳ; kwargs...) =  get_irf(𝓂; kwargs..., generalised_irf = true)









"""
$(SIGNATURES)
Return the (non-stochastic) steady state, calibrated parameters, and derivatives with respect to model parameters.

# Arguments
- $MODEL®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $DERIVATIVES®
- $PARAMETER_DERIVATIVES®
- `stochastic` [Default: `false`, Type: `Bool`]: return stochastic steady state using second order perturbation if no other higher order perturbation algorithm is provided in `algorithm`.
- `return_variables_only` [Default: `false`, Type: `Bool`]: return only variables and not calibrated parameters.
- $ALGORITHM®
- $QME®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows. The columns show the (non-stochastic) steady state and parameters for which derivatives are taken. 

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_steady_state(RBC)
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables_and_calibrated_parameters ∈ 4-element Vector{Symbol}
→   Steady_state_and_∂steady_state∂parameter ∈ 6-element Vector{Symbol}
And data, 4×6 Matrix{Float64}:
        (:Steady_state)  (:std_z)  (:ρ)     (:δ)      (:α)       (:β)
  (:c)   5.93625          0.0       0.0   -116.072    55.786     76.1014
  (:k)  47.3903           0.0       0.0  -1304.95    555.264   1445.93
  (:q)   6.88406          0.0       0.0    -94.7805   66.8912   105.02
  (:z)   0.0              0.0       0.0      0.0       0.0        0.0
```
"""
@unstable function get_steady_state(𝓂::ℳ; 
                            parameters::ParameterType = nothing,
                            steady_state_function::SteadyStateFunctionType = missing, 
                            derivatives::Bool = DEFAULT_DERIVATIVES_FLAG, 
                            stochastic::Bool = DEFAULT_STOCHASTIC_FLAG,
                            algorithm::Symbol = DEFAULT_ALGORITHM_SELECTOR(stochastic),
                            parameter_derivatives::Union{Symbol_input,String_input} = DEFAULT_VARIABLE_SELECTION,
                            return_variables_only::Bool = DEFAULT_RETURN_VARIABLES_ONLY,
                            verbose::Bool = DEFAULT_VERBOSE,
                            silent::Bool = DEFAULT_SILENT_FLAG,
                            tol::Tolerances = Tolerances(),
                            quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                            sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                            caching::Bool = DEFAULT_CACHING,
                            use_workspaces::Bool = DEFAULT_USE_WORKSPACES)::KeyedArray
    # @nospecialize # reduce compile time

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end
                            
    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? :bicgstab : sylvester_algorithm[2])
    
    if stochastic
        if algorithm == :first_order
            @info "Stochastic steady state requested but algorithm is $algorithm. Setting `algorithm = :second_order`." maxlog = DEFAULT_MAXLOG
            algorithm = :second_order
        end
    else
        if algorithm != :first_order
            @info "Non-stochastic steady state requested but algorithm is $algorithm. Setting `stochastic = true`." maxlog = DEFAULT_MAXLOG
            stochastic = true
        end
    end

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            opts = opts)

    vars_in_ss_equations = 𝓂.constants.post_model_macro.vars_in_ss_equations_no_aux
    
    parameter_derivatives = parameter_derivatives isa String_input ? parameter_derivatives .|> Meta.parse .|> replace_indices : parameter_derivatives

    if parameter_derivatives == :all
        length_par = length(𝓂.constants.post_complete_parameters.parameters)
        param_idx = 1:length_par
    elseif isa(parameter_derivatives,Symbol)
        @assert parameter_derivatives ∈ 𝓂.constants.post_complete_parameters.parameters string(parameter_derivatives) * " is not part of the free model parameters."

        param_idx = indexin([parameter_derivatives], 𝓂.constants.post_complete_parameters.parameters)
        length_par = 1
    else
        for p in vec(collect(parameter_derivatives))
            @assert p ∈ 𝓂.constants.post_complete_parameters.parameters string(p) * " is not part of the free model parameters."
        end
        param_idx = indexin(parameter_derivatives |> collect |> vec, 𝓂.constants.post_complete_parameters.parameters) |> sort
        length_par = length(parameter_derivatives)
    end

    SS, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)

    if solution_error > tol.nsss.acceptance_tol
        @warn "Could not find non-stochastic steady state. Solution error: $solution_error > $(tol.nsss.acceptance_tol)"
    end

    if stochastic
        solve!(𝓂, 
                opts = opts, 
                steady_state_function = steady_state_function, 
                dynamics = true, 
                algorithm = algorithm, 
                silent = silent, 
                obc = length(𝓂.equations.obc_violation) > 0)

        if  algorithm == :third_order
            SS[1:length(𝓂.constants.post_model_macro.var)] = 𝓂.caches.third_order_stochastic_steady_state
        elseif  algorithm == :pruned_third_order
            SS[1:length(𝓂.constants.post_model_macro.var)] = 𝓂.caches.pruned_third_order_stochastic_steady_state
        elseif  algorithm == :pruned_second_order
            SS[1:length(𝓂.constants.post_model_macro.var)] = 𝓂.caches.pruned_second_order_stochastic_steady_state
        else
            SS[1:length(𝓂.constants.post_model_macro.var)] = 𝓂.caches.second_order_stochastic_steady_state#[indexin(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_present)),sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.aux,𝓂.constants.post_model_macro.exo_present)))]
        end
    end

    ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    ms = 𝓂.constants.post_complete_parameters
    var_idx = ms.ss_var_idx_in_var_and_calib

    calib_idx = return_variables_only ? Int[] : ms.calib_idx_in_var_and_calib

    if length_par * length(var_idx) > 200 && derivatives
        @info "Most of the time is spent calculating derivatives wrt parameters. If they are not needed, add `derivatives = false` as an argument to the function call." maxlog = DEFAULT_MAXLOG
    #     derivatives = false
    end

    if parameter_derivatives != :all
        derivatives = true
    end

    ensure_name_display_constants!(𝓂)
    var_axis = 𝓂.constants.post_complete_parameters.var_axis
    calib_axis = 𝓂.constants.post_complete_parameters.calib_axis
    axis1 = return_variables_only ? var_axis[var_idx] : vcat(var_axis[var_idx], calib_axis)
    ss_names = return_variables_only ?
               𝓂.constants.post_model_macro.var[var_idx] :
               vcat(𝓂.constants.post_model_macro.var[var_idx], 𝓂.equations.calibration_parameters)
    bgp_model = is_bgp_model(𝓂)
    growth_column = bgp_model ? bgp_growth_column(𝓂, ss_names) : Float64[]

    axis2 = bgp_model ?
            vcat(:Steady_state, :Growth_rate, 𝓂.constants.post_complete_parameters.parameters[param_idx]) :
            vcat(:Steady_state, 𝓂.constants.post_complete_parameters.parameters[param_idx])

    if any(x -> contains(string(x), "◖"), axis2)
        axis2_decomposed = decompose_name.(axis2)
        axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
    end

    if derivatives 
        if stochastic
                n_tuple = algorithm ∈ (:third_order, :pruned_third_order) ? 10 : 8
                SSS_result, SSS_pb = rrule(calculate_stochastic_steady_state, Val(algorithm), 𝓂.parameter_values, 𝓂, opts = opts)
                SSS = SSS_result[1]
                n_sss = length(SSS)
                n_ss = length(SSS_result[3])
                nv = length(var_idx)
                nc = length(calib_idx)
                n_out = nv + nc
                np = length(𝓂.parameter_values)
                dSSS = zeros(n_out, np)
                for j in 1:n_out
                    if j ≤ nv
                        ∂sss = zeros(n_sss); ∂sss[var_idx[j]] = 1.0
                        seed = ntuple(k -> k == 1 ? ∂sss : NoTangent(), n_tuple)
                    else
                        ∂ss = zeros(n_ss); ∂ss[calib_idx[j - nv]] = 1.0
                        seed = ntuple(k -> k == 3 ? ∂ss : NoTangent(), n_tuple)
                    end
                    ∂p = SSS_pb(seed)[3]
                    if !(∂p isa AbstractZero); dSSS[j, :] .= ∂p; end
                end
                dSSS = dSSS[:, param_idx]

                SS_and_pars = SSS_result[3]
                steady_state_column = vcat(SSS[var_idx], SS_and_pars[calib_idx])
                if bgp_model
                    steady_state_column = hcat(steady_state_column, growth_column)
                end
                if !use_workspaces; 𝓂.workspaces = orig_ws; end
                return KeyedArray(hcat(steady_state_column, dSSS);  Variables_and_calibrated_parameters = axis1, Steady_state_and_∂steady_state∂parameter = axis2)
        else
            (nsss_result, nsss_pb) = rrule(get_NSSS_and_parameters, 𝓂, 𝓂.parameter_values, opts = opts)
            out_idx = [var_idx..., calib_idx...]
            n_ss = length(nsss_result[1])
            np = length(𝓂.parameter_values)
            n_out = length(out_idx)
            dSS = zeros(n_out, np)
            for j in 1:n_out
                ∂ss = zeros(n_ss); ∂ss[out_idx[j]] = 1.0
                ∂p = nsss_pb((∂ss, NoTangent()))[3]
                if !(∂p isa AbstractZero); dSS[j, :] .= ∂p; end
            end
            dSS = dSS[:, param_idx]

            # if length(𝓂.calibration_equations_parameters) == 0        
            #     return KeyedArray(hcat(collect(NSSS)[1:(end-1)],dNSSS);  Variables = [sort(union(𝓂.constants.post_model_macro.exo_present,var))...], Steady_state_and_∂steady_state∂parameter = vcat(:Steady_state, 𝓂.constants.post_complete_parameters.parameters))
            # else
            # return ComponentMatrix(hcat(collect(NSSS), dNSSS)',Axis(vcat(:SS, 𝓂.constants.post_complete_parameters.parameters)),Axis([sort(union(𝓂.constants.post_model_macro.exo_present,var))...,𝓂.calibration_equations_parameters...]))
            # return NamedArray(hcat(collect(NSSS), dNSSS), ([sort(union(𝓂.constants.post_model_macro.exo_present,var))..., 𝓂.calibration_equations_parameters...], vcat(:Steady_state, 𝓂.constants.post_complete_parameters.parameters)), ("Var. and par.", "∂x/∂y"))
            if !use_workspaces; 𝓂.workspaces = orig_ws; end
            steady_state_column = SS[[var_idx...,calib_idx...]]
            if bgp_model
                steady_state_column = hcat(steady_state_column, growth_column)
            end
            return KeyedArray(hcat(steady_state_column, dSS);  Variables_and_calibrated_parameters = axis1, Steady_state_and_∂steady_state∂parameter = axis2)
            # end
        end
    else
        # return ComponentVector(collect(NSSS),Axis([sort(union(𝓂.constants.post_model_macro.exo_present,var))...,𝓂.calibration_equations_parameters...]))
        # return NamedArray(collect(NSSS), [sort(union(𝓂.constants.post_model_macro.exo_present,var))..., 𝓂.calibration_equations_parameters...], ("Variables and calibrated parameters"))
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        steady_state_column = SS[[var_idx...,calib_idx...]]
        if bgp_model
            steady_state_column = hcat(steady_state_column, growth_column)
            return KeyedArray(steady_state_column;
                              Variables_and_calibrated_parameters = axis1,
                              Steady_state_and_∂steady_state∂parameter = [:Steady_state, :Growth_rate])
        end
        return KeyedArray(steady_state_column;  Variables_and_calibrated_parameters = axis1)
    end
    # ComponentVector(non_stochastic_steady_state = ComponentVector(NSSS.non_stochastic_steady_state, Axis(sort(union(𝓂.constants.post_model_macro.exo_present,var)))),
    #                 calibrated_parameters = ComponentVector(NSSS.non_stochastic_steady_state, Axis(𝓂.calibration_equations_parameters)),
    #                 stochastic = stochastic)

    # return (var .=> 𝓂.parameter_to_steady_state(𝓂.parameter_values...)[1:length(var)]),  (𝓂.par .=> 𝓂.parameter_to_steady_state(𝓂.parameter_values...)[length(var)+1:end])[getindex(1:length(𝓂.par),map(x->x ∈ collect(𝓂.calibration_equations_parameters),𝓂.par))]
end


"""
Wrapper for [`get_steady_state`](@ref) with `stochastic = false`.
"""
@unstable get_non_stochastic_steady_state(args...; kwargs...) = get_steady_state(args...; kwargs..., stochastic = false)


"""
Wrapper for [`get_steady_state`](@ref) with `stochastic = true`.
"""
@unstable get_stochastic_steady_state(args...; kwargs...) = get_steady_state(args...; kwargs..., stochastic = true)


"""
Wrapper for [`get_steady_state`](@ref) with `stochastic = true`.
"""
@unstable get_SSS(args...; kwargs...) = get_steady_state(args...; kwargs..., stochastic = true)


"""
Wrapper for [`get_steady_state`](@ref) with `stochastic = true`.
"""
@unstable SSS(args...; kwargs...) = get_steady_state(args...; kwargs..., stochastic = true)


"""
Wrapper for [`get_steady_state`](@ref) with `stochastic = true`.
"""
@unstable sss(args...; kwargs...) = get_steady_state(args...; kwargs..., stochastic = true)



"""
See [`get_steady_state`](@ref)
"""
@unstable SS(args...; kwargs...) = get_steady_state(args...; kwargs...)

"""
See [`get_steady_state`](@ref)
"""
@unstable steady_state(args...; kwargs...) = get_steady_state(args...; kwargs...)

"""
See [`get_steady_state`](@ref)
"""
@unstable get_SS(args...; kwargs...) = get_steady_state(args...; kwargs...)

"""
See [`get_steady_state`](@ref)
"""
@unstable get_ss(args...; kwargs...) = get_steady_state(args...; kwargs...)

"""
See [`get_steady_state`](@ref)
"""
@unstable ss(args...; kwargs...) = get_steady_state(args...; kwargs...)




"""
$(SIGNATURES)
Return the solution of the model. In the linear case it returns the non-stochastic steady state (NSSS) followed by the linearised solution of the model. In the nonlinear case (higher order perturbation) the function returns a multidimensional array with the endogenous variables as the second dimension and the state variables, shocks, and perturbation parameter (:Volatility) as the other dimensions.

The values of the output represent the NSSS in the case of a linear solution and below it the effect that deviations from the NSSS of the respective past states, shocks, and perturbation parameter have (perturbation parameter = 1) on the present value (NSSS deviation) of the model variables.

# Arguments
- $MODEL®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $QME®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with the endogenous variables including the auxiliary endogenous and exogenous variables (due to leads and lags > 1) as columns. The rows and other dimensions (depending on the chosen perturbation order) include the NSSS for the linear case only, followed by the states, and exogenous shocks. Subscripts following variable names indicate the timing (e.g. `variable₍₋₁₎`  indicates the variable being in the past). Superscripts indicate leads or lags (e.g. `variableᴸ⁽²⁾` indicates the variable being in lead by two periods). If no super- or subscripts follow the variable name, the variable is in the present.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_solution(RBC)
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Steady_state__States__Shocks ∈ 4-element Vector{Symbol}
→   Variables ∈ 4-element Vector{Symbol}
And data, 4×4 adjoint(::Matrix{Float64}) with eltype Float64:
                   (:c)         (:k)        (:q)        (:z)
  (:Steady_state)   5.93625     47.3903      6.88406     0.0
  (:k₍₋₁₎)          0.0957964    0.956835    0.0726316   0.0
  (:z₍₋₁₎)          0.134937     1.24187     1.37681     0.2
  (:eps_z₍ₓ₎)       0.00674687   0.0620937   0.0688406   0.01
```
"""
@unstable function get_solution(𝓂::ℳ; 
                        parameters::ParameterType = nothing,
                        steady_state_function::SteadyStateFunctionType = missing,
                        algorithm::Symbol = DEFAULT_ALGORITHM, 
                        silent::Bool = DEFAULT_SILENT_FLAG,
                        verbose::Bool = DEFAULT_VERBOSE,
                        tol::Tolerances = Tolerances(),
                        quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                        sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                        caching::Bool = DEFAULT_CACHING,
                        use_workspaces::Bool = DEFAULT_USE_WORKSPACES)::KeyedArray
    # @nospecialize # reduce compile time      

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? :bicgstab : sylvester_algorithm[2])

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            opts = opts,
            dynamics = true, 
            silent = silent, 
            algorithm = algorithm)

    solution_matrix = 𝓂.caches.first_order_solution_matrix

    axis1 = [𝓂.constants.post_model_macro.past_not_future_and_mixed; :Volatility; 𝓂.constants.post_model_macro.exo]

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        axis1[end-length(𝓂.constants.post_model_macro.exo)+1:end] = axis1[end-length(𝓂.constants.post_model_macro.exo)+1:end] .* "₍ₓ₎"
        axis1[1:length(𝓂.constants.post_model_macro.past_not_future_and_mixed)] = axis1[1:length(𝓂.constants.post_model_macro.past_not_future_and_mixed)] .* "₍₋₁₎"
    else
        axis1 = [map(x->Symbol(string(x) * "₍₋₁₎"),𝓂.constants.post_model_macro.past_not_future_and_mixed); :Volatility;map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)]
    end

    axis2 = 𝓂.constants.post_model_macro.var

    if any(x -> contains(string(x), "◖"), axis2)
        axis2_decomposed = decompose_name.(axis2)
        axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
    end

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    if algorithm == :second_order
        return KeyedArray(permutedims(reshape(𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂, 
                                    𝓂.constants.post_model_macro.nVars, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo),
                                [2,1,3]);
                            States__Shocks¹ = axis1,
                            Variables = axis2,
                            States__Shocks² = axis1)
    elseif algorithm == :pruned_second_order
        return KeyedArray(permutedims(reshape(𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂, 
                                    𝓂.constants.post_model_macro.nVars, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo),
                                [2,1,3]);
                            States__Shocks¹ = axis1,
                            Variables = axis2,
                            States__Shocks² = axis1)
    elseif algorithm == :third_order
        return KeyedArray(permutedims(reshape(𝓂.caches.third_order_solution * 𝓂.constants.third_order.𝐔₃, 
                                    𝓂.constants.post_model_macro.nVars, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo),
                                [2,1,3,4]);
                            States__Shocks¹ = axis1,
                            Variables = axis2,
                            States__Shocks² = axis1,
                            States__Shocks³ = axis1)
    elseif algorithm == :pruned_third_order
        return KeyedArray(permutedims(reshape(𝓂.caches.third_order_solution * 𝓂.constants.third_order.𝐔₃, 
                                    𝓂.constants.post_model_macro.nVars, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo),
                                [2,1,3,4]);
                            States__Shocks¹ = axis1,
                            Variables = axis2,
                            States__Shocks² = axis1,
                            States__Shocks³ = axis1)
    else
        axis1 = [:Steady_state; 𝓂.constants.post_model_macro.past_not_future_and_mixed; 𝓂.constants.post_model_macro.exo]

        if any(x -> contains(string(x), "◖"), axis1)
            axis1_decomposed = decompose_name.(axis1)
            axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
            axis1[end-length(𝓂.constants.post_model_macro.exo)+1:end] = axis1[end-length(𝓂.constants.post_model_macro.exo)+1:end] .* "₍ₓ₎"
            axis1[2:length(𝓂.constants.post_model_macro.past_not_future_and_mixed)+1] = axis1[2:length(𝓂.constants.post_model_macro.past_not_future_and_mixed)+1] .* "₍₋₁₎"
        else
            axis1 = [:Steady_state; map(x->Symbol(string(x) * "₍₋₁₎"),𝓂.constants.post_model_macro.past_not_future_and_mixed); map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)]
        end

        n_vars = length(𝓂.constants.post_model_macro.var)
        nsss = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)[1][1:n_vars]

        return KeyedArray([nsss solution_matrix]';
                            Steady_state__States__Shocks = axis1,
                            Variables = axis2)
    end
end


"""
Wrapper for [`get_solution`](@ref) with `algorithm = :first_order`.
"""
@unstable get_first_order_solution(args...; kwargs...) = get_solution(args...; kwargs..., algorithm = :first_order)

"""
Wrapper for [`get_solution`](@ref) with `algorithm = :second_order`.
"""
@unstable get_second_order_solution(args...; kwargs...) = get_solution(args...; kwargs..., algorithm = :second_order)

"""
Wrapper for [`get_solution`](@ref) with `algorithm = :third_order`.
"""
@unstable get_third_order_solution(args...; kwargs...) = get_solution(args...; kwargs..., algorithm = :third_order)

"""
See [`get_solution`](@ref)
"""
@unstable get_perturbation_solution(args...; kwargs...) = get_solution(args...; kwargs...)




"""
$(SIGNATURES)
Return the components of the solution of the model: non-stochastic steady state (NSSS), and solution martrices corresponding to the order of the solution. Note that all returned objects have the variables in rows and the solution matrices have as columns the state variables followed by the perturbation/volatility parameter for higher order solution matrices and lastly the exogenous shocks. Higher order perturbation matrices are sparse and have the Kronecker product of the forementioned elements as columns. The last element, a Boolean indicates whether the solution is numerically accurate.
Function to use when differentiating IRFs with respect to parameters.

# Arguments
- $MODEL®
- $PARAMETERS®
# Keyword Arguments
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $QME®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Tuple{Vector, Vector{AbstractMatrix}, Bool}` consisting of a `Vector` containing the NSSS, a `Vector` of solution matrices (one `Matrix` for first order, two for second order, three for third order), and a `Bool` indicating the correctness of the solution provided.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_solution(RBC, RBC.parameter_values)
# output
([5.936252888048734, 47.39025414828825, 6.8840579710144985, 0.0], AbstractMatrix{Float64}[[0.09579643002421026 0.1349373930517762 0.006746869652588118; 0.9568351489231574 1.2418742011511228 0.062093710057556865; 0.07263157894736799 1.3768115942028993 0.06884057971014498; 0.0 0.2 0.01]], true)
```
"""

# Construct a failure return value for get_solution with uniform tuple type.
# When 𝐒₁ is provided, it is included as the first solution matrix placeholder.
function get_solution_fail(algorithm::Symbol, SS::Vector{S}, nVar::Int, ::Type{S}) where S <: Real
    placeholder = zeros(S, nVar, 2)
    if algorithm in [:second_order, :pruned_second_order]
        return SS, AbstractMatrix{S}[placeholder, zeros(S, nVar, 2)], false
    elseif algorithm in [:third_order, :pruned_third_order]
        return SS, AbstractMatrix{S}[placeholder, zeros(S, nVar, 2), zeros(S, nVar, 2)], false
    else
        return SS, AbstractMatrix{S}[placeholder], false
    end
end

function get_solution_fail(algorithm::Symbol, SS::Vector{S}, nVar::Int, ::Type{S}, 𝐒₁::AbstractMatrix{S}) where S <: Real
    if algorithm in [:second_order, :pruned_second_order]
        return SS, AbstractMatrix{S}[𝐒₁, zeros(S, nVar, 2)], false
    elseif algorithm in [:third_order, :pruned_third_order]
        return SS, AbstractMatrix{S}[𝐒₁, zeros(S, nVar, 2), zeros(S, nVar, 2)], false
    else
        return SS, AbstractMatrix{S}[𝐒₁], false
    end
end

function get_solution(𝓂::ℳ, 
                        parameters::Vector{S}; 
                        steady_state_function::SteadyStateFunctionType = missing,
                        algorithm::Symbol = DEFAULT_ALGORITHM, 
                        verbose::Bool = DEFAULT_VERBOSE, 
                        tol::Tolerances = Tolerances(),
                        quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                        sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                        caching::Bool = DEFAULT_CACHING,
                        use_workspaces::Bool = DEFAULT_USE_WORKSPACES)::Tuple{Vector{S}, Vector{AbstractMatrix{S}}, Bool} where S <: Real

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? :bicgstab : sylvester_algorithm[2])

    estimation = true

    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)

    nVar = length(𝓂.constants.post_model_macro.var)

    solve!(𝓂, 
           opts = opts, 
           steady_state_function = steady_state_function,
           algorithm = algorithm)

    
    if check_bounds(parameters, 𝓂)
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return get_solution_fail(algorithm, fill(S(-Inf), nVar), nVar, S)
    end

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, opts = opts, estimation = estimation)

    if solution_error > tol.nsss.acceptance_tol || isnan(solution_error)
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return get_solution_fail(algorithm, SS_and_pars[1:nVar], nVar, S)
    end

    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces)# |> Matrix

    𝐒₁, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                        constants,
                                                        𝓂.workspaces,
                                                        𝓂.caches;
                                                        opts = opts,
                                                        initial_guess = 𝓂.caches.qme_solution,
                                                        parameter_values = parameters)
    
    update_perturbation_counter!(𝓂.counters, solved, estimation = estimation, order = 1)

    if !solved
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return get_solution_fail(algorithm, SS_and_pars[1:nVar], nVar, S, 𝐒₁)
    end

    if algorithm in [:second_order, :pruned_second_order]
        ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian, 𝓂.workspaces)
    
        𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                                                    initial_guess = 𝓂.caches.second_order_solution,
                                opts = opts, parameter_values = parameters)

        update_perturbation_counter!(𝓂.counters, solved2, estimation = estimation, order = 2)

        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return SS_and_pars[1:nVar], AbstractMatrix{S}[𝐒₁, 𝐒₂], true
    elseif algorithm in [:third_order, :pruned_third_order]
        ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian, 𝓂.workspaces)
    
        𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                                                    initial_guess = 𝓂.caches.second_order_solution,
                                opts = opts, parameter_values = parameters)
    
        update_perturbation_counter!(𝓂.counters, solved2, estimation = estimation, order = 2)

        ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives, 𝓂.workspaces)
                
        𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 
                                𝐒₁, 𝐒₂,
                                𝓂.constants,
                                𝓂.workspaces,
                                𝓂.caches;
                                initial_guess = 𝓂.caches.third_order_solution,
                                opts = opts, parameter_values = parameters)

        update_perturbation_counter!(𝓂.counters, solved3, estimation = estimation, order = 3)

        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return SS_and_pars[1:nVar], AbstractMatrix{S}[𝐒₁, 𝐒₂, 𝐒₃], true
    else
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return SS_and_pars[1:nVar], AbstractMatrix{S}[𝐒₁], true
    end
end


"""
$(SIGNATURES)
Return the conditional variance decomposition of endogenous variables with regards to the shocks using the linearised solution. 

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
# Keyword Arguments
- `periods` [Default: `[1:20...,Inf]`, Type: `Union{Vector{Int},Vector{Float64},UnitRange{Int64}}`]: vector of periods for which to calculate the conditional variance decomposition. If the vector contains `Inf`, also the unconditional variance decomposition is calculated (same output as [`get_variance_decomposition`](@ref)).
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $QME®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows, shocks in columns, and periods as the third dimension.

# Examples
```jldoctest part1
using MacroModelling

@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME  begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

get_conditional_variance_decomposition(RBC_CME)
# output
3-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 7-element Vector{Symbol}
→   Shocks ∈ 2-element Vector{Symbol}
◪   Periods ∈ 21-element Vector{Float64}
And data, 7×2×21 Array{Float64, 3}:
[showing 3 of 21 slices]
[:, :, 1] ~ (:, :, 1.0):
              (:delta_eps)  (:eps_z)
  (:A)         0.0           1.0
  (:Pi)        0.00158668    0.998413
  (:R)         0.00158668    0.998413
  (:c)         0.0277348     0.972265
  (:k)         0.00869568    0.991304
  (:y)         0.0           1.0
  (:z_delta)   1.0           0.0

[:, :, 11] ~ (:, :, 11.0):
              (:delta_eps)  (:eps_z)
  (:A)         0.0           1.0
  (:Pi)        0.0245641     0.975436
  (:R)         0.0245641     0.975436
  (:c)         0.0175249     0.982475
  (:k)         0.00869568    0.991304
  (:y)         7.63511e-5    0.999924
  (:z_delta)   1.0           0.0

[:, :, 21] ~ (:, :, Inf):
              (:delta_eps)  (:eps_z)
  (:A)         0.0           1.0
  (:Pi)        0.0156771     0.984323
  (:R)         0.0156771     0.984323
  (:c)         0.0134672     0.986533
  (:k)         0.00869568    0.991304
  (:y)         0.000313462   0.999687
  (:z_delta)   1.0           0.0
```
"""
@unstable function get_conditional_variance_decomposition(𝓂::ℳ; 
                                                periods::Union{Vector{Int},Vector{Float64},UnitRange{Int64}} = DEFAULT_CONDITIONAL_VARIANCE_PERIODS,
                                                parameters::ParameterType = nothing,
                                                steady_state_function::SteadyStateFunctionType = missing,  
                                                verbose::Bool = DEFAULT_VERBOSE,
                                                tol::Tolerances = Tolerances(),
                                                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                                                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                                                caching::Bool = DEFAULT_CACHING,
                                                use_workspaces::Bool = DEFAULT_USE_WORKSPACES)
    # @nospecialize # reduce compile time                                            

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                lyapunov_algorithm = lyapunov_algorithm)

    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)

    solve!(𝓂, 
            opts = opts,
            steady_state_function = steady_state_function,  
            parameters = parameters)

    # write_parameters_input!(𝓂,parameters, verbose = verbose)

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)
    
    ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces)# |> Matrix

    𝑺₁, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                        constants,
                                                        𝓂.workspaces,
                                                        𝓂.caches;
                                                        opts = opts,
                                                        initial_guess = 𝓂.caches.qme_solution,
                                                        parameter_values = 𝓂.parameter_values)
    
    update_perturbation_counter!(𝓂.counters, solved, order = 1)

    A = @views 𝑺₁[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.constants.post_model_macro.nVars))[indexin(𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,1:𝓂.constants.post_model_macro.nVars),:]
    
    sort!(periods)

    maxperiods = periods == [Inf] ? 0 : Int(maximum(periods[isfinite.(periods)]))

    var_container = zeros(size(𝑺₁)[1], 𝓂.constants.post_model_macro.nExo, length(periods))

    for i in 1:𝓂.constants.post_model_macro.nExo
        C = @views 𝑺₁[:,𝓂.constants.post_model_macro.nPast_not_future_and_mixed+i]
        CC = C * C'
        varr = zeros(size(C)[1],size(C)[1])
        for k in 1:maxperiods
            varr = A * varr * A' + CC
            if k ∈ periods
                var_container[:,i,indexin(k, periods)] = ℒ.diag(varr)
            end
        end
        if Inf in periods
            # Ensure lyapunov workspace is properly sized and get it
            lyap_ws = ensure_lyapunov_workspace!(𝓂.workspaces, 𝓂.constants.post_model_macro.nVars, :first_order)

            covar_raw, _ = solve_lyapunov_equation(A, CC, lyap_ws,
                                                    lyapunov_algorithm = opts.lyapunov_algorithm, 
                                                    tol = opts.tol.first_order.lyapunov,
                                                    verbose = opts.verbose)

            var_container[:,i,indexin(Inf,periods)] = ℒ.diag(covar_raw) # numerically more stable
        end
    end

    sum_var_container = max.(sum(var_container, dims=2),eps())
    
    var_container[var_container .< opts.tol.first_order.lyapunov.acceptance_tol] .= 0
    
    cond_var_decomp = var_container ./ sum_var_container

    axis1 = 𝓂.constants.post_model_macro.var

    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.var_axis
    axis2 = 𝓂.constants.post_complete_parameters.exo_axis_plain

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    KeyedArray(cond_var_decomp; Variables = axis1, Shocks = axis2, Periods = periods)
end


"""
See [`get_conditional_variance_decomposition`](@ref)
"""
@unstable get_fevd = get_conditional_variance_decomposition


"""
See [`get_conditional_variance_decomposition`](@ref)
"""
@unstable get_forecast_error_variance_decomposition = get_conditional_variance_decomposition


"""
See [`get_conditional_variance_decomposition`](@ref)
"""
fevd = get_conditional_variance_decomposition




"""
$(SIGNATURES)
Return the variance decomposition of endogenous variables with regards to the shocks. By default the linearised solution is used; with `algorithm = :pruned_second_order` or `algorithm = :pruned_third_order` the per-shock variance contributions are computed under the corresponding pruned higher-order solution and an extra column `:Cross_shock_interaction` is appended that captures the residual variance attributable to genuine cross-shock interaction terms in the centered higher moments (this column is zero whenever no products of distinct shocks appear in the model equations). Rows always sum to one.

Per-shock contributions are obtained by projecting the inner shock-cumulant block to a single shock `i` and re-solving the same pruned-state Lyapunov equation. At third order the projection is implemented as a binary mask on the augmented shock vector `ê`, retaining only those components whose exogenous-shock indices are all equal to `i`. Raw shares are returned without clipping; tiny negative entries can occur from numerical noise on near-zero contributions.

Setting `marginal_contribution = true` (only meaningful for the two pruned higher-order algorithms) instead allocates the cross-shock interaction across the individual shocks via marginal contributions (Shapley values). The result is a `nVars × nExo` table without a `:Cross_shock_interaction` column whose rows still sum to one (up to numerical noise; rows whose total variance is below `eps()` are reported as zero). The characteristic function `V(S)` is the same projected coalition variance used for the raw shares (state cumulants are kept at their full-shock values), so this is a marginal-contribution allocation of the projected higher-order variance, not a counterfactual recomputation of model variance under a sub-set of active shocks. Because `V(S)` need not be monotone, individual shares may be negative or exceed one. The allocation uses the Aumann–Shapley path-integral identity with Gauss–Legendre quadrature on the multilinear extension of `V`. The implementation starts from the low-order Gauss–Legendre rule (`2 * n_e` Lyapunov solves at second order, `3 * n_e` at third) and incrementally reruns with up to 7 nodes when the relative Shapley-efficiency closure error exceeds `1e-3`. At `:first_order` the option is silently ignored (with an `@info` notice) because the first-order decomposition is already additive across shocks.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- `algorithm` [Default: `:first_order`, Type: `Symbol`]: solution algorithm. Supports `:first_order`, `:pruned_second_order`, and `:pruned_third_order`.
- `marginal_contribution` [Default: `false`, Type: `Bool`]: when `true` and `algorithm` is one of the pruned higher-order solutions, return marginal-contribution (Shapley-allocated) per-shock variance shares via the Aumann–Shapley path-integral driver. The implementation starts from the low-order Gauss–Legendre rule (`2 * n_e` Lyapunov solves at second order, `3 * n_e` at third) and incrementally reruns with up to 7 nodes when the relative Shapley-efficiency closure error exceeds `1e-3`. At `:first_order` the option is silently ignored (with an `@info` notice).
- $QME®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows and shocks in columns. Under `:pruned_second_order` and `:pruned_third_order` an additional `:Cross_shock_interaction` column is appended unless `marginal_contribution = true`, in which case only the per-shock columns are returned.

# Examples
```jldoctest part1
using MacroModelling

@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME  begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

get_variance_decomposition(RBC_CME)
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 7-element Vector{Symbol}
→   Shocks ∈ 2-element Vector{Symbol}
And data, 7×2 Matrix{Float64}:
              (:delta_eps)  (:eps_z)
  (:A)         0.0           1.0
  (:Pi)        0.0156771     0.984323
  (:R)         0.0156771     0.984323
  (:c)         0.0134672     0.986533
  (:k)         0.00869568    0.991304
  (:y)         0.000313462   0.999687
  (:z_delta)   1.0           0.0
```

The higher-order variants are illustrated on the Caldara, Fernández-Villaverde & Yao (2012) model, which has a level shock `ϵᶻ` and a stochastic-volatility shock `ω`:

```jldoctest part2
using MacroModelling

@model Caldara_et_al_2012 begin
    V[0] = ((1 - β) * (c[0] ^ ν * (1 - l[0]) ^ (1 - ν)) ^ (1 - 1 / ψ) + β * V[1] ^ (1 - 1 / ψ)) ^ (1 / (1 - 1 / ψ))
    1 = (1 + ζ * exp(z[1]) * k[0] ^ (ζ - 1) * l[1] ^ (1 - ζ) - δ) * c[0] * β * (((1 - l[1]) / (1 - l[0])) ^ (1 - ν) * (c[1] / c[0]) ^ ν) ^ (1 - 1 / ψ) / c[1]
    Rᵏ[0] = ζ * exp(z[1]) * k[0] ^ (ζ - 1) * l[1] ^ (1 - ζ) - δ
    SDF⁺¹[0] = c[0] * β * (((1 - l[1]) / (1 - l[0])) ^ (1 - ν) * (c[1] / c[0]) ^ ν) ^ (1 - 1 / ψ) / c[1]
    1 + Rᶠ[0] = 1 / SDF⁺¹[0]
    (1 - ν) / ν * c[0] / (1 - l[0]) = (1 - ζ) * exp(z[0]) * k[-1] ^ ζ * l[0] ^ (-ζ)
    c[0] + i[0] = exp(z[0]) * k[-1] ^ ζ * l[0] ^ (1 - ζ)
    k[0] = i[0] + k[-1] * (1 - δ)
    z[0] = λ * z[-1] + σ[0] * ϵᶻ[x]
    y[0] = exp(z[0]) * k[-1] ^ ζ * l[0] ^ (1 - ζ)
    log(σ[0]) = (1 - ρ) * log(σ̄) + ρ * log(σ[-1]) + η * ω[x]
    dy[0] = 100 * (y[0] / y[-1] - 1) + dȳ
    dc[0] = 100 * (c[0] / c[-1] - 1) + dc̄
end

@parameters Caldara_et_al_2012 begin
    dȳ = 2.0
    dc̄ = 2.0
    β = 0.991
    l[ss] = 1/3 | ν
    ζ = 0.3
    δ = 0.0196
    λ = 0.95
    ψ = 0.5
    σ̄ = 0.021
    η = 0.1
    ρ = 0.9
end

get_variance_decomposition(Caldara_et_al_2012, algorithm = :pruned_second_order)
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 13-element Vector{Symbol}
→   Shocks ∈ 3-element Vector{Symbol}
And data, 13×3 Matrix{Float64}:
            (:ω)  (:ϵᶻ)      (:Cross_shock_interaction)
  (:Rᵏ)      0.0   0.990522   0.0094784
  (:Rᶠ)      0.0   0.990522   0.0094784
  (:SDF⁺¹)   0.0   0.99052    0.00948043
  (:V)       0.0   0.990504   0.00949593
  (:c)       0.0   0.990526   0.00947405
  (:dc)      0.0   0.990504   0.00949552
  (:dy)      0.0   0.990505   0.00949515
  (:i)       0.0   0.990592   0.00940813
  (:k)       0.0   0.990564   0.00943632
  (:l)       0.0   0.990506   0.0094941
  (:y)       0.0   0.990554   0.0094464
  (:z)       0.0   0.9905     0.0095
  (:σ)       1.0   0.0        0.0

get_variance_decomposition(Caldara_et_al_2012, algorithm = :pruned_second_order, marginal_contribution = true)
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 13-element Vector{Symbol}
→   Shocks ∈ 2-element Vector{Symbol}
And data, 13×2 Matrix{Float64}:
            (:ω)         (:ϵᶻ)
  (:Rᵏ)      0.0047392    0.995261
  (:Rᶠ)      0.0047392    0.995261
  (:SDF⁺¹)   0.00474022   0.99526
  (:V)       0.00474796   0.995252
  (:c)       0.00473702   0.995263
  (:dc)      0.00474776   0.995252
  (:dy)      0.00474757   0.995252
  (:i)       0.00470406   0.995296
  (:k)       0.00471816   0.995282
  (:l)       0.00474705   0.995253
  (:y)       0.0047232    0.995277
  (:z)       0.00475      0.99525
  (:σ)       1.0          0.0

get_variance_decomposition(Caldara_et_al_2012, algorithm = :pruned_third_order, marginal_contribution = true)
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 13-element Vector{Symbol}
→   Shocks ∈ 2-element Vector{Symbol}
And data, 13×2 Matrix{Float64}:
            (:ω)         (:ϵᶻ)
  (:Rᵏ)      0.0093913    0.990609
  (:Rᶠ)      0.00939245   0.990608
  (:SDF⁺¹)   0.00939535   0.990605
  (:V)       0.00933759   0.990662
  (:c)       0.00928112   0.990719
  (:dc)      0.00936725   0.990633
  (:dy)      0.00934246   0.990658
  (:i)       0.00917571   0.990824
  (:k)       0.00916176   0.990838
  (:l)       0.00933518   0.990665
  (:y)       0.0092273    0.990773
  (:z)       0.00935325   0.990647
  (:σ)       1.0          0.0
```
"""
@unstable function get_variance_decomposition(𝓂::ℳ; 
                                    parameters::ParameterType = nothing,
                                    steady_state_function::SteadyStateFunctionType = missing,
                                    algorithm::Symbol = :first_order,
                                    marginal_contribution::Bool = false,
                                    verbose::Bool = DEFAULT_VERBOSE,
                                    tol::Tolerances = Tolerances(),
                                    quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                                    lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                                    caching::Bool = DEFAULT_CACHING,
                                    use_workspaces::Bool = DEFAULT_USE_WORKSPACES)
    # @nospecialize # reduce compile time
                                    
    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    lyapunov_algorithm = lyapunov_algorithm)
    
    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)

    @assert algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] "algorithm must be :first_order, :pruned_second_order, or :pruned_third_order"

    if marginal_contribution && algorithm == :first_order
        @info "marginal_contribution = true has no effect for algorithm = :first_order. The first-order variance decomposition is already additive across shocks (no cross-shock interaction term to allocate); standard shares are returned."
        marginal_contribution = false
    end

    solve!(𝓂, 
            opts = opts, 
            steady_state_function = steady_state_function, 
            parameters = parameters,
            algorithm = algorithm,
            dynamics = algorithm != :first_order)

    if algorithm == :pruned_second_order || algorithm == :pruned_third_order
        if marginal_contribution
            if algorithm == :pruned_second_order
                shares_var, total_var, ok = calculate_aumann_shapley_second_order(𝓂.parameter_values, 𝓂; opts = opts)
                order_label = "second"
            else
                shares_var, total_var, ok = calculate_aumann_shapley_third_order(𝓂.parameter_values, 𝓂; opts = opts)
                order_label = "third"
            end

            if !ok
                if !use_workspaces; 𝓂.workspaces = orig_ws; end
                error("Marginal-contribution (Shapley) $order_label-order variance decomposition failed (Lyapunov did not converge for at least one coalition).")
            end

            # Normalise by total variance. Rows whose total variance is below
            # `eps()` are reported as zero shares (attribution is undefined).
            denom = max.(total_var, eps())
            var_decomp = shares_var ./ denom
            zero_rows = total_var .< eps()
            if any(zero_rows)
                var_decomp[zero_rows, :] .= zero(eltype(var_decomp))
            end

            ensure_name_display_constants!(𝓂)
            axis1 = 𝓂.constants.post_complete_parameters.var_axis
            axis2 = 𝓂.constants.post_complete_parameters.exo_axis_plain

            if !use_workspaces; 𝓂.workspaces = orig_ws; end

            return KeyedArray(var_decomp; Variables = axis1, Shocks = axis2)
        end

        if algorithm == :pruned_second_order
            per_shock_var, total_var, ok = calculate_per_shock_variance_second_order(𝓂.parameter_values, 𝓂; opts = opts)
            order_label = "second"
        else
            per_shock_var, total_var, ok = calculate_per_shock_variance_third_order(𝓂.parameter_values, 𝓂; opts = opts)
            order_label = "third"
        end

        if !ok
            if !use_workspaces; 𝓂.workspaces = orig_ws; end
            error("Per-shock $order_label-order variance decomposition failed (Lyapunov did not converge for at least one shock).")
        end

        # Use total variance for normalisation (rather than sum over shocks) so the
        # `:Cross_shock_interaction` column captures the residual exactly. Raw
        # shares are returned without clipping; tiny negative entries can occur
        # from numerical noise on near-zero contributions.
        denom = max.(total_var, eps())

        shares = per_shock_var ./ denom
        interaction = 1 .- vec(sum(shares, dims = 2))

        var_decomp = hcat(shares, interaction)

        ensure_name_display_constants!(𝓂)
        axis1 = 𝓂.constants.post_complete_parameters.var_axis
        axis2 = vcat(𝓂.constants.post_complete_parameters.exo_axis_plain, :Cross_shock_interaction)

        if !use_workspaces; 𝓂.workspaces = orig_ws; end

        return KeyedArray(var_decomp; Variables = axis1, Shocks = axis2)
    end

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)
    
    ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces)# |> Matrix

    sol, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                        constants,
                                                        𝓂.workspaces,
                                                        𝓂.caches;
                                                        opts = opts,
                                                        initial_guess = 𝓂.caches.qme_solution,
                                                        parameter_values = 𝓂.parameter_values)

    update_perturbation_counter!(𝓂.counters, solved, order = 1)
    
    variances_by_shock = zeros(𝓂.constants.post_model_macro.nVars, 𝓂.constants.post_model_macro.nExo)

    A = @views sol[:, 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.constants.post_model_macro.nVars))[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,:]

    # Use pre-detected unit root flag from QME solve when available.
    # Only compute Schur decomposition of A when unit roots are present (needed for deflation).
    has_unit_roots = 𝓂.caches.has_unit_roots
    unit_root_tol = 1e-8
    lyap_ws = ensure_lyapunov_workspace!(𝓂.workspaces, 𝓂.constants.post_model_macro.nVars, :first_order)

    if !has_unit_roots
        # Standard path: no unit roots, solve each shock directly
        for i in 1:𝓂.constants.post_model_macro.nExo
            C = @views sol[:, 𝓂.constants.post_model_macro.nPast_not_future_and_mixed + i]
            CC = C * C'
            covar_raw, _ = solve_lyapunov_equation(A, CC, lyap_ws,
                                                    lyapunov_algorithm = opts.lyapunov_algorithm, 
                                                    tol = opts.tol.first_order.lyapunov,
                                                    verbose = opts.verbose)
            variances_by_shock[:,i] = ℒ.diag(covar_raw)
        end
    else
        # Unit root path: compute Schur decomposition of A for deflation
        A_dense = collect(A)
        A_work = copy(A_dense)
        Tmat, U_schur, n_unstable = ordered_schur!(A_work, unit_root_tol, lyap_ws.schur_ws)
        n = size(A_dense, 1)

        if n_unstable == n
            # All eigenvalues unstable — all variances are NaN
            variances_by_shock .= NaN
        else
            n_stable = n - n_unstable
            stable_range = (n_unstable + 1):n
            T_ss = Tmat[stable_range, stable_range]
            U_s = U_schur[:, stable_range]

            # Identify unit-root variables
            U_u = @view U_schur[:, 1:n_unstable]
            unstable_loading = vec(sum(abs2, U_u; dims = 2))
            unit_root_vars = unstable_loading .> unit_root_tol

            ws_stable = Lyapunov_workspace(n_stable)

            if opts.verbose
                println("Variance decomposition: Schur pre-computed ($n_unstable unstable, $n_stable stable eigenvalues)")
            end

            for i in 1:𝓂.constants.post_model_macro.nExo
                C = @views sol[:, 𝓂.constants.post_model_macro.nPast_not_future_and_mixed + i]
                CC = C * C'

                # Transform to Schur basis and extract stable block
                CC_schur = U_schur' * CC * U_schur
                CC_ss = (CC_schur[stable_range, stable_range] + CC_schur[stable_range, stable_range]') / 2

                X_ss, _, sub_tol = solve_lyapunov_equation(T_ss, CC_ss, Val(:doubling), ws_stable;
                                                            tol = opts.tol.first_order.lyapunov)

                if sub_tol > opts.tol.first_order.lyapunov.acceptance_tol
                    X_ss, _, sub_tol = solve_lyapunov_equation(T_ss, CC_ss, Val(:bicgstab), ws_stable;
                                                                tol = opts.tol.first_order.lyapunov)
                end

                X_ss = collect(X_ss)

                # Map back: only need diagonal of U_s * X_ss * U_s'
                tmp = X_ss * U_s'
                var_i = vec(sum(U_s .* tmp', dims = 2))
                var_i[unit_root_vars] .= NaN
                variances_by_shock[:,i] = var_i
            end
        end
    end

    sum_variances_by_shock = max.(sum(variances_by_shock, dims=2), eps())
    
    variances_by_shock[variances_by_shock .< opts.tol.first_order.lyapunov.acceptance_tol] .= 0
    
    var_decomp = variances_by_shock ./ sum_variances_by_shock
    
    axis1 = 𝓂.constants.post_model_macro.var

    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.var_axis
    axis2 = 𝓂.constants.post_complete_parameters.exo_axis_plain

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    KeyedArray(var_decomp; Variables = axis1, Shocks = axis2)
end



"""
See [`get_variance_decomposition`](@ref)
"""
@unstable get_var_decomp = get_variance_decomposition




"""
$(SIGNATURES)
Return the correlations of endogenous variables using the first, pruned second, or pruned third order perturbation solution. 

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $QME®
- $LYAPUNOV®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows and columns.

# Examples
```jldoctest part1
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_correlation(RBC)
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 4-element Vector{Symbol}
→   𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 ∈ 4-element Vector{Symbol}
And data, 4×4 Matrix{Float64}:
        (:c)       (:k)       (:q)       (:z)
  (:c)   1.0        0.999812   0.550168   0.314562
  (:k)   0.999812   1.0        0.533879   0.296104
  (:q)   0.550168   0.533879   1.0        0.965726
  (:z)   0.314562   0.296104   0.965726   1.0
```
"""
@unstable get_correlation(args...; kwargs...) = get_moments(args...; kwargs..., variance = false, non_stochastic_steady_state = false, standard_deviation = false, covariance = false, correlation = true, derivatives = get(kwargs, :derivatives, false))[:correlation]

"""
See [`get_correlation`](@ref)
"""
@unstable get_corr = get_correlation


"""
See [`get_correlation`](@ref)
"""
corr = get_correlation




"""
$(SIGNATURES)
Return the autocorrelations of endogenous variables using the first, pruned second, or pruned third order perturbation solution. 

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
# Keyword Arguments
- `autocorrelation_periods` [Default: `1:5`, Type: `UnitRange{Int}`]: periods for which to return the autocorrelation
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $QME®
- $LYAPUNOV®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows and autocorrelation periods in columns.

# Examples
```jldoctest part1
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_autocorrelation(RBC)
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 4-element Vector{Symbol}
→   Autocorrelation_periods ∈ 5-element UnitRange{Int64}
And data, 4×5 Matrix{Float64}:
        (1)         (2)         (3)         (4)         (5)
  (:c)    0.966974    0.927263    0.887643    0.849409    0.812761
  (:k)    0.971015    0.931937    0.892277    0.853876    0.817041
  (:q)    0.32237     0.181562    0.148347    0.136867    0.129944
  (:z)    0.2         0.04        0.008       0.0016      0.00032
```
"""
@unstable function get_autocorrelation(𝓂::ℳ; 
                            autocorrelation_periods::UnitRange{Int} = DEFAULT_AUTOCORRELATION_PERIODS,
                            parameters::ParameterType = nothing,
                            steady_state_function::SteadyStateFunctionType = missing,  
                            algorithm::Symbol = DEFAULT_ALGORITHM,
                            quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                            sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                            lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM, 
                            verbose::Bool = DEFAULT_VERBOSE,
                            tol::Tolerances = Tolerances(),
                            caching::Bool = DEFAULT_CACHING,
                            use_workspaces::Bool = DEFAULT_USE_WORKSPACES)
    # @nospecialize # reduce compile time

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end
    
    opts = merge_calculation_options(tol = tol, verbose = verbose,
                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                            sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                            sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                            lyapunov_algorithm = lyapunov_algorithm)

    @assert algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] "Autocorrelation can only be calculated for first order perturbation or second and third order pruned perturbation solutions."

    solve!(𝓂, 
            opts = opts, 
            steady_state_function = steady_state_function, 
            parameters = parameters,
            algorithm = algorithm)

    if algorithm == :pruned_third_order
        covar_dcmp, state_μ, autocorr, SS_and_pars, solved = calculate_third_order_moments_with_autocorrelation(𝓂.parameter_values, 𝓂.constants.post_model_macro.var, 𝓂, 
                                                                                            opts = opts, 
                                                                                            autocorrelation_periods = autocorrelation_periods)

        autocorr[ℒ.diag(covar_dcmp) .< opts.tol.first_order.lyapunov.acceptance_tol,:] .= 0
    elseif algorithm == :pruned_second_order
        covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = calculate_second_order_moments_with_covariance(𝓂.parameter_values, 𝓂, opts = opts)

        ŝ_to_ŝ₂ⁱ = ℒ.diagm(ones(size(Σᶻ₂,1)))

        autocorr = zeros(size(covar_dcmp,1),length(autocorrelation_periods))

        covar_dcmp[abs.(covar_dcmp) .< opts.tol.first_order.lyapunov.acceptance_tol] .= 0

        for i in autocorrelation_periods
            autocorr[:,i] .= ℒ.diag(ŝ_to_y₂ * ŝ_to_ŝ₂ⁱ * autocorr_tmp) ./ ℒ.diag(covar_dcmp) 
            ŝ_to_ŝ₂ⁱ *= ŝ_to_ŝ₂
        end

        autocorr[ℒ.diag(covar_dcmp) .< opts.tol.first_order.lyapunov.acceptance_tol,:] .= 0
    else
        covar_dcmp, sol, _, SS_and_pars, solved = calculate_covariance(𝓂.parameter_values, 𝓂, opts = opts)

        if !solved
            @warn "Could not find covariance matrix. Results may contain NaN for unit-root variables."
        end

        A = @views sol[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.constants.post_model_macro.nVars))[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,:]
    
        autocorr = reduce(hcat,[ℒ.diag(A ^ i * covar_dcmp ./ ℒ.diag(covar_dcmp)) for i in autocorrelation_periods])

        autocorr[ℒ.diag(covar_dcmp) .< opts.tol.first_order.lyapunov.acceptance_tol,:] .= 0
    end
    
    axis1 = 𝓂.constants.post_model_macro.var

    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.var_axis

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    KeyedArray(collect(autocorr); Variables = axis1, Autocorrelation_periods = autocorrelation_periods)
end

"""
See [`get_autocorrelation`](@ref)
"""
@unstable get_autocorr(args...; kwargs...) = get_autocorrelation(args...; kwargs...)


"""
See [`get_autocorrelation`](@ref)
"""
@unstable autocorr(args...; kwargs...) = get_autocorrelation(args...; kwargs...)




"""
$(SIGNATURES)
Return the first and second moments of endogenous variables using the first, pruned second, or pruned third order perturbation solution. By default returns: non-stochastic steady state (NSSS), and standard deviations, but can optionally return variances, covariance matrix, and correlation matrix. Derivatives of the moments can also be provided by setting `derivatives` to `true`.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- `non_stochastic_steady_state` [Default: `true`, Type: `Bool`]: switch to return SS of endogenous variables
- `mean` [Default: `false`, Type: `Bool`]: switch to return mean of endogenous variables (the mean for the linearised solutoin is the NSSS)
- `standard_deviation` [Default: `true`, Type: `Bool`]: switch to return standard deviation of endogenous variables
- `variance` [Default: `false`, Type: `Bool`]: switch to return variance of endogenous variables
- `covariance` [Default: `false`, Type: `Bool`]: switch to return covariance matrix of endogenous variables
- `correlation` [Default: `false`, Type: `Bool`]: switch to return correlation matrix of endogenous variables
- $(VARIABLES®(DEFAULT_VARIABLES_EXCLUDING_OBC))
- $DERIVATIVES®
- $PARAMETER_DERIVATIVES®
- $ALGORITHM®
- $QME®
- $LYAPUNOV®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Dict{Symbol,KeyedArray}` containing the selected moments. All moments have variables as rows and the moment as the first column followed by partial derivatives wrt parameters. Covariance and correlation matrices are returned as 2D `KeyedArray`s (or 3D when `derivatives = true`). The `KeyedArray` type is provided by the `AxisKeys` package.

# Examples
```jldoctest part1
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

moments = get_moments(RBC);

moments[:non_stochastic_steady_state]
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 4-element Vector{Symbol}
→   Steady_state_and_∂steady_state∂parameter ∈ 6-element Vector{Symbol}
And data, 4×6 Matrix{Float64}:
        (:Steady_state)  (:std_z)  (:ρ)     (:δ)      (:α)       (:β)
  (:c)   5.93625          0.0       0.0   -116.072    55.786     76.1014
  (:k)  47.3903           0.0       0.0  -1304.95    555.264   1445.93
  (:q)   6.88406          0.0       0.0    -94.7805   66.8912   105.02
  (:z)   0.0              0.0       0.0      0.0       0.0        0.0
```


```jldoctest part1
moments[:standard_deviation]
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 4-element Vector{Symbol}
→   Standard_deviation_and_∂standard_deviation∂parameter ∈ 6-element Vector{Symbol}
And data, 4×6 Matrix{Float64}:
        (:Standard_deviation)  (:std_z)  …  (:δ)       (:α)       (:β)
  (:c)   0.0266642              2.66642     -0.384359   0.2626     0.144789
  (:k)   0.264677              26.4677      -5.74194    2.99332    6.30323
  (:q)   0.0739325              7.39325     -0.974722   0.726551   1.08
  (:z)   0.0102062              1.02062      0.0        0.0        0.0
```

Correlation matrix (returned when `correlation = true`):
```jldoctest part1
get_moments(RBC, non_stochastic_steady_state = false, standard_deviation = false, correlation = true, derivatives = false)[:correlation]
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 4-element Vector{Symbol}
→   𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 ∈ 4-element Vector{Symbol}
And data, 4×4 Matrix{Float64}:
        (:c)       (:k)       (:q)       (:z)
  (:c)   1.0        0.999812   0.550168   0.314562
  (:k)   0.999812   1.0        0.533879   0.296104
  (:q)   0.550168   0.533879   1.0        0.965726
  (:z)   0.314562   0.296104   0.965726   1.0
```
"""
@unstable function get_moments(𝓂::ℳ; 
                    parameters::ParameterType = nothing,
                    steady_state_function::SteadyStateFunctionType = missing,  
                    non_stochastic_steady_state::Bool = DEFAULT_NON_STOCHASTIC_STEADY_STATE_FLAG, 
                    mean::Bool = DEFAULT_MEAN_FLAG,
                    standard_deviation::Bool = DEFAULT_STANDARD_DEVIATION_FLAG, 
                    variance::Bool = DEFAULT_VARIANCE_FLAG, 
                    covariance::Bool = DEFAULT_COVARIANCE_FLAG, 
                    correlation::Bool = DEFAULT_CORRELATION_FLAG,
                    variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_OBC, 
                    derivatives::Bool = DEFAULT_DERIVATIVES_FLAG,
                    parameter_derivatives::Union{Symbol_input,String_input} = DEFAULT_VARIABLE_SELECTION,
                    algorithm::Symbol = DEFAULT_ALGORITHM,
                    silent::Bool = DEFAULT_SILENT_FLAG,
                    quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                    sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                    lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM, 
                    verbose::Bool = DEFAULT_VERBOSE,
                    tol::Tolerances = Tolerances(),
                    caching::Bool = DEFAULT_CACHING,
                    use_workspaces::Bool = DEFAULT_USE_WORKSPACES)#limit output by selecting pars and vars like for plots and irfs!?
    # @nospecialize # reduce compile time          

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                    lyapunov_algorithm = lyapunov_algorithm)

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            algorithm = algorithm, 
            opts = opts, 
            silent = silent)

    for (moment_name, condition) in [("Mean", mean), ("Standard deviation", standard_deviation), ("Variance", variance), ("Covariance", covariance), ("Correlation", correlation)]
        if condition
            @assert algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] moment_name * " only available for algorithms: `first_order`, `pruned_second_order`, and `pruned_third_order`."
        end
    end

    # write_parameters_input!(𝓂,parameters, verbose = verbose)

    var_idx = parse_variables_input_to_index(variables, 𝓂) |> sort

    parameter_derivatives = parameter_derivatives isa String_input ? parameter_derivatives .|> Meta.parse .|> replace_indices : parameter_derivatives
    length_par = 0

    param_idx = 0:0
    
    if parameter_derivatives == :all
        length_par = length(𝓂.constants.post_complete_parameters.parameters)
        param_idx = 1:length_par
    elseif isa(parameter_derivatives,Symbol)
        @assert parameter_derivatives ∈ 𝓂.constants.post_complete_parameters.parameters string(parameter_derivatives) * " is not part of the free model parameters."

        param_idx = indexin([parameter_derivatives], 𝓂.constants.post_complete_parameters.parameters)
        length_par = 1
    elseif length(parameter_derivatives) ≥ 1
        for p in vec(collect(parameter_derivatives))
            @assert p ∈ 𝓂.constants.post_complete_parameters.parameters string(p) * " is not part of the free model parameters."
        end
        param_idx = indexin(parameter_derivatives |> collect |> vec, 𝓂.constants.post_complete_parameters.parameters) |> sort
        length_par = length(parameter_derivatives)
    end

    NSSS, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)

    if solution_error >= tol.nsss.acceptance_tol
        @warn "Could not find non-stochastic steady state. Solution error: $solution_error > $(tol.nsss.acceptance_tol)" maxlog = DEFAULT_MAXLOG
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        inf_val = Inf * sum(abs2, 𝓂.parameter_values)
        var_idx_fail = parse_variables_input_to_index(variables, 𝓂) |> sort
        axis1_fail = 𝓂.constants.post_model_macro.var[var_idx_fail]
        ret = Dict{Symbol,KeyedArray}()
        if non_stochastic_steady_state
            axis1_nsss_fail = [axis1_fail..., 𝓂.equations.calibration_parameters...]
            ret[:non_stochastic_steady_state] = KeyedArray(fill(inf_val, length(axis1_nsss_fail)); Variables = axis1_nsss_fail)
        end
        if mean; ret[:mean] = KeyedArray(fill(inf_val, length(axis1_fail)); Variables = axis1_fail); end
        if standard_deviation; ret[:standard_deviation] = KeyedArray(fill(inf_val, length(axis1_fail)); Variables = axis1_fail); end
        if variance; ret[:variance] = KeyedArray(fill(inf_val, length(axis1_fail)); Variables = axis1_fail); end
        if covariance; ret[:covariance] = KeyedArray(fill(inf_val, length(var_idx_fail), length(var_idx_fail)); Variables = axis1_fail, Variables2 = axis1_fail); end
        if correlation; ret[:correlation] = KeyedArray(fill(inf_val, length(var_idx_fail), length(var_idx_fail)); Variables = axis1_fail, 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 = axis1_fail); end
        return ret
    end

    if length_par * length(NSSS) > 200 && derivatives
        @info "Most of the time is spent calculating derivatives wrt parameters. If they are not needed, add `derivatives = false` as an argument to the function call." maxlog = DEFAULT_MAXLOG
    end 

    if (!variance && !standard_deviation && !non_stochastic_steady_state && !mean && !covariance && !correlation)
        derivatives = false
    end

    if parameter_derivatives != :all && (variance || standard_deviation || non_stochastic_steady_state || mean || covariance || correlation)
        derivatives = true
    end


    axis1 = 𝓂.constants.post_model_macro.var

    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.var_axis
    axis2 = 𝓂.constants.post_complete_parameters.exo_axis_plain

    # Initialize variables used across derivative/non-derivative branches
    # to satisfy JET's definite-assignment analysis
    SS = KeyedArray(collect(NSSS)[var_idx]; Variables = 𝓂.constants.post_model_macro.var[var_idx])
    var_means = KeyedArray(collect(NSSS)[var_idx]; Variables = 𝓂.constants.post_model_macro.var[var_idx])
    st_dev = var_means
    varrs = var_means
    covar_dcmp = zeros(0, 0)
    dcovariance = zeros(0, 0)
    sol = zeros(0, 0)
    state_μ = Float64[]
    autocorr = zeros(0, 0)
    autocorr_tmp = zeros(0, 0)
    ŝ_to_ŝ₂ = zeros(0, 0)
    ŝ_to_y₂ = zeros(0, 0)
    SS_and_pars = Float64[]
    dvariance_full = zeros(0, 0)
    n_cov_tuple = 0
    cov_pb = nothing
    axis3 = Symbol[]

    if derivatives
        if non_stochastic_steady_state
            axis1 = [𝓂.constants.post_model_macro.var[var_idx]...,𝓂.equations.calibration_parameters...]
    
            if any(x -> contains(string(x), "◖"), axis1)
                axis1_decomposed = decompose_name.(axis1)
                axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
            end

            axis2 = vcat(:Steady_state, 𝓂.constants.post_complete_parameters.parameters[param_idx])
        
            if any(x -> contains(string(x), "◖"), axis2)
                axis2_decomposed = decompose_name.(axis2)
                axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
            end

            (nsss_d_result, nsss_d_pb) = rrule(get_NSSS_and_parameters, 𝓂, 𝓂.parameter_values, opts = opts)
            n_ss_full = length(nsss_d_result[1])
            np = length(𝓂.parameter_values)
            dNSSS = zeros(n_ss_full, np)
            for j in 1:n_ss_full
                ∂ss = zeros(n_ss_full); ∂ss[j] = 1.0
                ∂p = nsss_d_pb((∂ss, NoTangent()))[3]
                if !(∂p isa AbstractZero); dNSSS[j, :] .= ∂p; end
            end
            dNSSS = dNSSS[:, param_idx]
            
            if length(𝓂.equations.calibration_parameters) > 0
                var_idx_ext = vcat(var_idx, 𝓂.constants.post_model_macro.nVars .+ (1:length(𝓂.equations.calibration_parameters)))
            else
                var_idx_ext = var_idx
            end

            SS =  KeyedArray(hcat(collect(NSSS[var_idx_ext]),dNSSS[var_idx_ext,:]);  Variables = axis1, Steady_state_and_∂steady_state∂parameter = axis2)
        end
        
        axis1 = 𝓂.constants.post_model_macro.var[var_idx]
        if is_bgp_model(𝓂)
            axis1 = bgp_difference_labels(𝓂, axis1)
        end

        if any(x -> contains(string(x), "◖"), axis1)
            axis1_decomposed = decompose_name.(axis1)
            axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        end

        # Hoist covariance rrule call for shared use across variance/std_dev/covariance/correlation
        if variance || standard_deviation || covariance || correlation
            if algorithm == :pruned_second_order
                _cov_result, cov_pb = rrule(calculate_second_order_moments_with_covariance, 𝓂.parameter_values, 𝓂, opts = opts)
                covar_dcmp = _cov_result[1]
                n_cov_tuple = 15
            elseif algorithm == :pruned_third_order
                cov_obs = (covariance || correlation) ? :full_covar : variables
                _cov_result, cov_pb = rrule(calculate_third_order_moments, 𝓂.parameter_values, cov_obs, 𝓂, opts = opts)
                covar_dcmp = _cov_result[1]
                n_cov_tuple = 4
            else
                _cov_result, cov_pb = rrule(calculate_covariance, 𝓂.parameter_values, 𝓂, opts = opts)
                covar_dcmp = _cov_result[1]
                sol = _cov_result[2]
                if !_cov_result[5]
                    @warn "Could not find covariance matrix. Results may contain NaN for unit-root variables."
                end
                n_cov_tuple = 5
            end

            raw_covar_dcmp = covar_dcmp
            if algorithm == :first_order
                covar_dcmp = bgp_difference_covariance(covar_dcmp, sol, 𝓂)
            end

            covariance_pullback_seed = ΔΣ -> begin
                if algorithm == :first_order && is_bgp_model(𝓂)
                    Δraw, Δsol = bgp_difference_covariance_pullback(ΔΣ, raw_covar_dcmp, sol, 𝓂)
                    return ntuple(k -> k == 1 ? Δraw : k == 2 ? Δsol : NoTangent(), n_cov_tuple)
                end
                return ntuple(k -> k == 1 ? ΔΣ : NoTangent(), n_cov_tuple)
            end

            # Compute variance Jacobian via VJP (shared by variance & std_dev)
            if variance || standard_deviation
                np_cov = length(𝓂.parameter_values)
                nv_cov = size(covar_dcmp, 1)
                dvariance_full = zeros(nv_cov, np_cov)
                for j in 1:nv_cov
                    if covar_dcmp[j,j] > eps(Float64)
                        ∂Σ = zeros(nv_cov, nv_cov); ∂Σ[j,j] = 1.0
                        seed = covariance_pullback_seed(∂Σ)
                        ∂p = cov_pb(seed)[2]
                        if !(∂p isa AbstractZero); dvariance_full[j,:] .= ∂p; end
                    end
                end
            end
        end

        if variance
            axis1 = is_bgp_model(𝓂) ?
                    bgp_difference_labels(𝓂, 𝓂.constants.post_model_macro.var[var_idx]) :
                    axis1
            axis2 = vcat(:Variance, 𝓂.constants.post_complete_parameters.parameters[param_idx])
        
            if any(x -> contains(string(x), "◖"), axis2)
                axis2_decomposed = decompose_name.(axis2)
                axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
            end

            dvariance = dvariance_full[:, param_idx]

            vari = convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64)))
            
            varrs =  KeyedArray(hcat(vari[var_idx],dvariance[var_idx,:]);  Variables = axis1, Variance_and_∂variance∂parameter = axis2)

            if standard_deviation
                axis1 = is_bgp_model(𝓂) ?
                        bgp_difference_labels(𝓂, 𝓂.constants.post_model_macro.var[var_idx]) :
                        axis1
                axis2 = vcat(:Standard_deviation, 𝓂.constants.post_complete_parameters.parameters[param_idx])
            
                if any(x -> contains(string(x), "◖"), axis2)
                    axis2_decomposed = decompose_name.(axis2)
                    axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
                end
    
                standard_dev = sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))))
                # Analytical: d(sqrt(v))/d(params) = dv/d(params) / (2*sqrt(v))
                dst_dev = dvariance_full[:, param_idx] ./ (2 .* standard_dev)

                st_dev =  KeyedArray(hcat(standard_dev[var_idx], dst_dev[var_idx, :]);  Variables = axis1, Standard_deviation_and_∂standard_deviation∂parameter = axis2)
            end
        end

        if standard_deviation
            axis2 = vcat(:Standard_deviation, 𝓂.constants.post_complete_parameters.parameters[param_idx])
        
            if any(x -> contains(string(x), "◖"), axis2)
                axis2_decomposed = decompose_name.(axis2)
                axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
            end

            standard_dev = sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))))
            # Analytical: d(sqrt(v))/d(params) = dv/d(params) / (2*sqrt(v))
            dst_dev = dvariance_full[:, param_idx] ./ (2 .* standard_dev)

            st_dev =  KeyedArray(hcat(standard_dev[var_idx], dst_dev[var_idx, :]);  Variables = axis1, Standard_deviation_and_∂standard_deviation∂parameter = axis2)
        end


        if covariance || correlation
            # Compute full covariance Jacobian via VJP from hoisted rrule
            np_cov2 = length(𝓂.parameter_values)
            nv_cov2 = size(covar_dcmp, 1)
            dcovariance = zeros(nv_cov2 * nv_cov2, np_cov2)
            for j in 1:(nv_cov2 * nv_cov2)
                r = mod1(j, nv_cov2)
                c = div(j - 1, nv_cov2) + 1
                ∂Σ = zeros(nv_cov2, nv_cov2); ∂Σ[r,c] = 1.0
                seed = covariance_pullback_seed(∂Σ)
                ∂p = cov_pb(seed)[2]
                if !(∂p isa AbstractZero); dcovariance[j,:] .= ∂p; end
            end
            dcovariance = dcovariance[:, param_idx]
        end

        if covariance
            axis3 = vcat(:Covariance, 𝓂.constants.post_complete_parameters.parameters[param_idx])
        
            if any(x -> contains(string(x), "◖"), axis3)
                axis3_decomposed = decompose_name.(axis3)
                axis3 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis3_decomposed]
            end
        end

        if mean && algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
            axis2 = vcat(:Mean, 𝓂.constants.post_complete_parameters.parameters[param_idx])
        
            if any(x -> contains(string(x), "◖"), axis2)
                axis2_decomposed = decompose_name.(axis2)
                axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
            end

            (mean_result, mean_pb) = rrule(calculate_mean, 𝓂.parameter_values, 𝓂, algorithm = algorithm, opts = opts)
            state_μ = mean_result[1]
            
            if !mean_result[2]
                @warn "Mean not found." maxlog = DEFAULT_MAXLOG
                state_μ = fill(NaN, length(state_μ))
            end

            n_mean = length(state_μ)
            np_mean = length(𝓂.parameter_values)
            state_μ_dev = zeros(n_mean, np_mean)
            for j in 1:n_mean
                ∂mean = zeros(n_mean); ∂mean[j] = 1.0
                ∂p = mean_pb((∂mean, NoTangent()))[2]
                if !(∂p isa AbstractZero); state_μ_dev[j,:] .= ∂p; end
            end
            state_μ_dev = state_μ_dev[:, param_idx]
            
            var_means =  KeyedArray(hcat(state_μ[var_idx], state_μ_dev[var_idx, :]);  Variables = axis1, Mean_and_∂mean∂parameter = axis2)
        end
    else
        if non_stochastic_steady_state
            axis1 = [𝓂.constants.post_model_macro.var[var_idx]...,𝓂.equations.calibration_parameters...]
    
            if any(x -> contains(string(x), "◖"), axis1)
                axis1_decomposed = decompose_name.(axis1)
                axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
            end

            if length(𝓂.equations.calibration_parameters) > 0
                var_idx_ext = vcat(var_idx, 𝓂.constants.post_model_macro.nVars .+ (1:length(𝓂.equations.calibration_parameters)))
            else
                var_idx_ext = var_idx
            end

            if mean && algorithm == :first_order
                var_means = KeyedArray(collect(NSSS)[var_idx];  Variables = 𝓂.constants.post_model_macro.var[var_idx])
            end

            SS =  KeyedArray(collect(NSSS)[var_idx_ext];  Variables = axis1)
        end

        axis1 = 𝓂.constants.post_model_macro.var[var_idx]
        if is_bgp_model(𝓂)
            axis1 = bgp_difference_labels(𝓂, axis1)
        end

        if any(x -> contains(string(x), "◖"), axis1)
            axis1_decomposed = decompose_name.(axis1)
            axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        end

        var_means = KeyedArray(collect(NSSS)[var_idx];  Variables = 𝓂.constants.post_model_macro.var[var_idx])

        if mean && !(variance || standard_deviation || covariance)
            state_μ, solved = calculate_mean(𝓂.parameter_values, 𝓂, algorithm = algorithm, opts = opts)

            if !solved
                @warn "Mean not found."
            end

            var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
        end

        if variance
            if algorithm == :pruned_second_order
                covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = calculate_second_order_moments_with_covariance(𝓂.parameter_values, 𝓂, opts = opts)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            elseif algorithm == :pruned_third_order
                covar_dcmp, state_μ, _, solved = calculate_third_order_moments(𝓂.parameter_values, variables, 𝓂, opts = opts)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            else
                covar_dcmp, sol, __, _, solved = calculate_covariance(𝓂.parameter_values, 𝓂, opts = opts)
                if algorithm == :first_order
                    covar_dcmp = bgp_difference_covariance(covar_dcmp, sol, 𝓂)
                end

                if mean && algorithm == :first_order
                    var_means = KeyedArray(collect(NSSS)[var_idx];  Variables = 𝓂.constants.post_model_macro.var[var_idx])
                end
            end

            if !solved
                @warn "Could not find covariance matrix."
            end

            varr = convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64)))
            varrs = KeyedArray(varr[var_idx];  Variables = axis1)
            if standard_deviation
                st_dev = KeyedArray(sqrt.(varr)[var_idx];  Variables = axis1)
            end
        end

        if standard_deviation
            if algorithm == :pruned_second_order
                covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = calculate_second_order_moments_with_covariance(𝓂.parameter_values, 𝓂, opts = opts)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            elseif algorithm == :pruned_third_order
                covar_dcmp, state_μ, _, solved = calculate_third_order_moments(𝓂.parameter_values, variables, 𝓂, opts = opts)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            else
                covar_dcmp, sol, __, _, solved = calculate_covariance(𝓂.parameter_values, 𝓂, opts = opts)
                if algorithm == :first_order
                    covar_dcmp = bgp_difference_covariance(covar_dcmp, sol, 𝓂)
                end

                if mean && algorithm == :first_order
                    var_means = KeyedArray(collect(NSSS)[var_idx];  Variables = 𝓂.constants.post_model_macro.var[var_idx])
                end
            end

            if !solved
                @warn "Could not find covariance matrix."
            end

            st_dev = KeyedArray(sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))))[var_idx];  Variables = axis1)
        end

        if covariance
            if algorithm == :pruned_second_order
                covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = calculate_second_order_moments_with_covariance(𝓂.parameter_values, 𝓂, opts = opts)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            elseif algorithm == :pruned_third_order
                covar_dcmp, state_μ, _, solved = calculate_third_order_moments(𝓂.parameter_values, :full_covar, 𝓂, opts = opts)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            else
                covar_dcmp, sol, __, _, solved = calculate_covariance(𝓂.parameter_values, 𝓂, opts = opts)
                if algorithm == :first_order
                    covar_dcmp = bgp_difference_covariance(covar_dcmp, sol, 𝓂)
                end

                if mean && algorithm == :first_order
                    var_means = KeyedArray(collect(NSSS)[var_idx];  Variables = 𝓂.constants.post_model_macro.var[var_idx])
                end

                if !solved
                    @warn "Could not find covariance matrix."
                end
            end
        end

        if correlation
            if algorithm == :pruned_second_order
                covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = calculate_second_order_moments_with_covariance(𝓂.parameter_values, 𝓂, opts = opts)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            elseif algorithm == :pruned_third_order
                covar_dcmp, state_μ, _, solved = calculate_third_order_moments(𝓂.parameter_values, :full_covar, 𝓂, opts = opts)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            else
                covar_dcmp, sol, __, _, solved = calculate_covariance(𝓂.parameter_values, 𝓂, opts = opts)
                if algorithm == :first_order
                    covar_dcmp = bgp_difference_covariance(covar_dcmp, sol, 𝓂)
                end

                if mean && algorithm == :first_order
                    var_means = KeyedArray(collect(NSSS)[var_idx];  Variables = 𝓂.constants.post_model_macro.var[var_idx])
                end

                if !solved
                    @warn "Could not find covariance matrix."
                end
            end
        end
    end

    
    ret = Dict{Symbol,KeyedArray}()
    if non_stochastic_steady_state
        # push!(ret,SS)
        ret[:non_stochastic_steady_state] = SS
    end
    if mean
        # push!(ret,var_means)
        ret[:mean] = var_means
    end
    if standard_deviation
        # push!(ret,st_dev)
        ret[:standard_deviation] = st_dev
    end
    if variance
        # push!(ret,varrs)
        ret[:variance] = varrs
    end
    if covariance
        axis1 = is_bgp_model(𝓂) ?
                bgp_difference_labels(𝓂, 𝓂.constants.post_model_macro.var[var_idx]) :
                𝓂.constants.post_model_macro.var[var_idx]

        if any(x -> contains(string(x), "◖"), axis1)
            axis1_decomposed = decompose_name.(axis1)
            axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        end

        if derivatives
            # Determine dimensions
            n_full_vars = size(covar_dcmp, 1)        # Full number of variables (n)
            n_reduced_vars = length(var_idx)         # Reduced number of variables (k)
            n_params = length(param_idx)             # Number of parameters (p)

            # Pre-allocate array to hold reduced covariance and derivatives: k x k x (1 + p)
            covar_with_derivs = zeros(n_reduced_vars, n_reduced_vars, 1 + n_params)

            # First slice is the reduced covariance matrix
            # Take the slice of the covariance matrix
            covar_with_derivs[:, :, 1] = covar_dcmp[var_idx, var_idx]

            # Subsequent slices are reduced derivatives wrt each parameter
            # The key is to reshape the full n_full_vars x n_full_vars derivative
            # and then take the slice [var_idx, var_idx]
            for i in 1:n_params
                # dcovariance[:,i] is the vectorized full derivative (n_full_vars^2 length)
                # 1. Reshape to the full n_full_vars x n_full_vars derivative matrix
                full_deriv_matrix = reshape(dcovariance[:, i], n_full_vars, n_full_vars)

                # 2. Take the reduced slice [var_idx, var_idx] and assign to the pre-allocated array
                covar_with_derivs[:, :, i+1] = full_deriv_matrix[var_idx, var_idx]
            end

            # ---
            # Create axis names (unchanged from original)
            if !@isdefined axis3
                axis3 = vcat(:Covariance, 𝓂.constants.post_complete_parameters.parameters[param_idx])

                if any(x -> contains(string(x), "◖"), axis3)
                    axis3_decomposed = decompose_name.(axis3)
                    axis3 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis3_decomposed]
                end
            end
            # ---

            # Assign the result
            # The array is already sliced, so no need for covar_with_derivs[var_idx, var_idx, :]
            ret[:covariance] = KeyedArray(covar_with_derivs;
                Variables = axis1,         # Assuming axis1 holds the full variable names
                𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 = axis1,
                Covariance_and_∂covariance∂parameter = axis3
            )
        else
            # push!(ret,KeyedArray(covar_dcmp[var_idx, var_idx]; Variables = axis1, 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 = axis1))
            ret[:covariance] = KeyedArray(covar_dcmp[var_idx, var_idx]; Variables = axis1, 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 = axis1)
        end
    end
    if correlation
        axis1 = is_bgp_model(𝓂) ?
                bgp_difference_labels(𝓂, 𝓂.constants.post_model_macro.var[var_idx]) :
                𝓂.constants.post_model_macro.var[var_idx]

        if any(x -> contains(string(x), "◖"), axis1)
            axis1_decomposed = decompose_name.(axis1)
            axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        end

        corr_full_mat, covar_sym, diag_cov, std_corr = covariance_to_correlation(covar_dcmp)

        if derivatives
            n_full_vars = size(covar_dcmp, 1)
            n_reduced_vars = length(var_idx)
            n_params = length(param_idx)

            corr_with_derivs = zeros(n_reduced_vars, n_reduced_vars, 1 + n_params)
            corr_with_derivs[:, :, 1] = corr_full_mat[var_idx, var_idx]

            for p in 1:n_params
                dΣ_full = reshape(dcovariance[:, p], n_full_vars, n_full_vars)
                for (ri, i) in enumerate(var_idx)
                    for (rj, j) in enumerate(var_idx)
                        σi = std_corr[i]
                        σj = std_corr[j]
                        if !isfinite(σi) || !isfinite(σj) || diag_cov[i] <= 0 || diag_cov[j] <= 0
                            corr_with_derivs[ri, rj, p+1] = NaN
                        else
                            # dC[i,j]/dθ = dΣ[i,j]/(σi*σj) - C[i,j]*(dΣ[i,i]/(2*Σ[i,i]) + dΣ[j,j]/(2*Σ[j,j]))
                            corr_with_derivs[ri, rj, p+1] = dΣ_full[i,j] / (σi * σj) - corr_full_mat[i,j] * (dΣ_full[i,i] / (2 * diag_cov[i]) + dΣ_full[j,j] / (2 * diag_cov[j]))
                        end
                    end
                end
            end

            axis_corr = vcat(:Correlation, 𝓂.constants.post_complete_parameters.parameters[param_idx])

            if any(x -> contains(string(x), "◖"), axis_corr)
                axis_corr_decomposed = decompose_name.(axis_corr)
                axis_corr = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis_corr_decomposed]
            end

            ret[:correlation] = KeyedArray(corr_with_derivs;
                Variables = axis1,
                𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 = axis1,
                Correlation_and_∂correlation∂parameter = axis_corr
            )
        else
            ret[:correlation] = KeyedArray(corr_full_mat[var_idx, var_idx]; Variables = axis1, 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 = axis1)
        end
    end

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    return ret
end

"""
Wrapper for [`get_moments`](@ref) with `variance = true` and `non_stochastic_steady_state = false, standard_deviation = false, covariance = false`.
"""
@unstable get_variance(args...; kwargs...) =  get_moments(args...; kwargs..., variance = true, non_stochastic_steady_state = false, standard_deviation = false, covariance = false, derivatives = get(kwargs, :derivatives, true))[:variance]


"""
Wrapper for [`get_moments`](@ref) with `variance = true` and `non_stochastic_steady_state = false, standard_deviation = false, covariance = false`.
"""
@unstable get_var = get_variance


"""
Wrapper for [`get_moments`](@ref) with `variance = true` and `non_stochastic_steady_state = false, standard_deviation = false, covariance = false`.
"""
@unstable var = get_variance


"""
Wrapper for [`get_moments`](@ref) with `standard_deviation = true` and `non_stochastic_steady_state = false, variance = false, covariance = false`.
"""
@unstable get_standard_deviation(args...; kwargs...) =  get_moments(args...; kwargs..., variance = false, non_stochastic_steady_state = false, standard_deviation = true, covariance = false, derivatives = get(kwargs, :derivatives, true))[:standard_deviation]


"""
Wrapper for [`get_moments`](@ref) with `standard_deviation = true` and `non_stochastic_steady_state = false, variance = false, covariance = false`.
"""
@unstable get_std =  get_standard_deviation

"""
Wrapper for [`get_moments`](@ref) with `standard_deviation = true` and `non_stochastic_steady_state = false, variance = false, covariance = false`.
"""
@unstable get_stdev =  get_standard_deviation


"""
Wrapper for [`get_moments`](@ref) with `standard_deviation = true` and `non_stochastic_steady_state = false, variance = false, covariance = false`.
"""
@unstable stdev =  get_standard_deviation


"""
Wrapper for [`get_moments`](@ref) with `standard_deviation = true` and `non_stochastic_steady_state = false, variance = false, covariance = false`.
"""
@unstable std =  get_standard_deviation

"""
Wrapper for [`get_moments`](@ref) with `covariance = true` and `non_stochastic_steady_state = false, variance = false, standard_deviation = false, derivatives = false`.
"""
@unstable get_covariance(args...; kwargs...) =  get_moments(args...; kwargs..., variance = false, non_stochastic_steady_state = false, standard_deviation = false, covariance = true, derivatives = get(kwargs, :derivatives, false))[:covariance]


"""
Wrapper for [`get_moments`](@ref) with `covariance = true` and `non_stochastic_steady_state = false, variance = false, standard_deviation = false`.
"""
@unstable get_cov = get_covariance


"""
Wrapper for [`get_moments`](@ref) with `covariance = true` and `non_stochastic_steady_state = false, variance = false, standard_deviation = false`.
"""
@unstable cov = get_covariance


"""
Wrapper for [`get_moments`](@ref) with `mean = true`, and `non_stochastic_steady_state = false, variance = false, standard_deviation = false, covariance = false`
"""
@unstable get_mean(args...; kwargs...) =  get_moments(args...; kwargs..., variance = false, non_stochastic_steady_state = false, standard_deviation = false, covariance = false, mean = true, derivatives = get(kwargs, :derivatives, true))[:mean]


# """
# Wrapper for [`get_moments`](@ref) with `mean = true`, the default algorithm being `:pruned_second_order`, and `non_stochastic_steady_state = false, variance = false, standard_deviation = false, covariance = false`
# """
# mean(𝓂::ℳ; kwargs...) = get_mean(𝓂; kwargs...)



"""
$(SIGNATURES)
Return the first and second moments of endogenous variables using either the linearised solution or the pruned second or pruned third order perturbation solution. By default returns a `Dict` with: non-stochastic steady state (NSSS), and standard deviations, but can also return variances, covariance matrix, and correlation matrix. Values are returned in the order given for the specific moment.
Function to use when differentiating model moments with respect to parameters.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- `parameter_values` [Type: `Vector`]: Parameter values. If `parameter_names` is not explicitly defined, `parameter_values` are assumed to correspond to the parameters and the order of the parameters declared in the `@parameters` block.
# Keyword Arguments
- `parameters` [Type: `Vector{Symbol}`]: Corresponding names in the same order as `parameter_values`.
- `non_stochastic_steady_state` [Default: `Symbol[]`, Type: `Union{Symbol_input,String_input}`]: variables for which to show the NSSS of selected variables. Inputs can be a variable name passed on as either a `Symbol` or `String` (e.g. `:y` or `\"y\"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. Any variables not part of the model will trigger a warning. `:all_excluding_auxiliary_and_obc` contains all shocks less those related to auxiliary variables and related to occasionally binding constraints (obc). `:all_excluding_obc` contains all shocks less those related to auxiliary variables. `:all` will contain all variables.
- `mean` [Default: `Symbol[]`, Type: `Union{Symbol_input,String_input}`]: variables for which to show the mean of selected variables (the mean for the linearised solution is the NSSS). Inputs can be a variable name passed on as either a `Symbol` or `String` (e.g. `:y` or `\"y\"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. Any variables not part of the model will trigger a warning. `:all_excluding_auxiliary_and_obc` contains all shocks less those related to auxiliary variables and related to occasionally binding constraints (obc). `:all_excluding_obc` contains all shocks less those related to auxiliary variables. `:all` will contain all variables.
- `standard_deviation` [Default: `Symbol[]`, Type: `Union{Symbol_input,String_input}`]: variables for which to show the standard deviation of selected variables. Inputs can be a variable name passed on as either a `Symbol` or `String` (e.g. `:y` or `\"y\"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. Any variables not part of the model will trigger a warning. `:all_excluding_auxiliary_and_obc` contains all shocks less those related to auxiliary variables and related to occasionally binding constraints (obc). `:all_excluding_obc` contains all shocks less those related to auxiliary variables. `:all` will contain all variables.
- `variance` [Default: `Symbol[]`, Type: `Union{Symbol_input,String_input}`]: variables for which to show the variance of selected variables. Inputs can be a variable name passed on as either a `Symbol` or `String` (e.g. `:y` or `\"y\"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. Any variables not part of the model will trigger a warning. `:all_excluding_auxiliary_and_obc` contains all shocks less those related to auxiliary variables and related to occasionally binding constraints (obc). `:all_excluding_obc` contains all shocks less those related to auxiliary variables. `:all` will contain all variables.
- `covariance` [Default: `Symbol[]`, Type: `Union{Symbol_input,String_input}`]: variables for which to show the covariance of selected variables. Inputs can be a variable name passed on as either a `Symbol` or `String` (e.g. `:y` or `\"y\"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. For grouped covariance computation, pass a `Vector` of `Vector`s (e.g. `[[:y, :c], [:k, :i]]`) to compute covariances only within each group, returning a single covariance matrix where cross-group covariances are set to zero. This allows more granular control over which covariances to compute. Any variables not part of the model will trigger a warning. `:all_excluding_auxiliary_and_obc` contains all variables less those related to auxiliary variables and related to occasionally binding constraints (obc). `:all_excluding_obc` contains all variables less those related to occasionally binding constraints. `:all` will contain all variables.
- `correlation` [Default: `Symbol[]`, Type: `Union{Symbol_input,String_input}`]: variables for which to show the correlation matrix of selected variables. Inputs follow the same format as `covariance`, including grouped input (e.g. `[[:y, :c], [:k, :i]]`) which restricts the returned matrix to within-group correlations and sets cross-group entries to zero. Variables with non-positive variance produce `NaN` entries (left unchanged). `:all_excluding_auxiliary_and_obc` contains all variables less those related to auxiliary variables and related to occasionally binding constraints (obc). `:all_excluding_obc` contains all variables less those related to occasionally binding constraints. `:all` will contain all variables.
- `autocorrelation` [Default: `Symbol[]`, Type: `Union{Symbol_input,String_input}`]: variables for which to show the autocorrelation of selected variables. Inputs can be a variable name passed on as either a `Symbol` or `String` (e.g. `:y` or `\"y\"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. Any variables not part of the model will trigger a warning. `:all_excluding_auxiliary_and_obc` contains all shocks less those related to auxiliary variables and related to occasionally binding constraints (obc). `:all_excluding_obc` contains all shocks less those related to auxiliary variables. `:all` will contain all variables.
- `autocorrelation_periods` [Default: `1:5`, Type = `UnitRange{Int}`]: periods for which to return the autocorrelation of selected variables
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $QME®
- $LYAPUNOV®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Dict` with the name of the statistics and the corresponding vectors (NSSS, mean, standard deviation, variance) or matrices (covariance, correlation, autocorrelation).

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_statistics(RBC, RBC.parameter_values, standard_deviation = get_variables(RBC))
# output
Dict{Symbol, AbstractArray{Float64}} with 1 entry:
  :standard_deviation => [0.0266642, 0.264677, 0.0739325, 0.0102062]
```

For grouped covariance (computing covariances only within specified groups; cross-group
entries are set to zero):
```julia
get_statistics(RBC, RBC.parameter_values, covariance = [[:c, :k], [:q, :z]])
# Dict{Symbol, AbstractArray{Float64}} with 1 entry:
#   :covariance => [0.00071098 0.00705609 0.0 0.0; 0.0 0.0700541 0.0 0.0; 0.0 0.0…
```

For correlation (returns the correlation matrix among the selected variables;
diagonal is 1; supports the same grouped input as `covariance`, with cross-group
entries set to zero):
```julia
get_statistics(RBC, RBC.parameter_values, correlation = [:c, :k])
# Dict{Symbol, AbstractArray{Float64}} with 1 entry:
#   :correlation => [1.0 0.999812; 0.999812 1.0]
```
"""
function get_statistics(𝓂::ℳ,
                        parameter_values::Vector{T};
                        parameters::Union{Vector{Symbol},Vector{String}} = 𝓂.constants.post_complete_parameters.parameters,
                        steady_state_function::SteadyStateFunctionType = missing, 
                        non_stochastic_steady_state::Union{Symbol_input,String_input} = Symbol[],
                        mean::Union{Symbol_input,String_input} = Symbol[],
                        standard_deviation::Union{Symbol_input,String_input} = Symbol[],
                        variance::Union{Symbol_input,String_input} = Symbol[],
                        covariance::Union{Symbol_input,String_input, Vector{Vector{Symbol}},Vector{Tuple{Symbol,Vararg{Symbol}}},Vector{Vector{Symbol}},Tuple{Tuple{Symbol,Vararg{Symbol}},Vararg{Tuple{Symbol,Vararg{Symbol}}}}, Vector{Vector{String}},Vector{Tuple{String,Vararg{String}}},Vector{Vector{String}},Tuple{Tuple{String,Vararg{String}},Vararg{Tuple{String,Vararg{String}}}}} = Symbol[],
                        correlation::Union{Symbol_input,String_input, Vector{Vector{Symbol}},Vector{Tuple{Symbol,Vararg{Symbol}}},Vector{Vector{Symbol}},Tuple{Tuple{Symbol,Vararg{Symbol}},Vararg{Tuple{Symbol,Vararg{Symbol}}}}, Vector{Vector{String}},Vector{Tuple{String,Vararg{String}}},Vector{Vector{String}},Tuple{Tuple{String,Vararg{String}},Vararg{Tuple{String,Vararg{String}}}}} = Symbol[],
                        autocorrelation::Union{Symbol_input,String_input} = Symbol[],
                        autocorrelation_periods::UnitRange{Int} = DEFAULT_AUTOCORRELATION_PERIODS,
                        algorithm::Symbol = DEFAULT_ALGORITHM,
                        quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                        sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                        lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                        verbose::Bool = DEFAULT_VERBOSE,
                        tol::Tolerances = Tolerances(),
                        caching::Bool = DEFAULT_CACHING,
                        use_workspaces::Bool = DEFAULT_USE_WORKSPACES) where T

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                        sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                        sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                        lyapunov_algorithm = lyapunov_algorithm)

    @assert length(parameter_values) == length(parameters) "Vector of `parameters` must correspond to `parameter_values` in length and order. Define the parameter names in the `parameters` keyword argument."
    
    @assert algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] || !(!(standard_deviation == Symbol[]) || !(mean == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(correlation == Symbol[]) || !(autocorrelation == Symbol[])) "Statistics can only be provided for first order perturbation or second and third order pruned perturbation solutions."

    @assert !(non_stochastic_steady_state == Symbol[]) || !(standard_deviation == Symbol[]) || !(mean == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(correlation == Symbol[]) || !(autocorrelation == Symbol[]) "Provide variables for at least one output."

    SS_var_idx = parse_variables_input_to_index(non_stochastic_steady_state, 𝓂)

    mean_var_idx = parse_variables_input_to_index(mean, 𝓂)

    std_var_idx = parse_variables_input_to_index(standard_deviation, 𝓂)

    var_var_idx = parse_variables_input_to_index(variance, 𝓂)

    covar_var_idx = parse_variables_input_to_index(covariance, 𝓂)
    
    # Parse covariance groups if input is grouped format
    covar_groups = is_grouped_covariance_input(covariance) ? parse_covariance_groups(covariance, 𝓂.constants) : nothing

    corr_var_idx = parse_variables_input_to_index(correlation, 𝓂)

    corr_groups = is_grouped_covariance_input(correlation) ? parse_covariance_groups(correlation, 𝓂.constants) : nothing

    autocorr_var_idx = parse_variables_input_to_index(autocorrelation, 𝓂)


    other_parameter_values = 𝓂.parameter_values[indexin(setdiff(𝓂.constants.post_complete_parameters.parameters, parameters), 𝓂.constants.post_complete_parameters.parameters)]

    sort_idx = sortperm(vcat(indexin(setdiff(𝓂.constants.post_complete_parameters.parameters, parameters), 𝓂.constants.post_complete_parameters.parameters), indexin(parameters, 𝓂.constants.post_complete_parameters.parameters)))

    all_parameters = vcat(other_parameter_values, parameter_values)[sort_idx]

    solved = true

    if algorithm == :pruned_third_order && !(!(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(correlation == Symbol[]) || !(autocorrelation == Symbol[]))
        algorithm = :pruned_second_order
    end

    solve!(𝓂, 
           algorithm = algorithm, 
           steady_state_function = steady_state_function,
           opts = opts)

    if !(non_stochastic_steady_state == Symbol[]) && (standard_deviation == Symbol[]) && (variance == Symbol[]) && (covariance == Symbol[]) && (correlation == Symbol[]) && (autocorrelation == Symbol[])
        SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, all_parameters, opts = opts) # timer = timer, 
        
        SS = SS_and_pars[1:end - length(𝓂.equations.calibration)]

        ret = Dict{Symbol,AbstractArray{T}}()

        ret[:non_stochastic_steady_state] = solution_error < opts.tol.nsss.acceptance_tol ? SS[SS_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(SS_var_idx) ? 0 : length(SS_var_idx))

        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return ret
    end

    # Initialize variables that are conditionally assigned across algorithm branches
    # to satisfy JET's definite-assignment analysis. Each is overwritten in the
    # relevant branch below before it is actually used.
    nVars = 𝓂.constants.post_model_macro.nVars
    SS_and_pars = zeros(T, 0)
    covar_dcmp  = zeros(T, 0, 0)
    state_μ     = zeros(T, 0)
    sol         = zeros(T, 0, 0)
    autocorr_tmp = zeros(T, 0, 0)
    ŝ_to_ŝ₂    = zeros(T, 0, 0)
    ŝ_to_y₂    = zeros(T, 0, 0)
    autocorr    = zeros(T, 0, 0)
    varrs       = zeros(T, 0)
    st_dev      = zeros(T, 0)
    solved      = false

    if algorithm == :pruned_third_order

        if !(autocorrelation == Symbol[])
            second_mom_third_order = union(autocorr_var_idx, std_var_idx, var_var_idx, corr_var_idx)

            covar_dcmp, state_μ, autocorr, SS_and_pars, solved = calculate_third_order_moments_with_autocorrelation(all_parameters, 𝓂.constants.post_model_macro.var[second_mom_third_order], 𝓂, covariance = 𝓂.constants.post_model_macro.var[union(covar_var_idx, corr_var_idx)], opts = opts, autocorrelation_periods = autocorrelation_periods)

        elseif !(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(correlation == Symbol[])

            covar_dcmp, state_μ, SS_and_pars, solved = calculate_third_order_moments(all_parameters, 𝓂.constants.post_model_macro.var[union(std_var_idx, var_var_idx, corr_var_idx)], 𝓂, covariance = 𝓂.constants.post_model_macro.var[union(covar_var_idx, corr_var_idx)], opts = opts)

        end

    elseif algorithm == :pruned_second_order

        if !(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(correlation == Symbol[]) || !(autocorrelation == Symbol[])
            covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = calculate_second_order_moments_with_covariance(all_parameters, 𝓂, opts = opts)
        else
            state_μ, Δμˢ₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = calculate_second_order_moments(all_parameters, 𝓂, opts = opts)
        end

    else
        covar_dcmp, sol, _, SS_and_pars, solved = calculate_covariance(all_parameters, 𝓂, opts = opts)

        # @assert solved "Could not find covariance matrix."
    end

    if algorithm == :first_order
        covar_dcmp = bgp_difference_covariance(covar_dcmp, sol, 𝓂)
    end

    SS = SS_and_pars[1:end - length(𝓂.equations.calibration)]

    if solved && !(variance == Symbol[])
        varrs = convert(Vector{T},max.(ℒ.diag(covar_dcmp),eps(Float64)))
        if !(standard_deviation == Symbol[])
            st_dev = sqrt.(varrs)
        end
    end

    if solved && !(autocorrelation == Symbol[])
        if algorithm == :pruned_second_order
            ŝ_to_ŝ₂ⁱ = zero(ŝ_to_ŝ₂)
            ŝ_to_ŝ₂ⁱ += ℒ.diagm(ones(size(ŝ_to_ŝ₂,1)))

            autocorr = zeros(T,size(covar_dcmp,1),length(autocorrelation_periods))

            for i in autocorrelation_periods
                autocorr[:,i] .= ℒ.diag(ŝ_to_y₂ * ŝ_to_ŝ₂ⁱ * autocorr_tmp) ./ max.(ℒ.diag(covar_dcmp),eps(Float64))
                ŝ_to_ŝ₂ⁱ *= ŝ_to_ŝ₂
            end
            
            autocorr[ℒ.diag(covar_dcmp) .< opts.tol.first_order.lyapunov.acceptance_tol,:] .= 0
        elseif !(algorithm == :pruned_third_order)
            A = @views sol[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.constants.post_model_macro.nVars))[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,:]
        
            autocorr = reduce(hcat,[ℒ.diag(A ^ i * covar_dcmp ./ max.(ℒ.diag(covar_dcmp),eps(Float64))) for i in autocorrelation_periods])

            autocorr[ℒ.diag(covar_dcmp) .< opts.tol.first_order.lyapunov.acceptance_tol,:] .= 0
        end
    end

    if solved && !(standard_deviation == Symbol[])
        st_dev = sqrt.(abs.(convert(Vector{T}, max.(ℒ.diag(covar_dcmp),eps(Float64)))))
    end
        

    # ret = AbstractArray{T}[]
    ret = Dict{Symbol,AbstractArray{T}}()

    if !(non_stochastic_steady_state == Symbol[])
        # push!(ret,SS[SS_var_idx])
        ret[:non_stochastic_steady_state] = solved ? SS[SS_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(SS_var_idx) ? 0 : length(SS_var_idx))
    end
    if !(mean == Symbol[])
        if algorithm ∉ [:pruned_second_order,:pruned_third_order]
            # push!(ret,SS[mean_var_idx])
            ret[:mean] = solved ? SS[mean_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(mean_var_idx) ? 0 : length(mean_var_idx))
        else
            # push!(ret,state_μ[mean_var_idx])
            ret[:mean] = solved ? state_μ[mean_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(mean_var_idx) ? 0 : length(mean_var_idx))
        end
    end
    if !(standard_deviation == Symbol[])
        # push!(ret,st_dev[std_var_idx])
        ret[:standard_deviation] = solved ? st_dev[std_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(std_var_idx) ? 0 : length(std_var_idx))
    end
    if !(variance == Symbol[])
        # push!(ret,varrs[var_var_idx])
        ret[:variance] = solved ? varrs[var_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(var_var_idx) ? 0 : length(var_var_idx))
    end
    if !(covariance == Symbol[])
        covar_dcmp_sp = (ℒ.triu(covar_dcmp))

        # droptol!(covar_dcmp_sp,eps(Float64))

        if !isnothing(covar_groups)
            # Extract only the specified covariance groups (block diagonal structure)
            # Return a single matrix with zeros for non-computed covariances
            if solved
                # Initialize matrix with zeros
                covar_result = zeros(T, length(covar_var_idx), length(covar_var_idx))
                
                # Fill in only the specified groups
                for group in covar_groups
                    for (i_idx, i) in enumerate(group)
                        for (j_idx, j) in enumerate(group)
                            # Find position in covar_var_idx
                            i_pos = findfirst(==(i), covar_var_idx)
                            j_pos = findfirst(==(j), covar_var_idx)
                            if !isnothing(i_pos) && !isnothing(j_pos)
                                covar_result[i_pos, j_pos] = covar_dcmp_sp[i, j]
                            end
                        end
                    end
                end
                
                ret[:covariance] = covar_result
            else
                # Return matrix with Inf-filled diagonal and zeros elsewhere
                covar_result = fill(Inf * sum(abs2,parameter_values), length(covar_var_idx), length(covar_var_idx))
                ret[:covariance] = covar_result
            end
        else
            # Original behavior for non-grouped input
            # push!(ret,covar_dcmp_sp[covar_var_idx,covar_var_idx])
            ret[:covariance] = solved ? covar_dcmp_sp[covar_var_idx,covar_var_idx] : fill(Inf * sum(abs2,parameter_values),isnothing(covar_var_idx) ? 0 : length(covar_var_idx), isnothing(covar_var_idx) ? 0 : length(covar_var_idx))
        end
    end
    if !(correlation == Symbol[])
        if solved
            corr_full_mat, _, _, _ = covariance_to_correlation(covar_dcmp)

            if !isnothing(corr_groups)
                # Block-grouped correlation: cross-group entries left as zero
                corr_result = zeros(T, length(corr_var_idx), length(corr_var_idx))
                for group in corr_groups
                    for i in group
                        i_pos = findfirst(==(i), corr_var_idx)
                        isnothing(i_pos) && continue
                        for j in group
                            j_pos = findfirst(==(j), corr_var_idx)
                            isnothing(j_pos) && continue
                            corr_result[i_pos, j_pos] = corr_full_mat[i, j]
                        end
                    end
                end
                ret[:correlation] = corr_result
            else
                ret[:correlation] = corr_full_mat[corr_var_idx, corr_var_idx]
            end
        else
            ret[:correlation] = fill(Inf * sum(abs2, parameter_values), isnothing(corr_var_idx) ? 0 : length(corr_var_idx), isnothing(corr_var_idx) ? 0 : length(corr_var_idx))
        end
    end
    if !(autocorrelation == Symbol[]) 
        # push!(ret,autocorr[autocorr_var_idx,:] )
        ret[:autocorrelation] = solved ? autocorr[autocorr_var_idx,:] : fill(Inf * sum(abs2,parameter_values), isnothing(autocorr_var_idx) ? 0 : length(autocorr_var_idx), isnothing(autocorrelation_periods) ? 0 : length(autocorrelation_periods))
    end

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    return ret
end

"""
$(SIGNATURES)
Return the loglikelihood of the model given the data and parameters provided. The loglikelihood is either calculated based on the inversion or the Kalman filter (depending on the `filter` keyword argument). By default the package selects the Kalman filter for first order solutions and the inversion filter for nonlinear (higher order) solution algorithms. The data must be provided as a `KeyedArray{Float64}` with the names of the variables to be matched in rows and the periods in columns. The `KeyedArray` type is provided by the `AxisKeys` package.

This function is differentiable and supports both the Kalman and inversion likelihoods.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- $DATA®
- `parameter_values` [Type: `Vector`]: Parameter values.
# Keyword Arguments
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $FILTER®
- $WARMUP_ITERATIONS®
- `presample_periods` [Default: `0`, Type: `Int`]: periods at the beginning of the retained data sample for which the loglikelihood is discarded. Values above the retained sample length are clamped down automatically with an informational message.
- `initial_covariance` [Default: `:theoretical`, Type: `Union{Symbol,AbstractMatrix{<:Real}}`]: defines the method to initialise the Kalman filters covariance matrix. It can be initialised with the theoretical long run values (option `:theoretical`), large values (10.0) along the diagonal (option `:diagonal`), or a user-supplied matrix of appropriate size (number of observables and states).
- $INITIAL_STATE®
- `on_failure_loglikelihood` [Default: `-Inf`, Type: `AbstractFloat`]: value to return if the loglikelihood calculation fails. Setting this to a finite value can avoid errors in codes that rely on finite loglikelihood values, such as e.g. slice samplers (in Pigeons.jl).
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `<:AbstractFloat` loglikelihood 

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

import Random; Random.seed!(3)

simulated_data = simulate(RBC)

get_loglikelihood(RBC, simulated_data([:k], :, :simulate), RBC.parameter_values)
# output
53.76735680353869
```
"""
function get_loglikelihood(𝓂::ℳ, 
                            data::KeyedArray{T}, 
                            parameter_values::Vector{S}; 
                            steady_state_function::SteadyStateFunctionType = missing, 
                            algorithm::Symbol = DEFAULT_ALGORITHM, 
                            filter::Symbol = DEFAULT_FILTER_SELECTOR(algorithm), 
                            on_failure_loglikelihood::U = -Inf,
                            warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS, 
                            presample_periods::Int = DEFAULT_PRESAMPLE_PERIODS,
                            initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
                            filter_algorithm::Symbol = :LagrangeNewton,
                            tol::Tolerances = Tolerances(), 
                            quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂), 
                            lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM, 
                            sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                            verbose::Bool = DEFAULT_VERBOSE,
                            caching::Bool = DEFAULT_CACHING,
                            use_workspaces::Bool = DEFAULT_USE_WORKSPACES)::S where {T <: Union{Float64,Missing,Nothing}, S <: Real, U <: AbstractFloat}
    # Convenience method: no `initial_state` argument; uses the internal default.
    # To override the initial state (and get AD tangents through it), call the
    # positional method `get_loglikelihood(𝓂, data, p, initial_state; ...)`.
    return get_loglikelihood(𝓂, data, parameter_values, DEFAULT_INITIAL_STATE;
                             steady_state_function = steady_state_function,
                             algorithm = algorithm,
                             filter = filter,
                             on_failure_loglikelihood = on_failure_loglikelihood,
                             warmup_iterations = warmup_iterations,
                             presample_periods = presample_periods,
                             initial_covariance = initial_covariance,
                             filter_algorithm = filter_algorithm,
                             tol = tol,
                             quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                             lyapunov_algorithm = lyapunov_algorithm,
                             sylvester_algorithm = sylvester_algorithm,
                             verbose = verbose,
                             caching = caching,
                             use_workspaces = use_workspaces)
end

function get_loglikelihood(𝓂::ℳ,
                            data::KeyedArray{T},
                            parameter_values::Vector{S},
                            initial_state::InitialState;
                            steady_state_function::SteadyStateFunctionType = missing,
                            algorithm::Symbol = DEFAULT_ALGORITHM,
                            filter::Symbol = DEFAULT_FILTER_SELECTOR(algorithm),
                            on_failure_loglikelihood::U = -Inf,
                            warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS,
                            presample_periods::Int = DEFAULT_PRESAMPLE_PERIODS,
                            initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
                            filter_algorithm::Symbol = :LagrangeNewton,
                            tol::Tolerances = Tolerances(),
                            quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                            lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                            sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                            verbose::Bool = DEFAULT_VERBOSE,
                            caching::Bool = DEFAULT_CACHING,
                            use_workspaces::Bool = DEFAULT_USE_WORKSPACES)::promote_type(S, InitialStateScalar) where {T <: Union{Float64,Missing,Nothing}, S <: Real, InitialStateScalar <: Real, InitialState <: Union{AbstractVector{InitialStateScalar}, AbstractVector{<:AbstractVector{InitialStateScalar}}}, U <: AbstractFloat}

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    sylv²::Symbol = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1]
    sylv³::Symbol = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ?
        (sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM) :
        sylvester_algorithm[2]

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                            sylvester_algorithm² = sylv²,
                            sylvester_algorithm³ = sylv³,
                            lyapunov_algorithm = lyapunov_algorithm)

    estimation = true

    # if algorithm ∈ [:third_order,:pruned_third_order]
    #     sylvester_algorithm = :bicgstab
    # end

    @assert length(parameter_values) == length(𝓂.constants.post_complete_parameters.parameters) "The number of parameter values provided does not match the number of parameters in the model. If this function is used in the context of estimation and not all parameters are estimated, the estimated parameters need to be combined with the other model parameters in one `Vector`. Ensure they have the same order they were declared in the `@parameters` block (check by calling `get_parameters`)."

    # checks to avoid errors further down the line and inform the user
    @assert initial_covariance isa AbstractMatrix || initial_covariance ∈ [:theoretical, :diagonal] "Invalid method to initialise the Kalman filters covariance matrix. Supported methods are: the theoretical long run values (option `:theoretical`), large values (10.0) along the diagonal (option `:diagonal`), or a user-supplied matrix."

    if initial_state != DEFAULT_INITIAL_STATE
        nVars = 𝓂.constants.post_model_macro.nVars
        if eltype(initial_state) <: Real
            @assert length(initial_state) == nVars "initial_state must have length equal to the total number of variables ($nVars, see `show(model)`), got $(length(initial_state))."
        else
            @assert length(initial_state[1]) == nVars "Each vector in initial_state must have length equal to the total number of variables ($nVars, see `show(model)`), got $(length(initial_state[1]))."
            @assert all(v -> length(v) == length(initial_state[1]), initial_state) "All vectors in initial_state must have the same length, got lengths $(length.(initial_state))."
        end
    end
    
    filter, _, algorithm, _, _, warmup_iterations = normalize_filtering_options(filter, false, algorithm, false, warmup_iterations)

    observables = get_and_check_observables(𝓂.constants.post_model_macro, data)

    solve!(𝓂, 
           opts = opts,
           steady_state_function = steady_state_function,
           # timer = timer, 
           algorithm = algorithm)

    bounds_violated = check_bounds(parameter_values, 𝓂)

    if bounds_violated 
        # println("Bounds violated")
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return on_failure_loglikelihood
    end

    SS_and_pars_names = 𝓂.constants.post_complete_parameters.SS_and_pars_names

    obs_indices = convert(Vector{Int}, indexin(observables, SS_and_pars_names))

    # @timeit_debug timer "Get relevant steady state and solution" begin

    constants_obj, SS_and_pars, 𝐒, state, solved = get_relevant_steady_state_and_state_update(Val(algorithm), parameter_values, 𝓂, opts = opts, estimation = estimation)
                                                                                    # timer = timer,

    # end # timeit_debug

    if !solved 
        # println("Main call: 1st order solution not found")
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return on_failure_loglikelihood 
    end

    # Overwrite the solver-produced `state` with the user-supplied `initial_state`
    # (semantics match get_irf: Vector{Float64} = levels, Vector{Vector{Float64}} =
    # deviations from NSSS). The downstream filter recursions consume `state`
    # directly, so this is the only place initial_state needs to be applied.
    nVars = 𝓂.constants.post_model_macro.nVars
    if initial_state isa AbstractVector{<:Real}
        if length(initial_state) == nVars
            state_shift = state isa AbstractVector{<:AbstractVector{<:Real}} ? (length(state) == 1 ? zero(state[1]) : -state[2]) : zero(state)
            state = adjust_initial_state(initial_state, algorithm, nVars, state_shift, SS_and_pars[1:nVars])
            if algorithm == :first_order
                state = [state]
            end
        end
    elseif !isempty(initial_state)
        if state isa AbstractVector{<:AbstractVector{<:Real}}
            R_state = promote_type(eltype(eltype(state)), eltype(initial_state[1]))
            state = [convert(Vector{R_state}, i <= length(initial_state) ? initial_state[i] : state[i]) for i in eachindex(state)]
        else
            R_state = promote_type(eltype(state), eltype(initial_state[1]))
            state = convert(Vector{R_state}, initial_state[1])
        end
    end
 
    data_keyed::KeyedArray = collect(axiskeys(data, 1)) isa Vector{String} ?
        rekey(data, 1 => axiskeys(data, 1) .|> Meta.parse .|> replace_indices) :
        data

    # Canonicalise the raw observations to Float64 with NaN sentinels for
    # missing/nothing entries, then preserve the promoted element type after
    # subtracting steady-state values so AD inputs can flow through. Fully
    # unobserved periods at the sample boundaries are trimmed before the
    # filter kernels see the data.
    dt::Matrix{Float64} = missing_data_to_nan(collect(data_keyed(observables)))

    data_in_deviations, obs_idx_per_t, has_missing, _ = trim_informative_sample(dt .- SS_and_pars[obs_indices])

    # Keep the solution, data, and user-supplied state on one scalar type so
    # Dual numbers introduced solely through `initial_state` reach the filter.
    R = promote_type(S, InitialStateScalar)
    if 𝐒 isa AbstractVector{<:AbstractMatrix} && R !== eltype(eltype(𝐒))
        𝐒 = AbstractMatrix{R}[R.(Sᵢ) for Sᵢ in 𝐒]
    elseif 𝐒 isa AbstractMatrix && R !== eltype(𝐒)
        𝐒 = convert(Matrix{R}, 𝐒)
    end
    if R !== eltype(data_in_deviations)
        data_in_deviations = convert(Matrix{R}, data_in_deviations)
    end

    presample_periods = normalize_presample_periods(presample_periods, size(data_in_deviations, 2))

    if size(data_in_deviations, 2) == 0
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return zero(S)
    end

    # @timeit_debug timer "Filter" begin

    llh = if has_missing
        calculate_loglikelihood_with_missing(Val(filter),
                                    Val(algorithm),
                                    obs_indices,
                                    𝐒,
                                    data_in_deviations,
                                    constants_obj,
                                    state,
                                    𝓂.workspaces,
                                    obs_idx_per_t,
                                    warmup_iterations = warmup_iterations,
                                    presample_periods = presample_periods,
                                    initial_covariance = initial_covariance,
                                    filter_algorithm = filter_algorithm,
                                    opts = opts,
                                    on_failure_loglikelihood = on_failure_loglikelihood)
    else
        calculate_loglikelihood(Val(filter),
                                Val(algorithm),
                                obs_indices,
                                𝐒,
                                data_in_deviations,
                                constants_obj,
                                state,
                                𝓂.workspaces,
                                warmup_iterations = warmup_iterations,
                                presample_periods = presample_periods,
                                initial_covariance = initial_covariance,
                                filter_algorithm = filter_algorithm,
                                opts = opts,
                                on_failure_loglikelihood = on_failure_loglikelihood) # timer = timer
    end

    # end # timeit_debug

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    return llh
end

"""
$(SIGNATURES)
Return the *filter-free* loglikelihood of the model given the data, parameters, and a path of latent structural shocks. Unlike [`get_loglikelihood`](@ref) — which integrates out the latent shocks via a Kalman or inversion filter — this function evaluates the joint likelihood of data and shocks by forward-simulating the model with the supplied `shocks` and comparing the implied observable path to the data under a Gaussian measurement-error model. This is the building block needed to estimate (potentially nonlinear, non-Gaussian) DSGE models with HMC samplers by treating the shocks as additional latent parameters (Childers, Fernández-Villaverde, Perla, Rackauckas & Wu, 2025).

Only the *measurement* part of the joint loglikelihood is returned. The prior on the shocks (typically standard Normal) and the prior on `measurement_error_std` are expected to be declared by the user in their probabilistic-programming model.

The filter-free primal path and its analytical reverse-mode `rrule` are implemented for `:first_order`, `:second_order`, `:pruned_second_order`, `:third_order`, and `:pruned_third_order`.

# Arguments
- $MODEL®
- $DATA®
- If fully unobserved leading or trailing periods are discarded, any separately supplied shock path and any matrix-valued `measurement_error_std` input are aligned to the retained sample automatically.
- `parameter_values` [Type: `Vector`]: Parameter values.
- `shocks` [Type: `AbstractMatrix`]: Matrix of latent structural shocks with shape `nExo × (T + max(warmup_iterations - 1, 0))`, where `T` matches the number of observations in `data`. When `warmup_iterations > 1`, the leading `warmup_iterations - 1` shock columns are used only to warm the latent state before the first scored observation.
- `measurement_error_std` [Type: `Real`, `AbstractVector`, or `AbstractMatrix`]: Standard deviation(s) of the Gaussian measurement error added to each observable. Pass a scalar to use the same measurement-error std-dev on every observable, a vector of length equal to the number of observables for observable-specific std-devs, or a matrix of shape `(n_observables, n_periods)` for period-specific measurement error. If fully unobserved leading or trailing periods are discarded, matrix-valued inputs are trimmed to the same retained sample automatically.

# Keyword Arguments
- $STEADY_STATE_FUNCTION®
- `algorithm` [Default: `:second_order`, Type: `Symbol`]: solution algorithm. Supported perturbation algorithms are `:first_order`, `:second_order`, `:pruned_second_order`, `:third_order`, and `:pruned_third_order`.
- `warmup_iterations` [Default: `DEFAULT_WARMUP_ITERATIONS`, Type: `Int`]: Number of filter-style warmup iterations. In the filter-free case this prepends `max(warmup_iterations - 1, 0)` latent-shock periods before the first scored observation.
- $INITIAL_STATE®
- `on_failure_loglikelihood` [Default: `-Inf`, Type: `AbstractFloat`]: value to return if the loglikelihood calculation fails (e.g. solution did not converge or measurement-error std-dev is non-positive).
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `<:AbstractFloat` loglikelihood
"""
function get_loglikelihood(𝓂::ℳ,
                            data::KeyedArray{D},
                            parameter_values::Vector{S},
                            shocks::AbstractMatrix{T},
                            measurement_error_std::Union{T, AbstractVector{T}, AbstractMatrix{T}};
                            steady_state_function::SteadyStateFunctionType = missing,
                            algorithm::Symbol = :second_order,
                            warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS,
                            on_failure_loglikelihood::U = -Inf,
                            tol::Tolerances = Tolerances(),
                            quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                            lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                            sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                            verbose::Bool = DEFAULT_VERBOSE,
                            caching::Bool = DEFAULT_CACHING,
                            use_workspaces::Bool = DEFAULT_USE_WORKSPACES)::promote_type(S, T, Float64) where {D <: Union{Float64,Missing,Nothing}, S <: Real, T <: Real, U <: AbstractFloat}
    # Convenience method: no `initial_state` argument; uses the internal default.
    # To override the initial state (and get AD tangents through it), call the
    # positional method `get_loglikelihood(𝓂, data, p, shocks, me_std, initial_state; ...)`.
    return get_loglikelihood(𝓂, data, parameter_values, shocks, measurement_error_std, DEFAULT_INITIAL_STATE;
                            steady_state_function = steady_state_function,
                            algorithm = algorithm,
                            warmup_iterations = warmup_iterations,
                            on_failure_loglikelihood = on_failure_loglikelihood,
                            tol = tol,
                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                            lyapunov_algorithm = lyapunov_algorithm,
                            sylvester_algorithm = sylvester_algorithm,
                            verbose = verbose,
                            caching = caching,
                            use_workspaces = use_workspaces)
end



function get_loglikelihood(𝓂::ℳ,
                            data::KeyedArray{D},
                            parameter_values::Vector{S},
                            shocks::AbstractMatrix{T},
                            measurement_error_std::Union{T, AbstractVector{T}, AbstractMatrix{T}},
                            initial_state::Union{AbstractVector{IT}, AbstractVector{<:AbstractVector{IT}}};
                            steady_state_function::SteadyStateFunctionType = missing,
                            algorithm::Symbol = :second_order,
                            warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS,
                            on_failure_loglikelihood::U = -Inf,
                            tol::Tolerances = Tolerances(),
                            quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                            lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                            sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                            verbose::Bool = DEFAULT_VERBOSE,
                            caching::Bool = DEFAULT_CACHING,
                            use_workspaces::Bool = DEFAULT_USE_WORKSPACES)::promote_type(S, T, Float64, IT) where {D <: Union{Float64,Missing,Nothing}, S <: Real, T <: Real, U <: AbstractFloat, IT <: Real}

    @assert algorithm ∈ [:first_order, :second_order, :pruned_second_order, :third_order, :pruned_third_order] "`get_loglikelihood` only supports perturbation algorithms (`:first_order`, `:second_order`, `:pruned_second_order`, `:third_order`, `:pruned_third_order`)."

    @assert length(parameter_values) == length(𝓂.constants.post_complete_parameters.parameters) "The number of parameter values provided does not match the number of parameters in the model."

    if initial_state != DEFAULT_INITIAL_STATE
        nVars_check = 𝓂.constants.post_model_macro.nVars
        if eltype(initial_state) <: Real
            @assert length(initial_state) == nVars_check "initial_state must have length equal to the total number of variables ($nVars_check, see `show(model)`), got $(length(initial_state))."
        else
            @assert length(initial_state[1]) == nVars_check "Each vector in initial_state must have length equal to the total number of variables ($nVars_check, see `show(model)`), got $(length(initial_state[1]))."
            @assert all(v -> length(v) == length(initial_state[1]), initial_state) "All vectors in initial_state must have the same length, got lengths $(length.(initial_state))."
        end    
    end
    
    R = promote_type(S, T, Float64, IT)

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                            sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                            sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                            lyapunov_algorithm = lyapunov_algorithm)

    observables = get_and_check_observables(𝓂.constants.post_model_macro, data)

    solve!(𝓂,
           opts = opts,
           steady_state_function = steady_state_function,
           algorithm = algorithm)

    bounds_violated = check_bounds(parameter_values, 𝓂)

    if bounds_violated
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return convert(R, on_failure_loglikelihood)
    end

    me_std_is_vec = measurement_error_std isa AbstractVector
    me_std_is_mat = measurement_error_std isa AbstractMatrix
    n_obs = length(observables)
    nT_input = size(data, 2)
    @assert warmup_iterations >= 0 "`warmup_iterations` must be non-negative."
    n_warm = max(warmup_iterations - 1, 0)
    nT_total = nT_input + n_warm
    if me_std_is_vec
        @assert length(measurement_error_std) == n_obs || length(measurement_error_std) == 1 "`measurement_error_std` vector must have one entry per observable (got $(length(measurement_error_std)), expected $n_obs) or a single entry that is broadcast to all observables."
        if any(x -> !isfinite(x) || x <= zero(T), measurement_error_std)
            if !use_workspaces; 𝓂.workspaces = orig_ws; end
            return convert(R, on_failure_loglikelihood)
        end
        if length(measurement_error_std) == 1 && n_obs > 1
            measurement_error_std = fill(measurement_error_std[1], n_obs)
        end
    elseif me_std_is_mat
        @assert (size(measurement_error_std) == (n_obs, nT_input)) || (size(measurement_error_std) == (1, nT_input)) || (size(measurement_error_std) == (n_obs, 1)) || (size(measurement_error_std) == (1, 1)) "`measurement_error_std` matrix must have dimensions (n_observables, n_periods) = ($n_obs, $nT_input); got $(size(measurement_error_std)). A singleton dimension is broadcast."
        if any(x -> !isfinite(x) || x <= zero(T), measurement_error_std)
            if !use_workspaces; 𝓂.workspaces = orig_ws; end
            return convert(R, on_failure_loglikelihood)
        end
        if size(measurement_error_std) != (n_obs, nT_input)
            measurement_error_std = repeat(measurement_error_std,
                                           size(measurement_error_std, 1) == 1 ? n_obs : 1,
                                           size(measurement_error_std, 2) == 1 ? nT_input : 1)
        end
    else
        if !isfinite(measurement_error_std) || measurement_error_std <= zero(T)
            if !use_workspaces; 𝓂.workspaces = orig_ws; end
            return convert(R, on_failure_loglikelihood)
        end
    end

    SS_and_pars_names = 𝓂.constants.post_complete_parameters.SS_and_pars_names
    obs_indices = convert(Vector{Int}, indexin(observables, SS_and_pars_names))

    constants_obj, SS_and_pars, 𝐒, state, solved = get_relevant_steady_state_and_state_update(Val(algorithm), parameter_values, 𝓂, opts = opts, estimation = true)

    if !solved
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return convert(R, on_failure_loglikelihood)
    end

    # Overwrite the solver-produced `state` with any user-supplied `initial_state`
    # before any kept_rows reduction; the existing reduce_filter_free_surface then
    # handles slicing transparently.
    nVars = 𝓂.constants.post_model_macro.nVars
    if initial_state isa AbstractVector{<:Real}
        if length(initial_state) == nVars
            state_shift = state isa AbstractVector{<:AbstractVector{<:Real}} ? (length(state) == 1 ? zero(state[1]) : -state[2]) : zero(state)
            state = adjust_initial_state(initial_state, algorithm, nVars, state_shift, SS_and_pars[1:nVars])
            if algorithm == :first_order
                state = [state]
            end
        end
    elseif !isempty(initial_state)
        if state isa AbstractVector{<:AbstractVector{<:Real}}
            R_state = promote_type(eltype(eltype(state)), eltype(initial_state[1]))
            state = [convert(Vector{R_state}, i <= length(initial_state) ? initial_state[i] : state[i]) for i in eachindex(state)]
        else
            R_state = promote_type(eltype(state), eltype(initial_state[1]))
            state = convert(Vector{R_state}, initial_state[1])
        end
    end

    if collect(axiskeys(data,1)) isa Vector{String}
        data = rekey(data, 1 => axiskeys(data,1) .|> Meta.parse .|> replace_indices)
    end

    dt = missing_data_to_nan(collect(data(observables)))
    data_in_deviations, obs_idx_per_t, _, period_range = trim_informative_sample(dt .- SS_and_pars[obs_indices])

    if size(data_in_deviations, 2) == 0
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return zero(R)
    end

    nExo = 𝓂.constants.post_model_macro.nExo
    past_idx = 𝓂.constants.post_model_macro.past_not_future_and_mixed_idx
    nT = size(data_in_deviations, 2)

    @assert size(shocks, 1) == nExo "`shocks` must have one row per exogenous shock (got $(size(shocks, 1)), expected $nExo)."
    @assert size(shocks, 2) == nT_total "`shocks` must have $(nT_total) columns: $nT scored periods plus $n_warm filter-free warmup shock columns (got $(size(shocks, 2)))."

    visible_cols = isempty(period_range) ? Int[] : n_warm .+ collect(period_range)
    aligned_shocks = shocks[:, vcat(1:n_warm, visible_cols)]
    aligned_me_std = me_std_is_mat ? measurement_error_std[:, period_range] : measurement_error_std

    # Keep only the rows of the policy functions that we actually need:
    # the past-state slots required to propagate the recursion and the
    # observable rows required to form the residual. Everything else is
    # discarded so that the matmul, the kron, and the higher-order terms
    # operate on the minimum-necessary fraction of the policy functions.
    kept_rows, past_in_kept, obs_in_kept = filter_free_reduction_indices(past_idx, obs_indices)
    𝐒̂, statê = reduce_filter_free_surface(Val(algorithm), 𝐒, state, kept_rows)

    llh_raw = filter_free_loglikelihood_loop(Val(algorithm), 𝐒̂, statê, aligned_shocks, data_in_deviations, obs_in_kept, past_in_kept, obs_idx_per_t, aligned_me_std, n_warm)

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    llh = convert(R, llh_raw)

    if !isfinite(llh)
        return convert(R, on_failure_loglikelihood)
    end

    return llh
end


function filter_free_obs_logpdf(residual::AbstractVector{R}, me_std::Real) where R <: Real
    n = length(residual)
    σ² = abs2(me_std)
    return -R(0.5) * n * log(R(2π)) - n * log(me_std) - sum(abs2, residual) / (R(2) * σ²)
end

function filter_free_obs_logpdf(residual::AbstractVector{R}, me_std::AbstractVector{<:Real}) where R <: Real
    n = length(residual)
    ll = -R(0.5) * n * log(R(2π))
    @inbounds for i in 1:n
        σᵢ = me_std[i]
        ll += -log(σᵢ) - abs2(residual[i]) / (R(2) * abs2(σᵢ))
    end
    return ll
end

# Keep the shape dispatch for `measurement_error_std` in one place so the
# primal loops and the analytical pullback select the same observed subset.
period_me_std(me_std::Real, ::AbstractVector{Int}, ::Int) = me_std
period_me_std(me_std::AbstractVector, idx::AbstractVector{Int}, ::Int) = view(me_std, idx)
period_me_std(me_std::AbstractMatrix, idx::AbstractVector{Int}, t::Int) = view(me_std, idx, t)


function filter_free_reduction_indices(past_idx::Vector{Int}, obs_indices::Vector{Int})
    kept_rows = sort(unique(vcat(past_idx, obs_indices)))
    past_in_kept_raw = indexin(past_idx, kept_rows)
    obs_in_kept_raw = indexin(obs_indices, kept_rows)

    @assert all(!isnothing, past_in_kept_raw) "Failed to map all past-state indices into the reduced filter-free surface."
    @assert all(!isnothing, obs_in_kept_raw) "Failed to map all observable indices into the reduced filter-free surface."

    past_in_kept = Int[idx::Int for idx in past_in_kept_raw]
    obs_in_kept = Int[idx::Int for idx in obs_in_kept_raw]
    return kept_rows, past_in_kept, obs_in_kept
end


function reduce_filter_free_block(block::AbstractMatrix{S}, kept_rows::Vector{Int}) where S <: Real
    block̂ = Matrix{S}(undef, length(kept_rows), size(block, 2))

    @inbounds for j in axes(block̂, 2), i in eachindex(kept_rows)
        block̂[i, j] = block[kept_rows[i], j]
    end

    return block̂
end


function reduce_filter_free_surface(::Val{:first_order},
                                    𝐒::AbstractMatrix,
                                    state::AbstractVector{<:AbstractVector{<:Real}},
                                    kept_rows::Vector{Int})
    return 𝐒[kept_rows, :], [state[1][kept_rows]]
end


function reduce_filter_free_surface(::Val{:second_order},
                                    𝐒::Union{AbstractVector{M},Tuple{M,M}},
                                    state::AbstractVector{<:Real},
                                    kept_rows::Vector{Int}) where {S <: Real, M <: AbstractMatrix{S}}
    @assert length(𝐒) == 2 "Expected two policy-function blocks for second-order filter-free likelihood."
    𝐒̂₁ = reduce_filter_free_block(𝐒[1], kept_rows)
    𝐒̂₂ = reduce_filter_free_block(𝐒[2], kept_rows)
    return (𝐒̂₁, 𝐒̂₂), state[kept_rows]
end


function reduce_filter_free_surface(::Val{:third_order},
                                    𝐒::Union{AbstractVector{M},Tuple{M,M,M}},
                                    state::AbstractVector{<:Real},
                                    kept_rows::Vector{Int}) where {S <: Real, M <: AbstractMatrix{S}}
    @assert length(𝐒) == 3 "Expected three policy-function blocks for third-order filter-free likelihood."
    𝐒̂₁ = reduce_filter_free_block(𝐒[1], kept_rows)
    𝐒̂₂ = reduce_filter_free_block(𝐒[2], kept_rows)
    𝐒̂₃ = reduce_filter_free_block(𝐒[3], kept_rows)
    return (𝐒̂₁, 𝐒̂₂, 𝐒̂₃), state[kept_rows]
end


function reduce_filter_free_surface(::Val{:pruned_second_order},
                                    𝐒::Union{AbstractVector{M},Tuple{M,M}},
                                    state::AbstractVector{<:AbstractVector{<:Real}},
                                    kept_rows::Vector{Int}) where {S <: Real, M <: AbstractMatrix{S}}
    @assert length(𝐒) == 2 "Expected two policy-function blocks for pruned second-order filter-free likelihood."
    𝐒̂₁ = reduce_filter_free_block(𝐒[1], kept_rows)
    𝐒̂₂ = reduce_filter_free_block(𝐒[2], kept_rows)
    return (𝐒̂₁, 𝐒̂₂), [s[kept_rows] for s in state]
end


function reduce_filter_free_surface(::Val{:pruned_third_order},
                                    𝐒::Union{AbstractVector{M},Tuple{M,M,M}},
                                    state::AbstractVector{<:AbstractVector{<:Real}},
                                    kept_rows::Vector{Int}) where {S <: Real, M <: AbstractMatrix{S}}
    @assert length(𝐒) == 3 "Expected three policy-function blocks for pruned third-order filter-free likelihood."
    𝐒̂₁ = reduce_filter_free_block(𝐒[1], kept_rows)
    𝐒̂₂ = reduce_filter_free_block(𝐒[2], kept_rows)
    𝐒̂₃ = reduce_filter_free_block(𝐒[3], kept_rows)
    return (𝐒̂₁, 𝐒̂₂, 𝐒̂₃), [s[kept_rows] for s in state]
end


function filter_free_loglikelihood_loop(::Val{:first_order},
                                        𝐒::AbstractMatrix,
                                        state::AbstractVector{<:AbstractVector{<:Real}},
                                        shocks::AbstractMatrix{T},
                                        data_in_deviations::AbstractMatrix{<:Real},
                                        obs_indices::Vector{Int},
                                        past_idx::Vector{Int},
                                        obs_idx_per_t::Vector{Vector{Int}},
                                        me_std,
                                        n_warm::Int) where {T <: Real}
    𝐒₁ = 𝐒
    R = promote_type(eltype(𝐒₁), eltype(shocks), eltype(data_in_deviations), eltype(eltype(state)))
    nT = size(data_in_deviations, 2)

    cur_state = convert(Vector{R}, state[1])
    llh = zero(R)

    for t in 1:n_warm
        ϵ = view(shocks, :, t)
        aug = vcat(cur_state[past_idx], ϵ)
        cur_state = 𝐒₁ * aug
    end

    for t in 1:nT
        ϵ = view(shocks, :, n_warm + t)
        aug = vcat(cur_state[past_idx], ϵ)
        new_state = 𝐒₁ * aug
        idx = obs_idx_per_t[t]
        if !isempty(idx)
            obs_dev = new_state[obs_indices[idx]]
            residual = data_in_deviations[idx, t] - obs_dev
            llh += filter_free_obs_logpdf(residual, period_me_std(me_std, idx, t))
        end
        cur_state = new_state
    end

    return llh
end


function filter_free_loglikelihood_loop(::Val{:second_order},
                                        𝐒::Tuple{<:AbstractMatrix,<:AbstractMatrix},
                                        state::AbstractVector{<:Real},
                                        shocks::AbstractMatrix{T},
                                        data_in_deviations::AbstractMatrix{<:Real},
                                        obs_indices::Vector{Int},
                                        past_idx::Vector{Int},
                                        obs_idx_per_t::Vector{Vector{Int}},
                                        me_std,
                                        n_warm::Int) where {T <: Real}
    𝐒₁ = 𝐒[1]
    𝐒₂ = 𝐒[2]
    R = promote_type(eltype(𝐒₁), eltype(shocks), eltype(data_in_deviations), eltype(eltype(state)))
    nT = size(data_in_deviations, 2)

    cur_state = convert(Vector{R}, state)
    llh = zero(R)

    for t in 1:n_warm
        ϵ = view(shocks, :, t)
        aug = vcat(cur_state[past_idx], one(R), ϵ)
        cur_state = 𝐒₁ * aug + 𝐒₂ * ℒ.kron(aug, aug) / R(2)
    end

    for t in 1:nT
        ϵ = view(shocks, :, n_warm + t)
        aug = vcat(cur_state[past_idx], one(R), ϵ)
        new_state = 𝐒₁ * aug + 𝐒₂ * ℒ.kron(aug, aug) / R(2)
        idx = obs_idx_per_t[t]
        if !isempty(idx)
            obs_dev = new_state[obs_indices[idx]]
            residual = data_in_deviations[idx, t] - obs_dev
            llh += filter_free_obs_logpdf(residual, period_me_std(me_std, idx, t))
        end
        cur_state = new_state
    end

    return llh
end


function filter_free_loglikelihood_loop(::Val{:third_order},
                                        𝐒::Tuple{<:AbstractMatrix,<:AbstractMatrix,<:AbstractMatrix},
                                        state::AbstractVector{<:Real},
                                        shocks::AbstractMatrix{T},
                                        data_in_deviations::AbstractMatrix{<:Real},
                                        obs_indices::Vector{Int},
                                        past_idx::Vector{Int},
                                        obs_idx_per_t::Vector{Vector{Int}},
                                        me_std,
                                        n_warm::Int) where {T <: Real}
    𝐒₁ = 𝐒[1]
    𝐒₂ = 𝐒[2]
    𝐒₃ = 𝐒[3]
    R = promote_type(eltype(𝐒₁), eltype(shocks), eltype(data_in_deviations), eltype(eltype(state)))
    nT = size(data_in_deviations, 2)

    cur_state = convert(Vector{R}, state)
    llh = zero(R)

    for t in 1:n_warm
        ϵ = view(shocks, :, t)
        aug = vcat(cur_state[past_idx], one(R), ϵ)
        kaug = ℒ.kron(aug, aug)
        cur_state = 𝐒₁ * aug + 𝐒₂ * kaug / R(2) + 𝐒₃ * ℒ.kron(kaug, aug) / R(6)
    end

    for t in 1:nT
        ϵ = view(shocks, :, n_warm + t)
        aug = vcat(cur_state[past_idx], one(R), ϵ)
        kaug = ℒ.kron(aug, aug)
        new_state = 𝐒₁ * aug + 𝐒₂ * kaug / R(2) + 𝐒₃ * ℒ.kron(kaug, aug) / R(6)
        idx = obs_idx_per_t[t]
        if !isempty(idx)
            obs_dev = new_state[obs_indices[idx]]
            residual = data_in_deviations[idx, t] - obs_dev
            llh += filter_free_obs_logpdf(residual, period_me_std(me_std, idx, t))
        end
        cur_state = new_state
    end

    return llh
end


function filter_free_loglikelihood_loop(::Val{:pruned_second_order},
                                        𝐒::Tuple{<:AbstractMatrix,<:AbstractMatrix},
                                        state::AbstractVector{<:AbstractVector{<:Real}},
                                        shocks::AbstractMatrix{T},
                                        data_in_deviations::AbstractMatrix{<:Real},
                                        obs_indices::Vector{Int},
                                        past_idx::Vector{Int},
                                        obs_idx_per_t::Vector{Vector{Int}},
                                        me_std,
                                        n_warm::Int) where {T <: Real}
    𝐒₁ = 𝐒[1]
    𝐒₂ = 𝐒[2]
    R = promote_type(eltype(𝐒₁), eltype(shocks), eltype(data_in_deviations), eltype(eltype(state)))
    nVars = length(state[1])
    nT = size(data_in_deviations, 2)

    cur_state = [convert(Vector{R}, state[1]), convert(Vector{R}, state[2])]
    llh = zero(R)

    for t in 1:n_warm
        ϵ = collect(view(shocks, :, t))
        cur_state = pruned_second_order_state_update(cur_state, ϵ, past_idx, nVars, 𝐒₁, 𝐒₂)
    end

    for t in 1:nT
        ϵ = collect(view(shocks, :, n_warm + t))
        new_state = pruned_second_order_state_update(cur_state, ϵ, past_idx, nVars, 𝐒₁, 𝐒₂)
        idx = obs_idx_per_t[t]
        if !isempty(idx)
            obs_dev = new_state[1][obs_indices[idx]] + new_state[2][obs_indices[idx]]
            residual = data_in_deviations[idx, t] - obs_dev
            llh += filter_free_obs_logpdf(residual, period_me_std(me_std, idx, t))
        end
        cur_state = new_state
    end

    return llh
end


function filter_free_loglikelihood_loop(::Val{:pruned_third_order},
                                        𝐒::Tuple{<:AbstractMatrix,<:AbstractMatrix,<:AbstractMatrix},
                                        state::AbstractVector{<:AbstractVector{<:Real}},
                                        shocks::AbstractMatrix{T},
                                        data_in_deviations::AbstractMatrix{<:Real},
                                        obs_indices::Vector{Int},
                                        past_idx::Vector{Int},
                                        obs_idx_per_t::Vector{Vector{Int}},
                                        me_std,
                                        n_warm::Int) where {T <: Real}
    𝐒₁ = 𝐒[1]
    𝐒₂ = 𝐒[2]
    𝐒₃ = 𝐒[3]
    R = promote_type(eltype(𝐒₁), eltype(shocks), eltype(data_in_deviations), eltype(eltype(state)))
    nVars = length(state[1])
    nT = size(data_in_deviations, 2)

    cur_state = [convert(Vector{R}, state[1]), convert(Vector{R}, state[2]), convert(Vector{R}, state[3])]
    llh = zero(R)

    for t in 1:n_warm
        ϵ = collect(view(shocks, :, t))
        cur_state = pruned_third_order_state_update(cur_state, ϵ, past_idx, nVars, 𝐒₁, 𝐒₂, 𝐒₃)
    end

    for t in 1:nT
        ϵ = collect(view(shocks, :, n_warm + t))
        new_state = pruned_third_order_state_update(cur_state, ϵ, past_idx, nVars, 𝐒₁, 𝐒₂, 𝐒₃)
        idx = obs_idx_per_t[t]
        if !isempty(idx)
            obs_dev = new_state[1][obs_indices[idx]] + new_state[2][obs_indices[idx]] + new_state[3][obs_indices[idx]]
            residual = data_in_deviations[idx, t] - obs_dev
            llh += filter_free_obs_logpdf(residual, period_me_std(me_std, idx, t))
        end
        cur_state = new_state
    end

    return llh
end


function check_bounds(parameter_values::Vector{S}, 𝓂::ℳ)::Bool where S <: Real
    if !all(isfinite,parameter_values) return true end

    if length(𝓂.constants.post_parameters_macro.bounds) > 0 
        for (k,v) in 𝓂.constants.post_parameters_macro.bounds
            if k ∈ 𝓂.constants.post_complete_parameters.parameters
                if min(max(parameter_values[indexin([k], 𝓂.constants.post_complete_parameters.parameters)][1], v[1]), v[2]) != parameter_values[indexin([k], 𝓂.constants.post_complete_parameters.parameters)][1]
                    return true
                end
            end
        end
    end

    return false
end


function get_relevant_steady_state_and_state_update(::Val{:second_order}, 
                                                    parameter_values::Vector{S}, 
                                                    𝓂::ℳ; 
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    estimation::Bool = false)::Tuple{constants, Vector{S}, Vector{AbstractMatrix{S}}, Vector{S}, Bool} where S <: Real
                                                    # timer::TimerOutput = TimerOutput(), 
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_stochastic_steady_state(Val(:second_order), parameter_values, 𝓂, opts = opts, estimation = estimation) # timer = timer, 
    
    if !converged || solution_error > opts.tol.nsss.acceptance_tol
        if opts.verbose println("Could not find 2nd order stochastic steady state") end
        return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂], collect(sss), converged
    end

    ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    ms = 𝓂.constants.post_complete_parameters
    all_SS = expand_steady_state(SS_and_pars, ms)

    state = collect(sss) - all_SS

    return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂], state, converged
end



function get_relevant_steady_state_and_state_update(::Val{:pruned_second_order}, 
                                                    parameter_values::Vector{S}, 
                                                    𝓂::ℳ; 
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    estimation::Bool = false)::Tuple{constants, Vector{S}, Vector{AbstractMatrix{S}}, Vector{Vector{S}}, Bool} where S <: Real
                                                    # timer::TimerOutput = TimerOutput(), 
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_stochastic_steady_state(Val(:pruned_second_order), parameter_values, 𝓂, opts = opts, estimation = estimation) # timer = timer, 

    if !converged || solution_error > opts.tol.nsss.acceptance_tol
        if opts.verbose println("Could not find 2nd order stochastic steady state") end
        return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂], [zeros(S, 𝓂.constants.post_model_macro.nVars), zeros(S, 𝓂.constants.post_model_macro.nVars)], converged
    end

    ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    ms = 𝓂.constants.post_complete_parameters
    all_SS = expand_steady_state(SS_and_pars, ms)

    state = [zeros(S, 𝓂.constants.post_model_macro.nVars), collect(sss)::Vector{S} - all_SS]

    return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂], state, converged
end



function get_relevant_steady_state_and_state_update(::Val{:third_order}, 
                                                    parameter_values::Vector{S}, 
                                                    𝓂::ℳ; 
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    estimation::Bool = false)::Tuple{constants, Vector{S}, Vector{AbstractMatrix{S}}, Vector{S}, Bool} where S <: Real
                                                    # timer::TimerOutput = TimerOutput(), 
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_stochastic_steady_state(Val(:third_order), parameter_values, 𝓂, opts = opts, estimation = estimation) # timer = timer,  

    if !converged || solution_error > opts.tol.nsss.acceptance_tol
        if opts.verbose println("Could not find 3rd order stochastic steady state") end
        return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], collect(sss), converged
    end

    ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    ms = 𝓂.constants.post_complete_parameters
    all_SS = expand_steady_state(SS_and_pars, ms)

    state = collect(sss) - all_SS

    return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], state, converged
end



function get_relevant_steady_state_and_state_update(::Val{:pruned_third_order}, 
                                                    parameter_values::Vector{S}, 
                                                    𝓂::ℳ; 
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    estimation::Bool = false)::Tuple{constants, Vector{S}, Vector{AbstractMatrix{S}}, Vector{Vector{S}}, Bool} where S <: Real
                                                    # timer::TimerOutput = TimerOutput(), 
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_stochastic_steady_state(Val(:pruned_third_order), parameter_values, 𝓂, opts = opts, estimation = estimation) # timer = timer, 

    if !converged || solution_error > opts.tol.nsss.acceptance_tol
        if opts.verbose println("Could not find 3rd order stochastic steady state") end
        return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], [zeros(S, 𝓂.constants.post_model_macro.nVars), zeros(S, 𝓂.constants.post_model_macro.nVars), zeros(S, 𝓂.constants.post_model_macro.nVars)], converged
    end

    ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    ms = 𝓂.constants.post_complete_parameters
    all_SS = expand_steady_state(SS_and_pars, ms)

    state = [zeros(S, 𝓂.constants.post_model_macro.nVars), collect(sss)::Vector{S} - all_SS, zeros(S, 𝓂.constants.post_model_macro.nVars)]

    return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], state, converged
end


function get_relevant_steady_state_and_state_update(::Val{:first_order}, 
                                                    parameter_values::Vector{S}, 
                                                    𝓂::ℳ; 
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    estimation::Bool = false)::Tuple{constants, Vector{S}, Matrix{S}, Vector{Vector{Float64}}, Bool} where S <: Real
                                                    # timer::TimerOutput = TimerOutput(), 
    # Initialize constants at entry point
    constants_obj = initialise_constants!(𝓂)

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameter_values, opts = opts, estimation = estimation) # timer = timer,

    state = zeros(𝓂.constants.post_model_macro.nVars)

    if solution_error > opts.tol.nsss.acceptance_tol # || isnan(solution_error) if it's NaN the first condition is false anyway
        # println("NSSS not found")
        return 𝓂.constants, SS_and_pars, zeros(S, 0, 0), [state], solution_error < opts.tol.nsss.acceptance_tol
    end

    ∇₁ = calculate_jacobian(parameter_values, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces) # , timer = timer)# |> Matrix

    𝐒₁, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                        constants_obj,
                                                        𝓂.workspaces,
                                                        𝓂.caches;
                                                        opts = opts,
                                                        initial_guess = 𝓂.caches.qme_solution,
                                                        parameter_values = parameter_values)


    update_perturbation_counter!(𝓂.counters, solved, estimation = estimation, order = 1)

    if !solved
        # println("NSSS not found")
        return 𝓂.constants, SS_and_pars, zeros(S, 0, 0), [state], solved
    end

    return 𝓂.constants, SS_and_pars, 𝐒₁, [state], solved
end


"""
$(SIGNATURES)
Calculate the residuals of the non-stochastic steady state equations of the model for a given set of values. Values not provided, will be filled with the non-stochastic steady state values corresponding to the current parameters.

# Arguments
- $MODEL®
- `values` [Type: `Union{Vector{Float64}, Dict{Symbol, Float64}, Dict{String, Float64}, KeyedArray{Float64, 1}}`]: A Vector, Dict, or KeyedArray containing the values of the variables and calibrated parameters in the non-stochastic steady state equations (including calibration equations). The `KeyedArray` type is provided by the `AxisKeys` package.

# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) containing the absolute values of the residuals of the non-stochastic steady state equations.

# Examples
```jldoctest; filter = r"(Equation|CalibrationEquation)([^0-9+-]+)\\S+" => s"\\1\\2 0.0"
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC  begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    k[ss] / q[ss] = 2.5 | α
    β = 0.95
end

steady_state = SS(RBC, derivatives = false)

get_non_stochastic_steady_state_residuals(RBC, steady_state)
# output
1-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Equation ∈ 5-element Vector{Symbol}
And data, 5-element Vector{Float64}:
 (:Equation₁)             0.0
 (:Equation₂)             0.0
 (:Equation₃)             0.0
 (:Equation₄)             0.0
 (:CalibrationEquation₁)  0.0
```

Passing approximate values returns the residuals at those values:
```julia
get_non_stochastic_steady_state_residuals(RBC, [1.1641597, 3.0635781, 1.2254312, 0.0, 0.18157895])
# 1-dimensional KeyedArray(NamedDimsArray(...)) with keys:
# ↓   Equation ∈ 5-element Vector{Symbol}
# And data, 5-element Vector{Float64}:
#  (:Equation₁)             2.7360991250446887e-10
#  (:Equation₂)             6.199999980083248e-8
#  (:Equation₃)             2.7897102183871425e-8
#  (:Equation₄)             0.0
#  (:CalibrationEquation₁)  8.160392850342646e-8
```
"""
@unstable function get_non_stochastic_steady_state_residuals(𝓂::ℳ, 
                                                    values::Union{Vector{Float64}, Dict{Symbol, Float64}, Dict{String, Float64}, KeyedArray{Float64, 1}}; 
                                                    parameters::ParameterType = nothing,
                                                    steady_state_function::SteadyStateFunctionType = missing,
                                                    tol::Tolerances = Tolerances(),
                                                    verbose::Bool = DEFAULT_VERBOSE,
                                                    caching::Bool = DEFAULT_CACHING,
                                                    use_workspaces::Bool = DEFAULT_USE_WORKSPACES)
    # @nospecialize # reduce compile time                                             

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose)
    
    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            opts = opts)

    SS_and_pars, _ = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)

    axis1 = vcat(𝓂.constants.post_model_macro.var, 𝓂.equations.calibration_parameters)

    vars_in_ss_equations = sort(collect(setdiff(reduce(union, get_symbols.(𝓂.equations.steady_state)), union(𝓂.constants.post_model_macro.parameters_in_equations))))

    unknowns = vcat(vars_in_ss_equations, 𝓂.equations.calibration_parameters)

    combined_values = Dict(unknowns .=> SS_and_pars[indexin(unknowns, axis1)])

    if isa(values, Vector)
        @assert length(values) == length(unknowns) "Invalid input. Expected a vector of length $(length(unknowns))."
        for (i, value) in enumerate(values)
            combined_values[unknowns[i]] = value
        end
    elseif isa(values, Dict)
        for (key, value) in values
            if key isa String
                key = replace_indices(key)
            end
            combined_values[key] = value
        end
    elseif isa(values, KeyedArray)
        for (key, value) in zip(axiskeys(values, 1), collect(values))
            if key isa String
                key = replace_indices(key)
            end
            combined_values[key] = value
        end
    end

    vals = [combined_values[i] for i in unknowns]

    axis1 = vcat([Symbol("Equation" * sub(string(i))) for i in 1:length(vars_in_ss_equations)], [Symbol("CalibrationEquation" * sub(string(i))) for i in 1:length(𝓂.equations.calibration_parameters)])
    
    residual = zeros(length(vals))

    𝓂.functions.NSSS_check(residual, 𝓂.parameter_values, vals)

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    KeyedArray(abs.(residual), Equation = axis1)
end

"""
See [`get_non_stochastic_steady_state_residuals`](@ref)
"""
@unstable get_residuals = get_non_stochastic_steady_state_residuals

"""
See [`get_non_stochastic_steady_state_residuals`](@ref)
"""
check_residuals = get_non_stochastic_steady_state_residuals

end # @stable
