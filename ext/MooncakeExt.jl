module MooncakeExt

using Mooncake
using Mooncake: @is_primitive, MinimalCtx, CoDual, primal, zero_fcodual, NoPullback
import MacroModelling
import MacroModelling: _get_loglikelihood_internal, ℳ, CalculationOptions
import ChainRulesCore: rrule, NoTangent

# Mark _get_loglikelihood_internal as a primitive for Mooncake for all algorithm/filter combinations
# This allows Mooncake to use the ChainRulesCore rrule defined in the main package

# First order + kalman
@is_primitive MinimalCtx Tuple{
    typeof(_get_loglikelihood_internal),
    Vector{Float64}, Matrix{Float64}, Vector{Int},
    Val{:first_order}, Val{:kalman},
    Vector{Symbol}, ℳ, Int, Symbol, Int, Symbol, CalculationOptions, Float64
}

# First order + inversion
@is_primitive MinimalCtx Tuple{
    typeof(_get_loglikelihood_internal),
    Vector{Float64}, Matrix{Float64}, Vector{Int},
    Val{:first_order}, Val{:inversion},
    Vector{Symbol}, ℳ, Int, Symbol, Int, Symbol, CalculationOptions, Float64
}

# Second order + inversion
@is_primitive MinimalCtx Tuple{
    typeof(_get_loglikelihood_internal),
    Vector{Float64}, Matrix{Float64}, Vector{Int},
    Val{:second_order}, Val{:inversion},
    Vector{Symbol}, ℳ, Int, Symbol, Int, Symbol, CalculationOptions, Float64
}

# Pruned second order + inversion
@is_primitive MinimalCtx Tuple{
    typeof(_get_loglikelihood_internal),
    Vector{Float64}, Matrix{Float64}, Vector{Int},
    Val{:pruned_second_order}, Val{:inversion},
    Vector{Symbol}, ℳ, Int, Symbol, Int, Symbol, CalculationOptions, Float64
}

# Third order + inversion
@is_primitive MinimalCtx Tuple{
    typeof(_get_loglikelihood_internal),
    Vector{Float64}, Matrix{Float64}, Vector{Int},
    Val{:third_order}, Val{:inversion},
    Vector{Symbol}, ℳ, Int, Symbol, Int, Symbol, CalculationOptions, Float64
}

# Pruned third order + inversion
@is_primitive MinimalCtx Tuple{
    typeof(_get_loglikelihood_internal),
    Vector{Float64}, Matrix{Float64}, Vector{Int},
    Val{:pruned_third_order}, Val{:inversion},
    Vector{Symbol}, ℳ, Int, Symbol, Int, Symbol, CalculationOptions, Float64
}

# Helper function to create Mooncake rrule!! from ChainRulesCore rrule
# This is a generic implementation that works for any algorithm/filter combination
function _create_mooncake_rrule(
    parameter_values::CoDual{Vector{Float64}},
    data_raw::CoDual{Matrix{Float64}},
    obs_indices::CoDual{Vector{Int}},
    algorithm,
    filter,
    observables::CoDual{Vector{Symbol}},
    𝓂::CoDual{ℳ},
    presample_periods::CoDual{Int},
    initial_covariance::CoDual{Symbol},
    warmup_iterations::CoDual{Int},
    filter_algorithm::CoDual{Symbol},
    opts::CoDual{CalculationOptions},
    on_failure_loglikelihood::CoDual{Float64}
)
    # Extract primal values
    pv = primal(parameter_values)
    dr = primal(data_raw)
    oi = primal(obs_indices)
    alg = primal(algorithm)
    flt = primal(filter)
    obs = primal(observables)
    m = primal(𝓂)
    pp = primal(presample_periods)
    ic = primal(initial_covariance)
    wi = primal(warmup_iterations)
    fa = primal(filter_algorithm)
    op = primal(opts)
    ofl = primal(on_failure_loglikelihood)

    # Call the ChainRulesCore rrule
    result = rrule(_get_loglikelihood_internal, pv, dr, oi, alg, flt, obs, m, pp, ic, wi, fa, op, ofl)
    
    if result === nothing
        # No rrule defined, return no pullback
        llh = _get_loglikelihood_internal(pv, dr, oi, alg, flt, obs, m, pp, ic, wi, fa, op, ofl)
        return zero_fcodual(llh), NoPullback()
    end
    
    llh, cr_pullback = result
    
    # Create Mooncake pullback
    function mooncake_pullback(Δllh)
        # Call ChainRulesCore pullback
        cr_result = cr_pullback(Δllh)
        # cr_result is a tuple of (NoTangent(), ∂params, NoTangent(), ...)
        ∂params = cr_result[2]
        
        # Return tangent for parameter_values (mutable array)
        if !(∂params isa NoTangent) && parameter_values.dx !== nothing
            parameter_values.dx .+= ∂params
        end
        
        return Mooncake.NoRData()
    end
    
    return zero_fcodual(llh), mooncake_pullback
end

# Define Mooncake's rrule!! for first_order + kalman
function Mooncake.rrule!!(
    ::CoDual{typeof(_get_loglikelihood_internal)},
    parameter_values::CoDual{Vector{Float64}},
    data_raw::CoDual{Matrix{Float64}},
    obs_indices::CoDual{Vector{Int}},
    algorithm::CoDual{Val{:first_order}},
    filter::CoDual{Val{:kalman}},
    observables::CoDual{Vector{Symbol}},
    𝓂::CoDual{ℳ},
    presample_periods::CoDual{Int},
    initial_covariance::CoDual{Symbol},
    warmup_iterations::CoDual{Int},
    filter_algorithm::CoDual{Symbol},
    opts::CoDual{CalculationOptions},
    on_failure_loglikelihood::CoDual{Float64}
)
    return _create_mooncake_rrule(
        parameter_values, data_raw, obs_indices, algorithm, filter,
        observables, 𝓂, presample_periods, initial_covariance,
        warmup_iterations, filter_algorithm, opts, on_failure_loglikelihood
    )
end

# Define Mooncake's rrule!! for first_order + inversion
function Mooncake.rrule!!(
    ::CoDual{typeof(_get_loglikelihood_internal)},
    parameter_values::CoDual{Vector{Float64}},
    data_raw::CoDual{Matrix{Float64}},
    obs_indices::CoDual{Vector{Int}},
    algorithm::CoDual{Val{:first_order}},
    filter::CoDual{Val{:inversion}},
    observables::CoDual{Vector{Symbol}},
    𝓂::CoDual{ℳ},
    presample_periods::CoDual{Int},
    initial_covariance::CoDual{Symbol},
    warmup_iterations::CoDual{Int},
    filter_algorithm::CoDual{Symbol},
    opts::CoDual{CalculationOptions},
    on_failure_loglikelihood::CoDual{Float64}
)
    return _create_mooncake_rrule(
        parameter_values, data_raw, obs_indices, algorithm, filter,
        observables, 𝓂, presample_periods, initial_covariance,
        warmup_iterations, filter_algorithm, opts, on_failure_loglikelihood
    )
end

# Define Mooncake's rrule!! for second_order + inversion
function Mooncake.rrule!!(
    ::CoDual{typeof(_get_loglikelihood_internal)},
    parameter_values::CoDual{Vector{Float64}},
    data_raw::CoDual{Matrix{Float64}},
    obs_indices::CoDual{Vector{Int}},
    algorithm::CoDual{Val{:second_order}},
    filter::CoDual{Val{:inversion}},
    observables::CoDual{Vector{Symbol}},
    𝓂::CoDual{ℳ},
    presample_periods::CoDual{Int},
    initial_covariance::CoDual{Symbol},
    warmup_iterations::CoDual{Int},
    filter_algorithm::CoDual{Symbol},
    opts::CoDual{CalculationOptions},
    on_failure_loglikelihood::CoDual{Float64}
)
    return _create_mooncake_rrule(
        parameter_values, data_raw, obs_indices, algorithm, filter,
        observables, 𝓂, presample_periods, initial_covariance,
        warmup_iterations, filter_algorithm, opts, on_failure_loglikelihood
    )
end

# Define Mooncake's rrule!! for pruned_second_order + inversion
function Mooncake.rrule!!(
    ::CoDual{typeof(_get_loglikelihood_internal)},
    parameter_values::CoDual{Vector{Float64}},
    data_raw::CoDual{Matrix{Float64}},
    obs_indices::CoDual{Vector{Int}},
    algorithm::CoDual{Val{:pruned_second_order}},
    filter::CoDual{Val{:inversion}},
    observables::CoDual{Vector{Symbol}},
    𝓂::CoDual{ℳ},
    presample_periods::CoDual{Int},
    initial_covariance::CoDual{Symbol},
    warmup_iterations::CoDual{Int},
    filter_algorithm::CoDual{Symbol},
    opts::CoDual{CalculationOptions},
    on_failure_loglikelihood::CoDual{Float64}
)
    return _create_mooncake_rrule(
        parameter_values, data_raw, obs_indices, algorithm, filter,
        observables, 𝓂, presample_periods, initial_covariance,
        warmup_iterations, filter_algorithm, opts, on_failure_loglikelihood
    )
end

# Define Mooncake's rrule!! for third_order + inversion
function Mooncake.rrule!!(
    ::CoDual{typeof(_get_loglikelihood_internal)},
    parameter_values::CoDual{Vector{Float64}},
    data_raw::CoDual{Matrix{Float64}},
    obs_indices::CoDual{Vector{Int}},
    algorithm::CoDual{Val{:third_order}},
    filter::CoDual{Val{:inversion}},
    observables::CoDual{Vector{Symbol}},
    𝓂::CoDual{ℳ},
    presample_periods::CoDual{Int},
    initial_covariance::CoDual{Symbol},
    warmup_iterations::CoDual{Int},
    filter_algorithm::CoDual{Symbol},
    opts::CoDual{CalculationOptions},
    on_failure_loglikelihood::CoDual{Float64}
)
    return _create_mooncake_rrule(
        parameter_values, data_raw, obs_indices, algorithm, filter,
        observables, 𝓂, presample_periods, initial_covariance,
        warmup_iterations, filter_algorithm, opts, on_failure_loglikelihood
    )
end

# Define Mooncake's rrule!! for pruned_third_order + inversion
function Mooncake.rrule!!(
    ::CoDual{typeof(_get_loglikelihood_internal)},
    parameter_values::CoDual{Vector{Float64}},
    data_raw::CoDual{Matrix{Float64}},
    obs_indices::CoDual{Vector{Int}},
    algorithm::CoDual{Val{:pruned_third_order}},
    filter::CoDual{Val{:inversion}},
    observables::CoDual{Vector{Symbol}},
    𝓂::CoDual{ℳ},
    presample_periods::CoDual{Int},
    initial_covariance::CoDual{Symbol},
    warmup_iterations::CoDual{Int},
    filter_algorithm::CoDual{Symbol},
    opts::CoDual{CalculationOptions},
    on_failure_loglikelihood::CoDual{Float64}
)
    return _create_mooncake_rrule(
        parameter_values, data_raw, obs_indices, algorithm, filter,
        observables, 𝓂, presample_periods, initial_covariance,
        warmup_iterations, filter_algorithm, opts, on_failure_loglikelihood
    )
end

end # module
