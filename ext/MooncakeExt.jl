module MooncakeExt

using Mooncake
using Mooncake: @is_primitive, MinimalCtx, CoDual, NoRData
import MacroModelling: get_loglikelihood, get_NSSS_and_parameters_with_jacobian, ℳ, Tolerances, CalculationOptions, merge_calculation_options
import AxisKeys: KeyedArray
import ChainRulesCore
import ChainRulesCore: NoTangent

# ============================================================================
# get_NSSS_and_parameters_with_jacobian - Mooncake integration
# ============================================================================

# Mark get_NSSS_and_parameters_with_jacobian as a primitive for Mooncake
@is_primitive MinimalCtx Tuple{typeof(Core.kwcall), Any, typeof(get_NSSS_and_parameters_with_jacobian), ℳ, Vector{Float64}}
@is_primitive MinimalCtx Tuple{typeof(get_NSSS_and_parameters_with_jacobian), ℳ, Vector{Float64}}

# Helper function to compute gradients using the ChainRulesCore rrule for get_NSSS_and_parameters_with_jacobian
function _compute_gradient_nsss_with_jac(
    𝓂::ℳ,
    params::Vector{Float64};
    kwargs...
)
    rrule_result = ChainRulesCore.rrule(
        get_NSSS_and_parameters_with_jacobian,
        𝓂,
        params;
        kwargs...
    )
    
    if rrule_result === nothing
        error("ChainRulesCore.rrule returned nothing for get_NSSS_and_parameters_with_jacobian")
    end
    
    return rrule_result
end

# rrule!! for Mooncake with keyword arguments (via Core.kwcall)
function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kwargs::CoDual,
    ::CoDual{typeof(get_NSSS_and_parameters_with_jacobian)},
    𝓂::CoDual{<:ℳ},
    parameter_values::CoDual{<:Vector{Float64}}
)
    kwargs_primal = Mooncake.primal(kwargs)
    𝓂_primal = Mooncake.primal(𝓂)
    params_primal = Mooncake.primal(parameter_values)
    
    (SS_and_pars, jvp, solution_error, iters), chainrules_pullback = _compute_gradient_nsss_with_jac(
        𝓂_primal,
        params_primal;
        kwargs_primal...
    )
    
    function mooncake_pullback(Δ)
        # Δ is (∂SS_and_pars, ∂jvp, ∂solution_error, ∂iters)
        _, _, dparams, _ = chainrules_pullback(Δ)
        
        if dparams isa NoTangent || dparams === nothing
            return NoRData(), NoRData(), NoRData(), NoRData(), zero(params_primal)
        else
            return NoRData(), NoRData(), NoRData(), NoRData(), collect(dparams)
        end
    end
    
    result = (SS_and_pars, jvp, solution_error, iters)
    return CoDual(result, Mooncake.zero_tangent(result)), mooncake_pullback
end

# rrule!! for Mooncake without keyword arguments
function Mooncake.rrule!!(
    ::CoDual{typeof(get_NSSS_and_parameters_with_jacobian)},
    𝓂::CoDual{<:ℳ},
    parameter_values::CoDual{<:Vector{Float64}}
)
    𝓂_primal = Mooncake.primal(𝓂)
    params_primal = Mooncake.primal(parameter_values)
    
    (SS_and_pars, jvp, solution_error, iters), chainrules_pullback = _compute_gradient_nsss_with_jac(
        𝓂_primal,
        params_primal
    )
    
    function mooncake_pullback(Δ)
        _, _, dparams, _ = chainrules_pullback(Δ)
        
        if dparams isa NoTangent || dparams === nothing
            return NoRData(), NoRData(), zero(params_primal)
        else
            return NoRData(), NoRData(), collect(dparams)
        end
    end
    
    result = (SS_and_pars, jvp, solution_error, iters)
    return CoDual(result, Mooncake.zero_tangent(result)), mooncake_pullback
end

# ============================================================================
# get_loglikelihood - Mooncake integration (reuses ChainRulesCore rrule)
# ============================================================================

# Helper function to compute gradients using the ChainRulesCore rrule
function _compute_gradient_via_chainrules(
    𝓂::ℳ,
    data::KeyedArray{Float64},
    params::Vector{Float64};
    kwargs...
)
    rrule_result = ChainRulesCore.rrule(
        get_loglikelihood,
        𝓂,
        data,
        params;
        kwargs...
    )
    
    if rrule_result === nothing
        error("ChainRulesCore.rrule returned nothing for get_loglikelihood")
    end
    
    return rrule_result
end

# Mark get_loglikelihood as a primitive for Mooncake
@is_primitive MinimalCtx Tuple{typeof(Core.kwcall), Any, typeof(get_loglikelihood), ℳ, KeyedArray{Float64}, Vector{Float64}}
@is_primitive MinimalCtx Tuple{typeof(get_loglikelihood), ℳ, KeyedArray{Float64}, Vector{Float64}}

# rrule!! for Mooncake with keyword arguments (via Core.kwcall)
function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kwargs::CoDual,
    ::CoDual{typeof(get_loglikelihood)},
    𝓂::CoDual{<:ℳ},
    data::CoDual{<:KeyedArray{Float64}},
    parameter_values::CoDual{<:Vector{Float64}}
)
    kwargs_primal = Mooncake.primal(kwargs)
    𝓂_primal = Mooncake.primal(𝓂)
    data_primal = Mooncake.primal(data)
    params_primal = Mooncake.primal(parameter_values)
    
    loglikelihood, chainrules_pullback = _compute_gradient_via_chainrules(
        𝓂_primal,
        data_primal,
        params_primal;
        kwargs_primal...
    )
    
    function mooncake_pullback(dloglikelihood)
        _, _, _, dparams = chainrules_pullback(dloglikelihood)
        
        if dparams isa NoTangent || dparams === nothing
            return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), zero(params_primal)
        else
            return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), collect(dparams)
        end
    end
    
    return CoDual(loglikelihood, Mooncake.zero_tangent(loglikelihood)), mooncake_pullback
end

# rrule!! for Mooncake without keyword arguments
function Mooncake.rrule!!(
    ::CoDual{typeof(get_loglikelihood)},
    𝓂::CoDual{<:ℳ},
    data::CoDual{<:KeyedArray{Float64}},
    parameter_values::CoDual{<:Vector{Float64}}
)
    𝓂_primal = Mooncake.primal(𝓂)
    data_primal = Mooncake.primal(data)
    params_primal = Mooncake.primal(parameter_values)
    
    loglikelihood, chainrules_pullback = _compute_gradient_via_chainrules(
        𝓂_primal,
        data_primal,
        params_primal
    )
    
    function mooncake_pullback(dloglikelihood)
        _, _, _, dparams = chainrules_pullback(dloglikelihood)
        
        if dparams isa NoTangent || dparams === nothing
            return NoRData(), NoRData(), NoRData(), zero(params_primal)
        else
            return NoRData(), NoRData(), NoRData(), collect(dparams)
        end
    end
    
    return CoDual(loglikelihood, Mooncake.zero_tangent(loglikelihood)), mooncake_pullback
end

end # module