module MooncakeExt

using Mooncake
using Mooncake: @is_primitive, MinimalCtx, zero_fcodual, CoDual, tangent_type, NoPullback, NoRData
import MacroModelling: get_loglikelihood, ℳ, Tolerances
import MacroModelling: DEFAULT_ALGORITHM, DEFAULT_FILTER_SELECTOR, DEFAULT_WARMUP_ITERATIONS
import MacroModelling: DEFAULT_PRESAMPLE_PERIODS, DEFAULT_QME_ALGORITHM, DEFAULT_LYAPUNOV_ALGORITHM
import MacroModelling: DEFAULT_SYLVESTER_SELECTOR, DEFAULT_VERBOSE
import AxisKeys: KeyedArray
import ChainRulesCore
import ChainRulesCore: NoTangent

# Helper function to compute gradients using the ChainRulesCore rrule
# This is called from the Mooncake rrule!! to leverage the existing rrule
function _compute_gradient_via_chainrules(
    𝓂::ℳ,
    data::KeyedArray{Float64},
    params::Vector{Float64};
    kwargs...
)
    # Call the ChainRulesCore rrule to get result and pullback
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
# We need to handle the Core.kwcall signature for keyword arguments
@is_primitive MinimalCtx Tuple{typeof(Core.kwcall), Any, typeof(get_loglikelihood), ℳ, KeyedArray{Float64}, Vector{Float64}}

# Also mark the non-kwargs version as primitive
@is_primitive MinimalCtx Tuple{typeof(get_loglikelihood), ℳ, KeyedArray{Float64}, Vector{Float64}}

# Implement the rrule!! for Mooncake with keyword arguments (via Core.kwcall)
function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kwargs::CoDual,
    ::CoDual{typeof(get_loglikelihood)},
    𝓂::CoDual{<:ℳ},
    data::CoDual{<:KeyedArray{Float64}},
    parameter_values::CoDual{<:Vector{Float64}}
)
    # Extract primal values
    kwargs_primal = Mooncake.primal(kwargs)
    𝓂_primal = Mooncake.primal(𝓂)
    data_primal = Mooncake.primal(data)
    params_primal = Mooncake.primal(parameter_values)
    
    # Call the helper to get ChainRulesCore rrule result
    loglikelihood, chainrules_pullback = _compute_gradient_via_chainrules(
        𝓂_primal,
        data_primal,
        params_primal;
        kwargs_primal...
    )
    
    # Define the Mooncake pullback
    function mooncake_pullback(dloglikelihood)
        # Call the ChainRulesCore pullback
        _, _, _, dparams = chainrules_pullback(dloglikelihood)
        
        # Convert ChainRulesCore tangent to Mooncake format
        # Return NoRData for non-differentiable arguments (kwcall, kwargs, function, model, data)
        # Return the gradient for parameter_values
        if dparams isa NoTangent || dparams === nothing
            return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), zero(params_primal)
        else
            return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), collect(dparams)
        end
    end
    
    return CoDual(loglikelihood, Mooncake.zero_tangent(loglikelihood)), mooncake_pullback
end

# Implement the rrule!! for Mooncake without keyword arguments
function Mooncake.rrule!!(
    ::CoDual{typeof(get_loglikelihood)},
    𝓂::CoDual{<:ℳ},
    data::CoDual{<:KeyedArray{Float64}},
    parameter_values::CoDual{<:Vector{Float64}}
)
    # Extract primal values
    𝓂_primal = Mooncake.primal(𝓂)
    data_primal = Mooncake.primal(data)
    params_primal = Mooncake.primal(parameter_values)
    
    # Call the helper with default kwargs
    loglikelihood, chainrules_pullback = _compute_gradient_via_chainrules(
        𝓂_primal,
        data_primal,
        params_primal
    )
    
    # Define the Mooncake pullback
    function mooncake_pullback(dloglikelihood)
        # Call the ChainRulesCore pullback
        _, _, _, dparams = chainrules_pullback(dloglikelihood)
        
        # Convert ChainRulesCore tangent to Mooncake format
        if dparams isa NoTangent || dparams === nothing
            return NoRData(), NoRData(), NoRData(), zero(params_primal)
        else
            return NoRData(), NoRData(), NoRData(), collect(dparams)
        end
    end
    
    return CoDual(loglikelihood, Mooncake.zero_tangent(loglikelihood)), mooncake_pullback
end

end # module


