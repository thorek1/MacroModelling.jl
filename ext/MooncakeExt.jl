module MooncakeExt

using Mooncake
using Mooncake: @is_primitive, MinimalCtx, CoDual, NoRData, tangent, NoFData
import MacroModelling: get_loglikelihood, get_NSSS_and_parameters_with_jacobian, ℳ, Tolerances, CalculationOptions, merge_calculation_options
import AxisKeys: KeyedArray
import ChainRulesCore
import ChainRulesCore: NoTangent

# ============================================================================
# get_NSSS_and_parameters_with_jacobian - Mooncake integration
# ============================================================================

# Mark get_NSSS_and_parameters_with_jacobian as a primitive for Mooncake
@is_primitive MinimalCtx Tuple{typeof(get_NSSS_and_parameters_with_jacobian), ℳ, Vector{Float64}}

# rrule!! for Mooncake
# Returns: (SS_and_pars::Vector, jvp::Matrix, solution_error::Float64, iters::Int)
function Mooncake.rrule!!(
    ::CoDual{typeof(get_NSSS_and_parameters_with_jacobian)},
    𝓂_codual::CoDual{<:ℳ},
    params_codual::CoDual{Vector{Float64}}
)
    𝓂_primal = Mooncake.primal(𝓂_codual)
    params_primal = Mooncake.primal(params_codual)
    params_tangent = tangent(params_codual)  # Mutable tangent vector
    
    # Forward pass
    SS_and_pars, jvp, solution_error, iters = get_NSSS_and_parameters_with_jacobian(𝓂_primal, params_primal)
    
    function mooncake_pullback(Δ)
        # Δ is tangent for the result tuple: (∂SS_and_pars, ∂jvp, ∂solution_error, ∂iters)
        ∂SS_and_pars = Δ[1]
        
        if ∂SS_and_pars !== nothing && !(∂SS_and_pars isa NoRData) && !iszero(∂SS_and_pars)
            # ∂params = jvp' * ∂SS_and_pars
            dparams = jvp' * ∂SS_and_pars
            params_tangent .+= dparams
        end
        
        return NoRData(), NoRData(), NoRData()
    end
    
    # Build result with proper fdata types:
    # Vector{Float64} -> Vector{Float64} tangent
    # Matrix{Float64} -> Matrix{Float64} tangent  
    # Float64 -> NoFData
    # Int -> NoFData
    result = (SS_and_pars, jvp, solution_error, iters)
    result_fdata = (
        zeros(length(SS_and_pars)),  # tangent for SS_and_pars
        zeros(size(jvp)),             # tangent for jvp
        NoFData(),                    # Float64 has NoFData
        NoFData()                     # Int has NoFData
    )
    
    return CoDual(result, result_fdata), mooncake_pullback
end

# ============================================================================
# get_loglikelihood - Mooncake integration
# ============================================================================

# Mark get_loglikelihood as a primitive for Mooncake
@is_primitive MinimalCtx Tuple{typeof(get_loglikelihood), ℳ, KeyedArray{Float64}, Vector{Float64}}

# rrule!! for Mooncake - returns Float64
function Mooncake.rrule!!(
    ::CoDual{typeof(get_loglikelihood)},
    𝓂_codual::CoDual{<:ℳ},
    data_codual::CoDual{<:KeyedArray{Float64}},
    params_codual::CoDual{Vector{Float64}}
)
    𝓂_primal = Mooncake.primal(𝓂_codual)
    data_primal = Mooncake.primal(data_codual)
    params_primal = Mooncake.primal(params_codual)
    params_tangent = tangent(params_codual)
    
    # Call ChainRulesCore rrule
    rrule_result = ChainRulesCore.rrule(get_loglikelihood, 𝓂_primal, data_primal, params_primal)
    
    if rrule_result === nothing
        error("ChainRulesCore.rrule returned nothing for get_loglikelihood")
    end
    
    loglikelihood, chainrules_pullback = rrule_result
    
    function mooncake_pullback(dloglikelihood::Float64)
        _, _, _, dparams = chainrules_pullback(dloglikelihood)
        
        if !(dparams isa NoTangent) && dparams !== nothing
            params_tangent .+= dparams
        end
        
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    
    # Float64 result has NoFData tangent
    return CoDual(loglikelihood, NoFData()), mooncake_pullback
end

end # module