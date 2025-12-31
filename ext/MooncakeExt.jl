module MooncakeExt

using Mooncake
using Mooncake: @is_primitive, MinimalCtx, CoDual, primal, zero_fcodual, tangent
import MacroModelling
import MacroModelling: get_loglikelihood, ℳ, CalculationOptions, Tolerances
import MacroModelling: DEFAULT_ALGORITHM, DEFAULT_FILTER_SELECTOR, DEFAULT_WARMUP_ITERATIONS
import MacroModelling: DEFAULT_PRESAMPLE_PERIODS, DEFAULT_QME_ALGORITHM, DEFAULT_LYAPUNOV_ALGORITHM
import MacroModelling: DEFAULT_SYLVESTER_SELECTOR, DEFAULT_VERBOSE
import AxisKeys: KeyedArray
import Zygote

# Mark get_loglikelihood as a primitive for Mooncake
# This allows Mooncake to use a custom rrule!! that delegates to Zygote for gradient computation
@is_primitive MinimalCtx Tuple{
    typeof(get_loglikelihood),
    ℳ,
    KeyedArray{Float64},
    Vector{Float64}
}

# Define Mooncake's rrule!! for get_loglikelihood
# This delegates to Zygote to ensure identical gradients
function Mooncake.rrule!!(
    ::CoDual{typeof(get_loglikelihood)},
    𝓂::CoDual{ℳ},
    data::CoDual{KeyedArray{Float64}},
    parameter_values::CoDual{Vector{Float64}};
    algorithm::Symbol = DEFAULT_ALGORITHM, 
    filter::Symbol = DEFAULT_FILTER_SELECTOR(algorithm), 
    on_failure_loglikelihood::Float64 = -Inf,
    warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS, 
    presample_periods::Int = DEFAULT_PRESAMPLE_PERIODS,
    initial_covariance::Symbol = :theoretical,
    filter_algorithm::Symbol = :LagrangeNewton,
    tol::Tolerances = Tolerances(), 
    quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM, 
    lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM, 
    sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(primal(𝓂)),
    verbose::Bool = DEFAULT_VERBOSE
)
    # Extract primal values
    m = primal(𝓂)
    d = primal(data)
    pv = primal(parameter_values)

    # Use Zygote to compute the forward pass and get the pullback
    llh, zygote_pullback = Zygote.pullback(pv) do params
        get_loglikelihood(m, d, params,
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
            verbose = verbose
        )
    end
    
    # Create Mooncake pullback that delegates to Zygote's pullback
    function mooncake_pullback(Δllh)
        # Call Zygote pullback with the cotangent
        (∂params,) = zygote_pullback(Δllh)
        
        # Accumulate gradient into parameter_values tangent
        pv_tangent = tangent(parameter_values)
        if ∂params !== nothing && pv_tangent !== Mooncake.NoTangent()
            pv_tangent .+= ∂params
        end
        
        return Mooncake.NoRData()
    end
    
    return zero_fcodual(llh), mooncake_pullback
end

end # module
