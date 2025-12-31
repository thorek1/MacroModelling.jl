module MooncakeExt

using Mooncake
using Mooncake: @from_chainrules
import MacroModelling: get_loglikelihood, ℳ, Tolerances
import MacroModelling: DEFAULT_ALGORITHM, DEFAULT_FILTER_SELECTOR, DEFAULT_WARMUP_ITERATIONS
import MacroModelling: DEFAULT_PRESAMPLE_PERIODS, DEFAULT_QME_ALGORITHM, DEFAULT_LYAPUNOV_ALGORITHM
import MacroModelling: DEFAULT_SYLVESTER_SELECTOR, DEFAULT_VERBOSE
import AxisKeys: KeyedArray
import ChainRulesCore

# Import the ChainRulesCore rrule for get_loglikelihood into Mooncake
# This allows Mooncake to use the same gradient computation as Zygote
@from_chainrules ChainRulesCore.rrule(
    ::typeof(get_loglikelihood),
    𝓂::ℳ, 
    data::KeyedArray{Float64}, 
    parameter_values::Vector{Float64};
    algorithm::Symbol,
    filter::Symbol,
    on_failure_loglikelihood::Float64,
    warmup_iterations::Int,
    presample_periods::Int,
    initial_covariance::Symbol,
    filter_algorithm::Symbol,
    tol::Tolerances,
    quadratic_matrix_equation_algorithm::Symbol,
    lyapunov_algorithm::Symbol,
    sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}},
    verbose::Bool
)

end # module
