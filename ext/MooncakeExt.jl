module MooncakeExt

using Mooncake
using Mooncake: @from_chainrules, MinimalCtx, ReverseMode, @is_primitive
import MacroModelling
import MacroModelling: _get_loglikelihood_internal, ℳ, CalculationOptions

# Mark _get_loglikelihood_internal as a primitive for Mooncake and use ChainRules rrule
# For first_order algorithm with kalman filter
@from_chainrules MinimalCtx Tuple{
    typeof(_get_loglikelihood_internal),
    Vector{Float64},
    Matrix{Float64},
    Vector{Int},
    Val{:first_order},
    Val{:kalman},
    Vector{Symbol},
    ℳ,
    Int,
    Symbol,
    Int,
    Symbol,
    CalculationOptions,
    Float64
} true ReverseMode

# For first_order algorithm with inversion filter
@from_chainrules MinimalCtx Tuple{
    typeof(_get_loglikelihood_internal),
    Vector{Float64},
    Matrix{Float64},
    Vector{Int},
    Val{:first_order},
    Val{:inversion},
    Vector{Symbol},
    ℳ,
    Int,
    Symbol,
    Int,
    Symbol,
    CalculationOptions,
    Float64
} true ReverseMode

end # module
