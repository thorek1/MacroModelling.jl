module MooncakeExt

import MacroModelling
import Mooncake
import Mooncake: CoDual, NoRData, @is_primitive
import ChainRulesCore
import AxisKeys: KeyedArray

# ── Inference short-circuit for Mooncake primitives ──
# Mooncake's default abstract_call_gf_by_type runs full type inference on every
# call BEFORE checking whether it is a primitive.  For large primitives like
# get_statistics (thousands of transitive callees) this cascade takes 10+ min
# through Mooncake's fresh inference cache.
#
# Fix: check the primitive predicate FIRST.  When a call is recognised as a
# primitive, return a conservative CallMeta (return type Any, unknown effects)
# immediately, skipping the expensive inference cascade.  Correctness is
# preserved because Mooncake's AD tape uses CoDual values with concrete runtime
# types, and make_ad_stmts! already handles imprecise inferred types.
#
# Defined in __init__ to avoid "method overwriting during precompilation" error.
# function __init__()
#     @static if VERSION >= v"1.12-"
#         CC = Core.Compiler
#         @eval begin
#             function $CC.abstract_call_gf_by_type(
#                 interp::Mooncake.MooncakeInterpreter{C,M},
#                 @nospecialize(f),
#                 arginfo::$CC.ArgInfo,
#                 si::$CC.StmtInfo,
#                 @nospecialize(atype),
#                 sv::$CC.AbsIntState,
#                 max_methods::Int,
#             ) where {C,M}
#                 argtypes = arginfo.argtypes
#                 matches = $CC.find_method_matches(interp, argtypes, atype; max_methods)
#                 if !isa(matches, $CC.FailedMethodMatch)
#                     (; applicable) = matches
#                     if Mooncake.any_matches_primitive(applicable, C, M, interp.world)
#                         info = Mooncake.NoInlineCallInfo($CC.NoCallInfo(), atype)
#                         cm = $CC.CallMeta(Any, Any, $CC.Effects(), info)
#                         return $CC.Future(cm)
#                     end
#                 end
#                 return @invoke $CC.abstract_call_gf_by_type(
#                     interp::$CC.AbstractInterpreter,
#                     f::Any,
#                     arginfo::$CC.ArgInfo,
#                     si::$CC.StmtInfo,
#                     atype::Any,
#                     sv::$CC.AbsIntState,
#                     max_methods::Int,
#                 )
#             end
#         end
#     end
# end

Mooncake.tangent_type(::Type{MacroModelling.ℳ}) = Mooncake.NoTangent

# ── Scalar/Array-returning functions: @from_rrule works directly ──

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_solution), MacroModelling.ℳ, Vector{T}} where {T<:Base.IEEEFloat} true

# 3-arg kwarg-only path (no AD through initial_state, current behavior)
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, KeyedArray{Float64}, Vector{T}} where {T<:Base.IEEEFloat} true
# 4-arg positional path (AD through initial_state)
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, KeyedArray{Float64}, Vector{T}, Vector{Float64}} where {T<:Base.IEEEFloat} true
# Nested Vector{Vector} initial_state needs a manual rrule!! below because Mooncake
# cannot increment rdata for Vector{Vector{Float64}} through @from_rrule.

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_irf), MacroModelling.ℳ, Vector{T}} where {T<:Base.IEEEFloat} true

# get_loglikelihood:
#   (𝓂, data, parameter_values::Vector{T}, shocks::Matrix{T}, me_std::T_or_Vector{T})
# Two narrow @from_rrule generations cover scalar and vector me_std cases.
# 5-arg kwarg-only path (no AD through initial_state)
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, KeyedArray{Float64}, Vector{T}, Matrix{T}, T} where {T<:Base.IEEEFloat} true
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, KeyedArray{Float64}, Vector{T}, Matrix{T}, Vector{T}} where {T<:Base.IEEEFloat} true
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, KeyedArray{Float64}, Vector{T}, Matrix{T}, Matrix{T}} where {T<:Base.IEEEFloat} true
# 6-arg positional path (AD through initial_state)
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, KeyedArray{Float64}, Vector{T}, Matrix{T}, T, Vector{Float64}} where {T<:Base.IEEEFloat} true
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, KeyedArray{Float64}, Vector{T}, Matrix{T}, Vector{T}, Vector{Float64}} where {T<:Base.IEEEFloat} true
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, KeyedArray{Float64}, Vector{T}, Matrix{T}, Matrix{T}, Vector{Float64}} where {T<:Base.IEEEFloat} true
# Nested Vector{Vector} initial_state forms are implemented manually below.

# ── DynamicPPL compatibility: wider @is_primitive declarations ──
# Inside a Turing @model evaluated through DynamicPPL.logdensity_at,
# tilde_assume!! returns Any, so Julia's type inference widens the params
# argument to Any at the call site. The narrow signatures generated by
# @from_rrule do not match during Mooncake's abstract interpretation,
# causing it to trace into the full function body (~700s).
# These wider declarations ensure the primitive is recognized.
# At runtime, CoDual carries concrete types, so the narrow rrule!! methods
# auto-generated by @from_rrule still dispatch correctly.
@is_primitive Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, Any, Any}
@is_primitive Mooncake.DefaultCtx Tuple{typeof(Core.kwcall), <:NamedTuple, typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, Any, Any}
@is_primitive Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, Any, Any, Any}
@is_primitive Mooncake.DefaultCtx Tuple{typeof(Core.kwcall), <:NamedTuple, typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, Any, Any, Any}

@is_primitive Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_solution), MacroModelling.ℳ, Any}
@is_primitive Mooncake.DefaultCtx Tuple{typeof(Core.kwcall), <:NamedTuple, typeof(MacroModelling.get_solution), MacroModelling.ℳ, Any}

@is_primitive Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_irf), MacroModelling.ℳ, Any}
@is_primitive Mooncake.DefaultCtx Tuple{typeof(Core.kwcall), <:NamedTuple, typeof(MacroModelling.get_irf), MacroModelling.ℳ, Any}

@is_primitive Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_relevant_steady_state_and_state_update), Any, Any, MacroModelling.ℳ}
@is_primitive Mooncake.DefaultCtx Tuple{typeof(Core.kwcall), <:NamedTuple, typeof(MacroModelling.get_relevant_steady_state_and_state_update), Any, Any, MacroModelling.ℳ}

relevant_component_to_cr(::Mooncake.NoTangent) = ChainRulesCore.NoTangent()
relevant_component_to_cr(tangent) = Mooncake.to_cr_tangent(tangent)

function relevant_state_update_output_to_cr(tangent)
    tangent isa Mooncake.NoTangent && return ChainRulesCore.NoTangent()
    return (ChainRulesCore.NoTangent(),
            relevant_component_to_cr(tangent[2]),
            ChainRulesCore.NoTangent(),
            relevant_component_to_cr(tangent[4]),
            ChainRulesCore.NoTangent())
end

function Mooncake.rrule!!(
    f_cd::CoDual{typeof(MacroModelling.get_relevant_steady_state_and_state_update)},
    algorithm_cd::CoDual{<:Val},
    params_cd::CoDual{Vector{T}},
    model_cd::CoDual{MacroModelling.ℳ},
) where {T<:Base.IEEEFloat}
    fargs = (f_cd, algorithm_cd, params_cd, model_cd)
    primals = map(Mooncake.primal, fargs)
    lazy_rdata = map(Mooncake.lazy_zero_rdata, primals)
    y_primal, cr_pb = ChainRulesCore.rrule(primals...)
    y_fdata = Mooncake.fdata(Mooncake.zero_tangent(y_primal))
    function pb!!(y_rdata)
        cr_tangent = relevant_state_update_output_to_cr(Mooncake.tangent(y_fdata, y_rdata))
        cr_dfargs = cr_pb(cr_tangent)
        return map(fargs, lazy_rdata, cr_dfargs) do x, lr, cr_dx
            Mooncake.increment_and_get_rdata!(Mooncake.tangent(x), Mooncake.instantiate(lr), cr_dx)
        end
    end
    return CoDual(y_primal, y_fdata), pb!!
end

function Mooncake.rrule!!(
    kwcall_cd::CoDual{typeof(Core.kwcall)},
    kwargs_cd::CoDual{<:NamedTuple},
    f_cd::CoDual{typeof(MacroModelling.get_relevant_steady_state_and_state_update)},
    algorithm_cd::CoDual{<:Val},
    params_cd::CoDual{Vector{T}},
    model_cd::CoDual{MacroModelling.ℳ},
) where {T<:Base.IEEEFloat}
    kwargs = Mooncake.primal(kwargs_cd)
    algorithm = Mooncake.primal(algorithm_cd)
    params = Mooncake.primal(params_cd)
    model = Mooncake.primal(model_cd)
    y_primal, cr_pb = ChainRulesCore.rrule(MacroModelling.get_relevant_steady_state_and_state_update,
                                           algorithm, params, model; kwargs...)
    y_fdata = Mooncake.fdata(Mooncake.zero_tangent(y_primal))
    kwargs_lazy_rdata = Mooncake.lazy_zero_rdata(kwargs)
    inner_fargs = (f_cd, algorithm_cd, params_cd, model_cd)
    lazy_rdata = map(cd -> Mooncake.lazy_zero_rdata(Mooncake.primal(cd)), inner_fargs)
    function pb!!(y_rdata)
        cr_tangent = relevant_state_update_output_to_cr(Mooncake.tangent(y_fdata, y_rdata))
        cr_dfargs = cr_pb(cr_tangent)
        kwargs_rdata = Mooncake.increment_and_get_rdata!(
            Mooncake.tangent(kwargs_cd),
            Mooncake.instantiate(kwargs_lazy_rdata),
            ChainRulesCore.NoTangent(),
        )
        inner_rdata = map(inner_fargs, lazy_rdata, cr_dfargs) do x, lr, cr_dx
            Mooncake.increment_and_get_rdata!(Mooncake.tangent(x), Mooncake.instantiate(lr), cr_dx)
        end
        return (NoRData(), kwargs_rdata, inner_rdata...)
    end
    return CoDual(y_primal, y_fdata), pb!!
end

# Wide primitive declarations for get_loglikelihood so that the
# Turing/DynamicPPL call site (which widens argument types to Any) still
# matches the registered Mooncake primitive.
@is_primitive Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, Any, Any, Any, Any}
@is_primitive Mooncake.DefaultCtx Tuple{typeof(Core.kwcall), <:NamedTuple, typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, Any, Any, Any, Any}
@is_primitive Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, Any, Any, Any, Any, Any}
@is_primitive Mooncake.DefaultCtx Tuple{typeof(Core.kwcall), <:NamedTuple, typeof(MacroModelling.get_loglikelihood), MacroModelling.ℳ, Any, Any, Any, Any, Any}

function increment_nested_initial_state_rdata!(
    initial_state_cd::CoDual{<:Vector{<:AbstractVector}},
    cr_tangent,
)
    cr_tangent isa ChainRulesCore.AbstractZero && return NoRData()
    initial_state_fdata = Mooncake.tangent(initial_state_cd)
    @inbounds for i in eachindex(cr_tangent)
        component_tangent = cr_tangent[i]
        component_tangent isa ChainRulesCore.AbstractZero && continue
        Mooncake.increment_and_get_rdata!(initial_state_fdata[i], NoRData(), component_tangent)
    end
    return NoRData()
end

function increment_rrule_args_with_nested_initial_state(fargs::Tuple, lazy_rdata::Tuple, cr_dfargs::Tuple)
    regular_rdata = ntuple(i -> Mooncake.increment_and_get_rdata!(
        Mooncake.tangent(fargs[i]),
        Mooncake.instantiate(lazy_rdata[i]),
        cr_dfargs[i],
    ), length(fargs) - 1)
    initial_state_rdata = increment_nested_initial_state_rdata!(fargs[end], cr_dfargs[end])
    return (regular_rdata..., initial_state_rdata)
end

scalar_output_to_cr(y_fdata, y_rdata) = Mooncake.to_cr_tangent(Mooncake.tangent(y_fdata, y_rdata))

function Mooncake.rrule!!(
    f_cd::CoDual{typeof(MacroModelling.get_loglikelihood)},
    model_cd::CoDual{MacroModelling.ℳ},
    data_cd::CoDual{<:KeyedArray{Float64}},
    params_cd::CoDual{Vector{T}},
    initial_state_cd::CoDual{Vector{Vector{I}}},
) where {T<:Base.IEEEFloat,I<:Base.IEEEFloat}
    fargs = (f_cd, model_cd, data_cd, params_cd, initial_state_cd)
    primals = map(Mooncake.primal, fargs)
    lazy_rdata = map(Mooncake.lazy_zero_rdata, primals)
    y_primal, cr_pb = ChainRulesCore.rrule(primals...)
    y_fdata = Mooncake.fdata(Mooncake.zero_tangent(y_primal))
    function pb!!(y_rdata)
        cr_dfargs = cr_pb(scalar_output_to_cr(y_fdata, y_rdata))
        return increment_rrule_args_with_nested_initial_state(fargs, lazy_rdata, cr_dfargs)
    end
    return CoDual(y_primal, y_fdata), pb!!
end

function Mooncake.rrule!!(
    kwcall_cd::CoDual{typeof(Core.kwcall)},
    kwargs_cd::CoDual{<:NamedTuple},
    f_cd::CoDual{typeof(MacroModelling.get_loglikelihood)},
    model_cd::CoDual{MacroModelling.ℳ},
    data_cd::CoDual{<:KeyedArray{Float64}},
    params_cd::CoDual{Vector{T}},
    initial_state_cd::CoDual{Vector{Vector{I}}},
) where {T<:Base.IEEEFloat,I<:Base.IEEEFloat}
    kwargs = Mooncake.primal(kwargs_cd)
    model = Mooncake.primal(model_cd)
    data = Mooncake.primal(data_cd)
    params = Mooncake.primal(params_cd)
    initial_state = Mooncake.primal(initial_state_cd)
    y_primal, cr_pb = ChainRulesCore.rrule(MacroModelling.get_loglikelihood,
                                           model, data, params, initial_state; kwargs...)
    y_fdata = Mooncake.fdata(Mooncake.zero_tangent(y_primal))
    kwargs_lazy_rdata = Mooncake.lazy_zero_rdata(kwargs)
    inner_fargs = (f_cd, model_cd, data_cd, params_cd, initial_state_cd)
    lazy_rdata = map(cd -> Mooncake.lazy_zero_rdata(Mooncake.primal(cd)), inner_fargs)
    function pb!!(y_rdata)
        cr_dfargs = cr_pb(scalar_output_to_cr(y_fdata, y_rdata))
        kwargs_rdata = Mooncake.increment_and_get_rdata!(
            Mooncake.tangent(kwargs_cd),
            Mooncake.instantiate(kwargs_lazy_rdata),
            ChainRulesCore.NoTangent(),
        )
        inner_rdata = increment_rrule_args_with_nested_initial_state(inner_fargs, lazy_rdata, cr_dfargs)
        return (NoRData(), kwargs_rdata, inner_rdata...)
    end
    return CoDual(y_primal, y_fdata), pb!!
end

function Mooncake.rrule!!(
    f_cd::CoDual{typeof(MacroModelling.get_loglikelihood)},
    model_cd::CoDual{MacroModelling.ℳ},
    data_cd::CoDual{<:KeyedArray{Float64}},
    params_cd::CoDual{Vector{T}},
    shocks_cd::CoDual{Matrix{T}},
    measurement_error_std_cd::CoDual{M},
    initial_state_cd::CoDual{Vector{Vector{I}}},
) where {T<:Base.IEEEFloat,I<:Base.IEEEFloat,M<:Union{T,Vector{T},Matrix{T}}}
    fargs = (f_cd, model_cd, data_cd, params_cd, shocks_cd, measurement_error_std_cd, initial_state_cd)
    primals = map(Mooncake.primal, fargs)
    lazy_rdata = map(Mooncake.lazy_zero_rdata, primals)
    y_primal, cr_pb = ChainRulesCore.rrule(primals...)
    y_fdata = Mooncake.fdata(Mooncake.zero_tangent(y_primal))
    function pb!!(y_rdata)
        cr_dfargs = cr_pb(scalar_output_to_cr(y_fdata, y_rdata))
        return increment_rrule_args_with_nested_initial_state(fargs, lazy_rdata, cr_dfargs)
    end
    return CoDual(y_primal, y_fdata), pb!!
end

function Mooncake.rrule!!(
    kwcall_cd::CoDual{typeof(Core.kwcall)},
    kwargs_cd::CoDual{<:NamedTuple},
    f_cd::CoDual{typeof(MacroModelling.get_loglikelihood)},
    model_cd::CoDual{MacroModelling.ℳ},
    data_cd::CoDual{<:KeyedArray{Float64}},
    params_cd::CoDual{Vector{T}},
    shocks_cd::CoDual{Matrix{T}},
    measurement_error_std_cd::CoDual{M},
    initial_state_cd::CoDual{Vector{Vector{I}}},
) where {T<:Base.IEEEFloat,I<:Base.IEEEFloat,M<:Union{T,Vector{T},Matrix{T}}}
    kwargs = Mooncake.primal(kwargs_cd)
    model = Mooncake.primal(model_cd)
    data = Mooncake.primal(data_cd)
    params = Mooncake.primal(params_cd)
    shocks = Mooncake.primal(shocks_cd)
    measurement_error_std = Mooncake.primal(measurement_error_std_cd)
    initial_state = Mooncake.primal(initial_state_cd)
    y_primal, cr_pb = ChainRulesCore.rrule(MacroModelling.get_loglikelihood,
                                           model, data, params, shocks, measurement_error_std, initial_state; kwargs...)
    y_fdata = Mooncake.fdata(Mooncake.zero_tangent(y_primal))
    kwargs_lazy_rdata = Mooncake.lazy_zero_rdata(kwargs)
    inner_fargs = (f_cd, model_cd, data_cd, params_cd, shocks_cd, measurement_error_std_cd, initial_state_cd)
    lazy_rdata = map(cd -> Mooncake.lazy_zero_rdata(Mooncake.primal(cd)), inner_fargs)
    function pb!!(y_rdata)
        cr_dfargs = cr_pb(scalar_output_to_cr(y_fdata, y_rdata))
        kwargs_rdata = Mooncake.increment_and_get_rdata!(
            Mooncake.tangent(kwargs_cd),
            Mooncake.instantiate(kwargs_lazy_rdata),
            ChainRulesCore.NoTangent(),
        )
        inner_rdata = increment_rrule_args_with_nested_initial_state(inner_fargs, lazy_rdata, cr_dfargs)
        return (NoRData(), kwargs_rdata, inner_rdata...)
    end
    return CoDual(y_primal, y_fdata), pb!!
end

# ── get_statistics: manual rrule!! ──
# Returns Dict{Symbol,...} whose MutableTangent cannot be converted by to_cr_tangent.
# We mirror rrule_wrapper but reconstruct the Dict cotangent from MutableTangent fields.

# Convert MutableTangent (Dict internals: slots/keys/vals/...) → actual Dict cotangent
function mooncake_dict_to_cr_tangent(primal_dict::Dict, mt::Mooncake.MutableTangent)
    result = Dict{Symbol,Any}()
    raw_vals = mt.fields.vals
    vals_tangent = if raw_vals isa Mooncake.PossiblyUninitTangent
        Mooncake.is_init(raw_vals) ? raw_vals.tangent : return result
    else
        raw_vals
    end
    for (k, _) in primal_dict
        idx = Base.ht_keyindex(primal_dict, k)
        idx > 0 || continue
        isassigned(vals_tangent, idx) || continue
        vt = vals_tangent[idx]
        cr_vt = val_to_cr(vt)
        cr_vt isa ChainRulesCore.AbstractZero && continue
        result[k] = cr_vt
    end
    return result
end
mooncake_dict_to_cr_tangent(::Dict, ::Mooncake.NoTangent) = ChainRulesCore.NoTangent()

val_to_cr(x::AbstractArray{<:AbstractFloat}) = x
val_to_cr(::Mooncake.NoTangent) = ChainRulesCore.ZeroTangent()
val_to_cr(x::Mooncake.PossiblyUninitTangent) =
    Mooncake.is_init(x) ? val_to_cr(x.tangent) : ChainRulesCore.ZeroTangent()
val_to_cr(x) = Mooncake.to_cr_tangent(x)

# Positional: get_statistics(model, params)
@is_primitive Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_statistics), MacroModelling.ℳ, Vector{T}} where {T<:Base.IEEEFloat}
@is_primitive Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_statistics), MacroModelling.ℳ, Any}

function Mooncake.rrule!!(
    f_cd::CoDual{typeof(MacroModelling.get_statistics)},
    model_cd::CoDual{MacroModelling.ℳ},
    params_cd::CoDual{Vector{T}}
) where {T<:Base.IEEEFloat}
    fargs = (f_cd, model_cd, params_cd)
    primals = map(Mooncake.primal, fargs)
    lazy_rdata = map(Mooncake.lazy_zero_rdata, primals)
    y_primal, cr_pb = ChainRulesCore.rrule(primals...)
    y_fdata = Mooncake.fdata(Mooncake.zero_tangent(y_primal))
    function pb!!(y_rdata)
        cr_tangent = mooncake_dict_to_cr_tangent(y_primal, Mooncake.tangent(y_fdata, y_rdata))
        cr_dfargs = cr_pb(cr_tangent)
        return map(fargs, lazy_rdata, cr_dfargs) do x, lr, cr_dx
            Mooncake.increment_and_get_rdata!(Mooncake.tangent(x), Mooncake.instantiate(lr), cr_dx)
        end
    end
    return CoDual(y_primal, y_fdata), pb!!
end

# Keyword: Core.kwcall(kwargs, get_statistics, model, params)
@is_primitive Mooncake.DefaultCtx Tuple{typeof(Core.kwcall), <:NamedTuple, typeof(MacroModelling.get_statistics), MacroModelling.ℳ, Vector{T}} where {T<:Base.IEEEFloat}
@is_primitive Mooncake.DefaultCtx Tuple{typeof(Core.kwcall), <:NamedTuple, typeof(MacroModelling.get_statistics), MacroModelling.ℳ, Any}

function Mooncake.rrule!!(
    kwcall_cd::CoDual{typeof(Core.kwcall)},
    kwargs_cd::CoDual{<:NamedTuple},
    f_cd::CoDual{typeof(MacroModelling.get_statistics)},
    model_cd::CoDual{MacroModelling.ℳ},
    params_cd::CoDual{Vector{T}}
) where {T<:Base.IEEEFloat}
    kw = Mooncake.primal(kwargs_cd)
    model = Mooncake.primal(model_cd)
    params = Mooncake.primal(params_cd)
    # Call ChainRules rrule directly with kwargs (Core.kwcall has no rrule)
    y_primal, cr_pb = ChainRulesCore.rrule(MacroModelling.get_statistics, model, params; kw...)
    y_fdata = Mooncake.fdata(Mooncake.zero_tangent(y_primal))
    kwargs_lazy_rdata = Mooncake.lazy_zero_rdata(kw)
    inner_fargs = (f_cd, model_cd, params_cd)
    lazy_rdata = map(cd -> Mooncake.lazy_zero_rdata(Mooncake.primal(cd)), inner_fargs)
    function pb!!(y_rdata)
        cr_tangent = mooncake_dict_to_cr_tangent(y_primal, Mooncake.tangent(y_fdata, y_rdata))
        cr_dfargs = cr_pb(cr_tangent)
        kwargs_rdata = Mooncake.increment_and_get_rdata!(
            Mooncake.tangent(kwargs_cd),
            Mooncake.instantiate(kwargs_lazy_rdata),
            ChainRulesCore.NoTangent(),
        )
        inner_rdata = map(inner_fargs, lazy_rdata, cr_dfargs) do x, lr, cr_dx
            Mooncake.increment_and_get_rdata!(Mooncake.tangent(x), Mooncake.instantiate(lr), cr_dx)
        end
        return (NoRData(), kwargs_rdata, inner_rdata...)
    end
    return CoDual(y_primal, y_fdata), pb!!
end

# ── Dict getindex primitive ──
# Without this, Mooncake tries to compile a tape through Dict's hash table internals
# (hashing, slot probing, Memory access), which takes extremely long.
# For mutable containers, fdata is accumulated in-place so the pullback is a no-op.
@is_primitive Mooncake.DefaultCtx Tuple{typeof(Base.getindex), <:Dict{Symbol}, Symbol}

function Mooncake.rrule!!(
    ::CoDual{typeof(Base.getindex)},
    dict_cd::CoDual{<:Dict{Symbol}},
    key_cd::CoDual{Symbol}
)
    dict = Mooncake.primal(dict_cd)
    key = Mooncake.primal(key_cd)
    val = dict[key]
    dict_fdata = Mooncake.tangent(dict_cd)
    idx = Base.ht_keyindex(dict, key)
    val_fdata = dict_fdata.fields.vals[idx]
    function pb!!(::NoRData)
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(val, val_fdata), pb!!
end


# ── clear_solution_caches! primitive ──
# This function mutates model caches and has no differentiable effect on outputs.
# Registering it as a primitive with zero gradient prevents Mooncake from tracing
# through its internals when it is called inside differentiable closures.
@is_primitive Mooncake.DefaultCtx Tuple{typeof(MacroModelling.clear_solution_caches!), MacroModelling.ℳ, Symbol}

function Mooncake.rrule!!(
    ::CoDual{typeof(MacroModelling.clear_solution_caches!)},
    model_cd::CoDual{MacroModelling.ℳ},
    alg_cd::CoDual{Symbol}
)
    MacroModelling.clear_solution_caches!(Mooncake.primal(model_cd), Mooncake.primal(alg_cd))
    pb!!(::NoRData) = (NoRData(), NoRData(), NoRData())
    return CoDual(nothing, Mooncake.NoFData()), pb!!
end

end  # module MooncakeExt
