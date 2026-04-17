# Equation-modification reprocessing pipeline.
#
# The `@model` and `@parameters` macros on this branch fully construct a model
# struct and all its compiled helpers. To keep behaviour consistent after an
# equation modification, this file provides a rebuild pipeline that
# 1. serializes the current user-facing model state (equations + parameter
#    block) into the same expression form the macros expect,
# 2. re-runs those macros against a scratch global binding, and
# 3. copies the freshly built fields back into the user's model struct so
#    that existing references keep working and the revision history is
#    preserved.

"""
    reset_solver_state!(𝓂::ℳ)

Invalidate every cached solver result so the next `solve!` call recomputes
from scratch. Also marks the compiled model functions as outdated so the
rebuild pipeline rewrites them.
"""
function reset_solver_state!(𝓂::ℳ)
    𝓂.caches.valid_for = valid_for_caches()
    empty!(𝓂.caches.solver)
    𝓂.caches.non_stochastic_steady_state = Float64[]
    𝓂.functions.functions_written = false
    return nothing
end


"""
    reconstruct_parameter_block(𝓂; calibration_original_override = nothing,
                                   parameter_overrides = nothing)

Return a `:block` `Expr` that reproduces a valid `@parameters` body from the
current model state. Optional overrides let callers preview the effect of a
calibration change before committing it to the model.
"""
function reconstruct_parameter_block(𝓂::ℳ;
                                     calibration_original_override::Union{Nothing, Vector{Expr}} = nothing,
                                     parameter_overrides::Union{Nothing, AbstractDict{Symbol, <:Real}} = nothing)
    lines = Any[]

    calibration_original = calibration_original_override === nothing ?
        𝓂.equations.calibration_original : calibration_original_override

    new_calib_params = Set{Symbol}()
    for eq in calibration_original
        p = extract_calibrated_parameter(eq)
        if p !== nothing
            push!(new_calib_params, p)
        end
    end

    old_calib_params = Set{Symbol}(𝓂.equations.calibration_parameters)
    params_becoming_calibrated = setdiff(new_calib_params, old_calib_params)
    params_no_longer_calibrated = setdiff(old_calib_params, new_calib_params)

    fixed_params = 𝓂.constants.post_complete_parameters.parameters
    fixed_values = 𝓂.parameter_values
    for (p, v) in zip(fixed_params, fixed_values)
        (isnan(v) || p in params_becoming_calibrated) && continue
        val = (parameter_overrides !== nothing && haskey(parameter_overrides, p)) ? parameter_overrides[p] : v
        push!(lines, Expr(:(=), p, val))
    end

    if !isempty(params_no_longer_calibrated)
        n_vars = 𝓂.constants.post_model_macro.nVars
        old_calib_list = 𝓂.equations.calibration_parameters
        for p in params_no_longer_calibrated
            if parameter_overrides !== nothing && haskey(parameter_overrides, p)
                push!(lines, Expr(:(=), p, parameter_overrides[p]))
                continue
            end
            idx = findfirst(==(p), old_calib_list)
            if idx !== nothing && length(𝓂.caches.non_stochastic_steady_state) >= n_vars + idx
                val = 𝓂.caches.non_stochastic_steady_state[n_vars + idx]
                push!(lines, Expr(:(=), p, val))
            end
        end
    end

    for eq in 𝓂.equations.calibration_no_var
        push!(lines, eq)
    end

    for eq in calibration_original
        push!(lines, eq)
    end

    for (p, (lo, hi)) in 𝓂.constants.post_parameters_macro.bounds
        push!(lines, Expr(:comparison, lo, :(<), p, :(<), hi))
    end

    return Expr(:block, lines...)
end


"""
    extract_calibrated_parameter(eq::Expr) -> Union{Symbol, Nothing}

Return the parameter on the right of the `|` in a calibration equation such
as `k[ss] = 1.5 | δ`. Returns `nothing` if no calibration marker is found.
"""
function extract_calibrated_parameter(eq::Expr)::Union{Symbol, Nothing}
    result = Ref{Union{Symbol, Nothing}}(nothing)
    postwalk(eq) do x
        if x isa Expr && x.head == :call && !isempty(x.args) && x.args[1] == :|
            if length(x.args) >= 3 && x.args[end] isa Symbol
                result[] = x.args[end]
            end
        end
        x
    end
    return result[]
end


# --- internal macro re-evaluation helpers ---------------------------------

const _REBUILD_SCRATCH_SYM = :__macromodelling_rebuild_scratch__

function _rebuild_via_macros!(𝓂::ℳ,
                              new_equations::Vector{Expr},
                              parameter_block::Expr;
                              verbose::Bool = false,
                              silent::Bool = true)
    saved_history = copy(𝓂.revision_history)
    # Retain a stable handle to the model struct even if a scratch rebuild fails
    model_body = Expr(:block, new_equations...)

    # Build @model scratch begin ... end
    model_call = Expr(:macrocall,
        GlobalRef(@__MODULE__, Symbol("@model")),
        LineNumberNode(0),
        _REBUILD_SCRATCH_SYM,
        model_body,
    )

    # Build @parameters scratch silent=true begin ... end
    param_call = Expr(:macrocall,
        GlobalRef(@__MODULE__, Symbol("@parameters")),
        LineNumberNode(0),
        _REBUILD_SCRATCH_SYM,
        Expr(:(=), :silent, silent),
        Expr(:(=), :verbose, verbose),
        Expr(:(=), :report_missing_parameters, false),
        parameter_block,
    )

    # Run both macros inside Main so the global assignment in @model lands on
    # a well-known scratch binding we can read back. We use Core.eval to pick
    # up the freshly-created binding at the correct world age.
    Core.eval(Main, model_call)
    Core.eval(Main, param_call)

    new_model::ℳ = Core.eval(Main, _REBUILD_SCRATCH_SYM)

    # Copy all fields from the fresh model onto the user's struct, preserving
    # the model_name and revision_history.
    original_name = 𝓂.model_name
    for f in fieldnames(ℳ)
        f === :revision_history && continue
        f === :model_name && continue
        setfield!(𝓂, f, getfield(new_model, f))
    end
    𝓂.model_name = original_name
    𝓂.revision_history = saved_history

    # Drop the scratch binding
    Core.eval(Main, :($_REBUILD_SCRATCH_SYM = nothing))

    return nothing
end


"""
    reprocess_model_equations!(𝓂, new_equations; parameters, verbose, silent)

Rebuild the model from an updated equation list while preserving the
revision history. Equivalent to re-running `@model` and `@parameters` on the
current parameter state.
"""
function reprocess_model_equations!(𝓂::ℳ,
                                    new_equations::Vector{Expr};
                                    parameters::ParameterType = nothing,
                                    verbose::Bool = false,
                                    silent::Bool = true)
    if parameters !== nothing
        write_parameters_input!(𝓂, parameters, verbose = verbose)
    end

    parameter_block = reconstruct_parameter_block(𝓂)
    reset_solver_state!(𝓂)
    _rebuild_via_macros!(𝓂, new_equations, parameter_block; verbose = verbose, silent = silent)
    return nothing
end


"""
    reprocess_calibration_equations!(𝓂, updated_calibration_original; parameters,
                                     parameter_overrides, verbose, silent)

Rebuild the model's calibration with a modified `calibration_original` list.
`parameter_overrides` lets callers supply replacement fixed values for
parameters that leave the calibration set.
"""
function reprocess_calibration_equations!(𝓂::ℳ,
                                          updated_calibration_original::Vector{Expr};
                                          parameters::ParameterType = nothing,
                                          parameter_overrides::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
                                          verbose::Bool = false,
                                          silent::Bool = true)
    if parameters !== nothing
        write_parameters_input!(𝓂, parameters, verbose = verbose)
    end

    parameter_block = reconstruct_parameter_block(𝓂;
        calibration_original_override = updated_calibration_original,
        parameter_overrides = parameter_overrides,
    )
    reset_solver_state!(𝓂)
    _rebuild_via_macros!(𝓂, 𝓂.equations.original, parameter_block; verbose = verbose, silent = silent)
    return nothing
end


"""
    finalize_model_update!(𝓂; verbose, silent)

No-op on this branch: the rebuild via `@model`/`@parameters` already
finalises the steady-state solver and derivative functions. Provided so the
call sites mirror the reference implementation.
"""
function finalize_model_update!(𝓂::ℳ; verbose::Bool = false, silent::Bool = true)
    return nothing
end
