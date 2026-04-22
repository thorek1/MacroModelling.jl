# Equation-modification reprocessing pipeline.
#
# The `@model` and `@parameters` macros on this branch build a model struct
# and compile helpers. After an equation modification this file rebuilds the
# relevant internal state surgically — without re-evaluating the macros — by
# calling the pure functions `process_model_equations` and
# `process_parameter_definitions`.

"""
    reset_solver_state!(𝓂::ℳ)

Invalidate every cached solver result so the next `solve!` call recomputes
from scratch. Also marks the compiled model functions as outdated so the
rebuild pipeline rewrites them.
"""
function reset_solver_state!(𝓂::ℳ)
    𝓂.caches.valid_for = valid_for_caches()
    empty!(𝓂.caches.solver)

    # Reset size-dependent cache matrices so downstream code reallocates for
    # the (potentially) new model dimensions.
    𝓂.caches.jacobian = zeros(0, 0)
    𝓂.caches.jacobian_parameters = zeros(0, 0)
    𝓂.caches.jacobian_SS_and_pars = zeros(0, 0)
    𝓂.caches.hessian = zeros(0, 0)
    𝓂.caches.hessian_parameters = zeros(0, 0)
    𝓂.caches.hessian_SS_and_pars = zeros(0, 0)
    𝓂.caches.third_order_derivatives = zeros(0, 0)
    𝓂.caches.third_order_derivatives_parameters = zeros(0, 0)
    𝓂.caches.third_order_derivatives_SS_and_pars = zeros(0, 0)
    𝓂.caches.first_order_solution_matrix = zeros(0, 0)
    𝓂.caches.first_order_obc_solution_matrix = zeros(0, 0)
    𝓂.caches.qme_solution = zeros(0, 0)
    𝓂.caches.second_order_stochastic_steady_state = Float64[]
    𝓂.caches.second_order_solution = SparseMatrixCSC{Float64, Int64}(ℒ.I, 0, 0)
    𝓂.caches.pruned_second_order_stochastic_steady_state = Float64[]
    𝓂.caches.third_order_stochastic_steady_state = Float64[]
    𝓂.caches.third_order_solution = SparseMatrixCSC{Float64, Int64}(ℒ.I, 0, 0)
    𝓂.caches.pruned_third_order_stochastic_steady_state = Float64[]
    𝓂.caches.non_stochastic_steady_state = Float64[]
    𝓂.caches.covariance_first_order = zeros(0, 0)
    𝓂.caches.covariance_second_order = zeros(0, 0)
    𝓂.caches.covariance_third_order = zeros(0, 0)
    𝓂.caches.covariance_third_order_autocorr = zeros(0, 0)
    𝓂.caches.has_unit_roots = false

    𝓂.functions.functions_written = false

    # Invalidate derived caches that depend on the equations / calibration.
    # Empty axes / name tables are the sentinel used by the `ensure_*!`
    # helpers to decide whether to recompute.
    𝓂.constants.post_complete_parameters = update_post_complete_parameters(
        𝓂.constants.post_complete_parameters;
        var_axis = Symbol[],
        calib_axis = Symbol[],
        exo_axis_plain = Symbol[],
        exo_axis_with_subscript = Symbol[],
        full_NSSS_display = Symbol[],
        SS_and_pars_names = Symbol[],
        initialized = false,
    )

    # Reset the workspace buffers, forcing `ensure_*!` helpers to resize them
    # on next use.
    𝓂.workspaces = Workspaces()

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


"""
    finalize_model_update!(𝓂; verbose, silent)

Internal helper that finalises a model update by rebuilding the steady-state
solver and symbolic derivatives for the current `𝓂` state. Called by both
`reprocess_model_equations!` and `reprocess_calibration_equations!`.
"""
function finalize_model_update!(𝓂::ℳ; verbose::Bool = false, silent::Bool = true)
    has_missing_parameters = !isempty(𝓂.constants.post_complete_parameters.missing_parameters)
    missing_params = 𝓂.constants.post_complete_parameters.missing_parameters

    if !isnothing(𝓂.functions.NSSS_custom)
        write_ss_check_function!(𝓂)
    else
        if !has_missing_parameters
            set_up_steady_state_solver!(
                𝓂;
                verbose = verbose,
                silent = silent,
                ss_symbolic_mode = 𝓂.constants.post_parameters_macro.ss_symbolic_mode,
            )
        end
    end

    if !has_missing_parameters
        opts = merge_calculation_options(verbose = verbose)
        solve_steady_state!(
            𝓂,
            opts,
            𝓂.constants.post_parameters_macro.ss_solver_parameters_algorithm,
            𝓂.constants.post_parameters_macro.ss_solver_parameters_maxtime;
            silent = silent,
        )
        write_symbolic_derivatives!(𝓂; perturbation_order = 1, silent = silent)
        𝓂.functions.functions_written = true
    else
        if !silent
            @warn "Model has been set up with incomplete parameter definitions. Missing parameters: $(missing_params). The non-stochastic steady state and perturbation solution cannot be computed until all parameters are defined."
        end
    end

    return nothing
end


"""
    reprocess_model_equations!(𝓂, new_equations; parameters, verbose, silent)

Rebuild the model from an updated equation list while preserving the
revision history. Equivalent to re-running `@model` and `@parameters` on the
current parameter state, but without re-evaluating the macros.
"""
function reprocess_model_equations!(𝓂::ℳ,
                                    new_equations::Vector{Expr};
                                    parameters::ParameterType = nothing,
                                    verbose::Bool = false,
                                    silent::Bool = true)
    if parameters !== nothing
        write_parameters_input!(𝓂, parameters, verbose = verbose)
    end

    updated_block = Expr(:block, new_equations...)
    parameter_block = reconstruct_parameter_block(𝓂)

    T, equations_struct, ℂ, 𝓦 = process_model_equations(
        updated_block,
        𝓂.constants.post_model_macro.max_obc_horizon,
        𝓂.constants.post_parameters_macro.precompile,
    )

    𝓂.constants = ℂ
    𝓂.workspaces = 𝓦
    𝓂.equations = equations_struct

    reset_solver_state!(𝓂)

    parsed_parameters = process_parameter_definitions(parameter_block, 𝓂.constants.post_model_macro)

    𝓂.constants.post_parameters_macro = update_post_parameters_macro(
        𝓂.constants.post_parameters_macro;
        parameters_as_function_of_parameters = parsed_parameters.calib_parameters_no_var,
        ss_calib_list = parsed_parameters.ss_calib_list,
        par_calib_list = parsed_parameters.par_calib_list,
        bounds = parsed_parameters.bounds,
    )

    𝓂.equations.calibration = parsed_parameters.equations.calibration
    𝓂.equations.calibration_no_var = parsed_parameters.equations.calibration_no_var
    𝓂.equations.calibration_parameters = parsed_parameters.equations.calibration_parameters
    𝓂.equations.calibration_original = parsed_parameters.equations.calibration_original

    𝓂.constants.post_complete_parameters = update_post_complete_parameters(
        𝓂.constants.post_complete_parameters;
        parameters = parsed_parameters.parameters,
        missing_parameters = parsed_parameters.missing_parameters,
    )

    𝓂.parameter_values = parsed_parameters.parameter_values

    finalize_model_update!(𝓂; verbose = verbose, silent = silent)

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

    parameter_block = reconstruct_parameter_block(
        𝓂;
        calibration_original_override = updated_calibration_original,
        parameter_overrides = parameter_overrides,
    )

    parsed_parameters = process_parameter_definitions(parameter_block, 𝓂.constants.post_model_macro)

    𝓂.constants.post_parameters_macro = update_post_parameters_macro(
        𝓂.constants.post_parameters_macro;
        parameters_as_function_of_parameters = parsed_parameters.calib_parameters_no_var,
        ss_calib_list = parsed_parameters.ss_calib_list,
        par_calib_list = parsed_parameters.par_calib_list,
        bounds = parsed_parameters.bounds,
    )

    𝓂.equations.calibration = parsed_parameters.equations.calibration
    𝓂.equations.calibration_no_var = parsed_parameters.equations.calibration_no_var
    𝓂.equations.calibration_parameters = parsed_parameters.equations.calibration_parameters
    𝓂.equations.calibration_original = parsed_parameters.equations.calibration_original

    𝓂.constants.post_complete_parameters = update_post_complete_parameters(
        𝓂.constants.post_complete_parameters;
        parameters = parsed_parameters.parameters,
        missing_parameters = parsed_parameters.missing_parameters,
    )

    𝓂.parameter_values = parsed_parameters.parameter_values

    reset_solver_state!(𝓂)

    finalize_model_update!(𝓂; verbose = verbose, silent = silent)

    return nothing
end
