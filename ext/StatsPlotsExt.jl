module StatsPlotsExt

using MacroModelling

import MacroModelling: ParameterType, ℳ, Symbol_input, String_input, Tolerances, NsssTolerances, SolverTolerances, merge_calculation_options, MODEL®, DATA®, PARAMETERS®, ALGORITHM®, FILTER®, VARIABLES®, SMOOTH®, SHOW_PLOTS®, SAVE_PLOTS®, SAVE_PLOTS_NAME®, SAVE_PLOTS_FORMAT®, SAVE_PLOTS_PATH®, PLOTS_PER_PAGE®, MAX_ELEMENTS_PER_LEGENDS_ROW®, EXTRA_LEGEND_SPACE®, PLOT_ATTRIBUTES®, QME®, SYLVESTER®, LYAPUNOV®, TOLERANCES®, VERBOSE®, DATA_IN_LEVELS®, PERIODS®, SHOCKS®, SHOCK_SIZE®, NEGATIVE_SHOCK®, GENERALISED_IRF®, GENERALISED_IRF_WARMUP_ITERATIONS®, CONDITIONS_IN_LEVELS®, GENERALISED_IRF_DRAWS®, INITIAL_STATE®, IGNORE_OBC®, CONDITIONS®, SHOCK_CONDITIONS®, LEVELS®, LABEL®, RENAME_DICTIONARY®, STEADY_STATE_FUNCTION®, parse_shocks_input_to_index, parse_variables_input_to_index, replace_indices, replace_indices_special, filter_data_with_model, get_relevant_steady_states, replace_indices_in_symbol, parse_algorithm_to_state_update, girf, decompose_name, obc_objective_optim_fun, obc_constraint_optim_fun, compute_irf_responses, process_ignore_obc_flag, adjust_generalised_irf_flag, process_shocks_input, normalize_filtering_options, normalize_presample_periods, trim_informative_sample, adjust_initial_state, SteadyStateFunctionType
import MacroModelling: DEFAULT_CACHING, DEFAULT_USE_WORKSPACES, DEFAULT_ALGORITHM, DEFAULT_FILTER_SELECTOR, DEFAULT_WARMUP_ITERATIONS, DEFAULT_VARIABLES_EXCLUDING_OBC, DEFAULT_SHOCK_SELECTION, DEFAULT_PRESAMPLE_PERIODS, DEFAULT_DATA_IN_LEVELS, DEFAULT_SHOCK_DECOMPOSITION_SELECTOR, DEFAULT_SMOOTH_SELECTOR, DEFAULT_LABEL, DEFAULT_SHOW_PLOTS, DEFAULT_SAVE_PLOTS, DEFAULT_SAVE_PLOTS_FORMAT, DEFAULT_SAVE_PLOTS_PATH, DEFAULT_PLOTS_PER_PAGE_SMALL, DEFAULT_TRANSPARENCY, DEFAULT_MAX_ELEMENTS_PER_LEGEND_ROW, DEFAULT_EXTRA_LEGEND_SPACE, DEFAULT_VERBOSE, DEFAULT_QME_ALGORITHM, DEFAULT_SYLVESTER_SELECTOR, DEFAULT_SYLVESTER_THRESHOLD, DEFAULT_LARGE_SYLVESTER_ALGORITHM, DEFAULT_SYLVESTER_ALGORITHM, DEFAULT_LYAPUNOV_ALGORITHM, DEFAULT_PLOT_ATTRIBUTES, DEFAULT_ARGS_AND_KWARGS_NAMES, DEFAULT_PLOTS_PER_PAGE_LARGE, DEFAULT_SHOCKS_EXCLUDING_OBC, DEFAULT_VARIABLES_EXCLUDING_AUX_AND_OBC, DEFAULT_PERIODS, DEFAULT_SHOCK_SIZE, DEFAULT_NEGATIVE_SHOCK, DEFAULT_GENERALISED_IRF, DEFAULT_GENERALISED_IRF_WARMUP, DEFAULT_GENERALISED_IRF_DRAWS, DEFAULT_INITIAL_STATE, DEFAULT_IGNORE_OBC, DEFAULT_PLOT_TYPE, DEFAULT_CONDITIONS_IN_LEVELS, DEFAULT_SIGMA_RANGE, DEFAULT_FONT_SIZE, DEFAULT_VARIABLE_SELECTION, DEFAULT_FORECAST_PERIODS
import DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF, TYPEDSIGNATURES, TYPEDFIELDS
import Random
import LaTeXStrings

const irf_active_plot_container = Dict[]
const conditional_forecast_active_plot_container = Dict[]
const model_estimates_active_plot_container = Dict[]
const solution_active_plot_container = Dict[]

import StatsPlots
import Showoff
import DataStructures: OrderedSet
import SparseArrays: SparseMatrixCSC
import NLopt
import Dates
using DispatchDoctor

import MacroModelling: plot_irfs, plot_irf, plot_IRF, plot_simulations, plot_simulation, plot_solution, plot_girf, plot_conditional_forecast, plot_conditional_variance_decomposition, plot_forecast_error_variance_decomposition, plot_fevd, plot_model_estimates, plot_shock_decomposition, plotlyjs_backend, gr_backend, get_irf

import MacroModelling: plot_irfs!, plot_irf!, plot_IRF!, plot_girf!, plot_simulations!, plot_simulation!, plot_conditional_forecast!, plot_model_estimates!, plot_solution!


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions moved from core (only used by the plotting extension)
# ──────────────────────────────────────────────────────────────────────────────

const HIGHER_ORDER_ALGORITHMS = (:second_order, :pruned_second_order, :third_order, :pruned_third_order)
const THIRD_ORDER_ALGORITHMS  = (:third_order, :pruned_third_order)

const TOL_DISPLAY_NAMES = Dict{Symbol,String}(
    :tol => "Tolerance",
    :nsss => "NSSS",
    :first_order => "1st order",
    :second_order => "2nd order",
    :third_order => "3rd order",
    :qme => "QME",
    :sylvester => "Sylvester",
    :lyapunov => "Lyapunov",
    :atol => "atol",
    :rtol => "rtol",
    :initial_guess_acceptance_tol => "init. guess accept. tol",
    :acceptance_tol => "accept. tol",
    :xtol => "xtol",
    :ftol => "ftol",
    :rel_xtol => "rel. xtol",
    :droptol => "droptol",
    :dependencies_tol => "dep. tol",
)

function infer_step(x_axis::AbstractVector{T}) where {T<:Number}
    x_axis[end] - x_axis[end-1]
end

function infer_step(x_axis::AbstractVector{T}) where {T<:Dates.TimeType}
    d1 = x_axis[end-1]
    d2 = x_axis[end]

    # try to infer a monthly step if aligned by day-of-month
    if Dates.day(d1) == Dates.day(d2)
        m1 = 12 * Dates.year(d1) + Dates.month(d1)
        m2 = 12 * Dates.year(d2) + Dates.month(d2)
        mstep = m2 - m1
        if mstep != 0
            return Dates.Month(mstep)
        end
    end

    # fall back to the raw difference (in days, milliseconds, …)
    return d2 - d1
end

function apply_custom_name(symbol::R, custom_names::AbstractDict{S, T})::R where {R <: Union{Symbol, String}, S, T}
    # First, check for an exact match with the original symbol
    if haskey(custom_names, symbol)
        return R(custom_names[symbol])
    end
    
    # Handle cross-type check for exact match (String vs Symbol)
    if symbol isa Symbol && haskey(custom_names, String(replace_indices_in_symbol(symbol)))
        return R(custom_names[String(replace_indices_in_symbol(symbol))])
    elseif symbol isa String && haskey(custom_names, Symbol(symbol))
        return R(custom_names[Symbol(symbol)])
    end

    # If no exact match, strip lag operators and compare base names.
    s_str = string(symbol)
    lag_regex = r"^(.*)(ᴸ⁽.*⁾)$"
    m = match(lag_regex, s_str)

    base_symbol_str, lag_part = if m !== nothing
        (m.captures[1], m.captures[2])
    else
        (s_str, "")
    end

    for (key, value) in custom_names
        key_str = string(key)
        key_m = match(lag_regex, key_str)
        
        base_key_str = if key_m !== nothing
            key_m.captures[1]
        else
            key_str
        end

        if base_key_str == base_symbol_str
            return R(string(value) * lag_part)
        end
    end

    return symbol
end

function normalize_superscript(x::Symbol)
    return normalize_superscript(string(x))
end

function normalize_superscript(x::AbstractString)
    sub_map = Dict(
        '₀' => '0', '₁' => '1', '₂' => '2', '₃' => '3', '₄' => '4',
        '₅' => '5', '₆' => '6', '₇' => '7', '₈' => '8', '₉' => '9',
        '₊' => '+', '₋' => '-', '₌' => '=', '₍' => '(', '₎' => ')',
        'ₐ' => 'a', 'ₑ' => 'e', 'ₕ' => 'h', 'ᵢ' => 'i', 'ⱼ' => 'j',
        'ₖ' => 'k', 'ₗ' => 'l', 'ₘ' => 'm', 'ₙ' => 'n', 'ₒ' => 'o',
        'ₚ' => 'p', 'ᵣ' => 'r', 'ₛ' => 's', 'ₜ' => 't', 'ᵤ' => 'u',
        'ᵥ' => 'v', 'ₓ' => 'x'
    )
    super_map = Dict(
        '⁰' => '0', '¹' => '1', '²' => '2', '³' => '3', '⁴' => '4',
        '⁵' => '5', '⁶' => '6', '⁷' => '7', '⁸' => '8', '⁹' => '9',
        '⁺' => '+', '⁻' => '-', '⁼' => '=', '⁽' => '(', '⁾' => ')',
        'ᵃ' => 'a', 'ᵇ' => 'b', 'ᶜ' => 'c', 'ᵈ' => 'd', 'ᵉ' => 'e',
        'ᶠ' => 'f', 'ᵍ' => 'g', 'ʰ' => 'h', 'ᶦ' => 'i', 'ʲ' => 'j',
        'ᵏ' => 'k', 'ˡ' => 'l', 'ᵐ' => 'm', 'ⁿ' => 'n', 'ᵒ' => 'o',
        'ᵖ' => 'p', 'ʳ' => 'r', 'ˢ' => 's', 'ᵗ' => 't', 'ᵘ' => 'u',
        'ᵛ' => 'v', 'ʷ' => 'w', 'ˣ' => 'x', 'ʸ' => 'y', 'ᶻ' => 'z'
    )

    buf = IOBuffer()
    for c in x
        if haskey(sub_map, c)
            write(buf, sub_map[c])
        elseif haskey(super_map, c)
            write(buf, super_map[c])
        else
            write(buf, c)
        end
    end
    return String(take!(buf))
end

function solver_tol_to_dict(st::SolverTolerances)
    return Dict{Symbol,Any}(
        :atol => st.atol,
        :rtol => st.rtol,
        :initial_guess_acceptance_tol => st.initial_guess_acceptance_tol,
        :acceptance_tol => st.acceptance_tol,
    )
end

function nsss_tol_to_dict(nt::NsssTolerances)
    return Dict{Symbol,Any}(
        :acceptance_tol => nt.acceptance_tol,
        :initial_guess_acceptance_tol => nt.initial_guess_acceptance_tol,
        :xtol => nt.xtol,
        :ftol => nt.ftol,
        :rel_xtol => nt.rel_xtol,
    )
end

function tol_to_dict(tol::Tolerances, algorithm::Symbol; needs_covariance::Bool = false)
    d = Dict{Symbol,Any}()

    # NSSS — always relevant
    d[:nsss] = nsss_tol_to_dict(tol.nsss)

    # First-order — always relevant
    fo = Dict{Symbol,Any}(:qme => solver_tol_to_dict(tol.first_order.qme),
                           :droptol => tol.first_order.droptol)
    if needs_covariance
        fo[:lyapunov] = solver_tol_to_dict(tol.first_order.lyapunov)
        fo[:dependencies_tol] = tol.first_order.dependencies_tol
    end
    d[:first_order] = fo

    # Second-order — only for higher-order algorithms
    if algorithm in HIGHER_ORDER_ALGORITHMS
        so = Dict{Symbol,Any}(:sylvester => solver_tol_to_dict(tol.second_order.sylvester),
                               :droptol => tol.second_order.droptol)
        if needs_covariance
            so[:lyapunov] = solver_tol_to_dict(tol.second_order.lyapunov)
            so[:dependencies_tol] = tol.second_order.dependencies_tol
        end
        d[:second_order] = so
    end

    # Third-order — only for third-order algorithms
    if algorithm in THIRD_ORDER_ALGORITHMS
        to = Dict{Symbol,Any}(:sylvester => solver_tol_to_dict(tol.third_order.sylvester),
                               :droptol => tol.third_order.droptol)
        if needs_covariance
            to[:lyapunov] = solver_tol_to_dict(tol.third_order.lyapunov)
            to[:dependencies_tol] = tol.third_order.dependencies_tol
        end
        d[:third_order] = to
    end

    return d
end

function warn_irrelevant_tol(tol::Tolerances, algorithm::Symbol; needs_covariance::Bool = false)
    defaults = Tolerances()

    # --- order-based irrelevance ---
    if algorithm ∉ HIGHER_ORDER_ALGORITHMS
        if tol.second_order != defaults.second_order
            @info "Second-order tolerances have no effect with algorithm = :$algorithm and are ignored."
        end
    end

    if algorithm ∉ THIRD_ORDER_ALGORITHMS
        if tol.third_order != defaults.third_order
            @info "Third-order tolerances have no effect with algorithm = :$algorithm and are ignored."
        end
    end

    # --- covariance-based irrelevance ---
    if !needs_covariance
        if tol.first_order.lyapunov != defaults.first_order.lyapunov ||
           tol.first_order.dependencies_tol != defaults.first_order.dependencies_tol
            @info "First-order Lyapunov/dependencies tolerances have no effect without covariance computation (current operation does not require it) and are ignored."
        end

        if algorithm in HIGHER_ORDER_ALGORITHMS
            if tol.second_order.lyapunov != defaults.second_order.lyapunov ||
               tol.second_order.dependencies_tol != defaults.second_order.dependencies_tol
                @info "Second-order Lyapunov/dependencies tolerances have no effect without covariance computation (current operation does not require it) and are ignored."
            end
        end

        if algorithm in THIRD_ORDER_ALGORITHMS
            if tol.third_order.lyapunov != defaults.third_order.lyapunov ||
               tol.third_order.dependencies_tol != defaults.third_order.dependencies_tol
                @info "Third-order Lyapunov/dependencies tolerances have no effect without covariance computation (current operation does not require it) and are ignored."
            end
        end
    end
end

function flatten_tol_dict(d::Dict;
                           names::Dict{Symbol,String} = TOL_DISPLAY_NAMES,
                           prefix::String = "")
    result = Dict{String,Any}()
    for (k, v) in d
        seg = get(names, k, String(k))
        label = isempty(prefix) ? seg : prefix * " " * seg
        if v isa Dict
            merge!(result, flatten_tol_dict(v; names = names, prefix = label))
        else
            result[label] = v
        end
    end
    return result
end

function compare_args_and_kwargs(dicts::Vector{S}) where S <: Dict
    N = length(dicts)

    if N ≤ 1
        diffs = Dict{Symbol,Any}()
        if N == 1
            for k in keys(dicts[1])
                k in (:plot_data, :plot_type) && continue
                v = dicts[1][k]
                if v isa Dict
                    diffs[k] = compare_args_and_kwargs([v])
                else
                    diffs[k] = [v]
                end
            end
        end
        return diffs
    end

    diffs = Dict{Symbol,Any}()

    all_keys = reduce(union, keys.(dicts))

    for k in all_keys
        if k in [:plot_data, :plot_type]
            continue
        end

        if !all(haskey(d, k) for d in dicts)
            diffs[k] = [get(d, k, missing) for d in dicts]
            continue
        end

        vals = [d[k] for d in dicts]

        if all(v -> v isa Dict, vals)
            nested = compare_args_and_kwargs(vals)
            if !isempty(nested)
                diffs[k] = nested
            end

        elseif all(v -> v isa KeyedArray, vals)
            base = vals[1]
            identical = all(v -> length(v) == length(base) && all(collect(v) .== collect(base)), vals[2:end])
            if !identical
                diffs[k] = vals
            end

        elseif all(v -> v isa AbstractArray, vals)
            base = vals[1]
            identical = all(v -> length(v) == length(base) && all(v .== base), vals[2:end])
            if !identical
                diffs[k] = vals
            end

        else
            identical = all(v -> v == vals[1], vals[2:end])
            if !identical
                diffs[k] = vals
            end
        end
    end

    return diffs
end

function flatten_tol_diff(diff::Dict;
                          names::Dict{Symbol,String} = TOL_DISPLAY_NAMES,
                          prefix::String = "")
    result = Pair{String,Any}[]
    for (k, v) in sort(collect(diff), by = first)
        seg = get(names, k, String(k))
        label = isempty(prefix) ? seg : prefix * " " * seg
        if v isa Dict
            append!(result, flatten_tol_diff(v; names = names, prefix = label))
        else
            push!(result, label => reduce(vcat, v))
        end
    end
    return result
end

function setup_plot_attributes(plot_attributes::Dict)
    gr_back = StatsPlots.backend() == StatsPlots.Plots.GRBackend()
    attrbts = !gr_back ? merge(DEFAULT_PLOT_ATTRIBUTES, Dict(:framestyle => :box)) : merge(DEFAULT_PLOT_ATTRIBUTES, Dict())
    attributes = merge(attrbts, plot_attributes)
    attributes_redux = copy(attributes)
    delete!(attributes_redux, :framestyle)
    return gr_back, attributes, attributes_redux
end

function build_extended_palette(attributes_redux::Dict; total_pal_len::Int = 100, alpha_reduction_factor::Float64 = 0.7)
    orig_pal = StatsPlots.palette(attributes_redux[:palette])
    mapreduce(x -> StatsPlots.coloralpha.(orig_pal, alpha_reduction_factor ^ x), vcat, 0:(total_pal_len ÷ length(orig_pal)) - 1) |> StatsPlots.palette
end

function process_rename_dictionary(rename_dictionary::AbstractDict, 𝓂::ℳ)
    relevant_keys = [k for k in keys(rename_dictionary) if (k isa String ? replace_indices(k) : k) in vcat(𝓂.constants.post_model_macro.var, 𝓂.constants.post_model_macro.exo)] |> sort
    processed = Any[]
    for k in relevant_keys
        push!(processed, k => rename_dictionary[k])
    end
    return processed
end

function compute_diffdict(container::Vector{Dict}, ref_keys; include_label_in_reduced::Bool = true)
    label_keys = include_label_in_reduced ? [:run_id, :label] : [:run_id]
    reduced_vector = [
        Dict(k => d[k] for k in vcat(label_keys..., keys(DEFAULT_ARGS_AND_KWARGS_NAMES)...) if haskey(d, k))
        for d in container
    ]

    diffdict = compare_args_and_kwargs(reduced_vector)

    grouped_by_model = Dict{Any, Vector{Dict}}()

    for d in container
        model = d[:model_name]
        d_sub = Dict(k => d[k] for k in setdiff(ref_keys, keys(DEFAULT_ARGS_AND_KWARGS_NAMES), [:tol]) if haskey(d, k))
        push!(get!(grouped_by_model, model, Vector{Dict}()), d_sub)
    end

    model_names = unique([d[:model_name] for d in container])

    for model in model_names
        if length(grouped_by_model[model]) > 1
            diffdict_grouped = compare_args_and_kwargs(grouped_by_model[model])
            diffdict = merge_by_runid(diffdict, diffdict_grouped)
        end
    end

    return diffdict
end

function annotate_param_diff!(annotate_diff_input, diffdict)
    if haskey(diffdict, :parameters)
        param_nms = diffdict[:parameters] |> keys |> collect |> sort
        for param in param_nms
            result = [x === nothing ? "" : x for x in diffdict[:parameters][param]]
            push!(annotate_diff_input, String(param) => result)
        end
    end
end

function annotate_rename_dict_diff!(annotate_diff_input, diffdict)
    if haskey(diffdict, :rename_dictionary)
        non_nothing_dicts = [d for d in diffdict[:rename_dictionary] if !isnothing(d) && length(d) > 0]
        unique_dicts = unique(non_nothing_dicts)
        rename_idx = Int[]

        for init in diffdict[:rename_dictionary]
            if isnothing(init) || length(init) == 0
                push!(rename_idx, 0)
                continue
            end

            for (i,u) in enumerate(unique_dicts)
                if u == init
                    push!(rename_idx, i)
                    continue
                end
            end
        end

        push!(annotate_diff_input, "Rename dictionary" => [i > 0 ? "#$i" : "nothing" for i in rename_idx])
    end
end

function annotate_tol_diff!(annotate_diff_input, container)
    if length(container) > 1
        flat_tols = [flatten_tol_dict(d[:tol]) for d in container]
        shared_tol_keys = reduce(intersect, keys.(flat_tols))
        for fk in sort(collect(shared_tol_keys))
            fvals = [ft[fk] for ft in flat_tols]
            if !all(v -> v == fvals[1], fvals[2:end])
                push!(annotate_diff_input, fk => fvals)
            end
        end
    end
end

function should_use_label_switch(annotate_diff_input, container)
    ((length(annotate_diff_input) > 2) || (Dict(annotate_diff_input)["Plot label"] != collect(1:length(container)))) && length(container) > 1
end

function push_if_no_duplicate!(container, args_and_kwargs, specific_keys; collect_compare_keys = Symbol[])
    no_duplicate = all(
        !(all((
            all(get(dict, k, nothing) == args_and_kwargs[k] for k in specific_keys),
            all(
                k in collect_compare_keys ?
                    collect(get(dict, k, nothing)) == collect(get(args_and_kwargs, k, nothing)) :
                    get(dict, k, nothing) == get(args_and_kwargs, k, nothing)
                for k in setdiff(keys(DEFAULT_ARGS_AND_KWARGS_NAMES), [:label])
            )
        )))
        for dict in container
    )

    if no_duplicate
        push!(container, args_and_kwargs)
    else
        @info "Plot with same parameters already exists. Using previous plot data to create plot."
    end
end

function check_and_remove_duplicate!(container, specific_keys)
    if length(container) > 1
        ref = container[end]
        no_duplicate = all(
            !(all((
                all(get(dict, k, nothing) == ref[k] for k in specific_keys),
                all(get(dict, k, nothing) == get(ref, k, nothing) for k in setdiff(keys(DEFAULT_ARGS_AND_KWARGS_NAMES), [:label]))
            )))
            for dict in container[1:end-1]
        )

        if !no_duplicate
            @info "Plot with same parameters already exists. Using previous plot data to create plot."
            pop!(container)
        end
    end
end

function annotate_default_kwarg_diffs!(annotate_diff_input, args_and_kwargs, diffdict, exclude_keys)
    has_shock_direction_diff = false
    for k in setdiff(keys(args_and_kwargs), exclude_keys)
        if haskey(diffdict, k)
            push!(annotate_diff_input, DEFAULT_ARGS_AND_KWARGS_NAMES[k] => reduce(vcat, diffdict[k]))
            if k == :negative_shock
                has_shock_direction_diff = true
            end
        end
    end
    return has_shock_direction_diff
end

function assemble_and_emit_page!(
    return_plots, pp, legend_plot,
    annotate_diff_input, diffdict,
    attributes, attributes_redux,
    pane, n_subplots, plots_per_page,
    show_plots, save_plots, save_plots_path, save_plots_name, save_plots_format,
    default_model_name;
    title_extra::String = "",
    filename_extra::String = "",
    legend_height = length(annotate_diff_input),
    show_diff_table::Bool = false,
    annotate_ss = nothing,
    annotate_ss_page = nothing,
    plt_lab_switch::Bool = false,
    is_tail::Bool = false
)
    ppp = StatsPlots.plot(pp...; attributes...)

    if haskey(diffdict, :model_name)
        model_string = "multiple models"
        model_string_filename = "multiple_models"
    else
        model_string = string(default_model_name)
        model_string_filename = string(default_model_name)
    end

    plot_title = "Model: " * model_string * title_extra * "  (" * string(pane) * "/" * string(Int(ceil(n_subplots / plots_per_page))) * ")"

    plot_elements = [ppp, legend_plot]
    layout_heights = [15, legend_height]

    if annotate_ss !== nothing && annotate_ss_page !== nothing
        if plt_lab_switch
            annotate_diff_input_plot = plot_df(annotate_diff_input; fontsize = attributes[:annotationfontsize], title = "Relevant Input Differences")
            ppp_input_diff = StatsPlots.plot(annotate_diff_input_plot; attributes..., framestyle = :box)
            push!(plot_elements, ppp_input_diff)
            push!(layout_heights, 5)
            pushfirst!(annotate_ss_page, "Plot label" => reduce(vcat, diffdict[:label]))
        else
            pushfirst!(annotate_ss_page, annotate_diff_input[2][1] => annotate_diff_input[2][2])
        end

        push!(annotate_ss, annotate_ss_page)

        if length(annotate_ss[pane]) > 1
            annotate_ss_plot = plot_df(annotate_ss[pane]; fontsize = attributes[:annotationfontsize], title = "Relevant Steady States")
            ppp_ss = StatsPlots.plot(annotate_ss_plot; attributes..., framestyle = :box)
            push!(plot_elements, ppp_ss)
            push!(layout_heights, 5)
        end
    else
        if show_diff_table
            annotate_diff_input_plot = plot_df(annotate_diff_input; fontsize = attributes[:annotationfontsize], title = "Relevant Input Differences")
            ppp_input_diff = StatsPlots.plot(annotate_diff_input_plot; attributes..., framestyle = :box)
            push!(plot_elements, ppp_input_diff)
            push!(layout_heights, 5)
        end
    end

    p = StatsPlots.plot(plot_elements...,
                        layout = StatsPlots.grid(length(layout_heights), 1, heights = layout_heights ./ sum(layout_heights)),
                        plot_title = plot_title;
                        attributes_redux...)

    push!(return_plots, p)

    if show_plots
        display(p)
    end

    if save_plots
        if !isdir(save_plots_path) mkpath(save_plots_path) end
        fn = save_plots_path * "/" * string(save_plots_name) * "__" * model_string_filename
        if !isempty(filename_extra)
            fn *= "__" * string(filename_extra)
        end
        fn *= "__" * string(pane) * "." * string(save_plots_format)
        StatsPlots.savefig(p, fn)
    end

    if !is_tail
        pane += 1
        empty!(pp)
        if annotate_ss_page !== nothing
            empty!(annotate_ss_page)
        end
    end

    return pane
end

@stable default_mode = "disable" begin

"""
    gr_backend()
Renaming and reexport of StatsPlots function `gr()` to define GR.jl as backend.

# Returns
- `StatsPlots.GRBackend`: backend instance.
"""
gr_backend(args...; kwargs...) = StatsPlots.gr(args...; kwargs...)



"""
    plotlyjs_backend()
Renaming and reexport of StatsPlots function `plotlyjs()` to define PlotlyJS.jl as backend.

# Returns
- `StatsPlots.PlotlyJSBackend`: backend instance.
"""
plotlyjs_backend(args...; kwargs...) = StatsPlots.plotlyjs(args...; kwargs...)



"""
$(SIGNATURES)
Plot model estimates of the variables given the data. The default plot shows the estimated variables, shocks, the data underlying the estimates, and an unconditional forecast extending beyond the last data period. The estimates are based on the Kalman smoother or filter (depending on the `smooth` keyword argument) or inversion filter using the provided data and solution of the model. The unconditional forecast (shown as a dashed line) displays the model's expected path absent any exongeos shocks starting from the final filtered state.

The left axis shows the level, and the right the deviation from the relevant steady state. The non-stochastic steady state (NSSS) is relevant for first order solutions and the stochastic steady state for higher order solutions. The horizontal black line indicates the relevant steady state. Variable names are above the subplots and the title provides information about the model, shocks, and number of pages per shock.
In case `shock_decomposition = true`, the plot shows the variables, shocks, and data in absolute deviations from the relevant steady state as a stacked bar chart per period.

For higher order perturbation solutions the decomposition additionally contains a term `Nonlinearities`. This term represents the nonlinear interaction between the states in the periods after the shocks arrived and in the case of pruned third order, the interaction between (pruned second order) states and contemporaneous shocks. Setting `marginal_contribution = true` (only meaningful for `:pruned_second_order` and `:pruned_third_order` together with `shock_decomposition = true`) instead allocates this cross-shock interaction across shocks via marginal contributions (Shapley values) and omits the `Nonlinearities` bar.

If occasionally binding constraints are present in the model, they are not taken into account here.

# Arguments
- $MODEL®
- $DATA®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $FILTER®
$MacroModelling.PARTICLE_FILTER_KEYWORDS®
- $(VARIABLES®(DEFAULT_VARIABLES_EXCLUDING_OBC))
- `shocks` [Default: `:all`]: shocks for which to plot the estimates in the respective subplots and in the shock decompositions. Inputs can be either a `Symbol` or `String` (e.g. `:eps_a`, `\"eps_a\"`, or `:all`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. `:all` selects all shocks in the model. `:none` selects no shocks in the model. If not all shocks are shown, the ommitted shocks will be summarised and netted under the label `Other shocks (net)` in the shock decomposition.
- `presample_periods` [Default: `0`, Type: `Int`]: number of initial retained-sample periods omitted from the plot. Useful when filtering the full sample while focusing on a later subperiod. Values above the retained sample length are clamped down automatically with an informational message.
- `forecast_periods` [Default: `$DEFAULT_FORECAST_PERIODS`, Type: `Int`]: number of periods of unconditional forecast to add after the last period of data. The forecast is shown as a dotted line to distinguish it from the model estimates.
- $DATA_IN_LEVELS®
- `shock_decomposition` [Default: `true` for algorithms supporting shock decompositions (`:first_order`, `:pruned_second_order`, `:pruned_third_order`), otherwise `false`, Type: `Bool`]: whether to show the contribution of the shocks to the deviations from NSSS for each variable. If `false`, the plot shows the values of the selected variables, data, and shocks. When an unsupported algorithm is chosen the argument automatically falls back to `false`.
- $SMOOTH®
- `marginal_contribution` [Default: `false`, Type: `Bool`]: if `true` and the algorithm is `:pruned_second_order` or `:pruned_third_order` with `shock_decomposition = true`, attribute the cross-shock interaction across shocks via marginal contributions (Shapley values) and omit the `Nonlinearities` bar.
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMAT®
- $SAVE_PLOTS_PATH®
- $(SAVE_PLOTS_NAME®("estimation"))
- $(PLOTS_PER_PAGE®(DEFAULT_PLOTS_PER_PAGE_SMALL))
- `transparency` [Default: `$DEFAULT_TRANSPARENCY`, Type: `Float64`]: transparency of stacked bars. Only relevant if `shock_decomposition` is `true`.
- $MAX_ELEMENTS_PER_LEGENDS_ROW®
- $EXTRA_LEGEND_SPACE®
- `label` [Default: `1`, Type: `Union{Real, String, Symbol}`]: label to attribute to this function call in the plots.
- $RENAME_DICTIONARY®
- $PLOT_ATTRIBUTES®
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Vector{Plot}` of individual plots

# Examples
```julia
using MacroModelling, StatsPlots


@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

simulation = simulate(RBC_CME)

plot_model_estimates(RBC_CME, simulation([:k],:,:simulate))
```
"""
plot_algorithm_label(algorithm::Symbol) = algorithm === :first_order ? "FO" :
    algorithm === :second_order ? "SO" :
    algorithm === :third_order ? "TO" :
    algorithm === :pruned_second_order ? "PS2" :
    algorithm === :pruned_third_order ? "PS3" : string(algorithm)

plot_filter_label(filter::Symbol) = filter === :inversion ? "inv" :
    filter === :quadratic_kalman ? "QKF" :
    filter === :cubic_kalman ? "CKF" :
    filter === :ivashchenko_kalman ? "Iva" :
    filter === :bootstrap_particle ? "boot" :
    filter === :auxiliary_particle ? "aux" :
    filter === :tempered_particle ? "temp" :
    filter === :guided_particle ? "guide" : string(filter)

function plot_model_estimates(𝓂::ℳ,
                                data::KeyedArray;
                                parameters::ParameterType = nothing,
                                steady_state_function::SteadyStateFunctionType = missing,
                                algorithm::Symbol = DEFAULT_ALGORITHM, 
                                filter::Symbol = DEFAULT_FILTER_SELECTOR(algorithm),
                                initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
                                measurement_error::Union{Symbol,Real,AbstractVector{<:Real},AbstractMatrix{<:Real}} = MacroModelling.DEFAULT_MEASUREMENT_ERROR,
                                n_particles::Int = MacroModelling.DEFAULT_N_PARTICLES,
                                particle_resampling::Symbol = MacroModelling.DEFAULT_PARTICLE_RESAMPLING,
                                particle_resampling_threshold::Real = MacroModelling.DEFAULT_PARTICLE_RESAMPLING_THRESHOLD,
                                particle_initial_state_scaling::Real = MacroModelling.DEFAULT_PARTICLE_INITIAL_STATE_SCALING,
                                particle_rng::Random.AbstractRNG = Random.default_rng(), 
                                particle_target_ratio::Real = MacroModelling.DEFAULT_PARTICLE_TARGET_RATIO,
                                particle_mh_steps::Int = MacroModelling.DEFAULT_TEMPERED_MH_STEPS,
                                particle_max_stages::Int = MacroModelling.DEFAULT_PARTICLE_MAX_STAGES,
                                particle_mh_scale::Real = MacroModelling.DEFAULT_PARTICLE_MH_SCALE,
                                warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS,
                                variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_OBC, 
                                shocks::Union{Symbol_input,String_input} = DEFAULT_SHOCK_SELECTION, 
                                presample_periods::Int = DEFAULT_PRESAMPLE_PERIODS,
                                forecast_periods::Int = DEFAULT_FORECAST_PERIODS,
                                data_in_levels::Bool = DEFAULT_DATA_IN_LEVELS,
                                shock_decomposition::Bool = DEFAULT_SHOCK_DECOMPOSITION_SELECTOR(algorithm),
                                smooth::Bool = DEFAULT_SMOOTH_SELECTOR(filter),
                                marginal_contribution::Bool = false,
                                label::Union{Real, String, Symbol} = DEFAULT_LABEL,
                                show_plots::Bool = DEFAULT_SHOW_PLOTS,
                                save_plots::Bool = DEFAULT_SAVE_PLOTS,
                                save_plots_format::Symbol = DEFAULT_SAVE_PLOTS_FORMAT,
                                save_plots_name::Union{String, Symbol} = "estimation",
                                save_plots_path::String = DEFAULT_SAVE_PLOTS_PATH,
                                plots_per_page::Int = DEFAULT_PLOTS_PER_PAGE_SMALL,
                                transparency::Float64 = DEFAULT_TRANSPARENCY,
                                max_elements_per_legend_row::Int = DEFAULT_MAX_ELEMENTS_PER_LEGEND_ROW,
                                extra_legend_space::Float64 = DEFAULT_EXTRA_LEGEND_SPACE,
                                rename_dictionary::AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}} = Dict{Symbol, String}(),
                                plot_attributes::Dict = Dict(),
                                verbose::Bool = DEFAULT_VERBOSE,
                                tol::Tolerances = Tolerances(),
                                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                                caching::Bool = DEFAULT_CACHING,
                                use_workspaces::Bool = DEFAULT_USE_WORKSPACES)
    # @nospecialize # reduce compile time                            

    if !caching invalidate_cache_validity!(𝓂) end
    orig_ws = 𝓂.workspaces
    if !use_workspaces 𝓂.workspaces = fresh_workspaces(orig_ws) end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                                    lyapunov_algorithm = lyapunov_algorithm)
    warn_irrelevant_tol(tol, algorithm; needs_covariance = filter == :kalman)

    gr_back, attributes, attributes_redux = setup_plot_attributes(plot_attributes)


    # write_parameters_input!(𝓂, parameters, verbose = verbose)

    filter, smooth, algorithm, shock_decomposition, pruning, warmup_iterations = normalize_filtering_options(filter, smooth, algorithm, shock_decomposition, warmup_iterations)

    if marginal_contribution && shock_decomposition && !pruning
        @info "`marginal_contribution = true` is only meaningful for pruned higher-order solutions (`:pruned_second_order`, `:pruned_third_order`). Setting `marginal_contribution = false` for `algorithm = $(algorithm)`." maxlog = 3
        marginal_contribution = false
    end
    mc = marginal_contribution && shock_decomposition && pruning
    is_pruned = pruning
    pruning = pruning && !mc
    # Compact title: solution order, filter, smoothing mode, attribution mode.
    # `F/S` means filtered/smoothed and `seq/AS` means sequential/Aumann–Shapley.
    smooth_label = smooth ? "S" : "F"
    attribution_label = mc ? "AS" : "seq"
    plot_method_label = string(plot_algorithm_label(algorithm), " · ",
                               plot_filter_label(filter), " · ",
                               smooth_label, " · ", attribution_label)

    solve!(𝓂, 
            parameters = parameters, 
            steady_state_function = steady_state_function,
            algorithm = algorithm, 
            opts = opts, 
            dynamics = true)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)

    data = data(sort(axiskeys(data,1)))
    
    obs_axis = collect(axiskeys(data,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    variables = variables isa String_input ? variables .|> Meta.parse .|> replace_indices : variables

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    if shocks ∈ [:simulate, :all_excluding_obc] 
        @warn "Shocks input cannot be `:all_excluding_obc`, or `:simulate` in `plot_model_estimates`. Changed shocks to `:all`"
        shocks = :all
    end

    obs_idx     = parse_variables_input_to_index(obs_symbols, 𝓂.constants) |> unique |> sort
    var_idx     = parse_variables_input_to_index(variables, 𝓂.constants) |> unique  |> sort
    shock_idx   = shocks == :none ? Int64[] : parse_shocks_input_to_index(shocks, 𝓂.constants)

    # Create display names and sort alphabetically
    variable_names_display = [replace_indices_in_symbol.(apply_custom_name(𝓂.constants.post_model_macro.var[v], rename_dictionary)) for v in var_idx]
    @assert length(variable_names_display) == length(unique(variable_names_display)) "Renaming variables resulted in non-unique names. Please check the `rename_dictionary`."
    var_sort_perm = sortperm(variable_names_display, by = normalize_superscript)
    var_idx = var_idx[var_sort_perm]
    variable_names_display = variable_names_display[var_sort_perm]

    shock_names_display = [replace_indices_in_symbol.(apply_custom_name(𝓂.constants.post_model_macro.exo[s], rename_dictionary)) * "₍ₓ₎" for s in shock_idx]
    @assert length(shock_names_display) == length(unique(shock_names_display)) "Renaming shocks resulted in non-unique names. Please check the `rename_dictionary`."
    if length(shock_idx) > 1
        shock_sort_perm = sortperm(shock_names_display, by = normalize_superscript)
        shock_idx = shock_idx[shock_sort_perm]
        shock_names_display = shock_names_display[shock_sort_perm]
    end
    
    processed_rename_dictionary = process_rename_dictionary(rename_dictionary, 𝓂)

    legend_columns = 1

    legend_items = length(shock_idx) + 3 + pruning + (forecast_periods > 0 ? 1 : 0) + (mc ? 1 : 0)

    max_columns = min(legend_items, max_elements_per_legend_row)
    
    # Try from max_columns down to 1 to find the optimal solution
    for cols in max_columns:-1:1
        if legend_items % cols == 0 || legend_items % cols <= max_elements_per_legend_row
            legend_columns = cols
            break
        end
    end

    if data_in_levels
        data_in_deviations = MacroModelling.missing_data_to_nan(data) .- NSSS[obs_idx]
    else
        data_in_deviations = MacroModelling.missing_data_to_nan(data)
    end

    data_in_deviations, _, _, informative_periods = trim_informative_sample(data_in_deviations;
                                                                            require_informative_periods = true)
    presample_periods = normalize_presample_periods(presample_periods, size(data_in_deviations, 2))

    x_axis = axiskeys(data,2)[informative_periods]

    extra_legend_space += length(string(x_axis[1])) > 6 ? .1 : 0.0

    periods = presample_periods+1:size(data_in_deviations,2)

    x_axis = x_axis[periods]
    
    extra_kw = mc ? (; marginal_contribution = true) : NamedTuple()
    if filter ∈ MacroModelling.PARTICLE_FILTERS
        extra_kw = merge(extra_kw, (; measurement_error = MacroModelling.resolve_measurement_error(filter, measurement_error, data_in_deviations), n_particles, particle_resampling,
                                      particle_resampling_threshold, particle_initial_state_scaling,
                                      particle_rng, particle_target_ratio, particle_mh_steps,
                                      particle_max_stages, particle_mh_scale))
    elseif filter == :ivashchenko_kalman
        extra_kw = merge(extra_kw, (; initial_covariance,
                                      measurement_error = MacroModelling.resolve_measurement_error(filter, measurement_error, data_in_deviations)))
    elseif filter == :quadratic_kalman || filter == :cubic_kalman
        extra_kw = merge(extra_kw, (; initial_covariance, measurement_error))
    end
    if filter == :inversion && initial_covariance !== :theoretical
        @info "`initial_covariance` is not used by the inversion filter, which fixes the initial state and carries no state covariance. Ignoring input." maxlog = MacroModelling.DEFAULT_MAXLOG
    end
    # The Kalman and particle filters take a prior on the initial state; the
    # inversion filter has none, so only forward it where it means something.
    if filter == :kalman || filter == :ivashchenko_kalman || filter == :quadratic_kalman ||
       filter == :cubic_kalman || filter ∈ MacroModelling.PARTICLE_FILTERS
        extra_kw = merge(extra_kw, (; initial_covariance))
    end

    variables_to_plot, shocks_to_plot, standard_deviations, decomposition = filter_data_with_model(𝓂, data_in_deviations, Val(algorithm), Val(filter), warmup_iterations = warmup_iterations, smooth = smooth, opts = opts; extra_kw...)

    if is_pruned
        if mc
            decomposition[:, end - 1, :] .+= SSS_delta
        else
            decomposition[:,1:(end - 2 - pruning),:]    .+= SSS_delta
            decomposition[:,end - 2,:]                  .-= SSS_delta * (size(decomposition,2) - 4)
            decomposition[:,end,:]                      .+= SSS_delta
        end
    end
    
    variables_to_plot                           .+= SSS_delta
    data_in_deviations                          .+= SSS_delta[obs_idx]

    # Compute unconditional forecast if forecast_periods > 0
    forecast_irf = nothing
    forecast_data = nothing
    extended_x_axis = x_axis
    if forecast_periods > 0
        # Get the final state from the last period of filtered data
        final_filtered_state = variables_to_plot[:, end] .+ NSSS .- SSS_delta
        
        # Compute the unconditional forecast (IRF with no shocks from the final state)
        forecast_irf = get_irf(𝓂,
                               parameters = parameters,
                               algorithm = algorithm,
                               shocks = :none,
                               periods = forecast_periods,
                               variables = :all,
                               initial_state = final_filtered_state,
                               levels = false,
                               quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                               sylvester_algorithm = sylvester_algorithm,
                               lyapunov_algorithm = lyapunov_algorithm,
                               tol = tol,
                               verbose = verbose,
                               caching = caching,
                               use_workspaces = use_workspaces)

        forecast_data = collect(forecast_irf)
        
        # Create extended x-axis for plotting (including forecast periods)
        last_x = x_axis[end]
        
        period = infer_step(x_axis)
        extended_x_axis = vcat(x_axis, [last_x + i * period for i in 1:forecast_periods])
    end

    pal = build_extended_palette(attributes_redux)

    estimate_color = :navy

    data_color = :orangered

    while length(model_estimates_active_plot_container) > 0
        pop!(model_estimates_active_plot_container)
    end

    args_and_kwargs = Dict(:run_id => length(model_estimates_active_plot_container) + 1,
                           :model_name => 𝓂.model_name,
                           :label => label,
                           
                           :data => data,
                           :parameters => Dict(𝓂.constants.post_complete_parameters.parameters .=> 𝓂.parameter_values),
                           :algorithm => algorithm,
                           :filter => filter,
                           :warmup_iterations => warmup_iterations,
                           :variables => variables,
                           :shocks => shocks,
                           :presample_periods => presample_periods,
                           :data_in_levels => data_in_levels,
                        #    :shock_decomposition => shock_decomposition,
                           :smooth => smooth,
                           
                           :tol => tol_to_dict(tol, algorithm; needs_covariance = filter == :kalman),

                           :quadratic_matrix_equation_algorithm => quadratic_matrix_equation_algorithm,
                           :sylvester_algorithm => sylvester_algorithm,
                           :lyapunov_algorithm => lyapunov_algorithm,
                           
                           :decomposition => decomposition,
                           :variables_to_plot => variables_to_plot[var_idx, :],
                           :data_in_deviations => data_in_deviations,
                           :shocks_to_plot => shocks_to_plot,
                           :reference_steady_state => reference_steady_state[var_idx],
                           :variable_names => variable_names_display,
                           :shock_names => shock_names_display,
                           :x_axis => x_axis,
                           :extended_x_axis => extended_x_axis,
                           :forecast_data => isnothing(forecast_irf) ? forecast_irf : forecast_irf[var_idx, :, :],
                           :forecast_periods => forecast_periods,
                           :rename_dictionary => processed_rename_dictionary
                           )

    push!(model_estimates_active_plot_container, args_and_kwargs)

    return_plots = []

    n_subplots = length(var_idx) + length(shock_idx)
    pp = []
    pane = 1
    plot_count = 1

    for v in var_idx
        if all(isapprox.(variables_to_plot[v, periods], 0, atol = eps(Float32)))
            n_subplots -= 1
        end
    end

    non_zero_shock_names = String[]
    non_zero_shock_idx = Int[]

    for (i,s) in enumerate(shock_idx)
        if all(isapprox.(shocks_to_plot[s, periods], 0, atol = eps(Float32)))
            n_subplots -= 1
        elseif length(shock_idx) > 0
            push!(non_zero_shock_idx, s)
            push!(non_zero_shock_names, shock_names_display[i])
        end
    end
    
    for i in 1:length(var_idx) + length(non_zero_shock_idx)
        if i > length(var_idx) # Shock decomposition
            if !(all(isapprox.(shocks_to_plot[non_zero_shock_idx[i - length(var_idx)],periods], 0, atol = eps(Float32))))
                push!(pp,begin
                        p = standard_subplot(shocks_to_plot[non_zero_shock_idx[i - length(var_idx)],periods],
                                            0.0, 
                                            non_zero_shock_names[i - length(var_idx)], 
                                            gr_back,
                                            pal = shock_decomposition ? StatsPlots.palette([estimate_color]) : pal,
                                            xvals = x_axis)         
                end)
            else
                continue
            end
        else
            if !(all(isapprox.(variables_to_plot[var_idx[i],periods], 0, atol = eps(Float32))))
                SS = reference_steady_state[var_idx[i]]

                if shock_decomposition
                    if length(non_zero_shock_idx) < (size(decomposition,2) - 2 - pruning) # not showing all shocks
                        decomp_of_nonzero_shocks = decomposition[var_idx[i],non_zero_shock_idx,periods]
                        sum_of_other_shocks = decomposition[var_idx[i],[end],periods] .- decomposition[var_idx[i],[end-1],periods] .- sum(decomp_of_nonzero_shocks, dims = 1)

                        if pruning
                            sum_of_other_shocks .-= decomposition[var_idx[i],[end-2],periods]
                        end
                        
                        decomp = cat(decomp_of_nonzero_shocks, sum_of_other_shocks, decomposition[var_idx[i],(size(decomposition,2) - 1 - pruning):end,periods], dims = 1)
                    else
                        decomp = decomposition[var_idx[i],:,periods]
                    end

                    initial_value_idx = size(decomp,1) - 1
                    shock_component_idx = 1:(size(decomp,1) - 2 - Int(pruning && !mc))
                    component_order = vcat(initial_value_idx, shock_component_idx)
                    if pruning && !mc
                        component_order = vcat(component_order, size(decomp,1) - 2)
                    end
                    
                    # Prepare data with NaN padding for forecast extension
                    decomp_padded = if forecast_periods > 0
                        [vcat(decomp[k,:], fill(NaN, forecast_periods)) for k in component_order]
                    else
                        [decomp[k,:] for k in component_order]
                    end
                    
                    p = standard_subplot(Val(:stack),
                                        decomp_padded, 
                                        [SS for _ in component_order], 
                                        variable_names_display[i], 
                                        gr_back,
                                        true, # same_ss,
                                        transparency = transparency,
                                        xvals = extended_x_axis,
                                        pal = pal,
                                        color_total = estimate_color)
                    if var_idx[i] ∈ obs_idx
                        # Pad data with NaN for forecast period
                        data_padded = if forecast_periods > 0
                            vcat(vec(data_in_deviations[indexin([var_idx[i]],obs_idx),periods]), fill(NaN, forecast_periods))
                        else
                            vec(data_in_deviations[indexin([var_idx[i]],obs_idx),periods])
                        end
                        StatsPlots.plot!(p,
                            # extended_x_axis,
                            shock_decomposition ? data_padded : data_padded .+ SS,
                            label = "",
                            color = shock_decomposition ? data_color : pal[2])
                    end
                    
                    # Add forecast if available
                    if forecast_periods > 0 && !isnothing(forecast_data)
                        # Create full forecast array with NaN padding for historical periods
                        forecast_full = vcat(
                            fill(NaN, length(x_axis) - 1),  # NaN for all periods except the last
                            variables_to_plot[var_idx[i], end],  # Last filtered value for connection
                            forecast_data[var_idx[i], :]  # Forecast values
                        )
                        
                        StatsPlots.plot!(p,
                            # extended_x_axis,
                            forecast_full,
                            linestyle = :dash,
                            label = "",
                            color = estimate_color)
                    end
                else
                    # Pad variables_to_plot with NaN for forecast extension
                    var_data_padded = if forecast_periods > 0
                        vcat(variables_to_plot[var_idx[i],periods], fill(NaN, forecast_periods))
                    else
                        variables_to_plot[var_idx[i],periods]
                    end
                    p = standard_subplot(var_data_padded, 
                                        SS, 
                                        variable_names_display[i], 
                                        gr_back,
                                        pal = shock_decomposition ? StatsPlots.palette([estimate_color]) : pal,
                                        xvals = extended_x_axis)

                    if var_idx[i] ∈ obs_idx
                        # Pad data with NaN for forecast period
                        data_padded = if forecast_periods > 0
                            vcat(vec(data_in_deviations[indexin([var_idx[i]],obs_idx),periods]), fill(NaN, forecast_periods))
                        else
                            vec(data_in_deviations[indexin([var_idx[i]],obs_idx),periods])
                        end
                        StatsPlots.plot!(p,
                            extended_x_axis,
                            shock_decomposition ? data_padded : data_padded .+ SS,
                            label = "",
                            color = shock_decomposition ? data_color : pal[2])
                    end
                    
                    # Add forecast if available
                    if forecast_periods > 0 && !isnothing(forecast_data)
                        # Create full forecast array with NaN padding for historical periods
                        forecast_full = vcat(
                            fill(NaN, length(x_axis) - 1),  # NaN for all periods except the last
                            variables_to_plot[var_idx[i], end],  # Last filtered value for connection
                            forecast_data[var_idx[i], :]  # Forecast values
                        )
                        
                        StatsPlots.plot!(p,
                            extended_x_axis,
                            shock_decomposition ? forecast_full : forecast_full .+ SS,
                            linestyle = :dash,
                            label = "",
                            color = shock_decomposition ? estimate_color : pal[1])
                    end
                end
                        
                push!(pp, p)
            else
                continue
            end
        end

        if !(plot_count % plots_per_page == 0)
            plot_count += 1
        else
            plot_count = 1

            ppp = StatsPlots.plot(pp...; attributes...)

            pl = StatsPlots.plot(framestyle = :none,
                                legend = :inside, 
                                legend_columns = 2)

            StatsPlots.plot!(pl,
                            [NaN], 
                            label = "Estimate", 
                            color = shock_decomposition ? estimate_color : pal[1])

            if mc
                StatsPlots.plot!(pl,
                                [NaN],
                                label = "AS total",
                                color = :black)
            end

            if forecast_periods > 0
                StatsPlots.plot!(pl,
                                [NaN], 
                                label = "Forecast", 
                                linestyle = :dash,
                                color = shock_decomposition ? estimate_color : pal[1])
            end

            StatsPlots.plot!(pl,
                            [NaN], 
                            label = "Data", 
                            color = shock_decomposition ? data_color : pal[2])

            if shock_decomposition
                additional_labels_prefix = ["Initial value"]
                additional_labels_suffix = pruning && !mc ? ["Nonlinearities"] : String[]
                
                if length(non_zero_shock_idx) < (size(decomposition,2) - sum(contains.(string.(𝓂.constants.post_model_macro.exo), "ᵒᵇᶜ")) - 2 - pruning) # not showing all shocks
                    other_shocks = ["Other shocks (net)"]
                else
                    other_shocks = []
                end

                lbls_vec = vcat(additional_labels_prefix, string.(non_zero_shock_names), other_shocks, additional_labels_suffix)

                lbls = reshape(lbls_vec, 1, length(lbls_vec))

                StatsPlots.bar!(pl,
                                fill(NaN, 1, length(lbls_vec)), 
                                label = lbls, 
                                linewidth = 0,
                                alpha = transparency,
                                color = pal[mod1.(1:length(lbls), length(pal))]', 
                                legend_columns = legend_columns)
            end
            
            # Legend
            p = StatsPlots.plot(ppp,pl, 
                                    layout = StatsPlots.grid(2, 1, heights = [1 - legend_columns * 0.01 - extra_legend_space, legend_columns * 0.01 + extra_legend_space]),
                                    plot_title = "Model: "*𝓂.model_name*" ["*plot_method_label*"]  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")";
                                    attributes_redux...)

            push!(return_plots,p)

            if show_plots
                display(p)
            end

            if save_plots
                if !isdir(save_plots_path) mkpath(save_plots_path) end

                StatsPlots.savefig(p, save_plots_path * "/" * string(save_plots_name) * "__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
            end

            pane += 1
            pp = []
        end
    end

    if length(pp) > 0
        ppp = StatsPlots.plot(pp...; attributes...)

        pl = StatsPlots.plot(framestyle = :none,
                            legend = :inside, 
                            legend_columns = 2)

        StatsPlots.plot!(pl,
                        [NaN], 
                        label = "Estimate", 
                        color = shock_decomposition ? estimate_color : pal[1])

        if mc
            StatsPlots.plot!(pl,
                            [NaN],
                            label = "AS total",
                            color = :black)
        end

        if forecast_periods > 0
            StatsPlots.plot!(pl,
                            [NaN], 
                            label = "Forecast", 
                            linestyle = :dash,
                            color = shock_decomposition ? estimate_color : pal[1])
        end

        StatsPlots.plot!(pl,
                        [NaN], 
                        label = "Data", 
                        color = shock_decomposition ? data_color : pal[2])


        if shock_decomposition
            additional_labels_prefix = ["Initial value"]
            additional_labels_suffix = pruning && !mc ? ["Nonlinearities"] : String[]

            if length(non_zero_shock_idx) < (size(decomposition,2) - sum(contains.(string.(𝓂.constants.post_model_macro.exo), "ᵒᵇᶜ")) - 2 - pruning) # not showing all shocks
                other_shocks = ["Other shocks (net)"]
            else
                other_shocks = []
            end

            lbls_vec = vcat(additional_labels_prefix, string.(non_zero_shock_names), other_shocks, additional_labels_suffix)

            lbls = reshape(lbls_vec, 1, length(lbls_vec))

            StatsPlots.bar!(pl,
                            fill(NaN, 1, length(lbls_vec)), 
                            label = lbls, 
                            linewidth = 0,
                            alpha = transparency,
                            color = pal[mod1.(1:length(lbls), length(pal))]', 
                            legend_columns = legend_columns)
        end
        
        # Legend
        p = StatsPlots.plot(ppp,pl, 
                                layout = StatsPlots.grid(2, 1, heights = [1 - legend_columns * 0.01 - extra_legend_space, legend_columns * 0.01 + extra_legend_space]),
                                plot_title = "Model: "*𝓂.model_name*" ["*plot_method_label*"]  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")";
                                attributes_redux...)


        push!(return_plots,p)

        if show_plots
            display(p)
        end

        if save_plots
            if !isdir(save_plots_path) mkpath(save_plots_path) end

            StatsPlots.savefig(p, save_plots_path * "/" * string(save_plots_name) * "__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
        end
    end

    if !use_workspaces 𝓂.workspaces = orig_ws end

    return return_plots
end





"""
Wrapper for [`plot_model_estimates`](@ref) with `shock_decomposition = true`.

# Returns
- `Vector{Plot}` of individual plots
"""
plot_shock_decomposition(args...; kwargs...) =  plot_model_estimates(args...; kwargs..., shock_decomposition = true)


"""
$(SIGNATURES)
This function allows comparison of the estimated variables, shocks, the data underlying the estimates, and unconditional forecasts for any combination of inputs. The unconditional forecast (shown as a dashed line) displays the model's expected path absent any exongeos shocks starting from the final filtered state. In case the relevant steady state differs for a variable across the different calls, the plot shows the absolute deviations from the respective steady state for each call. The only exception being if the variable is observed in the data, in which case the data is always shown in levels, and the relevant steady states are indicated by black lines and mentioned in the table below the plot.

This function shares most of the signature and functionality of [`plot_model_estimates`](@ref). Its main purpose is to append plots based on the inputs to previous calls of this function and the last call of [`plot_model_estimates`](@ref). In the background it keeps a registry of the inputs and outputs and then plots the comparison.

# Arguments
- $MODEL®
- $DATA®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $FILTER®
$MacroModelling.PARTICLE_FILTER_KEYWORDS®
- $(VARIABLES®(DEFAULT_VARIABLES_EXCLUDING_OBC))
- `shocks` [Default: `:all`]: shocks for which to plot the estimates in the respective subplots. Inputs can be either a `Symbol` or `String` (e.g. `:eps_a`, `\"eps_a\"`, or `:all`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. `:all` selects all shocks in the model. `:none` selects no shocks in the model.
- `presample_periods` [Default: `0`, Type: `Int`]: number of initial retained-sample periods omitted from the plot. Useful when filtering the full sample while focusing on a later subperiod. Values above the retained sample length are clamped down automatically with an informational message.
- `forecast_periods` [Default: `$DEFAULT_FORECAST_PERIODS`, Type: `Int`]: number of periods of unconditional forecast to add after the last period of data. The forecast is shown as a dotted line to distinguish it from the model estimates.
- $DATA_IN_LEVELS®
- $LABEL®
- $RENAME_DICTIONARY®
- $SMOOTH®
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMAT®
- $SAVE_PLOTS_PATH®
- $(SAVE_PLOTS_NAME®("estimation"))
- $(PLOTS_PER_PAGE®(DEFAULT_PLOTS_PER_PAGE_SMALL))
- $MAX_ELEMENTS_PER_LEGENDS_ROW®
- $EXTRA_LEGEND_SPACE®
- $PLOT_ATTRIBUTES®
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Vector{Plot}` of individual plots

# Examples
```julia
using MacroModelling, StatsPlots


@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

simulation = simulate(RBC_CME)


plot_model_estimates(RBC_CME, simulation([:k],:,:simulate))

plot_model_estimates!(RBC_CME, simulation([:k,:c],:,:simulate))


plot_model_estimates(RBC_CME, simulation([:k],:,:simulate))

plot_model_estimates!(RBC_CME, simulation([:k],:,:simulate), smooth = false)

plot_model_estimates!(RBC_CME, simulation([:k],:,:simulate), filter = :inversion)


plot_model_estimates(RBC_CME, simulation([:c],:,:simulate))

plot_model_estimates!(RBC_CME, simulation([:c],:,:simulate), algorithm = :second_order)


plot_model_estimates(RBC_CME, simulation([:k],:,:simulate))

plot_model_estimates!(RBC_CME, simulation([:k],:,:simulate), parameters = :beta => .99)
```
"""
function plot_model_estimates!(𝓂::ℳ,
                                data::KeyedArray;
                                parameters::ParameterType = nothing,
                                steady_state_function::SteadyStateFunctionType = missing,
                                algorithm::Symbol = DEFAULT_ALGORITHM,
                                filter::Symbol = DEFAULT_FILTER_SELECTOR(algorithm),
                                initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
                                measurement_error::Union{Symbol,Real,AbstractVector{<:Real},AbstractMatrix{<:Real}} = MacroModelling.DEFAULT_MEASUREMENT_ERROR,
                                n_particles::Int = MacroModelling.DEFAULT_N_PARTICLES,
                                particle_resampling::Symbol = MacroModelling.DEFAULT_PARTICLE_RESAMPLING,
                                particle_resampling_threshold::Real = MacroModelling.DEFAULT_PARTICLE_RESAMPLING_THRESHOLD,
                                particle_initial_state_scaling::Real = MacroModelling.DEFAULT_PARTICLE_INITIAL_STATE_SCALING,
                                particle_rng::Random.AbstractRNG = Random.default_rng(),
                                particle_target_ratio::Real = MacroModelling.DEFAULT_PARTICLE_TARGET_RATIO,
                                particle_mh_steps::Int = MacroModelling.DEFAULT_TEMPERED_MH_STEPS,
                                particle_max_stages::Int = MacroModelling.DEFAULT_PARTICLE_MAX_STAGES,
                                particle_mh_scale::Real = MacroModelling.DEFAULT_PARTICLE_MH_SCALE,
                                warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS,
                                variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_OBC, 
                                shocks::Union{Symbol_input,String_input} = DEFAULT_SHOCK_SELECTION, 
                                presample_periods::Int = DEFAULT_PRESAMPLE_PERIODS,
                                forecast_periods::Int = DEFAULT_FORECAST_PERIODS,
                                data_in_levels::Bool = DEFAULT_DATA_IN_LEVELS,
                                smooth::Bool = DEFAULT_SMOOTH_SELECTOR(filter),
                                label::Union{Real, String, Symbol} = length(model_estimates_active_plot_container) + 1,
                                show_plots::Bool = DEFAULT_SHOW_PLOTS,
                                save_plots::Bool = DEFAULT_SAVE_PLOTS,
                                save_plots_format::Symbol = DEFAULT_SAVE_PLOTS_FORMAT,
                                save_plots_name::Union{String, Symbol} = "estimation",
                                save_plots_path::String = DEFAULT_SAVE_PLOTS_PATH,
                                plots_per_page::Int = DEFAULT_PLOTS_PER_PAGE_SMALL,
                                max_elements_per_legend_row::Int = DEFAULT_MAX_ELEMENTS_PER_LEGEND_ROW,
                                extra_legend_space::Float64 = DEFAULT_EXTRA_LEGEND_SPACE,
                                rename_dictionary::AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}} = Dict{Symbol, String}(),
                                plot_attributes::Dict = Dict(),
                                verbose::Bool = DEFAULT_VERBOSE,
                                tol::Tolerances = Tolerances(),
                                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                                caching::Bool = DEFAULT_CACHING,
                                use_workspaces::Bool = DEFAULT_USE_WORKSPACES)
    # @nospecialize # reduce compile time                            

    if !caching invalidate_cache_validity!(𝓂) end
    orig_ws = 𝓂.workspaces
    if !use_workspaces 𝓂.workspaces = fresh_workspaces(orig_ws) end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                                    lyapunov_algorithm = lyapunov_algorithm)

    warn_irrelevant_tol(tol, algorithm; needs_covariance = filter == :kalman)
    gr_back, attributes, attributes_redux = setup_plot_attributes(plot_attributes)


    # write_parameters_input!(𝓂, parameters, verbose = verbose)

    filter, smooth, algorithm, _, pruning, warmup_iterations = normalize_filtering_options(filter, smooth, algorithm, false, warmup_iterations)

    solve!(𝓂, 
            parameters = parameters, 
            steady_state_function = steady_state_function,
            algorithm = algorithm, 
            opts = opts, 
            dynamics = true)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)

    data = data(sort(axiskeys(data,1)))
    
    obs_axis = collect(axiskeys(data,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    variables = variables isa String_input ? variables .|> Meta.parse .|> replace_indices : variables

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    if shocks ∈ [:simulate, :all_excluding_obc] 
        @warn "Shocks input cannot be `:all_excluding_obc`, or `:simulate` in `plot_model_estimates`. Changed shocks to `:all`"
        shocks = :all
    end

    obs_idx     = parse_variables_input_to_index(obs_symbols, 𝓂.constants) |> unique |> sort
    var_idx     = parse_variables_input_to_index(variables, 𝓂.constants) |> unique  |> sort
    shock_idx   = shocks == :none ? Int64[] : parse_shocks_input_to_index(shocks, 𝓂.constants)

    # Create display names and sort alphabetically
    variable_names_display = [replace_indices_in_symbol.(apply_custom_name(𝓂.constants.post_model_macro.var[v], rename_dictionary)) for v in var_idx]
    @assert length(variable_names_display) == length(unique(variable_names_display)) "Renaming variables resulted in non-unique names. Please check the `rename_dictionary`."
    var_sort_perm = sortperm(variable_names_display, by = normalize_superscript)
    var_idx = var_idx[var_sort_perm]
    variable_names_display = variable_names_display[var_sort_perm]
    
    shock_names_display = [replace_indices_in_symbol.(apply_custom_name(𝓂.constants.post_model_macro.exo[s], rename_dictionary)) * "₍ₓ₎" for s in shock_idx]
    @assert length(shock_names_display) == length(unique(shock_names_display)) "Renaming shocks resulted in non-unique names. Please check the `rename_dictionary`."
    if length(shock_idx) > 1
        shock_sort_perm = sortperm(shock_names_display, by = normalize_superscript)
        shock_idx = shock_idx[shock_sort_perm]
        shock_names_display = shock_names_display[shock_sort_perm]
    end

    processed_rename_dictionary = process_rename_dictionary(rename_dictionary, 𝓂)

    legend_columns = 1

    legend_items = length(shock_idx) + 3 + pruning + (forecast_periods > 0 ? 1 : 0)

    max_columns = min(legend_items, max_elements_per_legend_row)
    
    # Try from max_columns down to 1 to find the optimal solution
    for cols in max_columns:-1:1
        if legend_items % cols == 0 || legend_items % cols <= max_elements_per_legend_row
            legend_columns = cols
            break
        end
    end

    if data_in_levels
        data_in_deviations = MacroModelling.missing_data_to_nan(data) .- NSSS[obs_idx]
    else
        data_in_deviations = MacroModelling.missing_data_to_nan(data)
    end

    data_in_deviations, _, _, informative_periods = trim_informative_sample(data_in_deviations;
                                                                            require_informative_periods = true)
    presample_periods = normalize_presample_periods(presample_periods, size(data_in_deviations, 2))

    x_axis = axiskeys(data,2)[informative_periods]

    extra_legend_space += length(string(x_axis[1])) > 6 ? .1 : 0.0

    periods = presample_periods+1:size(data_in_deviations,2)

    x_axis = x_axis[periods]
    
    particle_kw = filter ∈ MacroModelling.PARTICLE_FILTERS ?
        (; measurement_error = MacroModelling.resolve_measurement_error(filter, measurement_error, data_in_deviations), n_particles, particle_resampling, particle_resampling_threshold,
           particle_initial_state_scaling, particle_rng,
           particle_target_ratio, particle_mh_steps,
           particle_max_stages, particle_mh_scale) :
        filter == :ivashchenko_kalman ?
        (; initial_covariance,
           measurement_error = MacroModelling.resolve_measurement_error(filter, measurement_error, data_in_deviations)) :
        filter == :quadratic_kalman || filter == :cubic_kalman ?
        (; initial_covariance, measurement_error) :
        NamedTuple()

    if filter == :inversion && initial_covariance !== :theoretical
        @info "`initial_covariance` is not used by the inversion filter, which fixes the initial state and carries no state covariance. Ignoring input." maxlog = MacroModelling.DEFAULT_MAXLOG
    end
    # The Kalman and particle filters take a prior on the initial state; the
    # inversion filter has none (it fixes x₀ and clamps the covariance), so the
    # argument is only forwarded where it means something.
    if filter == :kalman || filter == :ivashchenko_kalman || filter == :quadratic_kalman ||
       filter == :cubic_kalman || filter ∈ MacroModelling.PARTICLE_FILTERS
        particle_kw = merge(particle_kw, (; initial_covariance))
    end

    variables_to_plot, shocks_to_plot, standard_deviations, decomposition = filter_data_with_model(𝓂, data_in_deviations, Val(algorithm), Val(filter), warmup_iterations = warmup_iterations, smooth = smooth, opts = opts; particle_kw...)
    
    if pruning
        decomposition[:,1:(end - 2 - pruning),:]    .+= SSS_delta
        decomposition[:,end - 2,:]                  .-= SSS_delta * (size(decomposition,2) - 4)
    end

    variables_to_plot                           .+= SSS_delta
    data_in_deviations                          .+= SSS_delta[obs_idx]

    # Compute unconditional forecast if forecast_periods > 0
    forecast_irf = nothing
    forecast_data = nothing
    extended_x_axis = x_axis
    if forecast_periods > 0
        # Get the final state from the last period of filtered data
        final_filtered_state = variables_to_plot[:, end] .+ NSSS .- SSS_delta
        
        # Compute the unconditional forecast (IRF with no shocks from the final state)
        forecast_irf = get_irf(𝓂,
                               parameters = parameters,
                               algorithm = algorithm,
                               shocks = :none,
                               periods = forecast_periods,
                               variables = :all,
                               initial_state = final_filtered_state,
                               levels = false,
                               quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                               sylvester_algorithm = sylvester_algorithm,
                               lyapunov_algorithm = lyapunov_algorithm,
                               tol = tol,
                               verbose = verbose,
                               caching = caching,
                               use_workspaces = use_workspaces)
        
        forecast_data = collect(forecast_irf)
        
        # Create extended x-axis for plotting (including forecast periods)
        last_x = x_axis[end]

        period = infer_step(x_axis)
        extended_x_axis = vcat(x_axis, [last_x + i * period for i in 1:forecast_periods])
    end

    pal = build_extended_palette(attributes_redux)

    estimate_color = :navy

    data_color = :orangered
    
    args_and_kwargs = Dict(:run_id => length(model_estimates_active_plot_container) + 1,
                           :model_name => 𝓂.model_name,
                           :label => label,
                           
                           :data => data,
                           :parameters => Dict(𝓂.constants.post_complete_parameters.parameters .=> 𝓂.parameter_values),
                           :algorithm => algorithm,
                           :filter => filter,
                           :warmup_iterations => warmup_iterations,
                           :variables => variables,
                           :shocks => shocks,
                           :presample_periods => presample_periods,
                           :data_in_levels => data_in_levels,
                        #    :shock_decomposition => shock_decomposition,
                           :smooth => smooth,
                           
                           :tol => tol_to_dict(tol, algorithm; needs_covariance = filter == :kalman),

                           :quadratic_matrix_equation_algorithm => quadratic_matrix_equation_algorithm,
                           :sylvester_algorithm => sylvester_algorithm,
                           :lyapunov_algorithm => lyapunov_algorithm,
                           
                           :decomposition => decomposition,
                           :variables_to_plot => variables_to_plot[var_idx, :],
                           :data_in_deviations => data_in_deviations,
                           :shocks_to_plot => shocks_to_plot,
                           :reference_steady_state => reference_steady_state[var_idx],
                           :variable_names => variable_names_display,
                           :shock_names => shock_names_display,
                           :x_axis => x_axis,
                           :extended_x_axis => extended_x_axis,
                           :forecast_data => isnothing(forecast_irf) ? forecast_irf : forecast_irf[var_idx, :, :],
                           :forecast_periods => forecast_periods,
                           :rename_dictionary => processed_rename_dictionary
                           )

    push_if_no_duplicate!(model_estimates_active_plot_container, args_and_kwargs,
        [:parameters, :rename_dictionary, :tol];
        collect_compare_keys = [:data])

    diffdict = compute_diffdict(model_estimates_active_plot_container, keys(args_and_kwargs), include_label_in_reduced = false)

    annotate_ss = Vector{Pair{String, Any}}[]

    annotate_ss_page = Pair{String,Any}[]

    annotate_diff_input = Pair{String,Any}[]

    push!(annotate_diff_input, "Plot label" => reduce(vcat, diffdict[:label]))

    len_diff = length(model_estimates_active_plot_container)

    annotate_param_diff!(annotate_diff_input, diffdict)

    common_axis = []

    data_idx = Int[]

    if haskey(diffdict, :data)
        unique_data = unique(collect.(diffdict[:data]))

        for init in diffdict[:data]
            for (i,u) in enumerate(unique_data)
                if u == init
                    push!(data_idx,i)
                    continue
                end
            end
        end

        push!(annotate_diff_input, "Data" => ["#$i" for i in data_idx])
    end

    annotate_rename_dict_diff!(annotate_diff_input, diffdict)
    
    # Determine common and combined x axis
    common_axis = mapreduce(k -> k[:x_axis], intersect, model_estimates_active_plot_container)

    if length(common_axis) > 0
        # Real x axis: collect all distinct points and sort them
        combined_x_axis = mapreduce(k -> k[:x_axis], union, model_estimates_active_plot_container) |> sort

        # For each container, compute the last x including its forecast extension
        required_last_x = maximum((
            let
                axis = k[:x_axis]
                step = infer_step(axis)
                last_observed = axis[end]
                forecast_periods = get(k, :forecast_periods, 0)
                last_observed + forecast_periods * step
            end
            for k in model_estimates_active_plot_container
        ))

        # Extend combined_x_axis up to required_last_x, if needed
        step = infer_step(combined_x_axis)
        last_combined = combined_x_axis[end]

        if required_last_x > last_combined
            xs = collect(combined_x_axis)

            next_x = last_combined
            while next_x < required_last_x
                next_x = next_x + step
                push!(xs, next_x)
            end

            extended_combined_x_axis = xs
        else
            extended_combined_x_axis = combined_x_axis
        end

    else
        # No common x axis: treat them as pure indices 1:N
        base_length = maximum(length(k[:x_axis]) for k in model_estimates_active_plot_container)
        combined_x_axis = 1:base_length

        max_extended_length = maximum(
            length(k[:x_axis]) + get(k, :forecast_periods, 0)
            for k in model_estimates_active_plot_container
        )

        combined_x_axis_length = length(combined_x_axis)
        needed_forecast_periods = max(0, max_extended_length - combined_x_axis_length)

        if needed_forecast_periods > 0
            extended_combined_x_axis = 1:(base_length + needed_forecast_periods)
        else
            extended_combined_x_axis = combined_x_axis
        end
    end

    annotate_default_kwarg_diffs!(annotate_diff_input, args_and_kwargs, diffdict,
        [:run_id, :parameters, :data, :data_in_levels,
         :decomposition, :variables_to_plot, :data_in_deviations, :shocks_to_plot, :reference_steady_state, :x_axis,
         :tol, :label,
         :shocks, :shock_names,
         :variables, :variable_names,
         :rename_dictionary, :forecast_periods, :forecast_data, :extended_x_axis])

    annotate_tol_diff!(annotate_diff_input, model_estimates_active_plot_container)
    
    if haskey(diffdict, :shock_names)
        if all(length.(diffdict[:shock_names]) .== 1)
            push!(annotate_diff_input, "Shock name" => map(x->x[1], diffdict[:shock_names]))
        end
    end
    
    legend_plot = StatsPlots.plot(framestyle = :none, 
                                    legend = :inside, 
                                    palette = pal,
                                    legend_columns = length(model_estimates_active_plot_container)) 
    
    joint_shocks = OrderedSet{String}()
    joint_variables = OrderedSet{String}()
    plt_lab_switch = should_use_label_switch(annotate_diff_input, model_estimates_active_plot_container)
    for (i,k) in enumerate(model_estimates_active_plot_container)
        StatsPlots.plot!(legend_plot,
                        [NaN],
                        color = pal[mod1.(i, length(pal))]',
                        legend_title = plt_lab_switch ? nothing : annotate_diff_input[2][1],
                        label = plt_lab_switch ? k[:label] isa Symbol ? string(k[:label]) : k[:label] : annotate_diff_input[2][2][i] isa String ? annotate_diff_input[2][2][i] : String(Symbol(annotate_diff_input[2][2][i])))

        foreach(n -> push!(joint_variables, String(apply_custom_name(n, Dict(k[:rename_dictionary])))), k[:variable_names] isa AbstractArray ? k[:variable_names] : (k[:variable_names],))
        foreach(n -> push!(joint_shocks, String(apply_custom_name(n, Dict(k[:rename_dictionary])))), k[:shock_names] isa AbstractArray ? k[:shock_names] : (k[:shock_names],))
    end

    # Add Forecast legend entries for scenarios that have forecasts
    for (i,k) in enumerate(model_estimates_active_plot_container)
        if k[:forecast_periods] > 0 && !isnothing(k[:forecast_data])
            lbl = plt_lab_switch ? k[:label] isa Symbol ? string(k[:label]) : k[:label] : annotate_diff_input[2][2][i] isa String ? annotate_diff_input[2][2][i] : String(Symbol(annotate_diff_input[2][2][i]))
            
            StatsPlots.plot!(legend_plot,
                            [NaN], 
                            linestyle = :dash,
                            label = "Forecast $lbl",
                            color = pal[mod1.(i, length(pal))]')
        end
    end

    if haskey(diffdict, :data) || haskey(diffdict, :presample_periods)
        for (i,k) in enumerate(model_estimates_active_plot_container)
            if length(data_idx) > 0
                lbl = "Data #$(data_idx[i])"
            else
                lbl = "Data $(k[:label])"
            end

            StatsPlots.plot!(legend_plot,
                                    [NaN], 
                                    label = lbl,
                                    color = pal[mod1.(length(model_estimates_active_plot_container) + i, length(pal))]',
                                    # color = pal[i]
                                    )
        end
    else
        StatsPlots.plot!(legend_plot,
                                [NaN], 
                                label = "Data",
                                color = data_color)
    end

    sort!(joint_shocks, by = normalize_superscript)
    sort!(joint_variables, by = normalize_superscript)

    return_plots = []

    n_subplots = length(joint_shocks) + length(joint_variables)
    pp = []
    pane = 1
    plot_count = 1

    joint_non_zero_variables = []
    joint_non_zero_shocks = []

    min_presample_periods = minimum([k[:presample_periods] for k in model_estimates_active_plot_container])

    for var in joint_variables
        not_zero_anywhere = false

        for k in model_estimates_active_plot_container
            var_idx = findfirst(==(var), apply_custom_name.(k[:variable_names], Ref(Dict(k[:rename_dictionary]))))
            periods = k[:presample_periods] + 1:size(k[:data], 2)

            if isnothing(var_idx) || not_zero_anywhere
                # If the variable or shock is not present in the current plot_container,
                # we skip this iteration.
                continue
            else
                if any(.!isapprox.(k[:variables_to_plot][var_idx, periods], 0, atol = eps(Float32)))
                    not_zero_anywhere = not_zero_anywhere || true
                    # break # If any irf data is not approximately zero, we set the flag to true.
                end
            end
        end
        
        if not_zero_anywhere 
            push!(joint_non_zero_variables, var)
        else
            # If all irf data for this variable and shock is approximately zero, we skip this subplot.
            n_subplots -= 1
        end
    end
    
    for shock in joint_shocks
        not_zero_anywhere = false

        for k in model_estimates_active_plot_container
            shock_idx = findfirst(==(shock), k[:shock_names])
            periods = k[:presample_periods] + 1:size(k[:data], 2)

            if isnothing(shock_idx) || not_zero_anywhere
                # If the variable or shock is not present in the current plot_container,
                # we skip this iteration.
                continue
            else
                if any(.!isapprox.(k[:shocks_to_plot][shock_idx, periods], 0, atol = eps(Float32)))
                    not_zero_anywhere = not_zero_anywhere || true
                    # break # If any irf data is not approximately zero, we set the flag to true.
                end
            end
        end
        
        if not_zero_anywhere 
            push!(joint_non_zero_shocks, shock)
        else
            # If all irf data for this variable and shock is approximately zero, we skip this subplot.
            n_subplots -= 1
        end
    end
    
    for (i,var) in enumerate(vcat(joint_non_zero_variables, joint_non_zero_shocks))
        SSs = eltype(model_estimates_active_plot_container[1][:reference_steady_state])[]

        shocks_to_plot_s = AbstractVector{eltype(model_estimates_active_plot_container[1][:shocks_to_plot])}[]

        variables_to_plot_s = AbstractVector{eltype(model_estimates_active_plot_container[1][:variables_to_plot])}[]

        for k in model_estimates_active_plot_container
            # periods = min_presample_periods + 1:length(combined_x_axis)
            periods = (1:length(k[:x_axis])) .+ k[:presample_periods]

            if i > length(joint_non_zero_variables)
                shock_idx = findfirst(==(var), apply_custom_name.(k[:shock_names], Ref(Dict(k[:rename_dictionary]))))
                if isnothing(shock_idx)
                    # If the variable or shock is not present in the current plot_container,
                    # we skip this iteration.
                    push!(SSs, NaN)
                    push!(shocks_to_plot_s, zeros(0))
                else
                    push!(SSs, 0.0)
                    
                    if common_axis == []
                        idx = 1:length(k[:x_axis])
                    else
                        idx = indexin(k[:x_axis], combined_x_axis)
                    end
                    
                    # Shocks use combined_x_axis only, not extended (no forecast for shocks)
                    shocks_to_plot = fill(NaN, length(combined_x_axis))
                    shocks_to_plot[idx] = k[:shocks_to_plot][shock_idx, periods]
                    # shocks_to_plot[idx][1:k[:presample_periods]] .= NaN
                    push!(shocks_to_plot_s, shocks_to_plot) # k[:shocks_to_plot][shock_idx, periods])
                end
            else
                var_idx = findfirst(==(var), apply_custom_name.(k[:variable_names], Ref(Dict(k[:rename_dictionary]))))
                if isnothing(var_idx)
                    # If the variable or shock is not present in the current plot_container,
                    # we skip this iteration.
                    push!(SSs, NaN)
                    push!(variables_to_plot_s, zeros(0))
                else
                    push!(SSs, k[:reference_steady_state][var_idx])

                    if common_axis == []
                        idx = 1:length(k[:x_axis])
                    else
                        idx = indexin(k[:x_axis], combined_x_axis)
                    end
                    
                    # Use extended_combined_x_axis length for padding (NaN for forecast periods)
                    variables_to_plot = fill(NaN, length(extended_combined_x_axis))
                    variables_to_plot[idx] = k[:variables_to_plot][var_idx, periods]

                    push!(variables_to_plot_s, variables_to_plot)#k[:variables_to_plot][var_idx, periods])
                end
            end
        end

        if i > length(joint_non_zero_variables)
            plot_data = shocks_to_plot_s
        else
            plot_data = variables_to_plot_s
        end

        same_ss = true

        if maximum(Base.filter(!isnan, SSs)) - minimum(Base.filter(!isnan, SSs)) > 1e-10
            push!(annotate_ss_page, var => minimal_sigfig_strings(SSs))
            same_ss = false
        end


        has_data = false

        for k in model_estimates_active_plot_container
            obs_axis = collect(axiskeys(k[:data],1))

            obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

            obs_symbols_display = [replace_indices_in_symbol.(apply_custom_name(v, Dict(k[:rename_dictionary]))) for v in obs_symbols]

            var_indx = findfirst(==(var), apply_custom_name.(k[:variable_names], Ref(Dict(k[:rename_dictionary]))))

            if var ∈ string.(obs_symbols_display) && !isnothing(var_indx)
                has_data = true || has_data
            end
        end

        # Use combined_x_axis for shocks, extended_combined_x_axis for variables
        subplot_x_axis = i > length(joint_non_zero_variables) ? combined_x_axis : extended_combined_x_axis
        
        p = standard_subplot(Val(:compare),
                                    plot_data, 
                                    SSs, 
                                    var, 
                                    gr_back,
                                    same_ss,
                                    pal = pal,
                                    xvals = subplot_x_axis,
                                    # transparency = transparency
                                    has_data = has_data
                                    )

        if haskey(diffdict, :data) || haskey(diffdict, :presample_periods)
            for (i,k) in enumerate(model_estimates_active_plot_container)
                # periods = min_presample_periods + 1:length(combined_x_axis)
                periods = (1:length(k[:x_axis])) .+ k[:presample_periods]

                obs_axis = collect(axiskeys(k[:data],1))

                obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

                obs_symbols_display = [replace_indices_in_symbol.(apply_custom_name(v, Dict(k[:rename_dictionary]))) for v in obs_symbols]

                var_indx = findfirst(==(var), apply_custom_name.(k[:variable_names], Ref(Dict(k[:rename_dictionary]))))

                if var ∈ string.(obs_symbols_display) && !isnothing(var_indx)
                    if common_axis == []
                        idx = 1:length(k[:x_axis])
                    else
                        idx = indexin(k[:x_axis], combined_x_axis)
                    end

                    # Use extended_combined_x_axis length for padding
                    data_in_deviations = fill(NaN, length(extended_combined_x_axis))
                    data_in_deviations[idx] = k[:data_in_deviations][indexin([var], string.(obs_symbols_display)), periods]
                    # data_in_deviations[idx][1:k[:presample_periods]] .= NaN

                    StatsPlots.plot!(p,
                        extended_combined_x_axis,
                        data_in_deviations .+ k[:reference_steady_state][var_indx],
                        label = "",
                        color = pal[length(model_estimates_active_plot_container) + i]
                        )
                end
            end
        else
            for k in model_estimates_active_plot_container
                periods = min_presample_periods + 1:size(k[:data], 2)

                obs_axis = collect(axiskeys(k[:data],1))

                obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

                obs_symbols_display = [replace_indices_in_symbol.(apply_custom_name(v, Dict(k[:rename_dictionary]))) for v in obs_symbols]

                var_indx = findfirst(==(var), apply_custom_name.(k[:variable_names], Ref(Dict(k[:rename_dictionary])))) 

                if var ∈ string.(obs_symbols_display) && !isnothing(var_indx)
                    # Use extended_combined_x_axis length for padding
                    data_in_deviations_padded = fill(NaN, length(extended_combined_x_axis))
                    data_vals = k[:data_in_deviations][indexin([var], string.(obs_symbols_display)),:]
                    data_vals[1:k[:presample_periods]] .= NaN
                    data_in_deviations_padded[1:length(combined_x_axis)] = data_vals[periods]
                    
                    StatsPlots.plot!(p,
                        extended_combined_x_axis,
                        data_in_deviations_padded .+ k[:reference_steady_state][var_indx],
                        label = "",
                        color = data_color
                    )
                end
            end
        end

        # Add forecast if available for any run - plot as dashed line
        if i <= length(joint_non_zero_variables)  # Only plot forecast for variables, not shocks
            for (idx, k) in enumerate(model_estimates_active_plot_container)
                if k[:forecast_periods] > 0 && !isnothing(k[:forecast_data])
                    var_indx = findfirst(==(var), apply_custom_name.(k[:variable_names], Ref(Dict(k[:rename_dictionary]))))
                    if !isnothing(var_indx)
                        # Create forecast array with NaN padding using extended_combined_x_axis
                        forecast_full = fill(NaN, length(extended_combined_x_axis))
                        
                        if common_axis == []
                            last_idx = length(k[:x_axis])
                        else
                            idx_x = indexin(k[:x_axis], combined_x_axis)
                            last_idx = maximum(idx_x)
                        end
                        
                        # Connection point (last filtered value)
                        forecast_full[last_idx] = k[:variables_to_plot][var_indx, end]
                        # Forecast values
                        forecast_start = last_idx + 1
                        forecast_end = last_idx + k[:forecast_periods]
                        forecast_full[forecast_start:forecast_end] = k[:forecast_data][var_indx, :]
                        
                        StatsPlots.plot!(p,
                            extended_combined_x_axis,
                            (has_data || same_ss) ? forecast_full .+ k[:reference_steady_state][var_indx] : forecast_full,
                            linestyle = :dash,
                            label = "",
                            color = pal[idx])
                    end
                end
            end
        end

        push!(pp, p)
        
        if !(plot_count % plots_per_page == 0)
            plot_count += 1
        else
            plot_count = 1

            pane = assemble_and_emit_page!(
                return_plots, pp, legend_plot,
                annotate_diff_input, diffdict,
                attributes, attributes_redux,
                pane, n_subplots, plots_per_page,
                show_plots, save_plots, save_plots_path, save_plots_name, save_plots_format,
                𝓂.model_name;
                annotate_ss = annotate_ss,
                annotate_ss_page = annotate_ss_page,
                plt_lab_switch = plt_lab_switch,
            )
        end
    end

    if length(pp) > 0
        assemble_and_emit_page!(
            return_plots, pp, legend_plot,
            annotate_diff_input, diffdict,
            attributes, attributes_redux,
            pane, n_subplots, plots_per_page,
            show_plots, save_plots, save_plots_path, save_plots_name, save_plots_format,
            𝓂.model_name;
            annotate_ss = annotate_ss,
            annotate_ss_page = annotate_ss_page,
            plt_lab_switch = plt_lab_switch,
            is_tail = true,
        )
    end

    if !use_workspaces 𝓂.workspaces = orig_ws end

    return return_plots
end




"""
$(SIGNATURES)
Plot impulse response functions (IRFs) of the model.

The left axis shows the level, and the right axis the deviation from the relevant steady state. The non-stochastic steady state is relevant for first order solutions and the stochastic steady state for higher order solutions. The horizontal black line indicates the relevant steady state. Variable names are above the subplots and the title provides information about the model, shocks and number of pages per shock.

If the model contains occasionally binding constraints and `ignore_obc = false` they are enforced using shocks.

# Arguments
- $MODEL®
# Keyword Arguments
- $PERIODS®
- $SHOCKS®
- $(VARIABLES®(DEFAULT_VARIABLES_EXCLUDING_AUX_AND_OBC))
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $SHOCK_SIZE®
- $NEGATIVE_SHOCK®
- $GENERALISED_IRF®
- $GENERALISED_IRF_WARMUP_ITERATIONS®
- $GENERALISED_IRF_DRAWS®
- $INITIAL_STATE®
- $IGNORE_OBC®
- `label` [Default: `1`, Type: `Union{Real, String, Symbol}`]: label to attribute to this function call in the plots.
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMAT®
- $SAVE_PLOTS_PATH®
- $(SAVE_PLOTS_NAME®("irf"))
- $(PLOTS_PER_PAGE®(DEFAULT_PLOTS_PER_PAGE_LARGE))
- $PLOT_ATTRIBUTES®
- $LABEL®
- $RENAME_DICTIONARY®
- $QME®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Vector{Plot}` of individual plots

# Examples
```julia
using MacroModelling, StatsPlots

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

plot_irf(RBC)
```
"""
function plot_irf(𝓂::ℳ;
                    periods::Int = DEFAULT_PERIODS, 
                    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = DEFAULT_SHOCKS_EXCLUDING_OBC, 
                    variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_AUX_AND_OBC,
                    parameters::ParameterType = nothing,
                    steady_state_function::SteadyStateFunctionType = missing,
                    label::Union{Real, String, Symbol} = DEFAULT_LABEL,
                    show_plots::Bool = DEFAULT_SHOW_PLOTS,
                    save_plots::Bool = DEFAULT_SAVE_PLOTS,
                    save_plots_format::Symbol = DEFAULT_SAVE_PLOTS_FORMAT,
                    save_plots_name::Union{String, Symbol} = "irf",
                    save_plots_path::String = DEFAULT_SAVE_PLOTS_PATH,
                    plots_per_page::Int = DEFAULT_PLOTS_PER_PAGE_LARGE, 
                    algorithm::Symbol = DEFAULT_ALGORITHM,
                    shock_size::Real = DEFAULT_SHOCK_SIZE,
                    negative_shock::Bool = DEFAULT_NEGATIVE_SHOCK,
                    generalised_irf::Bool = DEFAULT_GENERALISED_IRF,
                    generalised_irf_warmup_iterations::Int = DEFAULT_GENERALISED_IRF_WARMUP,
                    generalised_irf_draws::Int = DEFAULT_GENERALISED_IRF_DRAWS,
                    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}} = DEFAULT_INITIAL_STATE,
                    ignore_obc::Bool = DEFAULT_IGNORE_OBC,
                    rename_dictionary::AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}} = Dict{Symbol, String}(),
                    plot_attributes::Dict = Dict(),
                    verbose::Bool = DEFAULT_VERBOSE,
                    tol::Tolerances = Tolerances(),
                    quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                    sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                    caching::Bool = DEFAULT_CACHING,
                    use_workspaces::Bool = DEFAULT_USE_WORKSPACES)
    # @nospecialize # reduce compile time                

    if !caching invalidate_cache_validity!(𝓂) end
    orig_ws = 𝓂.workspaces
    if !use_workspaces 𝓂.workspaces = fresh_workspaces(orig_ws) end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2])

    warn_irrelevant_tol(tol, algorithm; needs_covariance = false)
    gr_back, attributes, attributes_redux = setup_plot_attributes(plot_attributes)

    shocks, negative_shock, shock_size, periods_extended, shock_idx, shock_history = process_shocks_input(shocks, negative_shock, shock_size, periods, 𝓂)

    variables = variables isa String_input ? variables .|> Meta.parse .|> replace_indices : variables

    var_idx = parse_variables_input_to_index(variables, 𝓂.constants) |> unique |> sort

    ignore_obc, occasionally_binding_constraints, obc_shocks_included = process_ignore_obc_flag(shocks, ignore_obc, 𝓂)

    generalised_irf = adjust_generalised_irf_flag(generalised_irf, generalised_irf_warmup_iterations, generalised_irf_draws, algorithm, occasionally_binding_constraints, shocks)

    solve!(𝓂, 
            parameters = parameters, 
            steady_state_function = steady_state_function,
            opts = opts, 
            dynamics = true, 
            algorithm = algorithm, 
            obc = occasionally_binding_constraints || obc_shocks_included)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)
    
    initial_state_input = copy(initial_state)

    initial_state = adjust_initial_state(initial_state, algorithm, 𝓂, SSS_delta, reference_steady_state)
    

    if occasionally_binding_constraints
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    elseif obc_shocks_included
        @assert algorithm ∉ [:pruned_second_order, :second_order, :pruned_third_order, :third_order] "Occasionally binding constraint shocks without enforcing the constraint is only compatible with first order perturbation solutions."

        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    else
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, false)
    end

    level = zeros(𝓂.constants.post_model_macro.nVars)

    Y = compute_irf_responses(𝓂,
                                state_update,
                                initial_state,
                                level;
                                periods = periods_extended,
                                shocks = shocks,
                                variables = variables,
                                shock_size = shock_size,
                                negative_shock = negative_shock,
                                generalised_irf = generalised_irf,
                                generalised_irf_warmup_iterations = generalised_irf_warmup_iterations,
                                generalised_irf_draws = generalised_irf_draws,
                                enforce_obc = occasionally_binding_constraints,
                                algorithm = algorithm)

    if !generalised_irf || occasionally_binding_constraints
        Y = Y .+ SSS_delta[var_idx]
    end

    shock_dir = negative_shock ? "Shock⁻" : "Shock⁺"

    if shocks == :none
        shock_dir = ""
    end
    if shocks == :simulate
        shock_dir = "Shocks"
    end
    if !(shocks isa Union{Symbol_input,String_input})
        shock_dir = ""
    end

    if shocks == :simulate
        shock_names_display = ["simulation"]
    elseif shocks == :none
        shock_names_display = ["no_shock"]
    elseif shocks isa Union{Symbol_input,String_input}
        shock_names_display = [replace_indices_in_symbol.(apply_custom_name(𝓂.constants.post_model_macro.exo[s], rename_dictionary)) for s in shock_idx]
        @assert length(shock_names_display) == length(unique(shock_names_display)) "Renaming shocks resulted in non-unique names. Please check the `rename_dictionary`."
        # Sort shocks alphabetically by display name
        if length(shock_idx) > 1
            shock_sort_perm = sortperm(shock_names_display, by = normalize_superscript)
            shock_idx = shock_idx[shock_sort_perm]
            shock_names_display = shock_names_display[shock_sort_perm]
        end
    else
        shock_names_display = ["shock_matrix"]
    end
    
    # Create display names and sort alphabetically
    variable_names_display = [replace_indices_in_symbol.(apply_custom_name(𝓂.constants.post_model_macro.var[v], rename_dictionary)) for v in var_idx]
    @assert length(variable_names_display) == length(unique(variable_names_display)) "Renaming variables resulted in non-unique names. Please check the `rename_dictionary`."
    var_sort_perm = sortperm(variable_names_display, by = normalize_superscript)
    var_idx = var_idx[var_sort_perm]
    variable_names_display = variable_names_display[var_sort_perm]
    
    Y = Y[var_sort_perm, :, :]

    processed_rename_dictionary = process_rename_dictionary(rename_dictionary, 𝓂)

    while length(irf_active_plot_container) > 0
        pop!(irf_active_plot_container)
    end
    
    args_and_kwargs = Dict(:run_id => length(irf_active_plot_container) + 1,
                           :model_name => 𝓂.model_name,
                           :label => label,

                           :periods => periods,
                           :shocks => shocks,
                           :variables => variables,
                           :parameters => Dict(𝓂.constants.post_complete_parameters.parameters .=> 𝓂.parameter_values),
                           :algorithm => algorithm,
                           :shock_size => shock_size,
                           :negative_shock => negative_shock,
                           :generalised_irf => generalised_irf,
                           :generalised_irf_warmup_iterations => generalised_irf_warmup_iterations,
                           :generalised_irf_draws => generalised_irf_draws,
                           :initial_state => initial_state_input,
                           :ignore_obc => ignore_obc,

                           :tol => tol_to_dict(tol, algorithm; needs_covariance = false),

                           :quadratic_matrix_equation_algorithm => quadratic_matrix_equation_algorithm,
                           :sylvester_algorithm => sylvester_algorithm,

                           :plot_data => Y,
                           :reference_steady_state => reference_steady_state[var_idx],
                           :variable_names => variable_names_display,
                           :shock_names => shock_names_display,
                           :rename_dictionary => processed_rename_dictionary
                           )
    
    push!(irf_active_plot_container, args_and_kwargs)

    pal = build_extended_palette(attributes_redux)

    return_plots = []

    for shock in 1:length(shock_idx)
        n_subplots = length(var_idx)
        pp = []
        pane = 1
        plot_count = 1

        for i in 1:length(var_idx)
            if all(isapprox.(Y[i,:,shock], 0, atol = eps(Float32)))
                n_subplots -= 1
            end
        end

        for (i,v) in enumerate(var_idx)
            SS = reference_steady_state[v]

            if !(all(isapprox.(Y[i,:,shock],0,atol = eps(Float32))))
                variable_name = variable_names_display[i]

                push!(pp, standard_subplot(Y[i,:,shock], SS, variable_name, gr_back, pal = pal))

                if !(plot_count % plots_per_page == 0)
                    plot_count += 1
                else
                    plot_count = 1

                    if shocks == :simulate
                        shock_string = ": simulate all"
                        shock_name = "simulation"
                    elseif shocks == :none
                        shock_string = ""
                        shock_name = "no_shock"
                    elseif shocks isa Union{Symbol_input,String_input}
                        shock_string = ": " * shock_names_display[shock]
                        shock_name = shock_names_display[shock]
                    else
                        shock_string = "Series of shocks"
                        shock_name = "shock_matrix"
                    end

                    p = StatsPlots.plot(pp..., plot_title = "Model: "*𝓂.model_name*"        " * shock_dir *  shock_string *"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"; attributes_redux...)

                    push!(return_plots,p)

                    if show_plots
                        display(p)
                    end

                    if save_plots
                        if !isdir(save_plots_path) mkpath(save_plots_path) end

                        StatsPlots.savefig(p, save_plots_path * "/" * string(save_plots_name) * "__" * 𝓂.model_name * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
                    end

                    pane += 1

                    pp = []
                end
            end
        end
        
        if length(pp) > 0
            if shocks == :simulate
                shock_string = ": simulate all"
                shock_name = "simulation"
            elseif shocks == :none
                shock_string = ""
                shock_name = "no_shock"
            elseif shocks isa Union{Symbol_input,String_input}
                shock_string = ": " * shock_names_display[shock]
                shock_name = shock_names_display[shock]
            else
                shock_string = "Series of shocks"
                shock_name = "shock_matrix"
            end

            p = StatsPlots.plot(pp..., plot_title = "Model: "*𝓂.model_name*"        " * shock_dir *  shock_string * "  (" * string(pane) * "/" * string(Int(ceil(n_subplots/plots_per_page)))*")"; attributes_redux...)

            push!(return_plots,p)

            if show_plots
                display(p)
            end

            if save_plots
                if !isdir(save_plots_path) mkpath(save_plots_path) end

                StatsPlots.savefig(p, save_plots_path * "/" * string(save_plots_name) * "__" * 𝓂.model_name * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
            end
        end
    end

    if !use_workspaces 𝓂.workspaces = orig_ws end

    return return_plots
end


function standard_subplot(irf_data::AbstractVector{S}, 
                            steady_state::S, 
                            variable_name::R, 
                            gr_back::Bool;
                            pal::StatsPlots.ColorPalette = StatsPlots.palette(:auto),
                            xvals = 1:length(irf_data)) where {S <: AbstractFloat, R <: Union{String, Symbol}}
    finite_vals = filter(isfinite, irf_data)
    can_dual_axis = gr_back && !isempty(finite_vals) && all((finite_vals .+ steady_state) .> eps(Float32)) && (steady_state > eps(Float32))

    xrotation = length(string(xvals[1])) > 5 ? 30 : 0

    p = StatsPlots.plot(xvals,
                        irf_data .+ steady_state,
                        title = variable_name,
                        ylabel = "Level",
                        xrotation = xrotation,
                        color = pal[1],
                        label = "")
                        
    StatsPlots.hline!([steady_state], 
                        color = :black, 
                        label = "")

    lo, hi = StatsPlots.ylims(p)

    # if !(xvals isa UnitRange)
        # low = 1
        # high = length(irf_data)

        # # Compute nice ticks on the shifted range
        # ticks_shifted, _ = StatsPlots.optimize_ticks(low, high, k_min = 4, k_max = 6)

        # ticks_shifted = Int.(ceil.(ticks_shifted))

        # labels = xvals[ticks_shifted]

        # StatsPlots.plot!(xticks = (ticks_shifted, labels))
    # end

    if can_dual_axis
        StatsPlots.plot!(StatsPlots.twinx(), 
                         ylims = (100 * (lo / steady_state - 1), 100 * (hi / steady_state - 1)),
                         xrotation = xrotation,
                         ylabel = LaTeXStrings.L"\% \Delta")                            
    end

    return p
end

function standard_subplot(::Val{:compare}, 
                            irf_data::Vector{<:AbstractVector{S}}, 
                            steady_state::Vector{S}, 
                            variable_name::R, 
                            gr_back::Bool, 
                            same_ss::Bool; 
                            xvals = 1:maximum(length.(irf_data)),
                            has_data::Bool = false,
                            pal::StatsPlots.ColorPalette = StatsPlots.palette(:auto),
                            transparency::Float64 = DEFAULT_TRANSPARENCY) where {S <: AbstractFloat, R <: Union{String, Symbol}}
    plot_dat = []
    plot_ss = 0
    
    pal_val = Int[]

    stst = 1.0

    xrotation = length(string(xvals[1])) > 5 ? 30 : 0

    can_dual_axis = gr_back
    
    for (y, ss) in zip(irf_data, steady_state)
        can_dual_axis = can_dual_axis && all((filter(!isnan, y) .+ ss) .> eps(Float32)) && ((ss > eps(Float32)) || isnan(ss))
    end
    
    for (i,(y, ss)) in enumerate(zip(irf_data, steady_state))
        if !isnan(ss)
            stst = ss
            
            if can_dual_axis && (same_ss || has_data)
                push!(plot_dat, y .+ ss)
                plot_ss = ss
            else
                if (same_ss || has_data)
                    push!(plot_dat, y .+ ss)
                else
                    push!(plot_dat, y)
                end
            end
            push!(pal_val, i)
        end
    end

    p = StatsPlots.plot(xvals,
                        plot_dat,
                        title = variable_name,
                        ylabel = (same_ss || has_data) ? "Level" : "abs. " * LaTeXStrings.L"\Delta",
                        color = pal[mod1.(pal_val, length(pal))]',
                        xrotation = xrotation,
                        label = "")

    if (same_ss || has_data)
        for ss in steady_state
            StatsPlots.hline!([ss], 
                            color = :black, 
                            label = "")
        end
    else
        StatsPlots.hline!([0], 
                        color = :black, 
                        label = "")
    end

    lo, hi = StatsPlots.ylims(p)

    # if !(xvals isa UnitRange)
    #     low = 1
    #     high = length(irf_data[1])

    #     # Compute nice ticks on the shifted range
    #     ticks_shifted, _ = StatsPlots.optimize_ticks(low, high, k_min = 4, k_max = 6)

    #     ticks_shifted = Int.(ceil.(ticks_shifted))

    #     labels = xvals[ticks_shifted]

    #     StatsPlots.plot!(xticks = (ticks_shifted, labels))
    # end

    if can_dual_axis && same_ss
        StatsPlots.plot!(StatsPlots.twinx(), 
                         ylims = (100 * (lo / plot_ss - 1), 100 * (hi / plot_ss - 1)),
                         ylabel = LaTeXStrings.L"\% \Delta")
    end
                      
    return p
end


function standard_subplot(::Val{:stack}, 
                            irf_data::Vector{<:AbstractVector{S}}, 
                            steady_state::Vector{S}, 
                            variable_name::String, 
                            gr_back::Bool, 
                            same_ss::Bool; 
                            color_total::Symbol = :black,
                            xvals = 1:length(irf_data[1]),
                            pal::StatsPlots.ColorPalette = StatsPlots.palette(:auto),
                            transparency::Float64 = DEFAULT_TRANSPARENCY) where S <: AbstractFloat
    plot_dat = []
    plot_ss = 0
    
    pal_val = Int[]

    stst = 1.0

    xrotation = length(string(xvals[1])) > 5 ? 30 : 0

    can_dual_axis = gr_back
    
    for (y, ss) in zip(irf_data, steady_state)
        if !isnan(ss)
            can_dual_axis = can_dual_axis && all((filter(!isnan, y) .+ ss) .> eps(Float32)) && ((ss > eps(Float32)) || isnan(ss))
        end
    end

    for (i,(y, ss)) in enumerate(zip(irf_data, steady_state))
        if !isnan(ss)
            stst = ss
            
            push!(plot_dat, y)

            if can_dual_axis && same_ss
                plot_ss = ss
            else
                if same_ss
                    plot_ss = ss
                end
            end
            push!(pal_val, i)
        end
    end

    # find maximum length
    maxlen = maximum(length.(plot_dat))

    # pad shorter vectors with 0
    padded = [vcat(collect(v), fill(NaN, maxlen - length(v))) for v in plot_dat]

    # now you can hcat
    plot_data = reduce(hcat, padded)

    p = StatsPlots.plot(xvals,
                    sum(x -> isfinite(x) ? x : NaN, plot_data, dims = 2), 
                    color = color_total, 
                    label = "",
                    xrotation = xrotation)
                        
    chosen_xticks = StatsPlots.xticks(p)

    p = StatsPlots.groupedbar(typeof(plot_data) <: AbstractVector ? hcat(plot_data) : plot_data,
                        title = variable_name,
                        bar_position = :stack,
                        linewidth = 0,
                        linealpha = transparency,
                        linecolor = pal[mod1.(pal_val, length(pal))]',
                        color = pal[mod1.(pal_val, length(pal))]',
                        alpha = transparency,
                        ylabel = same_ss ? "Level" : "abs. " * LaTeXStrings.L"\Delta",
                        label = "",
                        xrotation = xrotation
                        )
        
    chosen_xticks_bar = StatsPlots.xticks(p)

    if chosen_xticks_bar[1][1] == chosen_xticks[1][1]
        StatsPlots.xticks!(p, chosen_xticks_bar[1][1], chosen_xticks[1][2])
    else
        idxs = indexin(chosen_xticks[1][2], string.(xvals))

        if isnothing(idxs[1])
            idxs[1] = 0
        end

        if isnothing(idxs[end])
            idxs[end] = idxs[end-1] + (idxs[end-1] - idxs[end-2])
        end

        StatsPlots.xticks!(p, Int.(idxs), chosen_xticks[1][2])
        # StatsPlots.xticks!(p, chosen_xticks_bar[1][1], chosen_xticks_bar[1][2])
    end

    StatsPlots.hline!([0], 
                        color = :black, 
                        label = "")
                        
    StatsPlots.plot!(sum(x -> isfinite(x) ? x : NaN, plot_data, dims = 2), 
                    color = color_total, 
                    label = "")

    # Get the current y limits
    lo, hi = StatsPlots.ylims(p)

    # Compute nice ticks on the shifted range
    ticks_shifted, _ = StatsPlots.optimize_ticks(lo + plot_ss, hi + plot_ss, k_min = 4, k_max = 8)

    labels = Showoff.showoff(ticks_shifted, :auto)
    # Map tick positions back by subtracting the offset, keep shifted labels
    yticks_positions = ticks_shifted .- plot_ss
               
    StatsPlots.plot!(yticks = (yticks_positions, labels))
    
    # if !(xvals isa UnitRange)
    #     low = 1
    #     high = length(irf_data[1])

    #     # Compute nice ticks on the shifted range
    #     ticks_shifted, _ = StatsPlots.optimize_ticks(low, high, k_min = 4, k_max = 6)

    #     ticks_shifted = Int.(ceil.(ticks_shifted))

    #     labels = xvals[ticks_shifted]

    #     StatsPlots.plot!(xticks = (ticks_shifted, labels))
    # end

    if can_dual_axis && same_ss
        StatsPlots.plot!(
            StatsPlots.twinx(),
            ylims = (100 * ((lo + plot_ss) / plot_ss - 1), 100 * ((hi + plot_ss) / plot_ss - 1)),
            ylabel = LaTeXStrings.L"\% \Delta"
        )
    end
                    
    return p
end



"""
$(SIGNATURES)
This function allows comparison or stacking of impulse repsonse functions for any combination of inputs.

This function shares most of the signature and functionality of [`plot_irf`](@ref). Its main purpose is to append plots based on the inputs to previous calls of this function and the last call of [`plot_irf`](@ref). In the background it keeps a registry of the inputs and outputs and then plots the comparison or stacks the output.


# Arguments
- $MODEL®
# Keyword Arguments
- $PERIODS®
- $SHOCKS®
- $(VARIABLES®(DEFAULT_VARIABLES_EXCLUDING_AUX_AND_OBC))
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $SHOCK_SIZE®
- $NEGATIVE_SHOCK®
- $GENERALISED_IRF®
- $GENERALISED_IRF_WARMUP_ITERATIONS®
- $GENERALISED_IRF_DRAWS®
- $INITIAL_STATE®
- $IGNORE_OBC®
- $LABEL®
- $RENAME_DICTIONARY®
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMAT®
- $SAVE_PLOTS_PATH®
- $(SAVE_PLOTS_NAME®("irf"))
- $(PLOTS_PER_PAGE®(DEFAULT_PLOTS_PER_PAGE_SMALL))
- $PLOT_ATTRIBUTES®
- `plot_type` [Default: `:compare`, Type: `Symbol`]: plot type used to represent results. `:compare` means results are shown as separate lines. `:stack` means results are stacked.
- `transparency` [Default: `$DEFAULT_TRANSPARENCY`, Type: `Float64`]: transparency of stacked bars. Only relevant if `plot_type` is `:stack`.
- $QME®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®
# Returns
- `Vector{Plot}` of individual plots

# Examples
```julia
using MacroModelling, StatsPlots

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;


plot_irf(RBC)

plot_irf!(RBC, algorithm = :pruned_second_order)

plot_irf!(RBC, algorithm = :pruned_second_order, generalised_irf = true)


plot_irf(RBC)

plot_irf!(RBC, parameters = :β => 0.955)

plot_irf!(RBC, parameters = :α => 0.485)


plot_irf(RBC)

plot_irf!(RBC, negative_shock = true)


plot_irf(RBC, algorithm = :pruned_second_order)

plot_irf!(RBC, algorithm = :pruned_second_order, shock_size = 2)


plot_irf(RBC)

plot_irf!(RBC, shock_size = 2, plot_type = :stack)
```
"""
function plot_irf!(𝓂::ℳ;
                    periods::Int = DEFAULT_PERIODS, 
                    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = DEFAULT_SHOCKS_EXCLUDING_OBC, 
                    variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_AUX_AND_OBC,
                    parameters::ParameterType = nothing,
                    steady_state_function::SteadyStateFunctionType = missing,
                    label::Union{Real, String, Symbol} = length(irf_active_plot_container) + 1,
                    show_plots::Bool = DEFAULT_SHOW_PLOTS,
                    save_plots::Bool = DEFAULT_SAVE_PLOTS,
                    save_plots_format::Symbol = DEFAULT_SAVE_PLOTS_FORMAT,
                    save_plots_name::Union{String, Symbol} = "irf",
                    save_plots_path::String = DEFAULT_SAVE_PLOTS_PATH,
                    plots_per_page::Int = DEFAULT_PLOTS_PER_PAGE_SMALL, 
                    algorithm::Symbol = DEFAULT_ALGORITHM,
                    shock_size::Real = DEFAULT_SHOCK_SIZE,
                    negative_shock::Bool = DEFAULT_NEGATIVE_SHOCK,
                    generalised_irf::Bool = DEFAULT_GENERALISED_IRF,
                    generalised_irf_warmup_iterations::Int = DEFAULT_GENERALISED_IRF_WARMUP,
                    generalised_irf_draws::Int = DEFAULT_GENERALISED_IRF_DRAWS,
                    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}} = DEFAULT_INITIAL_STATE,
                    ignore_obc::Bool = DEFAULT_IGNORE_OBC,
                    plot_type::Symbol = DEFAULT_PLOT_TYPE,
                    rename_dictionary::AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}} = Dict{Symbol, String}(),
                    plot_attributes::Dict = Dict(),
                    transparency::Float64 = DEFAULT_TRANSPARENCY,
                    verbose::Bool = DEFAULT_VERBOSE,
                    tol::Tolerances = Tolerances(),
                    quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                    sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                    caching::Bool = DEFAULT_CACHING,
                    use_workspaces::Bool = DEFAULT_USE_WORKSPACES)
    # @nospecialize # reduce compile time                

    if !caching invalidate_cache_validity!(𝓂) end
    orig_ws = 𝓂.workspaces
    if !use_workspaces 𝓂.workspaces = fresh_workspaces(orig_ws) end

    @assert plot_type ∈ [:compare, :stack] "plot_type must be either :compare or :stack"

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2])

    warn_irrelevant_tol(tol, algorithm; needs_covariance = false)
    gr_back, attributes, attributes_redux = setup_plot_attributes(plot_attributes)

    pal = build_extended_palette(attributes_redux)

    shocks, negative_shock, shock_size, periods_extended, shock_idx, shock_history = process_shocks_input(shocks, negative_shock, shock_size, periods, 𝓂)
    
    variables = variables isa String_input ? variables .|> Meta.parse .|> replace_indices : variables

    var_idx = parse_variables_input_to_index(variables, 𝓂.constants) |> unique |> sort

    ignore_obc, occasionally_binding_constraints, obc_shocks_included = process_ignore_obc_flag(shocks, ignore_obc, 𝓂)

    generalised_irf = adjust_generalised_irf_flag(generalised_irf, generalised_irf_warmup_iterations, generalised_irf_draws, algorithm, occasionally_binding_constraints, shocks)

    solve!(𝓂, 
            parameters = parameters, 
            steady_state_function = steady_state_function,
            opts = opts, 
            dynamics = true, 
            algorithm = algorithm, 
            obc = occasionally_binding_constraints || obc_shocks_included)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)
    
    initial_state_input = copy(initial_state)

    initial_state = adjust_initial_state(initial_state, algorithm, 𝓂, SSS_delta, reference_steady_state)


    if occasionally_binding_constraints
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    elseif obc_shocks_included
        @assert algorithm ∉ [:pruned_second_order, :second_order, :pruned_third_order, :third_order] "Occasionally binding constraint shocks without enforcing the constraint is only compatible with first order perturbation solutions."

        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    else
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, false)
    end

    level = zeros(𝓂.constants.post_model_macro.nVars)

    Y = compute_irf_responses(𝓂,
                                state_update,
                                initial_state,
                                level;
                                periods = periods_extended,
                                shocks = shocks,
                                variables = variables,
                                shock_size = shock_size,
                                negative_shock = negative_shock,
                                generalised_irf = generalised_irf,
                                generalised_irf_warmup_iterations = generalised_irf_warmup_iterations,
                                generalised_irf_draws = generalised_irf_draws,
                                enforce_obc = occasionally_binding_constraints,
                                algorithm = algorithm)

    if !generalised_irf || occasionally_binding_constraints
        Y = Y .+ SSS_delta[var_idx]
    end

    if shocks == :simulate
        shock_names_display = ["simulation"]
    elseif shocks == :none
        shock_names_display = ["no_shock"]
    elseif shocks isa Union{Symbol_input,String_input}
        shock_names_display = [replace_indices_in_symbol.(apply_custom_name(𝓂.constants.post_model_macro.exo[s], rename_dictionary)) for s in shock_idx]
        # Sort shocks alphabetically by display name
        if length(shock_idx) > 1
            shock_sort_perm = sortperm(shock_names_display, by = normalize_superscript)
            shock_idx = shock_idx[shock_sort_perm]
            shock_names_display = shock_names_display[shock_sort_perm]
        end
    else
        shock_names_display = ["shock_matrix"]
    end
    
    # Create display names and sort alphabetically
    variable_names_display = [replace_indices_in_symbol.(apply_custom_name(𝓂.constants.post_model_macro.var[v], rename_dictionary)) for v in var_idx]
    @assert length(variable_names_display) == length(unique(variable_names_display)) "Renaming variables resulted in non-unique names. Please check the `rename_dictionary`."
    var_sort_perm = sortperm(variable_names_display, by = normalize_superscript)
    var_idx = var_idx[var_sort_perm]
    variable_names_display = variable_names_display[var_sort_perm]
    Y = Y[var_sort_perm, :, :]

    processed_rename_dictionary = process_rename_dictionary(rename_dictionary, 𝓂)

    args_and_kwargs = Dict(:run_id => length(irf_active_plot_container) + 1,
                           :model_name => 𝓂.model_name,
                           :label => label,

                           :periods => periods,
                           :shocks => shocks,
                           :variables => variables,
                           :parameters => Dict(𝓂.constants.post_complete_parameters.parameters .=> 𝓂.parameter_values),
                           :algorithm => algorithm,
                           :shock_size => shock_size,
                           :negative_shock => negative_shock,
                           :generalised_irf => generalised_irf,
                           :generalised_irf_warmup_iterations => generalised_irf_warmup_iterations,
                           :generalised_irf_draws => generalised_irf_draws,
                           :initial_state => initial_state_input,
                           :ignore_obc => ignore_obc,

                           :tol => tol_to_dict(tol, algorithm; needs_covariance = false),

                           :quadratic_matrix_equation_algorithm => quadratic_matrix_equation_algorithm,
                           :sylvester_algorithm => sylvester_algorithm,
                           :plot_data => Y,
                           :reference_steady_state => reference_steady_state[var_idx],
                           :variable_names => variable_names_display,
                           :shock_names => shock_names_display,
                           :rename_dictionary => processed_rename_dictionary
                           )

    push_if_no_duplicate!(irf_active_plot_container, args_and_kwargs,
        [:parameters, :shock_names, :rename_dictionary, :shocks, :initial_state, :tol])

    diffdict = compute_diffdict(irf_active_plot_container, keys(args_and_kwargs))

    # @assert haskey(diffdict, :parameters) || haskey(diffdict, :shock_names) || haskey(diffdict, :initial_state) || any(haskey.(Ref(diffdict), keys(DEFAULT_ARGS_AND_KWARGS_NAMES))) "New plot must be different from previous plot. Use the version without ! to plot."
    
    annotate_ss = Vector{Pair{String, Any}}[]

    annotate_ss_page = Pair{String,Any}[]

    annotate_diff_input = Pair{String,Any}[]

    push!(annotate_diff_input, "Plot label" => reduce(vcat, diffdict[:label]))

    len_diff = length(irf_active_plot_container)

    annotate_param_diff!(annotate_diff_input, diffdict)

    if haskey(diffdict, :shocks)
        # Build labels where matrices receive stable indices by content
        shcks = diffdict[:shocks]

        labels   = String[]                  # "" for trivial matrices, names pass through, "#k" for indexed matrices
        seen     = [] # distinct non-trivial normalised matrices
        next_idx = 0

        for (i,x) in enumerate(shcks)
            if x === nothing
                push!(labels, "")
            elseif typeof(x) <: AbstractMatrix
                # Assign running index by first appearance
                idx = findfirst(M -> M == x, seen)
                if idx === nothing
                    push!(seen, copy(x))
                    next_idx += 1
                    idx = next_idx
                end
                
                push!(labels, "Shock Matrix #$(idx)")

            elseif x isa AbstractVector || x isa Tuple
                # Pass through vector entries, flatten into labels
                push!(labels, "[" * join(string.(apply_custom_name.(x, Ref(Dict(irf_active_plot_container[i][:rename_dictionary])))), ", ") * "]")
            else
                # Pass through scalar names
                push!(labels, string(apply_custom_name(x, Dict(irf_active_plot_container[i][:rename_dictionary]))))
            end
        end
        
        # Respect existing shock_names logic: only add when no simple one-to-one names are present
        if haskey(diffdict, :shock_names)
            # if !all(length.(diffdict[:shock_names]) .== 1)
                push!(annotate_diff_input, "Shock" => labels)
            # end
        else
            push!(annotate_diff_input, "Shock" => labels)
        end
    end

    if haskey(diffdict, :initial_state)
        vals = diffdict[:initial_state]

        labels = String[]                                # "" for [0.0], "#k" otherwise
        seen   = []           # store distinct non-[0.0] values by content
        next_idx = 0

        for v in vals
            if v === nothing
                push!(labels, "")
            elseif v == [0.0]
                push!(labels, "nothing")
            else
                idx = findfirst(==(v), seen)             # content based lookup
                if idx === nothing
                    push!(seen, copy(v))                 # store by value
                    next_idx += 1
                    idx = next_idx
                end
                push!(labels, "#$(idx)")
            end
        end

        push!(annotate_diff_input, "Initial state" => labels)
    end
    
    annotate_rename_dict_diff!(annotate_diff_input, diffdict)

    same_shock_direction = true
    
    if annotate_default_kwarg_diffs!(annotate_diff_input, args_and_kwargs, diffdict,
            [:run_id, :parameters, :plot_data, :tol, :reference_steady_state, :initial_state, :label,
             :shocks, :shock_names,
             :variables, :variable_names,
             :rename_dictionary])
        same_shock_direction = false
    end

    annotate_tol_diff!(annotate_diff_input, irf_active_plot_container)



    legend_plot = StatsPlots.plot(framestyle = :none, 
                                    legend = :inside, 
                                    legend_columns = length(irf_active_plot_container)) 
    
    joint_shocks = OrderedSet{String}()
    joint_variables = OrderedSet{String}()
    single_shock_per_irf = true
    
    max_periods = 0
    plt_lab_switch = should_use_label_switch(annotate_diff_input, irf_active_plot_container)
    for (i,k) in enumerate(irf_active_plot_container)
        if plot_type == :stack
            StatsPlots.bar!(legend_plot,
                            [NaN], 
                            legend_title = plt_lab_switch ? nothing : annotate_diff_input[2][1],
                            alpha = transparency,
                            lw = 0,  # This removes the lines around the bars
                            linecolor = :transparent,
                            color = pal[mod1.(i, length(pal))]',
                            label = plt_lab_switch ? k[:label] isa Symbol ? string(k[:label]) : k[:label] : annotate_diff_input[2][2][i] isa String ? annotate_diff_input[2][2][i] : String(Symbol(annotate_diff_input[2][2][i])))
        elseif plot_type == :compare
            StatsPlots.plot!(legend_plot,
                            [NaN], 
                            color = pal[mod1.(i, length(pal))]',
                            legend_title = plt_lab_switch ? nothing : annotate_diff_input[2][1],
                            label = plt_lab_switch ? k[:label] isa Symbol ? string(k[:label]) : k[:label] : annotate_diff_input[2][2][i] isa String ? annotate_diff_input[2][2][i] : String(Symbol(annotate_diff_input[2][2][i])))
        end

        foreach(n -> push!(joint_variables, String(apply_custom_name(n, Dict(k[:rename_dictionary])))), k[:variable_names] isa AbstractArray ? k[:variable_names] : (k[:variable_names],))
        foreach(n -> push!(joint_shocks, String(apply_custom_name(n, Dict(k[:rename_dictionary])))), k[:shock_names] isa AbstractArray ? k[:shock_names] : (k[:shock_names],))

        single_shock_per_irf = single_shock_per_irf && length(k[:shock_names]) == 1

        max_periods = max(max_periods, size(k[:plot_data],2))
    end
    
    sort!(joint_shocks, by = normalize_superscript)
    sort!(joint_variables, by = normalize_superscript)

    if single_shock_per_irf && length(joint_shocks) > 1
        joint_shocks = [:single_shock_per_irf]
    end

    return_plots = []

    for shock in joint_shocks
        n_subplots = length(joint_variables)
        pp = []
        pane = 1
        plot_count = 1
        joint_non_zero_variables = []

        for var in joint_variables
            not_zero_anywhere = false

            for k in irf_active_plot_container
                var_idx = findfirst(==(var), apply_custom_name.(k[:variable_names], Ref(Dict(k[:rename_dictionary]))))
                shock_idx = shock == :single_shock_per_irf ? 1 : findfirst(==(shock), apply_custom_name.(k[:shock_names], Ref(Dict(k[:rename_dictionary]))))
                
                if isnothing(var_idx) || isnothing(shock_idx)
                    # If the variable or shock is not present in the current irf_active_plot_container,
                    # we skip this iteration.
                    continue
                else
                    if any(.!isapprox.(k[:plot_data][var_idx,:,shock_idx], 0, atol = eps(Float32)))
                        not_zero_anywhere = not_zero_anywhere || true
                        # break # If any irf data is not approximately zero, we set the flag to true.
                    end
                end
            end

            if not_zero_anywhere 
                push!(joint_non_zero_variables, var)
            else
                # If all irf data for this variable and shock is approximately zero, we skip this subplot.
                n_subplots -= 1
            end
        end

        for var in joint_non_zero_variables
            SSs = eltype(irf_active_plot_container[1][:reference_steady_state])[]
            Ys = AbstractVector{eltype(irf_active_plot_container[1][:plot_data])}[]

            for k in irf_active_plot_container
                var_idx = findfirst(==(var), apply_custom_name.(k[:variable_names], Ref(Dict(k[:rename_dictionary]))))
                shock_idx = shock == :single_shock_per_irf ? 1 : findfirst(==(shock), apply_custom_name.(k[:shock_names], Ref(Dict(k[:rename_dictionary]))))

                if isnothing(var_idx) || isnothing(shock_idx)
                    # If the variable or shock is not present in the current irf_active_plot_container,
                    # we skip this iteration.
                    push!(SSs, NaN)
                    push!(Ys, zeros(max_periods))
                else
                    dat = fill(NaN, max_periods)
                    dat[1:length(k[:plot_data][var_idx,:,shock_idx])] .= k[:plot_data][var_idx,:,shock_idx]
                    push!(SSs, k[:reference_steady_state][var_idx])
                    push!(Ys, dat) # k[:plot_data][var_idx,:,shock_idx])
                end
            end
            
            same_ss = true

            if maximum(filter(!isnan, SSs)) - minimum(filter(!isnan, SSs)) > 1e-10
                push!(annotate_ss_page, var => minimal_sigfig_strings(SSs))
                same_ss = false
            end

            push!(pp, standard_subplot(Val(plot_type),
                                    Ys, 
                                    SSs, 
                                    var, 
                                    gr_back,
                                    same_ss,
                                    pal = pal,
                                    transparency = transparency))
            
            if !(plot_count % plots_per_page == 0)
                plot_count += 1
            else
                plot_count = 1

                shock_dir = same_shock_direction ? negative_shock ? "Shock⁻" : "Shock⁺" : "Shock"

                if shock == :single_shock_per_irf
                    shock_string = ": multiple shocks"
                    shock_name = "multiple_shocks"
                elseif shock == "simulation"
                    shock_dir = "Shocks"
                    shock_string = ": simulate all"
                    shock_name = "simulation"
                elseif shock == "no_shock"
                    shock_dir = ""
                    shock_string = ""
                    shock_name = "no_shock"
                elseif shock == "shock_matrix"
                    shock_string = "Series of shocks"
                    shock_name = "shock_matrix"
                    shock_dir = ""
                elseif shock isa Union{Symbol_input,String_input}
                    shock_string = ": " * shock
                    shock_name = shock
                end

                pane = assemble_and_emit_page!(
                    return_plots, pp, legend_plot,
                    annotate_diff_input, diffdict,
                    attributes, attributes_redux,
                    pane, n_subplots, plots_per_page,
                    show_plots, save_plots, save_plots_path, save_plots_name, save_plots_format,
                    𝓂.model_name;
                    title_extra = "        " * shock_dir * shock_string,
                    filename_extra = shock_name,
                    legend_height = 1,
                    annotate_ss = annotate_ss,
                    annotate_ss_page = annotate_ss_page,
                    plt_lab_switch = plt_lab_switch,
                )
            end
        end


        if length(pp) > 0
            shock_dir = same_shock_direction ? negative_shock ? "Shock⁻" : "Shock⁺" : "Shock"

            if shock == :single_shock_per_irf
                shock_string = ": multiple shocks"
                shock_name = "multiple_shocks"
            elseif shock == "simulation"
                shock_dir = "Shocks"
                shock_string = ": simulate all"
                shock_name = "simulation"
            elseif shock == "no_shock"
                shock_dir = ""
                shock_string = ""
                shock_name = "no_shock"
            elseif shock == "shock_matrix"
                shock_string = "Series of shocks"
                shock_name = "shock_matrix"
                shock_dir = ""
            elseif shock isa Union{Symbol_input,String_input}
                shock_string = ": " * shock
                shock_name = shock
            end

            assemble_and_emit_page!(
                return_plots, pp, legend_plot,
                annotate_diff_input, diffdict,
                attributes, attributes_redux,
                pane, n_subplots, plots_per_page,
                show_plots, save_plots, save_plots_path, save_plots_name, save_plots_format,
                𝓂.model_name;
                title_extra = "        " * shock_dir * shock_string,
                filename_extra = shock_name,
                legend_height = 1,
                annotate_ss = annotate_ss,
                annotate_ss_page = annotate_ss_page,
                plt_lab_switch = plt_lab_switch,
                is_tail = true,
            )
        end

        annotate_ss = Vector{Pair{String, Any}}[]

        annotate_ss_page = Pair{String,Any}[]
    end

    if !use_workspaces 𝓂.workspaces = orig_ws end

    return return_plots
end


"""
See [`plot_irf!`](@ref)
"""
plot_IRF!(args...; kwargs...) = plot_irf!(args...; kwargs...)

"""
See [`plot_irf!`](@ref)
"""
plot_irfs!(args...; kwargs...) = plot_irf!(args...; kwargs...)


"""
Wrapper for [`plot_irf!`](@ref) with `shocks = :simulate` and `periods = 100`.
"""
plot_simulations!(args...; kwargs...) =  plot_irf!(args...; kwargs..., shocks = :simulate, periods = get(kwargs, :periods, 100))

"""
Wrapper for [`plot_irf!`](@ref) with `shocks = :simulate` and `periods = 100`.
"""
plot_simulation!(args...; kwargs...) =  plot_irf!(args...; kwargs..., shocks = :simulate, periods = get(kwargs, :periods, 100))

"""
Wrapper for [`plot_irf!`](@ref) with `generalised_irf = true`.
"""
plot_girf!(args...; kwargs...) =  plot_irf!(args...; kwargs..., generalised_irf = true)


function merge_by_runid(dicts::Dict...)
    @assert !isempty(dicts) "At least one dictionary is required"
    @assert all(haskey.(dicts, Ref(:run_id))) "Each dictionary must contain :run_id"

    # union of all run_ids, sorted
    all_runids = sort(unique(vcat([d[:run_id] for d in dicts]...)))
    n = length(all_runids)

    merged = Dict{Symbol,Any}()
    merged[:run_id] = all_runids
    
    # Initialize all vector-based keys in merged with appropriate length and type
    # This ensures subsequent passes can UPDATE the array instead of OVERWRITING it.
    for d in dicts
        for (k, v) in d
            k === :run_id && continue
            
            if v isa AbstractVector && length(v) == length(d[:run_id])
                # Initialize an array of appropriate type and length n, filled with nothing
                # This assumes we want Nothing to be the default for missing run_ids
                if !haskey(merged, k)
                    # Use Union{Nothing, eltype(v)} for the merged array's type
                    # For a vector of matrices, eltype(v) is Matrix{...}
                    T = Union{Nothing, eltype(v)} 
                    merged[k] = Vector{T}(nothing, n) 
                end
            elseif v isa Dict
                get!(merged, k, Dict{Symbol,Any}())
                for (kk, vv) in v
                    if vv isa AbstractVector && length(vv) == length(d[:run_id])
                         if !haskey(merged[k], kk)
                            T = Union{Nothing, eltype(vv)}
                            merged[k][kk] = Vector{T}(nothing, n)
                         end
                    else
                        # For non-vector/non-run_id-indexed values inside a Dict, overwrite or ignore on subsequent passes
                        # For this fix, we'll keep the current behavior of using a vector of the value
                        if !haskey(merged[k], kk)
                            merged[k][kk] = [vv for _ in 1:n]
                        end
                    end
                end
            else
                # For non-vector/non-dictionary values, if the key doesn't exist, initialize
                # Otherwise, the subsequent dicts will OVERWRITE the value.
                if !haskey(merged, k)
                    merged[k] = [v for _ in 1:n]
                end
            end
        end
    end

    # run_id → index map for each dict
    idx_maps = [Dict(r => i for (i, r) in enumerate(d[:run_id])) for d in dicts]

    # Fill in the initialized merged structure
    for (j, d) in enumerate(dicts)
        idx_map = idx_maps[j]
        
        # Mapping from all_runids index to d[:run_id] index
        current_runid_to_all_idx = Dict(r => i for (i, r) in enumerate(d[:run_id]))
        
        for (k, v) in d
            k === :run_id && continue

            if v isa AbstractVector && length(v) == length(d[:run_id])
                # UPDATE the existing merged[k] array
                for (i, r) in enumerate(d[:run_id])
                    # idx_map[r] is the index of run_id r in d[:run_id] (i)
                    # findfirst(==(r), all_runids) is the index of run_id r in all_runids
                    merged_idx = findfirst(==(r), all_runids)
                    merged[k][merged_idx] = v[i]
                end
            elseif v isa Dict
                sub = merged[k] # Already initialized by the pre-pass
                for (kk, vv) in v
                    if vv isa AbstractVector && length(vv) == length(d[:run_id])
                         # UPDATE the existing merged[k][kk] array
                         for (i, r) in enumerate(d[:run_id])
                            merged_idx = findfirst(==(r), all_runids)
                            sub[kk][merged_idx] = vv[i]
                         end
                    # Keep the original logic for non-vector values inside sub-dicts
                    # This overwrites the whole column for non-indexed values
                    elseif !haskey(sub, kk)
                        sub[kk] = [vv for _ in 1:n]
                    end
                end
            # Keep the original logic for non-vector/non-dictionary values
            # These are already initialized, no need to do anything here if we want the value from the *first* dict to win
            # If we want the value from the *last* dict to win, we would overwrite here.
            # Given the original code's structure (where it overwrites), let's stick to 'first' or 'last' value for simplicity:
            # The current setup will prioritize the FIRST dictionary's non-run_id-indexed scalar value.
            # If you want the LAST one to win, you'd add:
            # else 
            #   merged[k] = [v for _ in 1:n]
            # end
            end
        end
    end

    return merged
end

function minimal_sigfig_strings(v::AbstractVector{<:Real};
    min_sig::Int = 3, n::Int = 10, dup_tol::Float64 = 1e-13)

    idx = collect(eachindex(v))
    finite_mask = map(x -> isfinite(x), v) # && x != 0, v)
    work_idx = filter(i -> finite_mask[i], idx)
    sorted_idx = sort(work_idx, by = i -> v[i])
    mwork = length(sorted_idx)

    # Gaps to nearest neighbour
    gaps = Dict{Int,Float64}()
    for (k, i) in pairs(sorted_idx)
        x = float(v[i])
        if mwork == 1
            gaps[i] = Inf
        elseif k == 1
            gaps[i] = abs(v[sorted_idx[k+1]] - x)
        elseif k == mwork
            gaps[i] = abs(x - v[sorted_idx[k-1]])
        else
            g1 = abs(x - v[sorted_idx[k-1]])
            g2 = abs(v[sorted_idx[k+1]] - x)
            gaps[i] = min(g1, g2)
        end
    end

    # Duplicate clusters (within dup_tol)
    duplicate = Dict{Int,Bool}()
    k = 1
    while k <= mwork
        i = sorted_idx[k]
        cluster = [i]
        x = v[i]
        j = k + 1
        while j <= mwork && abs(v[sorted_idx[j]] - x) <= dup_tol
            push!(cluster, sorted_idx[j])
            j += 1
        end
        isdup = length(cluster) > 1
        for c in cluster
            duplicate[c] = isdup
        end
        k = j
    end

    # Required significant digits for distinction
    req_sig = Dict{Int,Int}()
    for i in sorted_idx
        if duplicate[i]
            req_sig[i] = min_sig  # will apply rule anyway
        else
            x = float(v[i])
            g = gaps[i]
            if g == 0.0
                req_sig[i] = min_sig
            else
                m = floor(log10(abs(x))) + 1

                m = max(typemin(Int), m)  # avoid negative indices

                s = max(min_sig, ceil(Int, m - log10(g)))
                # Apply rule: if they differ only after more than n sig digits
                if s > n
                    req_sig[i] = min_sig
                else
                    req_sig[i] = s
                end
            end
        end
    end

    # Format output
    out = Vector{String}(undef, length(v))
    for i in eachindex(v)
        x = v[i]
        if isnan(x)
            out[i] = ""
        elseif !(isfinite(x)) || x == 0
            # For zero or non finite just echo (rule does not change them)
            out[i] = string(x)
        elseif haskey(req_sig, i)
            s = req_sig[i]
            out[i] = string(round(x, sigdigits = s))
        else
            # Non finite or zero already handled; fallback
            out[i] = string(x)
        end
    end
    return out
end


function plot_df(plot_vector::Vector{Pair{String,Any}}; fontsize::Real = DEFAULT_FONT_SIZE, title::String = "")
    # Determine dimensions from plot_vector
    ncols = length(plot_vector)
    nrows = length(plot_vector[1].second)
        
    bg_matrix = ones(nrows + 1, ncols)
    bg_matrix[1, :] .= 0.35 # Header row
    for i in 3:2:nrows+1
        bg_matrix[i, :] .= 0.85
    end
 
    # draw the "cells"
    df_plot = StatsPlots.heatmap(bg_matrix;
                c = StatsPlots.cgrad([:lightgrey, :white]),      # Color gradient for background
                yflip = true,  
                tick = :none,
                legend = false,
                framestyle = :none,
                cbar = false)

    StatsPlots.title!(df_plot, title)

    # overlay the header and numeric values
    for j in 1:ncols
        StatsPlots.annotate!(df_plot, j, 1, StatsPlots.text(plot_vector[j].first, :center, fontsize)) # Header
        for i in 1:nrows
            StatsPlots.annotate!(df_plot, j, i + 1, StatsPlots.text(string(plot_vector[j].second[i]), :center, fontsize))
        end
    end

    StatsPlots.vline!(df_plot, [1.5], color=:black, lw=0.5)

    StatsPlots.hline!(df_plot, [1.5], color=:black, lw=0.5)

    return df_plot
end


# """
# See [`plot_irf`](@ref)
# """
# plot(𝓂::ℳ; kwargs...) = plot_irf(𝓂; kwargs...)

# plot(args...;kwargs...) = StatsPlots.plot(args...;kwargs...) #fallback

"""
See [`plot_irf`](@ref)
"""
plot_IRF(args...; kwargs...) = plot_irf(args...; kwargs...)


"""
See [`plot_irf`](@ref)
"""
plot_irfs(args...; kwargs...) = plot_irf(args...; kwargs...)


"""
Wrapper for [`plot_irf`](@ref) with `shocks = :simulate` and `periods = 100`.
"""
plot_simulations(args...; kwargs...) =  plot_irf(args...; kwargs..., shocks = :simulate, periods = get(kwargs, :periods, 100))

"""
Wrapper for [`plot_irf`](@ref) with `shocks = :simulate` and `periods = 100`.
"""
plot_simulation(args...; kwargs...) =  plot_irf(args...; kwargs..., shocks = :simulate, periods = get(kwargs, :periods, 100))

"""
Wrapper for [`plot_irf`](@ref) with `generalised_irf = true`.
"""
plot_girf(args...; kwargs...) =  plot_irf(args...; kwargs..., generalised_irf = true)





"""
$(SIGNATURES)
Plot conditional variance decomposition of the model.

The vertical axis shows the share of the shocks variance contribution, and horizontal axis the period of the variance decomposition. The stacked bars represent each shocks variance contribution at a specific time horizon.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
# Keyword Arguments
- $PERIODS®
- $(VARIABLES®(DEFAULT_VARIABLE_SELECTION))
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMAT®
- $SAVE_PLOTS_PATH®
- $(SAVE_PLOTS_NAME®("fevd"))
- $(PLOTS_PER_PAGE®(DEFAULT_PLOTS_PER_PAGE_LARGE))
- $PLOT_ATTRIBUTES®
- $MAX_ELEMENTS_PER_LEGENDS_ROW®
- $EXTRA_LEGEND_SPACE®
- $RENAME_DICTIONARY®
- $QME®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Vector{Plot}` of individual plots

# Examples
```julia
using MacroModelling, StatsPlots

@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .075
end

plot_conditional_variance_decomposition(RBC_CME)
```
"""
function plot_conditional_variance_decomposition(𝓂::ℳ;
                                                periods::Int = DEFAULT_PERIODS, 
                                                variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLE_SELECTION,
                                                parameters::ParameterType = nothing,
                                                steady_state_function::SteadyStateFunctionType = missing,
                                                show_plots::Bool = DEFAULT_SHOW_PLOTS,
                                                save_plots::Bool = DEFAULT_SAVE_PLOTS,
                                                save_plots_format::Symbol = DEFAULT_SAVE_PLOTS_FORMAT,
                                                save_plots_name::Union{String, Symbol} = "fevd",
                                                save_plots_path::String = DEFAULT_SAVE_PLOTS_PATH,
                                                plots_per_page::Int = DEFAULT_PLOTS_PER_PAGE_LARGE, 
                                                rename_dictionary::AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}} = Dict{Symbol, String}(),
                                                plot_attributes::Dict = Dict(),
                                                max_elements_per_legend_row::Int = DEFAULT_MAX_ELEMENTS_PER_LEGEND_ROW,
                                                extra_legend_space::Float64 = DEFAULT_EXTRA_LEGEND_SPACE,
                                                verbose::Bool = DEFAULT_VERBOSE,
                                                tol::Tolerances = Tolerances(),
                                                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                                                caching::Bool = DEFAULT_CACHING,
                                                use_workspaces::Bool = DEFAULT_USE_WORKSPACES)
    # @nospecialize # reduce compile time                                            

    if !caching invalidate_cache_validity!(𝓂) end
    orig_ws = 𝓂.workspaces
    if !use_workspaces 𝓂.workspaces = fresh_workspaces(orig_ws) end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm)

    gr_back, attributes, attributes_redux = setup_plot_attributes(plot_attributes)

    fevds = get_conditional_variance_decomposition(𝓂,
                                                    periods = 1:periods,
                                                    parameters = parameters,
                                                    steady_state_function = steady_state_function,
                                                    verbose = verbose,
                                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                    tol = tol,
                                                    caching = caching,
                                                    use_workspaces = use_workspaces)

    variables = variables isa String_input ? variables .|> Meta.parse .|> replace_indices : variables

    var_idx = parse_variables_input_to_index(variables, 𝓂.constants) |> unique |> sort

    fevds = fevds isa KeyedArray ? axiskeys(fevds,1) isa Vector{String} ? rekey(fevds, 1 => axiskeys(fevds,1) .|> Meta.parse .|> replace_indices_special) : fevds : fevds

    fevds = fevds isa KeyedArray ? axiskeys(fevds,2) isa Vector{String} ? rekey(fevds, 2 => axiskeys(fevds,2) .|> Meta.parse .|> replace_indices_special) : fevds : fevds

    vars_to_plot = intersect(axiskeys(fevds)[1], 𝓂.constants.post_model_macro.var[var_idx])
    
    # Sort variables alphabetically by display name
    variable_names_display = [replace_indices_in_symbol.(apply_custom_name(v, rename_dictionary)) for v in vars_to_plot]
    @assert length(variable_names_display) == length(unique(variable_names_display)) "Renaming variables resulted in non-unique names. Please check the `rename_dictionary`."
    vars_sort_perm = sortperm(variable_names_display, by = normalize_superscript)
    vars_to_plot = vars_to_plot[vars_sort_perm]
    
    shocks_to_plot = axiskeys(fevds)[2]
    
    # Sort shocks alphabetically by display name
    shock_names_display = [replace_indices_in_symbol(apply_custom_name(s, rename_dictionary)) for s in shocks_to_plot]
    @assert length(shock_names_display) == length(unique(shock_names_display)) "Renaming shocks resulted in non-unique names. Please check the `rename_dictionary`."
    shocks_sort_perm = sortperm(shock_names_display, by = normalize_superscript)
    shocks_to_plot = shocks_to_plot[shocks_sort_perm]

    legend_columns = 1

    legend_items = length(shocks_to_plot)

    max_columns = min(legend_items, max_elements_per_legend_row)
    
    # Try from max_columns down to 1 to find the optimal solution
    for cols in max_columns:-1:1
        if legend_items % cols == 0 || legend_items % cols <= max_elements_per_legend_row
            legend_columns = cols
            break
        end
    end

    pal = build_extended_palette(attributes_redux)

    n_subplots = length(var_idx)
    pp = []
    pane = 1
    plot_count = 1
    return_plots = []

    for k in vars_to_plot
        if gr_back
            push!(pp,StatsPlots.groupedbar(fevds(k,:,:)', 
            title = replace_indices_in_symbol(apply_custom_name(k, rename_dictionary)), 
            bar_position = :stack,
            color = pal[mod1.(1:length(shocks_to_plot), length(pal))]',
            linecolor = :transparent,
            legend = :none))
        else
            push!(pp,StatsPlots.groupedbar(fevds(k,:,:)', 
            title = replace_indices_in_symbol(apply_custom_name(k, rename_dictionary)), 
            bar_position = :stack, 
            color = pal[mod1.(1:length(shocks_to_plot), length(pal))]',
            linecolor = :transparent,
            label = reshape(string.([replace_indices_in_symbol(apply_custom_name(s, rename_dictionary)) for s in shocks_to_plot]),1,length(shocks_to_plot))))
        end

        if !(plot_count % plots_per_page == 0)
            plot_count += 1
        else
            plot_count = 1

            ppp = StatsPlots.plot(pp...; attributes...)
            
            pp = StatsPlots.bar(fill(NaN,1,length(shocks_to_plot)), 
                                label = reshape(string.([replace_indices_in_symbol(apply_custom_name(s, rename_dictionary)) for s in shocks_to_plot]),1,length(shocks_to_plot)), 
                                linewidth = 0 , 
                                linecolor = :transparent,
                                framestyle = :none, 
                                color = pal[mod1.(1:length(shocks_to_plot), length(pal))]',
                                legend = :inside, 
                                legend_columns = legend_columns)

            p = StatsPlots.plot(ppp,pp, 
                                layout = StatsPlots.grid(2, 1, heights = [1 - legend_columns * 0.01 - extra_legend_space, legend_columns * 0.01 + extra_legend_space]),
                                plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"; attributes_redux...)

            push!(return_plots,gr_back ? p : ppp)

            if show_plots
                display(p)
            end

            if save_plots
                if !isdir(save_plots_path) mkpath(save_plots_path) end

                StatsPlots.savefig(p, save_plots_path * "/" * string(save_plots_name) * "__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
            end

            pane += 1
            pp = []
        end
    end

    if length(pp) > 0
        ppp = StatsPlots.plot(pp...; attributes...)

        pp = StatsPlots.bar(fill(NaN,1,length(shocks_to_plot)), 
                            label = reshape(string.([replace_indices_in_symbol(apply_custom_name(s, rename_dictionary)) for s in shocks_to_plot]),1,length(shocks_to_plot)), 
                            linewidth = 0 , 
                            linecolor = :transparent,
                            framestyle = :none, 
                            color = pal[mod1.(1:length(shocks_to_plot), length(pal))]',
                            legend = :inside,
                            legend_columns = legend_columns)

        p = StatsPlots.plot(ppp,pp, 
                            layout = StatsPlots.grid(2, 1, heights = [1 - legend_columns * 0.01 - extra_legend_space, legend_columns * 0.01 + extra_legend_space]),
                            plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"; 
                            attributes_redux...)

        push!(return_plots,gr_back ? p : ppp)

        if show_plots
            display(p)
        end

        if save_plots
            if !isdir(save_plots_path) mkpath(save_plots_path) end

            StatsPlots.savefig(p, save_plots_path * "/" * string(save_plots_name) * "__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
        end
    end

    if !use_workspaces 𝓂.workspaces = orig_ws end

    return return_plots
end



"""
See [`plot_conditional_variance_decomposition`](@ref)
"""
plot_fevd(args...; kwargs...) = plot_conditional_variance_decomposition(args...; kwargs...)

"""
See [`plot_conditional_variance_decomposition`](@ref)
"""
plot_forecast_error_variance_decomposition(args...; kwargs...) = plot_conditional_variance_decomposition(args...; kwargs...)





"""
$(SIGNATURES)
Plot the solution of the model (mapping of past states to present variables) around the relevant steady state (e.g. higher order perturbation algorithms are centred around the stochastic steady state). Each plot shows the relationship between the chosen state (defined in `state`) and one of the chosen variables (defined in `variables`). 

The relevant steady state is plotted along with the mapping from the chosen past state to one present variable per plot. All other (non-chosen) states remain in the relevant steady state.

In the case of pruned higher order solutions there are as many (latent) state vectors as the perturbation order. The first and third order baseline state vectors are the non-stochastic steady state and the second order baseline state vector is the stochastic steady state. Deviations for the chosen state are only added to the first order baseline state. The plot shows the mapping from `σ` standard deviations (first order) added to the first order non-stochastic steady state and the present variables. Note that there is no unique mapping from the "pruned" states and the "actual" reported state. Hence, the plots shown are just one realisation of infinitely many possible mappings.

If the model contains occasionally binding constraints and `ignore_obc = false` they are enforced using shocks.

# Arguments
- $MODEL®
- `state` [Type: `Union{Symbol,String}`]: state variable to be shown on x-axis.
# Keyword Arguments
- $(VARIABLES®(DEFAULT_VARIABLE_SELECTION))
- $ALGORITHM®
- `σ` [Default: `2`, Type: `Union{Int64,Float64}`]: defines the range of the state variable around the (non) stochastic steady state in standard deviations. E.g. a value of 2 means that the state variable is plotted for values of the (non) stochastic steady state in standard deviations +/- 2 standard deviations.
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $IGNORE_OBC®
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMAT®
- $SAVE_PLOTS_PATH®
- $(SAVE_PLOTS_NAME®("solution"))
- `plots_per_page` [Default: `6`, Type: `Int`]: how many plots to show per page
- $PLOT_ATTRIBUTES®
- $RENAME_DICTIONARY®
- `label` [Default: `1`, Type: `Union{Real, String, Symbol}`]: label to attribute to this function call in the plots.
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Vector{Plot}` of individual plots

# Examples
```julia
using MacroModelling, StatsPlots

@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

plot_solution(RBC_CME, :k)
```
"""
function plot_solution(𝓂::ℳ,
                        state::Union{Symbol,String};
                        variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLE_SELECTION,
                        algorithm::Symbol = DEFAULT_ALGORITHM,
                        σ::Union{Int64,Float64} = DEFAULT_SIGMA_RANGE,
                        parameters::ParameterType = nothing,
                        steady_state_function::SteadyStateFunctionType = missing,
                        ignore_obc::Bool = DEFAULT_IGNORE_OBC,
                        label::Union{Real, String, Symbol} = DEFAULT_LABEL,
                        show_plots::Bool = DEFAULT_SHOW_PLOTS,
                        save_plots::Bool = DEFAULT_SAVE_PLOTS,
                        save_plots_format::Symbol = DEFAULT_SAVE_PLOTS_FORMAT,
                        save_plots_name::Union{String, Symbol} = "solution",
                        save_plots_path::String = DEFAULT_SAVE_PLOTS_PATH,
                        plots_per_page::Int = DEFAULT_PLOTS_PER_PAGE_SMALL,
                        rename_dictionary::AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}} = Dict{Symbol, String}(),
                        plot_attributes::Dict = Dict(),
                        verbose::Bool = DEFAULT_VERBOSE,
                        tol::Tolerances = Tolerances(),
                        quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                        sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                        lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                        caching::Bool = DEFAULT_CACHING,
                        use_workspaces::Bool = DEFAULT_USE_WORKSPACES)
    # @nospecialize # reduce compile time                    
    
    if !caching invalidate_cache_validity!(𝓂) end
    orig_ws = 𝓂.workspaces
    if !use_workspaces 𝓂.workspaces = fresh_workspaces(orig_ws) end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                        sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                        sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                        lyapunov_algorithm = lyapunov_algorithm)

    warn_irrelevant_tol(tol, algorithm; needs_covariance = true)
    gr_back, attributes, attributes_redux = setup_plot_attributes(plot_attributes)

    state = state isa Symbol ? state : state |> Meta.parse |> replace_indices

    @assert state ∈ 𝓂.constants.post_model_macro.past_not_future_and_mixed "Invalid state. Choose one from:"*repr(replace_indices_in_symbol.(𝓂.constants.post_model_macro.past_not_future_and_mixed))

    @assert algorithm ∈ [:third_order, :pruned_third_order, :second_order, :pruned_second_order, :first_order] "Invalid algorithm. Choose one of: :third_order, :pruned_third_order, :second_order, :pruned_second_order, :first_order"

    ignore_obc, occasionally_binding_constraints, _ = process_ignore_obc_flag(:all_excluding_obc, ignore_obc, 𝓂)
    
    solve!(𝓂, opts = opts, algorithm = algorithm, dynamics = true, parameters = parameters, obc = occasionally_binding_constraints)

    SS_and_std = get_moments(𝓂, 
                            derivatives = false,
                            parameters = parameters,
                            steady_state_function = steady_state_function,
                            variables = :all,
                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                            sylvester_algorithm = sylvester_algorithm,
                            lyapunov_algorithm = lyapunov_algorithm,
                            tol = tol,
                            verbose = verbose,
                            caching = caching,
                            use_workspaces = use_workspaces)

    SS_and_std[:non_stochastic_steady_state] = SS_and_std[:non_stochastic_steady_state] isa KeyedArray ? axiskeys(SS_and_std[:non_stochastic_steady_state],1) isa Vector{String} ? rekey(SS_and_std[:non_stochastic_steady_state], 1 => axiskeys(SS_and_std[:non_stochastic_steady_state],1).|> x->Symbol.(replace.(x, "{" => "◖", "}" => "◗"))) : SS_and_std[:non_stochastic_steady_state] : SS_and_std[:non_stochastic_steady_state]
    
    SS_and_std[:standard_deviation] = SS_and_std[:standard_deviation] isa KeyedArray ? axiskeys(SS_and_std[:standard_deviation],1) isa Vector{String} ? rekey(SS_and_std[:standard_deviation], 1 => axiskeys(SS_and_std[:standard_deviation],1).|> x->Symbol.(replace.(x, "{" => "◖", "}" => "◗"))) : SS_and_std[:standard_deviation] : SS_and_std[:standard_deviation]

    full_NSSS = sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.aux,𝓂.constants.post_model_macro.exo_present))

    full_NSSS[indexin(𝓂.constants.post_model_macro.aux,full_NSSS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.constants.post_model_macro.aux)

    full_SS = [s ∈ 𝓂.constants.post_model_macro.exo_present ? 0.0 : SS_and_std[:non_stochastic_steady_state](s) for s in full_NSSS]

    variables = variables isa String_input ? variables .|> Meta.parse .|> replace_indices : variables

    var_idx = parse_variables_input_to_index(variables, 𝓂.constants) |> unique |> sort

    vars_to_plot = intersect(axiskeys(SS_and_std[:non_stochastic_steady_state])[1],𝓂.constants.post_model_macro.var[var_idx])

    # Sort variables alphabetically by display name
    variable_names_display = [replace_indices_in_symbol.(apply_custom_name(v, rename_dictionary)) for v in vars_to_plot]
    vars_sort_perm = sortperm(variable_names_display, by = normalize_superscript)
    vars_to_plot = vars_to_plot[vars_sort_perm]

    processed_rename_dictionary = process_rename_dictionary(rename_dictionary, 𝓂)

    state_range = collect(range(-SS_and_std[:standard_deviation](state), SS_and_std[:standard_deviation](state), 100)) * σ
    
    state_selector = state .== 𝓂.constants.post_model_macro.var

    # Clear container for new plot
    while length(solution_active_plot_container) > 0
        pop!(solution_active_plot_container)
    end

    if any(x -> contains(string(x), "◖"), full_NSSS)
        full_NSSS_decomposed = decompose_name.(full_NSSS)
        full_NSSS = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in full_NSSS_decomposed]
    end

    # Get steady state for the algorithm
    relevant_SS = get_steady_state(𝓂, algorithm = algorithm, stochastic = algorithm != :first_order, return_variables_only = true, derivatives = false,
                                    tol = opts.tol,
                                    verbose = opts.verbose,
                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm = [opts.sylvester_algorithm², opts.sylvester_algorithm³],
                                    caching = caching,
                                    use_workspaces = use_workspaces)

    full_SS_current = [s ∈ 𝓂.constants.post_model_macro.exo_present ? 0.0 : relevant_SS(s) for s in full_NSSS]

    # Get NSSS (first order steady state) for reference
    NSSS_SS = algorithm == :first_order ? relevant_SS : get_steady_state(𝓂, algorithm = :first_order, return_variables_only = true, derivatives = false,
                                    tol = opts.tol,
                                    verbose = opts.verbose,
                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm = [opts.sylvester_algorithm², opts.sylvester_algorithm³],
                                    caching = caching,
                                    use_workspaces = use_workspaces)

    NSSS = [s ∈ 𝓂.constants.post_model_macro.exo_present ? 0.0 : NSSS_SS(s) for s in full_NSSS]

    SSS_delta = collect(NSSS - full_SS_current)

    # Compute variable responses across state range
    var_state_range = []

    for x in state_range
        if algorithm == :pruned_second_order
            initial_state = [state_selector * x, -SSS_delta]
        elseif algorithm == :pruned_third_order
            initial_state = [state_selector * x, -SSS_delta, zero(SSS_delta)]
        else
            initial_state = collect(full_SS_current) .+ state_selector * x
        end

        push!(var_state_range, get_irf(𝓂, algorithm = algorithm, periods = 1, ignore_obc = ignore_obc, initial_state = initial_state, shocks = :none, levels = true, variables = :all, caching = caching, use_workspaces = use_workspaces)[:,1,1] |> collect)
    end

    var_state_range = hcat(var_state_range...)

    variable_output = []
    has_impact = []

    for k in vars_to_plot
        idx = indexin([k], 𝓂.constants.post_model_macro.var)

        push!(variable_output,  k => var_state_range[idx,:]) 
        
        push!(has_impact,    k => any(abs.(sum(var_state_range[idx,:]) / size(var_state_range, 2) .- var_state_range[idx,:]) .> eps(Float32)))
    end

    # Store data in container
    labels = Dict(  :first_order            => ["1st order perturbation",           "Non-stochastic Steady State"],
                    :second_order           => ["2nd order perturbation",           "Stochastic Steady State (2nd order)"],
                    :pruned_second_order    => ["Pruned 2nd order perturbation",    "Stochastic Steady State (Pruned 2nd order)"],
                    :third_order            => ["3rd order perturbation",           "Stochastic Steady State (3rd order)"],
                    :pruned_third_order     => ["Pruned 3rd order perturbation",    "Stochastic Steady State (Pruned 3rd order)"])

    args_and_kwargs = Dict(:run_id => length(solution_active_plot_container) + 1,
                           :model_name => 𝓂.model_name,
                           :label => label,
                           :state => state,
                           :state_range => state_range,
                           :variables => variables,
                           :algorithm => algorithm,
                           :σ => σ,
                           :parameters => Dict(𝓂.constants.post_complete_parameters.parameters .=> 𝓂.parameter_values),
                           :ignore_obc => ignore_obc,
                           :tol => tol_to_dict(tol, algorithm; needs_covariance = true),
                           :variable_output => variable_output,
                           :has_impact => has_impact,
                           :vars_to_plot => vars_to_plot,
                           :full_SS_current => full_SS_current[indexin(sort(vcat(state, vars_to_plot)), 𝓂.constants.post_model_macro.var)],
                           :algorithm_label => labels[algorithm][1],
                           :ss_label => labels[algorithm][2],
                           :rename_dictionary => processed_rename_dictionary)

    push!(solution_active_plot_container, args_and_kwargs)

    # Generate plots from container
    if !use_workspaces 𝓂.workspaces = orig_ws end

    return plot_solution_from_container(;
                                         show_plots = show_plots,
                                         save_plots = save_plots,
                                         save_plots_format = save_plots_format,
                                         save_plots_name = save_plots_name,
                                         save_plots_path = save_plots_path,
                                         plots_per_page = plots_per_page,
                                         plot_attributes = plot_attributes)
end


# Helper function to generate plots from the solution container
function plot_solution_from_container(;
                                        show_plots::Bool = DEFAULT_SHOW_PLOTS,
                                        save_plots::Bool = DEFAULT_SAVE_PLOTS,
                                        save_plots_format::Symbol = DEFAULT_SAVE_PLOTS_FORMAT,
                                        save_plots_name::Union{String, Symbol} = "solution",
                                        save_plots_path::String = DEFAULT_SAVE_PLOTS_PATH,
                                        plots_per_page::Int = DEFAULT_PLOTS_PER_PAGE_SMALL,
                                        plot_attributes::Dict = Dict())
    
    if length(solution_active_plot_container) == 0
        @warn "No solution data to plot. Call plot_solution first."
        return []
    end
    
    # Get first container element for model info
    first_container = solution_active_plot_container[1]
    model_name = first_container[:model_name]
    
    # Collect all unique states from containers
    joint_states = OrderedSet{String}()
    for container in solution_active_plot_container
        push!(joint_states, string(apply_custom_name.(container[:state], Ref(Dict(container[:rename_dictionary])))))
    end
    
    gr_back, attributes, attributes_redux = setup_plot_attributes(plot_attributes)
    
    pal = build_extended_palette(attributes_redux)
    
    # Create comparison of containers to detect differences
    # Keep relevant keys for comparison: model_name, state, parameters, algorithm, ignore_obc, label
    # Only compare if there are multiple containers
    diffdict = Dict{Symbol,Any}()
    
    if length(solution_active_plot_container) > 1
        check_and_remove_duplicate!(solution_active_plot_container,
            [:parameters, :model_name, :algorithm, :ignore_obc, :tol])

        if length(solution_active_plot_container) == 0
            diffdict[:label] = [solution_active_plot_container[1][:label]]
        else
            diffdict = compute_diffdict(solution_active_plot_container, keys(solution_active_plot_container[end]))
        end
    else
        # For single container, create a diffdict with just the label
        diffdict[:label] = [solution_active_plot_container[1][:label]]
    end
    
    # Build annotation for relevant input differences
    annotate_diff_input = Pair{String,Any}[]
    
    push!(annotate_diff_input, "Plot label" => reduce(vcat, diffdict[:label]))
    
    # Add model name if different
    if haskey(diffdict, :model_name)
        push!(annotate_diff_input, "Model" => reduce(vcat, diffdict[:model_name]))
    end
    
    # Add state if different (though we generally expect same state)
    if haskey(diffdict, :state)
        push!(annotate_diff_input, "State" => reduce(vcat, diffdict[:state]))
    end
    
    # Add algorithm if different
    if haskey(diffdict, :algorithm)
        algo_labels = [String(a) for a in diffdict[:algorithm]]
        push!(annotate_diff_input, "Algorithm" => algo_labels)
    end
    
    # Add parameters if different
    annotate_param_diff!(annotate_diff_input, diffdict)
   
    annotate_rename_dict_diff!(annotate_diff_input, diffdict) 

    # Add ignore_obc if different
    if haskey(diffdict, :ignore_obc)
        push!(annotate_diff_input, "Ignore OBC" => reduce(vcat, diffdict[:ignore_obc]))
    end

    annotate_tol_diff!(annotate_diff_input, solution_active_plot_container)

    # Determine legend labels based on what differs
    # If more than one input differs (besides label), use custom labels from diffdict
    len_diff = length(solution_active_plot_container)
    
    any_custom_label = any([i != v[:label] for (i,v) in enumerate(solution_active_plot_container)])

    # Create legend with 2 columns so dynamics and steady state entries are side by side
    legend_plot = StatsPlots.plot(framestyle = :none, legend = :inside, legend_columns = 2) 
    plt_lab_switch = should_use_label_switch(annotate_diff_input, solution_active_plot_container)
    if plt_lab_switch
        # Multiple differences - use custom labels or plot labels
        for (i, container) in enumerate(solution_active_plot_container)
            label_text = container[:label] isa String ? container[:label] : string(container[:label])
            
            StatsPlots.plot!([NaN], 
                            color = pal[mod1(i, length(pal))],
                            label = string(label_text))
                            
            StatsPlots.scatter!([NaN], 
                                color = pal[mod1(i, length(pal))],
                                label = string(label_text) * " (relevant SS)")
        end
    else
        # Single difference (or just labels differ) - use the relevant input difference in legend
        # Get the legend title and labels from the second entry in annotate_diff_input
        legend_title_dynamics = any_custom_label ? nothing : length(annotate_diff_input) > 1 ? annotate_diff_input[2][1] : nothing
        legend_title_ss = legend_title_dynamics
        
        for (i, container) in enumerate(solution_active_plot_container)
            # For single difference, use the value of that difference as the label
            label_text = if any_custom_label
                container[:label] isa String ? container[:label] : string(container[:label])
            elseif length(annotate_diff_input) > 1
                val = annotate_diff_input[2][2][i]
                val isa String ? val : String(Symbol(val))
            else
                container[:algorithm_label]
            end
            
            StatsPlots.plot!([NaN], 
                            color = pal[mod1(i, length(pal))],
                            legend_title = legend_title_dynamics,
                            label = label_text)

            # For single difference, use the value of that difference as the label
            label_text = if any_custom_label
                (container[:label] isa String ? container[:label] : string(container[:label])) * " (relevant SS)"
            elseif length(annotate_diff_input) > 1
                val = annotate_diff_input[2][2][i]
                (val isa String ? val : String(Symbol(val))) * " (relevant SS)"
            else
                container[:ss_label]
            end
            
            StatsPlots.scatter!([NaN], 
                                color = pal[mod1(i, length(pal))],
                                legend_title = legend_title_ss,
                                label = label_text)
        end
    end
    
    # Collect all variables to plot across all containers
    all_vars = OrderedSet{String}()
    for container in solution_active_plot_container
        foreach(v -> push!(all_vars, v), string.(apply_custom_name.(container[:vars_to_plot], Ref(Dict(container[:rename_dictionary])))))
    end
    
    return_plots = []
    
    # Loop over each state (similar to how plot_irf loops over shocks)
    for state in joint_states
        # Filter containers for this state
        state_containers = [c for c in solution_active_plot_container if string(apply_custom_name.(c[:state], Ref(Dict(c[:rename_dictionary])))) == state]
        
        # Determine which variables have impact in at least one container for this state
        vars_with_impact = []
        for var in setdiff(all_vars, joint_states)
            has_any_impact = false
            for container in state_containers
                for (k,v) in Dict(container[:has_impact])
                    k_trans = string(apply_custom_name(k, (Dict(container[:rename_dictionary]))))
                    if k_trans == var && v
                        has_any_impact = true
                        break
                    end
                end
            end
            if has_any_impact
                push!(vars_with_impact, var)
            end
        end
        
        for var in intersect(joint_states, all_vars)
            push!(vars_with_impact, var)
        end

        n_subplots = length(vars_with_impact)
        pp = []
        pane = 1
        plot_count = 1
        
        # Plot each variable for this state
        for k in vars_with_impact
            Pl = StatsPlots.plot()
    

            # Plot line for each container with this state
            for (i, container) in enumerate(solution_active_plot_container)
                # return the key that corresponds to k in the original variable_output dictionary
                original_k_variable_output = nothing
                for key in keys(Dict(container[:variable_output]))
                    if string(apply_custom_name(key, (Dict(container[:rename_dictionary])))) == k
                        original_k_variable_output = key
                        break
                    end
                end

                # return the key that corresponds to k in the original has_impact dictionary
                original_k_has_impact = nothing
                for key in keys(Dict(container[:has_impact]))
                    if string(apply_custom_name(key, (Dict(container[:rename_dictionary])))) == k
                        original_k_has_impact = key
                        break
                    end
                end

                if string(apply_custom_name.(container[:state], Ref(Dict(container[:rename_dictionary])))) == state && !isnothing(original_k_variable_output) && !isnothing(original_k_has_impact)
                    # Create concatenated transformed variable names for indexing
                    concat_trans_vars = string.(apply_custom_name.(sort(vcat(container[:vars_to_plot], container[:state])), Ref(Dict(container[:rename_dictionary]))))

                    # Find state index in vars_to_plot
                    state_idx = findfirst(==(state), concat_trans_vars)
                    if !isnothing(state_idx)
                        state_ss = container[:full_SS_current][state_idx]
                    else
                        state_ss = 0.0  # fallback
                    end

                    StatsPlots.plot!(container[:state_range] .+ state_ss, 
                        Dict(container[:variable_output])[original_k_variable_output][1,:], 
                        ylabel = replace_indices_in_symbol(Symbol(k))*"₍₀₎", 
                        xlabel = replace_indices_in_symbol(Symbol(state))*"₍₋₁₎", 
                        color = pal[mod1(i, length(pal))],
                        label = "")
                end
            end
            
            # Plot SS markers for each container with this state
            for (i, container) in enumerate(solution_active_plot_container)
                # return the key that corresponds to k in the original variable_output dictionary
                original_k_variable_output = nothing
                for key in keys(Dict(container[:variable_output]))
                    if string(apply_custom_name(key, (Dict(container[:rename_dictionary])))) == k
                        original_k_variable_output = key
                        break
                    end
                end

                # return the key that corresponds to k in the original has_impact dictionary
                original_k_has_impact = nothing
                for key in keys(Dict(container[:has_impact]))
                    if string(apply_custom_name(key, (Dict(container[:rename_dictionary])))) == k
                        original_k_has_impact = key
                        break
                    end
                end

                if string(apply_custom_name.(container[:state], Ref(Dict(container[:rename_dictionary])))) == state && !isnothing(original_k_variable_output) && !isnothing(original_k_has_impact)
                    # Create concatenated transformed variable names for indexing
                    concat_trans_vars = string.(apply_custom_name.(sort(vcat(container[:vars_to_plot], container[:state])), Ref(Dict(container[:rename_dictionary]))))

                    # Get state and variable indices
                    state_idx = findfirst(==(state), concat_trans_vars)
                    var_idx = findfirst(==(k), concat_trans_vars)
                    
                    if !isnothing(state_idx) && !isnothing(var_idx)
                        state_ss = container[:full_SS_current][state_idx]
                        var_ss = container[:full_SS_current][var_idx]
                        
                        StatsPlots.scatter!([state_ss], [var_ss], 
                            color = pal[mod1(i, length(pal))],
                            label = "")
                    end
                end
            end
            
            push!(pp, Pl)
            
            if !(plot_count % plots_per_page == 0)
                plot_count += 1
            else
                plot_count = 1

                state_string = length(joint_states) > 1 ? " State: " * replace_indices_in_symbol(Symbol(state)) : ""
                state_name = replace_indices_in_symbol(Symbol(state))

                pane = assemble_and_emit_page!(
                    return_plots, pp, legend_plot,
                    annotate_diff_input, diffdict,
                    attributes, attributes_redux,
                    pane, n_subplots, plots_per_page,
                    show_plots, save_plots, save_plots_path, save_plots_name, save_plots_format,
                    solution_active_plot_container[1][:model_name];
                    title_extra = state_string,
                    filename_extra = string(state_name),
                    show_diff_table = plt_lab_switch || (any_custom_label && len_diff > 1),
                )
            end
        end
        
        # Handle remaining plots for this state
        if length(pp) > 0
            state_string = length(joint_states) > 1 ? " State: " * replace_indices_in_symbol(Symbol(state)) : ""
            state_name = replace_indices_in_symbol(Symbol(state))

            assemble_and_emit_page!(
                return_plots, pp, legend_plot,
                annotate_diff_input, diffdict,
                attributes, attributes_redux,
                pane, n_subplots, plots_per_page,
                show_plots, save_plots, save_plots_path, save_plots_name, save_plots_format,
                solution_active_plot_container[1][:model_name];
                title_extra = state_string,
                filename_extra = string(state_name),
                show_diff_table = plt_lab_switch || (any_custom_label && len_diff > 1),
                is_tail = true,
            )
        end
    end  # End of state loop
    
    return return_plots
end


"""

$(SIGNATURES)
Add another model variant to the previous plot of the solution. 

Each plot shows the relationship between the chosen state (defined in `state`) and one of the chosen variables (defined in `variables`).

The relevant steady state is plotted along with the mapping from the chosen past state to one present variable per plot. All other (non-chosen) states remain in the relevant steady state.

In the case of pruned higher order solutions there are as many (latent) state vectors as the perturbation order. The first and third order baseline state vectors are the non-stochastic steady state and the second order baseline state vector is the stochastic steady state. Deviations for the chosen state are only added to the first order baseline state. The plot shows the mapping from `σ` standard deviations (first order) added to the first order non-stochastic steady state and the present variables. Note that there is no unique mapping from the "pruned" states and the "actual" reported state. Hence, the plots shown are just one realisation of infinitely many possible mappings.

If the model contains occasionally binding constraints and `ignore_obc = false` they are enforced using shocks.

# Arguments
- $MODEL®
- `state` [Type: `Union{Symbol,String}`]: state variable to be shown on x-axis.
# Keyword Arguments
- $(VARIABLES®(DEFAULT_VARIABLE_SELECTION))
- $ALGORITHM®
- `σ` [Default: `2`, Type: `Union{Int64,Float64}`]: defines the range of the state variable around the (non) stochastic steady state in standard deviations. E.g. a value of 2 means that the state variable is plotted for values of the (non) stochastic steady state in standard deviations +/- 2 standard deviations.
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $IGNORE_OBC®
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMAT®
- $SAVE_PLOTS_PATH®
- $(SAVE_PLOTS_NAME®("solution"))
- `plots_per_page` [Default: `6`, Type: `Int`]: how many plots to show per page
- $PLOT_ATTRIBUTES®
- $RENAME_DICTIONARY®
- $LABEL®
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Vector{Plot}` of individual plots

# Examples
```julia
using MacroModelling, StatsPlots

@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

plot_solution(RBC_CME, :k)

plot_solution!(RBC_CME, :k, algorithm = :pruned_second_order)
```
"""
function plot_solution!(𝓂::ℳ,
                        state::Union{Symbol,String};
                        variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLE_SELECTION,
                        algorithm::Symbol = DEFAULT_ALGORITHM,
                        σ::Union{Int64,Float64} = DEFAULT_SIGMA_RANGE,
                        parameters::ParameterType = nothing,
                        steady_state_function::SteadyStateFunctionType = missing,
                        ignore_obc::Bool = DEFAULT_IGNORE_OBC,
                        label::Union{Real, String, Symbol} = length(solution_active_plot_container) + 1,
                        show_plots::Bool = DEFAULT_SHOW_PLOTS,
                        save_plots::Bool = DEFAULT_SAVE_PLOTS,
                        save_plots_format::Symbol = DEFAULT_SAVE_PLOTS_FORMAT,
                        save_plots_name::Union{String, Symbol} = "solution",
                        save_plots_path::String = DEFAULT_SAVE_PLOTS_PATH,
                        plots_per_page::Int = DEFAULT_PLOTS_PER_PAGE_SMALL,
                        rename_dictionary::AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}} = Dict{Symbol, String}(),
                        plot_attributes::Dict = Dict(),
                        verbose::Bool = DEFAULT_VERBOSE,
                        tol::Tolerances = Tolerances(),
                        quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                        sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                        lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                        caching::Bool = DEFAULT_CACHING,
                        use_workspaces::Bool = DEFAULT_USE_WORKSPACES)
    # @nospecialize # reduce compile time
    
    if !caching invalidate_cache_validity!(𝓂) end
    orig_ws = 𝓂.workspaces
    if !use_workspaces 𝓂.workspaces = fresh_workspaces(orig_ws) end

    # Do NOT clear container - add to existing
    
    opts = merge_calculation_options(tol = tol, verbose = verbose,
                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                        sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                        sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                        lyapunov_algorithm = lyapunov_algorithm)

    warn_irrelevant_tol(tol, algorithm; needs_covariance = true)
    gr_back, attributes, attributes_redux = setup_plot_attributes(plot_attributes)

    state = state isa Symbol ? state : state |> Meta.parse |> replace_indices

    @assert state ∈ 𝓂.constants.post_model_macro.past_not_future_and_mixed "Invalid state. Choose one from:"*repr(replace_indices_in_symbol.(𝓂.constants.post_model_macro.past_not_future_and_mixed))

    @assert algorithm ∈ [:third_order, :pruned_third_order, :second_order, :pruned_second_order, :first_order] "Invalid algorithm. Choose one of: :third_order, :pruned_third_order, :second_order, :pruned_second_order, :first_order"

    ignore_obc, occasionally_binding_constraints, _ = process_ignore_obc_flag(:all_excluding_obc, ignore_obc, 𝓂)
    
    solve!(𝓂, opts = opts, algorithm = algorithm, dynamics = true, parameters = parameters, obc = occasionally_binding_constraints)

    SS_and_std = get_moments(𝓂, 
                            derivatives = false,
                            parameters = parameters,
                            steady_state_function = steady_state_function,
                            variables = :all,
                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                            sylvester_algorithm = sylvester_algorithm,
                            lyapunov_algorithm = lyapunov_algorithm,
                            tol = tol,
                            verbose = verbose,
                            caching = caching,
                            use_workspaces = use_workspaces)

    SS_and_std[:non_stochastic_steady_state] = SS_and_std[:non_stochastic_steady_state] isa KeyedArray ? axiskeys(SS_and_std[:non_stochastic_steady_state],1) isa Vector{String} ? rekey(SS_and_std[:non_stochastic_steady_state], 1 => axiskeys(SS_and_std[:non_stochastic_steady_state],1).|> x->Symbol.(replace.(x, "{" => "◖", "}" => "◗"))) : SS_and_std[:non_stochastic_steady_state] : SS_and_std[:non_stochastic_steady_state]
    
    SS_and_std[:standard_deviation] = SS_and_std[:standard_deviation] isa KeyedArray ? axiskeys(SS_and_std[:standard_deviation],1) isa Vector{String} ? rekey(SS_and_std[:standard_deviation], 1 => axiskeys(SS_and_std[:standard_deviation],1).|> x->Symbol.(replace.(x, "{" => "◖", "}" => "◗"))) : SS_and_std[:standard_deviation] : SS_and_std[:standard_deviation]

    full_NSSS = sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.aux,𝓂.constants.post_model_macro.exo_present))

    full_NSSS[indexin(𝓂.constants.post_model_macro.aux,full_NSSS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.constants.post_model_macro.aux)

    full_SS = [s ∈ 𝓂.constants.post_model_macro.exo_present ? 0.0 : SS_and_std[:non_stochastic_steady_state](s) for s in full_NSSS]

    variables = variables isa String_input ? variables .|> Meta.parse .|> replace_indices : variables

    var_idx = parse_variables_input_to_index(variables, 𝓂.constants) |> unique |> sort

    vars_to_plot = intersect(axiskeys(SS_and_std[:non_stochastic_steady_state])[1],𝓂.constants.post_model_macro.var[var_idx])

    # Sort variables alphabetically by display name
    variable_names_display = [replace_indices_in_symbol.(apply_custom_name(v, rename_dictionary)) for v in vars_to_plot]
    vars_sort_perm = sortperm(variable_names_display, by = normalize_superscript)
    vars_to_plot = vars_to_plot[vars_sort_perm]

    processed_rename_dictionary = process_rename_dictionary(rename_dictionary, 𝓂)

    state_range = collect(range(-SS_and_std[:standard_deviation](state), SS_and_std[:standard_deviation](state), 100)) * σ
    
    state_selector = state .== 𝓂.constants.post_model_macro.var

    if any(x -> contains(string(x), "◖"), full_NSSS)
        full_NSSS_decomposed = decompose_name.(full_NSSS)
        full_NSSS = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in full_NSSS_decomposed]
    end

    # Get steady state for the algorithm
    relevant_SS = get_steady_state(𝓂, algorithm = algorithm, stochastic = algorithm != :first_order, return_variables_only = true, derivatives = false,
                                    tol = opts.tol,
                                    verbose = opts.verbose,
                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm = [opts.sylvester_algorithm², opts.sylvester_algorithm³],
                                    caching = caching,
                                    use_workspaces = use_workspaces)

    full_SS_current = [s ∈ 𝓂.constants.post_model_macro.exo_present ? 0.0 : relevant_SS(s) for s in full_NSSS]

    # Get NSSS (first order steady state) for reference
    NSSS_SS = algorithm == :first_order ? relevant_SS : get_steady_state(𝓂, algorithm = :first_order, return_variables_only = true, derivatives = false,
                                    tol = opts.tol,
                                    verbose = opts.verbose,
                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm = [opts.sylvester_algorithm², opts.sylvester_algorithm³],
                                    caching = caching,
                                    use_workspaces = use_workspaces)

    NSSS = [s ∈ 𝓂.constants.post_model_macro.exo_present ? 0.0 : NSSS_SS(s) for s in full_NSSS]

    SSS_delta = collect(NSSS - full_SS_current)

    # Compute variable responses across state range
    var_state_range = []

    for x in state_range
        if algorithm == :pruned_second_order
            initial_state = [state_selector * x, -SSS_delta]
        elseif algorithm == :pruned_third_order
            initial_state = [state_selector * x, -SSS_delta, zero(SSS_delta)]
        else
            initial_state = collect(full_SS_current) .+ state_selector * x
        end

        push!(var_state_range, get_irf(𝓂, algorithm = algorithm, periods = 1, ignore_obc = ignore_obc, initial_state = initial_state, shocks = :none, levels = true, variables = :all, caching = caching, use_workspaces = use_workspaces)[:,1,1] |> collect)
    end

    var_state_range = hcat(var_state_range...)

    variable_output = []
    has_impact = []

    for k in vars_to_plot
        idx = indexin([k], 𝓂.constants.post_model_macro.var)

        push!(variable_output,  k => var_state_range[idx,:]) 
        
        push!(has_impact,    k => any(abs.(sum(var_state_range[idx,:]) / size(var_state_range, 2) .- var_state_range[idx,:]) .> eps(Float32)))
    end

    # Store data in container
    labels = Dict(  :first_order            => ["1st order perturbation",           "Non-stochastic Steady State"],
                    :second_order           => ["2nd order perturbation",           "Stochastic Steady State (2nd order)"],
                    :pruned_second_order    => ["Pruned 2nd order perturbation",    "Stochastic Steady State (Pruned 2nd order)"],
                    :third_order            => ["3rd order perturbation",           "Stochastic Steady State (3rd order)"],
                    :pruned_third_order     => ["Pruned 3rd order perturbation",    "Stochastic Steady State (Pruned 3rd order)"])

    args_and_kwargs = Dict(:run_id => length(solution_active_plot_container) + 1,
                           :model_name => 𝓂.model_name,
                           :label => label,
                           :state => state,
                           :state_range => state_range,
                           :variables => variables,
                           :algorithm => algorithm,
                           :σ => σ,
                           :parameters => Dict(𝓂.constants.post_complete_parameters.parameters .=> 𝓂.parameter_values),
                           :ignore_obc => ignore_obc,
                           :tol => tol_to_dict(tol, algorithm; needs_covariance = true),
                           :variable_output => variable_output,
                           :has_impact => has_impact,
                           :vars_to_plot => vars_to_plot,
                           :full_SS_current => full_SS_current[indexin(sort(vcat(state, vars_to_plot)), 𝓂.constants.post_model_macro.var)],
                           :algorithm_label => labels[algorithm][1],
                           :ss_label => labels[algorithm][2],
                           :rename_dictionary => processed_rename_dictionary)

    push!(solution_active_plot_container, args_and_kwargs)

    if !use_workspaces 𝓂.workspaces = orig_ws end

    # Generate plots from container
    return plot_solution_from_container(;
                                         show_plots = show_plots,
                                         save_plots = save_plots,
                                         save_plots_format = save_plots_format,
                                         save_plots_name = save_plots_name,
                                         save_plots_path = save_plots_path,
                                         plots_per_page = plots_per_page,
                                         plot_attributes = plot_attributes)
end


"""
$(SIGNATURES)
Plot the conditional forecast given restrictions on endogenous variables and shocks (optional). By default, the values represent absolute deviations from the relevant steady state (see `levels` for details). The non-stochastic steady state (NSSS) is relevant for first order solutions and the stochastic steady state for higher order solutions. A constrained minimisation problem is solved to find the combination of shocks with the smallest squared magnitude fulfilling the conditions.

The left axis shows the level, and the right axis the deviation from the relevant steady state. The horizontal black line indicates the relevant steady state. Variable names are above the subplots and the title provides information about the model, shocks and number of pages per shock.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- $CONDITIONS®
# Keyword Arguments
- $SHOCK_CONDITIONS®
- $INITIAL_STATE®
- `periods` [Default: `40`, Type: `Int`]: the total number of periods is the sum of the argument provided here and the maximum of periods of the shocks or conditions argument.
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $(VARIABLES®(DEFAULT_VARIABLES_EXCLUDING_OBC))
- $CONDITIONS_IN_LEVELS®
- $ALGORITHM®
- `label` [Default: `1`, Type: `Union{Real, String, Symbol}`]: label to attribute to this function call in the plots.
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMAT®
- $SAVE_PLOTS_PATH®
- $(SAVE_PLOTS_NAME®("conditional_forecast"))
- $(PLOTS_PER_PAGE®(DEFAULT_PLOTS_PER_PAGE_LARGE))
- $RENAME_DICTIONARY®
- $PLOT_ATTRIBUTES®
- `label` [Default: `1`, Type: `Union{Real, String, Symbol}`]: label to attribute to this function call in the plots.
- $QME®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Vector{Plot}` of individual plots

# Examples
```julia
using MacroModelling, StatsPlots

@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

# c is conditioned to deviate by 0.01 in period 1 and y is conditioned to deviate by 0.02 in period 3
conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,3),Variables = [:c,:y], Periods = 1:3)
conditions[1,1] = .01
conditions[2,3] = .02

# in period 2 second shock (eps_z) is conditioned to take a value of 0.05
shocks = Matrix{Union{Nothing,Float64}}(undef,2,1)
shocks[1,1] = .05

plot_conditional_forecast(RBC_CME, conditions, shocks = shocks, conditions_in_levels = false)

# The same can be achieved with the other input formats:
# conditions = Matrix{Union{Nothing,Float64}}(undef,7,2)
# conditions[4,1] = .01
# conditions[6,2] = .02

# using SparseArrays
# conditions = spzeros(7,2)
# conditions[4,1] = .01
# conditions[6,2] = .02

# shocks = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,1),Variables = [:delta_eps], Periods = [1])
# shocks[1,1] = .05

# using SparseArrays
# shocks = spzeros(2,1)
# shocks[1,1] = .05
```
"""
function plot_conditional_forecast(𝓂::ℳ,
                                    conditions::Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}};
                                    shocks::Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}, Nothing} = nothing, 
                                    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}} = DEFAULT_INITIAL_STATE,
                                    periods::Int = DEFAULT_PERIODS, 
                                    parameters::ParameterType = nothing,
                                    steady_state_function::SteadyStateFunctionType = missing,
                                    variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_OBC, 
                                    conditions_in_levels::Bool = DEFAULT_CONDITIONS_IN_LEVELS,
                                    algorithm::Symbol = DEFAULT_ALGORITHM,
                                    label::Union{Real, String, Symbol} = DEFAULT_LABEL,
                                    show_plots::Bool = DEFAULT_SHOW_PLOTS,
                                    save_plots::Bool = DEFAULT_SAVE_PLOTS,
                                    save_plots_format::Symbol = DEFAULT_SAVE_PLOTS_FORMAT,
                                    save_plots_name::Union{String, Symbol} = "conditional_forecast",
                                    save_plots_path::String = DEFAULT_SAVE_PLOTS_PATH,
                                    plots_per_page::Int = DEFAULT_PLOTS_PER_PAGE_LARGE,
                                    rename_dictionary::AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}} = Dict{Symbol, String}(),
                                    plot_attributes::Dict = Dict(),
                                    verbose::Bool = DEFAULT_VERBOSE,
                                    tol::Tolerances = Tolerances(),
                                    quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                                    sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                                    caching::Bool = DEFAULT_CACHING,
                                    use_workspaces::Bool = DEFAULT_USE_WORKSPACES)
    # @nospecialize # reduce compile time
    
    if !caching invalidate_cache_validity!(𝓂) end
    orig_ws = 𝓂.workspaces
    if !use_workspaces 𝓂.workspaces = fresh_workspaces(orig_ws) end

    gr_back, attributes, attributes_redux = setup_plot_attributes(plot_attributes)

    initial_state_input = copy(initial_state)

    periods_input = max(periods, size(conditions,2), isnothing(shocks) ? 1 : size(shocks,2))

    conditions = conditions isa KeyedArray ? axiskeys(conditions,1) isa Vector{String} ? rekey(conditions, 1 => axiskeys(conditions,1) .|> Meta.parse .|> replace_indices) : conditions : conditions

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    Y = get_conditional_forecast(𝓂,
                                conditions,
                                shocks = shocks, 
                                initial_state = initial_state,
                                periods = periods, 
                                parameters = parameters,
                                steady_state_function = steady_state_function,
                                variables = variables, 
                                conditions_in_levels = conditions_in_levels,
                                algorithm = algorithm,
                                # levels = levels,
                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                sylvester_algorithm = sylvester_algorithm,
                                tol = tol,
                                verbose = verbose,
                                caching = caching,
                                use_workspaces = use_workspaces)
    warn_irrelevant_tol(tol, algorithm; needs_covariance = true)

    periods += max(size(conditions,2), isnothing(shocks) ? 1 : size(shocks,2))

    full_SS = vcat(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.aux,𝓂.constants.post_model_macro.exo_present)),map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo))

    full_var_SS = full_SS isa Vector{String} ? full_SS .|> Meta.parse .|> replace_indices : deepcopy(full_SS)

    var_names = axiskeys(Y,1)   

    var_names = var_names isa Vector{String} ? var_names .|> replace_indices : var_names

    var_idx = indexin(var_names,full_SS)

    # if length(intersect(𝓂.constants.post_model_macro.aux,var_names)) > 0
    #     for v in 𝓂.constants.post_model_macro.aux
    #         idx = indexin([v],var_names)
    #         if !isnothing(idx[1])
    #             var_names[idx[1]] = Symbol(replace(string(v), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
    #         end
    #     end
    #     # var_names[indexin(𝓂.constants.post_model_macro.aux,var_names)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.constants.post_model_macro.aux)
    # end
    
    relevant_SS = get_steady_state(𝓂, algorithm = algorithm, return_variables_only = true, derivatives = false,
                                    tol = tol,
                                    verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm = sylvester_algorithm,
                                    caching = caching,
                                    use_workspaces = use_workspaces)

    relevant_SS = relevant_SS isa KeyedArray ? axiskeys(relevant_SS,1) isa Vector{String} ? rekey(relevant_SS, 1 => axiskeys(relevant_SS,1) .|> Meta.parse .|> replace_indices) : relevant_SS : relevant_SS

    full_var_SS_copy = deepcopy(full_var_SS)

    if length(intersect(𝓂.constants.post_model_macro.aux,full_var_SS_copy)) > 0
        for v in 𝓂.constants.post_model_macro.aux
            idx = indexin([v],full_var_SS_copy)
            if !isnothing(idx[1])
                full_var_SS_copy[idx[1]] = Symbol(replace(string(v), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
            end
        end
        # var_names[indexin(𝓂.constants.post_model_macro.aux,var_names)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.constants.post_model_macro.aux)
    end

    reference_steady_state = [s ∈ union(map(x -> Symbol(string(x) * "₍ₓ₎"), 𝓂.constants.post_model_macro.exo), 𝓂.constants.post_model_macro.exo_present) ? 0.0 : relevant_SS(s) for s in full_var_SS_copy]

    var_length = length(full_SS) - 𝓂.constants.post_model_macro.nExo
    
    if conditions isa SparseMatrixCSC{Float64}
        @assert var_length == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(var_length) * " variables (including auxiliary variables): " * repr(full_var_SS)

        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,var_length,periods)
        nzs = findnz(conditions)
        for i in 1:length(nzs[1])
            cond_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
        end
        conditions = cond_tmp
    elseif conditions isa Matrix{Union{Nothing,Float64}}
        @assert var_length == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(var_length) * " variables (including auxiliary variables): " * repr(full_var_SS)

        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,var_length,periods)
        cond_tmp[:,axes(conditions,2)] = conditions
        conditions = cond_tmp
    elseif conditions isa KeyedArray{Union{Nothing,Float64}} || conditions isa KeyedArray{Float64}
        @assert length(setdiff(axiskeys(conditions,1),full_SS)) == 0 "The following symbols in the first axis of the conditions matrix are not part of the model: " * repr(setdiff(axiskeys(conditions,1),full_SS))
        
        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,var_length,periods)
        cond_tmp[indexin(sort(axiskeys(conditions,1)),full_SS),axes(conditions,2)] .= conditions(sort(axiskeys(conditions,1)))
        conditions = cond_tmp
    end
    
    if shocks isa SparseMatrixCSC{Float64}
        @assert length(𝓂.constants.post_model_macro.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.constants.post_model_macro.exo)) * " shocks: " * repr(𝓂.constants.post_model_macro.exo)

        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.constants.post_model_macro.exo),periods)
        nzs = findnz(shocks)
        for i in 1:length(nzs[1])
            shocks_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
        end
        shocks = shocks_tmp
    elseif shocks isa Matrix{Union{Nothing,Float64}}
        @assert length(𝓂.constants.post_model_macro.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.constants.post_model_macro.exo)) * " shocks: " * repr(𝓂.constants.post_model_macro.exo)

        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.constants.post_model_macro.exo),periods)
        shocks_tmp[:,axes(shocks,2)] = shocks
        shocks = shocks_tmp
    elseif shocks isa KeyedArray{Union{Nothing,Float64}} || shocks isa KeyedArray{Float64}
        @assert length(setdiff(axiskeys(shocks,1),𝓂.constants.post_model_macro.exo)) == 0 "The following symbols in the first axis of the shocks matrix are not part of the model: " * repr(setdiff(axiskeys(shocks,1),𝓂.constants.post_model_macro.exo))
        
        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.constants.post_model_macro.exo),periods)
        shocks_tmp[indexin(sort(axiskeys(shocks,1)),𝓂.constants.post_model_macro.exo),axes(shocks,2)] .= shocks(sort(axiskeys(shocks,1)))
        shocks = shocks_tmp
    elseif isnothing(shocks)
        shocks = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.constants.post_model_macro.exo),periods)
    end

    while length(conditional_forecast_active_plot_container) > 0
        pop!(conditional_forecast_active_plot_container)
    end

    # Create display names for variables and shocks
    full_variable_names_display = [(apply_custom_name(replace_indices_in_symbol(v), rename_dictionary)) for v in full_var_SS if v ∉ map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)]
    full_shock_names_display = [(apply_custom_name(replace_indices_in_symbol(s), rename_dictionary)) for s in full_var_SS if s ∈ map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)]

    @assert length(unique([v for v in full_var_SS if v ∉ map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)])) == length(unique(full_variable_names_display)) "Renaming variables resulted in non-unique names. Please check the `rename_dictionary`."
    @assert length(unique([v for v in full_var_SS if v ∈ map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)])) == length(unique(full_shock_names_display)) "Renaming shocks resulted in non-unique names. Please check the `rename_dictionary`."

    variable_names_display = [apply_custom_name(replace_indices_in_symbol(v), rename_dictionary) for v in var_names if v ∉ map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)]
    shock_names_display = [String(apply_custom_name(Symbol(replace(string(replace_indices_in_symbol(s)), "₍ₓ₎" => "")), rename_dictionary)) * "₍ₓ₎" for s in var_names if s ∈ map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)]
    
    # Get sorting permutations for variables and shocks separately
    var_sort_perm = sortperm(variable_names_display, by = normalize_superscript)
    shock_sort_perm = sortperm(shock_names_display, by = normalize_superscript)

    # Get sorting permutations for variables and shocks separately
    full_var_sort_perm = sortperm(full_variable_names_display, by = normalize_superscript)
    full_shock_sort_perm = sortperm(full_shock_names_display, by = normalize_superscript)

    # Process rename dictionary to only include relevant keys in sorted order
    processed_rename_dictionary = process_rename_dictionary(rename_dictionary, 𝓂)

    # Combine sorted indices
    combined_sort_perm = vcat(var_sort_perm, (length(variable_names_display) .+ (1:length(shock_names_display)))[shock_sort_perm])
    full_combined_sort_perm = vcat(full_var_sort_perm, (length(full_variable_names_display) .+ (1:length(full_shock_names_display)))[full_shock_sort_perm])

    # Apply the combined permutation to all relevant arrays
    Y = Y[combined_sort_perm, :]
    # conditions = conditions[full_var_sort_perm, :]
    # shocks = shocks[full_shock_sort_perm, :]
    # reference_steady_state = reference_steady_state[full_combined_sort_perm]
    var_idx = var_idx[combined_sort_perm]
    var_names_sorted = var_names[var_sort_perm]
    shock_names_sorted = var_names[(length(variable_names_display) .+ (1:length(shock_names_display)))[shock_sort_perm]]

    # Get the sorted display names
    # sorted_variable_names_display = sort(variable_names_display)
    sorted_shock_names_display = sort(shock_names_display)

    args_and_kwargs = Dict(:run_id => length(conditional_forecast_active_plot_container) + 1,
                           :model_name => 𝓂.model_name,
                           :label => label,

                           :conditions => conditions[:,1:periods_input],
                           :conditions_in_levels => conditions_in_levels,
                           :shocks => shocks[:,1:periods_input],
                           :initial_state => initial_state_input,
                           :periods => periods_input,
                           :parameters => Dict(𝓂.constants.post_complete_parameters.parameters .=> 𝓂.parameter_values),
                           :variables => variables,
                           :var_idx => var_idx,
                           :algorithm => algorithm,

                           :tol => tol_to_dict(tol, algorithm; needs_covariance = true),

                           :quadratic_matrix_equation_algorithm => quadratic_matrix_equation_algorithm,
                           :sylvester_algorithm => sylvester_algorithm,

                           :plot_data => Y,
                           :reference_steady_state => reference_steady_state,
                           :variable_names => var_names_sorted, # Use the new sorted variable names
                           :shock_names => shock_names_sorted,       # Use the new sorted shock names
                           :rename_dictionary => processed_rename_dictionary
                           )

    push!(conditional_forecast_active_plot_container, args_and_kwargs)

    pal = build_extended_palette(attributes_redux)

    n_subplots = length(var_idx)
    pp = []
    pane = 1
    plot_count = 1

    return_plots = []

    for (i,v) in enumerate(var_idx)
        if all(isapprox.(Y[i,:], 0, atol = eps(Float32))) && !(any(vcat(conditions,shocks)[v,:] .!= nothing))
            n_subplots -= 1
        end
    end
    
    for (i,v) in enumerate(var_idx)
        SS = reference_steady_state[v]

        if !(all(isapprox.(Y[i,:],0,atol = eps(Float32)))) || length(findall(vcat(conditions,shocks)[v,:] .!= nothing)) > 0

            cond_idx = findall(vcat(conditions,shocks)[v,:] .!= nothing)

            if replace(string(full_SS[v]), "₍ₓ₎" => "") == string(full_SS[v])
                subplot_title = apply_custom_name(replace_indices_in_symbol(full_SS[v]), rename_dictionary)
            else
                subplot_title = apply_custom_name(replace(string(replace_indices_in_symbol(full_SS[v])), "₍ₓ₎" => ""), rename_dictionary) * "₍ₓ₎"
            end

            p = standard_subplot(Y[i,:], SS, subplot_title, gr_back, pal = pal)
            
            if length(cond_idx) > 0
                StatsPlots.scatter!(p,
                                    cond_idx, 
                                    conditions_in_levels ? vcat(conditions,shocks)[v,cond_idx] : vcat(conditions,shocks)[v,cond_idx] .+ SS, 
                                    label = "",
                                    markerstrokewidth = 0,
                                    marker = gr_back ? :star8 : :pentagon, 
                                    markercolor = :black)
            end

            push!(pp, p)

            if !(plot_count % plots_per_page == 0)
                plot_count += 1
            else
                plot_count = 1

                shock_string = "Conditional forecast"

                ppp = StatsPlots.plot(pp...; attributes...)

                pp = StatsPlots.scatter([NaN], 
                                        label = "Condition", 
                                        marker = gr_back ? :star8 : :pentagon,
                                        markercolor = :black,
                                        markerstrokewidth = 0,
                                        framestyle = :none, 
                                        legend = :inside)
                                        
                p = StatsPlots.plot(ppp,pp, 
                                        layout = StatsPlots.grid(2, 1, heights=[0.99, 0.01]),
                                        plot_title = "Model: "*𝓂.model_name*"        " * shock_string * "  ("*string(pane) * "/" * string(Int(ceil(n_subplots/plots_per_page)))*")"; 
                                        attributes_redux...)
                
                push!(return_plots,p)

                if show_plots# & (length(pp) > 0)
                    display(p)
                end

                if save_plots# & (length(pp) > 0)
                    if !isdir(save_plots_path) mkpath(save_plots_path) end

                    StatsPlots.savefig(p, save_plots_path * "/" * string(save_plots_name) * "__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
                end

                pane += 1
                pp = []
            end
        end
    end

    if length(pp) > 0
        shock_string = "Conditional forecast"

        ppp = StatsPlots.plot(pp...; attributes...)

        pp = StatsPlots.scatter([NaN], 
                                label = "Condition", 
                                marker = gr_back ? :star8 : :pentagon,
                                markercolor = :black,
                                markerstrokewidth = 0,
                                framestyle = :none, 
                                legend = :inside)
                                
        p = StatsPlots.plot(ppp,pp, 
                                layout = StatsPlots.grid(2, 1, heights=[0.99, 0.01]),
                                plot_title = "Model: "*𝓂.model_name*"        " * shock_string * "  (" * string(pane) * "/" * string(Int(ceil(n_subplots/plots_per_page)))*")"; 
                                attributes_redux...)
        
        push!(return_plots,p)

        if show_plots
            display(p)
        end

        if save_plots
            if !isdir(save_plots_path) mkpath(save_plots_path) end

            StatsPlots.savefig(p, save_plots_path * "/" * string(save_plots_name) * "__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
        end
    end

    if !use_workspaces 𝓂.workspaces = orig_ws end

    return return_plots
end



"""
$(SIGNATURES)
This function allows comparison or stacking of conditional forecasts for any combination of inputs.

This function shares most of the signature and functionality of [`plot_conditional_forecast`](@ref). Its main purpose is to append plots based on the inputs to previous calls of this function and the last call of [`plot_conditional_forecast`](@ref). In the background it keeps a registry of the inputs and outputs and then plots the comparison or stacks the output.

# Arguments
- $MODEL®
- $CONDITIONS®
# Keyword Arguments
- $SHOCK_CONDITIONS®
- $INITIAL_STATE®
- `periods` [Default: `40`, Type: `Int`]: the total number of periods is the sum of the argument provided here and the maximum of periods of the shocks or conditions argument.
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $(VARIABLES®(DEFAULT_VARIABLES_EXCLUDING_OBC))
- $CONDITIONS_IN_LEVELS®
- $ALGORITHM®
- $LABEL®
- $RENAME_DICTIONARY®
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMAT®
- $SAVE_PLOTS_PATH®
- $(SAVE_PLOTS_NAME®("conditional_forecast"))
- $(PLOTS_PER_PAGE®(DEFAULT_PLOTS_PER_PAGE_SMALL))
- $PLOT_ATTRIBUTES®
- `plot_type` [Default: `:compare`, Type: `Symbol`]: plot type used to represent results. `:compare` means results are shown as separate lines. `:stack` means results are stacked.
- `transparency` [Default: `$DEFAULT_TRANSPARENCY`, Type: `Float64`]: transparency of stacked bars. Only relevant if `plot_type` is `:stack`.
- $QME®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Vector{Plot}` of individual plots

# Examples
```julia
using MacroModelling, StatsPlots

@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

# c is conditioned to deviate by 0.01 in period 1 and y is conditioned to deviate by 0.02 in period 3
conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,3),Variables = [:c,:y], Periods = 1:3)
conditions[1,1] = .01
conditions[2,3] = .02

# in period 2 second shock (eps_z) is conditioned to take a value of 0.05
shocks = Matrix{Union{Nothing,Float64}}(undef,2,1)
shocks[1,1] = .05

plot_conditional_forecast(RBC_CME, conditions, shocks = shocks, conditions_in_levels = false)

conditions = Matrix{Union{Nothing,Float64}}(undef,7,2)
conditions[4,2] = .01
conditions[6,1] = .03

plot_conditional_forecast!(RBC_CME, conditions, shocks = shocks, conditions_in_levels = false)

plot_conditional_forecast!(RBC_CME, conditions, shocks = shocks, conditions_in_levels = false, plot_type = :stack)


plot_conditional_forecast(RBC_CME, conditions, conditions_in_levels = false)

plot_conditional_forecast!(RBC_CME, conditions, conditions_in_levels = false, algorithm = :second_order)


plot_conditional_forecast(RBC_CME, conditions, conditions_in_levels = false)

plot_conditional_forecast!(RBC_CME, conditions, conditions_in_levels = false, parameters = :beta => 0.99)
```
"""
function plot_conditional_forecast!(𝓂::ℳ,
                                    conditions::Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}};
                                    shocks::Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}, Nothing} = nothing, 
                                    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}} = DEFAULT_INITIAL_STATE,
                                    periods::Int = DEFAULT_PERIODS, 
                                    parameters::ParameterType = nothing,
                                    steady_state_function::SteadyStateFunctionType = missing,
                                    variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_OBC, 
                                    conditions_in_levels::Bool = DEFAULT_CONDITIONS_IN_LEVELS,
                                    algorithm::Symbol = DEFAULT_ALGORITHM,
                                    label::Union{Real, String, Symbol} = length(conditional_forecast_active_plot_container) + 1,
                                    show_plots::Bool = DEFAULT_SHOW_PLOTS,
                                    save_plots::Bool = DEFAULT_SAVE_PLOTS,
                                    save_plots_format::Symbol = DEFAULT_SAVE_PLOTS_FORMAT,
                                    save_plots_name::Union{String, Symbol} = "conditional_forecast",
                                    save_plots_path::String = DEFAULT_SAVE_PLOTS_PATH,
                                    plots_per_page::Int = DEFAULT_PLOTS_PER_PAGE_SMALL,
                                    plot_attributes::Dict = Dict(),
                                    plot_type::Symbol = DEFAULT_PLOT_TYPE,
                                    transparency::Float64 = DEFAULT_TRANSPARENCY,
                                    rename_dictionary::AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}} = Dict{Symbol, String}(),
                                    verbose::Bool = DEFAULT_VERBOSE,
                                    tol::Tolerances = Tolerances(),
                                    quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                                    sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                                    caching::Bool = DEFAULT_CACHING,
                                    use_workspaces::Bool = DEFAULT_USE_WORKSPACES)
    # @nospecialize # reduce compile time
                 
    if !caching invalidate_cache_validity!(𝓂) end
    orig_ws = 𝓂.workspaces
    if !use_workspaces 𝓂.workspaces = fresh_workspaces(orig_ws) end

    @assert plot_type ∈ [:compare, :stack] "plot_type must be either :compare or :stack"
                   
    gr_back, attributes, attributes_redux = setup_plot_attributes(plot_attributes)

    initial_state_input = copy(initial_state)

    periods_input = max(periods, size(conditions,2), isnothing(shocks) ? 1 : size(shocks,2))

    conditions = conditions isa KeyedArray ? axiskeys(conditions,1) isa Vector{String} ? rekey(conditions, 1 => axiskeys(conditions,1) .|> Meta.parse .|> replace_indices) : conditions : conditions

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    Y = get_conditional_forecast(𝓂,
                                conditions,
                                shocks = shocks, 
                                initial_state = initial_state,
                                periods = periods, 
                                parameters = parameters,
                                steady_state_function = steady_state_function,
                                variables = variables, 
                                conditions_in_levels = conditions_in_levels,
                                algorithm = algorithm,
                                # levels = levels,
                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                sylvester_algorithm = sylvester_algorithm,
                                tol = tol,
                                verbose = verbose,
                                caching = caching,
                                use_workspaces = use_workspaces)

    warn_irrelevant_tol(tol, algorithm; needs_covariance = true)
    periods += max(size(conditions,2), isnothing(shocks) ? 1 : size(shocks,2))

    full_SS = vcat(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.aux,𝓂.constants.post_model_macro.exo_present)),map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo))

    full_var_SS = full_SS isa Vector{String} ? full_SS .|> Meta.parse .|> replace_indices : deepcopy(full_SS)

    var_names = axiskeys(Y,1)   

    var_names = var_names isa Vector{String} ? var_names .|> replace_indices : var_names

    var_idx = indexin(var_names,full_SS)

    # if length(intersect(𝓂.constants.post_model_macro.aux,var_names)) > 0
    #     for v in 𝓂.constants.post_model_macro.aux
    #         idx = indexin([v],var_names)
    #         if !isnothing(idx[1])
    #             var_names[idx[1]] = Symbol(replace(string(v), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
    #         end
    #     end
    #     # var_names[indexin(𝓂.constants.post_model_macro.aux,var_names)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.constants.post_model_macro.aux)
    # end
    
    relevant_SS = get_steady_state(𝓂, algorithm = algorithm, return_variables_only = true, derivatives = false,
                                    tol = tol,
                                    verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm = sylvester_algorithm,
                                    caching = caching,
                                    use_workspaces = use_workspaces)

    relevant_SS = relevant_SS isa KeyedArray ? axiskeys(relevant_SS,1) isa Vector{String} ? rekey(relevant_SS, 1 => axiskeys(relevant_SS,1) .|> Meta.parse .|> replace_indices) : relevant_SS : relevant_SS

    full_var_SS_copy = deepcopy(full_var_SS)

    if length(intersect(𝓂.constants.post_model_macro.aux,full_var_SS_copy)) > 0
        for v in 𝓂.constants.post_model_macro.aux
            idx = indexin([v],full_var_SS_copy)
            if !isnothing(idx[1])
                full_var_SS_copy[idx[1]] = Symbol(replace(string(v), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
            end
        end
        # var_names[indexin(𝓂.constants.post_model_macro.aux,var_names)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.constants.post_model_macro.aux)
    end

    reference_steady_state = [s ∈ union(map(x -> Symbol(string(x) * "₍ₓ₎"), 𝓂.constants.post_model_macro.exo), 𝓂.constants.post_model_macro.exo_present) ? 0.0 : relevant_SS(s) for s in full_var_SS_copy]

    var_length = length(full_SS) - 𝓂.constants.post_model_macro.nExo
    
    if conditions isa SparseMatrixCSC{Float64}
        @assert var_length == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(var_length) * " variables (including auxiliary variables): " * repr(var_names)

        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,var_length,periods)
        nzs = findnz(conditions)
        for i in 1:length(nzs[1])
            cond_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
        end
        conditions = cond_tmp
    elseif conditions isa Matrix{Union{Nothing,Float64}}
        @assert var_length == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(var_length) * " variables (including auxiliary variables): " * repr(var_names)

        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,var_length,periods)
        cond_tmp[:,axes(conditions,2)] = conditions
        conditions = cond_tmp
    elseif conditions isa KeyedArray{Union{Nothing,Float64}} || conditions isa KeyedArray{Float64}
        @assert length(setdiff(axiskeys(conditions,1),full_SS)) == 0 "The following symbols in the first axis of the conditions matrix are not part of the model: " * repr(setdiff(axiskeys(conditions,1),full_SS))
        
        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,var_length,periods)
        cond_tmp[indexin(sort(axiskeys(conditions,1)),full_SS),axes(conditions,2)] .= conditions(sort(axiskeys(conditions,1)))
        conditions = cond_tmp
    end

    if shocks isa SparseMatrixCSC{Float64}
        @assert length(𝓂.constants.post_model_macro.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.constants.post_model_macro.exo)) * " shocks: " * repr(𝓂.constants.post_model_macro.exo)

        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.constants.post_model_macro.exo),periods)
        nzs = findnz(shocks)
        for i in 1:length(nzs[1])
            shocks_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
        end
        shocks = shocks_tmp
    elseif shocks isa Matrix{Union{Nothing,Float64}}
        @assert length(𝓂.constants.post_model_macro.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.constants.post_model_macro.exo)) * " shocks: " * repr(𝓂.constants.post_model_macro.exo)

        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.constants.post_model_macro.exo),periods)
        shocks_tmp[:,axes(shocks,2)] = shocks
        shocks = shocks_tmp
    elseif shocks isa KeyedArray{Union{Nothing,Float64}} || shocks isa KeyedArray{Float64}
        @assert length(setdiff(axiskeys(shocks,1),𝓂.constants.post_model_macro.exo)) == 0 "The following symbols in the first axis of the shocks matrix are not part of the model: " * repr(setdiff(axiskeys(shocks,1),𝓂.constants.post_model_macro.exo))
        
        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.constants.post_model_macro.exo),periods)
        shocks_tmp[indexin(sort(axiskeys(shocks,1)),𝓂.constants.post_model_macro.exo),axes(shocks,2)] .= shocks(sort(axiskeys(shocks,1)))
        shocks = shocks_tmp
    elseif isnothing(shocks)
        shocks = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.constants.post_model_macro.exo),periods)
    end

    # Create display names for variables and shocks
    full_variable_names_display = [(apply_custom_name(replace_indices_in_symbol(v), rename_dictionary)) for v in full_var_SS if v ∉ map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)]
    full_shock_names_display = [(apply_custom_name(replace_indices_in_symbol(s), rename_dictionary)) for s in full_var_SS if s ∈ map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)]

    @assert length(unique([v for v in full_var_SS if v ∉ map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)])) == length(unique(full_variable_names_display)) "Renaming variables resulted in non-unique names. Please check the `rename_dictionary`."
    @assert length(unique([v for v in full_var_SS if v ∈ map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)])) == length(unique(full_shock_names_display)) "Renaming shocks resulted in non-unique names. Please check the `rename_dictionary`."

    variable_names_display = [apply_custom_name(replace_indices_in_symbol(v), rename_dictionary) for v in var_names if v ∉ map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)]
    shock_names_display = [String(apply_custom_name(Symbol(replace(string(replace_indices_in_symbol(s)), "₍ₓ₎" => "")), rename_dictionary)) * "₍ₓ₎" for s in var_names if s ∈ map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)]

    # Get sorting permutations for variables and shocks separately
    var_sort_perm = sortperm(variable_names_display, by = normalize_superscript)
    shock_sort_perm = sortperm(shock_names_display, by = normalize_superscript)

    # Get sorting permutations for variables and shocks separately
    full_var_sort_perm = sortperm(full_variable_names_display, by = normalize_superscript)
    full_shock_sort_perm = sortperm(full_shock_names_display, by = normalize_superscript)

    # Process rename dictionary to only include relevant keys in sorted order
    processed_rename_dictionary = process_rename_dictionary(rename_dictionary, 𝓂)

    # Combine sorted indices
    combined_sort_perm = vcat(var_sort_perm, (length(variable_names_display) .+ (1:length(shock_names_display)))[shock_sort_perm])
    full_combined_sort_perm = vcat(full_var_sort_perm, (length(full_variable_names_display) .+ (1:length(full_shock_names_display)))[full_shock_sort_perm])

    # Apply the combined permutation to all relevant arrays
    Y = Y[combined_sort_perm, :]
    # conditions = conditions[full_var_sort_perm, :]
    # shocks = shocks[full_shock_sort_perm, :]
    # reference_steady_state = reference_steady_state[full_combined_sort_perm]
    var_idx = var_idx[combined_sort_perm]
    var_names_sorted = var_names[var_sort_perm]
    shock_names_sorted = var_names[(length(variable_names_display) .+ (1:length(shock_names_display)))[shock_sort_perm]]

    # Get the sorted display names
    # sorted_variable_names_display = sort(variable_names_display)
    sorted_shock_names_display = sort(shock_names_display)

    pal = build_extended_palette(attributes_redux)

    args_and_kwargs = Dict(:run_id => length(conditional_forecast_active_plot_container) + 1,
                           :model_name => 𝓂.model_name,
                           :label => label,

                           :conditions => conditions[:,1:periods_input],
                           :conditions_in_levels => conditions_in_levels,
                           :shocks => shocks[:,1:periods_input],
                           :initial_state => initial_state_input,
                           :periods => periods_input,
                           :parameters => Dict(𝓂.constants.post_complete_parameters.parameters .=> 𝓂.parameter_values),
                           :variables => variables,
                           :var_idx => var_idx,
                           :algorithm => algorithm,

                           :tol => tol_to_dict(tol, algorithm; needs_covariance = true),

                           :quadratic_matrix_equation_algorithm => quadratic_matrix_equation_algorithm,
                           :sylvester_algorithm => sylvester_algorithm,

                           :plot_data => Y,
                           :reference_steady_state => reference_steady_state,
                           :variable_names => var_names_sorted, # Use the new sorted variable names
                           :shock_names => shock_names_sorted,       # Use the new sorted shock names
                           :rename_dictionary => processed_rename_dictionary
                           )
                           
    push_if_no_duplicate!(conditional_forecast_active_plot_container, args_and_kwargs,
        [:parameters, :rename_dictionary, :conditions, :shocks, :initial_state, :tol])

    diffdict = compute_diffdict(conditional_forecast_active_plot_container, keys(args_and_kwargs))
    
    annotate_ss = Vector{Pair{String, Any}}[]

    annotate_ss_page = Pair{String,Any}[]

    annotate_diff_input = Pair{String,Any}[]

    push!(annotate_diff_input, "Plot label" => reduce(vcat, diffdict[:label]))

    len_diff = length(conditional_forecast_active_plot_container)

    annotate_param_diff!(annotate_diff_input, diffdict)

    if haskey(diffdict, :shocks)
        shocks = diffdict[:shocks]
        
        labels = String[]                      # "" for trivial, "#k" otherwise
        seen   = []
        next_idx = 0

        for shock_mat in shocks
            if isnothing(shock_mat)
                push!(labels, "")
                continue
            end

            # Catch the all-nothing case here
            lastcol = findlast(j -> any(x -> x !== nothing, shock_mat[:, j]), axes(shock_mat, 2))
            
            if isnothing(lastcol)
                push!(labels, "nothing")
                continue
            end

            view_mat = shock_mat[:, 1:lastcol]

            # Normalise: replace `nothing` with NaN
            mat = map(x -> x === nothing ? NaN : float(x), view_mat)

            # Ignore leading all-zero rows for indexing
            firstrow = findfirst(i -> any(!=(NaN), mat[i, :]), axes(mat, 1))
            if firstrow === nothing
                push!(labels, "nothing")
                continue
            end

            norm_mat = mat[firstrow:end, :]

            # Assign running index by first appearance
            idx = findfirst(M -> M == norm_mat, seen)
            if idx === nothing
                push!(seen, copy(norm_mat))
                next_idx += 1
                idx = next_idx
            end
            push!(labels, "#$(idx)")
        end

        if length(labels) > 1
            push!(annotate_diff_input, "Shocks" => labels)
        end
    end

    if haskey(diffdict, :conditions)
        conds = diffdict[:conditions]

        labels = Vector{String}()
        seen   = []
        next_idx = 0

        for cond_mat in conds
            if cond_mat === nothing
                push!(labels, "")
                continue
            end

            # Catch the all-nothing case by column scan
            lastcol = findlast(j -> any(x -> x !== nothing, cond_mat[:, j]), axes(cond_mat, 2))
            if lastcol === nothing
                push!(labels, "nothing")
                continue
            end

            view_mat = cond_mat[:, 1:lastcol]

            # Replace `nothing` with 0.0 and work in Float64
            mat = map(x -> x === nothing ? 0.0 : float(x), view_mat)

            # Drop leading rows that are all zero
            firstrow = findfirst(i -> any(!=(0.0), mat[i, :]), axes(mat, 1))
            if firstrow === nothing
                push!(labels, "nothing")
                continue
            end

            norm_mat = mat[firstrow:end, :]

            # Assign running index by first appearance
            idx = findfirst(M -> M == norm_mat, seen)
            if idx === nothing
                push!(seen, copy(norm_mat))
                next_idx += 1
                idx = next_idx
            end
            push!(labels, "#$(idx)")
        end

        if length(labels) > 1
            push!(annotate_diff_input, "Conditions" => labels)
        end
    end

    if haskey(diffdict, :initial_state)
        vals = diffdict[:initial_state]

        labels = String[]                                # "" for [0.0], "#k" otherwise
        seen   = []                                      # store distinct non-[0.0] values by content
        next_idx = 0

        for v in vals
            if v === nothing
                push!(labels, "")
            elseif v == [0.0]
                push!(labels, "nothing")
            else
                idx = findfirst(==(v), seen)             # content based lookup
                if idx === nothing
                    push!(seen, copy(v))                 # store by value
                    next_idx += 1
                    idx = next_idx
                end
                push!(labels, "#$(idx)")
            end
        end

        push!(annotate_diff_input, "Initial state" => labels)
    end

    annotate_rename_dict_diff!(annotate_diff_input, diffdict)

    same_shock_direction = true
    
    if annotate_default_kwarg_diffs!(annotate_diff_input, args_and_kwargs, diffdict,
            [:run_id, :parameters, :plot_data, :tol, :reference_steady_state, :initial_state, :conditions, :conditions_in_levels, :label,
             :shocks, :shock_names,
             :variables, :variable_names, :var_idx,
             :rename_dictionary])
        same_shock_direction = false
    end

    annotate_tol_diff!(annotate_diff_input, conditional_forecast_active_plot_container)

    if haskey(diffdict, :shock_names)
        if all(length.(diffdict[:shock_names]) .== 1)
            push!(annotate_diff_input, "Shock name" => map(x->x[1], diffdict[:shock_names]))
        end
    end

    legend_plot = StatsPlots.plot(framestyle = :none, 
                                    legend = :inside, 
                                    legend_columns = min(4, length(conditional_forecast_active_plot_container))) 
    

    joint_shocks = OrderedSet{String}()
    joint_variables = OrderedSet{String}()
    single_shock_per_irf = true

    max_periods = 0
    plt_lab_switch = should_use_label_switch(annotate_diff_input, conditional_forecast_active_plot_container)
    for (i,k) in enumerate(conditional_forecast_active_plot_container)
        if plot_type == :stack
            StatsPlots.bar!(legend_plot,
                            [NaN], 
                            legend_title = plt_lab_switch ? nothing : annotate_diff_input[2][1],
                            linecolor = :transparent,
                            color = pal[mod1.(i, length(pal))]',
                            alpha = transparency,
                            linewidth = 0,
                            label = plt_lab_switch ? (k[:label] isa Symbol ? string(k[:label]) : k[:label]) : (annotate_diff_input[2][2][i] isa String ? annotate_diff_input[2][2][i] : String(Symbol(annotate_diff_input[2][2][i]))))
        elseif plot_type == :compare
            StatsPlots.plot!(legend_plot,
                            [NaN], 
                            legend_title = plt_lab_switch ? nothing : annotate_diff_input[2][1],
                            color = pal[mod1(i, length(pal))],
                            label = plt_lab_switch ? (k[:label] isa Symbol ? string(k[:label]) : k[:label]) : (annotate_diff_input[2][2][i] isa String ? annotate_diff_input[2][2][i] : String(Symbol(annotate_diff_input[2][2][i]))))
        end

        foreach(n -> push!(joint_variables, String(apply_custom_name(replace_indices_in_symbol(n), Dict(k[:rename_dictionary])))), k[:variable_names] isa AbstractArray ? k[:variable_names] : (k[:variable_names],))
        foreach(n -> push!(joint_shocks, String(apply_custom_name(Symbol(replace(string(replace_indices_in_symbol(n)), "₍ₓ₎" => "")), Dict(k[:rename_dictionary])))), k[:shock_names] isa AbstractArray ? k[:shock_names] : (k[:shock_names],))

        max_periods = max(max_periods, size(k[:plot_data],2))
    end
    
    for (i,k) in enumerate(conditional_forecast_active_plot_container)
        if plot_type == :compare
            StatsPlots.scatter!(legend_plot,
                                [NaN], 
                                label = "Condition", # * (length(annotate_diff_input) > 2 ? String(Symbol(i)) : annotate_diff_input[2][2][i] isa String ? annotate_diff_input[2][2][i] : String(Symbol(annotate_diff_input[2][2][i]))), 
                                marker = gr_back ? :star8 : :pentagon,
                                markerstrokewidth = 0,
                                markercolor = pal[mod1(i, length(pal))])

        end
    end
    
    sort!(joint_variables, by = normalize_superscript)
    sort!(joint_shocks, by = normalize_superscript)

    n_subplots = length(joint_variables) + length(joint_shocks)
    pp = []
    pane = 1
    plot_count = 1

    joint_non_zero_variables = []

    return_plots = []
    
    for var in vcat(collect(joint_variables), collect(joint_shocks))
        not_zero_in_any_cond_fcst = false

        for k in conditional_forecast_active_plot_container
            transformed_vars = String.(apply_custom_name.(replace_indices_in_symbol.(k[:variable_names]), Ref(Dict(k[:rename_dictionary]))))
            transformed_shocks = String.(apply_custom_name.(Symbol.(replace.(string.(replace_indices_in_symbol.(k[:shock_names])), Ref("₍ₓ₎" => ""))), Ref(Dict(k[:rename_dictionary]))))
            
            var_idx = findfirst(==(var), vcat(transformed_vars, transformed_shocks))
            if isnothing(var_idx)
                # If the variable or shock is not present in the current conditional_forecast_active_plot_container,
                # we skip this iteration.
                continue
            else
                if any(.!isapprox.(k[:plot_data][var_idx,:], 0, atol = eps(Float32))) || any(!=(nothing), vcat(k[:conditions], k[:shocks])[k[:var_idx][var_idx], :])
                    not_zero_in_any_cond_fcst = not_zero_in_any_cond_fcst || true
                    # break # If any cond_fcst data is not approximately zero, we set the flag to true.
                end
            end
        end

        if not_zero_in_any_cond_fcst 
            push!(joint_non_zero_variables, var)
        else
            # If all cond_fcst data for this variable and shock is approximately zero, we skip this subplot.
            n_subplots -= 1
        end
    end
    
    for var in joint_non_zero_variables
        SSs = eltype(conditional_forecast_active_plot_container[1][:reference_steady_state])[]
        Ys = AbstractVector{eltype(conditional_forecast_active_plot_container[1][:plot_data])}[]

        subplot_title = ""
        
        for k in conditional_forecast_active_plot_container
            transformed_vars = String.(apply_custom_name.(replace_indices_in_symbol.(k[:variable_names]), Ref(Dict(k[:rename_dictionary]))))
            transformed_shocks = String.(apply_custom_name.(Symbol.(replace.(string.(replace_indices_in_symbol.(k[:shock_names])), Ref("₍ₓ₎" => ""))), Ref(Dict(k[:rename_dictionary]))))
            
            var_idx = findfirst(==(var), vcat(transformed_vars, transformed_shocks))
            if isnothing(var_idx)
                # If the variable is not present in the current conditional_forecast_active_plot_container,
                # we skip this iteration.
                push!(SSs, NaN)
                push!(Ys, zeros(max_periods))
            else
                dat = fill(NaN, max_periods)
                dat[1:length(k[:plot_data][var_idx,:])] .= k[:plot_data][var_idx,:]
                push!(SSs, k[:reference_steady_state][k[:var_idx][var_idx]])
                push!(Ys, dat) # k[:plot_data][var_idx,:])
            end

        
            if var ∈ transformed_vars
                subplot_title = apply_custom_name(replace_indices_in_symbol(Symbol(var)), Dict(k[:rename_dictionary]))
            elseif var ∈ transformed_shocks
                subplot_title = String(apply_custom_name(Symbol(replace(string(replace_indices_in_symbol(Symbol(var))), "₍ₓ₎" => "")), Dict(k[:rename_dictionary]))) * "₍ₓ₎"
            end
        end

        same_ss = true

        if maximum(filter(!isnan, SSs)) - minimum(filter(!isnan, SSs)) > 1e-10
            push!(annotate_ss_page, var => minimal_sigfig_strings(SSs))
            same_ss = false
        end
        
        p = standard_subplot(Val(plot_type),
                                Ys, 
                                SSs, 
                                subplot_title, 
                                gr_back,
                                same_ss,
                                pal = pal,
                                transparency = transparency)

        if plot_type == :compare
            for (i,k) in enumerate(conditional_forecast_active_plot_container)   
                var_idx = findfirst(==(var), String.(apply_custom_name.(vcat(k[:variable_names], Symbol.(replace.(string.(k[:shock_names]), Ref("₍ₓ₎" => "")))), Ref(Dict(k[:rename_dictionary])))))

                if isnothing(var_idx) continue end
                cond_idx = findall(vcat(k[:conditions], k[:shocks])[k[:var_idx][var_idx],:] .!= nothing)

                if length(cond_idx) > 0
                    SS = k[:reference_steady_state][k[:var_idx][var_idx]]

                    vals = vcat(k[:conditions], k[:shocks])[k[:var_idx][var_idx], cond_idx]

                    if k[:conditions_in_levels]
                        vals .-= SS
                    end

                    if same_ss
                        vals .+= SS
                    end

                    StatsPlots.scatter!(p,
                                        cond_idx,
                                        vals,
                                        label = "",
                                        marker = gr_back ? :star8 : :pentagon, 
                                        markerstrokewidth = 0,
                                        markercolor = pal[mod1(i, length(pal))])
                end
            end
        end

        push!(pp, p)
        
        if !(plot_count % plots_per_page == 0)
            plot_count += 1
        else
            plot_count = 1

            pane = assemble_and_emit_page!(
                return_plots, pp, legend_plot,
                annotate_diff_input, diffdict,
                attributes, attributes_redux,
                pane, n_subplots, plots_per_page,
                show_plots, save_plots, save_plots_path, save_plots_name, save_plots_format,
                𝓂.model_name;
                title_extra = "        Conditional forecast",
                annotate_ss = annotate_ss,
                annotate_ss_page = annotate_ss_page,
                plt_lab_switch = plt_lab_switch,
            )
        end
    end

    if length(pp) > 0
        assemble_and_emit_page!(
            return_plots, pp, legend_plot,
            annotate_diff_input, diffdict,
            attributes, attributes_redux,
            pane, n_subplots, plots_per_page,
            show_plots, save_plots, save_plots_path, save_plots_name, save_plots_format,
            𝓂.model_name;
            title_extra = "        Conditional forecast",
            annotate_ss = annotate_ss,
            annotate_ss_page = annotate_ss_page,
            plt_lab_switch = plt_lab_switch,
            is_tail = true,
        )
    end

    if !use_workspaces 𝓂.workspaces = orig_ws end

    return return_plots
end


end # dispatch_doctor

end # module
