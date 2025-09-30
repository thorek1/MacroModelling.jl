module StatsPlotsExt

using MacroModelling
import MacroModelling: ParameterType, ℳ, Symbol_input, String_input, Tolerances, merge_calculation_options, MODEL®, DATA®, PARAMETERS®, ALGORITHM®, FILTER®, VARIABLES®, SMOOTH®, SHOW_PLOTS®, SAVE_PLOTS®, SAVE_PLOTS_FORMATH®, SAVE_PLOTS_PATH®, PLOTS_PER_PAGE®, MAX_ELEMENTS_PER_LEGENDS_ROW®, EXTRA_LEGEND_SPACE®, PLOT_ATTRIBUTES®, QME®, SYLVESTER®, LYAPUNOV®, TOLERANCES®, VERBOSE®, DATA_IN_LEVELS®, PERIODS®, SHOCKS®, SHOCK_SIZE®, NEGATIVE_SHOCK®, GENERALISED_IRF®, INITIAL_STATE®, IGNORE_OBC®, CONDITIONS®, SHOCK_CONDITIONS®, LEVELS®, LABEL®, parse_shocks_input_to_index, parse_variables_input_to_index, replace_indices, filter_data_with_model, get_relevant_steady_states, replace_indices_in_symbol, parse_algorithm_to_state_update, girf, decompose_name, obc_objective_optim_fun, obc_constraint_optim_fun, compute_irf_responses
import DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF, TYPEDSIGNATURES, TYPEDFIELDS
import LaTeXStrings

const irf_active_plot_container = Dict[]
const conditional_forecast_active_plot_container = Dict[]
const model_estimates_active_plot_container = Dict[]

import StatsPlots
import Showoff
import DataStructures: OrderedSet
import SparseArrays: SparseMatrixCSC
import NLopt
using DispatchDoctor

import MacroModelling: plot_irfs, plot_irf, plot_IRF, plot_simulations, plot_simulation, plot_solution, plot_girf, plot_conditional_forecast, plot_conditional_variance_decomposition, plot_forecast_error_variance_decomposition, plot_fevd, plot_model_estimates, plot_shock_decomposition, plotlyjs_backend, gr_backend, compare_args_and_kwargs

import MacroModelling: plot_irfs!, plot_irf!, plot_IRF!, plot_girf!, plot_simulations!, plot_simulation!, plot_conditional_forecast!, plot_model_estimates!

const default_plot_attributes = Dict(:size=>(700,500),
                                :plot_titlefont => 10, 
                                :titlefont => 8, 
                                :guidefont => 8,
                                :palette => :auto,
                                :legendfontsize => 8,
                                :annotationfontsize => 8,
                                :legend_title_font_pointsize => 8,
                                :tickfontsize => 8,
                                :framestyle => :semi)

# Elements whose difference between function calls should be highlighted across models.
const args_and_kwargs_names = Dict(:model_name => "Model",
                                    :algorithm => "Algorithm",
                                    :shock_names => "Shock",
                                    :shock_size => "Shock size",
                                    :negative_shock => "Negative shock",
                                    :generalised_irf => "Generalised IRF",
                                    :periods => "Periods",
                                    :presample_periods => "Presample Periods",
                                    :ignore_obc => "Ignore OBC",
                                    :smooth => "Smooth",
                                    :data => "Data",
                                    :label => "Label",
                                    :filter => "Filter",
                                    :warmup_iterations => "Warmup Iterations",
                                    :quadratic_matrix_equation_algorithm => "Quadratic Matrix Equation Algorithm",
                                    :sylvester_algorithm => "Sylvester Algorithm",
                                    :lyapunov_algorithm => "Lyapunov Algorithm",
                                    :NSSS_acceptance_tol => "NSSS acceptance tol",
                                    :NSSS_xtol => "NSSS xtol",
                                    :NSSS_ftol => "NSSS ftol",
                                    :NSSS_rel_xtol => "NSSS rel xtol",
                                    :qme_tol => "QME tol",
                                    :qme_acceptance_tol => "QME acceptance tol",
                                    :sylvester_tol => "Sylvester tol",
                                    :sylvester_acceptance_tol => "Sylvester acceptance tol",
                                    :lyapunov_tol => "Lyapunov tol",
                                    :lyapunov_acceptance_tol => "Lyapunov acceptance tol",
                                    :droptol => "Droptol",
                                    :dependencies_tol => "Dependencies tol"
                                    )
                        
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
Plot model estimates of the variables given the data. The default plot shows the estimated variables, shocks, and the data underlying the estimates. The estimates are based on the Kalman smoother or filter (depending on the `smooth` keyword argument) or inversion filter using the provided data and solution of the model.

The left axis shows the level, and the right the deviation from the relevant steady state. The non-stochastic steady state (NSSS) is relevant for first order solutions and the stochastic steady state for higher order solutions. The horizontal black line indicates the relevant steady state. Variable names are above the subplots and the title provides information about the model, shocks, and number of pages per shock.
In case `shock_decomposition = true`, the plot shows the variables, shocks, and data in absolute deviations from the relevant steady state as a stacked bar chart per period.

For higher order perturbation solutions the decomposition additionally contains a term `Nonlinearities`. This term represents the nonlinear interaction between the states in the periods after the shocks arrived and in the case of pruned third order, the interaction between (pruned second order) states and contemporaneous shocks.

If occasionally binding constraints are present in the model, they are not taken into account here.

# Arguments
- $MODEL®
- $DATA®
# Keyword Arguments
- $PARAMETERS®
- $ALGORITHM®
- $FILTER®
- $VARIABLES®
- `shocks` [Default: `:all`]: shocks for which to plot the estimates. Inputs can be either a `Symbol` (e.g. `:y`, or `:all`), `Tuple{Symbol, Vararg{Symbol}}`, `Matrix{Symbol}`, or `Vector{Symbol}`.
- `presample_periods` [Default: `0`, Type: `Int`]: periods at the beginning of the data which are not plotted. Useful if you want to filter for all periods but focus only on a certain period later in the sample.
- $DATA_IN_LEVELS®
- `shock_decomposition` [Default: `false`, Type: `Bool`]: whether to show the contribution of the shocks to the deviations from NSSS for each variable. If `false`, the plot shows the values of the selected variables, data, and shocks
- $SMOOTH®
- `label` [Default: `1`, Type: `Union{Real, String, Symbol}`]: label to attribute to this function call in the plots.
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMATH®
- $SAVE_PLOTS_PATH®
- $PLOTS_PER_PAGE®
- `transparency` [Default: `0.6`, Type: `Float64`]: transparency of stacked bars. Only relevant if `shock_decomposition` is `true`.
- $MAX_ELEMENTS_PER_LEGENDS_ROW®
- $EXTRA_LEGEND_SPACE®
- $LABEL®
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
function plot_model_estimates(𝓂::ℳ,
                                data::KeyedArray{Float64};
                                parameters::ParameterType = nothing,
                                algorithm::Symbol = :first_order, 
                                filter::Symbol = :kalman, 
                                warmup_iterations::Int = 0,
                                variables::Union{Symbol_input,String_input} = :all_excluding_obc, 
                                shocks::Union{Symbol_input,String_input} = :all, 
                                presample_periods::Int = 0,
                                data_in_levels::Bool = true,
                                shock_decomposition::Bool = false,
                                smooth::Bool = true,
                                label::Union{Real, String, Symbol} = 1,
                                show_plots::Bool = true,
                                save_plots::Bool = false,
                                save_plots_format::Symbol = :pdf,
                                save_plots_path::String = ".",
                                plots_per_page::Int = 6,
                                transparency::Float64 = .6,
                                max_elements_per_legend_row::Int = 4,
                                extra_legend_space::Float64 = 0.0,
                                plot_attributes::Dict = Dict(),
                                verbose::Bool = false,
                                tol::Tolerances = Tolerances(),
                                quadratic_matrix_equation_algorithm::Symbol = :schur,
                                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = sum(1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > 1000 ? :bicgstab : :doubling,
                                lyapunov_algorithm::Symbol = :doubling)
    # @nospecialize # reduce compile time                            

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > 1000 ? :bicgstab : :doubling : sylvester_algorithm[2],
                                    lyapunov_algorithm = lyapunov_algorithm)

    gr_back = StatsPlots.backend() == StatsPlots.Plots.GRBackend()

    if !gr_back
        attrbts = merge(default_plot_attributes, Dict(:framestyle => :box))
    else
        attrbts = merge(default_plot_attributes, Dict())
    end

    attributes = merge(attrbts, plot_attributes)

    attributes_redux = copy(attributes)

    delete!(attributes_redux, :framestyle)


    # write_parameters_input!(𝓂, parameters, verbose = verbose)

    @assert filter ∈ [:kalman, :inversion] "Currently only the kalman filter (:kalman) for linear models and the inversion filter (:inversion) for linear and nonlinear models are supported."

    pruning = false

    @assert !(algorithm ∈ [:second_order, :third_order] && shock_decomposition) "Decomposition implemented for first order, pruned second and third order. Second and third order solution decomposition is not yet implemented."
    
    if algorithm ∈ [:second_order, :third_order]
        filter = :inversion
    end

    if algorithm ∈ [:pruned_second_order, :pruned_third_order]
        filter = :inversion
        pruning = true
    end

    solve!(𝓂, parameters = parameters, algorithm = algorithm, opts = opts, dynamics = true)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)

    data = data(sort(axiskeys(data,1)))
    
    obs_axis = collect(axiskeys(data,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    variables = variables isa String_input ? variables .|> Meta.parse .|> replace_indices : variables

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    obs_idx     = parse_variables_input_to_index(obs_symbols, 𝓂.timings) |> sort
    var_idx     = parse_variables_input_to_index(variables, 𝓂.timings)  |> sort
    shock_idx   = shocks == :none ? [] : parse_shocks_input_to_index(shocks, 𝓂.timings)

    variable_names = replace_indices_in_symbol.(𝓂.timings.var[var_idx])
    
    shock_names = replace_indices_in_symbol.(𝓂.timings.exo[shock_idx]) .* "₍ₓ₎"
    
    legend_columns = 1

    legend_items = length(shock_idx) + 3 + pruning

    max_columns = min(legend_items, max_elements_per_legend_row)
    
    # Try from max_columns down to 1 to find the optimal solution
    for cols in max_columns:-1:1
        if legend_items % cols == 0 || legend_items % cols <= max_elements_per_legend_row
            legend_columns = cols
            break
        end
    end

    if data_in_levels
        data_in_deviations = data .- NSSS[obs_idx]
    else
        data_in_deviations = data
    end

    x_axis = axiskeys(data,2)

    extra_legend_space += length(string(x_axis[1])) > 6 ? .1 : 0.0

    @assert presample_periods < size(data,2) "The number of presample periods must be less than the number of periods in the data."

    periods = presample_periods+1:size(data,2)

    x_axis = x_axis[periods]
    
    variables_to_plot, shocks_to_plot, standard_deviations, decomposition = filter_data_with_model(𝓂, data_in_deviations, Val(algorithm), Val(filter), warmup_iterations = warmup_iterations, smooth = smooth, opts = opts)
    
    if pruning
        decomposition[:,1:(end - 2 - pruning),:]    .+= SSS_delta
        decomposition[:,end - 2,:]                  .-= SSS_delta * (size(decomposition,2) - 4)
        variables_to_plot                           .+= SSS_delta
        data_in_deviations                          .+= SSS_delta[obs_idx]
    end

    orig_pal = StatsPlots.palette(attributes_redux[:palette])

    total_pal_len = 100

    alpha_reduction_factor = 0.7

    pal = mapreduce(x -> StatsPlots.coloralpha.(orig_pal, alpha_reduction_factor ^ x), vcat, 0:(total_pal_len ÷ length(orig_pal)) - 1) |> StatsPlots.palette

    estimate_color = :navy

    data_color = :orangered

    while length(model_estimates_active_plot_container) > 0
        pop!(model_estimates_active_plot_container)
    end

    args_and_kwargs = Dict(:run_id => length(model_estimates_active_plot_container) + 1,
                           :model_name => 𝓂.model_name,
                           :label => label,
                           
                           :data => data,
                           :parameters => Dict(𝓂.parameters .=> 𝓂.parameter_values),
                           :algorithm => algorithm,
                           :filter => filter,
                           :warmup_iterations => warmup_iterations,
                           :variables => variables,
                           :shocks => shocks,
                           :presample_periods => presample_periods,
                           :data_in_levels => data_in_levels,
                        #    :shock_decomposition => shock_decomposition,
                           :smooth => smooth,
                           
                           :NSSS_acceptance_tol => tol.NSSS_acceptance_tol,
                           :NSSS_xtol => tol.NSSS_xtol,
                           :NSSS_ftol => tol.NSSS_ftol,
                           :NSSS_rel_xtol => tol.NSSS_rel_xtol,
                           :qme_tol => tol.qme_tol,
                           :qme_acceptance_tol => tol.qme_acceptance_tol,
                           :sylvester_tol => tol.sylvester_tol,
                           :sylvester_acceptance_tol => tol.sylvester_acceptance_tol,
                           :lyapunov_tol => tol.lyapunov_tol,
                           :lyapunov_acceptance_tol => tol.lyapunov_acceptance_tol,
                           :droptol => tol.droptol,
                           :dependencies_tol => tol.dependencies_tol,

                           :quadratic_matrix_equation_algorithm => quadratic_matrix_equation_algorithm,
                           :sylvester_algorithm => sylvester_algorithm,
                           :lyapunov_algorithm => lyapunov_algorithm,
                           
                           :decomposition => decomposition,
                           :variables_to_plot => variables_to_plot,
                           :data_in_deviations => data_in_deviations,
                           :shocks_to_plot => shocks_to_plot,
                           :reference_steady_state => reference_steady_state[var_idx],
                           :variable_names => variable_names,
                           :shock_names => shock_names,
                           :x_axis => x_axis
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
            push!(non_zero_shock_names, shock_names[i])
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

                p = standard_subplot(variables_to_plot[var_idx[i],periods], 
                                    SS, 
                                    variable_names[i], 
                                    gr_back,
                                    pal = shock_decomposition ? StatsPlots.palette([estimate_color]) : pal,
                                    xvals = x_axis)

                if shock_decomposition
                    additional_indices = pruning ? [size(decomposition,2)-1, size(decomposition,2)-2] : [size(decomposition,2)-1]

                    p = standard_subplot(Val(:stack),
                                        [decomposition[var_idx[i],k,periods] for k in vcat(additional_indices, non_zero_shock_idx)], 
                                        [SS for k in vcat(additional_indices, non_zero_shock_idx)], 
                                        variable_names[i], 
                                        gr_back,
                                        true, # same_ss,
                                        transparency = transparency,
                                        xvals = x_axis,
                                        pal = pal,
                                        color_total = estimate_color)
                                        
                    if var_idx[i] ∈ obs_idx 
                        StatsPlots.plot!(p,
                            # x_axis,
                            shock_decomposition ? data_in_deviations[indexin([var_idx[i]],obs_idx),periods]' : data_in_deviations[indexin([var_idx[i]],obs_idx),periods]' .+ SS,
                            label = "",
                            color = shock_decomposition ? data_color : pal[2])
                    end
                else
                    if var_idx[i] ∈ obs_idx 
                        StatsPlots.plot!(p,
                            x_axis,
                            shock_decomposition ? data_in_deviations[indexin([var_idx[i]],obs_idx),periods]' : data_in_deviations[indexin([var_idx[i]],obs_idx),periods]' .+ SS,
                            label = "",
                            color = shock_decomposition ? data_color : pal[2])
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

            StatsPlots.plot!(pl,
                            [NaN], 
                            label = "Data", 
                            color = shock_decomposition ? data_color : pal[2])

            if shock_decomposition
                additional_labels = pruning ? ["Initial value", "Nonlinearities"] : ["Initial value"]

                lbls = reshape(vcat(additional_labels, string.(replace_indices_in_symbol.(𝓂.exo[non_zero_shock_idx]))), 1, length(non_zero_shock_idx) + 1 + pruning)

                StatsPlots.bar!(pl,
                                fill(NaN, 1, length(non_zero_shock_idx) + 1 + pruning), 
                                label = lbls, 
                                linewidth = 0,
                                alpha = transparency,
                                color = pal[mod1.(1:length(lbls), length(pal))]', 
                                legend_columns = legend_columns)
            end
            
            # Legend
            p = StatsPlots.plot(ppp,pl, 
                                    layout = StatsPlots.grid(2, 1, heights = [1 - legend_columns * 0.01 - extra_legend_space, legend_columns * 0.01 + extra_legend_space]),
                                    plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"; 
                                    attributes_redux...)

            push!(return_plots,p)

            if show_plots
                display(p)
            end

            if save_plots
                StatsPlots.savefig(p, save_plots_path * "/estimation__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
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

        StatsPlots.plot!(pl,
                        [NaN], 
                        label = "Data", 
                        color = shock_decomposition ? data_color : pal[2])

        if shock_decomposition
            additional_labels = pruning ? ["Initial value", "Nonlinearities"] : ["Initial value"]

            lbls = reshape(vcat(additional_labels, string.(replace_indices_in_symbol.(𝓂.exo[non_zero_shock_idx]))), 1, length(non_zero_shock_idx) + 1 + pruning)

            StatsPlots.bar!(pl,
                            fill(NaN, 1, length(non_zero_shock_idx) + 1 + pruning), 
                            label = lbls, 
                            linewidth = 0,
                            alpha = transparency,
                            color = pal[mod1.(1:length(lbls), length(pal))]', 
                                legend_columns = legend_columns)
        end
        
        # Legend
        p = StatsPlots.plot(ppp,pl, 
                                layout = StatsPlots.grid(2, 1, heights = [1 - legend_columns * 0.01 - extra_legend_space, legend_columns * 0.01 + extra_legend_space]),
                                plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"; 
                                attributes_redux...)


        push!(return_plots,p)

        if show_plots
            display(p)
        end

        if save_plots
            StatsPlots.savefig(p, save_plots_path * "/estimation__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
        end
    end

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
This function allows comparison of the estimated variables, shocks, and the data underlying the estimates for any combination of inputs.

This function shares most of the signature and functionality of [`plot_model_estimates`](@ref). Its main purpose is to append plots based on the inputs to previous calls of this function and the last call of [`plot_model_estimates`](@ref). In the background it keeps a registry of the inputs and outputs and then plots the comparison.

# Arguments
- $MODEL®
- $DATA®
# Keyword Arguments
- $PARAMETERS®
- $ALGORITHM®
- $FILTER®
- $VARIABLES®
- `shocks` [Default: `:all`]: shocks for which to plot the estimates. Inputs can be either a `Symbol` (e.g. `:y`, or `:all`), `Tuple{Symbol, Vararg{Symbol}}`, `Matrix{Symbol}`, or `Vector{Symbol}`.
- `presample_periods` [Default: `0`, Type: `Int`]: periods at the beginning of the data which are not plotted. Useful if you want to filter for all periods but focus only on a certain period later in the sample.
- $DATA_IN_LEVELS®
- $LABEL®
- $SMOOTH®
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMATH®
- $SAVE_PLOTS_PATH®
- $PLOTS_PER_PAGE®
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
                                data::KeyedArray{Float64};
                                parameters::ParameterType = nothing,
                                algorithm::Symbol = :first_order,
                                filter::Symbol = :kalman,
                                warmup_iterations::Int = 0,
                                variables::Union{Symbol_input,String_input} = :all_excluding_obc, 
                                shocks::Union{Symbol_input,String_input} = :all, 
                                presample_periods::Int = 0,
                                data_in_levels::Bool = true,
                                smooth::Bool = true,
                                label::Union{Real, String, Symbol} = length(model_estimates_active_plot_container) + 1,
                                show_plots::Bool = true,
                                save_plots::Bool = false,
                                save_plots_format::Symbol = :pdf,
                                save_plots_path::String = ".",
                                plots_per_page::Int = 6,
                                max_elements_per_legend_row::Int = 4,
                                extra_legend_space::Float64 = 0.0,
                                plot_attributes::Dict = Dict(),
                                verbose::Bool = false,
                                tol::Tolerances = Tolerances(),
                                quadratic_matrix_equation_algorithm::Symbol = :schur,
                                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = sum(1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > 1000 ? :bicgstab : :doubling,
                                lyapunov_algorithm::Symbol = :doubling)
    # @nospecialize # reduce compile time                            

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > 1000 ? :bicgstab : :doubling : sylvester_algorithm[2],
                                    lyapunov_algorithm = lyapunov_algorithm)

    gr_back = StatsPlots.backend() == StatsPlots.Plots.GRBackend()

    if !gr_back
        attrbts = merge(default_plot_attributes, Dict(:framestyle => :box))
    else
        attrbts = merge(default_plot_attributes, Dict())
    end

    attributes = merge(attrbts, plot_attributes)

    attributes_redux = copy(attributes)

    delete!(attributes_redux, :framestyle)


    # write_parameters_input!(𝓂, parameters, verbose = verbose)

    @assert filter ∈ [:kalman, :inversion] "Currently only the kalman filter (:kalman) for linear models and the inversion filter (:inversion) for linear and nonlinear models are supported."

    pruning = false
    
    if algorithm ∈ [:second_order, :third_order]
        filter = :inversion
    end

    if algorithm ∈ [:pruned_second_order, :pruned_third_order]
        filter = :inversion
        pruning = true
    end

    solve!(𝓂, parameters = parameters, algorithm = algorithm, opts = opts, dynamics = true)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)

    data = data(sort(axiskeys(data,1)))
    
    obs_axis = collect(axiskeys(data,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    variables = variables isa String_input ? variables .|> Meta.parse .|> replace_indices : variables

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    obs_idx     = parse_variables_input_to_index(obs_symbols, 𝓂.timings) |> sort
    var_idx     = parse_variables_input_to_index(variables, 𝓂.timings)  |> sort
    shock_idx   = parse_shocks_input_to_index(shocks, 𝓂.timings)

    variable_names = replace_indices_in_symbol.(𝓂.timings.var[var_idx])
    
    shock_names = replace_indices_in_symbol.(𝓂.timings.exo[shock_idx]) .* "₍ₓ₎"
    
    legend_columns = 1

    legend_items = length(shock_idx) + 3 + pruning

    max_columns = min(legend_items, max_elements_per_legend_row)
    
    # Try from max_columns down to 1 to find the optimal solution
    for cols in max_columns:-1:1
        if legend_items % cols == 0 || legend_items % cols <= max_elements_per_legend_row
            legend_columns = cols
            break
        end
    end

    if data_in_levels
        data_in_deviations = data .- NSSS[obs_idx]
    else
        data_in_deviations = data
    end

    x_axis = axiskeys(data,2)

    extra_legend_space += length(string(x_axis[1])) > 6 ? .1 : 0.0

    @assert presample_periods < size(data,2) "The number of presample periods must be less than the number of periods in the data."

    periods = presample_periods+1:size(data,2)

    x_axis = x_axis[periods]
    
    variables_to_plot, shocks_to_plot, standard_deviations, decomposition = filter_data_with_model(𝓂, data_in_deviations, Val(algorithm), Val(filter), warmup_iterations = warmup_iterations, smooth = smooth, opts = opts)
    
    if pruning
        decomposition[:,1:(end - 2 - pruning),:]    .+= SSS_delta
        decomposition[:,end - 2,:]                  .-= SSS_delta * (size(decomposition,2) - 4)
        variables_to_plot                           .+= SSS_delta
        data_in_deviations                          .+= SSS_delta[obs_idx]
    end

    orig_pal = StatsPlots.palette(attributes_redux[:palette])

    total_pal_len = 100

    alpha_reduction_factor = 0.7

    pal = mapreduce(x -> StatsPlots.coloralpha.(orig_pal, alpha_reduction_factor ^ x), vcat, 0:(total_pal_len ÷ length(orig_pal)) - 1) |> StatsPlots.palette

    estimate_color = :navy

    data_color = :orangered

    args_and_kwargs = Dict(:run_id => length(model_estimates_active_plot_container) + 1,
                           :model_name => 𝓂.model_name,
                           :label => label,
                           
                           :data => data,
                           :parameters => Dict(𝓂.parameters .=> 𝓂.parameter_values),
                           :algorithm => algorithm,
                           :filter => filter,
                           :warmup_iterations => warmup_iterations,
                           :variables => variables,
                           :shocks => shocks,
                           :presample_periods => presample_periods,
                           :data_in_levels => data_in_levels,
                        #    :shock_decomposition => shock_decomposition,
                           :smooth => smooth,
                           
                           :NSSS_acceptance_tol => tol.NSSS_acceptance_tol,
                           :NSSS_xtol => tol.NSSS_xtol,
                           :NSSS_ftol => tol.NSSS_ftol,
                           :NSSS_rel_xtol => tol.NSSS_rel_xtol,
                           :qme_tol => tol.qme_tol,
                           :qme_acceptance_tol => tol.qme_acceptance_tol,
                           :sylvester_tol => tol.sylvester_tol,
                           :sylvester_acceptance_tol => tol.sylvester_acceptance_tol,
                           :lyapunov_tol => tol.lyapunov_tol,
                           :lyapunov_acceptance_tol => tol.lyapunov_acceptance_tol,
                           :droptol => tol.droptol,
                           :dependencies_tol => tol.dependencies_tol,

                           :quadratic_matrix_equation_algorithm => quadratic_matrix_equation_algorithm,
                           :sylvester_algorithm => sylvester_algorithm,
                           :lyapunov_algorithm => lyapunov_algorithm,
                           
                           :decomposition => decomposition,
                           :variables_to_plot => variables_to_plot,
                           :data_in_deviations => data_in_deviations,
                           :shocks_to_plot => shocks_to_plot,
                           :reference_steady_state => reference_steady_state[var_idx],
                           :variable_names => variable_names,
                           :shock_names => shock_names,
                           :x_axis => x_axis
                           )

    no_duplicate = all(
        !(all((
            get(dict, :parameters, nothing) == args_and_kwargs[:parameters],
            # get(dict, :data, nothing) == args_and_kwargs[:data],
            # get(dict, :filter, nothing) == args_and_kwargs[:filter],
            # get(dict, :warmup_iterations, nothing) == args_and_kwargs[:warmup_iterations],
            # get(dict, :smooth, nothing) == args_and_kwargs[:smooth],
            all(k == :data ? collect(get(dict, k, nothing)) == collect(get(args_and_kwargs, k, nothing)) : get(dict, k, nothing) == get(args_and_kwargs, k, nothing) for k in keys(args_and_kwargs_names))
        )))
        for dict in model_estimates_active_plot_container
    ) # "New plot must be different from previous plot. Use the version without ! to plot."

    if no_duplicate 
        push!(model_estimates_active_plot_container, args_and_kwargs)
    else
        @info "Plot with same parameters already exists. Using previous plot data to create plot."
    end

    # 1. Keep only certain keys from each dictionary
    reduced_vector = [
        Dict(k => d[k] for k in vcat(:run_id, keys(args_and_kwargs_names)...) if haskey(d, k))
        for d in model_estimates_active_plot_container
    ]

    diffdict = compare_args_and_kwargs(reduced_vector)

    # 2. Group the original vector by :model_name. Check difference for keys where they matter between models. Two different models might have different shocks so that difference is less important, but the same model with different shocks is a difference to highlight.
    grouped_by_model = Dict{Any, Vector{Dict}}()

    for d in model_estimates_active_plot_container
        model = d[:model_name]
        d_sub = Dict(k => d[k] for k in setdiff(keys(args_and_kwargs), keys(args_and_kwargs_names)) if haskey(d, k))
        push!(get!(grouped_by_model, model, Vector{Dict}()), d_sub)
    end

    model_names = []

    for d in model_estimates_active_plot_container
        push!(model_names, d[:model_name])
    end

    model_names = unique(model_names)

    for model in model_names
        if length(grouped_by_model[model]) > 1
            diffdict_grouped = compare_args_and_kwargs(grouped_by_model[model])
            diffdict = merge_by_runid(diffdict, diffdict_grouped)
        end
    end


    annotate_ss = Vector{Pair{String, Any}}[]

    annotate_ss_page = Pair{String,Any}[]

    annotate_diff_input = Pair{String,Any}[]

    push!(annotate_diff_input, "Plot label" => reduce(vcat, diffdict[:label]))

    len_diff = length(model_estimates_active_plot_container)

    if haskey(diffdict, :parameters)
        param_nms = diffdict[:parameters] |> keys |> collect |> sort
        for param in param_nms
            result = [x === nothing ? "" : x for x in diffdict[:parameters][param]]
            push!(annotate_diff_input, String(param) => result)
        end
    end

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

    common_axis = mapreduce(k -> k[:x_axis], intersect, model_estimates_active_plot_container)

    if length(common_axis) > 0
        combined_x_axis = mapreduce(k -> k[:x_axis], union, model_estimates_active_plot_container) |> sort
    else
        combined_x_axis = 1:maximum([length(k[:x_axis]) for k in model_estimates_active_plot_container]) # model_estimates_active_plot_container[end][:x_axis]
    end
       
    for k in setdiff(keys(args_and_kwargs), 
                        [
                            :run_id, :parameters, :data, :data_in_levels,
                            :decomposition, :variables_to_plot, :data_in_deviations,:shocks_to_plot, :reference_steady_state, :x_axis,
                            :tol, :label, #:presample_periods,
                            :shocks, :shock_names,
                            :variables, :variable_names,
                            # :periods, :quadratic_matrix_equation_algorithm, :sylvester_algorithm, :lyapunov_algorithm,
                        ]
                    )

        if haskey(diffdict, k)
            push!(annotate_diff_input, args_and_kwargs_names[k] => reduce(vcat, diffdict[k]))
        end
    end
    
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

    for (i,k) in enumerate(model_estimates_active_plot_container)
        StatsPlots.plot!(legend_plot,
                        [NaN], 
                        legend_title = length(annotate_diff_input) > 2 ? nothing : annotate_diff_input[2][1],
                        label = length(annotate_diff_input) > 2 ? k[:label] isa Symbol ? string(k[:label]) : k[:label] : annotate_diff_input[2][2][i] isa String ? annotate_diff_input[2][2][i] : String(Symbol(annotate_diff_input[2][2][i])))

        foreach(n -> push!(joint_variables, String(n)), k[:variable_names] isa AbstractVector ? k[:variable_names] : (k[:variable_names],))
        foreach(n -> push!(joint_shocks, String(n)), k[:shock_names] isa AbstractVector ? k[:shock_names] : (k[:shock_names],))
    end
    
    if haskey(diffdict, :data) || haskey(diffdict, :presample_periods)
        for (i,k) in enumerate(model_estimates_active_plot_container)
            if length(data_idx) > 0
                lbl = "Data $(data_idx[i])"
            else
                lbl = "Data $(k[:label])"
            end

            StatsPlots.plot!(legend_plot,
                                    [NaN], 
                                    label = lbl,
                                    # color = pal[i]
                                    )
        end
    else
        StatsPlots.plot!(legend_plot,
                                [NaN], 
                                label = "Data",
                                color = data_color)
    end

    sort!(joint_shocks)
    sort!(joint_variables)

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
            var_idx = findfirst(==(var), k[:variable_names])
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
                shock_idx = findfirst(==(var), k[:shock_names])
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
                    
                    shocks_to_plot = fill(NaN, length(combined_x_axis))
                    shocks_to_plot[idx] = k[:shocks_to_plot][shock_idx, periods]
                    # shocks_to_plot[idx][1:k[:presample_periods]] .= NaN
                    push!(shocks_to_plot_s, shocks_to_plot) # k[:shocks_to_plot][shock_idx, periods])
                end
            else
                var_idx = findfirst(==(var), k[:variable_names])
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
                    
                    variables_to_plot = fill(NaN, length(combined_x_axis))
                    variables_to_plot[idx] = k[:variables_to_plot][var_idx, periods]
                    # variables_to_plot[idx][1:k[:presample_periods]] .= NaN

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

        p = standard_subplot(Val(:compare),
                                    plot_data, 
                                    SSs, 
                                    var, 
                                    gr_back,
                                    same_ss,
                                    pal = pal,
                                    xvals = combined_x_axis, # TODO: check different data length or presample periods. to be fixed
                                    # transparency = transparency
                                    )

        if haskey(diffdict, :data) || haskey(diffdict, :presample_periods)
            for (i,k) in enumerate(model_estimates_active_plot_container)
                # periods = min_presample_periods + 1:length(combined_x_axis)
                periods = (1:length(k[:x_axis])) .+ k[:presample_periods]

                obs_axis = collect(axiskeys(k[:data],1))

                obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

                var_idx = findfirst(==(var), k[:variable_names])

                if var ∈ string.(obs_symbols)
                    if common_axis == []
                        idx = 1:length(k[:x_axis])
                    else
                        idx = indexin(k[:x_axis], combined_x_axis)
                    end

                    data_in_deviations = fill(NaN, length(combined_x_axis))
                    data_in_deviations[idx] = k[:data_in_deviations][indexin([var], string.(obs_symbols)), periods]
                    # data_in_deviations[idx][1:k[:presample_periods]] .= NaN

                    StatsPlots.plot!(p,
                        combined_x_axis,
                        data_in_deviations .+ k[:reference_steady_state][var_idx],
                        label = "",
                        color = pal[length(model_estimates_active_plot_container) + i]
                        )
                end
            end
        else
            k = model_estimates_active_plot_container[1]

            periods = min_presample_periods + 1:size(k[:data], 2)

            obs_axis = collect(axiskeys(k[:data],1))

            obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

            var_idx = findfirst(==(var), k[:variable_names]) 

            if var ∈ string.(obs_symbols)
                data_in_deviations = k[:data_in_deviations][indexin([var], string.(obs_symbols)),:]
                data_in_deviations[1:k[:presample_periods]] .= NaN
                
                StatsPlots.plot!(p,
                    combined_x_axis,
                    data_in_deviations[periods] .+ k[:reference_steady_state][var_idx],
                    label = "",
                    color = data_color
                    )

            end
        end

        push!(pp, p)
        
        if !(plot_count % plots_per_page == 0)
            plot_count += 1
        else
            plot_count = 1

            ppp = StatsPlots.plot(pp...; attributes...)

            pl = StatsPlots.plot(framestyle = :none)

            if haskey(diffdict, :model_name)
                model_string = "multiple models"
                model_string_filename = "multiple_models"
            else
                model_string = 𝓂.model_name
                model_string_filename = 𝓂.model_name
            end

            plot_title = "Model: "*model_string*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"
            
            plot_elements = [ppp, legend_plot]

            layout_heights = [15,1]
            
            if length(annotate_diff_input) > 2
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
                annotate_ss_plot = plot_df(annotate_ss[pane]; fontsize = attributes[:annotationfontsize], title = "Relevant Steady State")

                ppp_ss = StatsPlots.plot(annotate_ss_plot; attributes..., framestyle = :box)

                push!(plot_elements, ppp_ss)
                
                push!(layout_heights, 5)
            end

            p = StatsPlots.plot(plot_elements...,
                                layout = StatsPlots.grid(length(layout_heights), 1, heights = layout_heights ./ sum(layout_heights)),
                                plot_title = plot_title; 
                                attributes_redux...)

            push!(return_plots,p)

            if show_plots
                display(p)
            end

            if save_plots
                StatsPlots.savefig(p, save_plots_path * "/estimation__" * model_string_filename * "__" * string(pane) * "." * string(save_plots_format))
            end

            pane += 1

            annotate_ss_page = Pair{String,Any}[]

            pp = []
        end
    end

    if length(pp) > 0
        ppp = StatsPlots.plot(pp...; attributes...)

        pl = StatsPlots.plot(framestyle = :none)

        if haskey(diffdict, :model_name)
            model_string = "multiple models"
            model_string_filename = "multiple_models"
        else
            model_string = 𝓂.model_name
            model_string_filename = 𝓂.model_name
        end

        plot_title = "Model: "*model_string*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"
        
        plot_elements = [ppp, legend_plot]

        layout_heights = [15,1]
        
        if length(annotate_diff_input) > 2
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

        p = StatsPlots.plot(plot_elements...,
                            layout = StatsPlots.grid(length(layout_heights), 1, heights = layout_heights ./ sum(layout_heights)),
                            plot_title = plot_title; 
                            attributes_redux...)

        push!(return_plots,p)

        if show_plots
            display(p)
        end

        if save_plots
            StatsPlots.savefig(p, save_plots_path * "/estimation__" * model_string_filename * "__" * string(pane) * "." * string(save_plots_format))
        end
    end

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
- $VARIABLES®
- $PARAMETERS®
- $ALGORITHM®
- $SHOCK_SIZE®
- $NEGATIVE_SHOCK®
- $GENERALISED_IRF®
- $INITIAL_STATE®
- $IGNORE_OBC®
- `label` [Default: `1`, Type: `Union{Real, String, Symbol}`]: label to attribute to this function call in the plots.
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMATH®
- $SAVE_PLOTS_PATH®
- $PLOTS_PER_PAGE®
- $PLOT_ATTRIBUTES®
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
                    periods::Int = 40, 
                    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all_excluding_obc, 
                    variables::Union{Symbol_input,String_input} = :all_excluding_auxiliary_and_obc,
                    parameters::ParameterType = nothing,
                    label::Union{Real, String, Symbol} = 1,
                    show_plots::Bool = true,
                    save_plots::Bool = false,
                    save_plots_format::Symbol = :pdf,
                    save_plots_path::String = ".",
                    plots_per_page::Int = 9, 
                    algorithm::Symbol = :first_order,
                    shock_size::Real = 1,
                    negative_shock::Bool = false,
                    generalised_irf::Bool = false,
                    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}} = [0.0],
                    ignore_obc::Bool = false,
                    plot_attributes::Dict = Dict(),
                    verbose::Bool = false,
                    tol::Tolerances = Tolerances(),
                    quadratic_matrix_equation_algorithm::Symbol = :schur,
                    sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = sum(1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > 1000 ? :bicgstab : :doubling,
                    lyapunov_algorithm::Symbol = :doubling)
    # @nospecialize # reduce compile time                

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > 1000 ? :bicgstab : :doubling : sylvester_algorithm[2],
                    lyapunov_algorithm = lyapunov_algorithm)

    gr_back = StatsPlots.backend() == StatsPlots.Plots.GRBackend()

    if !gr_back
        attrbts = merge(default_plot_attributes, Dict(:framestyle => :box))
    else
        attrbts = merge(default_plot_attributes, Dict())
    end

    attributes = merge(attrbts, plot_attributes)
                
    attributes_redux = copy(attributes)

    delete!(attributes_redux, :framestyle)

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks
    
    shocks = 𝓂.timings.nExo == 0 ? :none : shocks

    stochastic_model = length(𝓂.timings.exo) > 0

    obc_model = length(𝓂.obc_violation_equations) > 0

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == 𝓂.timings.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        shock_idx = 1

        obc_shocks_included = stochastic_model && obc_model && sum(abs2,shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ"),:]) > 1e-10
    elseif shocks isa KeyedArray{Float64}
        shock_idx = 1

        obc_shocks = 𝓂.timings.exo[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")]

        obc_shocks_included = stochastic_model && obc_model && sum(abs2,shocks(intersect(obc_shocks, axiskeys(shocks,1)),:)) > 1e-10
    else
        shock_idx = parse_shocks_input_to_index(shocks,𝓂.timings)

        obc_shocks_included = stochastic_model && obc_model && (intersect((((shock_idx isa Vector) || (shock_idx isa UnitRange)) && (length(shock_idx) > 0)) ? 𝓂.timings.exo[shock_idx] : [𝓂.timings.exo[shock_idx]], 𝓂.timings.exo[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")]) != [])
    end

    if shocks isa KeyedArray{Float64} || shocks isa Matrix{Float64}  
        periods = max(periods, size(shocks)[2])
    end

    variables = variables isa String_input ? variables .|> Meta.parse .|> replace_indices : variables

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings) |> sort

    if ignore_obc
        occasionally_binding_constraints = false
    else
        occasionally_binding_constraints = length(𝓂.obc_violation_equations) > 0
    end

    solve!(𝓂, parameters = parameters, opts = opts, dynamics = true, algorithm = algorithm, obc = occasionally_binding_constraints || obc_shocks_included)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)
    
    initial_state_input = copy(initial_state)

    unspecified_initial_state = initial_state == [0.0]

    if unspecified_initial_state
        if algorithm == :pruned_second_order
            initial_state = [zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars) - SSS_delta]
        elseif algorithm == :pruned_third_order
            initial_state = [zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars) - SSS_delta, zeros(𝓂.timings.nVars)]
        else
            initial_state = zeros(𝓂.timings.nVars) - SSS_delta
        end
    else
        if initial_state isa Vector{Float64}
            if algorithm == :pruned_second_order
                initial_state = [initial_state - reference_steady_state[1:𝓂.timings.nVars], zeros(𝓂.timings.nVars) - SSS_delta]
            elseif algorithm == :pruned_third_order
                initial_state = [initial_state - reference_steady_state[1:𝓂.timings.nVars], zeros(𝓂.timings.nVars) - SSS_delta, zeros(𝓂.timings.nVars)]
            else
                initial_state = initial_state - reference_steady_state[1:𝓂.timings.nVars]
            end
        else
            if algorithm ∉ [:pruned_second_order, :pruned_third_order]
                @assert initial_state isa Vector{Float64} "The solution algorithm has one state vector: initial_state must be a Vector{Float64}."
            end
        end
    end
    

    if occasionally_binding_constraints
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    elseif obc_shocks_included
        @assert algorithm ∉ [:pruned_second_order, :second_order, :pruned_third_order, :third_order] "Occasionally binding constraint shocks without enforcing the constraint is only compatible with first order perturbation solutions."

        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    else
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, false)
    end

    level = zeros(𝓂.timings.nVars)

    Y = compute_irf_responses(𝓂,
                                state_update,
                                initial_state,
                                level;
                                periods = periods,
                                shocks = shocks,
                                variables = variables,
                                shock_size = shock_size,
                                negative_shock = negative_shock,
                                generalised_irf = generalised_irf,
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
        shock_names = ["simulation"]
    elseif shocks == :none
        shock_names = ["no_shock"]
    elseif shocks isa Union{Symbol_input,String_input}
        shock_names = replace_indices_in_symbol.(𝓂.timings.exo[shock_idx])
    else
        shock_names = ["shock_matrix"]
    end
    
    variable_names = replace_indices_in_symbol.(𝓂.timings.var[var_idx])

    while length(irf_active_plot_container) > 0
        pop!(irf_active_plot_container)
    end
    
    args_and_kwargs = Dict(:run_id => length(irf_active_plot_container) + 1,
                           :model_name => 𝓂.model_name,
                           :label => label,

                           :periods => periods,
                           :shocks => shocks,
                           :variables => variables,
                           :parameters => Dict(𝓂.parameters .=> 𝓂.parameter_values),
                           :algorithm => algorithm,
                           :shock_size => shock_size,
                           :negative_shock => negative_shock,
                           :generalised_irf => generalised_irf,
                           :initial_state => initial_state_input,
                           :ignore_obc => ignore_obc,

                           :NSSS_acceptance_tol => tol.NSSS_acceptance_tol,
                           :NSSS_xtol => tol.NSSS_xtol,
                           :NSSS_ftol => tol.NSSS_ftol,
                           :NSSS_rel_xtol => tol.NSSS_rel_xtol,
                           :qme_tol => tol.qme_tol,
                           :qme_acceptance_tol => tol.qme_acceptance_tol,
                           :sylvester_tol => tol.sylvester_tol,
                           :sylvester_acceptance_tol => tol.sylvester_acceptance_tol,
                           :lyapunov_tol => tol.lyapunov_tol,
                           :lyapunov_acceptance_tol => tol.lyapunov_acceptance_tol,
                           :droptol => tol.droptol,
                           :dependencies_tol => tol.dependencies_tol,

                           :quadratic_matrix_equation_algorithm => quadratic_matrix_equation_algorithm,
                           :sylvester_algorithm => sylvester_algorithm,
                           :lyapunov_algorithm => lyapunov_algorithm,

                           :plot_data => Y,
                           :reference_steady_state => reference_steady_state[var_idx],
                           :variable_names => variable_names,
                           :shock_names => shock_names
                           )
    
    push!(irf_active_plot_container, args_and_kwargs)

    orig_pal = StatsPlots.palette(attributes_redux[:palette])

    total_pal_len = 100

    alpha_reduction_factor = 0.7

    pal = mapreduce(x -> StatsPlots.coloralpha.(orig_pal, alpha_reduction_factor ^ x), vcat, 0:(total_pal_len ÷ length(orig_pal)) - 1) |> StatsPlots.palette

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
                variable_name = variable_names[i]

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
                        shock_string = ": " * replace_indices_in_symbol(𝓂.timings.exo[shock_idx[shock]])
                        shock_name = replace_indices_in_symbol(𝓂.timings.exo[shock_idx[shock]])
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
                        StatsPlots.savefig(p, save_plots_path * "/irf__" * 𝓂.model_name * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
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
                shock_string = ": " * replace_indices_in_symbol(𝓂.timings.exo[shock_idx[shock]])
                shock_name = replace_indices_in_symbol(𝓂.timings.exo[shock_idx[shock]])
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
                StatsPlots.savefig(p, save_plots_path * "/irf__" * 𝓂.model_name * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
            end
        end
    end

    return return_plots
end


function standard_subplot(irf_data::AbstractVector{S}, 
                            steady_state::S, 
                            variable_name::String, 
                            gr_back::Bool;
                            pal::StatsPlots.ColorPalette = StatsPlots.palette(:auto),
                            xvals = 1:length(irf_data)) where S <: AbstractFloat
    can_dual_axis = gr_back && all((irf_data .+ steady_state) .> eps(Float32)) && (steady_state > eps(Float32))
    
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
                            variable_name::String, 
                            gr_back::Bool, 
                            same_ss::Bool; 
                            xvals = 1:maximum(length.(irf_data)),
                            pal::StatsPlots.ColorPalette = StatsPlots.palette(:auto),
                            transparency::Float64 = .6) where S <: AbstractFloat
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
            
            if can_dual_axis && same_ss
                push!(plot_dat, y .+ ss)
                plot_ss = ss
            else
                if same_ss
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
                        ylabel = same_ss ? "Level" : "abs. " * LaTeXStrings.L"\Delta",
                        color = pal[mod1.(pal_val, length(pal))]',
                        xrotation = xrotation,
                        label = "")

    StatsPlots.hline!([same_ss ? stst : 0], 
                      color = :black, 
                      label = "")

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
                            transparency::Float64 = .6) where S <: AbstractFloat
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
    padded = [vcat(collect(v), fill(0, maxlen - length(v))) for v in plot_dat]

    # now you can hcat
    plot_data = reduce(hcat, padded)

    p = StatsPlots.plot(xvals,
                    sum(plot_data, dims = 2), 
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

    StatsPlots.xticks!(p, chosen_xticks_bar[1][1], chosen_xticks[1][2])

    StatsPlots.hline!([0], 
                        color = :black, 
                        label = "")
    
    StatsPlots.plot!(sum(plot_data, dims = 2), 
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
- $VARIABLES®
- $PARAMETERS®
- $ALGORITHM®
- $SHOCK_SIZE®
- $NEGATIVE_SHOCK®
- $GENERALISED_IRF®
- $INITIAL_STATE®
- $IGNORE_OBC®
- $LABEL®
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMATH®
- $SAVE_PLOTS_PATH®
- $PLOTS_PER_PAGE®
- $PLOT_ATTRIBUTES®
- `plot_type` [Default: `:compare`, Type: `Symbol`]: plot type used to represent results. `:compare` means results are shown as separate lines. `:stack` means results are stacked.
- `transparency` [Default: `0.6`, Type: `Float64`]: transparency of stacked bars. Only relevant if `plot_type` is `:stack`.
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
                    periods::Int = 40, 
                    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all_excluding_obc, 
                    variables::Union{Symbol_input,String_input} = :all_excluding_auxiliary_and_obc,
                    parameters::ParameterType = nothing,
                    label::Union{Real, String, Symbol} = length(irf_active_plot_container) + 1,
                    show_plots::Bool = true,
                    save_plots::Bool = false,
                    save_plots_format::Symbol = :pdf,
                    save_plots_path::String = ".",
                    plots_per_page::Int = 6, 
                    algorithm::Symbol = :first_order,
                    shock_size::Real = 1,
                    negative_shock::Bool = false,
                    generalised_irf::Bool = false,
                    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}} = [0.0],
                    ignore_obc::Bool = false,
                    plot_type::Symbol = :compare,
                    plot_attributes::Dict = Dict(),
                    transparency::Float64 = .6,
                    verbose::Bool = false,
                    tol::Tolerances = Tolerances(),
                    quadratic_matrix_equation_algorithm::Symbol = :schur,
                    sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = sum(1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > 1000 ? :bicgstab : :doubling,
                    lyapunov_algorithm::Symbol = :doubling)
    # @nospecialize # reduce compile time                

    @assert plot_type ∈ [:compare, :stack] "plot_type must be either :compare or :stack"

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > 1000 ? :bicgstab : :doubling : sylvester_algorithm[2],
                    lyapunov_algorithm = lyapunov_algorithm)

    gr_back = StatsPlots.backend() == StatsPlots.Plots.GRBackend()

    if !gr_back
        attrbts = merge(default_plot_attributes, Dict(:framestyle => :box))
    else
        attrbts = merge(default_plot_attributes, Dict())
    end

    attributes = merge(attrbts, plot_attributes)
                
    attributes_redux = copy(attributes)

    delete!(attributes_redux, :framestyle)

    orig_pal = StatsPlots.palette(attributes_redux[:palette])

    total_pal_len = 100

    alpha_reduction_factor = 0.7

    pal = mapreduce(x -> StatsPlots.coloralpha.(orig_pal, alpha_reduction_factor ^ x), vcat, 0:(total_pal_len ÷ length(orig_pal)) - 1) |> StatsPlots.palette

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks
    
    shocks = 𝓂.timings.nExo == 0 ? :none : shocks

    stochastic_model = length(𝓂.timings.exo) > 0

    obc_model = length(𝓂.obc_violation_equations) > 0

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == 𝓂.timings.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        shock_idx = 1

        obc_shocks_included = stochastic_model && obc_model && sum(abs2,shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ"),:]) > 1e-10
    elseif shocks isa KeyedArray{Float64}
        shock_idx = 1

        obc_shocks = 𝓂.timings.exo[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")]

        obc_shocks_included = stochastic_model && obc_model && sum(abs2,shocks(intersect(obc_shocks, axiskeys(shocks,1)),:)) > 1e-10
    else
        shock_idx = parse_shocks_input_to_index(shocks,𝓂.timings)

        obc_shocks_included = stochastic_model && obc_model && (intersect((((shock_idx isa Vector) || (shock_idx isa UnitRange)) && (length(shock_idx) > 0)) ? 𝓂.timings.exo[shock_idx] : [𝓂.timings.exo[shock_idx]], 𝓂.timings.exo[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")]) != [])
    end

    if shocks isa KeyedArray{Float64} || shocks isa Matrix{Float64}  
        periods = max(periods, size(shocks)[2])
    end

    variables = variables isa String_input ? variables .|> Meta.parse .|> replace_indices : variables

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings) |> sort

    if ignore_obc
        occasionally_binding_constraints = false
    else
        occasionally_binding_constraints = length(𝓂.obc_violation_equations) > 0
    end

    solve!(𝓂, parameters = parameters, opts = opts, dynamics = true, algorithm = algorithm, obc = occasionally_binding_constraints || obc_shocks_included)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)
    
    initial_state_input = copy(initial_state)

    unspecified_initial_state = initial_state == [0.0]

    if unspecified_initial_state
        if algorithm == :pruned_second_order
            initial_state = [zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars) - SSS_delta]
        elseif algorithm == :pruned_third_order
            initial_state = [zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars) - SSS_delta, zeros(𝓂.timings.nVars)]
        else
            initial_state = zeros(𝓂.timings.nVars) - SSS_delta
        end
    else
        if initial_state isa Vector{Float64}
            if algorithm == :pruned_second_order
                initial_state = [initial_state - reference_steady_state[1:𝓂.timings.nVars], zeros(𝓂.timings.nVars) - SSS_delta]
            elseif algorithm == :pruned_third_order
                initial_state = [initial_state - reference_steady_state[1:𝓂.timings.nVars], zeros(𝓂.timings.nVars) - SSS_delta, zeros(𝓂.timings.nVars)]
            else
                initial_state = initial_state - reference_steady_state[1:𝓂.timings.nVars]
            end
        else
            if algorithm ∉ [:pruned_second_order, :pruned_third_order]
                @assert initial_state isa Vector{Float64} "The solution algorithm has one state vector: initial_state must be a Vector{Float64}."
            end
        end
    end


    if occasionally_binding_constraints
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    elseif obc_shocks_included
        @assert algorithm ∉ [:pruned_second_order, :second_order, :pruned_third_order, :third_order] "Occasionally binding constraint shocks without enforcing the constraint is only compatible with first order perturbation solutions."

        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    else
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, false)
    end

    level = zeros(𝓂.timings.nVars)

    Y = compute_irf_responses(𝓂,
                                state_update,
                                initial_state,
                                level;
                                periods = periods,
                                shocks = shocks,
                                variables = variables,
                                shock_size = shock_size,
                                negative_shock = negative_shock,
                                generalised_irf = generalised_irf,
                                enforce_obc = occasionally_binding_constraints,
                                algorithm = algorithm)

    if !generalised_irf || occasionally_binding_constraints
        Y = Y .+ SSS_delta[var_idx]
    end

    if shocks == :simulate
        shock_names = ["simulation"]
    elseif shocks == :none
        shock_names = ["no_shock"]
    elseif shocks isa Union{Symbol_input,String_input}
        shock_names = replace_indices_in_symbol.(𝓂.timings.exo[shock_idx])
    else
        shock_names = ["shock_matrix"]
    end
    
    variable_names = replace_indices_in_symbol.(𝓂.timings.var[var_idx])

    args_and_kwargs = Dict(:run_id => length(irf_active_plot_container) + 1,
                           :model_name => 𝓂.model_name,
                           :label => label,

                           :periods => periods,
                           :shocks => shocks,
                           :variables => variables,
                           :parameters => Dict(𝓂.parameters .=> 𝓂.parameter_values),
                           :algorithm => algorithm,
                           :shock_size => shock_size,
                           :negative_shock => negative_shock,
                           :generalised_irf => generalised_irf,
                           :initial_state => initial_state_input,
                           :ignore_obc => ignore_obc,

                           :NSSS_acceptance_tol => tol.NSSS_acceptance_tol,
                           :NSSS_xtol => tol.NSSS_xtol,
                           :NSSS_ftol => tol.NSSS_ftol,
                           :NSSS_rel_xtol => tol.NSSS_rel_xtol,
                           :qme_tol => tol.qme_tol,
                           :qme_acceptance_tol => tol.qme_acceptance_tol,
                           :sylvester_tol => tol.sylvester_tol,
                           :sylvester_acceptance_tol => tol.sylvester_acceptance_tol,
                           :lyapunov_tol => tol.lyapunov_tol,
                           :lyapunov_acceptance_tol => tol.lyapunov_acceptance_tol,
                           :droptol => tol.droptol,
                           :dependencies_tol => tol.dependencies_tol,

                           :quadratic_matrix_equation_algorithm => quadratic_matrix_equation_algorithm,
                           :sylvester_algorithm => sylvester_algorithm,
                           :lyapunov_algorithm => lyapunov_algorithm,
                           :plot_data => Y,
                           :reference_steady_state => reference_steady_state[var_idx],
                           :variable_names => variable_names,
                           :shock_names => shock_names
                           )

    no_duplicate = all(
        !(all((
            get(dict, :parameters, nothing) == args_and_kwargs[:parameters],
            get(dict, :shock_names, nothing) == args_and_kwargs[:shock_names],
            get(dict, :shocks, nothing) == args_and_kwargs[:shocks],
            get(dict, :initial_state, nothing) == args_and_kwargs[:initial_state],
            all(get(dict, k, nothing) == get(args_and_kwargs, k, nothing) for k in keys(args_and_kwargs_names))
        )))
        for dict in irf_active_plot_container
    )# "New plot must be different from previous plot. Use the version without ! to plot."

    if no_duplicate 
        push!(irf_active_plot_container, args_and_kwargs)
    else
        @info "Plot with same parameters already exists. Using previous plot data to create plot."
    end

    # 1. Keep only certain keys from each dictionary
    reduced_vector = [
        Dict(k => d[k] for k in vcat(:run_id, :label, keys(args_and_kwargs_names)...) if haskey(d, k))
        for d in irf_active_plot_container
    ]

    diffdict = compare_args_and_kwargs(reduced_vector)

    # 2. Group the original vector by :model_name
    grouped_by_model = Dict{Any, Vector{Dict}}()

    for d in irf_active_plot_container
        model = d[:model_name]
        d_sub = Dict(k => d[k] for k in setdiff(keys(args_and_kwargs), keys(args_and_kwargs_names)) if haskey(d, k))
        push!(get!(grouped_by_model, model, Vector{Dict}()), d_sub)
    end

    model_names = []

    for d in irf_active_plot_container
        push!(model_names, d[:model_name])
    end

    model_names = unique(model_names)

    for model in model_names
        if length(grouped_by_model[model]) > 1
            diffdict_grouped = compare_args_and_kwargs(grouped_by_model[model])
            diffdict = merge_by_runid(diffdict, diffdict_grouped)
        end
    end

    # @assert haskey(diffdict, :parameters) || haskey(diffdict, :shock_names) || haskey(diffdict, :initial_state) || any(haskey.(Ref(diffdict), keys(args_and_kwargs_names))) "New plot must be different from previous plot. Use the version without ! to plot."
    
    annotate_ss = Vector{Pair{String, Any}}[]

    annotate_ss_page = Pair{String,Any}[]

    annotate_diff_input = Pair{String,Any}[]

    push!(annotate_diff_input, "Plot label" => reduce(vcat, diffdict[:label]))

    len_diff = length(irf_active_plot_container)

    if haskey(diffdict, :parameters)
        param_nms = diffdict[:parameters] |> keys |> collect |> sort
        for param in param_nms
            result = [x === nothing ? "" : x for x in diffdict[:parameters][param]]
            push!(annotate_diff_input, String(param) => result)
        end
    end
    
    if haskey(diffdict, :shocks)
        # if all(length.(diffdict[:shock_names]) .== 1)
            # push!(annotate_diff_input, "Shock" => reduce(vcat, map(x -> typeof(x) <: AbstractVector ? "Multiple shocks" : typeof(x) <: AbstractMatrix ? "Shock Matrix" : x, diffdict[:shocks])))
        # else
            push!(annotate_diff_input, "Shock" => [typeof(x) <: AbstractMatrix ? "Shock Matrix" : x for x in diffdict[:shocks]])
        # end
    end
    
    if haskey(diffdict, :initial_state)
        vals = diffdict[:initial_state]

        # Map each distinct non-[0.0] value to its running index
        seen = Dict{typeof(first(vals)), Int}()
        next_idx = 0

        labels = String[]
        for v in vals
            if v == [0.0]
                push!(labels, "")                  # put nothing
            else
                if !haskey(seen, v)
                    next_idx += 1                  # running index does not count [0.0]
                    seen[v] = next_idx
                end
                push!(labels, "#$(seen[v])")
            end
        end

        push!(annotate_diff_input, "Initial state" => labels)
    end
    
    same_shock_direction = true
    
    for k in setdiff(keys(args_and_kwargs), 
                        [
                            :run_id, :parameters, :plot_data, :tol, :reference_steady_state, :initial_state, :label,
                            :shocks, :shock_names,
                            :variables, :variable_names,
                            # :periods, :quadratic_matrix_equation_algorithm, :sylvester_algorithm, :lyapunov_algorithm,
                        ]
                    )

        if haskey(diffdict, k)
            push!(annotate_diff_input, args_and_kwargs_names[k] => reduce(vcat,diffdict[k]))
            
            if k == :negative_shock
                same_shock_direction = false
            end
        end
    end

    # if haskey(diffdict, :shock_names)
    #     if !all(length.(diffdict[:shock_names]) .== 1)
    #         push!(annotate_diff_input, "Shock name" => map(x->x[1], diffdict[:shock_names]))
    #     end
    # end

    legend_plot = StatsPlots.plot(framestyle = :none, 
                                    legend = :inside, 
                                    legend_columns = length(irf_active_plot_container)) 
    
    joint_shocks = OrderedSet{String}()
    joint_variables = OrderedSet{String}()
    single_shock_per_irf = true
    
    max_periods = 0
    for (i,k) in enumerate(irf_active_plot_container)
        if plot_type == :stack
            StatsPlots.bar!(legend_plot,
                            [NaN], 
                            legend_title = length(annotate_diff_input) > 2 ? nothing : annotate_diff_input[2][1],
                            alpha = transparency,
                            lw = 0,  # This removes the lines around the bars
                            linecolor = :transparent,
                            label = length(annotate_diff_input) > 2 ? k[:label] isa Symbol ? string(k[:label]) : k[:label] : annotate_diff_input[2][2][i] isa String ? annotate_diff_input[2][2][i] : String(Symbol(annotate_diff_input[2][2][i])))
        elseif plot_type == :compare
            StatsPlots.plot!(legend_plot,
                            [NaN], 
                            legend_title = length(annotate_diff_input) > 2 ? nothing : annotate_diff_input[2][1],
                            label = length(annotate_diff_input) > 2 ? k[:label] isa Symbol ? string(k[:label]) : k[:label] : annotate_diff_input[2][2][i] isa String ? annotate_diff_input[2][2][i] : String(Symbol(annotate_diff_input[2][2][i])))
        end

        foreach(n -> push!(joint_variables, String(n)), k[:variable_names] isa AbstractVector ? k[:variable_names] : (k[:variable_names],))
        foreach(n -> push!(joint_shocks, String(n)), k[:shock_names] isa AbstractVector ? k[:shock_names] : (k[:shock_names],))

        single_shock_per_irf = single_shock_per_irf && length(k[:shock_names]) == 1

        max_periods = max(max_periods, size(k[:plot_data],2))
    end

    sort!(joint_shocks)
    sort!(joint_variables)

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
                var_idx = findfirst(==(var), k[:variable_names])
                shock_idx = shock == :single_shock_per_irf ? 1 : findfirst(==(shock), k[:shock_names])
                
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
                var_idx = findfirst(==(var), k[:variable_names])
                shock_idx = shock == :single_shock_per_irf ? 1 : findfirst(==(shock), k[:shock_names])

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
                elseif shock isa Union{Symbol_input,String_input}
                    shock_string = ": " * shock
                    shock_name = shock
                else
                    shock_string = "Series of shocks"
                    shock_name = "shock_matrix"
                end

                ppp = StatsPlots.plot(pp...; attributes...)
                
                if haskey(diffdict, :model_name)
                    model_string = "multiple models"
                    model_string_filename = "multiple_models"
                else
                    model_string = 𝓂.model_name
                    model_string_filename = 𝓂.model_name
                end

                plot_title = "Model: "*model_string*"        " * shock_dir *  shock_string *"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"

                plot_elements = [ppp, legend_plot]

                layout_heights = [15,1]
                
                if length(annotate_diff_input) > 2
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

                p = StatsPlots.plot(plot_elements...,
                                    layout = StatsPlots.grid(length(layout_heights), 1, heights = layout_heights ./ sum(layout_heights)),
                                    plot_title = plot_title; 
                                    attributes_redux...)

                push!(return_plots,p)

                if show_plots
                    display(p)
                end

                if save_plots
                    StatsPlots.savefig(p, save_plots_path * "/irf__" * model_string_filename * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
                end

                pane += 1

                annotate_ss_page = Pair{String,Any}[]

                pp = []
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
            elseif shock isa Union{Symbol_input,String_input}
                shock_string = ": " * shock
                shock_name = shock
            else
                shock_string = "Series of shocks"
                shock_name = "shock_matrix"
            end

            ppp = StatsPlots.plot(pp...; attributes...)
            
            if haskey(diffdict, :model_name)
                model_string = "multiple models"
                model_string_filename = "multiple_models"
            else
                model_string = 𝓂.model_name
                model_string_filename = 𝓂.model_name
            end

            plot_title = "Model: "*model_string*"        " * shock_dir *  shock_string *"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"

            plot_elements = [ppp, legend_plot]

            layout_heights = [15,1]

            if length(annotate_diff_input) > 2
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

            p = StatsPlots.plot(plot_elements...,
                                layout = StatsPlots.grid(length(layout_heights), 1, heights = layout_heights ./ sum(layout_heights)),
                                plot_title = plot_title; 
                                attributes_redux...)

            push!(return_plots,p)

            if show_plots
                display(p)
            end

            if save_plots
                StatsPlots.savefig(p, save_plots_path * "/irf__" * model_string_filename * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
            end
        end

        annotate_ss = Vector{Pair{String, Any}}[]

        annotate_ss_page = Pair{String,Any}[]
    end

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

    # run_id → index map for each dict
    idx_maps = [Dict(r => i for (i, r) in enumerate(d[:run_id])) for d in dicts]

    for (j, d) in enumerate(dicts)
        idx_map = idx_maps[j]
        for (k, v) in d
            k === :run_id && continue

            if v isa AbstractVector && length(v) == length(d[:run_id])
                merged[k] = [haskey(idx_map, r) ? v[idx_map[r]] : nothing for r in all_runids]
            elseif v isa Dict
                sub = get!(merged, k, Dict{Symbol,Any}())
                for (kk, vv) in v
                    if vv isa AbstractVector && length(vv) == length(d[:run_id])
                        sub[kk] = [haskey(idx_map, r) ? vv[idx_map[r]] : nothing for r in all_runids]
                    else
                        sub[kk] = [vv for _ in 1:n]
                    end
                end
            else
                merged[k] = [v for _ in 1:n]
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


function plot_df(plot_vector::Vector{Pair{String,Any}}; fontsize::Real = 8, title::String = "")
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
- $VARIABLES®
- $PARAMETERS®
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMATH®
- $SAVE_PLOTS_PATH®
- $PLOTS_PER_PAGE®
- $PLOT_ATTRIBUTES®
- $MAX_ELEMENTS_PER_LEGENDS_ROW®
- $EXTRA_LEGEND_SPACE®
- $QME®
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

plot_conditional_variance_decomposition(RBC_CME)
```
"""
function plot_conditional_variance_decomposition(𝓂::ℳ;
                                                periods::Int = 40, 
                                                variables::Union{Symbol_input,String_input} = :all,
                                                parameters::ParameterType = nothing,
                                                show_plots::Bool = true,
                                                save_plots::Bool = false,
                                                save_plots_format::Symbol = :pdf,
                                                save_plots_path::String = ".",
                                                plots_per_page::Int = 9, 
                                                plot_attributes::Dict = Dict(),
                                                max_elements_per_legend_row::Int = 4,
                                                extra_legend_space::Float64 = 0.0,
                                                verbose::Bool = false,
                                                tol::Tolerances = Tolerances(),
                                                quadratic_matrix_equation_algorithm::Symbol = :schur,
                                                lyapunov_algorithm::Symbol = :doubling)
    # @nospecialize # reduce compile time                                            

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                lyapunov_algorithm = lyapunov_algorithm)

    gr_back = StatsPlots.backend() == StatsPlots.Plots.GRBackend()

    if !gr_back
        attrbts = merge(default_plot_attributes, Dict(:framestyle => :box))
    else
        attrbts = merge(default_plot_attributes, Dict())
    end

    attributes = merge(attrbts, plot_attributes)
                                            
    attributes_redux = copy(attributes)

    delete!(attributes_redux, :framestyle)

    fevds = get_conditional_variance_decomposition(𝓂,
                                                    periods = 1:periods,
                                                    parameters = parameters,
                                                    verbose = verbose,
                                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                    lyapunov_algorithm = lyapunov_algorithm,
                                                    tol = tol)

    variables = variables isa String_input ? variables .|> Meta.parse .|> replace_indices : variables

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings) |> sort

    fevds = fevds isa KeyedArray ? axiskeys(fevds,1) isa Vector{String} ? rekey(fevds, 1 => axiskeys(fevds,1) .|> Meta.parse .|> replace_indices) : fevds : fevds

    fevds = fevds isa KeyedArray ? axiskeys(fevds,2) isa Vector{String} ? rekey(fevds, 2 => axiskeys(fevds,2) .|> Meta.parse .|> replace_indices) : fevds : fevds

    vars_to_plot = intersect(axiskeys(fevds)[1],𝓂.timings.var[var_idx])
    
    shocks_to_plot = axiskeys(fevds)[2]

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

    orig_pal = StatsPlots.palette(attributes_redux[:palette])

    total_pal_len = 100

    alpha_reduction_factor = 0.7

    pal = mapreduce(x -> StatsPlots.coloralpha.(orig_pal, alpha_reduction_factor ^ x), vcat, 0:(total_pal_len ÷ length(orig_pal)) - 1) |> StatsPlots.palette

    n_subplots = length(var_idx)
    pp = []
    pane = 1
    plot_count = 1
    return_plots = []

    for k in vars_to_plot
        if gr_back
            push!(pp,StatsPlots.groupedbar(fevds(k,:,:)', 
            title = replace_indices_in_symbol(k), 
            bar_position = :stack,
            color = pal[mod1.(1:length(shocks_to_plot), length(pal))]',
            linecolor = :transparent,
            legend = :none))
        else
            push!(pp,StatsPlots.groupedbar(fevds(k,:,:)', 
            title = replace_indices_in_symbol(k), 
            bar_position = :stack, 
            color = pal[mod1.(1:length(shocks_to_plot), length(pal))]',
            linecolor = :transparent,
            label = reshape(string.(replace_indices_in_symbol.(shocks_to_plot)),1,length(shocks_to_plot))))
        end

        if !(plot_count % plots_per_page == 0)
            plot_count += 1
        else
            plot_count = 1

            ppp = StatsPlots.plot(pp...; attributes...)
            
            pp = StatsPlots.bar(fill(NaN,1,length(shocks_to_plot)), 
                                label = reshape(string.(replace_indices_in_symbol.(shocks_to_plot)),1,length(shocks_to_plot)), 
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
                StatsPlots.savefig(p, save_plots_path * "/fevd__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
            end

            pane += 1
            pp = []
        end
    end

    if length(pp) > 0
        ppp = StatsPlots.plot(pp...; attributes...)

        pp = StatsPlots.bar(fill(NaN,1,length(shocks_to_plot)), 
                            label = reshape(string.(replace_indices_in_symbol.(shocks_to_plot)),1,length(shocks_to_plot)), 
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
            StatsPlots.savefig(p, save_plots_path * "/fevd__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
        end
    end

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
- $VARIABLES®
- `algorithm` [Default: `:first_order`, Type: Union{Symbol,Vector{Symbol}}]: solution algorithm for which to show the IRFs. Can be more than one, e.g.: `[:second_order,:pruned_third_order]`"
- `σ` [Default: `2`, Type: `Union{Int64,Float64}`]: defines the range of the state variable around the (non) stochastic steady state in standard deviations. E.g. a value of 2 means that the state variable is plotted for values of the (non) stochastic steady state in standard deviations +/- 2 standard deviations.
- $PARAMETERS®
- $IGNORE_OBC®
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMATH®
- $SAVE_PLOTS_PATH®
- `plots_per_page` [Default: `6`, Type: `Int`]: how many plots to show per page
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

plot_solution(RBC_CME, :k)
```
"""
function plot_solution(𝓂::ℳ,
                        state::Union{Symbol,String};
                        variables::Union{Symbol_input,String_input} = :all,
                        algorithm::Union{Symbol,Vector{Symbol}} = :first_order,
                        σ::Union{Int64,Float64} = 2,
                        parameters::ParameterType = nothing,
                        ignore_obc::Bool = false,
                        show_plots::Bool = true,
                        save_plots::Bool = false,
                        save_plots_format::Symbol = :pdf,
                        save_plots_path::String = ".",
                        plots_per_page::Int = 6,
                        plot_attributes::Dict = Dict(),
                        verbose::Bool = false,
                        tol::Tolerances = Tolerances(),
                        quadratic_matrix_equation_algorithm::Symbol = :schur,
                        sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = sum(1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > 1000 ? :bicgstab : :doubling,
                        lyapunov_algorithm::Symbol = :doubling)
    # @nospecialize # reduce compile time                    
    
    opts = merge_calculation_options(tol = tol, verbose = verbose,
                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                        sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                        sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > 1000 ? :bicgstab : :doubling : sylvester_algorithm[2],
                        lyapunov_algorithm = lyapunov_algorithm)

    gr_back = StatsPlots.backend() == StatsPlots.Plots.GRBackend()

    if !gr_back
        attrbts = merge(default_plot_attributes, Dict(:framestyle => :box))
    else
        attrbts = merge(default_plot_attributes, Dict())
    end

    attributes = merge(attrbts, plot_attributes)
                    
    attributes_redux = copy(attributes)

    delete!(attributes_redux, :framestyle)

    state = state isa Symbol ? state : state |> Meta.parse |> replace_indices

    @assert state ∈ 𝓂.timings.past_not_future_and_mixed "Invalid state. Choose one from:"*repr(𝓂.timings.past_not_future_and_mixed)

    @assert length(setdiff(algorithm isa Symbol ? [algorithm] : algorithm, [:third_order, :pruned_third_order, :second_order, :pruned_second_order, :first_order])) == 0 "Invalid algorithm. Choose any combination of: :third_order, :pruned_third_order, :second_order, :pruned_second_order, :first_order"

    if algorithm isa Symbol
        algorithm = [algorithm]
    end

    if ignore_obc
        occasionally_binding_constraints = false
    else
        occasionally_binding_constraints = length(𝓂.obc_violation_equations) > 0
    end

    for a in algorithm
        solve!(𝓂, opts = opts, algorithm = a, dynamics = true, parameters = parameters, obc = occasionally_binding_constraints)
    end

    SS_and_std = get_moments(𝓂, 
                            derivatives = false,
                            parameters = parameters,
                            variables = :all,
                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                            sylvester_algorithm = sylvester_algorithm,
                            lyapunov_algorithm = lyapunov_algorithm,
                            tol = tol,
                            verbose = verbose)

    SS_and_std[:non_stochastic_steady_state] = SS_and_std[:non_stochastic_steady_state] isa KeyedArray ? axiskeys(SS_and_std[:non_stochastic_steady_state],1) isa Vector{String} ? rekey(SS_and_std[:non_stochastic_steady_state], 1 => axiskeys(SS_and_std[:non_stochastic_steady_state],1).|> x->Symbol.(replace.(x, "{" => "◖", "}" => "◗"))) : SS_and_std[:non_stochastic_steady_state] : SS_and_std[:non_stochastic_steady_state]
    
    SS_and_std[:standard_deviation] = SS_and_std[:standard_deviation] isa KeyedArray ? axiskeys(SS_and_std[:standard_deviation],1) isa Vector{String} ? rekey(SS_and_std[:standard_deviation], 1 => axiskeys(SS_and_std[:standard_deviation],1).|> x->Symbol.(replace.(x, "{" => "◖", "}" => "◗"))) : SS_and_std[:standard_deviation] : SS_and_std[:standard_deviation]

    full_NSSS = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

    full_NSSS[indexin(𝓂.aux,full_NSSS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)

    full_SS = [s ∈ 𝓂.exo_present ? 0 : SS_and_std[:non_stochastic_steady_state](s) for s in full_NSSS]

    variables = variables isa String_input ? variables .|> Meta.parse .|> replace_indices : variables

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings) |> sort

    vars_to_plot = intersect(axiskeys(SS_and_std[:non_stochastic_steady_state])[1],𝓂.timings.var[var_idx])

    state_range = collect(range(-SS_and_std[:standard_deviation](state), SS_and_std[:standard_deviation](state), 100)) * σ
    
    state_selector = state .== 𝓂.var

    n_subplots = length(var_idx)
    pp = []
    pane = 1
    plot_count = 1
    return_plots = []

    labels = Dict(  :first_order            => ["1st order perturbation",           "Non-stochastic Steady State"],
                    :second_order           => ["2nd order perturbation",           "Stochastic Steady State (2nd order)"],
                    :pruned_second_order    => ["Pruned 2nd order perturbation",    "Stochastic Steady State (Pruned 2nd order)"],
                    :third_order            => ["3rd order perturbation",           "Stochastic Steady State (3rd order)"],
                    :pruned_third_order     => ["Pruned 3rd order perturbation",    "Stochastic Steady State (Pruned 3rd order)"])

    orig_pal = StatsPlots.palette(attributes_redux[:palette])

    total_pal_len = 100

    alpha_reduction_factor = 0.7

    pal = mapreduce(x -> StatsPlots.coloralpha.(orig_pal, alpha_reduction_factor ^ x), vcat, 0:(total_pal_len ÷ length(orig_pal)) - 1) |> StatsPlots.palette

    legend_plot = StatsPlots.plot(framestyle = :none, 
                                    legend = :inside) 

    for (i,a) in enumerate(algorithm)
        StatsPlots.plot!([NaN], 
                        color = pal[mod1(i, length(pal))],
                        label = labels[a][1])
    end
    
    for (i,a) in enumerate(algorithm)
        StatsPlots.scatter!([NaN], 
                            color = pal[mod1(i, length(pal))],
                            label = labels[a][2])
    end

    if any(x -> contains(string(x), "◖"), full_NSSS)
        full_NSSS_decomposed = decompose_name.(full_NSSS)
        full_NSSS = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in full_NSSS_decomposed]
    end

    relevant_SS_dictionnary = Dict{Symbol,Vector{Float64}}()

    for a in algorithm
        relevant_SS = get_steady_state(𝓂, algorithm = a, return_variables_only = true, derivatives = false,
                                        tol = opts.tol,
                                        verbose = opts.verbose,
                                        quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                        sylvester_algorithm = [opts.sylvester_algorithm², opts.sylvester_algorithm³])

        full_SS = [s ∈ 𝓂.exo_present ? 0 : relevant_SS(s) for s in full_NSSS]

        push!(relevant_SS_dictionnary, a => full_SS)
    end

    if :first_order ∉ algorithm
        relevant_SS = get_steady_state(𝓂, algorithm = :first_order, return_variables_only = true, derivatives = false,
                                        tol = opts.tol,
                                        verbose = opts.verbose,
                                        quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                        sylvester_algorithm = [opts.sylvester_algorithm², opts.sylvester_algorithm³])

        full_SS = [s ∈ 𝓂.exo_present ? 0 : relevant_SS(s) for s in full_NSSS]

        push!(relevant_SS_dictionnary, :first_order => full_SS)
    end

    has_impact_dict = Dict()
    variable_dict = Dict()

    NSSS = relevant_SS_dictionnary[:first_order]

    for a in algorithm
        SSS_delta = collect(NSSS - relevant_SS_dictionnary[a])

        var_state_range = []

        for x in state_range
            if a == :pruned_second_order
                initial_state = [state_selector * x, -SSS_delta]
            elseif a == :pruned_third_order
                initial_state = [state_selector * x, -SSS_delta, zero(SSS_delta)]
            else
                initial_state = collect(relevant_SS_dictionnary[a]) .+ state_selector * x
            end

            push!(var_state_range, get_irf(𝓂, algorithm = a, periods = 1, ignore_obc = ignore_obc, initial_state = initial_state, shocks = :none, levels = true, variables = :all)[:,1,1] |> collect)
        end

        var_state_range = hcat(var_state_range...)

        variable_output = Dict()
        impact_output   = Dict()

        for k in vars_to_plot
            idx = indexin([k], 𝓂.var)

            push!(variable_output,  k => var_state_range[idx,:]) 
            
            push!(impact_output,    k => any(abs.(sum(var_state_range[idx,:]) / size(var_state_range, 2) .- var_state_range[idx,:]) .> eps(Float32)))
        end

        push!(variable_dict,    a => variable_output)
        push!(has_impact_dict,  a => impact_output)
    end

    has_impact_var_dict = Dict()

    for k in vars_to_plot
        has_impact = false

        for a in algorithm
            has_impact = has_impact || has_impact_dict[a][k]
        end

        if !has_impact
            n_subplots -= 1
        end

        push!(has_impact_var_dict, k => has_impact)
    end

    for k in vars_to_plot
        if !has_impact_var_dict[k] continue end

        Pl = StatsPlots.plot() 

        for (i,a) in enumerate(algorithm)
            StatsPlots.plot!(state_range .+ relevant_SS_dictionnary[a][indexin([state], 𝓂.var)][1], 
                variable_dict[a][k][1,:], 
                ylabel = replace_indices_in_symbol(k)*"₍₀₎", 
                xlabel = replace_indices_in_symbol(state)*"₍₋₁₎", 
                color = pal[mod1(i, length(pal))],
                label = "")
        end

        for (i,a) in enumerate(algorithm)
            StatsPlots.scatter!([relevant_SS_dictionnary[a][indexin([state], 𝓂.var)][1]], [relevant_SS_dictionnary[a][indexin([k], 𝓂.var)][1]], 
            color = pal[mod1(i, length(pal))],
            label = "")
        end

        push!(pp, Pl)

        if !(plot_count % plots_per_page == 0)
            plot_count += 1
        else
            plot_count = 1

            ppp = StatsPlots.plot(pp...; attributes...)
            
            p = StatsPlots.plot(ppp,
                            legend_plot, 
                            layout = StatsPlots.grid(2, 1, heights = length(algorithm) > 3 ? [0.65, 0.35] : [0.8, 0.2]),
                            plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"; 
                            attributes_redux...
            )

            push!(return_plots,p)

            if show_plots
                display(p)
            end

            if save_plots
                StatsPlots.savefig(p, save_plots_path * "/solution__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
            end

            pane += 1
            pp = []
        end
    end

    if length(pp) > 0
        ppp = StatsPlots.plot(pp...; attributes...)
            
        p = StatsPlots.plot(ppp,
                        legend_plot, 
                        layout = StatsPlots.grid(2, 1, heights = length(algorithm) > 3 ? [0.65, 0.35] : [0.8, 0.2]),
                        plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"; 
                        attributes_redux...
        )

        push!(return_plots,p)

        if show_plots
            display(p)
        end

        if save_plots
            StatsPlots.savefig(p, save_plots_path * "/solution__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
        end
    end

    return return_plots
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
- $VARIABLES®
- `conditions_in_levels` [Default: `true`, Type: `Bool`]: indicator whether the conditions are provided in levels. If `true` the input to the conditions argument will have the non-stochastic steady state subtracted.
- $ALGORITHM®
- `label` [Default: `1`, Type: `Union{Real, String, Symbol}`]: label to attribute to this function call in the plots.
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMATH®
- $SAVE_PLOTS_PATH®
- $PLOTS_PER_PAGE®
- $PLOT_ATTRIBUTES®
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
                                    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}} = [0.0],
                                    periods::Int = 40, 
                                    parameters::ParameterType = nothing,
                                    variables::Union{Symbol_input,String_input} = :all_excluding_obc, 
                                    conditions_in_levels::Bool = true,
                                    algorithm::Symbol = :first_order,
                                    label::Union{Real, String, Symbol} = 1,
                                    show_plots::Bool = true,
                                    save_plots::Bool = false,
                                    save_plots_format::Symbol = :pdf,
                                    save_plots_path::String = ".",
                                    plots_per_page::Int = 9,
                                    plot_attributes::Dict = Dict(),
                                    verbose::Bool = false,
                                    tol::Tolerances = Tolerances(),
                                    quadratic_matrix_equation_algorithm::Symbol = :schur,
                                    sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = sum(1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > 1000 ? :bicgstab : :doubling,
                                    lyapunov_algorithm::Symbol = :doubling)
    # @nospecialize # reduce compile time
    
    gr_back = StatsPlots.backend() == StatsPlots.Plots.GRBackend()

    if !gr_back
        attrbts = merge(default_plot_attributes, Dict(:framestyle => :box))
    else
        attrbts = merge(default_plot_attributes, Dict())
    end

    attributes = merge(attrbts, plot_attributes)

    attributes_redux = copy(attributes)

    delete!(attributes_redux, :framestyle)

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
                                variables = variables, 
                                conditions_in_levels = conditions_in_levels,
                                algorithm = algorithm,
                                # levels = levels,
                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                sylvester_algorithm = sylvester_algorithm,
                                lyapunov_algorithm = lyapunov_algorithm,
                                tol = tol,
                                verbose = verbose)

    periods += max(size(conditions,2), isnothing(shocks) ? 1 : size(shocks,2))

    full_SS = vcat(sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)),map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.timings.exo))

    var_names = axiskeys(Y,1)   

    var_names = var_names isa Vector{String} ? var_names .|> replace_indices : var_names

    var_idx = indexin(var_names,full_SS)

    if length(intersect(𝓂.aux,var_names)) > 0
        for v in 𝓂.aux
            idx = indexin([v],var_names)
            if !isnothing(idx[1])
                var_names[idx[1]] = Symbol(replace(string(v), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
            end
        end
        # var_names[indexin(𝓂.aux,var_names)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    end
    
    relevant_SS = get_steady_state(𝓂, algorithm = algorithm, return_variables_only = true, derivatives = false,
                                    tol = tol,
                                    verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm = sylvester_algorithm)

    relevant_SS = relevant_SS isa KeyedArray ? axiskeys(relevant_SS,1) isa Vector{String} ? rekey(relevant_SS, 1 => axiskeys(relevant_SS,1) .|> Meta.parse .|> replace_indices) : relevant_SS : relevant_SS

    reference_steady_state = [s ∈ union(map(x -> Symbol(string(x) * "₍ₓ₎"), 𝓂.timings.exo), 𝓂.exo_present) ? 0.0 : relevant_SS(s) for s in var_names]

    var_length = length(full_SS) - 𝓂.timings.nExo

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
        @assert length(𝓂.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.exo)) * " shocks: " * repr(𝓂.exo)

        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        nzs = findnz(shocks)
        for i in 1:length(nzs[1])
            shocks_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
        end
        shocks = shocks_tmp
    elseif shocks isa Matrix{Union{Nothing,Float64}}
        @assert length(𝓂.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.exo)) * " shocks: " * repr(𝓂.exo)

        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        shocks_tmp[:,axes(shocks,2)] = shocks
        shocks = shocks_tmp
    elseif shocks isa KeyedArray{Union{Nothing,Float64}} || shocks isa KeyedArray{Float64}
        @assert length(setdiff(axiskeys(shocks,1),𝓂.exo)) == 0 "The following symbols in the first axis of the shocks matrix are not part of the model: " * repr(setdiff(axiskeys(shocks,1),𝓂.exo))
        
        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        shocks_tmp[indexin(sort(axiskeys(shocks,1)),𝓂.exo),axes(shocks,2)] .= shocks(sort(axiskeys(shocks,1)))
        shocks = shocks_tmp
    elseif isnothing(shocks)
        shocks = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
    end

    while length(conditional_forecast_active_plot_container) > 0
        pop!(conditional_forecast_active_plot_container)
    end
    
    args_and_kwargs = Dict(:run_id => length(conditional_forecast_active_plot_container) + 1,
                           :model_name => 𝓂.model_name,
                           :label => label,

                           :conditions => conditions[:,1:periods_input],
                           :conditions_in_levels => conditions_in_levels,
                           :shocks => shocks[:,1:periods_input],
                           :initial_state => initial_state_input,
                           :periods => periods_input,
                           :parameters => Dict(𝓂.parameters .=> 𝓂.parameter_values),
                           :variables => variables,
                           :algorithm => algorithm,

                           :NSSS_acceptance_tol => tol.NSSS_acceptance_tol,
                           :NSSS_xtol => tol.NSSS_xtol,
                           :NSSS_ftol => tol.NSSS_ftol,
                           :NSSS_rel_xtol => tol.NSSS_rel_xtol,
                           :qme_tol => tol.qme_tol,
                           :qme_acceptance_tol => tol.qme_acceptance_tol,
                           :sylvester_tol => tol.sylvester_tol,
                           :sylvester_acceptance_tol => tol.sylvester_acceptance_tol,
                           :lyapunov_tol => tol.lyapunov_tol,
                           :lyapunov_acceptance_tol => tol.lyapunov_acceptance_tol,
                           :droptol => tol.droptol,
                           :dependencies_tol => tol.dependencies_tol,

                           :quadratic_matrix_equation_algorithm => quadratic_matrix_equation_algorithm,
                           :sylvester_algorithm => sylvester_algorithm,
                           :lyapunov_algorithm => lyapunov_algorithm,

                           :plot_data => Y,
                           :reference_steady_state => reference_steady_state,
                           :variable_names => var_names[1:end - 𝓂.timings.nExo],
                           :shock_names => var_names[end - 𝓂.timings.nExo + 1:end]
                           )
    
    push!(conditional_forecast_active_plot_container, args_and_kwargs)

    orig_pal = StatsPlots.palette(attributes_redux[:palette])

    total_pal_len = 100

    alpha_reduction_factor = 0.7

    pal = mapreduce(x -> StatsPlots.coloralpha.(orig_pal, alpha_reduction_factor ^ x), vcat, 0:(total_pal_len ÷ length(orig_pal)) - 1) |> StatsPlots.palette

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
        SS = reference_steady_state[i]

        if !(all(isapprox.(Y[i,:],0,atol = eps(Float32)))) || length(findall(vcat(conditions,shocks)[v,:] .!= nothing)) > 0
         
            cond_idx = findall(vcat(conditions,shocks)[v,:] .!= nothing)
                
            p = standard_subplot(Y[i,:], SS, replace_indices_in_symbol(full_SS[v]), gr_back, pal = pal)
            
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
                    StatsPlots.savefig(p, save_plots_path * "/conditional_forecast__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
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
            StatsPlots.savefig(p, save_plots_path * "/conditional_forecast__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
        end
    end

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
- $VARIABLES®
- `conditions_in_levels` [Default: `true`, Type: `Bool`]: indicator whether the conditions are provided in levels. If `true` the input to the conditions argument will have the non-stochastic steady state subtracted.
- $ALGORITHM®
- $LABEL®
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMATH®
- $SAVE_PLOTS_PATH®
- $PLOTS_PER_PAGE®
- $PLOT_ATTRIBUTES®
- `plot_type` [Default: `:compare`, Type: `Symbol`]: plot type used to represent results. `:compare` means results are shown as separate lines. `:stack` means results are stacked.
- `transparency` [Default: `0.6`, Type: `Float64`]: transparency of stacked bars. Only relevant if `plot_type` is `:stack`.
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
                                    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}} = [0.0],
                                    periods::Int = 40, 
                                    parameters::ParameterType = nothing,
                                    variables::Union{Symbol_input,String_input} = :all_excluding_obc, 
                                    conditions_in_levels::Bool = true,
                                    algorithm::Symbol = :first_order,
                                    label::Union{Real, String, Symbol} = length(conditional_forecast_active_plot_container) + 1,
                                    show_plots::Bool = true,
                                    save_plots::Bool = false,
                                    save_plots_format::Symbol = :pdf,
                                    save_plots_path::String = ".",
                                    plots_per_page::Int = 6,
                                    plot_attributes::Dict = Dict(),
                                    plot_type::Symbol = :compare,
                                    transparency::Float64 = .6,
                                    verbose::Bool = false,
                                    tol::Tolerances = Tolerances(),
                                    quadratic_matrix_equation_algorithm::Symbol = :schur,
                                    sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = sum(1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > 1000 ? :bicgstab : :doubling,
                                    lyapunov_algorithm::Symbol = :doubling)
    # @nospecialize # reduce compile time
                 
    @assert plot_type ∈ [:compare, :stack] "plot_type must be either :compare or :stack"
                   
    gr_back = StatsPlots.backend() == StatsPlots.Plots.GRBackend()

    if !gr_back
        attrbts = merge(default_plot_attributes, Dict(:framestyle => :box))
    else
        attrbts = merge(default_plot_attributes, Dict())
    end

    attributes = merge(attrbts, plot_attributes)

    attributes_redux = copy(attributes)

    delete!(attributes_redux, :framestyle)

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
                                variables = variables, 
                                conditions_in_levels = conditions_in_levels,
                                algorithm = algorithm,
                                # levels = levels,
                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                sylvester_algorithm = sylvester_algorithm,
                                lyapunov_algorithm = lyapunov_algorithm,
                                tol = tol,
                                verbose = verbose)

    periods += max(size(conditions,2), isnothing(shocks) ? 1 : size(shocks,2))

    full_SS = vcat(sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)),map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.timings.exo))

    var_names = axiskeys(Y,1)   

    var_names = var_names isa Vector{String} ? var_names .|> replace_indices : var_names

    var_idx = indexin(var_names,full_SS)

    if length(intersect(𝓂.aux,var_names)) > 0
        for v in 𝓂.aux
            idx = indexin([v],var_names)
            if !isnothing(idx[1])
                var_names[idx[1]] = Symbol(replace(string(v), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
            end
        end
        # var_names[indexin(𝓂.aux,var_names)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    end
    
    relevant_SS = get_steady_state(𝓂, algorithm = algorithm, return_variables_only = true, derivatives = false,
                                    tol = tol,
                                    verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm = sylvester_algorithm)

    relevant_SS = relevant_SS isa KeyedArray ? axiskeys(relevant_SS,1) isa Vector{String} ? rekey(relevant_SS, 1 => axiskeys(relevant_SS,1) .|> Meta.parse .|> replace_indices) : relevant_SS : relevant_SS

    reference_steady_state = [s ∈ union(map(x -> Symbol(string(x) * "₍ₓ₎"), 𝓂.timings.exo), 𝓂.exo_present) ? 0.0 : relevant_SS(s) for s in var_names]

    var_length = length(full_SS) - 𝓂.timings.nExo

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
        @assert length(𝓂.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.exo)) * " shocks: " * repr(𝓂.exo)

        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        nzs = findnz(shocks)
        for i in 1:length(nzs[1])
            shocks_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
        end
        shocks = shocks_tmp
    elseif shocks isa Matrix{Union{Nothing,Float64}}
        @assert length(𝓂.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.exo)) * " shocks: " * repr(𝓂.exo)

        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        shocks_tmp[:,axes(shocks,2)] = shocks
        shocks = shocks_tmp
    elseif shocks isa KeyedArray{Union{Nothing,Float64}} || shocks isa KeyedArray{Float64}
        @assert length(setdiff(axiskeys(shocks,1),𝓂.exo)) == 0 "The following symbols in the first axis of the shocks matrix are not part of the model: " * repr(setdiff(axiskeys(shocks,1),𝓂.exo))
        
        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        shocks_tmp[indexin(sort(axiskeys(shocks,1)),𝓂.exo),axes(shocks,2)] .= shocks(sort(axiskeys(shocks,1)))
        shocks = shocks_tmp
    elseif isnothing(shocks)
        shocks = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
    end

    orig_pal = StatsPlots.palette(attributes_redux[:palette])

    total_pal_len = 100

    alpha_reduction_factor = 0.7

    pal = mapreduce(x -> StatsPlots.coloralpha.(orig_pal, alpha_reduction_factor ^ x), vcat, 0:(total_pal_len ÷ length(orig_pal)) - 1) |> StatsPlots.palette

    args_and_kwargs = Dict(:run_id => length(conditional_forecast_active_plot_container) + 1,
                           :model_name => 𝓂.model_name,
                           :label => label,

                           :conditions => conditions[:,1:periods_input],
                           :conditions_in_levels => conditions_in_levels,
                           :shocks => shocks[:,1:periods_input],
                           :initial_state => initial_state_input,
                           :periods => periods_input,
                           :parameters => Dict(𝓂.parameters .=> 𝓂.parameter_values),
                           :variables => variables,
                           :algorithm => algorithm,

                           :NSSS_acceptance_tol => tol.NSSS_acceptance_tol,
                           :NSSS_xtol => tol.NSSS_xtol,
                           :NSSS_ftol => tol.NSSS_ftol,
                           :NSSS_rel_xtol => tol.NSSS_rel_xtol,
                           :qme_tol => tol.qme_tol,
                           :qme_acceptance_tol => tol.qme_acceptance_tol,
                           :sylvester_tol => tol.sylvester_tol,
                           :sylvester_acceptance_tol => tol.sylvester_acceptance_tol,
                           :lyapunov_tol => tol.lyapunov_tol,
                           :lyapunov_acceptance_tol => tol.lyapunov_acceptance_tol,
                           :droptol => tol.droptol,
                           :dependencies_tol => tol.dependencies_tol,

                           :quadratic_matrix_equation_algorithm => quadratic_matrix_equation_algorithm,
                           :sylvester_algorithm => sylvester_algorithm,
                           :lyapunov_algorithm => lyapunov_algorithm,

                           :plot_data => Y,
                           :reference_steady_state => reference_steady_state,
                           :variable_names => var_names[1:end - 𝓂.timings.nExo],
                           :shock_names => var_names[end - 𝓂.timings.nExo + 1:end]
                           )
                           
    no_duplicate = all(
        !(all((
            get(dict, :parameters, nothing) == args_and_kwargs[:parameters],
            get(dict, :conditions, nothing) == args_and_kwargs[:conditions],
            get(dict, :shocks, nothing) == args_and_kwargs[:shocks],
            get(dict, :initial_state, nothing) == args_and_kwargs[:initial_state],
            all(get(dict, k, nothing) == get(args_and_kwargs, k, nothing) for k in keys(args_and_kwargs_names))
        )))
        for dict in conditional_forecast_active_plot_container
    ) # "New plot must be different from previous plot. Use the version without ! to plot."

    if no_duplicate 
        push!(conditional_forecast_active_plot_container, args_and_kwargs)
    else
        @info "Plot with same parameters already exists. Using previous plot data to create plot."
    end

    # 1. Keep only certain keys from each dictionary
    reduced_vector = [
        Dict(k => d[k] for k in vcat(:run_id, :label, keys(args_and_kwargs_names)...) if haskey(d, k))
        for d in conditional_forecast_active_plot_container
    ]

    diffdict = compare_args_and_kwargs(reduced_vector)

    # 2. Group the original vector by :model_name
    grouped_by_model = Dict{Any, Vector{Dict}}()

    for d in conditional_forecast_active_plot_container
        model = d[:model_name]
        d_sub = Dict(k => d[k] for k in setdiff(keys(args_and_kwargs), keys(args_and_kwargs_names)) if haskey(d, k))
        push!(get!(grouped_by_model, model, Vector{Dict}()), d_sub)
    end

    model_names = []

    for d in conditional_forecast_active_plot_container
        push!(model_names, d[:model_name])
    end

    model_names = unique(model_names)

    for model in model_names
        if length(grouped_by_model[model]) > 1
            diffdict_grouped = compare_args_and_kwargs(grouped_by_model[model])
            diffdict = merge_by_runid(diffdict, diffdict_grouped)
        end
    end
    
    annotate_ss = Vector{Pair{String, Any}}[]

    annotate_ss_page = Pair{String,Any}[]

    annotate_diff_input = Pair{String,Any}[]

    push!(annotate_diff_input, "Plot label" => reduce(vcat, diffdict[:label]))

    len_diff = length(conditional_forecast_active_plot_container)

    if haskey(diffdict, :parameters)
        param_nms = diffdict[:parameters] |> keys |> collect |> sort
        for param in param_nms
            result = [x === nothing ? "" : x for x in diffdict[:parameters][param]]
            push!(annotate_diff_input, String(param) => result)
        end
    end
    
    if haskey(diffdict, :shocks)
        shock_mats_no_nothing = []

        for shock_mat in diffdict[:shocks]
            lastcol = findlast(j -> any(!isnothing, shock_mat[:, j]), 1:size(shock_mat, 2))
            lastcol = isnothing(lastcol) ? 1 : lastcol
            push!(shock_mats_no_nothing, shock_mat[:, 1:lastcol])
        end

        # Collect unique shocks excluding `nothing`
        unique_shocks = unique(shock_mats_no_nothing)

        shocks_idx = Union{Int,Nothing}[]

        for init in shock_mats_no_nothing
            if isnothing(init) || all(isnothing, init)
                push!(shocks_idx, nothing)
            else
                for (i,u) in enumerate(unique_shocks)
                    if u == init
                        push!(shocks_idx,i)
                        break
                    end
                end
            end
        end

        if length(unique_shocks) > 1
            push!(annotate_diff_input, "Shocks" => [isnothing(i) ? nothing : "#$i" for i in shocks_idx])
        end
    end
    
    if haskey(diffdict, :conditions)
        condition_mats_no_nothing = []

        for condition_mat in diffdict[:conditions]
            lastcol = findlast(j -> any(!isnothing, condition_mat[:, j]), 1:size(condition_mat, 2))
            lastcol = isnothing(lastcol) ? 1 : lastcol
            push!(condition_mats_no_nothing, condition_mat[:, 1:lastcol])
        end

        # Collect unique conditions excluding `nothing`
        unique_conditions = unique(condition_mats_no_nothing)

        conditions_idx = Union{Int,Nothing}[]

        for init in condition_mats_no_nothing
            if isnothing(init) || all(isnothing, init)
                push!(conditions_idx, nothing)
            else
                for (i,u) in enumerate(unique_conditions)
                    if u == init
                        push!(conditions_idx,i)
                        break
                    end
                end
            end
        end

        if length(unique_conditions) > 1
            push!(annotate_diff_input, "Conditions" => [isnothing(i) ? nothing : "#$i" for i in conditions_idx])
        end
    end

    if haskey(diffdict, :initial_state)
        unique_initial_state = unique(diffdict[:initial_state])

        initial_state_idx = Int[]

        for init in diffdict[:initial_state]
            for (i,u) in enumerate(unique_initial_state)
                if u == init
                    push!(initial_state_idx,i)
                    continue
                end
            end
        end

        push!(annotate_diff_input, "Initial state" => ["#$i" for i in initial_state_idx])
    end
    
    same_shock_direction = true
    
    for k in setdiff(keys(args_and_kwargs), 
                        [
                            :run_id, :parameters, :plot_data, :tol, :reference_steady_state, :initial_state, :conditions, :conditions_in_levels, :label,
                            :shocks, :shock_names,
                            :variables, :variable_names,
                            # :periods, :quadratic_matrix_equation_algorithm, :sylvester_algorithm, :lyapunov_algorithm,
                        ]
                    )

        if haskey(diffdict, k)
            push!(annotate_diff_input, args_and_kwargs_names[k] => reduce(vcat,diffdict[k]))
            
            if k == :negative_shock
                same_shock_direction = false
            end
        end
    end

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
    for (i,k) in enumerate(conditional_forecast_active_plot_container)
        if plot_type == :stack
            StatsPlots.bar!(legend_plot,
                            [NaN], 
                            legend_title = length(annotate_diff_input) > 2 ? nothing : annotate_diff_input[2][1],
                            linecolor = :transparent,
                            alpha = transparency,
                            linewidth = 0,
                            label = length(annotate_diff_input) > 2 ? k[:label] isa Symbol ? string(k[:label]) : k[:label] : annotate_diff_input[2][2][i] isa String ? annotate_diff_input[2][2][i] : String(Symbol(annotate_diff_input[2][2][i])))
        elseif plot_type == :compare
            StatsPlots.plot!(legend_plot,
                            [NaN], 
                            legend_title = length(annotate_diff_input) > 2 ? nothing : annotate_diff_input[2][1],
                            color = pal[mod1(i, length(pal))],
                            label = length(annotate_diff_input) > 2 ? k[:label] isa Symbol ? string(k[:label]) : k[:label] : annotate_diff_input[2][2][i] isa String ? annotate_diff_input[2][2][i] : String(Symbol(annotate_diff_input[2][2][i])))
        end

        foreach(n -> push!(joint_variables, String(n)), k[:variable_names] isa AbstractVector ? k[:variable_names] : (k[:variable_names],))
        foreach(n -> push!(joint_shocks, String(n)), k[:shock_names] isa AbstractVector ? k[:shock_names] : (k[:shock_names],))
        
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

    sort!(joint_variables)
    sort!(joint_shocks)

    n_subplots = length(joint_variables) + length(joint_shocks)
    pp = []
    pane = 1
    plot_count = 1

    joint_non_zero_variables = []

    return_plots = []
    
    for var in vcat(collect(joint_variables), collect(joint_shocks))
        not_zero_in_any_cond_fcst = false

        for k in conditional_forecast_active_plot_container
            var_idx = findfirst(==(var), String.(vcat(k[:variable_names], k[:shock_names])))
            if isnothing(var_idx)
                # If the variable or shock is not present in the current conditional_forecast_active_plot_container,
                # we skip this iteration.
                continue
            else
                if any(.!isapprox.(k[:plot_data][var_idx,:], 0, atol = eps(Float32))) || any(!=(nothing), vcat(k[:conditions], k[:shocks])[var_idx, :])
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

        for k in conditional_forecast_active_plot_container
            var_idx = findfirst(==(var), String.(vcat(k[:variable_names], k[:shock_names])))
            if isnothing(var_idx)
                # If the variable is not present in the current conditional_forecast_active_plot_container,
                # we skip this iteration.
                push!(SSs, NaN)
                push!(Ys, zeros(max_periods))
            else
                dat = fill(NaN, max_periods)
                dat[1:length(k[:plot_data][var_idx,:])] .= k[:plot_data][var_idx,:]
                push!(SSs, k[:reference_steady_state][var_idx])
                push!(Ys, dat) # k[:plot_data][var_idx,:])
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
                                        var, 
                                        gr_back,
                                        same_ss,
                                        pal = pal,
                                        transparency = transparency)

        if plot_type == :compare
            for (i,k) in enumerate(conditional_forecast_active_plot_container)   
                var_idx = findfirst(==(var), String.(vcat(k[:variable_names], k[:shock_names])))
                
                if isnothing(var_idx) continue end

                cond_idx = findall(vcat(k[:conditions], k[:shocks])[var_idx,:] .!= nothing)
                
                if length(cond_idx) > 0
                    SS = k[:reference_steady_state][var_idx]

                    vals = vcat(k[:conditions], k[:shocks])[var_idx, cond_idx]

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

            shock_string = "Conditional forecast"

            if haskey(diffdict, :model_name)
                model_string = "multiple models"
                model_string_filename = "multiple_models"
            else
                model_string = 𝓂.model_name
                model_string_filename = 𝓂.model_name
            end
            
            plot_title = "Model: "*model_string*"        " *  shock_string *"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"

            ppp = StatsPlots.plot(pp...; attributes...)

            plot_elements = [ppp, legend_plot]

            layout_heights = [15,1]
            
            if length(annotate_diff_input) > 2
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

            p = StatsPlots.plot(plot_elements...,
                                layout = StatsPlots.grid(length(layout_heights), 1, heights = layout_heights ./ sum(layout_heights)),
                                plot_title = plot_title; 
                                attributes_redux...)

            push!(return_plots,p)

            if show_plots# & (length(pp) > 0)
                display(p)
            end

            if save_plots# & (length(pp) > 0)
                StatsPlots.savefig(p, save_plots_path * "/conditional_forecast__" * model_string_filename * "__" * string(pane) * "." * string(save_plots_format))
            end

            pane += 1

            annotate_ss_page = Pair{String,Any}[]

            pp = []
        end
    end

    if length(pp) > 0

        shock_string = "Conditional forecast"

        if haskey(diffdict, :model_name)
            model_string = "multiple models"
            model_string_filename = "multiple_models"
        else
            model_string = 𝓂.model_name
            model_string_filename = 𝓂.model_name
        end
        
        plot_title = "Model: "*model_string*"        " *  shock_string *"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"

        ppp = StatsPlots.plot(pp...; attributes...)

        plot_elements = [ppp, legend_plot]

        layout_heights = [15,1]
        
        if length(annotate_diff_input) > 2
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

        p = StatsPlots.plot(plot_elements...,
                            layout = StatsPlots.grid(length(layout_heights), 1, heights = layout_heights ./ sum(layout_heights)),
                            plot_title = plot_title; 
                            attributes_redux...)
                                
        push!(return_plots,p)

        if show_plots# & (length(pp) > 0)
            display(p)
        end

        if save_plots# & (length(pp) > 0)
            StatsPlots.savefig(p, save_plots_path * "/conditional_forecast__" * model_string_filename * "__" * string(pane) * "." * string(save_plots_format))
        end
    end

    return return_plots
end


end # dispatch_doctor

end # module