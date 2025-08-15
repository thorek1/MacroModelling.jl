module StatsPlotsExt

using MacroModelling
import MacroModelling: ParameterType, ℳ, Symbol_input, String_input, Tolerances, merge_calculation_options, MODEL®, DATA®, PARAMETERS®, ALGORITHM®, FILTER®, VARIABLES®, SMOOTH®, SHOW_PLOTS®, SAVE_PLOTS®, SAVE_PLOTS_FORMATH®, SAVE_PLOTS_PATH®, PLOTS_PER_PAGE®, MAX_ELEMENTS_PER_LEGENDS_ROW®, EXTRA_LEGEND_SPACE®, PLOT_ATTRIBUTES®, QME®, SYLVESTER®, LYAPUNOV®, TOLERANCES®, VERBOSE®, DATA_IN_LEVELS®, PERIODS®, SHOCKS®, SHOCK_SIZE®, NEGATIVE_SHOCK®, GENERALISED_IRF®, INITIAL_STATE®, IGNORE_OBC®, CONDITIONS®, SHOCK_CONDITIONS®, LEVELS®, parse_shocks_input_to_index, parse_variables_input_to_index, replace_indices, filter_data_with_model, get_relevant_steady_states, replace_indices_in_symbol, parse_algorithm_to_state_update, girf, decompose_name, obc_objective_optim_fun, obc_constraint_optim_fun
import DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF, TYPEDSIGNATURES, TYPEDFIELDS
import LaTeXStrings

const irf_active_plot_container = Dict[]
const model_estimates_active_plot_container = Dict[]

import StatsPlots
import DataStructures: OrderedSet
import SparseArrays: SparseMatrixCSC
import NLopt
using DispatchDoctor

import MacroModelling: plot_irfs, plot_irf, plot_irf!, plot_IRF, plot_simulations, plot_simulation, plot_solution, plot_girf, plot_conditional_forecast, plot_conditional_variance_decomposition, plot_forecast_error_variance_decomposition, plot_fevd, plot_model_estimates, plot_shock_decomposition, plotlyjs_backend, gr_backend, compare_args_and_kwargs

const default_plot_attributes = Dict(:size=>(700,500),
                                :plot_titlefont => 10, 
                                :titlefont => 10, 
                                :guidefont => 8,
                                :legendfontsize => 8,
                                :legend_title_font_pointsize => 8,
                                :tickfontsize => 8,
                                :framestyle => :semi)


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
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMATH®
- $SAVE_PLOTS_PATH®
- $PLOTS_PER_PAGE®
- `transparency` [Default: `0.6`, Type: `Float64`]: transparency of bars
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
                                show_plots::Bool = true,
                                save_plots::Bool = false,
                                save_plots_format::Symbol = :pdf,
                                save_plots_path::String = ".",
                                plots_per_page::Int = 9,
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

    args_and_kwargs = Dict(:model_name => 𝓂.model_name,
                           :data => data,
                           :parameters => parameters,
                           :algorithm => algorithm,
                           :filter => filter,
                           :warmup_iterations => warmup_iterations,
                           :variables => variables,
                           :shocks => shocks,
                           :presample_periods => presample_periods,
                           :data_in_levels => data_in_levels,
                           :shock_decomposition => shock_decomposition,
                           :smooth => smooth,
                           :tol => tol,
                           :quadratic_matrix_equation_algorithm => quadratic_matrix_equation_algorithm,
                           :sylvester_algorithm => sylvester_algorithm,
                           :lyapunov_algorithm => lyapunov_algorithm)

    push!(model_estimates_active_plot_container, args_and_kwargs)

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

    @assert !(algorithm ∈ [:second_order, :third_order] && shock_decomposition) "Decomposition  implemented for first order, pruned second and third order. Second and third order solution decomposition is not yet implemented."
    
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
    shock_idx   = parse_shocks_input_to_index(shocks,𝓂.timings)

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

    date_axis = axiskeys(data,2)

    extra_legend_space += length(string(date_axis[1])) > 6 ? .1 : 0.0

    @assert presample_periods < size(data,2) "The number of presample periods must be less than the number of periods in the data."

    periods = presample_periods+1:size(data,2)

    date_axis = date_axis[periods]

    variables_to_plot, shocks_to_plot, standard_deviations, decomposition = filter_data_with_model(𝓂, data_in_deviations, Val(algorithm), Val(filter), warmup_iterations = warmup_iterations, smooth = smooth, opts = opts)
    
    if pruning
        decomposition[:,1:(end - 2 - pruning),:]    .+= SSS_delta
        decomposition[:,end - 2,:]                  .-= SSS_delta * (size(decomposition,2) - 4)
        variables_to_plot                           .+= SSS_delta
        data_in_deviations                          .+= SSS_delta[obs_idx]
    end

    return_plots = []

    estimate_color = :navy

    data_color = :orangered

    n_subplots = length(var_idx) + length(shock_idx)
    pp = []
    pane = 1
    plot_count = 1

    for i in 1:length(var_idx) + length(shock_idx)
        if i > length(var_idx) # Shock decomposition
            push!(pp,begin
                    StatsPlots.plot()
                    StatsPlots.plot!(#date_axis, 
                        shocks_to_plot[shock_idx[i - length(var_idx)],periods],
                        title = replace_indices_in_symbol(𝓂.timings.exo[shock_idx[i - length(var_idx)]]) * "₍ₓ₎", 
                        ylabel = shock_decomposition ? "Absolute Δ" : "Level",label = "", 
                        xformatter = x -> string(date_axis[max(1,min(ceil(Int,x),length(date_axis)))]),
                        xrotation = length(string(date_axis[1])) > 6 ? 30 : 0,
                        color = shock_decomposition ? estimate_color : :auto)
                    StatsPlots.hline!([0],
                        color = :black,
                        label = "")                               
            end)
        else
            SS = reference_steady_state[var_idx[i]]

            if shock_decomposition SS = zero(SS) end

            can_dual_axis = gr_back &&  all((variables_to_plot[var_idx[i],:] .+ SS) .> eps(Float32)) && (SS > eps(Float32)) && !shock_decomposition

            push!(pp,begin
                    StatsPlots.plot()

                    if shock_decomposition
                        additional_indices = pruning ? [size(decomposition,2)-1, size(decomposition,2)-2] : [size(decomposition,2)-1]

                        StatsPlots.groupedbar!(#date_axis,
                            decomposition[var_idx[i],[additional_indices..., shock_idx...],periods]', 
                            bar_position = :stack, 
                            xformatter = x -> string(date_axis[max(1,min(ceil(Int,x),length(date_axis)))]),
                            xrotation = length(string(date_axis[1])) > 6 ? 30 : 0,
                            lc = :transparent,  # Line color set to transparent
                            lw = 0,  # This removes the lines around the bars
                            legend = :none, 
                            # yformatter = y -> round(y + SS, digits = 1), # rm Absolute Δ in this case and fix SS additions
                            # xformatter = x -> string(date_axis[Int(x)]),
                            alpha = transparency)
                    end

                    StatsPlots.plot!(#date_axis,
                        variables_to_plot[var_idx[i],periods] .+ SS,
                        title = replace_indices_in_symbol(𝓂.timings.var[var_idx[i]]), 
                        ylabel = shock_decomposition ? "Absolute Δ" : "Level", 
                        xformatter = x -> string(date_axis[max(1,min(ceil(Int,x),length(date_axis)))]),
                        xrotation = length(string(date_axis[1])) > 6 ? 30 : 0,
                        label = "", 
                        # xformatter = x -> string(date_axis[Int(x)]),
                        color = shock_decomposition ? estimate_color : :auto)

                    if var_idx[i] ∈ obs_idx 
                        StatsPlots.plot!(#date_axis,
                            data_in_deviations[indexin([var_idx[i]],obs_idx),periods]' .+ SS,
                            title = replace_indices_in_symbol(𝓂.timings.var[var_idx[i]]),
                            ylabel = shock_decomposition ? "Absolute Δ" : "Level", 
                            label = "", 
                            xformatter = x -> string(date_axis[max(1,min(ceil(Int,x),length(date_axis)))]),
                            xrotation = length(string(date_axis[1])) > 6 ? 30 : 0,
                            # xformatter = x -> string(date_axis[Int(x)]),
                            color = shock_decomposition ? data_color : :auto) 
                    end

                    if can_dual_axis 
                        StatsPlots.plot!(StatsPlots.twinx(),
                            # date_axis, 
                            100*((variables_to_plot[var_idx[i],periods] .+ SS) ./ SS .- 1), 
                            ylabel = LaTeXStrings.L"\% \Delta", 
                            xformatter = x -> string(date_axis[max(1,min(ceil(Int,x),length(date_axis)))]),
                            xrotation = length(string(date_axis[1])) > 6 ? 30 : 0,
                            label = "") 

                        if var_idx[i] ∈ obs_idx 
                            StatsPlots.plot!(StatsPlots.twinx(),
                                # date_axis, 
                                100*((data_in_deviations[indexin([var_idx[i]],obs_idx),periods]' .+ SS) ./ SS .- 1), 
                                ylabel = LaTeXStrings.L"\% \Delta", 
                                xformatter = x -> string(date_axis[max(1,min(ceil(Int,x),length(date_axis)))]),
                                xrotation = length(string(date_axis[1])) > 6 ? 30 : 0,
                                label = "") 
                        end
                    end
                    
                    StatsPlots.hline!(can_dual_axis ? [SS 0] : [SS],
                        color = :black,
                        label = "")                               
            end)
        end

        if !(plot_count % plots_per_page == 0)
            plot_count += 1
        else
            plot_count = 1

            ppp = StatsPlots.plot(pp...; attributes...)

            # Legend
            p = StatsPlots.plot(ppp,begin
                                        StatsPlots.plot(framestyle = :none)
                                        if shock_decomposition
                                            additional_labels = pruning ? ["Initial value", "Nonlinearities"] : ["Initial value"]

                                            StatsPlots.bar!(fill(0, 1, length(shock_idx) + 1 + pruning), 
                                                                    label = reshape(vcat(additional_labels, string.(replace_indices_in_symbol.(𝓂.exo[shock_idx]))), 1, length(shock_idx) + 1 + pruning), 
                                                                    linewidth = 0,
                                                                    alpha = transparency,
                                                                    lw = 0,
                                                                    legend = :inside, 
                                                                    legend_columns = legend_columns)
                                        end
                                        StatsPlots.plot!(fill(0,1,1), 
                                        label = "Estimate", 
                                        color = shock_decomposition ? estimate_color : :auto,
                                        legend = :inside)
                                        StatsPlots.plot!(fill(0,1,1), 
                                        label = "Data", 
                                        color = shock_decomposition ? data_color : :auto,
                                        legend = :inside)
                                    end, 
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

        p = StatsPlots.plot(ppp,begin
                                    StatsPlots.plot(framestyle = :none)
                                    if shock_decomposition
                                        additional_labels = pruning ? ["Initial value", "Nonlinearities"] : ["Initial value"]
                                        
                                        StatsPlots.bar!(fill(0,1,length(shock_idx) + 1 + pruning), 
                                                                label = reshape(vcat(additional_labels..., string.(replace_indices_in_symbol.(𝓂.exo[shock_idx]))),1,length(shock_idx) + 1 + pruning), 
                                                                linewidth = 0,
                                                                alpha = transparency,
                                                                lw = 0,
                                                                legend = :inside, 
                                                                legend_columns = legend_columns)
                                    end
                                    StatsPlots.plot!(fill(0,1,1), 
                                    label = "Estimate", 
                                    color = shock_decomposition ? :black : :auto,
                                    legend = :inside)
                                    StatsPlots.plot!(fill(0,1,1), 
                                    label = "Data", 
                                    color = shock_decomposition ? :darkred : :auto,
                                    legend = :inside)
                                end, 
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
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMATH®
- $SAVE_PLOTS_PATH®
- $PLOTS_PER_PAGE®
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

    if generalised_irf
        Y = girf(state_update, 
                    initial_state, 
                    zeros(𝓂.timings.nVars), 
                    𝓂.timings; 
                    periods = periods, 
                    shocks = shocks, 
                    shock_size = shock_size,
                    variables = variables, 
                    negative_shock = negative_shock)#, warmup_periods::Int = 100, draws::Int = 50, iterations_to_steady_state::Int = 500)
    else
        if occasionally_binding_constraints
            function obc_state_update(present_states, present_shocks::Vector{R}, state_update::Function) where R <: Float64
                unconditional_forecast_horizon = 𝓂.max_obc_horizon

                reference_ss = 𝓂.solution.non_stochastic_steady_state

                obc_shock_idx = contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")

                periods_per_shock = 𝓂.max_obc_horizon + 1
                
                num_shocks = sum(obc_shock_idx) ÷ periods_per_shock
                
                p = (present_states, state_update, reference_ss, 𝓂, algorithm, unconditional_forecast_horizon, present_shocks)

                constraints_violated = any(𝓂.obc_violation_function(zeros(num_shocks*periods_per_shock), p) .> eps(Float32))

                if constraints_violated
                    opt = NLopt.Opt(NLopt.:LD_SLSQP, num_shocks*periods_per_shock)
                    # check whether auglag is more reliable and efficient here
                    opt.min_objective = obc_objective_optim_fun

                    opt.xtol_abs = eps(Float32)
                    opt.ftol_abs = eps(Float32)
                    opt.maxeval = 500
                    
                    # Adding constraints
                    # opt.upper_bounds = fill(eps(), num_shocks*periods_per_shock) 
                    # upper bounds don't work because it can be that bounds can only be enforced with offsetting (previous periods negative shocks) positive shocks. also in order to enforce the bound over the length of the forecasting horizon the shocks might be in the last period. that's why an approach whereby you increase the anticipation horizon of shocks can be more costly due to repeated computations.
                    # opt.lower_bounds = fill(-eps(), num_shocks*periods_per_shock)

                    upper_bounds = fill(eps(), 1 + 2*(max(num_shocks*periods_per_shock-1, 1)))
                    
                    NLopt.inequality_constraint!(opt, (res, x, jac) -> obc_constraint_optim_fun(res, x, jac, p), upper_bounds)

                    (minf,x,ret) = NLopt.optimize(opt, zeros(num_shocks*periods_per_shock))
                    
                    # solved = ret ∈ Symbol.([
                    #     NLopt.SUCCESS,
                    #     NLopt.STOPVAL_REACHED,
                    #     NLopt.FTOL_REACHED,
                    #     NLopt.XTOL_REACHED,
                    #     NLopt.ROUNDOFF_LIMITED,
                    # ])
                    
                    present_shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")] .= x

                    constraints_violated = any(𝓂.obc_violation_function(x, p) .> eps(Float32))

                    solved = !constraints_violated
                else
                    solved = true
                end
                # if constraints_violated
                #     obc_shock_timing = convert_superscript_to_integer.(string.(𝓂.timings.exo[obc_shock_idx]))
                
                #     for anticipated_shock_horizon in 1:periods_per_shock
                #         anticipated_shock_subset = obc_shock_timing .< anticipated_shock_horizon
                    
                #         function obc_violation_function_wrapper(x::Vector{T}) where T
                #             y = zeros(T, length(anticipated_shock_subset))
                        
                #             y[anticipated_shock_subset] = x
                        
                #             return 𝓂.obc_violation_function(y, p)
                #         end
                        
                #         opt = NLopt.Opt(NLopt.:LD_SLSQP, num_shocks * anticipated_shock_horizon)
                        
                #         opt.min_objective = obc_objective_optim_fun

                #         opt.xtol_rel = eps()
                        
                #         # Adding constraints
                #         # opt.upper_bounds = fill(eps(), num_shocks*periods_per_shock)
                #         # opt.lower_bounds = fill(-eps(), num_shocks*periods_per_shock)

                #         upper_bounds = fill(eps(), 1 + 2*(num_shocks*periods_per_shock-1))
                        
                #         NLopt.inequality_constraint!(opt, (res, x, jac) -> obc_constraint_optim_fun(res, x, jac, obc_violation_function_wrapper), upper_bounds)

                #         (minf,x,ret) = NLopt.optimize(opt, zeros(num_shocks * anticipated_shock_horizon))
                        
                #         solved = ret ∈ Symbol.([
                #             NLopt.SUCCESS,
                #             NLopt.STOPVAL_REACHED,
                #             NLopt.FTOL_REACHED,
                #             NLopt.XTOL_REACHED,
                #             NLopt.ROUNDOFF_LIMITED,
                #         ])
                        
                #         present_shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")][anticipated_shock_subset] .= x

                #         constraints_violated = any(𝓂.obc_violation_function(present_shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")], p) .> eps(Float32))
                        
                #         solved = solved && !constraints_violated

                #         if solved break end
                #     end

                #     solved = !any(𝓂.obc_violation_function(present_shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")], p) .> eps(Float32))
                # else
                #     solved = true
                # end

                present_states = state_update(present_states, present_shocks)

                return present_states, present_shocks, solved
            end

            Y =  irf(state_update,
                    obc_state_update,
                    initial_state, 
                    zeros(𝓂.timings.nVars),
                    𝓂.timings;
                    periods = periods, 
                    shocks = shocks, 
                    shock_size = shock_size,
                    variables = variables, 
                    negative_shock = negative_shock) .+ SSS_delta[var_idx]
        else
            Y = irf(state_update, 
                    initial_state, 
                    zeros(𝓂.timings.nVars),
                    𝓂.timings;
                    periods = periods, 
                    shocks = shocks, 
                    shock_size = shock_size,
                    variables = variables, 
                    negative_shock = negative_shock) .+ SSS_delta[var_idx]
        end
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

    args_and_kwargs = Dict(:model_name => 𝓂.model_name,
                           :periods => periods,
                           :shocks => shocks,
                           :variables => variables,
                           :parameters => Dict(𝓂.parameters .=> 𝓂.parameter_values),
                           :algorithm => algorithm,
                           :shock_size => shock_size,
                           :negative_shock => negative_shock,
                           :generalised_irf => generalised_irf,
                           :initial_state => initial_state,
                           :ignore_obc => ignore_obc,
                           :tol => tol,
                           :quadratic_matrix_equation_algorithm => quadratic_matrix_equation_algorithm,
                           :sylvester_algorithm => sylvester_algorithm,
                           :lyapunov_algorithm => lyapunov_algorithm,
                           :plot_data => Y,
                           :reference_steady_state => reference_steady_state[var_idx],
                           :variable_names => variable_names,
                           :shock_names => shock_names,
                           :shock_idx => shock_idx,
                           :var_idx => var_idx)

    while length(irf_active_plot_container) > 0
        pop!(irf_active_plot_container)
    end
    
    push!(irf_active_plot_container, args_and_kwargs)

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

        for i in 1:length(var_idx)
            SS = reference_steady_state[var_idx[i]]

            can_dual_axis = gr_back && all((Y[i,:,shock] .+ SS) .> eps(Float32)) && (SS > eps(Float32))

            if !(all(isapprox.(Y[i,:,shock],0,atol = eps(Float32))))
                variable_name = replace_indices_in_symbol(𝓂.timings.var[var_idx[i]])

                push!(pp, plot_irf_subplot(Y[i,:,shock], SS, variable_name, can_dual_axis))

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


function plot_irf_subplot(irf_data::AbstractVector{S}, steady_state::S, variable_name::String, can_dual_axis::Bool) where S <: AbstractFloat
    p = StatsPlots.plot(irf_data .+ steady_state,
                        title = variable_name,
                        ylabel = "Level",
                        label = "")

    if can_dual_axis
        StatsPlots.plot!(StatsPlots.twinx(), 
                         100*((irf_data .+ steady_state) ./ steady_state .- 1), 
                         ylabel = LaTeXStrings.L"\% \Delta", 
                         label = "") 
    end

    StatsPlots.hline!(can_dual_axis ? [steady_state 0] : [steady_state], 
                      color = :black, 
                      label = "")                               
    return p
end

function plot_irf_subplot(irf_data::Vector{<:AbstractVector{S}}, steady_state::Vector{S}, variable_name::String, can_dual_axis::Bool, same_ss::Bool; pal::StatsPlots.ColorPalette = StatsPlots.palette(:auto)) where S <: AbstractFloat
    plot_dat = []
    plot_dat_dual = []
    
    pal_val = Int[]

    stst = 1.0

    for (i,(y, ss)) in enumerate(zip(irf_data, steady_state))
        if !isnan(ss)
            stst = ss
            if can_dual_axis && same_ss
                push!(plot_dat, y .+ ss)
                push!(plot_dat_dual, 100 * ((y .+ ss) ./ ss .- 1))
            else
                push!(plot_dat, y)
            end
            push!(pal_val, i)
        end
    end

    p = StatsPlots.plot(plot_dat,
                        title = variable_name,
                        ylabel = same_ss ? "Level" : "abs. " * LaTeXStrings.L"\Delta",
                        color = pal[pal_val]',
                        label = "")

    if can_dual_axis && same_ss
        StatsPlots.plot!(StatsPlots.twinx(), 
                         plot_dat_dual, 
                         ylabel = LaTeXStrings.L"\% \Delta", 
                         color = pal[pal_val]',
                         label = "") 
    end
    StatsPlots.hline!(can_dual_axis && same_ss ? [stst 0] : [same_ss ? stst : 0], 
                      color = :black, 
                      label = "")
                      
    return p
end


function plot_irf!(𝓂::ℳ;
                    periods::Int = 40, 
                    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all_excluding_obc, 
                    variables::Union{Symbol_input,String_input} = :all_excluding_auxiliary_and_obc,
                    parameters::ParameterType = nothing,
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

    if generalised_irf
        Y = girf(state_update, 
                    initial_state, 
                    zeros(𝓂.timings.nVars), 
                    𝓂.timings; 
                    periods = periods, 
                    shocks = shocks, 
                    shock_size = shock_size,
                    variables = variables, 
                    negative_shock = negative_shock)#, warmup_periods::Int = 100, draws::Int = 50, iterations_to_steady_state::Int = 500)
    else
        if occasionally_binding_constraints
            function obc_state_update(present_states, present_shocks::Vector{R}, state_update::Function) where R <: Float64
                unconditional_forecast_horizon = 𝓂.max_obc_horizon

                reference_ss = 𝓂.solution.non_stochastic_steady_state

                obc_shock_idx = contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")

                periods_per_shock = 𝓂.max_obc_horizon + 1
                
                num_shocks = sum(obc_shock_idx) ÷ periods_per_shock
                
                p = (present_states, state_update, reference_ss, 𝓂, algorithm, unconditional_forecast_horizon, present_shocks)

                constraints_violated = any(𝓂.obc_violation_function(zeros(num_shocks*periods_per_shock), p) .> eps(Float32))

                if constraints_violated
                    opt = NLopt.Opt(NLopt.:LD_SLSQP, num_shocks*periods_per_shock)
                    # check whether auglag is more reliable and efficient here
                    opt.min_objective = obc_objective_optim_fun

                    opt.xtol_abs = eps(Float32)
                    opt.ftol_abs = eps(Float32)
                    opt.maxeval = 500
                    
                    # Adding constraints
                    # opt.upper_bounds = fill(eps(), num_shocks*periods_per_shock) 
                    # upper bounds don't work because it can be that bounds can only be enforced with offsetting (previous periods negative shocks) positive shocks. also in order to enforce the bound over the length of the forecasting horizon the shocks might be in the last period. that's why an approach whereby you increase the anticipation horizon of shocks can be more costly due to repeated computations.
                    # opt.lower_bounds = fill(-eps(), num_shocks*periods_per_shock)

                    upper_bounds = fill(eps(), 1 + 2*(max(num_shocks*periods_per_shock-1, 1)))
                    
                    NLopt.inequality_constraint!(opt, (res, x, jac) -> obc_constraint_optim_fun(res, x, jac, p), upper_bounds)

                    (minf,x,ret) = NLopt.optimize(opt, zeros(num_shocks*periods_per_shock))
                    
                    # solved = ret ∈ Symbol.([
                    #     NLopt.SUCCESS,
                    #     NLopt.STOPVAL_REACHED,
                    #     NLopt.FTOL_REACHED,
                    #     NLopt.XTOL_REACHED,
                    #     NLopt.ROUNDOFF_LIMITED,
                    # ])
                    
                    present_shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")] .= x

                    constraints_violated = any(𝓂.obc_violation_function(x, p) .> eps(Float32))

                    solved = !constraints_violated
                else
                    solved = true
                end
                # if constraints_violated
                #     obc_shock_timing = convert_superscript_to_integer.(string.(𝓂.timings.exo[obc_shock_idx]))
                
                #     for anticipated_shock_horizon in 1:periods_per_shock
                #         anticipated_shock_subset = obc_shock_timing .< anticipated_shock_horizon
                    
                #         function obc_violation_function_wrapper(x::Vector{T}) where T
                #             y = zeros(T, length(anticipated_shock_subset))
                        
                #             y[anticipated_shock_subset] = x
                        
                #             return 𝓂.obc_violation_function(y, p)
                #         end
                        
                #         opt = NLopt.Opt(NLopt.:LD_SLSQP, num_shocks * anticipated_shock_horizon)
                        
                #         opt.min_objective = obc_objective_optim_fun

                #         opt.xtol_rel = eps()
                        
                #         # Adding constraints
                #         # opt.upper_bounds = fill(eps(), num_shocks*periods_per_shock)
                #         # opt.lower_bounds = fill(-eps(), num_shocks*periods_per_shock)

                #         upper_bounds = fill(eps(), 1 + 2*(num_shocks*periods_per_shock-1))
                        
                #         NLopt.inequality_constraint!(opt, (res, x, jac) -> obc_constraint_optim_fun(res, x, jac, obc_violation_function_wrapper), upper_bounds)

                #         (minf,x,ret) = NLopt.optimize(opt, zeros(num_shocks * anticipated_shock_horizon))
                        
                #         solved = ret ∈ Symbol.([
                #             NLopt.SUCCESS,
                #             NLopt.STOPVAL_REACHED,
                #             NLopt.FTOL_REACHED,
                #             NLopt.XTOL_REACHED,
                #             NLopt.ROUNDOFF_LIMITED,
                #         ])
                        
                #         present_shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")][anticipated_shock_subset] .= x

                #         constraints_violated = any(𝓂.obc_violation_function(present_shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")], p) .> eps(Float32))
                        
                #         solved = solved && !constraints_violated

                #         if solved break end
                #     end

                #     solved = !any(𝓂.obc_violation_function(present_shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")], p) .> eps(Float32))
                # else
                #     solved = true
                # end

                present_states = state_update(present_states, present_shocks)

                return present_states, present_shocks, solved
            end

            Y =  irf(state_update,
                    obc_state_update,
                    initial_state, 
                    zeros(𝓂.timings.nVars),
                    𝓂.timings;
                    periods = periods, 
                    shocks = shocks, 
                    shock_size = shock_size,
                    variables = variables, 
                    negative_shock = negative_shock) .+ SSS_delta[var_idx]
        else
            Y = irf(state_update, 
                    initial_state, 
                    zeros(𝓂.timings.nVars),
                    𝓂.timings;
                    periods = periods, 
                    shocks = shocks, 
                    shock_size = shock_size,
                    variables = variables, 
                    negative_shock = negative_shock) .+ SSS_delta[var_idx]
        end
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

    args_and_kwargs = Dict(:model_name => 𝓂.model_name,
                           :periods => periods,
                           :shocks => shocks,
                           :variables => variables,
                           :parameters => Dict(𝓂.parameters .=> 𝓂.parameter_values),
                           :algorithm => algorithm,
                           :shock_size => shock_size,
                           :negative_shock => negative_shock,
                           :generalised_irf => generalised_irf,
                           :initial_state => initial_state,
                           :ignore_obc => ignore_obc,
                           :tol => tol,
                           :quadratic_matrix_equation_algorithm => quadratic_matrix_equation_algorithm,
                           :sylvester_algorithm => sylvester_algorithm,
                           :lyapunov_algorithm => lyapunov_algorithm,
                           :plot_data => Y,
                           :reference_steady_state => reference_steady_state[var_idx],
                           :variable_names => variable_names,
                           :shock_names => shock_names,
                           :shock_idx => shock_idx,
                           :var_idx => var_idx)

    push!(irf_active_plot_container, args_and_kwargs)
    
    diffdict = compare_args_and_kwargs(irf_active_plot_container)

    @assert haskey(diffdict, :parameters) || haskey(diffdict, :shock_names) "New plot must be different from previous plot. Use the version without ! to plot."
    
    annotate_ss = Vector{Pair{String, Any}}[]

    annotate_ss_page = Pair{String,Any}[]

    annotate_diff_input = Pair{String,Any}[]

    len_diff = length(irf_active_plot_container)

    if haskey(diffdict, :parameters)
        param_nms = diffdict[:parameters] |> keys |> collect |> sort
        for param in param_nms
            push!(annotate_diff_input, String(param) => diffdict[:parameters][param])
        end
    end
    
    if haskey(diffdict, :shock_names)
        if all(length.(diffdict[:shock_names]) .== 1)
            push!(annotate_diff_input, "Shock" => reduce(vcat,diffdict[:shock_names]))
        end
    end
    
    if haskey(diffdict, :negative_shock)
        push!(annotate_diff_input, "Negative shock" => reduce(vcat,diffdict[:negative_shock]))
        same_shock_direction = false
    else
        same_shock_direction = true
    end

    pushfirst!(annotate_diff_input, "Plot index" => 1:len_diff)
    

    shock_dir = same_shock_direction ? negative_shock ? "Shock⁻" : "Shock⁺" : "Shock"

    if shocks == :none
        shock_dir = ""
    end
    if shocks == :simulate
        shock_dir = "Shocks"
    end
    if !(shocks isa Union{Symbol_input,String_input})
        shock_dir = ""
    end

    legend_plot = StatsPlots.plot(framestyle = :none, legend_columns = length(irf_active_plot_container)) 
    
    joint_shocks = OrderedSet{String}()
    joint_variables = OrderedSet{String}()
    single_shock_per_irf = true

    for (i,k) in enumerate(irf_active_plot_container)
        StatsPlots.plot!(legend_plot,
                        fill(0,1,1), 
                        legend_title = length(annotate_diff_input) > 2 ? nothing : annotate_diff_input[2][1],
                        framestyle = :none, 
                        legend = :inside, 
                        label = length(annotate_diff_input) > 2 ? i : annotate_diff_input[2][2][i])

        push!(joint_shocks, k[:shock_names]...)
        push!(joint_variables, k[:variable_names]...)
        single_shock_per_irf = single_shock_per_irf && length(k[:shock_names]) == 1
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
        can_dual_axiss = Bool[]

        for var in joint_variables
            not_zero_in_any_irf = false
            can_dual_axis = gr_back

            for k in irf_active_plot_container
                var_idx = findfirst(==(var), k[:variable_names])
                shock_idx = shock == :single_shock_per_irf ? 1 : findfirst(==(shock), k[:shock_names])
                
                if isnothing(var_idx) || isnothing(shock_idx)
                    # If the variable or shock is not present in the current irf_active_plot_container,
                    # we skip this iteration.
                    continue
                else
                    if any(.!isapprox.(k[:plot_data][var_idx,:,shock_idx], 0, atol = eps(Float32)))
                        not_zero_in_any_irf = not_zero_in_any_irf || true
                        # break # If any irf data is not approximately zero, we set the flag to true.
                    end

                    SS = k[:reference_steady_state][var_idx]

                    if all((k[:plot_data][var_idx,:,shock_idx] .+ SS) .> eps(Float32)) && (SS > eps(Float32))
                        can_dual_axis = can_dual_axis && true
                    else
                        can_dual_axis = can_dual_axis && false
                    end
                end
            end

            if not_zero_in_any_irf 
                push!(joint_non_zero_variables, var)
                push!(can_dual_axiss, can_dual_axis)
            else
                # If all irf data for this variable and shock is approximately zero, we skip this subplot.
                n_subplots -= 1
            end
        end

        for (var, can_dual_axis) in zip(joint_non_zero_variables, can_dual_axiss)
            SSs = eltype(irf_active_plot_container[1][:reference_steady_state])[]
            Ys = AbstractVector{eltype(irf_active_plot_container[1][:plot_data])}[]

            for k in irf_active_plot_container
                var_idx = findfirst(==(var), k[:variable_names])
                shock_idx = shock == :single_shock_per_irf ? 1 : findfirst(==(shock), k[:shock_names])

                if isnothing(var_idx) || isnothing(shock_idx)
                    # If the variable or shock is not present in the current irf_active_plot_container,
                    # we skip this iteration.
                    push!(SSs, NaN)
                    push!(Ys, zeros(0))
                else
                    push!(SSs, k[:reference_steady_state][var_idx])
                    push!(Ys, k[:plot_data][var_idx,:,shock_idx])
                end
            end
            
            same_ss = true

            if maximum(filter(!isnan, SSs)) - minimum(filter(!isnan, SSs)) > 1e-10
                push!(annotate_ss_page, var => minimal_sigfig_strings(SSs))
                same_ss = false
            end

            push!(pp, plot_irf_subplot( Ys, 
                                    SSs, 
                                    var, 
                                    can_dual_axis,
                                    same_ss))
            
            if !(plot_count % plots_per_page == 0)
                plot_count += 1
            else
                plot_count = 1

                if shock == :single_shock_per_irf
                    shock_string = ": multiple shocks"
                    shock_name = "multiple_shocks"
                elseif shock == :simulate
                    shock_string = ": simulate all"
                    shock_name = "simulation"
                elseif shock == :none
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
                
                pushfirst!(annotate_ss_page, "Plot index" => 1:len_diff)
                
                push!(annotate_ss, annotate_ss_page)
                
                plot_title = "Model: "*𝓂.model_name*"        " * shock_dir *  shock_string *"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"

                plot_elements = [ppp, legend_plot]

                layout_heights = [15,1]

                if length(annotate_diff_input) > 2
                    annotate_diff_input_plot = plot_df(annotate_diff_input)

                    ppp_input_diff = StatsPlots.plot(annotate_diff_input_plot; attributes...)

                    push!(plot_elements, ppp_input_diff)

                    push!(layout_heights, 5)
                end

                if length(annotate_ss[pane]) > 1
                    annotate_ss_plot = plot_df(annotate_ss[pane])

                    ppp_ss = StatsPlots.plot(annotate_ss_plot; attributes...)

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
                    StatsPlots.savefig(p, save_plots_path * "/irf__" * 𝓂.model_name * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
                end

                pane += 1

                annotate_ss_page = Pair{String,Any}[]

                pp = []
            end
        end


        if length(pp) > 0
            if shock == :single_shock_per_irf
                    shock_string = ": multiple shocks"
                    shock_name = "multiple_shocks"
            elseif shock == :simulate
                shock_string = ": simulate all"
                shock_name = "simulation"
            elseif shock == :none
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
            
            pushfirst!(annotate_ss_page, "Plot index" => 1:len_diff)
            
            push!(annotate_ss, annotate_ss_page)
            
            plot_title = "Model: "*𝓂.model_name*"        " * shock_dir *  shock_string *"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"

            plot_elements = [ppp, legend_plot]

            layout_heights = [15,1]

            if length(annotate_diff_input) > 2
                annotate_diff_input_plot = plot_df(annotate_diff_input)

                ppp_input_diff = StatsPlots.plot(annotate_diff_input_plot; attributes...)

                push!(plot_elements, ppp_input_diff)

                push!(layout_heights, 5)
            end

            if length(annotate_ss[pane]) > 1
                annotate_ss_plot = plot_df(annotate_ss[pane])

                ppp_ss = StatsPlots.plot(annotate_ss_plot; attributes...)

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
                StatsPlots.savefig(p, save_plots_path * "/irf__" * 𝓂.model_name * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
            end
        end

        annotate_ss = Vector{Pair{String, Any}}[]
    end

    return return_plots
end

function minimal_sigfig_strings(v::AbstractVector{<:Real};
        min_sig::Int = 3, n::Int = 10, dup_tol::Float64 = 1e-13)

    idx = collect(eachindex(v))
    finite_mask = map(x -> isfinite(x) && x != 0, v)
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


function plot_df(plot_vector::Vector{Pair{String,Any}})
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
 
    # overlay the header and numeric values
    for j in 1:ncols
        StatsPlots.annotate!(df_plot, j, 1, StatsPlots.text(plot_vector[j].first, :center, 8)) # Header
        for i in 1:nrows
            StatsPlots.annotate!(df_plot, j, i + 1, StatsPlots.text(string(plot_vector[j].second[i]), :center, 8))
        end
    end

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

    n_subplots = length(var_idx)
    pp = []
    pane = 1
    plot_count = 1
    return_plots = []

    for k in vars_to_plot
        if gr_back
            push!(pp,StatsPlots.groupedbar(fevds(k,:,:)', title = replace_indices_in_symbol(k), bar_position = :stack, legend = :none))
        else
            push!(pp,StatsPlots.groupedbar(fevds(k,:,:)', title = replace_indices_in_symbol(k), bar_position = :stack, label = reshape(string.(replace_indices_in_symbol.(shocks_to_plot)),1,length(shocks_to_plot))))
        end

        if !(plot_count % plots_per_page == 0)
            plot_count += 1
        else
            plot_count = 1

            ppp = StatsPlots.plot(pp...; attributes...)

            p = StatsPlots.plot(ppp,StatsPlots.bar(fill(0,1,length(shocks_to_plot)), 
                                        label = reshape(string.(replace_indices_in_symbol.(shocks_to_plot)),1,length(shocks_to_plot)), 
                                        linewidth = 0 , 
                                        framestyle = :none, 
                                        legend = :inside, 
                                        legend_columns = legend_columns), 
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

        p = StatsPlots.plot(ppp,StatsPlots.bar(fill(0,1,length(shocks_to_plot)), 
                                    label = reshape(string.(replace_indices_in_symbol.(shocks_to_plot)),1,length(shocks_to_plot)), 
                                    linewidth = 0 , 
                                    framestyle = :none, 
                                    legend = :inside, 
                                    legend_columns = legend_columns), 
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
- $ALGORITHM®
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

    legend_plot = StatsPlots.plot(framestyle = :none) 

    for a in algorithm
        StatsPlots.plot!(fill(0,1,1), 
        framestyle = :none, 
        legend = :inside, 
        label = labels[a][1])
    end
    
    for a in algorithm
        StatsPlots.scatter!(fill(0,1,1), 
        framestyle = :none, 
        legend = :inside, 
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

    StatsPlots.scatter!(fill(0,1,1), 
                        label = "", 
                        marker = :rect,
                        markerstrokecolor = :white,
                        markerstrokewidth = 0, 
                        markercolor = :white,
                        linecolor = :white, 
                        linewidth = 0, 
                        framestyle = :none, 
                        legend = :inside)

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

        push!(pp,begin
                    Pl = StatsPlots.plot() 

                    for a in algorithm
                        StatsPlots.plot!(state_range .+ relevant_SS_dictionnary[a][indexin([state], 𝓂.var)][1], 
                            variable_dict[a][k][1,:], 
                            ylabel = replace_indices_in_symbol(k)*"₍₀₎", 
                            xlabel = replace_indices_in_symbol(state)*"₍₋₁₎", 
                            label = "")
                    end

                    for a in algorithm
                        StatsPlots.scatter!([relevant_SS_dictionnary[a][indexin([state], 𝓂.var)][1]], [relevant_SS_dictionnary[a][indexin([k], 𝓂.var)][1]], 
                        label = "")
                    end

                    Pl
        end)

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
- `levels` [Default: `false`, Type: `Bool`]: $LEVELS®
- $SHOW_PLOTS®
- $SAVE_PLOTS®
- $SAVE_PLOTS_FORMATH®
- $SAVE_PLOTS_PATH®
- $PLOTS_PER_PAGE®
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

# c is conditioned to deviate by 0.01 in period 1 and y is conditioned to deviate by 0.02 in period 3
conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,2),Variables = [:c,:y], Periods = 1:2)
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
                                    levels::Bool = false,
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
                                levels = levels,
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

    reference_steady_state = [s ∈ union(map(x -> Symbol(string(x) * "₍ₓ₎"), 𝓂.timings.exo), 𝓂.exo_present) ? 0 : relevant_SS(s) for s in var_names]

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

    n_subplots = length(var_idx)
    pp = []
    pane = 1
    plot_count = 1

    return_plots = []

    for i in 1:length(var_idx)
        if all(isapprox.(Y[i,:], 0, atol = eps(Float32))) && !(any(vcat(conditions,shocks)[var_idx[i],:] .!= nothing))
            n_subplots -= 1
        end
    end

    for i in 1:length(var_idx)
        SS = reference_steady_state[i]
        if !(all(isapprox.(Y[i,:],0,atol = eps(Float32)))) || length(findall(vcat(conditions,shocks)[var_idx[i],:] .!= nothing)) > 0
        
            if all((Y[i,:] .+ SS) .> eps(Float32)) & (SS > eps(Float32))
                cond_idx = findall(vcat(conditions,shocks)[var_idx[i],:] .!= nothing)

                if length(cond_idx) > 0
                    push!(pp,begin
                                StatsPlots.plot(1:periods, Y[i,:] .+ SS, title = replace_indices_in_symbol(full_SS[var_idx[i]]), ylabel = "Level", label = "")
                                if gr_back StatsPlots.plot!(StatsPlots.twinx(),1:periods, 100*((Y[i,:] .+ SS) ./ SS .- 1), ylabel = LaTeXStrings.L"\% \Delta", label = "") end
                                StatsPlots.hline!(gr_back ? [SS 0] : [SS],color = :black,label = "")   
                                StatsPlots.scatter!(cond_idx, conditions_in_levels ? vcat(conditions,shocks)[var_idx[i],cond_idx] : vcat(conditions,shocks)[var_idx[i],cond_idx] .+ SS, label = "",marker = gr_back ? :star8 : :pentagon, markercolor = :black)                            
                    end)
                else
                    push!(pp,begin
                                StatsPlots.plot(1:periods, Y[i,:] .+ SS, title = replace_indices_in_symbol(full_SS[var_idx[i]]), ylabel = "Level", label = "")
                                if gr_back StatsPlots.plot!(StatsPlots.twinx(),1:periods, 100*((Y[i,:] .+ SS) ./ SS .- 1), ylabel = LaTeXStrings.L"\% \Delta", label = "") end
                                StatsPlots.hline!(gr_back ? [SS 0] : [SS],color = :black,label = "")                              
                    end)
                end
            else
                cond_idx = findall(vcat(conditions,shocks)[var_idx[i],:] .!= nothing)
                if length(cond_idx) > 0
                    push!(pp,begin
                                StatsPlots.plot(1:periods, Y[i,:] .+ SS, title = replace_indices_in_symbol(full_SS[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                StatsPlots.hline!([SS], color = :black, label = "")
                                StatsPlots.scatter!(cond_idx, conditions_in_levels ? vcat(conditions,shocks)[var_idx[i],cond_idx] : vcat(conditions,shocks)[var_idx[i],cond_idx] .+ SS, label = "",marker = gr_back ? :star8 : :pentagon, markercolor = :black)  
                    end)
                else 
                    push!(pp,begin
                                StatsPlots.plot(1:periods, Y[i,:] .+ SS, title = replace_indices_in_symbol(full_SS[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                StatsPlots.hline!([SS], color = :black, label = "")
                    end)
                end

            end

            if !(plot_count % plots_per_page == 0)
                plot_count += 1
            else
                plot_count = 1

                shock_string = "Conditional forecast"

                ppp = StatsPlots.plot(pp...; attributes...)

                p = StatsPlots.plot(ppp,begin
                                            StatsPlots.scatter(fill(0,1,1), 
                                            label = "Condition", 
                                            marker = gr_back ? :star8 : :pentagon,
                                            markercolor = :black,
                                            linewidth = 0, 
                                            framestyle = :none, 
                                            legend = :inside)

                                            StatsPlots.scatter!(fill(0,1,1), 
                                            label = "", 
                                            marker = :rect,
                                            # markersize = 2,
                                            markerstrokecolor = :white,
                                            markerstrokewidth = 0, 
                                            markercolor = :white,
                                            linecolor = :white, 
                                            linewidth = 0, 
                                            framestyle = :none, 
                                            legend = :inside)
                                        end, 
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

        p = StatsPlots.plot(ppp,begin
                                StatsPlots.scatter(fill(0,1,1), 
                                label = "Condition", 
                                marker = gr_back ? :star8 : :pentagon,
                                markercolor = :black,
                                linewidth = 0, 
                                framestyle = :none, 
                                legend = :inside)

                                StatsPlots.scatter!(fill(0,1,1), 
                                label = "", 
                                marker = :rect,
                                # markersize = 2,
                                markerstrokecolor = :white,
                                markerstrokewidth = 0, 
                                markercolor = :white,
                                linecolor = :white, 
                                linewidth = 0, 
                                framestyle = :none, 
                                legend = :inside)
                                end, 
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

end # dispatch_doctor

end # module