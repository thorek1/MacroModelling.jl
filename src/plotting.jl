
import Plots, Plots.PlotMeasures
import LaTeXStrings
import StatsPlots

"""
```
gr_backend()
```
Renaming and reexport of Plot.jl function `gr()` to define GR.jl as backend
"""
gr_backend = Plots.gr



"""
```
plotlyjs_backend()
```
Renaming and reexport of Plot.jl function `plotlyjs()` to define PlotlyJS.jl as backend
"""
plotlyjs_backend = Plots.plotlyjs



"""
$(SIGNATURES)
Plot impulse response functions (IRFs) of the model.

The left axis shows the level, and the right the deviation from the reference steady state. Linear solutions have the non stochastic steady state as reference other solution the stochastic steady state. The horizontal black line indicates the reference steady state. Variable names are above the subplots and the title provides information about the model, shocks and number of pages per shock.

# Arguments
- $MODEL
# Keyword Arguments
- `plots_per_page` [Default: `9`, Type: `Int`]: how many plots to show per page
- `save_plots` [Default: `false`, Type: `Bool`]: switch to save plots using path and extension from `save_plots_path` and `save_plots_format`. Separate files per shocks and variables depending on number of variables and `plots_per_page`
- `save_plots_path` [Default: `pwd()`, Type: `String`]: path where to save plots
- `save_plots_format` [Default: `:pdf`, Type: `Symbol`]: output format of saved plots. See [input formats compatible with GR](https://docs.juliaplots.org/latest/output/#Supported-output-file-formats) for valid formats.
- `show_plots` [Default: `true`, Type: `Bool`]: show plots. Separate plots per shocks and varibles depending on number of variables and `plots_per_page`.
- $PERIODS
- $ALGORITHM
- $PARAMETERS
- $VARIABLES
- $SHOCKS
- $NEGATIVE_SHOCK
- $GENERALISED_IRF
- $INITIAL_STATE
- $VERBOSE

# Examples
```julia
using MacroModelling

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
    shocks::Union{Symbol_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Symbol_input = :all,
    parameters = nothing,
    show_plots::Bool = true,
    save_plots::Bool = false,
    save_plots_format::Symbol = :pdf,
    save_plots_path::String = ".",
    plots_per_page::Int = 9, 
    algorithm::Symbol = :first_order,
    negative_shock::Bool = false,
    generalised_irf::Bool = false,
    initial_state::Vector{Float64} = [0.0],
    verbose::Bool = false)

    gr_backend = Plots.backend() == Plots.GRBackend()

    Plots.default(size=(700,500),
                    plot_titlefont = 10, 
                    titlefont = 10, 
                    guidefont = 8, 
                    legendfontsize = 8, 
                    tickfontsize = 8,
                    framestyle = :box)

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    solve!(𝓂, verbose = verbose, dynamics = true, algorithm = algorithm)

    state_update = parse_algorithm_to_state_update(algorithm, 𝓂)

    NSSS, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (𝓂.solution.non_stochastic_steady_state, eps())

    full_SS = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))
    full_SS[indexin(𝓂.aux,full_SS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)

    NSSS_labels = [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]

    reference_steady_state = [s ∈ 𝓂.exo_present ? 0 : NSSS[indexin([s],NSSS_labels)...] for s in full_SS]

    if algorithm == :second_order
        SSS_delta = reference_steady_state - 𝓂.solution.perturbation.second_order.stochastic_steady_state
    elseif algorithm == :third_order
        SSS_delta = reference_steady_state - 𝓂.solution.perturbation.third_order.stochastic_steady_state
    else
        SSS_delta = zeros(length(reference_steady_state))
    end

    if algorithm == :second_order
        reference_steady_state = 𝓂.solution.perturbation.second_order.stochastic_steady_state
    elseif algorithm == :third_order
        reference_steady_state = 𝓂.solution.perturbation.third_order.stochastic_steady_state
    end

    initial_state = initial_state == [0.0] ? zeros(𝓂.timings.nVars) - SSS_delta : initial_state[indexin(full_SS, sort(union(𝓂.var,𝓂.exo_present)))] - reference_steady_state
    
    shocks = 𝓂.timings.nExo == 0 ? :none : shocks

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == 𝓂.timings.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        shock_idx = 1
    elseif shocks isa KeyedArray{Float64}
        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,𝓂.timings)
    end

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

    if generalised_irf
        Y = girf(state_update, zeros(𝓂.timings.nVars), 𝓂.timings; periods = periods, shocks = shocks, variables = variables, negative_shock = negative_shock)#, warmup_periods::Int = 100, draws::Int = 50, iterations_to_steady_state::Int = 500)
    else
        Y = irf(state_update, initial_state, zeros(𝓂.timings.nVars), 𝓂.timings; periods = periods, shocks = shocks, variables = variables, negative_shock = negative_shock) .+ SSS_delta[var_idx]
    end

    # fontt = "computer modern"#"serif-roman"#
    # fontt = "times roman"#"serif-roman"#
    # fontt = "symbol"#"serif-roman"#

    # plots = []
		if shocks isa KeyedArray{Float64} || shocks isa Matrix{Float64}  
				periods += size(shocks)[2]
		end

    shock_dir = negative_shock ? "Shock⁻" : "Shock⁺"

    if shocks == :none
        shock_dir = ""
    end
    if shocks == :simulate
        shock_dir = "Shocks"
    end
    if !(shocks isa Symbol_input)
        shock_dir = ""
    end

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
            if !(all(isapprox.(Y[i,:,shock],0,atol = eps(Float32))))
            # if !(plot_count ∈ unique(round.((1:𝓂.timings.timings.nVars)/plots_per_page))*plots_per_page)
                if !(plot_count % plots_per_page == 0)
                    plot_count += 1
                    if all((Y[i,:,shock] .+ SS) .> eps(Float32)) & (SS > eps(Float32))
                        push!(pp,begin
                                    Plots.plot(1:periods, Y[i,:,shock] .+ SS,title = string(𝓂.timings.var[var_idx[i]]),ylabel = "Level",label = "")
                                    if gr_backend Plots.plot!(Plots.twinx(),1:periods, 100*((Y[i,:,shock] .+ SS) ./ SS .- 1), ylabel = LaTeXStrings.L"\% \Delta", label = "") end
                                    Plots.hline!(gr_backend ? [SS 0] : [SS], color = :black, label = "")                               
                        end)
                    else
                        push!(pp,begin
                                    Plots.plot(1:periods, Y[i,:,shock] .+ SS, title = string(𝓂.timings.var[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                    Plots.hline!([SS], color = :black, label = "")
                        end)

                    end
                else

                    plot_count = 1
                    if all((Y[i,:,shock] .+ SS) .> eps(Float32)) & (SS > eps(Float32))
                        push!(pp,begin
                                    Plots.plot(1:periods, Y[i,:,shock] .+ SS,title = string(𝓂.timings.var[var_idx[i]]),ylabel = "Level",label = "")
                                    if gr_backend Plots.plot!(Plots.twinx(),1:periods, 100*((Y[i,:,shock] .+ SS) ./ SS .- 1), ylabel = LaTeXStrings.L"\% \Delta", label = "") end
                                    Plots.hline!(gr_backend ? [SS 0] : [SS],color = :black,label = "")                               
                        end)
                    else
                        push!(pp,begin
                                    Plots.plot(1:periods, Y[i,:,shock] .+ SS, title = string(𝓂.timings.var[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                    Plots.hline!([SS], color = :black, label = "")
                        end)

                    end

                    if shocks == :simulate
                        shock_string = ": simulate all"
                        shock_name = "simulation"
                    elseif shocks == :none
                        shock_string = ""
                        shock_name = "no_shock"
                    elseif shocks isa Symbol_input
                        shock_string = ": " * string(𝓂.timings.exo[shock_idx[shock]])
                        shock_name = string(𝓂.timings.exo[shock_idx[shock]])
                    else
                        shock_string = "Series of shocks"
                        shock_name = "shock_matrix"
                    end

                    p = Plots.plot(pp...,plot_title = "Model: "*𝓂.model_name*"        " * shock_dir *  shock_string *"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

                    # p[:plot_title] = String(𝓂.timings.exo[shock])

                    # end
                    push!(return_plots,p)

                    if show_plots# & (length(pp) > 0)
                        display(p)
                    end

                    if save_plots# & (length(pp) > 0)
                        Plots.savefig(p, save_plots_path * "/irf__" * 𝓂.model_name * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
                    end

                    pane += 1
                    pp = []
                end
            end
        end

        # if length(pp) == 1
        #     plot(pp)
        # elseif length(pp) > 1
        if length(pp) > 0


            if shocks == :simulate
                shock_string = ": simulate all"
                shock_name = "simulation"
            elseif shocks == :none
                shock_string = ""
                shock_name = "no_shock"
            elseif shocks isa Symbol_input
                shock_string = ": " * string(𝓂.timings.exo[shock_idx[shock]])
                shock_name = string(𝓂.timings.exo[shock_idx[shock]])
            else
                shock_string = "Series of shocks"
                shock_name = "shock_matrix"
            end

            p = Plots.plot(pp...,plot_title = "Model: "*𝓂.model_name*"        " * shock_dir *  shock_string*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

            push!(return_plots,p)

            if show_plots# & (length(pp) > 0)
                #println(length(pp))
                display(p)
            end

            if save_plots# & (length(pp) > 0)
                # savefig(p,"irf__"*string(𝓂.timings.exo[shock_idx[shock]])*"__"*string(pane)*".pdf")
                Plots.savefig(p, save_plots_path * "/irf__" * 𝓂.model_name * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
            end
        end
    end

    return return_plots
end




"""
See [`plot_irf`](@ref)
"""
plot_IRF = plot_irf


"""
See [`plot_irf`](@ref)
"""
plot_irfs = plot_irf


"""
Wrapper for [`plot_irf`](@ref) with `shocks = :simulate` and `periods = 100`.
"""
plot_simulations(args...; kwargs...) =  plot_irf(args...; kwargs..., shocks = :simulate, periods = 100)


"""
Wrapper for [`plot_irf`](@ref) with `generalised_irf = true`.
"""
plot_girf(args...; kwargs...) =  plot_irf(args...; kwargs..., generalised_irf = true)





"""
$(SIGNATURES)
Plot conditional variance decomposition of the model.

The vertical axis shows the share of the shocks variance contribution, and horizontal axis the period of the variance decomposition. The stacked bars represent each shocks variance contribution at a specific time horizon.

# Arguments
- $MODEL
# Keyword Arguments
- $PERIODS
- $VARIABLES
- $PARAMETERS
- `show_plots` [Default: `true`, Type: `Bool`]: show plots. Separate plots per shocks and varibles depending on number of variables and `plots_per_page`.
- `save_plots` [Default: `false`, Type: `Bool`]: switch to save plots using path and extension from `save_plots_path` and `save_plots_format`. Separate files per shocks and variables depending on number of variables and `plots_per_page`
- `save_plots_path` [Default: `pwd()`, Type: `String`]: path where to save plots
- `save_plots_format` [Default: `:pdf`, Type: `Symbol`]: output format of saved plots. See [input formats compatible with GR](https://docs.juliaplots.org/latest/output/#Supported-output-file-formats) for valid formats.
- `plots_per_page` [Default: `9`, Type: `Int`]: how many plots to show per page
- $VERBOSE

# Examples
```julia
using MacroModelling

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
    variables::Symbol_input = :all,
    parameters = nothing,
    show_plots::Bool = true,
    save_plots::Bool = false,
    save_plots_format::Symbol = :pdf,
    save_plots_path::String = ".",
    plots_per_page::Int = 9, 
    verbose::Bool = false)

    gr_backend = Plots.backend() == Plots.GRBackend()

    Plots.default(size=(700,500),
                    plot_titlefont = 10, 
                    titlefont = 10, 
                    guidefont = 8, 
                    legendfontsize = 8, 
                    tickfontsize = 8,
                    framestyle = :box)

    fevds = get_conditional_variance_decomposition(𝓂,
                                                    periods = 1:periods,
                                                    parameters = parameters,
                                                    verbose = verbose)

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

    vars_to_plot = intersect(axiskeys(fevds)[1],𝓂.timings.var[var_idx])
    
    shocks_to_plot = axiskeys(fevds)[2]

    n_subplots = length(var_idx)
    pp = []
    pane = 1
    plot_count = 1
    return_plots = []

    for k in vars_to_plot
        if !(plot_count % plots_per_page == 0)
            plot_count += 1
            if gr_backend
                push!(pp,StatsPlots.groupedbar(fevds(k,:,:)', title = string(k), bar_position = :stack, legend = :none))
            else
                push!(pp,StatsPlots.groupedbar(fevds(k,:,:)', title = string(k), bar_position = :stack, label = reshape(string.(shocks_to_plot),1,length(shocks_to_plot))))
            end
        else
            plot_count = 1

            if gr_backend
                push!(pp,StatsPlots.groupedbar(fevds(k,:,:)', title = string(k), bar_position = :stack, legend = :none))
            else
                push!(pp,StatsPlots.groupedbar(fevds(k,:,:)', title = string(k), bar_position = :stack, label = reshape(string.(shocks_to_plot),1,length(shocks_to_plot))))
            end
            ppp = Plots.plot(pp...)

            p = Plots.plot(ppp,Plots.bar(fill(0,1,length(shocks_to_plot)), 
                                        label = reshape(string.(shocks_to_plot),1,length(shocks_to_plot)), 
                                        linewidth = 0 , 
                                        framestyle = :none, 
                                        legend = :inside, 
                                        legend_columns = -1), 
                                        layout = Plots.grid(2, 1, heights=[0.99, 0.01]),
                                        plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

            push!(return_plots,gr_backend ? p : ppp)

            if show_plots
                display(p)
            end

            if save_plots
                Plots.savefig(p, save_plots_path * "/fevd__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
            end

            pane += 1
            pp = []
        end
    end

    if length(pp) > 0
        ppp = Plots.plot(pp...)

        p = Plots.plot(ppp,Plots.bar(fill(0,1,length(shocks_to_plot)), 
                                    label = reshape(string.(shocks_to_plot),1,length(shocks_to_plot)), 
                                    linewidth = 0 , 
                                    framestyle = :none, 
                                    legend = :inside, 
                                    legend_columns = -1), 
                                    layout = Plots.grid(2, 1, heights=[0.99, 0.01]),
                                    plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

        push!(return_plots,gr_backend ? p : ppp)

        if show_plots
            display(p)
        end

        if save_plots
            Plots.savefig(p, save_plots_path * "/fevd__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
        end
    end

    return return_plots
end



"""
See [`plot_conditional_variance_decomposition`](@ref)
"""
plot_fevd = plot_conditional_variance_decomposition

"""
See [`plot_conditional_variance_decomposition`](@ref)
"""
plot_forecast_error_variance_decomposition = plot_conditional_variance_decomposition





"""
$(SIGNATURES)
Plot the solution of the model (mapping of past states to present variables) around the (non) stochastic steady state (depending on chosen solution algorithm). Each plot shows the relationship between the chosen state (defined in `state`) and one of the chosen variables (defined in `variables`). 

The (non) stochastic steady state is plotted along with the mapping from the chosen past state to one present variable per plot. All other (non-chosen) states remain in the (non) stochastic steady state.

# Arguments
- $MODEL
- `state` [Type: `Symbol`]: state variable to be shown on x-axis.
# Keyword Arguments
- $VARIABLES
- `algorithm` [Default: `:first_order`, Type: Union{Symbol,Vector{Symbol}}]: solution algorithm for which to show the IRFs. Can be more than one: `[:second_order,:third_order]`"
- `σ` [Default: `2`, Type: `Union{Int64,Float64}`]: defines the range of the state variable around the (non) stochastic steady state in standard deviations. E.g. a value of 2 means that the state variable is plotted for values of the (non) stochastic steady state in standard deviations +/- 2 standard deviations.
- $PARAMETERS
- `show_plots` [Default: `true`, Type: `Bool`]: show plots. Separate plots per shocks and varibles depending on number of variables and `plots_per_page`.
- `save_plots` [Default: `false`, Type: `Bool`]: switch to save plots using path and extension from `save_plots_path` and `save_plots_format`. Separate files per shocks and variables depending on number of variables and `plots_per_page`
- `save_plots_path` [Default: `pwd()`, Type: `String`]: path where to save plots
- `save_plots_format` [Default: `:pdf`, Type: `Symbol`]: output format of saved plots. See [input formats compatible with GR](https://docs.juliaplots.org/latest/output/#Supported-output-file-formats) for valid formats.
- `plots_per_page` [Default: `4`, Type: `Int`]: how many plots to show per page
- $VERBOSE

# Examples
```julia
using MacroModelling

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
    state::Symbol;
    variables::Symbol_input = :all,
    algorithm::Union{Symbol,Vector{Symbol}} = :first_order,
    σ::Union{Int64,Float64} = 2,
    parameters = nothing,
    show_plots::Bool = true,
    save_plots::Bool = false,
    save_plots_format::Symbol = :pdf,
    save_plots_path::String = ".",
    plots_per_page::Int = 4,
    verbose::Bool = false)

    Plots.default(size=(700,500),
                    plot_titlefont = 10, 
                    titlefont = 10, 
                    guidefont = 8, 
                    legendfontsize = 8, 
                    tickfontsize = 8,
                    framestyle = :box)

    @assert state ∈ 𝓂.timings.past_not_future_and_mixed "Invalid state. Choose one from:"*repr(𝓂.timings.past_not_future_and_mixed)

    @assert length(setdiff(algorithm isa Symbol ? [algorithm] : algorithm, [:third_order, :second_order, :first_order])) == 0 "Invalid algorithm. Choose any combination of: :third_order, :second_order, :first_order"

    if algorithm isa Symbol
        max_algorithm = algorithm
        min_algorithm = algorithm
        algorithm = [algorithm]
    else
        if :third_order ∈ algorithm 
            max_algorithm = :third_order 
        elseif :second_order ∈ algorithm 
            max_algorithm = :second_order 
        else 
            max_algorithm = :first_order 
        end

        if :first_order ∈ algorithm 
            min_algorithm = :first_order 
        elseif :second_order ∈ algorithm 
            min_algorithm = :second_order 
        else 
            min_algorithm = :third_order 
        end
    end

    solve!(𝓂, verbose = verbose, algorithm = max_algorithm, dynamics = true)

    SS_and_std = get_moments(𝓂, 
                            derivatives = false,
                            parameters = parameters,
                            verbose = verbose)


    full_NSSS = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))
    full_NSSS[indexin(𝓂.aux,full_NSSS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    full_SS = [s ∈ 𝓂.exo_present ? 0 : SS_and_std[1](s) for s in full_NSSS]

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

    vars_to_plot = intersect(axiskeys(SS_and_std[1])[1],𝓂.timings.var[var_idx])

    state_range = collect(range(-SS_and_std[2](state), SS_and_std[2](state), 100)) * σ

    state_selector = state .== 𝓂.timings.var

    n_subplots = length(var_idx)
    pp = []
    pane = 1
    plot_count = 1
    return_plots = []

    
    legend_plot = Plots.plot(framestyle = :none) 
    if :first_order ∈ algorithm          
        Plots.plot!(fill(0,1,1), 
        framestyle = :none, 
        legend = :inside, 
        label = "1st order perturbation")
    end
    if :second_order ∈ algorithm    
        Plots.plot!(fill(0,1,1), 
        framestyle = :none, 
        legend = :inside, 
        label = "2nd order perturbation")
    end
    if :third_order ∈ algorithm    
        Plots.plot!(fill(0,1,1), 
        framestyle = :none, 
        legend = :inside, 
        label = "3rd order perturbation")
    end

    if :first_order ∈ algorithm   
        Plots.scatter!(fill(0,1,1), 
        framestyle = :none, 
        legend = :inside, 
        label = "Non Stochastic Steady State")
    end
    if :second_order ∈ algorithm    
        Plots.scatter!(fill(0,1,1), 
        framestyle = :none, 
        legend = :inside, 
        label = "Stochastic Steady State (2nd order)")
    end
    if :third_order ∈ algorithm    
        Plots.scatter!(fill(0,1,1), 
        framestyle = :none, 
        legend = :inside, 
        label = "Stochastic Steady State (3rd order)")
    end
    Plots.scatter!(fill(0,1,1), 
    label = "", 
    marker = :rect,
    markerstrokecolor = :white,
    markerstrokewidth = 0, 
    markercolor = :white,
    linecolor = :white, 
    linewidth = 0, 
    framestyle = :none, 
    legend = :inside)


    for k in vars_to_plot

        kk = Symbol(replace(string(k), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))

        if :first_order ∈ algorithm
            variable_first = [𝓂.solution.perturbation.first_order.state_update(state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

            variable_first = [(abs(x) > eps() ? x : 0.0) + SS_and_std[1](kk) for x in variable_first]
        end

        if :second_order ∈ algorithm
            SSS2 = 𝓂.solution.perturbation.second_order.stochastic_steady_state

            variable_second = [𝓂.solution.perturbation.second_order.state_update(SSS2 - full_SS .+ state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

            variable_second = [(abs(x) > eps() ? x : 0.0) + SS_and_std[1](kk) for x in variable_second]
        end

        if :third_order ∈ algorithm
            SSS3 = 𝓂.solution.perturbation.third_order.stochastic_steady_state

            variable_third = [𝓂.solution.perturbation.third_order.state_update(SSS3 - full_SS .+ state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

            variable_third = [(abs(x) > eps() ? x : 0.0) + SS_and_std[1](kk) for x in variable_third]
        end

        push!(pp,begin
                        Pl = Plots.plot() 
                        if :first_order ∈ algorithm
                                Plots.plot!(state_range .+ SS_and_std[1](state), 
                                variable_first, 
                                ylabel = string(k)*"₍₀₎", 
                                xlabel = string(state)*"₍₋₁₎", 
                                label = "")
                        end
                        if :second_order ∈ algorithm
                                Plots.plot!(state_range .+ SSS2[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1], 
                                variable_second, 
                                ylabel = string(k)*"₍₀₎", 
                                xlabel = string(state)*"₍₋₁₎", 
                                label = "")
                        end
                        if :third_order ∈ algorithm
                                Plots.plot!(state_range .+ SSS3[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1], 
                                variable_third, 
                                ylabel = string(k)*"₍₀₎", 
                                xlabel = string(state)*"₍₋₁₎", 
                                label = "")
                        end

                        if :first_order ∈ algorithm
                            Plots.scatter!([SS_and_std[1](state)], [SS_and_std[1](kk)], 
                            label = "")
                        end
                        if :second_order ∈ algorithm
                            Plots.scatter!([SSS2[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1]], [SSS2[indexin([k],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1]], 
                            label = "")
                        end
                        if :third_order ∈ algorithm
                            Plots.scatter!([SSS3[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1]], [SSS3[indexin([k],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1]], 
                            label = "")
                        end

                        Pl
        end)

        if !(plot_count % plots_per_page == 0)
            plot_count += 1
        else
            plot_count = 1

            ppp = Plots.plot(pp...)
            
            p = Plots.plot(ppp,
                            legend_plot, 
                            layout = Plots.grid(2, 1, heights=[0.9, 0.1]),
                            plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"
            )

            push!(return_plots,p)

            if show_plots
                display(p)
            end

            if save_plots
                Plots.savefig(p, save_plots_path * "/solution__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
            end

            pane += 1
            pp = []
        end
    end

    if length(pp) > 0
        ppp = Plots.plot(pp...)
            
        p = Plots.plot(ppp,
                        legend_plot, 
                        layout = Plots.grid(2, 1, heights=[0.9, 0.1]),
                        plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")"
        )

        push!(return_plots,p)

        if show_plots
            display(p)
        end

        if save_plots
            Plots.savefig(p, save_plots_path * "/solution__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
        end
    end

    return return_plots
end


"""
$(SIGNATURES)
Plot conditional forecast given restrictions on endogenous variables and shocks (optional) of the model. The algorithm finds the combinations of shocks with the smallest magnitude to match the conditions and plots both the endogenous variables and shocks.

The left axis shows the level, and the right axis the deviation from the non stochastic steady state. Variable names are above the subplots, conditioned values are marked, and the title provides information about the model, and number of pages.

Limited to the first order perturbation solution of the model.

# Arguments
- $MODEL
- $CONDITIONS
# Keyword Arguments
- $SHOCK_CONDITIONS
- `periods` [Default: `40`, Type: `Int`]: the total number of periods is the sum of the argument provided here and the maximum of periods of the shocks or conditions argument.
- $VARIABLES
`conditions_in_levels` [Default: `false`, Type: `Bool`]: indicator whether the conditions are provided in levels. If `true` the input to the conditions argument will have the non stochastic steady state substracted.
- $LEVELS
- `show_plots` [Default: `true`, Type: `Bool`]: show plots. Separate plots per shocks and varibles depending on number of variables and `plots_per_page`.
- `save_plots` [Default: `false`, Type: `Bool`]: switch to save plots using path and extension from `save_plots_path` and `save_plots_format`. Separate files per shocks and variables depending on number of variables and `plots_per_page`
- `save_plots_path` [Default: `pwd()`, Type: `String`]: path where to save plots
- `save_plots_format` [Default: `:pdf`, Type: `Symbol`]: output format of saved plots. See [input formats compatible with GR](https://docs.juliaplots.org/latest/output/#Supported-output-file-formats) for valid formats.
- `plots_per_page` [Default: `9`, Type: `Int`]: how many plots to show per page
- $VERBOSE

# Examples
```julia
using MacroModelling
using SparseArrays

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
conditions[2,2] = .02

# in period 2 second shock (eps_z) is conditioned to take a value of 0.05
shocks = Matrix{Union{Nothing,Float64}}(undef,2,1)
shocks[1,1] = .05

plot_conditional_forecast(RBC_CME, conditions, shocks = shocks)

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
    periods::Int = 40, 
    parameters = nothing,
    variables::Symbol_input = :all_including_auxilliary, 
    conditions_in_levels::Bool = false,
    levels::Bool = false,
    show_plots::Bool = true,
    save_plots::Bool = false,
    save_plots_format::Symbol = :pdf,
    save_plots_path::String = ".",
    plots_per_page::Int = 9,
    verbose::Bool = false)

    gr_backend = Plots.backend() == Plots.GRBackend()

    Plots.default(size=(700,500),
                    plot_titlefont = 10, 
                    titlefont = 10, 
                    guidefont = 8, 
                    legendfontsize = 8, 
                    tickfontsize = 8,
                    framestyle = :box)

    Y = get_conditional_forecast(𝓂,
                                conditions,
                                shocks = shocks, 
                                periods = periods, 
                                parameters = parameters,
                                variables = variables, 
                                conditions_in_levels = conditions_in_levels,
                                levels = levels,
                                verbose = verbose)

    periods += max(size(conditions,2), isnothing(shocks) ? 1 : size(shocks,2))

    full_SS = vcat(sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)),map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.timings.exo))

    NSSS, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (𝓂.solution.non_stochastic_steady_state, eps())
    
    var_names = axiskeys(Y,1)   

    var_idx = indexin(var_names,full_SS)

    NSSS_labels = [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]

    reference_steady_state = [s ∈ union(map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.timings.exo),𝓂.exo_present) ? 0 : NSSS[indexin([Symbol(replace(string(s), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))],NSSS_labels)...] for s in var_names]

    var_length = length(full_SS) - 𝓂.timings.nExo

    if conditions isa SparseMatrixCSC{Float64}
        @assert var_length == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(var_length) * " variables (including auxilliary variables): " * repr(var_names)

        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,var_length,periods)
        nzs = findnz(conditions)
        for i in 1:length(nzs[1])
            cond_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
        end
        conditions = cond_tmp
    elseif conditions isa Matrix{Union{Nothing,Float64}}
        @assert var_length == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(var_length) * " variables (including auxilliary variables): " * repr(var_names)

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

    # plots = []
    # for i in 1:length(var_idx)
        n_subplots = length(var_idx)
        pp = []
        pane = 1
        plot_count = 1

        return_plots = []

        for i in 1:length(var_idx)
            if all(isapprox.(Y[i,:], 0, atol = eps(Float32)))
                n_subplots -= 1
            end
        end

        for i in 1:length(var_idx)
            SS = reference_steady_state[i]
            if !(all(isapprox.(Y[i,:],0,atol = eps(Float32)))) || length(findall(vcat(conditions,shocks)[var_idx[i],:] .!= nothing)) > 0
            # if !(plot_count ∈ unique(round.((1:𝓂.timings.timings.nVars)/plots_per_page))*plots_per_page)
                if !(plot_count % plots_per_page == 0)
                    plot_count += 1
                    if all((Y[i,:] .+ SS) .> eps(Float32)) & (SS > eps(Float32))
                        cond_idx = findall(vcat(conditions,shocks)[var_idx[i],:] .!= nothing)
                        if length(cond_idx) > 0
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS,title = string(full_SS[var_idx[i]]),ylabel = "Level",label = "")
                                        if gr_backend Plots.plot!(Plots.twinx(),1:periods, 100*((Y[i,:] .+ SS) ./ SS .- 1), ylabel = LaTeXStrings.L"\% \Delta", label = "") end
                                        Plots.hline!(gr_backend ? [SS 0] : [SS], color = :black, label = "") 
                                        Plots.scatter!(cond_idx,vcat(conditions,shocks)[var_idx[i],cond_idx] .+ SS, label = "",marker = :star8, markercolor = :black)                             
                            end)
                        else
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS,title = string(full_SS[var_idx[i]]),ylabel = "Level",label = "")
                                        if gr_backend Plots.plot!(Plots.twinx(),1:periods, 100*((Y[i,:] .+ SS) ./ SS .- 1), ylabel = LaTeXStrings.L"\% \Delta", label = "") end
                                        Plots.hline!(gr_backend ? [SS 0] : [SS], color = :black, label = "")                         
                            end)
                        end
                    else
                        cond_idx = findall(vcat(conditions,shocks)[var_idx[i],:] .!= nothing)
                        if length(cond_idx) > 0
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS, title = string(full_SS[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                        Plots.hline!([SS], color = :black, label = "")
                                        Plots.scatter!(cond_idx,vcat(conditions,shocks)[var_idx[i],cond_idx] .+ SS, label = "",marker = :star8, markercolor = :black)   
                            end)
                        else
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS, title = string(full_SS[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                        Plots.hline!([SS], color = :black, label = "")
                            end)
                        end

                    end
                else

                    plot_count = 1
                    if all((Y[i,:] .+ SS) .> eps(Float32)) & (SS > eps(Float32))
                        cond_idx = findall(vcat(conditions,shocks)[var_idx[i],:] .!= nothing)
                        if length(cond_idx) > 0
                        push!(pp,begin
                                    Plots.plot(1:periods, Y[i,:] .+ SS,title = string(full_SS[var_idx[i]]),ylabel = "Level",label = "")
                                    if gr_backend Plots.plot!(Plots.twinx(),1:periods, 100*((Y[i,:] .+ SS) ./ SS .- 1), ylabel = LaTeXStrings.L"\% \Delta", label = "") end
                                    Plots.hline!(gr_backend ? [SS 0] : [SS],color = :black,label = "")   
                                    Plots.scatter!(cond_idx,vcat(conditions,shocks)[var_idx[i],cond_idx] .+ SS, label = "",marker = :star8, markercolor = :black)                            
                        end)
                    else
                        push!(pp,begin
                                    Plots.plot(1:periods, Y[i,:] .+ SS,title = string(full_SS[var_idx[i]]),ylabel = "Level",label = "")
                                    if gr_backend Plots.plot!(Plots.twinx(),1:periods, 100*((Y[i,:] .+ SS) ./ SS .- 1), ylabel = LaTeXStrings.L"\% \Delta", label = "") end
                                    Plots.hline!(gr_backend ? [SS 0] : [SS],color = :black,label = "")                              
                        end)
                    end
                    else
                        cond_idx = findall(vcat(conditions,shocks)[var_idx[i],:] .!= nothing)
                        if length(cond_idx) > 0
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS, title = string(full_SS[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                        Plots.hline!([SS], color = :black, label = "")
                                        Plots.scatter!(cond_idx,vcat(conditions,shocks)[var_idx[i],cond_idx] .+ SS, label = "",marker = :star8, markercolor = :black)  
                            end)
                        else 
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS, title = string(full_SS[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                        Plots.hline!([SS], color = :black, label = "")
                            end)
                        end

                    end

                    shock_string = "Conditional forecast"

                    ppp = Plots.plot(pp...)

                    p = Plots.plot(ppp,begin
                                                Plots.scatter(fill(0,1,1), 
                                                label = "Condition", 
                                                marker = :star8,
                                                markercolor = :black,
                                                linewidth = 0, 
                                                framestyle = :none, 
                                                legend = :inside)

                                                Plots.scatter!(fill(0,1,1), 
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
                                                layout = Plots.grid(2, 1, heights=[0.99, 0.01]),
                                                plot_title = "Model: "*𝓂.model_name*"        " * shock_string *"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")
                    
                    push!(return_plots,p)

                    if show_plots# & (length(pp) > 0)
                        display(p)
                    end

                    if save_plots# & (length(pp) > 0)
                        Plots.savefig(p, save_plots_path * "/conditional_forecast__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
                    end

                    pane += 1
                    pp = []
                end
            end
        end
        if length(pp) > 0

            shock_string = "Conditional forecast"

            # p = Plots.plot(pp...,plot_title = "Model: " * 𝓂.model_name * "        " * shock_string * "  (" * string(pane) * "/" * string(Int(ceil(n_subplots/plots_per_page))) * ")")

            ppp = Plots.plot(pp...)

            p = Plots.plot(ppp,begin
                                    Plots.scatter(fill(0,1,1), 
                                    label = "Condition", 
                                    marker = :star8,
                                    markercolor = :black,
                                    linewidth = 0, 
                                    framestyle = :none, 
                                    legend = :inside)

                                    Plots.scatter!(fill(0,1,1), 
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
                                        layout = Plots.grid(2, 1, heights=[0.99, 0.01]),
                                        plot_title = "Model: "*𝓂.model_name*"        " * shock_string *"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")
            
            push!(return_plots,p)

            if show_plots
                display(p)
            end

            if save_plots
                Plots.savefig(p, save_plots_path * "/conditional_forecast__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
            end
        end

    return return_plots

end