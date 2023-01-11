
using Plots, Plots.PlotMeasures, LaTeXStrings
using StatsPlots


"""
$(SIGNATURES)
Plot impulse response functions (IRFs) of the model.

The left axis shows the level, and the right the deviation from the reference steady state. Linear solutions have the non stochastic steady state as reference other solutoin the stochastic steady state. The horizontal black line indicates the reference steady state. Variable names are above the subplots and the title provides information about the model, shocks and number of pages per shock.

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

plot(RBC)
```
"""
function plot(𝓂::ℳ;
    periods::Int = 40, 
    shocks::Symbol_input = :all,
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
    verbose = false)

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    solve!(𝓂, verbose = verbose, dynamics = true, algorithm = algorithm)

    state_update = parse_algorithm_to_state_update(algorithm, 𝓂)

    NSSS, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, verbose) : (𝓂.solution.non_stochastic_steady_state, eps())

    full_SS = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))
    full_SS[indexin(𝓂.aux,full_SS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)

    if algorithm == :second_order
        reference_steady_state = 𝓂.solution.perturbation.second_order.stochastic_steady_state
    elseif algorithm == :third_order
        reference_steady_state = 𝓂.solution.perturbation.third_order.stochastic_steady_state
    elseif algorithm ∈ [:linear_time_iteration, :riccati, :first_order]
        reference_steady_state = [s ∈ 𝓂.exo_present ? 0 : NSSS[s] for s in full_SS]
    end

    initial_state = initial_state == [0.0] ? zeros(𝓂.timings.nVars) : initial_state[indexin(full_SS, sort(union(𝓂.var,𝓂.exo_present)))] - reference_steady_state
    
    shocks = 𝓂.timings.nExo == 0 ? :none : shocks

    shock_idx = parse_shocks_input_to_index(shocks,𝓂.timings)

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

    if generalised_irf
        Y = girf(state_update, 𝓂.timings; periods = periods, shocks = shocks, variables = variables, negative_shock = negative_shock)#, warmup_periods::Int = 100, draws::Int = 50, iterations_to_steady_state::Int = 500)
    else
        Y = irf(state_update, initial_state, 𝓂.timings; periods = periods, shocks = shocks, variables = variables, negative_shock = negative_shock)
    end

    # fontt = "computer modern"#"serif-roman"#
    # fontt = "times roman"#"serif-roman"#
    # fontt = "symbol"#"serif-roman"#

    # plots = []
    default(size=(700,500),
            # leg = false,
            # plot_titlefont = (10, fontt), 
            # titlefont = (10, fontt), 
            # guidefont = (8, fontt), 
            plot_titlefont = (10), 
            titlefont = (10), 
            guidefont = (8), 
            legendfontsize = 8, 
            # tickfont = (8, fontt),
            # tickfontfamily = fontt,
            tickfontsize = 8,
            # tickfontrotation = 9,
            # rotation = 90,
            # tickfontvalign = :center,
            # topmargin = 10mm,
            # rightmargin = 17mm, 
            framestyle = :box)


    shock_dir = negative_shock ? "Shock⁻" : "Shock⁺"

    if shocks == :none
        shock_dir = ""
    end
    if shocks == :simulate
        shock_dir = "Shocks"
    end

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
                                    Plots.plot!(twinx(),1:periods, 100*((Y[i,:,shock] .+ SS) ./ SS .- 1), ylabel = L"\% \Delta", label = "")
                                    hline!([SS 0], color = :black, label = "")                               
                        end)
                    else
                        push!(pp,begin
                                    Plots.plot(1:periods, Y[i,:,shock] .+ SS, title = string(𝓂.timings.var[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                    hline!([SS], color = :black, label = "")
                        end)

                    end
                else

                    plot_count = 1
                    if all((Y[i,:,shock] .+ SS) .> eps(Float32)) & (SS > eps(Float32))
                        push!(pp,begin
                                    Plots.plot(1:periods, Y[i,:,shock] .+ SS,title = string(𝓂.timings.var[var_idx[i]]),ylabel = "Level",label = "")
                                    Plots.plot!(twinx(),1:periods, 100*((Y[i,:,shock] .+ SS) ./ SS .- 1), ylabel = L"\% \Delta", label = "")
                                    hline!([SS 0],color = :black,label = "")                               
                        end)
                    else
                        push!(pp,begin
                                    Plots.plot(1:periods, Y[i,:,shock] .+ SS, title = string(𝓂.timings.var[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                    hline!([SS], color = :black, label = "")
                        end)

                    end

                    shock_string = ": " * string(𝓂.timings.exo[shock_idx[shock]])

                    if shocks == :simulate
                        shock_string = ": simulate all"
                        shock_name = "simulation"
                    elseif shocks == :none
                        shock_string = ""
                        shock_name = "no_shock"
                    else
                        shock_name = string(𝓂.timings.exo[shock_idx[shock]])
                    end

                    p = Plots.plot(pp...,plot_title = "Model: "*𝓂.model_name*"        " * shock_dir *  shock_string *"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

                    # p[:plot_title] = String(𝓂.timings.exo[shock])

                    # end


                    if show_plots# & (length(pp) > 0)
                        display(p)
                    end

                    if save_plots# & (length(pp) > 0)
                        savefig(p, save_plots_path * "/irf__" * 𝓂.model_name * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
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
            else
                shock_string = ": " * string(𝓂.timings.exo[shock_idx[shock]])
                shock_name = string(𝓂.timings.exo[shock_idx[shock]])
            end

            p = Plots.plot(pp...,plot_title = "Model: "*𝓂.model_name*"        " * shock_dir *  shock_string*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

            if show_plots# & (length(pp) > 0)
                #println(length(pp))
                display(p)
            end

            if save_plots# & (length(pp) > 0)
                # savefig(p,"irf__"*string(𝓂.timings.exo[shock_idx[shock]])*"__"*string(pane)*".pdf")
                savefig(p, save_plots_path * "/irf__" * 𝓂.model_name * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
            end
        end
    end
end



"""
See [`plot`](@ref)
"""
plot_irf = plot

"""
See [`plot`](@ref)
"""
plot_IRF = plot


"""
See [`plot`](@ref)
"""
plot_irfs = plot


"""
Wrapper for [`plot`](@ref) with `shocks = :simulate` and `periods = 100`.
"""
plot_simulations(args...; kwargs...) =  plot(args...; kwargs..., shocks = :simulate, periods = 100)






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
    verbose = false)

    fevds = get_conditional_variance_decomposition(𝓂,
                                                    periods = 1:periods,
                                                    parameters = parameters,
                                                    verbose = verbose)

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

    default(size=(700,500),
            plot_titlefont = (10), 
            titlefont = (10), 
            guidefont = (8), 
            legendfontsize = 8, 
            tickfontsize = 8,
            framestyle = :box)

    vars_to_plot = intersect(axiskeys(fevds)[1],𝓂.timings.var[var_idx])
    
    shocks_to_plot = axiskeys(fevds)[2]

    n_subplots = length(var_idx)
    pp = []
    pane = 1
    plot_count = 1

    for k in vars_to_plot
        if !(plot_count % plots_per_page == 0)
            plot_count += 1
            push!(pp,groupedbar(fevds(k,:,:)', title = string(k), bar_position = :stack, legend = :none))
        else
            plot_count = 1

            push!(pp,groupedbar(fevds(k,:,:)', title = string(k), bar_position = :stack, legend = :none))
            
            ppp = Plots.plot(pp...)

            p = Plots.plot(ppp,Plots.bar(fill(0,1,length(shocks_to_plot)), 
                                        label = reshape(string.(shocks_to_plot),1,length(shocks_to_plot)), 
                                        linewidth = 0 , 
                                        framestyle = :none, 
                                        legend = :inside, 
                                        legend_columns = -1), 
                                        layout = grid(2, 1, heights=[0.99, 0.01]),
                                        plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

            if show_plots
                display(p)
            end

            if save_plots
                savefig(p, save_plots_path * "/fevd__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
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
                                    layout = grid(2, 1, heights=[0.99, 0.01]),
                                    plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

        if show_plots
            display(p)
        end

        if save_plots
            savefig(p, save_plots_path * "/fevd__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
        end
    end
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
    verbose = false)

    @assert state ∈ 𝓂.timings.past_not_future_and_mixed "Invalid state. Choose one from:"*reduce(*," ".*string.(𝓂.timings.past_not_future_and_mixed))

    @assert length(setdiff(algorithm isa Symbol ? [algorithm] : algorithm, [:third_order, :second_order, :first_order])) == 0 "Invalid algorithm. Choose any combination of: :third_order, :second_order, :first_order"

    if algorithm isa Symbol
        max_algorithm = algorithm
        algorithm = [algorithm]
    else
        if :third_order ∈ algorithm 
            max_algorithm = :third_order 
        elseif :second_order ∈ algorithm 
            max_algorithm = :second_order 
        else 
            max_algorithm = :first_order 
        end
    end

    solve!(𝓂, verbose = verbose, algorithm = max_algorithm, dynamics = true)

    SS_and_std = get_moments(𝓂, 
                            derivatives = false,
                            parameters = parameters,
                            verbose = verbose)


    full_NSSS = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))
    full_NSSS[indexin(𝓂.aux,full_NSSS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    full_SS = [s ∈ 𝓂.exo_present ? 0 : SS_and_std[1](s) for s in full_NSSS]

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

    default(size=(700,500),
            plot_titlefont = (10), 
            titlefont = (10), 
            guidefont = (8), 
            legendfontsize = 8, 
            tickfontsize = 8,
            framestyle = :box)

    vars_to_plot = intersect(axiskeys(SS_and_std[1])[1],𝓂.timings.var[var_idx])

    state_range = collect(range(-SS_and_std[2](state), SS_and_std[2](state), 100)) * σ

    state_selector = state .== 𝓂.timings.var

    n_subplots = length(var_idx)
    pp = []
    pane = 1
    plot_count = 1

    for k in vars_to_plot

        kk = Symbol(replace(string(k), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))

        if !(plot_count % plots_per_page == 0)
            plot_count += 1
            
            if :first_order ∈ algorithm
                variable_first = [𝓂.solution.perturbation.first_order.state_update(state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

                variable_first = [(abs(x) > eps() ? x : 0.0) + SS_and_std[1](kk) for x in variable_first]
            end

            if :second_order ∈ algorithm
                SSS = 𝓂.solution.perturbation.second_order.stochastic_steady_state

                variable_second = [𝓂.solution.perturbation.second_order.state_update(SSS - full_SS .+ state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

                variable_second = [(abs(x) > eps() ? x : 0.0) + SS_and_std[1](kk) for x in variable_second]
            end

            if :third_order ∈ algorithm
                SSS = 𝓂.solution.perturbation.third_order.stochastic_steady_state

                variable_third = [𝓂.solution.perturbation.third_order.state_update(SSS - full_SS .+ state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

                variable_third = [(abs(x) > eps() ? x : 0.0) + SS_and_std[1](kk) for x in variable_third]
            end

            push!(pp,begin 
                        if :third_order ∈ algorithm 
                            Pl = Plots.plot(state_range .+ SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1], 
                                variable_third, 
                                ylabel = string(k)*"₍₀₎", 
                                xlabel = string(state)*"₍₋₁₎",
                                label = "3rd order perturbation")
                        end
                        if :second_order ∈ algorithm
                            if :second_order == max_algorithm 
                                Pl = Plots.plot(state_range .+ SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1], 
                                    variable_second, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "2nd order perturbation")
                            else
                                Plots.plot!(state_range .+ SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1], 
                                    variable_second, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "2nd order perturbation")
                            end
                        end
                        if :first_order ∈ algorithm
                            if :first_order  == max_algorithm 
                                Pl = Plots.plot(state_range .+ SS_and_std[1](state), 
                                    variable_first, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "1st order perturbation")

                                Plots.scatter!([SS_and_std[1](state)], [SS_and_std[1](kk)], label = "Non Stochastic Steady State")
                            else
                                Plots.plot!(state_range .+ SS_and_std[1](state), 
                                    variable_first, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "1st order perturbation")

                                Plots.scatter!([SS_and_std[1](state)], [SS_and_std[1](kk)], label = "Non Stochastic Steady State")
                            end
                        end

                        if :second_order ∈ algorithm || :third_order ∈ algorithm
                            Plots.scatter!([SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1]], [SSS[indexin([k],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1]], label = "Stochastic Steady State")
                        end

                        Pl
                    end)
        else
            plot_count = 1

            if :first_order ∈ algorithm
                variable_first = [𝓂.solution.perturbation.first_order.state_update(state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

                variable_first = [(abs(x) > eps() ? x : 0.0) + SS_and_std[1](kk) for x in variable_first]
            end

            if :second_order ∈ algorithm
                SSS = 𝓂.solution.perturbation.second_order.stochastic_steady_state

                variable_second = [𝓂.solution.perturbation.second_order.state_update(SSS - full_SS .+ state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

                variable_second = [(abs(x) > eps() ? x : 0.0) + SS_and_std[1](kk) for x in variable_second]
            end

            if :third_order ∈ algorithm
                SSS = 𝓂.solution.perturbation.third_order.stochastic_steady_state

                variable_third = [𝓂.solution.perturbation.third_order.state_update(SSS - full_SS .+ state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

                variable_third = [(abs(x) > eps() ? x : 0.0) + SS_and_std[1](kk) for x in variable_third]
            end

            push!(pp,begin 
                        if :third_order ∈ algorithm 
                            Pl = Plots.plot(state_range .+ SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1], 
                                variable_third, 
                                ylabel = string(k)*"₍₀₎", 
                                xlabel = string(state)*"₍₋₁₎",
                                label = "3rd order perturbation")
                        end
                        if :second_order ∈ algorithm
                            if :second_order == max_algorithm 
                                Pl = Plots.plot(state_range .+ SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1], 
                                    variable_second, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "2nd order perturbation")
                            else
                                Plots.plot!(state_range .+ SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1], 
                                    variable_second, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "2nd order perturbation")
                            end
                        end
                        if :first_order ∈ algorithm
                            if :first_order  == max_algorithm 
                                Pl = Plots.plot(state_range .+ SS_and_std[1](state), 
                                    variable_first, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "1st order perturbation")

                                Plots.scatter!([SS_and_std[1](state)], [SS_and_std[1](kk)], label = "Non Stochastic Steady State")
                            else
                                Plots.plot!(state_range .+ SS_and_std[1](state), 
                                    variable_first, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "1st order perturbation")

                                Plots.scatter!([SS_and_std[1](state)], [SS_and_std[1](kk)], label = "Non Stochastic Steady State")
                            end
                        end

                        if :second_order ∈ algorithm || :third_order ∈ algorithm
                            Plots.scatter!([SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1]], [SSS[indexin([k],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1]], label = "Stochastic Steady State")
                        end

                        Pl
                    end)

            p = Plots.plot(pp..., plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

            if show_plots
                display(p)
            end

            if save_plots
                savefig(p, save_plots_path * "/solution__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
            end

            pane += 1
            pp = []
        end
    end

    if length(pp) > 0
        p = Plots.plot(pp..., plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

        if show_plots
            display(p)
        end

        if save_plots
            savefig(p, save_plots_path * "/solution__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
        end
    end
end

