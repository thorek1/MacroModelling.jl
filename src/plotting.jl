
using Plots, Plots.PlotMeasures, LaTeXStrings



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

    full_NSSS = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))
    full_NSSS[indexin(𝓂.aux,full_NSSS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    SS = [NSSS[s] for s in full_NSSS]

    if algorithm == :second_order
        reference_steady_state = 𝓂.solution.perturbation.second_order.stochastic_steady_state
    elseif algorithm == :third_order
        reference_steady_state = 𝓂.solution.perturbation.third_order.stochastic_steady_state
    elseif algorithm ∈ [:linear_time_iteration, :riccati, :first_order]
        reference_steady_state = SS
    end

    initial_state = initial_state == [0.0] ? zeros(𝓂.timings.nVars) : initial_state - SS
    
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
