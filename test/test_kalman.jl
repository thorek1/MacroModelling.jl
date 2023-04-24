using MacroModelling
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import LinearAlgebra as ℒ
import RecursiveFactorization as RF


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


simulation = simulate(RBC_CME, parameters= :std_z_delta=>.05)


data = simulation([:k],:,:simulate)

using StatsPlots

get_estimated_variables(RBC_CME,data,data_in_levels = false)


plot_shock_decomposition(RBC_CME,data,data_in_levels = false, transparency = .5)

𝓂 = RBC_CME
verbose = true
parameters = 𝓂.parameter_values
data_in_levels = false
variables = :all
shocks = :all
observables = collect(axiskeys(data,1))


write_parameters_input!(𝓂, parameters, verbose = verbose)

solve!(𝓂, verbose = verbose, dynamics = true)

reference_steady_state, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())

obs_idx     = parse_variables_input_to_index(collect(axiskeys(data)[1]), 𝓂.timings)
var_idx     = parse_variables_input_to_index(variables, 𝓂.timings) 
shock_idx   = parse_shocks_input_to_index(shocks,𝓂.timings)

if data_in_levels
    data_in_deviations = data .- reference_steady_state[obs_idx]
else
    data_in_deviations = data
end

filtered_and_smoothed = filter_and_smooth(𝓂, data_in_deviations, sort(axiskeys(data)[1]); verbose = verbose)


sort!(observables)

solve!(𝓂, verbose = verbose)

parameters = 𝓂.parameter_values

SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, verbose)

# @assert solution_error < tol "Could not solve non stochastic steady state." 

∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)

sol = calculate_first_order_solution(∇₁; T = 𝓂.timings)

A = @views sol[:,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]

B = @views sol[:,𝓂.timings.nPast_not_future_and_mixed+1:end]

C = @views ℒ.diagm(ones(𝓂.timings.nVars))[sort(indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))),:]

𝐁 = B * B'

P̄ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)[1]

n_obs = size(data_in_deviations,2)

v = zeros(size(C,1), n_obs)
μ = zeros(size(A,1), n_obs+1) # filtered_states
P = zeros(size(A,1), size(A,1), n_obs+1) # filtered_covariances
σ = zeros(size(A,1), n_obs) # filtered_standard_deviations
iF= zeros(size(C,1), size(C,1), n_obs)
L = zeros(size(A,1), size(A,1), n_obs)
ϵ = zeros(size(B,1), n_obs) # filtered_shocks

P[:, :, 1] = P̄

# Kalman Filter
for t in axes(data_in_deviations,2)
    v[:, t]     .= data_in_deviations[:, t] - C * μ[:, t]
    iF[:, :, t] .= inv(C * P[:, :, t] * C')
    PCiF         = P[:, :, t] * C' * iF[:, :, t]
    L[:, :, t]  .= A - A * PCiF * C
    P[:, :, t+1].= A * P[:, :, t] * L[:, :, t]' + 𝐁
    σ[:, t] = sqrt.(ℒ.diag(P[:, :, t+1]))
    μ[:, t+1]   .= A * (μ[:, t] + PCiF * v[:, t])
    ϵ[:, t]     .= B' * C' * iF[:, :, t] * v[:, t]
end







plot_model_estimates(RBC_CME,simulation([:k],:,:simulate), data_in_levels = false)





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

simulation = simulate(RBC)

get_estimated_variable_standard_deviations(RBC,simulation([:c],:,:simulate))

get_estimated_shocks(RBC,simulation([:c],:,:simulate))
get_shock_decomposition(RBC,simulation([:c],:,:simulate))

get_estimated_variables(RBC,simulation([:c],:,:simulate))



include("models/FS2000.jl")

FS2000 = m
get_SS(FS2000)
get_covariance(FS2000)
# load data
dat = CSV.read("test/data/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)
axiskeys(data)[1]
# declare observables
observables = sort(Symbol.("log_".*names(dat)))

# subset observables in data
data = data(observables,:)


import StatsPlots

plot_model_estimates(FS2000, data)
plot_shock_decomposition(FS2000, data)
plot_shock_decomposition(FS2000, data, parameters = [0.403475267025427,0.990923010561409,0.004566214169879,1.014318555099325,0.845538800525148,0.689060025764850,0.001665380385476,0.013570417835562,0.003274145891950])

out = get_shock_decomposition(FS2000, data; data_in_levels = true, parameters = [0.403475267025427,0.990923010561409,0.004566214169879,1.014318555099325,0.845538800525148,0.689060025764850,0.001665380385476,0.013570417835562,0.003274145891950])

out2 = get_shock_decomposition(FS2000, data; data_in_levels = true)
out2 = get_shock_decomposition(FS2000, data; data_in_levels = true, smooth = false)

out3 = get_estimated_shocks(FS2000, data; data_in_levels = true)
out3 = get_estimated_shocks(FS2000, data; data_in_levels = true, smooth = false)

out3 = get_estimated_variables(FS2000, data; data_in_levels = true)
out3 = get_estimated_variables(FS2000, data; data_in_levels = true, smooth = false)


import StatsPlots
using LaTeXStrings

data_in_levels = true
𝓂 = FS2000
verbose = true
parameters = 𝓂.parameter_values
shocks = :all
variables = :all
plots_per_page = 9
save_plots = false
show_plots = true
shock_decomposition = true
# plot_model_estimates()

gr_back = StatsPlots.backend() == StatsPlots.Plots.GRBackend()

StatsPlots.default(size=(700,500),
plot_titlefont = 10, 
titlefont = 10, 
guidefont = 8, 
legendfontsize = 8, 
tickfontsize = 8,
framestyle = :box)

write_parameters_input!(𝓂, parameters, verbose = verbose)

solve!(𝓂, verbose = verbose, dynamics = true)

reference_steady_state, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())

obs_idx = parse_variables_input_to_index(collect(axiskeys(data)[1]), 𝓂.timings)

if data_in_levels
    data_in_deviations = data .- reference_steady_state[obs_idx]
else
    data_in_deviations = data
end

filtered_and_smoothed = filter_and_smooth(𝓂, data_in_deviations, sort(axiskeys(data)[1]); verbose = verbose)

smoothed_variables = filtered_and_smoothed[1]
smoothed_shocks = filtered_and_smoothed[3]
decomp = filtered_and_smoothed[4]


periods = size(smoothed_variables,2)
shock_idx = parse_shocks_input_to_index(shocks,𝓂.timings)


var_idx = parse_variables_input_to_index(variables, 𝓂.timings)


transparency = .4

return_plots = []

n_subplots = length(var_idx) + length(shock_idx)
pp = []
pane = 1
plot_count = 1

for i in 1:length(var_idx) + length(shock_idx)
    if i > length(var_idx)
        push!(pp,begin
                StatsPlots.plot()
                StatsPlots.plot!(smoothed_shocks[shock_idx[i - length(var_idx)],:],
                    title = string(𝓂.timings.exo[shock_idx[i - length(var_idx)]]) * "₍ₓ₎", 
                    ylabel = shock_decomposition ? "Absolute Δ" : "Level",label = "", 
                    color = shock_decomposition ? :black : :auto)
                StatsPlots.hline!([0],
                    color = :black,
                    label = "")                               
        end)
    else
        SS = reference_steady_state[var_idx[i]]

        if shock_decomposition SS = zero(SS) end

        can_double_axis = gr_back &&  all((smoothed_variables[var_idx[i],:] .+ SS) .> eps(Float32)) && (SS > eps(Float32)) && !shock_decomposition
        
        push!(pp,begin
                StatsPlots.plot()
                if shock_decomposition
                    StatsPlots.groupedbar!(decomp[var_idx[i],[end-1,shock_idx...],:]', 
                        bar_position = :stack, 
                        lw = 0,
                        legend = :none, 
                        alpha = transparency)
                end
                StatsPlots.plot!(smoothed_variables[var_idx[i],:] .+ SS,
                    title = string(𝓂.timings.var[var_idx[i]]), 
                    ylabel = shock_decomposition ? "Absolute Δ" : "Level",label = "", 
                    color = shock_decomposition ? :black : :auto)
                if var_idx[i] ∈ obs_idx 
                    StatsPlots.plot!(data_in_deviations[indexin([var_idx[i]],obs_idx),:]' .+ SS,
                        title = string(𝓂.timings.var[var_idx[i]]),
                        ylabel = shock_decomposition ? "Absolute Δ" : "Level", 
                        label = "", 
                        color = shock_decomposition ? :darkred : :auto) 
                end
                if can_double_axis 
                    StatsPlots.plot!(StatsPlots.twinx(),
                        100*((smoothed_variables[var_idx[i],:] .+ SS) ./ SS .- 1), 
                        ylabel = LaTeXStrings.L"\% \Delta", 
                        label = "") 
                    if var_idx[i] ∈ obs_idx 
                        StatsPlots.plot!(StatsPlots.twinx(),
                            100*((data_in_deviations[indexin([var_idx[i]],obs_idx),:]' .+ SS) ./ SS .- 1), 
                            ylabel = LaTeXStrings.L"\% \Delta", 
                            label = "") 
                    end
                end
                StatsPlots.hline!(can_double_axis ? [SS 0] : [SS],
                    color = :black,
                    label = "")                               
        end)
    end

    if !(plot_count % plots_per_page == 0)
        plot_count += 1
    else
        plot_count = 1

        ppp = StatsPlots.plot(pp...)

        p = StatsPlots.plot(ppp,begin
                                    StatsPlots.plot(framestyle = :none)
                                    if shock_decomposition
                                        StatsPlots.bar!(fill(0,1,size(decomp,2)-1), 
                                                                label = reshape(vcat("Initial value",string.(𝓂.exo[shock_idx])),1,size(decomp,2)-1), 
                                                                linewidth = 0,
                                                                bar_position = :stack,
                                                                alpha = transparency,
                                                                lw = 0,
                                                                legend = :inside, 
                                                                legend_columns = -1)
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
                                layout = StatsPlots.grid(2, 1, heights=[0.99, 0.01]),
            plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

        push!(return_plots,p)

        if show_plots# & (length(pp) > 0)
            display(p)
        end

        if save_plots# & (length(pp) > 0)
            StatsPlots.savefig(p, save_plots_path * "/estimation__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
        end

        pane += 1
        pp = []
    end
end

if length(pp) > 0
    ppp = StatsPlots.plot(pp...)


    p = StatsPlots.plot(ppp,begin
                                StatsPlots.plot(framestyle = :none)
                                if shock_decomposition
                                    StatsPlots.bar!(fill(0,1,size(decomp,2)-1), 
                                                            label = reshape(vcat("Initial value",string.(𝓂.exo[shock_idx])),1,size(decomp,2)-1), 
                                                            linewidth = 0,
                                                            bar_position = :stack,
                                                            alpha = transparency,
                                                            lw = 0,
                                                            legend = :inside, 
                                                            legend_columns = -1)
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
                            layout = StatsPlots.grid(2, 1, heights=[0.99, 0.01]),
        plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

    push!(return_plots,p)

    if show_plots
        display(p)
    end

    if save_plots
        StatsPlots.savefig(p, save_plots_path * "/estimation__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
    end
end





















function plot_model_estimates(𝓂::ℳ,
    data::AbstractArray{Float64};
    parameters = nothing,
    variables::Symbol_input = :all_including_auxilliary, 
    shocks::Symbol_input = :all, 
    data_in_levels::Bool = true,
    shock_decomposition = true,
    show_plots::Bool = true,
    save_plots::Bool = false,
    save_plots_format::Symbol = :pdf,
    save_plots_path::String = ".",
    plots_per_page::Int = 9,
    verbose::Bool = false)

    gr_backend = StatsPlots.backend() == StatsPlots.Plots.GRBackend()

    StatsPlots.default(size=(700,500),
                    plot_titlefont = 10, 
                    titlefont = 10, 
                    guidefont = 8, 
                    legendfontsize = 8, 
                    tickfontsize = 8,
                    framestyle = :box)

    write_parameters_input!(𝓂, parameters, verbose = verbose)

    solve!(𝓂, verbose = verbose, dynamics = true)

    reference_steady_state, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())

    obs_idx     = parse_variables_input_to_index(collect(axiskeys(data)[1]), 𝓂.timings)
    var_idx     = parse_variables_input_to_index(variables, 𝓂.timings) 
    shock_idx   = parse_shocks_input_to_index(shocks,𝓂.timings)

    if data_in_levels
        data_in_deviations = data .- reference_steady_state[obs_idx]
    else
        data_in_deviations = data
    end

    filtered_and_smoothed = filter_and_smooth(𝓂, data_in_deviations, sort(axiskeys(data)[1]); verbose = verbose)

    smoothed_variables  = filtered_and_smoothed[1]
    smoothed_shocks     = filtered_and_smoothed[3]
    decomp              = filtered_and_smoothed[4]

    periods = size(smoothed_variables,2)

    transparency = .4

    return_plots = []

    n_subplots = length(var_idx) + length(shock_idx)
    pp = []
    pane = 1
    plot_count = 1

    for i in 1:length(var_idx) + length(shock_idx)
        if i > length(var_idx)
            push!(pp,begin
                    StatsPlots.plot()
                    StatsPlots.plot!(smoothed_shocks[shock_idx[i - length(var_idx)],:],
                        title = string(𝓂.timings.exo[shock_idx[i - length(var_idx)]]) * "₍ₓ₎", 
                        ylabel = shock_decomposition ? "Absolute Δ" : "Level",label = "", 
                        color = shock_decomposition ? :black : :auto)
                    StatsPlots.hline!([0],
                        color = :black,
                        label = "")                               
            end)
        else
            SS = reference_steady_state[var_idx[i]]

            if shock_decomposition SS = zero(SS) end

            can_double_axis = gr_back &&  all((smoothed_variables[var_idx[i],:] .+ SS) .> eps(Float32)) && (SS > eps(Float32)) && !shock_decomposition
            
            push!(pp,begin
                    StatsPlots.plot()
                    if shock_decomposition
                        StatsPlots.groupedbar!(decomp[var_idx[i],[end-1,shock_idx...],:]', 
                            bar_position = :stack, 
                            lw = 0,
                            legend = :none, 
                            alpha = transparency)
                    end
                    StatsPlots.plot!(smoothed_variables[var_idx[i],:] .+ SS,
                        title = string(𝓂.timings.var[var_idx[i]]), 
                        ylabel = shock_decomposition ? "Absolute Δ" : "Level",label = "", 
                        color = shock_decomposition ? :black : :auto)
                    if var_idx[i] ∈ obs_idx 
                        StatsPlots.plot!(data_in_deviations[indexin([var_idx[i]],obs_idx),:]' .+ SS,
                            title = string(𝓂.timings.var[var_idx[i]]),
                            ylabel = shock_decomposition ? "Absolute Δ" : "Level", 
                            label = "", 
                            color = shock_decomposition ? :darkred : :auto) 
                    end
                    if can_double_axis 
                        StatsPlots.plot!(StatsPlots.twinx(),
                            100*((smoothed_variables[var_idx[i],:] .+ SS) ./ SS .- 1), 
                            ylabel = LaTeXStrings.L"\% \Delta", 
                            label = "") 
                        if var_idx[i] ∈ obs_idx 
                            StatsPlots.plot!(StatsPlots.twinx(),
                                100*((data_in_deviations[indexin([var_idx[i]],obs_idx),:]' .+ SS) ./ SS .- 1), 
                                ylabel = LaTeXStrings.L"\% \Delta", 
                                label = "") 
                        end
                    end
                    StatsPlots.hline!(can_double_axis ? [SS 0] : [SS],
                        color = :black,
                        label = "")                               
            end)
        end

        if !(plot_count % plots_per_page == 0)
            plot_count += 1
        else
            plot_count = 1

            ppp = StatsPlots.plot(pp...)

            p = StatsPlots.plot(ppp,begin
                                        StatsPlots.plot(framestyle = :none)
                                        if shock_decomposition
                                            StatsPlots.bar!(fill(0,1,size(decomp,2)-1), 
                                                                    label = reshape(vcat("Initial value",string.(𝓂.exo[shock_idx])),1,size(decomp,2)-1), 
                                                                    linewidth = 0,
                                                                    bar_position = :stack,
                                                                    alpha = transparency,
                                                                    lw = 0,
                                                                    legend = :inside, 
                                                                    legend_columns = -1)
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
                                    layout = StatsPlots.grid(2, 1, heights=[0.99, 0.01]),
                plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

            push!(return_plots,p)

            if show_plots# & (length(pp) > 0)
                display(p)
            end

            if save_plots# & (length(pp) > 0)
                StatsPlots.savefig(p, save_plots_path * "/estimation__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
            end

            pane += 1
            pp = []
        end
    end

    if length(pp) > 0
        ppp = StatsPlots.plot(pp...)


        p = StatsPlots.plot(ppp,begin
                                    StatsPlots.plot(framestyle = :none)
                                    if shock_decomposition
                                        StatsPlots.bar!(fill(0,1,size(decomp,2)-1), 
                                                                label = reshape(vcat("Initial value",string.(𝓂.exo[shock_idx])),1,size(decomp,2)-1), 
                                                                linewidth = 0,
                                                                bar_position = :stack,
                                                                alpha = transparency,
                                                                lw = 0,
                                                                legend = :inside, 
                                                                legend_columns = -1)
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
                                layout = StatsPlots.grid(2, 1, heights=[0.99, 0.01]),
            plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

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
















out(:log_gy_obs,:,:)
out(:,:,2)
out(:k,:Initial_values,:)


verbose = true
𝓂 = FS2000
parameters = [0.403475267025427,0.990923010561409,0.004566214169879,1.014318555099325,0.845538800525148,0.689060025764850,0.001665380385476,0.013570417835562,0.003274145891950]

write_parameters_input!(𝓂, parameters, verbose = verbose)

solve!(𝓂, verbose = verbose, dynamics = true)

reference_steady_state, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())

obs_idx = parse_variables_input_to_index(collect(axiskeys(data)[1]), 𝓂.timings)

if data_in_levels
    data .-= reference_steady_state[obs_idx]
end

filtered_and_smoothed = filter_and_smooth(𝓂, data, collect(axiskeys(data)[1]); verbose = verbose)

return KeyedArray(filtered_and_smoothed[3];  Shocks = map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.timings.exo), Periods = 1:size(data,2))


# get_SS(FS2000, parameters = [0.4027212142373724
# 0.9909438997461472
# 0.00455007831270222
# 1.014322728752977
# 0.8457081193818059
# 0.6910339118126667
# 0.0016353140797331237
# 0.013479922353054475
# 0.003257545969294338])

get_SS(FS2000,parameters = [0.403475267025427,0.990923010561409,0.004566214169879,1.014318555099325,0.845538800525148,0.689060025764850,0.001665380385476,0.013570417835562,0.003274145891950])


out = filter_and_smooth(FS2000, data(observables), observables)
out[3]

sqrt.(ℒ.diag(out[3][:,:,192]))

calculate_kalman_filter_loglikelihood(m, data(observables), observables)


𝓂 = FS2000
verbose = true
tol = eps()


sort!(observables)

solve!(𝓂, verbose = verbose)

parameters = [0.403475267025427,0.990923010561409,0.004566214169879,1.014318555099325,0.845538800525148,0.689060025764850,0.001665380385476,0.013570417835562,0.003274145891950]

SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, verbose)

if solution_error > tol || isnan(solution_error)
    return -Inf
end

NSSS_labels = [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]

obs_indices = indexin(observables,NSSS_labels)

data_in_deviations = collect(data(observables)) .- SS_and_pars[obs_indices]

∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)

sol = calculate_first_order_solution(∇₁; T = 𝓂.timings)

observables_and_states = sort(union(𝓂.timings.past_not_future_and_mixed_idx,indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))))

A = @views sol[observables_and_states,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(observables_and_states)))[(indexin(𝓂.timings.past_not_future_and_mixed_idx,observables_and_states)),:]
B = @views sol[observables_and_states,𝓂.timings.nPast_not_future_and_mixed+1:end]

C = @views ℒ.diagm(ones(length(observables_and_states)))[(indexin(sort(indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))),observables_and_states)),:]

𝐁 = B * B'

# Gaussian Prior

calculate_covariance_ = calculate_covariance_AD(sol, T = 𝓂.timings, subset_indices = Int64[observables_and_states...])

Pstar1 = calculate_covariance_(sol)
# P = reshape((ℒ.I - ℒ.kron(A, A)) \ reshape(𝐁, prod(size(A)), 1), size(A))
u = zeros(length(observables_and_states))
# u = SS_and_pars[sort(union(𝓂.timings.past_not_future_and_mixed,observables))] |> collect
z = C * u










n_obs = size(data_in_deviations,2)

nk = 1
d = 0
decomp = []
# spinf = size(Pinf1)
spstar = size(Pstar1)
v = zeros(size(C,1), size(data_in_deviations,2))
u = zeros(size(A,1), size(data_in_deviations,2)+1)
# û = zeros(size(A,1), size(data_in_deviations,2))
# uK = zeros(nk, size(A,1), size(data_in_deviations,2)+nk)
# PK = zeros(nk, size(A,1), size(A,1), size(data_in_deviations,2)+nk)
iF = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# Fstar = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# iFstar = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# iFinf = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# K = zeros(size(A,1), size(C,1), size(data_in_deviations,2))
L = zeros(size(A,1), size(A,1), size(data_in_deviations,2))
# Linf = zeros(size(A,1), size(A,1), size(data_in_deviations,2))
# Lstar = zeros(size(A,1), size(A,1), size(data_in_deviations,2))
# Kstar = zeros(size(A,1), size(C,1), size(data_in_deviations,2))
# Kinf = zeros(size(A,1), size(C,1), size(data_in_deviations,2))
P = zeros(size(A,1), size(A,1), size(data_in_deviations,2)+1)
# Pstar = zeros(spstar[1], spstar[2], size(data_in_deviations,2)+1)
# Pstar[:, :, 1] = Pstar1
# Pinf = zeros(spinf[1], spinf[2], size(data_in_deviations,2)+1)
# Pinf[:, :, 1] = Pinf1
rr = size(C,1)
# 𝐁 = R * Q * transpose(R)
ū = zeros(size(A,1), size(data_in_deviations,2))
ϵ̄ = zeros(rr, size(data_in_deviations,2))
ϵ = zeros(rr, size(data_in_deviations,2))
# epsilonhat = zeros(rr, size(data_in_deviations,2))
r = zeros(size(A,1))
# Finf_singular = zeros(1, size(data_in_deviations,2))

V = []

# t = 0

# d = t
P[:, :, 1] = Pstar1

    # Kalman Filter
    for t in axes(data_in_deviations,2)
        v[:, t]     .= data_in_deviations[:, t] - C * u[:, t]
        iF[:, :, t] .= inv(C * P[:, :, t] * C')
        PCiF         = P[:, :, t] * C' * iF[:, :, t]
        L[:, :, t]  .= A - A * PCiF * C
        P[:, :, t+1].= A * P[:, :, t] * L[:, :, t]' + 𝐁
        u[:, t+1]   .= A * (u[:, t] + PCiF * v[:, t])
        ϵ[:, t]     .= B' * C' * iF[:, :, t] * v[:, t]
    end

    ū = zeros(size(A,1), n_obs) # smoothed_states
    ϵ̄ = zeros(size(C,1), n_obs) # smoothed_shocks

    r = zeros(size(A,1))

    # Kalman Smoother
    for t in n_obs:-1:1
        r       .= C' * iF[:, :, t] * v[:, t] + L[:, :, t]' * r
        ū[:, t] .= u[:, t] + P[:, :, t] * r
        ϵ̄[:, t] .= B' * r
    end


for t in 1:size(data_in_deviations,2)
    v[:, t] = data_in_deviations[:, t] - C * u[:, t]
    F = C * P[:, :, t] * C'
    iF[:, :, t] = inv(F)
    PCiF = P[:, :, t] * C' * iF[:, :, t]
    û = u[:, t] + PCiF * v[:, t]
    K = A * PCiF
    L[:, :, t] = A - K * C
    P[:, :, t+1] = A * P[:, :, t] * L[:, :, t]' + 𝐁
    u[:, t+1] = A * û
    ϵ[:, t] = B' * C' * iF[:, :, t] * v[:, t]
    # Pf = P[:, :, t]
    # uK[1, :, t+1] = u[:, t+1]
    # for jnk in 1:nk
    #     Pf = A * Pf * A' + 𝐁
    #     PK[jnk, :, :, t+jnk] = Pf
    #     if jnk > 1
    #         uK[jnk, :, t+jnk] = A * uK[jnk-1, :, t+jnk-1]
    #     end
    # end
end

# backward pass; r_T and N_T, stored in entry (size(data_in_deviations,2)+1) were initialized at 0
# t = size(data_in_deviations,2) + 1
for t in size(data_in_deviations,2):-1:1
    r = C' * iF[:, :, t] * v[:, t] + L[:, :, t]' * r # compute r_{t-1}, DK (2012), eq. 4.38
    ū[:, t] = u[:, t] + P[:, :, t] * r # DK (2012), eq. 4.35
    ϵ̄[:, t] = B' * r # DK (2012), eq. 4.63
end

# if decomp_flag
    decomp = zeros(nk, size(A,1), rr, size(data_in_deviations,2)+nk)
    CBs = C' * inv(C * 𝐁 * C') * C * B
    for t in 1:size(data_in_deviations,2)
        # : = data_index[t]
        # calculate eta_tm1t
        eta_tm1t = B' * C' * iF[:, :, t] * v[:, t]
        AAA = P[:, :, t] * CBs .* eta_tm1t'
        # calculate decomposition
        decomp[1, :, :, t+1] = AAA
        for h = 2:nk
            AAA = A * AAA
            decomp[h, :, :, t+h] = AAA
        end
    end
# end

epsilonhat = data_in_deviations - C * ū





# outt = kalman_filter_and_smoother(data_in_deviations,A,𝐁,C,u,P)

# C*outt[1]
# C*outt[3]
# data_in_deviations

using StatsPlots

plot(data_in_deviations[1,:])
plot!((C*ū)[1,:])
plot!((C*u)[1,2:end])


plot(data_in_deviations[2,:])
plot!((C*ū)[2,:])
plot!((C*u)[2,2:end])

plot(ϵ̄[1,:])
plot(ϵ̄[2,:])

ϵ̄

plot(ϵ[1,:])
plot(ϵ[2,:])

ϵ

decomposition = zeros(size(A,1),size(B,2)+2,size(data_in_deviations,2))
decomposition[:,end,:] = ū


decomposition[:,1:end-2,1] = B .* repeat(ϵ̄[:, 1]', size(A,1))
decomposition[:,end-1,1] = decomposition[:,end,1] - sum(decomposition[:,1:end-2,1],dims=2)

for i in 2:size(data_in_deviations,2)
    decomposition[:,1:end-2,i] = A * decomposition[:,1:end-2,i-1]
    decomposition[:,1:end-2,i] += B .* repeat(ϵ̄[:, i]', size(A,1))
    decomposition[:,end-1,i] = decomposition[:,end,i] - sum(decomposition[:,1:end-2,i],dims=2)
end


# Assuming your 4x192 array is named "data"
data = decomposition[2,:,:]
# sum(data[1:3, :],dims=1)' .- data[4, :]
# Split the data into the relevant components
bar_data = data[1:3, :]
line_data = data[4, :]



# Create the stacked bar plot
bar_plot = groupedbar(bar_data[[end,1:end-1...],:]', label=["Bar1" "Bar2" "Bar3"], xlabel="Time", ylabel="Value", alpha=0.5, title="Stacked Bars with Line Overlay", bar_position = :stack)

plot!(line_data, label="Line", linewidth=2, color=:black, linestyle=:solid, legend=:topright)



ϵ̄
ū











loglik = 0.0

B̂ = RF.lu(C * B , check = false)

@assert ℒ.issuccess(B̂) "Numerical stabiltiy issues for restrictions in period 1."

B̂inv = inv(B̂)

n_timesteps = size(data, 2)
n_states = length(u)
filtered_states = zeros(n_states, n_timesteps)
updated_states = zeros(n_states, n_timesteps)
smoothed_states = zeros(n_states, n_timesteps)
filtered_covariance = zeros(n_states, n_states, n_timesteps)
filtered_shocks = zeros(size(C,1), n_timesteps)
smoothed_shocks = zeros(n_states, n_timesteps)
P_smoothed = copy(P)

# Kalman filter
for t in 1:n_timesteps
    v = data_in_deviations[:, t] - C * u
    filtered_shocks[:, t] = B̂inv * v
    F = C * P * C'
    K = P * C' / F
    P = A * (P - K * C * P) * A' + 𝐁
    filtered_covariance[:, :, t] = P
    u = A * (u + K * v)
    filtered_states[:, t] = u
    updated_states[:,t] = filtered_states[:, t] + K * v
end

smoothed_states = copy(filtered_states)
smoothed_covariance = copy(filtered_covariance)

for t in n_timesteps-1:-1:1
    J = filtered_covariance[:,:, t] * A * filtered_covariance[:,:, t + 1]
    smoothed_states[:, t] = filtered_states[:, t+ 1] + J * (smoothed_states[:, t + 1] - filtered_states[:, t])
    smoothed_covariance[:,:, t] = filtered_covariance[:,:, t] + J * (smoothed_covariance[:,:, t + 1] - filtered_covariance[:,:, t + 1]) * J'
end

v = data_in_deviations[:,t] - z

F = C * P * C'

K = P * C' / F

P = A * (P - K * C * P) * A' + 𝐁

u = A * (u + K * v)

z = C * u 
# Kalman smoother for states
smoothed_states[:, end] = filtered_states[:, end]
for t in (n_timesteps - 1):-1:1
    P_future = A * P * A' + 𝐁
    J = P * A' / P_future
    smoothed_states[:, t] = filtered_states[:, t] + J * (smoothed_states[:, t + 1] - A * filtered_states[:, t])
end





# Kalman smoother for states
smoothed_states[:, end] = filtered_states[:, end]
smoothed_covariances[:, :, end] = filtered_covariances[:, :, end]
for t in (n_timesteps - 1):-1:1
    P_future = A * P * A' + 𝐁
    J = P * A' / P_future
    smoothed_states[:, t] = filtered_states[:, t] + J * (smoothed_states[:, t + 1] - A * filtered_states[:, t])
    P = filtered_covariances[:, :, t] - J * (P_future - smoothed_covariances[:, :, t + 1]) * J'
    smoothed_covariances[:, :, t] = P
end




for t in 1:size(data)[2]
    v = data_in_deviations[:,t] - z

    F = C * P * C'

    # F = (F + F') / 2

    # loglik += log(max(eps(),ℒ.det(F))) + v' * ℒ.pinv(F) * v
    # K = P * C' * ℒ.pinv(F)

    # loglik += log(max(eps(),ℒ.det(F))) + v' / F  * v
    Fdet = ℒ.det(F)

    if Fdet < eps() return -Inf end

    loglik += log(Fdet) + v' / F  * v
    
    K = P * C' / F

    P = A * (P - K * C * P) * A' + 𝐁

    u = A * (u + K * v)
    
    z = C * u 
end

return -(loglik + length(data) * log(2 * 3.141592653589793)) / 2 # otherwise conflicts with model parameters assignment




function kalman_filter_and_smoother(data, A, B, C, u, P, 𝐁)
    n_timesteps = size(data, 2)
    n_states = length(u)
    filtered_states = zeros(n_states, n_timesteps)
    smoothed_states = zeros(n_states, n_timesteps)
    filtered_shocks = zeros(n_states, n_timesteps)
    smoothed_shocks = zeros(n_states, n_timesteps)
    P_smoothed = copy(P)

    # Kalman filter
    for t in 1:n_timesteps
        v = data[:, t] - C * u
        F = C * P * C'
        K = P * C' / F
        filtered_shocks[:, t] = K * v
        P = A * (P - K * C * P) * A' + 𝐁
        u = A * (u + filtered_shocks[:, t])
        filtered_states[:, t] = u
    end

    # Kalman smoother for states
    smoothed_states[:, end] = filtered_states[:, end]
    for t in (n_timesteps - 1):-1:1
        P_future = A * P * A' + 𝐁
        J = P * A' / P_future
        smoothed_states[:, t] = filtered_states[:, t] + J * (smoothed_states[:, t + 1] - A * filtered_states[:, t])
    end

    # Kalman smoother for shocks
    smoothed_shocks[:, end] = filtered_shocks[:, end]
    for t in (n_timesteps - 1):-1:1
        P_future = A * P * A' + 𝐁
        J = P * A' / P_future
        smoothed_shocks[:, t] = filtered_shocks[:, t] + J * (smoothed_shocks[:, t + 1] - A * filtered_shocks[:, t])
    end

    return filtered_states, smoothed_states, filtered_shocks, smoothed_shocks
end







using Distributions
using LinearAlgebra

for t in 1:size(data)[2]
    v = data_in_deviations[:,t] - z

    F = C * P * C'
    
    K = P * C' / F

    P = A * (P - K * C * P) * A' + 𝐁

    u = A * (u + K * v)
    u = A * (u + K * (data_in_deviations[:,t] - z))
    
    z = C * u 
end
# Kalman filter implementation
function kalman_filter(data_in_deviations, A, 𝐁, C, u0, P0)
    T = size(data_in_deviations, 2)
    n = size(A, 1)
    
    û = zeros(n, T)
    P = Array{Matrix{Float64}}(undef, T)
    û[:, 1] = u0
    P[1] = P0

    # Update
    F = C * P0 * C'
    K = P0 * C' / F
    û[:, 1] = K * (data_in_deviations[:, 1])
    P[1] = A * (P0 - K * C * P0) * A' + 𝐁

    for t in 2:T
        # Predict
        û⁻ = A * û[:, t - 1]
    
        # Update
        F = C * P[t - 1] * C'
        K = P[t - 1] * C' / F
        û[:, t] = û⁻ + K * (data_in_deviations[:, t] - C * û⁻)
        P[t] = A * (P[t - 1] - K * C * P[t - 1]) * A' + 𝐁
    end

    return û, P
end

# Kalman smoother implementation
function kalman_smoother(data_in_deviations, A, 𝐁, û, P)
    T = size(data_in_deviations, 2)
    n = size(A, 1)
    
    u_smoother = zeros(n, T)
    u_smoother[:, end] = û[:, end]
    P_smoother = Array{Matrix{Float64}}(undef, T)
    P_smoother[end] = P[end]

    for t in T-1:-1:1
        J = P[t] * A' / (A * P[t] * A' + 𝐁)
        u_smoother[:, t] = û[:, t] + J * (u_smoother[:, t + 1] - A * û[:, t])
        P_smoother[t] = P[t] + J * (P_smoother[t + 1] - (A * P[t] * A' + 𝐁)) * J'
    end

    return u_smoother, P_smoother
end



us
Ps

u0 = u
P0 = P

T = size(data_in_deviations, 2)
n = size(A, 1)

û = zeros(n, T)
P = Array{Matrix{Float64}}(undef, T)
û[:, 1] = u0
P[1] = P0


# Update
F = C * P0 * C'
K = P0 * C' / F
û[:, 1] = K * (data_in_deviations[:, 1])




for t in 2:T
    # Predict
    û⁻ = A * û[:, t - 1]

    # Update
    F = C * P[t - 1] * C'
    K = P[t - 1] * C' / F
    û[:, t] = û⁻ + K * (data_in_deviations[:, t] - C * û⁻)
    P[t] = A * (P[t - 1] - K * C * P[t - 1]) * A' + 𝐁
end




for t in 1:T
    v[:,t] = data_in_deviations[:,t] - C * A * û[:,t]
    F[:,:,t] = C * P[:,:,t] * C'
    K = P[:,:,t] * C' / F[:,:,t]
    L[:,:,t] = A - A * K * C
    if t < T
        û[:,t+1] = A * û[:,t] + K * v[:,t]
        P[:,:,t+1] = A * P[:,:,t] * L[:,:,t]' + 𝐁
    end
end



T = size(data_in_deviations, 2)
    n = size(A, 1)
    
    u_smoother = zeros(n, T)
    u_smoother[:, end] = û[:, end]
    P_smoother = Array{Matrix{Float64}}(undef, T)
    P_smoother[end] = P[end]

    for t in T-1:-1:1
        J = P[t] * A' / (A * P[t] * A')
        u_smoother[:, t] = û[:, t] + J * (u_smoother[:, t + 1] - A * û[:, t])
        P_smoother[t] = P[t] + J * (P_smoother[t + 1] - (A * P[t] * A')) * J'
    end

ufilter, Pfilter = kalman_filter(data_in_deviations,A,𝐁,C,u,P)

usmooth , Psmooth = kalman_smoother(data_in_deviations, A, 𝐁, ufilter, Pfilter)


Pfilter[end-1] * A' / (A * Pfilter[end-1] * A' + 𝐁)


using Distributions

function kalman_filter(data_in_deviations, A, 𝐁, C, u0, P0)
    T = size(data_in_deviations,2)
    n = size(u0,1)
    û = zeros(n,T)
    P = zeros(n,n,T)
    v = zeros(size(C,1),T)
    F = zeros(size(C,1),size(C,1),T)
    L = zeros(n,n,T)

    û[:,1] = u0
    P[:,:,1] = P0

    t = 1
    F[:,:,t] = C * P[:,:,t] * C'
    K = P[:,:,t] * C' / F[:,:,t]
    L[:,:,t] = A - A * K * C
    û[:,t] = K * data_in_deviations[:,t]
    P[:,:,t] = A * P[:,:,t] * L[:,:,t]' + 𝐁

    for t in 2:T
        v[:,t] = data_in_deviations[:,t] - C * A * û[:,t-1]
        F[:,:,t] = C * P[:,:,t-1] * C'
        K = P[:,:,t-1] * C' / F[:,:,t]
        L[:,:,t] = A - A * K * C
        û[:,t] = A * û[:,t-1] + K * v[:,t]
        P[:,:,t] = A * P[:,:,t-1] * L[:,:,t]' + 𝐁
    end

    r = zeros(n)
    N = 0.0
    u = zeros(n,T)
    U = zeros(n,n,T)

    for t in T:-1:1
        r = C' / F[:,:,t] * v[:,t] + L[:,:,t]' * r
        N = C' / F[:,:,t] * C + L[:,:,t]' * N * L[:,:,t]
        u[:,t] = û[:,t] + P[:,:,t] * r
        U[:,:,t] = P[:,:,t] - P[:,:,t] * N * P[:,:,t]
    end

    return û,P,u,U
end


out = kalman_filter(data_in_deviations,A,𝐁,C,u,P)

out[1]
out[2]
out[3]
out[4]

ufilter

data_in_deviations = rand(Normal(0,1),100) # example data
C = 1.0
H = 0.5^2
A = 0.8
𝐁 = 0.2^2
u0 = 0.0
P0 = 1.0

v,F,K,û,P,u,U = kalman_filter(data_in_deviations,C,H,A,𝐁,u0,P0)





Sure! Here is an example of a Kalman filter and smoother implemented in Julia:


using Distributions

function kalman_filter(y, Z, H, T, Q, R, a1, P1)
    n = size(y)[2]
    m = size(a1)[1]
    a = zeros(m,n)
    P = zeros(m,m,n)
    v = zeros(n)
    F = zeros(n)
    K = zeros(m,n)
    L = zeros(m,m,n)

    a[:,1] = a1
    P[:,:,1] = P1
    for t in 1:n
        v[t] = y[t] - Z*a[:,t]
        F[t] = Z*P[:,:,t]*Z' + H
        K[:,t] = T*P[:,:,t]*Z'*inv(F[t])
        L[:,:,t] = T - K[:,t]*Z
        if t < n
            a[:,t+1] = T*a[:,t] + K[:,t]*v[t]
            P[:,:,t+1] = T*P[:,:,t]*L[:,:,t]' + R
        end
    end

    r = zeros(m)
    N = 0.0
    u = zeros(n)
    U = zeros(n)
    for t in n:-1:1
        r = Z' / F[t] * v[t] + L[:,:,t]' * r
        N = Z' / F[t] * Z + L[:,:,t]' * N * L[:,:,t]
        u[t] = a[:,t] + P[:,:,t] * r
        U[t] = P[:,:,t] - P[:,:,t]*N*P[:,:,t]
    end

    return v,F,K,a,P,u,U
end

y = rand(Normal(0,1),100) # example data
Z = 1.0
H = 0.5^2
T = 0.8
Q = 0.2^2
R = 0.3^2
a1 = 0.0
P1 = 1.0

v,F,K,a,P,u,U = kalman_filter(y,Z,H,T,Q,R,a1,P1)


# This code defines a function `kalman_filter` that takes in the observed data `y`, the observation matrix `Z`, the observation variance `H`, the state transition matrix `T`, the state transition variance `Q`, the initial state variance `R`, the initial state mean `a1`, and the initial state variance `P1`. The function returns the prediction error `v`, the prediction error variance `F`, the Kalman gain `K`, the filtered state mean `a`, the filtered state variance `P`, the smoothed state mean `u`, and the smoothed state variance `U`.

# I hope this helps! Let me know if you have any questions or need further assistance.






# function calculate_kalman_filter_loglikelihood(𝓂::ℳ, data::AbstractArray{Float64}, observables::Vector{Symbol}; parameters = nothing, verbose::Bool = false, tol::AbstractFloat = eps())
#     sort!(observables)

#     solve!(𝓂, verbose = verbose)

#     parameters = 𝓂.parameter_values

#     SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, verbose)

#     NSSS_labels = [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]

#     obs_indices = indexin(observables,NSSS_labels)

#     data_in_deviations = collect(data(observables)) .- SS_and_pars[obs_indices]

# 	∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)

#     sol = calculate_first_order_solution(∇₁; T = 𝓂.timings)

#     observables_and_states = sort(union(𝓂.timings.past_not_future_and_mixed_idx,indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))))

#     A = @views sol[observables_and_states,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(𝓂.timings.past_not_future_and_mixed_idx,observables_and_states)),:]
#     B = @views sol[observables_and_states,𝓂.timings.nPast_not_future_and_mixed+1:end]

#     C = @views ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))),observables_and_states)),:]

#     𝐁 = B * B'

#     calculate_covariance_ = calculate_covariance_AD(sol, T = 𝓂.timings, subset_indices = Int64[observables_and_states...])

#     P = calculate_covariance_(sol)
#     u = zeros(length(observables_and_states))
#     z = C * u
    
#     loglik = 0.0

#     for t in 1:size(data)[2]
#         v = data_in_deviations[:,t] - z

#         F = C * P * C'
        
#         K = P * C' / F

#         P = A * (P - K * C * P) * A' + 𝐁

#         u = A * (u + K * v)
        
#         z = C * u 
#     end

#     return -(loglik + length(data) * log(2 * 3.141592653589793)) / 2 # otherwise conflicts with model parameters assignment
# end










function kalman_filter_and_smoother(data, A, 𝐁, C, u, P)
    n_timesteps = size(data, 2)
    n_states = length(u)
    filtered_states = zeros(n_states, n_timesteps)
    filtered_covariance = zeros(n_states, n_states, n_timesteps)
    smoothed_states = zeros(n_states, n_timesteps)
    smoothed_covariance = zeros(n_states, n_states, n_timesteps)

    # filtered_shocks = zeros(n_states, n_timesteps)
    # smoothed_shocks = zeros(n_states, n_timesteps)
    # P_smoothed = copy(P)

    # Kalman filter
    for t in 1:n_timesteps
        v = data[:, t] - C * u
        F = C * P * C'
        K = A * P * C' / F
        û = u + P * C' / F * v
        P̂ = P - P * C' / F * C * P
        u = A * û
        L = A - K * C
        P = A * P * L' + 𝐁

        filtered_states[:, t] = u
        filtered_covariance[:,:,t] = P
    end

    # Kalman smoother for states
    smoothed_states[:, end] = filtered_states[:, end]
    smoothed_covariance[:,:, end] = filtered_covariance[:,:, end]
    r = zero(u)

    for t in n_timesteps:-1:1
        u = filtered_states[:, t]
        P = filtered_covariance[:,:,t] 

        v = data[:, t] - C * u
        F = C * P * C'
        K = P * C' / F
        r = (C' * F)' \ v + r
        smoothed_states[:, t] = u + P * r
        smoothed_covariance[:,:, t] = P - r * r' * P'
        r = A' * r
    end


    return filtered_states, filtered_covariance, smoothed_states ,smoothed_covariance #, filtered_shocks, smoothed_shocks
end


outt = kalman_filter_and_smoother(data_in_deviations,A,𝐁,C,u,P)

C*outt[1]
C*outt[3]
data_in_deviations

using StatsPlots

plot(data_in_deviations[1,:])
plot!((C*outt[1])[1,:])
plot!((C*outt[3])[1,:])


plot(data_in_deviations[2,:])
plot!((C*outt[1])[2,:])
plot!((C*outt[3])[2,:])







𝐁 = B * B'

# Gaussian Prior

calculate_covariance_ = calculate_covariance_AD(sol, T = 𝓂.timings, subset_indices = Int64[observables_and_states...])

Pstar1 = calculate_covariance_(sol)
# P = reshape((ℒ.I - ℒ.kron(A, A)) \ reshape(𝐁, prod(size(A)), 1), size(A))
u = zeros(length(observables_and_states))
# u = SS_and_pars[sort(union(𝓂.timings.past_not_future_and_mixed,observables))] |> collect
z = C * u



using LinearAlgebra


nk = 1
d = 0
decomp = []
# spinf = size(Pinf1)
spstar = size(Pstar1)
v = zeros(size(C,1), size(data_in_deviations,2))
u = zeros(size(A,1), size(data_in_deviations,2)+1)
# û = zeros(size(A,1), size(data_in_deviations,2))
# uK = zeros(nk, size(A,1), size(data_in_deviations,2)+nk)
# PK = zeros(nk, size(A,1), size(A,1), size(data_in_deviations,2)+nk)
iF = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# Fstar = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# iFstar = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# iFinf = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# K = zeros(size(A,1), size(C,1), size(data_in_deviations,2))
L = zeros(size(A,1), size(A,1), size(data_in_deviations,2))
# Linf = zeros(size(A,1), size(A,1), size(data_in_deviations,2))
# Lstar = zeros(size(A,1), size(A,1), size(data_in_deviations,2))
# Kstar = zeros(size(A,1), size(C,1), size(data_in_deviations,2))
# Kinf = zeros(size(A,1), size(C,1), size(data_in_deviations,2))
P = zeros(size(A,1), size(A,1), size(data_in_deviations,2)+1)
# Pstar = zeros(spstar[1], spstar[2], size(data_in_deviations,2)+1)
# Pstar[:, :, 1] = Pstar1
# Pinf = zeros(spinf[1], spinf[2], size(data_in_deviations,2)+1)
# Pinf[:, :, 1] = Pinf1
rr = size(C,1)
# 𝐁 = R * Q * transpose(R)
ū = zeros(size(A,1), size(data_in_deviations,2))
ϵ̄ = zeros(rr, size(data_in_deviations,2))
ϵ = zeros(rr, size(data_in_deviations,2))
epsilonhat = zeros(rr, size(data_in_deviations,2))
r = zeros(size(A,1), size(data_in_deviations,2)+1)
Finf_singular = zeros(1, size(data_in_deviations,2))

V = []

# t = 0

# d = t
P[:, :, 1] = Pstar1
# iFinf = iFinf[:, :, 1:d]
# iFstar= iFstar[:, :, 1:d]
# Linf = Linf[:, :, 1:d]
# Lstar = Lstar[:, :, 1:d]
# Kstar = Kstar[:, :, 1:d]
# Pstar = Pstar[:, :, 1:d]
# Pinf = Pinf[:, :, 1:d]
# K
# û

for t in 1:size(data_in_deviations,2)
    v[:, t] = data_in_deviations[:, t] - C * u[:, t]
    F = C * P[:, :, t] * C'
    iF[:, :, t] = inv(F)
    PCiF = P[:, :, t] * C' * iF[:, :, t]
    û = u[:, t] + PCiF * v[:, t]
    K = A * PCiF
    L[:, :, t] = A - K * C
    P[:, :, t+1] = A * P[:, :, t] * L[:, :, t]' + 𝐁
    u[:, t+1] = A * û
    ϵ[:, t] = B' * C' * iF[:, :, t] * v[:, t]
    # Pf = P[:, :, t]
    # uK[1, :, t+1] = u[:, t+1]
    # for jnk in 1:nk
    #     Pf = A * Pf * A' + 𝐁
    #     PK[jnk, :, :, t+jnk] = Pf
    #     if jnk > 1
    #         uK[jnk, :, t+jnk] = A * uK[jnk-1, :, t+jnk-1]
    #     end
    # end
end

# backward pass; r_T and N_T, stored in entry (size(data_in_deviations,2)+1) were initialized at 0
# t = size(data_in_deviations,2) + 1
for t in size(data_in_deviations,2):-1:1
    r[:, t] = C' * iF[:, :, t] * v[:, t] + L[:, :, t]' * r[:, t+1] # compute r_{t-1}, DK (2012), eq. 4.38
    ū[:, t] = u[:, t] + P[:, :, t] * r[:, t] # DK (2012), eq. 4.35
    ϵ̄[:, t] = B' * r[:, t] # DK (2012), eq. 4.63
end

# if decomp_flag
    decomp = zeros(nk, size(A,1), rr, size(data_in_deviations,2)+nk)
    ZRQinv = inv(C * 𝐁 * C')
    for t in 1:size(data_in_deviations,2)
        # : = data_index[t]
        # calculate eta_tm1t
        eta_tm1t = B' * C' * iF[:, :, t] * v[:, t]
        AAA = P[:, :, t] * C' * ZRQinv[:, :] * (C * B .* eta_tm1t')
        # calculate decomposition
        decomp[1, :, :, t+1] = AAA
        for h = 2:nk
            AAA = A * AAA
            decomp[h, :, :, t+h] = AAA
        end
    end
# end

epsilonhat = data_in_deviations - C * ū





# outt = kalman_filter_and_smoother(data_in_deviations,A,𝐁,C,u,P)

# C*outt[1]
# C*outt[3]
# data_in_deviations

using StatsPlots

plot(data_in_deviations[1,:])
plot!((C*ū)[1,:])
plot!((C*u)[1,2:end])


plot(data_in_deviations[2,:])
plot!((C*ū)[2,:])
plot!((C*u)[2,2:end])

plot(ϵ̄[1,:])
plot(ϵ̄[2,:])

ϵ̄

plot(ϵ[1,:])
plot(ϵ[2,:])

ϵ