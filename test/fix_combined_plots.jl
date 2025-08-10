using Revise
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

plot_irf(RBC, parameters = [:std_z => 0.01, :β => 0.95, :ρ => 0.2])

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.012, :β => 0.95, :ρ => 0.75])

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.01, :β => 0.957, :ρ => 0.5])

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.01, :β => 0.97, :ρ => 0.5])

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.01, :β => 0.97, :ρ => 0.55])

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.021, :β => 0.97, :ρ => 0.55])


include("models/SW07_nonlinear.jl")

plot_irf(SW07_nonlinear, shocks = :ew, 
                        variables = [:gam1,:gam2,:gam3,
                        # :gamw1,:gamw2,:gamw3,
                        :inve,:kp,:k],
                        parameters = [:ctrend => .35, :curvw => 10, :calfa => 0.18003])

MacroModelling.plot_irf!(SW07_nonlinear, 
                        shocks = :ew,
                        variables = [:gam1,:gam2,:gam3,
                        # :gamw1,:gamw2,:gamw3,
                        :inve,:kp,:k],
                        parameters = :calfa => 0.15)

MacroModelling.plot_irf!(SW07_nonlinear, 
                        shocks = :ew,
                        variables = [:gam1,:gam2,:gam3,
                        :gamw1,:gamw2,:gamw3,
                        :inve,:kp,:k],
                        parameters = :curvw => 9)

MacroModelling.plot_irf!(SW07_nonlinear, 
                        shocks = :ew,
                        variables = [#:gam1,:gam2,:gam3,
                        :gamw1,:gamw2,:gamw3,
                        :inve,:kp,:k],
                        parameters = :cgy => .45)

MacroModelling.plot_irf!(SW07_nonlinear, 
                        shocks = :ew,
                        # plots_per_page = 4,
                        # variables = [:zcap,:gam1],
                        # variables = [:dy,:robs,:y,
                        # :xi,:ygap,
                        # :wnew,:xi,:ygap,
                        # :k,:kp,:r],
                        parameters = :ctrend => .5)

get_parameters(SW07_nonlinear, values = true)

diffdict = MacroModelling.compare_args_and_kwargs(MacroModelling.irf_active_plot_container)

using StatsPlots, DataFrames
using Plots

diffdict[:parameters]
mapreduce((x, y) -> x ∪ y, diffdict[:parameters])

df = diffdict[:parameters]|>DataFrame
param_nms = diffdict[:parameters]|>keys|>collect|>sort

plot_vector = Pair{String,Any}[]
for param in param_nms
    push!(plot_vector, String(param) => diffdict[:parameters][param])
end

pushfirst!(plot_vector, "Plot index" => 1:length(diffdict[:parameters][param_nms[1]]))


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
    df_plot = heatmap(bg_matrix;
                c = cgrad([:lightgrey, :white]),      # Color gradient for background
                yflip = true,  
                tick=:none,
                legend=false,
                framestyle = :none, # Keep the outer box 
                cbar=false)
 
    # overlay the header and numeric values
    for j in 1:ncols
        annotate!(df_plot, j, 1, text(plot_vector[j].first, :center, 8)) # Header
        for i in 1:nrows
            annotate!(df_plot, j, i+1, text(string(plot_vector[j].second[i]), :center, 8))
        end
    end
    return df_plot
end

plot_df(plot_vector)
