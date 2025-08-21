using Revise
using MacroModelling, StatsPlots

# TODO:
# fix other twinx situations (simplify it), especially bar plot in decomposition can have dual axis with the yticks trick
# put plots back in Docs 
# redo plots in docs

include("../models/GNSS_2010.jl")

model = GNSS_2010
get_shocks(model)

shcks = :e_y
vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

plot_irf(model, shocks = shcks, variables = vars)

plot_irf!(model, 
            shock_size = 1.2,
            plot_type = :stack,
            shocks = shcks, variables = vars)

plot_irf!(model, 
            shock_size = 1.2,
            plot_type = :stack,
            shocks = shcks, variables = vars)

plot_irf!(model, 
            shock_size = 0.2,
            plot_type = :stack,
            shocks = shcks, variables = vars)


vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

plot_irf(model, algorithm = :pruned_second_order, shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :pruned_second_order, 
            shock_size = 1.2,
            shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :pruned_second_order, 
            shock_size = 1.2,
            plot_type = :stack,
            shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :pruned_second_order, 
            shock_size = -1,
            shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :second_order, 
            shock_size = -1,
            plot_type = :stack,
            shocks = shcks, variables = vars)


vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

plot_irf(model, shocks = shcks, variables = vars)

plot_irf!(model, 
            shock_size = -1,
            shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :pruned_second_order, 
            periods = 5,
            plot_type = :stack,
            shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :second_order, 
            shock_size = -1,
            plot_type = :stack,
            shocks = shcks, variables = vars)


vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

get_shocks(model)

plot_irf(model, shocks = shcks, variables = vars)

plot_irf!(model, shocks = :e_j, 
            # plot_type = :stack,
            variables = vars)

plot_irf!(model, shocks = [:e_j, :e_me], 
            plot_type = :stack,
            variables = vars)

plot_irf!(model, 
            # plot_type = :stack,
            variables = vars)



vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

get_shocks(model)

plot_irf(model, shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :pruned_second_order, 
            shock_size = -1,
            plot_type = :stack,
            shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :second_order, 
            shock_size = -1,
            # plot_type = :stack,
            shocks = shcks, variables = vars)


vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

plot_irf(model, shocks = shcks, variables = vars)

plot_irf!(model, shocks = shcks, 
            # plot_type = :stack,
            variables = vars[2:end], shock_size = -1)


vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

plot_irf(model, shocks = shcks, variables = vars)

for a in [:second_order, :pruned_second_order, :third_order, :pruned_third_order]
    plot_irf!(model, shocks = shcks, variables = vars, algorithm = a)
end


vars = [:C, :K, :Y, :r_k, :w_p]

plot_irf(model, shocks = shcks, variables = vars)

for a in [:second_order, :pruned_second_order, :third_order, :pruned_third_order]
    plot_irf!(model, shocks = shcks, variables = vars, algorithm = a)
end



vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

plot_irf(model, shocks = shcks, variables = vars)

plot_irf!(model, shocks = shcks, 
            plot_type = :stack,
            variables = vars, negative_shock = true)

plot_irf!(model, shocks = :e_j, variables = vars, negative_shock = true)

plot_irf!(model, shocks = :e_j, shock_size = 2, variables = vars, negative_shock = true)

plot_irf!(model, shocks = :e_j, shock_size = -2, variables = vars, negative_shock = true, algorithm = :second_order)


vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

plot_irf(model, shocks = shcks, variables = vars, algorithm = :pruned_second_order)

plot_irf!(model, shocks = shcks, 
            # plot_type = :stack,
            variables = vars, generalised_irf = true, algorithm = :pruned_second_order)

plot_irf!(model, shocks = shcks, variables = vars, algorithm = :pruned_third_order)

plot_irf!(model, shocks = shcks, variables = vars, generalised_irf = true, algorithm = :pruned_third_order)




include("../models/Gali_2015_chapter_3_obc.jl")

model = Gali_2015_chapter_3_obc
get_shocks(model)
get_variables(model)[1:10]
shcks = :eps_z
vars = [:A, :C, :MC, :M_real, :N, :Pi, :Pi_star, :Q, :R, :S]

plot_irf(model, shocks = shcks, variables = vars, periods = 10)

plot_irf!(model, shocks = shcks, 
            # plot_type = :stack,
            variables = vars, periods = 10, ignore_obc = true)

plot_irf!(model, shocks = shcks, 
            # plot_type = :stack,
            variables = vars, periods = 10, shock_size = 2, ignore_obc = false)

plot_irf!(model, shocks = shcks, variables = vars, periods = 10, shock_size = 2, ignore_obc = true)

plot_irf!(model, shocks = :eps_a, variables = vars, periods = 10, shock_size = 4, ignore_obc = false)

plot_irf!(model, shocks = :eps_a, variables = vars, periods = 10, shock_size = 4, ignore_obc = true)



plot_irf(model, shocks = shcks, variables = vars, periods = 10)

plot_irf!(model, shocks = shcks, variables = vars, periods = 10, ignore_obc = true)

plot_irf!(model, shocks = shcks, variables = vars, periods = 10, algorithm = :pruned_second_order, ignore_obc = true)

plot_irf!(model, shocks = shcks, variables = vars, periods = 10, algorithm = :pruned_second_order, shock_size = 2, ignore_obc = false)

plot_irf!(model, shocks = shcks, variables = vars, periods = 10, algorithm = :pruned_second_order, shock_size = 2, ignore_obc = true)

plot_irf!(model, shocks = :eps_a, variables = vars, periods = 10, algorithm = :pruned_second_order, shock_size = 4, ignore_obc = false)

plot_irf!(model, shocks = :eps_a, variables = vars, periods = 10, algorithm = :pruned_second_order, shock_size = 4, ignore_obc = true)


plot_irf(model, shocks = shcks, variables = vars, algorithm = :pruned_second_order)

plot_irf!(model, shocks = shcks, variables = vars, algorithm = :pruned_second_order, quadratic_matrix_equation_algorithm = :doubling)

plot_irf!(model, shocks = shcks, variables = vars, algorithm = :pruned_second_order, sylvester_algorithm = :doubling)

plot_irf(model, shocks = shcks, variables = vars, algorithm = :pruned_third_order)

plot_irf!(model, shocks = shcks, variables = vars, algorithm = :pruned_third_order, quadratic_matrix_equation_algorithm = :doubling)


get_parameters(model, values = true)

plot_irf(model, shocks = shcks, variables = vars, parameters = :α => .25)

plot_irf!(model, shocks = shcks, variables = vars, parameters = :α => .2)

SS(model, derivatives = false, parameters = :α => .25)(:R)
SS(model, derivatives = false, parameters = :α => .2)(:R)


# DONE: handle initial state and tol

init_state = get_irf(model, shocks = :none, variables = :all,
       periods = 1, levels = true)

init_state[1] += 1

plot_irf(model, shocks = shcks, variables = vars, ignore_obc = true,
initial_state = vec(init_state))


plot_irf!(model, shocks = :none, variables = vars, ignore_obc = true,
       initial_state = vec(init_state), 
            plot_type = :stack,
    #    algorithm = :second_order
       )
       
init_state_2 = get_irf(model, shocks = :none, variables = :all, periods = 1, levels = true)

init_state_2[1] += 2

init_state[1] += 2

plot_irf!(model, shocks = :none, variables = vars, ignore_obc = true,initial_state = vec(init_state))

       
# init_state_2 = get_irf(model, shocks = :none, variables = :all, periods = 1, levels = true)

init_state[1] += .2

plot_irf!(model, shocks = shcks, variables = vars, ignore_obc = true,
       algorithm = :second_order,
initial_state = vec(init_state)
)


init_state_2 = get_irf(model, shocks = :none, variables = :all, algorithm = :pruned_second_order, periods = 1, levels = false)

plot_irf!(model, shocks = shcks, variables = vars, ignore_obc = true,
       algorithm = :pruned_second_order,
initial_state = vec(init_state_2)
)


plot_irf(model, shocks = shcks, variables = vars)

# plot_irf!(model, shocks = shcks, variables = vars, ignore_obc = true)

plot_irf!(model, shocks = shcks, variables = vars, tol = Tolerances(NSSS_acceptance_tol = 1e-8))

plot_irf!(model, shocks = shcks, variables = vars, quadratic_matrix_equation_algorithm = :doubling)


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

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.01, :β => 0.95, :ρ => 0.2], algorithm = :second_order)

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.01, :β => 0.95, :ρ => 0.2], algorithm = :pruned_second_order)

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.01, :β => 0.955, :ρ => 0.2], algorithm = :second_order)

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.01, :β => 0.957, :ρ => 0.5])

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.012, :β => 0.97, :ρ => 0.5])

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.01, :β => 0.97, :ρ => 0.55])

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.021, :β => 0.97, :ρ => 0.55])


include("models/SW07_nonlinear.jl")

hcat(SS(SW07_nonlinear, derivatives = false, parameters = [:ctrend => .35, :curvw => 10, :calfa => 0.18003])[30:end]
,SS(SW07_nonlinear, derivatives = false, parameters = :calfa => 0.15)[30:end])

get_shocks(SW07_nonlinear)
shock_series = KeyedArray(zeros(2,12), Shocks = [:eb, :ew], Periods = 1:12)
shock_series[1,2] = 1
shock_series[2,12] = -1
plot_irf(SW07_nonlinear, shocks = :ew, 
                        # negative_shock = true,
                        # generalised_irf = false,
                        # algorithm = :pruned_second_order,
                        # variables = [:robs,:ygap,:pinf,
                        # :gamw1,:gamw2,:gamw3,
                        # :inve,:c,:k],
                        # variables = [:ygap],
                        parameters = [:ctrend => .35, :curvw => 10, :calfa => 0.18003])


plot_irf!(SW07_nonlinear, shocks = shock_series, 
                        # negative_shock = true,
                        # generalised_irf = false,
                        # algorithm = :pruned_second_order,
                        # variables = [:robs,:ygap,:pinf,
                        # :gamw1,:gamw2,:gamw3,
                        # :inve,:c,:k],
                        # variables = [:ygap],
                        parameters = [:ctrend => .35, :curvw => 10, :calfa => 0.18003])

plot_irf!(SW07_nonlinear, shocks = :ew, 
                        # generalised_irf = true,
                        algorithm = :pruned_second_order,
                        # shock_size = 2,
                        # quadratic_matrix_equation_algorithm = :doubling,
                        # tol = MacroModelling.Tolerances(NSSS_acceptance_tol = 1e-10),
                        # negative_shock = true,
                        variables = [:robs,:ygap,:pinf,
                        # :gamw1,:gamw2,:gamw3,
                        :inve,:c,:k],
                        # variables = [:ygap],
                        parameters = [:ctrend => .365, :curvw => 10, :calfa => 0.18003])

for s in setdiff(get_shocks(SW07_nonlinear),["ew"])
    MacroModelling.plot_irf!(SW07_nonlinear, shocks = s,
                            # variables = [:robs,:ygap,:pinf,
                            # :gamw1,:gamw2,:gamw3,
                            # :inve,:c,:k],
                            # variables = [:ygap],
                            # plot_type = :stack,
                            parameters = [:ctrend => .35, :curvw => 10, :calfa => 0.18003])
end

# DONE: handle case where one plots had one shock, the other has multiple ones
# DONE: when difference is along one dimension dont use label but legend only
MacroModelling.plot_irf!(SW07_nonlinear, 
                        shocks = :epinf,
                        variables = [:gam1,:gam2,:gam3,
                        # :gamw1,:gamw2,:gamw3,
                        :inve,:kp,:k])

MacroModelling.plot_irf!(SW07_nonlinear, 
                        shocks = [:ew],
                        # plots_per_page = 9,
                        variables = [:gam1,:gam2,:gam3,
                        :gamw1,:gamw2,:gamw3,
                        :inve,:kp,:k],
                        parameters = :calfa => 0.15)

MacroModelling.plot_irf!(SW07_nonlinear, 
                        shocks = [:epinf,:ew],
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
