using Revise
using MacroModelling
import StatsPlots

include("models/RBC_baseline.jl")

algos = [:first_order, :second_order, :pruned_second_order, :third_order, :pruned_third_order]

# c is conditioned to deviate by 0.01 in period 1 and y is conditioned to deviate by 0.02 in period 2
conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,2),Variables = [:c,:y], Periods = 1:2)
conditions[1,1] = .01
conditions[2,2] = .02

# in period 2 second shock (eps_z) is conditioned to take a value of 0.05
shocks = Matrix{Union{Nothing,Float64}}(undef,2,1)
shocks[1,1] = .05

model = RBC_baseline

for algo in algos
    get_steady_state(model, algorithm = algo)

    get_conditional_forecast(model, conditions,  conditions_in_levels = false, algorithm = algo)

    if algo in [:first_order, :pruned_second_order, :pruned_third_order]
        get_mean(model, algorithm = algo)
        get_std(model, algorithm = algo)
    end
end


# include("models/Backus_Kehoe_Kydland_1992.jl")

# model = Backus_Kehoe_Kydland_1992

# algos = [:first_order, :second_order, :pruned_second_order]

# conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,2),Variables = ["C{H}","Y{F}"], Periods = 1:2)
# conditions[1,1] = .01
# conditions[2,2] = .02

# for algo in algos
#     get_steady_state(model, algorithm = algo)

#     get_conditional_forecast(model, conditions,  conditions_in_levels = false, algorithm = algo)

#     if algo in [:first_order, :pruned_second_order, :second_order]
#         get_mean(model, algorithm = algo)
#         get_std(model, algorithm = algo)
#     end
# end
