using MacroModelling
include("models/RBC_baseline.jl")
mean = get_mean(RBC_baseline, algorithm = :pruned_second_order)
println(mean)
