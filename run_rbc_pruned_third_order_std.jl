using MacroModelling

include("models/RBC_baseline.jl")

std = get_std(RBC_baseline, algorithm = :pruned_third_order)
println(std)
