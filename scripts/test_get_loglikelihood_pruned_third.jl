using MacroModelling
using Random
using Test

# minimal script to test get_loglikelihood with pruned third order
include(joinpath(@__DIR__, "..", "models", "RBC_baseline.jl"))

Random.seed!(123)

sim = simulate(RBC_baseline)
data = sim([:y], :, :simulate)

llh = get_loglikelihood(RBC_baseline, data, RBC_baseline.parameter_values, algorithm = :pruned_third_order, verbose = false)

@test isfinite(llh)

println("LLH (pruned_third_order): ", llh)
