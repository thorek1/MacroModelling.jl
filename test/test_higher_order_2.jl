using Test
using MacroModelling
using Random
import StatsPlots

include("functionality_tests.jl")

Random.seed!(18400875)
plots = true
# test_higher_order = true

include("models/Caldara_et_al_2012_estim.jl")

@testset verbose = true "RBC_CME with calibration equations, parameter definitions, special functions, variables in steady state, and leads/lag > 1 on endogenous and exogenous variables pruned second order" begin
    include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags.jl")
    functionality_test(m, Caldara_et_al_2012_estim, algorithm = :pruned_second_order, plots = plots)
end
# m = nothing
GC.gc()

@testset verbose = true "RBC_CME with calibration equations, parameter definitions, special functions, variables in steady state, and leads/lag > 1 on endogenous and exogenous variables pruned third order" begin
    # include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags.jl")
    functionality_test(m, Caldara_et_al_2012_estim, algorithm = :pruned_third_order, plots = plots)
end
m = nothing
GC.gc()
