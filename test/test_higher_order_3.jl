using Test
using MacroModelling
import MacroModelling: clear_solution_caches!
using Random

include("functionality_tests.jl")

plots = true
# test_higher_order = true

include("models/Caldara_et_al_2012_estim.jl")

@testset verbose = true "RBC_CME with calibration equations second order" begin
    include("models/RBC_CME_calibration_equations.jl")
    functionality_test(m, Caldara_et_al_2012_estim, algorithm = :second_order, plots = plots)
end
# m = nothing
GC.gc()

@testset verbose = true "RBC_CME with calibration equations third order" begin
    # include("models/RBC_CME_calibration_equations.jl")
    functionality_test(m, Caldara_et_al_2012_estim, algorithm = :third_order, plots = plots)
end
m = nothing
GC.gc()

@testset verbose = true "RBC_CME second order" begin
    include("models/RBC_CME.jl")
    functionality_test(m, Caldara_et_al_2012_estim, algorithm = :second_order, plots = plots)
end
# m = nothing
GC.gc()

@testset verbose = true "RBC_CME third order" begin
    # include("models/RBC_CME.jl")
    functionality_test(m, Caldara_et_al_2012_estim, algorithm = :third_order, plots = plots)
end
m = nothing
GC.gc()

@testset verbose = true "RBC_CME with calibration equations and parameter definitions second order" begin
    include("models/RBC_CME_calibration_equations_and_parameter_definitions.jl")
    functionality_test(m, Caldara_et_al_2012_estim, algorithm = :second_order, plots = plots)
end
# m = nothing
GC.gc()

@testset verbose = true "RBC_CME with calibration equations and parameter definitions third order" begin
    # include("models/RBC_CME_calibration_equations_and_parameter_definitions.jl")
    functionality_test(m, Caldara_et_al_2012_estim, algorithm = :third_order, plots = plots)
end
m = nothing
GC.gc()
