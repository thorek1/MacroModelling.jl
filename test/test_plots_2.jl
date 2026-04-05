using Test
using MacroModelling
import MacroModelling: clear_solution_caches!
using Random
import StatsPlots

include("functionality_tests.jl")

plots = true
Random.seed!(1)

include("models/Caldara_et_al_2012_estim.jl")

@testset verbose = true "Smets and Wouters (2007) nonlinear" begin
    include("../models/Smets_Wouters_2007.jl")
    functionality_test(Smets_Wouters_2007, Caldara_et_al_2012_estim, plots = plots)
end
Smets_Wouters_2007 = nothing
GC.gc()

@testset verbose = true "Smets_Wouters_2003 with calibration equations" begin
    include("../models/Smets_Wouters_2003.jl")
    functionality_test(Smets_Wouters_2003, Caldara_et_al_2012_estim, plots = plots)
end
Smets_Wouters_2003 = nothing
GC.gc()

@testset verbose = true "Smets and Wouters (2007) linear" begin
    include("../models/Smets_Wouters_2007_linear.jl")
    functionality_test(Smets_Wouters_2007_linear, Caldara_et_al_2012_estim, plots = plots)
end
Smets_Wouters_2007_linear = nothing
GC.gc()
