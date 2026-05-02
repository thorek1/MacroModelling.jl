using Test
using MacroModelling
import MacroModelling: clear_solution_caches!
using Random
Random.seed!(1234)

include("functionality_tests.jl")

plots = true
Random.seed!(1)

include("models/Caldara_et_al_2012_estim.jl")

@testset verbose = true "Backus_Kehoe_Kydland_1992" begin
    include("../models/Backus_Kehoe_Kydland_1992.jl")
    functionality_test(Backus_Kehoe_Kydland_1992, Caldara_et_al_2012_estim, plots = plots)
end
Backus_Kehoe_Kydland_1992 = nothing
GC.gc()

@testset verbose = true "FS2000" begin
    include("../models/FS2000.jl")
    functionality_test(FS2000, Caldara_et_al_2012_estim, plots = plots)
end
FS2000 = nothing
GC.gc()
