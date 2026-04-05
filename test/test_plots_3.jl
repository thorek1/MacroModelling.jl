using Test
using MacroModelling
import MacroModelling: clear_solution_caches!
using Random
import StatsPlots

include("functionality_tests.jl")

plots = true
Random.seed!(1)

include("models/Caldara_et_al_2012_estim.jl")

@testset verbose = true "Gali 2015 ELB" begin
    include("../models/Gali_2015_chapter_3_obc.jl")
    functionality_test(Gali_2015_chapter_3_obc, Caldara_et_al_2012_estim, plots = plots)
end
Gali_2015_chapter_3_obc = nothing
GC.gc()
