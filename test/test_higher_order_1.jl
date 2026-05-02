using Test
using MacroModelling
import MacroModelling: clear_solution_caches!
using Random
Random.seed!(1234)

include("functionality_tests.jl")

plots = true
# test_higher_order = true

include("models/Caldara_et_al_2012_estim.jl")

@testset verbose = true "FS2000 third order" begin
    include("../models/FS2000.jl")
    functionality_test(FS2000, Caldara_et_al_2012_estim, algorithm = :third_order, plots = plots)
end
FS2000 = nothing
GC.gc()

@testset verbose = true "FS2000 pruned third order" begin
    include("../models/FS2000.jl")
    functionality_test(FS2000, Caldara_et_al_2012_estim, algorithm = :pruned_third_order, plots = plots)
end
FS2000 = nothing
GC.gc()

@testset verbose = true "FS2000 second order" begin
    include("../models/FS2000.jl")
    functionality_test(FS2000, Caldara_et_al_2012_estim, algorithm = :second_order, plots = plots)
end
FS2000 = nothing
GC.gc()

@testset verbose = true "FS2000 pruned second order" begin
    include("../models/FS2000.jl")
    functionality_test(FS2000, Caldara_et_al_2012_estim, algorithm = :pruned_second_order, plots = plots)
end
FS2000 = nothing
GC.gc()
