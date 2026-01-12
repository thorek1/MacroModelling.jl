using Test
using MacroModelling

@testset "get_var_axis type stability" begin
    include(joinpath(@__DIR__, "..", "models", "RBC_baseline.jl"))

    MacroModelling.ensure_name_display_cache!(RBC_baseline)
    @test MacroModelling.get_var_axis(RBC_baseline) isa Vector{Symbol}
    @test MacroModelling.get_exo_axis(RBC_baseline) isa Vector{Symbol}
    @test MacroModelling.get_exo_axis(RBC_baseline; with_subscript = false) isa Vector{Symbol}
end

