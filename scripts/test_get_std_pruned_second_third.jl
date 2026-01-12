using Test
using MacroModelling

@testset "get_std pruned second/third order" begin
    include(joinpath(@__DIR__, "..", "test", "models", "RBC_CME.jl"))
    m_no_calibration_equations = m

    include(joinpath(@__DIR__, "..", "test", "models", "RBC_CME_calibration_equations.jl"))
    m_with_calibration_equations = m

    for (name, model) in [
        ("without calibration equations", m_no_calibration_equations),
        ("with calibration equations", m_with_calibration_equations),
    ]
        @testset "$name" begin
            std2 = get_std(model; algorithm = :pruned_second_order, silent = true, verbose = false, derivatives = false)
            @test !isempty(std2)
            @test all(isfinite, vec(Array(std2)))

            std3 = get_std(model; algorithm = :pruned_third_order, silent = true, verbose = false, derivatives = false)
            @test !isempty(std3)
            @test all(isfinite, vec(Array(std3)))
        end
    end
end
