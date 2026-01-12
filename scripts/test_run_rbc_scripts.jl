using Test
using MacroModelling

@testset "RBC pruned moments scripts" begin
    include(joinpath(@__DIR__, "..", "models", "RBC_baseline.jl"))

    mean = get_mean(RBC_baseline; algorithm = :pruned_second_order)
    @test !isempty(mean)
    @test all(isfinite, vec(mean))

    std = get_std(RBC_baseline; algorithm = :pruned_third_order)
    @test !isempty(std)
    @test all(isfinite, vec(std))
end

@testset "RBC calibration equations script" begin
    include(joinpath(@__DIR__, "..", "test", "models", "RBC_CME_calibration_equations.jl"))

    solve!(m)

    calibration_equations = get_calibration_equations(m)
    @test length(calibration_equations) == 2

    calibrated_parameters = get_calibrated_parameters(m; values = true)
    @test length(calibrated_parameters) == 2

    steady_state = get_steady_state(m; derivatives = false)
    @test !isempty(steady_state)
    @test all(isfinite, steady_state)
end

