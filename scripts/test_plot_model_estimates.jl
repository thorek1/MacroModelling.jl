using Test

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using MacroModelling
using StatsPlots

@testset "plot_model_estimates" begin
    include(joinpath(@__DIR__, "..", "test", "models", "RBC_CME.jl"))

    sim = simulate(m)
    data_k = sim([:k], :, :simulate)

    plots = plot_model_estimates(
        m,
        data_k;
        show_plots = false,
        save_plots = false,
        rename_dictionary = Dict(:k => "Capital")
    )
    @test !isempty(plots)

    plots2 = plot_model_estimates!(
        m,
        data_k;
        show_plots = false,
        save_plots = false,
        smooth = false,
        rename_dictionary = Dict(:k => "Capital")
    )
    @test !isempty(plots2)
end

