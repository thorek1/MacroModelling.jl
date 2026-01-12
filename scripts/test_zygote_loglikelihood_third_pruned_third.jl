using Pkg

Pkg.activate(temp = true)
Pkg.develop(path = joinpath(@__DIR__, ".."))
Pkg.add("Zygote")
Pkg.instantiate()

using Test
using Random
using MacroModelling
using Zygote

@testset "Zygote ∂get_loglikelihood/∂params" begin
    include(joinpath(@__DIR__, "..", "models", "RBC_baseline.jl"))

    Random.seed!(123)
    sim = simulate(RBC_baseline)
    data = sim([:y], :, :simulate)

    x0 = copy(RBC_baseline.parameter_values)

    for algorithm in (:third_order, :pruned_third_order)
        f(x) = get_loglikelihood(RBC_baseline, data, x; algorithm = algorithm, filter = :inversion, verbose = false)

        llh = f(x0)
        @test isfinite(llh)

        g = Zygote.gradient(f, x0)[1]
        @test g isa Vector{Float64}
        @test length(g) == length(x0)
        @test all(isfinite, g)
        @test maximum(abs, g) > 0
    end
end

