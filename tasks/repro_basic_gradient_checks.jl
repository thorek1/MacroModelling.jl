using MacroModelling
using Random
import LinearAlgebra as LA
import Zygote
import FiniteDifferences
import DifferentiationInterface
import ADTypes

const REPO_ROOT = length(ARGS) >= 1 ? abspath(ARGS[1]) : abspath(joinpath(@__DIR__, ".."))

function worst_relative_error(ad_grad::AbstractVector, fd_grad::AbstractVector)
    rel = similar(ad_grad, Float64)
    @inbounds for i in eachindex(ad_grad, fd_grad)
        scale = max(abs(ad_grad[i]), abs(fd_grad[i]), eps(Float64))
        rel[i] = abs(ad_grad[i] - fd_grad[i]) / scale
    end
    idx = argmax(rel)
    return idx, rel[idx], abs(ad_grad[idx] - fd_grad[idx])
end

function report_gradient_check(model_name, model, observables, fdm)
    Random.seed!(1)
    simulated = simulate(model)
    data = simulated(observables, :, :simulate)

    loglikelihood(x; verbose = false) = get_loglikelihood(model, data, x, verbose = verbose)

    baseline = loglikelihood(model.parameter_values; verbose = true)
    println("model=$model_name baseline_loglikelihood=$baseline")

    mooncake_grad = DifferentiationInterface.gradient(
        x -> loglikelihood(x; verbose = true),
        ADTypes.AutoMooncake(config = nothing),
        model.parameter_values,
    )
    zygote_grad = Zygote.gradient(x -> loglikelihood(x; verbose = true), model.parameter_values)[1]
    fd_grad = FiniteDifferences.grad(fdm, x -> loglikelihood(x), model.parameter_values)[1]

    mooncake_idx, mooncake_rel, mooncake_abs = worst_relative_error(mooncake_grad, fd_grad)
    zygote_idx, zygote_rel, zygote_abs = worst_relative_error(zygote_grad, fd_grad)

    println("model=$model_name mooncake_isapprox=", isapprox(mooncake_grad, fd_grad; rtol = 1e-4))
    println("model=$model_name zygote_isapprox=", isapprox(zygote_grad, fd_grad; rtol = 1e-4))
    println(
        "model=$model_name mooncake_worst idx=$mooncake_idx rel=$mooncake_rel abs=$mooncake_abs ad=$(mooncake_grad[mooncake_idx]) fd=$(fd_grad[mooncake_idx])",
    )
    println(
        "model=$model_name zygote_worst idx=$zygote_idx rel=$zygote_rel abs=$zygote_abs ad=$(zygote_grad[zygote_idx]) fd=$(fd_grad[zygote_idx])",
    )
    println(
        "model=$model_name norms mooncake_fd=$(LA.norm(mooncake_grad - fd_grad)) zygote_fd=$(LA.norm(zygote_grad - fd_grad)) fd=$(LA.norm(fd_grad))",
    )
end

include(joinpath(REPO_ROOT, "models", "QUEST3_2009.jl"))
report_gradient_check("QUEST3_2009", QUEST3_2009, [:outputgap, :inflation, :interest], FiniteDifferences.central_fdm(4, 1, max_range = 1e-5))

include(joinpath(REPO_ROOT, "models", "GNSS_2010.jl"))
report_gradient_check("GNSS_2010", GNSS_2010, [:C, :Y, :D, :BE], FiniteDifferences.forward_fdm(4, 1, max_range = 1e-4))
