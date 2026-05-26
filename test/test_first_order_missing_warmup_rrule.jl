using Test
using MacroModelling
using Random
import ForwardDiff
import Zygote
import FiniteDifferences

const FDM = FiniteDifferences.central_fdm(5, 1)
const RTOL = 1e-5

function make_llh_closure(model, data, base_params, idx, algorithm; kwargs...)
    n = length(base_params)
    pos = zeros(Int, n)
    @inbounds for (k, j) in enumerate(idx)
        pos[j] = k
    end
    return function (θ_subset)
        T = eltype(θ_subset)
        full = map(j -> pos[j] == 0 ? T(base_params[j]) : θ_subset[pos[j]], 1:n)
        return get_loglikelihood(model, data, full;
                                 filter = :inversion,
                                 algorithm = algorithm,
                                 on_failure_loglikelihood = -Inf,
                                 verbose = true,
                                 kwargs...)
    end
end

function ss_perturbed_data(model, observables; periods = 8, σ = 1e-4, seed = 42)
    SS = get_steady_state(model)
    ss_obs = collect(SS(observables, :Steady_state))
    Random.seed!(seed)
    dat = repeat(ss_obs, 1, periods) .+ σ .* randn(length(observables), periods)
    return KeyedArray(dat; Variables = observables, Time = 1:periods)
end

function data_with_missing_observations(data)
    dat_nan = Matrix{Float64}(collect(data))
    n_obs, n_time = size(dat_nan)
    @assert n_time >= 6

    partial_t1 = 2
    partial_t2 = n_time - 1
    full_t1 = n_time ÷ 2
    full_t2 = min(full_t1 + 1, n_time - 1)

    dat_nan[1, partial_t1] = NaN
    dat_nan[min(2, n_obs), partial_t2] = NaN
    dat_nan[:, full_t1] .= NaN
    dat_nan[:, full_t2] .= NaN

    dat_miss = Matrix{Union{Missing, Float64}}(copy(dat_nan))
    @inbounds for j in axes(dat_miss, 2), i in axes(dat_miss, 1)
        if !isfinite(Float64(coalesce(dat_miss[i, j], NaN)))
            dat_miss[i, j] = missing
        end
    end

    return KeyedArray(dat_miss; Variables = collect(axiskeys(data, 1)), Time = axes(dat_miss, 2))
end

include("../models/Gali_2015_chapter_3_nonlinear.jl")
const GALI = Gali_2015_chapter_3_nonlinear
const GALI_OBS_UNDER = [:log_y, :log_W_real]
const GALI_PARAM_SUBSET_NAMES = [:σ, :φ, :ϕᵖⁱ, :ρ_a, :ρ_z, :std_a, :std_z]

function gali_subset_indices()
    pnames = GALI.constants.post_complete_parameters.parameters
    return [findfirst(==(p), pnames) for p in GALI_PARAM_SUBSET_NAMES]
end

@testset "first-order missing warmup rrule" begin
    base_params = copy(GALI.parameter_values)
    p_subset = gali_subset_indices()
    data = ss_perturbed_data(GALI, GALI_OBS_UNDER; periods = 10, σ = 1e-4, seed = 19)
    data_missing = data_with_missing_observations(data)

    f = make_llh_closure(GALI, data_missing, base_params, p_subset, :first_order;
                         warmup_iterations = 2)
    θ = base_params[p_subset]
    llh = f(θ)

    @test isfinite(llh)

    fd_grad = first(FiniteDifferences.grad(FDM, f, θ))
    @test all(isfinite, fd_grad)

    fdiff_grad = ForwardDiff.gradient(f, θ)
    zg_grad, = Zygote.gradient(f, θ)

    @test isapprox(fdiff_grad, fd_grad; rtol = RTOL)
    @test isapprox(zg_grad, fd_grad; rtol = RTOL)
end