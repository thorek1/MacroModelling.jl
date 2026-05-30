using Test
using MacroModelling
import ADTypes
import DifferentiationInterface
import FiniteDifferences
import Mooncake
import LinearAlgebra as ℒ
using Random, DelimitedFiles, AxisKeys

include("test_helpers.jl")

include("../models/FS2000.jl")

# load data
dat, header = readdlm("data/FS2000_data.csv", ',', header = true)
dat = Float64.(dat)
names = vec(header)
data = KeyedArray(dat', Variable = Symbol.("log_".*names), Time = axes(dat, 1))
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names))

# subset observables in data
data = data(observables, :)

function prepend_initial_periods(data, n_periods)
    n_periods <= 0 && return data
    arr = collect(data)
    arr_prepended = hcat(arr[:, 1:n_periods], arr)
    return KeyedArray(arr_prepended, Variable = collect(axiskeys(data, 1)), Time = axes(arr_prepended, 2))
end

function blank_outer_periods(data, n_leading, n_trailing)
    arr = Matrix{Float64}(collect(data))
    n_leading > 0 && (arr[:, 1:n_leading] .= NaN)
    n_trailing > 0 && (arr[:, end-n_trailing+1:end] .= NaN)
    return KeyedArray(arr, Variable = collect(axiskeys(data, 1)), Time = axes(arr, 2))
end


@testset "Kalman with missing observations (mid-sample partial + fully missing periods)" begin
    n_obs   = size(data, 1)
    n_time  = size(data, 2)
    @assert n_time > 60 "FS2000 sample is shorter than expected"

    # Mid-sample mix of partial and fully-missing periods.  Periods 25 and 47
    # have a single observable missing; periods 30, 31 and 60 have ALL
    # observables missing (predict-only Kalman steps).  None at the boundaries.
    dat_nan = Matrix{Float64}(collect(data))
    dat_nan[1, 25]  = NaN                  # partial: drop obs 1
    dat_nan[2, 47]  = NaN                  # partial: drop obs 2
    dat_nan[:, 30] .= NaN                  # fully missing period
    dat_nan[:, 31] .= NaN                  # consecutive fully missing
    dat_nan[:, 60] .= NaN                  # another isolated fully missing

    data_nan = KeyedArray(dat_nan, Variable = collect(axiskeys(data, 1)), Time = axes(dat_nan, 2))

    # Equivalent Missing-typed array
    dat_miss = Matrix{Union{Missing,Float64}}(copy(dat_nan))
    @inbounds for j in axes(dat_miss, 2), i in axes(dat_miss, 1)
        if !isfinite(Float64(coalesce(dat_miss[i, j], NaN)))
            dat_miss[i, j] = missing
        end
    end
    data_missing = KeyedArray(dat_miss, Variable = collect(axiskeys(data, 1)), Time = axes(dat_miss, 2))

    dat_nothing = Matrix{Union{Nothing,Float64}}(copy(dat_nan))
    @inbounds for j in axes(dat_nothing, 2), i in axes(dat_nothing, 1)
        if !isfinite(coalesce(dat_nothing[i, j], NaN))
            dat_nothing[i, j] = nothing
        end
    end
    data_nothing = KeyedArray(dat_nothing, Variable = collect(axiskeys(data, 1)), Time = axes(dat_nothing, 2))

    # 1. forward filter: finite, agrees across NaN/Missing forms, differs from dense.
    ll_dense = get_loglikelihood(FS2000, data, FS2000.parameter_values)
    ll_nan   = get_loglikelihood(FS2000, data_nan, FS2000.parameter_values)
    ll_miss  = get_loglikelihood(FS2000, data_missing, FS2000.parameter_values)
    ll_nothing = get_loglikelihood(FS2000, data_nothing, FS2000.parameter_values)
    @test isfinite(ll_nan)
    @test isapprox(ll_nan, ll_miss)
    @test isapprox(ll_nan, ll_nothing)
    @test ll_nan != ll_dense

    # 2. AD gradient (Mooncake) matches finite differences on the missing-data input.
    back_grad = DifferentiationInterface.gradient(
        x -> get_loglikelihood(FS2000, data_nan, x),
        ADTypes.AutoMooncake(config = nothing),
        FS2000.parameter_values,
    )
    @test all(isfinite, back_grad)

    fin_grad = FiniteDifferences.grad(
        FiniteDifferences.central_fdm(4, 1),
        x -> get_loglikelihood(FS2000, data_nan, x),
        FS2000.parameter_values,
    )[1]
    @test isapprox(back_grad, fin_grad, rtol = 1e-4)

    # 3. smoother runs end-to-end and produces finite, well-shaped output, even
    #    at fully-missing periods (the smoother backfills shocks from the
    #    surrounding observations).
    sd = get_shock_decomposition(FS2000, data_nan)
    @test all(isfinite, collect(sd))
    @test size(sd, 3) == n_time

    sh = get_estimated_shocks(FS2000, data_nan)
    @test all(isfinite, collect(sh))
    @test size(sh, 2) == n_time

    vars = get_estimated_variables(FS2000, data_nan)
    @test all(isfinite, collect(vars))
    @test size(vars, 2) == n_time
end

@testset "Inversion filter with missing observations (mid-sample partial + fully missing periods)" begin
    n_time = size(data, 2)
    @assert n_time > 60 "FS2000 sample is shorter than expected"

    # Same missingness pattern as the Kalman test above: mid-sample partial and
    # fully-missing periods, none at the boundaries.
    dat_nan = Matrix{Float64}(collect(data))
    dat_nan[1, 25]  = NaN
    dat_nan[2, 47]  = NaN
    dat_nan[:, 30] .= NaN
    dat_nan[:, 31] .= NaN
    dat_nan[:, 60] .= NaN
    data_nan = KeyedArray(dat_nan, Variable = collect(axiskeys(data, 1)), Time = axes(dat_nan, 2))

    dat_miss = Matrix{Union{Missing,Float64}}(copy(dat_nan))
    @inbounds for j in axes(dat_miss, 2), i in axes(dat_miss, 1)
        if !isfinite(Float64(coalesce(dat_miss[i, j], NaN)))
            dat_miss[i, j] = missing
        end
    end
    data_missing = KeyedArray(dat_miss, Variable = collect(axiskeys(data, 1)), Time = axes(dat_miss, 2))

    dat_nothing = Matrix{Union{Nothing,Float64}}(copy(dat_nan))
    @inbounds for j in axes(dat_nothing, 2), i in axes(dat_nothing, 1)
        if !isfinite(coalesce(dat_nothing[i, j], NaN))
            dat_nothing[i, j] = nothing
        end
    end
    data_nothing = KeyedArray(dat_nothing, Variable = collect(axiskeys(data, 1)), Time = axes(dat_nothing, 2))

    inversion_algos = [:first_order, :pruned_second_order, :second_order, :pruned_third_order, :third_order]

    for algo in inversion_algos
        ll_dense = get_loglikelihood(FS2000, data,         FS2000.parameter_values; algorithm = algo, filter = :inversion)
        ll_nan   = get_loglikelihood(FS2000, data_nan,     FS2000.parameter_values; algorithm = algo, filter = :inversion)
        ll_miss  = get_loglikelihood(FS2000, data_missing, FS2000.parameter_values; algorithm = algo, filter = :inversion)
        ll_nothing = get_loglikelihood(FS2000, data_nothing, FS2000.parameter_values; algorithm = algo, filter = :inversion)
        @test isfinite(ll_nan)
        @test isapprox(ll_nan, ll_miss)
        @test isapprox(ll_nan, ll_nothing)
        @test ll_nan != ll_dense

        # filter_data_with_model end-to-end via the smoother accessors.
        sh = get_estimated_shocks(FS2000, data_nan; algorithm = algo, filter = :inversion)
        @test all(isfinite, collect(sh))
        @test size(sh, 2) == n_time
    end

    # AD gradients are available for all inversion algorithms in this loop.
    for algo in (:first_order, :pruned_second_order, :second_order, :pruned_third_order, :third_order)
        back_grad = DifferentiationInterface.gradient(
            x -> get_loglikelihood(FS2000, data_nan, x; algorithm = algo, filter = :inversion),
            ADTypes.AutoMooncake(config = nothing),
            FS2000.parameter_values,
        )
        @test all(isfinite, back_grad)

        fin_grad = FiniteDifferences.grad(
            FiniteDifferences.central_fdm(4, 1),
            x -> get_loglikelihood(FS2000, data_nan, x; algorithm = algo, filter = :inversion),
            FS2000.parameter_values,
        )[1]
        @test isapprox(back_grad, fin_grad, rtol = 1e-4)
    end
end

@testset "Higher-order inversion output warmup differs from prepended-data run" begin
    warmup_iterations = 2
    n_warm = warmup_iterations - 1
    higher_order_algos = (:pruned_second_order, :second_order, :pruned_third_order, :third_order)

    dat_nan = Matrix{Float64}(collect(data))
    dat_nan[1, 25]  = NaN
    dat_nan[2, 47]  = NaN
    dat_nan[:, 30] .= NaN
    dat_nan[:, 31] .= NaN
    dat_nan[:, 60] .= NaN
    data_nan = KeyedArray(dat_nan, Variable = collect(axiskeys(data, 1)), Time = axes(dat_nan, 2))

    for (label, dataset) in (("dense", data), ("missing", data_nan))
        dataset_prepended = prepend_initial_periods(dataset, n_warm)

        for algo in higher_order_algos
            sh_warm = get_estimated_shocks(FS2000, dataset;
                                           algorithm = algo,
                                           filter = :inversion,
                                           warmup_iterations = warmup_iterations)
            sh_prepended = get_estimated_shocks(FS2000, dataset_prepended;
                                                algorithm = algo,
                                                filter = :inversion)
            @test maximum(abs.(collect(sh_warm) .- collect(sh_prepended)[:, n_warm+1:end])) > 1e-8

            vars_warm = get_estimated_variables(FS2000, dataset;
                                                algorithm = algo,
                                                filter = :inversion,
                                                warmup_iterations = warmup_iterations)
            vars_prepended = get_estimated_variables(FS2000, dataset_prepended;
                                                     algorithm = algo,
                                                     filter = :inversion)
            @test maximum(abs.(collect(vars_warm) .- collect(vars_prepended)[:, n_warm+1:end])) > 1e-8
        end

        for algo in (:pruned_second_order, :pruned_third_order)
            sd_warm = get_shock_decomposition(FS2000, dataset;
                                              algorithm = algo,
                                              filter = :inversion,
                                              warmup_iterations = warmup_iterations)
            sd_prepended = get_shock_decomposition(FS2000, dataset_prepended;
                                                   algorithm = algo,
                                                   filter = :inversion)
            @test maximum(abs.(collect(sd_warm) .- collect(sd_prepended)[:, :, n_warm+1:end])) > 1e-8
        end
    end
end

@testset "Inversion output warmup one is a no-op" begin
    dat_nan = Matrix{Float64}(collect(data))
    dat_nan[1, 25]  = NaN
    dat_nan[2, 47]  = NaN
    dat_nan[:, 30] .= NaN
    dat_nan[:, 31] .= NaN
    dat_nan[:, 60] .= NaN
    data_nan_local = KeyedArray(dat_nan, Variable = collect(axiskeys(data, 1)), Time = axes(dat_nan, 2))

    for dataset in (data, data_nan_local)
        for algo in (:first_order, :pruned_second_order, :second_order, :pruned_third_order, :third_order)
            shocks_no_warmup = get_estimated_shocks(FS2000, dataset;
                                                    algorithm = algo,
                                                    filter = :inversion,
                                                    warmup_iterations = 0)
            shocks_one_warmup = get_estimated_shocks(FS2000, dataset;
                                                     algorithm = algo,
                                                     filter = :inversion,
                                                     warmup_iterations = 1)
            @test isapprox(collect(shocks_no_warmup), collect(shocks_one_warmup); rtol = 1e-10, atol = 1e-10)

            variables_no_warmup = get_estimated_variables(FS2000, dataset;
                                                          algorithm = algo,
                                                          filter = :inversion,
                                                          warmup_iterations = 0)
            variables_one_warmup = get_estimated_variables(FS2000, dataset;
                                                           algorithm = algo,
                                                           filter = :inversion,
                                                           warmup_iterations = 1)
            @test isapprox(collect(variables_no_warmup), collect(variables_one_warmup); rtol = 1e-10, atol = 1e-10)
        end

        for algo in (:first_order, :pruned_second_order, :pruned_third_order)
            decomposition_no_warmup = get_shock_decomposition(FS2000, dataset;
                                                              algorithm = algo,
                                                              filter = :inversion,
                                                              warmup_iterations = 0)
            decomposition_one_warmup = get_shock_decomposition(FS2000, dataset;
                                                               algorithm = algo,
                                                               filter = :inversion,
                                                               warmup_iterations = 1)
            @test isapprox(collect(decomposition_no_warmup), collect(decomposition_one_warmup); rtol = 1e-10, atol = 1e-10)
        end
    end
end

@testset "Outer fully missing periods are ignored" begin
    warmup_iterations = 2
    outer = blank_outer_periods(data, 2, 3)
    trimmed = data[:, 3:size(data, 2)-3]
    inversion_kwargs = (; algorithm = :pruned_second_order, filter = :inversion, warmup_iterations = warmup_iterations)

    ll_outer = get_loglikelihood(FS2000, outer, FS2000.parameter_values; inversion_kwargs...)
    ll_trimmed = get_loglikelihood(FS2000, trimmed, FS2000.parameter_values; inversion_kwargs...)
    @test isapprox(ll_outer, ll_trimmed; rtol = 1e-10, atol = 1e-10)

    grad_outer = DifferentiationInterface.gradient(
        x -> get_loglikelihood(FS2000, outer, x; inversion_kwargs...),
        ADTypes.AutoMooncake(config = nothing),
        FS2000.parameter_values,
    )
    grad_trimmed = DifferentiationInterface.gradient(
        x -> get_loglikelihood(FS2000, trimmed, x; inversion_kwargs...),
        ADTypes.AutoMooncake(config = nothing),
        FS2000.parameter_values,
    )
    @test isapprox(grad_outer, grad_trimmed; rtol = 1e-9, atol = 1e-9)

    sh_outer = get_estimated_shocks(FS2000, outer; inversion_kwargs...)
    sh_trimmed = get_estimated_shocks(FS2000, trimmed; inversion_kwargs...)
    @test size(sh_outer, 2) == size(trimmed, 2)
    @test isapprox(collect(sh_outer), collect(sh_trimmed); rtol = 1e-10, atol = 1e-10)

    vars_outer = get_estimated_variables(FS2000, outer; inversion_kwargs...)
    vars_trimmed = get_estimated_variables(FS2000, trimmed; inversion_kwargs...)
    @test size(vars_outer, 2) == size(trimmed, 2)
    @test isapprox(collect(vars_outer), collect(vars_trimmed); rtol = 1e-10, atol = 1e-10)

    sd_outer = get_shock_decomposition(FS2000, outer; inversion_kwargs...)
    sd_trimmed = get_shock_decomposition(FS2000, trimmed; inversion_kwargs...)
    @test size(sd_outer, 3) == size(trimmed, 2)
    @test isapprox(collect(sd_outer), collect(sd_trimmed); rtol = 1e-10, atol = 1e-10)

    ll_kalman_outer = get_loglikelihood(FS2000, outer, FS2000.parameter_values)
    ll_kalman_trimmed = get_loglikelihood(FS2000, trimmed, FS2000.parameter_values)
    @test isapprox(ll_kalman_outer, ll_kalman_trimmed; rtol = 1e-10, atol = 1e-10)

    std_outer = get_estimated_variable_standard_deviations(FS2000, outer)
    std_trimmed = get_estimated_variable_standard_deviations(FS2000, trimmed)
    @test size(std_outer, 2) == size(trimmed, 2)
    @test isapprox(collect(std_outer), collect(std_trimmed); rtol = 1e-10, atol = 1e-10)

    all_missing = blank_outer_periods(data, size(data, 2), 0)

    @test iszero(get_loglikelihood(FS2000, all_missing, FS2000.parameter_values; inversion_kwargs...))
    @test size(get_estimated_shocks(FS2000, all_missing; inversion_kwargs...), 2) == 0
    @test size(get_estimated_variables(FS2000, all_missing; inversion_kwargs...), 2) == 0
    @test size(get_shock_decomposition(FS2000, all_missing; inversion_kwargs...), 3) == 0
    @test size(get_estimated_variable_standard_deviations(FS2000, all_missing), 2) == 0
end
