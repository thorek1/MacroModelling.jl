using Dates
using AxisKeys


"""
    inject_missing_observations(data) -> KeyedArray

Return a copy of the KeyedArray `data` with a mid-sample mix of partial and
fully-missing periods (NaN-coded), suitable for exercising missing-observation
code paths during estimation tests.  None of the injected gaps sit at the
sample boundaries, so the resulting series can still be filtered and smoothed.

Two periods have a single observable missing (partial missingness) and two or
three periods have ALL observables missing (predict-only filter steps).
Dim names and axis keys of the input are preserved.
"""
function inject_missing_observations(data)
    dat_nan = Matrix{Float64}(collect(data))
    n_obs, n_time = size(dat_nan)
    @assert n_time >= 6 "Need at least 6 periods to inject non-boundary missing observations"

    partial_t1 = max(2, n_time ÷ 4)
    partial_t2 = min(n_time - 1, max(partial_t1 + 1, 3 * n_time ÷ 4))
    full_t1    = clamp(n_time ÷ 2, 2, n_time - 1)
    full_t2    = min(full_t1 + 1, n_time - 1)
    full_t3    = min(full_t1 + 5, n_time - 1)

    dat_nan[1, partial_t1]              = NaN
    dat_nan[min(2, n_obs), partial_t2]  = NaN
    dat_nan[:, full_t1]                .= NaN
    dat_nan[:, full_t2]                .= NaN
    if full_t3 != full_t2 && full_t3 != full_t1
        dat_nan[:, full_t3]            .= NaN
    end

    dn = dimnames(data)
    keys_by_dim = NamedTuple{dn}((collect(axiskeys(data, 1)), axes(dat_nan, 2)))
    return KeyedArray(dat_nan; keys_by_dim...)
end


function blank_outer_periods(data; n_leading::Int = 0, n_trailing::Int = 0)
    dat_nan = Matrix{Float64}(collect(data))
    n_leading > 0 && (dat_nan[:, 1:n_leading] .= NaN)
    n_trailing > 0 && (dat_nan[:, end - n_trailing + 1:end] .= NaN)

    dn = dimnames(data)
    keys_by_dim = NamedTuple{dn}((collect(axiskeys(data, 1)), axes(dat_nan, 2)))
    return KeyedArray(dat_nan; keys_by_dim...)
end


trim_outer_periods(data, n_leading::Int, n_trailing::Int) = data[:, n_leading + 1:size(data, 2) - n_trailing]


function trim_filter_free_shocks(shocks::AbstractMatrix, n_leading::Int, n_trailing::Int, warmup_iterations::Int)
    n_warm = max(warmup_iterations - 1, 0)
    n_visible = size(shocks, 2) - n_warm
    visible_cols = n_warm .+ collect(n_leading + 1:n_visible - n_trailing)
    warm_cols = n_warm > 0 ? collect(1:n_warm) : Int[]
    return shocks[:, vcat(warm_cols, visible_cols)]
end


function boundary_coverage_data(data; n_periods::Int = min(size(data, 2), 12), n_leading::Int = 2, n_trailing::Int = 2)
    @assert n_periods >= max(6, n_leading + n_trailing + 1) "Need at least $(max(6, n_leading + n_trailing + 1)) periods to build boundary-missing coverage."

    base = data[:, 1:n_periods]

    return (base = base,
            missing = inject_missing_observations(base),
            leading = blank_outer_periods(base; n_leading = n_leading),
            trailing = blank_outer_periods(base; n_trailing = n_trailing),
            boundary = blank_outer_periods(base; n_leading = n_leading, n_trailing = n_trailing),
            trimmed_leading = trim_outer_periods(base, n_leading, 0),
            trimmed_trailing = trim_outer_periods(base, 0, n_trailing),
            trimmed_boundary = trim_outer_periods(base, n_leading, n_trailing),
            n_leading = n_leading,
            n_trailing = n_trailing)
end


function check_loglikelihood_boundary_cases(model, coverage, parameter_values;
                                           algorithm::Symbol = :first_order,
                                           filter::Symbol = :kalman,
                                           presample_periods::Int = 0,
                                           initial_covariance::Symbol = :theoretical,
                                           warmup_iterations::Union{Nothing,Int} = nothing,
                                           tol = MacroModelling.Tolerances(),
                                           verbose::Bool = false)
    presample_periods = MacroModelling.normalize_presample_periods(presample_periods, size(coverage.base, 2))
    base_kwargs = (; algorithm = algorithm,
                    filter = filter,
                    presample_periods = presample_periods,
                    initial_covariance = initial_covariance,
                    tol = tol,
                    verbose = verbose)

    @test isfinite(get_loglikelihood(model, coverage.base, parameter_values; base_kwargs...))
    @test isfinite(get_loglikelihood(model, coverage.missing, parameter_values; base_kwargs...))

    if !isnothing(warmup_iterations)
        @test isapprox(get_loglikelihood(model, coverage.base, parameter_values; base_kwargs..., warmup_iterations = 0),
                       get_loglikelihood(model, coverage.base, parameter_values; base_kwargs..., warmup_iterations = 1);
                       rtol = 1e-10, atol = 1e-10)
        @test isfinite(get_loglikelihood(model, coverage.base, parameter_values; base_kwargs..., warmup_iterations = warmup_iterations))
        @test isfinite(get_loglikelihood(model, coverage.missing, parameter_values; base_kwargs..., warmup_iterations = warmup_iterations))
    end

    for (outer_data, trimmed_data) in ((coverage.leading, coverage.trimmed_leading),
                                       (coverage.trailing, coverage.trimmed_trailing),
                                       (coverage.boundary, coverage.trimmed_boundary))
        outer_llh = isnothing(warmup_iterations) ?
            get_loglikelihood(model, outer_data, parameter_values; base_kwargs...) :
            get_loglikelihood(model, outer_data, parameter_values; base_kwargs..., warmup_iterations = warmup_iterations)

        trimmed_llh = isnothing(warmup_iterations) ?
            get_loglikelihood(model, trimmed_data, parameter_values; base_kwargs...) :
            get_loglikelihood(model, trimmed_data, parameter_values; base_kwargs..., warmup_iterations = warmup_iterations)

        @test isapprox(outer_llh, trimmed_llh; rtol = 1e-10, atol = 1e-10)
    end

    return nothing
end


function check_filter_free_boundary_cases(model, coverage, parameter_values;
                                          algorithm::Symbol = :first_order,
                                          warmup_iterations::Int = 0,
                                          measurement_error_std = 0.05)
    n_exo = length(get_shocks(model))
    n_warm = max(warmup_iterations - 1, 0)
    shocks = zeros(n_exo, size(coverage.base, 2) + n_warm)
    kwargs = warmup_iterations > 0 ? (; algorithm = algorithm, warmup_iterations = warmup_iterations) : (; algorithm = algorithm)

    @test isfinite(get_filter_free_loglikelihood(model, coverage.base, parameter_values, shocks, measurement_error_std; kwargs...))
    @test isfinite(get_filter_free_loglikelihood(model, coverage.missing, parameter_values, shocks, measurement_error_std; kwargs...))

    @test isapprox(get_filter_free_loglikelihood(model, coverage.leading, parameter_values, shocks, measurement_error_std; kwargs...),
                   get_filter_free_loglikelihood(model, coverage.trimmed_leading, parameter_values,
                                                 trim_filter_free_shocks(shocks, coverage.n_leading, 0, warmup_iterations),
                                                 measurement_error_std; kwargs...);
                   rtol = 1e-10, atol = 1e-10)
    @test isapprox(get_filter_free_loglikelihood(model, coverage.trailing, parameter_values, shocks, measurement_error_std; kwargs...),
                   get_filter_free_loglikelihood(model, coverage.trimmed_trailing, parameter_values,
                                                 trim_filter_free_shocks(shocks, 0, coverage.n_trailing, warmup_iterations),
                                                 measurement_error_std; kwargs...);
                   rtol = 1e-10, atol = 1e-10)
    @test isapprox(get_filter_free_loglikelihood(model, coverage.boundary, parameter_values, shocks, measurement_error_std; kwargs...),
                   get_filter_free_loglikelihood(model, coverage.trimmed_boundary, parameter_values,
                                                 trim_filter_free_shocks(shocks, coverage.n_leading, coverage.n_trailing, warmup_iterations),
                                                 measurement_error_std; kwargs...);
                   rtol = 1e-10, atol = 1e-10)

    return nothing
end


function maybe_print_loglikelihood(verbose::Bool, llh, dists, all_params)
    verbose || return nothing
    prior_llh = Turing.logpdf(Turing.product_distribution(dists), all_params)
    println("Loglikelihood: $(llh) and prior llh: $(prior_llh) with params $(all_params)")
    return nothing
end

function quarterly_dates(start_date::Date, len::Int)
    dates = Vector{Date}(undef, len)
    current_date = start_date
    for i in 1:len
        dates[i] = current_date
        current_date = current_date + Dates.Month(3)
    end
    return dates
end