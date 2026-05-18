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