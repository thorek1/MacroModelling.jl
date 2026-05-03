using Dates


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



# function pigeons_flexichain(samples::AbstractArray{<:Real,3}, names::AbstractVector)
#     n_iters, _, n_chains = size(samples)
#     symbol_names = Symbol.(collect(names))
#     data = OrderedDict{FlexiChains.ParameterOrExtra{Symbol}, Matrix{eltype(samples)}}()
#     for (column, name) in pairs(symbol_names)
#         key = name == :log_density ? Extra(name) : Parameter(name)
#         data[key] = Matrix(@view samples[:, column, :])
#     end
#     return FlexiChain{Symbol}(n_iters, n_chains, data)
# end
