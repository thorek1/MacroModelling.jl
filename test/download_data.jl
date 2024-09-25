using Downloads
using CSV
using DataFrames
import Dates
using AxisKeys
using Statistics

function download_fred_data(series_names::Vector{String}; 
                            wide_format::Bool=false, 
                            quarterly_only::Bool=true)::DataFrame
    all_data = DataFrame()
    
    for series_name in series_names
        # Construct the URL for the FRED series data
        url = "https://fred.stlouisfed.org/series/$series_name/downloaddata/$series_name.csv"
        
        # Download the CSV data from the URL
        http_response = Downloads.download(url)
        
        # Convert the downloaded data into a DataFrame
        data = CSV.read(http_response, DataFrame)
        
        # Add a column for the series name
        data[!, :series_name] .= series_name
        
        # Append to the final DataFrame
        all_data = vcat(all_data, data)
    end
    
    # Sort the DataFrame by date
    sort!(all_data, [:series_name, :DATE])

    if quarterly_only
        # Filter to include only quarterly data (1st of Jan, Apr, Jul, Oct)
        quarterly_dates = filter(row -> Dates.month(row.DATE) in [1, 4, 7, 10], all_data)
        all_data = quarterly_dates
    end

    if wide_format
        # Pivot the DataFrame to wide format
        # all_data_wide = unstack(all_data, :DATE, :series_name, :VALUE)
        all_data_wide = unstack(all_data, :series_name, :DATE, :VALUE)
        return all_data_wide
    else
        return all_data
    end
end

# https://users.cla.umn.edu/~erm/data/sr409/sw_orig/readme.pdf
# https://fredaccount.stlouisfed.org/public/datalist/803
# https://www.wiwi.uni-bonn.de/bgsepapers/boncrc/CRCTR224_2022_356.pdf   p37
# https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf  p48

series = ["GDPC1","CE16OV","PCEC","FEDFUNDS","FPI","GDPDEF","PRS85006023","COMPNFB","CNP16OV"] 
# "PCECC96","PRS84006023","COMPRNFB","B007RG3Q086SBEA","PRS85006152","PRS84006152","HOABS","AWHNONAG","HOANBS","GDPC96"
# Example usage:
df = download_fred_data(series, wide_format = false, quarterly_only = true)


all_data_wide = unstack(df, :DATE, :series_name, :VALUE)
sort!(all_data_wide, :DATE)


all_data_wide.interest_rate = all_data_wide.FEDFUNDS / 4

all_data_wide.real_GDP_per_capita = all_data_wide.GDPC1 ./ all_data_wide.CNP16OV
all_data_wide.real_GDP_per_capita_growth = [missing; diff(log.(all_data_wide.real_GDP_per_capita)) * 100]

all_data_wide.inflation = [missing; diff(log.(all_data_wide.GDPDEF)) * 100]

all_data_wide.real_investment_per_capita = all_data_wide.FPI ./ all_data_wide.GDPDEF ./ all_data_wide.CNP16OV
all_data_wide.real_investment_per_capita_growth = [missing; diff(log.(all_data_wide.real_investment_per_capita)) * 100]

all_data_wide.real_consumption_per_capita = all_data_wide.PCEC ./ all_data_wide.GDPDEF ./ all_data_wide.CNP16OV
all_data_wide.real_consumption_per_capita_growth = [missing; diff(log.(all_data_wide.real_consumption_per_capita)) * 100]

all_data_wide.hours_worked = log.(all_data_wide.PRS85006023 .* all_data_wide.CE16OV ./ all_data_wide.CNP16OV) .* 100
all_data_wide.hours_worked .-= all_data_wide.hours_worked[1:232] |> skipmissing |> mean

all_data_wide.real_wage_per_capita = all_data_wide.COMPNFB ./ all_data_wide.GDPDEF
all_data_wide.real_wage_per_capita_growth = [missing; diff(log.(all_data_wide.real_wage_per_capita)) * 100]
# this series is quite different for past values


subset_data_wide = all_data_wide[:,[:DATE, 
                                    :real_GDP_per_capita_growth, 
                                    :real_consumption_per_capita_growth, 
                                    :real_investment_per_capita_growth, 
                                    :hours_worked, 
                                    :inflation, 
                                    :real_wage_per_capita_growth,
                                    :interest_rate
                                    ]]

complete_subset_data_wide = subset_data_wide[completecases(subset_data_wide),:]

CSV.write("us_data.csv", complete_subset_data_wide)

data = KeyedArray(Float64.(Matrix(complete_subset_data_wide[:, Not(:DATE)])'), Variable = Symbol.(names(complete_subset_data_wide)[2:end]), Time = complete_subset_data_wide[:, :DATE])


# declare observables as written in model
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

data = rekey(data, :Variable => observables)

## Check with original file
# # load data
# dat = CSV.read("test/data/usmodel.csv", DataFrame)

# # load data
# data1 = KeyedArray(Array(dat)',Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

# # declare observables as written in csv file
# observables_old = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

# data1 = data1(observables_old, :)

# # declare observables as written in model
# observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

# data1 = rekey(data1, :Variable => observables)


# using StatsPlots
# using LinearAlgebra

# for i in 1:7
#     p=plot(data1[i,29:end], title = axiskeys(data,1)[i])
#     plot!(p,data[i,<(Dates.Date("2005-01-01"))])

#     display(p)

#     relative_difference = norm(collect(data1[i,29:end]) .- collect(data[i,<(Dates.Date("2005-01-01"))])) / max(norm(data[i,<(Dates.Date("2005-01-01"))]), norm(data1[i,29:end]))

#     println("Relative difference for series $(axiskeys(data,1)[i]): $relative_difference")
# end


# all_data_wide.hours_worked = all_data_wide.PRS84006023 .* all_data_wide.CE16OV ./ all_data_wide.CNP16OV
# all_data_wide.hours_worked = all_data_wide.AWHNONAG .* all_data_wide.CE16OV ./ all_data_wide.CNP16OV
# all_data_wide.real_wage_index_per_capita = all_data_wide.COMPRNFB
# all_data_wide.real_wage_per_capita = all_data_wide.COMPNFB ./ all_data_wide.GDPDEF
# all_data_wide.real_wage_index_per_capita = all_data_wide.CE16OV .* all_data_wide.COMPRNFB ./ all_data_wide.CNP16OV
# all_data_wide.real_wage_per_capita_growth = [missing; diff(all_data_wide.real_wage_per_capita) ./ all_data_wide.real_wage_per_capita[2:end] * 100]

# using StatsPlots
# plot(all_data_wide.real_wage_per_capita_growth[1:230])
# plot(all_data_wide.hours_worked[1:230] .- 415)
# plot((all_data_wide.hours_worked) .- 415)
# plot(all_data_wide.interest_rate)
# plot(all_data_wide.real_GDP_per_capita_growth[1:230])
# plot(all_data_wide.real_investment)
# plot(all_data_wide.real_consumption_per_capita_growth[1:230])
# plot(all_data_wide.inflation)

# (all_data_wide.hours_worked[5:230] |> sum) / 226
# (all_data_wide.hours_worked[5:end-1] |> sum) / 306

# dy[0] = ctrend + 100 * (y[0] / y[-1] - 1) # fine

# dc[0] = ctrend + 100 * (c[0] / c[-1] - 1) # fine

# dinve[0] = ctrend + 100 * (inve[0] / inve[-1] - 1) # fine

# pinfobs[0] = 100 * (pinf[0] - 1) # fine except for some data issues somewhere in the middle

# robs[0] = 100 * (r[0] - 1) # fine

# dwobs[0] = ctrend + 100 * (w[0] / w[-1] - 1)

# labobs[0] = constelab + 100 * (lab[0] / lab[ss] - 1) # fine, except for constant
