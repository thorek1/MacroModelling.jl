using Downloads
using CSV
using DataFrames
import Dates
using AxisKeys
using Statistics

dat = CSV.read("./Github/MacroModelling.jl/test/data/EA_SW_rawdata.csv", DataFrame, types = Dict(8=>Float64))

dat.interest_rate = dat.shortrate / 4

dat.real_GDP_per_capita = dat.gdp ./ dat.pop
dat.real_GDP_per_capita_growth = [missing; diff(log.(dat.real_GDP_per_capita)) * 100]

dat.inflation = [missing; diff(log.(dat.defgdp)) * 100]

dat.real_investment_per_capita = dat.inves ./ dat.pop .* dat.definves ./ dat.defgdp
dat.real_investment_per_capita_growth = [missing; diff(log.(dat.real_investment_per_capita)) * 100]

dat.real_consumption_per_capita = dat.conso ./ dat.pop .* dat.defconso ./ dat.defgdp
dat.real_consumption_per_capita_growth = [missing; diff(log.(dat.real_consumption_per_capita)) * 100]

dat.real_wage_per_capita = dat.wage ./ dat.hours ./ dat.defgdp
dat.real_wage_per_capita_growth = [missing; diff(log.(dat.real_wage_per_capita)) * 100]

dat.hours_worked = log.(dat.hours ./ dat.pop) .* 100
dat.hours_growth = [missing; diff(log.(dat.hours_worked)) * 100]
dat.hours_worked .-= dat.hours_worked[121:200] |> skipmissing |> mean # avg. between 2000Q1 and 2019Q4

subset_data_wide = dat[:,[:period, 
                        :real_GDP_per_capita_growth, 
                        :real_consumption_per_capita_growth, 
                        :real_investment_per_capita_growth, 
                        :hours_worked, 
                        :hours_growth,
                        :inflation, 
                        :real_wage_per_capita_growth,
                        :interest_rate
                        ]]
                                    
complete_subset_data_wide = subset_data_wide[completecases(subset_data_wide),:]

CSV.write("ea_data.csv", complete_subset_data_wide)

data = KeyedArray(Float64.(Matrix(complete_subset_data_wide[:, Not(:period)])'), 
                    Variable = Symbol.(names(complete_subset_data_wide)[2:end]), 
                    Time = complete_subset_data_wide[:, :period])

# declare observables as written in model
obs = [:dy, :dc, :dinve, 
:labobs, 
:dlabobs,
:pinfobs, :dwobs, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

data = rekey(data, :Variable => obs)
