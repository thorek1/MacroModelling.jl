
# using Revise
using MacroModelling
using Zygote
import Turing
# import MicroCanonicalHMC
import Pigeons
import Turing: NUTS, sample, logpdf, AutoZygote
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL
# import InferenceReport
import Dates
using Serialization
using StatsPlots

using LinearAlgebra
BLAS.set_num_threads(Threads.nthreads() ÷ 2)

println("Threads used: ", Threads.nthreads())
println("BLAS threads used: ", BLAS.get_num_threads())

geo = try ENV["geo"] catch 
    "US" 
end

priors = try ENV["priors"] catch 
    "original" 
end

smple = try ENV["sample"] catch
    "original" 
end

smplr = try ENV["sampler"] catch
    "NUTS" 
end

fltr = try Symbol(ENV["filter"]) catch
    :kalman
end

algo = try Symbol(ENV["algorithm"]) catch 
    :first_order
end 

smpls = try Meta.parse(ENV["samples"]) catch 
    1000
end

labor = try ENV["labor"] catch 
    "level" 
end 

rnds = try Meta.parse(ENV["rounds"]) catch
    4
end

msrmt_err = try Meta.parse(ENV["measurement_error"]) catch 
    # if fltr == :inversion || !(algo == :first_order)
    #     true
    # else
        false
    # end
end 

geo = "EA"
# smple = "full"
# fltr = :inversion
algo = :second_order
# smpls = 2000
priors = "open"#"original"
# labor = "growth"
# msrmt_err = true
smplr = "pigeons"
rnds = 10
if !(pwd() == "/home/cdsw") cd("/home/cdsw") end


## Load data
if geo == "EA"
    include("download_EA_data.jl") # 1970Q4 - 2024Q2
    if smple == "original"
        # subset observables in data
        sample_idx = 78:size(data,2) # 1990Q1-2024Q4

        data = data[:, sample_idx]
    # elseif smple == "full" # 1970Q4 - 2024Q2
    end
elseif geo == "US"
    if smple == "original"
        # load data
        dat = CSV.read("./Github/MacroModelling.jl/test/data/usmodel.csv", DataFrame)
        dat.dlabobs = [0.0; diff(dat.labobs)] # 0.0 is wrong but i get a float64 matrix

        # load data
        data = KeyedArray(Array(dat)',Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

        # declare observables as written in csv file
        observables_old = [:dy, :dc, :dinve, :labobs, :dlabobs, :pinfobs, :dw, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

        # Subsample
        # subset observables in data
        sample_idx = 47:230 # 1960Q1-2004Q4

        data = data(observables_old, sample_idx)

        # declare observables as written in model
        observables_new = [:dy, :dc, :dinve, :labobs, :dlabobs, :pinfobs, :dwobs, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

        data = rekey(data, :Variable => observables_new)
    elseif smple == "original_new_data" # 1960Q1 - 2004Q4
        include("download_US_data.jl") 
        data = data[:,Interval(Dates.Date("1960-01-01"), Dates.Date("2004-10-01"))]
    elseif smple == "full" # 1954Q4 - 2024Q2
        include("download_US_data.jl") 
    elseif smple == "no_pandemic" # 1954Q4 - 2020Q1
        include("download_US_data.jl") 
        data = data[:,<(Dates.Date("2020-04-01"))]
    elseif smple == "update" # 1960Q1 - 2024Q2
        include("download_US_data.jl") 
        data = data[:,>(Dates.Date("1959-10-01"))]
    end
end

data = data(observables)

## Load model
if msrmt_err
    include("../models/Smets_Wouters_2007_estim_measurement_errors.jl")
else
    include("../models/Smets_Wouters_2007_estim.jl")
end


estimated_parameters = [0.5892333767395813, 0.1383043564523832, 0.5734335837898152, 1.6747138283452097, 0.22753410291407916, 0.23438858806256252, 0.4791193958568236, 0.9757262468832444, 0.9362218801318333, 0.9903083458341554, 0.9880960128100245, 0.21267559166262834, 0.9781519455924899, 0.9131384874977679, 0.3645817399997867, 0.49895915379271527, 0.8458909508102286, 1.0609598880052062, 0.29523699432674916, 0.7244538936848905, 0.8507090560337105, 0.5048905967461411, 0.5921367360318939, 0.5146519588267263, 0.546009006773398, 1.1201054724253559, 2.4512064706409364, 0.8739816739243392, 0.049008985850282724, 0.07668702995055902, 1.0013799190230839, 0.6822690911300185, -6.361462857900299, 0.43121606593839035, 0.3983190599928122, 0.322307218836979, 0.01883568339881853, 2.869571924435338, 0.18203698328536416, 11.877115723345769, 4.122906938256878]
varnames = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]
varnames = vcat(varnames,[:ctou, :clandaw, :cg, :curvp, :curvw])

plot_model_estimates(Smets_Wouters_2007, data, algorithm = algo, parameters = Dict(varnames .=> estimated_parameters))

state_estims = get_estimated_variables(Smets_Wouters_2007, data, algorithm = algo)


# data[:,260:end]
state_dependent_irf_inf = zeros(size(state_estims,2), 40)
i = 1
for c in eachcol(state_estims)
    with_shock = get_irf(Smets_Wouters_2007, initial_state = collect(c), algorithm = algo, shocks = :em, variables = [:pinfobs], negative_shock = true)

    without_shock = get_irf(Smets_Wouters_2007, initial_state = collect(c), algorithm = algo, shocks = :none, variables = [:pinfobs], negative_shock = true)
    
    state_dependent_irf_inf[i,:] = with_shock[1,:,1] - without_shock[1,:,1]
    i += 1
end

surface(state_dependent_irf_inf[end-50:end,1:20]')

plot(state_dependent_irf_inf[105:end,4])

plot(state_dependent_irf_inf[:,4])
data[:,120:125]

states = get_state_variables(Smets_Wouters_2007) .|> Symbol

state_estims(states,120:125)

k = 1 
for i in eachrow(state_estims(states,120:125))
    println("Row: $k has z-value $(sum(abs2,i)/sum(i))")
    k+=1
end
state_estims(states,120:125)[[8],:]
state_estims(states,110:130)[[12],:]' |> plot
state_estims(states,120:125)[[13],:]' |> plot
state_estims(states,110:130)[[22],:]' |> plot

get_mean(Smets_Wouters_2007, algorithm = :pruned_second_order, derivatives = false)(:sw)

plot_solution(Smets_Wouters_2007, :kp, algorithm = algo, variables = [:robs,:pinfobs,:dy,:ygap])
plot_solution(Smets_Wouters_2007, :sw, algorithm = algo, variables = [:robs,:pinfobs,:dy,:ygap])
plot_solution(Smets_Wouters_2007, :r,  algorithm = algo, variables = [:robs,:pinfobs,:dy,:ygap])


state_dependent_irf_gdp = zeros(size(state_estims,2), 40)
i = 1
for c in eachcol(state_estims)
    with_shock = get_irf(Smets_Wouters_2007, initial_state = collect(c), algorithm = algo, shocks = :em, variables = [:dy])

    without_shock = get_irf(Smets_Wouters_2007, initial_state = collect(c), algorithm = algo, shocks = :none, variables = [:dy])
    
    state_dependent_irf_gdp[i,:] = with_shock[1,:,1] - without_shock[1,:,1]
    i += 1
end

surface(state_dependent_irf_gdp[260:end,3:10]')

plot(state_dependent_irf_gdp[1:end,3])



with_shock = get_irf(Smets_Wouters_2007, initial_state = collect(state_estims[:,1]), algorithm = algo, shocks = :em, variables = [:pinfobs])

without_shock = get_irf(Smets_Wouters_2007, initial_state = collect(state_estims[:,1]), algorithm = algo, shocks = :none, variables = [:pinfobs])

state_dependent_irf = with_shock[1,:,1] - without_shock[1,:,1]

plot(net_irf)

