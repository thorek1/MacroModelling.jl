# TODO: 
# check higher order correctness; 
# check measurement error parameters;

# check why test fail for SS - low tols -> fixed it
# check EA data (labor is off, rest is more or less ok)
# check why Pigeons is so much slower (more allocs) compared to a few days ago - because SS criteria were seldomly met and he spent unnecessary time trying to find it
# rewrite inversion_filter to take into account constrained optim problem - done


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
    "EA" 
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

# geo = "EA"
# smple = "full"
# fltr = :inversion
# algo = :second_order
# smpls = 2000
# priors = "open"#"original"
# labor = "growth"
# msrmt_err = true
# smplr = "pigeons"
# rnds = 10
if !(pwd() == "/home/cdsw") cd("/home/cdsw") end
# smpler = ENV["sampler"] # "pigeons" #
# mdl = ENV["model"] # "linear" # 
# chns = Meta.parse(ENV["chains"]) # "4" # 
# scns = Meta.parse(ENV["scans"]) # "4" # 

# println("Model: $mdl")
# println("Chains: $chns")
# println("Scans: $scns")

println("Sampler: $smplr")
println("Economy: $geo")
println("Priors: $priors")
println("Estimation Sample: $smple")
println("Samples: $smpls")
println("Filter: $fltr")
println("Algorithm: $algo")
println("Labor: $labor")
println("Measurement errors: $msrmt_err")
println("Rounds (Pigeons): $rnds")

println(pwd())

## Observables
if labor == "growth"
    observables = [:dy, :dc, :dinve, :dlabobs, :pinfobs, :dwobs, :robs]
elseif labor == "level"
    observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
elseif labor == "none"
    observables = [:dy, :dc, :dinve, :pinfobs, :dwobs, :robs]
end

## Load data
if geo == "EA"
    include("download_EA_data.jl") # 1990Q1 - 2024Q2
    if smple == "original"
        # subset observables in data
        sample_idx = 78:size(data,2) # 1990Q1-2024Q4

        data = data[:, sample_idx]
    elseif smple == "no_pandemic" # 1990Q1 - 2020Q1
        # subset observables in data
        sample_idx = 78:198 # 1990Q1-2020Q1

        data = data[:, sample_idx]
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

fixed_parameters = Vector{Vector{Float64}}(undef,0)

if priors == "original"
    push!(fixed_parameters, Smets_Wouters_2007.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw], Smets_Wouters_2007.parameters)])
elseif priors ∈ ["all", "open"]
    push!(fixed_parameters, Float64[])
end

if msrmt_err
    if labor == "growth"
        push!(fixed_parameters, Smets_Wouters_2007.parameter_values[indexin([:z_labobs], Smets_Wouters_2007.parameters)])
    elseif labor == "level"
        push!(fixed_parameters, Smets_Wouters_2007.parameter_values[indexin([:z_dlabobs], Smets_Wouters_2007.parameters)])
    end
end

## Priors
# Handling distributions with varying parameters using arraydist
# if priors == "open"
    dists = [
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_ea
    InverseGamma(0.1, 2.0, 0.025,5.0, μσ = true),   # z_eb
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_eg
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_eqs
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_em
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_epinf
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_ew
    # Beta(0.5, 0.2, μσ = true),        # crhoa
    # Beta(0.5, 0.2, μσ = true),        # crhob
    # Beta(0.5, 0.2, μσ = true),        # crhog
    # Beta(0.5, 0.2, μσ = true),        # crhoqs
    # Beta(0.5, 0.2, μσ = true),        # crhoms
    # Beta(0.5, 0.2, μσ = true),        # crhopinf
    # Beta(0.5, 0.2, μσ = true),        # crhow
    # Beta(0.5, 0.2, μσ = true),        # cmap
    # Beta(0.5, 0.2, μσ = true),        # cmaw
    # Gamma(4.0, 1.5, μσ = true),                  # csadjcost
    # Normal(1.50, 0.375),                  # csigma
    # Beta(0.7, 0.1, μσ = true),         # chabb
    # Beta(0.5, 0.1, μσ = true),           # cprobw
    # Normal(2.0, 0.75),                  # csigl
    # Beta(0.5, 0.10, μσ = true),          # cprobp
    # Beta(0.5, 0.15, μσ = true),         # cindw
    # Beta(0.5, 0.15, μσ = true),         # cindp
    # Beta(0.5, 0.15, μσ = true),      # czcap
    # Normal(1.25, 0.125),                  # cfc
    # Gamma(1.5, 0.25, μσ = true),                    # crpi
    # Beta(0.75, 0.10, μσ = true),        # crr
    # Gamma(0.125, 0.05, μσ = true),                # cry
    # Gamma(0.125, 0.05, μσ = true),                # crdy
    # Gamma(0.625, 0.1, μσ = true),         # constepinf
    # Gamma(0.25, 0.1, μσ = true),         # constebeta
    # Normal(0.0, 2.0),                  # constelab
    # Normal(0.3, 0.1),                    # ctrend
    # Normal(0.5, 0.25),                   # cgy
    # Beta(0.3, 0.05, μσ = true),                   # calfa
    # Beta(0.025, 0.005, μσ = true),     # ctou    = 0.025;       % depreciation rate; AER page 592
    # Normal(1.5, 1.0),                               # clandaw = 1.5;         % average wage markup
    # Beta(0.18, 0.01, μσ = true),          # cg      = 0.18;        % exogenous spending gdp ratio; AER page 592      
    # Gamma(10, 5, μσ = true),                        # curvp   = 10;          % Kimball curvature in the goods market; AER page 592
    # Gamma(10, 5, μσ = true)                        # curvw   = 10;          % Kimball curvature in the labor market; AER page 592
    ]
# elseif priors == "tighter"
#     dists = [
#     Cauchy(0.0, 2.0, 0.0, 10.0),   # z_ea
#     Cauchy(0.0, 2.0, 0.0, 10.0),   # z_eb
#     Cauchy(0.0, 2.0, 0.0, 10.0),   # z_eg
#     Cauchy(0.0, 2.0, 0.0, 10.0),   # z_eqs
#     Cauchy(0.0, 2.0, 0.0, 10.0),   # z_em
#     Cauchy(0.0, 2.0, 0.0, 10.0),   # z_epinf
#     Cauchy(0.0, 2.0, 0.0, 10.0),   # z_ew
#     Beta(0.5, 0.2, μσ = true),        # crhoa
#     Beta(0.5, 0.2, μσ = true),        # crhob
#     Beta(0.5, 0.2, μσ = true),        # crhog
#     Beta(0.5, 0.2, μσ = true),        # crhoqs
#     Beta(0.5, 0.2, μσ = true),        # crhoms
#     Beta(0.5, 0.2, μσ = true),        # crhopinf
#     Beta(0.5, 0.2, μσ = true),        # crhow
#     Beta(0.5, 0.2, μσ = true),        # cmap
#     Beta(0.5, 0.2, μσ = true),        # cmaw
#     Gamma(4.0, 1.5, μσ = true),                  # csadjcost
#     Normal(1.50, 0.375),                  # csigma
#     Beta(0.7, 0.1, μσ = true),         # chabb
#     Beta(0.5, 0.1, μσ = true),           # cprobw
#     Normal(2.0, 0.75),                  # csigl
#     Beta(0.5, 0.10, μσ = true),          # cprobp
#     Beta(0.5, 0.15, μσ = true),         # cindw
#     Beta(0.5, 0.15, μσ = true),         # cindp
#     Beta(0.5, 0.15, μσ = true),      # czcap
#     Normal(1.25, 0.125),                  # cfc
#     Gamma(1.5, 0.25, μσ = true),                    # crpi
#     Beta(0.75, 0.10, μσ = true),        # crr
#     Gamma(0.125, 0.05, μσ = true),                # cry
#     # Gamma(0.0001, 0.0000001,0.0, 0.000001, μσ = true),                # crdy
#     Gamma(0.475, 0.025, μσ = true),         # constepinf
#     Gamma(0.25, 0.1, μσ = true),         # constebeta
#     Normal(0.0, 2.0),                  # constelab
#     Normal(0.3, 0.1),                    # ctrend
#     Normal(0.5, 0.25),                   # cgy
#     Beta(0.3, 0.05, μσ = true),                   # calfa
#     Beta(0.025, 0.005, μσ = true),     # ctou    = 0.025;       % depreciation rate; AER page 592
#     Normal(1.5, 1.0),                               # clandaw = 1.5;         % average wage markup
#     Beta(0.18, 0.01, μσ = true),          # cg      = 0.18;        % exogenous spending gdp ratio; AER page 592      
#     Gamma(10, 5, μσ = true),                        # curvp   = 10;          % Kimball curvature in the goods market; AER page 592
#     Gamma(10, 5, μσ = true)                        # curvw   = 10;          % Kimball curvature in the labor market; AER page 592
#         ]
# else
#     dists = [
#     InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_ea
#     InverseGamma(0.1, 2.0, 0.025,5.0, μσ = true),   # z_eb
#     InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_eg
#     InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_eqs
#     InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_em
#     InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_epinf
#     InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_ew
#     Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhoa
#     Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhob
#     Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhog
#     Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhoqs
#     Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhoms
#     Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhopinf
#     Beta(0.5, 0.2, 0.001,0.9999, μσ = true),        # crhow
#     Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # cmap
#     Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # cmaw
#     Normal(4.0, 1.5,   2.0, 15.0),                  # csadjcost
#     Normal(1.50,0.375, 0.25, 3.0),                  # csigma
#     Beta(0.7, 0.1, 0.001, 0.99, μσ = true),         # chabb
#     Beta(0.5, 0.1, 0.3, 0.95, μσ = true),           # cprobw
#     Normal(2.0, 0.75, 0.25, 10.0),                  # csigl
#     Beta(0.5, 0.10, 0.5, 0.95, μσ = true),          # cprobp
#     Beta(0.5, 0.15, 0.01, 0.99, μσ = true),         # cindw
#     Beta(0.5, 0.15, 0.01, 0.99, μσ = true),         # cindp
#     Beta(0.5, 0.15, 0.01, 0.99999, μσ = true),      # czcap
#     Normal(1.25, 0.125, 1.0, 3.0),                  # cfc
#     Normal(1.5, 0.25, 1.0, 3.0),                    # crpi
#     Beta(0.75, 0.10, 0.5, 0.975, μσ = true),        # crr
#     Normal(0.125, 0.05, 0.001, 0.5),                # cry
#     Normal(0.125, 0.05, 0.001, 0.5),                # crdy
#     Gamma(0.625, 0.1, 0.1, 2.0, μσ = true),         # constepinf
#     Gamma(0.25, 0.1, 0.01, 2.0, μσ = true),         # constebeta
#     Normal(0.0, 2.0, -10.0, 10.0),                  # constelab
#     Normal(0.4, 0.10, 0.1, 0.8),                    # ctrend
#     Normal(0.5, 0.25, 0.01, 2.0),                   # cgy
#     Normal(0.3, 0.05, 0.01, 1.0),                   # calfa
#     ]
# end

# if priors == "all"
#     dists = vcat(dists,[
#         Beta(0.025, 0.005, 0.0025, .05, μσ = true),     # ctou    = 0.025;       % depreciation rate; AER page 592
#         Normal(1.5, 1.0),                               # clandaw = 1.5;         % average wage markup
#         Beta(0.18, 0.01, 0.1, 0.3, μσ = true),          # cg      = 0.18;        % exogenous spending gdp ratio; AER page 592      
#         Gamma(10, 5, μσ = true),                        # curvp   = 10;          % Kimball curvature in the goods market; AER page 592
#         Gamma(10, 5, μσ = true)                        # curvw   = 10;          % Kimball curvature in the labor market; AER page 592
#                         ])
# end

# if msrmt_err
#     dists = vcat(dists,[
#         Cauchy(0.0, 0.5, 0.0, 10.0),   # z_dy
#         Cauchy(0.0, 0.5, 0.0, 10.0),   # z_dc
#         Cauchy(0.0, 0.5, 0.0, 10.0),   # z_dinve
#         Cauchy(0.0, 0.5, 0.0, 10.0),   # z_pinfobs
#         Cauchy(0.0, 0.5, 0.0, 10.0),   # z_robs
#         Cauchy(0.0, 0.5, 0.0, 10.0)    # z_dwobs
#                         ])

#     if labor == "growth"
#         dists = vcat(dists,[Cauchy(0.0, 0.5, 0.0, 10.0)])   # z_dlabobs
#     elseif labor == "level"
#         dists = vcat(dists,[Cauchy(0.0, 0.5, 0.0, 10.0)])   # z_labobs
#     end
# end

## Turing callback
dir_name1 = "estimation_results"
if !isdir(dir_name1) mkdir(dir_name1) end

# cd(dir_name1)

dir_name = dir_name1 * "/$(geo)_$(algo)__$(smple)_sample__$(priors)_priors__$(fltr)_filter__$(smpls)_samples__$(smplr)_sampler"

if !isdir(dir_name) mkdir(dir_name) end

# cd(dir_name)

# println("Current working directory: ", pwd())


# define callback
# Define the path for the CSV file
dt = Dates.format(Dates.now(), "yyyy-mm-dd_HH")
csv_file_path = dir_name * "/samples_$(dt)h.csv"

# # Initialize a DataFrame to store the data
# df = DataFrame(iteration = Float64[])
# function callback(rng, model, sampler, sample, i)
# function callback(rng, model, sampler, sample, state, i; kwargs...)
function callback(rng, model, sampler, sample, state, i; kwargs...)
    # println(sample)
    df = DataFrame(iteration = Float64[])

    # Prepare a row for the DataFrame
    row = Dict("iteration" => Float64(i))
    for (name, value) in sample.θ
        row[string(name)] = value
    end
    
    # If the DataFrame `df` does not have columns for these names, add them
    for name in keys(row)
        if !any(name .== names(df))
            df[!, name] = Union{Missing, Any}[missing for _ in 1:nrow(df)]
        end
    end
    
    # Append the new data to the DataFrame
    push!(df, row)
    # println(df)
    # Write the updated DataFrame to the CSV file
    # Note: To avoid performance issues, consider writing periodically instead of on every callback
    CSV.write(csv_file_path, df, append=true)
end

Smets_Wouters_2007.parameters

## Turing model definition
# bounds
# lo_bnds = vcat(zeros(7), zeros(7), zeros(18), -30, zeros(3), zeros(5))
# up_bnds = vcat(fill(2, 7), fill(1, 7), fill(1, 2), 30, 10, fill(1, 2), 30, fill(1, 4), 10, 5, 1, 2, 2, 100, 100, 30, 5, 1, 1, 1, 10, 1, 100, 100)

fixed_parameters = [0.9941945010996004, 0.9900258724661307, 0.944447821651772, 0.09136974979681929, 0.5469941169752605, 0.9839879182859345, 0.8176542834158012, 0.46404242788618344, 0.7277828188461039,  6.207074468776051, 0.5342528174391462, 0.560325003881225, 0.6329231385353169, 0.8484146558715042, 0.7618268755139341, 0.7816314780804516, 0.07816721962903334, 0.817115418052766, 0.9812936465960612, 2.2188852317152006, 0.626915938550924, 0.02363305569575591, 0.4237043241955714, 0.28392007131192487, -0.7476344687461959, 0.30542058428439206, 0.5032209567712396, 0.2993769847124837, 0.034103710249185064, 4.119095036926654, 0.19636391880348672, 8.22514103090019, 14.633481496900645]

Turing.@model function SW07_loglikelihood_function(data, m, observables, fixed_parameters, algorithm, filter, dists)
    all_params ~ Turing.arraydist(dists)
    
    z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew = all_params
    
    crdy = 0

    crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, constepinf, constebeta, constelab, ctrend, cgy, calfa, ctou, clandaw, cg, curvp, curvw = fixed_parameters

    parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]
    
    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext() 
        llh = get_loglikelihood(m, data(observables), parameters_combined, 
                                filter = filter,
                                # timer = timer,
                                # presample_periods = 4, initial_covariance = :diagonal, 
                                algorithm = algorithm)

        # if isfinite(llh) println(all_params) end

        Turing.@addlogprob! llh
    end
end

init_params = [0.6078559133318278, 0.06836618238325545, 0.4203898197505046, 1.1088241818556892, 0.6541387441075293, 0.1267035923202942, 0.4528480216201151] #, 0.9941945010996004, 0.9900258724661307, 0.944447821651772, 0.09136974979681929, 0.5469941169752605, 0.9839879182859345, 0.8176542834158012, 0.46404242788618344, 0.7277828188461039]

SW07_llh = SW07_loglikelihood_function(data, Smets_Wouters_2007, observables, fixed_parameters, algo, fltr, dists)

LLH = Turing.logjoint(SW07_llh, (all_params = init_params,))
# Turing.logjoint(SW07_loglikelihood_function(data, Smets_Wouters_2007, observables, fixed_parameters, :pruned_second_order, fltr), (all_params = init_params,))
# Turing.logjoint(SW07_loglikelihood_function(data, Smets_Wouters_2007, observables, fixed_parameters, :first_order, :inversion), (all_params = init_params,))

# using OptimizationNLopt

# modeSW2007NM = try Turing.maximum_a_posteriori(SW07_llh, NLopt.LN_NELDERMEAD(), 
#     # initial_params = init_params, 
#     # maxtime = 4*60^2)
#     maxeval = 10000)#, show_trace = true, iterations = 100)
# catch
#     1
# end

# modeSW2007NM = try Turing.maximum_a_posteriori(SW07_llh, NLopt.LD_LBFGS(), 
#     initial_params = init_params, 
#     adtype = AutoZygote(),
#     maxeval = 100)#, show_trace = true, iterations = 100)
# catch
#     1
# end

if !isfinite(LLH)
    println("Initial values have infinite loglikelihood. Trying to find finite starting point.")
    modeSW2007NM = try Turing.maximum_a_posteriori(SW07_llh, Optim.SimulatedAnnealing(), initial_params = init_params)#, show_trace = true, iterations = 100)
    catch
        1
    end

    if !(modeSW2007NM == 1) && isfinite(modeSW2007NM.lp)
        global LLH = modeSW2007NM.lp
        global init_params = modeSW2007NM.values
        println("Nelder Mead: $LLH")
    end

    if !(smplr == "pigeons")
        modeSW2007 = try Turing.maximum_a_posteriori(SW07_llh, 
                                                Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)),
                                                adtype = AutoZygote(),
                                                initial_params = init_params)#,
                                                # show_trace = true)
        catch
            1
        end

        if !(modeSW2007 == 1) && isfinite(modeSW2007.lp)
            global LLH = modeSW2007.lp
            global init_params = modeSW2007.values
            println("LBFGS: $LLH")
        end
    end
end

println("Mode variable values: $(init_params); Mode loglikelihood: $(LLH)")

if smplr == "NUTS"
    samps = @time Turing.sample(SW07_llh,
                                # Turing.externalsampler(MicroCanonicalHMC.MCHMC(10_000,.01), adtype = AutoZygote()), # worse quality
                                NUTS(1000, 0.65, adtype = AutoZygote()),
                                smpls;
                                initial_params = isfinite(LLH) ? init_params : nothing,
                                progress = true,
                                callback = callback)

    # InferenceReport.report(samps; max_moving_plot_iters = 0, view = false, render = true, exec_folder = "/home/cdsw")
elseif smplr == "pigeons"
    # generate a Pigeons log potential
    sw07_lp = Pigeons.TuringLogPotential(SW07_llh)

    const SW07_LP = typeof(sw07_lp)
    
    function Pigeons.initialization(target::SW07_LP, rng::AbstractRNG, _::Int64)
        result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.SampleFromPrior(), DynamicPPL.PriorContext())
        # DynamicPPL.link!!(result, DynamicPPL.SampleFromPrior(), target.model)
        
        result = DynamicPPL.initialize_parameters!!(result, init_params, DynamicPPL.SampleFromPrior(), target.model)

        return result
    end
    
    cd(dir_name)

    # Get list of folders only (ignoring files)
    # folders = filter(isdir, joinpath.("results/all", readdir("results/all")))
    # latest_folder = folders |> sort |> last

    # full_rounds = length(filter(isdir, joinpath.(latest_folder, readdir(latest_folder))))

    # if full_rounds < rnds
    #     pt = Pigeons.PT(latest_folder)

    #     # do two more rounds of sampling
    #     pt = Pigeons.increment_n_rounds!(pt, 1)
    #     pt = Pigeons.pigeons(pt)
    # else
        pt = Pigeons.pigeons(target = sw07_lp, n_rounds = 0, n_chains = 2)

        pt = Pigeons.pigeons(target = sw07_lp,
                            # explorer = Pigeons.AutoMALA(default_autodiff_backend = :Zygote),
                            checkpoint = true,
                            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default(); Pigeons.disk],
                            multithreaded = false,
                            n_chains = 2,
                            n_rounds = rnds)
    # end

    cd("../..")

    # InferenceReport.report(pt; max_moving_plot_iters = 0, view = false, render = true, exec_folder = "/home/cdsw")

    samps = MCMCChains.Chains(pt)
end


println(samps)
println("Mean variable values: $(mean(samps).nt.mean)")

varnames = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew] #, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw]

nms = copy(names(samps))
samps = replacenames(samps, Dict(nms[1:length(varnames)] .=> varnames))

dt = Dates.format(Dates.now(), "yyyy-mm-dd_HH")
serialize(dir_name * "/samples_$(dt)h.jls", samps)

my_plot = StatsPlots.plot(samps)
StatsPlots.savefig(my_plot, dir_name * "/samples_$(dt)h.png")
StatsPlots.savefig(my_plot, "samples_latest.png")

Base.show(stdout, MIME"text/plain"(), samps)