#TODO: check why Pigeons is so much slower (more allocs) compared to a few days ago; check higher order correctness; check EA data

# using Revise
using MacroModelling
using Zygote
import Turing
# import MicroCanonicalHMC
# import Pigeons
import Turing: NUTS, sample, logpdf, AutoZygote
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL

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

# cd("/home/cdsw")
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

println(pwd())

if labor == "growth"
    observables = [:dy, :dc, :dinve, :dlabobs, :pinfobs, :dwobs, :robs]
elseif labor == "level"
    observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
end


if geo == "EA"
    include("download_EA_data.jl") # 1970Q4 - 2024Q2
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

# Handling distributions with varying parameters using arraydist
if priors == "open"
    dists = [
    InverseGamma(0.1, 2.0, μσ = true),   # z_ea
    InverseGamma(0.1, 2.0, μσ = true),   # z_eb
    InverseGamma(0.1, 2.0, μσ = true),   # z_eg
    InverseGamma(0.1, 2.0, μσ = true),   # z_eqs
    InverseGamma(0.1, 2.0, μσ = true),   # z_em
    InverseGamma(0.1, 2.0, μσ = true),   # z_epinf
    InverseGamma(0.1, 2.0, μσ = true),   # z_ew
    Beta(0.5, 0.2, μσ = true),        # crhoa
    Beta(0.5, 0.2, μσ = true),        # crhob
    Beta(0.5, 0.2, μσ = true),        # crhog
    Beta(0.5, 0.2, μσ = true),        # crhoqs
    Beta(0.5, 0.2, μσ = true),        # crhoms
    Beta(0.5, 0.2, μσ = true),        # crhopinf
    Beta(0.5, 0.2, μσ = true),        # crhow
    Beta(0.5, 0.2, μσ = true),        # cmap
    Beta(0.5, 0.2, μσ = true),        # cmaw
    Normal(4.0, 1.5),                  # csadjcost
    Normal(1.50,0.375),                  # csigma
    Beta(0.7, 0.1, μσ = true),         # chabb
    Beta(0.5, 0.1, μσ = true),           # cprobw
    Normal(2.0, 0.75),                  # csigl
    Beta(0.5, 0.10, μσ = true),          # cprobp
    Beta(0.5, 0.15, μσ = true),         # cindw
    Beta(0.5, 0.15, μσ = true),         # cindp
    Beta(0.5, 0.15, μσ = true),      # czcap
    Normal(1.25, 0.125),                  # cfc
    Gamma(1.5, 0.25, μσ = true),                    # crpi
    Beta(0.75, 0.10, μσ = true),        # crr
    Gamma(0.125, 0.05, μσ = true),                # cry
    Gamma(0.125, 0.05, μσ = true),                # crdy
    Gamma(0.625, 0.1, μσ = true),         # constepinf
    Gamma(0.25, 0.1, μσ = true),         # constebeta
    Normal(0.0, 2.0),                  # constelab
    Normal(0.4, 0.10),                    # ctrend
    Normal(0.5, 0.25),                   # cgy
    Normal(0.3, 0.05),                   # calfa
    Beta(0.025, 0.005, μσ = true),     # ctou    = 0.025;       % depreciation rate; AER page 592
    Normal(1.5, 1.0),                               # clandaw = 1.5;         % average wage markup
    Beta(0.18, 0.01, μσ = true),          # cg      = 0.18;        % exogenous spending gdp ratio; AER page 592      
    Gamma(10, 5, μσ = true),                        # curvp   = 10;          % Kimball curvature in the goods market; AER page 592
    Gamma(10, 5, μσ = true)                        # curvw   = 10;          % Kimball curvature in the labor market; AER page 592
    ]
else
    dists = [
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_ea
    InverseGamma(0.1, 2.0, 0.025,5.0, μσ = true),   # z_eb
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_eg
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_eqs
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_em
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_epinf
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_ew
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhoa
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhob
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhog
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhoqs
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhoms
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhopinf
    Beta(0.5, 0.2, 0.001,0.9999, μσ = true),        # crhow
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # cmap
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # cmaw
    Normal(4.0, 1.5,   2.0, 15.0),                  # csadjcost
    Normal(1.50,0.375, 0.25, 3.0),                  # csigma
    Beta(0.7, 0.1, 0.001, 0.99, μσ = true),         # chabb
    Beta(0.5, 0.1, 0.3, 0.95, μσ = true),           # cprobw
    Normal(2.0, 0.75, 0.25, 10.0),                  # csigl
    Beta(0.5, 0.10, 0.5, 0.95, μσ = true),          # cprobp
    Beta(0.5, 0.15, 0.01, 0.99, μσ = true),         # cindw
    Beta(0.5, 0.15, 0.01, 0.99, μσ = true),         # cindp
    Beta(0.5, 0.15, 0.01, 0.99999, μσ = true),      # czcap
    Normal(1.25, 0.125, 1.0, 3.0),                  # cfc
    Normal(1.5, 0.25, 1.0, 3.0),                    # crpi
    Beta(0.75, 0.10, 0.5, 0.975, μσ = true),        # crr
    Normal(0.125, 0.05, 0.001, 0.5),                # cry
    Normal(0.125, 0.05, 0.001, 0.5),                # crdy
    Gamma(0.625, 0.1, 0.1, 2.0, μσ = true),         # constepinf
    Gamma(0.25, 0.1, 0.01, 2.0, μσ = true),         # constebeta
    Normal(0.0, 2.0, -10.0, 10.0),                  # constelab
    Normal(0.4, 0.10, 0.1, 0.8),                    # ctrend
    Normal(0.5, 0.25, 0.01, 2.0),                   # cgy
    Normal(0.3, 0.05, 0.01, 1.0),                   # calfa
    ]
end

if priors == "all"
    dists = vcat(dists,[
        Beta(0.025, 0.005, 0.0025, .05, μσ = true),     # ctou    = 0.025;       % depreciation rate; AER page 592
        Normal(1.5, 0.5),                               # clandaw = 1.5;         % average wage markup
        Beta(0.18, 0.01, 0.1, 0.3, μσ = true),          # cg      = 0.18;        % exogenous spending gdp ratio; AER page 592      
        Gamma(10, 2, μσ = true),                        # curvp   = 10;          % Kimball curvature in the goods market; AER page 592
        Gamma(10, 2, μσ = true)                        # curvw   = 10;          % Kimball curvature in the labor market; AER page 592
                        ])
end

if msrmt_err
    dists = vcat(dists,[
        InverseGamma(0.1, 2.0, μσ = true),   # z_dy
        InverseGamma(0.1, 2.0, μσ = true),   # z_dc
        InverseGamma(0.1, 2.0, μσ = true),   # z_dinve
        InverseGamma(0.1, 2.0, μσ = true),   # z_pinfobs
        InverseGamma(0.1, 2.0, μσ = true),   # z_robs
        InverseGamma(0.1, 2.0, μσ = true)    # z_dwobs
                        ])

    if labor == "growth"
        dists = vcat(dists,[InverseGamma(0.1, 2.0, μσ = true)])   # z_dlabobs
    elseif labor == "level"
        dists = vcat(dists,[InverseGamma(0.1, 2.0, μσ = true)])   # z_labobs
    end
end

Turing.@model function SW07_loglikelihood_function(data, m, observables, fixed_parameters, algorithm, filter)
    all_params ~ Turing.arraydist(dists)
    
    z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa = all_params[1:36]

    if priors == "original"
        ctou, clandaw, cg, curvp, curvw = fixed_parameters[1]
    elseif priors ∈ ["all", "open"]
        ctou, clandaw, cg, curvp, curvw = all_params[36 .+ (1:5)]
    end

    if msrmt_err
        z_dy, z_dc, z_dinve, z_pinfobs, z_robs, z_dwobs = all_params[36 + length(fixed_parameters[1]) - 5 .+ (1:6)]

        if labor == "growth"
            z_labobs = fixed_parameters[2][1]
            z_dlabobs = all_params[36 + length(fixed_parameters[1]) - 5 + 6 + length(fixed_parameters[2])]
        elseif labor == "level"
            z_labobs = all_params[36 + length(fixed_parameters[1]) - 5 + 6 + length(fixed_parameters[2])]
            z_dlabobs = fixed_parameters[2][1]
        end

        parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf, z_dy, z_dc, z_dinve, z_pinfobs, z_robs, z_dwobs, z_dlabobs, z_labobs]
    else
        parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]
    end
    
    
    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext() 
        llh = get_loglikelihood(m, data(observables), parameters_combined, 
                                filter = filter,
                                # timer = timer,
                                # presample_periods = 4, initial_covariance = :diagonal, 
                                algorithm = algorithm)

        Turing.@addlogprob! llh
    end
end

# estimate nonlinear model
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


LLH = -1e6
init_params = []

if geo == "EA"
    if !msrmt_err
        if priors == "original"
            init_params = [0.9402697855190816, 1.0641239635812076, 0.6011436583516284, 0.8670182620644392, 0.20640860226785046, 0.1856857208217943, 0.5255835232525442, 0.9960962084440508, 0.08160402420965861, 0.8653512010135149, 0.9744072435163497, 0.5291848669042267, 0.8967104272151635, 0.8959743162864988, 0.6058602943584073, 0.7723502513424746, 6.857865255029908, 1.5303217160632798, 0.8555085690933164, 0.8730857626622777, 1.4949673614587475, 0.7886765516330885, 0.5599881504710812, 0.11871254613452384, 0.8148513410110766, 1.2350588282243782, 1.6069552041393589, 0.8917708276180196, 0.06630827109294861, 0.019885519304454532, 0.6071403118071758, 0.2286762642462631, 9.839321503302711, 0.5082099734632451, 0.23876321194697436, 0.17299221069235832]
        else
            init_params = [0.9402697855190816, 1.0641239635812076, 0.6011436583516284, 0.8670182620644392, 0.20640860226785046, 0.1856857208217943, 0.5255835232525442, 0.9960962084440508, 0.08160402420965861, 0.8653512010135149, 0.9744072435163497, 0.5291848669042267, 0.8967104272151635, 0.8959743162864988, 0.6058602943584073, 0.7723502513424746, 6.857865255029908, 1.5303217160632798, 0.8555085690933164, 0.8730857626622777, 1.4949673614587475, 0.7886765516330885, 0.5599881504710812, 0.11871254613452384, 0.8148513410110766, 1.2350588282243782, 1.6069552041393589, 0.8917708276180196, 0.06630827109294861, 0.019885519304454532, 0.6071403118071758, 0.2286762642462631, 9.839321503302711, 0.5082099734632451, 0.23876321194697436, 0.17299221069235832, 0.025214321221851643, 2.3115002282432133, 0.19484090283654273, 10.832587309591032, 14.19876357891681]
        end
    end
end

for l in range(50, size(data,2), 4)
    l = floor(Int,l)
    
    SW07_llh = SW07_loglikelihood_function(data[:,1:l], Smets_Wouters_2007, observables, fixed_parameters, algo, fltr)

    for i in 1:50
        modeSW2007SA = try 
            if length(init_params) > 0 && isfinite(LLH)
                Turing.maximum_a_posteriori(SW07_llh, Optim.SimulatedAnnealing(), initial_params = init_params)
            else
                Turing.maximum_a_posteriori(SW07_llh, Optim.SimulatedAnnealing())
            end
        catch
            1
        end

        if !(modeSW2007SA == 1)
            global LLH = modeSW2007SA.lp
            global init_params = modeSW2007SA.values
            println("Iter $i, data up to $l, found loglikelihood $LLH from Simulated Annealing")
        end

        modeSW2007NM = try 
            if length(init_params) > 0 && isfinite(LLH)
                Turing.maximum_a_posteriori(SW07_llh, Optim.NelderMead(), initial_params = init_params)
            else
                Turing.maximum_a_posteriori(SW07_llh, Optim.NelderMead())
            end
        catch
            1
        end

        if !(modeSW2007NM == 1)
            global LLH = modeSW2007NM.lp
            global init_params = modeSW2007NM.values
            println("Iter $i, data up to $l, found loglikelihood $LLH from Nelder Mead")
        end

        modeSW2007 = try Turing.maximum_a_posteriori(SW07_llh, 
                                                Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)),
                                                adtype = AutoZygote(),
                                                initial_params = modeSW2007NM.values)
        catch
            1
        end

        if !(modeSW2007 == 1)
            global LLH = modeSW2007.lp
            global init_params = modeSW2007.values
            println("Iter $i, data up to $l, found loglikelihood $LLH from LBFGS")
        end
        
        if LLH > -3000 
            println("Mode variable values: $(init_params); Mode loglikelihood: $(LLH)")
            break
        end
    end
end

SW07_loglikelihood = SW07_loglikelihood_function(data, Smets_Wouters_2007, observables, fixed_parameters, algo, fltr)

samps = @time Turing.sample(SW07_loglikelihood, 
                            # Turing.externalsampler(MicroCanonicalHMC.MCHMC(10_000,.01), adtype = AutoZygote()), # worse quality
                            NUTS(1000, 0.65, adtype = AutoZygote()), 
                            smpls, 
                            initial_params = init_params, 
                            progress = true)

println(samps)
println("Mean variable values: $(mean(samps).nt.mean)")


varnames = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

if priors ∈ ["all", "open"]
    varnames = vcat(varnames,[:ctou, :clandaw, :cg, :curvp, :curvw])
end

if msrmt_err
    varnames = vcat(varnames,[:z_dy, :z_dc, :z_dinve, :z_pinfobs, :z_robs, :z_dwobs])

    if labor == "growth"
        varnames = vcat(varnames,[:z_dlabobs])
        z_dlabobs = all_params[36 + length(fixed_parameters[1]) - 5 + 6 + length(fixed_parameters[2])]
    elseif labor == "level"
        varnames = vcat(varnames,[:z_labobs])
    end
end


nms = copy(names(samps))
samps = replacenames(samps, Dict(nms[1:length(varnames)] .=> varnames))

if !isdir("estimation_results") mkdir("estimation_results") end

cd("estimation_results")

dir_name = "$(geo)_$(algo)__$(smple)_sample__$(priors)_priors__$(fltr)_filter__$(smpls)_samples__$(smplr)_sampler"

if !isdir(dir_name) mkdir(dir_name) end

cd(dir_name)

println("Current working directory: ", pwd())

dt = Dates.format(Dates.now(), "yyyy-mm-dd_HH")
serialize("samples_$(dt)h.jls", samps)

my_plot = StatsPlots.plot(samps)
StatsPlots.savefig(my_plot, "samples_$(dt)h.png")
StatsPlots.savefig(my_plot, "../../samples_latest.png")

Base.show(stdout, MIME"text/plain"(), samps)