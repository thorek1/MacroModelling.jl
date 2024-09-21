using MacroModelling
using Zygote
import Turing
# import Pigeons
import Turing: NUTS, sample, logpdf, AutoZygote
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL

using LinearAlgebra
BLAS.set_num_threads(Threads.nthreads() ÷ 2)

println("Threads used: ", Threads.nthreads())
println("BLAS threads used: ", BLAS.get_num_threads())

# smpler = ENV["sampler"] # "pigeons" #
# smple = ENV["sample"] # "original" #
# mdl = ENV["model"] # "linear" # 
# fltr = ENV["filter"] # "kalman" # 
algo = ENV["algorithm"] # "kalman" # 
# chns = Meta.parse(ENV["chains"]) # "4" # 
# scns = Meta.parse(ENV["scans"]) # "4" # 
smpls = Meta.parse(ENV["samples"]) # "4" # 

# println("Sampler: $smpler")
println("Samples: $smpls")
# println("Model: $mdl")
# println("Chains: $chns")
# println("Filter: $fltr")
println("Algorithm: $algo")
# println("Scans: $scns")
println(pwd())

# load data
dat = CSV.read("./Github/MacroModelling.jl/test/data/usmodel.csv", DataFrame)

# load data
data = KeyedArray(Array(dat)',Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

# declare observables as written in csv file
observables_old = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

# Subsample
# subset observables in data
sample_idx = 47:230 # 1960Q1-2004Q4

data = data(observables_old, sample_idx)

# declare observables as written in model
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

data = rekey(data, :Variable => observables)


# Handling distributions with varying parameters using arraydist
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

Turing.@model function SW07_loglikelihood_function(data, m, observables, fixed_parameters, algorithm)
    all_params ~ Turing.arraydist(dists)

    z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa = all_params

    ctou, clandaw, cg, curvp, curvw = fixed_parameters

    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext() 
        parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]

        llh = get_loglikelihood(m, data(observables), parameters_combined, 
                                # presample_periods = 4, initial_covariance = :diagonal, 
                                algorithm = algorithm)

        Turing.@addlogprob! llh
    end
end

# estimate nonlinear model
include("../models/Smets_Wouters_2007.jl")

fixed_parameters = Smets_Wouters_2007.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw], Smets_Wouters_2007.parameters)]

SS(Smets_Wouters_2007, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01, :cmap => 0.01, :cmaw => 0.01])(observables,:)

SW07_loglikelihood_short = SW07_loglikelihood_function(data[:,150:end], Smets_Wouters_2007, observables, fixed_parameters, Symbol(algo))

SW07_loglikelihood_middle = SW07_loglikelihood_function(data[:,100:end], Smets_Wouters_2007, observables, fixed_parameters, Symbol(algo))

SW07_loglikelihood = SW07_loglikelihood_function(data, Smets_Wouters_2007, observables, fixed_parameters, Symbol(algo))

# modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
#                                         Optim.SimulatedAnnealing())

modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood_short, 
                                        Optim.NelderMead())

println("Mode loglikelihood (short sample): $(modeSW2007.lp)")

modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood_middle, 
                                        Optim.NelderMead(),
                                        initial_params = modeSW2007.values)

println("Mode loglikelihood (middle sample): $(modeSW2007.lp)")

modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
                                        Optim.NelderMead(),
                                        initial_params = modeSW2007.values)

println("Mode loglikelihood (long sample - first try): $(modeSW2007.lp)")

modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
                                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)),
                                        adtype = AutoZygote(),
                                        initial_params = modeSW2007.values)

println("Mode variable values: $(modeSW2007.values); Mode loglikelihood: $(modeSW2007.lp)")

if !isfinite(modeSW2007.lp)
    init_params = [0.7248701429457346, 0.31875152313990046, 0.7651867698318467, 0.6620223369539874, 0.316885143386958, 0.18643344917495216, 0.32397804943356734, 0.9759668805261548, 0.3105313295181706, 0.9706557936572335, 0.723241634807532, 0.17477940951141493, 0.9896361329526329, 0.9762803469828244, 0.8699807995786517, 0.8981055737966092, 4.126271804145603, 1.4997776024972307, 0.6772054796410071, 0.7386629360844853, 2.054189805699832, 0.6547852112595888, 0.5723464191765094, 0.28496127578987235, 0.30894743946090913, 1.5713304554796519, 1.913829101556369, 0.8181038764433114, 0.09967962270593514, 0.17943164999901604, 0.6828556381658487, 0.1456360631102682, 0.535222453466881, 0.4973571394378191, 0.5200653049555408, 0.20176976786183218]
else
    init_params = modeSW2007.values
end

samps = @time Turing.sample(SW07_loglikelihood, 
                            NUTS(adtype = AutoZygote()), 
                            smpls, 
                            progress = true, 
                            initial_params = init_params)

println(samps)
println("Mean variable values: $(mean(samps).nt.mean)")

dir_name = "sw07_$(algo)_$(smpls)_samples"

if !isdir(dir_name) mkdir(dir_name) end

cd(dir_name)

println("Current working directory: ", pwd())

dt = Dates.format(Dates.now(), "yyyy-mm-dd_HH")
serialize("samples_$(dt)h.jls", samps)

my_plot = StatsPlots.plot(samps)
StatsPlots.savefig(my_plot, "samples_$(dt)h.png")
StatsPlots.savefig(my_plot, "../samples_latest.png")

#Base.show(samps)
#println(Base.show(samps))
Base.show(stdout, MIME"text/plain"(), samps)