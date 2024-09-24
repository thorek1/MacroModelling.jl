using MacroModelling
using Zygote
import Turing
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

smple = "full"
fltr = "inversion"
algo = "pruned_second_order"
smpls = 1000

# smpler = ENV["sampler"] # "pigeons" #
smple = ENV["sample"] # "original" #
# mdl = ENV["model"] # "linear" # 
fltr = ENV["filter"] # "kalman" # 
algo = ENV["algorithm"] # "first_order" # 
# chns = Meta.parse(ENV["chains"]) # "4" # 
# scns = Meta.parse(ENV["scans"]) # "4" # 
smpls = Meta.parse(ENV["samples"]) # "4" # 

# println("Sampler: $smpler")
println("Estimation Sample: $smple")
println("Samples: $smpls")
# println("Model: $mdl")
# println("Chains: $chns")
println("Filter: $fltr")
println("Algorithm: $algo")
# println("Scans: $scns")
println(pwd())

if smple == "original"
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
elseif smple == "original_new_data" # 1960Q1 - 2004Q4
    include("download_data.jl") 
    data = data[:,Interval(Dates.Date("1960-01-01"), Dates.Date("2004-10-01"))]
elseif smple == "full" # 1954Q4 - 2024Q2
    include("download_data.jl") 
elseif smple == "no_pandemic" # 1954Q4 - 2020Q1
    include("download_data.jl") 
    data = data[:,<(Dates.Date("2020-04-01"))]
elseif smple == "update" # 1960Q1 - 2024Q2
    include("download_data.jl") 
    data = data[:,>(Dates.Date("1959-10-01"))]
end


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

Turing.@model function SW07_loglikelihood_function(data, m, observables, fixed_parameters, algorithm, filter)
    all_params ~ Turing.arraydist(dists)

    z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa = all_params

    ctou, clandaw, cg, curvp, curvw = fixed_parameters

    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext() 
        parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]

        llh = get_loglikelihood(m, data(observables), parameters_combined, 
                                filter = filter,
                                # presample_periods = 4, initial_covariance = :diagonal, 
                                algorithm = algorithm)

        Turing.@addlogprob! llh
    end
end

# estimate nonlinear model
include("../models/Smets_Wouters_2007.jl")

fixed_parameters = Smets_Wouters_2007.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw], Smets_Wouters_2007.parameters)]

SS(Smets_Wouters_2007, 
    parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01, :cmap => 0.01, :cmaw => 0.01], 
    derivatives = false) #(observables,:)

SW07_loglikelihood_short = SW07_loglikelihood_function(data[:,1:50], Smets_Wouters_2007, observables, fixed_parameters, Symbol(algo), Symbol(fltr))

init_params = [0.7336875859606048, 0.29635793589906373, 0.7885127259084761, 0.627589469317792, 0.33279740477340736, 0.18071963802733587, 0.3136301504425597, 0.9716682344909233, 0.4212412387517263, 0.9714124687443985, 0.7670313366057671, 0.21992355568749347, 0.9803757371138285, 0.9515717168366892, 0.8107798813529196, 0.8197332384776822, 4.178359103188282, 1.3532376868676554, 0.6728358960879459, 0.7234419205397767, 2.1069886214607654, 0.6485994561807753, 0.5726327821945625, 0.2750069399589326, 0.37392325592474474, 1.5618305945104742, 1.9566618052670857, 0.7982525395708631, 0.10123079234482239, 0.18389934495956148, 0.7110007143377491, 0.1734645242219083, 0.26768709319895306, 0.5067914157714182, 0.5356423461417883, 0.19654471991293343]
    
modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood_short, 
                                        Optim.NelderMead(),
                                        initial_params = init_params)


for t in 50:25:size(data,2)
    SW07_loglikelihood = SW07_loglikelihood_function(data[:,1:t], Smets_Wouters_2007, observables, fixed_parameters, Symbol(algo), Symbol(fltr))

    global modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
                                            Optim.NelderMead(),
                                            initial_params = modeSW2007.values)
    
    println("Sample up to $t out of $(size(data,2))\nMode variable values:\n$(modeSW2007.values)\nMode loglikelihood: $(modeSW2007.lp)")
end

# modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood_short, 
#                                         Optim.SimulatedAnnealing())

# println("Mode loglikelihood (Simulated Annealing): $(modeSW2007.lp)")

# println("Mode loglikelihood (short sample): $(modeSW2007.lp)")

# modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood_middle, 
#                                         Optim.NelderMead(),
#                                         initial_params = modeSW2007.values)

# println("Mode loglikelihood (middle sample): $(modeSW2007.lp)")

# modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
#                                         Optim.NelderMead(),
#                                         initial_params = modeSW2007.values)

# println("Mode loglikelihood (long sample - first try): $(modeSW2007.lp)")

SW07_loglikelihood = SW07_loglikelihood_function(data, Smets_Wouters_2007, observables, fixed_parameters, Symbol(algo), Symbol(fltr))

modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
                                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)),
                                        adtype = AutoZygote(),
                                        initial_params = modeSW2007.values)

println("Mode variable values: $(modeSW2007.values); Mode loglikelihood: $(modeSW2007.lp)")

if !isfinite(modeSW2007.lp)
    # 1st order   [0.4894876019295857, 0.24788318367812182, 0.5182931118128361, 0.4513465115416531, 0.22526689500106026, 0.15156894421117362, 0.30227045144284304, 0.9694204574216405, 0.19152866619905887, 0.977733247208875, 0.7256812814930197, 0.12609364808227475, 0.9627930764736579, 0.9608723271236311, 0.7918895227761636, 0.9816154316311907, 6.000605704278623, 1.4274383858919004, 0.7334170320281084, 0.8979844503570699, 2.576335249409614, 0.5713813352786876, 0.3081196258828236, 0.31293462158465574, 0.5679755856499227, 1.5761869461822966, 1.9889804684367596, 0.8445471809593218, 0.08869515037144081, 0.21867275755222343, 0.6866872620656241, 0.1445162195125813, 0.3900118440603694, 0.4321094915800651, 0.5525562884207029, 0.19795760655291933]
    init_params = [0.7336875859606048, 0.29635793589906373, 0.7885127259084761, 0.627589469317792, 0.33279740477340736, 0.18071963802733587, 0.3136301504425597, 0.9716682344909233, 0.4212412387517263, 0.9714124687443985, 0.7670313366057671, 0.21992355568749347, 0.9803757371138285, 0.9515717168366892, 0.8107798813529196, 0.8197332384776822, 4.178359103188282, 1.3532376868676554, 0.6728358960879459, 0.7234419205397767, 2.1069886214607654, 0.6485994561807753, 0.5726327821945625, 0.2750069399589326, 0.37392325592474474, 1.5618305945104742, 1.9566618052670857, 0.7982525395708631, 0.10123079234482239, 0.18389934495956148, 0.7110007143377491, 0.1734645242219083, 0.26768709319895306, 0.5067914157714182, 0.5356423461417883, 0.19654471991293343]
    # init_params = [0.7248701429457346, 0.31875152313990046, 0.7651867698318467, 0.6620223369539874, 0.316885143386958, 0.18643344917495216, 0.32397804943356734, 0.9759668805261548, 0.3105313295181706, 0.9706557936572335, 0.723241634807532, 0.17477940951141493, 0.9896361329526329, 0.9762803469828244, 0.8699807995786517, 0.8981055737966092, 4.126271804145603, 1.4997776024972307, 0.6772054796410071, 0.7386629360844853, 2.054189805699832, 0.6547852112595888, 0.5723464191765094, 0.28496127578987235, 0.30894743946090913, 1.5713304554796519, 1.913829101556369, 0.8181038764433114, 0.09967962270593514, 0.17943164999901604, 0.6828556381658487, 0.1456360631102682, 0.535222453466881, 0.4973571394378191, 0.5200653049555408, 0.20176976786183218]
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

varnames = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]
nms = copy(names(samps))
samps = replacenames(samps, Dict(nms[1:length(varnames)] .=> varnames))

dir_name = "sw07_$(algo)_$(smpls)_samples_$(fltr)_filter_$(smple)_sample"

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