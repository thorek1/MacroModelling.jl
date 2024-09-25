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

priors = try ENV["priors"] catch 
    "original" 
end

smple = try ENV["sample"] catch
     "original" 
end

fltr = try ENV["filter"] catch
     "kalman" 
end

algo = try ENV["algorithm"] catch 
    "first_order" 
end 

smpls = try Meta.parse(ENV["samples"]) catch 
    1000
end

# smple = "full"
# fltr = "kalman"#"inversion"
# algo = "pruned_second_order"
# smpls = 1000
# priors = "all"#"original"

# smpler = ENV["sampler"] # "pigeons" #
# mdl = ENV["model"] # "linear" # 
# chns = Meta.parse(ENV["chains"]) # "4" # 
# scns = Meta.parse(ENV["scans"]) # "4" # 

# println("Sampler: $smpler")
# println("Model: $mdl")
# println("Chains: $chns")
# println("Scans: $scns")

println("Priors: $priors")
println("Estimation Sample: $smple")
println("Samples: $smpls")
println("Filter: $fltr")
println("Algorithm: $algo")

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

if priors == "all"
    dists = vcat(dists,[
                        Beta(0.025, 0.005, 0.0025, .05, μσ = true),     # ctou    = 0.025;       % depreciation rate; AER page 592
                        Normal(1.5, 0.5),                               # clandaw = 1.5;         % average wage markup
                        Beta(0.18, 0.01, 0.1, 0.3, μσ = true),          # cg      = 0.18;        % exogenous spending gdp ratio; AER page 592      
                        Gamma(10, 2, μσ = true),                        # curvp   = 10;          % Kimball curvature in the goods market; AER page 592
                        Gamma(10, 2, μσ = true)                        # curvw   = 10;          % Kimball curvature in the labor market; AER page 592
                        ])
end

Turing.@model function SW07_loglikelihood_function(data, m, observables, fixed_parameters, algorithm, filter)
    all_params ~ Turing.arraydist(dists)
    
    if priors == "original"
        z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa = all_params
        
        ctou, clandaw, cg, curvp, curvw = fixed_parameters
    elseif priors == "all"
        z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa, ctou, clandaw, cg, curvp, curvw = all_params
    end
    
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

SW07_loglikelihood_short = SW07_loglikelihood_function(data[:,1:50], Smets_Wouters_2007, observables, fixed_parameters, Symbol(algo), Symbol(fltr))


if priors == "original"
    init_params = [1.2680371911986956, 0.19422730839970156, 0.8164198615725822, 0.830620735939253, 0.4222177288145677, 0.23158456157711735, 0.5571822075182862, 0.9373633692011772, 0.887470499213972, 0.9566106288585369, 0.9807623964505654, 0.10862603585993574, 0.9924614220268047, 0.9968262903012788, 0.8917241421447357, 0.942253368323001, 3.2923846313674403, 1.1502510627415017, 0.39232016579286266, 0.6688620249471491, 0.33809587030742533, 0.7495414227540258, 0.7074829122844127, 0.3612250007066853, 0.8338901114503362, 1.2192605440957358, 1.8234557710893085, 0.8151160974914622, 0.024667051014476388, 0.10298045019430382, 0.6839828089187178, 0.12151983475697328, -4.673610201875958, 0.4220665873237083, 0.2516440238692975, 0.23813252215055805]
elseif priors == "all"
    # init_params = [1.4671411333705768, 0.46344227140065425, 0.5900873571109617, 0.7707360605032056, 0.2473940195676655, 0.20986227059095883, 0.47263890985003354, 0.602213107777379, 0.3992489263711089, 0.424870761635276, 0.6091475220512262, 0.6493471521510309, 0.7783951334071859, 0.718946448785881, 0.5284591997533923, 0.6388458908004027, 5.497750970804447, 2.01168618062827, 0.739676340346658, 0.5384676222948235, 1.7144436884486272, 0.6176676331124016, 0.5663736165312281, 0.384256933339992, 0.412998510277241, 1.4185679379830725, 1.2206351355024725, 0.8466442254037725, 0.20635611488224723, 0.1279070537691375, 0.6415832545112821, 0.11963725207766585, -0.5606694788189568, 0.5740273683158224, 0.8276293149452258, 0.1864850155427461, 0.026913021996649984, 2.41882803238468, 0.1726751347129797, 11.86173163234974, 10.746145708994563]
    init_params = [0.6903311071443926, 0.23619919346626797, 0.4083698788698424, 0.40325989845471827, 0.10143716370378969, 0.18068502675840892, 0.3058916878514176, 0.9955555237550752, 0.26671389363124226, 0.9236689505457307, 0.674731006441746, 0.28408249851850514, 0.9683465610132509, 0.933378182883533, 0.925027986231054, 0.903046143979153, 7.896155496461486, 1.6455318200429114, 0.8087247835156663, 0.6810460886306969, 1.9841461267764804, 0.760636151573123, 0.6382455491225867, 0.19142190630093547, 0.2673044965954749, 1.420182575070305, 1.4691204685875094, 0.9116901265919812, 0.038876128828754096, 0.06853866339690745, 0.5268979929068206, 0.10560163114659846, 0.066430553115568, 0.2533046323843139, 0.4757377383382505, 0.19839420502490226, 0.021640395369881125, 2.421353949984134, 0.18231926720206218, 10.692905462142393, 10.871558909176551]
end

modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood_short, 
                                        Optim.NelderMead(),
                                        initial_params = init_params)


for t in 75:25:size(data,2)
    SW07_loglikelihood = SW07_loglikelihood_function(data[:,1:t], Smets_Wouters_2007, observables, fixed_parameters, Symbol(algo), Symbol(fltr))

    global modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
                                            Optim.NelderMead(),
                                            initial_params = modeSW2007.values)
    
    println("Sample up to $t out of $(size(data,2))\nMode variable values:\n$(modeSW2007.values)\nMode loglikelihood: $(modeSW2007.lp)")
end

# modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood_short, 
#                                         Optim.SimulatedAnnealing())

# println("Mode loglikelihood (Simulated Annealing): $(modeSW2007.lp)")

SW07_loglikelihood = SW07_loglikelihood_function(data, Smets_Wouters_2007, observables, fixed_parameters, Symbol(algo), Symbol(fltr))

modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
                                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)),
                                        adtype = AutoZygote(),
                                        initial_params = modeSW2007.values)

println("Mode variable values: $(modeSW2007.values); Mode loglikelihood: $(modeSW2007.lp)")

if !isfinite(modeSW2007.lp)
    if priors == "original"
        init_params = [0.7336875859606048, 0.29635793589906373, 0.7885127259084761, 0.627589469317792, 0.33279740477340736, 0.18071963802733587, 0.3136301504425597, 0.9716682344909233, 0.4212412387517263, 0.9714124687443985, 0.7670313366057671, 0.21992355568749347, 0.9803757371138285, 0.9515717168366892, 0.8107798813529196, 0.8197332384776822, 4.178359103188282, 1.3532376868676554, 0.6728358960879459, 0.7234419205397767, 2.1069886214607654, 0.6485994561807753, 0.5726327821945625, 0.2750069399589326, 0.37392325592474474, 1.5618305945104742, 1.9566618052670857, 0.7982525395708631, 0.10123079234482239, 0.18389934495956148, 0.7110007143377491, 0.1734645242219083, 0.26768709319895306, 0.5067914157714182, 0.5356423461417883, 0.19654471991293343]
    elseif priors == "all"
        init_params = [0.9529733496330073, 0.42614535606982823, 0.4742577388945689, 0.3716366060389797, 0.24696728630860904, 0.1218130930403994, 0.4365362234507848, 0.9903848613028288, 0.1465634562548954, 0.9158671302404774, 0.663841236340643, 0.0655266169629801, 0.984704283269366, 0.9304890554643118, 0.9265705630036396, 0.9224904712859519, 7.97805743362492, 1.6312629433175978, 0.7999601457959883, 0.8485606954365501, 2.7372427463274724, 0.9031957988375723, 0.643006173765684, 0.13100697840871936, 0.2309504617062573, 1.1274603062499469, 1.2727349627156925, 0.8428930767435494, 0.057023802333110836, 0.07039191727967987, 0.6354356309438446, 0.10195057994063485, 0.054764316413866254, 0.28787999102966416, 0.259743211415718, 0.15927262539033032, 0.022988567748339406, 2.4904127252600885, 0.1829056817570428, 10.689914744490899, 11.178677844919884]
    end
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

if priors == "original"
    varnames = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]
elseif priors == "all"
    varnames = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa, :ctou, :clandaw, :cg, :curvp, :curvw]
end

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