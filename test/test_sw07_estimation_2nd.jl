#TODO: check why Pigeons is so much slower (more allocs) compared to a few days ago; check higher order correctness; check EA data (labor is off, rest is more or less ok)

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
elseif labor == "none"
    observables = [:dy, :dc, :dinve, :pinfobs, :dwobs, :robs]
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
        Normal(1.5, 1.0),                               # clandaw = 1.5;         % average wage markup
        Beta(0.18, 0.01, 0.1, 0.3, μσ = true),          # cg      = 0.18;        % exogenous spending gdp ratio; AER page 592      
        Gamma(10, 5, μσ = true),                        # curvp   = 10;          % Kimball curvature in the goods market; AER page 592
        Gamma(10, 5, μσ = true)                        # curvw   = 10;          % Kimball curvature in the labor market; AER page 592
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


if !msrmt_err
    if geo == "EA"
        if priors == "original"
            init_params = [0.9402697855190816, 1.0641239635812076, 0.6011436583516284, 0.8670182620644392, 0.20640860226785046, 0.1856857208217943, 0.5255835232525442, 0.9960962084440508, 0.08160402420965861, 0.8653512010135149, 0.9744072435163497, 0.5291848669042267, 0.8967104272151635, 0.8959743162864988, 0.6058602943584073, 0.7723502513424746, 6.857865255029908, 1.5303217160632798, 0.8555085690933164, 0.8730857626622777, 1.4949673614587475, 0.7886765516330885, 0.5599881504710812, 0.11871254613452384, 0.8148513410110766, 1.2350588282243782, 1.6069552041393589, 0.8917708276180196, 0.06630827109294861, 0.019885519304454532, 0.6071403118071758, 0.2286762642462631, 9.839321503302711, 0.5082099734632451, 0.23876321194697436, 0.17299221069235832]
        elseif priors ∈ ["all", "open"]
            # init_params = [0.9402697855190816, 1.0641239635812076, 0.6011436583516284, 0.8670182620644392, 0.20640860226785046, 0.1856857208217943, 0.5255835232525442, 0.9960962084440508, 0.08160402420965861, 0.8653512010135149, 0.9744072435163497, 0.5291848669042267, 0.8967104272151635, 0.8959743162864988, 0.6058602943584073, 0.7723502513424746, 6.857865255029908, 1.5303217160632798, 0.8555085690933164, 0.8730857626622777, 1.4949673614587475, 0.7886765516330885, 0.5599881504710812, 0.11871254613452384, 0.8148513410110766, 1.2350588282243782, 1.6069552041393589, 0.8917708276180196, 0.06630827109294861, 0.019885519304454532, 0.6071403118071758, 0.2286762642462631, 9.839321503302711, 0.5082099734632451, 0.23876321194697436, 0.17299221069235832, 0.018760530782850515, 2.2522903639437537, 0.19962605331130512, 13.089980156684542, 17.68418026510776]
            init_params = [0.7313106904457178, 0.09700895799848093, 0.5920640699440656, 1.265361545704499, 0.27297506922164955, 0.195953368384533, 0.5442225915559922, 0.9955500976192675, 0.9725651688203177, 0.8866959058966631, 0.37038470114816974, 0.5775705282809267, 0.9873994492006416, 0.890191474044162, 0.5274740205652197, 0.7215671001774427, 5.4994704577984965, 1.2486546742085023, 0.42324160155361734, 0.618867961816553, 0.07760841574345216, 0.6101273580974104, 0.6893261256964629, 0.19027202373062296, 0.8062387668960043, 1.1899316448115973, 2.0445068173053076, 0.8402388639664063, 0.024030692850014076, 0.07557280727782609, 0.7888835120052056, 0.3196699135160988, 16.1094568737249, 0.5609215600824993, 0.3145230845675469, 0.17801096063423938, 0.018195041145224043, 3.050466433163007, 0.1957715762720714, 12.013367486745231, 14.358322495821245]
        end
    elseif geo == "US"
        if priors == "original"
            init_params = [1.2680371911986956, 0.19422730839970156, 0.8164198615725822, 0.830620735939253, 0.4222177288145677, 0.23158456157711735, 0.5571822075182862, 0.9373633692011772, 0.887470499213972, 0.9566106288585369, 0.9807623964505654, 0.10862603585993574, 0.9924614220268047, 0.9968262903012788, 0.8917241421447357, 0.942253368323001, 3.2923846313674403, 1.1502510627415017, 0.39232016579286266, 0.6688620249471491, 0.33809587030742533, 0.7495414227540258, 0.7074829122844127, 0.3612250007066853, 0.8338901114503362, 1.2192605440957358, 1.8234557710893085, 0.8151160974914622, 0.024667051014476388, 0.10298045019430382, 0.6839828089187178, 0.12151983475697328, -4.673610201875958, 0.4220665873237083, 0.2516440238692975, 0.23813252215055805]
        elseif priors ∈ ["all", "open"]
            # init_params = [1.4671411333705768, 0.46344227140065425, 0.5900873571109617, 0.7707360605032056, 0.2473940195676655, 0.20986227059095883, 0.47263890985003354, 0.602213107777379, 0.3992489263711089, 0.424870761635276, 0.6091475220512262, 0.6493471521510309, 0.7783951334071859, 0.718946448785881, 0.5284591997533923, 0.6388458908004027, 5.497750970804447, 2.01168618062827, 0.739676340346658, 0.5384676222948235, 1.7144436884486272, 0.6176676331124016, 0.5663736165312281, 0.384256933339992, 0.412998510277241, 1.4185679379830725, 1.2206351355024725, 0.8466442254037725, 0.20635611488224723, 0.1279070537691375, 0.6415832545112821, 0.11963725207766585, -0.5606694788189568, 0.5740273683158224, 0.8276293149452258, 0.1864850155427461, 0.026913021996649984, 2.41882803238468, 0.1726751347129797, 11.86173163234974, 10.746145708994563]
            init_params = [0.6903311071443926, 0.23619919346626797, 0.4083698788698424, 0.40325989845471827, 0.10143716370378969, 0.18068502675840892, 0.3058916878514176, 0.9955555237550752, 0.26671389363124226, 0.9236689505457307, 0.674731006441746, 0.28408249851850514, 0.9683465610132509, 0.933378182883533, 0.925027986231054, 0.903046143979153, 7.896155496461486, 1.6455318200429114, 0.8087247835156663, 0.6810460886306969, 1.9841461267764804, 0.760636151573123, 0.6382455491225867, 0.19142190630093547, 0.2673044965954749, 1.420182575070305, 1.4691204685875094, 0.9116901265919812, 0.038876128828754096, 0.06853866339690745, 0.5268979929068206, 0.10560163114659846, 0.066430553115568, 0.2533046323843139, 0.4757377383382505, 0.19839420502490226, 0.021640395369881125, 2.421353949984134, 0.18231926720206218, 10.692905462142393, 10.871558909176551]
        end
    end
end
# init_params = [0.44782103619412217, 0.2744045109202037, 0.4749131045737315, 0.5500013887684249, 0.20797862809478998, 0.16403206635329395, 0.2265429204859803, 0.43525313943541133, 0.2997483599917961, 0.9929064436262415, 0.8910717163839491, 0.32001072751135523, 0.9470322491063278, 0.975570664813966, 0.9093679009913245, 0.9322716703495983, 7.983148082915397, 1.9442481261064553, 0.8765975237541827, 0.7327948767708871, 1.4574975375554169, 0.879978391524078, 0.49747177843774315, 0.16832219162867687, 0.3137327219049285, 1.2454711526959057, 1.5641396410403388, 0.9350501638586194, 0.09977324400428086, 0.10712340142214574, 0.8677773593341505, 0.255711488563984, 0.1575767917768001, 0.5234796158982757, 0.4860719623974521, 0.16549041204343068, 0.01900863729797648, 2.3150642268481483, 0.1865874146108375, 19.373555985211755, 15.149288245159438]

# init_params = [0.5385394820973827, 0.3197180175495133, 0.5601931993394005, 0.6404791252524147, 0.2424743956393476, 0.1888604855414084, 0.2656934310737569, 0.43614085208448194, 0.2924168276337342, 0.9934910232091974, 0.898389230124756, 0.3140630989689976, 0.951502286664987, 0.9778291138619446, 0.9142552081269052, 0.935374453492536, 8.077789204450177, 2.128646028889402, 0.9598880970288731, 0.7870113675994888, 1.407903348898891, 0.9207149636808479, 0.4977393687273778, 0.14873103987877914, 0.2903579447271281, 1.265849679351675, 1.7920551594029805, 0.9425433398382976, 0.09771702491956362, 0.10749885452252637, 1.7800508083595947, 0.26943375031410555, 0.29887846718348093, 0.943799603684421, 0.4574889743375664, 0.06254784879918907, 0.01488851937088571, 2.325954394386562, 0.2141982188195456, 25.14997472497331, 18.00738280331338]
LLH = -1e6


SW07_llh = SW07_loglikelihood_function(data, Smets_Wouters_2007, observables, fixed_parameters, algo, fltr)

# Turing.logjoint(SW07_loglikelihood_function(data[:,1:150], Smets_Wouters_2007, observables, fixed_parameters, algo, fltr), (all_params = init_params,))
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

modeSW2007NM = try Turing.maximum_a_posteriori(SW07_llh, Optim.NelderMead(), initial_params = init_params)#, show_trace = true, iterations = 100)
catch
    1
end

if !(modeSW2007NM == 1) && isfinite(LLH)
    global LLH = modeSW2007NM.lp
    global init_params = modeSW2007NM.values
    println("Nelder Mead: $LLH")
end

modeSW2007 = try Turing.maximum_a_posteriori(SW07_llh, 
                                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)),
                                        adtype = AutoZygote(),
                                        initial_params = init_params)#,
                                        # show_trace = true)
catch
    1
end

if !(modeSW2007 == 1) && isfinite(LLH)
    global LLH = modeSW2007.lp
    global init_params = modeSW2007.values
    println("LBFGS: $LLH")
end

println("Mode variable values: $(init_params); Mode loglikelihood: $(LLH)")


samps = @time Turing.sample(SW07_llh, 
                            # Turing.externalsampler(MicroCanonicalHMC.MCHMC(10_000,.01), adtype = AutoZygote()), # worse quality
                            NUTS(1000, 0.65, adtype = AutoZygote()), 
                            smpls, 
                            initial_params = isfinite(LLH) ? init_params : nothing, 
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