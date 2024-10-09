using Revise
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

geo = "EA"
# smple = "full"
fltr = :inversion
algo = :second_order
# algo = :pruned_second_order
# algo = :third_order
# algo = :pruned_third_order
# smpls = 2000
priors = "open"#"original"
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

## Load model
include("../models/Smets_Wouters_2007.jl")

SS(Smets_Wouters_2007, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01, :cmap => 0.01, :cmaw => 0.01], derivatives = false)

## Load initial parameters found to be useful from previous runs
if !msrmt_err
    if geo == "EA"
        if priors == "original"
            if algo ∈ [:first_order, :second_order, :third_order]
                init_params = [1.0539813334864006, 1.790429623465026, 0.9612222743988239, 2.245019416637746, 1.8289708051323972, 2.358601316186861, 1.0789156261697992, 0.8568570601272402, 0.5139803858707833, 0.5086054051557977, 0.746210025525195, 0.2216391679710211, 0.19408834744356726, 0.25466129806674787, 0.30052398003282593, 0.24194224912107154, 3.765673288590403, 1.0977662596765847, 0.15765142087879247, 0.5912277797105701, 2.4184271279443204, 0.674638689237271, 0.2907071570692985, 0.2739634874939473, 0.41260499766712827, 2.430495316537354, 1.6681265382886468, 0.6276427598244203, 0.18302843368856803, 0.16329242389824458, 0.6958098238409626, 0.8548003732242354, -6.430609194624573, 0.6565910012337266, 0.6119056581112686, 0.1572641393297738]
                # init_params = [0.9402697855190816, 1.0641239635812076, 0.6011436583516284, 0.8670182620644392, 0.20640860226785046, 0.1856857208217943, 0.5255835232525442, 0.9960962084440508, 0.08160402420965861, 0.8653512010135149, 0.9744072435163497, 0.5291848669042267, 0.8967104272151635, 0.8959743162864988, 0.6058602943584073, 0.7723502513424746, 6.857865255029908, 1.5303217160632798, 0.8555085690933164, 0.8730857626622777, 1.4949673614587475, 0.7886765516330885, 0.5599881504710812, 0.11871254613452384, 0.8148513410110766, 1.2350588282243782, 1.6069552041393589, 0.8917708276180196, 0.06630827109294861, 0.019885519304454532, 0.6071403118071758, 0.2286762642462631, 9.839321503302711, 0.5082099734632451, 0.23876321194697436, 0.17299221069235832]
            else
                init_params = [2.5989822151536535, 0.6578044451220171, 0.9281310531821928, 1.8852968089406528, 0.8186057477776528, 1.624749728115599, 1.369464287917585, 0.41507938838632535, 0.721658080649216, 0.6664692672787901, 0.8514498174369488, 0.4890983016708977, 0.5624499122468842, 0.5414859476317512, 0.18714280089686092, 0.24684006940138484, 8.398955355725677, 1.0006070838317087, 0.12254688311712678, 0.5509201816084156, 2.195979154320753, 0.5881723445941589, 0.8508888247348659, 0.8686808809823324, 0.4526443895917511, 2.1036428455911094, 2.618645870807919, 0.5776965506871476, 0.30373468432509065, 0.0638034232215675, 1.5107565101005929, 0.8497814956065451, -6.561209938565711, 0.5803935558464064, 1.7440087116410714, 0.2280110157450484]
            end
        elseif priors ∈ ["all", "open"]
            if algo ∈ [:first_order, :second_order, :third_order]
                init_params = [0.7634182382527728, 0.1051593631358384, 0.5380206989183525, 1.035247090648674, 0.3474545268183535, 0.18872162635728051, 0.4523188756702942, 0.9644401368733749, 0.9335270233922365, 0.9885668070980861, 0.9684490999461778, 0.22892889166430036, 0.9609347238927085, 0.8563480603419, 0.7604059535773245, 0.25908482910986136, 2.154457596381628, 1.4090318009629395, 0.2772068624512342, 0.732701012535399, -0.10355717771601747, 0.6660073416651998, 0.6923607650084501, 0.5346492654810947, 0.7524181410736868, 1.1383683356266345, 1.7428849969716322, 0.8795707523003566, 0.016368759430859702, 0.1399538642493827, 0.8242234478192142, 0.16528650409117301, 2.582610947255007, 0.5416171646957421, 0.32393603224059236, 0.23639416133278487, 0.01969851344949738, 2.4812097313961896, 0.1909830159681015, 26.106440595605864, 4.183827130216384]
                # init_params = [1.0684367030087487, 0.7413356790824348, 0.651645804415184, 0.909399723044257, 0.3541011741346618, 0.21378397360041512, 0.7655347845840065, 0.9463238660984851, 0.7730496130813369, 0.7538080421583919, 0.5676524690984942, 0.5734812450102826, 0.6411252781905724, 0.7915403743191631, 0.17280602018316096, 0.18056800909027879, 3.335314258445025, 0.5264396196821965, 0.4192536942946363, 0.6693818083526992, 0.825850104046445, 0.8543679796931302, 0.360355717761473, 0.2590608874343108, 0.6940031864762307, 1.0874252276356324, 1.4790255355741952, 0.7522186927508887, 0.03386035566548007, 0.10533741748125172, 1.0188924595382955, 0.47197941113894964, -4.084600344798764, 0.29064934865697245, 0.25728904097993166, 0.1688815982672292, 0.022706560973330424, 2.1060507855204866, 0.17956382791143666, 32.03995589253832, 5.205262088423767]
                # init_params = [1.0539813334864006, 1.790429623465026, 0.9612222743988239, 2.245019416637746, 1.8289708051323972, 2.358601316186861, 1.0789156261697992, 0.8568570601272402, 0.5139803858707833, 0.5086054051557977, 0.746210025525195, 0.2216391679710211, 0.19408834744356726, 0.25466129806674787, 0.30052398003282593, 0.24194224912107154, 3.765673288590403, 1.0977662596765847, 0.15765142087879247, 0.5912277797105701, 2.4184271279443204, 0.674638689237271, 0.2907071570692985, 0.2739634874939473, 0.41260499766712827, 2.430495316537354, 1.6681265382886468, 0.6276427598244203, 0.18302843368856803, 0.16329242389824458, 0.6958098238409626, 0.8548003732242354, -6.430609194624573, 0.6565910012337266, 0.6119056581112686, 0.1572641393297738, 0.025, 1.5, 0.18, 10, 10]
                # init_params = [0.7313106904457178, 0.09700895799848093, 0.5920640699440656, 1.265361545704499, 0.27297506922164955, 0.195953368384533, 0.5442225915559922, 0.9955500976192675, 0.9725651688203177, 0.8866959058966631, 0.37038470114816974, 0.5775705282809267, 0.9873994492006416, 0.890191474044162, 0.5274740205652197, 0.7215671001774427, 5.4994704577984965, 1.2486546742085023, 0.42324160155361734, 0.618867961816553, 0.07760841574345216, 0.6101273580974104, 0.6893261256964629, 0.19027202373062296, 0.8062387668960043, 1.1899316448115973, 2.0445068173053076, 0.8402388639664063, 0.024030692850014076, 0.07557280727782609, 0.7888835120052056, 0.3196699135160988, 16.1094568737249, 0.5609215600824993, 0.3145230845675469, 0.17801096063423938, 0.018195041145224043, 3.050466433163007, 0.1957715762720714, 12.013367486745231, 14.358322495821245]
                # init_params = [1.4096491114728449, 1.276469479507767, 0.8199421197596538, 1.1492490144710634, 0.8570493433823514, 2.104831086575707, 1.3970751593570685, 0.9091701812225143, 0.5561703142710693, 0.5985458295400763, 0.7467908716874889, 0.34119468647156004, 0.26584271843503193, 0.4645085470204899, 0.21999679024742438, 0.1234682520969361, 3.438270946191947, 0.9370990212137913, 0.19858354659530447, 0.7151750386624385, 0.8471731085782387, 0.7866200028088226, 0.3102064673383781, 0.26755220075226094, 0.45264868381611284, 1.1122392496164437, 1.8356356903083675, 0.6920881545468558, 0.042495178087510146, 0.21319527987296272, 0.6674680044330801, 0.5677206677632957, -5.6455030664907415, 0.17797051330607852, 0.3392885166384611, 0.1785819301869544, 0.01521171472071183, 1.8149225637371782, 0.1935894680586804, 16.393321278178554, 5.576132237342907]
            else
                init_params = [2.7749852410138702, 0.8456673107681664, 0.8193867630407516, 1.741416552345618, 0.914872705790209, 1.18695818062067, 1.2528679322551275, 0.40491114421769414, 0.7373768296231142, 0.6909661001343012, 0.8344336581861671, 0.488743166195874, 0.525692918226704, 0.5357304799885511, 0.22046533559816237, 0.21400559014083304, 8.46093342090205, 0.9696483822433017, 0.13091505956911606, 0.5742334844273536, 2.087263881526341, 0.611539849422541, 0.8651508487613061, 0.8662238419365927, 0.44570575381670136, 1.74218169018084, 2.405472755923317, 0.5590205671329227, 0.2589985535732698, 0.05587005191768583, 1.0807045784619862, 0.7570290094634802, -6.591114604426181, 0.7135443306435844, 1.6808911462102745, 0.27736369269355865, 0.02569284966819406, 1.4429570504223437, 0.170773328665838, 10.346943630023226, 10.467680279242824]
                # init_params = [2.5989822151536535, 0.6578044451220171, 0.9281310531821928, 1.8852968089406528, 0.8186057477776528, 1.624749728115599, 1.369464287917585, 0.41507938838632535, 0.721658080649216, 0.6664692672787901, 0.8514498174369488, 0.4890983016708977, 0.5624499122468842, 0.5414859476317512, 0.18714280089686092, 0.24684006940138484, 8.398955355725677, 1.0006070838317087, 0.12254688311712678, 0.5509201816084156, 2.195979154320753, 0.5881723445941589, 0.8508888247348659, 0.8686808809823324, 0.4526443895917511, 2.1036428455911094, 2.618645870807919, 0.5776965506871476, 0.30373468432509065, 0.0638034232215675, 1.5107565101005929, 0.8497814956065451, -6.561209938565711, 0.5803935558464064, 1.7440087116410714, 0.2280110157450484, 0.025, 1.5, 0.18, 10, 10]
                # init_params = [2.4234476767279203, 0.7990666288267709, 0.8136306400120275, 1.892259433528891, 0.8282835303556103, 1.3698087264415044, 1.269302438048303, 0.43335193492928953, 0.732369097969046, 0.6826684995233729, 0.8410480532591216, 0.4701707864131535, 0.5384388773666394, 0.5284920558095482, 0.20876007722463555, 0.237853989339499, 8.43303979995329, 0.9426736660718141, 0.1250251820316393, 0.585409208379986, 2.1552175865500076, 0.587014310766227, 0.8442575339340547, 0.8635171903414101, 0.4407994842182499, 1.833815904510263, 2.425377469485058, 0.5751445401394171, 0.2788927211737701, 0.05939365209736599, 1.2292102874414594, 0.8375375827974578, -6.675720000473695, 0.6331874236197592, 1.7171969712430373, 0.2490816690363107, 0.02377224622650648, 1.5308709555684354, 0.19858467478653455, 10.993324319418779, 10.684914581142905]
            end
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
# init_params = [1.3266536536624582, 0.19834888933864725, 0.8164146806125434, 0.43983175175589523, 0.4288868840783121, 0.21909106271329742, 0.5887924311214535, 0.8767199200222564, 0.8765325582596163, 0.9480636820046106, 0.8485465028640758, 0.12766130856147417, 0.9772163596785366, 0.9737140623437552, 0.8574432726436533, 0.9319338839180734, 4.809237466680534, 1.1352915711738845, 0.43328064314076487, 0.7399417434303438, 0.332704858539258, 0.7310580299866732, 0.7075597434239054, 0.29744607563422165, 0.7952874494399615, 1.273274044369838, 1.8249122287729622, 0.8157471314759212, 0.019247235416889278, 0.10748442246495876, 0.7160521307849335, 0.15161081843619933, -5.04056349410976, 0.42434547340259676, 0.20684410298510936, 0.21802052973575092]

z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa = init_params[1:36]

if priors == "original"
    ctou, clandaw, cg, curvp, curvw = fixed_parameters[1]
elseif priors ∈ ["all", "open"]
    ctou, clandaw, cg, curvp, curvw = init_params[36 .+ (1:5)]
end

if msrmt_err
    z_dy, z_dc, z_dinve, z_pinfobs, z_robs, z_dwobs = init_params[36 + length(fixed_parameters[1]) - 5 .+ (1:6)]

    if labor == "growth"
        z_labobs = fixed_parameters[2][1]
        z_dlabobs = init_params[36 + length(fixed_parameters[1]) - 5 + 6 + length(fixed_parameters[2])]
    elseif labor == "level"
        z_labobs = init_params[36 + length(fixed_parameters[1]) - 5 + 6 + length(fixed_parameters[2])]
        z_dlabobs = fixed_parameters[2][1]
    end

    parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf, z_dy, z_dc, z_dinve, z_pinfobs, z_robs, z_dwobs, z_dlabobs, z_labobs]
else
    parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]
end



SS(Smets_Wouters_2007, parameters = parameters_combined, derivatives = false)

SS(Smets_Wouters_2007, derivatives = false)

SS(Smets_Wouters_2007, derivatives = false, algorithm = algo)

state_estims = get_estimated_variables(Smets_Wouters_2007, data[:,1:15], algorithm = algo, verbose = true)

state_estims = get_estimated_variables(Smets_Wouters_2007, data, algorithm = algo, filter = fltr)

state_estims = get_estimated_variables(Smets_Wouters_2007, data, algorithm = :pruned_second_order)

shck_dcmp = get_shock_decomposition(Smets_Wouters_2007, data[:,1:15], algorithm = algo, verbose = true)

plot_model_estimates(Smets_Wouters_2007, 
                    data, 
                    algorithm = :pruned_second_order, 
                    filter = fltr, 
                    shock_decomposition = true, 
                    variables = :pinf, 
                    plots_per_page = 1, 
                    save_plots = true, 
                    save_plots_format = :png, 
                    show_plots = false)

plot_model_estimates(Smets_Wouters_2007, data[:,1:150], algorithm = :second_order, filter = fltr, shock_decomposition = false)

plot_model_estimates(Smets_Wouters_2007, data, algorithm = :first_order, filter = fltr, shock_decomposition = true)

plot_model_estimates(Smets_Wouters_2007, data, algorithm = :first_order, shock_decomposition = true)

plot_model_estimates(Smets_Wouters_2007, data, algorithm = :first_order, shock_decomposition = false)

plot_model_estimates(Smets_Wouters_2007, data, algorithm = :pruned_second_order, filter = fltr, shock_decomposition = true)

plot_model_estimates(Smets_Wouters_2007, data, algorithm = :pruned_second_order, shock_decomposition = true, variables = :pinf, plots_per_page = 1)

sck_dcmp = get_shock_decomposition(Smets_Wouters_2007, data, algorithm = :pruned_second_order)
sck_dcmp(:pinf,:,:)

plot_model_estimates(Smets_Wouters_2007, data, algorithm = :first_order, 
# filter = fltr, 
shock_decomposition = true, variables = :pinf, plots_per_page = 1, save_plots = true, show_plots = false)

plot_model_estimates(Smets_Wouters_2007, data, algorithm = algo, filter = fltr, shock_decomposition = true, variables = :pinf, plots_per_page = 1, save_plots = true, show_plots = false)

plot_model_estimates(Smets_Wouters_2007, data[:,1:150], algorithm = :second_order, shock_decomposition = false, variables = :pinf, plots_per_page = 1)

plot_model_estimates(Smets_Wouters_2007, data, algorithm = algo, shock_decomposition = true, variables = :pinf, plots_per_page = 1)

plot_model_estimates(Smets_Wouters_2007, collect(data), algorithm = algo, shock_decomposition = true, variables = :pinf, plots_per_page = 1, save_plots = true, show_plots=false)

plot_model_estimates(Smets_Wouters_2007, data, algorithm = algo)

plot_irf(Smets_Wouters_2007, algorithm = algo, shocks = :em, variables = [:dy, :pinfobs])

plot_irf(Smets_Wouters_2007, algorithm = algo, shocks = :em, negative_shock = true)

plot_irf(Smets_Wouters_2007, algorithm = algo, shocks = :em, generalised_irf = true)


algo = :pruned_second_order
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





# plotting
using Dates
dts = dt[1:150]
range(dts[1],dts[end],10)
plot(dts, rand(150), xticks = (dts, Dates.format.(dts, "m/d/yyyy")), rotation=45)

tm_ticks = round.(dts, Year(1)) |> unique
(dts[end] - dts[1]) ÷ 8

dts_range = range(dts[1],dts[end],step = (dts[end] - dts[1]) ÷ 8)
plot(dts, rand(150), xticks = (dts_range, Dates.format.(dts, "uu/yyyy")), rotation=45)
groupedbar!(dts, rand(150,7), bar_position = :stack, lw = 0, lc=:transparent, xticks = (dts_range, Dates.format.(dts, "uu/yyyy")), transparency = .2)


using StatsPlots
using Dates

# Sample Data
dates = [Date(2023, 1, 1), Date(2023, 2, 1), Date(2023, 3, 1)]
category1 = [10, 20, 30]
category2 = [15, 25, 35]

# Prepare data for grouped bar plot
data = hcat(category1, category2)

# Create the grouped bar plot with dates on the x-axis

groupedbar(dates, data, bar_position=:stack, label=["Category 1" "Category 2"])

groupedbar(data, bar_position=:stack, label=["Category 1" "Category 2"], xformatter= x-> begin println(x)
string(dates[min(ceil(Int,x),length(dates))])
end)

groupedbar(dts, rand(150,7), bar_position = :stack, lw = 0, lc=:transparent, xticks = (dts_range, Dates.format.(dts, "uu/yyyy")), transparency = .2)

groupedbar(rand(150,7), bar_position=:stack, xformatter= x-> begin println(x)
string(dts[max(1,min(ceil(Int,x),length(dts)))])
end)