
import Dates
using MacroModelling
using Serialization
using StatsPlots
import Turing
import Turing: NUTS, PG, IS, sample, logpdf, Truncated#, Normal, Beta, Gamma, InverseGamma,
using CSV, DataFrames, AxisKeys
# using Zygote, MCMCChains
# using ComponentArrays, Optimization, OptimizationNLopt, OptimizationOptimisers
# import DynamicPPL: logjoint
# import DynamicPPL
# import ChainRulesCore: @ignore_derivatives, ignore_derivatives
# import Pigeons
import Random
Random.seed!(1)
# ]add CSV, DataFrames, Zygote, AxisKeys, MCMCChains, Turing, DynamicPPL, Pigeons, StatsPlots
println("Threads used: ", Threads.nthreads())

smpler = "nuts" #
smple = "original" #
mdl = "linear" # 
chns = 1 # 
scns = 1000

println("Sampler: $smpler")
println("Sample: $smple")
println("Model: $mdl")
println("Chains: $chns")
println("Scans: $scns")


if smple == "extended"
    smpl = "1966Q1-2020Q1"
    sample_idx = 75:291
    dat = CSV.read("test/data/usmodel_extended.csv", DataFrame)
elseif smple == "original"
    smpl = "1966Q1-2004Q4"
    sample_idx = 1:230
    dat = CSV.read("test/data/usmodel.csv", DataFrame)
elseif smple == "full"
    smpl = "1966Q1-2023Q4"
    sample_idx = 75:306
    dat = CSV.read("test/data/usmodel_extended.csv", DataFrame)
end

if mdl == "linear"
    include("../models/Smets_Wouters_2007_linear.jl")
    Smets_Wouters_2007 = Smets_Wouters_2007_linear
elseif mdl == "nonlinear"
    include("../models/Smets_Wouters_2007.jl")
end

# get_solution(SW07_nonlinear)
# get_solution(SW07_nonlinear, algorithm = :quadratic_iteration)
# import MacroTools: postwalk
# postwalk(x -> x isa Expr ? x.args[1] == :^ ? :(NaNMath.pow($(x.args[2:end]...))) : x : x, :((^)((+)(1, (*)(1//100, ctrend)), (*)(-1, csigma))))

# load data
data = KeyedArray(Array(dat)',Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

# declare observables
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs]

# Subsample
# subset observables in data
data = data(observables, sample_idx)

SS(Smets_Wouters_2007)(observables)

SS(Smets_Wouters_2007, parameters = 
[:z_ea        =>   0.451788281662122,
:z_eb         =>   0.242460701013770,
:z_eg         =>   0.520010319208288,
:z_eqs        =>   0.450106906080831,
:z_em         =>   0.239839325484002,
:z_epinf      =>   0.141123850778673,
:z_ew         =>   0.244391601233500,
:crhoa        =>   0.958774095336246,
:crhob        =>   0.182439345125560,
:crhog        =>   0.976161415046499,
:crhoqs       =>   0.709569323873602,
:crhoms       =>   0.127131476313068,
:crhopinf     =>   0.903807340558011,
:crhow        =>   0.971853774024447,
:cmap         =>   0.744871846683131,
:cmaw         =>   0.888145926618249,
:csadjcost    =>   5.48819700906062,
:csigma       =>   1.39519289795144,
:chabb        =>   0.712400635178752,
:cprobw       =>   0.737541323772002,
:csigl        =>   1.91988384168640,
:cprobp       =>   0.656266260297550,
:cindw        =>   0.591998309497386,
:cindp        =>   0.228354019115349,
:czcap        =>   0.547213129238992,
:cfc          =>   1.61497958797633,
:crpi         =>   2.02946740344113,
:crr          =>   0.815324872021385,
:cry          =>   0.0846869053285818,
:crdy         =>   0.222925708063948,
:constepinf   =>   0.817982220538172,
:constebeta   =>   0.160654114713215,
:constelab    =>   -0.103065166985808,
:ctrend       =>   0.432026374810516,
:cgy          =>   0.526121219470843,
:calfa        =>   0.192800456418155,

# :crhoms => 0.02, 
# :crhopinf   => 0.02, 
# :crhow  => 0.02, 
# :cmap   => 0.02, 
# :cmaw => 0.02
])

kalman_prob = get_loglikelihood(Smets_Wouters_2007, data(observables), Smets_Wouters_2007.parameter_values)

# SS(SW07, parameter_derivatives = :constelab)(observables,:)