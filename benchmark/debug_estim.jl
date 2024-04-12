
using MacroModelling
import Turing: NUTS, HMC, PG, IS, sample, logpdf, Truncated#, Normal, Beta, Gamma, InverseGamma,
using CSV, DataFrames, AxisKeys
import Zygote
import ForwardDiff
import FiniteDifferences
import ChainRulesCore: @ignore_derivatives, ignore_derivatives
using Random
import BenchmarkTools: @benchmark
Random.seed!(1)
# ]add CSV, DataFrames, Zygote, AxisKeys, MCMCChains, Turing, DynamicPPL, Pigeons, StatsPlots
println("Threads used: ", Threads.nthreads())

smpler = "nuts" #
mdl = "linear" # 
fltr = :kalman
algo = :first_order

sample_idx = 47:52
dat = CSV.read("benchmark/usmodel.csv", DataFrame)

# Initialize a DataFrame to store the data
df = DataFrame(iteration = Float64[])

if mdl == "linear"
    include("../models/Smets_Wouters_2007_linear.jl")
    Smets_Wouters_2007 = Smets_Wouters_2007_linear
elseif mdl == "nonlinear"
    include("../models/Smets_Wouters_2007.jl")
end


# load data
data = KeyedArray(Array(dat)',Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

# declare observables
observables_old = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

# Subsample
# subset observables in data
data = data(observables_old, sample_idx)

observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

data = rekey(data, :Variable => observables)

# Handling distributions with varying parameters using arraydist
dists = [
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # 1     z_ea
InverseGamma(0.1, 2.0, 0.025,5.0, μσ = true),   # 2     z_eb
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # 3     z_eg
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # 4     z_eqs
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # 5     z_em
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # 6     z_epinf
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # 7     z_ew
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # 8     crhoa
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # 9     crhob
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # 10    crhog
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # 11    crhoqs
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # 12    crhoms
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # 13    crhopinf
Beta(0.5, 0.2, 0.001,0.9999, μσ = true),        # 14    crhow
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # 15    cmap
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # 16    cmaw
Normal(4.0, 1.5,   2.0, 15.0),                  # 17    csadjcost
Normal(1.50,0.375, 0.25, 3.0),                  # 18    csigma 
Beta(0.7, 0.1, 0.001, 0.99, μσ = true),         # 19    chabb
Beta(0.5, 0.1, 0.3, 0.95, μσ = true),           # 20    cprobw
Normal(2.0, 0.75, 0.25, 10.0),                  # 21    csigl
Beta(0.5, 0.10, 0.5, 0.95, μσ = true),          # 22    cprobp 
Beta(0.5, 0.15, 0.01, 0.99, μσ = true),         # 23    cindw      -> ιʷ
Beta(0.5, 0.15, 0.01, 0.99, μσ = true),         # 24    cindp      -> ιᵖ
Beta(0.5, 0.15, 0.01, 0.99999, μσ = true),      # 25    czcap      -> ψ
Normal(1.25, 0.125, 1.0, 3.0),                  # 26    cfc        -> Φ
Normal(1.5, 0.25, 1.0, 3.0),                    # 27    crpi
Beta(0.75, 0.10, 0.5, 0.975, μσ = true),        # 28    crr
Normal(0.125, 0.05, 0.001, 0.5),                # 29    cry
Normal(0.125, 0.05, 0.001, 0.5),                # 30    crdy
Gamma(0.625, 0.1, 0.1, 2.0, μσ = true),         # 31    constepinf
Gamma(0.25, 0.1, 0.01, 2.0, μσ = true),         # 32    constebeta -> 100(β⁻¹ - 1)
Normal(0.0, 2.0, -10.0, 10.0),                  # 33    constelab  -> l̄
Normal(0.4, 0.10, 0.1, 0.8),                    # 34    ctrend     -> γ̄
Normal(0.5, 0.25, 0.01, 2.0),                   # 35    cgy        -> ρᵍᵃ
Normal(0.3, 0.05, 0.01, 1.0),                   # 36    calfa
]

fixed_parameters = Smets_Wouters_2007.parameter_values[indexin([:ctou,:clandaw,:cg,:curvp,:curvw],Smets_Wouters_2007.parameters)]

SS(Smets_Wouters_2007, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01,:cmap => 0.01,:cmaw => 0.01], algorithm = algo)(observables)

inits = [Dict(get_parameters(Smets_Wouters_2007, values = true))[string(i)] for i in [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]]

function get_prior_llh(prior_distribuions, parameters::Vector{S})::S where S <: Real
    return sum(logpdf.(prior_distribuions, parameters))
end

function calculate_posterior_loglikelihood(parameters::Vector{S}, fixed_parameters, prior_distribuions, model, data, filter::Symbol, algorithm::Symbol)::S where S <: Real
    prior_llh = get_prior_llh(prior_distribuions, parameters)

    z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa = parameters

    ctou, clandaw, cg, curvp, curvw = fixed_parameters

    parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]

    model_llh = get_loglikelihood(model, data, parameters_combined, verbose = false, presample_periods = 4, filter = filter, algorithm = algorithm, initial_covariance = :diagonal)

    return prior_llh + model_llh
end


inits = [  0.5295766584252728
0.25401999781328677
0.5555813987579575
0.3654903601830364
0.2294564856713931
0.12294028349908431
0.20767050150368016
0.9674674841230338
0.20993223738088435
0.9888169549988175
0.8669340301385475
0.07818383624087137
0.6105112778170307
0.37671694996404337
0.2187231627543815
0.1362385298510586
6.3886101979474015
1.6678696241559958
0.6799655079831786
0.9424292929726574
2.502826072472096
0.6570767721691694
0.6729083298930368
0.23408903978575385
0.6457362272648652
1.4738116352107862
2.088069269612668
0.8655409607264644
0.0895375194503755
0.18792207697672325
0.696046453737325
0.1899464169442222
-0.5748023731804703
0.3683194328119635
0.5101771887138438
0.17425592648706756]

calculate_posterior_loglikelihood(inits, fixed_parameters, dists, Smets_Wouters_2007, data, fltr, algo) # -1114.047468890962

@time back_grad = Zygote.gradient(x -> calculate_posterior_loglikelihood(x, fixed_parameters, dists, Smets_Wouters_2007, data, fltr, algo), inits)

@time forw_grad = ForwardDiff.gradient(x -> calculate_posterior_loglikelihood(x, fixed_parameters, dists, Smets_Wouters_2007, data, fltr, algo), inits)

fini_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), x -> calculate_posterior_loglikelihood(x, fixed_parameters, dists, Smets_Wouters_2007, data, fltr, algo), inits)[1]


@benchmark calculate_posterior_loglikelihood(inits, fixed_parameters, dists, Smets_Wouters_2007, data, fltr, algo)

@benchmark Zygote.gradient(x -> calculate_posterior_loglikelihood(x, fixed_parameters, dists, Smets_Wouters_2007, data, fltr, algo), inits)

@benchmark ForwardDiff.gradient(x -> calculate_posterior_loglikelihood(x, fixed_parameters, dists, Smets_Wouters_2007, data, fltr, algo), inits)

@profview for i in 1:100 calculate_posterior_loglikelihood(inits, fixed_parameters, dists, Smets_Wouters_2007, data, fltr, algo) end

@profview for i in 1:10 Zygote.gradient(x -> calculate_posterior_loglikelihood(x, fixed_parameters, dists, Smets_Wouters_2007, data, fltr, algo), inits) end
