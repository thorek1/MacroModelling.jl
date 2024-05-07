using MacroModelling
using Zygote
import Turing, Pigeons
import Turing: NUTS, sample, logpdf, AutoZygote
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL

include("../models/Smets_Wouters_2007_linear.jl")

# load data
dat = CSV.read("data/usmodel.csv", DataFrame)

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

Turing.@model function SW07_loglikelihood_function(data, m, observables, fixed_parameters)
    all_params ~ Turing.arraydist(dists)

    z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa = all_params

    ctou, clandaw, cg, curvp, curvw = fixed_parameters

    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext() 
        parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]

        kalman_prob = get_loglikelihood(m, data(observables), parameters_combined, presample_periods = 4, initial_covariance = :diagonal)

        Turing.@addlogprob! kalman_prob
    end
end

fixed_parameters = Smets_Wouters_2007_linear.parameter_values[indexin([:ctou,:clandaw,:cg,:curvp,:curvw],Smets_Wouters_2007_linear.parameters)]


SW07_loglikelihood = SW07_loglikelihood_function(data, Smets_Wouters_2007_linear, observables, fixed_parameters)

SS(Smets_Wouters_2007_linear, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01,:cmap => 0.01,:cmaw => 0.01])

inits = [Dict(get_parameters(Smets_Wouters_2007_linear, values = true))[string(i)] for i in [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]]




#find starting value

function calculate_posterior_loglikelihood(parameters, fixed_parameters, prior_distribuions, model, data)
    log_lik = 0.0

    for (dist, val) in zip(prior_distribuions, parameters)
        log_lik -= logpdf(dist, val)
    end

    z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa = parameters

    ctou, clandaw, cg, curvp, curvw = fixed_parameters

    parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]

    log_lik -= get_loglikelihood(model, data, parameters_combined, verbose = false, presample_periods = 4, initial_covariance = :diagonal)

    return log_lik
end

calculate_posterior_loglikelihood(inits, fixed_parameters, dists, Smets_Wouters_2007_linear, data)



bounds = [0.01 3.0
0.025 5.0
0.01 3.0
0.01 3.0
0.01 3.0
0.01 3.0
0.01 3.0
0.01 0.9999
0.01 0.9999
0.01 0.9999
0.01 0.9999
0.01 0.9999
0.01 0.9999
0.00 10.9999
0.01 0.9999
0.01 0.9999
2.0 15.0
0.25 3.0
0.001 0.99
0.3 0.95
0.25 10.0
0.5 0.95
0.01 0.99
0.01 0.99
0.01 0.9999
1.0 3.0
1.0 3.0
0.5 0.975
0.001 0.5
0.001 0.5
0.1 2.0
0.01 2.0
-10.0 10.0
0.1 0.8
0.01 2.0
0.01 1.0]

lbs = bounds[:,1] # .+ 1e-7
ubs = bounds[:,2] # .- 1e-7

sol = Optim.optimize(x -> calculate_posterior_loglikelihood(x, fixed_parameters, dists, Smets_Wouters_2007_linear, data),
                            lbs, ubs, inits, 
                            Optim.SAMIN(verbosity = 2), 
                            # Optim.ParticleSwarm(lower = lbs, upper = ubs), 
                            # Optim.NelderMead(), 
                            Optim.Options(#f_abstol = eps(), 
                                            # g_tol= 1e-30,
                                            # iterations = 3000,
                                            show_trace = false,
                                            extended_trace = false)
                            )


inits = sol.minimizer

z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa = inits

ctou, clandaw, cg, curvp, curvw = fixed_parameters

parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]

get_loglikelihood(Smets_Wouters_2007_linear, data(observables), parameters_combined, presample_periods = 4, initial_covariance = :diagonal)

n_samples = 1000

samps = @time Turing.sample(SW07_loglikelihood, NUTS(adtype = AutoZygote()), n_samples, progress = true, initial_params = inits)

println(mean(samps).nt.mean)

# estimate nonlinear model

include("../models/Smets_Wouters_2007.jl")

fixed_parameters = Smets_Wouters_2007.parameter_values[indexin([:ctou,:clandaw,:cg,:curvp,:curvw],Smets_Wouters_2007.parameters)]

SW07_loglikelihood = SW07_loglikelihood_function(data, Smets_Wouters_2007, observables, fixed_parameters)

SS(Smets_Wouters_2007, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01,:cmap => 0.01,:cmaw => 0.01])(observables)

inits = [Dict(get_parameters(Smets_Wouters_2007, values = true))[string(i)] for i in [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]]

sol = Optim.optimize(x -> calculate_posterior_loglikelihood(x, fixed_parameters, dists, Smets_Wouters_2007, data),
                            lbs, ubs, inits, 
                            Optim.SAMIN(verbosity = 2), 
                            # Optim.ParticleSwarm(lower = lbs, upper = ubs), 
                            # Optim.NelderMead(), 
                            Optim.Options(#f_abstol = eps(), 
                                            # g_tol= 1e-30,
                                            # iterations = 3000,
                                            show_trace = false,
                                            extended_trace = false)
                            )


inits = sol.minimizer

z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa = inits

ctou, clandaw, cg, curvp, curvw = fixed_parameters

parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]

get_loglikelihood(Smets_Wouters_2007, data(observables), parameters_combined, presample_periods = 4, initial_covariance = :diagonal)

n_samples = 1000

samps = @time Turing.sample(SW07_loglikelihood, NUTS(adtype = AutoZygote()), n_samples, progress = true, initial_params = inits)
