using MacroModelling
using Zygote
import ForwardDiff
import Turing
# import Pigeons
import Turing: NUTS, sample, logpdf, AutoZygote
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL
using StatsPlots

# load data
dat = CSV.read("test/data/usmodel.csv", DataFrame)

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

        # llh = get_loglikelihood(m, data(observables), parameters_combined, presample_periods = 4, initial_covariance = :diagonal, filter = filter)
        llh = get_loglikelihood(m, data(observables), parameters_combined, algorithm = :pruned_second_order)

        Turing.@addlogprob! llh
    end
end

# estimate linear model

include("../models/Smets_Wouters_2007_linear.jl")

fixed_parameters = Smets_Wouters_2007_linear.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw], Smets_Wouters_2007_linear.parameters)]

SS(Smets_Wouters_2007_linear, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01,:cmap => 0.01,:cmaw => 0.01])

SW07_loglikelihood = SW07_loglikelihood_function(data, Smets_Wouters_2007_linear, observables, fixed_parameters)
# inversion filter delivers similar results

# par_names = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

# inits = [Dict(get_parameters(Smets_Wouters_2007_linear, values = true))[string(i)] for i in par_names]

# modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
#                                         Optim.SimulatedAnnealing(),
#                                         initial_params = inits)

# modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
#                                         Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)),
#                                         initial_params = modeSW2007.values)

modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
                                        Optim.NelderMead())

println("Mode variable values (linear): $(modeSW2007.values); Mode loglikelihood: $(modeSW2007.lp)")

n_samples = 1000

samps = Turing.sample(SW07_loglikelihood, NUTS(adtype = AutoZygote()), n_samples, progress = true, initial_params = modeSW2007.values)

println(samps)
println("Mean variable values (linear): $(mean(samps).nt.mean)")

# estimate nonlinear model

include("../models/Smets_Wouters_2007.jl")

fixed_parameters = Smets_Wouters_2007.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw], Smets_Wouters_2007.parameters)]

SS(Smets_Wouters_2007, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01, :cmap => 0.01, :cmaw => 0.01])(observables,:)

SW07_loglikelihood = SW07_loglikelihood_function(data, Smets_Wouters_2007, observables, fixed_parameters)

par_names = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

inits = [Dict(get_parameters(Smets_Wouters_2007, values = true))[string(i)] for i in par_names]

prelim_calib = [0.4813670475270772, 0.24320737018621832, 0.6142082826448849, 0.5053618080961291, 0.19035600057021243, 0.16314529405961975, 0.24962377895319873, 0.9639698677543815, 0.18770419581251246, 0.9584316098996059, 0.805430406165429, 0.31911234852327763, 0.7529475086575895, 0.9676287425668599, 0.5404369107137618, 0.5559877036206856, 5.209936567404959, 1.4862089326367263, 0.6840941852578368, 0.5953446850610999, 1.8900934869133503, 0.71586139228798, 0.5121652475370841, 0.3024732916676549, 0.5913207527941786, 1.4914409882484962, 1.9190803125304088, 0.8226818966969396, 0.01977038820192296, 0.15405567411538829, 0.9398984721990804, 0.2824474802594214, -5.485555453566931, 0.47121328130643825, 0.47260279427286916, 0.23099216672363482]

SS(Smets_Wouters_2007, parameters = par_names .=> collect(prelim_calib))

modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
                                        Optim.SimulatedAnnealing(),
                                        initial_params = prelim_calib)

modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
                                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)),
                                        initial_params = modeSW2007.values,
                                        adtype = AutoZygote())

get_loglikelihood(Smets_Wouters_2007, data(observables)[:,100:end], Smets_Wouters_2007.parameter_values, algorithm = :pruned_second_order)

Zygote.gradient(x -> get_loglikelihood(Smets_Wouters_2007, data(observables), x, algorithm = :pruned_second_order), Smets_Wouters_2007.parameter_values)[1]

SS(Smets_Wouters_2007, parameters = par_names .=> collect(modeSW2007.values))

get_loglikelihood(Smets_Wouters_2007, data(observables), Smets_Wouters_2007.parameter_values, algorithm = :pruned_second_order)

modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
                                        Optim.NelderMead(),
                                        initial_params = modeSW2007.values)

println("Mode variable values (linear): $(modeSW2007.values); Mode loglikelihood: $(modeSW2007.lp)")
# [0.4813670475270772, 0.24320737018621832, 0.6142082826448849, 0.5053618080961291, 0.19035600057021243, 0.16314529405961975, 0.24962377895319873, 0.9639698677543815, 0.18770419581251246, 0.9584316098996059, 0.805430406165429, 0.31911234852327763, 0.7529475086575895, 0.9676287425668599, 0.5404369107137618, 0.5559877036206856, 5.209936567404959, 1.4862089326367263, 0.6840941852578368, 0.5953446850610999, 1.8900934869133503, 0.71586139228798, 0.5121652475370841, 0.3024732916676549, 0.5913207527941786, 1.4914409882484962, 1.9190803125304088, 0.8226818966969396, 0.01977038820192296, 0.15405567411538829, 0.9398984721990804, 0.2824474802594214, -5.485555453566931, 0.47121328130643825, 0.47260279427286916, 0.23099216672363482]
n_samples = 1000

samps = Turing.sample(SW07_loglikelihood, NUTS(adtype = AutoZygote()), n_samples, progress = true, initial_params = modeSW2007.values)

println(samps)
println("Mean variable values (nonlinear): $(mean(samps).nt.mean)")




# Optimal simple rule

include("../models/Smets_Wouters_2007.jl")

SS(Smets_Wouters_2007, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01, :cmap => 0.01, :cmaw => 0.01])(observables,:)

# npar_names = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

# some_mode = [0.4813670475270772, 0.24320737018621832, 0.6142082826448849, 0.5053618080961291, 0.19035600057021243, 0.16314529405961975, 0.24962377895319873, 0.9639698677543815, 0.18770419581251246, 0.9584316098996059, 0.805430406165429, 0.31911234852327763, 0.7529475086575895, 0.9676287425668599, 0.5404369107137618, 0.5559877036206856, 5.209936567404959, 1.4862089326367263, 0.6840941852578368, 0.5953446850610999, 1.8900934869133503, 0.71586139228798, 0.5121652475370841, 0.3024732916676549, 0.5913207527941786, 1.4914409882484962, 1.9190803125304088, 0.8226818966969396, 0.01977038820192296, 0.15405567411538829, 0.9398984721990804, 0.2824474802594214, -5.485555453566931, 0.47121328130643825, 0.47260279427286916, 0.23099216672363482]

# SS(Smets_Wouters_2007, parameters = par_names .=> collect(some_mode))

# get_loglikelihood(Smets_Wouters_2007, data(observables)[:,100:end], Smets_Wouters_2007.parameter_values, algorithm = :pruned_second_order)

# Zygote.gradient(x -> get_loglikelihood(Smets_Wouters_2007, data(observables), x, algorithm = :pruned_second_order), Smets_Wouters_2007.parameter_values)[1]

parameter_value_inputs = [0.8762, 1.488, 0.0593, 0.2347]

model_statistics = get_statistics(Smets_Wouters_2007, 
                                parameter_value_inputs, 
                                algorithm = :pruned_second_order, 
                                parameters = [:crr, :crpi, :cry, :crdy], 
                                variance = [:drobs, :dpinfobs, :ygap])

                                
function distance_to_target(parameter_value_inputs)
    model_statistics = get_statistics(Smets_Wouters_2007, 
                                        parameter_value_inputs, 
                                        algorithm = :pruned_second_order, 
                                        parameters = [:crr, :crpi, :cry, :crdy], 
                                        variance = [:drobs, :dpinfobs, :ygap])

    weights = [4,10,3]

    return model_statistics[1]' * weights ./ sum(weights)
end

distance_to_target(sol.minimizer)

ForwardDiff.gradient(distance_to_target, parameter_value_inputs)

parameter_value_inputs = [0.9918780015748438, 49.08252729389275, 0.8700701653905272, 0.5507441553256847]

sol = Optim.optimize(distance_to_target,
                               [0.5,1,0.001,0.001],
                               [.975,3,.5,.5],
                               parameter_value_inputs,
                               Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))),
                            #    Optim.Fminbox(Optim.NelderMead()),
                               Optim.Options(outer_iterations = 1,
                                                time_limit = 600,
                                                show_trace = true))

                 
sol = Optim.optimize(distance_to_target,
[0.5,1,0.001,0.001],
[.975,3,.5,.5],
                               sol.minimizer,
                                Optim.Fminbox(Optim.NelderMead()),
                            #    Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))),
                               Optim.Options(outer_iterations = 1,
                                            show_trace = true,
                                            time_limit = 600))


model_statistics = get_statistics(Smets_Wouters_2007,
                                # sol.minimizer, 
                                parameter_value_inputs,
                                algorithm = :pruned_second_order, 
                                parameters = [:crr, :crpi, :cry, :crdy], 
                                variance = [:drobs, :dpinfobs, :ygap])
                                # 4-element Vector{Float64}:
                                # 0.9803094444557999
                                # 8.639241288770886
                                # 0.9171416346407335
                                # 0.9972515638017101

distance_to_target(sol.minimizer)

producivity_shock_size = []
OSR_coefficients = []

for i in range(.4, .6, 10)
    push!(producivity_shock_size, i)
    SS(Smets_Wouters_2007, parameters = :z_ea => i, derivatives = false)
    sol = Optim.optimize(distance_to_target,
    [0.5,1,0.001,0.001],
    [.975,3,.5,.5],
                        sol.minimizer,
                        Optim.Fminbox(Optim.NelderMead()),
                        # Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))),
                        Optim.Options(outer_iterations = 3,
                                    # show_trace = true,
                                    time_limit = 600))
    push!(OSR_coefficients, sol.minimizer)
    println("z_ea: $i => OSR coeffs: $(sol.minimizer)")
end



function distance_to_target(parameter_value_inputs)
    model_statistics = get_statistics(Smets_Wouters_2007, 
                                        parameter_value_inputs, 
                                        algorithm = :first_order, 
                                        parameters = [:crr, :crpi, :cry, :crdy], 
                                        standard_deviation = [:drobs, :pinfobs, :ygap])

    weights = [4,10,3]

    return model_statistics[1]' * weights ./ sum(weights)
end

distance_to_target(parameter_value_inputs)

sol = Optim.optimize(distance_to_target,
                    [0.5,1,0.001,0.001],
                    [.975,3,.5,.5],
                    parameter_value_inputs,
                    Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))),
                    # Optim.Fminbox(Optim.NelderMead()),
                    Optim.Options(outer_iterations = 10,
                                        time_limit = 600,
                                        show_trace = true))

sol.minimizer

distance_to_target(sol.minimizer)




producivity_shock_size = []
inflation_coefficient = []
pinfobs_std = []

get_std(Smets_Wouters_2007)(:pinfobs,:)

get_std(Smets_Wouters_2007)(:pinfobs)

for i in range(.4, .9, 10)
    for k in range(1.45,1.5,10)
        push!(producivity_shock_size, i)
        push!(inflation_coefficient, k)
        push!(pinfobs_std, get_std(Smets_Wouters_2007, 
                                    parameters = (:z_ea => i, :crpi => k), 
                                    derivatives = false, 
                                    algorithm = :pruned_second_order)(:pinfobs))
    end
end

StatsPlots.surface(producivity_shock_size, 
                    inflation_coefficient, 
                    pinfobs_std, 
                    camera = (55,25),
                    # xrotation = 45,
                    xlabel = "Productivity Shock \nStd. Dev.", 
                    ylabel = "Taylor Rule: \nInflation Coefficient", 
                    zlabel = "Inflation Std. Dev.")




StatsPlots.contour(producivity_shock_size, 
                    inflation_coefficient, 
                    pinfobs_std, 
                    # camera = (45,-45),
                    # xrotation = 45,
                    # xlabel = "Productivity Shock Std. Dev.", 
                    # ylabel = "Taylor Rule: \nInflation Coefficient", 
                    # zlabel = "Inflation Std. Dev."
                    )




cprobp_uncertainty = []
inflation_coefficient = []
pinfobs_std = []

for i in range(.01, .5, 5)
    for k in range(1.45,1.5,5)
        push!(cprobp_uncertainty, i)
        push!(inflation_coefficient, k)
        push!(pinfobs_std, get_std(Smets_Wouters_2007, 
                                    parameters = (:z_e_cprobp => i, :crpi => k), 
                                    algorithm = :pruned_second_order, 
                                    derivatives = false)(:pinfobs))
    end
end

StatsPlots.surface(cprobp_uncertainty, 
                    inflation_coefficient, 
                    pinfobs_std, 
                    camera = (45,35),
                    # xrotation = 45,
                    xlabel = "Calvo probability (Goods)\nUncertainty (Std. Dev.)", 
                    ylabel = "Taylor Rule: \nInflation Coefficient", 
                    zlabel = "Inflation Std. Dev."
                    )

StatsPlots.savefig("Calvo_prob_TR_inf.pdf")

StatsPlots.savefig("Calvo_prob_TR_inf.png")

get_std(Smets_Wouters_2007, algorithm = :pruned_second_order, derivatives = true)(:pinfobs,:)

stoch_vol_prod_shock = []
inflation_coefficient = []
pinfobs_std = []

for i in range(.01, .5, 5)
    for k in range(1.45,1.5,5)
        push!(stoch_vol_prod_shock, i)
        push!(inflation_coefficient, k)
        push!(pinfobs_std, get_std(Smets_Wouters_2007, 
                                    parameters = (:z_ez_ea => i, :crpi => k), 
                                    algorithm = :pruned_third_order, 
                                    derivatives = false)(:pinfobs))
    end
end

StatsPlots.surface(stoch_vol_prod_shock, 
                    inflation_coefficient, 
                    pinfobs_std, 
                    camera = (45,35),
                    # xrotation = 45,
                    xlabel = "Stoch. Volatility \nProductivity Shock", 
                    ylabel = "Taylor Rule: \nInflation Coefficient", 
                    zlabel = "Inflation Std. Dev."
                    )

StatsPlots.savefig("Stoch_vol_TR_inf.pdf")

StatsPlots.savefig("Stoch_vol_TR_inf.png")

