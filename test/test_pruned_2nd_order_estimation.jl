using MacroModelling
import Turing
import Pigeons
import ADTypes: AutoZygote
import Turing: NUTS, sample, logpdf, Beta, Normal, InverseGamma
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL

include("../models/FS2000.jl")

# load data
dat = CSV.read("data/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names(dat)))

# subset observables in data
data = data(observables,:)


dists = [
    Beta(0.356, 0.02, Val(:μσ)),           # alp
    Beta(0.993, 0.002, Val(:μσ)),          # bet
    Normal(0.0085, 0.003),                  # gam
    Normal(1.0002, 0.007),                  # mst
    Beta(0.129, 0.223, Val(:μσ)),          # rho
    Beta(0.65, 0.05, Val(:μσ)),            # psi
    Beta(0.01, 0.005, Val(:μσ)),           # del
    InverseGamma(0.035449, Inf, Val(:μσ)), # z_e_a
    InverseGamma(0.008862, Inf, Val(:μσ))  # z_e_m
]

Turing.@model function FS2000_loglikelihood_function(data, m, algorithm)
    all_params ~ Turing.arraydist(dists)

    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext() 
        Turing.@addlogprob! get_loglikelihood(m, data, all_params, algorithm = algorithm)
    end
end


Random.seed!(30)

n_samples = 500

samps = @time sample(FS2000_loglikelihood_function(data, FS2000, :pruned_second_order), NUTS(adtype = AutoZygote()), n_samples, progress = true, initial_params = FS2000.parameter_values)


println("Mean variable values (Zygote): $(mean(samps).nt.mean)")

sample_nuts = mean(samps).nt.mean

# generate a Pigeons log potential
FS2000_pruned2nd_lp = Pigeons.TuringLogPotential(FS2000_loglikelihood_function(data, FS2000, :pruned_second_order))

init_params = sample_nuts

LLH = Turing.logjoint(FS2000_loglikelihood_function(data, FS2000, :pruned_second_order), (all_params = init_params,))

if isfinite(LLH)
    const FS2000_pruned2nd_LP = typeof(FS2000_pruned2nd_lp)

    function Pigeons.initialization(target::FS2000_pruned2nd_LP, rng::AbstractRNG, _::Int64)
        result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.SampleFromPrior(), DynamicPPL.PriorContext())
        # DynamicPPL.link!!(result, DynamicPPL.SampleFromPrior(), target.model)
        
        result = DynamicPPL.initialize_parameters!!(result, init_params, target.model)

        return result
    end

    pt = Pigeons.pigeons(target = FS2000_pruned2nd_lp, n_rounds = 0, n_chains = 1)
else
    replica = pt.replicas[end]
    XMAX = deepcopy(replica.state)
    LPmax = FS2000_pruned2nd_lp(XMAX)

    i = 0

    while !isfinite(LPmax) && i < 1000
        Pigeons.sample_iid!(FS2000_pruned2nd_lp, replica, pt.shared)
        new_LP = FS2000_pruned2nd_lp(replica.state)
        if new_LP > LPmax
            global LPmax = new_LP
            global XMAX  = deepcopy(replica.state)
        end
        global i += 1
    end

    # define a specific initialization for this model
    Pigeons.initialization(::Pigeons.TuringLogPotential{typeof(FS2000_loglikelihood_function)}, ::AbstractRNG, ::Int64) = deepcopy(XMAX)
end

pt = @time Pigeons.pigeons(target = FS2000_pruned2nd_lp,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 1,
            n_rounds = 8,
            multithreaded = false)

samps = MCMCChains.Chains(pt)


println("Mean variable values (pruned second order): $(mean(samps).nt.mean)")

# # estimate highly nonlinear model


# # load data
# dat = CSV.read("data/usmodel.csv", DataFrame)
# data = KeyedArray(Array(dat)',Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

# # declare observables
# observables = [:dy, :dc]#, :dinve, :labobs, :pinfobs, :dw, :robs]

# # Subsample from 1966Q1 - 2004Q4
# # subset observables in data
# data = data(observables,75:230)


# include("models/Caldara_et_al_2012_estim.jl")


# # get_loglikelihood(Caldara_et_al_2012_estim, data, Caldara_et_al_2012_estim.parameter_values, algorithm = :pruned_third_order)

# # get_loglikelihood(Caldara_et_al_2012_estim, data, Caldara_et_al_2012_estim.parameter_values*0.99, algorithm = :pruned_third_order)


# # get_parameters(Caldara_et_al_2012_estim, values = true)

# Turing.@model function Caldara_et_al_2012_loglikelihood_function(data, m)
#     dȳ  ~ Normal(0, 1)
#     dc̄  ~ Normal(0, 1)
#     β   ~ Beta(0.95, 0.005, Val(:μσ))
#     ζ   ~ Beta(0.33, 0.05, Val(:μσ))
#     δ   ~ Beta(0.02, 0.01, Val(:μσ))
#     λ   ~ Beta(0.75, 0.01, Val(:μσ))
#     ψ   ~ Normal(1, .25)#, Val(:μσ))
#     σ̄   ~ InverseGamma(0.021, Inf, Val(:μσ))
#     η   ~ InverseGamma(0.1, Inf, Val(:μσ))
#     ρ   ~ Beta(0.75, 0.02, Val(:μσ))

#     Turing.@addlogprob! get_loglikelihood(m, data, [dȳ, dc̄, β, ζ, δ, λ, ψ, σ̄, η, ρ], algorithm = :pruned_third_order)
# end


# Random.seed!(3)

# pt = @time Pigeons.pigeons(target = Pigeons.TuringLogPotential(Caldara_et_al_2012_loglikelihood_function(data, Caldara_et_al_2012_estim)),
#             record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
#             n_chains = 1,
#             n_rounds = 6,
#             multithreaded = true)

# samps = MCMCChains.Chains(Pigeons.get_sample(pt))


# println(mean(samps).nt.mean)


# Random.seed!(30)

# function calculate_posterior_llkh(parameters, grad)
#     if length(grad)>0
#         grad .= ForwardDiff.gradient(x->begin
#             dȳ, dc̄, β, ζ, δ, λ, ψ, σ̄, η, ρ = x
#             # println(parameters)
#             log_lik = 0
#             log_lik -= get_loglikelihood(Caldara_et_al_2012_estim, data, x, algorithm = :pruned_third_order)
#             log_lik -= logpdf(Normal(0, 1),dȳ)
#             log_lik -= logpdf(Normal(0, 1),dc̄)
#             log_lik -= logpdf(Beta(0.993, 0.05, Val(:μσ)),β)
#             log_lik -= logpdf(Beta(0.356, 0.05, Val(:μσ)),ζ)
#             log_lik -= logpdf(Beta(0.02, 0.01, Val(:μσ)),δ)
#             log_lik -= logpdf(Beta(0.5, 0.25, Val(:μσ)),λ)
#             log_lik -= logpdf(Normal(1, .25),ψ)
#             # log_lik -= logpdf(Normal(40, 10),γ)
#             log_lik -= logpdf(InverseGamma(0.021, Inf, Val(:μσ)),σ̄)
#             log_lik -= logpdf(InverseGamma(0.1, Inf, Val(:μσ)),η)
#             log_lik -= logpdf(Beta(0.5, 0.25, Val(:μσ)),ρ)
        
#             return log_lik
#         end, parameters)
#     end
#     dȳ, dc̄, β, ζ, δ, λ, ψ, σ̄, η, ρ = parameters
#     # println(parameters)
#     log_lik = 0
#     log_lik -= get_loglikelihood(Caldara_et_al_2012_estim, data, parameters, algorithm = :pruned_third_order)
#     log_lik -= logpdf(Normal(0, 1),dȳ)
#     log_lik -= logpdf(Normal(0, 1),dc̄)
#     log_lik -= logpdf(Beta(0.95, 0.005, Val(:μσ)),β)
#     log_lik -= logpdf(Beta(0.33, 0.05, Val(:μσ)),ζ)
#     log_lik -= logpdf(Beta(0.02, 0.01, Val(:μσ)),δ)
#     log_lik -= logpdf(Beta(0.75, 0.01, Val(:μσ)),λ)
#     log_lik -= logpdf(Normal(1, .25),ψ)
#     # log_lik -= logpdf(Normal(40, 10),γ)
#     log_lik -= logpdf(InverseGamma(0.021, Inf, Val(:μσ)),σ̄)
#     log_lik -= logpdf(InverseGamma(0.1, Inf, Val(:μσ)),η)
#     log_lik -= logpdf(Beta(0.75, 0.02, Val(:μσ)),ρ)
#     println(log_lik)
#     return log_lik
# end

# init_params = deepcopy(Caldara_et_al_2012_estim.parameter_values)
# using NLopt, ForwardDiff
# grad = zeros(0)
# calculate_posterior_llkh(Caldara_et_al_2012_estim.parameter_values, grad)

# grad = zeros(length(Caldara_et_al_2012_estim.parameter_values))
# calculate_posterior_llkh(Caldara_et_al_2012_estim.parameter_values, grad)


# opt = NLopt.Opt(NLopt.:LN_NELDERMEAD, length(get_parameters(Caldara_et_al_2012_estim)))
# opt = NLopt.Opt(NLopt.:LN_SBPLX, length(get_parameters(Caldara_et_al_2012_estim)))
# opt = NLopt.Opt(NLopt.:LN_PRAXIS, length(get_parameters(Caldara_et_al_2012_estim)))
# opt = NLopt.Opt(NLopt.:LN_COBYLA, length(get_parameters(Caldara_et_al_2012_estim)))
# opt = NLopt.Opt(NLopt.:LN_BOBYQA, length(get_parameters(Caldara_et_al_2012_estim)))
# opt = NLopt.Opt(NLopt.:LD_LBFGS, length(get_parameters(Caldara_et_al_2012_estim)))
# opt = NLopt.Opt(NLopt.:LD_SLSQP, length(get_parameters(Caldara_et_al_2012_estim)))
# opt = NLopt.Opt(NLopt.:LD_MMA, length(get_parameters(Caldara_et_al_2012_estim)))
# opt = NLopt.Opt(NLopt.:LD_VAR2, length(get_parameters(Caldara_et_al_2012_estim)))

# opt = NLopt.Opt(NLopt.:GN_CRS2_LM, length(get_parameters(Caldara_et_al_2012_estim)))

# opt.min_objective = calculate_posterior_llkh

# opt.upper_bounds = [5,5,1,1,1,1,100,100,100,1]
# opt.lower_bounds = [-3,-3,0,0,0,0,0,0,0,0]

# opt.xtol_rel = eps()

# opt.maxeval = 50000

# (minf,x,ret) = NLopt.optimize(opt, Caldara_et_al_2012_estim.parameter_values)

# opt.numevals

# using StatsPlots

# plot_irf(Caldara_et_al_2012_estim, parameters = x, algorithm = :pruned_third_order, periods = 1000)
# plot_irf(Caldara_et_al_2012_estim, parameters = :ψ => .05, algorithm = :pruned_third_order, periods = 1000)


# get_irf(Caldara_et_al_2012_estim, parameters = x, algorithm = :pruned_third_order)

# get_parameters(Caldara_et_al_2012_estim, values= true)

# calculate_posterior_loglikelihood(Caldara_et_al_2012_estim.parameter_values)

# sol = Optim.optimize(calculate_posterior_loglikelihood, 
# [-3,-3,0,0,0,0,-10,-10,0,0,0], [5,5,1,1,1,1,100,100,100,100,1] ,Caldara_et_al_2012_estim.parameter_values, 
# Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))); autodiff = :forward)


# sol = Optim.optimize(calculate_posterior_loglikelihood, 
# [-3,-3,0,0,0,0,-10,-10,0,0,0], [5,5,1,1,1,1,100,100,100,100,1] ,Caldara_et_al_2012_estim.parameter_values, 
# Optim.Fminbox(Optim.NelderMead()))


# 1
# @testset "Estimation results" begin
#     @test isapprox(sol.minimum, -1343.7491257498598, rtol = eps(Float32))
#     @test isapprox(mean(samps).nt.mean, [0.40248024934137033, 0.9905235783816697, 0.004618184988033483, 1.014268215459915, 0.8459140293740781, 0.6851143053372912, 0.0025570276255960107, 0.01373547787288702, 0.003343985776134218], rtol = 1e-2)
# end



# plot_model_estimates(FS2000, data, parameters = sol.minimizer)
# plot_shock_decomposition(FS2000, data)

# FS2000 = nothing
# m = nothing
# @profview sample(FS2000_loglikelihood, NUTS(), n_samples, progress = true)


# chain_NUTS  = sample(FS2000_loglikelihood, NUTS(), n_samples, init_params = FS2000.parameter_values, progress = true)#, init_params = FS2000.parameter_values)#init_theta = FS2000.parameter_values)

# StatsPlots.plot(chain_NUTS)

# parameter_mean = mean(chain_NUTS)

# pars = ComponentArray(parameter_mean.nt[2],Axis(parameter_mean.nt[1]))

# logjoint(FS2000_loglikelihood, pars)

# function calculate_log_probability(par1, par2, pars_syms, orig_pars, model)
#     orig_pars[pars_syms] = [par1, par2]
#     logjoint(model, orig_pars)
# end

# granularity = 32;

# par1 = :del;
# par2 = :gam;
# par_range1 = collect(range(minimum(chain_NUTS[par1]), stop = maximum(chain_NUTS[par1]), length = granularity));
# par_range2 = collect(range(minimum(chain_NUTS[par2]), stop = maximum(chain_NUTS[par2]), length = granularity));

# p = surface(par_range1, par_range2, 
#             (x,y) -> calculate_log_probability(x, y, [par1, par2], pars, FS2000_loglikelihood),
#             camera=(30, 65),
#             colorbar=false,
#             color=:inferno);


# joint_loglikelihood = [logjoint(FS2000_loglikelihood, ComponentArray(reduce(hcat, get(chain_NUTS, FS2000.parameters)[FS2000.parameters])[s,:], Axis(FS2000.parameters))) for s in 1:length(chain_NUTS)]

# scatter3d!(vec(collect(chain_NUTS[par1])),
#            vec(collect(chain_NUTS[par2])),
#            joint_loglikelihood,
#             mc = :viridis, 
#             marker_z = collect(1:length(chain_NUTS)), 
#             msw = 0,
#             legend = false, 
#             colorbar = false, 
#             xlabel = string(par1),
#             ylabel = string(par2),
#             zlabel = "Log probability",
#             alpha = 0.5);

# p

