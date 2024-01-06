using MacroModelling
import Turing, Pigeons
import Turing: NUTS, sample, logpdf
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL: logjoint

include("../models/FS2000.jl")

# load data
dat = CSV.read("data/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names(dat)))

# subset observables in data
data = data(observables,:)


Turing.@model function FS2000_loglikelihood_function(data, m)
    alp     ~ Beta(0.356, 0.02, μσ = true)
    bet     ~ Beta(0.993, 0.002, μσ = true)
    gam     ~ Normal(0.0085, 0.003)
    mst     ~ Normal(1.0002, 0.007)
    rho     ~ Beta(0.129, 0.223, μσ = true)
    psi     ~ Beta(0.65, 0.05, μσ = true)
    del     ~ Beta(0.01, 0.005, μσ = true)
    z_e_a   ~ InverseGamma(0.035449, Inf, μσ = true)
    z_e_m   ~ InverseGamma(0.008862, Inf, μσ = true)
    # println([alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
    Turing.@addlogprob! get_loglikelihood(m, [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m], data)
end

FS2000_loglikelihood = FS2000_loglikelihood_function(data, FS2000)


pt = @time Pigeons.pigeons(target = Pigeons.TuringLogPotential(FS2000_loglikelihood_function(data, FS2000)),
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 3,
            n_rounds = 7,
            multithreaded = true)

samps = MCMCChains.Chains(Pigeons.get_sample(pt))

println(mean(samps).nt.mean)

Random.seed!(30)

function calculate_posterior_loglikelihood(parameters)
    alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = parameters
    log_lik = 0
    log_lik -= get_loglikelihood(FS2000, parameters, data)
    log_lik -= logpdf(Beta(0.356, 0.02, μσ = true),alp)
    log_lik -= logpdf(Beta(0.993, 0.002, μσ = true),bet)
    log_lik -= logpdf(Normal(0.0085, 0.003),gam)
    log_lik -= logpdf(Normal(1.0002, 0.007),mst)
    log_lik -= logpdf(Beta(0.129, 0.223, μσ = true),rho)
    log_lik -= logpdf(Beta(0.65, 0.05, μσ = true),psi)
    log_lik -= logpdf(Beta(0.01, 0.005, μσ = true),del)
    log_lik -= logpdf(InverseGamma(0.035449, Inf, μσ = true),z_e_a)
    log_lik -= logpdf(InverseGamma(0.008862, Inf, μσ = true),z_e_m)
    return log_lik
end

sol = Optim.optimize(calculate_posterior_loglikelihood, 
[0,0,-10,-10,0,0,0,0,0], [1,1,10,10,1,1,1,100,100] ,FS2000.parameter_values, 
Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))); autodiff = :forward)

@testset "Estimation results" begin
    @test isapprox(sol.minimum, -1343.7491257498598, rtol = eps(Float32))
    @test isapprox(mean(samps).nt.mean, [0.40248024934137033, 0.9905235783816697, 0.004618184988033483, 1.014268215459915, 0.8459140293740781, 0.6851143053372912, 0.0025570276255960107, 0.01373547787288702, 0.003343985776134218], rtol = 1e-2)
end



plot_model_estimates(FS2000, data, parameters = sol.minimizer)
plot_shock_decomposition(FS2000, data)

FS2000 = nothing
m = nothing
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
