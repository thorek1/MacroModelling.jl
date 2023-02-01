using MacroModelling
import Turing
import Turing: Normal, Beta, InverseGamma, NUTS, sample, logpdf
using Random, CSV, DataFrames, ComponentArrays, MCMCChains#, Optimization, OptimizationNLopt, OptimizationOptimisers
import DynamicPPL: logjoint

include("models/FS2000.jl")

FS2000 = m

# load data
dat = CSV.read("data/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names(dat)))

# subset observables in data
data = data(observables,:)

# functions to map mean and standard deviations to distribution parameters
function beta_map(μ, σ) 
    α = ((1 - μ) / σ ^ 2 - 1 / μ) * μ ^ 2
    β = α * (1 / μ - 1)
    return α, β
end

function inv_gamma_map(μ, σ)
    α = (μ / σ) ^ 2 + 2
    β = μ * ((μ / σ) ^ 2 + 1)
    return α, β
end

function gamma_map(μ, σ)
    k = μ^2/σ^2 
    θ = σ^2 / μ
    return k, θ
end


Turing.@model function FS2000_loglikelihood_function(data, m, observables)
    alp     ~ Beta(beta_map(0.356, 0.02)...)
    bet     ~ Beta(beta_map(0.993, 0.002)...)
    gam     ~ Normal(0.0085, 0.003)
    mst     ~ Normal(1.0002, 0.007)
    rho     ~ Beta(beta_map(0.129, 0.223)...)
    psi     ~ Beta(beta_map(0.65, 0.05)...)
    del     ~ Beta(beta_map(0.01, 0.005)...)
    z_e_a   ~ InverseGamma(inv_gamma_map(0.035449, Inf)...)
    z_e_m   ~ InverseGamma(inv_gamma_map(0.008862, Inf)...)
    # println([alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
end

FS2000_loglikelihood = FS2000_loglikelihood_function(data, FS2000, observables)


# function calculate_posterior_loglikelihood(parameters, u)
#     alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = parameters
#     log_lik = 0
#     log_lik -= calculate_kalman_filter_loglikelihood(FS2000, data(observables), observables; parameters = parameters)
#     log_lik -= logpdf(Beta(beta_map(0.356, 0.02)...),alp)
#     log_lik -= logpdf(Beta(beta_map(0.993, 0.002)...),bet)
#     log_lik -= logpdf(Normal(0.0085, 0.003),gam)
#     log_lik -= logpdf(Normal(1.0002, 0.007),mst)
#     log_lik -= logpdf(Beta(beta_map(0.129, 0.223)...),rho)
#     log_lik -= logpdf(Beta(beta_map(0.65, 0.05)...),psi)
#     log_lik -= logpdf(Beta(beta_map(0.01, 0.005)...),del)
#     log_lik -= logpdf(InverseGamma(inv_gamma_map(0.035449, Inf)...),z_e_a)
#     log_lik -= logpdf(InverseGamma(inv_gamma_map(0.008862, Inf)...),z_e_m)
#     return log_lik
# end

# f = OptimizationFunction(calculate_posterior_loglikelihood, Optimization.AutoForwardDiff())

# prob = OptimizationProblem(f, Float64.(FS2000.parameter_values), []);
# sol = solve(prob, Optimisers.ADAM(), maxiters = 1000)
# sol.minimum

# lbs = fill(-1e12, length(FS2000.parameters));
# ubs = fill(1e12, length(FS2000.parameters));

# bounds_index_in_pars = indexin(intersect(FS2000.bounded_vars,FS2000.parameters),FS2000.parameters);
# bounds_index_in_bounds = indexin(intersect(FS2000.bounded_vars,FS2000.parameters),FS2000.bounded_vars);

# lbs[bounds_index_in_pars] = max.(-1e12,FS2000.lower_bounds[bounds_index_in_bounds]);
# ubs[bounds_index_in_pars] = min.(1e12,FS2000.upper_bounds[bounds_index_in_bounds]);

# prob = OptimizationProblem(f, min.(max.(sol.u,lbs),ubs), [], lb = lbs, ub = ubs);
# sol = solve(prob, NLopt.LD_LBFGS())
# sol.minimum


n_samples = 1000

Random.seed!(30)
# using Zygote
# Turing.setadbackend(:zygote)
samps = sample(FS2000_loglikelihood, NUTS(), n_samples, progress = true)#, init_params = sol)

println(mean(samps))

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