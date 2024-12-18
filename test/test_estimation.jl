using Revise
using MacroModelling
import Turing, Pigeons, Zygote
import Turing: NUTS, sample, logpdf, AutoZygote
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL

include("../models/FS2000.jl")

# load data
dat = CSV.read("test/data/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names(dat)))

# subset observables in data
data = data(observables,:)


# Handling distributions with varying parameters using arraydist
dists = [
    Beta(0.356, 0.02, μσ = true),           # alp
    Beta(0.993, 0.002, μσ = true),          # bet
    Normal(0.0085, 0.003),                  # gam
    Normal(1.0002, 0.007),                  # mst
    Beta(0.129, 0.223, μσ = true),          # rho
    Beta(0.65, 0.05, μσ = true),            # psi
    Beta(0.01, 0.005, μσ = true),           # del
    InverseGamma(0.035449, Inf, μσ = true), # z_e_a
    InverseGamma(0.008862, Inf, μσ = true)  # z_e_m
]

Turing.@model function FS2000_loglikelihood_function(data, m)
    all_params ~ Turing.arraydist(dists)

    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext() 
        Turing.@addlogprob! get_loglikelihood(m, data, all_params, verbose = false)
    end
end

FS2000_loglikelihood = FS2000_loglikelihood_function(data, FS2000)


n_samples = 1000

# using Zygote
# Turing.setadbackend(:zygote)
# samps = @time sample(FS2000_loglikelihood, NUTS(), n_samples, progress = true, initial_params = FS2000.parameter_values)

# println("Mean variable values (ForwardDiff): $(mean(samps).nt.mean)")

# samps = @time sample(FS2000_loglikelihood, NUTS(adtype = Turing.AutoZygote()), n_samples, progress = true, initial_params = FS2000.parameter_values)

# println("Mean variable values (Zygote): $(mean(samps).nt.mean)")

# sample_nuts = mean(samps).nt.mean


# generate a Pigeons log potential
FS2000_lp = Pigeons.TuringLogPotential(FS2000_loglikelihood_function(data, FS2000))

# find a feasible starting point
pt = Pigeons.pigeons(target = FS2000_lp, n_rounds = 0, n_chains = 1)

replica = pt.replicas[end]
XMAX = deepcopy(replica.state)
LPmax = FS2000_lp(XMAX)

i = 0

while !isfinite(LPmax) && i < 1000
    Pigeons.sample_iid!(FS2000_lp, replica, pt.shared)
    new_LP = FS2000_lp(replica.state)
    if new_LP > LPmax
        LPmax = new_LP
        XMAX  = deepcopy(replica.state)
    end
    i += 1
end

# define a specific initialization for this model
Pigeons.initialization(::Pigeons.TuringLogPotential{typeof(FS2000_loglikelihood_function)}, ::AbstractRNG, ::Int64) = deepcopy(XMAX)

Pigeons.pigeons(target = FS2000_lp,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 1,
            n_rounds = 10,
            multithreaded = true)

pt = @profview Pigeons.pigeons(target = FS2000_lp,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 1,
            n_rounds = 1,
            multithreaded = false)
        #     ────────────────────────────────────────────────────────────────────────────
        #     scans     restarts    time(s)    allc(B)  log(Z₁/Z₀)   min(αₑ)   mean(αₑ) 
        #   ────────── ────────── ────────── ────────── ────────── ────────── ──────────
        #           2          0        4.8   3.21e+09          0      0.982      0.982 
        #           4          0       7.17   4.99e+09          0          1          1 
        #           8          0       12.5   8.73e+09          0          1          1 
        #          16          0       28.5   1.97e+10          0          1          1 
        #          32          0       56.2   3.89e+10          0          1          1 
        #          64          0        109   7.47e+10          0          1          1 
        #         128          0        230   1.57e+11          0          1          1 
samps = MCMCChains.Chains(pt)

println("Mean variable values (Pigeons): $(mean(samps).nt.mean)")

sample_pigeons = mean(samps).nt.mean


modeFS2000 = Turing.maximum_a_posteriori(FS2000_loglikelihood, 
                                        # Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 2)), 
                                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                                        # Optim.NelderMead(), 
                                        adtype = AutoZygote(), 
                                        # maxiters = 100,
                                        # lb = [0,0,-10,-10,0,0,0,0,0], 
                                        # ub = [1,1,10,10,1,1,1,100,100], 
                                        initial_params = FS2000.parameter_values)

println("Mode variable values: $(modeFS2000.values); Mode loglikelihood: $(modeFS2000.lp)")

@testset "Estimation results" begin
    # @test isapprox(modeFS2000.lp, 1281.669108730447, rtol = eps(Float32))
    @test isapprox(sample_nuts, [0.40248024934137033, 0.9905235783816697, 0.004618184988033483, 1.014268215459915, 0.8459140293740781, 0.6851143053372912, 0.0025570276255960107, 0.01373547787288702, 0.003343985776134218], rtol = 1e-2)
    @test isapprox(sample_pigeons[1:length(sample_nuts)], [0.40248024934137033, 0.9905235783816697, 0.004618184988033483, 1.014268215459915, 0.8459140293740781, 0.6851143053372912, 0.0025570276255960107, 0.01373547787288702, 0.003343985776134218], rtol = 1e-2)
end



plot_model_estimates(FS2000, data, parameters = sample_nuts)
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
