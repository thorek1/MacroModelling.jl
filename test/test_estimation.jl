using Test
using MacroModelling
import Turing
import Mooncake
import ADTypes
import ADTypes: AutoMooncake
import DifferentiationInterface
import FiniteDifferences
import Turing: NUTS, sample
import Optim, LineSearches
import LinearAlgebra as ℒ
using Random, DelimitedFiles, AxisKeys
import StatsPlots

using FlexiChains
include("test_helpers.jl")

include("../models/FS2000.jl")

# load data
dat, header = readdlm("data/FS2000_data.csv", ',', header = true)
dat = Float64.(dat)
names = vec(header)
data = KeyedArray(dat', Variable = Symbol.("log_".*names), Time = axes(dat, 1))
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names))

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

n_samples = 1000

# ---------------------------------------------------------------------------
# Filter-free estimation (joint sampling of parameters and latent shocks)
# First-order — analytical rrule + Mooncake AD
# ---------------------------------------------------------------------------
import ADTypes: AutoForwardDiff
import Turing: MvNormal
import LinearAlgebra: I as LinearAlgebraI

const T_ff_1st = size(data, 2)
const data_ff_1st = data
const nExo_ff_1st = length(get_shocks(FS2000))

Turing.@model function FS2000_filter_free_function_1st(data, m, algorithm, nExo, nT, on_failure_loglikelihood)
    all_params  ~ Turing.product_distribution(dists)
    me_std      ~ InverseGamma(0.05, Inf, μσ = true)
    shocks_vec  ~ MvNormal(zeros(nExo * nT), LinearAlgebraI)
    shocks      = collect(reshape(shocks_vec, nExo, nT))
    Turing.@addlogprob! get_filter_free_loglikelihood(m, data, all_params, shocks, me_std;
                                                      algorithm = algorithm,
                                                      on_failure_loglikelihood = on_failure_loglikelihood)
end

Random.seed!(30)

@testset "Filter-free NUTS (first order)" begin
    init_ff = (; all_params = FS2000.parameter_values,
                 me_std     = 0.05,
                 shocks_vec = zeros(nExo_ff_1st * T_ff_1st))
    ff_samps = @time sample(
        FS2000_filter_free_function_1st(data_ff_1st, FS2000, :first_order, nExo_ff_1st, T_ff_1st, -Inf),
        NUTS(adtype = AutoMooncake(; config=nothing)),
        n_samples,
        progress = true,
        initial_params = Turing.InitFromParams(init_ff))
    posterior_summary = FlexiChains.summarystats(ff_samps)
    show(stdout, MIME"text/plain"(), posterior_summary)
    println()
    println("Mean variable values (filter-free, first order): $(collect(values(FlexiChains.mean(ff_samps); parameters_only = true)))")
    @test size(ff_samps, 1) == n_samples
end


Turing.@model function FS2000_loglikelihood_function(data, m, on_failure_loglikelihood; verbose = false)
    all_params ~ Turing.product_distribution(dists)

    llh = get_loglikelihood(m, 
                             data, 
                             all_params, 
                             on_failure_loglikelihood = on_failure_loglikelihood)
    maybe_print_loglikelihood(verbose, llh, dists, all_params)

    Turing.@addlogprob! llh
    # with Turing >= 0.40 this becomes: Turing.@addlogprob! (; loglikelihood = llh)
end

FS2000_loglikelihood = FS2000_loglikelihood_function(data, FS2000, -Inf)



samps = @time sample(FS2000_loglikelihood, NUTS(adtype = AutoMooncake(; config=nothing)), n_samples, progress = true, initial_params = Turing.InitFromParams((; all_params = FS2000.parameter_values)))
posterior_summary = FlexiChains.summarystats(samps)
show(stdout, MIME"text/plain"(), posterior_summary)
println()
println("Mean variable values (Mooncake): $(collect(values(FlexiChains.mean(samps); parameters_only = true)))")

get_steady_state(FS2000, steady_state_function = FS2000_custom_steady_state_function!)

samps = @time sample(FS2000_loglikelihood, NUTS(adtype = AutoMooncake(; config=nothing)), n_samples, progress = true, initial_params = Turing.InitFromParams((; all_params = FS2000.parameter_values)))
posterior_summary = FlexiChains.summarystats(samps)
show(stdout, MIME"text/plain"(), posterior_summary)
println()
println("Mean variable values (Mooncake + custom steady state): $(collect(values(FlexiChains.mean(samps); parameters_only = true)))")

get_steady_state(FS2000, steady_state_function = nothing)

samps = @time sample(FS2000_loglikelihood, NUTS(), n_samples, progress = true, initial_params = Turing.InitFromParams((; all_params = FS2000.parameter_values)))


posterior_summary = FlexiChains.summarystats(samps)
show(stdout, MIME"text/plain"(), posterior_summary)
println()
println("Mean variable values (ForwardDiff): $(collect(values(FlexiChains.mean(samps); parameters_only = true)))")

sample_nuts = collect(values(FlexiChains.mean(samps); parameters_only = true))


modeFS2000 = Turing.maximum_a_posteriori(FS2000_loglikelihood, 
                                        # Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 2)), 
                                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                                        # Optim.NelderMead(), 
                                        adtype = AutoMooncake(; config=nothing), 
                                        # maxiters = 100,
                                        # lb = [0,0,-10,-10,0,0,0,0,0], 
                                        # ub = [1,1,10,10,1,1,1,100,100], 
                                        initial_params = Turing.InitFromParams((; all_params = FS2000.parameter_values)))

println("Mode variable values: $(modeFS2000.params); Mode loglikelihood: $(modeFS2000.lp)")

@testset "Estimation results" begin
    # @test isapprox(modeFS2000.lp, 1281.669108730447, rtol = eps(Float32))
    @test isapprox(sample_nuts, [0.40248024934137033, 0.9905235783816697, 0.004618184988033483, 1.014268215459915, 0.8459140293740781, 0.6851143053372912, 0.0025570276255960107, 0.01373547787288702, 0.003343985776134218], rtol = 1e-2)
end

@testset "Mooncake vs FiniteDifferences gradient (1st order Kalman)" begin
    back_grad = DifferentiationInterface.gradient(x -> get_loglikelihood(FS2000, data, x), ADTypes.AutoMooncake(config = nothing), FS2000.parameter_values)
    @test !isnothing(back_grad)
    @test all(isfinite, back_grad)

    for i in 1:100
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4, 1), x -> get_loglikelihood(FS2000, data, x), FS2000.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences converged after $i iterations")
            @test isapprox(back_grad, fin_grad[1], rtol = 1e-4)
            break
        end
    end
end

plot_model_estimates(FS2000, data, parameters = sample_nuts)
plot_shock_decomposition(FS2000, data)

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

