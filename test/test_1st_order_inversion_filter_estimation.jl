using Test
using MacroModelling
import Turing
import Mooncake
import Turing: NUTS, sample
import ADTypes
import ADTypes: AutoMooncake
import Turing: MvNormal
import LinearAlgebra: I as LinearAlgebraI
import DifferentiationInterface
import FiniteDifferences
import Optim, LineSearches
import LinearAlgebra as ℒ
using Random, DelimitedFiles, AxisKeys

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

@testset "Filter-free NUTS (first order, inversion script)" begin
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


# ---------------------------------------------------------------------------
# Replicate the estimation problem on data with missing observations.
# ---------------------------------------------------------------------------

Turing.@model function FS2000_loglikelihood_function(data, m, filter, on_failure_loglikelihood; verbose = false)
    all_params ~ Turing.product_distribution(dists)

    llh = get_loglikelihood(m, 
                            data, 
                            all_params, 
                            filter = filter,
                            on_failure_loglikelihood = on_failure_loglikelihood)
    maybe_print_loglikelihood(verbose, llh, dists, all_params)

    Turing.@addlogprob! llh
end


samps = @time sample(FS2000_loglikelihood_function(data, FS2000, :inversion, -Inf), NUTS(adtype = AutoMooncake(; config=nothing)), n_samples, progress = true, initial_params = Turing.InitFromParams((; all_params = FS2000.parameter_values)))


posterior_summary = FlexiChains.summarystats(samps)
show(stdout, MIME"text/plain"(), posterior_summary)
println()
println("Mean variable values (Mooncake): $(collect(values(FlexiChains.mean(samps); parameters_only = true)))")

sample_nuts = collect(values(FlexiChains.mean(samps); parameters_only = true))

modeFS2000i = Turing.maximum_a_posteriori(FS2000_loglikelihood_function(data, FS2000, :inversion, -Inf), 
                                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                                        adtype = AutoMooncake(; config=nothing), 
                                        initial_params = Turing.InitFromParams((; all_params = FS2000.parameter_values)))

println("Mode variable values: $(modeFS2000i.params); Mode loglikelihood: $(modeFS2000i.lp)")

@testset "Mooncake vs FiniteDifferences gradient (1st order inversion)" begin
    back_grad = DifferentiationInterface.gradient(x -> get_loglikelihood(FS2000, data, x, filter = :inversion), ADTypes.AutoMooncake(config = nothing), FS2000.parameter_values)
    @test !isnothing(back_grad)
    @test all(isfinite, back_grad)

    for i in 1:100
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4, 1), x -> get_loglikelihood(FS2000, data, x, filter = :inversion), FS2000.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences converged after $i iterations")
            @test isapprox(back_grad, fin_grad[1], rtol = 1e-4)
            break
        end
    end
end


# ---------------------------------------------------------------------------
# Replicate the estimation problem on data with missing observations.
# ---------------------------------------------------------------------------
data_missing = inject_missing_observations(data)

samps_missing = @time sample(FS2000_loglikelihood_function(data_missing, FS2000, :inversion, -Inf), NUTS(adtype = AutoMooncake(; config=nothing)), n_samples, progress = true, initial_params = Turing.InitFromParams((; all_params = FS2000.parameter_values)))


posterior_summary_missing = FlexiChains.summarystats(samps_missing)
show(stdout, MIME"text/plain"(), posterior_summary_missing)
println()
println("Mean variable values (Mooncake, missing data): $(collect(values(FlexiChains.mean(samps_missing); parameters_only = true)))")

sample_nuts_missing = collect(values(FlexiChains.mean(samps_missing); parameters_only = true))

modeFS2000i_missing = Turing.maximum_a_posteriori(FS2000_loglikelihood_function(data_missing, FS2000, :inversion, -Inf),
                                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)),
                                        adtype = AutoMooncake(; config=nothing),
                                        initial_params = Turing.InitFromParams((; all_params = FS2000.parameter_values)))

println("Mode variable values (missing data): $(modeFS2000i_missing.params); Mode loglikelihood: $(modeFS2000i_missing.lp)")

@testset "Estimation results (1st order inversion, missing data)" begin
    @test all(isfinite, sample_nuts_missing)
    @test length(sample_nuts_missing) == length(FS2000.parameter_values)
    @test isfinite(modeFS2000i_missing.lp)
end

@testset "Mooncake vs FiniteDifferences gradient (1st order inversion, missing data)" begin
    # Pass model and data_missing as Constant contexts (not closure captures)
    # so Mooncake doesn't run `__verify_const` against the captured globals.
    # That check uses `==`, which returns `false` for NaN-bearing arrays even
    # when the array is the same object — triggering an assertion failure.
    loglik_target(x, m, d) = get_loglikelihood(m, d, x, filter = :inversion)
    back_grad = DifferentiationInterface.gradient(loglik_target,
        ADTypes.AutoMooncake(config = nothing), FS2000.parameter_values,
        DifferentiationInterface.Constant(FS2000),
        DifferentiationInterface.Constant(data_missing))
    @test !isnothing(back_grad)
    @test all(isfinite, back_grad)

    for i in 1:100
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4, 1), x -> get_loglikelihood(FS2000, data_missing, x, filter = :inversion), FS2000.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences converged after $i iterations")
            @test isapprox(back_grad, fin_grad[1], rtol = 1e-4)
            break
        end
    end
end


# # estimate highly nonlinear model


# # load data
# dat, header = readdlm("data/usmodel.csv", ',', header = true)
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
#     β   ~ Beta(0.95, 0.005, μσ = true)
#     ζ   ~ Beta(0.33, 0.05, μσ = true)
#     δ   ~ Beta(0.02, 0.01, μσ = true)
#     λ   ~ Beta(0.75, 0.01, μσ = true)
#     ψ   ~ Normal(1, .25)#, μσ = true)
#     σ̄   ~ InverseGamma(0.021, Inf, μσ = true)
#     η   ~ InverseGamma(0.1, Inf, μσ = true)
#     ρ   ~ Beta(0.75, 0.02, μσ = true)

#     Turing.@addlogprob! get_loglikelihood(m, data, [dȳ, dc̄, β, ζ, δ, λ, ψ, σ̄, η, ρ], algorithm = :pruned_third_order)
# end


# Random.seed!(3)

# pt = @time Pigeons.pigeons(target = Pigeons.TuringLogPotential(Caldara_et_al_2012_loglikelihood_function(data, Caldara_et_al_2012_estim)),
#             record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
#             n_chains = 1,
#             n_rounds = 6,
#             multithreaded = false)

# samps = pigeons_flexichain(Pigeons.sample_array(pt), Pigeons.sample_names(pt))


# println(collect(values(FlexiChains.mean(samps); parameters_only = true)))


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
#             log_lik -= logpdf(Beta(0.993, 0.05, μσ = true),β)
#             log_lik -= logpdf(Beta(0.356, 0.05, μσ = true),ζ)
#             log_lik -= logpdf(Beta(0.02, 0.01, μσ = true),δ)
#             log_lik -= logpdf(Beta(0.5, 0.25, μσ = true),λ)
#             log_lik -= logpdf(Normal(1, .25),ψ)
#             # log_lik -= logpdf(Normal(40, 10),γ)
#             log_lik -= logpdf(InverseGamma(0.021, Inf, μσ = true),σ̄)
#             log_lik -= logpdf(InverseGamma(0.1, Inf, μσ = true),η)
#             log_lik -= logpdf(Beta(0.5, 0.25, μσ = true),ρ)
        
#             return log_lik
#         end, parameters)
#     end
#     dȳ, dc̄, β, ζ, δ, λ, ψ, σ̄, η, ρ = parameters
#     # println(parameters)
#     log_lik = 0
#     log_lik -= get_loglikelihood(Caldara_et_al_2012_estim, data, parameters, algorithm = :pruned_third_order)
#     log_lik -= logpdf(Normal(0, 1),dȳ)
#     log_lik -= logpdf(Normal(0, 1),dc̄)
#     log_lik -= logpdf(Beta(0.95, 0.005, μσ = true),β)
#     log_lik -= logpdf(Beta(0.33, 0.05, μσ = true),ζ)
#     log_lik -= logpdf(Beta(0.02, 0.01, μσ = true),δ)
#     log_lik -= logpdf(Beta(0.75, 0.01, μσ = true),λ)
#     log_lik -= logpdf(Normal(1, .25),ψ)
#     # log_lik -= logpdf(Normal(40, 10),γ)
#     log_lik -= logpdf(InverseGamma(0.021, Inf, μσ = true),σ̄)
#     log_lik -= logpdf(InverseGamma(0.1, Inf, μσ = true),η)
#     log_lik -= logpdf(Beta(0.75, 0.02, μσ = true),ρ)
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
#     @test isapprox(collect(values(FlexiChains.mean(samps); parameters_only = true)), [0.40248024934137033, 0.9905235783816697, 0.004618184988033483, 1.014268215459915, 0.8459140293740781, 0.6851143053372912, 0.0025570276255960107, 0.01373547787288702, 0.003343985776134218], rtol = 1e-2)
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

