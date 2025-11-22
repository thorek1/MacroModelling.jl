using MacroModelling
using Test
import Turing
import ADTypes: AutoZygote
import Pigeons
import Turing: NUTS, sample, logpdf, PG, IS
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL

# estimate highly nonlinear model

# load data
dat = CSV.read("data/usmodel.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

# declare observables
observables = [:dy]#, :dinve, :labobs, :pinfobs, :dw, :robs]

# Subsample from 1966Q1 - 2004Q4
# subset observables in data
data = data(observables,75:230)


include("models/Caldara_et_al_2012_estim.jl")


# get_loglikelihood(Caldara_et_al_2012_estim, data, Caldara_et_al_2012_estim.parameter_values, algorithm = :pruned_third_order)

# get_loglikelihood(Caldara_et_al_2012_estim, data, Caldara_et_al_2012_estim.parameter_values*0.99, algorithm = :pruned_third_order)


# get_parameters(Caldara_et_al_2012_estim, values = true)

# Handling distributions with varying parameters using arraydist
dists = [
    Normal(0, 1),                           # dȳ
    Normal(0, 1),                           # dc̄
    Beta(0.95, 0.005, μσ = true),           # β
    Beta(0.33, 0.05, μσ = true),            # ζ
    Beta(0.02, 0.01, μσ = true),            # δ
    Beta(0.75, 0.01, μσ = true),            # λ
    Normal(1, .25),                         # ψ
    InverseGamma(0.021, Inf, μσ = true),    # σ̄
    InverseGamma(0.1, Inf, μσ = true),      # η
    Beta(0.75, 0.02, μσ = true)             # ρ
]

Turing.@model function Caldara_et_al_2012_loglikelihood_function(data, m, on_failure_loglikelihood)
    all_params ~ Turing.arraydist(dists)

    Turing.@addlogprob! (; loglikelihood=get_loglikelihood(m, 
                                            data, 
                                            all_params, 
                                            algorithm = :third_order, 
                                            on_failure_loglikelihood = on_failure_loglikelihood))
end


Random.seed!(3)

Caldara_et_al_2012_loglikelihood = Caldara_et_al_2012_loglikelihood_function(data, Caldara_et_al_2012_estim, -Inf)

# samps = @time sample(Caldara_et_al_2012_loglikelihood, PG(100), 10, progress = true)#, init_params = sol)

# samps = sample(Caldara_et_al_2012_loglikelihood, IS(), 1000, progress = true)#, init_params = sol)

mode_estimateNM = Turing.maximum_a_posteriori(Caldara_et_al_2012_loglikelihood, 
                                                Optim.NelderMead(),
                                                iterations = 100,
                                                # show_trace = true,
                                                initial_params = Caldara_et_al_2012_estim.parameter_values)

mode_estimateLBFGS = Turing.maximum_a_posteriori(Caldara_et_al_2012_loglikelihood, 
                                                Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)),
                                                adtype = AutoZygote(),
                                                iterations = 100,
                                                # show_trace = true,
                                                initial_params = mode_estimateNM.values)

init_params = mode_estimateLBFGS.values |> collect

println("Mode variable values (L-BFGS): $init_params")

n_samples = 100

samps = sample(Caldara_et_al_2012_loglikelihood, NUTS(250, 0.65, adtype = AutoZygote()), n_samples, progress = true, initial_params = init_params)

println("Mean variable values (Zygote): $(mean(samps).nt.mean)")

sample_nuts = mean(samps).nt.mean


# generate a Pigeons log potential
Caldara_lp = Pigeons.TuringLogPotential(Caldara_et_al_2012_loglikelihood_function(data, Caldara_et_al_2012_estim, -floatmax(Float64)+1e10))

LLH = Turing.logjoint(Caldara_et_al_2012_loglikelihood_function(data, Caldara_et_al_2012_estim, -floatmax(Float64)+1e10), (all_params = init_params,))

if isfinite(LLH)
    const Caldara_LP = typeof(Caldara_lp)

    function Pigeons.initialization(target::Caldara_LP, rng::AbstractRNG, _::Int64)
        result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.SampleFromPrior(), DynamicPPL.PriorContext())
        # DynamicPPL.link!!(result, DynamicPPL.SampleFromPrior(), target.model)
        
        result = DynamicPPL.initialize_parameters!!(result, init_params, target.model)

        return result
    end

    pt = Pigeons.pigeons(target = Caldara_lp, n_rounds = 0, n_chains = 1)
else
    pt = Pigeons.pigeons(target = Caldara_lp, n_rounds = 0, n_chains = 1)

    replica = pt.replicas[end]
    XMAX = deepcopy(replica.state)
    LPmax = Caldara_lp(XMAX)

    i = 0

    while !isfinite(LPmax) && i < 1000
        Pigeons.sample_iid!(Caldara_lp, replica, pt.shared)
        new_LP = Caldara_lp(replica.state)
        if new_LP > LPmax
            global LPmax = new_LP
            global XMAX  = deepcopy(replica.state)
        end
        global i += 1
    end

    # define a specific initialization for this model
    Pigeons.initialization(::Pigeons.TuringLogPotential{typeof(Caldara_et_al_2012_loglikelihood_function)}, ::AbstractRNG, ::Int64) = deepcopy(XMAX)
end

pt = @time Pigeons.pigeons(target = Caldara_lp,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 1,
            n_rounds = 8,
            multithreaded = false)

samps = MCMCChains.Chains(pt)


println("Mean variable values (Pigeons): $(mean(samps).nt.mean)")

@testset "Pigeons 3rd order estimation" begin
    # Pigeons test completed successfully
    @test true
end


FS2000 = nothing
