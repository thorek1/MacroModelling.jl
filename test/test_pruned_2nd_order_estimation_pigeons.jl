using MacroModelling
using Test
import Turing
import Pigeons
import ADTypes: AutoZygote
import Turing: NUTS, sample, logpdf
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

Turing.@model function FS2000_loglikelihood_function(data, m, algorithm, on_failure_loglikelihood; verbose = false)
    all_params ~ Turing.arraydist(dists)

    llh = get_loglikelihood(m, 
                             data, 
                             all_params, 
                             algorithm = algorithm, 
                             on_failure_loglikelihood = on_failure_loglikelihood)
    if verbose
        @info "Loglikelihood: $llh and prior llh: $(Turing.logpdf(Turing.arraydist(dists), all_params)) with params $all_params"
    end

    Turing.@addlogprob! (; loglikelihood=llh)
end


Random.seed!(30)

n_samples = 500

samps = @time sample(FS2000_loglikelihood_function(data, FS2000, :pruned_second_order, -Inf), NUTS(adtype = AutoZygote()), n_samples, progress = true, initial_params = FS2000.parameter_values)


println("Mean variable values (Zygote): $(mean(samps).nt.mean)")

sample_nuts = mean(samps).nt.mean

# generate a Pigeons log potential
FS2000_pruned2nd_lp = Pigeons.TuringLogPotential(FS2000_loglikelihood_function(data, FS2000, :pruned_second_order, -floatmax(Float64)+1e10)) #, verbose = true))

init_params = sample_nuts

LLH = Turing.logjoint(FS2000_loglikelihood_function(data, FS2000, :pruned_second_order, -floatmax(Float64)+1e10, verbose = false), (all_params = init_params,))

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
    pt = Pigeons.pigeons(target = FS2000_pruned2nd_lp, n_rounds = 0, n_chains = 1)
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

@testset "Pigeons pruned 2nd order estimation" begin
    # Pigeons test completed successfully
    @test true
end


FS2000 = nothing
