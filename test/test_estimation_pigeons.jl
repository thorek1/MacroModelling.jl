using MacroModelling
using Test
import Turing, Pigeons
import ADTypes: AutoZygote
import Turing: NUTS, sample, logpdf
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

Turing.@model function FS2000_loglikelihood_function(data, m, on_failure_loglikelihood; verbose = false)
    all_params ~ Turing.arraydist(dists)

    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext() 
        llh = get_loglikelihood(m, 
                                 data, 
                                 all_params, 
                                 on_failure_loglikelihood = on_failure_loglikelihood)
        if verbose
            @info "Loglikelihood: $llh and prior llh: $(Turing.logpdf(Turing.arraydist(dists), all_params)) with params $all_params"
        end

        Turing.@addlogprob! llh
    end
end

# generate a Pigeons log potential
FS2000_lp = Pigeons.TuringLogPotential(FS2000_loglikelihood_function(data, FS2000, -floatmax(Float64)+1e10))

init_params = FS2000.parameter_values

const FS2000_LP = typeof(FS2000_lp)

function Pigeons.initialization(target::FS2000_LP, rng::AbstractRNG, _::Int64)
    result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.SampleFromPrior(), DynamicPPL.PriorContext())
    
    result = DynamicPPL.initialize_parameters!!(result, init_params, target.model)

    return result
end

pt = Pigeons.pigeons(target = FS2000_lp, n_rounds = 0, n_chains = 1)

pt = @time Pigeons.pigeons(target = FS2000_lp,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 1,
            n_rounds = 10,
            multithreaded = false) # tests fail on multithreaded

samps = MCMCChains.Chains(pt)

println("Mean variable values (Pigeons): $(mean(samps).nt.mean)")

sample_pigeons = mean(samps).nt.mean

@testset "Pigeons Estimation results" begin
    @test isapprox(sample_pigeons[1:9], [0.40248024934137033, 0.9905235783816697, 0.004618184988033483, 1.014268215459915, 0.8459140293740781, 0.6851143053372912, 0.0025570276255960107, 0.01373547787288702, 0.003343985776134218], rtol = 1e-2)
end

plot_model_estimates(FS2000, data, parameters = sample_pigeons)

FS2000 = nothing
