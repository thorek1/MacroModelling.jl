using MacroModelling
using Test
import Turing
import Pigeons
using Random, DelimitedFiles, MCMCChains, AxisKeys
import DynamicPPL

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

Turing.@model function FS2000_loglikelihood_function(data, m, filter, on_failure_loglikelihood; verbose = false)
    all_params ~ Turing.product_distribution(dists)

    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext() 
        llh = get_loglikelihood(m, 
                                data, 
                                all_params, 
                                filter = filter,
                                on_failure_loglikelihood = on_failure_loglikelihood)
        maybe_print_loglikelihood(verbose, llh, dists, all_params)

        Turing.@addlogprob! llh
    end
end

# generate a Pigeons log potential
FS2000_lp = Pigeons.TuringLogPotential(FS2000_loglikelihood_function(data, FS2000, :inversion, -floatmax(Float64)+1e10)) #, verbose = true))

init_params = FS2000.parameter_values
const PIGEONS_SEED = 30

const FS2000_LP = typeof(FS2000_lp)

function Pigeons.initialization(target::FS2000_LP, rng::AbstractRNG, _::Int64)
    result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.SampleFromPrior(), DynamicPPL.PriorContext())
    result = DynamicPPL.link!!(result, target.model)

    result = DynamicPPL.initialize_parameters!!(result, init_params, target.model)

    return result
end

pt = Pigeons.pigeons(target = FS2000_lp, n_rounds = 0, n_chains = 1, seed = PIGEONS_SEED)

# ---------------------------------------------------------------------------
# Run the missing-data Pigeons estimation FIRST so failures surface early.
# ---------------------------------------------------------------------------
data_missing = inject_missing_observations(data)

FS2000_lp_missing = Pigeons.TuringLogPotential(FS2000_loglikelihood_function(data_missing, FS2000, :inversion, -floatmax(Float64)+1e10))

pt_missing = Pigeons.pigeons(target = FS2000_lp_missing, n_rounds = 0, n_chains = 1, seed = PIGEONS_SEED+1)

pt_missing = @time Pigeons.pigeons(target = FS2000_lp_missing,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 2,
            n_rounds = 10,
            seed = PIGEONS_SEED+1,
            multithreaded = false)

samps_missing = MCMCChains.Chains(pt_missing)

sample_pigeons_missing = mean(samps_missing).nt.mean
println("Mean variable values (Pigeons, missing data): $(sample_pigeons_missing)")

@testset "Pigeons Estimation results (1st order inversion, missing data)" begin
    @test length(sample_pigeons_missing) >= 9
    @test all(isfinite, sample_pigeons_missing[1:9])
end

pt = @time Pigeons.pigeons(target = FS2000_lp,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 2,
            n_rounds = 10,
            seed = PIGEONS_SEED,
            multithreaded = false) # tests fail on multithreaded

samps = MCMCChains.Chains(pt)

println("Mean variable values (Pigeons): $(mean(samps).nt.mean)")
