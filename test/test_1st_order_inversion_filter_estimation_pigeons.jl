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

const PIGEONS_SEED = 30

# ---------------------------------------------------------------------------
# Filter-free estimation via Pigeons (gradient-free MCMC; joint sampling of
# parameters + latent shocks).  First order (inversion-filter script).
# ---------------------------------------------------------------------------
import Turing: MvNormal
import LinearAlgebra: I as LinearAlgebraI

const T_ff_1st = size(data, 2)
const data_ff_1st = data
const nExo_ff_1st = length(get_shocks(FS2000))

Turing.@model function FS2000_filter_free_function_1st(data, m, algorithm, nExo, nT, on_failure_loglikelihood)
    all_params  ~ Turing.product_distribution(dists)
    me_std      ~ InverseGamma(0.05, Inf, μσ = true)
    shocks_vec  ~ MvNormal(zeros(nExo * nT), LinearAlgebraI)
    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
        shocks  = reshape(shocks_vec, nExo, nT)
        Turing.@addlogprob! get_filter_free_loglikelihood(m, data, all_params, shocks, me_std;
                                                          algorithm = algorithm,
                                                          on_failure_loglikelihood = on_failure_loglikelihood)
    end
end

FS2000_ff_lp = Pigeons.TuringLogPotential(
    FS2000_filter_free_function_1st(data_ff_1st, FS2000, :first_order, nExo_ff_1st, T_ff_1st, -floatmax(Float64)+1e10))

init_ff_params = (; all_params = FS2000.parameter_values,
                    me_std     = 0.05,
                    shocks_vec = zeros(nExo_ff_1st * T_ff_1st))

const FS2000_FF_LP = typeof(FS2000_ff_lp)

function Pigeons.initialization(target::FS2000_FF_LP, rng::AbstractRNG, _::Int64)
    result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.SampleFromPrior(), DynamicPPL.PriorContext())
    result = DynamicPPL.initialize_parameters!!(result, init_ff_params, target.model)
    return result
end

pt_ff = @time Pigeons.pigeons(target = FS2000_ff_lp,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 2,
            n_rounds = 10,
            seed = PIGEONS_SEED,
            multithreaded = false)

samps_ff = MCMCChains.Chains(pt_ff)
println("Filter-free (Pigeons, first order, inversion script) — mean: $(mean(samps_ff).nt.mean)")
@test size(samps_ff, 1) > 0


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

const FS2000_LP = typeof(FS2000_lp)

function Pigeons.initialization(target::FS2000_LP, rng::AbstractRNG, _::Int64)
    result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.SampleFromPrior(), DynamicPPL.PriorContext())
    # DynamicPPL.link!!(result, DynamicPPL.SampleFromPrior(), target.model)
    
    result = DynamicPPL.initialize_parameters!!(result, init_params, target.model)

    return result
end

pt = Pigeons.pigeons(target = FS2000_lp, n_rounds = 0, n_chains = 1, seed = PIGEONS_SEED)

pt = @time Pigeons.pigeons(target = FS2000_lp,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 2,
            n_rounds = 10,
            seed = PIGEONS_SEED,
            multithreaded = false) # tests fail on multithreaded

samps = MCMCChains.Chains(pt)

println("Mean variable values (Pigeons): $(mean(samps).nt.mean)")
