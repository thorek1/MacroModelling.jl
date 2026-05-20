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
    Beta(0.356, 0.02, eps(Float64), 1 - eps(Float64), μσ = true),           # alp
    Beta(0.993, 0.002, eps(Float64), 1 - eps(Float64), μσ = true),          # bet
    Normal(0.0085, 0.003),                  # gam
    Normal(1.0002, 0.007),                  # mst
    Beta(0.129, 0.223, eps(Float64), 1 - eps(Float64), μσ = true),          # rho
    Beta(0.65, 0.05, eps(Float64), 1 - eps(Float64), μσ = true),            # psi
    Beta(0.01, 0.005, eps(Float64), 1 - eps(Float64), μσ = true),           # del
    InverseGamma(0.035449, Inf, eps(Float64), Inf, μσ = true),              # z_e_a
    InverseGamma(0.008862, Inf, eps(Float64), Inf, μσ = true)               # z_e_m
]

const PIGEONS_SEED = 30

# ---------------------------------------------------------------------------
# Filter-free estimation via Pigeons (gradient-free MCMC; joint sampling of
# parameters + latent shocks).
# ---------------------------------------------------------------------------
import Turing: MvNormal
import LinearAlgebra: I as LinearAlgebraI

const T_ff_pruned2nd = size(data, 2)
const data_ff_pruned2nd = data
const nExo_ff_pruned2nd = length(get_shocks(FS2000))

Turing.@model function FS2000_filter_free_function(data, m, algorithm, nExo, nT, on_failure_loglikelihood)
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
    FS2000_filter_free_function(data_ff_pruned2nd, FS2000, :pruned_second_order, nExo_ff_pruned2nd, T_ff_pruned2nd, -floatmax(Float64)+1e10))

init_ff_params = (; all_params = FS2000.parameter_values,
                    me_std     = 0.05,
                    shocks_vec = zeros(nExo_ff_pruned2nd * T_ff_pruned2nd))

const FS2000_FF_LP = typeof(FS2000_ff_lp)

function Pigeons.initialization(target::FS2000_FF_LP, rng::AbstractRNG, _::Int64)
    result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.SampleFromPrior(), DynamicPPL.PriorContext())
    result = DynamicPPL.initialize_parameters!!(result, init_ff_params, target.model)
    return result
end

pt_ff = @time Pigeons.pigeons(target = FS2000_ff_lp,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 1,
            n_rounds = 8,
            seed = PIGEONS_SEED,
            multithreaded = false)

samps_ff = MCMCChains.Chains(pt_ff)
println("Filter-free (Pigeons, pruned second order) — mean: $(mean(samps_ff).nt.mean)")
@test size(samps_ff, 1) > 0


Turing.@model function FS2000_loglikelihood_function(data, m, algorithm, on_failure_loglikelihood; verbose = false)
    all_params ~ Turing.product_distribution(dists)

    llh = get_loglikelihood(m,
                             data,
                             all_params,
                             algorithm = algorithm,
                             on_failure_loglikelihood = on_failure_loglikelihood)
    maybe_print_loglikelihood(verbose, llh, dists, all_params)

    Turing.@addlogprob! llh
end



# generate a Pigeons log potential
FS2000_pruned2nd_lp = Pigeons.TuringLogPotential(FS2000_loglikelihood_function(data, FS2000, :pruned_second_order, -floatmax(Float64)+1e10)) #, verbose = true))

#=
const FS2000_pruned2nd_LP = typeof(FS2000_pruned2nd_lp)

init_params = FS2000.parameter_values

LLH = Turing.logjoint(FS2000_loglikelihood_function(data, FS2000, :pruned_second_order, -floatmax(Float64)+1e10, verbose = false), (all_params = init_params,))

if isfinite(LLH)
    function Pigeons.initialization(target::FS2000_pruned2nd_LP, rng::AbstractRNG, _::Int64)
        result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.InitFromParams((; all_params = init_params)))
        result = DynamicPPL.link!!(result, target.model)

        # result = DynamicPPL.initialize_parameters(result, init_params, target.model)

        return result
    end

    pt = Pigeons.pigeons(target = FS2000_pruned2nd_lp, n_rounds = 0, n_chains = 1, seed = PIGEONS_SEED)
else
    pt = Pigeons.pigeons(target = FS2000_pruned2nd_lp, n_rounds = 0, n_chains = 1, seed = PIGEONS_SEED)
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
    Pigeons.initialization(::FS2000_pruned2nd_LP, ::AbstractRNG, ::Int64) = deepcopy(XMAX)
end
=#

# ---------------------------------------------------------------------------
# Run the missing-data Pigeons estimation FIRST so failures surface early.
# ---------------------------------------------------------------------------
data_missing = inject_missing_observations(data)

FS2000_pruned2nd_lp_missing = Pigeons.TuringLogPotential(FS2000_loglikelihood_function(data_missing, FS2000, :pruned_second_order, -floatmax(Float64)+1e10))

#=
const FS2000_pruned2nd_LP_MISSING = typeof(FS2000_pruned2nd_lp_missing)

function Pigeons.initialization(target::FS2000_pruned2nd_LP_MISSING, rng::AbstractRNG, _::Int64)
    result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.InitFromParams((; all_params = init_params)))
    result = DynamicPPL.link!!(result, target.model)

    # result = DynamicPPL.initialize_parameters(result, init_params, target.model)

    return result
end

pt_missing = Pigeons.pigeons(target = FS2000_pruned2nd_lp_missing, n_rounds = 0, n_chains = 1, seed = PIGEONS_SEED)
=#

pt_missing = @time Pigeons.pigeons(target = FS2000_pruned2nd_lp_missing,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 1,
            n_rounds = 8,
            seed = PIGEONS_SEED,
            multithreaded = false)

samps_missing = MCMCChains.Chains(pt_missing)

sample_pigeons_missing = mean(samps_missing).nt.mean
println("Mean variable values (Pigeons, pruned 2nd order, missing data): $(sample_pigeons_missing)")

@testset "Pigeons Estimation results (pruned 2nd order, missing data)" begin
    @test length(sample_pigeons_missing) >= 9
    @test all(isfinite, sample_pigeons_missing[1:9])
end

pt = Pigeons.pigeons(target = FS2000_pruned2nd_lp, n_rounds = 0, n_chains = 1, seed = PIGEONS_SEED + 2)

pt = @time Pigeons.pigeons(target = FS2000_pruned2nd_lp,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 1,
            n_rounds = 8,
            seed = PIGEONS_SEED + 2,
            multithreaded = false)

samps = MCMCChains.Chains(pt)


println("Mean variable values (pruned second order): $(mean(samps).nt.mean)")
