using MacroModelling
using Test
import Turing
import Pigeons
using Random, DelimitedFiles, MCMCChains, AxisKeys
import DynamicPPL

include("test_helpers.jl")

# estimate highly nonlinear model

# load data
dat, header = readdlm("data/usmodel.csv", ',', header = true)
dat = Float64.(dat)
names = vec(Symbol.(strip.(header)))
data = KeyedArray(dat', Variable = names, Time = axes(dat, 1))

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
    Beta(0.95, 0.005, eps(Float64), 1 - eps(Float64), μσ = true),           # β
    Beta(0.33, 0.05, eps(Float64), 1 - eps(Float64), μσ = true),            # ζ
    Beta(0.02, 0.01, eps(Float64), 1 - eps(Float64), μσ = true),            # δ
    Beta(0.75, 0.01, eps(Float64), 1 - eps(Float64), μσ = true),            # λ
    Normal(1, .25),                         # ψ
    InverseGamma(0.021, Inf, eps(Float64), Inf, μσ = true),                 # σ̄
    InverseGamma(0.1, Inf, eps(Float64), Inf, μσ = true),                   # η
    Beta(0.75, 0.02, eps(Float64), 1 - eps(Float64), μσ = true)             # ρ
]

Turing.@model function Caldara_et_al_2012_loglikelihood_function(data, m, on_failure_loglikelihood)
    all_params ~ Turing.product_distribution(dists)

    llh = get_loglikelihood(m,
                            data,
                            all_params,
                            algorithm = :third_order,
                            on_failure_loglikelihood = on_failure_loglikelihood)
    Turing.@addlogprob! llh
end


const PIGEONS_SEED = 3

Caldara_et_al_2012_loglikelihood = Caldara_et_al_2012_loglikelihood_function(data, Caldara_et_al_2012_estim, -Inf)

# samps = @time sample(Caldara_et_al_2012_loglikelihood, PG(100), 10, progress = true)#, init_params = sol)

# generate a Pigeons log potential
Caldara_lp = Pigeons.TuringLogPotential(Caldara_et_al_2012_loglikelihood_function(data, Caldara_et_al_2012_estim, -floatmax(Float64)+1e10))

#=
const Caldara_LP = typeof(Caldara_lp)

init_params = Caldara_et_al_2012_estim.parameter_values

LLH = Turing.logjoint(Caldara_et_al_2012_loglikelihood_function(data, Caldara_et_al_2012_estim, -floatmax(Float64)+1e10), (all_params = init_params,))

if isfinite(LLH)
    function Pigeons.initialization(target::Caldara_LP, rng::AbstractRNG, _::Int64)
        result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.InitFromParams((; all_params = init_params)))
        result = DynamicPPL.link!!(result, target.model)

        # result = DynamicPPL.initialize_parameters(result, init_params, target.model)

        return result
    end

    pt = Pigeons.pigeons(target = Caldara_lp, n_rounds = 0, n_chains = 1, seed = PIGEONS_SEED)
else
    pt = Pigeons.pigeons(target = Caldara_lp, n_rounds = 0, n_chains = 1, seed = PIGEONS_SEED)

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
    Pigeons.initialization(::Caldara_LP, ::AbstractRNG, ::Int64) = deepcopy(XMAX)
end
=#

# ---------------------------------------------------------------------------
# Run the missing-data Pigeons estimation FIRST so failures surface early.
# ---------------------------------------------------------------------------
data_missing = inject_missing_observations(data)

Caldara_lp_missing = Pigeons.TuringLogPotential(Caldara_et_al_2012_loglikelihood_function(data_missing, Caldara_et_al_2012_estim, -floatmax(Float64)+1e10))

#=
const Caldara_LP_MISSING = typeof(Caldara_lp_missing)

function Pigeons.initialization(target::Caldara_LP_MISSING, rng::AbstractRNG, _::Int64)
    result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.InitFromParams((; all_params = init_params)))
    result = DynamicPPL.link!!(result, target.model)

    # result = DynamicPPL.initialize_parameters(result, init_params, target.model)

    return result
end

pt_missing = Pigeons.pigeons(target = Caldara_lp_missing, n_rounds = 0, n_chains = 1, seed = PIGEONS_SEED)
=#

pt_missing = @time Pigeons.pigeons(target = Caldara_lp_missing,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 4,
            n_rounds = 8,
            seed = PIGEONS_SEED,
            multithreaded = false)

samps_missing = MCMCChains.Chains(pt_missing)

sample_pigeons_missing = mean(samps_missing).nt.mean
println("Mean variable values (Pigeons, 3rd order, missing data): $(sample_pigeons_missing)")

@testset "Pigeons Estimation results (3rd order, missing data)" begin
    n_params = length(Caldara_et_al_2012_estim.parameter_values)
    @test length(sample_pigeons_missing) >= n_params
    @test all(isfinite, sample_pigeons_missing[1:n_params])
end

pt = Pigeons.pigeons(target = Caldara_lp, n_rounds = 0, n_chains = 1, seed = PIGEONS_SEED)

pt = @time Pigeons.pigeons(target = Caldara_lp,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 4,
            n_rounds = 8,
            seed = PIGEONS_SEED,
            multithreaded = false) # tests fail on multithreaded

samps = MCMCChains.Chains(pt)


println("Mean variable values (Pigeons): $(mean(samps).nt.mean)")
