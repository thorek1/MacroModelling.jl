using MacroModelling
using Test
import Turing
import Pigeons
import Turing: NUTS, sample, logpdf
import ADTypes: AutoZygote
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

Turing.@model function FS2000_loglikelihood_function(data, m, filter, on_failure_loglikelihood; verbose = false)
    all_params ~ Turing.arraydist(dists)

    llh = get_loglikelihood(m, 
                            data, 
                            all_params, 
                            filter = filter,
                            on_failure_loglikelihood = on_failure_loglikelihood)
    if verbose
        @info "Loglikelihood: $llh and prior llh: $(Turing.logpdf(Turing.arraydist(dists), all_params)) with params $all_params"
    end

    Turing.@addlogprob! (; loglikelihood=llh)
end

n_samples = 1000

samps = @time sample(FS2000_loglikelihood_function(data, FS2000, :inversion, -Inf), NUTS(adtype = AutoZygote()), n_samples, progress = true, initial_params = (all_params = FS2000.parameter_values,))


println("Mean variable values (Zygote): $(mean(samps).nt.mean)")

sample_nuts = mean(samps).nt.mean

modeFS2000i = Turing.maximum_a_posteriori(FS2000_loglikelihood_function(data, FS2000, :inversion, -Inf), 
                                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                                        adtype = AutoZygote(), 
                                        initial_params = (all_params = FS2000.parameter_values,))

println("Mode variable values: $(modeFS2000i.values); Mode loglikelihood: $(modeFS2000i.lp)")

FS2000_lp = Pigeons.TuringLogPotential(FS2000_loglikelihood_function(data, FS2000, :inversion, -floatmax(Float64)+1e10)) #, verbose = true))

init_params = FS2000.parameter_values

const FS2000_LP = typeof(FS2000_lp)

function Pigeons.initialization(target::FS2000_LP, rng::AbstractRNG, _::Int64)
    result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.SampleFromPrior(), DynamicPPL.PriorContext())
    # DynamicPPL.link!!(result, DynamicPPL.SampleFromPrior(), target.model)
    
    result = DynamicPPL.initialize_parameters!!(result, init_params, target.model)

    return result
end

pt = Pigeons.pigeons(target = FS2000_lp, n_rounds = 0, n_chains = 1)

pt = @time Pigeons.pigeons(target = FS2000_lp,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 2,
            n_rounds = 10,
            multithreaded = false)

samps = MCMCChains.Chains(pt)

println("Mean variable values (Pigeons): $(mean(samps).nt.mean)")

@testset "Pigeons 1st order inversion filter estimation" begin
    # Pigeons test completed successfully
    @test true
end


FS2000 = nothing
