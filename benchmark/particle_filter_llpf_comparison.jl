# Benchmark: MacroModelling's particle filter vs LowLevelParticleFilters.jl
#
# Compares the log-likelihood and per-evaluation wall time of MacroModelling's
# built-in particle filters (`filter = :particle`) against a bootstrap
# `ParticleFilter` from LowLevelParticleFilters.jl, on the same first-order
# (linear) DSGE state space, cross-checked against the exact Kalman likelihood.
#
# LowLevelParticleFilters and Distributions are NOT dependencies of the package;
# run this in a throwaway environment, e.g.
#
#   julia --project=/tmp/pfbench -e '
#       using Pkg; Pkg.develop(path="."); Pkg.add(["LowLevelParticleFilters","Distributions","AxisKeys","DelimitedFiles"])'
#   julia --project=/tmp/pfbench benchmark/particle_filter_llpf_comparison.jl
#
# The DSGE structure is mapped onto LLPF's generic interface as
#   xₜ = A·xₜ₋₁ + wₜ,   wₜ ~ N(0, B Bᵀ)      (rank-deficient process noise)
#   yₜ = xₜ[observables] + vₜ,  vₜ ~ N(0, H)
# with the initial cloud drawn from the ergodic covariance Σ (discrete Lyapunov).

using MacroModelling
using LowLevelParticleFilters
using Distributions
using Random, DelimitedFiles, AxisKeys
import Statistics
import LinearAlgebra as ℒ
import MacroModelling: get_relevant_steady_state_and_state_update,
                       particle_initial_state_covariance, merge_calculation_options

# Pull the exact first-order state space (deviation form) the package filter uses.
# The initial-state covariance comes from the package's own Lyapunov solver via
# `particle_initial_state_covariance`, i.e. the very routine the particle filter
# uses to spread its initial cloud, so both filters start from the same prior.
function extract_linear(m, data_levels, observables, params)
    constants, SS_and_pars, 𝐒, _, _ = get_relevant_steady_state_and_state_update(Val(:first_order), params, m)
    T = constants.post_model_macro
    nVars = T.nVars; nPast = T.nPast_not_future_and_mixed
    ssnames = constants.post_complete_parameters.SS_and_pars_names
    obs_idx = convert(Vector{Int}, indexin(observables, ssnames))
    A = zeros(nVars, nVars); A[:, T.past_not_future_and_mixed_idx] .= 𝐒[:, 1:nPast]
    B = Matrix(𝐒[:, nPast+1:end])
    dev = collect(data_levels) .- SS_and_pars[obs_idx]
    Σ, _ = particle_initial_state_covariance(m, T, merge_calculation_options(), :theoretical)
    return A, B, obs_idx, Σ, dev, nVars
end

function bench_model(name, m, data, observables, me; N = 20000, nseed = 8)
    params = m.parameter_values
    kal = get_loglikelihood(m, data(observables), params; filter = :kalman,
                            presample_periods = 0, initial_covariance = :theoretical,
                            measurement_error = me .^ 2)
    println("\n==== $name (N=$N) ====")
    println("Kalman+ME = ", round(kal, digits = 3))

    for pf_filter in (:bootstrap_particle, :tempered_particle)
        Nn = pf_filter == :tempered_particle ? N ÷ 3 : N
        t0 = time()
        lls = [get_loglikelihood(m, data(observables), params; filter = pf_filter,
                    algorithm = :first_order, presample_periods = 0, initial_covariance = :theoretical,
                    measurement_error = me .^ 2,
                    n_particles = Nn, particle_rng = Random.Xoshiro(s)) for s in 1:nseed]
        println("MacroModelling ", rpad(String(pf_filter), 19), " N=$Nn  mean=", round(Statistics.mean(lls), digits = 2),
                "  std=", round(Statistics.std(lls), digits = 2), "  time/run=", round((time() - t0) / nseed, digits = 3), "s")
    end

    A, B, obs_idx, Σ, dev, nVars = extract_linear(m, data(observables), observables, params)
    me_var = (me isa AbstractVector ? collect(me) : fill(me, length(observables))) .^ 2
    nObs = length(observables); nT = size(dev, 2)
    df = MvNormal(zeros(nVars), ℒ.Symmetric(B * B') + 1e-10ℒ.I)
    dg = MvNormal(zeros(nObs), ℒ.Diagonal(me_var))
    d0 = MvNormal(zeros(nVars), ℒ.Symmetric(Σ) + 1e-10ℒ.I)
    u = [Float64[] for _ in 1:nT]
    y = [collect(dev[:, t]) for t in 1:nT]
    t0 = time()
    lls = map(1:nseed) do s
        Random.seed!(s)
        loglik(ParticleFilter(N, (x, u, p, t) -> A * x, (x, u, p, t) -> x[obs_idx], df, dg, d0), u, y)
    end
    println("LLPF           ", rpad("bootstrap", 10), " N=$N  mean=", round(Statistics.mean(lls), digits = 2),
            "  std=", round(Statistics.std(lls), digits = 2), "  time/run=", round((time() - t0) / nseed, digits = 3), "s")
end

@model RBC2 begin
    1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α * exp(g[0])
    z[0] = ρz * z[-1] + std_z * eps_z[x]
    g[0] = ρg * g[-1] + std_g * eps_g[x]
end
@parameters RBC2 begin
    std_z = 0.01; std_g = 0.01; ρz = 0.4; ρg = 0.6; δ = 0.02; α = 0.5; β = 0.95
end
Random.seed!(12345)
data_rbc = MacroModelling.simulate(RBC2, periods = 40)([:c, :q], :, :simulate)
bench_model("RBC (2 obs)", RBC2, data_rbc, [:c, :q], 0.002; N = 20000, nseed = 8)

dat, header = readdlm(joinpath(@__DIR__, "..", "test", "data", "usmodel.csv"), ',', header = true)
dat = Float64.(dat); csv = vec(Symbol.(strip.(header)))
dsw = KeyedArray(dat', Variable = csv, Time = axes(dat, 1))([:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs], 47:230)
obs_sw = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
dsw = rekey(dsw, :Variable => obs_sw)
include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007_linear.jl"))
SS(Smets_Wouters_2007_linear, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01, :cmap => 0.01, :cmaw => 0.01])
bench_model("SW07 (7 obs)", Smets_Wouters_2007_linear, dsw, obs_sw,
            2.0 .* [Statistics.std(collect(dsw(o))) for o in obs_sw]; N = 20000, nseed = 6)
