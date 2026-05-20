using Test
using MacroModelling
import Turing
using PythonCall
using Random, AxisKeys

using FlexiChains
using FlexiChains: Parameter, FlexiChain
using DataStructures: OrderedDict


include("test_helpers.jl")

# ──────────────────────────────────────────────────────────────────────────────
# Nested sampling (nessai FlowSampler) for Smets-Wouters 2007 nonlinear at
# pruned-second-order with the FILTER-FREE likelihood
# (joint sampling of parameters, latent shocks, and measurement-error std).
#
# Mirrors test_sw07_estimation_nested_sampling.jl (which targets the linear
# model + Kalman filter), but:
#   * model      : Smets_Wouters_2007 (nonlinear)
#   * algorithm  : :pruned_second_order
#   * likelihood : get_filter_free_loglikelihood
#   * parameter space : the full SW07 estimation parameter vector used in the
#                       other SW07 estimation tests, plus observable-specific
#                       measurement-error stds, plus all latent shocks over the
#                       synthetic sample window.
# ──────────────────────────────────────────────────────────────────────────────

USE_NESSAI = true

NESSAI_NLIVE = 1500
NESSAI_FLOW_POOLSIZE = 128
NESSAI_FLOW_DRAWSIZE = NESSAI_FLOW_POOLSIZE
NESSAI_UNINFORMED_POOLSIZE = NESSAI_FLOW_POOLSIZE
NESSAI_MAXIMUM_UNINFORMED = 2 * NESSAI_NLIVE
NESSAI_LOG_LEVEL = "INFO"
NESSAI_LOGGING_INTERVAL = 500
NESSAI_IMPORTANCE_NESTED_SAMPLER = false
NESSAI_RESET_FLOW = false

NESSAI_FLOW_CONFIG = Dict{String,Any}(
    "ftype" => "nsf",
)

# ──────────────────────────────────────────────────────────────────────────────
# Install nessai
# ──────────────────────────────────────────────────────────────────────────────
println("Installing nessai...")
using CondaPkg
USE_NESSAI && CondaPkg.add_pip("nessai")
CondaPkg.resolve()
println("nessai installed")

# ──────────────────────────────────────────────────────────────────────────────
# Load nonlinear SW07
# ──────────────────────────────────────────────────────────────────────────────
include("../models/Smets_Wouters_2007.jl")
const SW07_FF = Smets_Wouters_2007

# Use a small under-identified observable set to keep the model well-conditioned.
const OBS_FF = [:dy, :dc, :dinve]

# Synthetic data: steady-state with small noise (deterministic, reproducible).
const T_FF   = 6
function ss_perturbed_data(model, observables; periods, σ, seed)
    SS     = get_steady_state(model)
    ss_obs = collect(SS(observables, :Steady_state))
    Random.seed!(seed)
    dat = repeat(ss_obs, 1, periods) .+ σ .* randn(length(observables), periods)
    return KeyedArray(dat; Variables = observables, Time = 1:periods)
end
const DATA_FF = ss_perturbed_data(SW07_FF, OBS_FF; periods = T_FF, σ = 1e-4, seed = 17)

# ──────────────────────────────────────────────────────────────────────────────
# Sampled SW07 parameter vector + priors
# ──────────────────────────────────────────────────────────────────────────────
const SW07_PARAM_NAMES_FF = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew,
                             :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow,
                             :cmap, :cmaw,
                             :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap,
                             :cfc, :crpi, :crr, :cry, :crdy,
                             :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

function sw07_ff_parameter_indices()
    pnames = SW07_FF.constants.post_complete_parameters.parameters
    idx    = Int[]
    for p in SW07_PARAM_NAMES_FF
        j = findfirst(==(p), pnames)
        j === nothing || push!(idx, j)
    end
    @assert length(idx) == length(SW07_PARAM_NAMES_FF) "Could not resolve all SW07 parameter names"
    return idx
end
const PARAM_IDX_FF = sw07_ff_parameter_indices()

const BASE_PARAMS = copy(SW07_FF.parameter_values)
const N_EXO       = length(get_shocks(SW07_FF))
const N_SHOCKS_FF = N_EXO * T_FF
const N_OBS_FF    = length(OBS_FF)
const ME_STD_NAMES_FF = [Symbol("me_std_", obs) for obs in OBS_FF]

param_dists_ff = [
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),     # z_ea
    InverseGamma(0.1, 2.0, 0.025, 5.0, μσ = true),    # z_eb
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),     # z_eg
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),     # z_eqs
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),     # z_em
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),     # z_epinf
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),     # z_ew
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),          # crhoa
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),          # crhob
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),          # crhog
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),          # crhoqs
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),          # crhoms
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),          # crhopinf
    Beta(0.5, 0.2, 0.001, 0.9999, μσ = true),         # crhow
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),          # cmap
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),          # cmaw
    Normal(4.0, 1.5, 2.0, 15.0),                      # csadjcost
    Normal(1.50, 0.375, 0.25, 3.0),                   # csigma
    Beta(0.7, 0.1, 0.001, 0.99, μσ = true),          # chabb
    Beta(0.5, 0.1, 0.3, 0.95, μσ = true),            # cprobw
    Normal(2.0, 0.75, 0.25, 10.0),                    # csigl
    Beta(0.5, 0.10, 0.5, 0.95, μσ = true),           # cprobp
    Beta(0.5, 0.15, 0.01, 0.99, μσ = true),          # cindw
    Beta(0.5, 0.15, 0.01, 0.99, μσ = true),          # cindp
    Beta(0.5, 0.15, 0.01, 0.99999, μσ = true),       # czcap
    Normal(1.25, 0.125, 1.0, 3.0),                    # cfc
    Normal(1.5, 0.25, 1.0, 3.0),                      # crpi
    Beta(0.75, 0.10, 0.5, 0.975, μσ = true),         # crr
    Normal(0.125, 0.05, 0.001, 0.5),                  # cry
    Normal(0.125, 0.05, 0.001, 0.5),                  # crdy
    Gamma(0.625, 0.1, 0.1, 2.0, μσ = true),           # constepinf
    Gamma(0.25, 0.1, 0.01, 2.0, μσ = true),           # constebeta
    Normal(0.0, 2.0, -10.0, 10.0),                    # constelab
    Normal(0.4, 0.10, 0.1, 0.8),                      # ctrend
    Normal(0.5, 0.25, 0.01, 2.0),                     # cgy
    Normal(0.3, 0.05, 0.01, 1.0),                     # calfa
]
me_std_dists_ff = fill(InverseGamma(0.05, Inf, 1e-6, 1.0, μσ = true), N_OBS_FF)
shock_dist  = Normal(0.0, 1.0, -5.0, 5.0)
shock_dists_ff = fill(shock_dist, N_SHOCKS_FF)

dists_ff = vcat(param_dists_ff, me_std_dists_ff, shock_dists_ff)

const PARAM_NAMES_FF = vcat(
    SW07_PARAM_NAMES_FF,
    ME_STD_NAMES_FF,
    [Symbol("shock_", i) for i in 1:N_SHOCKS_FF],
)
@assert length(dists_ff) == length(PARAM_NAMES_FF)

Turing.@model function SW07_filter_free_function(data, m, algorithm, on_failure_loglikelihood)
    all_params ~ Turing.product_distribution(param_dists_ff)
    me_std     ~ Turing.product_distribution(me_std_dists_ff)
    shocks_vec ~ Turing.product_distribution(shock_dists_ff)
    shocks      = collect(reshape(shocks_vec, N_EXO, T_FF))
    Turing.@addlogprob! get_filter_free_loglikelihood(m, data, all_params, shocks, me_std;
                                                      algorithm = algorithm,
                                                      on_failure_loglikelihood = on_failure_loglikelihood)
end

# ──────────────────────────────────────────────────────────────────────────────
# Julia callbacks for nessai
# ──────────────────────────────────────────────────────────────────────────────
function sw07_ff_log_prior_density(params::Vector{Float64})
    lp = 0.0
    @inbounds for i in eachindex(dists_ff)
        lp += Turing.logpdf(dists_ff[i], params[i])
    end
    return lp
end

function sw07_ff_log_likelihood(params::Vector{Float64})
    nP     = length(PARAM_IDX_FF)
    θ      = params[1:nP]
    nM     = N_OBS_FF
    me_std = params[nP + 1:nP + nM]
    shocks = reshape(params[nP + nM + 1:end], N_EXO, T_FF)

    full = copy(BASE_PARAMS)
    @inbounds for (k, j) in enumerate(PARAM_IDX_FF)
        full[j] = θ[k]
    end

    return get_filter_free_loglikelihood(
        SW07_FF, DATA_FF, full, shocks, me_std;
        algorithm = :pruned_second_order,
        on_failure_loglikelihood = -1e10,
    )
end

names_py  = [string(n) for n in PARAM_NAMES_FF]
bounds_py = Dict(string(n) => (Float64(minimum(d)), Float64(maximum(d)))
                 for (n, d) in zip(PARAM_NAMES_FF, dists_ff))

function posterior_matrix_from_named_samples(named_samples)
    return reduce(hcat, [
        pyconvert(Vector{Float64}, named_samples[string(name)]) for name in PARAM_NAMES_FF
    ])
end

function summarize_posterior_matrix(label::String, posterior_matrix::Matrix{Float64})
    n_posterior = size(posterior_matrix, 1)
    println("$label number of posterior samples: $n_posterior")
    if n_posterior == 0
        println("$label returned no posterior samples")
        return n_posterior, nothing
    end

    println("$label posterior means:")
    posterior_means = Dict{Symbol,Float64}()
    for (i, name) in pairs(PARAM_NAMES_FF)
        col = @view posterior_matrix[:, i]
        mean_value = sum(col) / length(col)
        posterior_means[Symbol(name)] = mean_value
        println("  $name: $mean_value")
    end

    if !HAVE_FLEXICHAINS
        println("$label FlexiChains summary skipped: FlexiChains not available in the active project environment")
        return n_posterior, posterior_means
    end

    n_iters, _ = size(posterior_matrix)
    symbol_names = Symbol.(collect(PARAM_NAMES_FF))
    chain_data = OrderedDict{FlexiChains.ParameterOrExtra{Symbol}, Matrix{eltype(posterior_matrix)}}()
    for (column, name) in pairs(symbol_names)
        chain_data[Parameter(name)] = reshape(collect(@view posterior_matrix[:, column]), n_iters, 1)
    end
    posterior_chain = FlexiChain{Symbol}(n_iters, 1, chain_data)
    posterior_summary = FlexiChains.summarystats(posterior_chain)
    println("$label FlexiChains summary:")
    show(stdout, MIME"text/plain"(), posterior_summary)
    println()
    return n_posterior, posterior_summary
end

# ──────────────────────────────────────────────────────────────────────────────
# nessai FlowSampler
# ──────────────────────────────────────────────────────────────────────────────
if USE_NESSAI

    function nessai_log_prior(params_py)
        return sw07_ff_log_prior_density(pyconvert(Vector{Float64}, params_py))
    end

    function nessai_log_likelihood(params_py)
        return sw07_ff_log_likelihood(pyconvert(Vector{Float64}, params_py))
    end

    nessai_tmpdir = mktempdir()
    write(joinpath(nessai_tmpdir, "sw07_ff_nessai_model.py"), """
    import numpy as np
    from nessai.model import Model

    class SW07FFNessaiModel(Model):

        allow_vectorised = False
        allow_vectorised_prior = False
        likelihood_chunksize = 1

        def __init__(self, param_names, param_bounds, jl_log_prior, jl_log_likelihood):
            self.names = list(param_names)
            self.bounds = dict(param_bounds)
            self._jl_log_prior = jl_log_prior
            self._jl_log_likelihood = jl_log_likelihood

        def _as_points(self, x):
            x_array = np.asarray(x)
            if x_array.shape == ():
                return [x], True
            return x_array, False

        def log_prior(self, x):
            points, scalar_input = self._as_points(x)
            log_p = np.zeros(len(points))
            for i, point in enumerate(points):
                params = [float(point[n]) for n in self.names]
                log_p[i] = float(self._jl_log_prior(params))
            if scalar_input:
                return log_p[0]
            return log_p

        def log_likelihood(self, x):
            points, scalar_input = self._as_points(x)
            log_l = np.zeros(len(points))
            for i, point in enumerate(points):
                params = [float(point[n]) for n in self.names]
                log_l[i] = float(self._jl_log_likelihood(params))
            if scalar_input:
                return log_l[0]
            return log_l
    """)

    sys_mod = pyimport("sys")
    sys_mod.path.insert(0, nessai_tmpdir)
    sw07_ff_module = pyimport("sw07_ff_nessai_model")

    FlowSampler            = pyimport("nessai.flowsampler").FlowSampler
    RejectionProposal      = pyimport("nessai.proposal").RejectionProposal
    configure_nessai_logger = pyimport("nessai.utils.logging").configure_logger

    nessai_model = sw07_ff_module.SW07FFNessaiModel(
        names_py, bounds_py, nessai_log_prior, nessai_log_likelihood)

    nessai_output_dir = pwd()
    println("Running nessai FlowSampler on SW07 (nonlinear, pruned-2nd, filter-free)...")
    configure_nessai_logger(
        output    = nessai_output_dir,
        label     = "",
        log_level = NESSAI_LOG_LEVEL,
        stream    = "stdout",
    )
    nessai_fs = FlowSampler(nessai_model;
        output = nessai_output_dir,
        # importance_nested_sampler = NESSAI_IMPORTANCE_NESTED_SAMPLER,
        nlive = NESSAI_NLIVE,
        seed = 1234,
        # pytorch_threads = 1,
        resume = false,
        disable_vectorisation = true,
        logging_interval = NESSAI_LOGGING_INTERVAL,
        log_on_iteration = true,
        maximum_uninformed = NESSAI_MAXIMUM_UNINFORMED,
        uninformed_proposal = RejectionProposal,
        uninformed_proposal_kwargs = pydict(Dict("poolsize" => NESSAI_UNINFORMED_POOLSIZE)),
        flow_config = pydict(NESSAI_FLOW_CONFIG),
        # training_config = pydict(NESSAI_TRAINING_CONFIG),
        # reset_flow = NESSAI_RESET_FLOW,
        # retrain_acceptance = NESSAI_RETRAIN_ACCEPTANCE,
        # acceptance_threshold = NESSAI_ACCEPTANCE_THRESHOLD,
        poolsize = NESSAI_FLOW_POOLSIZE,
        drawsize = NESSAI_FLOW_DRAWSIZE,
        plot = false,
        proposal_plots = false,
    )
    nessai_fs.run(plot = false, save = false)
    println("nessai filter-free pruned-2nd estimation completed")

    nessai_log_evidence       = pyconvert(Float64, nessai_fs.logZ)
    nessai_posterior_samples  = nessai_fs.posterior_samples
    nessai_n_posterior, nessai_posterior_summary = summarize_posterior_matrix(
        "nessai (FF pruned-2nd)",
        posterior_matrix_from_named_samples(nessai_posterior_samples),
    )
    println("nessai (FF pruned-2nd) log evidence: $nessai_log_evidence")

    @testset "nessai SW07 pruned-2nd filter-free estimation" begin
        @test isfinite(nessai_log_evidence)
        @test nessai_n_posterior > 0
        @test !isnothing(nessai_posterior_summary)
        @test !isnothing(nessai_fs)
    end

end # USE_NESSAI
