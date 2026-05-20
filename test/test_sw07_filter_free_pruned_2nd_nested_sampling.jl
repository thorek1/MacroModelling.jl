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
#   * parameter space : a representative subset of structural parameters
#                       plus me_std plus a small panel of latent shocks
#
# Dimensionality is kept modest so nested sampling stays tractable on CI.
# ──────────────────────────────────────────────────────────────────────────────

USE_NESSAI = true

NESSAI_NLIVE              = 200
NESSAI_FLOW_POOLSIZE      = 64
NESSAI_FLOW_DRAWSIZE      = NESSAI_FLOW_POOLSIZE
NESSAI_UNINFORMED_POOLSIZE = NESSAI_FLOW_POOLSIZE
NESSAI_MAXIMUM_UNINFORMED  = 2 * NESSAI_NLIVE
NESSAI_LOG_LEVEL           = "INFO"
NESSAI_LOGGING_INTERVAL    = 200

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
# Sampled parameter subset + priors
# ──────────────────────────────────────────────────────────────────────────────
const SW07_SUBSET = [:crhoa, :crhob, :crhog, :csadjcost, :chabb, :csigma, :cprobw]

function sw07_ff_subset_indices()
    pnames = SW07_FF.constants.post_complete_parameters.parameters
    idx    = Int[]
    for p in SW07_SUBSET
        j = findfirst(==(p), pnames)
        j === nothing || push!(idx, j)
    end
    @assert length(idx) == length(SW07_SUBSET) "Could not resolve all SW07 subset names"
    return idx
end
const SUBSET_IDX = sw07_ff_subset_indices()

const BASE_PARAMS = copy(SW07_FF.parameter_values)
const N_EXO       = length(get_shocks(SW07_FF))
const N_SHOCKS_FF = N_EXO * T_FF
const N_OBS_FF    = length(OBS_FF)

# Structural-parameter priors (loosely-informative Beta/Gamma around defaults).
function bounded_beta(μ, σ, lo, hi)
    return Beta(μ, σ, lo, hi, μσ = true)
end

structural_dists = [
    bounded_beta(0.95,  0.04, 0.01, 0.999),  # crhoa
    bounded_beta(0.18,  0.10, 0.01, 0.999),  # crhob
    bounded_beta(0.97,  0.02, 0.01, 0.999),  # crhog
    Normal(5.74, 1.50, 1.0, 15.0),           # csadjcost
    bounded_beta(0.71,  0.10, 0.01, 0.999),  # chabb
    Normal(1.38, 0.50, 0.25, 5.0),           # csigma
    bounded_beta(0.74,  0.10, 0.30, 0.95),   # cprobw
]
me_std_dist = InverseGamma(0.05, 2.0, 1e-4, 1.0, μσ = true)
shock_dist  = Normal(0.0, 1.0, -5.0, 5.0)

dists_ff = vcat(structural_dists, [me_std_dist], fill(shock_dist, N_SHOCKS_FF))

const PARAM_NAMES_FF = vcat(
    SW07_SUBSET,
    [:me_std],
    [Symbol("shock_", i) for i in 1:N_SHOCKS_FF],
)
@assert length(dists_ff) == length(PARAM_NAMES_FF)

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
    nP     = length(SUBSET_IDX)
    θ      = params[1:nP]
    me_std = params[nP + 1]
    shocks = reshape(params[nP + 2:end], N_EXO, T_FF)

    full = copy(BASE_PARAMS)
    @inbounds for (k, j) in enumerate(SUBSET_IDX)
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
        output                     = nessai_output_dir,
        nlive                      = NESSAI_NLIVE,
        seed                       = 1234,
        resume                     = false,
        disable_vectorisation      = true,
        logging_interval           = NESSAI_LOGGING_INTERVAL,
        log_on_iteration           = true,
        maximum_uninformed         = NESSAI_MAXIMUM_UNINFORMED,
        uninformed_proposal        = RejectionProposal,
        uninformed_proposal_kwargs = pydict(Dict("poolsize" => NESSAI_UNINFORMED_POOLSIZE)),
        flow_config                = pydict(NESSAI_FLOW_CONFIG),
        poolsize                   = NESSAI_FLOW_POOLSIZE,
        drawsize                   = NESSAI_FLOW_DRAWSIZE,
        plot                       = false,
        proposal_plots             = false,
    )
    nessai_fs.run(plot = false, save = false)
    println("nessai filter-free pruned-2nd estimation completed")

    nessai_log_evidence       = pyconvert(Float64, nessai_fs.logZ)
    nessai_posterior_samples  = nessai_fs.posterior_samples
    nessai_posterior_matrix   = reduce(hcat, [
        pyconvert(Vector{Float64}, nessai_posterior_samples[string(name)])
        for name in PARAM_NAMES_FF
    ])
    n_posterior = size(nessai_posterior_matrix, 1)
    println("nessai (FF pruned-2nd) log evidence: $nessai_log_evidence")
    println("nessai (FF pruned-2nd) number of posterior samples: $n_posterior")

    @testset "nessai SW07 pruned-2nd filter-free estimation" begin
        @test isfinite(nessai_log_evidence)
        @test n_posterior > 0
        @test !isnothing(nessai_fs)
    end

end # USE_NESSAI
