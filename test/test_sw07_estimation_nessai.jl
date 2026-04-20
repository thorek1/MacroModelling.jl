using Test
using MacroModelling
import Turing
using MCMCChains
using PythonCall
using DelimitedFiles, AxisKeys

const NESSAI_NLIVE = 1000
const NESSAI_UNINFORMED_POOLSIZE = 1000 # 64
const NESSAI_FLOW_POOLSIZE = 1000 # 64
const NESSAI_FLOW_DRAWSIZE = 1000 # 64
const NESSAI_LOG_LEVEL = "INFO"
const NESSAI_LOGGING_INTERVAL = 500

# ──────────────────────────────────────────────────────────────────────────────
# Install nessai into PythonCall's Python environment
# ──────────────────────────────────────────────────────────────────────────────
println("Installing nessai...")
using CondaPkg
CondaPkg.add_pip("nessai")
CondaPkg.resolve()
println("nessai installed successfully")

# ──────────────────────────────────────────────────────────────────────────────
# Load data (identical to test_sw07_estimation.jl)
# ──────────────────────────────────────────────────────────────────────────────
dat, header = readdlm("data/usmodel.csv", ',', header = true)
dat = Float64.(dat)
col_names = vec(Symbol.(strip.(header)))

data = KeyedArray(dat', Variable = col_names, Time = axes(dat, 1))

observables_old = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs]
sample_idx = 47:230 # 1960Q1-2004Q4
data = data(observables_old, sample_idx)

observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
data = rekey(data, :Variable => observables)

# ──────────────────────────────────────────────────────────────────────────────
# Define priors (identical to test_sw07_estimation.jl)
# ──────────────────────────────────────────────────────────────────────────────
dists = [
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_ea
    InverseGamma(0.1, 2.0, 0.025,5.0, μσ = true),    # z_eb
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),    # z_eg
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),    # z_eqs
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),    # z_em
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),    # z_epinf
    InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),    # z_ew
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),         # crhoa
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),         # crhob
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),         # crhog
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),         # crhoqs
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),         # crhoms
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),         # crhopinf
    Beta(0.5, 0.2, 0.001,0.9999, μσ = true),         # crhow
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),         # cmap
    Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),         # cmaw
    Normal(4.0, 1.5,   2.0, 15.0),                   # csadjcost
    Normal(1.50,0.375, 0.25, 3.0),                   # csigma
    Beta(0.7, 0.1, 0.001, 0.99, μσ = true),          # chabb
    Beta(0.5, 0.1, 0.3, 0.95, μσ = true),            # cprobw
    Normal(2.0, 0.75, 0.25, 10.0),                   # csigl
    Beta(0.5, 0.10, 0.5, 0.95, μσ = true),           # cprobp
    Beta(0.5, 0.15, 0.01, 0.99, μσ = true),          # cindw
    Beta(0.5, 0.15, 0.01, 0.99, μσ = true),          # cindp
    Beta(0.5, 0.15, 0.01, 0.99999, μσ = true),       # czcap
    Normal(1.25, 0.125, 1.0, 3.0),                   # cfc
    Normal(1.5, 0.25, 1.0, 3.0),                     # crpi
    Beta(0.75, 0.10, 0.5, 0.975, μσ = true),         # crr
    Normal(0.125, 0.05, 0.001, 0.5),                 # cry
    Normal(0.125, 0.05, 0.001, 0.5),                 # crdy
    Gamma(0.625, 0.1, 0.1, 2.0, μσ = true),          # constepinf
    Gamma(0.25, 0.1, 0.01, 2.0, μσ = true),          # constebeta
    Normal(0.0, 2.0, -10.0, 10.0),                   # constelab
    Normal(0.4, 0.10, 0.1, 0.8),                     # ctrend
    Normal(0.5, 0.25, 0.01, 2.0),                    # cgy
    Normal(0.3, 0.05, 0.01, 1.0),                    # calfa
]

# Parameter names in dists order
const param_names = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew,
                     :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow,
                     :cmap, :cmaw,
                     :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap,
                     :cfc, :crpi, :crr, :cry, :crdy,
                     :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

# ──────────────────────────────────────────────────────────────────────────────
# Include linear model and set up fixed parameters
# ──────────────────────────────────────────────────────────────────────────────
include("../models/Smets_Wouters_2007_linear.jl")

fixed_parameters = Smets_Wouters_2007_linear.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw], Smets_Wouters_2007_linear.constants.post_complete_parameters.parameters)]

SS(Smets_Wouters_2007_linear, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01, :cmap => 0.01, :cmaw => 0.01])

# ──────────────────────────────────────────────────────────────────────────────
# Reorder index: maps dists order → parameters_combined order (after fixed)
# parameters_combined = [ctou, clandaw, cg, curvp, curvw,
#   calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp,
#   cindw, cindp, czcap, crpi, crr, cry, crdy,
#   crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw,
#   constelab, constepinf, constebeta, ctrend,
#   z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]
# ──────────────────────────────────────────────────────────────────────────────
const reorder_idx = [36, 18, 26, 35, 17, 19, 20, 21, 22, 23, 24, 25,
                     27, 28, 29, 30, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                     33, 31, 32, 34, 1, 2, 3, 5, 7, 4, 6]

# ──────────────────────────────────────────────────────────────────────────────
# Julia callback functions for nessai
# ──────────────────────────────────────────────────────────────────────────────
function nessai_log_prior(params_py)
    params = pyconvert(Vector{Float64}, params_py)
    lp = 0.0
    for i in eachindex(dists)
        lp += Turing.logpdf(dists[i], params[i])
    end
    return lp
end

function nessai_log_likelihood(params_py)
    params = pyconvert(Vector{Float64}, params_py)
    parameters_combined = vcat(fixed_parameters, params[reorder_idx])
    llh = get_loglikelihood(Smets_Wouters_2007_linear, data(observables), parameters_combined,
                           presample_periods = 4, initial_covariance = :diagonal,
                           filter = :kalman, on_failure_loglikelihood = -1e10)
    return llh
end

# ──────────────────────────────────────────────────────────────────────────────
# Define Python nessai Model subclass via temporary module
# ──────────────────────────────────────────────────────────────────────────────
nessai_tmpdir = mktempdir()
write(joinpath(nessai_tmpdir, "sw07_nessai_model.py"), """
import numpy as np
from nessai.model import Model

class SW07NessaiModel(Model):
    # SW07 DSGE model for nessai nested sampling.

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
sw07_nessai = pyimport("sw07_nessai_model")

# ──────────────────────────────────────────────────────────────────────────────
# Set up and run nessai FlowSampler
# ──────────────────────────────────────────────────────────────────────────────
FlowSampler = pyimport("nessai.flowsampler").FlowSampler
RejectionProposal = pyimport("nessai.proposal").RejectionProposal
configure_nessai_logger = pyimport("nessai.utils.logging").configure_logger
np = pyimport("numpy")

names_py = [string(n) for n in param_names]
bounds_py = Dict(string(n) => (Float64(minimum(d)), Float64(maximum(d)))
                 for (n, d) in zip(param_names, dists))

model = sw07_nessai.SW07NessaiModel(names_py, bounds_py, nessai_log_prior, nessai_log_likelihood)

skip_checkpoint(::Any) = nothing

log_evidence = NaN
posterior_samples = nothing
n_posterior = 0
mcmcchains_summary = nothing
fs = nothing

mktempdir() do output_dir
    println("Running full nessai estimation on SW07 linear model...")
    configure_nessai_logger(
        output = output_dir,
        label = "",
        log_level = NESSAI_LOG_LEVEL,
        stream = "stdout",
    )
    fs = FlowSampler(model;
        output = output_dir,
        nlive = NESSAI_NLIVE,
        seed = 1234,
        pytorch_threads = 1,
        resume = false,
        disable_vectorisation = true,
        logging_interval = NESSAI_LOGGING_INTERVAL,
        log_on_iteration = true,
        # checkpointing = false,
        # checkpoint_callback = skip_checkpoint,
        # uninformed_proposal = RejectionProposal, # this is ok
        
        # uninformed_proposal_kwargs = Dict("poolsize" => NESSAI_UNINFORMED_POOLSIZE), # this is ok
        poolsize = NESSAI_FLOW_POOLSIZE,
        drawsize = NESSAI_FLOW_DRAWSIZE,
        # update_poolsize = false,
        # max_poolsize_scale = 1,
        plot = false,
        proposal_plots = false,
        # memory = false, # this is ok
    )
    fs.run(plot = false, save = false)
    println("nessai estimation completed")

    log_evidence = pyconvert(Float64, fs.logZ)
    posterior_samples = fs.posterior_samples
    n_posterior = pyconvert(Int, posterior_samples.size)
end

# ──────────────────────────────────────────────────────────────────────────────
# Extract results and test
# ──────────────────────────────────────────────────────────────────────────────
println("Log evidence: $log_evidence")
println("Number of posterior samples: $n_posterior")
if n_posterior > 0
    println("Posterior means:")
    for name in param_names
        param_mean = pyconvert(Float64, np.mean(posterior_samples[string(name)]))
        println("  $name: $param_mean")
    end

    posterior_matrix = reduce(hcat, [
        pyconvert(Vector{Float64}, posterior_samples[string(name)]) for name in param_names
    ])
    posterior_chain = MCMCChains.Chains(posterior_matrix, param_names)
    mcmcchains_summary = MCMCChains.summarize(posterior_chain; sections = [:parameters])
    println("MCMCChains summary:")
    show(stdout, MIME"text/plain"(), mcmcchains_summary)
    println()
else
    println("No posterior samples returned")
end

@testset "nessai SW07 linear estimation" begin
    @test isfinite(log_evidence)
    @test n_posterior > 0
    @test !isnothing(mcmcchains_summary)
    @test !isnothing(fs)
end
