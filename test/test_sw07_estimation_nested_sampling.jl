using Test
using MacroModelling
import Turing
using PythonCall
using DelimitedFiles, AxisKeys
using FlexiChains
using FlexiChains: Parameter, FlexiChain
using DataStructures: OrderedDict

include("test_helpers.jl")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration switches
# ──────────────────────────────────────────────────────────────────────────────
USE_NESSAI = true
USE_DYNESTY = false
USE_ULTRANEST = false
USE_FLAT_PRIOR = false
USE_NONLINEAR_MODEL = false

NESSAI_NLIVE = 1500
NESSAI_FLOW_POOLSIZE = 128
NESSAI_FLOW_DRAWSIZE = NESSAI_FLOW_POOLSIZE
NESSAI_UNINFORMED_POOLSIZE = NESSAI_FLOW_POOLSIZE
NESSAI_MAXIMUM_UNINFORMED = 2 * NESSAI_NLIVE
NESSAI_LOG_LEVEL = "INFO"
NESSAI_LOGGING_INTERVAL = 500
NESSAI_IMPORTANCE_NESTED_SAMPLER = false
NESSAI_RESET_FLOW = false
# NESSAI_RETRAIN_ACCEPTANCE = true
# NESSAI_ACCEPTANCE_THRESHOLD = 0.1

# NSF flow configuration sized for SW07's 36-dimensional posterior.
# Wrapped in pydict() at the call site so nessai receives native Python dicts.
NESSAI_FLOW_CONFIG = Dict{String,Any}(
    "ftype"                      => "nsf",
    # "n_blocks"                   => 10,
    # "n_neurons"                  => 64,
    # "n_layers"                   => 4,
    # "batch_norm_between_layers"  => true,
    # "use_random_permutations"    => true,
    # "use_residual_blocks"        => true,
    # "dropout_probability"        => 0.01,
    # "activation"                 => "relu",
)

# Longer training schedule so the flow can learn the complex posterior shape
# const NESSAI_TRAINING_CONFIG = Dict{String,Any}(
#     "max_epochs" => 1000,
#     "patience"   => 50,
# )

DYNESTY_NLIVE_INIT = NESSAI_NLIVE
DYNESTY_NLIVE_BATCH = max(500, DYNESTY_NLIVE_INIT ÷ 2)
DYNESTY_BOUND = "multi"
DYNESTY_SAMPLE = "rslice"
DYNESTY_DLOGZ_INIT = 0.1
DYNESTY_BOOTSTRAP = 0
DYNESTY_WEIGHT_PFRAC = 1.0

ULTRANEST_MIN_NUM_LIVE_POINTS = 400

# ──────────────────────────────────────────────────────────────────────────────
# Install nested-sampling Python packages into PythonCall's Python environment
# ──────────────────────────────────────────────────────────────────────────────
println("Installing nested-sampling Python packages...")
using CondaPkg
USE_NESSAI    && CondaPkg.add_pip("nessai")
USE_DYNESTY   && CondaPkg.add_pip("dynesty")
USE_ULTRANEST && CondaPkg.add_pip("ultranest")
CondaPkg.resolve()
println("Nested-sampling Python packages installed successfully")

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
informative_dists = [
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

dists = if USE_FLAT_PRIOR
    [Turing.Uniform(minimum(d), maximum(d)) for d in informative_dists]
else
    informative_dists
end

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
if USE_NONLINEAR_MODEL
    include("../models/Smets_Wouters_2007.jl")

    fixed_parameters = Smets_Wouters_2007.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw], Smets_Wouters_2007.constants.post_complete_parameters.parameters)]

    SS(Smets_Wouters_2007, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01, :cmap => 0.01, :cmaw => 0.01])(observables,:)

    model = Smets_Wouters_2007
else
    include("../models/Smets_Wouters_2007_linear.jl")

    fixed_parameters = Smets_Wouters_2007_linear.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw], Smets_Wouters_2007_linear.constants.post_complete_parameters.parameters)]

    SS(Smets_Wouters_2007_linear, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01, :cmap => 0.01, :cmaw => 0.01], derivatives = false)

    model = Smets_Wouters_2007_linear
end

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
# Shared Julia callback functions for nested samplers
# ──────────────────────────────────────────────────────────────────────────────
function sw07_log_prior_density(params::Vector{Float64})
    lp = 0.0
    for i in eachindex(dists)
        lp += Turing.logpdf(dists[i], params[i])
    end
    return lp
end

function sw07_log_likelihood(params::Vector{Float64})
    parameters_combined = vcat(fixed_parameters, params[reorder_idx])
    llh = get_loglikelihood(model, data(observables), parameters_combined,
                           presample_periods = 4, initial_covariance = :diagonal,
                           filter = :kalman, on_failure_loglikelihood = -1e10)
    return llh
end

function sw07_prior_transform(unit_params::Vector{Float64})
    transformed_params = Vector{Float64}(undef, length(unit_params))
    for i in eachindex(dists)
        transformed_params[i] = Turing.quantile(dists[i], clamp(unit_params[i], eps(Float64), prevfloat(1.0)))
    end
    return transformed_params
end

function posterior_matrix_from_named_samples(named_samples)
    return reduce(hcat, [
        pyconvert(Vector{Float64}, named_samples[string(name)]) for name in param_names
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
    for (i, name) in pairs(param_names)
        col = @view posterior_matrix[:, i]
        println("  $name: $(sum(col) / length(col))")
    end

    n_iters, _ = size(posterior_matrix)
    symbol_names = Symbol.(collect(param_names))
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

# Shared Python helpers
names_py = [string(n) for n in param_names]
bounds_py = Dict(string(n) => (Float64(minimum(d)), Float64(maximum(d)))
                 for (n, d) in zip(param_names, dists))
np = (USE_DYNESTY || USE_ULTRANEST) ? pyimport("numpy") : nothing

# ──────────────────────────────────────────────────────────────────────────────
# nessai FlowSampler
# ──────────────────────────────────────────────────────────────────────────────
if USE_NESSAI

    function nessai_log_prior(params_py)
        return sw07_log_prior_density(pyconvert(Vector{Float64}, params_py))
    end

    function nessai_log_likelihood(params_py)
        return sw07_log_likelihood(pyconvert(Vector{Float64}, params_py))
    end

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

    FlowSampler = pyimport("nessai.flowsampler").FlowSampler
    RejectionProposal = pyimport("nessai.proposal").RejectionProposal
    configure_nessai_logger = pyimport("nessai.utils.logging").configure_logger

    nessai_model = sw07_nessai.SW07NessaiModel(names_py, bounds_py, nessai_log_prior, nessai_log_likelihood)
    nessai_output_dir = pwd()

    println("Running full nessai estimation on SW07 linear model...")
    configure_nessai_logger(
        output = nessai_output_dir,
        label = "",
        log_level = NESSAI_LOG_LEVEL,
        stream = "stdout",
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
    println("nessai estimation completed")

    nessai_log_evidence = pyconvert(Float64, nessai_fs.logZ)
    nessai_posterior_samples = nessai_fs.posterior_samples
    nessai_n_posterior, nessai_posterior_summary = summarize_posterior_matrix(
        "nessai",
        posterior_matrix_from_named_samples(nessai_posterior_samples),
    )

    println("nessai log evidence: $nessai_log_evidence")

    @testset "nessai SW07 linear estimation" begin
        @test isfinite(nessai_log_evidence)
        @test nessai_n_posterior > 0
        @test !isnothing(nessai_posterior_summary)
        @test !isnothing(nessai_fs)
    end

# USE_FLAT_PRIOR = true
# 05-16 23:18 nessai INFO    : Final ln-evidence: -1000.984 +/- 0.179
# 05-16 23:18 nessai INFO    : Information: 64.29
# 05-16 23:19 nessai INFO    : Final KS test: D=0.03676, p-value=1.217e-178
# 05-16 23:19 nessai WARNING : Final p-value for the insertion indices is less than 0.05, this could be an indication of problems during sampling. Consider checking the diagnostic plots.
# 05-16 23:19 nessai INFO    : Checkpointing nested sampling
# 05-16 23:19 nessai INFO    : Total sampling time: 5:51:29.037943
# 05-16 23:19 nessai INFO    : Total training time: 2:30:57.819565
# 05-16 23:19 nessai INFO    : Total population time: 1:48:37.768095
# 05-16 23:19 nessai INFO    : Total likelihood evaluations: 1575272
# 05-16 23:19 nessai INFO    : Time spent evaluating likelihood: 1:31:42.081410
# 05-16 23:19 nessai INFO    : Total sampling time: 5:51:29.037943
# 05-16 23:19 nessai INFO    : Total likelihood evaluations: 1575272
# 05-16 23:19 nessai INFO    : Starting post processing
# 05-16 23:19 nessai INFO    : Computing posterior samples
# 05-16 23:19 nessai INFO    : Effective sample size: 25888.8
# 05-16 23:19 nessai INFO    : Producing posterior samples using rejection sampling
# 05-16 23:19 nessai INFO    : Expect 18102.563355074064 samples from rejection sampling
# 05-16 23:19 nessai INFO    : Returned 18091 posterior samples
# nessai estimation completed
# nessai number of posterior samples: 18091
# nessai posterior means:
#   z_ea: 0.45597505067412014
#   z_eb: 0.2634471634240015
#   z_eg: 0.5399781177240325
#   z_eqs: 0.4613615701545901
#   z_em: 0.22045420331736487
#   z_epinf: 0.13056396971061376
#   z_ew: 0.25179550470796747
#   crhoa: 0.9786408574522782
#   crhob: 0.15199680428936876
#   crhog: 0.9780564007164783
#   crhoqs: 0.6564116015571109
#   crhoms: 0.06738391108495864
#   crhopinf: 0.9558515467489069
#   crhow: 0.9629644930386038
#   cmap: 0.8494662208479085
#   cmaw: 0.9441767978898936
#   csadjcost: 10.640862613235852
#   csigma: 1.5896792904211503
#   chabb: 0.7875637746883587
#   cprobw: 0.9123398699555444
#   csigl: 4.998929930146837
#   cprobp: 0.7089127968773836
#   cindw: 0.614639209280305
#   cindp: 0.10747340317835892
#   czcap: 0.5774969844402321
#   cfc: 1.9619671340448266
#   crpi: 2.651779898533512
#   crr: 0.8866870134207688
#   cry: 0.14560572712053682
#   crdy: 0.2085582121253571
#   constepinf: 1.1080946420338906
#   constebeta: 0.0845741693625765
#   constelab: -0.7359719855369656
#   ctrend: 0.44698207929208417
#   cgy: 0.5229994110626076
#   calfa: 0.21057521428483572
# nessai FlexiChains summary:
# ╭─FlexiSummary (9 statistics) ─────────────────────────────────────────────────────────────────────────────────────────╮
# │   iter    collapsed                                                                                                  │
# │   chain   collapsed                                                                                                  │
# │ ↓ stat  = [mean, std, mcse, ess_bulk, ess_tail, rhat, q5, q50, q95]                                                  │
# │                                                                                                                      │
# │ Parameters (36) ── Symbol                                                                                            │
# │  Float64  z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap,  │
# │           cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy,     │
# │           constepinf, constebeta, constelab, ctrend, cgy, calfa                                                      │
# │                                                                                                                      │
# │ Extras (0)                                                                                                           │
# │  (none)                                                                                                              │
# │                                                                                                                      │
# │ Summary                                                                                                              │
# │        param     mean     std    mcse    ess_bulk    ess_tail    rhat       q5      q50      q95                     │
# │         z_ea   0.4560  0.0240  0.0002  17737.3098   2424.5258  1.0201   0.4182   0.4551   0.4969                     │
# │         z_eb   0.2634  0.0200  0.0002  17209.3386   1400.1095  1.0199   0.2304   0.2634   0.2966                     │
# │         z_eg   0.5400  0.0260  0.0002  18202.0418   2259.5431  1.0221   0.4994   0.5387   0.5845                     │
# │        z_eqs   0.4614  0.0375  0.0004   7517.6694   1452.8103  1.0149   0.4012   0.4602   0.5251                     │
# │         z_em   0.2205  0.0106  0.0001  10922.9755   1135.3528  1.0199   0.2039   0.2200   0.2388                     │
# │      z_epinf   0.1306  0.0145  0.0001  15695.6764   2569.9921  1.0124   0.1071   0.1302   0.1547                     │
# │         z_ew   0.2518  0.0151  0.0006    654.2906    703.1263  1.0295   0.2271   0.2519   0.2764                     │
# │        crhoa   0.9786  0.0044  0.0000  17328.3740   1270.9016  1.0254   0.9711   0.9788   0.9855                     │
# │        crhob   0.1520  0.0688  0.0019   1836.2886    846.4405  1.0175   0.0463   0.1475   0.2718                     │
# │        crhog   0.9781  0.0068  0.0001  15740.3369   1311.1854  1.0180   0.9663   0.9784   0.9887                     │
# │       crhoqs   0.6564  0.0470  0.0004  16916.0572   3407.4362  1.0121   0.5779   0.6569   0.7338                     │
# │       crhoms   0.0674  0.0400  0.0022    405.3113    712.6945  1.0410   0.0176   0.0596   0.1448                     │
# │     crhopinf   0.9559  0.0179  0.0002  15713.7123   1436.5666  1.0165   0.9244   0.9570   0.9828                     │
# │        crhow   0.9630  0.0162  0.0007    865.8765    917.2468  1.0225   0.9324   0.9661   0.9831                     │
# │         cmap   0.8495  0.0475  0.0004  16660.5245   4637.4201  1.0172   0.7633   0.8546   0.9171                     │
# │         cmaw   0.9442  0.0199  0.0013    252.2650    553.2506  1.0674   0.9069   0.9482   0.9680                     │
# │    csadjcost  10.6409  1.8610  0.0251   5947.1725  15547.3654  1.0055   7.6668  10.5965  13.7493                     │
# │       csigma   1.5897  0.1465  0.0033   1783.2334   1193.9868  1.0115   1.3537   1.5868   1.8376                     │
# │        chabb   0.7876  0.0312  0.0005   3671.2221   1863.0511  1.0145   0.7354   0.7878   0.8384                     │
# │       cprobw   0.9123  0.0229  0.0007   1511.9196   1268.1527  1.0125   0.8691   0.9160   0.9423                     │
# │        csigl   4.9989  1.5756  0.0152   4305.8474   2073.4637  1.0087   2.6721   4.8444   7.8543                     │
# │       cprobp   0.7089  0.0479  0.0004  18070.2996   2874.3563  1.0121   0.6278   0.7100   0.7874                     │
# │        cindw   0.6146  0.1461  0.0049   1016.0963   1216.2424  1.0191   0.3594   0.6223   0.8418                     │
# │        cindp   0.1075  0.0691  0.0028    745.8185   1086.8971  1.0245   0.0208   0.0942   0.2397                     │
# │        czcap   0.5775  0.1100  0.0008  17784.2685   3122.7813  1.0131   0.3988   0.5768   0.7607                     │
# │          cfc   1.9620  0.1318  0.0010  17516.9688   3312.8420  1.0154   1.7570   1.9556   2.1893                     │
# │         crpi   2.6518  0.2079  0.0021  11329.3786   3925.3179  1.0057   2.2701   2.6805   2.9403                     │
# │          crr   0.8867  0.0146  0.0003   4505.8139   1726.8373  1.0169   0.8610   0.8876   0.9088                     │
# │          cry   0.1456  0.0275  0.0003   6882.1499   1901.3740  1.0095   0.1012   0.1448   0.1921                     │
# │         crdy   0.2086  0.0243  0.0002  16764.8131   2203.2354  1.0197   0.1708   0.2074   0.2502                     │
# │   constepinf   1.1081  0.1504  0.0011  18162.9990  13464.1905  1.0073   0.8652   1.1051   1.3571                     │
# │   constebeta   0.0846  0.0588  0.0029    485.9809    976.6815  1.0325   0.0171   0.0704   0.1993                     │
# │    constelab  -0.7360  1.0574  0.0078  18170.9297  15111.6223  1.0120  -2.5156  -0.7086   0.9596                     │
# │       ctrend   0.4470  0.0142  0.0001  15757.8298   1948.6520  1.0251   0.4232   0.4471   0.4698                     │
# │          cgy   0.5230  0.0788  0.0006  17064.2716   2355.5780  1.0147   0.3916   0.5232   0.6503                     │
# │        calfa   0.2106  0.0164  0.0001  18208.5176   2030.2823  1.0202   0.1838   0.2103   0.2378                     │
# ╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
# nessai log evidence: -1000.9839809407748
end # USE_NESSAI

# ──────────────────────────────────────────────────────────────────────────────
# dynesty DynamicNestedSampler
# ──────────────────────────────────────────────────────────────────────────────
if USE_DYNESTY

    dynesty = pyimport("dynesty")

    function dynesty_log_likelihood(params_py)
        return sw07_log_likelihood(pyconvert(Vector{Float64}, params_py))
    end

    function dynesty_prior_transform(unit_params_py)
        transformed_params = sw07_prior_transform(pyconvert(Vector{Float64}, unit_params_py))
        return np.asarray(pylist(transformed_params), dtype = np.float64)
    end

    println("Running dynesty dynamic nested sampling on SW07 linear model...")
    dynesty_sampler = dynesty.DynamicNestedSampler(
        dynesty_log_likelihood,
        dynesty_prior_transform,
        length(param_names);
        bound = DYNESTY_BOUND,
        sample = DYNESTY_SAMPLE,
        slices = length(param_names) + 3,
        bootstrap = DYNESTY_BOOTSTRAP,
        queue_size = 1,
    )
    dynesty_sampler.run_nested(
        nlive_init = DYNESTY_NLIVE_INIT,
        nlive_batch = DYNESTY_NLIVE_BATCH,
        dlogz_init = DYNESTY_DLOGZ_INIT,
        wt_kwargs = Dict("pfrac" => DYNESTY_WEIGHT_PFRAC),
        stop_kwargs = Dict("pfrac" => DYNESTY_WEIGHT_PFRAC),
        print_progress = true,
        save_bounds = false,
    )
    println("dynesty dynamic nested sampling completed")

    dynesty_results = dynesty_sampler.results
    println("dynesty summary:")
    dynesty_results.summary()
    dynesty_log_evidence = pyconvert(Vector{Float64}, dynesty_results.logz)[end]
    dynesty_posterior_matrix = pyconvert(Matrix{Float64}, dynesty_results.samples_equal())
    dynesty_n_posterior, dynesty_posterior_summary = summarize_posterior_matrix(
        "dynesty dynamic",
        dynesty_posterior_matrix,
    )

    println("dynesty log evidence: $dynesty_log_evidence")

    @testset "dynesty dynamic SW07 linear estimation" begin
        @test isfinite(dynesty_log_evidence)
        @test dynesty_n_posterior > 0
        @test !isnothing(dynesty_posterior_summary)
        @test !isnothing(dynesty_sampler)
        @test !isnothing(dynesty_results)
    end

end # USE_DYNESTY

# ──────────────────────────────────────────────────────────────────────────────
# UltraNest ReactiveNestedSampler
# ──────────────────────────────────────────────────────────────────────────────
if USE_ULTRANEST

    ultranest = pyimport("ultranest")
    ultranest_stepsampler = pyimport("ultranest.stepsampler")
    ReactiveNestedSampler = ultranest.ReactiveNestedSampler

    function ultranest_log_likelihood(params_py)
        return sw07_log_likelihood(pyconvert(Vector{Float64}, params_py))
    end

    function ultranest_prior_transform(unit_params_py)
        transformed_params = sw07_prior_transform(pyconvert(Vector{Float64}, unit_params_py))
        return np.asarray(pylist(transformed_params), dtype = np.float64)
    end

    ultranest_log_dir = mktempdir()

    println("Running UltraNest nested sampling on SW07 linear model...")
    ultranest_sampler = ReactiveNestedSampler(
        pylist(names_py),
        ultranest_log_likelihood,
        ultranest_prior_transform;
        log_dir = ultranest_log_dir,
        resume = "overwrite",
    )

    nsteps = length(param_names)
    ultranest_sampler.stepsampler = ultranest_stepsampler.SliceSampler(;
        nsteps = nsteps,
        generate_direction = ultranest_stepsampler.generate_mixture_random_direction,
    )

    ultranest_result = ultranest_sampler.run(;
        min_num_live_points = ULTRANEST_MIN_NUM_LIVE_POINTS,
        show_status = true,
    )
    ultranest_sampler.print_results()
    println("UltraNest nested sampling completed")

    ultranest_log_evidence = pyconvert(Float64, ultranest_result["logz"])
    ultranest_posterior_matrix = pyconvert(Matrix{Float64}, ultranest_result["samples"])
    @assert size(ultranest_posterior_matrix, 2) == length(param_names) "UltraNest samples have $(size(ultranest_posterior_matrix, 2)) columns but expected $(length(param_names))"

    ultranest_n_posterior, ultranest_posterior_summary = summarize_posterior_matrix(
        "UltraNest",
        ultranest_posterior_matrix,
    )

    println("UltraNest log evidence: $ultranest_log_evidence")

    @testset "UltraNest SW07 linear estimation" begin
        @test isfinite(ultranest_log_evidence)
        @test ultranest_n_posterior > 0
        @test !isnothing(ultranest_posterior_summary)
        @test !isnothing(ultranest_result)
    end

end # USE_ULTRANEST
