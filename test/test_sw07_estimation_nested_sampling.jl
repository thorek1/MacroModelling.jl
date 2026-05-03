using Test
using MacroModelling
import Turing
using PythonCall
using DelimitedFiles, AxisKeys

include("test_helpers.jl")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration switches
# ──────────────────────────────────────────────────────────────────────────────
USE_NESSAI = true
USE_DYNESTY = false
USE_ULTRANEST = false
USE_FLAT_PRIOR = false

NESSAI_NLIVE = 2000
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
include("../models/Smets_Wouters_2007_linear.jl")

fixed_parameters = Smets_Wouters_2007_linear.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw], Smets_Wouters_2007_linear.constants.post_complete_parameters.parameters)]

SS(Smets_Wouters_2007_linear, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01, :cmap => 0.01, :cmaw => 0.01], derivatives = false)

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
    llh = get_loglikelihood(Smets_Wouters_2007_linear, data(observables), parameters_combined,
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

    posterior_chain = flexichain_from_matrix(posterior_matrix, param_names)
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
