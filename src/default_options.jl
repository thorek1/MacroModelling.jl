# Default option constants shared across MacroModelling components.

# General algorithm and filtering defaults
const DEFAULT_ALGORITHM = :first_order
const DEFAULT_ALGORITHM_SELECTOR = stochastic -> stochastic ? :second_order : :first_order
const DEFAULT_FILTER_SELECTOR = algorithm -> algorithm == :first_order ? :kalman : :inversion
const DEFAULT_SHOCK_DECOMPOSITION_SELECTOR = algorithm -> algorithm ∉ (:second_order, :third_order)
const DEFAULT_SMOOTH_SELECTOR = filter -> filter == :kalman
const DEFAULT_WARMUP_ITERATIONS = 0
const DEFAULT_PRESAMPLE_PERIODS = 0

# ── Filter registry ──────────────────────────────────────────────────────────
# Each particle-filter variant is its own `filter` value, so the filter is fully
# identified by a single symbol (no separate "which particle filter" argument).
const PARTICLE_FILTERS = (:bootstrap_particle, :auxiliary_particle, :tempered_particle, :guided_particle)
# The quadratic and cubic Kalman filters apply to the pruned second- and
# third-order solutions. Ivashchenko's filter is a separate Gaussian
# moment-closure filter for raw or pruned second- and third-order solutions.
const SUPPORTED_FILTERS = (:kalman, :inversion, :quadratic_kalman, :cubic_kalman,
                           :ivashchenko_kalman, PARTICLE_FILTERS...)
# `:particle` is accepted as a convenience alias for the variant a user asking for
# "a particle filter" should get. That is the guided filter: it draws each shock from
# its own conditional rather than blindly, which on a Smets-Wouters-sized problem is
# both an order of magnitude cheaper and several times more accurate than the
# bootstrap filter the alias used to point at.
const PARTICLE_FILTER_ALIASES = Dict(:particle => :guided_particle)
# Maps a filter symbol onto the internal variant tag used for dispatch.
const PARTICLE_FILTER_VARIANT = Dict(:bootstrap_particle => :bootstrap,
                                     :auxiliary_particle => :auxiliary,
                                     :tempered_particle  => :tempered,
                                     :guided_particle    => :guided)

# ── Measurement error ────────────────────────────────────────────────────────
# `measurement_error` is the covariance H of ηₜ ~ N(0, H) in yₜ = C xₜ + ηₜ. It is
# *not* a standard deviation: a scalar is the common variance of every observable,
# a vector the per-observable variances, and a matrix the full covariance.
# `:auto` resolves per filter: no measurement error for the Kalman, inversion,
# and Ivashchenko filters (their historical/deterministic-filter behaviour), and
# a small data-driven value for the particle filters, which are degenerate without it.
const DEFAULT_MEASUREMENT_ERROR = :auto
# Auto measurement-error *standard deviation* as a fraction of each observable's
# sample standard deviation (squared into a variance before it reaches a filter).
# 0.1 puts ~1% of each observable's variance into measurement error: enough to
# keep the particle weights well spread on a Smets-Wouters-sized problem, small
# enough that the likelihood still reflects the model rather than the noise.
const DEFAULT_PARTICLE_MEASUREMENT_ERROR_FRACTION = 0.1

# `-Inf` is the right failure value for the deterministic filters: it tells a
# sampler the draw is impossible. A particle filter, though, can fail for purely
# stochastic reasons (every particle far off in one period), and an `-Inf` there
# would permanently kill the chain state rather than just reject one proposal —
# so it returns a large finite penalty instead.
const DEFAULT_ON_FAILURE_LOGLIKELIHOOD_SELECTOR = filter -> get(PARTICLE_FILTER_ALIASES, filter, filter) ∈ PARTICLE_FILTERS ? -1e6 : -Inf

# ── Particle filter defaults (see `src/filter/particle.jl`) ──────────────────
#
# Naming: a setting shared by more than one particle filter is `PARTICLE_…` and
# its keyword argument is `particle_…`; a setting that belongs to one variant
# carries that variant's name instead (`GUIDED_…`, `TEMPERED_…`). The one
# deliberate exception is `n_particles`, kept under its conventional name.
#
# 10_000 particles keeps a Smets-Wouters-sized problem (7 observables, ~180
# periods) accurate to a couple of log-likelihood points in well under a second
# per evaluation; raise it when the likelihood is used inside a sampler.
const DEFAULT_N_PARTICLES = 10_000
const DEFAULT_PARTICLE_RESAMPLING = :systematic
const DEFAULT_PARTICLE_RESAMPLING_THRESHOLD = 0.5
const DEFAULT_PARTICLE_INITIAL_STATE_SCALING = 1.0

# ── Bridging controls, shared by `:tempered_particle` and `:guided_particle` ──
# Both filters reach the period's target through a sequence of intermediate
# distributions, reweighting, resampling and mutating along the way; these set how
# finely they step and how hard they mutate.

# How much weight inefficiency one bridging step is allowed to add, which is what
# picks the step sizes. Lower means more, smaller steps. Set below Herbst &
# Schorfheide's own value of 2 because for the tempered filter — which has to
# bridge all the way from the prior — the bridging, not the particle count, is
# what limits accuracy, and buying accuracy here is cheaper per unit of compute
# than raising `n_particles`.
const DEFAULT_PARTICLE_TARGET_RATIO = 1.5

# Metropolis-Hastings mutation steps per bridging stage. The two filters want
# different amounts and get their own defaults, resolved by the selector below.
#
# The tempered filter is highly sensitive to this: it bridges from the prior, so
# mutation is what rejuvenates a cloud that would otherwise be badly degenerate,
# and going from one step to four roughly halves the run-to-run spread of both the
# estimates and the likelihood.
const DEFAULT_TEMPERED_MH_STEPS = 4
# The guided filter bridges from a proposal already close to the target, so it
# needs far less. Its *estimates* are flat in this knob — any value from zero
# upwards is within measurement noise — and only the likelihood discriminates,
# putting the optimum at two at both perturbation orders. Anyone re-measuring this
# should know the likelihood dispersion is a much noisier statistic than the
# estimates one (it is the log of an average of heavy-tailed weights) and needs
# paired seeds, and plenty of them, to resolve at all.
const DEFAULT_GUIDED_MH_STEPS = 2
# Which of the two a call gets, from the filter it selected.
const DEFAULT_PARTICLE_MH_STEPS_SELECTOR = filter -> get(PARTICLE_FILTER_ALIASES, filter, filter) == :guided_particle ? DEFAULT_GUIDED_MH_STEPS : DEFAULT_TEMPERED_MH_STEPS

const DEFAULT_PARTICLE_MAX_STAGES = 100
# Starting value for the Metropolis mutation step, in units of the stage's own
# posterior scale (the proposal is preconditioned by it, see `src/filter/particle.jl`).
# 2.38/sqrt(d) is the textbook optimum for a d-dimensional random walk; the filter
# adapts from here towards the target acceptance rate below, so this only sets
# where it starts.
const DEFAULT_PARTICLE_MH_SCALE = 1.0
# Adaptation of that scale: the target acceptance rate, the gain of the log-scale
# update towards it, and the bounds the scale is clamped to.
const DEFAULT_PARTICLE_MH_TARGET_ACCEPTANCE = 0.25
const DEFAULT_PARTICLE_MH_ADAPTATION_GAIN = 1.0
const DEFAULT_PARTICLE_MH_SCALE_BOUNDS = (1e-8, 1e2)

# ── Guided-filter specifics ──────────────────────────────────────────────────
# Newton steps refining the proposal's centre, the width of the proposal as a
# multiple of the Laplace scale, and whether filtered shock estimates are reported
# from the proposal mean rather than the draw. The reasoning behind each value is
# at its point of use in `src/filter/particle.jl`.
const DEFAULT_GUIDED_NEWTON_STEPS = 2
const DEFAULT_GUIDED_PROPOSAL_SCALE = 1.0
const DEFAULT_GUIDED_RAO_BLACKWELL = true

# ── Internal sizing and diagnostics, common to every particle filter ──────────
# Transition scratch budget, smallest block worth a `gemm`, and the arithmetic per
# sweep below which the swarm is propagated on the calling thread.
const DEFAULT_PARTICLE_SCRATCH_BYTES = 256 * 2^20
const DEFAULT_PARTICLE_MIN_BLOCK = 64
const DEFAULT_PARTICLE_PARALLEL_MIN_WORK = 1 << 20
# Chunking for the memory-bound copy passes (resampling gather, Metropolis accept).
const DEFAULT_PARTICLE_COPY_CHUNK = 2048
const DEFAULT_PARTICLE_COPY_MAX_TASKS = 8
# Mean effective sample size below which the filter warns that its proposal is a
# poor fit to the data.
const DEFAULT_PARTICLE_LOW_ESS_FRACTION = 0.05

const DEFAULT_DATA_IN_LEVELS = true
const DEFAULT_LEVELS = true
const DEFAULT_CONDITIONS_IN_LEVELS = true
const DEFAULT_IGNORE_OBC = false
const DEFAULT_SMOOTH_FLAG = true

# Plotting defaults
const DEFAULT_LABEL = 1
const DEFAULT_SHOW_PLOTS = true
const DEFAULT_SAVE_PLOTS = false
const DEFAULT_SAVE_PLOTS_FORMAT = :pdf
const DEFAULT_SAVE_PLOTS_PATH = "."
const DEFAULT_PLOTS_PER_PAGE_SMALL = 6
const DEFAULT_PLOTS_PER_PAGE_LARGE = 9
const DEFAULT_TRANSPARENCY = 1.0
const DEFAULT_MAX_ELEMENTS_PER_LEGEND_ROW = 4
const DEFAULT_EXTRA_LEGEND_SPACE = 0.0
const DEFAULT_PLOT_TYPE = :compare
const DEFAULT_FONT_SIZE = 8

# Time horizon defaults
const DEFAULT_PERIODS = 40
const DEFAULT_CONDITIONAL_VARIANCE_PERIODS = [1:20..., Inf]
const DEFAULT_AUTOCORRELATION_PERIODS = 1:5
const DEFAULT_FORECAST_PERIODS = 12

# Shock and variable selections
const DEFAULT_SHOCK_SELECTION = :all
const DEFAULT_SHOCKS_EXCLUDING_OBC = :all_excluding_obc
const DEFAULT_VARIABLE_SELECTION = :all
const DEFAULT_VARIABLES_EXCLUDING_OBC = :all_excluding_obc
const DEFAULT_VARIABLES_EXCLUDING_AUX_AND_OBC = :all_excluding_auxiliary_and_obc

# IRF and GIRF defaults
const DEFAULT_SHOCK_SIZE = 1
const DEFAULT_NEGATIVE_SHOCK = false
const DEFAULT_GENERALISED_IRF = false
const DEFAULT_GENERALISED_IRF_WARMUP = 100
const DEFAULT_GENERALISED_IRF_DRAWS = 50
const DEFAULT_INITIAL_STATE = [0.0]

# Moment and statistics defaults
const DEFAULT_SIGMA_RANGE = 2
const DEFAULT_NON_STOCHASTIC_STEADY_STATE_FLAG = true
const DEFAULT_MEAN_FLAG = false
const DEFAULT_STANDARD_DEVIATION_FLAG = true
const DEFAULT_VARIANCE_FLAG = false
const DEFAULT_COVARIANCE_FLAG = false
const DEFAULT_CORRELATION_FLAG = false
const DEFAULT_AUTOCORRELATION_FLAG = false
const DEFAULT_DERIVATIVES_FLAG = true
const DEFAULT_STOCHASTIC_FLAG = false
const DEFAULT_RETURN_VARIABLES_ONLY = false
const DEFAULT_SILENT_FLAG = false

# Solver and tolerance defaults
const DEFAULT_VERBOSE = false
const DEFAULT_QME_ALGORITHM = :schur
const DEFAULT_QME_THRESHOLD = 1000000
const DEFAULT_LARGE_QME_ALGORITHM = :doubling
const DEFAULT_QME_SELECTOR = 𝓂 -> (𝓂.constants.post_model_macro.nVars - 𝓂.constants.post_model_macro.nPresent_only)^2 > DEFAULT_QME_THRESHOLD ? DEFAULT_LARGE_QME_ALGORITHM : DEFAULT_QME_ALGORITHM
const DEFAULT_LYAPUNOV_ALGORITHM = :doubling
const DEFAULT_SYLVESTER_ALGORITHM = :doubling
const DEFAULT_SYLVESTER_THRESHOLD = 10000
const DEFAULT_LARGE_SYLVESTER_ALGORITHM = :bicgstab
const DEFAULT_SYLVESTER_SELECTOR = 𝓂 -> sum(1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM

# StatsPlots specific constants
const DEFAULT_PLOT_ATTRIBUTES = Dict(
    :size => (700, 500),
    :plot_titlefont => DEFAULT_FONT_SIZE + 2,
    :titlefont => DEFAULT_FONT_SIZE,
    :guidefont => DEFAULT_FONT_SIZE,
    :palette => :auto,
    :legendfontsize => DEFAULT_FONT_SIZE,
    :annotationfontsize => DEFAULT_FONT_SIZE,
    :legend_title_font_pointsize => DEFAULT_FONT_SIZE,
    :tickfontsize => DEFAULT_FONT_SIZE,
    :framestyle => :semi,
)

const DEFAULT_ARGS_AND_KWARGS_NAMES = Dict(
    :model_name => "Model",
    :algorithm => "Algorithm",
    :shock_names => "Shock",
    :shock_size => "Shock size",
    :negative_shock => "Negative shock",
    :generalised_irf => "Generalised IRF",
    :generalised_irf_warmup_iterations => "Generalised IRF warmup iterations",
    :generalised_irf_draws => "Generalised IRF draws",
    :periods => "Periods",
    :presample_periods => "Presample Periods",
    :ignore_obc => "Ignore OBC",
    :smooth => "Smooth",
    :data => "Data",
    :label => "Label",
    :filter => "Filter",
    :warmup_iterations => "Warmup Iterations",
    :quadratic_matrix_equation_algorithm => "Quadratic Matrix Equation Algorithm",
    :sylvester_algorithm => "Sylvester Algorithm",
    :lyapunov_algorithm => "Lyapunov Algorithm",
)

# Turing distribution wrapper defaults
const DEFAULT_TURING_USE_MEAN_STD = false

const DEFAULT_MAXLOG = 3

# Caching and workspace defaults
const DEFAULT_CACHING = true
const DEFAULT_USE_WORKSPACES = true
