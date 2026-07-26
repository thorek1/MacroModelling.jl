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
const PARTICLE_FILTERS = (:bootstrap_particle, :auxiliary_particle, :tempered_particle)
const SUPPORTED_FILTERS = (:kalman, :inversion, PARTICLE_FILTERS...)
# `:particle` is accepted as a convenience alias for the bootstrap filter.
const PARTICLE_FILTER_ALIASES = Dict(:particle => :bootstrap_particle)
# Maps a filter symbol onto the internal variant tag used for dispatch.
const PARTICLE_FILTER_VARIANT = Dict(:bootstrap_particle => :bootstrap,
                                     :auxiliary_particle => :auxiliary,
                                     :tempered_particle  => :tempered)

# ── Measurement error ────────────────────────────────────────────────────────
# `:auto` resolves per filter: no measurement error for the Kalman and inversion
# filters (their historical behaviour), and a small data-driven value for the
# particle filters, which are degenerate without it.
const DEFAULT_MEASUREMENT_ERROR_STD = :auto
# Auto measurement-error standard deviation as a fraction of each observable's
# sample standard deviation.
const DEFAULT_PARTICLE_MEASUREMENT_ERROR_FRACTION = 0.1

# ── Particle filter defaults (see `src/filter/particle.jl`) ──────────────────
# 10_000 particles keeps a Smets-Wouters-sized problem (7 observables, ~180
# periods) accurate to a couple of log-likelihood points in well under a second
# per evaluation; raise it when the likelihood is used inside a sampler.
const DEFAULT_N_PARTICLES = 10_000
const DEFAULT_PARTICLE_RESAMPLING = :systematic
const DEFAULT_PARTICLE_RESAMPLING_THRESHOLD = 0.5
const DEFAULT_PARTICLE_INITIAL_STATE_SCALING = 1.0
# Tempered particle filter (Herbst & Schorfheide, 2019) controls
const DEFAULT_TEMPERING_TARGET_RATIO = 2.0
const DEFAULT_TEMPERING_MH_STEPS = 1
const DEFAULT_TEMPERING_MAX_STAGES = 100
const DEFAULT_TEMPERING_MH_SCALE = 0.3

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