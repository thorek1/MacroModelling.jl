# Default option constants shared across MacroModelling components.

# General algorithm and filtering defaults
const DEFAULT_ALGORITHM = :first_order
const DEFAULT_ALGORITHM_SELECTOR = stochastic -> stochastic ? :second_order : :first_order
const DEFAULT_FILTER_SELECTOR = algorithm -> algorithm == :first_order ? :kalman : :inversion
const DEFAULT_SHOCK_DECOMPOSITION_SELECTOR = algorithm -> algorithm ∉ (:second_order, :third_order)
const DEFAULT_SMOOTH_SELECTOR = filter -> filter == :kalman
const DEFAULT_WARMUP_ITERATIONS = 0
const DEFAULT_PRESAMPLE_PERIODS = 0
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
const DEFAULT_TRANSPARENCY = 0.6
const DEFAULT_MAX_ELEMENTS_PER_LEGEND_ROW = 4
const DEFAULT_EXTRA_LEGEND_SPACE = 0.0
const DEFAULT_PLOT_TYPE = :compare
const DEFAULT_FONT_SIZE = 8

# Time horizon defaults
const DEFAULT_PERIODS = 40
const DEFAULT_CONDITIONAL_VARIANCE_PERIODS = [1:20..., Inf]
const DEFAULT_AUTOCORRELATION_PERIODS = 1:5

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
const DEFAULT_AUTOCORRELATION_FLAG = false
const DEFAULT_DERIVATIVES_FLAG = true
const DEFAULT_STOCHASTIC_FLAG = false
const DEFAULT_RETURN_VARIABLES_ONLY = false
const DEFAULT_SILENT_FLAG = false

# Solver and tolerance defaults
const DEFAULT_VERBOSE = false
const DEFAULT_QME_ALGORITHM = :schur
const DEFAULT_LYAPUNOV_ALGORITHM = :doubling
const DEFAULT_SYLVESTER_ALGORITHM = :doubling
const DEFAULT_SYLVESTER_THRESHOLD = 1000
const DEFAULT_LARGE_SYLVESTER_ALGORITHM = :bicgstab
const DEFAULT_SYLVESTER_SELECTOR = 𝓂 -> sum(1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM

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
    :NSSS_acceptance_tol => "NSSS acceptance tol",
    :NSSS_xtol => "NSSS xtol",
    :NSSS_ftol => "NSSS ftol",
    :NSSS_rel_xtol => "NSSS rel xtol",
    :qme_tol => "QME tol",
    :qme_acceptance_tol => "QME acceptance tol",
    :sylvester_tol => "Sylvester tol",
    :sylvester_acceptance_tol => "Sylvester acceptance tol",
    :lyapunov_tol => "Lyapunov tol",
    :lyapunov_acceptance_tol => "Lyapunov acceptance tol",
    :droptol => "Droptol",
    :dependencies_tol => "Dependencies tol",
)

# Turing distribution wrapper defaults
const DEFAULT_TURING_USE_MEAN_STD = false

const DEFAULT_MAXLOG = 3