module FS2000NsssResiduals
using MacroModelling

const normcdf = MacroModelling.SymPyWorkspace.normcdf
const pnorm = MacroModelling.SymPyWorkspace.pnorm
const normpdf = MacroModelling.SymPyWorkspace.normpdf
const dnorm = MacroModelling.SymPyWorkspace.dnorm
const normlogpdf = MacroModelling.SymPyWorkspace.normlogpdf
const norminvcdf = MacroModelling.SymPyWorkspace.norminvcdf
const norminv = MacroModelling.SymPyWorkspace.norminv
const qnorm = MacroModelling.SymPyWorkspace.qnorm
const erfcinv = MacroModelling.SymPyWorkspace.erfcinv
const erfc = MacroModelling.SymPyWorkspace.erfc
const Max = max
const Min = min

const MODEL_NAME = "FS2000"
const SOURCE_MODEL_FILE = "models/FS2000.jl"
const NSSS_SOLUTION_ERROR = 4.965068306494546e-16
const NSSS_RESIDUAL_NORM = 5.874748045952207e-16

const PARAMETER_NAMES = [
    "alp",
    "bet",
    "gam",
    "mst",
    "rho",
    "psi",
    "del",
    "z_e_a",
    "z_e_m",
]
const PARAMETER_VALUES = Float64[
    0.356,
    0.993,
    0.0085,
    1.0002,
    0.129,
    0.65,
    0.01,
    0.035449,
    0.008862,
]
const COMPLETE_PARAMETER_NAMES = [
    "alp",
    "bet",
    "gam",
    "mst",
    "rho",
    "psi",
    "del",
    "z_e_a",
    "z_e_m",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    0.356,
    0.993,
    0.0085,
    1.0002,
    0.129,
    0.65,
    0.01,
    0.035449,
    0.008862,
]
const ORIGINAL_SOLUTION_NAMES = [
    "P",
    "R",
    "W",
    "c",
    "d",
    "dA",
    "e",
    "gp_obs",
    "gy_obs",
    "k",
    "l",
    "log_gp_obs",
    "log_gy_obs",
    "m",
    "n",
    "y",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    0.9932505545675542,
    1.0072507552870087,
    2.7185621689742447,
    1.0069966690685328,
    0.8608478832599588,
    1.0085362275720395,
    1.0,
    0.9917343300675393,
    1.0085362275720395,
    18.982191719109345,
    0.8610478832599587,
    -0.008300019997333728,
    0.008499999999999985,
    1.0002,
    0.3167291493594371,
    1.3558767746137388,
]
const ORIGINAL_INITIAL_SOLUTION_VALUES = Float64[
    0.9932505545675542,
    1.0072507552870087,
    2.7185621689742447,
    1.0069966690685328,
    0.8608478832599588,
    1.0085362275720395,
    1.0,
    0.9917343300675393,
    1.0085362275720395,
    18.982191719109345,
    0.8610478832599587,
    -0.008300019997333728,
    0.008499999999999985,
    1.0002,
    0.3167291493594371,
    1.3558767746137388,
]
const AUXILIARY_SOLUTION_NAMES = [
    "P",
    "R",
    "W",
    "c",
    "d",
    "dA",
    "e",
    "gp_obs",
    "gy_obs",
    "k",
    "l",
    "log_gp_obs",
    "log_gy_obs",
    "m",
    "n",
    "y",
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    0.9932505545675542,
    1.0072507552870087,
    2.7185621689742447,
    1.0069966690685328,
    0.8608478832599588,
    1.0085362275720395,
    1.0,
    0.9917343300675393,
    1.0085362275720395,
    18.982191719109345,
    0.8610478832599587,
    -0.008300019997333728,
    0.008499999999999985,
    1.0002,
    0.3167291493594371,
    1.3558767746137388,
    0.0085,
    -0.003026,
    -0.0085,
    -0.003026,
    -0.0085,
]
const AUXILIARY_INITIAL_SOLUTION_VALUES = Float64[
    0.9932505545675542,
    1.0072507552870087,
    2.7185621689742447,
    1.0069966690685328,
    0.8608478832599588,
    1.0085362275720395,
    1.0,
    0.9917343300675393,
    1.0085362275720395,
    18.982191719109345,
    0.8610478832599587,
    -0.008300019997333728,
    0.008499999999999985,
    1.0002,
    0.3167291493594371,
    1.3558767746137388,
    0.0085,
    -0.003026,
    -0.0085,
    -0.003026,
    -0.0085,
]
const ALL_AUXILIARY_VARIABLE_NAMES = [
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
    "➕₆",
    "➕₇",
    "➕₈",
    "➕₉",
    "➕₁₀",
    "➕₁₁",
    "➕₁₂",
    "➕₁₃",
    "➕₁₄",
    "➕₁₅",
    "➕₁₆",
    "➕₁₇",
    "➕₁₈",
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    0.0085,
    -0.003026,
    -0.0085,
    -0.003026,
    -0.0085,
    0.9917343300675393,
    18.982191719109345,
    0.3167291493594371,
    1.0085362275720395,
    1.0002,
    1.0,
    1.0,
    18.982191719109345,
    0.3167291493594371,
    0.9917343300675393,
    1.0085362275720395,
    18.982191719109345,
    0.3167291493594371,
]
const ALL_AUXILIARY_VARIABLE_INITIAL_VALUES = Float64[
    0.0085,
    -0.003026,
    -0.0085,
    -0.003026,
    -0.0085,
    0.9917343300675393,
    18.982191719109345,
    0.3167291493594371,
    1.0085362275720395,
    1.0002,
    1.0,
    1.0,
    18.982191719109345,
    0.3167291493594371,
    0.9917343300675393,
    1.0085362275720395,
    18.982191719109345,
    0.3167291493594371,
]
const DEFAULTED_NSSS_SOLUTION_NAMES = [
]
const CALIBRATION_PARAMETER_NAMES = [
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(dA - exp(gam + z_e_a * 0)),
    :(log(m) - ((1 - rho) * log(mst) + rho * log(m) + z_e_m * 0)),
    :((-P / (c * P * m) + (bet * P * (alp * exp(-alp * (gam + log(e))) * k ^ (alp - 1) * n ^ (1 - alp) + (1 - del) * exp(-((gam + log(e)))))) / (c * P * m)) - 0),
    :(W - l / n),
    :((-(psi / (1 - psi)) * ((c * P) / (1 - n)) + l / n) - 0),
    :(R - (P * (1 - alp) * exp(-alp * (gam + z_e_a * 0)) * k ^ alp * n ^ -alp) / W),
    :((1 / (c * P) - (bet * P * (1 - alp) * exp(-alp * (gam + z_e_a * 0)) * k ^ alp * n ^ (1 - alp)) / (m * l * c * P)) - 0),
    :((c + k) - (exp(-alp * (gam + z_e_a * 0)) * k ^ alp * n ^ (1 - alp) + (1 - del) * exp(-((gam + z_e_a * 0))) * k)),
    :(P * c - m),
    :(((m - 1) + d) - l),
    :(e - exp(z_e_a * 0)),
    :(y - k ^ alp * n ^ (1 - alp) * exp(-alp * (gam + z_e_a * 0))),
    :(gy_obs - (dA * y) / y),
    :(gp_obs - ((P / P) * m) / dA),
    :(log_gy_obs - log(gy_obs)),
    :(log_gp_obs - log(gp_obs)),
]
const CALIBRATION_EQUATIONS = Expr[
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :(➕₁ - gam),
    :(dA - exp(➕₁)),
    :((-rho * log(m) - (1 - rho) * log(mst)) + log(m)),
    :(➕₂ - -alp * (gam + log(e))),
    :(➕₃ - (-gam - log(e))),
    :((bet * (alp * k ^ (alp - 1) * n ^ (1 - alp) * exp(➕₂) + (1 - del) * exp(➕₃))) / (c * m) - 1 / (c * m)),
    :(W - l / n),
    :((-P * c * psi) / ((1 - n) * (1 - psi)) + l / n),
    :(➕₄ - -alp * gam),
    :((-P * k ^ alp * (1 - alp) * exp(➕₄)) / (W * n ^ alp) + R),
    :((-bet * k ^ alp * n ^ (1 - alp) * (1 - alp) * exp(➕₄)) / (c * l * m) + 1 / (P * c)),
    :(➕₅ - -gam),
    :(((c - k * (1 - del) * exp(➕₅)) + k) - k ^ alp * n ^ (1 - alp) * exp(➕₄)),
    :(P * c - m),
    :(((d - l) + m) - 1),
    :(e - 1.0),
    :(-(k ^ alp) * n ^ (1 - alp) * exp(➕₄) + y),
    :(-dA + gy_obs),
    :(gp_obs - m / dA),
    :(log_gy_obs - log(gy_obs)),
    :(log_gp_obs - log(gp_obs)),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(dA - exp(gam + z_e_a * 0)),
    :(log(m) - ((1 - rho) * log(mst) + rho * log(m) + z_e_m * 0)),
    :((-P / (c * P * m) + (bet * P * (alp * exp(-alp * (gam + log(e))) * k ^ (alp - 1) * n ^ (1 - alp) + (1 - del) * exp(-((gam + log(e)))))) / (c * P * m)) - 0),
    :(W - l / n),
    :((-(psi / (1 - psi)) * ((c * P) / (1 - n)) + l / n) - 0),
    :(R - (P * (1 - alp) * exp(-alp * (gam + z_e_a * 0)) * k ^ alp * n ^ -alp) / W),
    :((1 / (c * P) - (bet * P * (1 - alp) * exp(-alp * (gam + z_e_a * 0)) * k ^ alp * n ^ (1 - alp)) / (m * l * c * P)) - 0),
    :((c + k) - (exp(-alp * (gam + z_e_a * 0)) * k ^ alp * n ^ (1 - alp) + (1 - del) * exp(-((gam + z_e_a * 0))) * k)),
    :(P * c - m),
    :(((m - 1) + d) - l),
    :(e - exp(z_e_a * 0)),
    :(y - k ^ alp * n ^ (1 - alp) * exp(-alp * (gam + z_e_a * 0))),
    :(gy_obs - (dA * y) / y),
    :(gp_obs - ((P / P) * m) / dA),
    :(log_gy_obs - log(gy_obs)),
    :(log_gp_obs - log(gp_obs)),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :(➕₁ - gam),
    :(dA - exp(➕₁)),
    :((-rho * log(m) - (1 - rho) * log(mst)) + log(m)),
    :(➕₂ - -alp * (gam + log(e))),
    :(➕₃ - (-gam - log(e))),
    :((bet * (alp * k ^ (alp - 1) * n ^ (1 - alp) * exp(➕₂) + (1 - del) * exp(➕₃))) / (c * m) - 1 / (c * m)),
    :(W - l / n),
    :((-P * c * psi) / ((1 - n) * (1 - psi)) + l / n),
    :(➕₄ - -alp * gam),
    :((-P * k ^ alp * (1 - alp) * exp(➕₄)) / (W * n ^ alp) + R),
    :((-bet * k ^ alp * n ^ (1 - alp) * (1 - alp) * exp(➕₄)) / (c * l * m) + 1 / (P * c)),
    :(➕₅ - -gam),
    :(((c - k * (1 - del) * exp(➕₅)) + k) - k ^ alp * n ^ (1 - alp) * exp(➕₄)),
    :(P * c - m),
    :(((d - l) + m) - 1),
    :(e - 1.0),
    :(-(k ^ alp) * n ^ (1 - alp) * exp(➕₄) + y),
    :(-dA + gy_obs),
    :(gp_obs - m / dA),
    :(log_gy_obs - log(gy_obs)),
    :(log_gp_obs - log(gp_obs)),
]

const PARAMETER_DEFINITION_NAMES = [
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "alp",
    "bet",
    "gam",
    "mst",
    "rho",
    "psi",
    "del",
    "z_e_a",
    "z_e_m",
]
const PARAMETER_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
]
const PARAMETER_BOX_UPPER_BOUNDS = Float64[
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
]
const ORIGINAL_BOX_CONSTRAINT_NAMES = [
    "P",
    "R",
    "W",
    "c",
    "d",
    "dA",
    "e",
    "gp_obs",
    "gy_obs",
    "k",
    "l",
    "log_gp_obs",
    "log_gy_obs",
    "m",
    "n",
    "y",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -Inf,
]
const ORIGINAL_BOX_UPPER_BOUNDS = Float64[
    1.0e12,
    Inf,
    Inf,
    1.0e12,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    Inf,
    1.0e12,
    Inf,
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "P",
    "R",
    "W",
    "c",
    "d",
    "dA",
    "e",
    "gp_obs",
    "gy_obs",
    "k",
    "l",
    "log_gp_obs",
    "log_gy_obs",
    "m",
    "n",
    "y",
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
]
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    1.0e12,
    Inf,
    Inf,
    1.0e12,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    Inf,
    1.0e12,
    Inf,
    600.0,
    600.0,
    600.0,
    600.0,
    600.0,
]
const ALL_AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
    "➕₆",
    "➕₇",
    "➕₈",
    "➕₉",
    "➕₁₀",
    "➕₁₁",
    "➕₁₂",
    "➕₁₃",
    "➕₁₄",
    "➕₁₅",
    "➕₁₆",
    "➕₁₇",
    "➕₁₈",
]
const ALL_AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
]
const ALL_AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    600.0,
    600.0,
    600.0,
    600.0,
    600.0,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
]

const BLOCKS = [
    (
        index = 1,
        solve_order = 17,
        variables = ["y"],
        previous_solution_names = ["k", "n", "➕₄"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₇", "➕₁₈"],
        equation_indices = [17],
        equations = Expr[
            :(-(➕₁₇ ^ alp) * ➕₁₈ ^ (1 - alp) * exp(➕₄) + y),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₇ = min(1.0e12, max(eps(), k))),
            :(➕₁₈ = min(1.0e12, max(eps(), n))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₇ - k)),
            :(abs(➕₁₈ - n)),
        ],
        solution_names = ["y", "➕₁₇", "➕₁₈"],
        previous_solution_values = [18.982191719109345, 0.3167291493594371, -0.003026],
        external_solution_values = Float64[],
        solution_values = [1.3558767746137388, 18.982191719109345, 0.3167291493594371],
        previous_solution_initial_values = [18.982191719109345, 0.3167291493594371, -0.003026],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.3558767746137388, 18.982191719109345, 0.3167291493594371],
        box_lower_bounds = [-Inf, 2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12, 1.0e12],
    ),
    (
        index = 2,
        solve_order = 16,
        variables = ["log_gy_obs"],
        previous_solution_names = ["gy_obs"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₆"],
        equation_indices = [20],
        equations = Expr[
            :(log_gy_obs - log(➕₁₆)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₆ = min(1.0e12, max(eps(), gy_obs))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₆ - gy_obs)),
        ],
        solution_names = ["log_gy_obs", "➕₁₆"],
        previous_solution_values = [1.0085362275720395],
        external_solution_values = Float64[],
        solution_values = [0.008499999999999985, 1.0085362275720395],
        previous_solution_initial_values = [1.0085362275720395],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.008499999999999985, 1.0085362275720395],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 3,
        solve_order = 15,
        variables = ["log_gp_obs"],
        previous_solution_names = ["gp_obs"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₅"],
        equation_indices = [21],
        equations = Expr[
            :(log_gp_obs - log(➕₁₅)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₅ = min(1.0e12, max(eps(), gp_obs))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₅ - gp_obs)),
        ],
        solution_names = ["log_gp_obs", "➕₁₅"],
        previous_solution_values = [0.9917343300675393],
        external_solution_values = Float64[],
        solution_values = [-0.008300019997333728, 0.9917343300675393],
        previous_solution_initial_values = [0.9917343300675393],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-0.008300019997333728, 0.9917343300675393],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 4,
        solve_order = 14,
        variables = ["gy_obs"],
        previous_solution_names = ["dA"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [18],
        equations = Expr[
            :(-dA + gy_obs),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["gy_obs"],
        previous_solution_values = [1.0085362275720395],
        external_solution_values = Float64[],
        solution_values = [1.0085362275720395],
        previous_solution_initial_values = [1.0085362275720395],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0085362275720395],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 5,
        solve_order = 13,
        variables = ["gp_obs"],
        previous_solution_names = ["dA", "m"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [19],
        equations = Expr[
            :(gp_obs - m / dA),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["gp_obs"],
        previous_solution_values = [1.0085362275720395, 1.0002],
        external_solution_values = Float64[],
        solution_values = [0.9917343300675393],
        previous_solution_initial_values = [1.0085362275720395, 1.0002],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.9917343300675393],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 6,
        solve_order = 12,
        variables = ["dA"],
        previous_solution_names = ["➕₁"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [2],
        equations = Expr[
            :(dA - exp(➕₁)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["dA"],
        previous_solution_values = [0.0085],
        external_solution_values = Float64[],
        solution_values = [1.0085362275720395],
        previous_solution_initial_values = [0.0085],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0085362275720395],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 7,
        solve_order = 11,
        variables = ["➕₁"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [1],
        equations = Expr[
            :(➕₁ - gam),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["➕₁"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0085],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0085],
        box_lower_bounds = [-1.0e12],
        box_upper_bounds = [600.0],
    ),
    (
        index = 8,
        solve_order = 10,
        variables = ["d"],
        previous_solution_names = ["l", "m"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [15],
        equations = Expr[
            :(((d - l) + m) - 1),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["d"],
        previous_solution_values = [0.8610478832599587, 1.0002],
        external_solution_values = Float64[],
        solution_values = [0.8608478832599588],
        previous_solution_initial_values = [0.8610478832599587, 1.0002],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.8608478832599588],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 9,
        solve_order = 9,
        variables = ["R"],
        previous_solution_names = ["P", "W", "k", "n", "➕₄"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₃", "➕₁₄"],
        equation_indices = [10],
        equations = Expr[
            :((-P * ➕₁₃ ^ alp * (1 - alp) * exp(➕₄)) / (W * ➕₁₄ ^ alp) + R),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₃ = min(1.0e12, max(eps(), k))),
            :(➕₁₄ = min(1.0e12, max(eps(), n))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₃ - k)),
            :(abs(➕₁₄ - n)),
        ],
        solution_names = ["R", "➕₁₃", "➕₁₄"],
        previous_solution_values = [0.9932505545675542, 2.7185621689742447, 18.982191719109345, 0.3167291493594371, -0.003026],
        external_solution_values = Float64[],
        solution_values = [1.0072507552870087, 18.982191719109345, 0.3167291493594371],
        previous_solution_initial_values = [0.9932505545675542, 2.7185621689742447, 18.982191719109345, 0.3167291493594371, -0.003026],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0072507552870087, 18.982191719109345, 0.3167291493594371],
        box_lower_bounds = [-Inf, 2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12, 1.0e12],
    ),
    (
        index = 10,
        solve_order = 8,
        variables = ["W"],
        previous_solution_names = ["l", "n"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [7],
        equations = Expr[
            :(W - l / n),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["W"],
        previous_solution_values = [0.8610478832599587, 0.3167291493594371],
        external_solution_values = Float64[],
        solution_values = [2.7185621689742447],
        previous_solution_initial_values = [0.8610478832599587, 0.3167291493594371],
        external_solution_initial_values = Float64[],
        solution_initial_values = [2.7185621689742447],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 11,
        solve_order = 7,
        variables = ["P", "c", "k", "l", "n"],
        previous_solution_names = ["m", "➕₂", "➕₃", "➕₄", "➕₅"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [14, 6, 11, 8, 13],
        equations = Expr[
            :(P * c - m),
            :((bet * (alp * k ^ (alp - 1) * n ^ (1 - alp) * exp(➕₂) + (1 - del) * exp(➕₃))) / (c * m) - 1 / (c * m)),
            :((-bet * k ^ alp * n ^ (1 - alp) * (1 - alp) * exp(➕₄)) / (c * l * m) + 1 / (P * c)),
            :((-P * c * psi) / ((1 - n) * (1 - psi)) + l / n),
            :(((c - k * (1 - del) * exp(➕₅)) + k) - k ^ alp * n ^ (1 - alp) * exp(➕₄)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["P", "c", "k", "l", "n"],
        previous_solution_values = [1.0002, -0.003026, -0.0085, -0.003026, -0.0085],
        external_solution_values = Float64[],
        solution_values = [0.9932505545675542, 1.0069966690685328, 18.982191719109345, 0.8610478832599587, 0.3167291493594371],
        previous_solution_initial_values = [1.0002, -0.003026, -0.0085, -0.003026, -0.0085],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.9932505545675542, 1.0069966690685328, 18.982191719109345, 0.8610478832599587, 0.3167291493594371],
        box_lower_bounds = [-1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 12,
        solve_order = 6,
        variables = ["➕₃"],
        previous_solution_names = ["e"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₂"],
        equation_indices = [5],
        equations = Expr[
            :(➕₃ - (-gam - log(➕₁₂))),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₂ = min(1.0e12, max(eps(), e))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₂ - e)),
        ],
        solution_names = ["➕₃", "➕₁₂"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [-0.0085, 1.0],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-0.0085, 1.0],
        box_lower_bounds = [-1.0e12, 2.220446049250313e-16],
        box_upper_bounds = [600.0, 1.0e12],
    ),
    (
        index = 13,
        solve_order = 5,
        variables = ["➕₂"],
        previous_solution_names = ["e"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₁"],
        equation_indices = [4],
        equations = Expr[
            :(➕₂ - -alp * (gam + log(➕₁₁))),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₁ = min(1.0e12, max(eps(), e))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₁ - e)),
        ],
        solution_names = ["➕₂", "➕₁₁"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [-0.003026, 1.0],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-0.003026, 1.0],
        box_lower_bounds = [-1.0e12, 2.220446049250313e-16],
        box_upper_bounds = [600.0, 1.0e12],
    ),
    (
        index = 14,
        solve_order = 4,
        variables = ["e"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [16],
        equations = Expr[
            :(e - 1.0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["e"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 15,
        solve_order = 3,
        variables = ["m"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₀"],
        equation_indices = [3],
        equations = Expr[
            :((-rho * log(m) - (1 - rho) * log(➕₁₀)) + log(m)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₀ = min(1.0e12, max(eps(), mst))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₀ - mst)),
        ],
        solution_names = ["m", "➕₁₀"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.0002, 1.0002],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0002, 1.0002],
        box_lower_bounds = [2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12],
    ),
    (
        index = 16,
        solve_order = 2,
        variables = ["➕₅"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [12],
        equations = Expr[
            :(➕₅ - -gam),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["➕₅"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [-0.0085],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-0.0085],
        box_lower_bounds = [-1.0e12],
        box_upper_bounds = [600.0],
    ),
    (
        index = 17,
        solve_order = 1,
        variables = ["➕₄"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [9],
        equations = Expr[
            :(➕₄ - -alp * gam),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["➕₄"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [-0.003026],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-0.003026],
        box_lower_bounds = [-1.0e12],
        box_upper_bounds = [600.0],
    ),
]
const BLOCK_EQUATION_ORDER = [17, 20, 21, 18, 19, 2, 1, 15, 10, 7, 14, 6, 11, 8, 13, 5, 4, 16, 3, 12, 9]
const BLOCK_SOLVE_ORDER = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    ["k", "n", "➕₄"],
    ["gy_obs"],
    ["gp_obs"],
    ["dA"],
    ["dA", "m"],
    ["➕₁"],
    String[],
    ["l", "m"],
    ["P", "W", "k", "n", "➕₄"],
    ["l", "n"],
    ["m", "➕₂", "➕₃", "➕₄", "➕₅"],
    ["e"],
    ["e"],
    String[],
    String[],
    String[],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    [18.982191719109345, 0.3167291493594371, -0.003026],
    [1.0085362275720395],
    [0.9917343300675393],
    [1.0085362275720395],
    [1.0085362275720395, 1.0002],
    [0.0085],
    Float64[],
    [0.8610478832599587, 1.0002],
    [0.9932505545675542, 2.7185621689742447, 18.982191719109345, 0.3167291493594371, -0.003026],
    [0.8610478832599587, 0.3167291493594371],
    [1.0002, -0.003026, -0.0085, -0.003026, -0.0085],
    [1.0],
    [1.0],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
]
const BLOCK_EXTERNAL_SOLUTION_NAMES = [
    String[],
    String[],
    String[],
    String[],
    String[],
    String[],
    String[],
    String[],
    String[],
    String[],
    String[],
    String[],
    String[],
    String[],
    String[],
    String[],
    String[],
]
const BLOCK_EXTERNAL_SOLUTION_VALUES = [
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
]
const BLOCK_SOLUTION_NAMES = [
    ["y", "➕₁₇", "➕₁₈"],
    ["log_gy_obs", "➕₁₆"],
    ["log_gp_obs", "➕₁₅"],
    ["gy_obs"],
    ["gp_obs"],
    ["dA"],
    ["➕₁"],
    ["d"],
    ["R", "➕₁₃", "➕₁₄"],
    ["W"],
    ["P", "c", "k", "l", "n"],
    ["➕₃", "➕₁₂"],
    ["➕₂", "➕₁₁"],
    ["e"],
    ["m", "➕₁₀"],
    ["➕₅"],
    ["➕₄"],
]
const BLOCK_SOLUTION_VALUES = [
    [1.3558767746137388, 18.982191719109345, 0.3167291493594371],
    [0.008499999999999985, 1.0085362275720395],
    [-0.008300019997333728, 0.9917343300675393],
    [1.0085362275720395],
    [0.9917343300675393],
    [1.0085362275720395],
    [0.0085],
    [0.8608478832599588],
    [1.0072507552870087, 18.982191719109345, 0.3167291493594371],
    [2.7185621689742447],
    [0.9932505545675542, 1.0069966690685328, 18.982191719109345, 0.8610478832599587, 0.3167291493594371],
    [-0.0085, 1.0],
    [-0.003026, 1.0],
    [1.0],
    [1.0002, 1.0002],
    [-0.0085],
    [-0.003026],
]
const BLOCK_PREVIOUS_SOLUTION_INITIAL_VALUES = [
    [18.982191719109345, 0.3167291493594371, -0.003026],
    [1.0085362275720395],
    [0.9917343300675393],
    [1.0085362275720395],
    [1.0085362275720395, 1.0002],
    [0.0085],
    Float64[],
    [0.8610478832599587, 1.0002],
    [0.9932505545675542, 2.7185621689742447, 18.982191719109345, 0.3167291493594371, -0.003026],
    [0.8610478832599587, 0.3167291493594371],
    [1.0002, -0.003026, -0.0085, -0.003026, -0.0085],
    [1.0],
    [1.0],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
]
const BLOCK_EXTERNAL_SOLUTION_INITIAL_VALUES = [
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
]
const BLOCK_SOLUTION_INITIAL_VALUES = [
    [1.3558767746137388, 18.982191719109345, 0.3167291493594371],
    [0.008499999999999985, 1.0085362275720395],
    [-0.008300019997333728, 0.9917343300675393],
    [1.0085362275720395],
    [0.9917343300675393],
    [1.0085362275720395],
    [0.0085],
    [0.8608478832599588],
    [1.0072507552870087, 18.982191719109345, 0.3167291493594371],
    [2.7185621689742447],
    [0.9932505545675542, 1.0069966690685328, 18.982191719109345, 0.8610478832599587, 0.3167291493594371],
    [-0.0085, 1.0],
    [-0.003026, 1.0],
    [1.0],
    [1.0002, 1.0002],
    [-0.0085],
    [-0.003026],
]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[4] = parameters[4]
    complete_parameters[8] = parameters[8]
    complete_parameters[9] = parameters[9]
    complete_parameters[3] = parameters[3]
    complete_parameters[5] = parameters[5]
    complete_parameters[1] = parameters[1]
    complete_parameters[2] = parameters[2]
    complete_parameters[7] = parameters[7]
    complete_parameters[6] = parameters[6]
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[6] - exp(complete_parameters[3] + complete_parameters[8] * 0),
        log(solution[14]) - ((1 - complete_parameters[5]) * log(complete_parameters[4]) + complete_parameters[5] * log(solution[14]) + complete_parameters[9] * 0),
        (-(solution[1]) / (solution[4] * solution[1] * solution[14]) + (complete_parameters[2] * solution[1] * (complete_parameters[1] * exp(-(complete_parameters[1]) * (complete_parameters[3] + log(solution[7]))) * solution[10] ^ (complete_parameters[1] - 1) * solution[15] ^ (1 - complete_parameters[1]) + (1 - complete_parameters[7]) * exp(-((complete_parameters[3] + log(solution[7])))))) / (solution[4] * solution[1] * solution[14])) - 0,
        solution[3] - solution[11] / solution[15],
        (-(complete_parameters[6] / (1 - complete_parameters[6])) * ((solution[4] * solution[1]) / (1 - solution[15])) + solution[11] / solution[15]) - 0,
        solution[2] - (solution[1] * (1 - complete_parameters[1]) * exp(-(complete_parameters[1]) * (complete_parameters[3] + complete_parameters[8] * 0)) * solution[10] ^ complete_parameters[1] * solution[15] ^ -(complete_parameters[1])) / solution[3],
        (1 / (solution[4] * solution[1]) - (complete_parameters[2] * solution[1] * (1 - complete_parameters[1]) * exp(-(complete_parameters[1]) * (complete_parameters[3] + complete_parameters[8] * 0)) * solution[10] ^ complete_parameters[1] * solution[15] ^ (1 - complete_parameters[1])) / (solution[14] * solution[11] * solution[4] * solution[1])) - 0,
        (solution[4] + solution[10]) - (exp(-(complete_parameters[1]) * (complete_parameters[3] + complete_parameters[8] * 0)) * solution[10] ^ complete_parameters[1] * solution[15] ^ (1 - complete_parameters[1]) + (1 - complete_parameters[7]) * exp(-((complete_parameters[3] + complete_parameters[8] * 0))) * solution[10]),
        solution[1] * solution[4] - solution[14],
        ((solution[14] - 1) + solution[5]) - solution[11],
        solution[7] - exp(complete_parameters[8] * 0),
        solution[16] - solution[10] ^ complete_parameters[1] * solution[15] ^ (1 - complete_parameters[1]) * exp(-(complete_parameters[1]) * (complete_parameters[3] + complete_parameters[8] * 0)),
        solution[9] - (solution[6] * solution[16]) / solution[16],
        solution[8] - ((solution[1] / solution[1]) * solution[14]) / solution[6],
        solution[13] - log(solution[9]),
        solution[12] - log(solution[8]),
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[17] - complete_parameters[3],
        solution[6] - exp(solution[17]),
        (-(complete_parameters[5]) * log(solution[14]) - (1 - complete_parameters[5]) * log(complete_parameters[4])) + log(solution[14]),
        solution[18] - -(complete_parameters[1]) * (complete_parameters[3] + log(solution[7])),
        solution[19] - (-(complete_parameters[3]) - log(solution[7])),
        (complete_parameters[2] * (complete_parameters[1] * solution[10] ^ (complete_parameters[1] - 1) * solution[15] ^ (1 - complete_parameters[1]) * exp(solution[18]) + (1 - complete_parameters[7]) * exp(solution[19]))) / (solution[4] * solution[14]) - 1 / (solution[4] * solution[14]),
        solution[3] - solution[11] / solution[15],
        (-(solution[1]) * solution[4] * complete_parameters[6]) / ((1 - solution[15]) * (1 - complete_parameters[6])) + solution[11] / solution[15],
        solution[20] - -(complete_parameters[1]) * complete_parameters[3],
        (-(solution[1]) * solution[10] ^ complete_parameters[1] * (1 - complete_parameters[1]) * exp(solution[20])) / (solution[3] * solution[15] ^ complete_parameters[1]) + solution[2],
        (-(complete_parameters[2]) * solution[10] ^ complete_parameters[1] * solution[15] ^ (1 - complete_parameters[1]) * (1 - complete_parameters[1]) * exp(solution[20])) / (solution[4] * solution[11] * solution[14]) + 1 / (solution[1] * solution[4]),
        solution[21] - -(complete_parameters[3]),
        ((solution[4] - solution[10] * (1 - complete_parameters[7]) * exp(solution[21])) + solution[10]) - solution[10] ^ complete_parameters[1] * solution[15] ^ (1 - complete_parameters[1]) * exp(solution[20]),
        solution[1] * solution[4] - solution[14],
        ((solution[5] - solution[11]) + solution[14]) - 1,
        solution[7] - 1.0,
        -(solution[10] ^ complete_parameters[1]) * solution[15] ^ (1 - complete_parameters[1]) * exp(solution[20]) + solution[16],
        -(solution[6]) + solution[9],
        solution[8] - solution[14] / solution[6],
        solution[13] - log(solution[9]),
        solution[12] - log(solution[8]),
    ]
end

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 3
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[2] ^ complete_parameters[1]) * solution[3] ^ (1 - complete_parameters[1]) * exp(previous_solution[3]) + solution[1],
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
        solution[3] - min(1.0e12, max(eps(), previous_solution[2])),
    ]
end

function residuals_block_2(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - log(solution[2]),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - log(solution[2]),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_4(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_5(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[2] / previous_solution[1],
    ]
end

function residuals_block_6(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - exp(previous_solution[1]),
    ]
end

function residuals_block_7(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - complete_parameters[3],
    ]
end

function residuals_block_8(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((solution[1] - previous_solution[1]) + previous_solution[2]) - 1,
    ]
end

function residuals_block_9(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 5
    @assert length(external_solution) == 0
    @assert length(solution) == 3
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(previous_solution[1]) * solution[2] ^ complete_parameters[1] * (1 - complete_parameters[1]) * exp(previous_solution[5])) / (previous_solution[2] * solution[3] ^ complete_parameters[1]) + solution[1],
        solution[2] - min(1.0e12, max(eps(), previous_solution[3])),
        solution[3] - min(1.0e12, max(eps(), previous_solution[4])),
    ]
end

function residuals_block_10(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1] / previous_solution[2],
    ]
end

function residuals_block_11(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 5
    @assert length(external_solution) == 0
    @assert length(solution) == 5
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] * solution[2] - previous_solution[1],
        (complete_parameters[2] * (complete_parameters[1] * solution[3] ^ (complete_parameters[1] - 1) * solution[5] ^ (1 - complete_parameters[1]) * exp(previous_solution[2]) + (1 - complete_parameters[7]) * exp(previous_solution[3]))) / (solution[2] * previous_solution[1]) - 1 / (solution[2] * previous_solution[1]),
        (-(complete_parameters[2]) * solution[3] ^ complete_parameters[1] * solution[5] ^ (1 - complete_parameters[1]) * (1 - complete_parameters[1]) * exp(previous_solution[4])) / (solution[2] * solution[4] * previous_solution[1]) + 1 / (solution[1] * solution[2]),
        (-(solution[1]) * solution[2] * complete_parameters[6]) / ((1 - solution[5]) * (1 - complete_parameters[6])) + solution[4] / solution[5],
        ((solution[2] - solution[3] * (1 - complete_parameters[7]) * exp(previous_solution[5])) + solution[3]) - solution[3] ^ complete_parameters[1] * solution[5] ^ (1 - complete_parameters[1]) * exp(previous_solution[4]),
    ]
end

function residuals_block_12(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - (-(complete_parameters[3]) - log(solution[2])),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_13(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - -(complete_parameters[1]) * (complete_parameters[3] + log(solution[2])),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_14(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 1.0,
    ]
end

function residuals_block_15(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(complete_parameters[5]) * log(solution[1]) - (1 - complete_parameters[5]) * log(solution[2])) + log(solution[1]),
        solution[2] - min(1.0e12, max(eps(), complete_parameters[4])),
    ]
end

function residuals_block_16(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - -(complete_parameters[3]),
    ]
end

function residuals_block_17(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - -(complete_parameters[1]) * complete_parameters[3],
    ]
end

function residuals_blocks(parameters::AbstractVector, previous_solutions::AbstractVector, external_solutions::AbstractVector, solutions::AbstractVector)
    @assert length(previous_solutions) == length(BLOCKS)
    @assert length(external_solutions) == length(BLOCKS)
    @assert length(solutions) == length(BLOCKS)
    return vcat(
        residuals_block_1(parameters, previous_solutions[1], external_solutions[1], solutions[1]),
        residuals_block_2(parameters, previous_solutions[2], external_solutions[2], solutions[2]),
        residuals_block_3(parameters, previous_solutions[3], external_solutions[3], solutions[3]),
        residuals_block_4(parameters, previous_solutions[4], external_solutions[4], solutions[4]),
        residuals_block_5(parameters, previous_solutions[5], external_solutions[5], solutions[5]),
        residuals_block_6(parameters, previous_solutions[6], external_solutions[6], solutions[6]),
        residuals_block_7(parameters, previous_solutions[7], external_solutions[7], solutions[7]),
        residuals_block_8(parameters, previous_solutions[8], external_solutions[8], solutions[8]),
        residuals_block_9(parameters, previous_solutions[9], external_solutions[9], solutions[9]),
        residuals_block_10(parameters, previous_solutions[10], external_solutions[10], solutions[10]),
        residuals_block_11(parameters, previous_solutions[11], external_solutions[11], solutions[11]),
        residuals_block_12(parameters, previous_solutions[12], external_solutions[12], solutions[12]),
        residuals_block_13(parameters, previous_solutions[13], external_solutions[13], solutions[13]),
        residuals_block_14(parameters, previous_solutions[14], external_solutions[14], solutions[14]),
        residuals_block_15(parameters, previous_solutions[15], external_solutions[15], solutions[15]),
        residuals_block_16(parameters, previous_solutions[16], external_solutions[16], solutions[16]),
        residuals_block_17(parameters, previous_solutions[17], external_solutions[17], solutions[17]),
    )
end

export MODEL_NAME, SOURCE_MODEL_FILE, NSSS_SOLUTION_ERROR, NSSS_RESIDUAL_NORM
export PARAMETER_NAMES, PARAMETER_VALUES, COMPLETE_PARAMETER_NAMES, COMPLETE_PARAMETER_VALUES
export ORIGINAL_SOLUTION_NAMES, ORIGINAL_SOLUTION_VALUES, ORIGINAL_INITIAL_SOLUTION_VALUES
export AUXILIARY_SOLUTION_NAMES, AUXILIARY_SOLUTION_VALUES, AUXILIARY_INITIAL_SOLUTION_VALUES
export ALL_AUXILIARY_VARIABLE_NAMES, ALL_AUXILIARY_VARIABLE_VALUES, ALL_AUXILIARY_VARIABLE_INITIAL_VALUES
export DEFAULTED_NSSS_SOLUTION_NAMES
export ORIGINAL_NSSS_EQUATIONS, AUXILIARY_NSSS_EQUATIONS, CALIBRATION_EQUATIONS
export BLOCKS, BLOCK_EQUATION_ORDER, BLOCK_SOLVE_ORDER
export BLOCK_PREVIOUS_SOLUTION_NAMES, BLOCK_PREVIOUS_SOLUTION_VALUES
export BLOCK_PREVIOUS_SOLUTION_INITIAL_VALUES
export BLOCK_EXTERNAL_SOLUTION_NAMES, BLOCK_EXTERNAL_SOLUTION_VALUES
export BLOCK_EXTERNAL_SOLUTION_INITIAL_VALUES
export BLOCK_SOLUTION_NAMES, BLOCK_SOLUTION_VALUES, BLOCK_SOLUTION_INITIAL_VALUES
export residuals_original, residuals_auxiliary, residuals_blocks
export residuals_block_1, residuals_block_2, residuals_block_3, residuals_block_4, residuals_block_5, residuals_block_6, residuals_block_7, residuals_block_8, residuals_block_9, residuals_block_10, residuals_block_11, residuals_block_12, residuals_block_13, residuals_block_14, residuals_block_15, residuals_block_16, residuals_block_17
end
