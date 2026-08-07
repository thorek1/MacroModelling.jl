module Ascari_Sbordone_2014NsssResiduals
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

const MODEL_NAME = "Ascari_Sbordone_2014"
const SOURCE_MODEL_FILE = "models/Ascari_Sbordone_2014.jl"
const NSSS_SOLUTION_ERROR = 1.8914564400730272e-15
const NSSS_RESIDUAL_NORM = 1.8318679906315083e-15

const PARAMETER_NAMES = [
    "beta",
    "trend_inflation",
    "alpha",
    "theta",
    "epsilon",
    "sigma",
    "rho_v",
    "rho_a",
    "rho_zeta",
    "phi_par",
    "phi_pi",
    "phi_y",
    "rho_i",
    "var_rho",
    "σ_zeta",
    "σₐ",
    "σᵥ",
]
const PARAMETER_VALUES = Float64[
    0.99,
    0.0,
    0.0,
    0.75,
    10.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    2.0,
    0.125,
    0.8,
    0.0,
    0.01,
    0.01,
    0.01,
]
const COMPLETE_PARAMETER_NAMES = [
    "beta",
    "trend_inflation",
    "alpha",
    "theta",
    "epsilon",
    "sigma",
    "rho_v",
    "rho_a",
    "rho_zeta",
    "phi_par",
    "phi_pi",
    "phi_y",
    "rho_i",
    "var_rho",
    "σ_zeta",
    "σₐ",
    "σᵥ",
    "Pi_bar",
    "i_bar",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    0.99,
    0.0,
    0.0,
    0.75,
    10.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    2.0,
    0.125,
    0.8,
    0.0,
    0.01,
    0.01,
    0.01,
    1.0,
    0.010101010101010166,
]
const ORIGINAL_SOLUTION_NAMES = [
    "A",
    "A_tilde",
    "Average_markup",
    "MC_real",
    "Marginal_markup",
    "N",
    "Utility",
    "i",
    "p_star",
    "phi",
    "pi",
    "price_adjustment_gap",
    "psi",
    "real_interest",
    "s",
    "v",
    "w",
    "y",
    "zeta",
    "d_n",
    "Y_bar",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    0.0,
    1.000000000000001,
    1.1111111132484524,
    0.8999999982687535,
    1.1111110582123165,
    0.3333333333333333,
    -154.86122878024847,
    0.010100993423387527,
    0.9999999504674777,
    3.8834934816299174,
    0.9999999834891536,
    1.0000000495325247,
    3.4951439603436207,
    1.0101010101010102,
    0.9999999999999989,
    0.0,
    0.8999999982687535,
    0.3333333333333333,
    0.0,
    8.099999984418782,
    0.3333332893044127,
]
const AUXILIARY_SOLUTION_NAMES = [
    "A",
    "A_tilde",
    "Average_markup",
    "MC_real",
    "Marginal_markup",
    "N",
    "Utility",
    "i",
    "p_star",
    "phi",
    "pi",
    "price_adjustment_gap",
    "psi",
    "real_interest",
    "s",
    "v",
    "w",
    "y",
    "zeta",
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
    "➕₆",
    "➕₇",
    "d_n",
    "Y_bar",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    0.0,
    1.000000000000001,
    1.1111111132484524,
    0.8999999982687535,
    1.1111110582123152,
    0.3333333333333329,
    -154.86122878024847,
    0.010100993423387527,
    0.9999999504674765,
    3.8834934816299174,
    0.9999999834891536,
    1.000000049532526,
    3.4951439603436207,
    1.0101010101010102,
    0.9999999999999989,
    0.0,
    0.8999999982687535,
    0.3333333333333333,
    0.0,
    1.0000004457928222,
    1.0,
    0.3333333333333333,
    0.9999999834891536,
    0.9999999834891536,
    1.0000001320867793,
    0.9999999834891534,
    8.099999984418782,
    0.3333332893044127,
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
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    1.0000004457928222,
    1.0,
    0.3333333333333333,
    0.9999999834891536,
    0.9999999834891536,
    1.0000001320867793,
    0.9999999834891534,
    0.9999999834891536,
    0.9999999504674777,
    0.9999999834891536,
    0.3333333333333333,
]
const DEFAULTED_NSSS_SOLUTION_NAMES = [
]
const CALIBRATION_PARAMETER_NAMES = [
    "d_n",
    "Y_bar",
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(1 / y ^ sigma - (beta * (1 + i)) / (pi * y ^ sigma)),
    :(w - y ^ sigma * d_n * exp(zeta) * N ^ phi_par),
    :(p_star - ((1 - theta * pi ^ ((1 - epsilon) * var_rho) * pi ^ (epsilon - 1)) / (1 - theta)) ^ (1 / (1 - epsilon))),
    :(p_star ^ (1 + (epsilon * alpha) / (1 - alpha)) - ((epsilon / ((epsilon - 1) * (1 - alpha))) * psi) / phi),
    :(psi - (w * exp(A) ^ (-1 / (1 - alpha)) * y ^ (1 / (1 - alpha) - sigma) + beta * theta * pi ^ ((epsilon * -var_rho) / (1 - alpha)) * pi ^ (epsilon / (1 - alpha)) * psi)),
    :(phi - (y ^ (1 - sigma) + beta * theta * pi ^ ((1 - epsilon) * var_rho) * pi ^ (epsilon - 1) * phi)),
    :(N - s * (y / exp(A)) ^ (1 / (1 - alpha))),
    :(s - ((1 - theta) * p_star ^ (-epsilon / (1 - alpha)) + theta * pi ^ ((var_rho * -epsilon) / (1 - alpha)) * pi ^ (epsilon / (1 - alpha)) * s)),
    :((1 + i) / (1 + i_bar) - ((1 + i) / (1 + i_bar)) ^ rho_i * ((pi / Pi_bar) ^ phi_pi * (y / Y_bar) ^ phi_y) ^ (1 - rho_i) * exp(v)),
    :(MC_real - ((w * 1) / (1 - alpha)) * exp(A) ^ (1 / (alpha - 1)) * y ^ (alpha / (1 - alpha))),
    :(real_interest - (1 + i) / pi),
    :(Utility - ((log(y) - (d_n * exp(zeta) * N ^ (1 + phi_par)) / (1 + phi_par)) + beta * Utility)),
    :(v - (rho_v * v + σᵥ * 0)),
    :(A - (rho_a * A + σₐ * 0)),
    :(zeta - (rho_zeta * zeta + σ_zeta * 0)),
    :(A_tilde - exp(A) / s),
    :(Average_markup - 1 / MC_real),
    :(Marginal_markup - p_star / MC_real),
    :(price_adjustment_gap - 1 / p_star),
]
const CALIBRATION_EQUATIONS = Expr[
    :(N - 1 / 3),
    :((1 / 3) ^ (1 - alpha) - y),
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :((-beta * (i + 1)) / (pi * y ^ sigma) + y ^ -sigma),
    :(-(N ^ phi_par) * d_n * y ^ sigma * exp(zeta) + w),
    :(➕₁ - (-(pi ^ (var_rho * (1 - epsilon))) * pi ^ (epsilon - 1) * theta + 1) / (1 - theta)),
    :(p_star - ➕₁ ^ (1 / (1 - epsilon))),
    :((-epsilon * psi) / (phi * (1 - alpha) * (epsilon - 1)) + p_star ^ ((alpha * epsilon) / (1 - alpha) + 1)),
    :(➕₂ - exp(A)),
    :(((-beta * pi ^ (epsilon / (1 - alpha)) * psi * theta) / pi ^ ((epsilon * var_rho) / (1 - alpha)) + psi) - (w * y ^ (-sigma + 1 / (1 - alpha))) / ➕₂ ^ (1 / (1 - alpha))),
    :((-beta * phi * pi ^ (var_rho * (1 - epsilon)) * pi ^ (epsilon - 1) * theta + phi) - y ^ (1 - sigma)),
    :(➕₃ - y * exp(-A)),
    :(N - s * ➕₃ ^ (1 / (1 - alpha))),
    :(((-(pi ^ (epsilon / (1 - alpha))) * s * theta) / pi ^ ((epsilon * var_rho) / (1 - alpha)) + s) - (1 - theta) / p_star ^ (epsilon / (1 - alpha))),
    :(➕₄ - (i + 1) / (i_bar + 1)),
    :(➕₅ - pi / Pi_bar),
    :(➕₆ - y / Y_bar),
    :(➕₇ - ➕₅ ^ phi_pi * ➕₆ ^ phi_y),
    :(-(➕₄ ^ rho_i) * ➕₇ ^ (1 - rho_i) * exp(v) + (i + 1) / (i_bar + 1)),
    :(MC_real - (w * y ^ (alpha / (1 - alpha)) * ➕₂ ^ (1 / (alpha - 1))) / (1 - alpha)),
    :(real_interest - (i + 1) / pi),
    :((((N ^ (phi_par + 1) * d_n * exp(zeta)) / (phi_par + 1) - Utility * beta) + Utility) - log(y)),
    :(-rho_v * v + v),
    :(-A * rho_a + A),
    :(-rho_zeta * zeta + zeta),
    :(A_tilde - exp(A) / s),
    :(Average_markup - 1 / MC_real),
    :(Marginal_markup - p_star / MC_real),
    :(price_adjustment_gap - 1 / p_star),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(1 / y ^ sigma - (beta * (1 + i)) / (pi * y ^ sigma)),
    :(w - y ^ sigma * d_n * exp(zeta) * N ^ phi_par),
    :(p_star - ((1 - theta * pi ^ ((1 - epsilon) * var_rho) * pi ^ (epsilon - 1)) / (1 - theta)) ^ (1 / (1 - epsilon))),
    :(p_star ^ (1 + (epsilon * alpha) / (1 - alpha)) - ((epsilon / ((epsilon - 1) * (1 - alpha))) * psi) / phi),
    :(psi - (w * exp(A) ^ (-1 / (1 - alpha)) * y ^ (1 / (1 - alpha) - sigma) + beta * theta * pi ^ ((epsilon * -var_rho) / (1 - alpha)) * pi ^ (epsilon / (1 - alpha)) * psi)),
    :(phi - (y ^ (1 - sigma) + beta * theta * pi ^ ((1 - epsilon) * var_rho) * pi ^ (epsilon - 1) * phi)),
    :(N - s * (y / exp(A)) ^ (1 / (1 - alpha))),
    :(s - ((1 - theta) * p_star ^ (-epsilon / (1 - alpha)) + theta * pi ^ ((var_rho * -epsilon) / (1 - alpha)) * pi ^ (epsilon / (1 - alpha)) * s)),
    :((1 + i) / (1 + i_bar) - ((1 + i) / (1 + i_bar)) ^ rho_i * ((pi / Pi_bar) ^ phi_pi * (y / Y_bar) ^ phi_y) ^ (1 - rho_i) * exp(v)),
    :(MC_real - ((w * 1) / (1 - alpha)) * exp(A) ^ (1 / (alpha - 1)) * y ^ (alpha / (1 - alpha))),
    :(real_interest - (1 + i) / pi),
    :(Utility - ((log(y) - (d_n * exp(zeta) * N ^ (1 + phi_par)) / (1 + phi_par)) + beta * Utility)),
    :(v - (rho_v * v + σᵥ * 0)),
    :(A - (rho_a * A + σₐ * 0)),
    :(zeta - (rho_zeta * zeta + σ_zeta * 0)),
    :(A_tilde - exp(A) / s),
    :(Average_markup - 1 / MC_real),
    :(Marginal_markup - p_star / MC_real),
    :(price_adjustment_gap - 1 / p_star),
    :(N - 1 / 3),
    :((1 / 3) ^ (1 - alpha) - y),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :((-beta * (i + 1)) / (pi * y ^ sigma) + y ^ -sigma),
    :(-(N ^ phi_par) * d_n * y ^ sigma * exp(zeta) + w),
    :(➕₁ - (-(pi ^ (var_rho * (1 - epsilon))) * pi ^ (epsilon - 1) * theta + 1) / (1 - theta)),
    :(p_star - ➕₁ ^ (1 / (1 - epsilon))),
    :((-epsilon * psi) / (phi * (1 - alpha) * (epsilon - 1)) + p_star ^ ((alpha * epsilon) / (1 - alpha) + 1)),
    :(➕₂ - exp(A)),
    :(((-beta * pi ^ (epsilon / (1 - alpha)) * psi * theta) / pi ^ ((epsilon * var_rho) / (1 - alpha)) + psi) - (w * y ^ (-sigma + 1 / (1 - alpha))) / ➕₂ ^ (1 / (1 - alpha))),
    :((-beta * phi * pi ^ (var_rho * (1 - epsilon)) * pi ^ (epsilon - 1) * theta + phi) - y ^ (1 - sigma)),
    :(➕₃ - y * exp(-A)),
    :(N - s * ➕₃ ^ (1 / (1 - alpha))),
    :(((-(pi ^ (epsilon / (1 - alpha))) * s * theta) / pi ^ ((epsilon * var_rho) / (1 - alpha)) + s) - (1 - theta) / p_star ^ (epsilon / (1 - alpha))),
    :(➕₄ - (i + 1) / (i_bar + 1)),
    :(➕₅ - pi / Pi_bar),
    :(➕₆ - y / Y_bar),
    :(➕₇ - ➕₅ ^ phi_pi * ➕₆ ^ phi_y),
    :(-(➕₄ ^ rho_i) * ➕₇ ^ (1 - rho_i) * exp(v) + (i + 1) / (i_bar + 1)),
    :(MC_real - (w * y ^ (alpha / (1 - alpha)) * ➕₂ ^ (1 / (alpha - 1))) / (1 - alpha)),
    :(real_interest - (i + 1) / pi),
    :((((N ^ (phi_par + 1) * d_n * exp(zeta)) / (phi_par + 1) - Utility * beta) + Utility) - log(y)),
    :(-rho_v * v + v),
    :(-A * rho_a + A),
    :(-rho_zeta * zeta + zeta),
    :(A_tilde - exp(A) / s),
    :(Average_markup - 1 / MC_real),
    :(Marginal_markup - p_star / MC_real),
    :(price_adjustment_gap - 1 / p_star),
    :(N - 1 / 3),
    :((1 / 3) ^ (1 - alpha) - y),
]

const PARAMETER_DEFINITION_NAMES = [
    "Pi_bar",
    "i_bar",
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
    "(1 + trend_inflation / 100) ^ (1 / 4)",
    "Pi_bar / beta - 1",
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "beta",
    "trend_inflation",
    "alpha",
    "theta",
    "epsilon",
    "sigma",
    "rho_v",
    "rho_a",
    "rho_zeta",
    "phi_par",
    "phi_pi",
    "phi_y",
    "rho_i",
    "var_rho",
    "σ_zeta",
    "σₐ",
    "σᵥ",
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
    "A",
    "A_tilde",
    "Average_markup",
    "MC_real",
    "Marginal_markup",
    "N",
    "Utility",
    "i",
    "p_star",
    "phi",
    "pi",
    "price_adjustment_gap",
    "psi",
    "real_interest",
    "s",
    "v",
    "w",
    "y",
    "zeta",
    "d_n",
    "Y_bar",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -Inf,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
]
const ORIGINAL_BOX_UPPER_BOUNDS = Float64[
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    1.0e12,
    Inf,
    1.0e12,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    1.0e12,
    Inf,
    Inf,
    Inf,
    Inf,
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "A",
    "A_tilde",
    "Average_markup",
    "MC_real",
    "Marginal_markup",
    "N",
    "Utility",
    "i",
    "p_star",
    "phi",
    "pi",
    "price_adjustment_gap",
    "psi",
    "real_interest",
    "s",
    "v",
    "w",
    "y",
    "zeta",
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
    "➕₆",
    "➕₇",
    "d_n",
    "Y_bar",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -Inf,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -Inf,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -Inf,
    -Inf,
]
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    1.0e12,
    Inf,
    1.0e12,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    1.0e12,
    Inf,
    Inf,
    1.0e12,
    Inf,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
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
]
const ALL_AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    2.220446049250313e-16,
    -Inf,
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
    1.0e12,
    Inf,
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
        variables = ["Y_bar"],
        equation_indices = [14],
        equations = Expr[
            :(➕₆ - y / Y_bar),
        ],
    ),
    (
        index = 2,
        variables = ["➕₆"],
        equation_indices = [15],
        equations = Expr[
            :(➕₇ - ➕₅ ^ phi_pi * ➕₆ ^ phi_y),
        ],
    ),
    (
        index = 3,
        variables = ["➕₇"],
        equation_indices = [16],
        equations = Expr[
            :(-(➕₄ ^ rho_i) * ➕₇ ^ (1 - rho_i) * exp(v) + (i + 1) / (i_bar + 1)),
        ],
    ),
    (
        index = 4,
        variables = ["➕₅"],
        equation_indices = [13],
        equations = Expr[
            :(➕₅ - pi / Pi_bar),
        ],
    ),
    (
        index = 5,
        variables = ["➕₄"],
        equation_indices = [12],
        equations = Expr[
            :(➕₄ - (i + 1) / (i_bar + 1)),
        ],
    ),
    (
        index = 6,
        variables = ["v"],
        equation_indices = [20],
        equations = Expr[
            :(-rho_v * v + v),
        ],
    ),
    (
        index = 7,
        variables = ["real_interest"],
        equation_indices = [18],
        equations = Expr[
            :(real_interest - (i + 1) / pi),
        ],
    ),
    (
        index = 8,
        variables = ["price_adjustment_gap"],
        equation_indices = [26],
        equations = Expr[
            :(price_adjustment_gap - 1 / p_star),
        ],
    ),
    (
        index = 9,
        variables = ["i"],
        equation_indices = [1],
        equations = Expr[
            :((-beta * (i + 1)) / (pi * y ^ sigma) + y ^ -sigma),
        ],
    ),
    (
        index = 10,
        variables = ["Utility"],
        equation_indices = [19],
        equations = Expr[
            :((((N ^ (phi_par + 1) * d_n * exp(zeta)) / (phi_par + 1) - Utility * beta) + Utility) - log(y)),
        ],
    ),
    (
        index = 11,
        variables = ["d_n"],
        equation_indices = [2],
        equations = Expr[
            :(-(N ^ phi_par) * d_n * y ^ sigma * exp(zeta) + w),
        ],
    ),
    (
        index = 12,
        variables = ["zeta"],
        equation_indices = [22],
        equations = Expr[
            :(-rho_zeta * zeta + zeta),
        ],
    ),
    (
        index = 13,
        variables = ["Marginal_markup"],
        equation_indices = [25],
        equations = Expr[
            :(Marginal_markup - p_star / MC_real),
        ],
    ),
    (
        index = 14,
        variables = ["Average_markup"],
        equation_indices = [24],
        equations = Expr[
            :(Average_markup - 1 / MC_real),
        ],
    ),
    (
        index = 15,
        variables = ["MC_real"],
        equation_indices = [17],
        equations = Expr[
            :(MC_real - (w * y ^ (alpha / (1 - alpha)) * ➕₂ ^ (1 / (alpha - 1))) / (1 - alpha)),
        ],
    ),
    (
        index = 16,
        variables = ["w"],
        equation_indices = [7],
        equations = Expr[
            :(((-beta * pi ^ (epsilon / (1 - alpha)) * psi * theta) / pi ^ ((epsilon * var_rho) / (1 - alpha)) + psi) - (w * y ^ (-sigma + 1 / (1 - alpha))) / ➕₂ ^ (1 / (1 - alpha))),
        ],
    ),
    (
        index = 17,
        variables = ["➕₂"],
        equation_indices = [6],
        equations = Expr[
            :(➕₂ - exp(A)),
        ],
    ),
    (
        index = 18,
        variables = ["psi"],
        equation_indices = [5],
        equations = Expr[
            :((-epsilon * psi) / (phi * (1 - alpha) * (epsilon - 1)) + p_star ^ ((alpha * epsilon) / (1 - alpha) + 1)),
        ],
    ),
    (
        index = 19,
        variables = ["phi"],
        equation_indices = [8],
        equations = Expr[
            :((-beta * phi * pi ^ (var_rho * (1 - epsilon)) * pi ^ (epsilon - 1) * theta + phi) - y ^ (1 - sigma)),
        ],
    ),
    (
        index = 20,
        variables = ["p_star", "pi", "➕₁"],
        equation_indices = [11, 3, 4],
        equations = Expr[
            :(((-(pi ^ (epsilon / (1 - alpha))) * s * theta) / pi ^ ((epsilon * var_rho) / (1 - alpha)) + s) - (1 - theta) / p_star ^ (epsilon / (1 - alpha))),
            :(➕₁ - (-(pi ^ (var_rho * (1 - epsilon))) * pi ^ (epsilon - 1) * theta + 1) / (1 - theta)),
            :(p_star - ➕₁ ^ (1 / (1 - epsilon))),
        ],
    ),
    (
        index = 21,
        variables = ["A_tilde"],
        equation_indices = [23],
        equations = Expr[
            :(A_tilde - exp(A) / s),
        ],
    ),
    (
        index = 22,
        variables = ["s"],
        equation_indices = [10],
        equations = Expr[
            :(N - s * ➕₃ ^ (1 / (1 - alpha))),
        ],
    ),
    (
        index = 23,
        variables = ["➕₃"],
        equation_indices = [9],
        equations = Expr[
            :(➕₃ - y * exp(-A)),
        ],
    ),
    (
        index = 24,
        variables = ["y"],
        equation_indices = [28],
        equations = Expr[
            :((1 / 3) ^ (1 - alpha) - y),
        ],
    ),
    (
        index = 25,
        variables = ["N"],
        equation_indices = [27],
        equations = Expr[
            :(N - 1 / 3),
        ],
    ),
    (
        index = 26,
        variables = ["A"],
        equation_indices = [21],
        equations = Expr[
            :(-A * rho_a + A),
        ],
    ),
]
const BLOCK_EQUATION_ORDER = [14, 15, 16, 13, 12, 20, 18, 26, 1, 19, 2, 22, 25, 24, 17, 7, 6, 5, 8, 11, 3, 4, 23, 10, 9, 28, 27, 21]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[3] = parameters[3]
    complete_parameters[15] = parameters[15]
    complete_parameters[13] = parameters[13]
    complete_parameters[17] = parameters[17]
    complete_parameters[2] = parameters[2]
    complete_parameters[16] = parameters[16]
    complete_parameters[1] = parameters[1]
    complete_parameters[7] = parameters[7]
    complete_parameters[9] = parameters[9]
    complete_parameters[10] = parameters[10]
    complete_parameters[11] = parameters[11]
    complete_parameters[5] = parameters[5]
    complete_parameters[4] = parameters[4]
    complete_parameters[6] = parameters[6]
    complete_parameters[14] = parameters[14]
    complete_parameters[8] = parameters[8]
    complete_parameters[12] = parameters[12]
    complete_parameters[18] = (1 + complete_parameters[2] / 100) ^ (1 / 4)
    complete_parameters[19] = complete_parameters[18] / complete_parameters[1] - 1
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        1 / solution[18] ^ complete_parameters[6] - (complete_parameters[1] * (1 + solution[8])) / (solution[11] * solution[18] ^ complete_parameters[6]),
        solution[17] - solution[18] ^ complete_parameters[6] * solution[20] * exp(solution[19]) * solution[6] ^ complete_parameters[10],
        solution[9] - ((1 - complete_parameters[4] * solution[11] ^ ((1 - complete_parameters[5]) * complete_parameters[14]) * solution[11] ^ (complete_parameters[5] - 1)) / (1 - complete_parameters[4])) ^ (1 / (1 - complete_parameters[5])),
        solution[9] ^ (1 + (complete_parameters[5] * complete_parameters[3]) / (1 - complete_parameters[3])) - ((complete_parameters[5] / ((complete_parameters[5] - 1) * (1 - complete_parameters[3]))) * solution[13]) / solution[10],
        solution[13] - (solution[17] * exp(solution[1]) ^ (-1 / (1 - complete_parameters[3])) * solution[18] ^ (1 / (1 - complete_parameters[3]) - complete_parameters[6]) + complete_parameters[1] * complete_parameters[4] * solution[11] ^ ((complete_parameters[5] * -(complete_parameters[14])) / (1 - complete_parameters[3])) * solution[11] ^ (complete_parameters[5] / (1 - complete_parameters[3])) * solution[13]),
        solution[10] - (solution[18] ^ (1 - complete_parameters[6]) + complete_parameters[1] * complete_parameters[4] * solution[11] ^ ((1 - complete_parameters[5]) * complete_parameters[14]) * solution[11] ^ (complete_parameters[5] - 1) * solution[10]),
        solution[6] - solution[15] * (solution[18] / exp(solution[1])) ^ (1 / (1 - complete_parameters[3])),
        solution[15] - ((1 - complete_parameters[4]) * solution[9] ^ (-(complete_parameters[5]) / (1 - complete_parameters[3])) + complete_parameters[4] * solution[11] ^ ((complete_parameters[14] * -(complete_parameters[5])) / (1 - complete_parameters[3])) * solution[11] ^ (complete_parameters[5] / (1 - complete_parameters[3])) * solution[15]),
        (1 + solution[8]) / (1 + complete_parameters[19]) - ((1 + solution[8]) / (1 + complete_parameters[19])) ^ complete_parameters[13] * ((solution[11] / complete_parameters[18]) ^ complete_parameters[11] * (solution[18] / solution[21]) ^ complete_parameters[12]) ^ (1 - complete_parameters[13]) * exp(solution[16]),
        solution[4] - ((solution[17] * 1) / (1 - complete_parameters[3])) * exp(solution[1]) ^ (1 / (complete_parameters[3] - 1)) * solution[18] ^ (complete_parameters[3] / (1 - complete_parameters[3])),
        solution[14] - (1 + solution[8]) / solution[11],
        solution[7] - ((log(solution[18]) - (solution[20] * exp(solution[19]) * solution[6] ^ (1 + complete_parameters[10])) / (1 + complete_parameters[10])) + complete_parameters[1] * solution[7]),
        solution[16] - (complete_parameters[7] * solution[16] + complete_parameters[17] * 0),
        solution[1] - (complete_parameters[8] * solution[1] + complete_parameters[16] * 0),
        solution[19] - (complete_parameters[9] * solution[19] + complete_parameters[15] * 0),
        solution[2] - exp(solution[1]) / solution[15],
        solution[3] - 1 / solution[4],
        solution[5] - solution[9] / solution[4],
        solution[12] - 1 / solution[9],
        solution[6] - 1 / 3,
        (1 / 3) ^ (1 - complete_parameters[3]) - solution[18],
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(complete_parameters[1]) * (solution[8] + 1)) / (solution[11] * solution[18] ^ complete_parameters[6]) + solution[18] ^ -(complete_parameters[6]),
        -(solution[6] ^ complete_parameters[10]) * solution[27] * solution[18] ^ complete_parameters[6] * exp(solution[19]) + solution[17],
        solution[20] - (-(solution[11] ^ (complete_parameters[14] * (1 - complete_parameters[5]))) * solution[11] ^ (complete_parameters[5] - 1) * complete_parameters[4] + 1) / (1 - complete_parameters[4]),
        solution[9] - solution[20] ^ (1 / (1 - complete_parameters[5])),
        (-(complete_parameters[5]) * solution[13]) / (solution[10] * (1 - complete_parameters[3]) * (complete_parameters[5] - 1)) + solution[9] ^ ((complete_parameters[3] * complete_parameters[5]) / (1 - complete_parameters[3]) + 1),
        solution[21] - exp(solution[1]),
        ((-(complete_parameters[1]) * solution[11] ^ (complete_parameters[5] / (1 - complete_parameters[3])) * solution[13] * complete_parameters[4]) / solution[11] ^ ((complete_parameters[5] * complete_parameters[14]) / (1 - complete_parameters[3])) + solution[13]) - (solution[17] * solution[18] ^ (-(complete_parameters[6]) + 1 / (1 - complete_parameters[3]))) / solution[21] ^ (1 / (1 - complete_parameters[3])),
        (-(complete_parameters[1]) * solution[10] * solution[11] ^ (complete_parameters[14] * (1 - complete_parameters[5])) * solution[11] ^ (complete_parameters[5] - 1) * complete_parameters[4] + solution[10]) - solution[18] ^ (1 - complete_parameters[6]),
        solution[22] - solution[18] * exp(-(solution[1])),
        solution[6] - solution[15] * solution[22] ^ (1 / (1 - complete_parameters[3])),
        ((-(solution[11] ^ (complete_parameters[5] / (1 - complete_parameters[3]))) * solution[15] * complete_parameters[4]) / solution[11] ^ ((complete_parameters[5] * complete_parameters[14]) / (1 - complete_parameters[3])) + solution[15]) - (1 - complete_parameters[4]) / solution[9] ^ (complete_parameters[5] / (1 - complete_parameters[3])),
        solution[23] - (solution[8] + 1) / (complete_parameters[19] + 1),
        solution[24] - solution[11] / complete_parameters[18],
        solution[25] - solution[18] / solution[28],
        solution[26] - solution[24] ^ complete_parameters[11] * solution[25] ^ complete_parameters[12],
        -(solution[23] ^ complete_parameters[13]) * solution[26] ^ (1 - complete_parameters[13]) * exp(solution[16]) + (solution[8] + 1) / (complete_parameters[19] + 1),
        solution[4] - (solution[17] * solution[18] ^ (complete_parameters[3] / (1 - complete_parameters[3])) * solution[21] ^ (1 / (complete_parameters[3] - 1))) / (1 - complete_parameters[3]),
        solution[14] - (solution[8] + 1) / solution[11],
        (((solution[6] ^ (complete_parameters[10] + 1) * solution[27] * exp(solution[19])) / (complete_parameters[10] + 1) - solution[7] * complete_parameters[1]) + solution[7]) - log(solution[18]),
        -(complete_parameters[7]) * solution[16] + solution[16],
        -(solution[1]) * complete_parameters[8] + solution[1],
        -(complete_parameters[9]) * solution[19] + solution[19],
        solution[2] - exp(solution[1]) / solution[15],
        solution[3] - 1 / solution[4],
        solution[5] - solution[9] / solution[4],
        solution[12] - 1 / solution[9],
        solution[6] - 1 / 3,
        (1 / 3) ^ (1 - complete_parameters[3]) - solution[18],
    ]
end

function residuals_blocks(parameters::AbstractVector, solution::AbstractVector)
    return residuals_auxiliary(parameters, solution)[BLOCK_EQUATION_ORDER]
end

export MODEL_NAME, SOURCE_MODEL_FILE, NSSS_SOLUTION_ERROR, NSSS_RESIDUAL_NORM
export PARAMETER_NAMES, PARAMETER_VALUES, COMPLETE_PARAMETER_NAMES, COMPLETE_PARAMETER_VALUES
export ORIGINAL_SOLUTION_NAMES, ORIGINAL_SOLUTION_VALUES
export AUXILIARY_SOLUTION_NAMES, AUXILIARY_SOLUTION_VALUES
export ALL_AUXILIARY_VARIABLE_NAMES, ALL_AUXILIARY_VARIABLE_VALUES
export DEFAULTED_NSSS_SOLUTION_NAMES
export ORIGINAL_NSSS_EQUATIONS, AUXILIARY_NSSS_EQUATIONS, CALIBRATION_EQUATIONS
export BLOCKS, BLOCK_EQUATION_ORDER, residuals_original, residuals_auxiliary, residuals_blocks
end
