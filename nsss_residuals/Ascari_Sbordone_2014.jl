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
const ORIGINAL_INITIAL_SOLUTION_VALUES = Float64[
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
const AUXILIARY_INITIAL_SOLUTION_VALUES = Float64[
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
    1.0000004457928229,
    1.0,
    0.3333333333333333,
    0.9999999834891536,
    0.9999999834891536,
    1.0000001320867793,
    0.9999999834891535,
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
    "➕₁₂",
    "➕₁₃",
    "➕₁₄",
    "➕₁₅",
    "➕₁₆",
    "➕₁₇",
    "➕₁₈",
    "➕₁₉",
    "➕₂₀",
    "➕₂₁",
    "➕₂₂",
    "➕₂₃",
    "➕₂₄",
    "➕₂₅",
    "➕₂₆",
    "➕₂₇",
    "➕₂₈",
    "➕₂₉",
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
    0.3333333333333333,
    -0.0,
    0.0,
    0.9999999834891536,
    0.3333333333333333,
    0.9999999504674765,
    0.0,
    0.9999999834891536,
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333329,
    0.3333333333333333,
    0.0,
    0.3333333333333329,
    0.0,
    0.3333333333333333,
    0.3333333333333333,
    0.0,
]
const ALL_AUXILIARY_VARIABLE_INITIAL_VALUES = Float64[
    1.0000004457928229,
    1.0,
    0.3333333333333333,
    0.9999999834891536,
    0.9999999834891536,
    1.0000001320867793,
    0.9999999834891535,
    0.9999999834891536,
    0.9999999504674777,
    0.9999999834891536,
    0.3333333333333333,
    0.3333333333333333,
    0.0,
    0.0,
    0.9999999834891536,
    0.3333333333333333,
    0.9999999504674777,
    0.0,
    0.9999999834891536,
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333333,
    0.0,
    0.3333333333333333,
    0.0,
    0.3333333333333333,
    0.3333333333333333,
    0.0,
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
    "➕₁₂",
    "➕₁₃",
    "➕₁₄",
    "➕₁₅",
    "➕₁₆",
    "➕₁₇",
    "➕₁₈",
    "➕₁₉",
    "➕₂₀",
    "➕₂₁",
    "➕₂₂",
    "➕₂₃",
    "➕₂₄",
    "➕₂₅",
    "➕₂₆",
    "➕₂₇",
    "➕₂₈",
    "➕₂₉",
]
const ALL_AUXILIARY_BOX_LOWER_BOUNDS = Float64[
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
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -1.0e12,
    2.220446049250313e-16,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -1.0e12,
]
const ALL_AUXILIARY_BOX_UPPER_BOUNDS = Float64[
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
    600.0,
    600.0,
    1.0e12,
    1.0e12,
    1.0e12,
    600.0,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    600.0,
    1.0e12,
    600.0,
    1.0e12,
    1.0e12,
    600.0,
]

const BLOCKS = [
    (
        index = 1,
        solve_order = 26,
        variables = ["Y_bar"],
        previous_solution_names = ["y", "➕₆"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [14],
        equations = Expr[
            :(➕₆ - y / Y_bar),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["Y_bar"],
        previous_solution_values = [0.3333333333333333, 1.0000001320867793],
        external_solution_values = Float64[],
        solution_values = [0.3333332893044127],
        previous_solution_initial_values = [0.3333333333333333, 1.0000001320867793],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.3333332893044127],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 2,
        solve_order = 25,
        variables = ["➕₆"],
        previous_solution_names = ["➕₅", "➕₇"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [15],
        equations = Expr[
            :(➕₇ - ➕₅ ^ phi_pi * ➕₆ ^ phi_y),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["➕₆"],
        previous_solution_values = [0.9999999834891536, 0.9999999834891534],
        external_solution_values = Float64[],
        solution_values = [1.0000001320867793],
        previous_solution_initial_values = [0.9999999834891536, 0.9999999834891535],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0000001320867793],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 3,
        solve_order = 24,
        variables = ["➕₇"],
        previous_solution_names = ["i", "v", "➕₄"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₉"],
        equation_indices = [16],
        equations = Expr[
            :(-(➕₄ ^ rho_i) * ➕₇ ^ (1 - rho_i) * exp(➕₂₉) + (i + 1) / (i_bar + 1)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₉ = min(600, max(-1.0e12, v))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₉ - v)),
        ],
        solution_names = ["➕₇", "➕₂₉"],
        previous_solution_values = [0.010100993423387527, 0.0, 0.9999999834891536],
        external_solution_values = Float64[],
        solution_values = [0.9999999834891534, 0.0],
        previous_solution_initial_values = [0.010100993423387527, 0.0, 0.9999999834891536],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.9999999834891535, 0.0],
        box_lower_bounds = [2.220446049250313e-16, -1.0e12],
        box_upper_bounds = [1.0e12, 600.0],
    ),
    (
        index = 4,
        solve_order = 23,
        variables = ["➕₅"],
        previous_solution_names = ["pi"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [13],
        equations = Expr[
            :(➕₅ - pi / Pi_bar),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["➕₅"],
        previous_solution_values = [0.9999999834891536],
        external_solution_values = Float64[],
        solution_values = [0.9999999834891536],
        previous_solution_initial_values = [0.9999999834891536],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.9999999834891536],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 5,
        solve_order = 22,
        variables = ["➕₄"],
        previous_solution_names = ["i"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [12],
        equations = Expr[
            :(➕₄ - (i + 1) / (i_bar + 1)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["➕₄"],
        previous_solution_values = [0.010100993423387527],
        external_solution_values = Float64[],
        solution_values = [0.9999999834891536],
        previous_solution_initial_values = [0.010100993423387527],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.9999999834891536],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 6,
        solve_order = 21,
        variables = ["v"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [20],
        equations = Expr[
            :(-rho_v * v + v),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["v"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 7,
        solve_order = 20,
        variables = ["real_interest"],
        previous_solution_names = ["i", "pi"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [18],
        equations = Expr[
            :(real_interest - (i + 1) / pi),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["real_interest"],
        previous_solution_values = [0.010100993423387527, 0.9999999834891536],
        external_solution_values = Float64[],
        solution_values = [1.0101010101010102],
        previous_solution_initial_values = [0.010100993423387527, 0.9999999834891536],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0101010101010102],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 8,
        solve_order = 19,
        variables = ["price_adjustment_gap"],
        previous_solution_names = ["p_star"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [26],
        equations = Expr[
            :(price_adjustment_gap - 1 / p_star),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["price_adjustment_gap"],
        previous_solution_values = [0.9999999504674765],
        external_solution_values = Float64[],
        solution_values = [1.000000049532526],
        previous_solution_initial_values = [0.9999999504674777],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0000000495325247],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 9,
        solve_order = 18,
        variables = ["i"],
        previous_solution_names = ["pi", "y"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₈"],
        equation_indices = [1],
        equations = Expr[
            :((-beta * (i + 1)) / (pi * ➕₂₈ ^ sigma) + ➕₂₈ ^ -sigma),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₈ = min(1.0e12, max(eps(), y))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₈ - y)),
        ],
        solution_names = ["i", "➕₂₈"],
        previous_solution_values = [0.9999999834891536, 0.3333333333333333],
        external_solution_values = Float64[],
        solution_values = [0.010100993423387527, 0.3333333333333333],
        previous_solution_initial_values = [0.9999999834891536, 0.3333333333333333],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.010100993423387527, 0.3333333333333333],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 10,
        solve_order = 17,
        variables = ["Utility"],
        previous_solution_names = ["N", "d_n", "y", "zeta"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₅", "➕₂₆", "➕₂₇"],
        equation_indices = [19],
        equations = Expr[
            :((((➕₂₅ ^ (phi_par + 1) * d_n * exp(➕₂₆)) / (phi_par + 1) - Utility * beta) + Utility) - log(➕₂₇)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₅ = min(1.0e12, max(eps(), N))),
            :(➕₂₆ = min(600, max(-1.0e12, zeta))),
            :(➕₂₇ = min(1.0e12, max(eps(), y))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₅ - N)),
            :(abs(➕₂₆ - zeta)),
            :(abs(➕₂₇ - y)),
        ],
        solution_names = ["Utility", "➕₂₅", "➕₂₆", "➕₂₇"],
        previous_solution_values = [0.3333333333333329, 8.099999984418782, 0.3333333333333333, 0.0],
        external_solution_values = Float64[],
        solution_values = [-154.86122878024847, 0.3333333333333329, 0.0, 0.3333333333333333],
        previous_solution_initial_values = [0.3333333333333333, 8.099999984418782, 0.3333333333333333, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-154.86122878024847, 0.3333333333333333, 0.0, 0.3333333333333333],
        box_lower_bounds = [-Inf, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12, 600.0, 1.0e12],
    ),
    (
        index = 11,
        solve_order = 16,
        variables = ["d_n"],
        previous_solution_names = ["N", "w", "y", "zeta"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₂", "➕₂₃", "➕₂₄"],
        equation_indices = [2],
        equations = Expr[
            :(-(➕₂₂ ^ phi_par) * d_n * ➕₂₃ ^ sigma * exp(➕₂₄) + w),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₂ = min(1.0e12, max(eps(), N))),
            :(➕₂₃ = min(1.0e12, max(eps(), y))),
            :(➕₂₄ = min(600, max(-1.0e12, zeta))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₂ - N)),
            :(abs(➕₂₃ - y)),
            :(abs(➕₂₄ - zeta)),
        ],
        solution_names = ["d_n", "➕₂₂", "➕₂₃", "➕₂₄"],
        previous_solution_values = [0.3333333333333329, 0.8999999982687535, 0.3333333333333333, 0.0],
        external_solution_values = Float64[],
        solution_values = [8.099999984418782, 0.3333333333333329, 0.3333333333333333, 0.0],
        previous_solution_initial_values = [0.3333333333333333, 0.8999999982687535, 0.3333333333333333, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [8.099999984418782, 0.3333333333333333, 0.3333333333333333, 0.0],
        box_lower_bounds = [-Inf, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12],
        box_upper_bounds = [Inf, 1.0e12, 1.0e12, 600.0],
    ),
    (
        index = 12,
        solve_order = 15,
        variables = ["zeta"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [22],
        equations = Expr[
            :(-rho_zeta * zeta + zeta),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["zeta"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 13,
        solve_order = 14,
        variables = ["Marginal_markup"],
        previous_solution_names = ["MC_real", "p_star"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [25],
        equations = Expr[
            :(Marginal_markup - p_star / MC_real),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["Marginal_markup"],
        previous_solution_values = [0.8999999982687535, 0.9999999504674765],
        external_solution_values = Float64[],
        solution_values = [1.1111110582123152],
        previous_solution_initial_values = [0.8999999982687535, 0.9999999504674777],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.1111110582123165],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 14,
        solve_order = 13,
        variables = ["Average_markup"],
        previous_solution_names = ["MC_real"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [24],
        equations = Expr[
            :(Average_markup - 1 / MC_real),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["Average_markup"],
        previous_solution_values = [0.8999999982687535],
        external_solution_values = Float64[],
        solution_values = [1.1111111132484524],
        previous_solution_initial_values = [0.8999999982687535],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.1111111132484524],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 15,
        solve_order = 12,
        variables = ["MC_real"],
        previous_solution_names = ["w", "y", "➕₂"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₁"],
        equation_indices = [17],
        equations = Expr[
            :(MC_real - (w * ➕₂₁ ^ (alpha / (1 - alpha)) * ➕₂ ^ (1 / (alpha - 1))) / (1 - alpha)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₁ = min(1.0e12, max(eps(), y))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₁ - y)),
        ],
        solution_names = ["MC_real", "➕₂₁"],
        previous_solution_values = [0.8999999982687535, 0.3333333333333333, 1.0],
        external_solution_values = Float64[],
        solution_values = [0.8999999982687535, 0.3333333333333333],
        previous_solution_initial_values = [0.8999999982687535, 0.3333333333333333, 1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.8999999982687535, 0.3333333333333333],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 16,
        solve_order = 11,
        variables = ["w"],
        previous_solution_names = ["pi", "psi", "y", "➕₂"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₉", "➕₂₀"],
        equation_indices = [7],
        equations = Expr[
            :(((-beta * ➕₁₉ ^ (epsilon / (1 - alpha)) * psi * theta) / ➕₁₉ ^ ((epsilon * var_rho) / (1 - alpha)) + psi) - (w * ➕₂₀ ^ (-sigma + 1 / (1 - alpha))) / ➕₂ ^ (1 / (1 - alpha))),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₉ = min(1.0e12, max(eps(), pi))),
            :(➕₂₀ = min(1.0e12, max(eps(), y))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₉ - pi)),
            :(abs(➕₂₀ - y)),
        ],
        solution_names = ["w", "➕₁₉", "➕₂₀"],
        previous_solution_values = [0.9999999834891536, 3.4951439603436207, 0.3333333333333333, 1.0],
        external_solution_values = Float64[],
        solution_values = [0.8999999982687535, 0.9999999834891536, 0.3333333333333333],
        previous_solution_initial_values = [0.9999999834891536, 3.4951439603436207, 0.3333333333333333, 1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.8999999982687535, 0.9999999834891536, 0.3333333333333333],
        box_lower_bounds = [-1.0e12, 2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 17,
        solve_order = 10,
        variables = ["➕₂"],
        previous_solution_names = ["A"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₈"],
        equation_indices = [6],
        equations = Expr[
            :(➕₂ - exp(➕₁₈)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₈ = min(600, max(-1.0e12, A))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₈ - A)),
        ],
        solution_names = ["➕₂", "➕₁₈"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [1.0, 0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0, 0.0],
        box_lower_bounds = [2.220446049250313e-16, -1.0e12],
        box_upper_bounds = [1.0e12, 600.0],
    ),
    (
        index = 18,
        solve_order = 9,
        variables = ["psi"],
        previous_solution_names = ["p_star", "phi"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₇"],
        equation_indices = [5],
        equations = Expr[
            :((-epsilon * psi) / (phi * (1 - alpha) * (epsilon - 1)) + ➕₁₇ ^ ((alpha * epsilon) / (1 - alpha) + 1)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₇ = min(1.0e12, max(eps(), p_star))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₇ - p_star)),
        ],
        solution_names = ["psi", "➕₁₇"],
        previous_solution_values = [0.9999999504674765, 3.8834934816299174],
        external_solution_values = Float64[],
        solution_values = [3.4951439603436207, 0.9999999504674765],
        previous_solution_initial_values = [0.9999999504674777, 3.8834934816299174],
        external_solution_initial_values = Float64[],
        solution_initial_values = [3.4951439603436207, 0.9999999504674777],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 19,
        solve_order = 8,
        variables = ["phi"],
        previous_solution_names = ["pi", "y"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₅", "➕₁₆"],
        equation_indices = [8],
        equations = Expr[
            :((-beta * phi * ➕₁₅ ^ (var_rho * (1 - epsilon)) * ➕₁₅ ^ (epsilon - 1) * theta + phi) - ➕₁₆ ^ (1 - sigma)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₅ = min(1.0e12, max(eps(), pi))),
            :(➕₁₆ = min(1.0e12, max(eps(), y))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₅ - pi)),
            :(abs(➕₁₆ - y)),
        ],
        solution_names = ["phi", "➕₁₅", "➕₁₆"],
        previous_solution_values = [0.9999999834891536, 0.3333333333333333],
        external_solution_values = Float64[],
        solution_values = [3.8834934816299174, 0.9999999834891536, 0.3333333333333333],
        previous_solution_initial_values = [0.9999999834891536, 0.3333333333333333],
        external_solution_initial_values = Float64[],
        solution_initial_values = [3.8834934816299174, 0.9999999834891536, 0.3333333333333333],
        box_lower_bounds = [-Inf, 2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12, 1.0e12],
    ),
    (
        index = 20,
        solve_order = 7,
        variables = ["p_star", "pi", "➕₁"],
        previous_solution_names = ["s"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [11, 3, 4],
        equations = Expr[
            :(((-(pi ^ (epsilon / (1 - alpha))) * s * theta) / pi ^ ((epsilon * var_rho) / (1 - alpha)) + s) - (1 - theta) / p_star ^ (epsilon / (1 - alpha))),
            :(➕₁ - (-(pi ^ (var_rho * (1 - epsilon))) * pi ^ (epsilon - 1) * theta + 1) / (1 - theta)),
            :(p_star - ➕₁ ^ (1 / (1 - epsilon))),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["p_star", "pi", "➕₁"],
        previous_solution_values = [0.9999999999999989],
        external_solution_values = Float64[],
        solution_values = [0.9999999504674765, 0.9999999834891536, 1.0000004457928222],
        previous_solution_initial_values = [0.9999999999999989],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.9999999504674777, 0.9999999834891536, 1.0000004457928229],
        box_lower_bounds = [2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 21,
        solve_order = 6,
        variables = ["A_tilde"],
        previous_solution_names = ["A", "s"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₄"],
        equation_indices = [23],
        equations = Expr[
            :(A_tilde - exp(➕₁₄) / s),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₄ = min(600, max(-1.0e12, A))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₄ - A)),
        ],
        solution_names = ["A_tilde", "➕₁₄"],
        previous_solution_values = [0.0, 0.9999999999999989],
        external_solution_values = Float64[],
        solution_values = [1.000000000000001, 0.0],
        previous_solution_initial_values = [0.0, 0.9999999999999989],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.000000000000001, 0.0],
        box_lower_bounds = [-Inf, -1.0e12],
        box_upper_bounds = [Inf, 600.0],
    ),
    (
        index = 22,
        solve_order = 5,
        variables = ["s"],
        previous_solution_names = ["N", "➕₃"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [10],
        equations = Expr[
            :(N - s * ➕₃ ^ (1 / (1 - alpha))),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["s"],
        previous_solution_values = [0.3333333333333329, 0.3333333333333333],
        external_solution_values = Float64[],
        solution_values = [0.9999999999999989],
        previous_solution_initial_values = [0.3333333333333333, 0.3333333333333333],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.9999999999999989],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 23,
        solve_order = 4,
        variables = ["➕₃"],
        previous_solution_names = ["A", "y"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₃"],
        equation_indices = [9],
        equations = Expr[
            :(➕₃ - y * exp(➕₁₃)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₃ = min(600, max(-1.0e12, -A))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₃ - -A)),
        ],
        solution_names = ["➕₃", "➕₁₃"],
        previous_solution_values = [0.0, 0.3333333333333333],
        external_solution_values = Float64[],
        solution_values = [0.3333333333333333, -0.0],
        previous_solution_initial_values = [0.0, 0.3333333333333333],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.3333333333333333, 0.0],
        box_lower_bounds = [2.220446049250313e-16, -1.0e12],
        box_upper_bounds = [1.0e12, 600.0],
    ),
    (
        index = 24,
        solve_order = 3,
        variables = ["y"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₂"],
        equation_indices = [28],
        equations = Expr[
            :(➕₁₂ ^ (1 - alpha) - y),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₂ = min(1.0e12, max(eps(), 1 / 3))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₂ - 1 / 3)),
        ],
        solution_names = ["y", "➕₁₂"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.3333333333333333, 0.3333333333333333],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.3333333333333333, 0.3333333333333333],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 25,
        solve_order = 2,
        variables = ["N"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [27],
        equations = Expr[
            :(N - 1 / 3),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["N"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.3333333333333329],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.3333333333333333],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 26,
        solve_order = 1,
        variables = ["A"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [21],
        equations = Expr[
            :(-A * rho_a + A),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["A"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
]
const BLOCK_EQUATION_ORDER = [14, 15, 16, 13, 12, 20, 18, 26, 1, 19, 2, 22, 25, 24, 17, 7, 6, 5, 8, 11, 3, 4, 23, 10, 9, 28, 27, 21]
const BLOCK_SOLVE_ORDER = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    ["y", "➕₆"],
    ["➕₅", "➕₇"],
    ["i", "v", "➕₄"],
    ["pi"],
    ["i"],
    String[],
    ["i", "pi"],
    ["p_star"],
    ["pi", "y"],
    ["N", "d_n", "y", "zeta"],
    ["N", "w", "y", "zeta"],
    String[],
    ["MC_real", "p_star"],
    ["MC_real"],
    ["w", "y", "➕₂"],
    ["pi", "psi", "y", "➕₂"],
    ["A"],
    ["p_star", "phi"],
    ["pi", "y"],
    ["s"],
    ["A", "s"],
    ["N", "➕₃"],
    ["A", "y"],
    String[],
    String[],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    [0.3333333333333333, 1.0000001320867793],
    [0.9999999834891536, 0.9999999834891534],
    [0.010100993423387527, 0.0, 0.9999999834891536],
    [0.9999999834891536],
    [0.010100993423387527],
    Float64[],
    [0.010100993423387527, 0.9999999834891536],
    [0.9999999504674765],
    [0.9999999834891536, 0.3333333333333333],
    [0.3333333333333329, 8.099999984418782, 0.3333333333333333, 0.0],
    [0.3333333333333329, 0.8999999982687535, 0.3333333333333333, 0.0],
    Float64[],
    [0.8999999982687535, 0.9999999504674765],
    [0.8999999982687535],
    [0.8999999982687535, 0.3333333333333333, 1.0],
    [0.9999999834891536, 3.4951439603436207, 0.3333333333333333, 1.0],
    [0.0],
    [0.9999999504674765, 3.8834934816299174],
    [0.9999999834891536, 0.3333333333333333],
    [0.9999999999999989],
    [0.0, 0.9999999999999989],
    [0.3333333333333329, 0.3333333333333333],
    [0.0, 0.3333333333333333],
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
    ["Y_bar"],
    ["➕₆"],
    ["➕₇", "➕₂₉"],
    ["➕₅"],
    ["➕₄"],
    ["v"],
    ["real_interest"],
    ["price_adjustment_gap"],
    ["i", "➕₂₈"],
    ["Utility", "➕₂₅", "➕₂₆", "➕₂₇"],
    ["d_n", "➕₂₂", "➕₂₃", "➕₂₄"],
    ["zeta"],
    ["Marginal_markup"],
    ["Average_markup"],
    ["MC_real", "➕₂₁"],
    ["w", "➕₁₉", "➕₂₀"],
    ["➕₂", "➕₁₈"],
    ["psi", "➕₁₇"],
    ["phi", "➕₁₅", "➕₁₆"],
    ["p_star", "pi", "➕₁"],
    ["A_tilde", "➕₁₄"],
    ["s"],
    ["➕₃", "➕₁₃"],
    ["y", "➕₁₂"],
    ["N"],
    ["A"],
]
const BLOCK_SOLUTION_VALUES = [
    [0.3333332893044127],
    [1.0000001320867793],
    [0.9999999834891534, 0.0],
    [0.9999999834891536],
    [0.9999999834891536],
    [0.0],
    [1.0101010101010102],
    [1.000000049532526],
    [0.010100993423387527, 0.3333333333333333],
    [-154.86122878024847, 0.3333333333333329, 0.0, 0.3333333333333333],
    [8.099999984418782, 0.3333333333333329, 0.3333333333333333, 0.0],
    [0.0],
    [1.1111110582123152],
    [1.1111111132484524],
    [0.8999999982687535, 0.3333333333333333],
    [0.8999999982687535, 0.9999999834891536, 0.3333333333333333],
    [1.0, 0.0],
    [3.4951439603436207, 0.9999999504674765],
    [3.8834934816299174, 0.9999999834891536, 0.3333333333333333],
    [0.9999999504674765, 0.9999999834891536, 1.0000004457928222],
    [1.000000000000001, 0.0],
    [0.9999999999999989],
    [0.3333333333333333, -0.0],
    [0.3333333333333333, 0.3333333333333333],
    [0.3333333333333329],
    [0.0],
]
const BLOCK_PREVIOUS_SOLUTION_INITIAL_VALUES = [
    [0.3333333333333333, 1.0000001320867793],
    [0.9999999834891536, 0.9999999834891535],
    [0.010100993423387527, 0.0, 0.9999999834891536],
    [0.9999999834891536],
    [0.010100993423387527],
    Float64[],
    [0.010100993423387527, 0.9999999834891536],
    [0.9999999504674777],
    [0.9999999834891536, 0.3333333333333333],
    [0.3333333333333333, 8.099999984418782, 0.3333333333333333, 0.0],
    [0.3333333333333333, 0.8999999982687535, 0.3333333333333333, 0.0],
    Float64[],
    [0.8999999982687535, 0.9999999504674777],
    [0.8999999982687535],
    [0.8999999982687535, 0.3333333333333333, 1.0],
    [0.9999999834891536, 3.4951439603436207, 0.3333333333333333, 1.0],
    [0.0],
    [0.9999999504674777, 3.8834934816299174],
    [0.9999999834891536, 0.3333333333333333],
    [0.9999999999999989],
    [0.0, 0.9999999999999989],
    [0.3333333333333333, 0.3333333333333333],
    [0.0, 0.3333333333333333],
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
    [0.3333332893044127],
    [1.0000001320867793],
    [0.9999999834891535, 0.0],
    [0.9999999834891536],
    [0.9999999834891536],
    [0.0],
    [1.0101010101010102],
    [1.0000000495325247],
    [0.010100993423387527, 0.3333333333333333],
    [-154.86122878024847, 0.3333333333333333, 0.0, 0.3333333333333333],
    [8.099999984418782, 0.3333333333333333, 0.3333333333333333, 0.0],
    [0.0],
    [1.1111110582123165],
    [1.1111111132484524],
    [0.8999999982687535, 0.3333333333333333],
    [0.8999999982687535, 0.9999999834891536, 0.3333333333333333],
    [1.0, 0.0],
    [3.4951439603436207, 0.9999999504674777],
    [3.8834934816299174, 0.9999999834891536, 0.3333333333333333],
    [0.9999999504674777, 0.9999999834891536, 1.0000004457928229],
    [1.000000000000001, 0.0],
    [0.9999999999999989],
    [0.3333333333333333, 0.0],
    [0.3333333333333333, 0.3333333333333333],
    [0.3333333333333333],
    [0.0],
]

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

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        previous_solution[2] - previous_solution[1] / solution[1],
    ]
end

function residuals_block_2(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        previous_solution[2] - previous_solution[1] ^ complete_parameters[11] * solution[1] ^ complete_parameters[12],
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[3] ^ complete_parameters[13]) * solution[1] ^ (1 - complete_parameters[13]) * exp(solution[2]) + (previous_solution[1] + 1) / (complete_parameters[19] + 1),
        solution[2] - min(600, max(-1.0e12, previous_solution[2])),
    ]
end

function residuals_block_4(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1] / complete_parameters[18],
    ]
end

function residuals_block_5(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - (previous_solution[1] + 1) / (complete_parameters[19] + 1),
    ]
end

function residuals_block_6(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[7]) * solution[1] + solution[1],
    ]
end

function residuals_block_7(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - (previous_solution[1] + 1) / previous_solution[2],
    ]
end

function residuals_block_8(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 1 / previous_solution[1],
    ]
end

function residuals_block_9(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(complete_parameters[1]) * (solution[1] + 1)) / (previous_solution[1] * solution[2] ^ complete_parameters[6]) + solution[2] ^ -(complete_parameters[6]),
        solution[2] - min(1.0e12, max(eps(), previous_solution[2])),
    ]
end

function residuals_block_10(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 4
    @assert length(external_solution) == 0
    @assert length(solution) == 4
    complete_parameters = complete_parameter_values(parameters)
    return [
        (((solution[2] ^ (complete_parameters[10] + 1) * previous_solution[2] * exp(solution[3])) / (complete_parameters[10] + 1) - solution[1] * complete_parameters[1]) + solution[1]) - log(solution[4]),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
        solution[3] - min(600, max(-1.0e12, previous_solution[4])),
        solution[4] - min(1.0e12, max(eps(), previous_solution[3])),
    ]
end

function residuals_block_11(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 4
    @assert length(external_solution) == 0
    @assert length(solution) == 4
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[2] ^ complete_parameters[10]) * solution[1] * solution[3] ^ complete_parameters[6] * exp(solution[4]) + previous_solution[2],
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
        solution[3] - min(1.0e12, max(eps(), previous_solution[3])),
        solution[4] - min(600, max(-1.0e12, previous_solution[4])),
    ]
end

function residuals_block_12(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[9]) * solution[1] + solution[1],
    ]
end

function residuals_block_13(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[2] / previous_solution[1],
    ]
end

function residuals_block_14(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 1 / previous_solution[1],
    ]
end

function residuals_block_15(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - (previous_solution[1] * solution[2] ^ (complete_parameters[3] / (1 - complete_parameters[3])) * previous_solution[3] ^ (1 / (complete_parameters[3] - 1))) / (1 - complete_parameters[3]),
        solution[2] - min(1.0e12, max(eps(), previous_solution[2])),
    ]
end

function residuals_block_16(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 4
    @assert length(external_solution) == 0
    @assert length(solution) == 3
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((-(complete_parameters[1]) * solution[2] ^ (complete_parameters[5] / (1 - complete_parameters[3])) * previous_solution[2] * complete_parameters[4]) / solution[2] ^ ((complete_parameters[5] * complete_parameters[14]) / (1 - complete_parameters[3])) + previous_solution[2]) - (solution[1] * solution[3] ^ (-(complete_parameters[6]) + 1 / (1 - complete_parameters[3]))) / previous_solution[4] ^ (1 / (1 - complete_parameters[3])),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
        solution[3] - min(1.0e12, max(eps(), previous_solution[3])),
    ]
end

function residuals_block_17(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - exp(solution[2]),
        solution[2] - min(600, max(-1.0e12, previous_solution[1])),
    ]
end

function residuals_block_18(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(complete_parameters[5]) * solution[1]) / (previous_solution[2] * (1 - complete_parameters[3]) * (complete_parameters[5] - 1)) + solution[2] ^ ((complete_parameters[3] * complete_parameters[5]) / (1 - complete_parameters[3]) + 1),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_19(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 3
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(complete_parameters[1]) * solution[1] * solution[2] ^ (complete_parameters[14] * (1 - complete_parameters[5])) * solution[2] ^ (complete_parameters[5] - 1) * complete_parameters[4] + solution[1]) - solution[3] ^ (1 - complete_parameters[6]),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
        solution[3] - min(1.0e12, max(eps(), previous_solution[2])),
    ]
end

function residuals_block_20(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 3
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((-(solution[2] ^ (complete_parameters[5] / (1 - complete_parameters[3]))) * previous_solution[1] * complete_parameters[4]) / solution[2] ^ ((complete_parameters[5] * complete_parameters[14]) / (1 - complete_parameters[3])) + previous_solution[1]) - (1 - complete_parameters[4]) / solution[1] ^ (complete_parameters[5] / (1 - complete_parameters[3])),
        solution[3] - (-(solution[2] ^ (complete_parameters[14] * (1 - complete_parameters[5]))) * solution[2] ^ (complete_parameters[5] - 1) * complete_parameters[4] + 1) / (1 - complete_parameters[4]),
        solution[1] - solution[3] ^ (1 / (1 - complete_parameters[5])),
    ]
end

function residuals_block_21(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - exp(solution[2]) / previous_solution[2],
        solution[2] - min(600, max(-1.0e12, previous_solution[1])),
    ]
end

function residuals_block_22(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        previous_solution[1] - solution[1] * previous_solution[2] ^ (1 / (1 - complete_parameters[3])),
    ]
end

function residuals_block_23(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[2] * exp(solution[2]),
        solution[2] - min(600, max(-1.0e12, -(previous_solution[1]))),
    ]
end

function residuals_block_24(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[2] ^ (1 - complete_parameters[3]) - solution[1],
        solution[2] - min(1.0e12, max(eps(), 1 / 3)),
    ]
end

function residuals_block_25(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 1 / 3,
    ]
end

function residuals_block_26(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) * complete_parameters[8] + solution[1],
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
        residuals_block_18(parameters, previous_solutions[18], external_solutions[18], solutions[18]),
        residuals_block_19(parameters, previous_solutions[19], external_solutions[19], solutions[19]),
        residuals_block_20(parameters, previous_solutions[20], external_solutions[20], solutions[20]),
        residuals_block_21(parameters, previous_solutions[21], external_solutions[21], solutions[21]),
        residuals_block_22(parameters, previous_solutions[22], external_solutions[22], solutions[22]),
        residuals_block_23(parameters, previous_solutions[23], external_solutions[23], solutions[23]),
        residuals_block_24(parameters, previous_solutions[24], external_solutions[24], solutions[24]),
        residuals_block_25(parameters, previous_solutions[25], external_solutions[25], solutions[25]),
        residuals_block_26(parameters, previous_solutions[26], external_solutions[26], solutions[26]),
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
export residuals_block_1, residuals_block_2, residuals_block_3, residuals_block_4, residuals_block_5, residuals_block_6, residuals_block_7, residuals_block_8, residuals_block_9, residuals_block_10, residuals_block_11, residuals_block_12, residuals_block_13, residuals_block_14, residuals_block_15, residuals_block_16, residuals_block_17, residuals_block_18, residuals_block_19, residuals_block_20, residuals_block_21, residuals_block_22, residuals_block_23, residuals_block_24, residuals_block_25, residuals_block_26
end
