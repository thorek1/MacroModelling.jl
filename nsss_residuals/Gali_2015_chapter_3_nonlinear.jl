module Gali_2015_chapter_3_nonlinearNsssResiduals
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

const MODEL_NAME = "Gali_2015_chapter_3_nonlinear"
const SOURCE_MODEL_FILE = "models/Gali_2015_chapter_3_nonlinear.jl"
const NSSS_SOLUTION_ERROR = 8.598191765838457e-16
const NSSS_RESIDUAL_NORM = 4.577566798522237e-16

const PARAMETER_NAMES = [
    "σ",
    "φ",
    "ϕᵖⁱ",
    "ϕʸ",
    "θ",
    "ρ_ν",
    "ρ_z",
    "ρ_a",
    "β",
    "η",
    "α",
    "ϵ",
    "τ",
    "std_a",
    "std_z",
    "std_nu",
]
const PARAMETER_VALUES = Float64[
    1.0,
    5.0,
    1.5,
    0.125,
    0.75,
    0.5,
    0.5,
    0.9,
    0.99,
    3.77,
    0.25,
    9.0,
    0.0,
    0.01,
    0.05,
    0.0025,
]
const COMPLETE_PARAMETER_NAMES = [
    "σ",
    "φ",
    "ϕᵖⁱ",
    "ϕʸ",
    "θ",
    "ρ_ν",
    "ρ_z",
    "ρ_a",
    "β",
    "η",
    "α",
    "ϵ",
    "τ",
    "std_a",
    "std_z",
    "std_nu",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    1.0,
    5.0,
    1.5,
    0.125,
    0.75,
    0.5,
    0.5,
    0.9,
    0.99,
    3.77,
    0.25,
    9.0,
    0.0,
    0.01,
    0.05,
    0.0025,
]
const ORIGINAL_SOLUTION_NAMES = [
    "A",
    "C",
    "MC",
    "M_real",
    "N",
    "Pi",
    "Pi_star",
    "Q",
    "R",
    "S",
    "W_real",
    "Y",
    "Z",
    "i_ann",
    "log_N",
    "log_W_real",
    "log_y",
    "nu",
    "pi_ann",
    "r_real_ann",
    "realinterest",
    "x_aux_1",
    "x_aux_2",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    1.0,
    0.9505798249541406,
    0.8888888888888886,
    0.915236383286892,
    0.934655265184067,
    0.9999999999999996,
    0.9999999999999987,
    0.9900000000000004,
    1.0101010101010095,
    1.0,
    0.6780252644037243,
    0.9505798249541406,
    1.0,
    0.04020134341400339,
    -0.06757751801802749,
    -0.38857072860365793,
    -0.050683138513520666,
    0.0,
    -1.776356839400251e-15,
    0.04020134341400514,
    1.01010101010101,
    3.4519956850053406,
    3.8834951456310276,
]
const ORIGINAL_INITIAL_SOLUTION_VALUES = Float64[
    1.0,
    0.9505798249541406,
    0.8888888888888886,
    0.915236383286892,
    0.934655265184067,
    0.9999999999999996,
    0.9999999999999987,
    0.9900000000000004,
    1.0101010101010095,
    1.0,
    0.6780252644037243,
    0.9505798249541406,
    1.0,
    0.04020134341400339,
    -0.06757751801802749,
    -0.38857072860365793,
    -0.050683138513520666,
    0.0,
    -1.776356839400251e-15,
    0.04020134341400514,
    1.01010101010101,
    3.4519956850053406,
    3.8834951456310276,
]
const AUXILIARY_SOLUTION_NAMES = [
    "A",
    "C",
    "MC",
    "M_real",
    "N",
    "Pi",
    "Pi_star",
    "Q",
    "R",
    "S",
    "W_real",
    "Y",
    "Z",
    "i_ann",
    "log_N",
    "log_W_real",
    "log_y",
    "nu",
    "pi_ann",
    "r_real_ann",
    "realinterest",
    "x_aux_1",
    "x_aux_2",
    "➕₁",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    1.0,
    0.9505798249541406,
    0.8888888888888886,
    0.9152363832868913,
    0.934655265184067,
    0.9999999999999996,
    0.9999999999999987,
    0.9900000000000004,
    1.0101010101010097,
    1.0,
    0.6780252644037243,
    0.9505798249541406,
    1.0,
    0.04020134341400426,
    -0.06757751801802749,
    -0.38857072860365793,
    -0.050683138513520666,
    0.0,
    -1.776356839400251e-15,
    0.04020134341400514,
    1.01010101010101,
    3.4519956850053406,
    3.8834951456310276,
    0.934655265184067,
]
const AUXILIARY_INITIAL_SOLUTION_VALUES = Float64[
    1.0,
    0.9505798249541406,
    0.8888888888888886,
    0.915236383286892,
    0.934655265184067,
    0.9999999999999996,
    0.9999999999999987,
    0.9900000000000004,
    1.0101010101010095,
    1.0,
    0.6780252644037243,
    0.9505798249541406,
    1.0,
    0.04020134341400339,
    -0.06757751801802749,
    -0.38857072860365793,
    -0.050683138513520666,
    0.0,
    -1.776356839400251e-15,
    0.04020134341400514,
    1.01010101010101,
    3.4519956850053406,
    3.8834951456310276,
    0.934655265184067,
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
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    0.934655265184067,
    0.9999999999999996,
    1.0000000000000107,
    0.9999999999999987,
    0.9999999999999996,
    0.9999999999999987,
    1.0101010101010095,
    1.01010101010101,
    0.9505798249541406,
    0.934655265184067,
    0.6780252644037243,
    0.0,
    0.9999999999999996,
    0.9999999999999996,
    0.9999999999999987,
    0.9999999999999996,
    0.9999999999999987,
    1.0101010101010097,
    1.0101010101010097,
    0.934655265184067,
    0.6780252644037243,
    0.9505798249541406,
    0.9999999999999996,
    1.01010101010101,
]
const ALL_AUXILIARY_VARIABLE_INITIAL_VALUES = Float64[
    0.934655265184067,
    0.9999999999999996,
    1.0000000000000107,
    0.9999999999999987,
    0.9999999999999996,
    0.9999999999999987,
    1.0101010101010095,
    1.01010101010101,
    0.9505798249541406,
    0.934655265184067,
    0.6780252644037243,
    0.0,
    0.9999999999999996,
    0.9999999999999996,
    0.9999999999999987,
    0.9999999999999996,
    0.9999999999999987,
    1.0101010101010095,
    1.0101010101010095,
    0.934655265184067,
    0.6780252644037243,
    0.9505798249541406,
    0.9999999999999996,
    1.01010101010101,
]
const DEFAULTED_NSSS_SOLUTION_NAMES = [
]
const CALIBRATION_PARAMETER_NAMES = [
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(W_real - C ^ σ * N ^ φ),
    :(Q - ((β * (C / C) ^ -σ * Z) / Z) / Pi),
    :(R - 1 / Q),
    :(Y - A * (N / S) ^ (1 - α)),
    :(R - Pi * realinterest),
    :(R - (1 / β) * Pi ^ ϕᵖⁱ * (Y / Y) ^ ϕʸ * exp(nu)),
    :(C - Y),
    :(log(A) - (ρ_a * log(A) + std_a * 0)),
    :(log(Z) - (ρ_z * log(Z) - std_z * 0)),
    :(nu - (ρ_ν * nu + std_nu * 0)),
    :(MC - W_real / ((S * Y * (1 - α)) / N)),
    :(1 - (θ * Pi ^ (ϵ - 1) + (1 - θ) * Pi_star ^ (1 - ϵ))),
    :(S - ((1 - θ) * Pi_star ^ (-ϵ / (1 - α)) + θ * Pi ^ (ϵ / (1 - α)) * S)),
    :(Pi_star ^ (1 + (ϵ * α) / (1 - α)) - (((ϵ * x_aux_1) / x_aux_2) * (1 - τ)) / (ϵ - 1)),
    :(x_aux_1 - (MC * Y * Z * C ^ -σ + β * θ * Pi ^ (ϵ + (α * ϵ) / (1 - α)) * x_aux_1)),
    :(x_aux_2 - (Y * Z * C ^ -σ + β * θ * Pi ^ (ϵ - 1) * x_aux_2)),
    :(log_y - log(Y)),
    :(log_W_real - log(W_real)),
    :(log_N - log(N)),
    :(pi_ann - log(Pi) * 4),
    :(i_ann - log(R) * 4),
    :(r_real_ann - log(realinterest) * 4),
    :(M_real - Y / R ^ η),
]
const CALIBRATION_EQUATIONS = Expr[
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :(-(C ^ σ) * N ^ φ + W_real),
    :(Q - β / Pi),
    :(R - 1 / Q),
    :(➕₁ - N / S),
    :(-A * ➕₁ ^ (1 - α) + Y),
    :(-Pi * realinterest + R),
    :((-(Pi ^ ϕᵖⁱ) * exp(nu)) / β + R),
    :(C - Y),
    :(-ρ_a * log(A) + log(A)),
    :(-ρ_z * log(Z) + log(Z)),
    :(-nu * ρ_ν + nu),
    :(MC - (N * W_real) / (S * Y * (1 - α))),
    :((-(Pi ^ (ϵ - 1)) * θ - Pi_star ^ (1 - ϵ) * (1 - θ)) + 1),
    :((-(Pi ^ (ϵ / (1 - α))) * S * θ + S) - (1 - θ) / Pi_star ^ (ϵ / (1 - α))),
    :(Pi_star ^ ((α * ϵ) / (1 - α) + 1) - (x_aux_1 * ϵ * (1 - τ)) / (x_aux_2 * (ϵ - 1))),
    :((-(Pi ^ ((α * ϵ) / (1 - α) + ϵ)) * x_aux_1 * β * θ + x_aux_1) - (MC * Y * Z) / C ^ σ),
    :((-(Pi ^ (ϵ - 1)) * x_aux_2 * β * θ + x_aux_2) - (Y * Z) / C ^ σ),
    :(log_y - log(Y)),
    :(log_W_real - log(W_real)),
    :(log_N - log(N)),
    :(pi_ann - 4 * log(Pi)),
    :(i_ann - 4 * log(R)),
    :(r_real_ann - 4 * log(realinterest)),
    :(M_real - Y / R ^ η),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(W_real - C ^ σ * N ^ φ),
    :(Q - ((β * (C / C) ^ -σ * Z) / Z) / Pi),
    :(R - 1 / Q),
    :(Y - A * (N / S) ^ (1 - α)),
    :(R - Pi * realinterest),
    :(R - (1 / β) * Pi ^ ϕᵖⁱ * (Y / Y) ^ ϕʸ * exp(nu)),
    :(C - Y),
    :(log(A) - (ρ_a * log(A) + std_a * 0)),
    :(log(Z) - (ρ_z * log(Z) - std_z * 0)),
    :(nu - (ρ_ν * nu + std_nu * 0)),
    :(MC - W_real / ((S * Y * (1 - α)) / N)),
    :(1 - (θ * Pi ^ (ϵ - 1) + (1 - θ) * Pi_star ^ (1 - ϵ))),
    :(S - ((1 - θ) * Pi_star ^ (-ϵ / (1 - α)) + θ * Pi ^ (ϵ / (1 - α)) * S)),
    :(Pi_star ^ (1 + (ϵ * α) / (1 - α)) - (((ϵ * x_aux_1) / x_aux_2) * (1 - τ)) / (ϵ - 1)),
    :(x_aux_1 - (MC * Y * Z * C ^ -σ + β * θ * Pi ^ (ϵ + (α * ϵ) / (1 - α)) * x_aux_1)),
    :(x_aux_2 - (Y * Z * C ^ -σ + β * θ * Pi ^ (ϵ - 1) * x_aux_2)),
    :(log_y - log(Y)),
    :(log_W_real - log(W_real)),
    :(log_N - log(N)),
    :(pi_ann - log(Pi) * 4),
    :(i_ann - log(R) * 4),
    :(r_real_ann - log(realinterest) * 4),
    :(M_real - Y / R ^ η),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :(-(C ^ σ) * N ^ φ + W_real),
    :(Q - β / Pi),
    :(R - 1 / Q),
    :(➕₁ - N / S),
    :(-A * ➕₁ ^ (1 - α) + Y),
    :(-Pi * realinterest + R),
    :((-(Pi ^ ϕᵖⁱ) * exp(nu)) / β + R),
    :(C - Y),
    :(-ρ_a * log(A) + log(A)),
    :(-ρ_z * log(Z) + log(Z)),
    :(-nu * ρ_ν + nu),
    :(MC - (N * W_real) / (S * Y * (1 - α))),
    :((-(Pi ^ (ϵ - 1)) * θ - Pi_star ^ (1 - ϵ) * (1 - θ)) + 1),
    :((-(Pi ^ (ϵ / (1 - α))) * S * θ + S) - (1 - θ) / Pi_star ^ (ϵ / (1 - α))),
    :(Pi_star ^ ((α * ϵ) / (1 - α) + 1) - (x_aux_1 * ϵ * (1 - τ)) / (x_aux_2 * (ϵ - 1))),
    :((-(Pi ^ ((α * ϵ) / (1 - α) + ϵ)) * x_aux_1 * β * θ + x_aux_1) - (MC * Y * Z) / C ^ σ),
    :((-(Pi ^ (ϵ - 1)) * x_aux_2 * β * θ + x_aux_2) - (Y * Z) / C ^ σ),
    :(log_y - log(Y)),
    :(log_W_real - log(W_real)),
    :(log_N - log(N)),
    :(pi_ann - 4 * log(Pi)),
    :(i_ann - 4 * log(R)),
    :(r_real_ann - 4 * log(realinterest)),
    :(M_real - Y / R ^ η),
]

const PARAMETER_DEFINITION_NAMES = [
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "σ",
    "φ",
    "ϕᵖⁱ",
    "ϕʸ",
    "θ",
    "ρ_ν",
    "ρ_z",
    "ρ_a",
    "β",
    "η",
    "α",
    "ϵ",
    "τ",
    "std_a",
    "std_z",
    "std_nu",
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
]
const ORIGINAL_BOX_CONSTRAINT_NAMES = [
    "A",
    "C",
    "MC",
    "M_real",
    "N",
    "Pi",
    "Pi_star",
    "Q",
    "R",
    "S",
    "W_real",
    "Y",
    "Z",
    "i_ann",
    "log_N",
    "log_W_real",
    "log_y",
    "nu",
    "pi_ann",
    "r_real_ann",
    "realinterest",
    "x_aux_1",
    "x_aux_2",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    2.220446049250313e-16,
    -1.0e12,
    -Inf,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -Inf,
    -1.0e12,
    -1.0e12,
    -Inf,
    -1.0e12,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
]
const ORIGINAL_BOX_UPPER_BOUNDS = Float64[
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "A",
    "C",
    "MC",
    "M_real",
    "N",
    "Pi",
    "Pi_star",
    "Q",
    "R",
    "S",
    "W_real",
    "Y",
    "Z",
    "i_ann",
    "log_N",
    "log_W_real",
    "log_y",
    "nu",
    "pi_ann",
    "r_real_ann",
    "realinterest",
    "x_aux_1",
    "x_aux_2",
    "➕₁",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    2.220446049250313e-16,
    -1.0e12,
    -Inf,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -Inf,
    -1.0e12,
    -1.0e12,
    -Inf,
    -1.0e12,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
]
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
    1.0e12,
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
]

const BLOCKS = [
    (
        index = 1,
        solve_order = 15,
        variables = ["r_real_ann"],
        previous_solution_names = ["realinterest"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₄"],
        equation_indices = [23],
        equations = Expr[
            :(r_real_ann - log(➕₂₄) * 4),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₄ = min(1.0e12, max(eps(), realinterest))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₄ - realinterest)),
        ],
        solution_names = ["r_real_ann", "➕₂₄"],
        previous_solution_values = [1.01010101010101],
        external_solution_values = Float64[],
        solution_values = [0.04020134341400514, 1.01010101010101],
        previous_solution_initial_values = [1.01010101010101],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.04020134341400514, 1.01010101010101],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 2,
        solve_order = 14,
        variables = ["realinterest"],
        previous_solution_names = ["Pi", "R"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [6],
        equations = Expr[
            :(-Pi * realinterest + R),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["realinterest"],
        previous_solution_values = [0.9999999999999996, 1.0101010101010097],
        external_solution_values = Float64[],
        solution_values = [1.01010101010101],
        previous_solution_initial_values = [0.9999999999999996, 1.0101010101010095],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.01010101010101],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 3,
        solve_order = 13,
        variables = ["pi_ann"],
        previous_solution_names = ["Pi"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₃"],
        equation_indices = [21],
        equations = Expr[
            :(pi_ann - log(➕₂₃) * 4),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₃ = min(1.0e12, max(eps(), Pi))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₃ - Pi)),
        ],
        solution_names = ["pi_ann", "➕₂₃"],
        previous_solution_values = [0.9999999999999996],
        external_solution_values = Float64[],
        solution_values = [-1.776356839400251e-15, 0.9999999999999996],
        previous_solution_initial_values = [0.9999999999999996],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-1.776356839400251e-15, 0.9999999999999996],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 4,
        solve_order = 12,
        variables = ["log_y"],
        previous_solution_names = ["Y"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₂"],
        equation_indices = [18],
        equations = Expr[
            :(log_y - log(➕₂₂)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₂ = min(1.0e12, max(eps(), Y))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₂ - Y)),
        ],
        solution_names = ["log_y", "➕₂₂"],
        previous_solution_values = [0.9505798249541406],
        external_solution_values = Float64[],
        solution_values = [-0.050683138513520666, 0.9505798249541406],
        previous_solution_initial_values = [0.9505798249541406],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-0.050683138513520666, 0.9505798249541406],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 5,
        solve_order = 11,
        variables = ["log_W_real"],
        previous_solution_names = ["W_real"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₁"],
        equation_indices = [19],
        equations = Expr[
            :(log_W_real - log(➕₂₁)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₁ = min(1.0e12, max(eps(), W_real))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₁ - W_real)),
        ],
        solution_names = ["log_W_real", "➕₂₁"],
        previous_solution_values = [0.6780252644037243],
        external_solution_values = Float64[],
        solution_values = [-0.38857072860365793, 0.6780252644037243],
        previous_solution_initial_values = [0.6780252644037243],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-0.38857072860365793, 0.6780252644037243],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 6,
        solve_order = 10,
        variables = ["log_N"],
        previous_solution_names = ["N"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₀"],
        equation_indices = [20],
        equations = Expr[
            :(log_N - log(➕₂₀)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₀ = min(1.0e12, max(eps(), N))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₀ - N)),
        ],
        solution_names = ["log_N", "➕₂₀"],
        previous_solution_values = [0.934655265184067],
        external_solution_values = Float64[],
        solution_values = [-0.06757751801802749, 0.934655265184067],
        previous_solution_initial_values = [0.934655265184067],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-0.06757751801802749, 0.934655265184067],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 7,
        solve_order = 9,
        variables = ["i_ann"],
        previous_solution_names = ["R"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₉"],
        equation_indices = [22],
        equations = Expr[
            :(i_ann - log(➕₁₉) * 4),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₉ = min(1.0e12, max(eps(), R))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₉ - R)),
        ],
        solution_names = ["i_ann", "➕₁₉"],
        previous_solution_values = [1.0101010101010097],
        external_solution_values = Float64[],
        solution_values = [0.04020134341400426, 1.0101010101010097],
        previous_solution_initial_values = [1.0101010101010095],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.04020134341400339, 1.0101010101010095],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 8,
        solve_order = 8,
        variables = ["M_real"],
        previous_solution_names = ["R", "Y"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₈"],
        equation_indices = [24],
        equations = Expr[
            :(M_real - Y / ➕₁₈ ^ η),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₈ = min(1.0e12, max(eps(), R))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₈ - R)),
        ],
        solution_names = ["M_real", "➕₁₈"],
        previous_solution_values = [1.0101010101010097, 0.9505798249541406],
        external_solution_values = Float64[],
        solution_values = [0.9152363832868913, 1.0101010101010097],
        previous_solution_initial_values = [1.0101010101010095, 0.9505798249541406],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.915236383286892, 1.0101010101010095],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 9,
        solve_order = 7,
        variables = ["C", "MC", "N", "W_real", "Y", "x_aux_1", "x_aux_2", "➕₁"],
        previous_solution_names = ["A", "Pi", "Pi_star", "S", "Z"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₆", "➕₁₇"],
        equation_indices = [16, 12, 4, 1, 8, 15, 17, 5],
        equations = Expr[
            :((-(➕₁₆ ^ ((α * ϵ) / (1 - α) + ϵ)) * x_aux_1 * β * θ + x_aux_1) - (MC * Y * Z) / C ^ σ),
            :(MC - (N * W_real) / (S * Y * (1 - α))),
            :(➕₁ - N / S),
            :(-(C ^ σ) * N ^ φ + W_real),
            :(C - Y),
            :(➕₁₇ ^ ((α * ϵ) / (1 - α) + 1) - (x_aux_1 * ϵ * (1 - τ)) / (x_aux_2 * (ϵ - 1))),
            :((-(➕₁₆ ^ (ϵ - 1)) * x_aux_2 * β * θ + x_aux_2) - (Y * Z) / C ^ σ),
            :(-A * ➕₁ ^ (1 - α) + Y),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₆ = min(1.0e12, max(eps(), Pi))),
            :(➕₁₇ = min(1.0e12, max(eps(), Pi_star))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₆ - Pi)),
            :(abs(➕₁₇ - Pi_star)),
        ],
        solution_names = ["C", "MC", "N", "W_real", "Y", "x_aux_1", "x_aux_2", "➕₁", "➕₁₆", "➕₁₇"],
        previous_solution_values = [1.0, 0.9999999999999996, 0.9999999999999987, 1.0, 1.0],
        external_solution_values = Float64[],
        solution_values = [0.9505798249541406, 0.8888888888888886, 0.934655265184067, 0.6780252644037243, 0.9505798249541406, 3.4519956850053406, 3.8834951456310276, 0.934655265184067, 0.9999999999999996, 0.9999999999999987],
        previous_solution_initial_values = [1.0, 0.9999999999999996, 0.9999999999999987, 1.0, 1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.9505798249541406, 0.8888888888888886, 0.934655265184067, 0.6780252644037243, 0.9505798249541406, 3.4519956850053406, 3.8834951456310276, 0.934655265184067, 0.9999999999999996, 0.9999999999999987],
        box_lower_bounds = [2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 10,
        solve_order = 6,
        variables = ["Z"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [10],
        equations = Expr[
            :(-ρ_z * log(Z) + log(Z)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["Z"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 11,
        solve_order = 5,
        variables = ["S"],
        previous_solution_names = ["Pi", "Pi_star"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₄", "➕₁₅"],
        equation_indices = [14],
        equations = Expr[
            :((-(➕₁₄ ^ (ϵ / (1 - α))) * S * θ + S) - (1 - θ) / ➕₁₅ ^ (ϵ / (1 - α))),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₄ = min(1.0e12, max(eps(), Pi))),
            :(➕₁₅ = min(1.0e12, max(eps(), Pi_star))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₄ - Pi)),
            :(abs(➕₁₅ - Pi_star)),
        ],
        solution_names = ["S", "➕₁₄", "➕₁₅"],
        previous_solution_values = [0.9999999999999996, 0.9999999999999987],
        external_solution_values = Float64[],
        solution_values = [1.0, 0.9999999999999996, 0.9999999999999987],
        previous_solution_initial_values = [0.9999999999999996, 0.9999999999999987],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0, 0.9999999999999996, 0.9999999999999987],
        box_lower_bounds = [-Inf, 2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12, 1.0e12],
    ),
    (
        index = 12,
        solve_order = 4,
        variables = ["Pi_star"],
        previous_solution_names = ["Pi"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₃"],
        equation_indices = [13],
        equations = Expr[
            :((-(➕₁₃ ^ (ϵ - 1)) * θ - Pi_star ^ (1 - ϵ) * (1 - θ)) + 1),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₃ = min(1.0e12, max(eps(), Pi))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₃ - Pi)),
        ],
        solution_names = ["Pi_star", "➕₁₃"],
        previous_solution_values = [0.9999999999999996],
        external_solution_values = Float64[],
        solution_values = [0.9999999999999987, 0.9999999999999996],
        previous_solution_initial_values = [0.9999999999999996],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.9999999999999987, 0.9999999999999996],
        box_lower_bounds = [2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12],
    ),
    (
        index = 13,
        solve_order = 3,
        variables = ["Pi", "Q", "R"],
        previous_solution_names = ["nu"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₂"],
        equation_indices = [7, 2, 3],
        equations = Expr[
            :((-(Pi ^ ϕᵖⁱ) * exp(➕₁₂)) / β + R),
            :(Q - β / Pi),
            :(R - 1 / Q),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₂ = min(600, max(-1.0e12, nu))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₂ - nu)),
        ],
        solution_names = ["Pi", "Q", "R", "➕₁₂"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.9999999999999996, 0.9900000000000004, 1.0101010101010097, 0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.9999999999999996, 0.9900000000000004, 1.0101010101010095, 0.0],
        box_lower_bounds = [2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12, 600.0],
    ),
    (
        index = 14,
        solve_order = 2,
        variables = ["nu"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [11],
        equations = Expr[
            :(-nu * ρ_ν + nu),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["nu"],
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
        index = 15,
        solve_order = 1,
        variables = ["A"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [9],
        equations = Expr[
            :(-ρ_a * log(A) + log(A)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["A"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
]
const BLOCK_EQUATION_ORDER = [23, 6, 21, 18, 19, 20, 22, 24, 16, 12, 4, 1, 8, 15, 17, 5, 10, 14, 13, 7, 2, 3, 11, 9]
const BLOCK_SOLVE_ORDER = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    ["realinterest"],
    ["Pi", "R"],
    ["Pi"],
    ["Y"],
    ["W_real"],
    ["N"],
    ["R"],
    ["R", "Y"],
    ["A", "Pi", "Pi_star", "S", "Z"],
    String[],
    ["Pi", "Pi_star"],
    ["Pi"],
    ["nu"],
    String[],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    [1.01010101010101],
    [0.9999999999999996, 1.0101010101010097],
    [0.9999999999999996],
    [0.9505798249541406],
    [0.6780252644037243],
    [0.934655265184067],
    [1.0101010101010097],
    [1.0101010101010097, 0.9505798249541406],
    [1.0, 0.9999999999999996, 0.9999999999999987, 1.0, 1.0],
    Float64[],
    [0.9999999999999996, 0.9999999999999987],
    [0.9999999999999996],
    [0.0],
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
]
const BLOCK_SOLUTION_NAMES = [
    ["r_real_ann", "➕₂₄"],
    ["realinterest"],
    ["pi_ann", "➕₂₃"],
    ["log_y", "➕₂₂"],
    ["log_W_real", "➕₂₁"],
    ["log_N", "➕₂₀"],
    ["i_ann", "➕₁₉"],
    ["M_real", "➕₁₈"],
    ["C", "MC", "N", "W_real", "Y", "x_aux_1", "x_aux_2", "➕₁", "➕₁₆", "➕₁₇"],
    ["Z"],
    ["S", "➕₁₄", "➕₁₅"],
    ["Pi_star", "➕₁₃"],
    ["Pi", "Q", "R", "➕₁₂"],
    ["nu"],
    ["A"],
]
const BLOCK_SOLUTION_VALUES = [
    [0.04020134341400514, 1.01010101010101],
    [1.01010101010101],
    [-1.776356839400251e-15, 0.9999999999999996],
    [-0.050683138513520666, 0.9505798249541406],
    [-0.38857072860365793, 0.6780252644037243],
    [-0.06757751801802749, 0.934655265184067],
    [0.04020134341400426, 1.0101010101010097],
    [0.9152363832868913, 1.0101010101010097],
    [0.9505798249541406, 0.8888888888888886, 0.934655265184067, 0.6780252644037243, 0.9505798249541406, 3.4519956850053406, 3.8834951456310276, 0.934655265184067, 0.9999999999999996, 0.9999999999999987],
    [1.0],
    [1.0, 0.9999999999999996, 0.9999999999999987],
    [0.9999999999999987, 0.9999999999999996],
    [0.9999999999999996, 0.9900000000000004, 1.0101010101010097, 0.0],
    [0.0],
    [1.0],
]
const BLOCK_PREVIOUS_SOLUTION_INITIAL_VALUES = [
    [1.01010101010101],
    [0.9999999999999996, 1.0101010101010095],
    [0.9999999999999996],
    [0.9505798249541406],
    [0.6780252644037243],
    [0.934655265184067],
    [1.0101010101010095],
    [1.0101010101010095, 0.9505798249541406],
    [1.0, 0.9999999999999996, 0.9999999999999987, 1.0, 1.0],
    Float64[],
    [0.9999999999999996, 0.9999999999999987],
    [0.9999999999999996],
    [0.0],
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
]
const BLOCK_SOLUTION_INITIAL_VALUES = [
    [0.04020134341400514, 1.01010101010101],
    [1.01010101010101],
    [-1.776356839400251e-15, 0.9999999999999996],
    [-0.050683138513520666, 0.9505798249541406],
    [-0.38857072860365793, 0.6780252644037243],
    [-0.06757751801802749, 0.934655265184067],
    [0.04020134341400339, 1.0101010101010095],
    [0.915236383286892, 1.0101010101010095],
    [0.9505798249541406, 0.8888888888888886, 0.934655265184067, 0.6780252644037243, 0.9505798249541406, 3.4519956850053406, 3.8834951456310276, 0.934655265184067, 0.9999999999999996, 0.9999999999999987],
    [1.0],
    [1.0, 0.9999999999999996, 0.9999999999999987],
    [0.9999999999999987, 0.9999999999999996],
    [0.9999999999999996, 0.9900000000000004, 1.0101010101010095, 0.0],
    [0.0],
    [1.0],
]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[2] = parameters[2]
    complete_parameters[11] = parameters[11]
    complete_parameters[6] = parameters[6]
    complete_parameters[15] = parameters[15]
    complete_parameters[9] = parameters[9]
    complete_parameters[5] = parameters[5]
    complete_parameters[8] = parameters[8]
    complete_parameters[4] = parameters[4]
    complete_parameters[7] = parameters[7]
    complete_parameters[12] = parameters[12]
    complete_parameters[1] = parameters[1]
    complete_parameters[16] = parameters[16]
    complete_parameters[13] = parameters[13]
    complete_parameters[3] = parameters[3]
    complete_parameters[14] = parameters[14]
    complete_parameters[10] = parameters[10]
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[11] - solution[2] ^ complete_parameters[1] * solution[5] ^ complete_parameters[2],
        solution[8] - ((complete_parameters[9] * (solution[2] / solution[2]) ^ -(complete_parameters[1]) * solution[13]) / solution[13]) / solution[6],
        solution[9] - 1 / solution[8],
        solution[12] - solution[1] * (solution[5] / solution[10]) ^ (1 - complete_parameters[11]),
        solution[9] - solution[6] * solution[21],
        solution[9] - (1 / complete_parameters[9]) * solution[6] ^ complete_parameters[3] * (solution[12] / solution[12]) ^ complete_parameters[4] * exp(solution[18]),
        solution[2] - solution[12],
        log(solution[1]) - (complete_parameters[8] * log(solution[1]) + complete_parameters[14] * 0),
        log(solution[13]) - (complete_parameters[7] * log(solution[13]) - complete_parameters[15] * 0),
        solution[18] - (complete_parameters[6] * solution[18] + complete_parameters[16] * 0),
        solution[3] - solution[11] / ((solution[10] * solution[12] * (1 - complete_parameters[11])) / solution[5]),
        1 - (complete_parameters[5] * solution[6] ^ (complete_parameters[12] - 1) + (1 - complete_parameters[5]) * solution[7] ^ (1 - complete_parameters[12])),
        solution[10] - ((1 - complete_parameters[5]) * solution[7] ^ (-(complete_parameters[12]) / (1 - complete_parameters[11])) + complete_parameters[5] * solution[6] ^ (complete_parameters[12] / (1 - complete_parameters[11])) * solution[10]),
        solution[7] ^ (1 + (complete_parameters[12] * complete_parameters[11]) / (1 - complete_parameters[11])) - (((complete_parameters[12] * solution[22]) / solution[23]) * (1 - complete_parameters[13])) / (complete_parameters[12] - 1),
        solution[22] - (solution[3] * solution[12] * solution[13] * solution[2] ^ -(complete_parameters[1]) + complete_parameters[9] * complete_parameters[5] * solution[6] ^ (complete_parameters[12] + (complete_parameters[11] * complete_parameters[12]) / (1 - complete_parameters[11])) * solution[22]),
        solution[23] - (solution[12] * solution[13] * solution[2] ^ -(complete_parameters[1]) + complete_parameters[9] * complete_parameters[5] * solution[6] ^ (complete_parameters[12] - 1) * solution[23]),
        solution[17] - log(solution[12]),
        solution[16] - log(solution[11]),
        solution[15] - log(solution[5]),
        solution[19] - log(solution[6]) * 4,
        solution[14] - log(solution[9]) * 4,
        solution[20] - log(solution[21]) * 4,
        solution[4] - solution[12] / solution[9] ^ complete_parameters[10],
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[2] ^ complete_parameters[1]) * solution[5] ^ complete_parameters[2] + solution[11],
        solution[8] - complete_parameters[9] / solution[6],
        solution[9] - 1 / solution[8],
        solution[24] - solution[5] / solution[10],
        -(solution[1]) * solution[24] ^ (1 - complete_parameters[11]) + solution[12],
        -(solution[6]) * solution[21] + solution[9],
        (-(solution[6] ^ complete_parameters[3]) * exp(solution[18])) / complete_parameters[9] + solution[9],
        solution[2] - solution[12],
        -(complete_parameters[8]) * log(solution[1]) + log(solution[1]),
        -(complete_parameters[7]) * log(solution[13]) + log(solution[13]),
        -(solution[18]) * complete_parameters[6] + solution[18],
        solution[3] - (solution[5] * solution[11]) / (solution[10] * solution[12] * (1 - complete_parameters[11])),
        (-(solution[6] ^ (complete_parameters[12] - 1)) * complete_parameters[5] - solution[7] ^ (1 - complete_parameters[12]) * (1 - complete_parameters[5])) + 1,
        (-(solution[6] ^ (complete_parameters[12] / (1 - complete_parameters[11]))) * solution[10] * complete_parameters[5] + solution[10]) - (1 - complete_parameters[5]) / solution[7] ^ (complete_parameters[12] / (1 - complete_parameters[11])),
        solution[7] ^ ((complete_parameters[11] * complete_parameters[12]) / (1 - complete_parameters[11]) + 1) - (solution[22] * complete_parameters[12] * (1 - complete_parameters[13])) / (solution[23] * (complete_parameters[12] - 1)),
        (-(solution[6] ^ ((complete_parameters[11] * complete_parameters[12]) / (1 - complete_parameters[11]) + complete_parameters[12])) * solution[22] * complete_parameters[9] * complete_parameters[5] + solution[22]) - (solution[3] * solution[12] * solution[13]) / solution[2] ^ complete_parameters[1],
        (-(solution[6] ^ (complete_parameters[12] - 1)) * solution[23] * complete_parameters[9] * complete_parameters[5] + solution[23]) - (solution[12] * solution[13]) / solution[2] ^ complete_parameters[1],
        solution[17] - log(solution[12]),
        solution[16] - log(solution[11]),
        solution[15] - log(solution[5]),
        solution[19] - 4 * log(solution[6]),
        solution[14] - 4 * log(solution[9]),
        solution[20] - 4 * log(solution[21]),
        solution[4] - solution[12] / solution[9] ^ complete_parameters[10],
    ]
end

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - log(solution[2]) * 4,
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_2(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) * solution[1] + previous_solution[2],
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - log(solution[2]) * 4,
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_4(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
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

function residuals_block_5(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
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

function residuals_block_6(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
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

function residuals_block_7(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - log(solution[2]) * 4,
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_8(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[2] / solution[2] ^ complete_parameters[10],
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_9(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 5
    @assert length(external_solution) == 0
    @assert length(solution) == 10
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[9] ^ ((complete_parameters[11] * complete_parameters[12]) / (1 - complete_parameters[11]) + complete_parameters[12])) * solution[6] * complete_parameters[9] * complete_parameters[5] + solution[6]) - (solution[2] * solution[5] * previous_solution[5]) / solution[1] ^ complete_parameters[1],
        solution[2] - (solution[3] * solution[4]) / (previous_solution[4] * solution[5] * (1 - complete_parameters[11])),
        solution[8] - solution[3] / previous_solution[4],
        -(solution[1] ^ complete_parameters[1]) * solution[3] ^ complete_parameters[2] + solution[4],
        solution[1] - solution[5],
        solution[10] ^ ((complete_parameters[11] * complete_parameters[12]) / (1 - complete_parameters[11]) + 1) - (solution[6] * complete_parameters[12] * (1 - complete_parameters[13])) / (solution[7] * (complete_parameters[12] - 1)),
        (-(solution[9] ^ (complete_parameters[12] - 1)) * solution[7] * complete_parameters[9] * complete_parameters[5] + solution[7]) - (solution[5] * previous_solution[5]) / solution[1] ^ complete_parameters[1],
        -(previous_solution[1]) * solution[8] ^ (1 - complete_parameters[11]) + solution[5],
        solution[9] - min(1.0e12, max(eps(), previous_solution[2])),
        solution[10] - min(1.0e12, max(eps(), previous_solution[3])),
    ]
end

function residuals_block_10(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[7]) * log(solution[1]) + log(solution[1]),
    ]
end

function residuals_block_11(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 3
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[2] ^ (complete_parameters[12] / (1 - complete_parameters[11]))) * solution[1] * complete_parameters[5] + solution[1]) - (1 - complete_parameters[5]) / solution[3] ^ (complete_parameters[12] / (1 - complete_parameters[11])),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
        solution[3] - min(1.0e12, max(eps(), previous_solution[2])),
    ]
end

function residuals_block_12(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[2] ^ (complete_parameters[12] - 1)) * complete_parameters[5] - solution[1] ^ (1 - complete_parameters[12]) * (1 - complete_parameters[5])) + 1,
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_13(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 4
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[1] ^ complete_parameters[3]) * exp(solution[4])) / complete_parameters[9] + solution[3],
        solution[2] - complete_parameters[9] / solution[1],
        solution[3] - 1 / solution[2],
        solution[4] - min(600, max(-1.0e12, previous_solution[1])),
    ]
end

function residuals_block_14(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) * complete_parameters[6] + solution[1],
    ]
end

function residuals_block_15(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[8]) * log(solution[1]) + log(solution[1]),
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
export residuals_block_1, residuals_block_2, residuals_block_3, residuals_block_4, residuals_block_5, residuals_block_6, residuals_block_7, residuals_block_8, residuals_block_9, residuals_block_10, residuals_block_11, residuals_block_12, residuals_block_13, residuals_block_14, residuals_block_15
end
