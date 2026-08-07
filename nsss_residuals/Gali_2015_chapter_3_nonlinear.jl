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
const NSSS_SOLUTION_ERROR = 9.153786615810231e-16
const NSSS_RESIDUAL_NORM = 7.28021923224409e-16

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
    0.8888888888888883,
    0.9152363832868922,
    0.934655265184067,
    0.9999999999999996,
    0.9999999999999987,
    0.9900000000000004,
    1.0101010101010095,
    1.0,
    0.6780252644037243,
    0.9505798249541407,
    1.0,
    0.04020134341400339,
    -0.06757751801802749,
    -0.38857072860365793,
    -0.05068313851352055,
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
    0.9505798249541407,
    0.8888888888888885,
    0.9152363832868914,
    0.934655265184067,
    0.9999999999999996,
    0.9999999999999987,
    0.9900000000000004,
    1.0101010101010097,
    1.0,
    0.6780252644037243,
    0.9505798249541407,
    1.0,
    0.04020134341400426,
    -0.06757751801802749,
    -0.38857072860365793,
    -0.05068313851352055,
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
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    0.934655265184067,
    0.9999999999999996,
    1.0000000000000107,
    0.9999999999999987,
    0.9999999999999987,
    0.9999999999999996,
    0.9505798249541407,
    0.6780252644037243,
    1.01010101010101,
    0.934655265184067,
    1.0101010101010095,
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
]

const BLOCKS = [
    (
        index = 1,
        variables = ["r_real_ann"],
        equation_indices = [23],
        equations = Expr[
            :(r_real_ann - 4 * log(realinterest)),
        ],
    ),
    (
        index = 2,
        variables = ["realinterest"],
        equation_indices = [6],
        equations = Expr[
            :(-Pi * realinterest + R),
        ],
    ),
    (
        index = 3,
        variables = ["pi_ann"],
        equation_indices = [21],
        equations = Expr[
            :(pi_ann - 4 * log(Pi)),
        ],
    ),
    (
        index = 4,
        variables = ["log_y"],
        equation_indices = [18],
        equations = Expr[
            :(log_y - log(Y)),
        ],
    ),
    (
        index = 5,
        variables = ["log_W_real"],
        equation_indices = [19],
        equations = Expr[
            :(log_W_real - log(W_real)),
        ],
    ),
    (
        index = 6,
        variables = ["log_N"],
        equation_indices = [20],
        equations = Expr[
            :(log_N - log(N)),
        ],
    ),
    (
        index = 7,
        variables = ["i_ann"],
        equation_indices = [22],
        equations = Expr[
            :(i_ann - 4 * log(R)),
        ],
    ),
    (
        index = 8,
        variables = ["M_real"],
        equation_indices = [24],
        equations = Expr[
            :(M_real - Y / R ^ η),
        ],
    ),
    (
        index = 9,
        variables = ["C", "MC", "N", "W_real", "Y", "x_aux_1", "x_aux_2", "➕₁"],
        equation_indices = [16, 12, 4, 1, 8, 15, 17, 5],
        equations = Expr[
            :((-(Pi ^ ((α * ϵ) / (1 - α) + ϵ)) * x_aux_1 * β * θ + x_aux_1) - (MC * Y * Z) / C ^ σ),
            :(MC - (N * W_real) / (S * Y * (1 - α))),
            :(➕₁ - N / S),
            :(-(C ^ σ) * N ^ φ + W_real),
            :(C - Y),
            :(Pi_star ^ ((α * ϵ) / (1 - α) + 1) - (x_aux_1 * ϵ * (1 - τ)) / (x_aux_2 * (ϵ - 1))),
            :((-(Pi ^ (ϵ - 1)) * x_aux_2 * β * θ + x_aux_2) - (Y * Z) / C ^ σ),
            :(-A * ➕₁ ^ (1 - α) + Y),
        ],
    ),
    (
        index = 10,
        variables = ["Z"],
        equation_indices = [10],
        equations = Expr[
            :(-ρ_z * log(Z) + log(Z)),
        ],
    ),
    (
        index = 11,
        variables = ["S"],
        equation_indices = [14],
        equations = Expr[
            :((-(Pi ^ (ϵ / (1 - α))) * S * θ + S) - (1 - θ) / Pi_star ^ (ϵ / (1 - α))),
        ],
    ),
    (
        index = 12,
        variables = ["Pi_star"],
        equation_indices = [13],
        equations = Expr[
            :((-(Pi ^ (ϵ - 1)) * θ - Pi_star ^ (1 - ϵ) * (1 - θ)) + 1),
        ],
    ),
    (
        index = 13,
        variables = ["Pi", "Q", "R"],
        equation_indices = [7, 2, 3],
        equations = Expr[
            :((-(Pi ^ ϕᵖⁱ) * exp(nu)) / β + R),
            :(Q - β / Pi),
            :(R - 1 / Q),
        ],
    ),
    (
        index = 14,
        variables = ["nu"],
        equation_indices = [11],
        equations = Expr[
            :(-nu * ρ_ν + nu),
        ],
    ),
    (
        index = 15,
        variables = ["A"],
        equation_indices = [9],
        equations = Expr[
            :(-ρ_a * log(A) + log(A)),
        ],
    ),
]
const BLOCK_EQUATION_ORDER = [23, 6, 21, 18, 19, 20, 22, 24, 16, 12, 4, 1, 8, 15, 17, 5, 10, 14, 13, 7, 2, 3, 11, 9]

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
