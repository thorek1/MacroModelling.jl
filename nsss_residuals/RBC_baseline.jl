module RBC_baselineNsssResiduals
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

const MODEL_NAME = "RBC_baseline"
const SOURCE_MODEL_FILE = "models/RBC_baseline.jl"
const NSSS_SOLUTION_ERROR = 0.0
const NSSS_RESIDUAL_NORM = 4.476983626029388e-16

const PARAMETER_NAMES = [
    "σᶻ",
    "σᵍ",
    "σ",
    "i_y",
    "k_y",
    "ρᶻ",
    "ρᵍ",
    "g_y",
    "α",
]
const PARAMETER_VALUES = Float64[
    0.066,
    0.104,
    1.0,
    0.25,
    10.4,
    0.97,
    0.989,
    0.2038,
    0.3333333333333333,
]
const COMPLETE_PARAMETER_NAMES = [
    "σᶻ",
    "σᵍ",
    "σ",
    "i_y",
    "k_y",
    "ρᶻ",
    "ρᵍ",
    "g_y",
    "α",
    "β",
    "δ",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    0.066,
    0.104,
    1.0,
    0.25,
    10.4,
    0.97,
    0.989,
    0.2038,
    0.3333333333333333,
    0.9920508744038156,
    0.024038461538461536,
]
const ORIGINAL_SOLUTION_NAMES = [
    "c",
    "g",
    "i",
    "k",
    "l",
    "r",
    "w",
    "y",
    "z",
    "ḡ",
    "ψ",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    0.5871473576160889,
    0.21907841721376572,
    0.26874192494328436,
    11.17966407764063,
    0.3333333333333333,
    0.1282051282051284,
    2.1499353995462784,
    1.074967699773139,
    1.0,
    0.21907841721376572,
    2.4411082631514707,
]
const AUXILIARY_SOLUTION_NAMES = [
    "c",
    "g",
    "i",
    "k",
    "l",
    "r",
    "w",
    "y",
    "z",
    "➕₁",
    "ḡ",
    "ψ",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    0.5871473576160889,
    0.21907841721376572,
    0.26874192494328436,
    11.17966407764063,
    0.3333333333333333,
    0.1282051282051284,
    2.1499353995462784,
    1.074967699773139,
    1.0,
    33.53899223292189,
    0.21907841721376572,
    2.4411082631514707,
]
const ALL_AUXILIARY_VARIABLE_NAMES = [
    "➕₁",
    "➕₂",
    "➕₃",
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    33.53899223292189,
    11.17966407764063,
    0.5871473576160889,
]
const DEFAULTED_NSSS_SOLUTION_NAMES = [
]
const CALIBRATION_PARAMETER_NAMES = [
    "ḡ",
    "ψ",
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(c ^ -σ - β * c ^ -σ * ((α * z * (k / l) ^ (α - 1) + 1) - δ)),
    :((ψ * c ^ σ) / (1 - l) - w),
    :(k - ((1 - δ) * k + i)),
    :(y - (c + i + g)),
    :(y - z * k ^ α * l ^ (1 - α)),
    :(w - (y * (1 - α)) / l),
    :(r - (y * α * 4) / k),
    :(z - ((1 - ρᶻ) + ρᶻ * z + σᶻ * 0)),
    :(g - ((1 - ρᵍ) * ḡ + ρᵍ * g + σᵍ * 0)),
]
const CALIBRATION_EQUATIONS = Expr[
    :(ḡ - g_y * y),
    :(l - 1 / 3),
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :(➕₁ - k / l),
    :((-β * ((z * α * ➕₁ ^ (α - 1) - δ) + 1)) / c ^ σ + c ^ -σ),
    :((c ^ σ * ψ) / (1 - l) - w),
    :((-i - k * (1 - δ)) + k),
    :(((-c - g) - i) + y),
    :(-(k ^ α) * l ^ (1 - α) * z + y),
    :(w - (y * (1 - α)) / l),
    :(r - (4 * y * α) / k),
    :((-z * ρᶻ + z + ρᶻ) - 1),
    :((-g * ρᵍ + g) - ḡ * (1 - ρᵍ)),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(c ^ -σ - β * c ^ -σ * ((α * z * (k / l) ^ (α - 1) + 1) - δ)),
    :((ψ * c ^ σ) / (1 - l) - w),
    :(k - ((1 - δ) * k + i)),
    :(y - (c + i + g)),
    :(y - z * k ^ α * l ^ (1 - α)),
    :(w - (y * (1 - α)) / l),
    :(r - (y * α * 4) / k),
    :(z - ((1 - ρᶻ) + ρᶻ * z + σᶻ * 0)),
    :(g - ((1 - ρᵍ) * ḡ + ρᵍ * g + σᵍ * 0)),
    :(ḡ - g_y * y),
    :(l - 1 / 3),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :(➕₁ - k / l),
    :((-β * ((z * α * ➕₁ ^ (α - 1) - δ) + 1)) / c ^ σ + c ^ -σ),
    :((c ^ σ * ψ) / (1 - l) - w),
    :((-i - k * (1 - δ)) + k),
    :(((-c - g) - i) + y),
    :(-(k ^ α) * l ^ (1 - α) * z + y),
    :(w - (y * (1 - α)) / l),
    :(r - (4 * y * α) / k),
    :((-z * ρᶻ + z + ρᶻ) - 1),
    :((-g * ρᵍ + g) - ḡ * (1 - ρᵍ)),
    :(ḡ - g_y * y),
    :(l - 1 / 3),
]

const PARAMETER_DEFINITION_NAMES = [
    "δ",
    "β",
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
    "i_y / k_y",
    "1 / (α / k_y + (1 - δ))",
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "σᶻ",
    "σᵍ",
    "σ",
    "i_y",
    "k_y",
    "ρᶻ",
    "ρᵍ",
    "g_y",
    "α",
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
    "c",
    "g",
    "i",
    "k",
    "l",
    "r",
    "w",
    "y",
    "z",
    "ḡ",
    "ψ",
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
    Inf,
    Inf,
    Inf,
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "c",
    "g",
    "i",
    "k",
    "l",
    "r",
    "w",
    "y",
    "z",
    "➕₁",
    "ḡ",
    "ψ",
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
    -Inf,
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
    Inf,
    1.0e12,
    Inf,
    Inf,
]
const ALL_AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "➕₁",
    "➕₂",
    "➕₃",
]
const ALL_AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
]
const ALL_AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    1.0e12,
    1.0e12,
    1.0e12,
]

const BLOCKS = [
    (
        index = 1,
        variables = ["ψ"],
        equation_indices = [3],
        equations = Expr[
            :((c ^ σ * ψ) / (1 - l) - w),
        ],
    ),
    (
        index = 2,
        variables = ["w"],
        equation_indices = [7],
        equations = Expr[
            :(w - (y * (1 - α)) / l),
        ],
    ),
    (
        index = 3,
        variables = ["r"],
        equation_indices = [8],
        equations = Expr[
            :(r - (4 * y * α) / k),
        ],
    ),
    (
        index = 4,
        variables = ["c", "g", "i", "k", "y", "➕₁", "ḡ"],
        equation_indices = [2, 5, 4, 6, 11, 1, 10],
        equations = Expr[
            :((-β * ((z * α * ➕₁ ^ (α - 1) - δ) + 1)) / c ^ σ + c ^ -σ),
            :(((-c - g) - i) + y),
            :((-i - k * (1 - δ)) + k),
            :(-(k ^ α) * l ^ (1 - α) * z + y),
            :(ḡ - g_y * y),
            :(➕₁ - k / l),
            :((-g * ρᵍ + g) - ḡ * (1 - ρᵍ)),
        ],
    ),
    (
        index = 5,
        variables = ["l"],
        equation_indices = [12],
        equations = Expr[
            :(l - 1 / 3),
        ],
    ),
    (
        index = 6,
        variables = ["z"],
        equation_indices = [9],
        equations = Expr[
            :((-z * ρᶻ + z + ρᶻ) - 1),
        ],
    ),
]
const BLOCK_EQUATION_ORDER = [3, 7, 8, 2, 5, 4, 6, 11, 1, 10, 12, 9]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[9] = parameters[9]
    complete_parameters[3] = parameters[3]
    complete_parameters[5] = parameters[5]
    complete_parameters[2] = parameters[2]
    complete_parameters[6] = parameters[6]
    complete_parameters[4] = parameters[4]
    complete_parameters[1] = parameters[1]
    complete_parameters[7] = parameters[7]
    complete_parameters[8] = parameters[8]
    complete_parameters[11] = complete_parameters[4] / complete_parameters[5]
    complete_parameters[10] = 1 / (complete_parameters[9] / complete_parameters[5] + (1 - complete_parameters[11]))
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] ^ -(complete_parameters[3]) - complete_parameters[10] * solution[1] ^ -(complete_parameters[3]) * ((complete_parameters[9] * solution[9] * (solution[4] / solution[5]) ^ (complete_parameters[9] - 1) + 1) - complete_parameters[11]),
        (solution[11] * solution[1] ^ complete_parameters[3]) / (1 - solution[5]) - solution[7],
        solution[4] - ((1 - complete_parameters[11]) * solution[4] + solution[3]),
        solution[8] - (solution[1] + solution[3] + solution[2]),
        solution[8] - solution[9] * solution[4] ^ complete_parameters[9] * solution[5] ^ (1 - complete_parameters[9]),
        solution[7] - (solution[8] * (1 - complete_parameters[9])) / solution[5],
        solution[6] - (solution[8] * complete_parameters[9] * 4) / solution[4],
        solution[9] - ((1 - complete_parameters[6]) + complete_parameters[6] * solution[9] + complete_parameters[1] * 0),
        solution[2] - ((1 - complete_parameters[7]) * solution[10] + complete_parameters[7] * solution[2] + complete_parameters[2] * 0),
        solution[10] - complete_parameters[8] * solution[8],
        solution[5] - 1 / 3,
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[10] - solution[4] / solution[5],
        (-(complete_parameters[10]) * ((solution[9] * complete_parameters[9] * solution[10] ^ (complete_parameters[9] - 1) - complete_parameters[11]) + 1)) / solution[1] ^ complete_parameters[3] + solution[1] ^ -(complete_parameters[3]),
        (solution[1] ^ complete_parameters[3] * solution[12]) / (1 - solution[5]) - solution[7],
        (-(solution[3]) - solution[4] * (1 - complete_parameters[11])) + solution[4],
        ((-(solution[1]) - solution[2]) - solution[3]) + solution[8],
        -(solution[4] ^ complete_parameters[9]) * solution[5] ^ (1 - complete_parameters[9]) * solution[9] + solution[8],
        solution[7] - (solution[8] * (1 - complete_parameters[9])) / solution[5],
        solution[6] - (4 * solution[8] * complete_parameters[9]) / solution[4],
        (-(solution[9]) * complete_parameters[6] + solution[9] + complete_parameters[6]) - 1,
        (-(solution[2]) * complete_parameters[7] + solution[2]) - solution[11] * (1 - complete_parameters[7]),
        solution[11] - complete_parameters[8] * solution[8],
        solution[5] - 1 / 3,
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
