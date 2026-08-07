module Ireland_2004NsssResiduals
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

const MODEL_NAME = "Ireland_2004"
const SOURCE_MODEL_FILE = "models/Ireland_2004.jl"
const NSSS_SOLUTION_ERROR = 0.0
const NSSS_RESIDUAL_NORM = 0.0

const PARAMETER_NAMES = [
    "β",
    "ψ",
    "ω",
    "αˣ",
    "αᵖ",
    "ρᵖ",
    "ρᵍ",
    "ρˣ",
    "ρᵃ",
    "ρᵉ",
    "σʳ",
    "σᵃ",
    "σᵉ",
    "σᶻ",
]
const PARAMETER_VALUES = Float64[
    0.99,
    0.1,
    0.0581,
    1.0e-5,
    1.0e-5,
    0.3866,
    0.396,
    0.1654,
    0.9048,
    0.9907,
    0.0028,
    0.0302,
    0.0002,
    0.0089,
]
const COMPLETE_PARAMETER_NAMES = [
    "β",
    "ψ",
    "ω",
    "αˣ",
    "αᵖ",
    "ρᵖ",
    "ρᵍ",
    "ρˣ",
    "ρᵃ",
    "ρᵉ",
    "σʳ",
    "σᵃ",
    "σᵉ",
    "σᶻ",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    0.99,
    0.1,
    0.0581,
    1.0e-5,
    1.0e-5,
    0.3866,
    0.396,
    0.1654,
    0.9048,
    0.9907,
    0.0028,
    0.0302,
    0.0002,
    0.0089,
]
const ORIGINAL_SOLUTION_NAMES = [
    "a",
    "e",
    "r̂",
    "x",
    "ĝ",
    "ŷ",
    "π̂",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
const AUXILIARY_SOLUTION_NAMES = [
    "a",
    "e",
    "r̂",
    "x",
    "ĝ",
    "ŷ",
    "π̂",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
const ALL_AUXILIARY_VARIABLE_NAMES = [
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
]
const DEFAULTED_NSSS_SOLUTION_NAMES = [
]
const CALIBRATION_PARAMETER_NAMES = [
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(a - (ρᵃ * a + σᵃ * 0)),
    :(e - (ρᵉ * e + σᵉ * 0)),
    :(x - (((αˣ * x + (1 - αˣ) * x) - (r̂ - π̂)) + a * (1 - ω) * (1 - ρᵃ))),
    :(π̂ - ((β * (αᵖ * π̂ + π̂ * (1 - αᵖ)) + x * ψ) - e)),
    :(x - (ŷ - a * ω)),
    :(ĝ - ((σᶻ * 0 + ŷ) - ŷ)),
    :((r̂ - r̂) - (π̂ * ρᵖ + ĝ * ρᵍ + x * ρˣ + σʳ * 0)),
]
const CALIBRATION_EQUATIONS = Expr[
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :(-a * ρᵃ + a),
    :(-e * ρᵉ + e),
    :(((((-a * (1 - ρᵃ) * (1 - ω) + r̂) - x * αˣ) - x * (1 - αˣ)) + x) - π̂),
    :(((e - x * ψ) - β * (αᵖ * π̂ + π̂ * (1 - αᵖ))) + π̂),
    :((a * ω + x) - ŷ),
    :(ĝ - 0),
    :((-x * ρˣ - ĝ * ρᵍ) - π̂ * ρᵖ),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(a - (ρᵃ * a + σᵃ * 0)),
    :(e - (ρᵉ * e + σᵉ * 0)),
    :(x - (((αˣ * x + (1 - αˣ) * x) - (r̂ - π̂)) + a * (1 - ω) * (1 - ρᵃ))),
    :(π̂ - ((β * (αᵖ * π̂ + π̂ * (1 - αᵖ)) + x * ψ) - e)),
    :(x - (ŷ - a * ω)),
    :(ĝ - ((σᶻ * 0 + ŷ) - ŷ)),
    :((r̂ - r̂) - (π̂ * ρᵖ + ĝ * ρᵍ + x * ρˣ + σʳ * 0)),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :(-a * ρᵃ + a),
    :(-e * ρᵉ + e),
    :(((((-a * (1 - ρᵃ) * (1 - ω) + r̂) - x * αˣ) - x * (1 - αˣ)) + x) - π̂),
    :(((e - x * ψ) - β * (αᵖ * π̂ + π̂ * (1 - αᵖ))) + π̂),
    :((a * ω + x) - ŷ),
    :(ĝ - 0),
    :((-x * ρˣ - ĝ * ρᵍ) - π̂ * ρᵖ),
]

const PARAMETER_DEFINITION_NAMES = [
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "β",
    "ψ",
    "ω",
    "αˣ",
    "αᵖ",
    "ρᵖ",
    "ρᵍ",
    "ρˣ",
    "ρᵃ",
    "ρᵉ",
    "σʳ",
    "σᵃ",
    "σᵉ",
    "σᶻ",
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
]
const ORIGINAL_BOX_CONSTRAINT_NAMES = [
    "a",
    "e",
    "r̂",
    "x",
    "ĝ",
    "ŷ",
    "π̂",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
]
const ORIGINAL_BOX_UPPER_BOUNDS = Float64[
    Inf,
    Inf,
    Inf,
    1.0e12,
    Inf,
    Inf,
    1.0e12,
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "a",
    "e",
    "r̂",
    "x",
    "ĝ",
    "ŷ",
    "π̂",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
]
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    Inf,
    Inf,
    Inf,
    1.0e12,
    Inf,
    Inf,
    1.0e12,
]
const ALL_AUXILIARY_BOX_CONSTRAINT_NAMES = [
]
const ALL_AUXILIARY_BOX_LOWER_BOUNDS = Float64[
]
const ALL_AUXILIARY_BOX_UPPER_BOUNDS = Float64[
]

const BLOCKS = [
    (
        index = 1,
        variables = ["ŷ"],
        equation_indices = [5],
        equations = Expr[
            :((a * ω + x) - ŷ),
        ],
    ),
    (
        index = 2,
        variables = ["r̂"],
        equation_indices = [3],
        equations = Expr[
            :(((((-a * (1 - ρᵃ) * (1 - ω) + r̂) - x * αˣ) - x * (1 - αˣ)) + x) - π̂),
        ],
    ),
    (
        index = 3,
        variables = ["x", "π̂"],
        equation_indices = [4, 7],
        equations = Expr[
            :(((e - x * ψ) - β * (αᵖ * π̂ + π̂ * (1 - αᵖ))) + π̂),
            :((-x * ρˣ - ĝ * ρᵍ) - π̂ * ρᵖ),
        ],
    ),
    (
        index = 4,
        variables = ["ĝ"],
        equation_indices = [6],
        equations = Expr[
            :(ĝ - 0),
        ],
    ),
    (
        index = 5,
        variables = ["e"],
        equation_indices = [2],
        equations = Expr[
            :(-e * ρᵉ + e),
        ],
    ),
    (
        index = 6,
        variables = ["a"],
        equation_indices = [1],
        equations = Expr[
            :(-a * ρᵃ + a),
        ],
    ),
]
const BLOCK_EQUATION_ORDER = [5, 3, 4, 7, 6, 2, 1]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[12] = parameters[12]
    complete_parameters[13] = parameters[13]
    complete_parameters[7] = parameters[7]
    complete_parameters[14] = parameters[14]
    complete_parameters[1] = parameters[1]
    complete_parameters[4] = parameters[4]
    complete_parameters[9] = parameters[9]
    complete_parameters[6] = parameters[6]
    complete_parameters[8] = parameters[8]
    complete_parameters[3] = parameters[3]
    complete_parameters[5] = parameters[5]
    complete_parameters[2] = parameters[2]
    complete_parameters[10] = parameters[10]
    complete_parameters[11] = parameters[11]
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - (complete_parameters[9] * solution[1] + complete_parameters[12] * 0),
        solution[2] - (complete_parameters[10] * solution[2] + complete_parameters[13] * 0),
        solution[4] - (((complete_parameters[4] * solution[4] + (1 - complete_parameters[4]) * solution[4]) - (solution[3] - solution[7])) + solution[1] * (1 - complete_parameters[3]) * (1 - complete_parameters[9])),
        solution[7] - ((complete_parameters[1] * (complete_parameters[5] * solution[7] + solution[7] * (1 - complete_parameters[5])) + solution[4] * complete_parameters[2]) - solution[2]),
        solution[4] - (solution[6] - solution[1] * complete_parameters[3]),
        solution[5] - ((complete_parameters[14] * 0 + solution[6]) - solution[6]),
        (solution[3] - solution[3]) - (solution[7] * complete_parameters[6] + solution[5] * complete_parameters[7] + solution[4] * complete_parameters[8] + complete_parameters[11] * 0),
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) * complete_parameters[9] + solution[1],
        -(solution[2]) * complete_parameters[10] + solution[2],
        ((((-(solution[1]) * (1 - complete_parameters[9]) * (1 - complete_parameters[3]) + solution[3]) - solution[4] * complete_parameters[4]) - solution[4] * (1 - complete_parameters[4])) + solution[4]) - solution[7],
        ((solution[2] - solution[4] * complete_parameters[2]) - complete_parameters[1] * (complete_parameters[5] * solution[7] + solution[7] * (1 - complete_parameters[5]))) + solution[7],
        (solution[1] * complete_parameters[3] + solution[4]) - solution[6],
        solution[5] - 0,
        (-(solution[4]) * complete_parameters[8] - solution[5] * complete_parameters[7]) - solution[7] * complete_parameters[6],
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
