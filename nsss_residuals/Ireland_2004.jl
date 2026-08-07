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
        solve_order = 6,
        variables = ["ŷ"],
        previous_solution_names = ["a", "x"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [5],
        equations = Expr[
            :((a * ω + x) - ŷ),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ŷ"],
        previous_solution_values = [0.0, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 2,
        solve_order = 5,
        variables = ["r̂"],
        previous_solution_names = ["a", "x", "π̂"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [3],
        equations = Expr[
            :(((((-a * (1 - ρᵃ) * (1 - ω) + r̂) - x * αˣ) - x * (1 - αˣ)) + x) - π̂),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["r̂"],
        previous_solution_values = [0.0, 0.0, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 3,
        solve_order = 4,
        variables = ["x", "π̂"],
        previous_solution_names = ["e", "ĝ"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [4, 7],
        equations = Expr[
            :(((e - x * ψ) - β * (αᵖ * π̂ + π̂ * (1 - αᵖ))) + π̂),
            :((-x * ρˣ - ĝ * ρᵍ) - π̂ * ρᵖ),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["x", "π̂"],
        previous_solution_values = [0.0, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.0, 0.0],
        box_lower_bounds = [-1.0e12, -1.0e12],
        box_upper_bounds = [1.0e12, 1.0e12],
    ),
    (
        index = 4,
        solve_order = 3,
        variables = ["ĝ"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [6],
        equations = Expr[
            :(ĝ - 0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ĝ"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 5,
        solve_order = 2,
        variables = ["e"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [2],
        equations = Expr[
            :(-e * ρᵉ + e),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["e"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 6,
        solve_order = 1,
        variables = ["a"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [1],
        equations = Expr[
            :(-a * ρᵃ + a),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["a"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
]
const BLOCK_EQUATION_ORDER = [5, 3, 4, 7, 6, 2, 1]
const BLOCK_SOLVE_ORDER = [6, 5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    ["a", "x"],
    ["a", "x", "π̂"],
    ["e", "ĝ"],
    String[],
    String[],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    [0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0],
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
]
const BLOCK_EXTERNAL_SOLUTION_VALUES = [
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
]
const BLOCK_SOLUTION_NAMES = [
    ["ŷ"],
    ["r̂"],
    ["x", "π̂"],
    ["ĝ"],
    ["e"],
    ["a"],
]
const BLOCK_SOLUTION_VALUES = [
    [0.0],
    [0.0],
    [0.0, 0.0],
    [0.0],
    [0.0],
    [0.0],
]

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

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (previous_solution[1] * complete_parameters[3] + previous_solution[2]) - solution[1],
    ]
end

function residuals_block_2(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((((-(previous_solution[1]) * (1 - complete_parameters[9]) * (1 - complete_parameters[3]) + solution[1]) - previous_solution[2] * complete_parameters[4]) - previous_solution[2] * (1 - complete_parameters[4])) + previous_solution[2]) - previous_solution[3],
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((previous_solution[1] - solution[1] * complete_parameters[2]) - complete_parameters[1] * (complete_parameters[5] * solution[2] + solution[2] * (1 - complete_parameters[5]))) + solution[2],
        (-(solution[1]) * complete_parameters[8] - previous_solution[2] * complete_parameters[7]) - solution[2] * complete_parameters[6],
    ]
end

function residuals_block_4(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0,
    ]
end

function residuals_block_5(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) * complete_parameters[10] + solution[1],
    ]
end

function residuals_block_6(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) * complete_parameters[9] + solution[1],
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
    )
end

export MODEL_NAME, SOURCE_MODEL_FILE, NSSS_SOLUTION_ERROR, NSSS_RESIDUAL_NORM
export PARAMETER_NAMES, PARAMETER_VALUES, COMPLETE_PARAMETER_NAMES, COMPLETE_PARAMETER_VALUES
export ORIGINAL_SOLUTION_NAMES, ORIGINAL_SOLUTION_VALUES
export AUXILIARY_SOLUTION_NAMES, AUXILIARY_SOLUTION_VALUES
export ALL_AUXILIARY_VARIABLE_NAMES, ALL_AUXILIARY_VARIABLE_VALUES
export DEFAULTED_NSSS_SOLUTION_NAMES
export ORIGINAL_NSSS_EQUATIONS, AUXILIARY_NSSS_EQUATIONS, CALIBRATION_EQUATIONS
export BLOCKS, BLOCK_EQUATION_ORDER, BLOCK_SOLVE_ORDER
export BLOCK_PREVIOUS_SOLUTION_NAMES, BLOCK_PREVIOUS_SOLUTION_VALUES
export BLOCK_EXTERNAL_SOLUTION_NAMES, BLOCK_EXTERNAL_SOLUTION_VALUES
export BLOCK_SOLUTION_NAMES, BLOCK_SOLUTION_VALUES
export residuals_original, residuals_auxiliary, residuals_blocks
export residuals_block_1, residuals_block_2, residuals_block_3, residuals_block_4, residuals_block_5, residuals_block_6
end
