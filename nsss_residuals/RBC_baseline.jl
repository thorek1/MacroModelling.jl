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
const ORIGINAL_INITIAL_SOLUTION_VALUES = Float64[
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
const AUXILIARY_INITIAL_SOLUTION_VALUES = Float64[
    0.5871473576160889,
    0.21907841721376572,
    0.26874192494328436,
    11.17966407764063,
    0.3333333333333333,
    0.1282051282051284,
    2.1499353995462784,
    1.074967699773139,
    1.0,
    33.53899223292193,
    0.21907841721376572,
    2.4411082631514707,
]
const ALL_AUXILIARY_VARIABLE_NAMES = [
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    33.53899223292189,
    11.17966407764063,
    0.5871473576160889,
    0.3333333333333333,
    0.5871473576160889,
]
const ALL_AUXILIARY_VARIABLE_INITIAL_VALUES = Float64[
    33.53899223292193,
    11.17966407764063,
    0.5871473576160889,
    0.3333333333333333,
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
    "➕₄",
    "➕₅",
]
const ALL_AUXILIARY_BOX_LOWER_BOUNDS = Float64[
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
]

const BLOCKS = [
    (
        index = 1,
        solve_order = 6,
        variables = ["ψ"],
        previous_solution_names = ["c", "l", "w"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₅"],
        equation_indices = [3],
        equations = Expr[
            :((➕₅ ^ σ * ψ) / (1 - l) - w),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₅ = min(1.0e12, max(eps(), c))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₅ - c)),
        ],
        solution_names = ["ψ", "➕₅"],
        previous_solution_values = [0.5871473576160889, 0.3333333333333333, 2.1499353995462784],
        external_solution_values = Float64[],
        solution_values = [2.4411082631514707, 0.5871473576160889],
        previous_solution_initial_values = [0.5871473576160889, 0.3333333333333333, 2.1499353995462784],
        external_solution_initial_values = Float64[],
        solution_initial_values = [2.4411082631514707, 0.5871473576160889],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 2,
        solve_order = 5,
        variables = ["w"],
        previous_solution_names = ["l", "y"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [7],
        equations = Expr[
            :(w - (y * (1 - α)) / l),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["w"],
        previous_solution_values = [0.3333333333333333, 1.074967699773139],
        external_solution_values = Float64[],
        solution_values = [2.1499353995462784],
        previous_solution_initial_values = [0.3333333333333333, 1.074967699773139],
        external_solution_initial_values = Float64[],
        solution_initial_values = [2.1499353995462784],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 3,
        solve_order = 4,
        variables = ["r"],
        previous_solution_names = ["k", "y"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [8],
        equations = Expr[
            :(r - (4 * y * α) / k),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["r"],
        previous_solution_values = [11.17966407764063, 1.074967699773139],
        external_solution_values = Float64[],
        solution_values = [0.1282051282051284],
        previous_solution_initial_values = [11.17966407764063, 1.074967699773139],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.1282051282051284],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 4,
        solve_order = 3,
        variables = ["c", "g", "i", "k", "y", "ḡ", "➕₁"],
        previous_solution_names = ["l", "z"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₄"],
        equation_indices = [2, 5, 4, 6, 11, 1, 10],
        equations = Expr[
            :((-β * ((z * α * ➕₁ ^ (α - 1) - δ) + 1)) / c ^ σ + c ^ -σ),
            :(((-c - g) - i) + y),
            :((-i - k * (1 - δ)) + k),
            :(-(k ^ α) * ➕₄ ^ (1 - α) * z + y),
            :(ḡ - g_y * y),
            :(➕₁ - k / l),
            :((-g * ρᵍ + g) - ḡ * (1 - ρᵍ)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₄ = min(1.0e12, max(eps(), l))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₄ - l)),
        ],
        solution_names = ["c", "g", "i", "k", "y", "ḡ", "➕₁", "➕₄"],
        previous_solution_values = [0.3333333333333333, 1.0],
        external_solution_values = Float64[],
        solution_values = [0.5871473576160889, 0.21907841721376572, 0.26874192494328436, 11.17966407764063, 1.074967699773139, 0.21907841721376572, 33.53899223292189, 0.3333333333333333],
        previous_solution_initial_values = [0.3333333333333333, 1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.5871473576160889, 0.21907841721376572, 0.26874192494328436, 11.17966407764063, 1.074967699773139, 0.21907841721376572, 33.53899223292193, 0.3333333333333333],
        box_lower_bounds = [2.220446049250313e-16, -Inf, -Inf, 2.220446049250313e-16, -Inf, -Inf, 2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, Inf, Inf, 1.0e12, Inf, Inf, 1.0e12, 1.0e12],
    ),
    (
        index = 5,
        solve_order = 2,
        variables = ["l"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [12],
        equations = Expr[
            :(l - 1 / 3),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["l"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.3333333333333333],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.3333333333333333],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 6,
        solve_order = 1,
        variables = ["z"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [9],
        equations = Expr[
            :((-z * ρᶻ + z + ρᶻ) - 1),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["z"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
]
const BLOCK_EQUATION_ORDER = [3, 7, 8, 2, 5, 4, 6, 11, 1, 10, 12, 9]
const BLOCK_SOLVE_ORDER = [6, 5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    ["c", "l", "w"],
    ["l", "y"],
    ["k", "y"],
    ["l", "z"],
    String[],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    [0.5871473576160889, 0.3333333333333333, 2.1499353995462784],
    [0.3333333333333333, 1.074967699773139],
    [11.17966407764063, 1.074967699773139],
    [0.3333333333333333, 1.0],
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
    ["ψ", "➕₅"],
    ["w"],
    ["r"],
    ["c", "g", "i", "k", "y", "ḡ", "➕₁", "➕₄"],
    ["l"],
    ["z"],
]
const BLOCK_SOLUTION_VALUES = [
    [2.4411082631514707, 0.5871473576160889],
    [2.1499353995462784],
    [0.1282051282051284],
    [0.5871473576160889, 0.21907841721376572, 0.26874192494328436, 11.17966407764063, 1.074967699773139, 0.21907841721376572, 33.53899223292189, 0.3333333333333333],
    [0.3333333333333333],
    [1.0],
]
const BLOCK_PREVIOUS_SOLUTION_INITIAL_VALUES = [
    [0.5871473576160889, 0.3333333333333333, 2.1499353995462784],
    [0.3333333333333333, 1.074967699773139],
    [11.17966407764063, 1.074967699773139],
    [0.3333333333333333, 1.0],
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
]
const BLOCK_SOLUTION_INITIAL_VALUES = [
    [2.4411082631514707, 0.5871473576160889],
    [2.1499353995462784],
    [0.1282051282051284],
    [0.5871473576160889, 0.21907841721376572, 0.26874192494328436, 11.17966407764063, 1.074967699773139, 0.21907841721376572, 33.53899223292193, 0.3333333333333333],
    [0.3333333333333333],
    [1.0],
]

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

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        (solution[2] ^ complete_parameters[3] * solution[1]) / (1 - previous_solution[2]) - previous_solution[3],
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
        solution[1] - (previous_solution[2] * (1 - complete_parameters[9])) / previous_solution[1],
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - (4 * previous_solution[2] * complete_parameters[9]) / previous_solution[1],
    ]
end

function residuals_block_4(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 8
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(complete_parameters[10]) * ((previous_solution[2] * complete_parameters[9] * solution[7] ^ (complete_parameters[9] - 1) - complete_parameters[11]) + 1)) / solution[1] ^ complete_parameters[3] + solution[1] ^ -(complete_parameters[3]),
        ((-(solution[1]) - solution[2]) - solution[3]) + solution[5],
        (-(solution[3]) - solution[4] * (1 - complete_parameters[11])) + solution[4],
        -(solution[4] ^ complete_parameters[9]) * solution[8] ^ (1 - complete_parameters[9]) * previous_solution[2] + solution[5],
        solution[6] - complete_parameters[8] * solution[5],
        solution[7] - solution[4] / previous_solution[1],
        (-(solution[2]) * complete_parameters[7] + solution[2]) - solution[6] * (1 - complete_parameters[7]),
        solution[8] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_5(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 1 / 3,
    ]
end

function residuals_block_6(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[1]) * complete_parameters[6] + solution[1] + complete_parameters[6]) - 1,
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
export residuals_block_1, residuals_block_2, residuals_block_3, residuals_block_4, residuals_block_5, residuals_block_6
end
