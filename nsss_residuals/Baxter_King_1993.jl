module Baxter_King_1993NsssResiduals
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

const MODEL_NAME = "Baxter_King_1993"
const SOURCE_MODEL_FILE = "models/Baxter_King_1993.jl"
const NSSS_SOLUTION_ERROR = 9.254518065620725e-16
const NSSS_RESIDUAL_NORM = 9.254518065620725e-16

const PARAMETER_NAMES = [
    "A",
    "γ_x",
    "θ_k",
    "δ_k",
    "N",
    "R",
    "sG",
    "τBAR",
    "τ",
]
const PARAMETER_VALUES = Float64[
    1.0,
    1.016,
    0.42,
    0.1,
    0.2,
    0.065,
    0.2,
    0.2,
    0.2,
]
const COMPLETE_PARAMETER_NAMES = [
    "A",
    "γ_x",
    "θ_k",
    "δ_k",
    "N",
    "R",
    "sG",
    "τBAR",
    "τ",
    "GB_BAR",
    "β",
    "θ_l",
    "θ_n",
    "L",
    "Q",
    "FK",
    "K",
    "FN",
    "IV",
    "Y",
    "C",
    "UC",
    "UL",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    1.0,
    1.016,
    0.42,
    0.1,
    0.2,
    0.065,
    0.2,
    0.2,
    0.2,
    0.06694433831170692,
    0.9539906103286385,
    3.2920536635706927,
    0.5800000000000001,
    0.8,
    0.165,
    0.20625,
    0.6816150809919249,
    0.9706929055197504,
    0.07906734939506331,
    0.33472169155853454,
    0.1887100038517643,
    5.299136132631957,
    4.1150670794633655,
]
const ORIGINAL_SOLUTION_NAMES = [
    "c",
    "check_walras",
    "fk",
    "fn",
    "gb",
    "iv",
    "k",
    "l",
    "n",
    "q",
    "r",
    "tr",
    "uc",
    "ul",
    "w",
    "y",
    "λ",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    0.18871000385176434,
    1.3877787807814457e-16,
    0.2062499999999999,
    0.9706929055197503,
    0.06694433831170692,
    0.07906734939506334,
    0.6816150809919251,
    0.8,
    0.19999999999999998,
    0.16499999999999992,
    0.06499999999999999,
    -1.3877787807814457e-17,
    5.299136132631956,
    4.1150670794633655,
    0.9706929055197503,
    0.3347216915585345,
    5.299136132631956,
]
const AUXILIARY_SOLUTION_NAMES = [
    "c",
    "check_walras",
    "fk",
    "fn",
    "gb",
    "iv",
    "k",
    "l",
    "n",
    "q",
    "r",
    "tr",
    "uc",
    "ul",
    "w",
    "y",
    "λ",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    0.18871000385176434,
    1.3877787807814457e-16,
    0.2062499999999999,
    0.9706929055197503,
    0.06694433831170692,
    0.07906734939506334,
    0.6816150809919251,
    0.8,
    0.19999999999999998,
    0.16499999999999992,
    0.06499999999999999,
    -1.3877787807814457e-17,
    5.299136132631956,
    4.1150670794633655,
    0.9706929055197503,
    0.3347216915585345,
    5.299136132631956,
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
    :(uc - c ^ -1),
    :(ul - θ_l * l ^ -1),
    :(y - A * k ^ θ_k * n ^ θ_n),
    :(fk - θ_k * A * k ^ (θ_k - 1) * n ^ θ_n),
    :(fn - θ_n * A * k ^ θ_k * n ^ (θ_n - 1)),
    :(γ_x * k - ((1 - δ_k) * k + iv)),
    :((l + n) - 1),
    :((c + iv) - ((1 - τ) * y + tr + check_walras)),
    :((c + iv + gb) - y),
    :(τ * y - (gb + tr)),
    :(uc - λ),
    :(ul - λ * (1 - τ) * fn),
    :(β * λ * ((q + 1) - δ_k) - γ_x * λ),
    :(q - (1 - τ) * fk),
    :(gb - (GB_BAR + 0)),
    :((1 + r) - (γ_x * λ) / (λ * β)),
    :(w - fn),
]
const CALIBRATION_EQUATIONS = Expr[
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :(uc - 1 / c),
    :(ul - θ_l / l),
    :(-A * k ^ θ_k * n ^ θ_n + y),
    :(-A * k ^ (θ_k - 1) * n ^ θ_n * θ_k + fk),
    :(-A * k ^ θ_k * n ^ (θ_n - 1) * θ_n + fn),
    :((-iv + k * γ_x) - k * (1 - δ_k)),
    :((l + n) - 1),
    :((((c - check_walras) + iv) - tr) - y * (1 - τ)),
    :((c + gb + iv) - y),
    :((-gb - tr) + y * τ),
    :(uc - λ),
    :(-fn * λ * (1 - τ) + ul),
    :(β * λ * ((q - δ_k) + 1) - γ_x * λ),
    :(-fk * (1 - τ) + q),
    :(-GB_BAR + gb),
    :((r + 1) - γ_x / β),
    :(-fn + w),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(uc - c ^ -1),
    :(ul - θ_l * l ^ -1),
    :(y - A * k ^ θ_k * n ^ θ_n),
    :(fk - θ_k * A * k ^ (θ_k - 1) * n ^ θ_n),
    :(fn - θ_n * A * k ^ θ_k * n ^ (θ_n - 1)),
    :(γ_x * k - ((1 - δ_k) * k + iv)),
    :((l + n) - 1),
    :((c + iv) - ((1 - τ) * y + tr + check_walras)),
    :((c + iv + gb) - y),
    :(τ * y - (gb + tr)),
    :(uc - λ),
    :(ul - λ * (1 - τ) * fn),
    :(β * λ * ((q + 1) - δ_k) - γ_x * λ),
    :(q - (1 - τ) * fk),
    :(gb - (GB_BAR + 0)),
    :((1 + r) - (γ_x * λ) / (λ * β)),
    :(w - fn),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :(uc - 1 / c),
    :(ul - θ_l / l),
    :(-A * k ^ θ_k * n ^ θ_n + y),
    :(-A * k ^ (θ_k - 1) * n ^ θ_n * θ_k + fk),
    :(-A * k ^ θ_k * n ^ (θ_n - 1) * θ_n + fn),
    :((-iv + k * γ_x) - k * (1 - δ_k)),
    :((l + n) - 1),
    :((((c - check_walras) + iv) - tr) - y * (1 - τ)),
    :((c + gb + iv) - y),
    :((-gb - tr) + y * τ),
    :(uc - λ),
    :(-fn * λ * (1 - τ) + ul),
    :(β * λ * ((q - δ_k) + 1) - γ_x * λ),
    :(-fk * (1 - τ) + q),
    :(-GB_BAR + gb),
    :((r + 1) - γ_x / β),
    :(-fn + w),
]

const PARAMETER_DEFINITION_NAMES = [
    "L",
    "β",
    "θ_n",
    "Q",
    "FK",
    "K",
    "Y",
    "FN",
    "GB_BAR",
    "IV",
    "C",
    "UC",
    "UL",
    "θ_l",
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
    "1 - N",
    "γ_x / (1 + R)",
    "1 - θ_k",
    "(γ_x / β - 1) + δ_k",
    "Q / (1 - τBAR)",
    "(FK / (θ_k * A * N ^ θ_n)) ^ (1 / (θ_k - 1))",
    "A * N ^ (1 - θ_k) * K ^ θ_k",
    "θ_n * A * K ^ θ_k * N ^ (θ_n - 1)",
    "sG * Y",
    "((γ_x - 1) + δ_k) * K",
    "(Y - IV) - GB_BAR",
    "C ^ -1",
    "UC * (1 - τBAR) * FN",
    "UL * L",
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "A",
    "γ_x",
    "θ_k",
    "δ_k",
    "N",
    "R",
    "sG",
    "τBAR",
    "τ",
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
    "check_walras",
    "fk",
    "fn",
    "gb",
    "iv",
    "k",
    "l",
    "n",
    "q",
    "r",
    "tr",
    "uc",
    "ul",
    "w",
    "y",
    "λ",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
    -Inf,
    -1.0e12,
    2.220446049250313e-16,
    -1.0e12,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -Inf,
    -1.0e12,
    -1.0e12,
]
const ORIGINAL_BOX_UPPER_BOUNDS = Float64[
    1.0e12,
    Inf,
    Inf,
    1.0e12,
    Inf,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    1.0e12,
    1.0e12,
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "c",
    "check_walras",
    "fk",
    "fn",
    "gb",
    "iv",
    "k",
    "l",
    "n",
    "q",
    "r",
    "tr",
    "uc",
    "ul",
    "w",
    "y",
    "λ",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
    -Inf,
    -1.0e12,
    2.220446049250313e-16,
    -1.0e12,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -Inf,
    -1.0e12,
    -1.0e12,
]
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    1.0e12,
    Inf,
    Inf,
    1.0e12,
    Inf,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    1.0e12,
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
        variables = ["w"],
        equation_indices = [17],
        equations = Expr[
            :(-fn + w),
        ],
    ),
    (
        index = 2,
        variables = ["r"],
        equation_indices = [16],
        equations = Expr[
            :((r + 1) - γ_x / β),
        ],
    ),
    (
        index = 3,
        variables = ["check_walras"],
        equation_indices = [8],
        equations = Expr[
            :((((c - check_walras) + iv) - tr) - y * (1 - τ)),
        ],
    ),
    (
        index = 4,
        variables = ["tr"],
        equation_indices = [10],
        equations = Expr[
            :((-gb - tr) + y * τ),
        ],
    ),
    (
        index = 5,
        variables = ["c", "fk", "fn", "iv", "k", "l", "n", "q", "uc", "ul", "y", "λ"],
        equation_indices = [9, 14, 5, 6, 4, 2, 7, 13, 1, 12, 3, 11],
        equations = Expr[
            :((c + gb + iv) - y),
            :(-fk * (1 - τ) + q),
            :(-A * k ^ θ_k * n ^ (θ_n - 1) * θ_n + fn),
            :((-iv + k * γ_x) - k * (1 - δ_k)),
            :(-A * k ^ (θ_k - 1) * n ^ θ_n * θ_k + fk),
            :(ul - θ_l / l),
            :((l + n) - 1),
            :(β * λ * ((q - δ_k) + 1) - γ_x * λ),
            :(uc - 1 / c),
            :(-fn * λ * (1 - τ) + ul),
            :(-A * k ^ θ_k * n ^ θ_n + y),
            :(uc - λ),
        ],
    ),
    (
        index = 6,
        variables = ["gb"],
        equation_indices = [15],
        equations = Expr[
            :(-GB_BAR + gb),
        ],
    ),
]
const BLOCK_EQUATION_ORDER = [17, 16, 8, 10, 9, 14, 5, 6, 4, 2, 7, 13, 1, 12, 3, 11, 15]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[2] = parameters[2]
    complete_parameters[4] = parameters[4]
    complete_parameters[1] = parameters[1]
    complete_parameters[3] = parameters[3]
    complete_parameters[5] = parameters[5]
    complete_parameters[6] = parameters[6]
    complete_parameters[7] = parameters[7]
    complete_parameters[9] = parameters[9]
    complete_parameters[8] = parameters[8]
    complete_parameters[14] = 1 - complete_parameters[5]
    complete_parameters[11] = complete_parameters[2] / (1 + complete_parameters[6])
    complete_parameters[13] = 1 - complete_parameters[3]
    complete_parameters[15] = (complete_parameters[2] / complete_parameters[11] - 1) + complete_parameters[4]
    complete_parameters[16] = complete_parameters[15] / (1 - complete_parameters[8])
    complete_parameters[17] = (complete_parameters[16] / (complete_parameters[3] * complete_parameters[1] * complete_parameters[5] ^ complete_parameters[13])) ^ (1 / (complete_parameters[3] - 1))
    complete_parameters[20] = complete_parameters[1] * complete_parameters[5] ^ (1 - complete_parameters[3]) * complete_parameters[17] ^ complete_parameters[3]
    complete_parameters[18] = complete_parameters[13] * complete_parameters[1] * complete_parameters[17] ^ complete_parameters[3] * complete_parameters[5] ^ (complete_parameters[13] - 1)
    complete_parameters[10] = complete_parameters[7] * complete_parameters[20]
    complete_parameters[19] = ((complete_parameters[2] - 1) + complete_parameters[4]) * complete_parameters[17]
    complete_parameters[21] = (complete_parameters[20] - complete_parameters[19]) - complete_parameters[10]
    complete_parameters[22] = complete_parameters[21] ^ -1
    complete_parameters[23] = complete_parameters[22] * (1 - complete_parameters[8]) * complete_parameters[18]
    complete_parameters[12] = complete_parameters[23] * complete_parameters[14]
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[13] - solution[1] ^ -1,
        solution[14] - complete_parameters[12] * solution[8] ^ -1,
        solution[16] - complete_parameters[1] * solution[7] ^ complete_parameters[3] * solution[9] ^ complete_parameters[13],
        solution[3] - complete_parameters[3] * complete_parameters[1] * solution[7] ^ (complete_parameters[3] - 1) * solution[9] ^ complete_parameters[13],
        solution[4] - complete_parameters[13] * complete_parameters[1] * solution[7] ^ complete_parameters[3] * solution[9] ^ (complete_parameters[13] - 1),
        complete_parameters[2] * solution[7] - ((1 - complete_parameters[4]) * solution[7] + solution[6]),
        (solution[8] + solution[9]) - 1,
        (solution[1] + solution[6]) - ((1 - complete_parameters[9]) * solution[16] + solution[12] + solution[2]),
        (solution[1] + solution[6] + solution[5]) - solution[16],
        complete_parameters[9] * solution[16] - (solution[5] + solution[12]),
        solution[13] - solution[17],
        solution[14] - solution[17] * (1 - complete_parameters[9]) * solution[4],
        complete_parameters[11] * solution[17] * ((solution[10] + 1) - complete_parameters[4]) - complete_parameters[2] * solution[17],
        solution[10] - (1 - complete_parameters[9]) * solution[3],
        solution[5] - (complete_parameters[10] + 0),
        (1 + solution[11]) - (complete_parameters[2] * solution[17]) / (solution[17] * complete_parameters[11]),
        solution[15] - solution[4],
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[13] - 1 / solution[1],
        solution[14] - complete_parameters[12] / solution[8],
        -(complete_parameters[1]) * solution[7] ^ complete_parameters[3] * solution[9] ^ complete_parameters[13] + solution[16],
        -(complete_parameters[1]) * solution[7] ^ (complete_parameters[3] - 1) * solution[9] ^ complete_parameters[13] * complete_parameters[3] + solution[3],
        -(complete_parameters[1]) * solution[7] ^ complete_parameters[3] * solution[9] ^ (complete_parameters[13] - 1) * complete_parameters[13] + solution[4],
        (-(solution[6]) + solution[7] * complete_parameters[2]) - solution[7] * (1 - complete_parameters[4]),
        (solution[8] + solution[9]) - 1,
        (((solution[1] - solution[2]) + solution[6]) - solution[12]) - solution[16] * (1 - complete_parameters[9]),
        (solution[1] + solution[5] + solution[6]) - solution[16],
        (-(solution[5]) - solution[12]) + solution[16] * complete_parameters[9],
        solution[13] - solution[17],
        -(solution[4]) * solution[17] * (1 - complete_parameters[9]) + solution[14],
        complete_parameters[11] * solution[17] * ((solution[10] - complete_parameters[4]) + 1) - complete_parameters[2] * solution[17],
        -(solution[3]) * (1 - complete_parameters[9]) + solution[10],
        -(complete_parameters[10]) + solution[5],
        (solution[11] + 1) - complete_parameters[2] / complete_parameters[11],
        -(solution[4]) + solution[15],
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
