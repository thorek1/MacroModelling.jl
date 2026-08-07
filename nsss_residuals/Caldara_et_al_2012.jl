module Caldara_et_al_2012NsssResiduals
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

const MODEL_NAME = "Caldara_et_al_2012"
const SOURCE_MODEL_FILE = "models/Caldara_et_al_2012.jl"
const NSSS_SOLUTION_ERROR = 2.8929862278582974e-16
const NSSS_RESIDUAL_NORM = 9.313225746172272e-10

const PARAMETER_NAMES = [
    "β",
    "ζ",
    "δ",
    "λ",
    "ψ",
    "γ",
    "σ̄",
    "η",
    "ρ",
]
const PARAMETER_VALUES = Float64[
    0.991,
    0.3,
    0.0196,
    0.95,
    0.5,
    40.0,
    0.021,
    0.1,
    0.9,
]
const COMPLETE_PARAMETER_NAMES = [
    "β",
    "ζ",
    "δ",
    "λ",
    "ψ",
    "γ",
    "σ̄",
    "η",
    "ρ",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    0.991,
    0.3,
    0.0196,
    0.95,
    0.5,
    40.0,
    0.021,
    0.1,
    0.9,
]
const ORIGINAL_SOLUTION_NAMES = [
    "Rᵏ",
    "Rᶠ",
    "SDF⁺¹",
    "V",
    "c",
    "i",
    "k",
    "l",
    "s",
    "y",
    "z",
    "σ",
    "ν",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    0.009081735620585375,
    0.009081735620585276,
    0.991,
    0.6871386578565624,
    0.7247305637488348,
    0.18688997126148366,
    9.53520261538182,
    0.3333333333333333,
    14.633547871167153,
    0.9116205350103185,
    0.0,
    0.021,
    0.3621843141705121,
]
const AUXILIARY_SOLUTION_NAMES = [
    "Rᵏ",
    "Rᶠ",
    "SDF⁺¹",
    "V",
    "c",
    "i",
    "k",
    "l",
    "s",
    "y",
    "z",
    "σ",
    "➕₁",
    "➕₂",
    "➕₃",
    "ν",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    0.009081735620585375,
    0.009081735620585276,
    0.991,
    0.6871386578565624,
    0.7247305637488348,
    0.18688997126148366,
    9.53520261538182,
    0.3333333333333333,
    14.633547871167153,
    0.9116205350103185,
    0.0,
    0.021,
    0.6666666666666667,
    0.6871386578565634,
    1.4553103490340173,
    0.3621843141705121,
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
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    0.6666666666666667,
    0.6871386578565634,
    1.4553103490340173,
    0.2062856661496336,
    9.53520261538182,
    28.60560784614546,
    0.6871386578565624,
    2.266047927759795e6,
]
const DEFAULTED_NSSS_SOLUTION_NAMES = [
]
const CALIBRATION_PARAMETER_NAMES = [
    "ν",
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(V - ((1 - β) * (c ^ ν * (1 - l) ^ (1 - ν)) ^ (1 - 1 / ψ) + β * V ^ (1 - 1 / ψ)) ^ (1 / (1 - 1 / ψ))),
    :(exp(s) - V ^ (1 - γ)),
    :(1 - (((1 + ζ * exp(z) * k ^ (ζ - 1) * l ^ (1 - ζ)) - δ) * c * β * (((1 - l) / (1 - l)) ^ (1 - ν) * (c / c) ^ ν) ^ (1 - 1 / ψ)) / c),
    :(Rᵏ - (ζ * exp(z) * k ^ (ζ - 1) * l ^ (1 - ζ) - δ)),
    :(SDF⁺¹ - (c * β * (((1 - l) / (1 - l)) ^ (1 - ν) * (c / c) ^ ν) ^ (1 - 1 / ψ)) / c),
    :((1 + Rᶠ) - 1 / SDF⁺¹),
    :((((1 - ν) / ν) * c) / (1 - l) - (1 - ζ) * exp(z) * k ^ ζ * l ^ -ζ),
    :((c + i) - exp(z) * k ^ ζ * l ^ (1 - ζ)),
    :(k - (i + k * (1 - δ))),
    :(z - (λ * z + σ * 0)),
    :(y - exp(z) * k ^ ζ * l ^ (1 - ζ)),
    :(log(σ) - ((1 - ρ) * log(σ̄) + ρ * log(σ) + η * 0)),
]
const CALIBRATION_EQUATIONS = Expr[
    :(l - 1 / 3),
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :(➕₁ - (1 - l)),
    :(➕₂ - c ^ ν * ➕₁ ^ (1 - ν)),
    :(➕₃ - (V ^ (1 - 1 / ψ) * β + ➕₂ ^ (1 - 1 / ψ) * (1 - β))),
    :(V - ➕₃ ^ (1 / (1 - 1 / ψ))),
    :(-(V ^ (1 - γ)) + exp(s)),
    :(-β * ((k ^ (ζ - 1) * l ^ (1 - ζ) * ζ * exp(z) - δ) + 1) + 1),
    :((Rᵏ - k ^ (ζ - 1) * l ^ (1 - ζ) * ζ * exp(z)) + δ),
    :(SDF⁺¹ - β),
    :((Rᶠ + 1) - 1 / SDF⁺¹),
    :((c * (1 - ν)) / (ν * (1 - l)) - (k ^ ζ * (1 - ζ) * exp(z)) / l ^ ζ),
    :((c + i) - k ^ ζ * l ^ (1 - ζ) * exp(z)),
    :((-i - k * (1 - δ)) + k),
    :(-z * λ + z),
    :(-(k ^ ζ) * l ^ (1 - ζ) * exp(z) + y),
    :((-ρ * log(σ) - (1 - ρ) * log(σ̄)) + log(σ)),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(V - ((1 - β) * (c ^ ν * (1 - l) ^ (1 - ν)) ^ (1 - 1 / ψ) + β * V ^ (1 - 1 / ψ)) ^ (1 / (1 - 1 / ψ))),
    :(exp(s) - V ^ (1 - γ)),
    :(1 - (((1 + ζ * exp(z) * k ^ (ζ - 1) * l ^ (1 - ζ)) - δ) * c * β * (((1 - l) / (1 - l)) ^ (1 - ν) * (c / c) ^ ν) ^ (1 - 1 / ψ)) / c),
    :(Rᵏ - (ζ * exp(z) * k ^ (ζ - 1) * l ^ (1 - ζ) - δ)),
    :(SDF⁺¹ - (c * β * (((1 - l) / (1 - l)) ^ (1 - ν) * (c / c) ^ ν) ^ (1 - 1 / ψ)) / c),
    :((1 + Rᶠ) - 1 / SDF⁺¹),
    :((((1 - ν) / ν) * c) / (1 - l) - (1 - ζ) * exp(z) * k ^ ζ * l ^ -ζ),
    :((c + i) - exp(z) * k ^ ζ * l ^ (1 - ζ)),
    :(k - (i + k * (1 - δ))),
    :(z - (λ * z + σ * 0)),
    :(y - exp(z) * k ^ ζ * l ^ (1 - ζ)),
    :(log(σ) - ((1 - ρ) * log(σ̄) + ρ * log(σ) + η * 0)),
    :(l - 1 / 3),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :(➕₁ - (1 - l)),
    :(➕₂ - c ^ ν * ➕₁ ^ (1 - ν)),
    :(➕₃ - (V ^ (1 - 1 / ψ) * β + ➕₂ ^ (1 - 1 / ψ) * (1 - β))),
    :(V - ➕₃ ^ (1 / (1 - 1 / ψ))),
    :(-(V ^ (1 - γ)) + exp(s)),
    :(-β * ((k ^ (ζ - 1) * l ^ (1 - ζ) * ζ * exp(z) - δ) + 1) + 1),
    :((Rᵏ - k ^ (ζ - 1) * l ^ (1 - ζ) * ζ * exp(z)) + δ),
    :(SDF⁺¹ - β),
    :((Rᶠ + 1) - 1 / SDF⁺¹),
    :((c * (1 - ν)) / (ν * (1 - l)) - (k ^ ζ * (1 - ζ) * exp(z)) / l ^ ζ),
    :((c + i) - k ^ ζ * l ^ (1 - ζ) * exp(z)),
    :((-i - k * (1 - δ)) + k),
    :(-z * λ + z),
    :(-(k ^ ζ) * l ^ (1 - ζ) * exp(z) + y),
    :((-ρ * log(σ) - (1 - ρ) * log(σ̄)) + log(σ)),
    :(l - 1 / 3),
]

const PARAMETER_DEFINITION_NAMES = [
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "β",
    "ζ",
    "δ",
    "λ",
    "ψ",
    "γ",
    "σ̄",
    "η",
    "ρ",
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
    "Rᵏ",
    "Rᶠ",
    "SDF⁺¹",
    "V",
    "c",
    "i",
    "k",
    "l",
    "s",
    "y",
    "z",
    "σ",
    "ν",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
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
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "Rᵏ",
    "Rᶠ",
    "SDF⁺¹",
    "V",
    "c",
    "i",
    "k",
    "l",
    "s",
    "y",
    "z",
    "σ",
    "➕₁",
    "➕₂",
    "➕₃",
    "ν",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
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
    2.220446049250313e-16,
    -Inf,
]
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    Inf,
    Inf,
    Inf,
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
]
const ALL_AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
]
const ALL_AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    Inf,
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
        variables = ["σ"],
        equation_indices = [15],
        equations = Expr[
            :((-ρ * log(σ) - (1 - ρ) * log(σ̄)) + log(σ)),
        ],
    ),
    (
        index = 2,
        variables = ["y"],
        equation_indices = [14],
        equations = Expr[
            :(-(k ^ ζ) * l ^ (1 - ζ) * exp(z) + y),
        ],
    ),
    (
        index = 3,
        variables = ["s"],
        equation_indices = [5],
        equations = Expr[
            :(-(V ^ (1 - γ)) + exp(s)),
        ],
    ),
    (
        index = 4,
        variables = ["V", "➕₃"],
        equation_indices = [3, 4],
        equations = Expr[
            :(➕₃ - (V ^ (1 - 1 / ψ) * β + ➕₂ ^ (1 - 1 / ψ) * (1 - β))),
            :(V - ➕₃ ^ (1 / (1 - 1 / ψ))),
        ],
    ),
    (
        index = 5,
        variables = ["➕₂"],
        equation_indices = [2],
        equations = Expr[
            :(➕₂ - c ^ ν * ➕₁ ^ (1 - ν)),
        ],
    ),
    (
        index = 6,
        variables = ["ν"],
        equation_indices = [10],
        equations = Expr[
            :((c * (1 - ν)) / (ν * (1 - l)) - (k ^ ζ * (1 - ζ) * exp(z)) / l ^ ζ),
        ],
    ),
    (
        index = 7,
        variables = ["➕₁"],
        equation_indices = [1],
        equations = Expr[
            :(➕₁ - (1 - l)),
        ],
    ),
    (
        index = 8,
        variables = ["c"],
        equation_indices = [11],
        equations = Expr[
            :((c + i) - k ^ ζ * l ^ (1 - ζ) * exp(z)),
        ],
    ),
    (
        index = 9,
        variables = ["i"],
        equation_indices = [12],
        equations = Expr[
            :((-i - k * (1 - δ)) + k),
        ],
    ),
    (
        index = 10,
        variables = ["Rᶠ"],
        equation_indices = [9],
        equations = Expr[
            :((Rᶠ + 1) - 1 / SDF⁺¹),
        ],
    ),
    (
        index = 11,
        variables = ["SDF⁺¹"],
        equation_indices = [8],
        equations = Expr[
            :(SDF⁺¹ - β),
        ],
    ),
    (
        index = 12,
        variables = ["Rᵏ"],
        equation_indices = [7],
        equations = Expr[
            :((Rᵏ - k ^ (ζ - 1) * l ^ (1 - ζ) * ζ * exp(z)) + δ),
        ],
    ),
    (
        index = 13,
        variables = ["k"],
        equation_indices = [6],
        equations = Expr[
            :(-β * ((k ^ (ζ - 1) * l ^ (1 - ζ) * ζ * exp(z) - δ) + 1) + 1),
        ],
    ),
    (
        index = 14,
        variables = ["z"],
        equation_indices = [13],
        equations = Expr[
            :(-z * λ + z),
        ],
    ),
    (
        index = 15,
        variables = ["l"],
        equation_indices = [16],
        equations = Expr[
            :(l - 1 / 3),
        ],
    ),
]
const BLOCK_EQUATION_ORDER = [15, 14, 5, 3, 4, 2, 10, 1, 11, 12, 9, 8, 7, 6, 13, 16]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[6] = parameters[6]
    complete_parameters[9] = parameters[9]
    complete_parameters[4] = parameters[4]
    complete_parameters[3] = parameters[3]
    complete_parameters[2] = parameters[2]
    complete_parameters[7] = parameters[7]
    complete_parameters[1] = parameters[1]
    complete_parameters[5] = parameters[5]
    complete_parameters[8] = parameters[8]
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[4] - ((1 - complete_parameters[1]) * (solution[5] ^ solution[13] * (1 - solution[8]) ^ (1 - solution[13])) ^ (1 - 1 / complete_parameters[5]) + complete_parameters[1] * solution[4] ^ (1 - 1 / complete_parameters[5])) ^ (1 / (1 - 1 / complete_parameters[5])),
        exp(solution[9]) - solution[4] ^ (1 - complete_parameters[6]),
        1 - (((1 + complete_parameters[2] * exp(solution[11]) * solution[7] ^ (complete_parameters[2] - 1) * solution[8] ^ (1 - complete_parameters[2])) - complete_parameters[3]) * solution[5] * complete_parameters[1] * (((1 - solution[8]) / (1 - solution[8])) ^ (1 - solution[13]) * (solution[5] / solution[5]) ^ solution[13]) ^ (1 - 1 / complete_parameters[5])) / solution[5],
        solution[1] - (complete_parameters[2] * exp(solution[11]) * solution[7] ^ (complete_parameters[2] - 1) * solution[8] ^ (1 - complete_parameters[2]) - complete_parameters[3]),
        solution[3] - (solution[5] * complete_parameters[1] * (((1 - solution[8]) / (1 - solution[8])) ^ (1 - solution[13]) * (solution[5] / solution[5]) ^ solution[13]) ^ (1 - 1 / complete_parameters[5])) / solution[5],
        (1 + solution[2]) - 1 / solution[3],
        (((1 - solution[13]) / solution[13]) * solution[5]) / (1 - solution[8]) - (1 - complete_parameters[2]) * exp(solution[11]) * solution[7] ^ complete_parameters[2] * solution[8] ^ -(complete_parameters[2]),
        (solution[5] + solution[6]) - exp(solution[11]) * solution[7] ^ complete_parameters[2] * solution[8] ^ (1 - complete_parameters[2]),
        solution[7] - (solution[6] + solution[7] * (1 - complete_parameters[3])),
        solution[11] - (complete_parameters[4] * solution[11] + solution[12] * 0),
        solution[10] - exp(solution[11]) * solution[7] ^ complete_parameters[2] * solution[8] ^ (1 - complete_parameters[2]),
        log(solution[12]) - ((1 - complete_parameters[9]) * log(complete_parameters[7]) + complete_parameters[9] * log(solution[12]) + complete_parameters[8] * 0),
        solution[8] - 1 / 3,
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[13] - (1 - solution[8]),
        solution[14] - solution[5] ^ solution[16] * solution[13] ^ (1 - solution[16]),
        solution[15] - (solution[4] ^ (1 - 1 / complete_parameters[5]) * complete_parameters[1] + solution[14] ^ (1 - 1 / complete_parameters[5]) * (1 - complete_parameters[1])),
        solution[4] - solution[15] ^ (1 / (1 - 1 / complete_parameters[5])),
        -(solution[4] ^ (1 - complete_parameters[6])) + exp(solution[9]),
        -(complete_parameters[1]) * ((solution[7] ^ (complete_parameters[2] - 1) * solution[8] ^ (1 - complete_parameters[2]) * complete_parameters[2] * exp(solution[11]) - complete_parameters[3]) + 1) + 1,
        (solution[1] - solution[7] ^ (complete_parameters[2] - 1) * solution[8] ^ (1 - complete_parameters[2]) * complete_parameters[2] * exp(solution[11])) + complete_parameters[3],
        solution[3] - complete_parameters[1],
        (solution[2] + 1) - 1 / solution[3],
        (solution[5] * (1 - solution[16])) / (solution[16] * (1 - solution[8])) - (solution[7] ^ complete_parameters[2] * (1 - complete_parameters[2]) * exp(solution[11])) / solution[8] ^ complete_parameters[2],
        (solution[5] + solution[6]) - solution[7] ^ complete_parameters[2] * solution[8] ^ (1 - complete_parameters[2]) * exp(solution[11]),
        (-(solution[6]) - solution[7] * (1 - complete_parameters[3])) + solution[7],
        -(solution[11]) * complete_parameters[4] + solution[11],
        -(solution[7] ^ complete_parameters[2]) * solution[8] ^ (1 - complete_parameters[2]) * exp(solution[11]) + solution[10],
        (-(complete_parameters[9]) * log(solution[12]) - (1 - complete_parameters[9]) * log(complete_parameters[7])) + log(solution[12]),
        solution[8] - 1 / 3,
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
