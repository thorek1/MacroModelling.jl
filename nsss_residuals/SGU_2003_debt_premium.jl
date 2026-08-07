module SGU_2003_debt_premiumNsssResiduals
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

const MODEL_NAME = "SGU_2003_debt_premium"
const SOURCE_MODEL_FILE = "models/SGU_2003_debt_premium.jl"
const NSSS_SOLUTION_ERROR = 1.3653937842860002e-15
const NSSS_RESIDUAL_NORM = 3.3383219781502926e-15

const PARAMETER_NAMES = [
    "γ",
    "ω",
    "α",
    "ϕ",
    "r̄",
    "δ",
    "ρ",
    "σ_tfp",
    "ψ²",
    "d̄",
]
const PARAMETER_VALUES = Float64[
    2.0,
    1.455,
    0.32,
    0.028,
    0.04,
    0.1,
    0.42,
    0.0129,
    0.000742,
    0.7442,
]
const COMPLETE_PARAMETER_NAMES = [
    "γ",
    "ω",
    "α",
    "ϕ",
    "r̄",
    "δ",
    "ρ",
    "σ_tfp",
    "ψ²",
    "d̄",
    "β",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    2.0,
    1.455,
    0.32,
    0.028,
    0.04,
    0.1,
    0.42,
    0.0129,
    0.000742,
    0.7442,
    0.9615384615384615,
]
const ORIGINAL_SOLUTION_NAMES = [
    "a",
    "c",
    "caʸ",
    "d",
    "h",
    "i",
    "k",
    "r",
    "riskpremium",
    "tbʸ",
    "util",
    "y",
    "λ",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    0.0,
    1.1169507819117181,
    0.0,
    0.7442000000001217,
    1.0074179936054626,
    0.33976852797384277,
    3.397685279738427,
    0.04000000000000009,
    9.020562075079397e-17,
    0.020025734361833705,
    -1.3683490243936776,
    1.486487309885566,
    5.609077101346483,
]
const AUXILIARY_SOLUTION_NAMES = [
    "a",
    "c",
    "caʸ",
    "d",
    "h",
    "i",
    "k",
    "r",
    "riskpremium",
    "tbʸ",
    "util",
    "y",
    "λ",
    "➕₁",
    "➕₂",
    "➕₃",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    0.0,
    1.1169507819117181,
    0.0,
    0.7442000000001217,
    1.0074179936054626,
    0.33976852797384277,
    3.397685279738427,
    0.04000000000000009,
    9.028688907619654e-17,
    0.020025734361833705,
    -1.3683490243936776,
    1.486487309885566,
    5.609077101346483,
    0.42223506320231263,
    1.2168044349891716e-13,
    0.42223506320231263,
]
const ALL_AUXILIARY_VARIABLE_NAMES = [
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    0.42223506320231263,
    1.2168044349891716e-13,
    0.42223506320231263,
    0.0,
    1.0074179936054626,
]
const DEFAULTED_NSSS_SOLUTION_NAMES = [
]
const CALIBRATION_PARAMETER_NAMES = [
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(d - (((1 + r) * d - y) + c + i + (ϕ / 2) * (k - k) ^ 2)),
    :(y - exp(a) * k ^ α * h ^ (1 - α)),
    :(k - (i + k * (1 - δ))),
    :(λ - β * (1 + r) * λ),
    :((c - h ^ ω / ω) ^ -γ - λ),
    :((c - h ^ ω / ω) ^ -γ * h ^ (ω - 1) - (y * (1 - α) * λ) / h),
    :(λ * (1 + ϕ * (k - k)) - β * λ * (((1 + (α * y) / k) - δ) + ϕ * (k - k))),
    :(a - (ρ * a + σ_tfp * 0)),
    :(r - (r̄ + riskpremium)),
    :(riskpremium - ψ² * (exp(d - d̄) - 1)),
    :(tbʸ - (1 - ((ϕ / 2) * (k - k) ^ 2 + c + i) / y)),
    :(caʸ - (1 / y) * (d - d)),
    :(util - ((c - h ^ ω * ω ^ -1) ^ (1 - γ) - 1) / (1 - γ)),
]
const CALIBRATION_EQUATIONS = Expr[
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :((((-c - d * (r + 1)) + d) - i) + y),
    :(-(h ^ (1 - α)) * k ^ α * exp(a) + y),
    :((-i - k * (1 - δ)) + k),
    :(-β * λ * (r + 1) + λ),
    :(➕₁ - (c - h ^ ω / ω)),
    :(-λ + ➕₁ ^ -γ),
    :(h ^ (ω - 1) / ➕₁ ^ γ - (y * λ * (1 - α)) / h),
    :(-β * λ * (-δ + 1 + (y * α) / k) + λ),
    :(-a * ρ + a),
    :((r - riskpremium) - r̄),
    :(➕₂ - (d - d̄)),
    :(riskpremium - ψ² * (exp(➕₂) - 1)),
    :((tbʸ - 1) + (c + i) / y),
    :(caʸ - 0),
    :(➕₃ - (c - h ^ ω / ω)),
    :(util - (➕₃ ^ (1 - γ) - 1) / (1 - γ)),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(d - (((1 + r) * d - y) + c + i + (ϕ / 2) * (k - k) ^ 2)),
    :(y - exp(a) * k ^ α * h ^ (1 - α)),
    :(k - (i + k * (1 - δ))),
    :(λ - β * (1 + r) * λ),
    :((c - h ^ ω / ω) ^ -γ - λ),
    :((c - h ^ ω / ω) ^ -γ * h ^ (ω - 1) - (y * (1 - α) * λ) / h),
    :(λ * (1 + ϕ * (k - k)) - β * λ * (((1 + (α * y) / k) - δ) + ϕ * (k - k))),
    :(a - (ρ * a + σ_tfp * 0)),
    :(r - (r̄ + riskpremium)),
    :(riskpremium - ψ² * (exp(d - d̄) - 1)),
    :(tbʸ - (1 - ((ϕ / 2) * (k - k) ^ 2 + c + i) / y)),
    :(caʸ - (1 / y) * (d - d)),
    :(util - ((c - h ^ ω * ω ^ -1) ^ (1 - γ) - 1) / (1 - γ)),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :((((-c - d * (r + 1)) + d) - i) + y),
    :(-(h ^ (1 - α)) * k ^ α * exp(a) + y),
    :((-i - k * (1 - δ)) + k),
    :(-β * λ * (r + 1) + λ),
    :(➕₁ - (c - h ^ ω / ω)),
    :(-λ + ➕₁ ^ -γ),
    :(h ^ (ω - 1) / ➕₁ ^ γ - (y * λ * (1 - α)) / h),
    :(-β * λ * (-δ + 1 + (y * α) / k) + λ),
    :(-a * ρ + a),
    :((r - riskpremium) - r̄),
    :(➕₂ - (d - d̄)),
    :(riskpremium - ψ² * (exp(➕₂) - 1)),
    :((tbʸ - 1) + (c + i) / y),
    :(caʸ - 0),
    :(➕₃ - (c - h ^ ω / ω)),
    :(util - (➕₃ ^ (1 - γ) - 1) / (1 - γ)),
]

const PARAMETER_DEFINITION_NAMES = [
    "β",
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
    "1 / (1 + r̄)",
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "γ",
    "ω",
    "α",
    "ϕ",
    "r̄",
    "δ",
    "ρ",
    "σ_tfp",
    "ψ²",
    "d̄",
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
]
const ORIGINAL_BOX_CONSTRAINT_NAMES = [
    "a",
    "c",
    "caʸ",
    "d",
    "h",
    "i",
    "k",
    "r",
    "riskpremium",
    "tbʸ",
    "util",
    "y",
    "λ",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -1.0e12,
    2.220446049250313e-16,
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
    Inf,
    Inf,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "a",
    "c",
    "caʸ",
    "d",
    "h",
    "i",
    "k",
    "r",
    "riskpremium",
    "tbʸ",
    "util",
    "y",
    "λ",
    "➕₁",
    "➕₂",
    "➕₃",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -1.0e12,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
]
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    Inf,
    1.0e12,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
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
]
const ALL_AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -1.0e12,
    2.220446049250313e-16,
]
const ALL_AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    1.0e12,
    600.0,
    1.0e12,
    600.0,
    1.0e12,
]

const BLOCKS = [
    (
        index = 1,
        solve_order = 6,
        variables = ["util"],
        previous_solution_names = ["➕₃"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [16],
        equations = Expr[
            :(util - (➕₃ ^ (1 - γ) - 1) / (1 - γ)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["util"],
        previous_solution_values = [0.42223506320231263],
        external_solution_values = Float64[],
        solution_values = [-1.3683490243936776],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 2,
        solve_order = 5,
        variables = ["➕₃"],
        previous_solution_names = ["c", "h"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₅"],
        equation_indices = [15],
        equations = Expr[
            :(➕₃ - (c - ➕₅ ^ ω / ω)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₅ = min(1.0e12, max(eps(), h))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₅ - h)),
        ],
        solution_names = ["➕₃", "➕₅"],
        previous_solution_values = [1.1169507819117181, 1.0074179936054626],
        external_solution_values = Float64[],
        solution_values = [0.42223506320231263, 1.0074179936054626],
        box_lower_bounds = [2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12],
    ),
    (
        index = 3,
        solve_order = 4,
        variables = ["tbʸ"],
        previous_solution_names = ["c", "i", "y"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [13],
        equations = Expr[
            :((tbʸ - 1) + (c + i) / y),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["tbʸ"],
        previous_solution_values = [1.1169507819117181, 0.33976852797384277, 1.486487309885566],
        external_solution_values = Float64[],
        solution_values = [0.020025734361833705],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 4,
        solve_order = 3,
        variables = ["caʸ"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [14],
        equations = Expr[
            :(caʸ - 0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["caʸ"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 5,
        solve_order = 2,
        variables = ["c", "d", "h", "i", "k", "r", "riskpremium", "y", "λ", "➕₁", "➕₂"],
        previous_solution_names = ["a"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₄"],
        equation_indices = [1, 11, 2, 3, 8, 4, 10, 7, 6, 5, 12],
        equations = Expr[
            :((((-c - d * (r + 1)) + d) - i) + y),
            :(➕₂ - (d - d̄)),
            :(-(h ^ (1 - α)) * k ^ α * exp(➕₄) + y),
            :((-i - k * (1 - δ)) + k),
            :(-β * λ * (-δ + 1 + (y * α) / k) + λ),
            :(-β * λ * (r + 1) + λ),
            :((r - riskpremium) - r̄),
            :(h ^ (ω - 1) / ➕₁ ^ γ - (y * λ * (1 - α)) / h),
            :(-λ + ➕₁ ^ -γ),
            :(➕₁ - (c - h ^ ω / ω)),
            :(riskpremium - ψ² * (exp(➕₂) - 1)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₄ = min(600, max(-1.0e12, a))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₄ - a)),
        ],
        solution_names = ["c", "d", "h", "i", "k", "r", "riskpremium", "y", "λ", "➕₁", "➕₂", "➕₄"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [1.1169507819117181, 0.7442000000001217, 1.0074179936054626, 0.33976852797384277, 3.397685279738427, 0.04000000000000009, 9.028688907619654e-17, 1.486487309885566, 5.609077101346483, 0.42223506320231263, 1.2168044349891716e-13, 0.0],
        box_lower_bounds = [-1.0e12, -Inf, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, -Inf, -Inf, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12],
        box_upper_bounds = [1.0e12, Inf, 1.0e12, 1.0e12, 1.0e12, Inf, Inf, 1.0e12, 1.0e12, 1.0e12, 600.0, 600.0],
    ),
    (
        index = 6,
        solve_order = 1,
        variables = ["a"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [9],
        equations = Expr[
            :(-a * ρ + a),
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
const BLOCK_EQUATION_ORDER = [16, 15, 13, 14, 1, 11, 2, 3, 8, 4, 10, 7, 6, 5, 12, 9]
const BLOCK_SOLVE_ORDER = [6, 5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    ["➕₃"],
    ["c", "h"],
    ["c", "i", "y"],
    String[],
    ["a"],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    [0.42223506320231263],
    [1.1169507819117181, 1.0074179936054626],
    [1.1169507819117181, 0.33976852797384277, 1.486487309885566],
    Float64[],
    [0.0],
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
    ["util"],
    ["➕₃", "➕₅"],
    ["tbʸ"],
    ["caʸ"],
    ["c", "d", "h", "i", "k", "r", "riskpremium", "y", "λ", "➕₁", "➕₂", "➕₄"],
    ["a"],
]
const BLOCK_SOLUTION_VALUES = [
    [-1.3683490243936776],
    [0.42223506320231263, 1.0074179936054626],
    [0.020025734361833705],
    [0.0],
    [1.1169507819117181, 0.7442000000001217, 1.0074179936054626, 0.33976852797384277, 3.397685279738427, 0.04000000000000009, 9.028688907619654e-17, 1.486487309885566, 5.609077101346483, 0.42223506320231263, 1.2168044349891716e-13, 0.0],
    [0.0],
]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[1] = parameters[1]
    complete_parameters[3] = parameters[3]
    complete_parameters[7] = parameters[7]
    complete_parameters[2] = parameters[2]
    complete_parameters[6] = parameters[6]
    complete_parameters[9] = parameters[9]
    complete_parameters[5] = parameters[5]
    complete_parameters[10] = parameters[10]
    complete_parameters[8] = parameters[8]
    complete_parameters[4] = parameters[4]
    complete_parameters[11] = 1 / (1 + complete_parameters[5])
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[4] - (((1 + solution[8]) * solution[4] - solution[12]) + solution[2] + solution[6] + (complete_parameters[4] / 2) * (solution[7] - solution[7]) ^ 2),
        solution[12] - exp(solution[1]) * solution[7] ^ complete_parameters[3] * solution[5] ^ (1 - complete_parameters[3]),
        solution[7] - (solution[6] + solution[7] * (1 - complete_parameters[6])),
        solution[13] - complete_parameters[11] * (1 + solution[8]) * solution[13],
        (solution[2] - solution[5] ^ complete_parameters[2] / complete_parameters[2]) ^ -(complete_parameters[1]) - solution[13],
        (solution[2] - solution[5] ^ complete_parameters[2] / complete_parameters[2]) ^ -(complete_parameters[1]) * solution[5] ^ (complete_parameters[2] - 1) - (solution[12] * (1 - complete_parameters[3]) * solution[13]) / solution[5],
        solution[13] * (1 + complete_parameters[4] * (solution[7] - solution[7])) - complete_parameters[11] * solution[13] * (((1 + (complete_parameters[3] * solution[12]) / solution[7]) - complete_parameters[6]) + complete_parameters[4] * (solution[7] - solution[7])),
        solution[1] - (complete_parameters[7] * solution[1] + complete_parameters[8] * 0),
        solution[8] - (complete_parameters[5] + solution[9]),
        solution[9] - complete_parameters[9] * (exp(solution[4] - complete_parameters[10]) - 1),
        solution[10] - (1 - ((complete_parameters[4] / 2) * (solution[7] - solution[7]) ^ 2 + solution[2] + solution[6]) / solution[12]),
        solution[3] - (1 / solution[12]) * (solution[4] - solution[4]),
        solution[11] - ((solution[2] - solution[5] ^ complete_parameters[2] * complete_parameters[2] ^ -1) ^ (1 - complete_parameters[1]) - 1) / (1 - complete_parameters[1]),
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        (((-(solution[2]) - solution[4] * (solution[8] + 1)) + solution[4]) - solution[6]) + solution[12],
        -(solution[5] ^ (1 - complete_parameters[3])) * solution[7] ^ complete_parameters[3] * exp(solution[1]) + solution[12],
        (-(solution[6]) - solution[7] * (1 - complete_parameters[6])) + solution[7],
        -(complete_parameters[11]) * solution[13] * (solution[8] + 1) + solution[13],
        solution[14] - (solution[2] - solution[5] ^ complete_parameters[2] / complete_parameters[2]),
        -(solution[13]) + solution[14] ^ -(complete_parameters[1]),
        solution[5] ^ (complete_parameters[2] - 1) / solution[14] ^ complete_parameters[1] - (solution[12] * solution[13] * (1 - complete_parameters[3])) / solution[5],
        -(complete_parameters[11]) * solution[13] * (-(complete_parameters[6]) + 1 + (solution[12] * complete_parameters[3]) / solution[7]) + solution[13],
        -(solution[1]) * complete_parameters[7] + solution[1],
        (solution[8] - solution[9]) - complete_parameters[5],
        solution[15] - (solution[4] - complete_parameters[10]),
        solution[9] - complete_parameters[9] * (exp(solution[15]) - 1),
        (solution[10] - 1) + (solution[2] + solution[6]) / solution[12],
        solution[3] - 0,
        solution[16] - (solution[2] - solution[5] ^ complete_parameters[2] / complete_parameters[2]),
        solution[11] - (solution[16] ^ (1 - complete_parameters[1]) - 1) / (1 - complete_parameters[1]),
    ]
end

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - (previous_solution[1] ^ (1 - complete_parameters[1]) - 1) / (1 - complete_parameters[1]),
    ]
end

function residuals_block_2(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - (previous_solution[1] - solution[2] ^ complete_parameters[2] / complete_parameters[2]),
        solution[2] - min(1.0e12, max(eps(), previous_solution[2])),
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (solution[1] - 1) + (previous_solution[1] + previous_solution[2]) / previous_solution[3],
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
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 12
    complete_parameters = complete_parameter_values(parameters)
    return [
        (((-(solution[1]) - solution[2] * (solution[6] + 1)) + solution[2]) - solution[4]) + solution[8],
        solution[11] - (solution[2] - complete_parameters[10]),
        -(solution[3] ^ (1 - complete_parameters[3])) * solution[5] ^ complete_parameters[3] * exp(solution[12]) + solution[8],
        (-(solution[4]) - solution[5] * (1 - complete_parameters[6])) + solution[5],
        -(complete_parameters[11]) * solution[9] * (-(complete_parameters[6]) + 1 + (solution[8] * complete_parameters[3]) / solution[5]) + solution[9],
        -(complete_parameters[11]) * solution[9] * (solution[6] + 1) + solution[9],
        (solution[6] - solution[7]) - complete_parameters[5],
        solution[3] ^ (complete_parameters[2] - 1) / solution[10] ^ complete_parameters[1] - (solution[8] * solution[9] * (1 - complete_parameters[3])) / solution[3],
        -(solution[9]) + solution[10] ^ -(complete_parameters[1]),
        solution[10] - (solution[1] - solution[3] ^ complete_parameters[2] / complete_parameters[2]),
        solution[7] - complete_parameters[9] * (exp(solution[11]) - 1),
        solution[12] - min(600, max(-1.0e12, previous_solution[1])),
    ]
end

function residuals_block_6(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) * complete_parameters[7] + solution[1],
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
