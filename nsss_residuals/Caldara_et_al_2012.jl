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
const ORIGINAL_INITIAL_SOLUTION_VALUES = Float64[
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
const AUXILIARY_INITIAL_SOLUTION_VALUES = Float64[
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
    0.666666666666667,
    0.6871386578565634,
    1.4553103490340171,
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
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    0.6666666666666667,
    0.6871386578565634,
    1.4553103490340173,
    0.2062856661496336,
    9.53520261538182,
    0.6871386578565624,
    2.266047927759795e6,
    28.60560784614546,
    0.3333333333333333,
    0.0,
    9.53520261538182,
    0.3333333333333333,
    0.0,
    9.53520261538182,
    0.3333333333333333,
    0.0,
    9.53520261538182,
    0.0,
    0.3333333333333333,
    0.7247305637488348,
    0.6871386578565624,
    9.53520261538182,
    0.3333333333333333,
    0.0,
    0.021,
]
const ALL_AUXILIARY_VARIABLE_INITIAL_VALUES = Float64[
    0.666666666666667,
    0.6871386578565634,
    1.4553103490340171,
    0.2062856661496336,
    9.53520261538182,
    0.6871386578565624,
    2.266047927759795e6,
    28.60560784614546,
    0.3333333333333333,
    0.0,
    9.53520261538182,
    0.3333333333333333,
    0.0,
    9.53520261538182,
    0.3333333333333333,
    0.0,
    9.53520261538182,
    0.0,
    0.3333333333333333,
    0.7247305637488348,
    0.6871386578565624,
    9.53520261538182,
    0.3333333333333333,
    0.0,
    0.021,
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
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -1.0e12,
    2.220446049250313e-16,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -1.0e12,
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
    600.0,
    1.0e12,
    1.0e12,
    600.0,
    1.0e12,
    1.0e12,
    600.0,
    1.0e12,
    600.0,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    600.0,
    1.0e12,
]

const BLOCKS = [
    (
        index = 1,
        solve_order = 15,
        variables = ["σ"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₅"],
        equation_indices = [15],
        equations = Expr[
            :((-ρ * log(σ) - (1 - ρ) * log(➕₂₅)) + log(σ)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₅ = min(1.0e12, max(eps(), σ̄))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₅ - σ̄)),
        ],
        solution_names = ["σ", "➕₂₅"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.021, 0.021],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.021, 0.021],
        box_lower_bounds = [2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12],
    ),
    (
        index = 2,
        solve_order = 14,
        variables = ["y"],
        previous_solution_names = ["k", "l", "z"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₂", "➕₂₃", "➕₂₄"],
        equation_indices = [14],
        equations = Expr[
            :(-(➕₂₂ ^ ζ) * ➕₂₃ ^ (1 - ζ) * exp(➕₂₄) + y),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₂ = min(1.0e12, max(eps(), k))),
            :(➕₂₃ = min(1.0e12, max(eps(), l))),
            :(➕₂₄ = min(600, max(-1.0e12, z))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₂ - k)),
            :(abs(➕₂₃ - l)),
            :(abs(➕₂₄ - z)),
        ],
        solution_names = ["y", "➕₂₂", "➕₂₃", "➕₂₄"],
        previous_solution_values = [9.53520261538182, 0.3333333333333333, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.9116205350103185, 9.53520261538182, 0.3333333333333333, 0.0],
        previous_solution_initial_values = [9.53520261538182, 0.3333333333333333, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.9116205350103185, 9.53520261538182, 0.3333333333333333, 0.0],
        box_lower_bounds = [-Inf, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12],
        box_upper_bounds = [Inf, 1.0e12, 1.0e12, 600.0],
    ),
    (
        index = 3,
        solve_order = 13,
        variables = ["s"],
        previous_solution_names = ["V"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₁"],
        equation_indices = [5],
        equations = Expr[
            :(-(➕₂₁ ^ (1 - γ)) + exp(s)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₁ = min(1.0e12, max(eps(), V))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₁ - V)),
        ],
        solution_names = ["s", "➕₂₁"],
        previous_solution_values = [0.6871386578565624],
        external_solution_values = Float64[],
        solution_values = [14.633547871167153, 0.6871386578565624],
        previous_solution_initial_values = [0.6871386578565624],
        external_solution_initial_values = Float64[],
        solution_initial_values = [14.633547871167153, 0.6871386578565624],
        box_lower_bounds = [-1.0e12, 2.220446049250313e-16],
        box_upper_bounds = [600.0, 1.0e12],
    ),
    (
        index = 4,
        solve_order = 12,
        variables = ["V", "➕₃"],
        previous_solution_names = ["➕₂"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [3, 4],
        equations = Expr[
            :(➕₃ - (V ^ (1 - 1 / ψ) * β + ➕₂ ^ (1 - 1 / ψ) * (1 - β))),
            :(V - ➕₃ ^ (1 / (1 - 1 / ψ))),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["V", "➕₃"],
        previous_solution_values = [0.6871386578565634],
        external_solution_values = Float64[],
        solution_values = [0.6871386578565624, 1.4553103490340173],
        previous_solution_initial_values = [0.6871386578565634],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.6871386578565624, 1.4553103490340171],
        box_lower_bounds = [2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12],
    ),
    (
        index = 5,
        solve_order = 11,
        variables = ["➕₂"],
        previous_solution_names = ["c", "ν", "➕₁"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₀"],
        equation_indices = [2],
        equations = Expr[
            :(➕₂ - ➕₂₀ ^ ν * ➕₁ ^ (1 - ν)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₀ = min(1.0e12, max(eps(), c))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₀ - c)),
        ],
        solution_names = ["➕₂", "➕₂₀"],
        previous_solution_values = [0.7247305637488348, 0.3621843141705121, 0.6666666666666667],
        external_solution_values = Float64[],
        solution_values = [0.6871386578565634, 0.7247305637488348],
        previous_solution_initial_values = [0.7247305637488348, 0.3621843141705121, 0.666666666666667],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.6871386578565634, 0.7247305637488348],
        box_lower_bounds = [2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12],
    ),
    (
        index = 6,
        solve_order = 10,
        variables = ["ν"],
        previous_solution_names = ["c", "k", "l", "z"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₇", "➕₁₈", "➕₁₉"],
        equation_indices = [10],
        equations = Expr[
            :((c * (1 - ν)) / (ν * (1 - l)) - (➕₁₇ ^ ζ * (1 - ζ) * exp(➕₁₈)) / ➕₁₉ ^ ζ),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₇ = min(1.0e12, max(eps(), k))),
            :(➕₁₈ = min(600, max(-1.0e12, z))),
            :(➕₁₉ = min(1.0e12, max(eps(), l))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₇ - k)),
            :(abs(➕₁₈ - z)),
            :(abs(➕₁₉ - l)),
        ],
        solution_names = ["ν", "➕₁₇", "➕₁₈", "➕₁₉"],
        previous_solution_values = [0.7247305637488348, 9.53520261538182, 0.3333333333333333, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.3621843141705121, 9.53520261538182, 0.0, 0.3333333333333333],
        previous_solution_initial_values = [0.7247305637488348, 9.53520261538182, 0.3333333333333333, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.3621843141705121, 9.53520261538182, 0.0, 0.3333333333333333],
        box_lower_bounds = [-Inf, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12, 600.0, 1.0e12],
    ),
    (
        index = 7,
        solve_order = 9,
        variables = ["➕₁"],
        previous_solution_names = ["l"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [1],
        equations = Expr[
            :(➕₁ - (1 - l)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["➕₁"],
        previous_solution_values = [0.3333333333333333],
        external_solution_values = Float64[],
        solution_values = [0.6666666666666667],
        previous_solution_initial_values = [0.3333333333333333],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.666666666666667],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 8,
        solve_order = 8,
        variables = ["c"],
        previous_solution_names = ["i", "k", "l", "z"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₄", "➕₁₅", "➕₁₆"],
        equation_indices = [11],
        equations = Expr[
            :((c + i) - ➕₁₄ ^ ζ * ➕₁₅ ^ (1 - ζ) * exp(➕₁₆)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₄ = min(1.0e12, max(eps(), k))),
            :(➕₁₅ = min(1.0e12, max(eps(), l))),
            :(➕₁₆ = min(600, max(-1.0e12, z))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₄ - k)),
            :(abs(➕₁₅ - l)),
            :(abs(➕₁₆ - z)),
        ],
        solution_names = ["c", "➕₁₄", "➕₁₅", "➕₁₆"],
        previous_solution_values = [0.18688997126148366, 9.53520261538182, 0.3333333333333333, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.7247305637488348, 9.53520261538182, 0.3333333333333333, 0.0],
        previous_solution_initial_values = [0.18688997126148366, 9.53520261538182, 0.3333333333333333, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.7247305637488348, 9.53520261538182, 0.3333333333333333, 0.0],
        box_lower_bounds = [-Inf, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12],
        box_upper_bounds = [Inf, 1.0e12, 1.0e12, 600.0],
    ),
    (
        index = 9,
        solve_order = 7,
        variables = ["i"],
        previous_solution_names = ["k"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [12],
        equations = Expr[
            :((-i - k * (1 - δ)) + k),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["i"],
        previous_solution_values = [9.53520261538182],
        external_solution_values = Float64[],
        solution_values = [0.18688997126148366],
        previous_solution_initial_values = [9.53520261538182],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.18688997126148366],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 10,
        solve_order = 6,
        variables = ["Rᶠ"],
        previous_solution_names = ["SDF⁺¹"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [9],
        equations = Expr[
            :((Rᶠ + 1) - 1 / SDF⁺¹),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["Rᶠ"],
        previous_solution_values = [0.991],
        external_solution_values = Float64[],
        solution_values = [0.009081735620585276],
        previous_solution_initial_values = [0.991],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.009081735620585276],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 11,
        solve_order = 5,
        variables = ["SDF⁺¹"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [8],
        equations = Expr[
            :(SDF⁺¹ - β),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["SDF⁺¹"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.991],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.991],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 12,
        solve_order = 4,
        variables = ["Rᵏ"],
        previous_solution_names = ["k", "l", "z"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₁", "➕₁₂", "➕₁₃"],
        equation_indices = [7],
        equations = Expr[
            :((Rᵏ - ➕₁₁ ^ (ζ - 1) * ➕₁₂ ^ (1 - ζ) * ζ * exp(➕₁₃)) + δ),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₁ = min(1.0e12, max(eps(), k))),
            :(➕₁₂ = min(1.0e12, max(eps(), l))),
            :(➕₁₃ = min(600, max(-1.0e12, z))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₁ - k)),
            :(abs(➕₁₂ - l)),
            :(abs(➕₁₃ - z)),
        ],
        solution_names = ["Rᵏ", "➕₁₁", "➕₁₂", "➕₁₃"],
        previous_solution_values = [9.53520261538182, 0.3333333333333333, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.009081735620585375, 9.53520261538182, 0.3333333333333333, 0.0],
        previous_solution_initial_values = [9.53520261538182, 0.3333333333333333, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.009081735620585375, 9.53520261538182, 0.3333333333333333, 0.0],
        box_lower_bounds = [-Inf, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12],
        box_upper_bounds = [Inf, 1.0e12, 1.0e12, 600.0],
    ),
    (
        index = 13,
        solve_order = 3,
        variables = ["k"],
        previous_solution_names = ["l", "z"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₉", "➕₁₀"],
        equation_indices = [6],
        equations = Expr[
            :(-β * ((k ^ (ζ - 1) * ➕₉ ^ (1 - ζ) * ζ * exp(➕₁₀) - δ) + 1) + 1),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₉ = min(1.0e12, max(eps(), l))),
            :(➕₁₀ = min(600, max(-1.0e12, z))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₉ - l)),
            :(abs(➕₁₀ - z)),
        ],
        solution_names = ["k", "➕₉", "➕₁₀"],
        previous_solution_values = [0.3333333333333333, 0.0],
        external_solution_values = Float64[],
        solution_values = [9.53520261538182, 0.3333333333333333, 0.0],
        previous_solution_initial_values = [0.3333333333333333, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [9.53520261538182, 0.3333333333333333, 0.0],
        box_lower_bounds = [2.220446049250313e-16, 2.220446049250313e-16, -1.0e12],
        box_upper_bounds = [1.0e12, 1.0e12, 600.0],
    ),
    (
        index = 14,
        solve_order = 2,
        variables = ["z"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [13],
        equations = Expr[
            :(-z * λ + z),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["z"],
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
        variables = ["l"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [16],
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
]
const BLOCK_EQUATION_ORDER = [15, 14, 5, 3, 4, 2, 10, 1, 11, 12, 9, 8, 7, 6, 13, 16]
const BLOCK_SOLVE_ORDER = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    String[],
    ["k", "l", "z"],
    ["V"],
    ["➕₂"],
    ["c", "ν", "➕₁"],
    ["c", "k", "l", "z"],
    ["l"],
    ["i", "k", "l", "z"],
    ["k"],
    ["SDF⁺¹"],
    String[],
    ["k", "l", "z"],
    ["l", "z"],
    String[],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    Float64[],
    [9.53520261538182, 0.3333333333333333, 0.0],
    [0.6871386578565624],
    [0.6871386578565634],
    [0.7247305637488348, 0.3621843141705121, 0.6666666666666667],
    [0.7247305637488348, 9.53520261538182, 0.3333333333333333, 0.0],
    [0.3333333333333333],
    [0.18688997126148366, 9.53520261538182, 0.3333333333333333, 0.0],
    [9.53520261538182],
    [0.991],
    Float64[],
    [9.53520261538182, 0.3333333333333333, 0.0],
    [0.3333333333333333, 0.0],
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
    ["σ", "➕₂₅"],
    ["y", "➕₂₂", "➕₂₃", "➕₂₄"],
    ["s", "➕₂₁"],
    ["V", "➕₃"],
    ["➕₂", "➕₂₀"],
    ["ν", "➕₁₇", "➕₁₈", "➕₁₉"],
    ["➕₁"],
    ["c", "➕₁₄", "➕₁₅", "➕₁₆"],
    ["i"],
    ["Rᶠ"],
    ["SDF⁺¹"],
    ["Rᵏ", "➕₁₁", "➕₁₂", "➕₁₃"],
    ["k", "➕₉", "➕₁₀"],
    ["z"],
    ["l"],
]
const BLOCK_SOLUTION_VALUES = [
    [0.021, 0.021],
    [0.9116205350103185, 9.53520261538182, 0.3333333333333333, 0.0],
    [14.633547871167153, 0.6871386578565624],
    [0.6871386578565624, 1.4553103490340173],
    [0.6871386578565634, 0.7247305637488348],
    [0.3621843141705121, 9.53520261538182, 0.0, 0.3333333333333333],
    [0.6666666666666667],
    [0.7247305637488348, 9.53520261538182, 0.3333333333333333, 0.0],
    [0.18688997126148366],
    [0.009081735620585276],
    [0.991],
    [0.009081735620585375, 9.53520261538182, 0.3333333333333333, 0.0],
    [9.53520261538182, 0.3333333333333333, 0.0],
    [0.0],
    [0.3333333333333333],
]
const BLOCK_PREVIOUS_SOLUTION_INITIAL_VALUES = [
    Float64[],
    [9.53520261538182, 0.3333333333333333, 0.0],
    [0.6871386578565624],
    [0.6871386578565634],
    [0.7247305637488348, 0.3621843141705121, 0.666666666666667],
    [0.7247305637488348, 9.53520261538182, 0.3333333333333333, 0.0],
    [0.3333333333333333],
    [0.18688997126148366, 9.53520261538182, 0.3333333333333333, 0.0],
    [9.53520261538182],
    [0.991],
    Float64[],
    [9.53520261538182, 0.3333333333333333, 0.0],
    [0.3333333333333333, 0.0],
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
    [0.021, 0.021],
    [0.9116205350103185, 9.53520261538182, 0.3333333333333333, 0.0],
    [14.633547871167153, 0.6871386578565624],
    [0.6871386578565624, 1.4553103490340171],
    [0.6871386578565634, 0.7247305637488348],
    [0.3621843141705121, 9.53520261538182, 0.0, 0.3333333333333333],
    [0.666666666666667],
    [0.7247305637488348, 9.53520261538182, 0.3333333333333333, 0.0],
    [0.18688997126148366],
    [0.009081735620585276],
    [0.991],
    [0.009081735620585375, 9.53520261538182, 0.3333333333333333, 0.0],
    [9.53520261538182, 0.3333333333333333, 0.0],
    [0.0],
    [0.3333333333333333],
]

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

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(complete_parameters[9]) * log(solution[1]) - (1 - complete_parameters[9]) * log(solution[2])) + log(solution[1]),
        solution[2] - min(1.0e12, max(eps(), complete_parameters[7])),
    ]
end

function residuals_block_2(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 4
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[2] ^ complete_parameters[2]) * solution[3] ^ (1 - complete_parameters[2]) * exp(solution[4]) + solution[1],
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
        solution[3] - min(1.0e12, max(eps(), previous_solution[2])),
        solution[4] - min(600, max(-1.0e12, previous_solution[3])),
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[2] ^ (1 - complete_parameters[6])) + exp(solution[1]),
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
        solution[2] - (solution[1] ^ (1 - 1 / complete_parameters[5]) * complete_parameters[1] + previous_solution[1] ^ (1 - 1 / complete_parameters[5]) * (1 - complete_parameters[1])),
        solution[1] - solution[2] ^ (1 / (1 - 1 / complete_parameters[5])),
    ]
end

function residuals_block_5(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - solution[2] ^ previous_solution[2] * previous_solution[3] ^ (1 - previous_solution[2]),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_6(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 4
    @assert length(external_solution) == 0
    @assert length(solution) == 4
    complete_parameters = complete_parameter_values(parameters)
    return [
        (previous_solution[1] * (1 - solution[1])) / (solution[1] * (1 - previous_solution[3])) - (solution[2] ^ complete_parameters[2] * (1 - complete_parameters[2]) * exp(solution[3])) / solution[4] ^ complete_parameters[2],
        solution[2] - min(1.0e12, max(eps(), previous_solution[2])),
        solution[3] - min(600, max(-1.0e12, previous_solution[4])),
        solution[4] - min(1.0e12, max(eps(), previous_solution[3])),
    ]
end

function residuals_block_7(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - (1 - previous_solution[1]),
    ]
end

function residuals_block_8(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 4
    @assert length(external_solution) == 0
    @assert length(solution) == 4
    complete_parameters = complete_parameter_values(parameters)
    return [
        (solution[1] + previous_solution[1]) - solution[2] ^ complete_parameters[2] * solution[3] ^ (1 - complete_parameters[2]) * exp(solution[4]),
        solution[2] - min(1.0e12, max(eps(), previous_solution[2])),
        solution[3] - min(1.0e12, max(eps(), previous_solution[3])),
        solution[4] - min(600, max(-1.0e12, previous_solution[4])),
    ]
end

function residuals_block_9(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[1]) - previous_solution[1] * (1 - complete_parameters[3])) + previous_solution[1],
    ]
end

function residuals_block_10(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (solution[1] + 1) - 1 / previous_solution[1],
    ]
end

function residuals_block_11(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - complete_parameters[1],
    ]
end

function residuals_block_12(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 4
    complete_parameters = complete_parameter_values(parameters)
    return [
        (solution[1] - solution[2] ^ (complete_parameters[2] - 1) * solution[3] ^ (1 - complete_parameters[2]) * complete_parameters[2] * exp(solution[4])) + complete_parameters[3],
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
        solution[3] - min(1.0e12, max(eps(), previous_solution[2])),
        solution[4] - min(600, max(-1.0e12, previous_solution[3])),
    ]
end

function residuals_block_13(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 3
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[1]) * ((solution[1] ^ (complete_parameters[2] - 1) * solution[2] ^ (1 - complete_parameters[2]) * complete_parameters[2] * exp(solution[3]) - complete_parameters[3]) + 1) + 1,
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
        solution[3] - min(600, max(-1.0e12, previous_solution[2])),
    ]
end

function residuals_block_14(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) * complete_parameters[4] + solution[1],
    ]
end

function residuals_block_15(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 1 / 3,
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
