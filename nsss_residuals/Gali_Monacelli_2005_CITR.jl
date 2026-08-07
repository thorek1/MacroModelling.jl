module Gali_Monacelli_2005_CITRNsssResiduals
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

const MODEL_NAME = "Gali_Monacelli_2005_CITR"
const SOURCE_MODEL_FILE = "models/Gali_Monacelli_2005_CITR.jl"
const NSSS_SOLUTION_ERROR = 0.0
const NSSS_RESIDUAL_NORM = 0.0

const PARAMETER_NAMES = [
    "σ",
    "η",
    "γ",
    "ϕ",
    "θ",
    "β",
    "α",
    "ϕᵖⁱ",
    "ρᵃ",
    "ρʸ",
]
const PARAMETER_VALUES = Float64[
    1.0,
    1.0,
    1.0,
    3.0,
    0.75,
    0.99,
    0.4,
    1.5,
    0.9,
    0.86,
]
const COMPLETE_PARAMETER_NAMES = [
    "σ",
    "η",
    "γ",
    "ϕ",
    "θ",
    "β",
    "α",
    "ϕᵖⁱ",
    "ρᵃ",
    "ρʸ",
    "Γ",
    "Θ",
    "Ψ",
    "κᵃ",
    "σᵃ",
    "ω",
    "ρ",
    "λ",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    1.0,
    1.0,
    1.0,
    3.0,
    0.75,
    0.99,
    0.4,
    1.5,
    0.9,
    0.86,
    1.0,
    0.0,
    -0.0,
    0.34333333333333343,
    1.0000000000000002,
    1.0,
    0.010101010101010166,
    0.08583333333333336,
]
const ORIGINAL_SOLUTION_NAMES = [
    "a",
    "c",
    "deprec_rate",
    "n",
    "nx",
    "pi",
    "pih",
    "r",
    "real_wage",
    "rnat",
    "s",
    "x",
    "y",
    "ynat",
    "ystar",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
const ORIGINAL_INITIAL_SOLUTION_VALUES = Float64[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
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
    "c",
    "deprec_rate",
    "n",
    "nx",
    "pi",
    "pih",
    "r",
    "real_wage",
    "rnat",
    "s",
    "x",
    "y",
    "ynat",
    "ystar",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
const AUXILIARY_INITIAL_SOLUTION_VALUES = Float64[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
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
const ALL_AUXILIARY_VARIABLE_INITIAL_VALUES = Float64[
]
const DEFAULTED_NSSS_SOLUTION_NAMES = [
]
const CALIBRATION_PARAMETER_NAMES = [
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(x - (x - σᵃ ^ -1 * ((r - pih) - rnat))),
    :(pih - (pih * β + x * κᵃ)),
    :(rnat - (-σᵃ * Γ * (1 - ρᵃ) * a + σᵃ * α * (Θ + Ψ) * (ystar - ystar))),
    :(ynat - (Γ * a + ystar * α * Ψ)),
    :(x - (y - ynat)),
    :(y - (ystar + σᵃ ^ -1 * s)),
    :(pi - (pih + α * (s - s))),
    :(s - ((s + deprec_rate) - pih)),
    :(y - (a + n)),
    :(nx - s * α * (ω / σ - 1)),
    :(y - (c + (s * α * ω) / σ)),
    :(real_wage - (σ * c + n * ϕ)),
    :(a - (ρᵃ * a + 0)),
    :(ystar - (ρʸ * ystar + 0)),
    :(r - pi * ϕᵖⁱ),
]
const CALIBRATION_EQUATIONS = Expr[
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :(((-pih + r) - rnat) / σᵃ),
    :((-pih * β + pih) - x * κᵃ),
    :(a * Γ * σᵃ * (1 - ρᵃ) + rnat),
    :((-a * Γ + ynat) - ystar * Ψ * α),
    :((x - y) + ynat),
    :((-s / σᵃ + y) - ystar),
    :(pi - pih),
    :(-deprec_rate + pih),
    :((-a - n) + y),
    :(nx - s * α * (-1 + ω / σ)),
    :((-c - (s * α * ω) / σ) + y),
    :((-c * σ - n * ϕ) + real_wage),
    :(-a * ρᵃ + a),
    :(-ystar * ρʸ + ystar),
    :(-pi * ϕᵖⁱ + r),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(x - (x - σᵃ ^ -1 * ((r - pih) - rnat))),
    :(pih - (pih * β + x * κᵃ)),
    :(rnat - (-σᵃ * Γ * (1 - ρᵃ) * a + σᵃ * α * (Θ + Ψ) * (ystar - ystar))),
    :(ynat - (Γ * a + ystar * α * Ψ)),
    :(x - (y - ynat)),
    :(y - (ystar + σᵃ ^ -1 * s)),
    :(pi - (pih + α * (s - s))),
    :(s - ((s + deprec_rate) - pih)),
    :(y - (a + n)),
    :(nx - s * α * (ω / σ - 1)),
    :(y - (c + (s * α * ω) / σ)),
    :(real_wage - (σ * c + n * ϕ)),
    :(a - (ρᵃ * a + 0)),
    :(ystar - (ρʸ * ystar + 0)),
    :(r - pi * ϕᵖⁱ),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :(((-pih + r) - rnat) / σᵃ),
    :((-pih * β + pih) - x * κᵃ),
    :(a * Γ * σᵃ * (1 - ρᵃ) + rnat),
    :((-a * Γ + ynat) - ystar * Ψ * α),
    :((x - y) + ynat),
    :((-s / σᵃ + y) - ystar),
    :(pi - pih),
    :(-deprec_rate + pih),
    :((-a - n) + y),
    :(nx - s * α * (-1 + ω / σ)),
    :((-c - (s * α * ω) / σ) + y),
    :((-c * σ - n * ϕ) + real_wage),
    :(-a * ρᵃ + a),
    :(-ystar * ρʸ + ystar),
    :(-pi * ϕᵖⁱ + r),
]

const PARAMETER_DEFINITION_NAMES = [
    "Θ",
    "λ",
    "ρ",
    "ω",
    "σᵃ",
    "Γ",
    "Ψ",
    "κᵃ",
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
    "((1 - α) * (σ * η - 1) + σ * γ) - 1",
    "((1 - β * θ) * (1 - θ)) / θ",
    "1 / β - 1",
    "σ * γ + (1 - α) * (σ * η - 1)",
    "σ / ((1 - α) + α * ω)",
    "(1 + ϕ) / (σᵃ + ϕ)",
    "(-σᵃ * Θ) / (σᵃ + ϕ)",
    "λ * (σᵃ + ϕ)",
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "σ",
    "η",
    "γ",
    "ϕ",
    "θ",
    "β",
    "α",
    "ϕᵖⁱ",
    "ρᵃ",
    "ρʸ",
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
    "deprec_rate",
    "n",
    "nx",
    "pi",
    "pih",
    "r",
    "real_wage",
    "rnat",
    "s",
    "x",
    "y",
    "ynat",
    "ystar",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
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
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "a",
    "c",
    "deprec_rate",
    "n",
    "nx",
    "pi",
    "pih",
    "r",
    "real_wage",
    "rnat",
    "s",
    "x",
    "y",
    "ynat",
    "ystar",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
]
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
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
        solve_order = 13,
        variables = ["real_wage"],
        previous_solution_names = ["c", "n"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [12],
        equations = Expr[
            :((-c * σ - n * ϕ) + real_wage),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["real_wage"],
        previous_solution_values = [0.0, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 2,
        solve_order = 12,
        variables = ["nx"],
        previous_solution_names = ["s"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [10],
        equations = Expr[
            :(nx - s * α * (-1 + ω / σ)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["nx"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 3,
        solve_order = 11,
        variables = ["n"],
        previous_solution_names = ["a", "y"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [9],
        equations = Expr[
            :((-a - n) + y),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["n"],
        previous_solution_values = [0.0, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 4,
        solve_order = 10,
        variables = ["deprec_rate"],
        previous_solution_names = ["pih"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [8],
        equations = Expr[
            :(-deprec_rate + pih),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["deprec_rate"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 5,
        solve_order = 9,
        variables = ["c"],
        previous_solution_names = ["s", "y"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [11],
        equations = Expr[
            :((-c - (s * α * ω) / σ) + y),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["c"],
        previous_solution_values = [0.0, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 6,
        solve_order = 8,
        variables = ["s"],
        previous_solution_names = ["y", "ystar"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [6],
        equations = Expr[
            :((-s / σᵃ + y) - ystar),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["s"],
        previous_solution_values = [0.0, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 7,
        solve_order = 7,
        variables = ["y"],
        previous_solution_names = ["x", "ynat"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [5],
        equations = Expr[
            :((x - y) + ynat),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["y"],
        previous_solution_values = [0.0, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 8,
        solve_order = 6,
        variables = ["ynat"],
        previous_solution_names = ["a", "ystar"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [4],
        equations = Expr[
            :((-a * Γ + ynat) - ystar * Ψ * α),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ynat"],
        previous_solution_values = [0.0, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 9,
        solve_order = 5,
        variables = ["ystar"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [14],
        equations = Expr[
            :(-ystar * ρʸ + ystar),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ystar"],
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
        index = 10,
        solve_order = 4,
        variables = ["x"],
        previous_solution_names = ["pih"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [2],
        equations = Expr[
            :((-pih * β + pih) - x * κᵃ),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["x"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 11,
        solve_order = 3,
        variables = ["pi", "pih", "r"],
        previous_solution_names = ["rnat"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [7, 1, 15],
        equations = Expr[
            :(pi - pih),
            :(((-pih + r) - rnat) / σᵃ),
            :(-pi * ϕᵖⁱ + r),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["pi", "pih", "r"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0, 0.0, 0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 0.0, 0.0],
        box_lower_bounds = [-1.0e12, -1.0e12, -1.0e12],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 12,
        solve_order = 2,
        variables = ["rnat"],
        previous_solution_names = ["a"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [3],
        equations = Expr[
            :(a * Γ * σᵃ * (1 - ρᵃ) + rnat),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["rnat"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 13,
        solve_order = 1,
        variables = ["a"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [13],
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
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
]
const BLOCK_EQUATION_ORDER = [12, 10, 9, 8, 11, 6, 5, 4, 14, 2, 7, 1, 15, 3, 13]
const BLOCK_SOLVE_ORDER = [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    ["c", "n"],
    ["s"],
    ["a", "y"],
    ["pih"],
    ["s", "y"],
    ["y", "ystar"],
    ["x", "ynat"],
    ["a", "ystar"],
    String[],
    ["pih"],
    ["rnat"],
    ["a"],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    [0.0, 0.0],
    [0.0],
    [0.0, 0.0],
    [0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    Float64[],
    [0.0],
    [0.0],
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
]
const BLOCK_SOLUTION_NAMES = [
    ["real_wage"],
    ["nx"],
    ["n"],
    ["deprec_rate"],
    ["c"],
    ["s"],
    ["y"],
    ["ynat"],
    ["ystar"],
    ["x"],
    ["pi", "pih", "r"],
    ["rnat"],
    ["a"],
]
const BLOCK_SOLUTION_VALUES = [
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0, 0.0, 0.0],
    [0.0],
    [0.0],
]
const BLOCK_PREVIOUS_SOLUTION_INITIAL_VALUES = [
    [0.0, 0.0],
    [0.0],
    [0.0, 0.0],
    [0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    Float64[],
    [0.0],
    [0.0],
    [0.0],
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
]
const BLOCK_SOLUTION_INITIAL_VALUES = [
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0, 0.0, 0.0],
    [0.0],
    [0.0],
]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[3] = parameters[3]
    complete_parameters[7] = parameters[7]
    complete_parameters[9] = parameters[9]
    complete_parameters[1] = parameters[1]
    complete_parameters[10] = parameters[10]
    complete_parameters[8] = parameters[8]
    complete_parameters[6] = parameters[6]
    complete_parameters[2] = parameters[2]
    complete_parameters[5] = parameters[5]
    complete_parameters[4] = parameters[4]
    complete_parameters[12] = ((1 - complete_parameters[7]) * (complete_parameters[1] * complete_parameters[2] - 1) + complete_parameters[1] * complete_parameters[3]) - 1
    complete_parameters[18] = ((1 - complete_parameters[6] * complete_parameters[5]) * (1 - complete_parameters[5])) / complete_parameters[5]
    complete_parameters[17] = 1 / complete_parameters[6] - 1
    complete_parameters[16] = complete_parameters[1] * complete_parameters[3] + (1 - complete_parameters[7]) * (complete_parameters[1] * complete_parameters[2] - 1)
    complete_parameters[15] = complete_parameters[1] / ((1 - complete_parameters[7]) + complete_parameters[7] * complete_parameters[16])
    complete_parameters[11] = (1 + complete_parameters[4]) / (complete_parameters[15] + complete_parameters[4])
    complete_parameters[13] = (-(complete_parameters[15]) * complete_parameters[12]) / (complete_parameters[15] + complete_parameters[4])
    complete_parameters[14] = complete_parameters[18] * (complete_parameters[15] + complete_parameters[4])
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[12] - (solution[12] - complete_parameters[15] ^ -1 * ((solution[8] - solution[7]) - solution[10])),
        solution[7] - (solution[7] * complete_parameters[6] + solution[12] * complete_parameters[14]),
        solution[10] - (-(complete_parameters[15]) * complete_parameters[11] * (1 - complete_parameters[9]) * solution[1] + complete_parameters[15] * complete_parameters[7] * (complete_parameters[12] + complete_parameters[13]) * (solution[15] - solution[15])),
        solution[14] - (complete_parameters[11] * solution[1] + solution[15] * complete_parameters[7] * complete_parameters[13]),
        solution[12] - (solution[13] - solution[14]),
        solution[13] - (solution[15] + complete_parameters[15] ^ -1 * solution[11]),
        solution[6] - (solution[7] + complete_parameters[7] * (solution[11] - solution[11])),
        solution[11] - ((solution[11] + solution[3]) - solution[7]),
        solution[13] - (solution[1] + solution[4]),
        solution[5] - solution[11] * complete_parameters[7] * (complete_parameters[16] / complete_parameters[1] - 1),
        solution[13] - (solution[2] + (solution[11] * complete_parameters[7] * complete_parameters[16]) / complete_parameters[1]),
        solution[9] - (complete_parameters[1] * solution[2] + solution[4] * complete_parameters[4]),
        solution[1] - (complete_parameters[9] * solution[1] + 0),
        solution[15] - (complete_parameters[10] * solution[15] + 0),
        solution[8] - solution[6] * complete_parameters[8],
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((-(solution[7]) + solution[8]) - solution[10]) / complete_parameters[15],
        (-(solution[7]) * complete_parameters[6] + solution[7]) - solution[12] * complete_parameters[14],
        solution[1] * complete_parameters[11] * complete_parameters[15] * (1 - complete_parameters[9]) + solution[10],
        (-(solution[1]) * complete_parameters[11] + solution[14]) - solution[15] * complete_parameters[13] * complete_parameters[7],
        (solution[12] - solution[13]) + solution[14],
        (-(solution[11]) / complete_parameters[15] + solution[13]) - solution[15],
        solution[6] - solution[7],
        -(solution[3]) + solution[7],
        (-(solution[1]) - solution[4]) + solution[13],
        solution[5] - solution[11] * complete_parameters[7] * (-1 + complete_parameters[16] / complete_parameters[1]),
        (-(solution[2]) - (solution[11] * complete_parameters[7] * complete_parameters[16]) / complete_parameters[1]) + solution[13],
        (-(solution[2]) * complete_parameters[1] - solution[4] * complete_parameters[4]) + solution[9],
        -(solution[1]) * complete_parameters[9] + solution[1],
        -(solution[15]) * complete_parameters[10] + solution[15],
        -(solution[6]) * complete_parameters[8] + solution[8],
    ]
end

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(previous_solution[1]) * complete_parameters[1] - previous_solution[2] * complete_parameters[4]) + solution[1],
    ]
end

function residuals_block_2(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1] * complete_parameters[7] * (-1 + complete_parameters[16] / complete_parameters[1]),
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(previous_solution[1]) - solution[1]) + previous_solution[2],
    ]
end

function residuals_block_4(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) + previous_solution[1],
    ]
end

function residuals_block_5(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[1]) - (previous_solution[1] * complete_parameters[7] * complete_parameters[16]) / complete_parameters[1]) + previous_solution[2],
    ]
end

function residuals_block_6(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[1]) / complete_parameters[15] + previous_solution[1]) - previous_solution[2],
    ]
end

function residuals_block_7(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (previous_solution[1] - solution[1]) + previous_solution[2],
    ]
end

function residuals_block_8(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(previous_solution[1]) * complete_parameters[11] + solution[1]) - previous_solution[2] * complete_parameters[13] * complete_parameters[7],
    ]
end

function residuals_block_9(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) * complete_parameters[10] + solution[1],
    ]
end

function residuals_block_10(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(previous_solution[1]) * complete_parameters[6] + previous_solution[1]) - solution[1] * complete_parameters[14],
    ]
end

function residuals_block_11(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 3
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - solution[2],
        ((-(solution[2]) + solution[3]) - previous_solution[1]) / complete_parameters[15],
        -(solution[1]) * complete_parameters[8] + solution[3],
    ]
end

function residuals_block_12(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        previous_solution[1] * complete_parameters[11] * complete_parameters[15] * (1 - complete_parameters[9]) + solution[1],
    ]
end

function residuals_block_13(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
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
        residuals_block_7(parameters, previous_solutions[7], external_solutions[7], solutions[7]),
        residuals_block_8(parameters, previous_solutions[8], external_solutions[8], solutions[8]),
        residuals_block_9(parameters, previous_solutions[9], external_solutions[9], solutions[9]),
        residuals_block_10(parameters, previous_solutions[10], external_solutions[10], solutions[10]),
        residuals_block_11(parameters, previous_solutions[11], external_solutions[11], solutions[11]),
        residuals_block_12(parameters, previous_solutions[12], external_solutions[12], solutions[12]),
        residuals_block_13(parameters, previous_solutions[13], external_solutions[13], solutions[13]),
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
export residuals_block_1, residuals_block_2, residuals_block_3, residuals_block_4, residuals_block_5, residuals_block_6, residuals_block_7, residuals_block_8, residuals_block_9, residuals_block_10, residuals_block_11, residuals_block_12, residuals_block_13
end
