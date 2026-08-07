module JQ_2012_RBCNsssResiduals
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

const MODEL_NAME = "JQ_2012_RBC"
const SOURCE_MODEL_FILE = "models/JQ_2012_RBC.jl"
const NSSS_SOLUTION_ERROR = 2.220462989844635e-16
const NSSS_RESIDUAL_NORM = 5.09868547704787e-16

const PARAMETER_NAMES = [
    "BY_ratio",
    "n̄",
    "z̄",
    "β",
    "σ",
    "θ",
    "δ",
    "τ",
    "κ",
    "A¹¹",
    "A¹²",
    "A²¹",
    "A²²",
    "σᶻ",
    "σˣⁱ",
]
const PARAMETER_VALUES = Float64[
    3.36,
    0.3,
    1.0,
    0.9825,
    1.0,
    0.36,
    0.025,
    0.35,
    0.146,
    0.9457,
    -0.0091,
    0.0321,
    0.9703,
    0.0045,
    0.0098,
]
const COMPLETE_PARAMETER_NAMES = [
    "BY_ratio",
    "n̄",
    "z̄",
    "β",
    "σ",
    "θ",
    "δ",
    "τ",
    "κ",
    "A¹¹",
    "A¹²",
    "A²¹",
    "A²²",
    "σᶻ",
    "σˣⁱ",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    3.36,
    0.3,
    1.0,
    0.9825,
    1.0,
    0.36,
    0.025,
    0.35,
    0.146,
    0.9457,
    -0.0091,
    0.0321,
    0.9703,
    0.0045,
    0.0098,
]
const ORIGINAL_SOLUTION_NAMES = [
    "R",
    "b",
    "c",
    "d",
    "i",
    "k",
    "n",
    "r",
    "v",
    "w",
    "y",
    "z",
    "μ",
    "ξ",
    "ξ̄",
    "α",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    1.0115776081424936,
    3.6358259916618088,
    0.8111657946199858,
    0.11480054308526577,
    0.2519886806204091,
    10.079547224816363,
    0.3,
    0.017811704834605608,
    6.5600310334437735,
    2.1825095165016286,
    1.063154475240395,
    1.0,
    0.03772089598850499,
    0.16337753022029997,
    0.16337753022029997,
    1.8834086344418184,
]
const ORIGINAL_INITIAL_SOLUTION_VALUES = Float64[
    1.0115776081424936,
    0.0,
    -1.25e10,
    -1.25e10,
    1.25e10,
    5.0e11,
    0.3,
    0.017811704834605608,
    -7.14285714285716e11,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
const AUXILIARY_SOLUTION_NAMES = [
    "R",
    "b",
    "c",
    "d",
    "i",
    "k",
    "n",
    "r",
    "v",
    "w",
    "y",
    "z",
    "μ",
    "ξ",
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "ξ̄",
    "α",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    1.0115776081424936,
    3.6358259916618088,
    0.8111657946199858,
    0.11480054308526577,
    0.2519886806204091,
    10.079547224816363,
    0.3,
    0.017811704834605608,
    6.5600310334437735,
    2.1825095165016286,
    1.063154475240395,
    1.0,
    0.03772089598850499,
    0.16337753022029997,
    1.0,
    1.0,
    1.0,
    1.0,
    0.16337753022029997,
    1.8834086344418184,
]
const AUXILIARY_INITIAL_SOLUTION_VALUES = Float64[
    1.0115776081424936,
    0.0,
    -1.25e10,
    -1.25e10,
    1.25e10,
    5.0e11,
    0.3,
    0.017811704834605608,
    -7.14285714285716e11,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    0.0,
    0.0,
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
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    1.0,
    1.0,
    1.0,
    1.0,
    0.3,
    10.079547224816363,
    0.3,
    0.8111657946199858,
    0.3,
    0.8111657946199858,
]
const ALL_AUXILIARY_VARIABLE_INITIAL_VALUES = Float64[
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    0.3,
    5.0e11,
    0.3,
    2.220446049250313e-16,
    0.3,
    2.220446049250313e-16,
]
const DEFAULTED_NSSS_SOLUTION_NAMES = [
]
const CALIBRATION_PARAMETER_NAMES = [
    "ξ̄",
    "α",
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(w / c ^ σ - α / (1 - n)),
    :(c ^ -σ - ((β * (R - τ)) / (1 - τ)) * c ^ -σ),
    :((((w * n + b) - b / R) + d) - c),
    :((1 - θ) * z * k ^ θ * n ^ -θ - w / (1 - μ * (1 + κ * (d - d) * 2))),
    :((((β * (c / c) ^ σ * (1 + κ * (d - d) * 2)) / (1 + κ * (d - d) * 2)) * ((1 - δ) + θ * (1 - (1 + κ * (d - d) * 2) * μ) * z * k ^ (θ - 1) * n ^ (1 - θ)) + (1 + κ * (d - d) * 2) * μ * ξ) - 1),
    :((((1 + κ * (d - d) * 2) / (1 + κ * (d - d) * 2)) * (c / c) ^ σ * β * R + ((1 + κ * (d - d) * 2) * μ * ξ * R * (1 - τ)) / (R - τ)) - 1),
    :(((((b / R + k * (1 - δ) + z * k ^ θ * n ^ (1 - θ)) - w * n) - b) - k) - (d + κ * (d - d) ^ 2)),
    :(ξ * (k - (b * (1 - τ)) / (R - τ)) - z * k ^ θ * n ^ (1 - θ)),
    :(log(z / z̄) - (A¹¹ * log(z / z̄) + A¹² * log(ξ / ξ̄) + σᶻ * 0)),
    :(log(ξ / ξ̄) - (log(z / z̄) * A²¹ + log(ξ / ξ̄) * A²² + σˣⁱ * 0)),
    :(y - z * k ^ θ * n ^ (1 - θ)),
    :(k - (k * (1 - δ) + i)),
    :(v - (d + ((c * β) / c) * v)),
    :((1 + r) - (R - τ) / (1 - τ)),
]
const CALIBRATION_EQUATIONS = Expr[
    :(b / (y * (1 + r)) - BY_ratio),
    :(n - n̄),
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :(-α / (1 - n) + w / c ^ σ),
    :((-β * (R - τ)) / (c ^ σ * (1 - τ)) + c ^ -σ),
    :(((b - c) + d + n * w) - b / R),
    :((k ^ θ * z * (1 - θ)) / n ^ θ - w / (1 - μ)),
    :((β * ((k ^ (θ - 1) * n ^ (1 - θ) * z * θ * (1 - μ) - δ) + 1) + μ * ξ) - 1),
    :((R * β + (R * μ * ξ * (1 - τ)) / (R - τ)) - 1),
    :((((((-b - d) + k * (1 - δ)) - k) + k ^ θ * n ^ (1 - θ) * z) - n * w) + b / R),
    :(-(k ^ θ) * n ^ (1 - θ) * z + ξ * ((-b * (1 - τ)) / (R - τ) + k)),
    :(➕₁ - z / z̄),
    :(➕₂ - z / z̄),
    :(➕₃ - ξ / ξ̄),
    :((-A¹² * log(➕₃) - A¹¹ * log(➕₂)) + log(➕₁)),
    :(➕₄ - ξ / ξ̄),
    :((-A²² * log(➕₃) - A²¹ * log(➕₂)) + log(➕₄)),
    :(-(k ^ θ) * n ^ (1 - θ) * z + y),
    :((-i - k * (1 - δ)) + k),
    :((-d - v * β) + v),
    :((r + 1) - (R - τ) / (1 - τ)),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(w / c ^ σ - α / (1 - n)),
    :(c ^ -σ - ((β * (R - τ)) / (1 - τ)) * c ^ -σ),
    :((((w * n + b) - b / R) + d) - c),
    :((1 - θ) * z * k ^ θ * n ^ -θ - w / (1 - μ * (1 + κ * (d - d) * 2))),
    :((((β * (c / c) ^ σ * (1 + κ * (d - d) * 2)) / (1 + κ * (d - d) * 2)) * ((1 - δ) + θ * (1 - (1 + κ * (d - d) * 2) * μ) * z * k ^ (θ - 1) * n ^ (1 - θ)) + (1 + κ * (d - d) * 2) * μ * ξ) - 1),
    :((((1 + κ * (d - d) * 2) / (1 + κ * (d - d) * 2)) * (c / c) ^ σ * β * R + ((1 + κ * (d - d) * 2) * μ * ξ * R * (1 - τ)) / (R - τ)) - 1),
    :(((((b / R + k * (1 - δ) + z * k ^ θ * n ^ (1 - θ)) - w * n) - b) - k) - (d + κ * (d - d) ^ 2)),
    :(ξ * (k - (b * (1 - τ)) / (R - τ)) - z * k ^ θ * n ^ (1 - θ)),
    :(log(z / z̄) - (A¹¹ * log(z / z̄) + A¹² * log(ξ / ξ̄) + σᶻ * 0)),
    :(log(ξ / ξ̄) - (log(z / z̄) * A²¹ + log(ξ / ξ̄) * A²² + σˣⁱ * 0)),
    :(y - z * k ^ θ * n ^ (1 - θ)),
    :(k - (k * (1 - δ) + i)),
    :(v - (d + ((c * β) / c) * v)),
    :((1 + r) - (R - τ) / (1 - τ)),
    :(b / (y * (1 + r)) - BY_ratio),
    :(n - n̄),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :(-α / (1 - n) + w / c ^ σ),
    :((-β * (R - τ)) / (c ^ σ * (1 - τ)) + c ^ -σ),
    :(((b - c) + d + n * w) - b / R),
    :((k ^ θ * z * (1 - θ)) / n ^ θ - w / (1 - μ)),
    :((β * ((k ^ (θ - 1) * n ^ (1 - θ) * z * θ * (1 - μ) - δ) + 1) + μ * ξ) - 1),
    :((R * β + (R * μ * ξ * (1 - τ)) / (R - τ)) - 1),
    :((((((-b - d) + k * (1 - δ)) - k) + k ^ θ * n ^ (1 - θ) * z) - n * w) + b / R),
    :(-(k ^ θ) * n ^ (1 - θ) * z + ξ * ((-b * (1 - τ)) / (R - τ) + k)),
    :(➕₁ - z / z̄),
    :(➕₂ - z / z̄),
    :(➕₃ - ξ / ξ̄),
    :((-A¹² * log(➕₃) - A¹¹ * log(➕₂)) + log(➕₁)),
    :(➕₄ - ξ / ξ̄),
    :((-A²² * log(➕₃) - A²¹ * log(➕₂)) + log(➕₄)),
    :(-(k ^ θ) * n ^ (1 - θ) * z + y),
    :((-i - k * (1 - δ)) + k),
    :((-d - v * β) + v),
    :((r + 1) - (R - τ) / (1 - τ)),
    :(b / (y * (1 + r)) - BY_ratio),
    :(n - n̄),
]

const PARAMETER_DEFINITION_NAMES = [
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "BY_ratio",
    "n̄",
    "z̄",
    "β",
    "σ",
    "θ",
    "δ",
    "τ",
    "κ",
    "A¹¹",
    "A¹²",
    "A²¹",
    "A²²",
    "σᶻ",
    "σˣⁱ",
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
]
const ORIGINAL_BOX_CONSTRAINT_NAMES = [
    "R",
    "b",
    "c",
    "d",
    "i",
    "k",
    "n",
    "r",
    "v",
    "w",
    "y",
    "z",
    "μ",
    "ξ",
    "ξ̄",
    "α",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
]
const ORIGINAL_BOX_UPPER_BOUNDS = Float64[
    Inf,
    1.0e12,
    Inf,
    Inf,
    Inf,
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
    Inf,
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "R",
    "b",
    "c",
    "d",
    "i",
    "k",
    "n",
    "r",
    "v",
    "w",
    "y",
    "z",
    "μ",
    "ξ",
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "ξ̄",
    "α",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -1.0e12,
    -Inf,
]
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    Inf,
    1.0e12,
    Inf,
    Inf,
    Inf,
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
    1.0e12,
    1.0e12,
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
]

const BLOCKS = [
    (
        index = 1,
        solve_order = 5,
        variables = ["α"],
        previous_solution_names = ["c", "n", "w"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₀"],
        equation_indices = [1],
        equations = Expr[
            :(-α / (1 - n) + w / ➕₁₀ ^ σ),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₀ = min(1.0e12, max(eps(), c))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₀ - c)),
        ],
        solution_names = ["α", "➕₁₀"],
        previous_solution_values = [0.8111657946199858, 0.3, 2.1825095165016286],
        external_solution_values = Float64[],
        solution_values = [1.8834086344418184, 0.8111657946199858],
        previous_solution_initial_values = [-1.25e10, 0.3, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 2.220446049250313e-16],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 2,
        solve_order = 4,
        variables = ["v"],
        previous_solution_names = ["d"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [17],
        equations = Expr[
            :((-d - v * β) + v),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["v"],
        previous_solution_values = [0.11480054308526577],
        external_solution_values = Float64[],
        solution_values = [6.5600310334437735],
        previous_solution_initial_values = [-1.25e10],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-7.14285714285716e11],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 3,
        solve_order = 3,
        variables = ["i"],
        previous_solution_names = ["k"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [16],
        equations = Expr[
            :((-i - k * (1 - δ)) + k),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["i"],
        previous_solution_values = [10.079547224816363],
        external_solution_values = Float64[],
        solution_values = [0.2519886806204091],
        previous_solution_initial_values = [5.0e11],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.25e10],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 4,
        solve_order = 2,
        variables = ["R", "b", "c", "d", "k", "r", "w", "y", "z", "μ", "ξ", "ξ̄", "➕₁", "➕₂", "➕₃", "➕₄"],
        previous_solution_names = ["n"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₉"],
        equation_indices = [8, 19, 2, 3, 4, 18, 7, 15, 5, 6, 11, 9, 10, 12, 14, 13],
        equations = Expr[
            :(-(k ^ θ) * ➕₉ ^ (1 - θ) * z + ξ * ((-b * (1 - τ)) / (R - τ) + k)),
            :(b / (y * (1 + r)) - BY_ratio),
            :((-β * (R - τ)) / (c ^ σ * (1 - τ)) + c ^ -σ),
            :(((b - c) + d + n * w) - b / R),
            :((k ^ θ * z * (1 - θ)) / ➕₉ ^ θ - w / (1 - μ)),
            :((r + 1) - (R - τ) / (1 - τ)),
            :((((((-b - d) + k * (1 - δ)) - k) + k ^ θ * ➕₉ ^ (1 - θ) * z) - n * w) + b / R),
            :(-(k ^ θ) * ➕₉ ^ (1 - θ) * z + y),
            :((β * ((k ^ (θ - 1) * ➕₉ ^ (1 - θ) * z * θ * (1 - μ) - δ) + 1) + μ * ξ) - 1),
            :((R * β + (R * μ * ξ * (1 - τ)) / (R - τ)) - 1),
            :(➕₃ - ξ / ξ̄),
            :(➕₁ - z / z̄),
            :(➕₂ - z / z̄),
            :((-A¹² * log(➕₃) - A¹¹ * log(➕₂)) + log(➕₁)),
            :((-A²² * log(➕₃) - A²¹ * log(➕₂)) + log(➕₄)),
            :(➕₄ - ξ / ξ̄),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₉ = min(1.0e12, max(eps(), n))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₉ - n)),
        ],
        solution_names = ["R", "b", "c", "d", "k", "r", "w", "y", "z", "μ", "ξ", "ξ̄", "➕₁", "➕₂", "➕₃", "➕₄", "➕₉"],
        previous_solution_values = [0.3],
        external_solution_values = Float64[],
        solution_values = [1.0115776081424936, 3.6358259916618088, 0.8111657946199858, 0.11480054308526577, 10.079547224816363, 0.017811704834605608, 2.1825095165016286, 1.063154475240395, 1.0, 0.03772089598850499, 0.16337753022029997, 0.16337753022029997, 1.0, 1.0, 1.0, 1.0, 0.3],
        previous_solution_initial_values = [0.3],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0115776081424936, 0.0, -1.25e10, -1.25e10, 5.0e11, 0.017811704834605608, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 0.3],
        box_lower_bounds = [-Inf, -1.0e12, 2.220446049250313e-16, -Inf, 2.220446049250313e-16, -Inf, -Inf, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12, 1.0e12, Inf, 1.0e12, Inf, Inf, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 5,
        solve_order = 1,
        variables = ["n"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [20],
        equations = Expr[
            :(n - n̄),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["n"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.3],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.3],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
]
const BLOCK_EQUATION_ORDER = [1, 17, 16, 8, 19, 2, 3, 4, 18, 7, 15, 5, 6, 11, 9, 10, 12, 14, 13, 20]
const BLOCK_SOLVE_ORDER = [5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    ["c", "n", "w"],
    ["d"],
    ["k"],
    ["n"],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    [0.8111657946199858, 0.3, 2.1825095165016286],
    [0.11480054308526577],
    [10.079547224816363],
    [0.3],
    Float64[],
]
const BLOCK_EXTERNAL_SOLUTION_NAMES = [
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
]
const BLOCK_SOLUTION_NAMES = [
    ["α", "➕₁₀"],
    ["v"],
    ["i"],
    ["R", "b", "c", "d", "k", "r", "w", "y", "z", "μ", "ξ", "ξ̄", "➕₁", "➕₂", "➕₃", "➕₄", "➕₉"],
    ["n"],
]
const BLOCK_SOLUTION_VALUES = [
    [1.8834086344418184, 0.8111657946199858],
    [6.5600310334437735],
    [0.2519886806204091],
    [1.0115776081424936, 3.6358259916618088, 0.8111657946199858, 0.11480054308526577, 10.079547224816363, 0.017811704834605608, 2.1825095165016286, 1.063154475240395, 1.0, 0.03772089598850499, 0.16337753022029997, 0.16337753022029997, 1.0, 1.0, 1.0, 1.0, 0.3],
    [0.3],
]
const BLOCK_PREVIOUS_SOLUTION_INITIAL_VALUES = [
    [-1.25e10, 0.3, 0.0],
    [-1.25e10],
    [5.0e11],
    [0.3],
    Float64[],
]
const BLOCK_EXTERNAL_SOLUTION_INITIAL_VALUES = [
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
]
const BLOCK_SOLUTION_INITIAL_VALUES = [
    [0.0, 2.220446049250313e-16],
    [-7.14285714285716e11],
    [1.25e10],
    [1.0115776081424936, 0.0, -1.25e10, -1.25e10, 5.0e11, 0.017811704834605608, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 0.3],
    [0.3],
]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[10] = parameters[10]
    complete_parameters[2] = parameters[2]
    complete_parameters[11] = parameters[11]
    complete_parameters[1] = parameters[1]
    complete_parameters[3] = parameters[3]
    complete_parameters[7] = parameters[7]
    complete_parameters[14] = parameters[14]
    complete_parameters[4] = parameters[4]
    complete_parameters[6] = parameters[6]
    complete_parameters[5] = parameters[5]
    complete_parameters[13] = parameters[13]
    complete_parameters[8] = parameters[8]
    complete_parameters[9] = parameters[9]
    complete_parameters[12] = parameters[12]
    complete_parameters[15] = parameters[15]
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[10] / solution[3] ^ complete_parameters[5] - solution[16] / (1 - solution[7]),
        solution[3] ^ -(complete_parameters[5]) - ((complete_parameters[4] * (solution[1] - complete_parameters[8])) / (1 - complete_parameters[8])) * solution[3] ^ -(complete_parameters[5]),
        (((solution[10] * solution[7] + solution[2]) - solution[2] / solution[1]) + solution[4]) - solution[3],
        (1 - complete_parameters[6]) * solution[12] * solution[6] ^ complete_parameters[6] * solution[7] ^ -(complete_parameters[6]) - solution[10] / (1 - solution[13] * (1 + complete_parameters[9] * (solution[4] - solution[4]) * 2)),
        (((complete_parameters[4] * (solution[3] / solution[3]) ^ complete_parameters[5] * (1 + complete_parameters[9] * (solution[4] - solution[4]) * 2)) / (1 + complete_parameters[9] * (solution[4] - solution[4]) * 2)) * ((1 - complete_parameters[7]) + complete_parameters[6] * (1 - (1 + complete_parameters[9] * (solution[4] - solution[4]) * 2) * solution[13]) * solution[12] * solution[6] ^ (complete_parameters[6] - 1) * solution[7] ^ (1 - complete_parameters[6])) + (1 + complete_parameters[9] * (solution[4] - solution[4]) * 2) * solution[13] * solution[14]) - 1,
        (((1 + complete_parameters[9] * (solution[4] - solution[4]) * 2) / (1 + complete_parameters[9] * (solution[4] - solution[4]) * 2)) * (solution[3] / solution[3]) ^ complete_parameters[5] * complete_parameters[4] * solution[1] + ((1 + complete_parameters[9] * (solution[4] - solution[4]) * 2) * solution[13] * solution[14] * solution[1] * (1 - complete_parameters[8])) / (solution[1] - complete_parameters[8])) - 1,
        ((((solution[2] / solution[1] + solution[6] * (1 - complete_parameters[7]) + solution[12] * solution[6] ^ complete_parameters[6] * solution[7] ^ (1 - complete_parameters[6])) - solution[10] * solution[7]) - solution[2]) - solution[6]) - (solution[4] + complete_parameters[9] * (solution[4] - solution[4]) ^ 2),
        solution[14] * (solution[6] - (solution[2] * (1 - complete_parameters[8])) / (solution[1] - complete_parameters[8])) - solution[12] * solution[6] ^ complete_parameters[6] * solution[7] ^ (1 - complete_parameters[6]),
        log(solution[12] / complete_parameters[3]) - (complete_parameters[10] * log(solution[12] / complete_parameters[3]) + complete_parameters[11] * log(solution[14] / solution[15]) + complete_parameters[14] * 0),
        log(solution[14] / solution[15]) - (log(solution[12] / complete_parameters[3]) * complete_parameters[12] + log(solution[14] / solution[15]) * complete_parameters[13] + complete_parameters[15] * 0),
        solution[11] - solution[12] * solution[6] ^ complete_parameters[6] * solution[7] ^ (1 - complete_parameters[6]),
        solution[6] - (solution[6] * (1 - complete_parameters[7]) + solution[5]),
        solution[9] - (solution[4] + ((solution[3] * complete_parameters[4]) / solution[3]) * solution[9]),
        (1 + solution[8]) - (solution[1] - complete_parameters[8]) / (1 - complete_parameters[8]),
        solution[2] / (solution[11] * (1 + solution[8])) - complete_parameters[1],
        solution[7] - complete_parameters[2],
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[20]) / (1 - solution[7]) + solution[10] / solution[3] ^ complete_parameters[5],
        (-(complete_parameters[4]) * (solution[1] - complete_parameters[8])) / (solution[3] ^ complete_parameters[5] * (1 - complete_parameters[8])) + solution[3] ^ -(complete_parameters[5]),
        ((solution[2] - solution[3]) + solution[4] + solution[7] * solution[10]) - solution[2] / solution[1],
        (solution[6] ^ complete_parameters[6] * solution[12] * (1 - complete_parameters[6])) / solution[7] ^ complete_parameters[6] - solution[10] / (1 - solution[13]),
        (complete_parameters[4] * ((solution[6] ^ (complete_parameters[6] - 1) * solution[7] ^ (1 - complete_parameters[6]) * solution[12] * complete_parameters[6] * (1 - solution[13]) - complete_parameters[7]) + 1) + solution[13] * solution[14]) - 1,
        (solution[1] * complete_parameters[4] + (solution[1] * solution[13] * solution[14] * (1 - complete_parameters[8])) / (solution[1] - complete_parameters[8])) - 1,
        (((((-(solution[2]) - solution[4]) + solution[6] * (1 - complete_parameters[7])) - solution[6]) + solution[6] ^ complete_parameters[6] * solution[7] ^ (1 - complete_parameters[6]) * solution[12]) - solution[7] * solution[10]) + solution[2] / solution[1],
        -(solution[6] ^ complete_parameters[6]) * solution[7] ^ (1 - complete_parameters[6]) * solution[12] + solution[14] * ((-(solution[2]) * (1 - complete_parameters[8])) / (solution[1] - complete_parameters[8]) + solution[6]),
        solution[15] - solution[12] / complete_parameters[3],
        solution[16] - solution[12] / complete_parameters[3],
        solution[17] - solution[14] / solution[19],
        (-(complete_parameters[11]) * log(solution[17]) - complete_parameters[10] * log(solution[16])) + log(solution[15]),
        solution[18] - solution[14] / solution[19],
        (-(complete_parameters[13]) * log(solution[17]) - complete_parameters[12] * log(solution[16])) + log(solution[18]),
        -(solution[6] ^ complete_parameters[6]) * solution[7] ^ (1 - complete_parameters[6]) * solution[12] + solution[11],
        (-(solution[5]) - solution[6] * (1 - complete_parameters[7])) + solution[6],
        (-(solution[4]) - solution[9] * complete_parameters[4]) + solution[9],
        (solution[8] + 1) - (solution[1] - complete_parameters[8]) / (1 - complete_parameters[8]),
        solution[2] / (solution[11] * (1 + solution[8])) - complete_parameters[1],
        solution[7] - complete_parameters[2],
    ]
end

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) / (1 - previous_solution[2]) + previous_solution[3] / solution[2] ^ complete_parameters[5],
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_2(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(previous_solution[1]) - solution[1] * complete_parameters[4]) + solution[1],
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[1]) - previous_solution[1] * (1 - complete_parameters[7])) + previous_solution[1],
    ]
end

function residuals_block_4(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 17
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[5] ^ complete_parameters[6]) * solution[17] ^ (1 - complete_parameters[6]) * solution[9] + solution[11] * ((-(solution[2]) * (1 - complete_parameters[8])) / (solution[1] - complete_parameters[8]) + solution[5]),
        solution[2] / (solution[8] * (1 + solution[6])) - complete_parameters[1],
        (-(complete_parameters[4]) * (solution[1] - complete_parameters[8])) / (solution[3] ^ complete_parameters[5] * (1 - complete_parameters[8])) + solution[3] ^ -(complete_parameters[5]),
        ((solution[2] - solution[3]) + solution[4] + previous_solution[1] * solution[7]) - solution[2] / solution[1],
        (solution[5] ^ complete_parameters[6] * solution[9] * (1 - complete_parameters[6])) / solution[17] ^ complete_parameters[6] - solution[7] / (1 - solution[10]),
        (solution[6] + 1) - (solution[1] - complete_parameters[8]) / (1 - complete_parameters[8]),
        (((((-(solution[2]) - solution[4]) + solution[5] * (1 - complete_parameters[7])) - solution[5]) + solution[5] ^ complete_parameters[6] * solution[17] ^ (1 - complete_parameters[6]) * solution[9]) - previous_solution[1] * solution[7]) + solution[2] / solution[1],
        -(solution[5] ^ complete_parameters[6]) * solution[17] ^ (1 - complete_parameters[6]) * solution[9] + solution[8],
        (complete_parameters[4] * ((solution[5] ^ (complete_parameters[6] - 1) * solution[17] ^ (1 - complete_parameters[6]) * solution[9] * complete_parameters[6] * (1 - solution[10]) - complete_parameters[7]) + 1) + solution[10] * solution[11]) - 1,
        (solution[1] * complete_parameters[4] + (solution[1] * solution[10] * solution[11] * (1 - complete_parameters[8])) / (solution[1] - complete_parameters[8])) - 1,
        solution[15] - solution[11] / solution[12],
        solution[13] - solution[9] / complete_parameters[3],
        solution[14] - solution[9] / complete_parameters[3],
        (-(complete_parameters[11]) * log(solution[15]) - complete_parameters[10] * log(solution[14])) + log(solution[13]),
        (-(complete_parameters[13]) * log(solution[15]) - complete_parameters[12] * log(solution[14])) + log(solution[16]),
        solution[16] - solution[11] / solution[12],
        solution[17] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_5(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - complete_parameters[2],
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
export residuals_block_1, residuals_block_2, residuals_block_3, residuals_block_4, residuals_block_5
end
