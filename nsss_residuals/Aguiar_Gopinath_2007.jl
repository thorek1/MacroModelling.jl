module Aguiar_Gopinath_2007NsssResiduals
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

const MODEL_NAME = "Aguiar_Gopinath_2007"
const SOURCE_MODEL_FILE = "models/Aguiar_Gopinath_2007.jl"
const NSSS_SOLUTION_ERROR = 5.978733960281817e-16
const NSSS_RESIDUAL_NORM = 5.332751299267466e-16

const PARAMETER_NAMES = [
    "gamma",
    "b_share",
    "psi",
    "alpha",
    "sigma",
    "delta",
    "phi",
    "rho_z",
    "rho_g",
    "σᶻ",
    "σᵍ",
    "beta",
    "mu_g",
]
const PARAMETER_VALUES = Float64[
    0.36,
    0.1,
    0.001,
    0.68,
    2.0,
    0.05,
    4.0,
    0.95,
    0.01,
    0.01,
    0.0005,
    0.9803921568627451,
    0.006578315360122507,
]
const COMPLETE_PARAMETER_NAMES = [
    "gamma",
    "b_share",
    "psi",
    "alpha",
    "sigma",
    "delta",
    "phi",
    "rho_z",
    "rho_g",
    "σᶻ",
    "σᵍ",
    "beta",
    "mu_g",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    0.36,
    0.1,
    0.001,
    0.68,
    2.0,
    0.05,
    4.0,
    0.95,
    0.01,
    0.01,
    0.0005,
    0.9803921568627451,
    0.006578315360122507,
]
const ORIGINAL_SOLUTION_NAMES = [
    "b",
    "c",
    "c_y",
    "delta_y",
    "g",
    "i_y",
    "invest",
    "k",
    "l",
    "nx",
    "q",
    "u",
    "uc",
    "ul",
    "y",
    "z",
    "b_star",
    "r_star",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    0.06451768392335203,
    0.4961560429642257,
    0.7690233325085192,
    0.006578315360122507,
    0.006578315360122507,
    0.22878398204327802,
    0.14760612640180695,
    2.607882091904717,
    0.3321686927235321,
    0.0021926854482023114,
    0.9716601882753793,
    -1.6664438374559893,
    1.2091352911878412,
    -1.5969961940258588,
    0.6451768392329362,
    0.0,
    0.06451768392329363,
    0.029166381484582272,
]
const ORIGINAL_INITIAL_SOLUTION_VALUES = Float64[
    0.0,
    5.0e11,
    0.0,
    0.006578315360122507,
    0.006578315360122507,
    0.0,
    2.8299999999999992e10,
    5.0e11,
    0.0,
    0.0,
    0.9716601882753793,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.029166381484582272,
]
const AUXILIARY_SOLUTION_NAMES = [
    "b",
    "c",
    "c_y",
    "delta_y",
    "g",
    "i_y",
    "invest",
    "k",
    "l",
    "nx",
    "q",
    "u",
    "uc",
    "ul",
    "y",
    "z",
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
    "b_star",
    "r_star",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    0.06451768392335203,
    0.4961560429642257,
    0.7690233325085192,
    0.006578315360122507,
    0.006578315360122507,
    0.22878398204327802,
    0.14760612640180695,
    2.607882091904717,
    0.3321686927235321,
    0.002192685448202314,
    0.9716601882753793,
    -1.6664438374559893,
    1.2091352911878412,
    -1.5969961940258588,
    0.6451768392329362,
    0.0,
    0.3343610060955074,
    0.6678313072764679,
    0.6000802292422951,
    5.839773109528323e-14,
    -0.0023681935296441022,
    0.06451768392329363,
    0.029166381484582272,
]
const AUXILIARY_INITIAL_SOLUTION_VALUES = Float64[
    0.0,
    5.0e11,
    0.0,
    0.006578315360122507,
    0.006578315360122507,
    0.0,
    2.8299999999999992e10,
    5.0e11,
    0.0,
    0.0,
    0.9716601882753793,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    5.0e11,
    5.0e11,
    5.0e11,
    5.839773109528153e-14,
    -0.0023681935296441022,
    0.0,
    0.029166381484582272,
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
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    0.3343610060955074,
    0.6678313072764679,
    0.6000802292422951,
    5.839773109528323e-14,
    -0.0023681935296441022,
    0.006578315360122507,
    0.006578315360122507,
    0.006578315360122507,
    0.006578315360122507,
    0.006578315360122507,
    0.006578315360122507,
    0.0,
    0.006578315360122507,
    0.006578315360122507,
    0.006578315360122507,
]
const ALL_AUXILIARY_VARIABLE_INITIAL_VALUES = Float64[
    5.0e11,
    5.0e11,
    5.0e11,
    5.839773109528153e-14,
    -0.0023681935296441022,
    0.006578315360122507,
    0.006578315360122507,
    0.006578315360122507,
    0.006578315360122507,
    0.006578315360122507,
    0.006578315360122507,
    0.0,
    0.006578315360122507,
    0.006578315360122507,
    0.006578315360122507,
]
const DEFAULTED_NSSS_SOLUTION_NAMES = [
]
const CALIBRATION_PARAMETER_NAMES = [
    "b_star",
    "r_star",
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(y - (exp(g) * l) ^ alpha * exp(z) * k ^ (1 - alpha)),
    :(z - (rho_z * z + σᶻ * 0)),
    :(g - ((1 - rho_g) * mu_g + rho_g * g + σᵍ * 0)),
    :(u - (c ^ gamma * (1 - l) ^ (1 - gamma)) ^ (1 - sigma) / (1 - sigma)),
    :(uc - ((1 - sigma) * u * gamma) / c),
    :(ul - ((1 - sigma) * u * -((1 - gamma))) / (1 - l)),
    :((c + k * exp(g)) - ((((y + (1 - delta) * k) - ((k * phi) / 2) * ((k * exp(g)) / k - exp(mu_g)) ^ 2) - b) + b * exp(g) * q)),
    :(1 / q - (1 + r_star + psi * (exp(b - b_star) - 1))),
    :(exp(g) * uc * (1 + phi * ((k * exp(g)) / k - exp(mu_g))) - beta * exp(g * gamma * (1 - sigma)) * uc * (((1 - delta) + ((1 - alpha) * y) / k) - (phi / 2) * ((k * exp(g) * -(((k * exp(g)) / k - exp(mu_g)) * 2)) / k + ((k * exp(g)) / k - exp(mu_g)) ^ 2))),
    :((ul + (y * alpha * uc) / l) - 0),
    :(q * exp(g) * uc - beta * exp(g * gamma * (1 - sigma)) * uc),
    :(invest - ((((k * phi) / 2) * ((k * exp(g)) / k - exp(mu_g)) ^ 2 + k * exp(g)) - (1 - delta) * k)),
    :(c_y - c / y),
    :(i_y - invest / y),
    :(nx - (b - b * exp(g) * q) / y),
    :(delta_y - ((g + log(y)) - log(y))),
]
const CALIBRATION_EQUATIONS = Expr[
    :(b_share * y - b_star),
    :((1 + r_star) - 1 / q),
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :(➕₁ - l * exp(g)),
    :(-(k ^ (1 - alpha)) * ➕₁ ^ alpha * exp(z) + y),
    :(-rho_z * z + z),
    :((-g * rho_g + g) - mu_g * (1 - rho_g)),
    :(➕₂ - (1 - l)),
    :(➕₃ - c ^ gamma * ➕₂ ^ (1 - gamma)),
    :(u - ➕₃ ^ (1 - sigma) / (1 - sigma)),
    :(uc - (gamma * u * (1 - sigma)) / c),
    :((-u * (1 - sigma) * (gamma - 1)) / (1 - l) + ul),
    :((((-b * q * exp(g) + b + c + (k * phi * (exp(g) - exp(mu_g)) ^ 2) / 2) - k * (1 - delta)) + k * exp(g)) - y),
    :(➕₄ - (b - b_star)),
    :(((-psi * (exp(➕₄) - 1) - r_star) - 1) + 1 / q),
    :(➕₅ - g * gamma * (1 - sigma)),
    :(-beta * uc * ((-delta - (phi * ((-2 * exp(g) + 2 * exp(mu_g)) * exp(g) + (exp(g) - exp(mu_g)) ^ 2)) / 2) + 1 + (y * (1 - alpha)) / k) * exp(➕₅) + uc * (phi * (exp(g) - exp(mu_g)) + 1) * exp(g)),
    :((alpha * uc * y) / l + ul),
    :(-beta * uc * exp(➕₅) + q * uc * exp(g)),
    :(((invest - (k * phi * (exp(g) - exp(mu_g)) ^ 2) / 2) + k * (1 - delta)) - k * exp(g)),
    :(-c / y + c_y),
    :(i_y - invest / y),
    :(nx - (-b * q * exp(g) + b) / y),
    :(delta_y - g),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(y - (exp(g) * l) ^ alpha * exp(z) * k ^ (1 - alpha)),
    :(z - (rho_z * z + σᶻ * 0)),
    :(g - ((1 - rho_g) * mu_g + rho_g * g + σᵍ * 0)),
    :(u - (c ^ gamma * (1 - l) ^ (1 - gamma)) ^ (1 - sigma) / (1 - sigma)),
    :(uc - ((1 - sigma) * u * gamma) / c),
    :(ul - ((1 - sigma) * u * -((1 - gamma))) / (1 - l)),
    :((c + k * exp(g)) - ((((y + (1 - delta) * k) - ((k * phi) / 2) * ((k * exp(g)) / k - exp(mu_g)) ^ 2) - b) + b * exp(g) * q)),
    :(1 / q - (1 + r_star + psi * (exp(b - b_star) - 1))),
    :(exp(g) * uc * (1 + phi * ((k * exp(g)) / k - exp(mu_g))) - beta * exp(g * gamma * (1 - sigma)) * uc * (((1 - delta) + ((1 - alpha) * y) / k) - (phi / 2) * ((k * exp(g) * -(((k * exp(g)) / k - exp(mu_g)) * 2)) / k + ((k * exp(g)) / k - exp(mu_g)) ^ 2))),
    :((ul + (y * alpha * uc) / l) - 0),
    :(q * exp(g) * uc - beta * exp(g * gamma * (1 - sigma)) * uc),
    :(invest - ((((k * phi) / 2) * ((k * exp(g)) / k - exp(mu_g)) ^ 2 + k * exp(g)) - (1 - delta) * k)),
    :(c_y - c / y),
    :(i_y - invest / y),
    :(nx - (b - b * exp(g) * q) / y),
    :(delta_y - ((g + log(y)) - log(y))),
    :(b_share * y - b_star),
    :((1 + r_star) - 1 / q),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :(➕₁ - l * exp(g)),
    :(-(k ^ (1 - alpha)) * ➕₁ ^ alpha * exp(z) + y),
    :(-rho_z * z + z),
    :((-g * rho_g + g) - mu_g * (1 - rho_g)),
    :(➕₂ - (1 - l)),
    :(➕₃ - c ^ gamma * ➕₂ ^ (1 - gamma)),
    :(u - ➕₃ ^ (1 - sigma) / (1 - sigma)),
    :(uc - (gamma * u * (1 - sigma)) / c),
    :((-u * (1 - sigma) * (gamma - 1)) / (1 - l) + ul),
    :((((-b * q * exp(g) + b + c + (k * phi * (exp(g) - exp(mu_g)) ^ 2) / 2) - k * (1 - delta)) + k * exp(g)) - y),
    :(➕₄ - (b - b_star)),
    :(((-psi * (exp(➕₄) - 1) - r_star) - 1) + 1 / q),
    :(➕₅ - g * gamma * (1 - sigma)),
    :(-beta * uc * ((-delta - (phi * ((-2 * exp(g) + 2 * exp(mu_g)) * exp(g) + (exp(g) - exp(mu_g)) ^ 2)) / 2) + 1 + (y * (1 - alpha)) / k) * exp(➕₅) + uc * (phi * (exp(g) - exp(mu_g)) + 1) * exp(g)),
    :((alpha * uc * y) / l + ul),
    :(-beta * uc * exp(➕₅) + q * uc * exp(g)),
    :(((invest - (k * phi * (exp(g) - exp(mu_g)) ^ 2) / 2) + k * (1 - delta)) - k * exp(g)),
    :(-c / y + c_y),
    :(i_y - invest / y),
    :(nx - (-b * q * exp(g) + b) / y),
    :(delta_y - g),
    :(b_share * y - b_star),
    :((1 + r_star) - 1 / q),
]

const PARAMETER_DEFINITION_NAMES = [
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "gamma",
    "b_share",
    "psi",
    "alpha",
    "sigma",
    "delta",
    "phi",
    "rho_z",
    "rho_g",
    "σᶻ",
    "σᵍ",
    "beta",
    "mu_g",
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
]
const ORIGINAL_BOX_CONSTRAINT_NAMES = [
    "b",
    "c",
    "c_y",
    "delta_y",
    "g",
    "i_y",
    "invest",
    "k",
    "l",
    "nx",
    "q",
    "u",
    "uc",
    "ul",
    "y",
    "z",
    "b_star",
    "r_star",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
    -1.0e12,
    -Inf,
]
const ORIGINAL_BOX_UPPER_BOUNDS = Float64[
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
    1.0e12,
    Inf,
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "b",
    "c",
    "c_y",
    "delta_y",
    "g",
    "i_y",
    "invest",
    "k",
    "l",
    "nx",
    "q",
    "u",
    "uc",
    "ul",
    "y",
    "z",
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
    "b_star",
    "r_star",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -1.0e12,
    -1.0e12,
    -Inf,
]
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    600.0,
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
]
const ALL_AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
]
const ALL_AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    1.0e12,
    1.0e12,
    1.0e12,
    600.0,
    600.0,
    600.0,
    600.0,
    600.0,
    600.0,
    600.0,
    600.0,
    600.0,
    600.0,
    600.0,
    600.0,
]

const BLOCKS = [
    (
        index = 1,
        solve_order = 9,
        variables = ["nx"],
        previous_solution_names = ["b", "g", "q", "y"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₅"],
        equation_indices = [20],
        equations = Expr[
            :(nx - (-b * q * exp(➕₁₅) + b) / y),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₅ = min(600, max(-1.0e12, g))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₅ - g)),
        ],
        solution_names = ["nx", "➕₁₅"],
        previous_solution_values = [0.06451768392335203, 0.006578315360122507, 0.9716601882753793, 0.6451768392329362],
        external_solution_values = Float64[],
        solution_values = [0.002192685448202314, 0.006578315360122507],
        previous_solution_initial_values = [0.0, 0.006578315360122507, 0.9716601882753793, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 0.006578315360122507],
        box_lower_bounds = [-Inf, -1.0e12],
        box_upper_bounds = [Inf, 600.0],
    ),
    (
        index = 2,
        solve_order = 8,
        variables = ["i_y"],
        previous_solution_names = ["invest", "y"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [19],
        equations = Expr[
            :(i_y - invest / y),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["i_y"],
        previous_solution_values = [0.14760612640180695, 0.6451768392329362],
        external_solution_values = Float64[],
        solution_values = [0.22878398204327802],
        previous_solution_initial_values = [2.8299999999999992e10, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 3,
        solve_order = 7,
        variables = ["invest"],
        previous_solution_names = ["g", "k"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₃", "➕₁₄"],
        equation_indices = [17],
        equations = Expr[
            :(((invest - (k * phi * (exp(➕₁₃) - exp(➕₁₄)) ^ 2) / 2) + k * (1 - delta)) - k * exp(➕₁₃)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₃ = min(600, max(-1.0e12, g))),
            :(➕₁₄ = min(600, max(-1.0e12, mu_g))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₃ - g)),
            :(abs(➕₁₄ - mu_g)),
        ],
        solution_names = ["invest", "➕₁₃", "➕₁₄"],
        previous_solution_values = [0.006578315360122507, 2.607882091904717],
        external_solution_values = Float64[],
        solution_values = [0.14760612640180695, 0.006578315360122507, 0.006578315360122507],
        previous_solution_initial_values = [0.006578315360122507, 5.0e11],
        external_solution_initial_values = Float64[],
        solution_initial_values = [2.8299999999999992e10, 0.006578315360122507, 0.006578315360122507],
        box_lower_bounds = [-Inf, -1.0e12, -1.0e12],
        box_upper_bounds = [Inf, 600.0, 600.0],
    ),
    (
        index = 4,
        solve_order = 6,
        variables = ["delta_y"],
        previous_solution_names = ["g"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [21],
        equations = Expr[
            :(delta_y - g),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["delta_y"],
        previous_solution_values = [0.006578315360122507],
        external_solution_values = Float64[],
        solution_values = [0.006578315360122507],
        previous_solution_initial_values = [0.006578315360122507],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.006578315360122507],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 5,
        solve_order = 5,
        variables = ["c_y"],
        previous_solution_names = ["c", "y"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [18],
        equations = Expr[
            :(-c / y + c_y),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["c_y"],
        previous_solution_values = [0.4961560429642257, 0.6451768392329362],
        external_solution_values = Float64[],
        solution_values = [0.7690233325085192],
        previous_solution_initial_values = [5.0e11, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 6,
        solve_order = 4,
        variables = ["b", "b_star", "c", "k", "l", "q", "r_star", "u", "uc", "ul", "y", "➕₁", "➕₂", "➕₃", "➕₄"],
        previous_solution_names = ["g", "z", "➕₅"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₀", "➕₁₁", "➕₁₂"],
        equation_indices = [10, 8, 2, 15, 23, 7, 16, 9, 14, 1, 5, 6, 11, 22, 12],
        equations = Expr[
            :((((-b * q * exp(➕₁₀) + b + c + (k * phi * (exp(➕₁₀) - exp(➕₁₁)) ^ 2) / 2) - k * (1 - delta)) + k * exp(➕₁₀)) - y),
            :(uc - (gamma * u * (1 - sigma)) / c),
            :(-(k ^ (1 - alpha)) * ➕₁ ^ alpha * exp(➕₁₂) + y),
            :((alpha * uc * y) / l + ul),
            :((1 + r_star) - 1 / q),
            :(u - ➕₃ ^ (1 - sigma) / (1 - sigma)),
            :(-beta * uc * exp(➕₅) + q * uc * exp(➕₁₀)),
            :((-u * (1 - sigma) * (gamma - 1)) / (1 - l) + ul),
            :(-beta * uc * ((-delta - (phi * ((exp(➕₁₀) * -2 + exp(➕₁₁) * 2) * exp(➕₁₀) + (exp(➕₁₀) - exp(➕₁₁)) ^ 2)) / 2) + 1 + (y * (1 - alpha)) / k) * exp(➕₅) + uc * (phi * (exp(➕₁₀) - exp(➕₁₁)) + 1) * exp(➕₁₀)),
            :(➕₁ - l * exp(➕₁₀)),
            :(➕₂ - (1 - l)),
            :(➕₃ - c ^ gamma * ➕₂ ^ (1 - gamma)),
            :(➕₄ - (b - b_star)),
            :(b_share * y - b_star),
            :(((-psi * (exp(➕₄) - 1) - r_star) - 1) + 1 / q),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₀ = min(600, max(-1.0e12, g))),
            :(➕₁₁ = min(600, max(-1.0e12, mu_g))),
            :(➕₁₂ = min(600, max(-1.0e12, z))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₀ - g)),
            :(abs(➕₁₁ - mu_g)),
            :(abs(➕₁₂ - z)),
        ],
        solution_names = ["b", "b_star", "c", "k", "l", "q", "r_star", "u", "uc", "ul", "y", "➕₁", "➕₂", "➕₃", "➕₄", "➕₁₀", "➕₁₁", "➕₁₂"],
        previous_solution_values = [0.006578315360122507, 0.0, -0.0023681935296441022],
        external_solution_values = Float64[],
        solution_values = [0.06451768392335203, 0.06451768392329363, 0.4961560429642257, 2.607882091904717, 0.3321686927235321, 0.9716601882753793, 0.029166381484582272, -1.6664438374559893, 1.2091352911878412, -1.5969961940258588, 0.6451768392329362, 0.3343610060955074, 0.6678313072764679, 0.6000802292422951, 5.839773109528323e-14, 0.006578315360122507, 0.006578315360122507, 0.0],
        previous_solution_initial_values = [0.006578315360122507, 0.0, -0.0023681935296441022],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 0.0, 5.0e11, 5.0e11, 0.0, 0.9716601882753793, 0.029166381484582272, 0.0, 0.0, 0.0, 0.0, 5.0e11, 5.0e11, 5.0e11, 5.839773109528153e-14, 0.006578315360122507, 0.006578315360122507, 0.0],
        box_lower_bounds = [-1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -Inf, -Inf, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, Inf, Inf, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 600.0, 600.0, 600.0, 600.0],
    ),
    (
        index = 7,
        solve_order = 3,
        variables = ["➕₅"],
        previous_solution_names = ["g"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [13],
        equations = Expr[
            :(➕₅ - g * gamma * (1 - sigma)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["➕₅"],
        previous_solution_values = [0.006578315360122507],
        external_solution_values = Float64[],
        solution_values = [-0.0023681935296441022],
        previous_solution_initial_values = [0.006578315360122507],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-0.0023681935296441022],
        box_lower_bounds = [-1.0e12],
        box_upper_bounds = [600.0],
    ),
    (
        index = 8,
        solve_order = 2,
        variables = ["z"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [3],
        equations = Expr[
            :(-rho_z * z + z),
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
        index = 9,
        solve_order = 1,
        variables = ["g"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [4],
        equations = Expr[
            :((-g * rho_g + g) - mu_g * (1 - rho_g)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["g"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.006578315360122507],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.006578315360122507],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
]
const BLOCK_EQUATION_ORDER = [20, 19, 17, 21, 18, 10, 8, 2, 15, 23, 7, 16, 9, 14, 1, 5, 6, 11, 22, 12, 13, 3, 4]
const BLOCK_SOLVE_ORDER = [9, 8, 7, 6, 5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    ["b", "g", "q", "y"],
    ["invest", "y"],
    ["g", "k"],
    ["g"],
    ["c", "y"],
    ["g", "z", "➕₅"],
    ["g"],
    String[],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    [0.06451768392335203, 0.006578315360122507, 0.9716601882753793, 0.6451768392329362],
    [0.14760612640180695, 0.6451768392329362],
    [0.006578315360122507, 2.607882091904717],
    [0.006578315360122507],
    [0.4961560429642257, 0.6451768392329362],
    [0.006578315360122507, 0.0, -0.0023681935296441022],
    [0.006578315360122507],
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
]
const BLOCK_SOLUTION_NAMES = [
    ["nx", "➕₁₅"],
    ["i_y"],
    ["invest", "➕₁₃", "➕₁₄"],
    ["delta_y"],
    ["c_y"],
    ["b", "b_star", "c", "k", "l", "q", "r_star", "u", "uc", "ul", "y", "➕₁", "➕₂", "➕₃", "➕₄", "➕₁₀", "➕₁₁", "➕₁₂"],
    ["➕₅"],
    ["z"],
    ["g"],
]
const BLOCK_SOLUTION_VALUES = [
    [0.002192685448202314, 0.006578315360122507],
    [0.22878398204327802],
    [0.14760612640180695, 0.006578315360122507, 0.006578315360122507],
    [0.006578315360122507],
    [0.7690233325085192],
    [0.06451768392335203, 0.06451768392329363, 0.4961560429642257, 2.607882091904717, 0.3321686927235321, 0.9716601882753793, 0.029166381484582272, -1.6664438374559893, 1.2091352911878412, -1.5969961940258588, 0.6451768392329362, 0.3343610060955074, 0.6678313072764679, 0.6000802292422951, 5.839773109528323e-14, 0.006578315360122507, 0.006578315360122507, 0.0],
    [-0.0023681935296441022],
    [0.0],
    [0.006578315360122507],
]
const BLOCK_PREVIOUS_SOLUTION_INITIAL_VALUES = [
    [0.0, 0.006578315360122507, 0.9716601882753793, 0.0],
    [2.8299999999999992e10, 0.0],
    [0.006578315360122507, 5.0e11],
    [0.006578315360122507],
    [5.0e11, 0.0],
    [0.006578315360122507, 0.0, -0.0023681935296441022],
    [0.006578315360122507],
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
]
const BLOCK_SOLUTION_INITIAL_VALUES = [
    [0.0, 0.006578315360122507],
    [0.0],
    [2.8299999999999992e10, 0.006578315360122507, 0.006578315360122507],
    [0.006578315360122507],
    [0.0],
    [0.0, 0.0, 5.0e11, 5.0e11, 0.0, 0.9716601882753793, 0.029166381484582272, 0.0, 0.0, 0.0, 0.0, 5.0e11, 5.0e11, 5.0e11, 5.839773109528153e-14, 0.006578315360122507, 0.006578315360122507, 0.0],
    [-0.0023681935296441022],
    [0.0],
    [0.006578315360122507],
]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[4] = parameters[4]
    complete_parameters[1] = parameters[1]
    complete_parameters[9] = parameters[9]
    complete_parameters[11] = parameters[11]
    complete_parameters[10] = parameters[10]
    complete_parameters[13] = parameters[13]
    complete_parameters[6] = parameters[6]
    complete_parameters[12] = parameters[12]
    complete_parameters[2] = parameters[2]
    complete_parameters[5] = parameters[5]
    complete_parameters[7] = parameters[7]
    complete_parameters[3] = parameters[3]
    complete_parameters[8] = parameters[8]
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[15] - (exp(solution[5]) * solution[9]) ^ complete_parameters[4] * exp(solution[16]) * solution[8] ^ (1 - complete_parameters[4]),
        solution[16] - (complete_parameters[8] * solution[16] + complete_parameters[10] * 0),
        solution[5] - ((1 - complete_parameters[9]) * complete_parameters[13] + complete_parameters[9] * solution[5] + complete_parameters[11] * 0),
        solution[12] - (solution[2] ^ complete_parameters[1] * (1 - solution[9]) ^ (1 - complete_parameters[1])) ^ (1 - complete_parameters[5]) / (1 - complete_parameters[5]),
        solution[13] - ((1 - complete_parameters[5]) * solution[12] * complete_parameters[1]) / solution[2],
        solution[14] - ((1 - complete_parameters[5]) * solution[12] * -((1 - complete_parameters[1]))) / (1 - solution[9]),
        (solution[2] + solution[8] * exp(solution[5])) - ((((solution[15] + (1 - complete_parameters[6]) * solution[8]) - ((solution[8] * complete_parameters[7]) / 2) * ((solution[8] * exp(solution[5])) / solution[8] - exp(complete_parameters[13])) ^ 2) - solution[1]) + solution[1] * exp(solution[5]) * solution[11]),
        1 / solution[11] - (1 + solution[18] + complete_parameters[3] * (exp(solution[1] - solution[17]) - 1)),
        exp(solution[5]) * solution[13] * (1 + complete_parameters[7] * ((solution[8] * exp(solution[5])) / solution[8] - exp(complete_parameters[13]))) - complete_parameters[12] * exp(solution[5] * complete_parameters[1] * (1 - complete_parameters[5])) * solution[13] * (((1 - complete_parameters[6]) + ((1 - complete_parameters[4]) * solution[15]) / solution[8]) - (complete_parameters[7] / 2) * ((solution[8] * exp(solution[5]) * -(((solution[8] * exp(solution[5])) / solution[8] - exp(complete_parameters[13])) * 2)) / solution[8] + ((solution[8] * exp(solution[5])) / solution[8] - exp(complete_parameters[13])) ^ 2)),
        (solution[14] + (solution[15] * complete_parameters[4] * solution[13]) / solution[9]) - 0,
        solution[11] * exp(solution[5]) * solution[13] - complete_parameters[12] * exp(solution[5] * complete_parameters[1] * (1 - complete_parameters[5])) * solution[13],
        solution[7] - ((((solution[8] * complete_parameters[7]) / 2) * ((solution[8] * exp(solution[5])) / solution[8] - exp(complete_parameters[13])) ^ 2 + solution[8] * exp(solution[5])) - (1 - complete_parameters[6]) * solution[8]),
        solution[3] - solution[2] / solution[15],
        solution[6] - solution[7] / solution[15],
        solution[10] - (solution[1] - solution[1] * exp(solution[5]) * solution[11]) / solution[15],
        solution[4] - ((solution[5] + log(solution[15])) - log(solution[15])),
        complete_parameters[2] * solution[15] - solution[17],
        (1 + solution[18]) - 1 / solution[11],
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[17] - solution[9] * exp(solution[5]),
        -(solution[8] ^ (1 - complete_parameters[4])) * solution[17] ^ complete_parameters[4] * exp(solution[16]) + solution[15],
        -(complete_parameters[8]) * solution[16] + solution[16],
        (-(solution[5]) * complete_parameters[9] + solution[5]) - complete_parameters[13] * (1 - complete_parameters[9]),
        solution[18] - (1 - solution[9]),
        solution[19] - solution[2] ^ complete_parameters[1] * solution[18] ^ (1 - complete_parameters[1]),
        solution[12] - solution[19] ^ (1 - complete_parameters[5]) / (1 - complete_parameters[5]),
        solution[13] - (complete_parameters[1] * solution[12] * (1 - complete_parameters[5])) / solution[2],
        (-(solution[12]) * (1 - complete_parameters[5]) * (complete_parameters[1] - 1)) / (1 - solution[9]) + solution[14],
        (((-(solution[1]) * solution[11] * exp(solution[5]) + solution[1] + solution[2] + (solution[8] * complete_parameters[7] * (exp(solution[5]) - exp(complete_parameters[13])) ^ 2) / 2) - solution[8] * (1 - complete_parameters[6])) + solution[8] * exp(solution[5])) - solution[15],
        solution[20] - (solution[1] - solution[22]),
        ((-(complete_parameters[3]) * (exp(solution[20]) - 1) - solution[23]) - 1) + 1 / solution[11],
        solution[21] - solution[5] * complete_parameters[1] * (1 - complete_parameters[5]),
        -(complete_parameters[12]) * solution[13] * ((-(complete_parameters[6]) - (complete_parameters[7] * ((-2 * exp(solution[5]) + 2 * exp(complete_parameters[13])) * exp(solution[5]) + (exp(solution[5]) - exp(complete_parameters[13])) ^ 2)) / 2) + 1 + (solution[15] * (1 - complete_parameters[4])) / solution[8]) * exp(solution[21]) + solution[13] * (complete_parameters[7] * (exp(solution[5]) - exp(complete_parameters[13])) + 1) * exp(solution[5]),
        (complete_parameters[4] * solution[13] * solution[15]) / solution[9] + solution[14],
        -(complete_parameters[12]) * solution[13] * exp(solution[21]) + solution[11] * solution[13] * exp(solution[5]),
        ((solution[7] - (solution[8] * complete_parameters[7] * (exp(solution[5]) - exp(complete_parameters[13])) ^ 2) / 2) + solution[8] * (1 - complete_parameters[6])) - solution[8] * exp(solution[5]),
        -(solution[2]) / solution[15] + solution[3],
        solution[6] - solution[7] / solution[15],
        solution[10] - (-(solution[1]) * solution[11] * exp(solution[5]) + solution[1]) / solution[15],
        solution[4] - solution[5],
        complete_parameters[2] * solution[15] - solution[22],
        (1 + solution[23]) - 1 / solution[11],
    ]
end

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 4
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - (-(previous_solution[1]) * previous_solution[3] * exp(solution[2]) + previous_solution[1]) / previous_solution[4],
        solution[2] - min(600, max(-1.0e12, previous_solution[2])),
    ]
end

function residuals_block_2(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1] / previous_solution[2],
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 3
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((solution[1] - (previous_solution[2] * complete_parameters[7] * (exp(solution[2]) - exp(solution[3])) ^ 2) / 2) + previous_solution[2] * (1 - complete_parameters[6])) - previous_solution[2] * exp(solution[2]),
        solution[2] - min(600, max(-1.0e12, previous_solution[1])),
        solution[3] - min(600, max(-1.0e12, complete_parameters[13])),
    ]
end

function residuals_block_4(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_5(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) / previous_solution[2] + solution[1],
    ]
end

function residuals_block_6(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 18
    complete_parameters = complete_parameter_values(parameters)
    return [
        (((-(solution[1]) * solution[6] * exp(solution[16]) + solution[1] + solution[3] + (solution[4] * complete_parameters[7] * (exp(solution[16]) - exp(solution[17])) ^ 2) / 2) - solution[4] * (1 - complete_parameters[6])) + solution[4] * exp(solution[16])) - solution[11],
        solution[9] - (complete_parameters[1] * solution[8] * (1 - complete_parameters[5])) / solution[3],
        -(solution[4] ^ (1 - complete_parameters[4])) * solution[12] ^ complete_parameters[4] * exp(solution[18]) + solution[11],
        (complete_parameters[4] * solution[9] * solution[11]) / solution[5] + solution[10],
        (1 + solution[7]) - 1 / solution[6],
        solution[8] - solution[14] ^ (1 - complete_parameters[5]) / (1 - complete_parameters[5]),
        -(complete_parameters[12]) * solution[9] * exp(previous_solution[3]) + solution[6] * solution[9] * exp(solution[16]),
        (-(solution[8]) * (1 - complete_parameters[5]) * (complete_parameters[1] - 1)) / (1 - solution[5]) + solution[10],
        -(complete_parameters[12]) * solution[9] * ((-(complete_parameters[6]) - (complete_parameters[7] * ((exp(solution[16]) * -2 + exp(solution[17]) * 2) * exp(solution[16]) + (exp(solution[16]) - exp(solution[17])) ^ 2)) / 2) + 1 + (solution[11] * (1 - complete_parameters[4])) / solution[4]) * exp(previous_solution[3]) + solution[9] * (complete_parameters[7] * (exp(solution[16]) - exp(solution[17])) + 1) * exp(solution[16]),
        solution[12] - solution[5] * exp(solution[16]),
        solution[13] - (1 - solution[5]),
        solution[14] - solution[3] ^ complete_parameters[1] * solution[13] ^ (1 - complete_parameters[1]),
        solution[15] - (solution[1] - solution[2]),
        complete_parameters[2] * solution[11] - solution[2],
        ((-(complete_parameters[3]) * (exp(solution[15]) - 1) - solution[7]) - 1) + 1 / solution[6],
        solution[16] - min(600, max(-1.0e12, previous_solution[1])),
        solution[17] - min(600, max(-1.0e12, complete_parameters[13])),
        solution[18] - min(600, max(-1.0e12, previous_solution[2])),
    ]
end

function residuals_block_7(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1] * complete_parameters[1] * (1 - complete_parameters[5]),
    ]
end

function residuals_block_8(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[8]) * solution[1] + solution[1],
    ]
end

function residuals_block_9(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[1]) * complete_parameters[9] + solution[1]) - complete_parameters[13] * (1 - complete_parameters[9]),
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
export residuals_block_1, residuals_block_2, residuals_block_3, residuals_block_4, residuals_block_5, residuals_block_6, residuals_block_7, residuals_block_8, residuals_block_9
end
