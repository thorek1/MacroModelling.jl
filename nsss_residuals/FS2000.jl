module FS2000NsssResiduals
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

const MODEL_NAME = "FS2000"
const SOURCE_MODEL_FILE = "models/FS2000.jl"
const NSSS_SOLUTION_ERROR = 9.678699938266253e-16
const NSSS_RESIDUAL_NORM = 1.3732700395566711e-15

const PARAMETER_NAMES = [
    "alp",
    "bet",
    "gam",
    "mst",
    "rho",
    "psi",
    "del",
    "z_e_a",
    "z_e_m",
]
const PARAMETER_VALUES = Float64[
    0.356,
    0.993,
    0.0085,
    1.0002,
    0.129,
    0.65,
    0.01,
    0.035449,
    0.008862,
]
const COMPLETE_PARAMETER_NAMES = [
    "alp",
    "bet",
    "gam",
    "mst",
    "rho",
    "psi",
    "del",
    "z_e_a",
    "z_e_m",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    0.356,
    0.993,
    0.0085,
    1.0002,
    0.129,
    0.65,
    0.01,
    0.035449,
    0.008862,
]
const ORIGINAL_SOLUTION_NAMES = [
    "P",
    "R",
    "W",
    "c",
    "d",
    "dA",
    "e",
    "gp_obs",
    "gy_obs",
    "k",
    "l",
    "log_gp_obs",
    "log_gy_obs",
    "m",
    "n",
    "y",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    0.993250554567554,
    1.0072507552870091,
    2.7185621689742434,
    1.0069966690685328,
    0.8608478832599584,
    1.0085362275720395,
    1.0,
    0.9917343300675393,
    1.0085362275720395,
    18.982191719109345,
    0.8610478832599584,
    -0.008300019997333728,
    0.008499999999999985,
    1.0002,
    0.3167291493594371,
    1.3558767746137388,
]
const AUXILIARY_SOLUTION_NAMES = [
    "P",
    "R",
    "W",
    "c",
    "d",
    "dA",
    "e",
    "gp_obs",
    "gy_obs",
    "k",
    "l",
    "log_gp_obs",
    "log_gy_obs",
    "m",
    "n",
    "y",
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    0.993250554567554,
    1.0072507552870091,
    2.7185621689742434,
    1.0069966690685328,
    0.8608478832599584,
    1.0085362275720395,
    1.0,
    0.9917343300675393,
    1.0085362275720395,
    18.982191719109345,
    0.8610478832599584,
    -0.008300019997333728,
    0.008499999999999985,
    1.0002,
    0.3167291493594371,
    1.3558767746137388,
    0.0085,
    -0.003026,
    -0.0085,
    -0.003026,
    -0.0085,
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
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    0.0085,
    -0.003026,
    -0.0085,
    -0.003026,
    -0.0085,
    1.0085362275720395,
    0.9917343300675393,
    18.982191719109345,
    0.3167291493594371,
]
const DEFAULTED_NSSS_SOLUTION_NAMES = [
]
const CALIBRATION_PARAMETER_NAMES = [
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(dA - exp(gam + z_e_a * 0)),
    :(log(m) - ((1 - rho) * log(mst) + rho * log(m) + z_e_m * 0)),
    :((-P / (c * P * m) + (bet * P * (alp * exp(-alp * (gam + log(e))) * k ^ (alp - 1) * n ^ (1 - alp) + (1 - del) * exp(-((gam + log(e)))))) / (c * P * m)) - 0),
    :(W - l / n),
    :((-(psi / (1 - psi)) * ((c * P) / (1 - n)) + l / n) - 0),
    :(R - (P * (1 - alp) * exp(-alp * (gam + z_e_a * 0)) * k ^ alp * n ^ -alp) / W),
    :((1 / (c * P) - (bet * P * (1 - alp) * exp(-alp * (gam + z_e_a * 0)) * k ^ alp * n ^ (1 - alp)) / (m * l * c * P)) - 0),
    :((c + k) - (exp(-alp * (gam + z_e_a * 0)) * k ^ alp * n ^ (1 - alp) + (1 - del) * exp(-((gam + z_e_a * 0))) * k)),
    :(P * c - m),
    :(((m - 1) + d) - l),
    :(e - exp(z_e_a * 0)),
    :(y - k ^ alp * n ^ (1 - alp) * exp(-alp * (gam + z_e_a * 0))),
    :(gy_obs - (dA * y) / y),
    :(gp_obs - ((P / P) * m) / dA),
    :(log_gy_obs - log(gy_obs)),
    :(log_gp_obs - log(gp_obs)),
]
const CALIBRATION_EQUATIONS = Expr[
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :(➕₁ - gam),
    :(dA - exp(➕₁)),
    :((-rho * log(m) - (1 - rho) * log(mst)) + log(m)),
    :(➕₂ - -alp * (gam + log(e))),
    :(➕₃ - (-gam - log(e))),
    :((bet * (alp * k ^ (alp - 1) * n ^ (1 - alp) * exp(➕₂) + (1 - del) * exp(➕₃))) / (c * m) - 1 / (c * m)),
    :(W - l / n),
    :((-P * c * psi) / ((1 - n) * (1 - psi)) + l / n),
    :(➕₄ - -alp * gam),
    :((-P * k ^ alp * (1 - alp) * exp(➕₄)) / (W * n ^ alp) + R),
    :((-bet * k ^ alp * n ^ (1 - alp) * (1 - alp) * exp(➕₄)) / (c * l * m) + 1 / (P * c)),
    :(➕₅ - -gam),
    :(((c - k * (1 - del) * exp(➕₅)) + k) - k ^ alp * n ^ (1 - alp) * exp(➕₄)),
    :(P * c - m),
    :(((d - l) + m) - 1),
    :(e - 1.0),
    :(-(k ^ alp) * n ^ (1 - alp) * exp(➕₄) + y),
    :(-dA + gy_obs),
    :(gp_obs - m / dA),
    :(log_gy_obs - log(gy_obs)),
    :(log_gp_obs - log(gp_obs)),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(dA - exp(gam + z_e_a * 0)),
    :(log(m) - ((1 - rho) * log(mst) + rho * log(m) + z_e_m * 0)),
    :((-P / (c * P * m) + (bet * P * (alp * exp(-alp * (gam + log(e))) * k ^ (alp - 1) * n ^ (1 - alp) + (1 - del) * exp(-((gam + log(e)))))) / (c * P * m)) - 0),
    :(W - l / n),
    :((-(psi / (1 - psi)) * ((c * P) / (1 - n)) + l / n) - 0),
    :(R - (P * (1 - alp) * exp(-alp * (gam + z_e_a * 0)) * k ^ alp * n ^ -alp) / W),
    :((1 / (c * P) - (bet * P * (1 - alp) * exp(-alp * (gam + z_e_a * 0)) * k ^ alp * n ^ (1 - alp)) / (m * l * c * P)) - 0),
    :((c + k) - (exp(-alp * (gam + z_e_a * 0)) * k ^ alp * n ^ (1 - alp) + (1 - del) * exp(-((gam + z_e_a * 0))) * k)),
    :(P * c - m),
    :(((m - 1) + d) - l),
    :(e - exp(z_e_a * 0)),
    :(y - k ^ alp * n ^ (1 - alp) * exp(-alp * (gam + z_e_a * 0))),
    :(gy_obs - (dA * y) / y),
    :(gp_obs - ((P / P) * m) / dA),
    :(log_gy_obs - log(gy_obs)),
    :(log_gp_obs - log(gp_obs)),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :(➕₁ - gam),
    :(dA - exp(➕₁)),
    :((-rho * log(m) - (1 - rho) * log(mst)) + log(m)),
    :(➕₂ - -alp * (gam + log(e))),
    :(➕₃ - (-gam - log(e))),
    :((bet * (alp * k ^ (alp - 1) * n ^ (1 - alp) * exp(➕₂) + (1 - del) * exp(➕₃))) / (c * m) - 1 / (c * m)),
    :(W - l / n),
    :((-P * c * psi) / ((1 - n) * (1 - psi)) + l / n),
    :(➕₄ - -alp * gam),
    :((-P * k ^ alp * (1 - alp) * exp(➕₄)) / (W * n ^ alp) + R),
    :((-bet * k ^ alp * n ^ (1 - alp) * (1 - alp) * exp(➕₄)) / (c * l * m) + 1 / (P * c)),
    :(➕₅ - -gam),
    :(((c - k * (1 - del) * exp(➕₅)) + k) - k ^ alp * n ^ (1 - alp) * exp(➕₄)),
    :(P * c - m),
    :(((d - l) + m) - 1),
    :(e - 1.0),
    :(-(k ^ alp) * n ^ (1 - alp) * exp(➕₄) + y),
    :(-dA + gy_obs),
    :(gp_obs - m / dA),
    :(log_gy_obs - log(gy_obs)),
    :(log_gp_obs - log(gp_obs)),
]

const PARAMETER_DEFINITION_NAMES = [
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "alp",
    "bet",
    "gam",
    "mst",
    "rho",
    "psi",
    "del",
    "z_e_a",
    "z_e_m",
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
    "P",
    "R",
    "W",
    "c",
    "d",
    "dA",
    "e",
    "gp_obs",
    "gy_obs",
    "k",
    "l",
    "log_gp_obs",
    "log_gy_obs",
    "m",
    "n",
    "y",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -Inf,
]
const ORIGINAL_BOX_UPPER_BOUNDS = Float64[
    1.0e12,
    Inf,
    Inf,
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
    Inf,
    1.0e12,
    Inf,
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "P",
    "R",
    "W",
    "c",
    "d",
    "dA",
    "e",
    "gp_obs",
    "gy_obs",
    "k",
    "l",
    "log_gp_obs",
    "log_gy_obs",
    "m",
    "n",
    "y",
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
]
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    1.0e12,
    Inf,
    Inf,
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
    Inf,
    1.0e12,
    Inf,
    600.0,
    600.0,
    600.0,
    600.0,
    600.0,
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
]
const ALL_AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
]
const ALL_AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    600.0,
    600.0,
    600.0,
    600.0,
    600.0,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
]

const BLOCKS = [
    (
        index = 1,
        variables = ["y"],
        equation_indices = [17],
        equations = Expr[
            :(-(k ^ alp) * n ^ (1 - alp) * exp(➕₄) + y),
        ],
    ),
    (
        index = 2,
        variables = ["log_gy_obs"],
        equation_indices = [20],
        equations = Expr[
            :(log_gy_obs - log(gy_obs)),
        ],
    ),
    (
        index = 3,
        variables = ["log_gp_obs"],
        equation_indices = [21],
        equations = Expr[
            :(log_gp_obs - log(gp_obs)),
        ],
    ),
    (
        index = 4,
        variables = ["gy_obs"],
        equation_indices = [18],
        equations = Expr[
            :(-dA + gy_obs),
        ],
    ),
    (
        index = 5,
        variables = ["gp_obs"],
        equation_indices = [19],
        equations = Expr[
            :(gp_obs - m / dA),
        ],
    ),
    (
        index = 6,
        variables = ["dA"],
        equation_indices = [2],
        equations = Expr[
            :(dA - exp(➕₁)),
        ],
    ),
    (
        index = 7,
        variables = ["➕₁"],
        equation_indices = [1],
        equations = Expr[
            :(➕₁ - gam),
        ],
    ),
    (
        index = 8,
        variables = ["d"],
        equation_indices = [15],
        equations = Expr[
            :(((d - l) + m) - 1),
        ],
    ),
    (
        index = 9,
        variables = ["R"],
        equation_indices = [10],
        equations = Expr[
            :((-P * k ^ alp * (1 - alp) * exp(➕₄)) / (W * n ^ alp) + R),
        ],
    ),
    (
        index = 10,
        variables = ["W"],
        equation_indices = [7],
        equations = Expr[
            :(W - l / n),
        ],
    ),
    (
        index = 11,
        variables = ["P", "c", "k", "l", "n"],
        equation_indices = [14, 6, 11, 8, 13],
        equations = Expr[
            :(P * c - m),
            :((bet * (alp * k ^ (alp - 1) * n ^ (1 - alp) * exp(➕₂) + (1 - del) * exp(➕₃))) / (c * m) - 1 / (c * m)),
            :((-bet * k ^ alp * n ^ (1 - alp) * (1 - alp) * exp(➕₄)) / (c * l * m) + 1 / (P * c)),
            :((-P * c * psi) / ((1 - n) * (1 - psi)) + l / n),
            :(((c - k * (1 - del) * exp(➕₅)) + k) - k ^ alp * n ^ (1 - alp) * exp(➕₄)),
        ],
    ),
    (
        index = 12,
        variables = ["➕₃"],
        equation_indices = [5],
        equations = Expr[
            :(➕₃ - (-gam - log(e))),
        ],
    ),
    (
        index = 13,
        variables = ["➕₂"],
        equation_indices = [4],
        equations = Expr[
            :(➕₂ - -alp * (gam + log(e))),
        ],
    ),
    (
        index = 14,
        variables = ["e"],
        equation_indices = [16],
        equations = Expr[
            :(e - 1.0),
        ],
    ),
    (
        index = 15,
        variables = ["m"],
        equation_indices = [3],
        equations = Expr[
            :((-rho * log(m) - (1 - rho) * log(mst)) + log(m)),
        ],
    ),
    (
        index = 16,
        variables = ["➕₅"],
        equation_indices = [12],
        equations = Expr[
            :(➕₅ - -gam),
        ],
    ),
    (
        index = 17,
        variables = ["➕₄"],
        equation_indices = [9],
        equations = Expr[
            :(➕₄ - -alp * gam),
        ],
    ),
]
const BLOCK_EQUATION_ORDER = [17, 20, 21, 18, 19, 2, 1, 15, 10, 7, 14, 6, 11, 8, 13, 5, 4, 16, 3, 12, 9]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[4] = parameters[4]
    complete_parameters[8] = parameters[8]
    complete_parameters[9] = parameters[9]
    complete_parameters[3] = parameters[3]
    complete_parameters[5] = parameters[5]
    complete_parameters[1] = parameters[1]
    complete_parameters[2] = parameters[2]
    complete_parameters[7] = parameters[7]
    complete_parameters[6] = parameters[6]
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[6] - exp(complete_parameters[3] + complete_parameters[8] * 0),
        log(solution[14]) - ((1 - complete_parameters[5]) * log(complete_parameters[4]) + complete_parameters[5] * log(solution[14]) + complete_parameters[9] * 0),
        (-(solution[1]) / (solution[4] * solution[1] * solution[14]) + (complete_parameters[2] * solution[1] * (complete_parameters[1] * exp(-(complete_parameters[1]) * (complete_parameters[3] + log(solution[7]))) * solution[10] ^ (complete_parameters[1] - 1) * solution[15] ^ (1 - complete_parameters[1]) + (1 - complete_parameters[7]) * exp(-((complete_parameters[3] + log(solution[7])))))) / (solution[4] * solution[1] * solution[14])) - 0,
        solution[3] - solution[11] / solution[15],
        (-(complete_parameters[6] / (1 - complete_parameters[6])) * ((solution[4] * solution[1]) / (1 - solution[15])) + solution[11] / solution[15]) - 0,
        solution[2] - (solution[1] * (1 - complete_parameters[1]) * exp(-(complete_parameters[1]) * (complete_parameters[3] + complete_parameters[8] * 0)) * solution[10] ^ complete_parameters[1] * solution[15] ^ -(complete_parameters[1])) / solution[3],
        (1 / (solution[4] * solution[1]) - (complete_parameters[2] * solution[1] * (1 - complete_parameters[1]) * exp(-(complete_parameters[1]) * (complete_parameters[3] + complete_parameters[8] * 0)) * solution[10] ^ complete_parameters[1] * solution[15] ^ (1 - complete_parameters[1])) / (solution[14] * solution[11] * solution[4] * solution[1])) - 0,
        (solution[4] + solution[10]) - (exp(-(complete_parameters[1]) * (complete_parameters[3] + complete_parameters[8] * 0)) * solution[10] ^ complete_parameters[1] * solution[15] ^ (1 - complete_parameters[1]) + (1 - complete_parameters[7]) * exp(-((complete_parameters[3] + complete_parameters[8] * 0))) * solution[10]),
        solution[1] * solution[4] - solution[14],
        ((solution[14] - 1) + solution[5]) - solution[11],
        solution[7] - exp(complete_parameters[8] * 0),
        solution[16] - solution[10] ^ complete_parameters[1] * solution[15] ^ (1 - complete_parameters[1]) * exp(-(complete_parameters[1]) * (complete_parameters[3] + complete_parameters[8] * 0)),
        solution[9] - (solution[6] * solution[16]) / solution[16],
        solution[8] - ((solution[1] / solution[1]) * solution[14]) / solution[6],
        solution[13] - log(solution[9]),
        solution[12] - log(solution[8]),
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[17] - complete_parameters[3],
        solution[6] - exp(solution[17]),
        (-(complete_parameters[5]) * log(solution[14]) - (1 - complete_parameters[5]) * log(complete_parameters[4])) + log(solution[14]),
        solution[18] - -(complete_parameters[1]) * (complete_parameters[3] + log(solution[7])),
        solution[19] - (-(complete_parameters[3]) - log(solution[7])),
        (complete_parameters[2] * (complete_parameters[1] * solution[10] ^ (complete_parameters[1] - 1) * solution[15] ^ (1 - complete_parameters[1]) * exp(solution[18]) + (1 - complete_parameters[7]) * exp(solution[19]))) / (solution[4] * solution[14]) - 1 / (solution[4] * solution[14]),
        solution[3] - solution[11] / solution[15],
        (-(solution[1]) * solution[4] * complete_parameters[6]) / ((1 - solution[15]) * (1 - complete_parameters[6])) + solution[11] / solution[15],
        solution[20] - -(complete_parameters[1]) * complete_parameters[3],
        (-(solution[1]) * solution[10] ^ complete_parameters[1] * (1 - complete_parameters[1]) * exp(solution[20])) / (solution[3] * solution[15] ^ complete_parameters[1]) + solution[2],
        (-(complete_parameters[2]) * solution[10] ^ complete_parameters[1] * solution[15] ^ (1 - complete_parameters[1]) * (1 - complete_parameters[1]) * exp(solution[20])) / (solution[4] * solution[11] * solution[14]) + 1 / (solution[1] * solution[4]),
        solution[21] - -(complete_parameters[3]),
        ((solution[4] - solution[10] * (1 - complete_parameters[7]) * exp(solution[21])) + solution[10]) - solution[10] ^ complete_parameters[1] * solution[15] ^ (1 - complete_parameters[1]) * exp(solution[20]),
        solution[1] * solution[4] - solution[14],
        ((solution[5] - solution[11]) + solution[14]) - 1,
        solution[7] - 1.0,
        -(solution[10] ^ complete_parameters[1]) * solution[15] ^ (1 - complete_parameters[1]) * exp(solution[20]) + solution[16],
        -(solution[6]) + solution[9],
        solution[8] - solution[14] / solution[6],
        solution[13] - log(solution[9]),
        solution[12] - log(solution[8]),
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
