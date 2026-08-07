module Backus_Kehoe_Kydland_1992NsssResiduals
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

const MODEL_NAME = "Backus_Kehoe_Kydland_1992"
const SOURCE_MODEL_FILE = "models/Backus_Kehoe_Kydland_1992.jl"
const NSSS_SOLUTION_ERROR = 1.1504097182708125e-15
const NSSS_RESIDUAL_NORM = 9.411201764041564e-16

const PARAMETER_NAMES = [
    "K_ss",
    "mu{F}",
    "mu{H}",
    "gamma{F}",
    "gamma{H}",
    "alpha{F}",
    "alpha{H}",
    "eta{F}",
    "eta{H}",
    "theta{F}",
    "theta{H}",
    "nu{F}",
    "nu{H}",
    "sigma{F}",
    "sigma{H}",
    "delta{F}",
    "delta{H}",
    "psi{F}",
    "psi{H}",
    "Z_E{F}",
    "Z_E{H}",
    "rho{H}{H}",
    "rho{H}{F}",
    "phi{F}",
    "phi{H}",
]
const PARAMETER_VALUES = Float64[
    11.0,
    0.34,
    0.34,
    -1.0,
    -1.0,
    1.0,
    1.0,
    0.5,
    0.5,
    0.36,
    0.36,
    3.0,
    3.0,
    0.01,
    0.01,
    0.025,
    0.025,
    0.5,
    0.5,
    0.00852,
    0.00852,
    0.906,
    0.088,
    0.25,
    0.25,
]
const COMPLETE_PARAMETER_NAMES = [
    "K_ss",
    "mu{F}",
    "mu{H}",
    "gamma{F}",
    "gamma{H}",
    "alpha{F}",
    "alpha{H}",
    "eta{F}",
    "eta{H}",
    "theta{F}",
    "theta{H}",
    "nu{F}",
    "nu{H}",
    "sigma{F}",
    "sigma{H}",
    "delta{F}",
    "delta{H}",
    "psi{F}",
    "psi{H}",
    "Z_E{F}",
    "Z_E{H}",
    "rho{H}{H}",
    "rho{H}{F}",
    "phi{F}",
    "phi{H}",
    "rho{F}{F}",
    "rho{F}{H}",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    11.0,
    0.34,
    0.34,
    -1.0,
    -1.0,
    1.0,
    1.0,
    0.5,
    0.5,
    0.36,
    0.36,
    3.0,
    3.0,
    0.01,
    0.01,
    0.025,
    0.025,
    0.5,
    0.5,
    0.00852,
    0.00852,
    0.906,
    0.088,
    0.25,
    0.25,
    0.906,
    0.088,
]
const ORIGINAL_SOLUTION_NAMES = [
    "A{F}",
    "A{H}",
    "C{F}",
    "C{H}",
    "K{F}",
    "K{H}",
    "LAMBDA{F}",
    "LAMBDA{H}",
    "LGM",
    "L{F}",
    "L{H}",
    "NX{F}",
    "NX{H}",
    "N{F}",
    "N{H}",
    "S{F}",
    "S{H}",
    "U{F}",
    "U{H}",
    "X{F}",
    "X{H}",
    "Y{F}",
    "Y{H}",
    "Z{F}",
    "Z{H}",
    "beta{F}",
    "beta{H}",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    0.6063247747434655,
    0.6063247747434652,
    0.8257911472128492,
    0.825791147212849,
    11.0,
    11.0,
    1.0000000000000009,
    1.0000000000000009,
    0.2788534213100842,
    0.6968376126282674,
    0.6968376126282674,
    1.0085682714983567e-16,
    -2.017136542996714e-16,
    0.30316238737173273,
    0.3031623873717327,
    0.275,
    0.275,
    1.354556980516955,
    1.354556980516955,
    0.275,
    0.275,
    1.1007911472128493,
    1.1007911472128489,
    1.0973730994934199,
    1.0973730994934117,
    0.9899763180304794,
    0.9899763180304794,
]
const ORIGINAL_INITIAL_SOLUTION_VALUES = Float64[
    0.0,
    0.0,
    5.0e11,
    5.0e11,
    11.0,
    11.0,
    0.0,
    0.0,
    0.0,
    5.0e11,
    5.0e11,
    -5.50048828125e-13,
    -5.50048828125e-13,
    5.0e11,
    5.0e11,
    0.275,
    0.275,
    0.0,
    0.0,
    0.275,
    0.275,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    0.0,
    0.0,
]
const AUXILIARY_SOLUTION_NAMES = [
    "A{F}",
    "A{H}",
    "C{F}",
    "C{H}",
    "K{F}",
    "K{H}",
    "LAMBDA{F}",
    "LAMBDA{H}",
    "LGM",
    "L{F}",
    "L{H}",
    "NX{F}",
    "NX{H}",
    "N{F}",
    "N{H}",
    "S{F}",
    "S{H}",
    "U{F}",
    "U{H}",
    "X{F}",
    "X{H}",
    "Y{F}",
    "Y{H}",
    "Z{F}",
    "Z{H}",
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
    "➕₆",
    "➕₇",
    "➕₈",
    "beta{F}",
    "beta{H}",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    0.6063247747434655,
    0.6063247747434652,
    0.8257911472128492,
    0.825791147212849,
    11.0,
    11.0,
    1.0000000000000009,
    1.0000000000000009,
    0.2788534213100842,
    0.6968376126282674,
    0.6968376126282674,
    4.034273085993426e-16,
    0.0,
    0.30316238737173273,
    0.3031623873717327,
    0.275,
    0.275,
    1.3545569805169553,
    1.3545569805169555,
    0.275,
    0.275,
    1.1007911472128495,
    1.100791147212849,
    1.0973730994934199,
    1.0973730994934117,
    1.1045199649645074,
    0.7496960373944875,
    0.738248751719812,
    1.1045199649645074,
    1.1045199649645077,
    0.7496960373944869,
    0.7382487517198121,
    1.1045199649645077,
    0.9899763180304794,
    0.9899763180304794,
]
const AUXILIARY_INITIAL_SOLUTION_VALUES = Float64[
    0.0,
    0.0,
    5.0e11,
    5.0e11,
    11.0,
    11.0,
    0.0,
    0.0,
    0.0,
    5.0e11,
    5.0e11,
    -5.50048828125e-13,
    -5.50048828125e-13,
    5.0e11,
    5.0e11,
    0.275,
    0.275,
    0.0,
    0.0,
    0.275,
    0.275,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
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
    "➕₁₁",
    "➕₁₂",
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    1.1045199649645074,
    0.7496960373944875,
    0.738248751719812,
    1.1045199649645074,
    1.1045199649645077,
    0.7496960373944869,
    0.7382487517198121,
    1.1045199649645077,
    11.0,
    11.0,
    11.0,
    11.0,
]
const ALL_AUXILIARY_VARIABLE_INITIAL_VALUES = Float64[
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    11.0,
    11.0,
    11.0,
    11.0,
]
const DEFAULTED_NSSS_SOLUTION_NAMES = [
]
const CALIBRATION_PARAMETER_NAMES = [
    "beta{F}",
    "beta{H}",
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(Y◖H◗ - ((LAMBDA◖H◗ * K◖H◗ ^ theta◖H◗ * N◖H◗ ^ (1 - theta◖H◗)) ^ -nu◖H◗ + sigma◖H◗ * Z◖H◗ ^ -nu◖H◗) ^ (-1 / nu◖H◗)),
    :(K◖H◗ - ((1 - delta◖H◗) * K◖H◗ + S◖H◗)),
    :(X◖H◗ - (phi◖H◗ * S◖H◗ + phi◖H◗ * S◖H◗ + phi◖H◗ * S◖H◗ + phi◖H◗ * S◖H◗)),
    :(A◖H◗ - ((1 - eta◖H◗) * A◖H◗ + N◖H◗)),
    :(L◖H◗ - ((1 - alpha◖H◗ * N◖H◗) - (1 - alpha◖H◗) * eta◖H◗ * A◖H◗)),
    :(U◖H◗ - (C◖H◗ ^ mu◖H◗ * L◖H◗ ^ (1 - mu◖H◗)) ^ gamma◖H◗),
    :(((psi◖H◗ * mu◖H◗) / C◖H◗) * U◖H◗ - LGM),
    :(((psi◖H◗ * (1 - mu◖H◗)) / L◖H◗) * U◖H◗ * -alpha◖H◗ - ((-LGM * (1 - theta◖H◗)) / N◖H◗) * (LAMBDA◖H◗ * K◖H◗ ^ theta◖H◗ * N◖H◗ ^ (1 - theta◖H◗)) ^ -nu◖H◗ * Y◖H◗ ^ (1 + nu◖H◗)),
    :(((beta◖H◗ ^ 0 * LGM * phi◖H◗ + beta◖H◗ ^ 1 * LGM * phi◖H◗ + beta◖H◗ ^ 2 * LGM * phi◖H◗ + beta◖H◗ ^ 3 * LGM * phi◖H◗) + (-(beta◖H◗ ^ 1) * LGM * phi◖H◗ * (1 - delta◖H◗) + -(beta◖H◗ ^ 2) * LGM * phi◖H◗ * (1 - delta◖H◗) + -(beta◖H◗ ^ 3) * LGM * phi◖H◗ * (1 - delta◖H◗) + -(beta◖H◗ ^ 4) * LGM * phi◖H◗ * (1 - delta◖H◗))) - ((beta◖H◗ ^ 4 * LGM * theta◖H◗) / K◖H◗) * (LAMBDA◖H◗ * K◖H◗ ^ theta◖H◗ * N◖H◗ ^ (1 - theta◖H◗)) ^ -nu◖H◗ * Y◖H◗ ^ (1 + nu◖H◗)),
    :(LGM - beta◖H◗ * LGM * (1 + sigma◖H◗ * Z◖H◗ ^ (-nu◖H◗ - 1) * Y◖H◗ ^ (1 + nu◖H◗))),
    :(NX◖H◗ - (Y◖H◗ - ((C◖H◗ + X◖H◗ + Z◖H◗) - Z◖H◗)) / Y◖H◗),
    :(Y◖F◗ - ((LAMBDA◖F◗ * K◖F◗ ^ theta◖F◗ * N◖F◗ ^ (1 - theta◖F◗)) ^ -nu◖F◗ + sigma◖F◗ * Z◖F◗ ^ -nu◖F◗) ^ (-1 / nu◖F◗)),
    :(K◖F◗ - ((1 - delta◖F◗) * K◖F◗ + S◖F◗)),
    :(X◖F◗ - (phi◖F◗ * S◖F◗ + phi◖F◗ * S◖F◗ + phi◖F◗ * S◖F◗ + phi◖F◗ * S◖F◗)),
    :(A◖F◗ - ((1 - eta◖F◗) * A◖F◗ + N◖F◗)),
    :(L◖F◗ - ((1 - alpha◖F◗ * N◖F◗) - (1 - alpha◖F◗) * eta◖F◗ * A◖F◗)),
    :(U◖F◗ - (C◖F◗ ^ mu◖F◗ * L◖F◗ ^ (1 - mu◖F◗)) ^ gamma◖F◗),
    :(((psi◖F◗ * mu◖F◗) / C◖F◗) * U◖F◗ - LGM),
    :(((psi◖F◗ * (1 - mu◖F◗)) / L◖F◗) * U◖F◗ * -alpha◖F◗ - ((-LGM * (1 - theta◖F◗)) / N◖F◗) * (LAMBDA◖F◗ * K◖F◗ ^ theta◖F◗ * N◖F◗ ^ (1 - theta◖F◗)) ^ -nu◖F◗ * Y◖F◗ ^ (1 + nu◖F◗)),
    :(((beta◖F◗ ^ 0 * LGM * phi◖F◗ + beta◖F◗ ^ 1 * LGM * phi◖F◗ + beta◖F◗ ^ 2 * LGM * phi◖F◗ + beta◖F◗ ^ 3 * LGM * phi◖F◗) + (-(beta◖F◗ ^ 1) * LGM * phi◖F◗ * (1 - delta◖F◗) + -(beta◖F◗ ^ 2) * LGM * phi◖F◗ * (1 - delta◖F◗) + -(beta◖F◗ ^ 3) * LGM * phi◖F◗ * (1 - delta◖F◗) + -(beta◖F◗ ^ 4) * LGM * phi◖F◗ * (1 - delta◖F◗))) - ((beta◖F◗ ^ 4 * LGM * theta◖F◗) / K◖F◗) * (LAMBDA◖F◗ * K◖F◗ ^ theta◖F◗ * N◖F◗ ^ (1 - theta◖F◗)) ^ -nu◖F◗ * Y◖F◗ ^ (1 + nu◖F◗)),
    :(LGM - beta◖F◗ * LGM * (1 + sigma◖F◗ * Z◖F◗ ^ (-nu◖F◗ - 1) * Y◖F◗ ^ (1 + nu◖F◗))),
    :(NX◖F◗ - (Y◖F◗ - ((C◖F◗ + X◖F◗ + Z◖F◗) - Z◖F◗)) / Y◖F◗),
    :((LAMBDA◖H◗ - 1) - (rho◖H◗◖H◗ * (LAMBDA◖H◗ - 1) + rho◖H◗◖F◗ * (LAMBDA◖F◗ - 1) + Z_E◖H◗ * 0)),
    :((LAMBDA◖F◗ - 1) - (rho◖F◗◖F◗ * (LAMBDA◖F◗ - 1) + rho◖F◗◖H◗ * (LAMBDA◖H◗ - 1) + Z_E◖F◗ * 0)),
    :((((C◖H◗ + X◖H◗ + Z◖H◗) - Z◖H◗) + ((C◖F◗ + X◖F◗ + Z◖F◗) - Z◖F◗)) - (Y◖H◗ + Y◖F◗)),
]
const CALIBRATION_EQUATIONS = Expr[
    :(K◖F◗ - K_ss),
    :(K◖H◗ - K_ss),
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :(➕₁ - K◖H◗ ^ theta◖H◗ * LAMBDA◖H◗ * N◖H◗ ^ (1 - theta◖H◗)),
    :(➕₂ - (➕₁ ^ -nu◖H◗ + sigma◖H◗ / Z◖H◗ ^ nu◖H◗)),
    :(Y◖H◗ - 1 / ➕₂ ^ (1 / nu◖H◗)),
    :((-K◖H◗ * (1 - delta◖H◗) + K◖H◗) - S◖H◗),
    :(-4 * S◖H◗ * phi◖H◗ + X◖H◗),
    :((-A◖H◗ * (1 - eta◖H◗) + A◖H◗) - N◖H◗),
    :((A◖H◗ * eta◖H◗ * (1 - alpha◖H◗) + L◖H◗ + N◖H◗ * alpha◖H◗) - 1),
    :(➕₃ - C◖H◗ ^ mu◖H◗ * L◖H◗ ^ (1 - mu◖H◗)),
    :(U◖H◗ - ➕₃ ^ gamma◖H◗),
    :(-LGM + (U◖H◗ * mu◖H◗ * psi◖H◗) / C◖H◗),
    :((LGM * Y◖H◗ ^ (nu◖H◗ + 1) * (1 - theta◖H◗)) / (N◖H◗ * ➕₁ ^ nu◖H◗) - (U◖H◗ * alpha◖H◗ * psi◖H◗ * (1 - mu◖H◗)) / L◖H◗),
    :(➕₄ - K◖H◗ ^ theta◖H◗ * LAMBDA◖H◗ * N◖H◗ ^ (1 - theta◖H◗)),
    :(((((((-LGM * beta◖H◗ ^ 4 * phi◖H◗ * (1 - delta◖H◗) - LGM * beta◖H◗ ^ 3 * phi◖H◗ * (1 - delta◖H◗)) + LGM * beta◖H◗ ^ 3 * phi◖H◗) - LGM * beta◖H◗ ^ 2 * phi◖H◗ * (1 - delta◖H◗)) + LGM * beta◖H◗ ^ 2 * phi◖H◗) - LGM * beta◖H◗ * phi◖H◗ * (1 - delta◖H◗)) + LGM * beta◖H◗ * phi◖H◗ + LGM * phi◖H◗) - (LGM * Y◖H◗ ^ (nu◖H◗ + 1) * beta◖H◗ ^ 4 * theta◖H◗) / (K◖H◗ * ➕₄ ^ nu◖H◗)),
    :(-LGM * beta◖H◗ * (Y◖H◗ ^ (nu◖H◗ + 1) * Z◖H◗ ^ (-nu◖H◗ - 1) * sigma◖H◗ + 1) + LGM),
    :(NX◖H◗ - ((-C◖H◗ - X◖H◗) + Y◖H◗) / Y◖H◗),
    :(➕₅ - K◖F◗ ^ theta◖F◗ * LAMBDA◖F◗ * N◖F◗ ^ (1 - theta◖F◗)),
    :(➕₆ - (➕₅ ^ -nu◖F◗ + sigma◖F◗ / Z◖F◗ ^ nu◖F◗)),
    :(Y◖F◗ - 1 / ➕₆ ^ (1 / nu◖F◗)),
    :((-K◖F◗ * (1 - delta◖F◗) + K◖F◗) - S◖F◗),
    :(-4 * S◖F◗ * phi◖F◗ + X◖F◗),
    :((-A◖F◗ * (1 - eta◖F◗) + A◖F◗) - N◖F◗),
    :((A◖F◗ * eta◖F◗ * (1 - alpha◖F◗) + L◖F◗ + N◖F◗ * alpha◖F◗) - 1),
    :(➕₇ - C◖F◗ ^ mu◖F◗ * L◖F◗ ^ (1 - mu◖F◗)),
    :(U◖F◗ - ➕₇ ^ gamma◖F◗),
    :(-LGM + (U◖F◗ * mu◖F◗ * psi◖F◗) / C◖F◗),
    :((LGM * Y◖F◗ ^ (nu◖F◗ + 1) * (1 - theta◖F◗)) / (N◖F◗ * ➕₅ ^ nu◖F◗) - (U◖F◗ * alpha◖F◗ * psi◖F◗ * (1 - mu◖F◗)) / L◖F◗),
    :(➕₈ - K◖F◗ ^ theta◖F◗ * LAMBDA◖F◗ * N◖F◗ ^ (1 - theta◖F◗)),
    :(((((((-LGM * beta◖F◗ ^ 4 * phi◖F◗ * (1 - delta◖F◗) - LGM * beta◖F◗ ^ 3 * phi◖F◗ * (1 - delta◖F◗)) + LGM * beta◖F◗ ^ 3 * phi◖F◗) - LGM * beta◖F◗ ^ 2 * phi◖F◗ * (1 - delta◖F◗)) + LGM * beta◖F◗ ^ 2 * phi◖F◗) - LGM * beta◖F◗ * phi◖F◗ * (1 - delta◖F◗)) + LGM * beta◖F◗ * phi◖F◗ + LGM * phi◖F◗) - (LGM * Y◖F◗ ^ (nu◖F◗ + 1) * beta◖F◗ ^ 4 * theta◖F◗) / (K◖F◗ * ➕₈ ^ nu◖F◗)),
    :(-LGM * beta◖F◗ * (Y◖F◗ ^ (nu◖F◗ + 1) * Z◖F◗ ^ (-nu◖F◗ - 1) * sigma◖F◗ + 1) + LGM),
    :(NX◖F◗ - ((-C◖F◗ - X◖F◗) + Y◖F◗) / Y◖F◗),
    :(((LAMBDA◖H◗ - rho◖H◗◖F◗ * (LAMBDA◖F◗ - 1)) - rho◖H◗◖H◗ * (LAMBDA◖H◗ - 1)) - 1),
    :(((LAMBDA◖F◗ - rho◖F◗◖F◗ * (LAMBDA◖F◗ - 1)) - rho◖F◗◖H◗ * (LAMBDA◖H◗ - 1)) - 1),
    :(((C◖F◗ + C◖H◗ + X◖F◗ + X◖H◗) - Y◖F◗) - Y◖H◗),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(Y◖H◗ - ((LAMBDA◖H◗ * K◖H◗ ^ theta◖H◗ * N◖H◗ ^ (1 - theta◖H◗)) ^ -nu◖H◗ + sigma◖H◗ * Z◖H◗ ^ -nu◖H◗) ^ (-1 / nu◖H◗)),
    :(K◖H◗ - ((1 - delta◖H◗) * K◖H◗ + S◖H◗)),
    :(X◖H◗ - (phi◖H◗ * S◖H◗ + phi◖H◗ * S◖H◗ + phi◖H◗ * S◖H◗ + phi◖H◗ * S◖H◗)),
    :(A◖H◗ - ((1 - eta◖H◗) * A◖H◗ + N◖H◗)),
    :(L◖H◗ - ((1 - alpha◖H◗ * N◖H◗) - (1 - alpha◖H◗) * eta◖H◗ * A◖H◗)),
    :(U◖H◗ - (C◖H◗ ^ mu◖H◗ * L◖H◗ ^ (1 - mu◖H◗)) ^ gamma◖H◗),
    :(((psi◖H◗ * mu◖H◗) / C◖H◗) * U◖H◗ - LGM),
    :(((psi◖H◗ * (1 - mu◖H◗)) / L◖H◗) * U◖H◗ * -alpha◖H◗ - ((-LGM * (1 - theta◖H◗)) / N◖H◗) * (LAMBDA◖H◗ * K◖H◗ ^ theta◖H◗ * N◖H◗ ^ (1 - theta◖H◗)) ^ -nu◖H◗ * Y◖H◗ ^ (1 + nu◖H◗)),
    :(((beta◖H◗ ^ 0 * LGM * phi◖H◗ + beta◖H◗ ^ 1 * LGM * phi◖H◗ + beta◖H◗ ^ 2 * LGM * phi◖H◗ + beta◖H◗ ^ 3 * LGM * phi◖H◗) + (-(beta◖H◗ ^ 1) * LGM * phi◖H◗ * (1 - delta◖H◗) + -(beta◖H◗ ^ 2) * LGM * phi◖H◗ * (1 - delta◖H◗) + -(beta◖H◗ ^ 3) * LGM * phi◖H◗ * (1 - delta◖H◗) + -(beta◖H◗ ^ 4) * LGM * phi◖H◗ * (1 - delta◖H◗))) - ((beta◖H◗ ^ 4 * LGM * theta◖H◗) / K◖H◗) * (LAMBDA◖H◗ * K◖H◗ ^ theta◖H◗ * N◖H◗ ^ (1 - theta◖H◗)) ^ -nu◖H◗ * Y◖H◗ ^ (1 + nu◖H◗)),
    :(LGM - beta◖H◗ * LGM * (1 + sigma◖H◗ * Z◖H◗ ^ (-nu◖H◗ - 1) * Y◖H◗ ^ (1 + nu◖H◗))),
    :(NX◖H◗ - (Y◖H◗ - ((C◖H◗ + X◖H◗ + Z◖H◗) - Z◖H◗)) / Y◖H◗),
    :(Y◖F◗ - ((LAMBDA◖F◗ * K◖F◗ ^ theta◖F◗ * N◖F◗ ^ (1 - theta◖F◗)) ^ -nu◖F◗ + sigma◖F◗ * Z◖F◗ ^ -nu◖F◗) ^ (-1 / nu◖F◗)),
    :(K◖F◗ - ((1 - delta◖F◗) * K◖F◗ + S◖F◗)),
    :(X◖F◗ - (phi◖F◗ * S◖F◗ + phi◖F◗ * S◖F◗ + phi◖F◗ * S◖F◗ + phi◖F◗ * S◖F◗)),
    :(A◖F◗ - ((1 - eta◖F◗) * A◖F◗ + N◖F◗)),
    :(L◖F◗ - ((1 - alpha◖F◗ * N◖F◗) - (1 - alpha◖F◗) * eta◖F◗ * A◖F◗)),
    :(U◖F◗ - (C◖F◗ ^ mu◖F◗ * L◖F◗ ^ (1 - mu◖F◗)) ^ gamma◖F◗),
    :(((psi◖F◗ * mu◖F◗) / C◖F◗) * U◖F◗ - LGM),
    :(((psi◖F◗ * (1 - mu◖F◗)) / L◖F◗) * U◖F◗ * -alpha◖F◗ - ((-LGM * (1 - theta◖F◗)) / N◖F◗) * (LAMBDA◖F◗ * K◖F◗ ^ theta◖F◗ * N◖F◗ ^ (1 - theta◖F◗)) ^ -nu◖F◗ * Y◖F◗ ^ (1 + nu◖F◗)),
    :(((beta◖F◗ ^ 0 * LGM * phi◖F◗ + beta◖F◗ ^ 1 * LGM * phi◖F◗ + beta◖F◗ ^ 2 * LGM * phi◖F◗ + beta◖F◗ ^ 3 * LGM * phi◖F◗) + (-(beta◖F◗ ^ 1) * LGM * phi◖F◗ * (1 - delta◖F◗) + -(beta◖F◗ ^ 2) * LGM * phi◖F◗ * (1 - delta◖F◗) + -(beta◖F◗ ^ 3) * LGM * phi◖F◗ * (1 - delta◖F◗) + -(beta◖F◗ ^ 4) * LGM * phi◖F◗ * (1 - delta◖F◗))) - ((beta◖F◗ ^ 4 * LGM * theta◖F◗) / K◖F◗) * (LAMBDA◖F◗ * K◖F◗ ^ theta◖F◗ * N◖F◗ ^ (1 - theta◖F◗)) ^ -nu◖F◗ * Y◖F◗ ^ (1 + nu◖F◗)),
    :(LGM - beta◖F◗ * LGM * (1 + sigma◖F◗ * Z◖F◗ ^ (-nu◖F◗ - 1) * Y◖F◗ ^ (1 + nu◖F◗))),
    :(NX◖F◗ - (Y◖F◗ - ((C◖F◗ + X◖F◗ + Z◖F◗) - Z◖F◗)) / Y◖F◗),
    :((LAMBDA◖H◗ - 1) - (rho◖H◗◖H◗ * (LAMBDA◖H◗ - 1) + rho◖H◗◖F◗ * (LAMBDA◖F◗ - 1) + Z_E◖H◗ * 0)),
    :((LAMBDA◖F◗ - 1) - (rho◖F◗◖F◗ * (LAMBDA◖F◗ - 1) + rho◖F◗◖H◗ * (LAMBDA◖H◗ - 1) + Z_E◖F◗ * 0)),
    :((((C◖H◗ + X◖H◗ + Z◖H◗) - Z◖H◗) + ((C◖F◗ + X◖F◗ + Z◖F◗) - Z◖F◗)) - (Y◖H◗ + Y◖F◗)),
    :(K◖F◗ - K_ss),
    :(K◖H◗ - K_ss),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :(➕₁ - K◖H◗ ^ theta◖H◗ * LAMBDA◖H◗ * N◖H◗ ^ (1 - theta◖H◗)),
    :(➕₂ - (➕₁ ^ -nu◖H◗ + sigma◖H◗ / Z◖H◗ ^ nu◖H◗)),
    :(Y◖H◗ - 1 / ➕₂ ^ (1 / nu◖H◗)),
    :((-K◖H◗ * (1 - delta◖H◗) + K◖H◗) - S◖H◗),
    :(-4 * S◖H◗ * phi◖H◗ + X◖H◗),
    :((-A◖H◗ * (1 - eta◖H◗) + A◖H◗) - N◖H◗),
    :((A◖H◗ * eta◖H◗ * (1 - alpha◖H◗) + L◖H◗ + N◖H◗ * alpha◖H◗) - 1),
    :(➕₃ - C◖H◗ ^ mu◖H◗ * L◖H◗ ^ (1 - mu◖H◗)),
    :(U◖H◗ - ➕₃ ^ gamma◖H◗),
    :(-LGM + (U◖H◗ * mu◖H◗ * psi◖H◗) / C◖H◗),
    :((LGM * Y◖H◗ ^ (nu◖H◗ + 1) * (1 - theta◖H◗)) / (N◖H◗ * ➕₁ ^ nu◖H◗) - (U◖H◗ * alpha◖H◗ * psi◖H◗ * (1 - mu◖H◗)) / L◖H◗),
    :(➕₄ - K◖H◗ ^ theta◖H◗ * LAMBDA◖H◗ * N◖H◗ ^ (1 - theta◖H◗)),
    :(((((((-LGM * beta◖H◗ ^ 4 * phi◖H◗ * (1 - delta◖H◗) - LGM * beta◖H◗ ^ 3 * phi◖H◗ * (1 - delta◖H◗)) + LGM * beta◖H◗ ^ 3 * phi◖H◗) - LGM * beta◖H◗ ^ 2 * phi◖H◗ * (1 - delta◖H◗)) + LGM * beta◖H◗ ^ 2 * phi◖H◗) - LGM * beta◖H◗ * phi◖H◗ * (1 - delta◖H◗)) + LGM * beta◖H◗ * phi◖H◗ + LGM * phi◖H◗) - (LGM * Y◖H◗ ^ (nu◖H◗ + 1) * beta◖H◗ ^ 4 * theta◖H◗) / (K◖H◗ * ➕₄ ^ nu◖H◗)),
    :(-LGM * beta◖H◗ * (Y◖H◗ ^ (nu◖H◗ + 1) * Z◖H◗ ^ (-nu◖H◗ - 1) * sigma◖H◗ + 1) + LGM),
    :(NX◖H◗ - ((-C◖H◗ - X◖H◗) + Y◖H◗) / Y◖H◗),
    :(➕₅ - K◖F◗ ^ theta◖F◗ * LAMBDA◖F◗ * N◖F◗ ^ (1 - theta◖F◗)),
    :(➕₆ - (➕₅ ^ -nu◖F◗ + sigma◖F◗ / Z◖F◗ ^ nu◖F◗)),
    :(Y◖F◗ - 1 / ➕₆ ^ (1 / nu◖F◗)),
    :((-K◖F◗ * (1 - delta◖F◗) + K◖F◗) - S◖F◗),
    :(-4 * S◖F◗ * phi◖F◗ + X◖F◗),
    :((-A◖F◗ * (1 - eta◖F◗) + A◖F◗) - N◖F◗),
    :((A◖F◗ * eta◖F◗ * (1 - alpha◖F◗) + L◖F◗ + N◖F◗ * alpha◖F◗) - 1),
    :(➕₇ - C◖F◗ ^ mu◖F◗ * L◖F◗ ^ (1 - mu◖F◗)),
    :(U◖F◗ - ➕₇ ^ gamma◖F◗),
    :(-LGM + (U◖F◗ * mu◖F◗ * psi◖F◗) / C◖F◗),
    :((LGM * Y◖F◗ ^ (nu◖F◗ + 1) * (1 - theta◖F◗)) / (N◖F◗ * ➕₅ ^ nu◖F◗) - (U◖F◗ * alpha◖F◗ * psi◖F◗ * (1 - mu◖F◗)) / L◖F◗),
    :(➕₈ - K◖F◗ ^ theta◖F◗ * LAMBDA◖F◗ * N◖F◗ ^ (1 - theta◖F◗)),
    :(((((((-LGM * beta◖F◗ ^ 4 * phi◖F◗ * (1 - delta◖F◗) - LGM * beta◖F◗ ^ 3 * phi◖F◗ * (1 - delta◖F◗)) + LGM * beta◖F◗ ^ 3 * phi◖F◗) - LGM * beta◖F◗ ^ 2 * phi◖F◗ * (1 - delta◖F◗)) + LGM * beta◖F◗ ^ 2 * phi◖F◗) - LGM * beta◖F◗ * phi◖F◗ * (1 - delta◖F◗)) + LGM * beta◖F◗ * phi◖F◗ + LGM * phi◖F◗) - (LGM * Y◖F◗ ^ (nu◖F◗ + 1) * beta◖F◗ ^ 4 * theta◖F◗) / (K◖F◗ * ➕₈ ^ nu◖F◗)),
    :(-LGM * beta◖F◗ * (Y◖F◗ ^ (nu◖F◗ + 1) * Z◖F◗ ^ (-nu◖F◗ - 1) * sigma◖F◗ + 1) + LGM),
    :(NX◖F◗ - ((-C◖F◗ - X◖F◗) + Y◖F◗) / Y◖F◗),
    :(((LAMBDA◖H◗ - rho◖H◗◖F◗ * (LAMBDA◖F◗ - 1)) - rho◖H◗◖H◗ * (LAMBDA◖H◗ - 1)) - 1),
    :(((LAMBDA◖F◗ - rho◖F◗◖F◗ * (LAMBDA◖F◗ - 1)) - rho◖F◗◖H◗ * (LAMBDA◖H◗ - 1)) - 1),
    :(((C◖F◗ + C◖H◗ + X◖F◗ + X◖H◗) - Y◖F◗) - Y◖H◗),
    :(K◖F◗ - K_ss),
    :(K◖H◗ - K_ss),
]

const PARAMETER_DEFINITION_NAMES = [
    "rho{F}{F}",
    "rho{F}{H}",
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
    "rho{H}{H}",
    "rho{H}{F}",
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "K_ss",
    "mu{F}",
    "mu{H}",
    "gamma{F}",
    "gamma{H}",
    "alpha{F}",
    "alpha{H}",
    "eta{F}",
    "eta{H}",
    "theta{F}",
    "theta{H}",
    "nu{F}",
    "nu{H}",
    "sigma{F}",
    "sigma{H}",
    "delta{F}",
    "delta{H}",
    "psi{F}",
    "psi{H}",
    "Z_E{F}",
    "Z_E{H}",
    "rho{H}{H}",
    "rho{H}{F}",
    "phi{F}",
    "phi{H}",
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
    "A{F}",
    "A{H}",
    "C{F}",
    "C{H}",
    "K{F}",
    "K{H}",
    "LAMBDA{F}",
    "LAMBDA{H}",
    "LGM",
    "L{F}",
    "L{H}",
    "NX{F}",
    "NX{H}",
    "N{F}",
    "N{H}",
    "S{F}",
    "S{H}",
    "U{F}",
    "U{H}",
    "X{F}",
    "X{H}",
    "Y{F}",
    "Y{H}",
    "Z{F}",
    "Z{H}",
    "beta{F}",
    "beta{H}",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -1.0e12,
    -1.0e12,
]
const ORIGINAL_BOX_UPPER_BOUNDS = Float64[
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
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
    1.0e12,
    1.0e12,
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "A{F}",
    "A{H}",
    "C{F}",
    "C{H}",
    "K{F}",
    "K{H}",
    "LAMBDA{F}",
    "LAMBDA{H}",
    "LGM",
    "L{F}",
    "L{H}",
    "NX{F}",
    "NX{H}",
    "N{F}",
    "N{H}",
    "S{F}",
    "S{H}",
    "U{F}",
    "U{H}",
    "X{F}",
    "X{H}",
    "Y{F}",
    "Y{H}",
    "Z{F}",
    "Z{H}",
    "➕₁",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
    "➕₆",
    "➕₇",
    "➕₈",
    "beta{F}",
    "beta{H}",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -Inf,
    -Inf,
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
    2.220446049250313e-16,
    2.220446049250313e-16,
    -1.0e12,
    -1.0e12,
]
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
    Inf,
    1.0e12,
    1.0e12,
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
    1.0e12,
    1.0e12,
]

const BLOCKS = [
    (
        index = 1,
        solve_order = 10,
        variables = ["NX{H}"],
        previous_solution_names = ["C{H}", "X{H}", "Y{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [15],
        equations = Expr[
            :(NX◖H◗ - ((-C◖H◗ - X◖H◗) + Y◖H◗) / Y◖H◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["NX{H}"],
        previous_solution_values = [0.825791147212849, 0.275, 1.100791147212849],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [5.0e11, 0.275, 5.0e11],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-5.50048828125e-13],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 2,
        solve_order = 9,
        variables = ["NX{F}"],
        previous_solution_names = ["C{F}", "X{F}", "Y{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [30],
        equations = Expr[
            :(NX◖F◗ - ((-C◖F◗ - X◖F◗) + Y◖F◗) / Y◖F◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["NX{F}"],
        previous_solution_values = [0.8257911472128492, 0.275, 1.1007911472128495],
        external_solution_values = Float64[],
        solution_values = [4.034273085993426e-16],
        previous_solution_initial_values = [5.0e11, 0.275, 5.0e11],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-5.50048828125e-13],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 3,
        solve_order = 8,
        variables = ["A{F}", "A{H}", "C{F}", "C{H}", "LGM", "L{F}", "L{H}", "N{F}", "N{H}", "U{F}", "U{H}", "Y{F}", "Y{H}", "Z{F}", "Z{H}", "beta{F}", "beta{H}", "➕₁", "➕₂", "➕₃", "➕₄", "➕₅", "➕₆", "➕₇", "➕₈"],
        previous_solution_names = ["K{F}", "K{H}", "LAMBDA{F}", "LAMBDA{H}", "X{F}", "X{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₁", "➕₁₂"],
        equation_indices = [22, 6, 33, 8, 25, 26, 7, 21, 12, 24, 10, 28, 11, 17, 2, 1, 3, 9, 13, 16, 18, 23, 27, 29, 14],
        equations = Expr[
            :((A◖F◗ * eta◖F◗ * (1 - alpha◖F◗) + L◖F◗ + N◖F◗ * alpha◖F◗) - 1),
            :((-A◖H◗ * (1 - eta◖H◗) + A◖H◗) - N◖H◗),
            :(((C◖F◗ + C◖H◗ + X◖F◗ + X◖H◗) - Y◖F◗) - Y◖H◗),
            :(➕₃ - C◖H◗ ^ mu◖H◗ * L◖H◗ ^ (1 - mu◖H◗)),
            :(-LGM + (U◖F◗ * mu◖F◗ * psi◖F◗) / C◖F◗),
            :((LGM * Y◖F◗ ^ (nu◖F◗ + 1) * (1 - theta◖F◗)) / (N◖F◗ * ➕₅ ^ nu◖F◗) - (U◖F◗ * alpha◖F◗ * psi◖F◗ * (1 - mu◖F◗)) / L◖F◗),
            :((A◖H◗ * eta◖H◗ * (1 - alpha◖H◗) + L◖H◗ + N◖H◗ * alpha◖H◗) - 1),
            :((-A◖F◗ * (1 - eta◖F◗) + A◖F◗) - N◖F◗),
            :(➕₄ - ➕₁₁ ^ theta◖H◗ * LAMBDA◖H◗ * N◖H◗ ^ (1 - theta◖H◗)),
            :(U◖F◗ - ➕₇ ^ gamma◖F◗),
            :(-LGM + (U◖H◗ * mu◖H◗ * psi◖H◗) / C◖H◗),
            :(((((((-LGM * beta◖F◗ ^ 4 * phi◖F◗ * (1 - delta◖F◗) - LGM * beta◖F◗ ^ 3 * phi◖F◗ * (1 - delta◖F◗)) + LGM * beta◖F◗ ^ 3 * phi◖F◗) - LGM * beta◖F◗ ^ 2 * phi◖F◗ * (1 - delta◖F◗)) + LGM * beta◖F◗ ^ 2 * phi◖F◗) - LGM * beta◖F◗ * phi◖F◗ * (1 - delta◖F◗)) + LGM * beta◖F◗ * phi◖F◗ + LGM * phi◖F◗) - (LGM * Y◖F◗ ^ (nu◖F◗ + 1) * beta◖F◗ ^ 4 * theta◖F◗) / (K◖F◗ * ➕₈ ^ nu◖F◗)),
            :((LGM * Y◖H◗ ^ (nu◖H◗ + 1) * (1 - theta◖H◗)) / (N◖H◗ * ➕₁ ^ nu◖H◗) - (U◖H◗ * alpha◖H◗ * psi◖H◗ * (1 - mu◖H◗)) / L◖H◗),
            :(➕₆ - (➕₅ ^ -nu◖F◗ + sigma◖F◗ / Z◖F◗ ^ nu◖F◗)),
            :(➕₂ - (➕₁ ^ -nu◖H◗ + sigma◖H◗ / Z◖H◗ ^ nu◖H◗)),
            :(➕₁ - ➕₁₁ ^ theta◖H◗ * LAMBDA◖H◗ * N◖H◗ ^ (1 - theta◖H◗)),
            :(Y◖H◗ - 1 / ➕₂ ^ (1 / nu◖H◗)),
            :(U◖H◗ - ➕₃ ^ gamma◖H◗),
            :(((((((-LGM * beta◖H◗ ^ 4 * phi◖H◗ * (1 - delta◖H◗) - LGM * beta◖H◗ ^ 3 * phi◖H◗ * (1 - delta◖H◗)) + LGM * beta◖H◗ ^ 3 * phi◖H◗) - LGM * beta◖H◗ ^ 2 * phi◖H◗ * (1 - delta◖H◗)) + LGM * beta◖H◗ ^ 2 * phi◖H◗) - LGM * beta◖H◗ * phi◖H◗ * (1 - delta◖H◗)) + LGM * beta◖H◗ * phi◖H◗ + LGM * phi◖H◗) - (LGM * Y◖H◗ ^ (nu◖H◗ + 1) * beta◖H◗ ^ 4 * theta◖H◗) / (K◖H◗ * ➕₄ ^ nu◖H◗)),
            :(➕₅ - ➕₁₂ ^ theta◖F◗ * LAMBDA◖F◗ * N◖F◗ ^ (1 - theta◖F◗)),
            :(Y◖F◗ - 1 / ➕₆ ^ (1 / nu◖F◗)),
            :(➕₇ - C◖F◗ ^ mu◖F◗ * L◖F◗ ^ (1 - mu◖F◗)),
            :(➕₈ - ➕₁₂ ^ theta◖F◗ * LAMBDA◖F◗ * N◖F◗ ^ (1 - theta◖F◗)),
            :(-LGM * beta◖F◗ * (Y◖F◗ ^ (nu◖F◗ + 1) * Z◖F◗ ^ (-nu◖F◗ - 1) * sigma◖F◗ + 1) + LGM),
            :(-LGM * beta◖H◗ * (Y◖H◗ ^ (nu◖H◗ + 1) * Z◖H◗ ^ (-nu◖H◗ - 1) * sigma◖H◗ + 1) + LGM),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₁ = min(1.0e12, max(eps(), K◖H◗))),
            :(➕₁₂ = min(1.0e12, max(eps(), K◖F◗))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₁ - K◖H◗)),
            :(abs(➕₁₂ - K◖F◗)),
        ],
        solution_names = ["A{F}", "A{H}", "C{F}", "C{H}", "LGM", "L{F}", "L{H}", "N{F}", "N{H}", "U{F}", "U{H}", "Y{F}", "Y{H}", "Z{F}", "Z{H}", "beta{F}", "beta{H}", "➕₁", "➕₂", "➕₃", "➕₄", "➕₅", "➕₆", "➕₇", "➕₈", "➕₁₁", "➕₁₂"],
        previous_solution_values = [11.0, 11.0, 1.0000000000000009, 1.0000000000000009, 0.275, 0.275],
        external_solution_values = Float64[],
        solution_values = [0.6063247747434655, 0.6063247747434652, 0.8257911472128492, 0.825791147212849, 0.2788534213100842, 0.6968376126282674, 0.6968376126282674, 0.30316238737173273, 0.3031623873717327, 1.3545569805169553, 1.3545569805169555, 1.1007911472128495, 1.100791147212849, 1.0973730994934199, 1.0973730994934117, 0.9899763180304794, 0.9899763180304794, 1.1045199649645074, 0.7496960373944875, 0.738248751719812, 1.1045199649645074, 1.1045199649645077, 0.7496960373944869, 0.7382487517198121, 1.1045199649645077, 11.0, 11.0],
        previous_solution_initial_values = [11.0, 11.0, 0.0, 0.0, 0.275, 0.275],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 0.0, 5.0e11, 5.0e11, 0.0, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 0.0, 0.0, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 0.0, 0.0, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 11.0, 11.0],
        box_lower_bounds = [-1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 4,
        solve_order = 7,
        variables = ["X{H}"],
        previous_solution_names = ["S{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [5],
        equations = Expr[
            :(-4 * S◖H◗ * phi◖H◗ + X◖H◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["X{H}"],
        previous_solution_values = [0.275],
        external_solution_values = Float64[],
        solution_values = [0.275],
        previous_solution_initial_values = [0.275],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.275],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 5,
        solve_order = 6,
        variables = ["S{H}"],
        previous_solution_names = ["K{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [4],
        equations = Expr[
            :((-K◖H◗ * (1 - delta◖H◗) + K◖H◗) - S◖H◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["S{H}"],
        previous_solution_values = [11.0],
        external_solution_values = Float64[],
        solution_values = [0.275],
        previous_solution_initial_values = [11.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.275],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 6,
        solve_order = 5,
        variables = ["X{F}"],
        previous_solution_names = ["S{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [20],
        equations = Expr[
            :(-4 * S◖F◗ * phi◖F◗ + X◖F◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["X{F}"],
        previous_solution_values = [0.275],
        external_solution_values = Float64[],
        solution_values = [0.275],
        previous_solution_initial_values = [0.275],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.275],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 7,
        solve_order = 4,
        variables = ["S{F}"],
        previous_solution_names = ["K{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [19],
        equations = Expr[
            :((-K◖F◗ * (1 - delta◖F◗) + K◖F◗) - S◖F◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["S{F}"],
        previous_solution_values = [11.0],
        external_solution_values = Float64[],
        solution_values = [0.275],
        previous_solution_initial_values = [11.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.275],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 8,
        solve_order = 3,
        variables = ["K{F}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [34],
        equations = Expr[
            :(K◖F◗ - K_ss),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["K{F}"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [11.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [11.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 9,
        solve_order = 2,
        variables = ["LAMBDA{F}", "LAMBDA{H}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [32, 31],
        equations = Expr[
            :(((LAMBDA◖F◗ - rho◖F◗◖F◗ * (LAMBDA◖F◗ - 1)) - rho◖F◗◖H◗ * (LAMBDA◖H◗ - 1)) - 1),
            :(((LAMBDA◖H◗ - rho◖H◗◖F◗ * (LAMBDA◖F◗ - 1)) - rho◖H◗◖H◗ * (LAMBDA◖H◗ - 1)) - 1),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["LAMBDA{F}", "LAMBDA{H}"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.0000000000000009, 1.0000000000000009],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 0.0],
        box_lower_bounds = [-1.0e12, -1.0e12],
        box_upper_bounds = [1.0e12, 1.0e12],
    ),
    (
        index = 10,
        solve_order = 1,
        variables = ["K{H}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [35],
        equations = Expr[
            :(K◖H◗ - K_ss),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["K{H}"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [11.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [11.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
]
const BLOCK_EQUATION_ORDER = [15, 30, 22, 6, 33, 8, 25, 26, 7, 21, 12, 24, 10, 28, 11, 17, 2, 1, 3, 9, 13, 16, 18, 23, 27, 29, 14, 5, 4, 20, 19, 34, 32, 31, 35]
const BLOCK_SOLVE_ORDER = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    ["C{H}", "X{H}", "Y{H}"],
    ["C{F}", "X{F}", "Y{F}"],
    ["K{F}", "K{H}", "LAMBDA{F}", "LAMBDA{H}", "X{F}", "X{H}"],
    ["S{H}"],
    ["K{H}"],
    ["S{F}"],
    ["K{F}"],
    String[],
    String[],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    [0.825791147212849, 0.275, 1.100791147212849],
    [0.8257911472128492, 0.275, 1.1007911472128495],
    [11.0, 11.0, 1.0000000000000009, 1.0000000000000009, 0.275, 0.275],
    [0.275],
    [11.0],
    [0.275],
    [11.0],
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
]
const BLOCK_SOLUTION_NAMES = [
    ["NX{H}"],
    ["NX{F}"],
    ["A{F}", "A{H}", "C{F}", "C{H}", "LGM", "L{F}", "L{H}", "N{F}", "N{H}", "U{F}", "U{H}", "Y{F}", "Y{H}", "Z{F}", "Z{H}", "beta{F}", "beta{H}", "➕₁", "➕₂", "➕₃", "➕₄", "➕₅", "➕₆", "➕₇", "➕₈", "➕₁₁", "➕₁₂"],
    ["X{H}"],
    ["S{H}"],
    ["X{F}"],
    ["S{F}"],
    ["K{F}"],
    ["LAMBDA{F}", "LAMBDA{H}"],
    ["K{H}"],
]
const BLOCK_SOLUTION_VALUES = [
    [0.0],
    [4.034273085993426e-16],
    [0.6063247747434655, 0.6063247747434652, 0.8257911472128492, 0.825791147212849, 0.2788534213100842, 0.6968376126282674, 0.6968376126282674, 0.30316238737173273, 0.3031623873717327, 1.3545569805169553, 1.3545569805169555, 1.1007911472128495, 1.100791147212849, 1.0973730994934199, 1.0973730994934117, 0.9899763180304794, 0.9899763180304794, 1.1045199649645074, 0.7496960373944875, 0.738248751719812, 1.1045199649645074, 1.1045199649645077, 0.7496960373944869, 0.7382487517198121, 1.1045199649645077, 11.0, 11.0],
    [0.275],
    [0.275],
    [0.275],
    [0.275],
    [11.0],
    [1.0000000000000009, 1.0000000000000009],
    [11.0],
]
const BLOCK_PREVIOUS_SOLUTION_INITIAL_VALUES = [
    [5.0e11, 0.275, 5.0e11],
    [5.0e11, 0.275, 5.0e11],
    [11.0, 11.0, 0.0, 0.0, 0.275, 0.275],
    [0.275],
    [11.0],
    [0.275],
    [11.0],
    Float64[],
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
]
const BLOCK_SOLUTION_INITIAL_VALUES = [
    [-5.50048828125e-13],
    [-5.50048828125e-13],
    [0.0, 0.0, 5.0e11, 5.0e11, 0.0, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 0.0, 0.0, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 0.0, 0.0, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 5.0e11, 11.0, 11.0],
    [0.275],
    [0.275],
    [0.275],
    [0.275],
    [11.0],
    [0.0, 0.0],
    [11.0],
]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[4] = parameters[4]
    complete_parameters[6] = parameters[6]
    complete_parameters[11] = parameters[11]
    complete_parameters[23] = parameters[23]
    complete_parameters[16] = parameters[16]
    complete_parameters[17] = parameters[17]
    complete_parameters[21] = parameters[21]
    complete_parameters[25] = parameters[25]
    complete_parameters[1] = parameters[1]
    complete_parameters[3] = parameters[3]
    complete_parameters[19] = parameters[19]
    complete_parameters[22] = parameters[22]
    complete_parameters[10] = parameters[10]
    complete_parameters[13] = parameters[13]
    complete_parameters[9] = parameters[9]
    complete_parameters[24] = parameters[24]
    complete_parameters[20] = parameters[20]
    complete_parameters[18] = parameters[18]
    complete_parameters[12] = parameters[12]
    complete_parameters[7] = parameters[7]
    complete_parameters[8] = parameters[8]
    complete_parameters[2] = parameters[2]
    complete_parameters[15] = parameters[15]
    complete_parameters[14] = parameters[14]
    complete_parameters[5] = parameters[5]
    complete_parameters[26] = complete_parameters[22]
    complete_parameters[27] = complete_parameters[23]
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[23] - ((solution[8] * solution[6] ^ complete_parameters[11] * solution[15] ^ (1 - complete_parameters[11])) ^ -(complete_parameters[13]) + complete_parameters[15] * solution[25] ^ -(complete_parameters[13])) ^ (-1 / complete_parameters[13]),
        solution[6] - ((1 - complete_parameters[17]) * solution[6] + solution[17]),
        solution[21] - (complete_parameters[25] * solution[17] + complete_parameters[25] * solution[17] + complete_parameters[25] * solution[17] + complete_parameters[25] * solution[17]),
        solution[2] - ((1 - complete_parameters[9]) * solution[2] + solution[15]),
        solution[11] - ((1 - complete_parameters[7] * solution[15]) - (1 - complete_parameters[7]) * complete_parameters[9] * solution[2]),
        solution[19] - (solution[4] ^ complete_parameters[3] * solution[11] ^ (1 - complete_parameters[3])) ^ complete_parameters[5],
        ((complete_parameters[19] * complete_parameters[3]) / solution[4]) * solution[19] - solution[9],
        ((complete_parameters[19] * (1 - complete_parameters[3])) / solution[11]) * solution[19] * -(complete_parameters[7]) - ((-(solution[9]) * (1 - complete_parameters[11])) / solution[15]) * (solution[8] * solution[6] ^ complete_parameters[11] * solution[15] ^ (1 - complete_parameters[11])) ^ -(complete_parameters[13]) * solution[23] ^ (1 + complete_parameters[13]),
        ((solution[27] ^ 0 * solution[9] * complete_parameters[25] + solution[27] ^ 1 * solution[9] * complete_parameters[25] + solution[27] ^ 2 * solution[9] * complete_parameters[25] + solution[27] ^ 3 * solution[9] * complete_parameters[25]) + (-(solution[27] ^ 1) * solution[9] * complete_parameters[25] * (1 - complete_parameters[17]) + -(solution[27] ^ 2) * solution[9] * complete_parameters[25] * (1 - complete_parameters[17]) + -(solution[27] ^ 3) * solution[9] * complete_parameters[25] * (1 - complete_parameters[17]) + -(solution[27] ^ 4) * solution[9] * complete_parameters[25] * (1 - complete_parameters[17]))) - ((solution[27] ^ 4 * solution[9] * complete_parameters[11]) / solution[6]) * (solution[8] * solution[6] ^ complete_parameters[11] * solution[15] ^ (1 - complete_parameters[11])) ^ -(complete_parameters[13]) * solution[23] ^ (1 + complete_parameters[13]),
        solution[9] - solution[27] * solution[9] * (1 + complete_parameters[15] * solution[25] ^ (-(complete_parameters[13]) - 1) * solution[23] ^ (1 + complete_parameters[13])),
        solution[13] - (solution[23] - ((solution[4] + solution[21] + solution[25]) - solution[25])) / solution[23],
        solution[22] - ((solution[7] * solution[5] ^ complete_parameters[10] * solution[14] ^ (1 - complete_parameters[10])) ^ -(complete_parameters[12]) + complete_parameters[14] * solution[24] ^ -(complete_parameters[12])) ^ (-1 / complete_parameters[12]),
        solution[5] - ((1 - complete_parameters[16]) * solution[5] + solution[16]),
        solution[20] - (complete_parameters[24] * solution[16] + complete_parameters[24] * solution[16] + complete_parameters[24] * solution[16] + complete_parameters[24] * solution[16]),
        solution[1] - ((1 - complete_parameters[8]) * solution[1] + solution[14]),
        solution[10] - ((1 - complete_parameters[6] * solution[14]) - (1 - complete_parameters[6]) * complete_parameters[8] * solution[1]),
        solution[18] - (solution[3] ^ complete_parameters[2] * solution[10] ^ (1 - complete_parameters[2])) ^ complete_parameters[4],
        ((complete_parameters[18] * complete_parameters[2]) / solution[3]) * solution[18] - solution[9],
        ((complete_parameters[18] * (1 - complete_parameters[2])) / solution[10]) * solution[18] * -(complete_parameters[6]) - ((-(solution[9]) * (1 - complete_parameters[10])) / solution[14]) * (solution[7] * solution[5] ^ complete_parameters[10] * solution[14] ^ (1 - complete_parameters[10])) ^ -(complete_parameters[12]) * solution[22] ^ (1 + complete_parameters[12]),
        ((solution[26] ^ 0 * solution[9] * complete_parameters[24] + solution[26] ^ 1 * solution[9] * complete_parameters[24] + solution[26] ^ 2 * solution[9] * complete_parameters[24] + solution[26] ^ 3 * solution[9] * complete_parameters[24]) + (-(solution[26] ^ 1) * solution[9] * complete_parameters[24] * (1 - complete_parameters[16]) + -(solution[26] ^ 2) * solution[9] * complete_parameters[24] * (1 - complete_parameters[16]) + -(solution[26] ^ 3) * solution[9] * complete_parameters[24] * (1 - complete_parameters[16]) + -(solution[26] ^ 4) * solution[9] * complete_parameters[24] * (1 - complete_parameters[16]))) - ((solution[26] ^ 4 * solution[9] * complete_parameters[10]) / solution[5]) * (solution[7] * solution[5] ^ complete_parameters[10] * solution[14] ^ (1 - complete_parameters[10])) ^ -(complete_parameters[12]) * solution[22] ^ (1 + complete_parameters[12]),
        solution[9] - solution[26] * solution[9] * (1 + complete_parameters[14] * solution[24] ^ (-(complete_parameters[12]) - 1) * solution[22] ^ (1 + complete_parameters[12])),
        solution[12] - (solution[22] - ((solution[3] + solution[20] + solution[24]) - solution[24])) / solution[22],
        (solution[8] - 1) - (complete_parameters[22] * (solution[8] - 1) + complete_parameters[23] * (solution[7] - 1) + complete_parameters[21] * 0),
        (solution[7] - 1) - (complete_parameters[26] * (solution[7] - 1) + complete_parameters[27] * (solution[8] - 1) + complete_parameters[20] * 0),
        (((solution[4] + solution[21] + solution[25]) - solution[25]) + ((solution[3] + solution[20] + solution[24]) - solution[24])) - (solution[23] + solution[22]),
        solution[5] - complete_parameters[1],
        solution[6] - complete_parameters[1],
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[26] - solution[6] ^ complete_parameters[11] * solution[8] * solution[15] ^ (1 - complete_parameters[11]),
        solution[27] - (solution[26] ^ -(complete_parameters[13]) + complete_parameters[15] / solution[25] ^ complete_parameters[13]),
        solution[23] - 1 / solution[27] ^ (1 / complete_parameters[13]),
        (-(solution[6]) * (1 - complete_parameters[17]) + solution[6]) - solution[17],
        -4 * solution[17] * complete_parameters[25] + solution[21],
        (-(solution[2]) * (1 - complete_parameters[9]) + solution[2]) - solution[15],
        (solution[2] * complete_parameters[9] * (1 - complete_parameters[7]) + solution[11] + solution[15] * complete_parameters[7]) - 1,
        solution[28] - solution[4] ^ complete_parameters[3] * solution[11] ^ (1 - complete_parameters[3]),
        solution[19] - solution[28] ^ complete_parameters[5],
        -(solution[9]) + (solution[19] * complete_parameters[3] * complete_parameters[19]) / solution[4],
        (solution[9] * solution[23] ^ (complete_parameters[13] + 1) * (1 - complete_parameters[11])) / (solution[15] * solution[26] ^ complete_parameters[13]) - (solution[19] * complete_parameters[7] * complete_parameters[19] * (1 - complete_parameters[3])) / solution[11],
        solution[29] - solution[6] ^ complete_parameters[11] * solution[8] * solution[15] ^ (1 - complete_parameters[11]),
        ((((((-(solution[9]) * solution[35] ^ 4 * complete_parameters[25] * (1 - complete_parameters[17]) - solution[9] * solution[35] ^ 3 * complete_parameters[25] * (1 - complete_parameters[17])) + solution[9] * solution[35] ^ 3 * complete_parameters[25]) - solution[9] * solution[35] ^ 2 * complete_parameters[25] * (1 - complete_parameters[17])) + solution[9] * solution[35] ^ 2 * complete_parameters[25]) - solution[9] * solution[35] * complete_parameters[25] * (1 - complete_parameters[17])) + solution[9] * solution[35] * complete_parameters[25] + solution[9] * complete_parameters[25]) - (solution[9] * solution[23] ^ (complete_parameters[13] + 1) * solution[35] ^ 4 * complete_parameters[11]) / (solution[6] * solution[29] ^ complete_parameters[13]),
        -(solution[9]) * solution[35] * (solution[23] ^ (complete_parameters[13] + 1) * solution[25] ^ (-(complete_parameters[13]) - 1) * complete_parameters[15] + 1) + solution[9],
        solution[13] - ((-(solution[4]) - solution[21]) + solution[23]) / solution[23],
        solution[30] - solution[5] ^ complete_parameters[10] * solution[7] * solution[14] ^ (1 - complete_parameters[10]),
        solution[31] - (solution[30] ^ -(complete_parameters[12]) + complete_parameters[14] / solution[24] ^ complete_parameters[12]),
        solution[22] - 1 / solution[31] ^ (1 / complete_parameters[12]),
        (-(solution[5]) * (1 - complete_parameters[16]) + solution[5]) - solution[16],
        -4 * solution[16] * complete_parameters[24] + solution[20],
        (-(solution[1]) * (1 - complete_parameters[8]) + solution[1]) - solution[14],
        (solution[1] * complete_parameters[8] * (1 - complete_parameters[6]) + solution[10] + solution[14] * complete_parameters[6]) - 1,
        solution[32] - solution[3] ^ complete_parameters[2] * solution[10] ^ (1 - complete_parameters[2]),
        solution[18] - solution[32] ^ complete_parameters[4],
        -(solution[9]) + (solution[18] * complete_parameters[2] * complete_parameters[18]) / solution[3],
        (solution[9] * solution[22] ^ (complete_parameters[12] + 1) * (1 - complete_parameters[10])) / (solution[14] * solution[30] ^ complete_parameters[12]) - (solution[18] * complete_parameters[6] * complete_parameters[18] * (1 - complete_parameters[2])) / solution[10],
        solution[33] - solution[5] ^ complete_parameters[10] * solution[7] * solution[14] ^ (1 - complete_parameters[10]),
        ((((((-(solution[9]) * solution[34] ^ 4 * complete_parameters[24] * (1 - complete_parameters[16]) - solution[9] * solution[34] ^ 3 * complete_parameters[24] * (1 - complete_parameters[16])) + solution[9] * solution[34] ^ 3 * complete_parameters[24]) - solution[9] * solution[34] ^ 2 * complete_parameters[24] * (1 - complete_parameters[16])) + solution[9] * solution[34] ^ 2 * complete_parameters[24]) - solution[9] * solution[34] * complete_parameters[24] * (1 - complete_parameters[16])) + solution[9] * solution[34] * complete_parameters[24] + solution[9] * complete_parameters[24]) - (solution[9] * solution[22] ^ (complete_parameters[12] + 1) * solution[34] ^ 4 * complete_parameters[10]) / (solution[5] * solution[33] ^ complete_parameters[12]),
        -(solution[9]) * solution[34] * (solution[22] ^ (complete_parameters[12] + 1) * solution[24] ^ (-(complete_parameters[12]) - 1) * complete_parameters[14] + 1) + solution[9],
        solution[12] - ((-(solution[3]) - solution[20]) + solution[22]) / solution[22],
        ((solution[8] - complete_parameters[23] * (solution[7] - 1)) - complete_parameters[22] * (solution[8] - 1)) - 1,
        ((solution[7] - complete_parameters[26] * (solution[7] - 1)) - complete_parameters[27] * (solution[8] - 1)) - 1,
        ((solution[3] + solution[4] + solution[20] + solution[21]) - solution[22]) - solution[23],
        solution[5] - complete_parameters[1],
        solution[6] - complete_parameters[1],
    ]
end

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - ((-(previous_solution[1]) - previous_solution[2]) + previous_solution[3]) / previous_solution[3],
    ]
end

function residuals_block_2(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - ((-(previous_solution[1]) - previous_solution[2]) + previous_solution[3]) / previous_solution[3],
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 6
    @assert length(external_solution) == 0
    @assert length(solution) == 27
    complete_parameters = complete_parameter_values(parameters)
    return [
        (solution[1] * complete_parameters[8] * (1 - complete_parameters[6]) + solution[6] + solution[8] * complete_parameters[6]) - 1,
        (-(solution[2]) * (1 - complete_parameters[9]) + solution[2]) - solution[9],
        ((solution[3] + solution[4] + previous_solution[5] + previous_solution[6]) - solution[12]) - solution[13],
        solution[20] - solution[4] ^ complete_parameters[3] * solution[7] ^ (1 - complete_parameters[3]),
        -(solution[5]) + (solution[10] * complete_parameters[2] * complete_parameters[18]) / solution[3],
        (solution[5] * solution[12] ^ (complete_parameters[12] + 1) * (1 - complete_parameters[10])) / (solution[8] * solution[22] ^ complete_parameters[12]) - (solution[10] * complete_parameters[6] * complete_parameters[18] * (1 - complete_parameters[2])) / solution[6],
        (solution[2] * complete_parameters[9] * (1 - complete_parameters[7]) + solution[7] + solution[9] * complete_parameters[7]) - 1,
        (-(solution[1]) * (1 - complete_parameters[8]) + solution[1]) - solution[8],
        solution[21] - solution[26] ^ complete_parameters[11] * previous_solution[4] * solution[9] ^ (1 - complete_parameters[11]),
        solution[10] - solution[24] ^ complete_parameters[4],
        -(solution[5]) + (solution[11] * complete_parameters[3] * complete_parameters[19]) / solution[4],
        ((((((-(solution[5]) * solution[16] ^ 4 * complete_parameters[24] * (1 - complete_parameters[16]) - solution[5] * solution[16] ^ 3 * complete_parameters[24] * (1 - complete_parameters[16])) + solution[5] * solution[16] ^ 3 * complete_parameters[24]) - solution[5] * solution[16] ^ 2 * complete_parameters[24] * (1 - complete_parameters[16])) + solution[5] * solution[16] ^ 2 * complete_parameters[24]) - solution[5] * solution[16] * complete_parameters[24] * (1 - complete_parameters[16])) + solution[5] * solution[16] * complete_parameters[24] + solution[5] * complete_parameters[24]) - (solution[5] * solution[12] ^ (complete_parameters[12] + 1) * solution[16] ^ 4 * complete_parameters[10]) / (previous_solution[1] * solution[25] ^ complete_parameters[12]),
        (solution[5] * solution[13] ^ (complete_parameters[13] + 1) * (1 - complete_parameters[11])) / (solution[9] * solution[18] ^ complete_parameters[13]) - (solution[11] * complete_parameters[7] * complete_parameters[19] * (1 - complete_parameters[3])) / solution[7],
        solution[23] - (solution[22] ^ -(complete_parameters[12]) + complete_parameters[14] / solution[14] ^ complete_parameters[12]),
        solution[19] - (solution[18] ^ -(complete_parameters[13]) + complete_parameters[15] / solution[15] ^ complete_parameters[13]),
        solution[18] - solution[26] ^ complete_parameters[11] * previous_solution[4] * solution[9] ^ (1 - complete_parameters[11]),
        solution[13] - 1 / solution[19] ^ (1 / complete_parameters[13]),
        solution[11] - solution[20] ^ complete_parameters[5],
        ((((((-(solution[5]) * solution[17] ^ 4 * complete_parameters[25] * (1 - complete_parameters[17]) - solution[5] * solution[17] ^ 3 * complete_parameters[25] * (1 - complete_parameters[17])) + solution[5] * solution[17] ^ 3 * complete_parameters[25]) - solution[5] * solution[17] ^ 2 * complete_parameters[25] * (1 - complete_parameters[17])) + solution[5] * solution[17] ^ 2 * complete_parameters[25]) - solution[5] * solution[17] * complete_parameters[25] * (1 - complete_parameters[17])) + solution[5] * solution[17] * complete_parameters[25] + solution[5] * complete_parameters[25]) - (solution[5] * solution[13] ^ (complete_parameters[13] + 1) * solution[17] ^ 4 * complete_parameters[11]) / (previous_solution[2] * solution[21] ^ complete_parameters[13]),
        solution[22] - solution[27] ^ complete_parameters[10] * previous_solution[3] * solution[8] ^ (1 - complete_parameters[10]),
        solution[12] - 1 / solution[23] ^ (1 / complete_parameters[12]),
        solution[24] - solution[3] ^ complete_parameters[2] * solution[6] ^ (1 - complete_parameters[2]),
        solution[25] - solution[27] ^ complete_parameters[10] * previous_solution[3] * solution[8] ^ (1 - complete_parameters[10]),
        -(solution[5]) * solution[16] * (solution[12] ^ (complete_parameters[12] + 1) * solution[14] ^ (-(complete_parameters[12]) - 1) * complete_parameters[14] + 1) + solution[5],
        -(solution[5]) * solution[17] * (solution[13] ^ (complete_parameters[13] + 1) * solution[15] ^ (-(complete_parameters[13]) - 1) * complete_parameters[15] + 1) + solution[5],
        solution[26] - min(1.0e12, max(eps(), previous_solution[2])),
        solution[27] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_4(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -4 * previous_solution[1] * complete_parameters[25] + solution[1],
    ]
end

function residuals_block_5(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(previous_solution[1]) * (1 - complete_parameters[17]) + previous_solution[1]) - solution[1],
    ]
end

function residuals_block_6(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -4 * previous_solution[1] * complete_parameters[24] + solution[1],
    ]
end

function residuals_block_7(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(previous_solution[1]) * (1 - complete_parameters[16]) + previous_solution[1]) - solution[1],
    ]
end

function residuals_block_8(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - complete_parameters[1],
    ]
end

function residuals_block_9(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((solution[1] - complete_parameters[26] * (solution[1] - 1)) - complete_parameters[27] * (solution[2] - 1)) - 1,
        ((solution[2] - complete_parameters[23] * (solution[1] - 1)) - complete_parameters[22] * (solution[2] - 1)) - 1,
    ]
end

function residuals_block_10(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - complete_parameters[1],
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
export residuals_block_1, residuals_block_2, residuals_block_3, residuals_block_4, residuals_block_5, residuals_block_6, residuals_block_7, residuals_block_8, residuals_block_9, residuals_block_10
end
