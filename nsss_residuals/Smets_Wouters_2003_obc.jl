module Smets_Wouters_2003_obcNsssResiduals
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

const MODEL_NAME = "Smets_Wouters_2003_obc"
const SOURCE_MODEL_FILE = "models/Smets_Wouters_2003_obc.jl"
const NSSS_SOLUTION_ERROR = 2.577300025439547e-14
const NSSS_RESIDUAL_NORM = 7.30653933512871e-14

const PARAMETER_NAMES = [
    "R̄",
    "lambda_p",
    "G_bar",
    "lambda_w",
    "Phi",
    "alpha",
    "beta",
    "gamma_w",
    "gamma_p",
    "h",
    "omega",
    "psi",
    "r_pi",
    "r_Y",
    "r_Delta_pi",
    "r_Delta_y",
    "sigma_c",
    "sigma_l",
    "tau",
    "varphi",
    "xi_w",
    "xi_p",
    "rho",
    "rho_b",
    "rho_L",
    "rho_I",
    "rho_a",
    "rho_G",
    "rho_pi_bar",
    "std_scaling_factor",
    "σ_eta_b",
    "σ_eta_L",
    "σ_eta_I",
    "σ_eta_a",
    "σ_eta_w",
    "σ_eta_p",
    "σ_eta_G",
    "σ_eta_R",
    "σ_eta_pi",
    "activeᵒᵇᶜshocks",
]
const PARAMETER_VALUES = Float64[
    0.0,
    0.368,
    0.362,
    0.5,
    0.819,
    0.3,
    0.99,
    0.763,
    0.469,
    0.573,
    1.0,
    0.169,
    1.684,
    0.099,
    0.14,
    0.159,
    1.353,
    2.4,
    0.025,
    6.771,
    0.737,
    0.908,
    0.961,
    0.855,
    0.889,
    0.927,
    0.823,
    0.949,
    0.924,
    10.0,
    0.336,
    3.52,
    0.085,
    0.598,
    0.6853261,
    0.7896512,
    0.325,
    0.081,
    0.017,
    0.0,
]
const COMPLETE_PARAMETER_NAMES = [
    "R̄",
    "lambda_p",
    "G_bar",
    "lambda_w",
    "Phi",
    "alpha",
    "beta",
    "gamma_w",
    "gamma_p",
    "h",
    "omega",
    "psi",
    "r_pi",
    "r_Y",
    "r_Delta_pi",
    "r_Delta_y",
    "sigma_c",
    "sigma_l",
    "tau",
    "varphi",
    "xi_w",
    "xi_p",
    "rho",
    "rho_b",
    "rho_L",
    "rho_I",
    "rho_a",
    "rho_G",
    "rho_pi_bar",
    "std_scaling_factor",
    "σ_eta_b",
    "σ_eta_L",
    "σ_eta_I",
    "σ_eta_a",
    "σ_eta_w",
    "σ_eta_p",
    "σ_eta_G",
    "σ_eta_R",
    "σ_eta_pi",
    "activeᵒᵇᶜshocks",
    "std_eta_G",
    "std_eta_I",
    "std_eta_L",
    "std_eta_R",
    "std_eta_a",
    "std_eta_b",
    "std_eta_p",
    "std_eta_pi",
    "std_eta_w",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    0.0,
    0.368,
    0.362,
    0.5,
    0.819,
    0.3,
    0.99,
    0.763,
    0.469,
    0.573,
    1.0,
    0.169,
    1.684,
    0.099,
    0.14,
    0.159,
    1.353,
    2.4,
    0.025,
    6.771,
    0.737,
    0.908,
    0.961,
    0.855,
    0.889,
    0.927,
    0.823,
    0.949,
    0.924,
    10.0,
    0.336,
    3.52,
    0.085,
    0.598,
    0.6853261,
    0.7896512,
    0.325,
    0.081,
    0.017,
    0.0,
    0.0325,
    0.0085,
    0.352,
    0.0081,
    0.0598,
    0.033600000000000005,
    0.07896512,
    0.0017000000000000001,
    0.06853261000000001,
]
const ORIGINAL_SOLUTION_NAMES = [
    "C",
    "C_f",
    "G",
    "G_f",
    "I",
    "I_f",
    "K",
    "K_f",
    "L",
    "L_f",
    "L_s",
    "L_s_f",
    "P_j_f",
    "Pi_ps_f",
    "Pi_ws_f",
    "Q",
    "Q_f",
    "R",
    "R_f",
    "T",
    "T_f",
    "U",
    "U_f",
    "W",
    "W_disutil_f",
    "W_f",
    "W_i_f",
    "Y",
    "Y_f",
    "Y_s",
    "Y_s_f",
    "epsilon_G",
    "epsilon_I",
    "epsilon_L",
    "epsilon_a",
    "epsilon_b",
    "f_1",
    "f_2",
    "g_1",
    "g_2",
    "mc",
    "mc_f",
    "nu_p",
    "nu_w",
    "pi",
    "pi_obj",
    "pi_star",
    "q",
    "q_f",
    "r_k",
    "r_k_f",
    "w_star",
    "z",
    "z_f",
    "Χᵒᵇᶜ⁺ꜝ¹ꜝ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝʳ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝˡ",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝ",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾",
    "calibr_pi_obj",
    "calibr_pi",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    1.2043777490745378,
    1.2043777490745378,
    0.362,
    0.362,
    0.4415383986956471,
    0.4415383986956452,
    17.6615359478258,
    17.661535947825776,
    1.2891159432112824,
    1.289115943211282,
    1.2891159432112824,
    1.289115943211282,
    1.0,
    0.5401411859498743,
    0.48217380688087924,
    1.0000000000000002,
    0.9999999999999998,
    1.0101010101010097,
    1.0101010101010102,
    0.362,
    0.362,
    -427.98589108169506,
    -427.98589108169494,
    1.1221034292999648,
    0.7480689528666431,
    1.1221034292999645,
    1.1221034292999645,
    2.007916147770185,
    2.007916147770184,
    2.007916147770185,
    2.007916147770184,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    8.77069946048825,
    8.77069946048825,
    48.847175846537034,
    35.70699988781946,
    0.7309941520467836,
    0.7309941520467835,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    2.4590033503396467,
    2.459003350339646,
    0.035101010101010174,
    0.03510101010101021,
    1.1221034292999648,
    1.0,
    0.9999999999999999,
    0.0,
    0.0,
    -0.010050335853501065,
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
    1.0,
    4.85722573273506e-17,
]
const ORIGINAL_INITIAL_SOLUTION_VALUES = Float64[
    0.0,
    0.0,
    0.362,
    0.362,
    0.0,
    0.0,
    0.0,
    0.0,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0101010101010102,
    1.0101010101010102,
    0.362,
    0.362,
    -1.7580009187316713e41,
    -1.7580009187316713e41,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.010050335853501506,
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
    1.0,
    -5.551115123125783e-17,
]
const AUXILIARY_SOLUTION_NAMES = [
    "C",
    "C_f",
    "G",
    "G_f",
    "I",
    "I_f",
    "K",
    "K_f",
    "L",
    "L_f",
    "L_s",
    "L_s_f",
    "P_j_f",
    "Pi_ps_f",
    "Pi_ws_f",
    "Q",
    "Q_f",
    "R",
    "R_f",
    "T",
    "T_f",
    "U",
    "U_f",
    "W",
    "W_disutil_f",
    "W_f",
    "W_i_f",
    "Y",
    "Y_f",
    "Y_s",
    "Y_s_f",
    "epsilon_G",
    "epsilon_I",
    "epsilon_L",
    "epsilon_a",
    "epsilon_b",
    "f_1",
    "f_2",
    "g_1",
    "g_2",
    "mc",
    "mc_f",
    "nu_p",
    "nu_w",
    "pi",
    "pi_obj",
    "pi_star",
    "q",
    "q_f",
    "r_k",
    "r_k_f",
    "w_star",
    "z",
    "z_f",
    "Χᵒᵇᶜ⁺ꜝ¹ꜝ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝʳ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝˡ",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝ",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾",
    "➕₁",
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
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
    "➕₆",
    "➕₇",
    "➕₈",
    "➕₉",
    "calibr_pi_obj",
    "calibr_pi",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    1.2043777490745378,
    1.2043777490745378,
    0.362,
    0.362,
    0.4415383986956471,
    0.4415383986956452,
    17.6615359478258,
    17.661535947825776,
    1.2891159432112824,
    1.289115943211282,
    1.2891159432112824,
    1.289115943211282,
    1.0,
    0.5401411859498743,
    0.48217380688087924,
    1.0000000000000002,
    0.9999999999999998,
    1.0101010101010097,
    1.0101010101010102,
    0.362,
    0.362,
    -427.98589108169506,
    -427.98589108169494,
    1.1221034292999648,
    0.7480689528666431,
    1.1221034292999645,
    1.1221034292999645,
    2.007916147770185,
    2.007916147770184,
    2.007916147770185,
    2.007916147770184,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    8.77069946048825,
    8.77069946048825,
    48.847175846537034,
    35.70699988781946,
    0.7309941520467836,
    0.7309941520467835,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    2.4590033503396467,
    2.459003350339646,
    0.035101010101010174,
    0.03510101010101021,
    1.1221034292999648,
    1.0,
    0.9999999999999999,
    0.0,
    0.0,
    -0.010050335853501065,
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
    1.0,
    1.0,
    1.0,
    1.0,
    1.2891159432112824,
    1.0,
    1.0,
    0.0,
    -1.8762769116165147e-17,
    0.9900000000000003,
    0.5142692988548276,
    -1.8762769116165147e-17,
    0.5142692988548276,
    17.6615359478258,
    17.661535947825772,
    1.0,
    0.5142692988548276,
    0.5142692988548276,
    1.0,
    4.85722573273506e-17,
]
const AUXILIARY_INITIAL_SOLUTION_VALUES = Float64[
    0.0,
    0.0,
    0.362,
    0.362,
    0.0,
    0.0,
    0.0,
    0.0,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0101010101010102,
    1.0101010101010102,
    0.362,
    0.362,
    -1.7580009187316713e41,
    -1.7580009187316713e41,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.010050335853501506,
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
    -4.999999997e11,
    1.0,
    1.0,
    1.0,
    1.0,
    5.0e11,
    1.0,
    1.0,
    -4.999999997e11,
    -4.999999997e11,
    0.99,
    5.0e11,
    -4.999999997e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    1.0,
    -5.551115123125783e-17,
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
    "➕₂₆",
    "➕₂₇",
    "➕₂₈",
    "➕₂₉",
    "➕₃₀",
    "➕₃₁",
    "➕₃₂",
    "➕₃₃",
    "➕₃₄",
    "➕₃₅",
    "➕₃₆",
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    0.0,
    0.5142692988548276,
    -1.8762769116165147e-17,
    0.5142692988548276,
    17.6615359478258,
    17.661535947825772,
    1.0,
    0.5142692988548276,
    0.5142692988548276,
    1.0,
    1.0,
    1.0,
    1.0,
    1.2891159432112824,
    1.0,
    1.0,
    0.0,
    -1.8762769116165147e-17,
    0.9900000000000003,
    1.0101010101010097,
    1.0,
    1.2891159432112824,
    1.289115943211282,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.2891159432112824,
    1.289115943211282,
    1.0101010101010097,
    1.0,
    1.0,
    1.0101010101010097,
]
const ALL_AUXILIARY_VARIABLE_INITIAL_VALUES = Float64[
    -4.999999997e11,
    5.0e11,
    -4.999999997e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    5.0e11,
    1.0,
    1.0,
    1.0,
    1.0,
    5.0e11,
    1.0,
    1.0,
    -4.999999997e11,
    -4.999999997e11,
    0.99,
    1.0101010101010102,
    5.0e11,
    5.0e11,
    5.0e11,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    5.0e11,
    5.0e11,
    5.0e11,
    1.0101010101010102,
    1.0,
    1.0,
    1.0101010101010102,
]
const DEFAULTED_NSSS_SOLUTION_NAMES = [
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾",
]
const CALIBRATION_PARAMETER_NAMES = [
    "calibr_pi_obj",
    "calibr_pi",
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(-q + beta * ((1 - tau) * q + epsilon_b * (r_k * z - psi ^ -1 * r_k * (-1 + exp(psi * (-1 + z)))) * (C - h * C) ^ -sigma_c)),
    :(-q_f + beta * ((1 - tau) * q_f + epsilon_b * (r_k_f * z_f - psi ^ -1 * r_k_f * (-1 + exp(psi * (-1 + z_f)))) * (C_f - h * C_f) ^ -sigma_c)),
    :(-r_k + alpha * epsilon_a * mc * L ^ (1 - alpha) * (K * z) ^ (-1 + alpha)),
    :(-r_k_f + alpha * epsilon_a * mc_f * L_f ^ (1 - alpha) * (K_f * z_f) ^ (-1 + alpha)),
    :(-G + T),
    :(-G + G_bar * epsilon_G),
    :(-G_f + T_f),
    :(-G_f + G_bar * epsilon_G),
    :(-L + nu_w ^ -1 * L_s),
    :(-L_s_f + L_f * (W_i_f * W_f ^ -1) ^ (lambda_w ^ -1 * (-1 - lambda_w))),
    :(L_s_f - L_f),
    :(L_s_f + lambda_w ^ -1 * L_f * W_f ^ -1 * (-1 - lambda_w) * (-W_disutil_f + W_i_f) * (W_i_f * W_f ^ -1) ^ (-1 + lambda_w ^ -1 * (-1 - lambda_w))),
    :(Pi_ws_f - L_s_f * (-W_disutil_f + W_i_f)),
    :(Pi_ps_f - Y_f * (-mc_f + P_j_f) * P_j_f ^ (-(lambda_p ^ -1) * (1 + lambda_p))),
    :(-Q + epsilon_b ^ -1 * q * (C - h * C) ^ sigma_c),
    :(-Q_f + epsilon_b ^ -1 * q_f * (C_f - h * C_f) ^ sigma_c),
    :(-W + epsilon_a * mc * (1 - alpha) * L ^ -alpha * (K * z) ^ alpha),
    :(-W_f + epsilon_a * mc_f * (1 - alpha) * L_f ^ -alpha * (K_f * z_f) ^ alpha),
    :(-Y_f + Y_s_f),
    :(Y_s - nu_p * Y),
    :(-Y_s_f + Y_f * P_j_f ^ (-(lambda_p ^ -1) * (1 + lambda_p))),
    :(beta * epsilon_b * (C_f - h * C_f) ^ -sigma_c - epsilon_b * R_f ^ -1 * (C_f - h * C_f) ^ -sigma_c),
    :(beta * epsilon_b * pi ^ -1 * (C - h * C) ^ -sigma_c - epsilon_b * R ^ -1 * (C - h * C) ^ -sigma_c),
    :(Y_f * P_j_f ^ (-(lambda_p ^ -1) * (1 + lambda_p)) - lambda_p ^ -1 * Y_f * (1 + lambda_p) * (-mc_f + P_j_f) * P_j_f ^ (-1 - lambda_p ^ -1 * (1 + lambda_p))),
    :(epsilon_b * W_disutil_f * (C_f - h * C_f) ^ -sigma_c - omega * epsilon_b * epsilon_L * L_s_f ^ sigma_l),
    :(-1 + xi_p * (pi ^ -1 * pi ^ gamma_p) ^ -(lambda_p ^ -1) + (1 - xi_p) * pi_star ^ -(lambda_p ^ -1)),
    :(-1 + (1 - xi_w) * (w_star * W ^ -1) ^ -(lambda_w ^ -1) + xi_w * (W * W ^ -1) ^ -(lambda_w ^ -1) * (pi ^ -1 * pi ^ gamma_w) ^ -(lambda_w ^ -1)),
    :((-Phi - Y_s) + epsilon_a * L ^ (1 - alpha) * (K * z) ^ alpha),
    :((-Phi - Y_f * P_j_f ^ (-(lambda_p ^ -1) * (1 + lambda_p))) + epsilon_a * L_f ^ (1 - alpha) * (K_f * z_f) ^ alpha),
    :((std_eta_b * 0 - log(epsilon_b)) + rho_b * log(epsilon_b)),
    :((-std_eta_L * 0 - log(epsilon_L)) + rho_L * log(epsilon_L)),
    :((std_eta_I * 0 - log(epsilon_I)) + rho_I * log(epsilon_I)),
    :((std_eta_w * 0 - f_1) + f_2),
    :((std_eta_a * 0 - log(epsilon_a)) + rho_a * log(epsilon_a)),
    :((std_eta_p * 0 - g_1) + g_2 * (1 + lambda_p)),
    :((std_eta_G * 0 - log(epsilon_G)) + rho_G * log(epsilon_G)),
    :(-f_1 + beta * xi_w * f_1 * (w_star ^ -1 * w_star) ^ (lambda_w ^ -1) * (pi ^ -1 * pi ^ gamma_w) ^ -(lambda_w ^ -1) + epsilon_b * w_star * L * (1 + lambda_w) ^ -1 * (C - h * C) ^ -sigma_c * (w_star * W ^ -1) ^ (-(lambda_w ^ -1) * (1 + lambda_w))),
    :(-f_2 + beta * xi_w * f_2 * (w_star ^ -1 * w_star) ^ (lambda_w ^ -1 * (1 + lambda_w) * (1 + sigma_l)) * (pi ^ -1 * pi ^ gamma_w) ^ (-(lambda_w ^ -1) * (1 + lambda_w) * (1 + sigma_l)) + omega * epsilon_b * epsilon_L * (L * (w_star * W ^ -1) ^ (-(lambda_w ^ -1) * (1 + lambda_w))) ^ (1 + sigma_l)),
    :(-g_1 + beta * xi_p * pi_star * g_1 * pi_star ^ -1 * (pi ^ -1 * pi ^ gamma_p) ^ -(lambda_p ^ -1) + epsilon_b * pi_star * Y * (C - h * C) ^ -sigma_c),
    :(-g_2 + beta * xi_p * g_2 * (pi ^ -1 * pi ^ gamma_p) ^ (-(lambda_p ^ -1) * (1 + lambda_p)) + epsilon_b * mc * Y * (C - h * C) ^ -sigma_c),
    :(-nu_w + (1 - xi_w) * (w_star * W ^ -1) ^ (-(lambda_w ^ -1) * (1 + lambda_w)) + xi_w * nu_w * (W * pi ^ -1 * W ^ -1 * pi ^ gamma_w) ^ (-(lambda_w ^ -1) * (1 + lambda_w))),
    :(-nu_p + (1 - xi_p) * pi_star ^ (-(lambda_p ^ -1) * (1 + lambda_p)) + xi_p * nu_p * (pi ^ -1 * pi ^ gamma_p) ^ (-(lambda_p ^ -1) * (1 + lambda_p))),
    :(-K + K * (1 - tau) + I * (1 - 0.5 * varphi * (-1 + I ^ -1 * epsilon_I * I) ^ 2)),
    :(-K_f + K_f * (1 - tau) + I_f * (1 - 0.5 * varphi * (-1 + I_f ^ -1 * epsilon_I * I_f) ^ 2)),
    :((U - beta * U) - epsilon_b * ((1 - sigma_c) ^ -1 * (C - h * C) ^ (1 - sigma_c) - omega * epsilon_L * (1 + sigma_l) ^ -1 * L_s ^ (1 + sigma_l))),
    :((U_f - beta * U_f) - epsilon_b * ((1 - sigma_c) ^ -1 * (C_f - h * C_f) ^ (1 - sigma_c) - omega * epsilon_L * (1 + sigma_l) ^ -1 * L_s_f ^ (1 + sigma_l))),
    :(-epsilon_b * (C - h * C) ^ -sigma_c + q * ((1 - 0.5 * varphi * (-1 + I ^ -1 * epsilon_I * I) ^ 2) - varphi * I ^ -1 * epsilon_I * I * (-1 + I ^ -1 * epsilon_I * I)) + beta * varphi * I ^ -2 * epsilon_I * q * I ^ 2 * (-1 + I ^ -1 * epsilon_I * I)),
    :(-epsilon_b * (C_f - h * C_f) ^ -sigma_c + q_f * ((1 - 0.5 * varphi * (-1 + I_f ^ -1 * epsilon_I * I_f) ^ 2) - varphi * I_f ^ -1 * epsilon_I * I_f * (-1 + I_f ^ -1 * epsilon_I * I_f)) + beta * varphi * I_f ^ -2 * epsilon_I * q_f * I_f ^ 2 * (-1 + I_f ^ -1 * epsilon_I * I_f)),
    :((((-C - I) - T) + Y) - psi ^ -1 * r_k * K * (-1 + exp(psi * (-1 + z)))),
    :((((((-C_f - I_f) + Pi_ws_f) - T_f) + Y_f + L_s_f * W_disutil_f) - L_f * W_f) - psi ^ -1 * r_k_f * K_f * (-1 + exp(psi * (-1 + z_f)))),
    :(epsilon_b * (K * r_k - r_k * K * exp(psi * (-1 + z))) * (C - h * C) ^ -sigma_c),
    :(epsilon_b * (K_f * r_k_f - r_k_f * K_f * exp(psi * (-1 + z_f))) * (C_f - h * C_f) ^ -sigma_c),
    :((std_eta_pi * 0 - log(pi_obj)) + rho_pi_bar * log(pi_obj) + log(calibr_pi_obj) * (1 - rho_pi_bar)),
    :(χᵒᵇᶜ⁺ꜝ¹ꜝˡ - (R̄ - log(R))),
    :(χᵒᵇᶜ⁺ꜝ¹ꜝʳ - (((((r_Delta_pi * (-(log(pi ^ -1 * pi)) + log(pi ^ -1 * pi)) + r_Delta_y * ((-(log(Y ^ -1 * Y)) + log(Y ^ -1 * Y) + log(Y_f ^ -1 * Y_f)) - log(Y_f ^ -1 * Y_f)) + rho * log(R ^ -1 * R) + (1 - rho) * (log(pi_obj) + r_pi * (-(log(pi_obj)) + log(pi ^ -1 * pi)) + r_Y * (log(Y ^ -1 * Y) - log(Y_f ^ -1 * Y_f)))) - calibr_pi) + std_eta_R * 0) - log(R ^ -1)) - log(R))),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - max(χᵒᵇᶜ⁺ꜝ¹ꜝˡ, χᵒᵇᶜ⁺ꜝ¹ꜝʳ)),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝ),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾ - activeᵒᵇᶜshocks * 0),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾ + activeᵒᵇᶜshocks * 0)),
]
const CALIBRATION_EQUATIONS = Expr[
    :(1 - pi_obj),
    :(pi - pi_obj),
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :(➕₁ - psi * (z - 1)),
    :(➕₂ - (-C * h + C)),
    :(beta * ((epsilon_b * (r_k * z - (r_k * (exp(➕₁) - 1)) / psi)) / ➕₂ ^ sigma_c + q * (1 - tau)) - q),
    :(➕₃ - psi * (z_f - 1)),
    :(➕₄ - (-C_f * h + C_f)),
    :(beta * ((epsilon_b * (r_k_f * z_f - (r_k_f * (exp(➕₃) - 1)) / psi)) / ➕₄ ^ sigma_c + q_f * (1 - tau)) - q_f),
    :(➕₅ - K * z),
    :(L ^ (1 - alpha) * alpha * epsilon_a * mc * ➕₅ ^ (alpha - 1) - r_k),
    :(➕₆ - K_f * z_f),
    :(L_f ^ (1 - alpha) * alpha * epsilon_a * mc_f * ➕₆ ^ (alpha - 1) - r_k_f),
    :(-G + T),
    :(-G + G_bar * epsilon_G),
    :(-G_f + T_f),
    :(G_bar * epsilon_G - G_f),
    :(-L + L_s / nu_w),
    :(➕₇ - W_i_f / W_f),
    :(L_f * ➕₇ ^ ((-lambda_w - 1) / lambda_w) - L_s_f),
    :(-L_f + L_s_f),
    :((L_f * ➕₇ ^ (-1 + (-lambda_w - 1) / lambda_w) * (-W_disutil_f + W_i_f) * (-lambda_w - 1)) / (W_f * lambda_w) + L_s_f),
    :(-L_s_f * (-W_disutil_f + W_i_f) + Pi_ws_f),
    :(Pi_ps_f - (Y_f * (P_j_f - mc_f)) / P_j_f ^ ((lambda_p + 1) / lambda_p)),
    :(➕₈ - (-C * h + C)),
    :(-Q + (q * ➕₈ ^ sigma_c) / epsilon_b),
    :(➕₉ - (-C_f * h + C_f)),
    :(-Q_f + (q_f * ➕₉ ^ sigma_c) / epsilon_b),
    :(-W + (epsilon_a * mc * ➕₅ ^ alpha * (1 - alpha)) / L ^ alpha),
    :(-W_f + (epsilon_a * mc_f * ➕₆ ^ alpha * (1 - alpha)) / L_f ^ alpha),
    :(-Y_f + Y_s_f),
    :(-Y * nu_p + Y_s),
    :(-Y_s_f + Y_f / P_j_f ^ ((lambda_p + 1) / lambda_p)),
    :((beta * epsilon_b) / ➕₄ ^ sigma_c - epsilon_b / (R_f * ➕₉ ^ sigma_c)),
    :((beta * epsilon_b) / (pi * ➕₂ ^ sigma_c) - epsilon_b / (R * ➕₈ ^ sigma_c)),
    :((-(P_j_f ^ (-1 - (lambda_p + 1) / lambda_p)) * Y_f * (P_j_f - mc_f) * (lambda_p + 1)) / lambda_p + Y_f / P_j_f ^ ((lambda_p + 1) / lambda_p)),
    :(-(L_s_f ^ sigma_l) * epsilon_L * epsilon_b * omega + (W_disutil_f * epsilon_b) / ➕₉ ^ sigma_c),
    :(➕₁₀ - pi ^ gamma_p / pi),
    :((xi_p / ➕₁₀ ^ (1 / lambda_p) - 1) + (1 - xi_p) / pi_star ^ (1 / lambda_p)),
    :(➕₁₁ - w_star / W),
    :(➕₁₂ - pi ^ gamma_w / pi),
    :((xi_w / ➕₁₂ ^ (1 / lambda_w) - 1) + (1 - xi_w) / ➕₁₁ ^ (1 / lambda_w)),
    :((L ^ (1 - alpha) * epsilon_a * ➕₅ ^ alpha - Phi) - Y_s),
    :((L_f ^ (1 - alpha) * epsilon_a * ➕₆ ^ alpha - Phi) - Y_f / P_j_f ^ ((lambda_p + 1) / lambda_p)),
    :(rho_b * log(epsilon_b) - log(epsilon_b)),
    :(rho_L * log(epsilon_L) - log(epsilon_L)),
    :(rho_I * log(epsilon_I) - log(epsilon_I)),
    :(-f_1 + f_2),
    :(rho_a * log(epsilon_a) - log(epsilon_a)),
    :(-g_1 + g_2 * (lambda_p + 1)),
    :(rho_G * log(epsilon_G) - log(epsilon_G)),
    :(➕₁₃ - pi ^ gamma_w / pi),
    :(((L * epsilon_b * w_star) / (➕₁₁ ^ ((lambda_w + 1) / lambda_w) * ➕₈ ^ sigma_c * (lambda_w + 1)) + (beta * f_1 * xi_w) / ➕₁₃ ^ (1 / lambda_w)) - f_1),
    :(➕₁₄ - L / ➕₁₁ ^ ((lambda_w + 1) / lambda_w)),
    :(((beta * f_2 * xi_w) / ➕₁₃ ^ (((lambda_w + 1) * (sigma_l + 1)) / lambda_w) + epsilon_L * epsilon_b * omega * ➕₁₄ ^ (sigma_l + 1)) - f_2),
    :(➕₁₅ - pi ^ gamma_p / pi),
    :(((Y * epsilon_b * pi_star) / ➕₈ ^ sigma_c + (beta * g_1 * xi_p) / ➕₁₅ ^ (1 / lambda_p)) - g_1),
    :(((Y * epsilon_b * mc) / ➕₈ ^ sigma_c + (beta * g_2 * xi_p) / ➕₁₅ ^ ((lambda_p + 1) / lambda_p)) - g_2),
    :(➕₁₆ - pi ^ gamma_w / pi),
    :(((nu_w * xi_w) / ➕₁₆ ^ ((lambda_w + 1) / lambda_w) - nu_w) + (1 - xi_w) / ➕₁₁ ^ ((lambda_w + 1) / lambda_w)),
    :(((nu_p * xi_p) / ➕₁₀ ^ ((lambda_p + 1) / lambda_p) - nu_p) + (1 - xi_p) / pi_star ^ ((lambda_p + 1) / lambda_p)),
    :((I * (-0.5 * varphi * (epsilon_I - 1) ^ 2 + 1) + K * (1 - tau)) - K),
    :((I_f * (-0.5 * varphi * (epsilon_I - 1) ^ 2 + 1) + K_f * (1 - tau)) - K_f),
    :((-U * beta + U) - epsilon_b * ((-(L_s ^ (sigma_l + 1)) * epsilon_L * omega) / (sigma_l + 1) + ➕₈ ^ (1 - sigma_c) / (1 - sigma_c))),
    :((-U_f * beta + U_f) - epsilon_b * ((-(L_s_f ^ (sigma_l + 1)) * epsilon_L * omega) / (sigma_l + 1) + ➕₉ ^ (1 - sigma_c) / (1 - sigma_c))),
    :((beta * epsilon_I * q * varphi * (epsilon_I - 1) - epsilon_b / ➕₈ ^ sigma_c) + q * ((-epsilon_I * varphi * (epsilon_I - 1) - 0.5 * varphi * (epsilon_I - 1) ^ 2) + 1)),
    :((beta * epsilon_I * q_f * varphi * (epsilon_I - 1) - epsilon_b / ➕₉ ^ sigma_c) + q_f * ((-epsilon_I * varphi * (epsilon_I - 1) - 0.5 * varphi * (epsilon_I - 1) ^ 2) + 1)),
    :(➕₁₇ - psi * (z - 1)),
    :((((-C - I) - (K * r_k * (exp(➕₁₇) - 1)) / psi) - T) + Y),
    :(➕₁₈ - psi * (z_f - 1)),
    :((((((-C_f - I_f) - (K_f * r_k_f * (exp(➕₁₈) - 1)) / psi) - L_f * W_f) + L_s_f * W_disutil_f + Pi_ws_f) - T_f) + Y_f),
    :((epsilon_b * (-K * r_k * exp(➕₁₇) + K * r_k)) / ➕₈ ^ sigma_c),
    :((epsilon_b * (-K_f * r_k_f * exp(➕₁₈) + K_f * r_k_f)) / ➕₉ ^ sigma_c),
    :((rho_pi_bar * log(pi_obj) + (1 - rho_pi_bar) * log(calibr_pi_obj)) - log(pi_obj)),
    :(-R̄ + χᵒᵇᶜ⁺ꜝ¹ꜝˡ + log(R)),
    :(➕₁₉ - 1 / R),
    :(((calibr_pi + χᵒᵇᶜ⁺ꜝ¹ꜝʳ) - (1 - rho) * (-r_pi * log(pi_obj) + log(pi_obj))) + log(R) + log(➕₁₉)),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - Max(χᵒᵇᶜ⁺ꜝ¹ꜝʳ, χᵒᵇᶜ⁺ꜝ¹ꜝˡ)),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝ),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾ - 0),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(-q + beta * ((1 - tau) * q + epsilon_b * (r_k * z - psi ^ -1 * r_k * (-1 + exp(psi * (-1 + z)))) * (C - h * C) ^ -sigma_c)),
    :(-q_f + beta * ((1 - tau) * q_f + epsilon_b * (r_k_f * z_f - psi ^ -1 * r_k_f * (-1 + exp(psi * (-1 + z_f)))) * (C_f - h * C_f) ^ -sigma_c)),
    :(-r_k + alpha * epsilon_a * mc * L ^ (1 - alpha) * (K * z) ^ (-1 + alpha)),
    :(-r_k_f + alpha * epsilon_a * mc_f * L_f ^ (1 - alpha) * (K_f * z_f) ^ (-1 + alpha)),
    :(-G + T),
    :(-G + G_bar * epsilon_G),
    :(-G_f + T_f),
    :(-G_f + G_bar * epsilon_G),
    :(-L + nu_w ^ -1 * L_s),
    :(-L_s_f + L_f * (W_i_f * W_f ^ -1) ^ (lambda_w ^ -1 * (-1 - lambda_w))),
    :(L_s_f - L_f),
    :(L_s_f + lambda_w ^ -1 * L_f * W_f ^ -1 * (-1 - lambda_w) * (-W_disutil_f + W_i_f) * (W_i_f * W_f ^ -1) ^ (-1 + lambda_w ^ -1 * (-1 - lambda_w))),
    :(Pi_ws_f - L_s_f * (-W_disutil_f + W_i_f)),
    :(Pi_ps_f - Y_f * (-mc_f + P_j_f) * P_j_f ^ (-(lambda_p ^ -1) * (1 + lambda_p))),
    :(-Q + epsilon_b ^ -1 * q * (C - h * C) ^ sigma_c),
    :(-Q_f + epsilon_b ^ -1 * q_f * (C_f - h * C_f) ^ sigma_c),
    :(-W + epsilon_a * mc * (1 - alpha) * L ^ -alpha * (K * z) ^ alpha),
    :(-W_f + epsilon_a * mc_f * (1 - alpha) * L_f ^ -alpha * (K_f * z_f) ^ alpha),
    :(-Y_f + Y_s_f),
    :(Y_s - nu_p * Y),
    :(-Y_s_f + Y_f * P_j_f ^ (-(lambda_p ^ -1) * (1 + lambda_p))),
    :(beta * epsilon_b * (C_f - h * C_f) ^ -sigma_c - epsilon_b * R_f ^ -1 * (C_f - h * C_f) ^ -sigma_c),
    :(beta * epsilon_b * pi ^ -1 * (C - h * C) ^ -sigma_c - epsilon_b * R ^ -1 * (C - h * C) ^ -sigma_c),
    :(Y_f * P_j_f ^ (-(lambda_p ^ -1) * (1 + lambda_p)) - lambda_p ^ -1 * Y_f * (1 + lambda_p) * (-mc_f + P_j_f) * P_j_f ^ (-1 - lambda_p ^ -1 * (1 + lambda_p))),
    :(epsilon_b * W_disutil_f * (C_f - h * C_f) ^ -sigma_c - omega * epsilon_b * epsilon_L * L_s_f ^ sigma_l),
    :(-1 + xi_p * (pi ^ -1 * pi ^ gamma_p) ^ -(lambda_p ^ -1) + (1 - xi_p) * pi_star ^ -(lambda_p ^ -1)),
    :(-1 + (1 - xi_w) * (w_star * W ^ -1) ^ -(lambda_w ^ -1) + xi_w * (W * W ^ -1) ^ -(lambda_w ^ -1) * (pi ^ -1 * pi ^ gamma_w) ^ -(lambda_w ^ -1)),
    :((-Phi - Y_s) + epsilon_a * L ^ (1 - alpha) * (K * z) ^ alpha),
    :((-Phi - Y_f * P_j_f ^ (-(lambda_p ^ -1) * (1 + lambda_p))) + epsilon_a * L_f ^ (1 - alpha) * (K_f * z_f) ^ alpha),
    :((std_eta_b * 0 - log(epsilon_b)) + rho_b * log(epsilon_b)),
    :((-std_eta_L * 0 - log(epsilon_L)) + rho_L * log(epsilon_L)),
    :((std_eta_I * 0 - log(epsilon_I)) + rho_I * log(epsilon_I)),
    :((std_eta_w * 0 - f_1) + f_2),
    :((std_eta_a * 0 - log(epsilon_a)) + rho_a * log(epsilon_a)),
    :((std_eta_p * 0 - g_1) + g_2 * (1 + lambda_p)),
    :((std_eta_G * 0 - log(epsilon_G)) + rho_G * log(epsilon_G)),
    :(-f_1 + beta * xi_w * f_1 * (w_star ^ -1 * w_star) ^ (lambda_w ^ -1) * (pi ^ -1 * pi ^ gamma_w) ^ -(lambda_w ^ -1) + epsilon_b * w_star * L * (1 + lambda_w) ^ -1 * (C - h * C) ^ -sigma_c * (w_star * W ^ -1) ^ (-(lambda_w ^ -1) * (1 + lambda_w))),
    :(-f_2 + beta * xi_w * f_2 * (w_star ^ -1 * w_star) ^ (lambda_w ^ -1 * (1 + lambda_w) * (1 + sigma_l)) * (pi ^ -1 * pi ^ gamma_w) ^ (-(lambda_w ^ -1) * (1 + lambda_w) * (1 + sigma_l)) + omega * epsilon_b * epsilon_L * (L * (w_star * W ^ -1) ^ (-(lambda_w ^ -1) * (1 + lambda_w))) ^ (1 + sigma_l)),
    :(-g_1 + beta * xi_p * pi_star * g_1 * pi_star ^ -1 * (pi ^ -1 * pi ^ gamma_p) ^ -(lambda_p ^ -1) + epsilon_b * pi_star * Y * (C - h * C) ^ -sigma_c),
    :(-g_2 + beta * xi_p * g_2 * (pi ^ -1 * pi ^ gamma_p) ^ (-(lambda_p ^ -1) * (1 + lambda_p)) + epsilon_b * mc * Y * (C - h * C) ^ -sigma_c),
    :(-nu_w + (1 - xi_w) * (w_star * W ^ -1) ^ (-(lambda_w ^ -1) * (1 + lambda_w)) + xi_w * nu_w * (W * pi ^ -1 * W ^ -1 * pi ^ gamma_w) ^ (-(lambda_w ^ -1) * (1 + lambda_w))),
    :(-nu_p + (1 - xi_p) * pi_star ^ (-(lambda_p ^ -1) * (1 + lambda_p)) + xi_p * nu_p * (pi ^ -1 * pi ^ gamma_p) ^ (-(lambda_p ^ -1) * (1 + lambda_p))),
    :(-K + K * (1 - tau) + I * (1 - 0.5 * varphi * (-1 + I ^ -1 * epsilon_I * I) ^ 2)),
    :(-K_f + K_f * (1 - tau) + I_f * (1 - 0.5 * varphi * (-1 + I_f ^ -1 * epsilon_I * I_f) ^ 2)),
    :((U - beta * U) - epsilon_b * ((1 - sigma_c) ^ -1 * (C - h * C) ^ (1 - sigma_c) - omega * epsilon_L * (1 + sigma_l) ^ -1 * L_s ^ (1 + sigma_l))),
    :((U_f - beta * U_f) - epsilon_b * ((1 - sigma_c) ^ -1 * (C_f - h * C_f) ^ (1 - sigma_c) - omega * epsilon_L * (1 + sigma_l) ^ -1 * L_s_f ^ (1 + sigma_l))),
    :(-epsilon_b * (C - h * C) ^ -sigma_c + q * ((1 - 0.5 * varphi * (-1 + I ^ -1 * epsilon_I * I) ^ 2) - varphi * I ^ -1 * epsilon_I * I * (-1 + I ^ -1 * epsilon_I * I)) + beta * varphi * I ^ -2 * epsilon_I * q * I ^ 2 * (-1 + I ^ -1 * epsilon_I * I)),
    :(-epsilon_b * (C_f - h * C_f) ^ -sigma_c + q_f * ((1 - 0.5 * varphi * (-1 + I_f ^ -1 * epsilon_I * I_f) ^ 2) - varphi * I_f ^ -1 * epsilon_I * I_f * (-1 + I_f ^ -1 * epsilon_I * I_f)) + beta * varphi * I_f ^ -2 * epsilon_I * q_f * I_f ^ 2 * (-1 + I_f ^ -1 * epsilon_I * I_f)),
    :((((-C - I) - T) + Y) - psi ^ -1 * r_k * K * (-1 + exp(psi * (-1 + z)))),
    :((((((-C_f - I_f) + Pi_ws_f) - T_f) + Y_f + L_s_f * W_disutil_f) - L_f * W_f) - psi ^ -1 * r_k_f * K_f * (-1 + exp(psi * (-1 + z_f)))),
    :(epsilon_b * (K * r_k - r_k * K * exp(psi * (-1 + z))) * (C - h * C) ^ -sigma_c),
    :(epsilon_b * (K_f * r_k_f - r_k_f * K_f * exp(psi * (-1 + z_f))) * (C_f - h * C_f) ^ -sigma_c),
    :((std_eta_pi * 0 - log(pi_obj)) + rho_pi_bar * log(pi_obj) + log(calibr_pi_obj) * (1 - rho_pi_bar)),
    :(χᵒᵇᶜ⁺ꜝ¹ꜝˡ - (R̄ - log(R))),
    :(χᵒᵇᶜ⁺ꜝ¹ꜝʳ - (((((r_Delta_pi * (-(log(pi ^ -1 * pi)) + log(pi ^ -1 * pi)) + r_Delta_y * ((-(log(Y ^ -1 * Y)) + log(Y ^ -1 * Y) + log(Y_f ^ -1 * Y_f)) - log(Y_f ^ -1 * Y_f)) + rho * log(R ^ -1 * R) + (1 - rho) * (log(pi_obj) + r_pi * (-(log(pi_obj)) + log(pi ^ -1 * pi)) + r_Y * (log(Y ^ -1 * Y) - log(Y_f ^ -1 * Y_f)))) - calibr_pi) + std_eta_R * 0) - log(R ^ -1)) - log(R))),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - max(χᵒᵇᶜ⁺ꜝ¹ꜝˡ, χᵒᵇᶜ⁺ꜝ¹ꜝʳ)),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝ),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾ - activeᵒᵇᶜshocks * 0),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾ + activeᵒᵇᶜshocks * 0)),
    :(1 - pi_obj),
    :(pi - pi_obj),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :(➕₁ - psi * (z - 1)),
    :(➕₂ - (-C * h + C)),
    :(beta * ((epsilon_b * (r_k * z - (r_k * (exp(➕₁) - 1)) / psi)) / ➕₂ ^ sigma_c + q * (1 - tau)) - q),
    :(➕₃ - psi * (z_f - 1)),
    :(➕₄ - (-C_f * h + C_f)),
    :(beta * ((epsilon_b * (r_k_f * z_f - (r_k_f * (exp(➕₃) - 1)) / psi)) / ➕₄ ^ sigma_c + q_f * (1 - tau)) - q_f),
    :(➕₅ - K * z),
    :(L ^ (1 - alpha) * alpha * epsilon_a * mc * ➕₅ ^ (alpha - 1) - r_k),
    :(➕₆ - K_f * z_f),
    :(L_f ^ (1 - alpha) * alpha * epsilon_a * mc_f * ➕₆ ^ (alpha - 1) - r_k_f),
    :(-G + T),
    :(-G + G_bar * epsilon_G),
    :(-G_f + T_f),
    :(G_bar * epsilon_G - G_f),
    :(-L + L_s / nu_w),
    :(➕₇ - W_i_f / W_f),
    :(L_f * ➕₇ ^ ((-lambda_w - 1) / lambda_w) - L_s_f),
    :(-L_f + L_s_f),
    :((L_f * ➕₇ ^ (-1 + (-lambda_w - 1) / lambda_w) * (-W_disutil_f + W_i_f) * (-lambda_w - 1)) / (W_f * lambda_w) + L_s_f),
    :(-L_s_f * (-W_disutil_f + W_i_f) + Pi_ws_f),
    :(Pi_ps_f - (Y_f * (P_j_f - mc_f)) / P_j_f ^ ((lambda_p + 1) / lambda_p)),
    :(➕₈ - (-C * h + C)),
    :(-Q + (q * ➕₈ ^ sigma_c) / epsilon_b),
    :(➕₉ - (-C_f * h + C_f)),
    :(-Q_f + (q_f * ➕₉ ^ sigma_c) / epsilon_b),
    :(-W + (epsilon_a * mc * ➕₅ ^ alpha * (1 - alpha)) / L ^ alpha),
    :(-W_f + (epsilon_a * mc_f * ➕₆ ^ alpha * (1 - alpha)) / L_f ^ alpha),
    :(-Y_f + Y_s_f),
    :(-Y * nu_p + Y_s),
    :(-Y_s_f + Y_f / P_j_f ^ ((lambda_p + 1) / lambda_p)),
    :((beta * epsilon_b) / ➕₄ ^ sigma_c - epsilon_b / (R_f * ➕₉ ^ sigma_c)),
    :((beta * epsilon_b) / (pi * ➕₂ ^ sigma_c) - epsilon_b / (R * ➕₈ ^ sigma_c)),
    :((-(P_j_f ^ (-1 - (lambda_p + 1) / lambda_p)) * Y_f * (P_j_f - mc_f) * (lambda_p + 1)) / lambda_p + Y_f / P_j_f ^ ((lambda_p + 1) / lambda_p)),
    :(-(L_s_f ^ sigma_l) * epsilon_L * epsilon_b * omega + (W_disutil_f * epsilon_b) / ➕₉ ^ sigma_c),
    :(➕₁₀ - pi ^ gamma_p / pi),
    :((xi_p / ➕₁₀ ^ (1 / lambda_p) - 1) + (1 - xi_p) / pi_star ^ (1 / lambda_p)),
    :(➕₁₁ - w_star / W),
    :(➕₁₂ - pi ^ gamma_w / pi),
    :((xi_w / ➕₁₂ ^ (1 / lambda_w) - 1) + (1 - xi_w) / ➕₁₁ ^ (1 / lambda_w)),
    :((L ^ (1 - alpha) * epsilon_a * ➕₅ ^ alpha - Phi) - Y_s),
    :((L_f ^ (1 - alpha) * epsilon_a * ➕₆ ^ alpha - Phi) - Y_f / P_j_f ^ ((lambda_p + 1) / lambda_p)),
    :(rho_b * log(epsilon_b) - log(epsilon_b)),
    :(rho_L * log(epsilon_L) - log(epsilon_L)),
    :(rho_I * log(epsilon_I) - log(epsilon_I)),
    :(-f_1 + f_2),
    :(rho_a * log(epsilon_a) - log(epsilon_a)),
    :(-g_1 + g_2 * (lambda_p + 1)),
    :(rho_G * log(epsilon_G) - log(epsilon_G)),
    :(➕₁₃ - pi ^ gamma_w / pi),
    :(((L * epsilon_b * w_star) / (➕₁₁ ^ ((lambda_w + 1) / lambda_w) * ➕₈ ^ sigma_c * (lambda_w + 1)) + (beta * f_1 * xi_w) / ➕₁₃ ^ (1 / lambda_w)) - f_1),
    :(➕₁₄ - L / ➕₁₁ ^ ((lambda_w + 1) / lambda_w)),
    :(((beta * f_2 * xi_w) / ➕₁₃ ^ (((lambda_w + 1) * (sigma_l + 1)) / lambda_w) + epsilon_L * epsilon_b * omega * ➕₁₄ ^ (sigma_l + 1)) - f_2),
    :(➕₁₅ - pi ^ gamma_p / pi),
    :(((Y * epsilon_b * pi_star) / ➕₈ ^ sigma_c + (beta * g_1 * xi_p) / ➕₁₅ ^ (1 / lambda_p)) - g_1),
    :(((Y * epsilon_b * mc) / ➕₈ ^ sigma_c + (beta * g_2 * xi_p) / ➕₁₅ ^ ((lambda_p + 1) / lambda_p)) - g_2),
    :(➕₁₆ - pi ^ gamma_w / pi),
    :(((nu_w * xi_w) / ➕₁₆ ^ ((lambda_w + 1) / lambda_w) - nu_w) + (1 - xi_w) / ➕₁₁ ^ ((lambda_w + 1) / lambda_w)),
    :(((nu_p * xi_p) / ➕₁₀ ^ ((lambda_p + 1) / lambda_p) - nu_p) + (1 - xi_p) / pi_star ^ ((lambda_p + 1) / lambda_p)),
    :((I * (-0.5 * varphi * (epsilon_I - 1) ^ 2 + 1) + K * (1 - tau)) - K),
    :((I_f * (-0.5 * varphi * (epsilon_I - 1) ^ 2 + 1) + K_f * (1 - tau)) - K_f),
    :((-U * beta + U) - epsilon_b * ((-(L_s ^ (sigma_l + 1)) * epsilon_L * omega) / (sigma_l + 1) + ➕₈ ^ (1 - sigma_c) / (1 - sigma_c))),
    :((-U_f * beta + U_f) - epsilon_b * ((-(L_s_f ^ (sigma_l + 1)) * epsilon_L * omega) / (sigma_l + 1) + ➕₉ ^ (1 - sigma_c) / (1 - sigma_c))),
    :((beta * epsilon_I * q * varphi * (epsilon_I - 1) - epsilon_b / ➕₈ ^ sigma_c) + q * ((-epsilon_I * varphi * (epsilon_I - 1) - 0.5 * varphi * (epsilon_I - 1) ^ 2) + 1)),
    :((beta * epsilon_I * q_f * varphi * (epsilon_I - 1) - epsilon_b / ➕₉ ^ sigma_c) + q_f * ((-epsilon_I * varphi * (epsilon_I - 1) - 0.5 * varphi * (epsilon_I - 1) ^ 2) + 1)),
    :(➕₁₇ - psi * (z - 1)),
    :((((-C - I) - (K * r_k * (exp(➕₁₇) - 1)) / psi) - T) + Y),
    :(➕₁₈ - psi * (z_f - 1)),
    :((((((-C_f - I_f) - (K_f * r_k_f * (exp(➕₁₈) - 1)) / psi) - L_f * W_f) + L_s_f * W_disutil_f + Pi_ws_f) - T_f) + Y_f),
    :((epsilon_b * (-K * r_k * exp(➕₁₇) + K * r_k)) / ➕₈ ^ sigma_c),
    :((epsilon_b * (-K_f * r_k_f * exp(➕₁₈) + K_f * r_k_f)) / ➕₉ ^ sigma_c),
    :((rho_pi_bar * log(pi_obj) + (1 - rho_pi_bar) * log(calibr_pi_obj)) - log(pi_obj)),
    :(-R̄ + χᵒᵇᶜ⁺ꜝ¹ꜝˡ + log(R)),
    :(➕₁₉ - 1 / R),
    :(((calibr_pi + χᵒᵇᶜ⁺ꜝ¹ꜝʳ) - (1 - rho) * (-r_pi * log(pi_obj) + log(pi_obj))) + log(R) + log(➕₁₉)),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - Max(χᵒᵇᶜ⁺ꜝ¹ꜝʳ, χᵒᵇᶜ⁺ꜝ¹ꜝˡ)),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝ),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾ - 0),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾),
    :(1 - pi_obj),
    :(pi - pi_obj),
]

const PARAMETER_DEFINITION_NAMES = [
    "std_eta_G",
    "std_eta_I",
    "std_eta_L",
    "std_eta_R",
    "std_eta_a",
    "std_eta_b",
    "std_eta_p",
    "std_eta_pi",
    "std_eta_w",
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
    "σ_eta_G / std_scaling_factor",
    "σ_eta_I / std_scaling_factor",
    "σ_eta_L / std_scaling_factor",
    "σ_eta_R / std_scaling_factor",
    "σ_eta_a / std_scaling_factor",
    "σ_eta_b / std_scaling_factor",
    "σ_eta_p / std_scaling_factor",
    "σ_eta_pi / std_scaling_factor",
    "σ_eta_w / std_scaling_factor",
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "R̄",
    "lambda_p",
    "G_bar",
    "lambda_w",
    "Phi",
    "alpha",
    "beta",
    "gamma_w",
    "gamma_p",
    "h",
    "omega",
    "psi",
    "r_pi",
    "r_Y",
    "r_Delta_pi",
    "r_Delta_y",
    "sigma_c",
    "sigma_l",
    "tau",
    "varphi",
    "xi_w",
    "xi_p",
    "rho",
    "rho_b",
    "rho_L",
    "rho_I",
    "rho_a",
    "rho_G",
    "rho_pi_bar",
    "std_scaling_factor",
    "σ_eta_b",
    "σ_eta_L",
    "σ_eta_I",
    "σ_eta_a",
    "σ_eta_w",
    "σ_eta_p",
    "σ_eta_G",
    "σ_eta_R",
    "σ_eta_pi",
    "activeᵒᵇᶜshocks",
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
    "C",
    "C_f",
    "G",
    "G_f",
    "I",
    "I_f",
    "K",
    "K_f",
    "L",
    "L_f",
    "L_s",
    "L_s_f",
    "P_j_f",
    "Pi_ps_f",
    "Pi_ws_f",
    "Q",
    "Q_f",
    "R",
    "R_f",
    "T",
    "T_f",
    "U",
    "U_f",
    "W",
    "W_disutil_f",
    "W_f",
    "W_i_f",
    "Y",
    "Y_f",
    "Y_s",
    "Y_s_f",
    "epsilon_G",
    "epsilon_I",
    "epsilon_L",
    "epsilon_a",
    "epsilon_b",
    "f_1",
    "f_2",
    "g_1",
    "g_2",
    "mc",
    "mc_f",
    "nu_p",
    "nu_w",
    "pi",
    "pi_obj",
    "pi_star",
    "q",
    "q_f",
    "r_k",
    "r_k_f",
    "w_star",
    "z",
    "z_f",
    "Χᵒᵇᶜ⁺ꜝ¹ꜝ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝʳ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝˡ",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝ",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾",
    "calibr_pi_obj",
    "calibr_pi",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -Inf,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
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
    -Inf,
    -Inf,
]
const ORIGINAL_BOX_UPPER_BOUNDS = Float64[
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
    Inf,
    1.0e12,
    1.0e12,
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
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
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
    Inf,
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
    Inf,
    1.0e12,
    Inf,
    Inf,
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
    Inf,
    Inf,
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "C",
    "C_f",
    "G",
    "G_f",
    "I",
    "I_f",
    "K",
    "K_f",
    "L",
    "L_f",
    "L_s",
    "L_s_f",
    "P_j_f",
    "Pi_ps_f",
    "Pi_ws_f",
    "Q",
    "Q_f",
    "R",
    "R_f",
    "T",
    "T_f",
    "U",
    "U_f",
    "W",
    "W_disutil_f",
    "W_f",
    "W_i_f",
    "Y",
    "Y_f",
    "Y_s",
    "Y_s_f",
    "epsilon_G",
    "epsilon_I",
    "epsilon_L",
    "epsilon_a",
    "epsilon_b",
    "f_1",
    "f_2",
    "g_1",
    "g_2",
    "mc",
    "mc_f",
    "nu_p",
    "nu_w",
    "pi",
    "pi_obj",
    "pi_star",
    "q",
    "q_f",
    "r_k",
    "r_k_f",
    "w_star",
    "z",
    "z_f",
    "Χᵒᵇᶜ⁺ꜝ¹ꜝ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝʳ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝˡ",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝ",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾",
    "➕₁",
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
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
    "➕₆",
    "➕₇",
    "➕₈",
    "➕₉",
    "calibr_pi_obj",
    "calibr_pi",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -Inf,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
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
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    2.220446049250313e-16,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -Inf,
    -Inf,
]
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
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
    Inf,
    1.0e12,
    1.0e12,
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
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
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
    Inf,
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
    Inf,
    1.0e12,
    Inf,
    Inf,
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
    600.0,
    Inf,
    Inf,
    Inf,
    Inf,
    1.0e12,
    Inf,
    Inf,
    600.0,
    600.0,
    1.0e12,
    1.0e12,
    600.0,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
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
    "➕₂₆",
    "➕₂₇",
    "➕₂₈",
    "➕₂₉",
    "➕₃₀",
    "➕₃₁",
    "➕₃₂",
    "➕₃₃",
    "➕₃₄",
    "➕₃₅",
    "➕₃₆",
]
const ALL_AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    2.220446049250313e-16,
    -1.0e12,
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
    2.220446049250313e-16,
    -1.0e12,
    -1.0e12,
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
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
]
const ALL_AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    600.0,
    1.0e12,
    600.0,
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
    600.0,
    600.0,
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
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
]

const BLOCKS = [
    (
        index = 1,
        solve_order = 78,
        variables = ["calibr_pi"],
        previous_solution_names = ["R", "pi_obj", "χᵒᵇᶜ⁺ꜝ¹ꜝʳ", "➕₁₉"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₃₅", "➕₃₆"],
        equation_indices = [74],
        equations = Expr[
            :(((calibr_pi + χᵒᵇᶜ⁺ꜝ¹ꜝʳ) - (1 - rho) * (-r_pi * log(➕₃₅) + log(➕₃₅))) + log(➕₃₆) + log(➕₁₉)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₃₅ = min(1.0e12, max(eps(), pi_obj))),
            :(➕₃₆ = min(1.0e12, max(eps(), R))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₃₅ - pi_obj)),
            :(abs(➕₃₆ - R)),
        ],
        solution_names = ["calibr_pi", "➕₃₅", "➕₃₆"],
        previous_solution_values = [1.0101010101010097, 1.0, 0.0, 0.9900000000000003],
        external_solution_values = Float64[],
        solution_values = [4.85722573273506e-17, 1.0, 1.0101010101010097],
        previous_solution_initial_values = [1.0101010101010102, 1.0, 0.0, 0.99],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-5.551115123125783e-17, 1.0, 1.0101010101010102],
        box_lower_bounds = [-Inf, 2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12, 1.0e12],
    ),
    (
        index = 2,
        solve_order = 77,
        variables = ["calibr_pi_obj"],
        previous_solution_names = ["pi_obj"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₃₄"],
        equation_indices = [71],
        equations = Expr[
            :((rho_pi_bar * log(➕₃₄) + (1 - rho_pi_bar) * log(calibr_pi_obj)) - log(➕₃₄)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₃₄ = min(1.0e12, max(eps(), pi_obj))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₃₄ - pi_obj)),
        ],
        solution_names = ["calibr_pi_obj", "➕₃₄"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [1.0, 1.0],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0, 1.0],
        box_lower_bounds = [2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12],
    ),
    (
        index = 3,
        solve_order = 76,
        variables = ["➕₁₉"],
        previous_solution_names = ["R"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [73],
        equations = Expr[
            :(➕₁₉ - 1 / R),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["➕₁₉"],
        previous_solution_values = [1.0101010101010097],
        external_solution_values = Float64[],
        solution_values = [0.9900000000000003],
        previous_solution_initial_values = [1.0101010101010102],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.99],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 4,
        solve_order = 75,
        variables = ["χᵒᵇᶜ⁺ꜝ¹ꜝʳ"],
        previous_solution_names = ["Χᵒᵇᶜ⁺ꜝ¹ꜝ", "χᵒᵇᶜ⁺ꜝ¹ꜝˡ"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [75],
        equations = Expr[
            :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - Max(χᵒᵇᶜ⁺ꜝ¹ꜝʳ, χᵒᵇᶜ⁺ꜝ¹ꜝˡ)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["χᵒᵇᶜ⁺ꜝ¹ꜝʳ"],
        previous_solution_values = [0.0, -0.010050335853501065],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0, -0.010050335853501506],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-1.0e12],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 5,
        solve_order = 74,
        variables = ["χᵒᵇᶜ⁺ꜝ¹ꜝˡ"],
        previous_solution_names = ["R"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₃₃"],
        equation_indices = [72],
        equations = Expr[
            :(-R̄ + χᵒᵇᶜ⁺ꜝ¹ꜝˡ + log(➕₃₃)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₃₃ = min(1.0e12, max(eps(), R))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₃₃ - R)),
        ],
        solution_names = ["χᵒᵇᶜ⁺ꜝ¹ꜝˡ", "➕₃₃"],
        previous_solution_values = [1.0101010101010097],
        external_solution_values = Float64[],
        solution_values = [-0.010050335853501065, 1.0101010101010097],
        previous_solution_initial_values = [1.0101010101010102],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-0.010050335853501506, 1.0101010101010102],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 6,
        solve_order = 73,
        variables = ["Χᵒᵇᶜ⁺ꜝ¹ꜝ"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝ"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [76],
        equations = Expr[
            :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝ),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["Χᵒᵇᶜ⁺ꜝ¹ꜝ"],
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
        index = 7,
        solve_order = 72,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝ"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [77],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝ"],
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
        index = 8,
        solve_order = 71,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [118],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 9,
        solve_order = 70,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [117],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 10,
        solve_order = 69,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [116],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 11,
        solve_order = 68,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [115],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 12,
        solve_order = 67,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [114],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 13,
        solve_order = 66,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [113],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 14,
        solve_order = 65,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [112],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 15,
        solve_order = 64,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [111],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 16,
        solve_order = 63,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [110],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 17,
        solve_order = 62,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [109],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 18,
        solve_order = 61,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [108],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 19,
        solve_order = 60,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [107],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 20,
        solve_order = 59,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [106],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 21,
        solve_order = 58,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [105],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 22,
        solve_order = 57,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [104],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 23,
        solve_order = 56,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [103],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 24,
        solve_order = 55,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [102],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 25,
        solve_order = 54,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [101],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 26,
        solve_order = 53,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [100],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 27,
        solve_order = 52,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [99],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 28,
        solve_order = 51,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [98],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 29,
        solve_order = 50,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [97],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 30,
        solve_order = 49,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [96],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 31,
        solve_order = 48,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [95],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 32,
        solve_order = 47,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [94],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 33,
        solve_order = 46,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [93],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 34,
        solve_order = 45,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [92],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 35,
        solve_order = 44,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [91],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 36,
        solve_order = 43,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [90],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 37,
        solve_order = 42,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [89],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 38,
        solve_order = 41,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [88],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 39,
        solve_order = 40,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [87],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 40,
        solve_order = 39,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [86],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 41,
        solve_order = 38,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [85],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 42,
        solve_order = 37,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [84],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 43,
        solve_order = 36,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [83],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 44,
        solve_order = 35,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [82],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 45,
        solve_order = 34,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [81],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 46,
        solve_order = 33,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [80],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 47,
        solve_order = 32,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [79],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 48,
        solve_order = 31,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [78],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾ - 0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [0.0],
        box_upper_bounds = [0.0],
    ),
    (
        index = 49,
        solve_order = 30,
        variables = ["U_f"],
        previous_solution_names = ["L_s_f", "epsilon_L", "epsilon_b", "➕₉"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₃₂"],
        equation_indices = [62],
        equations = Expr[
            :((-U_f * beta + U_f) - epsilon_b * ((-(➕₃₂ ^ (sigma_l + 1)) * epsilon_L * omega) / (sigma_l + 1) + ➕₉ ^ (1 - sigma_c) / (1 - sigma_c))),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₃₂ = min(1.0e12, max(eps(), L_s_f))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₃₂ - L_s_f)),
        ],
        solution_names = ["U_f", "➕₃₂"],
        previous_solution_values = [1.289115943211282, 1.0, 1.0, 0.5142692988548276],
        external_solution_values = Float64[],
        solution_values = [-427.98589108169494, 1.289115943211282],
        previous_solution_initial_values = [5.0e11, 1.0, 1.0, 5.0e11],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-1.7580009187316713e41, 5.0e11],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 50,
        solve_order = 29,
        variables = ["U"],
        previous_solution_names = ["L_s", "epsilon_L", "epsilon_b", "➕₈"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₃₁"],
        equation_indices = [61],
        equations = Expr[
            :((-U * beta + U) - epsilon_b * ((-(➕₃₁ ^ (sigma_l + 1)) * epsilon_L * omega) / (sigma_l + 1) + ➕₈ ^ (1 - sigma_c) / (1 - sigma_c))),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₃₁ = min(1.0e12, max(eps(), L_s))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₃₁ - L_s)),
        ],
        solution_names = ["U", "➕₃₁"],
        previous_solution_values = [1.2891159432112824, 1.0, 1.0, 0.5142692988548276],
        external_solution_values = Float64[],
        solution_values = [-427.98589108169506, 1.2891159432112824],
        previous_solution_initial_values = [5.0e11, 1.0, 1.0, 5.0e11],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-1.7580009187316713e41, 5.0e11],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 51,
        solve_order = 28,
        variables = ["R_f"],
        previous_solution_names = ["epsilon_b", "➕₄", "➕₉"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [31],
        equations = Expr[
            :((beta * epsilon_b) / ➕₄ ^ sigma_c - epsilon_b / (R_f * ➕₉ ^ sigma_c)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["R_f"],
        previous_solution_values = [1.0, 0.5142692988548276, 0.5142692988548276],
        external_solution_values = Float64[],
        solution_values = [1.0101010101010102],
        previous_solution_initial_values = [1.0, 5.0e11, 5.0e11],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0101010101010102],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 52,
        solve_order = 27,
        variables = ["R"],
        previous_solution_names = ["epsilon_b", "pi", "➕₂", "➕₈"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [32],
        equations = Expr[
            :((beta * epsilon_b) / (pi * ➕₂ ^ sigma_c) - epsilon_b / (R * ➕₈ ^ sigma_c)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["R"],
        previous_solution_values = [1.0, 1.0, 0.5142692988548276, 0.5142692988548276],
        external_solution_values = Float64[],
        solution_values = [1.0101010101010097],
        previous_solution_initial_values = [1.0, 1.0, 5.0e11, 5.0e11],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0101010101010102],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 53,
        solve_order = 26,
        variables = ["Q_f"],
        previous_solution_names = ["epsilon_b", "q_f", "➕₉"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [25],
        equations = Expr[
            :(-Q_f + (q_f * ➕₉ ^ sigma_c) / epsilon_b),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["Q_f"],
        previous_solution_values = [1.0, 2.459003350339646, 0.5142692988548276],
        external_solution_values = Float64[],
        solution_values = [0.9999999999999998],
        previous_solution_initial_values = [1.0, 0.0, 5.0e11],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 54,
        solve_order = 25,
        variables = ["Q"],
        previous_solution_names = ["epsilon_b", "q", "➕₈"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [23],
        equations = Expr[
            :(-Q + (q * ➕₈ ^ sigma_c) / epsilon_b),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["Q"],
        previous_solution_values = [1.0, 2.4590033503396467, 0.5142692988548276],
        external_solution_values = Float64[],
        solution_values = [1.0000000000000002],
        previous_solution_initial_values = [1.0, 0.0, 5.0e11],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 55,
        solve_order = 24,
        variables = ["Pi_ps_f"],
        previous_solution_names = ["P_j_f", "Y_f", "mc_f"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₃₀"],
        equation_indices = [21],
        equations = Expr[
            :(Pi_ps_f - (Y_f * (P_j_f - mc_f)) / ➕₃₀ ^ ((lambda_p + 1) / lambda_p)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₃₀ = min(1.0e12, max(eps(), P_j_f))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₃₀ - P_j_f)),
        ],
        solution_names = ["Pi_ps_f", "➕₃₀"],
        previous_solution_values = [1.0, 2.007916147770184, 0.7309941520467835],
        external_solution_values = Float64[],
        solution_values = [0.5401411859498743, 1.0],
        previous_solution_initial_values = [5.0e11, 0.0, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 5.0e11],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 56,
        solve_order = 23,
        variables = ["L_s"],
        previous_solution_names = ["L", "nu_w"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [15],
        equations = Expr[
            :(-L + L_s / nu_w),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["L_s"],
        previous_solution_values = [1.2891159432112824, 1.0],
        external_solution_values = Float64[],
        solution_values = [1.2891159432112824],
        previous_solution_initial_values = [5.0e11, 1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [5.0e11],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 57,
        solve_order = 22,
        variables = ["nu_w"],
        previous_solution_names = ["➕₁₁", "➕₁₆"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [57],
        equations = Expr[
            :(((nu_w * xi_w) / ➕₁₆ ^ ((lambda_w + 1) / lambda_w) - nu_w) + (1 - xi_w) / ➕₁₁ ^ ((lambda_w + 1) / lambda_w)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["nu_w"],
        previous_solution_values = [1.0, 1.0],
        external_solution_values = Float64[],
        solution_values = [1.0],
        previous_solution_initial_values = [1.0, 1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 58,
        solve_order = 21,
        variables = ["➕₁₆"],
        previous_solution_names = ["pi"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₉"],
        equation_indices = [56],
        equations = Expr[
            :(➕₁₆ - ➕₂₉ ^ gamma_w / pi),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₉ = min(1.0e12, max(eps(), pi))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₉ - pi)),
        ],
        solution_names = ["➕₁₆", "➕₂₉"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [1.0, 1.0],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0, 1.0],
        box_lower_bounds = [2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12],
    ),
    (
        index = 59,
        solve_order = 20,
        variables = ["C_f", "I_f", "K_f", "L_f", "L_s_f", "P_j_f", "Pi_ws_f", "W_disutil_f", "W_f", "W_i_f", "Y_f", "Y_s_f", "mc_f", "q_f", "r_k_f", "z_f", "➕₁₈", "➕₃", "➕₄", "➕₆", "➕₇", "➕₉"],
        previous_solution_names = ["T_f", "epsilon_I", "epsilon_L", "epsilon_a", "epsilon_b"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [24, 60, 70, 68, 18, 33, 20, 34, 16, 19, 28, 30, 27, 6, 10, 9, 67, 4, 5, 41, 17, 64],
        equations = Expr[
            :(➕₉ - (-C_f * h + C_f)),
            :((I_f * (-0.5 * varphi * (epsilon_I - 1) ^ 2 + 1) + K_f * (1 - tau)) - K_f),
            :((epsilon_b * (-K_f * r_k_f * exp(➕₁₈) + K_f * r_k_f)) / ➕₉ ^ sigma_c),
            :((((((-C_f - I_f) - (K_f * r_k_f * (exp(➕₁₈) - 1)) / psi) - L_f * W_f) + L_s_f * W_disutil_f + Pi_ws_f) - T_f) + Y_f),
            :(-L_f + L_s_f),
            :((-(P_j_f ^ (-1 - (lambda_p + 1) / lambda_p)) * Y_f * (P_j_f - mc_f) * (lambda_p + 1)) / lambda_p + Y_f / P_j_f ^ ((lambda_p + 1) / lambda_p)),
            :(-L_s_f * (-W_disutil_f + W_i_f) + Pi_ws_f),
            :(-(L_s_f ^ sigma_l) * epsilon_L * epsilon_b * omega + (W_disutil_f * epsilon_b) / ➕₉ ^ sigma_c),
            :(➕₇ - W_i_f / W_f),
            :((L_f * ➕₇ ^ (-1 + (-lambda_w - 1) / lambda_w) * (-W_disutil_f + W_i_f) * (-lambda_w - 1)) / (W_f * lambda_w) + L_s_f),
            :(-Y_f + Y_s_f),
            :(-Y_s_f + Y_f / P_j_f ^ ((lambda_p + 1) / lambda_p)),
            :(-W_f + (epsilon_a * mc_f * ➕₆ ^ alpha * (1 - alpha)) / L_f ^ alpha),
            :(beta * ((epsilon_b * (r_k_f * z_f - (r_k_f * (exp(➕₃) - 1)) / psi)) / ➕₄ ^ sigma_c + q_f * (1 - tau)) - q_f),
            :(L_f ^ (1 - alpha) * alpha * epsilon_a * mc_f * ➕₆ ^ (alpha - 1) - r_k_f),
            :(➕₆ - K_f * z_f),
            :(➕₁₈ - psi * (z_f - 1)),
            :(➕₃ - psi * (z_f - 1)),
            :(➕₄ - (-C_f * h + C_f)),
            :((L_f ^ (1 - alpha) * epsilon_a * ➕₆ ^ alpha - Phi) - Y_f / P_j_f ^ ((lambda_p + 1) / lambda_p)),
            :(L_f * ➕₇ ^ ((-lambda_w - 1) / lambda_w) - L_s_f),
            :((beta * epsilon_I * q_f * varphi * (epsilon_I - 1) - epsilon_b / ➕₉ ^ sigma_c) + q_f * ((-epsilon_I * varphi * (epsilon_I - 1) - 0.5 * varphi * (epsilon_I - 1) ^ 2) + 1)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["C_f", "I_f", "K_f", "L_f", "L_s_f", "P_j_f", "Pi_ws_f", "W_disutil_f", "W_f", "W_i_f", "Y_f", "Y_s_f", "mc_f", "q_f", "r_k_f", "z_f", "➕₁₈", "➕₃", "➕₄", "➕₆", "➕₇", "➕₉"],
        previous_solution_values = [0.362, 1.0, 1.0, 1.0, 1.0],
        external_solution_values = Float64[],
        solution_values = [1.2043777490745378, 0.4415383986956452, 17.661535947825776, 1.289115943211282, 1.289115943211282, 1.0, 0.48217380688087924, 0.7480689528666431, 1.1221034292999645, 1.1221034292999645, 2.007916147770184, 2.007916147770184, 0.7309941520467835, 2.459003350339646, 0.03510101010101021, 0.9999999999999999, -1.8762769116165147e-17, -1.8762769116165147e-17, 0.5142692988548276, 17.661535947825772, 1.0, 0.5142692988548276],
        previous_solution_initial_values = [0.362, 1.0, 1.0, 1.0, 1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 0.0, 0.0, 5.0e11, 5.0e11, 5.0e11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.999999997e11, -4.999999997e11, 5.0e11, 5.0e11, 5.0e11, 5.0e11],
        box_lower_bounds = [-1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 600.0, 600.0, 1.0e12, 1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 60,
        solve_order = 19,
        variables = ["T_f"],
        previous_solution_names = ["G_f"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [13],
        equations = Expr[
            :(-G_f + T_f),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["T_f"],
        previous_solution_values = [0.362],
        external_solution_values = Float64[],
        solution_values = [0.362],
        previous_solution_initial_values = [0.362],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.362],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 61,
        solve_order = 18,
        variables = ["G_f"],
        previous_solution_names = ["epsilon_G"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [14],
        equations = Expr[
            :(G_bar * epsilon_G - G_f),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["G_f"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [0.362],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.362],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 62,
        solve_order = 17,
        variables = ["C", "I", "K", "L", "W", "Y", "Y_s", "f_1", "f_2", "g_1", "g_2", "mc", "q", "r_k", "w_star", "z", "➕₁", "➕₁₄", "➕₁₇", "➕₂", "➕₅", "➕₈"],
        previous_solution_names = ["T", "epsilon_I", "epsilon_L", "epsilon_a", "epsilon_b", "nu_p", "pi_star", "➕₁₁", "➕₁₃", "➕₁₅"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [66, 59, 7, 8, 37, 29, 40, 45, 52, 54, 47, 55, 63, 69, 50, 1, 3, 51, 65, 2, 26, 22],
        equations = Expr[
            :((((-C - I) - (K * r_k * (exp(➕₁₇) - 1)) / psi) - T) + Y),
            :((I * (-0.5 * varphi * (epsilon_I - 1) ^ 2 + 1) + K * (1 - tau)) - K),
            :(➕₅ - K * z),
            :(L ^ (1 - alpha) * alpha * epsilon_a * mc * ➕₅ ^ (alpha - 1) - r_k),
            :(➕₁₁ - w_star / W),
            :(-Y * nu_p + Y_s),
            :((L ^ (1 - alpha) * epsilon_a * ➕₅ ^ alpha - Phi) - Y_s),
            :(-f_1 + f_2),
            :(((beta * f_2 * xi_w) / ➕₁₃ ^ (((lambda_w + 1) * (sigma_l + 1)) / lambda_w) + epsilon_L * epsilon_b * omega * ➕₁₄ ^ (sigma_l + 1)) - f_2),
            :(((Y * epsilon_b * pi_star) / ➕₈ ^ sigma_c + (beta * g_1 * xi_p) / ➕₁₅ ^ (1 / lambda_p)) - g_1),
            :(-g_1 + g_2 * (lambda_p + 1)),
            :(((Y * epsilon_b * mc) / ➕₈ ^ sigma_c + (beta * g_2 * xi_p) / ➕₁₅ ^ ((lambda_p + 1) / lambda_p)) - g_2),
            :((beta * epsilon_I * q * varphi * (epsilon_I - 1) - epsilon_b / ➕₈ ^ sigma_c) + q * ((-epsilon_I * varphi * (epsilon_I - 1) - 0.5 * varphi * (epsilon_I - 1) ^ 2) + 1)),
            :((epsilon_b * (-K * r_k * exp(➕₁₇) + K * r_k)) / ➕₈ ^ sigma_c),
            :(((L * epsilon_b * w_star) / (➕₁₁ ^ ((lambda_w + 1) / lambda_w) * ➕₈ ^ sigma_c * (lambda_w + 1)) + (beta * f_1 * xi_w) / ➕₁₃ ^ (1 / lambda_w)) - f_1),
            :(➕₁ - psi * (z - 1)),
            :(beta * ((epsilon_b * (r_k * z - (r_k * (exp(➕₁) - 1)) / psi)) / ➕₂ ^ sigma_c + q * (1 - tau)) - q),
            :(➕₁₄ - L / ➕₁₁ ^ ((lambda_w + 1) / lambda_w)),
            :(➕₁₇ - psi * (z - 1)),
            :(➕₂ - (-C * h + C)),
            :(-W + (epsilon_a * mc * ➕₅ ^ alpha * (1 - alpha)) / L ^ alpha),
            :(➕₈ - (-C * h + C)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["C", "I", "K", "L", "W", "Y", "Y_s", "f_1", "f_2", "g_1", "g_2", "mc", "q", "r_k", "w_star", "z", "➕₁", "➕₁₄", "➕₁₇", "➕₂", "➕₅", "➕₈"],
        previous_solution_values = [0.362, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        external_solution_values = Float64[],
        solution_values = [1.2043777490745378, 0.4415383986956471, 17.6615359478258, 1.2891159432112824, 1.1221034292999648, 2.007916147770185, 2.007916147770185, 8.77069946048825, 8.77069946048825, 48.847175846537034, 35.70699988781946, 0.7309941520467836, 2.4590033503396467, 0.035101010101010174, 1.1221034292999648, 1.0, 0.0, 1.2891159432112824, 0.0, 0.5142692988548276, 17.6615359478258, 0.5142692988548276],
        previous_solution_initial_values = [0.362, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 0.0, 0.0, 5.0e11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.999999997e11, 5.0e11, -4.999999997e11, 5.0e11, 5.0e11, 5.0e11],
        box_lower_bounds = [-1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 600.0, 1.0e12, 600.0, 1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 63,
        solve_order = 16,
        variables = ["T"],
        previous_solution_names = ["G"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [11],
        equations = Expr[
            :(-G + T),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["T"],
        previous_solution_values = [0.362],
        external_solution_values = Float64[],
        solution_values = [0.362],
        previous_solution_initial_values = [0.362],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.362],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 64,
        solve_order = 15,
        variables = ["G"],
        previous_solution_names = ["epsilon_G"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [12],
        equations = Expr[
            :(-G + G_bar * epsilon_G),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["G"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [0.362],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.362],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 65,
        solve_order = 14,
        variables = ["epsilon_G"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [48],
        equations = Expr[
            :(rho_G * log(epsilon_G) - log(epsilon_G)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["epsilon_G"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 66,
        solve_order = 13,
        variables = ["➕₁₁"],
        previous_solution_names = ["➕₁₂"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [39],
        equations = Expr[
            :((xi_w / ➕₁₂ ^ (1 / lambda_w) - 1) + (1 - xi_w) / ➕₁₁ ^ (1 / lambda_w)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["➕₁₁"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [1.0],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 67,
        solve_order = 12,
        variables = ["➕₁₂"],
        previous_solution_names = ["pi"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₈"],
        equation_indices = [38],
        equations = Expr[
            :(➕₁₂ - ➕₂₈ ^ gamma_w / pi),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₈ = min(1.0e12, max(eps(), pi))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₈ - pi)),
        ],
        solution_names = ["➕₁₂", "➕₂₈"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [1.0, 1.0],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0, 1.0],
        box_lower_bounds = [2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12],
    ),
    (
        index = 68,
        solve_order = 11,
        variables = ["➕₁₃"],
        previous_solution_names = ["pi"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₇"],
        equation_indices = [49],
        equations = Expr[
            :(➕₁₃ - ➕₂₇ ^ gamma_w / pi),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₇ = min(1.0e12, max(eps(), pi))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₇ - pi)),
        ],
        solution_names = ["➕₁₃", "➕₂₇"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [1.0, 1.0],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0, 1.0],
        box_lower_bounds = [2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12],
    ),
    (
        index = 69,
        solve_order = 10,
        variables = ["epsilon_L"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [43],
        equations = Expr[
            :(rho_L * log(epsilon_L) - log(epsilon_L)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["epsilon_L"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 70,
        solve_order = 9,
        variables = ["➕₁₅"],
        previous_solution_names = ["pi"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₆"],
        equation_indices = [53],
        equations = Expr[
            :(➕₁₅ - ➕₂₆ ^ gamma_p / pi),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₆ = min(1.0e12, max(eps(), pi))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₆ - pi)),
        ],
        solution_names = ["➕₁₅", "➕₂₆"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [1.0, 1.0],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0, 1.0],
        box_lower_bounds = [2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12],
    ),
    (
        index = 71,
        solve_order = 8,
        variables = ["nu_p"],
        previous_solution_names = ["pi_star", "➕₁₀"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₅"],
        equation_indices = [58],
        equations = Expr[
            :(((nu_p * xi_p) / ➕₁₀ ^ ((lambda_p + 1) / lambda_p) - nu_p) + (1 - xi_p) / ➕₂₅ ^ ((lambda_p + 1) / lambda_p)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₅ = min(1.0e12, max(eps(), pi_star))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₅ - pi_star)),
        ],
        solution_names = ["nu_p", "➕₂₅"],
        previous_solution_values = [1.0, 1.0],
        external_solution_values = Float64[],
        solution_values = [1.0, 1.0],
        previous_solution_initial_values = [1.0, 1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0, 1.0],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 72,
        solve_order = 7,
        variables = ["pi_star"],
        previous_solution_names = ["➕₁₀"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [36],
        equations = Expr[
            :((xi_p / ➕₁₀ ^ (1 / lambda_p) - 1) + (1 - xi_p) / pi_star ^ (1 / lambda_p)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["pi_star"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [1.0],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 73,
        solve_order = 6,
        variables = ["➕₁₀"],
        previous_solution_names = ["pi"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₂₄"],
        equation_indices = [35],
        equations = Expr[
            :(➕₁₀ - ➕₂₄ ^ gamma_p / pi),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₂₄ = min(1.0e12, max(eps(), pi))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₂₄ - pi)),
        ],
        solution_names = ["➕₁₀", "➕₂₄"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [1.0, 1.0],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0, 1.0],
        box_lower_bounds = [2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12],
    ),
    (
        index = 74,
        solve_order = 5,
        variables = ["pi"],
        previous_solution_names = ["pi_obj"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [120],
        equations = Expr[
            :(pi - pi_obj),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["pi"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [1.0],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 75,
        solve_order = 4,
        variables = ["pi_obj"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [119],
        equations = Expr[
            :(1 - pi_obj),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["pi_obj"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 76,
        solve_order = 3,
        variables = ["epsilon_a"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [46],
        equations = Expr[
            :(rho_a * log(epsilon_a) - log(epsilon_a)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["epsilon_a"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 77,
        solve_order = 2,
        variables = ["epsilon_I"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [44],
        equations = Expr[
            :(rho_I * log(epsilon_I) - log(epsilon_I)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["epsilon_I"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 78,
        solve_order = 1,
        variables = ["epsilon_b"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [42],
        equations = Expr[
            :(rho_b * log(epsilon_b) - log(epsilon_b)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["epsilon_b"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
]
const BLOCK_EQUATION_ORDER = [74, 71, 73, 75, 72, 76, 77, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 62, 61, 31, 32, 25, 23, 21, 15, 57, 56, 24, 60, 70, 68, 18, 33, 20, 34, 16, 19, 28, 30, 27, 6, 10, 9, 67, 4, 5, 41, 17, 64, 13, 14, 66, 59, 7, 8, 37, 29, 40, 45, 52, 54, 47, 55, 63, 69, 50, 1, 3, 51, 65, 2, 26, 22, 11, 12, 48, 39, 38, 49, 43, 53, 58, 36, 35, 120, 119, 46, 44, 42]
const BLOCK_SOLVE_ORDER = [78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    ["R", "pi_obj", "χᵒᵇᶜ⁺ꜝ¹ꜝʳ", "➕₁₉"],
    ["pi_obj"],
    ["R"],
    ["Χᵒᵇᶜ⁺ꜝ¹ꜝ", "χᵒᵇᶜ⁺ꜝ¹ꜝˡ"],
    ["R"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝ"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾"],
    String[],
    ["L_s_f", "epsilon_L", "epsilon_b", "➕₉"],
    ["L_s", "epsilon_L", "epsilon_b", "➕₈"],
    ["epsilon_b", "➕₄", "➕₉"],
    ["epsilon_b", "pi", "➕₂", "➕₈"],
    ["epsilon_b", "q_f", "➕₉"],
    ["epsilon_b", "q", "➕₈"],
    ["P_j_f", "Y_f", "mc_f"],
    ["L", "nu_w"],
    ["➕₁₁", "➕₁₆"],
    ["pi"],
    ["T_f", "epsilon_I", "epsilon_L", "epsilon_a", "epsilon_b"],
    ["G_f"],
    ["epsilon_G"],
    ["T", "epsilon_I", "epsilon_L", "epsilon_a", "epsilon_b", "nu_p", "pi_star", "➕₁₁", "➕₁₃", "➕₁₅"],
    ["G"],
    ["epsilon_G"],
    String[],
    ["➕₁₂"],
    ["pi"],
    ["pi"],
    String[],
    ["pi"],
    ["pi_star", "➕₁₀"],
    ["➕₁₀"],
    ["pi"],
    ["pi_obj"],
    String[],
    String[],
    String[],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    [1.0101010101010097, 1.0, 0.0, 0.9900000000000003],
    [1.0],
    [1.0101010101010097],
    [0.0, -0.010050335853501065],
    [1.0101010101010097],
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
    [0.0],
    [0.0],
    Float64[],
    [1.289115943211282, 1.0, 1.0, 0.5142692988548276],
    [1.2891159432112824, 1.0, 1.0, 0.5142692988548276],
    [1.0, 0.5142692988548276, 0.5142692988548276],
    [1.0, 1.0, 0.5142692988548276, 0.5142692988548276],
    [1.0, 2.459003350339646, 0.5142692988548276],
    [1.0, 2.4590033503396467, 0.5142692988548276],
    [1.0, 2.007916147770184, 0.7309941520467835],
    [1.2891159432112824, 1.0],
    [1.0, 1.0],
    [1.0],
    [0.362, 1.0, 1.0, 1.0, 1.0],
    [0.362],
    [1.0],
    [0.362, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.362],
    [1.0],
    Float64[],
    [1.0],
    [1.0],
    [1.0],
    Float64[],
    [1.0],
    [1.0, 1.0],
    [1.0],
    [1.0],
    [1.0],
    Float64[],
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
    Float64[],
    Float64[],
    Float64[],
]
const BLOCK_SOLUTION_NAMES = [
    ["calibr_pi", "➕₃₅", "➕₃₆"],
    ["calibr_pi_obj", "➕₃₄"],
    ["➕₁₉"],
    ["χᵒᵇᶜ⁺ꜝ¹ꜝʳ"],
    ["χᵒᵇᶜ⁺ꜝ¹ꜝˡ", "➕₃₃"],
    ["Χᵒᵇᶜ⁺ꜝ¹ꜝ"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝ"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁴⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³³⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³²⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³¹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁰⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁸⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁷⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁶⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁵⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁴⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²³⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²²⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²¹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁰⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁸⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁷⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁶⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁵⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁴⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹³⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹²⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹¹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁰⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁸⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁷⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁶⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁵⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾"],
    ["U_f", "➕₃₂"],
    ["U", "➕₃₁"],
    ["R_f"],
    ["R"],
    ["Q_f"],
    ["Q"],
    ["Pi_ps_f", "➕₃₀"],
    ["L_s"],
    ["nu_w"],
    ["➕₁₆", "➕₂₉"],
    ["C_f", "I_f", "K_f", "L_f", "L_s_f", "P_j_f", "Pi_ws_f", "W_disutil_f", "W_f", "W_i_f", "Y_f", "Y_s_f", "mc_f", "q_f", "r_k_f", "z_f", "➕₁₈", "➕₃", "➕₄", "➕₆", "➕₇", "➕₉"],
    ["T_f"],
    ["G_f"],
    ["C", "I", "K", "L", "W", "Y", "Y_s", "f_1", "f_2", "g_1", "g_2", "mc", "q", "r_k", "w_star", "z", "➕₁", "➕₁₄", "➕₁₇", "➕₂", "➕₅", "➕₈"],
    ["T"],
    ["G"],
    ["epsilon_G"],
    ["➕₁₁"],
    ["➕₁₂", "➕₂₈"],
    ["➕₁₃", "➕₂₇"],
    ["epsilon_L"],
    ["➕₁₅", "➕₂₆"],
    ["nu_p", "➕₂₅"],
    ["pi_star"],
    ["➕₁₀", "➕₂₄"],
    ["pi"],
    ["pi_obj"],
    ["epsilon_a"],
    ["epsilon_I"],
    ["epsilon_b"],
]
const BLOCK_SOLUTION_VALUES = [
    [4.85722573273506e-17, 1.0, 1.0101010101010097],
    [1.0, 1.0],
    [0.9900000000000003],
    [0.0],
    [-0.010050335853501065, 1.0101010101010097],
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
    [0.0],
    [0.0],
    [0.0],
    [-427.98589108169494, 1.289115943211282],
    [-427.98589108169506, 1.2891159432112824],
    [1.0101010101010102],
    [1.0101010101010097],
    [0.9999999999999998],
    [1.0000000000000002],
    [0.5401411859498743, 1.0],
    [1.2891159432112824],
    [1.0],
    [1.0, 1.0],
    [1.2043777490745378, 0.4415383986956452, 17.661535947825776, 1.289115943211282, 1.289115943211282, 1.0, 0.48217380688087924, 0.7480689528666431, 1.1221034292999645, 1.1221034292999645, 2.007916147770184, 2.007916147770184, 0.7309941520467835, 2.459003350339646, 0.03510101010101021, 0.9999999999999999, -1.8762769116165147e-17, -1.8762769116165147e-17, 0.5142692988548276, 17.661535947825772, 1.0, 0.5142692988548276],
    [0.362],
    [0.362],
    [1.2043777490745378, 0.4415383986956471, 17.6615359478258, 1.2891159432112824, 1.1221034292999648, 2.007916147770185, 2.007916147770185, 8.77069946048825, 8.77069946048825, 48.847175846537034, 35.70699988781946, 0.7309941520467836, 2.4590033503396467, 0.035101010101010174, 1.1221034292999648, 1.0, 0.0, 1.2891159432112824, 0.0, 0.5142692988548276, 17.6615359478258, 0.5142692988548276],
    [0.362],
    [0.362],
    [1.0],
    [1.0],
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0],
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0],
    [1.0, 1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
]
const BLOCK_PREVIOUS_SOLUTION_INITIAL_VALUES = [
    [1.0101010101010102, 1.0, 0.0, 0.99],
    [1.0],
    [1.0101010101010102],
    [0.0, -0.010050335853501506],
    [1.0101010101010102],
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
    [0.0],
    [0.0],
    Float64[],
    [5.0e11, 1.0, 1.0, 5.0e11],
    [5.0e11, 1.0, 1.0, 5.0e11],
    [1.0, 5.0e11, 5.0e11],
    [1.0, 1.0, 5.0e11, 5.0e11],
    [1.0, 0.0, 5.0e11],
    [1.0, 0.0, 5.0e11],
    [5.0e11, 0.0, 0.0],
    [5.0e11, 1.0],
    [1.0, 1.0],
    [1.0],
    [0.362, 1.0, 1.0, 1.0, 1.0],
    [0.362],
    [1.0],
    [0.362, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.362],
    [1.0],
    Float64[],
    [1.0],
    [1.0],
    [1.0],
    Float64[],
    [1.0],
    [1.0, 1.0],
    [1.0],
    [1.0],
    [1.0],
    Float64[],
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
    [-5.551115123125783e-17, 1.0, 1.0101010101010102],
    [1.0, 1.0],
    [0.99],
    [0.0],
    [-0.010050335853501506, 1.0101010101010102],
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
    [0.0],
    [0.0],
    [0.0],
    [-1.7580009187316713e41, 5.0e11],
    [-1.7580009187316713e41, 5.0e11],
    [1.0101010101010102],
    [1.0101010101010102],
    [0.0],
    [0.0],
    [0.0, 5.0e11],
    [5.0e11],
    [1.0],
    [1.0, 1.0],
    [0.0, 0.0, 0.0, 5.0e11, 5.0e11, 5.0e11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.999999997e11, -4.999999997e11, 5.0e11, 5.0e11, 5.0e11, 5.0e11],
    [0.362],
    [0.362],
    [0.0, 0.0, 0.0, 5.0e11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.999999997e11, 5.0e11, -4.999999997e11, 5.0e11, 5.0e11, 5.0e11],
    [0.362],
    [0.362],
    [1.0],
    [1.0],
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0],
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0],
    [1.0, 1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[33] = parameters[33]
    complete_parameters[6] = parameters[6]
    complete_parameters[37] = parameters[37]
    complete_parameters[39] = parameters[39]
    complete_parameters[40] = parameters[40]
    complete_parameters[18] = parameters[18]
    complete_parameters[23] = parameters[23]
    complete_parameters[29] = parameters[29]
    complete_parameters[4] = parameters[4]
    complete_parameters[10] = parameters[10]
    complete_parameters[20] = parameters[20]
    complete_parameters[13] = parameters[13]
    complete_parameters[22] = parameters[22]
    complete_parameters[24] = parameters[24]
    complete_parameters[28] = parameters[28]
    complete_parameters[31] = parameters[31]
    complete_parameters[14] = parameters[14]
    complete_parameters[32] = parameters[32]
    complete_parameters[38] = parameters[38]
    complete_parameters[15] = parameters[15]
    complete_parameters[16] = parameters[16]
    complete_parameters[35] = parameters[35]
    complete_parameters[12] = parameters[12]
    complete_parameters[36] = parameters[36]
    complete_parameters[17] = parameters[17]
    complete_parameters[8] = parameters[8]
    complete_parameters[9] = parameters[9]
    complete_parameters[2] = parameters[2]
    complete_parameters[19] = parameters[19]
    complete_parameters[25] = parameters[25]
    complete_parameters[21] = parameters[21]
    complete_parameters[11] = parameters[11]
    complete_parameters[7] = parameters[7]
    complete_parameters[30] = parameters[30]
    complete_parameters[34] = parameters[34]
    complete_parameters[1] = parameters[1]
    complete_parameters[5] = parameters[5]
    complete_parameters[26] = parameters[26]
    complete_parameters[3] = parameters[3]
    complete_parameters[27] = parameters[27]
    complete_parameters[41] = complete_parameters[37] / complete_parameters[30]
    complete_parameters[42] = complete_parameters[33] / complete_parameters[30]
    complete_parameters[43] = complete_parameters[32] / complete_parameters[30]
    complete_parameters[44] = complete_parameters[38] / complete_parameters[30]
    complete_parameters[45] = complete_parameters[34] / complete_parameters[30]
    complete_parameters[46] = complete_parameters[31] / complete_parameters[30]
    complete_parameters[47] = complete_parameters[36] / complete_parameters[30]
    complete_parameters[48] = complete_parameters[39] / complete_parameters[30]
    complete_parameters[49] = complete_parameters[35] / complete_parameters[30]
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[48]) + complete_parameters[7] * ((1 - complete_parameters[19]) * solution[48] + solution[36] * (solution[50] * solution[53] - complete_parameters[12] ^ -1 * solution[50] * (-1 + exp(complete_parameters[12] * (-1 + solution[53])))) * (solution[1] - complete_parameters[10] * solution[1]) ^ -(complete_parameters[17])),
        -(solution[49]) + complete_parameters[7] * ((1 - complete_parameters[19]) * solution[49] + solution[36] * (solution[51] * solution[54] - complete_parameters[12] ^ -1 * solution[51] * (-1 + exp(complete_parameters[12] * (-1 + solution[54])))) * (solution[2] - complete_parameters[10] * solution[2]) ^ -(complete_parameters[17])),
        -(solution[50]) + complete_parameters[6] * solution[35] * solution[41] * solution[9] ^ (1 - complete_parameters[6]) * (solution[7] * solution[53]) ^ (-1 + complete_parameters[6]),
        -(solution[51]) + complete_parameters[6] * solution[35] * solution[42] * solution[10] ^ (1 - complete_parameters[6]) * (solution[8] * solution[54]) ^ (-1 + complete_parameters[6]),
        -(solution[3]) + solution[20],
        -(solution[3]) + complete_parameters[3] * solution[32],
        -(solution[4]) + solution[21],
        -(solution[4]) + complete_parameters[3] * solution[32],
        -(solution[9]) + solution[44] ^ -1 * solution[11],
        -(solution[12]) + solution[10] * (solution[27] * solution[26] ^ -1) ^ (complete_parameters[4] ^ -1 * (-1 - complete_parameters[4])),
        solution[12] - solution[10],
        solution[12] + complete_parameters[4] ^ -1 * solution[10] * solution[26] ^ -1 * (-1 - complete_parameters[4]) * (-(solution[25]) + solution[27]) * (solution[27] * solution[26] ^ -1) ^ (-1 + complete_parameters[4] ^ -1 * (-1 - complete_parameters[4])),
        solution[15] - solution[12] * (-(solution[25]) + solution[27]),
        solution[14] - solution[29] * (-(solution[42]) + solution[13]) * solution[13] ^ (-(complete_parameters[2] ^ -1) * (1 + complete_parameters[2])),
        -(solution[16]) + solution[36] ^ -1 * solution[48] * (solution[1] - complete_parameters[10] * solution[1]) ^ complete_parameters[17],
        -(solution[17]) + solution[36] ^ -1 * solution[49] * (solution[2] - complete_parameters[10] * solution[2]) ^ complete_parameters[17],
        -(solution[24]) + solution[35] * solution[41] * (1 - complete_parameters[6]) * solution[9] ^ -(complete_parameters[6]) * (solution[7] * solution[53]) ^ complete_parameters[6],
        -(solution[26]) + solution[35] * solution[42] * (1 - complete_parameters[6]) * solution[10] ^ -(complete_parameters[6]) * (solution[8] * solution[54]) ^ complete_parameters[6],
        -(solution[29]) + solution[31],
        solution[30] - solution[43] * solution[28],
        -(solution[31]) + solution[29] * solution[13] ^ (-(complete_parameters[2] ^ -1) * (1 + complete_parameters[2])),
        complete_parameters[7] * solution[36] * (solution[2] - complete_parameters[10] * solution[2]) ^ -(complete_parameters[17]) - solution[36] * solution[19] ^ -1 * (solution[2] - complete_parameters[10] * solution[2]) ^ -(complete_parameters[17]),
        complete_parameters[7] * solution[36] * solution[45] ^ -1 * (solution[1] - complete_parameters[10] * solution[1]) ^ -(complete_parameters[17]) - solution[36] * solution[18] ^ -1 * (solution[1] - complete_parameters[10] * solution[1]) ^ -(complete_parameters[17]),
        solution[29] * solution[13] ^ (-(complete_parameters[2] ^ -1) * (1 + complete_parameters[2])) - complete_parameters[2] ^ -1 * solution[29] * (1 + complete_parameters[2]) * (-(solution[42]) + solution[13]) * solution[13] ^ (-1 - complete_parameters[2] ^ -1 * (1 + complete_parameters[2])),
        solution[36] * solution[25] * (solution[2] - complete_parameters[10] * solution[2]) ^ -(complete_parameters[17]) - complete_parameters[11] * solution[36] * solution[34] * solution[12] ^ complete_parameters[18],
        -1 + complete_parameters[22] * (solution[45] ^ -1 * solution[45] ^ complete_parameters[9]) ^ -(complete_parameters[2] ^ -1) + (1 - complete_parameters[22]) * solution[47] ^ -(complete_parameters[2] ^ -1),
        -1 + (1 - complete_parameters[21]) * (solution[52] * solution[24] ^ -1) ^ -(complete_parameters[4] ^ -1) + complete_parameters[21] * (solution[24] * solution[24] ^ -1) ^ -(complete_parameters[4] ^ -1) * (solution[45] ^ -1 * solution[45] ^ complete_parameters[8]) ^ -(complete_parameters[4] ^ -1),
        (-(complete_parameters[5]) - solution[30]) + solution[35] * solution[9] ^ (1 - complete_parameters[6]) * (solution[7] * solution[53]) ^ complete_parameters[6],
        (-(complete_parameters[5]) - solution[29] * solution[13] ^ (-(complete_parameters[2] ^ -1) * (1 + complete_parameters[2]))) + solution[35] * solution[10] ^ (1 - complete_parameters[6]) * (solution[8] * solution[54]) ^ complete_parameters[6],
        (complete_parameters[46] * 0 - log(solution[36])) + complete_parameters[24] * log(solution[36]),
        (-(complete_parameters[43]) * 0 - log(solution[34])) + complete_parameters[25] * log(solution[34]),
        (complete_parameters[42] * 0 - log(solution[33])) + complete_parameters[26] * log(solution[33]),
        (complete_parameters[49] * 0 - solution[37]) + solution[38],
        (complete_parameters[45] * 0 - log(solution[35])) + complete_parameters[27] * log(solution[35]),
        (complete_parameters[47] * 0 - solution[39]) + solution[40] * (1 + complete_parameters[2]),
        (complete_parameters[41] * 0 - log(solution[32])) + complete_parameters[28] * log(solution[32]),
        -(solution[37]) + complete_parameters[7] * complete_parameters[21] * solution[37] * (solution[52] ^ -1 * solution[52]) ^ (complete_parameters[4] ^ -1) * (solution[45] ^ -1 * solution[45] ^ complete_parameters[8]) ^ -(complete_parameters[4] ^ -1) + solution[36] * solution[52] * solution[9] * (1 + complete_parameters[4]) ^ -1 * (solution[1] - complete_parameters[10] * solution[1]) ^ -(complete_parameters[17]) * (solution[52] * solution[24] ^ -1) ^ (-(complete_parameters[4] ^ -1) * (1 + complete_parameters[4])),
        -(solution[38]) + complete_parameters[7] * complete_parameters[21] * solution[38] * (solution[52] ^ -1 * solution[52]) ^ (complete_parameters[4] ^ -1 * (1 + complete_parameters[4]) * (1 + complete_parameters[18])) * (solution[45] ^ -1 * solution[45] ^ complete_parameters[8]) ^ (-(complete_parameters[4] ^ -1) * (1 + complete_parameters[4]) * (1 + complete_parameters[18])) + complete_parameters[11] * solution[36] * solution[34] * (solution[9] * (solution[52] * solution[24] ^ -1) ^ (-(complete_parameters[4] ^ -1) * (1 + complete_parameters[4]))) ^ (1 + complete_parameters[18]),
        -(solution[39]) + complete_parameters[7] * complete_parameters[22] * solution[47] * solution[39] * solution[47] ^ -1 * (solution[45] ^ -1 * solution[45] ^ complete_parameters[9]) ^ -(complete_parameters[2] ^ -1) + solution[36] * solution[47] * solution[28] * (solution[1] - complete_parameters[10] * solution[1]) ^ -(complete_parameters[17]),
        -(solution[40]) + complete_parameters[7] * complete_parameters[22] * solution[40] * (solution[45] ^ -1 * solution[45] ^ complete_parameters[9]) ^ (-(complete_parameters[2] ^ -1) * (1 + complete_parameters[2])) + solution[36] * solution[41] * solution[28] * (solution[1] - complete_parameters[10] * solution[1]) ^ -(complete_parameters[17]),
        -(solution[44]) + (1 - complete_parameters[21]) * (solution[52] * solution[24] ^ -1) ^ (-(complete_parameters[4] ^ -1) * (1 + complete_parameters[4])) + complete_parameters[21] * solution[44] * (solution[24] * solution[45] ^ -1 * solution[24] ^ -1 * solution[45] ^ complete_parameters[8]) ^ (-(complete_parameters[4] ^ -1) * (1 + complete_parameters[4])),
        -(solution[43]) + (1 - complete_parameters[22]) * solution[47] ^ (-(complete_parameters[2] ^ -1) * (1 + complete_parameters[2])) + complete_parameters[22] * solution[43] * (solution[45] ^ -1 * solution[45] ^ complete_parameters[9]) ^ (-(complete_parameters[2] ^ -1) * (1 + complete_parameters[2])),
        -(solution[7]) + solution[7] * (1 - complete_parameters[19]) + solution[5] * (1 - 0.5 * complete_parameters[20] * (-1 + solution[5] ^ -1 * solution[33] * solution[5]) ^ 2),
        -(solution[8]) + solution[8] * (1 - complete_parameters[19]) + solution[6] * (1 - 0.5 * complete_parameters[20] * (-1 + solution[6] ^ -1 * solution[33] * solution[6]) ^ 2),
        (solution[22] - complete_parameters[7] * solution[22]) - solution[36] * ((1 - complete_parameters[17]) ^ -1 * (solution[1] - complete_parameters[10] * solution[1]) ^ (1 - complete_parameters[17]) - complete_parameters[11] * solution[34] * (1 + complete_parameters[18]) ^ -1 * solution[11] ^ (1 + complete_parameters[18])),
        (solution[23] - complete_parameters[7] * solution[23]) - solution[36] * ((1 - complete_parameters[17]) ^ -1 * (solution[2] - complete_parameters[10] * solution[2]) ^ (1 - complete_parameters[17]) - complete_parameters[11] * solution[34] * (1 + complete_parameters[18]) ^ -1 * solution[12] ^ (1 + complete_parameters[18])),
        -(solution[36]) * (solution[1] - complete_parameters[10] * solution[1]) ^ -(complete_parameters[17]) + solution[48] * ((1 - 0.5 * complete_parameters[20] * (-1 + solution[5] ^ -1 * solution[33] * solution[5]) ^ 2) - complete_parameters[20] * solution[5] ^ -1 * solution[33] * solution[5] * (-1 + solution[5] ^ -1 * solution[33] * solution[5])) + complete_parameters[7] * complete_parameters[20] * solution[5] ^ -2 * solution[33] * solution[48] * solution[5] ^ 2 * (-1 + solution[5] ^ -1 * solution[33] * solution[5]),
        -(solution[36]) * (solution[2] - complete_parameters[10] * solution[2]) ^ -(complete_parameters[17]) + solution[49] * ((1 - 0.5 * complete_parameters[20] * (-1 + solution[6] ^ -1 * solution[33] * solution[6]) ^ 2) - complete_parameters[20] * solution[6] ^ -1 * solution[33] * solution[6] * (-1 + solution[6] ^ -1 * solution[33] * solution[6])) + complete_parameters[7] * complete_parameters[20] * solution[6] ^ -2 * solution[33] * solution[49] * solution[6] ^ 2 * (-1 + solution[6] ^ -1 * solution[33] * solution[6]),
        (((-(solution[1]) - solution[5]) - solution[20]) + solution[28]) - complete_parameters[12] ^ -1 * solution[50] * solution[7] * (-1 + exp(complete_parameters[12] * (-1 + solution[53]))),
        (((((-(solution[2]) - solution[6]) + solution[15]) - solution[21]) + solution[29] + solution[12] * solution[25]) - solution[10] * solution[26]) - complete_parameters[12] ^ -1 * solution[51] * solution[8] * (-1 + exp(complete_parameters[12] * (-1 + solution[54]))),
        solution[36] * (solution[7] * solution[50] - solution[50] * solution[7] * exp(complete_parameters[12] * (-1 + solution[53]))) * (solution[1] - complete_parameters[10] * solution[1]) ^ -(complete_parameters[17]),
        solution[36] * (solution[8] * solution[51] - solution[51] * solution[8] * exp(complete_parameters[12] * (-1 + solution[54]))) * (solution[2] - complete_parameters[10] * solution[2]) ^ -(complete_parameters[17]),
        (complete_parameters[48] * 0 - log(solution[46])) + complete_parameters[29] * log(solution[46]) + log(solution[100]) * (1 - complete_parameters[29]),
        solution[57] - (complete_parameters[1] - log(solution[18])),
        solution[56] - (((((complete_parameters[15] * (-(log(solution[45] ^ -1 * solution[45])) + log(solution[45] ^ -1 * solution[45])) + complete_parameters[16] * ((-(log(solution[28] ^ -1 * solution[28])) + log(solution[28] ^ -1 * solution[28]) + log(solution[29] ^ -1 * solution[29])) - log(solution[29] ^ -1 * solution[29])) + complete_parameters[23] * log(solution[18] ^ -1 * solution[18]) + (1 - complete_parameters[23]) * (log(solution[46]) + complete_parameters[13] * (-(log(solution[46])) + log(solution[45] ^ -1 * solution[45])) + complete_parameters[14] * (log(solution[28] ^ -1 * solution[28]) - log(solution[29] ^ -1 * solution[29])))) - solution[101]) + complete_parameters[44] * 0) - log(solution[18] ^ -1)) - log(solution[18])),
        solution[55] - max(solution[57], solution[56]),
        solution[55] - solution[58],
        solution[58] - solution[93],
        solution[92] - complete_parameters[40] * 0,
        solution[91] - (solution[92] + complete_parameters[40] * 0),
        solution[69] - (solution[91] + complete_parameters[40] * 0),
        solution[80] - (solution[69] + complete_parameters[40] * 0),
        solution[94] - (solution[80] + complete_parameters[40] * 0),
        solution[95] - (solution[94] + complete_parameters[40] * 0),
        solution[96] - (solution[95] + complete_parameters[40] * 0),
        solution[97] - (solution[96] + complete_parameters[40] * 0),
        solution[98] - (solution[97] + complete_parameters[40] * 0),
        solution[99] - (solution[98] + complete_parameters[40] * 0),
        solution[84] - (solution[99] + complete_parameters[40] * 0),
        solution[83] - (solution[84] + complete_parameters[40] * 0),
        solution[81] - (solution[83] + complete_parameters[40] * 0),
        solution[82] - (solution[81] + complete_parameters[40] * 0),
        solution[85] - (solution[82] + complete_parameters[40] * 0),
        solution[86] - (solution[85] + complete_parameters[40] * 0),
        solution[87] - (solution[86] + complete_parameters[40] * 0),
        solution[88] - (solution[87] + complete_parameters[40] * 0),
        solution[89] - (solution[88] + complete_parameters[40] * 0),
        solution[90] - (solution[89] + complete_parameters[40] * 0),
        solution[62] - (solution[90] + complete_parameters[40] * 0),
        solution[61] - (solution[62] + complete_parameters[40] * 0),
        solution[59] - (solution[61] + complete_parameters[40] * 0),
        solution[60] - (solution[59] + complete_parameters[40] * 0),
        solution[63] - (solution[60] + complete_parameters[40] * 0),
        solution[64] - (solution[63] + complete_parameters[40] * 0),
        solution[65] - (solution[64] + complete_parameters[40] * 0),
        solution[66] - (solution[65] + complete_parameters[40] * 0),
        solution[67] - (solution[66] + complete_parameters[40] * 0),
        solution[68] - (solution[67] + complete_parameters[40] * 0),
        solution[73] - (solution[68] + complete_parameters[40] * 0),
        solution[72] - (solution[73] + complete_parameters[40] * 0),
        solution[70] - (solution[72] + complete_parameters[40] * 0),
        solution[71] - (solution[70] + complete_parameters[40] * 0),
        solution[74] - (solution[71] + complete_parameters[40] * 0),
        solution[75] - (solution[74] + complete_parameters[40] * 0),
        solution[76] - (solution[75] + complete_parameters[40] * 0),
        solution[77] - (solution[76] + complete_parameters[40] * 0),
        solution[78] - (solution[77] + complete_parameters[40] * 0),
        solution[79] - (solution[78] + complete_parameters[40] * 0),
        solution[93] - (solution[79] + complete_parameters[40] * 0),
        1 - solution[46],
        solution[45] - solution[46],
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[100] - complete_parameters[12] * (solution[53] - 1),
        solution[111] - (-(solution[1]) * complete_parameters[10] + solution[1]),
        complete_parameters[7] * ((solution[36] * (solution[50] * solution[53] - (solution[50] * (exp(solution[100]) - 1)) / complete_parameters[12])) / solution[111] ^ complete_parameters[17] + solution[48] * (1 - complete_parameters[19])) - solution[48],
        solution[112] - complete_parameters[12] * (solution[54] - 1),
        solution[113] - (-(solution[2]) * complete_parameters[10] + solution[2]),
        complete_parameters[7] * ((solution[36] * (solution[51] * solution[54] - (solution[51] * (exp(solution[112]) - 1)) / complete_parameters[12])) / solution[113] ^ complete_parameters[17] + solution[49] * (1 - complete_parameters[19])) - solution[49],
        solution[114] - solution[7] * solution[53],
        solution[9] ^ (1 - complete_parameters[6]) * complete_parameters[6] * solution[35] * solution[41] * solution[114] ^ (complete_parameters[6] - 1) - solution[50],
        solution[115] - solution[8] * solution[54],
        solution[10] ^ (1 - complete_parameters[6]) * complete_parameters[6] * solution[35] * solution[42] * solution[115] ^ (complete_parameters[6] - 1) - solution[51],
        -(solution[3]) + solution[20],
        -(solution[3]) + complete_parameters[3] * solution[32],
        -(solution[4]) + solution[21],
        complete_parameters[3] * solution[32] - solution[4],
        -(solution[9]) + solution[11] / solution[44],
        solution[116] - solution[27] / solution[26],
        solution[10] * solution[116] ^ ((-(complete_parameters[4]) - 1) / complete_parameters[4]) - solution[12],
        -(solution[10]) + solution[12],
        (solution[10] * solution[116] ^ (-1 + (-(complete_parameters[4]) - 1) / complete_parameters[4]) * (-(solution[25]) + solution[27]) * (-(complete_parameters[4]) - 1)) / (solution[26] * complete_parameters[4]) + solution[12],
        -(solution[12]) * (-(solution[25]) + solution[27]) + solution[15],
        solution[14] - (solution[29] * (solution[13] - solution[42])) / solution[13] ^ ((complete_parameters[2] + 1) / complete_parameters[2]),
        solution[117] - (-(solution[1]) * complete_parameters[10] + solution[1]),
        -(solution[16]) + (solution[48] * solution[117] ^ complete_parameters[17]) / solution[36],
        solution[118] - (-(solution[2]) * complete_parameters[10] + solution[2]),
        -(solution[17]) + (solution[49] * solution[118] ^ complete_parameters[17]) / solution[36],
        -(solution[24]) + (solution[35] * solution[41] * solution[114] ^ complete_parameters[6] * (1 - complete_parameters[6])) / solution[9] ^ complete_parameters[6],
        -(solution[26]) + (solution[35] * solution[42] * solution[115] ^ complete_parameters[6] * (1 - complete_parameters[6])) / solution[10] ^ complete_parameters[6],
        -(solution[29]) + solution[31],
        -(solution[28]) * solution[43] + solution[30],
        -(solution[31]) + solution[29] / solution[13] ^ ((complete_parameters[2] + 1) / complete_parameters[2]),
        (complete_parameters[7] * solution[36]) / solution[113] ^ complete_parameters[17] - solution[36] / (solution[19] * solution[118] ^ complete_parameters[17]),
        (complete_parameters[7] * solution[36]) / (solution[45] * solution[111] ^ complete_parameters[17]) - solution[36] / (solution[18] * solution[117] ^ complete_parameters[17]),
        (-(solution[13] ^ (-1 - (complete_parameters[2] + 1) / complete_parameters[2])) * solution[29] * (solution[13] - solution[42]) * (complete_parameters[2] + 1)) / complete_parameters[2] + solution[29] / solution[13] ^ ((complete_parameters[2] + 1) / complete_parameters[2]),
        -(solution[12] ^ complete_parameters[18]) * solution[34] * solution[36] * complete_parameters[11] + (solution[25] * solution[36]) / solution[118] ^ complete_parameters[17],
        solution[101] - solution[45] ^ complete_parameters[9] / solution[45],
        (complete_parameters[22] / solution[101] ^ (1 / complete_parameters[2]) - 1) + (1 - complete_parameters[22]) / solution[47] ^ (1 / complete_parameters[2]),
        solution[102] - solution[52] / solution[24],
        solution[103] - solution[45] ^ complete_parameters[8] / solution[45],
        (complete_parameters[21] / solution[103] ^ (1 / complete_parameters[4]) - 1) + (1 - complete_parameters[21]) / solution[102] ^ (1 / complete_parameters[4]),
        (solution[9] ^ (1 - complete_parameters[6]) * solution[35] * solution[114] ^ complete_parameters[6] - complete_parameters[5]) - solution[30],
        (solution[10] ^ (1 - complete_parameters[6]) * solution[35] * solution[115] ^ complete_parameters[6] - complete_parameters[5]) - solution[29] / solution[13] ^ ((complete_parameters[2] + 1) / complete_parameters[2]),
        complete_parameters[24] * log(solution[36]) - log(solution[36]),
        complete_parameters[25] * log(solution[34]) - log(solution[34]),
        complete_parameters[26] * log(solution[33]) - log(solution[33]),
        -(solution[37]) + solution[38],
        complete_parameters[27] * log(solution[35]) - log(solution[35]),
        -(solution[39]) + solution[40] * (complete_parameters[2] + 1),
        complete_parameters[28] * log(solution[32]) - log(solution[32]),
        solution[104] - solution[45] ^ complete_parameters[8] / solution[45],
        ((solution[9] * solution[36] * solution[52]) / (solution[102] ^ ((complete_parameters[4] + 1) / complete_parameters[4]) * solution[117] ^ complete_parameters[17] * (complete_parameters[4] + 1)) + (complete_parameters[7] * solution[37] * complete_parameters[21]) / solution[104] ^ (1 / complete_parameters[4])) - solution[37],
        solution[105] - solution[9] / solution[102] ^ ((complete_parameters[4] + 1) / complete_parameters[4]),
        ((complete_parameters[7] * solution[38] * complete_parameters[21]) / solution[104] ^ (((complete_parameters[4] + 1) * (complete_parameters[18] + 1)) / complete_parameters[4]) + solution[34] * solution[36] * complete_parameters[11] * solution[105] ^ (complete_parameters[18] + 1)) - solution[38],
        solution[106] - solution[45] ^ complete_parameters[9] / solution[45],
        ((solution[28] * solution[36] * solution[47]) / solution[117] ^ complete_parameters[17] + (complete_parameters[7] * solution[39] * complete_parameters[22]) / solution[106] ^ (1 / complete_parameters[2])) - solution[39],
        ((solution[28] * solution[36] * solution[41]) / solution[117] ^ complete_parameters[17] + (complete_parameters[7] * solution[40] * complete_parameters[22]) / solution[106] ^ ((complete_parameters[2] + 1) / complete_parameters[2])) - solution[40],
        solution[107] - solution[45] ^ complete_parameters[8] / solution[45],
        ((solution[44] * complete_parameters[21]) / solution[107] ^ ((complete_parameters[4] + 1) / complete_parameters[4]) - solution[44]) + (1 - complete_parameters[21]) / solution[102] ^ ((complete_parameters[4] + 1) / complete_parameters[4]),
        ((solution[43] * complete_parameters[22]) / solution[101] ^ ((complete_parameters[2] + 1) / complete_parameters[2]) - solution[43]) + (1 - complete_parameters[22]) / solution[47] ^ ((complete_parameters[2] + 1) / complete_parameters[2]),
        (solution[5] * (-0.5 * complete_parameters[20] * (solution[33] - 1) ^ 2 + 1) + solution[7] * (1 - complete_parameters[19])) - solution[7],
        (solution[6] * (-0.5 * complete_parameters[20] * (solution[33] - 1) ^ 2 + 1) + solution[8] * (1 - complete_parameters[19])) - solution[8],
        (-(solution[22]) * complete_parameters[7] + solution[22]) - solution[36] * ((-(solution[11] ^ (complete_parameters[18] + 1)) * solution[34] * complete_parameters[11]) / (complete_parameters[18] + 1) + solution[117] ^ (1 - complete_parameters[17]) / (1 - complete_parameters[17])),
        (-(solution[23]) * complete_parameters[7] + solution[23]) - solution[36] * ((-(solution[12] ^ (complete_parameters[18] + 1)) * solution[34] * complete_parameters[11]) / (complete_parameters[18] + 1) + solution[118] ^ (1 - complete_parameters[17]) / (1 - complete_parameters[17])),
        (complete_parameters[7] * solution[33] * solution[48] * complete_parameters[20] * (solution[33] - 1) - solution[36] / solution[117] ^ complete_parameters[17]) + solution[48] * ((-(solution[33]) * complete_parameters[20] * (solution[33] - 1) - 0.5 * complete_parameters[20] * (solution[33] - 1) ^ 2) + 1),
        (complete_parameters[7] * solution[33] * solution[49] * complete_parameters[20] * (solution[33] - 1) - solution[36] / solution[118] ^ complete_parameters[17]) + solution[49] * ((-(solution[33]) * complete_parameters[20] * (solution[33] - 1) - 0.5 * complete_parameters[20] * (solution[33] - 1) ^ 2) + 1),
        solution[108] - complete_parameters[12] * (solution[53] - 1),
        (((-(solution[1]) - solution[5]) - (solution[7] * solution[50] * (exp(solution[108]) - 1)) / complete_parameters[12]) - solution[20]) + solution[28],
        solution[109] - complete_parameters[12] * (solution[54] - 1),
        (((((-(solution[2]) - solution[6]) - (solution[8] * solution[51] * (exp(solution[109]) - 1)) / complete_parameters[12]) - solution[10] * solution[26]) + solution[12] * solution[25] + solution[15]) - solution[21]) + solution[29],
        (solution[36] * (-(solution[7]) * solution[50] * exp(solution[108]) + solution[7] * solution[50])) / solution[117] ^ complete_parameters[17],
        (solution[36] * (-(solution[8]) * solution[51] * exp(solution[109]) + solution[8] * solution[51])) / solution[118] ^ complete_parameters[17],
        (complete_parameters[29] * log(solution[46]) + (1 - complete_parameters[29]) * log(solution[119])) - log(solution[46]),
        -(complete_parameters[1]) + solution[57] + log(solution[18]),
        solution[110] - 1 / solution[18],
        ((solution[120] + solution[56]) - (1 - complete_parameters[23]) * (-(complete_parameters[13]) * log(solution[46]) + log(solution[46]))) + log(solution[18]) + log(solution[110]),
        solution[55] - Max(solution[56], solution[57]),
        solution[55] - solution[58],
        solution[58] - solution[93],
        solution[92] - 0,
        solution[91] - solution[92],
        solution[69] - solution[91],
        -(solution[69]) + solution[80],
        -(solution[80]) + solution[94],
        -(solution[94]) + solution[95],
        -(solution[95]) + solution[96],
        -(solution[96]) + solution[97],
        -(solution[97]) + solution[98],
        -(solution[98]) + solution[99],
        solution[84] - solution[99],
        solution[83] - solution[84],
        solution[81] - solution[83],
        -(solution[81]) + solution[82],
        -(solution[82]) + solution[85],
        -(solution[85]) + solution[86],
        -(solution[86]) + solution[87],
        -(solution[87]) + solution[88],
        -(solution[88]) + solution[89],
        -(solution[89]) + solution[90],
        solution[62] - solution[90],
        solution[61] - solution[62],
        solution[59] - solution[61],
        -(solution[59]) + solution[60],
        -(solution[60]) + solution[63],
        -(solution[63]) + solution[64],
        -(solution[64]) + solution[65],
        -(solution[65]) + solution[66],
        -(solution[66]) + solution[67],
        -(solution[67]) + solution[68],
        -(solution[68]) + solution[73],
        solution[72] - solution[73],
        solution[70] - solution[72],
        -(solution[70]) + solution[71],
        -(solution[71]) + solution[74],
        -(solution[74]) + solution[75],
        -(solution[75]) + solution[76],
        -(solution[76]) + solution[77],
        -(solution[77]) + solution[78],
        -(solution[78]) + solution[79],
        -(solution[79]) + solution[93],
        1 - solution[46],
        solution[45] - solution[46],
    ]
end

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 4
    @assert length(external_solution) == 0
    @assert length(solution) == 3
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((solution[1] + previous_solution[3]) - (1 - complete_parameters[23]) * (-(complete_parameters[13]) * log(solution[2]) + log(solution[2]))) + log(solution[3]) + log(previous_solution[4]),
        solution[2] - min(1.0e12, max(eps(), previous_solution[2])),
        solution[3] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_2(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        (complete_parameters[29] * log(solution[2]) + (1 - complete_parameters[29]) * log(solution[1])) - log(solution[2]),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 1 / previous_solution[1],
    ]
end

function residuals_block_4(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        previous_solution[1] - Max(solution[1], previous_solution[2]),
    ]
end

function residuals_block_5(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[1]) + solution[1] + log(solution[2]),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_6(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_7(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_8(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_9(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_10(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_11(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_12(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_13(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_14(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_15(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_16(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_17(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_18(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_19(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_20(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_21(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_22(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_23(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_24(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_25(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_26(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_27(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_28(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_29(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_30(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_31(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_32(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_33(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_34(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_35(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_36(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_37(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_38(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_39(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_40(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_41(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_42(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_43(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_44(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_45(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_46(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_47(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_48(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0,
    ]
end

function residuals_block_49(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 4
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[1]) * complete_parameters[7] + solution[1]) - previous_solution[3] * ((-(solution[2] ^ (complete_parameters[18] + 1)) * previous_solution[2] * complete_parameters[11]) / (complete_parameters[18] + 1) + previous_solution[4] ^ (1 - complete_parameters[17]) / (1 - complete_parameters[17])),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_50(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 4
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[1]) * complete_parameters[7] + solution[1]) - previous_solution[3] * ((-(solution[2] ^ (complete_parameters[18] + 1)) * previous_solution[2] * complete_parameters[11]) / (complete_parameters[18] + 1) + previous_solution[4] ^ (1 - complete_parameters[17]) / (1 - complete_parameters[17])),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_51(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (complete_parameters[7] * previous_solution[1]) / previous_solution[2] ^ complete_parameters[17] - previous_solution[1] / (solution[1] * previous_solution[3] ^ complete_parameters[17]),
    ]
end

function residuals_block_52(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 4
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (complete_parameters[7] * previous_solution[1]) / (previous_solution[2] * previous_solution[3] ^ complete_parameters[17]) - previous_solution[1] / (solution[1] * previous_solution[4] ^ complete_parameters[17]),
    ]
end

function residuals_block_53(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) + (previous_solution[2] * previous_solution[3] ^ complete_parameters[17]) / previous_solution[1],
    ]
end

function residuals_block_54(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) + (previous_solution[2] * previous_solution[3] ^ complete_parameters[17]) / previous_solution[1],
    ]
end

function residuals_block_55(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - (previous_solution[2] * (previous_solution[1] - previous_solution[3])) / solution[2] ^ ((complete_parameters[2] + 1) / complete_parameters[2]),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_56(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1] / previous_solution[2],
    ]
end

function residuals_block_57(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((solution[1] * complete_parameters[21]) / previous_solution[2] ^ ((complete_parameters[4] + 1) / complete_parameters[4]) - solution[1]) + (1 - complete_parameters[21]) / previous_solution[1] ^ ((complete_parameters[4] + 1) / complete_parameters[4]),
    ]
end

function residuals_block_58(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - solution[2] ^ complete_parameters[8] / previous_solution[1],
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_59(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 5
    @assert length(external_solution) == 0
    @assert length(solution) == 22
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[22] - (-(solution[1]) * complete_parameters[10] + solution[1]),
        (solution[2] * (-0.5 * complete_parameters[20] * (previous_solution[2] - 1) ^ 2 + 1) + solution[3] * (1 - complete_parameters[19])) - solution[3],
        (previous_solution[5] * (-(solution[3]) * solution[15] * exp(solution[17]) + solution[3] * solution[15])) / solution[22] ^ complete_parameters[17],
        (((((-(solution[1]) - solution[2]) - (solution[3] * solution[15] * (exp(solution[17]) - 1)) / complete_parameters[12]) - solution[4] * solution[9]) + solution[5] * solution[8] + solution[7]) - previous_solution[1]) + solution[11],
        -(solution[4]) + solution[5],
        (-(solution[6] ^ (-1 - (complete_parameters[2] + 1) / complete_parameters[2])) * solution[11] * (solution[6] - solution[13]) * (complete_parameters[2] + 1)) / complete_parameters[2] + solution[11] / solution[6] ^ ((complete_parameters[2] + 1) / complete_parameters[2]),
        -(solution[5]) * (-(solution[8]) + solution[10]) + solution[7],
        -(solution[5] ^ complete_parameters[18]) * previous_solution[3] * previous_solution[5] * complete_parameters[11] + (solution[8] * previous_solution[5]) / solution[22] ^ complete_parameters[17],
        solution[21] - solution[10] / solution[9],
        (solution[4] * solution[21] ^ (-1 + (-(complete_parameters[4]) - 1) / complete_parameters[4]) * (-(solution[8]) + solution[10]) * (-(complete_parameters[4]) - 1)) / (solution[9] * complete_parameters[4]) + solution[5],
        -(solution[11]) + solution[12],
        -(solution[12]) + solution[11] / solution[6] ^ ((complete_parameters[2] + 1) / complete_parameters[2]),
        -(solution[9]) + (previous_solution[4] * solution[13] * solution[20] ^ complete_parameters[6] * (1 - complete_parameters[6])) / solution[4] ^ complete_parameters[6],
        complete_parameters[7] * ((previous_solution[5] * (solution[15] * solution[16] - (solution[15] * (exp(solution[18]) - 1)) / complete_parameters[12])) / solution[19] ^ complete_parameters[17] + solution[14] * (1 - complete_parameters[19])) - solution[14],
        solution[4] ^ (1 - complete_parameters[6]) * complete_parameters[6] * previous_solution[4] * solution[13] * solution[20] ^ (complete_parameters[6] - 1) - solution[15],
        solution[20] - solution[3] * solution[16],
        solution[17] - complete_parameters[12] * (solution[16] - 1),
        solution[18] - complete_parameters[12] * (solution[16] - 1),
        solution[19] - (-(solution[1]) * complete_parameters[10] + solution[1]),
        (solution[4] ^ (1 - complete_parameters[6]) * previous_solution[4] * solution[20] ^ complete_parameters[6] - complete_parameters[5]) - solution[11] / solution[6] ^ ((complete_parameters[2] + 1) / complete_parameters[2]),
        solution[4] * solution[21] ^ ((-(complete_parameters[4]) - 1) / complete_parameters[4]) - solution[5],
        (complete_parameters[7] * previous_solution[2] * solution[14] * complete_parameters[20] * (previous_solution[2] - 1) - previous_solution[5] / solution[22] ^ complete_parameters[17]) + solution[14] * ((-(previous_solution[2]) * complete_parameters[20] * (previous_solution[2] - 1) - 0.5 * complete_parameters[20] * (previous_solution[2] - 1) ^ 2) + 1),
    ]
end

function residuals_block_60(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_61(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        complete_parameters[3] * previous_solution[1] - solution[1],
    ]
end

function residuals_block_62(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 10
    @assert length(external_solution) == 0
    @assert length(solution) == 22
    complete_parameters = complete_parameter_values(parameters)
    return [
        (((-(solution[1]) - solution[2]) - (solution[3] * solution[14] * (exp(solution[19]) - 1)) / complete_parameters[12]) - previous_solution[1]) + solution[6],
        (solution[2] * (-0.5 * complete_parameters[20] * (previous_solution[2] - 1) ^ 2 + 1) + solution[3] * (1 - complete_parameters[19])) - solution[3],
        solution[21] - solution[3] * solution[16],
        solution[4] ^ (1 - complete_parameters[6]) * complete_parameters[6] * previous_solution[4] * solution[12] * solution[21] ^ (complete_parameters[6] - 1) - solution[14],
        previous_solution[8] - solution[15] / solution[5],
        -(solution[6]) * previous_solution[6] + solution[7],
        (solution[4] ^ (1 - complete_parameters[6]) * previous_solution[4] * solution[21] ^ complete_parameters[6] - complete_parameters[5]) - solution[7],
        -(solution[8]) + solution[9],
        ((complete_parameters[7] * solution[9] * complete_parameters[21]) / previous_solution[9] ^ (((complete_parameters[4] + 1) * (complete_parameters[18] + 1)) / complete_parameters[4]) + previous_solution[3] * previous_solution[5] * complete_parameters[11] * solution[18] ^ (complete_parameters[18] + 1)) - solution[9],
        ((solution[6] * previous_solution[5] * previous_solution[7]) / solution[22] ^ complete_parameters[17] + (complete_parameters[7] * solution[10] * complete_parameters[22]) / previous_solution[10] ^ (1 / complete_parameters[2])) - solution[10],
        -(solution[10]) + solution[11] * (complete_parameters[2] + 1),
        ((solution[6] * previous_solution[5] * solution[12]) / solution[22] ^ complete_parameters[17] + (complete_parameters[7] * solution[11] * complete_parameters[22]) / previous_solution[10] ^ ((complete_parameters[2] + 1) / complete_parameters[2])) - solution[11],
        (complete_parameters[7] * previous_solution[2] * solution[13] * complete_parameters[20] * (previous_solution[2] - 1) - previous_solution[5] / solution[22] ^ complete_parameters[17]) + solution[13] * ((-(previous_solution[2]) * complete_parameters[20] * (previous_solution[2] - 1) - 0.5 * complete_parameters[20] * (previous_solution[2] - 1) ^ 2) + 1),
        (previous_solution[5] * (-(solution[3]) * solution[14] * exp(solution[19]) + solution[3] * solution[14])) / solution[22] ^ complete_parameters[17],
        ((solution[4] * previous_solution[5] * solution[15]) / (previous_solution[8] ^ ((complete_parameters[4] + 1) / complete_parameters[4]) * solution[22] ^ complete_parameters[17] * (complete_parameters[4] + 1)) + (complete_parameters[7] * solution[8] * complete_parameters[21]) / previous_solution[9] ^ (1 / complete_parameters[4])) - solution[8],
        solution[17] - complete_parameters[12] * (solution[16] - 1),
        complete_parameters[7] * ((previous_solution[5] * (solution[14] * solution[16] - (solution[14] * (exp(solution[17]) - 1)) / complete_parameters[12])) / solution[20] ^ complete_parameters[17] + solution[13] * (1 - complete_parameters[19])) - solution[13],
        solution[18] - solution[4] / previous_solution[8] ^ ((complete_parameters[4] + 1) / complete_parameters[4]),
        solution[19] - complete_parameters[12] * (solution[16] - 1),
        solution[20] - (-(solution[1]) * complete_parameters[10] + solution[1]),
        -(solution[5]) + (previous_solution[4] * solution[12] * solution[21] ^ complete_parameters[6] * (1 - complete_parameters[6])) / solution[4] ^ complete_parameters[6],
        solution[22] - (-(solution[1]) * complete_parameters[10] + solution[1]),
    ]
end

function residuals_block_63(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) + solution[1],
    ]
end

function residuals_block_64(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) + complete_parameters[3] * previous_solution[1],
    ]
end

function residuals_block_65(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        complete_parameters[28] * log(solution[1]) - log(solution[1]),
    ]
end

function residuals_block_66(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (complete_parameters[21] / previous_solution[1] ^ (1 / complete_parameters[4]) - 1) + (1 - complete_parameters[21]) / solution[1] ^ (1 / complete_parameters[4]),
    ]
end

function residuals_block_67(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - solution[2] ^ complete_parameters[8] / previous_solution[1],
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_68(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - solution[2] ^ complete_parameters[8] / previous_solution[1],
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_69(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        complete_parameters[25] * log(solution[1]) - log(solution[1]),
    ]
end

function residuals_block_70(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - solution[2] ^ complete_parameters[9] / previous_solution[1],
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_71(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((solution[1] * complete_parameters[22]) / previous_solution[2] ^ ((complete_parameters[2] + 1) / complete_parameters[2]) - solution[1]) + (1 - complete_parameters[22]) / solution[2] ^ ((complete_parameters[2] + 1) / complete_parameters[2]),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_72(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (complete_parameters[22] / previous_solution[1] ^ (1 / complete_parameters[2]) - 1) + (1 - complete_parameters[22]) / solution[1] ^ (1 / complete_parameters[2]),
    ]
end

function residuals_block_73(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - solution[2] ^ complete_parameters[9] / previous_solution[1],
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_74(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_75(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        1 - solution[1],
    ]
end

function residuals_block_76(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        complete_parameters[27] * log(solution[1]) - log(solution[1]),
    ]
end

function residuals_block_77(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        complete_parameters[26] * log(solution[1]) - log(solution[1]),
    ]
end

function residuals_block_78(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        complete_parameters[24] * log(solution[1]) - log(solution[1]),
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
        residuals_block_16(parameters, previous_solutions[16], external_solutions[16], solutions[16]),
        residuals_block_17(parameters, previous_solutions[17], external_solutions[17], solutions[17]),
        residuals_block_18(parameters, previous_solutions[18], external_solutions[18], solutions[18]),
        residuals_block_19(parameters, previous_solutions[19], external_solutions[19], solutions[19]),
        residuals_block_20(parameters, previous_solutions[20], external_solutions[20], solutions[20]),
        residuals_block_21(parameters, previous_solutions[21], external_solutions[21], solutions[21]),
        residuals_block_22(parameters, previous_solutions[22], external_solutions[22], solutions[22]),
        residuals_block_23(parameters, previous_solutions[23], external_solutions[23], solutions[23]),
        residuals_block_24(parameters, previous_solutions[24], external_solutions[24], solutions[24]),
        residuals_block_25(parameters, previous_solutions[25], external_solutions[25], solutions[25]),
        residuals_block_26(parameters, previous_solutions[26], external_solutions[26], solutions[26]),
        residuals_block_27(parameters, previous_solutions[27], external_solutions[27], solutions[27]),
        residuals_block_28(parameters, previous_solutions[28], external_solutions[28], solutions[28]),
        residuals_block_29(parameters, previous_solutions[29], external_solutions[29], solutions[29]),
        residuals_block_30(parameters, previous_solutions[30], external_solutions[30], solutions[30]),
        residuals_block_31(parameters, previous_solutions[31], external_solutions[31], solutions[31]),
        residuals_block_32(parameters, previous_solutions[32], external_solutions[32], solutions[32]),
        residuals_block_33(parameters, previous_solutions[33], external_solutions[33], solutions[33]),
        residuals_block_34(parameters, previous_solutions[34], external_solutions[34], solutions[34]),
        residuals_block_35(parameters, previous_solutions[35], external_solutions[35], solutions[35]),
        residuals_block_36(parameters, previous_solutions[36], external_solutions[36], solutions[36]),
        residuals_block_37(parameters, previous_solutions[37], external_solutions[37], solutions[37]),
        residuals_block_38(parameters, previous_solutions[38], external_solutions[38], solutions[38]),
        residuals_block_39(parameters, previous_solutions[39], external_solutions[39], solutions[39]),
        residuals_block_40(parameters, previous_solutions[40], external_solutions[40], solutions[40]),
        residuals_block_41(parameters, previous_solutions[41], external_solutions[41], solutions[41]),
        residuals_block_42(parameters, previous_solutions[42], external_solutions[42], solutions[42]),
        residuals_block_43(parameters, previous_solutions[43], external_solutions[43], solutions[43]),
        residuals_block_44(parameters, previous_solutions[44], external_solutions[44], solutions[44]),
        residuals_block_45(parameters, previous_solutions[45], external_solutions[45], solutions[45]),
        residuals_block_46(parameters, previous_solutions[46], external_solutions[46], solutions[46]),
        residuals_block_47(parameters, previous_solutions[47], external_solutions[47], solutions[47]),
        residuals_block_48(parameters, previous_solutions[48], external_solutions[48], solutions[48]),
        residuals_block_49(parameters, previous_solutions[49], external_solutions[49], solutions[49]),
        residuals_block_50(parameters, previous_solutions[50], external_solutions[50], solutions[50]),
        residuals_block_51(parameters, previous_solutions[51], external_solutions[51], solutions[51]),
        residuals_block_52(parameters, previous_solutions[52], external_solutions[52], solutions[52]),
        residuals_block_53(parameters, previous_solutions[53], external_solutions[53], solutions[53]),
        residuals_block_54(parameters, previous_solutions[54], external_solutions[54], solutions[54]),
        residuals_block_55(parameters, previous_solutions[55], external_solutions[55], solutions[55]),
        residuals_block_56(parameters, previous_solutions[56], external_solutions[56], solutions[56]),
        residuals_block_57(parameters, previous_solutions[57], external_solutions[57], solutions[57]),
        residuals_block_58(parameters, previous_solutions[58], external_solutions[58], solutions[58]),
        residuals_block_59(parameters, previous_solutions[59], external_solutions[59], solutions[59]),
        residuals_block_60(parameters, previous_solutions[60], external_solutions[60], solutions[60]),
        residuals_block_61(parameters, previous_solutions[61], external_solutions[61], solutions[61]),
        residuals_block_62(parameters, previous_solutions[62], external_solutions[62], solutions[62]),
        residuals_block_63(parameters, previous_solutions[63], external_solutions[63], solutions[63]),
        residuals_block_64(parameters, previous_solutions[64], external_solutions[64], solutions[64]),
        residuals_block_65(parameters, previous_solutions[65], external_solutions[65], solutions[65]),
        residuals_block_66(parameters, previous_solutions[66], external_solutions[66], solutions[66]),
        residuals_block_67(parameters, previous_solutions[67], external_solutions[67], solutions[67]),
        residuals_block_68(parameters, previous_solutions[68], external_solutions[68], solutions[68]),
        residuals_block_69(parameters, previous_solutions[69], external_solutions[69], solutions[69]),
        residuals_block_70(parameters, previous_solutions[70], external_solutions[70], solutions[70]),
        residuals_block_71(parameters, previous_solutions[71], external_solutions[71], solutions[71]),
        residuals_block_72(parameters, previous_solutions[72], external_solutions[72], solutions[72]),
        residuals_block_73(parameters, previous_solutions[73], external_solutions[73], solutions[73]),
        residuals_block_74(parameters, previous_solutions[74], external_solutions[74], solutions[74]),
        residuals_block_75(parameters, previous_solutions[75], external_solutions[75], solutions[75]),
        residuals_block_76(parameters, previous_solutions[76], external_solutions[76], solutions[76]),
        residuals_block_77(parameters, previous_solutions[77], external_solutions[77], solutions[77]),
        residuals_block_78(parameters, previous_solutions[78], external_solutions[78], solutions[78]),
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
export residuals_block_1, residuals_block_2, residuals_block_3, residuals_block_4, residuals_block_5, residuals_block_6, residuals_block_7, residuals_block_8, residuals_block_9, residuals_block_10, residuals_block_11, residuals_block_12, residuals_block_13, residuals_block_14, residuals_block_15, residuals_block_16, residuals_block_17, residuals_block_18, residuals_block_19, residuals_block_20, residuals_block_21, residuals_block_22, residuals_block_23, residuals_block_24, residuals_block_25, residuals_block_26, residuals_block_27, residuals_block_28, residuals_block_29, residuals_block_30, residuals_block_31, residuals_block_32, residuals_block_33, residuals_block_34, residuals_block_35, residuals_block_36, residuals_block_37, residuals_block_38, residuals_block_39, residuals_block_40, residuals_block_41, residuals_block_42, residuals_block_43, residuals_block_44, residuals_block_45, residuals_block_46, residuals_block_47, residuals_block_48, residuals_block_49, residuals_block_50, residuals_block_51, residuals_block_52, residuals_block_53, residuals_block_54, residuals_block_55, residuals_block_56, residuals_block_57, residuals_block_58, residuals_block_59, residuals_block_60, residuals_block_61, residuals_block_62, residuals_block_63, residuals_block_64, residuals_block_65, residuals_block_66, residuals_block_67, residuals_block_68, residuals_block_69, residuals_block_70, residuals_block_71, residuals_block_72, residuals_block_73, residuals_block_74, residuals_block_75, residuals_block_76, residuals_block_77, residuals_block_78
end
