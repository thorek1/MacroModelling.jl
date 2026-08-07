module ParserFeaturesNsssResiduals
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

const MODEL_NAME = "ParserFeatures"
const SOURCE_MODEL_FILE = "models/ParserFeatures.jl"
const NSSS_SOLUTION_ERROR = 7.194635805153547e-16
const NSSS_RESIDUAL_NORM = 1.3497887713973342e-15

const PARAMETER_NAMES = [
    "rho",
    "sigma",
    "consumption_share",
    "foreign_gap",
    "inflation_bar",
    "inflation_shock",
    "rate_floor",
    "tax_cap",
    "tax_base",
    "tax_rate",
    "ifelse_switch",
    "shock_scale",
    "trade_weight{H}{F}",
    "trade_weight{F}{H}",
    "λ{A}",
    "λ{B}",
    "χ{A}",
    "χ{B}",
    "alpha{F}",
    "output_target",
    "inflation_target",
    "delta",
    "activeᵒᵇᶜshocks",
]
const PARAMETER_VALUES = Float64[
    0.8,
    0.01,
    0.7,
    0.05,
    1.0,
    0.01,
    0.95,
    0.4,
    0.1,
    0.01,
    1.0,
    0.001,
    0.2,
    0.2,
    0.2,
    0.3,
    0.75,
    0.5,
    0.6,
    5.2,
    0.02,
    0.05,
    0.0,
]
const COMPLETE_PARAMETER_NAMES = [
    "rho",
    "sigma",
    "consumption_share",
    "foreign_gap",
    "inflation_bar",
    "inflation_shock",
    "rate_floor",
    "tax_cap",
    "tax_base",
    "tax_rate",
    "ifelse_switch",
    "shock_scale",
    "trade_weight{H}{F}",
    "trade_weight{F}{H}",
    "λ{A}",
    "λ{B}",
    "χ{A}",
    "χ{B}",
    "alpha{F}",
    "output_target",
    "inflation_target",
    "delta",
    "activeᵒᵇᶜshocks",
    "foreign_scale",
    "investment_share",
    "rate_target",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    0.8,
    0.01,
    0.7,
    0.05,
    1.0,
    0.01,
    0.95,
    0.4,
    0.1,
    0.01,
    1.0,
    0.001,
    0.2,
    0.2,
    0.2,
    0.3,
    0.75,
    0.5,
    0.6,
    5.2,
    0.02,
    0.05,
    0.0,
    0.95,
    0.15000000000000002,
    1.02,
]
const ORIGINAL_SOLUTION_NAMES = [
    "a{F}",
    "a{H}",
    "cdf_signal{F}",
    "cdf_signal{H}",
    "c{F}",
    "c{H}",
    "dnorm_signal{F}",
    "dnorm_signal{H}",
    "forward_anchor{F}",
    "forward_anchor{H}",
    "g{F}",
    "g{H}",
    "inflation_product{F}",
    "inflation_product{H}",
    "inflation{F}",
    "inflation{H}",
    "inverse_signal{F}",
    "inverse_signal{H}",
    "i{F}",
    "i{H}",
    "k{F}",
    "k{H}",
    "logpdf_signal{F}",
    "logpdf_signal{H}",
    "net_exports{F}",
    "net_exports{H}",
    "norminv_signal{F}",
    "norminv_signal{H}",
    "pdf_signal{F}",
    "pdf_signal{H}",
    "pnorm_signal{F}",
    "pnorm_signal{H}",
    "probability_signal{F}",
    "probability_signal{H}",
    "qnorm_signal{F}",
    "qnorm_signal{H}",
    "relative_output",
    "r{F}",
    "r{H}",
    "sales_window{F}",
    "sales_window{H}",
    "steady_gap{F}",
    "steady_gap{H}",
    "tax{F}",
    "tax{H}",
    "world_output",
    "y{F}",
    "y{H}",
    "Χᵒᵇᶜ⁺ꜝ³ꜝ",
    "Χᵒᵇᶜ⁺ꜝ¹ꜝ",
    "Χᵒᵇᶜ⁻ꜝ²ꜝ",
    "Χᵒᵇᶜ⁻ꜝ⁴ꜝ",
    "κ{F}{A}",
    "κ{F}{B}",
    "κ{H}{A}",
    "κ{H}{B}",
    "χᵒᵇᶜ⁺ꜝ³ꜝʳ",
    "χᵒᵇᶜ⁺ꜝ³ꜝˡ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝʳ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝˡ",
    "χᵒᵇᶜ⁻ꜝ²ꜝʳ",
    "χᵒᵇᶜ⁻ꜝ²ꜝˡ",
    "χᵒᵇᶜ⁻ꜝ⁴ꜝʳ",
    "χᵒᵇᶜ⁻ꜝ⁴ꜝˡ",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝ",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝ",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝ",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝ",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾",
    "beta",
    "alpha{H}",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    1.0,
    1.0,
    0.5,
    0.5,
    3.1995502501328206,
    3.6399999999999997,
    0.3989422804014327,
    0.3989422804014327,
    0.0,
    0.0,
    0.6856179107427472,
    0.7800000000000004,
    1.0,
    1.0,
    1.0,
    1.0,
    1.46849953349524e-32,
    -4.1843292356759943e-32,
    0.6856179107427475,
    0.7800000000000001,
    13.712358214854945,
    15.599999999999982,
    -0.9189385332046727,
    -0.9189385332046727,
    -0.12584278567633703,
    0.12584278567633703,
    1.46849953349524e-32,
    -4.1843292356759943e-32,
    0.3989422804014327,
    0.3989422804014327,
    0.5,
    0.5,
    0.5,
    0.5,
    1.46849953349524e-32,
    -4.1843292356759943e-32,
    1.0,
    1.02,
    1.02,
    13.712358214854945,
    15.600000000000001,
    0.0,
    0.0,
    0.14570786071618314,
    0.15200000000000002,
    9.770786071618316,
    4.570786071618315,
    5.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.30000000000000004,
    0.3,
    0.30000000000000004,
    0.3,
    0.0,
    -0.07000000000000006,
    0.0,
    -0.07000000000000006,
    0.0,
    0.24799999999999997,
    0.0,
    0.25429213928381683,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.9803921568627451,
    0.6001077713277388,
]
const ORIGINAL_INITIAL_SOLUTION_VALUES = Float64[
    1.0,
    1.0,
    0.5,
    0.5,
    0.0,
    3.6399999999999997,
    0.398942280401433,
    0.398942280401433,
    0.0,
    0.0,
    0.0,
    0.7800000000000002,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.7800000000000001,
    5.0e11,
    15.600000000000001,
    -0.918938533204673,
    -0.918938533204673,
    -1.04,
    1.04,
    0.0,
    0.0,
    0.398942280401433,
    0.398942280401433,
    0.5,
    0.5,
    0.5,
    0.5,
    0.0,
    0.0,
    1.0,
    0.0,
    1.02,
    0.0,
    15.600000000000001,
    0.0,
    0.0,
    0.0,
    0.0,
    5.2,
    0.0,
    5.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.1,
    0.3,
    0.1,
    0.3,
    0.0,
    0.0,
    0.0,
    -0.07000000000000006,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.9803921568627451,
    0.6001077713277385,
]
const AUXILIARY_SOLUTION_NAMES = [
    "a{F}",
    "a{H}",
    "cdf_signal{F}",
    "cdf_signal{H}",
    "c{F}",
    "c{H}",
    "dnorm_signal{F}",
    "dnorm_signal{H}",
    "forward_anchor{F}",
    "forward_anchor{H}",
    "g{F}",
    "g{H}",
    "inflation_product{F}",
    "inflation_product{H}",
    "inflation{F}",
    "inflation{H}",
    "inverse_signal{F}",
    "inverse_signal{H}",
    "i{F}",
    "i{H}",
    "k{F}",
    "k{H}",
    "logpdf_signal{F}",
    "logpdf_signal{H}",
    "net_exports{F}",
    "net_exports{H}",
    "norminv_signal{F}",
    "norminv_signal{H}",
    "pdf_signal{F}",
    "pdf_signal{H}",
    "pnorm_signal{F}",
    "pnorm_signal{H}",
    "probability_signal{F}",
    "probability_signal{H}",
    "qnorm_signal{F}",
    "qnorm_signal{H}",
    "relative_output",
    "r{F}",
    "r{H}",
    "sales_window{F}",
    "sales_window{H}",
    "steady_gap{F}",
    "steady_gap{H}",
    "tax{F}",
    "tax{H}",
    "world_output",
    "y{F}",
    "y{H}",
    "Χᵒᵇᶜ⁺ꜝ³ꜝ",
    "Χᵒᵇᶜ⁺ꜝ¹ꜝ",
    "Χᵒᵇᶜ⁻ꜝ²ꜝ",
    "Χᵒᵇᶜ⁻ꜝ⁴ꜝ",
    "κ{F}{A}",
    "κ{F}{B}",
    "κ{H}{A}",
    "κ{H}{B}",
    "χᵒᵇᶜ⁺ꜝ³ꜝʳ",
    "χᵒᵇᶜ⁺ꜝ³ꜝˡ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝʳ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝˡ",
    "χᵒᵇᶜ⁻ꜝ²ꜝʳ",
    "χᵒᵇᶜ⁻ꜝ²ꜝˡ",
    "χᵒᵇᶜ⁻ꜝ⁴ꜝʳ",
    "χᵒᵇᶜ⁻ꜝ⁴ꜝˡ",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝ",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝ",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝ",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝ",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾",
    "beta",
    "alpha{H}",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    1.0,
    1.0,
    0.5,
    0.5,
    3.1995502501328197,
    3.6399999999999997,
    0.398942280401433,
    0.398942280401433,
    0.0,
    0.0,
    0.6856179107427471,
    0.7800000000000002,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.6856179107427471,
    0.7800000000000001,
    13.71235821485494,
    15.600000000000001,
    -0.918938533204673,
    -0.918938533204673,
    -0.12584278567633725,
    0.12584278567633725,
    0.0,
    0.0,
    0.398942280401433,
    0.398942280401433,
    0.5,
    0.5,
    0.5,
    0.5,
    0.0,
    0.0,
    1.0,
    1.0199999999999998,
    1.02,
    13.712358214854941,
    15.600000000000001,
    0.0,
    0.0,
    0.14570786071618316,
    0.15200000000000002,
    9.770786071618314,
    4.570786071618314,
    5.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.1,
    0.3,
    0.1,
    0.3,
    1.6431625272369705e-16,
    -0.06999999999999967,
    0.0,
    -0.07000000000000006,
    -3.2590301201736453e-21,
    0.24799999999999997,
    -3.479470926141543e-21,
    0.25429213928381683,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.9803921568627451,
    0.6001077713277385,
]
const AUXILIARY_INITIAL_SOLUTION_VALUES = Float64[
    1.0,
    1.0,
    0.5,
    0.5,
    0.0,
    3.6399999999999997,
    0.398942280401433,
    0.398942280401433,
    0.0,
    0.0,
    0.0,
    0.7800000000000002,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.7800000000000001,
    5.0e11,
    15.600000000000001,
    -0.918938533204673,
    -0.918938533204673,
    -1.04,
    1.04,
    0.0,
    0.0,
    0.398942280401433,
    0.398942280401433,
    0.5,
    0.5,
    0.5,
    0.5,
    0.0,
    0.0,
    1.0,
    0.0,
    1.02,
    0.0,
    15.600000000000001,
    0.0,
    0.0,
    0.0,
    0.0,
    5.2,
    0.0,
    5.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.1,
    0.3,
    0.1,
    0.3,
    0.0,
    0.0,
    0.0,
    -0.07000000000000006,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.9803921568627451,
    0.6001077713277385,
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
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    5.2,
    15.600000000000001,
    -0.0,
    -0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    -0.0,
    -0.0,
    1.0,
    1.0,
    15.600000000000001,
]
const ALL_AUXILIARY_VARIABLE_INITIAL_VALUES = Float64[
    5.2,
    15.600000000000001,
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    0.0,
    1.0,
    1.0,
    15.600000000000001,
]
const DEFAULTED_NSSS_SOLUTION_NAMES = [
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾",
]
const CALIBRATION_PARAMETER_NAMES = [
    "beta",
    "alpha{H}",
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(y◖H◗ - a◖H◗ * k◖H◗ ^ alpha◖H◗),
    :(c◖H◗ - consumption_share * y◖H◗),
    :(i◖H◗ - investment_share * y◖H◗),
    :(((y◖H◗ - c◖H◗) - i◖H◗) - g◖H◗),
    :(k◖H◗ - ((1 - delta) * k◖H◗ + i◖H◗)),
    :(a◖H◗ - ((1 - rho) + rho * a◖H◗ + sigma * 0 + sigma * 0 + sigma * 0 + sigma * 0)),
    :(χᵒᵇᶜ⁺ꜝ¹ꜝˡ - (rate_floor - r◖H◗)),
    :(χᵒᵇᶜ⁺ꜝ¹ꜝʳ - (c◖H◗ / (beta * c◖H◗) - r◖H◗)),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - max(χᵒᵇᶜ⁺ꜝ¹ꜝˡ, χᵒᵇᶜ⁺ꜝ¹ꜝʳ)),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝ),
    :(χᵒᵇᶜ⁻ꜝ²ꜝˡ - (tax_cap - tax◖H◗)),
    :(χᵒᵇᶜ⁻ꜝ²ꜝʳ - ((tax_base + tax_rate * y◖H◗) - tax◖H◗)),
    :(Χᵒᵇᶜ⁻ꜝ²ꜝ - min(χᵒᵇᶜ⁻ꜝ²ꜝˡ, χᵒᵇᶜ⁻ꜝ²ꜝʳ)),
    :(Χᵒᵇᶜ⁻ꜝ²ꜝ - ϵᵒᵇᶜ⁻ꜝ²ꜝ),
    :(sales_window◖H◗ - (y◖H◗ + y◖H◗ + y◖H◗)),
    :(forward_anchor◖H◗ - (y◖H◗ - y◖H◗)),
    :(inflation_product◖H◗ - inflation◖H◗ * inflation◖H◗ * inflation◖H◗),
    :(inflation◖H◗ - (inflation_bar + inflation_shock * 0)),
    :(cdf_signal◖H◗ - normcdf(a◖H◗ - 1)),
    :(pdf_signal◖H◗ - normpdf(a◖H◗ - 1)),
    :(logpdf_signal◖H◗ - normlogpdf(a◖H◗ - 1)),
    :(probability_signal◖H◗ - (0.5 + 0.25 * tanh(a◖H◗ - 1))),
    :(inverse_signal◖H◗ - norminvcdf(probability_signal◖H◗)),
    :(norminv_signal◖H◗ - norminv(probability_signal◖H◗)),
    :(qnorm_signal◖H◗ - qnorm(probability_signal◖H◗)),
    :(pnorm_signal◖H◗ - pnorm(a◖H◗ - 1)),
    :(dnorm_signal◖H◗ - dnorm(a◖H◗ - 1)),
    :(steady_gap◖H◗ - (y◖H◗ - y◖H◗)),
    :(κ◖H◗◖A◗ - (0.25κ◖H◗◖A◗ + 0.25κ◖H◗◖A◗ + λ◖A◗ * ifelse(ifelse_switch > 0.5, χ◖A◗, 1 - χ◖A◗) + shock_scale * 0 + shock_scale * 0)),
    :(κ◖H◗◖B◗ - (0.25κ◖H◗◖B◗ + 0.25κ◖H◗◖B◗ + λ◖B◗ * ifelse(ifelse_switch > 0.5, χ◖B◗, 1 - χ◖B◗) + shock_scale * 0 + shock_scale * 0)),
    :(y◖F◗ - foreign_scale * a◖F◗ * k◖F◗ ^ alpha◖F◗),
    :(c◖F◗ - consumption_share * y◖F◗),
    :(i◖F◗ - investment_share * y◖F◗),
    :(((y◖F◗ - c◖F◗) - i◖F◗) - g◖F◗),
    :(k◖F◗ - ((1 - delta) * k◖F◗ + i◖F◗)),
    :(a◖F◗ - ((1 - rho) + rho * a◖F◗ + sigma * 0 + sigma * 0 + sigma * 0 + sigma * 0)),
    :(χᵒᵇᶜ⁺ꜝ³ꜝˡ - (rate_floor - r◖F◗)),
    :(χᵒᵇᶜ⁺ꜝ³ꜝʳ - (c◖F◗ / (beta * c◖F◗) - r◖F◗)),
    :(Χᵒᵇᶜ⁺ꜝ³ꜝ - max(χᵒᵇᶜ⁺ꜝ³ꜝˡ, χᵒᵇᶜ⁺ꜝ³ꜝʳ)),
    :(Χᵒᵇᶜ⁺ꜝ³ꜝ - ϵᵒᵇᶜ⁺ꜝ³ꜝ),
    :(χᵒᵇᶜ⁻ꜝ⁴ꜝˡ - (tax_cap - tax◖F◗)),
    :(χᵒᵇᶜ⁻ꜝ⁴ꜝʳ - ((tax_base + tax_rate * y◖F◗) - tax◖F◗)),
    :(Χᵒᵇᶜ⁻ꜝ⁴ꜝ - min(χᵒᵇᶜ⁻ꜝ⁴ꜝˡ, χᵒᵇᶜ⁻ꜝ⁴ꜝʳ)),
    :(Χᵒᵇᶜ⁻ꜝ⁴ꜝ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝ),
    :(sales_window◖F◗ - (y◖F◗ + y◖F◗ + y◖F◗)),
    :(forward_anchor◖F◗ - (y◖F◗ - y◖F◗)),
    :(inflation_product◖F◗ - inflation◖F◗ * inflation◖F◗ * inflation◖F◗),
    :(inflation◖F◗ - (inflation_bar + inflation_shock * 0)),
    :(cdf_signal◖F◗ - normcdf(a◖F◗ - 1)),
    :(pdf_signal◖F◗ - normpdf(a◖F◗ - 1)),
    :(logpdf_signal◖F◗ - normlogpdf(a◖F◗ - 1)),
    :(probability_signal◖F◗ - (0.5 + 0.25 * tanh(a◖F◗ - 1))),
    :(inverse_signal◖F◗ - norminvcdf(probability_signal◖F◗)),
    :(norminv_signal◖F◗ - norminv(probability_signal◖F◗)),
    :(qnorm_signal◖F◗ - qnorm(probability_signal◖F◗)),
    :(pnorm_signal◖F◗ - pnorm(a◖F◗ - 1)),
    :(dnorm_signal◖F◗ - dnorm(a◖F◗ - 1)),
    :(steady_gap◖F◗ - (y◖F◗ - y◖F◗)),
    :(κ◖F◗◖A◗ - (0.25κ◖F◗◖A◗ + 0.25κ◖F◗◖A◗ + λ◖A◗ * ifelse(ifelse_switch > 0.5, χ◖A◗, 1 - χ◖A◗) + shock_scale * 0 + shock_scale * 0)),
    :(κ◖F◗◖B◗ - (0.25κ◖F◗◖B◗ + 0.25κ◖F◗◖B◗ + λ◖B◗ * ifelse(ifelse_switch > 0.5, χ◖B◗, 1 - χ◖B◗) + shock_scale * 0 + shock_scale * 0)),
    :(world_output - (y◖H◗ + y◖F◗)),
    :(relative_output - (y◖H◗ / y◖F◗) / (y◖H◗ / y◖F◗)),
    :(net_exports◖H◗ - trade_weight◖H◗◖F◗ * (y◖H◗ - y◖F◗)),
    :(net_exports◖F◗ - trade_weight◖F◗◖H◗ * (y◖F◗ - y◖H◗)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾ - activeᵒᵇᶜshocks * 0),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝ - ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾ - activeᵒᵇᶜshocks * 0),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾ - (ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾ - (ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾ - (ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾ - (ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝ - ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾ - activeᵒᵇᶜshocks * 0),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾ - (ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾ - (ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾ - (ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾ - (ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾ - activeᵒᵇᶜshocks * 0),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾ - (ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾ - (ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾ - (ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾ - (ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾ + activeᵒᵇᶜshocks * 0)),
]
const CALIBRATION_EQUATIONS = Expr[
    :(r◖H◗ - rate_target),
    :(y◖H◗ - output_target),
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :(-a◖H◗ * k◖H◗ ^ alpha◖H◗ + y◖H◗),
    :(-consumption_share * y◖H◗ + c◖H◗),
    :(-investment_share * y◖H◗ + i◖H◗),
    :(((-c◖H◗ - g◖H◗) - i◖H◗) + y◖H◗),
    :((-i◖H◗ - k◖H◗ * (1 - delta)) + k◖H◗),
    :((-a◖H◗ * rho + a◖H◗ + rho) - 1),
    :(-rate_floor + r◖H◗ + χᵒᵇᶜ⁺ꜝ¹ꜝˡ),
    :((r◖H◗ + χᵒᵇᶜ⁺ꜝ¹ꜝʳ) - 1 / beta),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - Max(χᵒᵇᶜ⁺ꜝ¹ꜝʳ, χᵒᵇᶜ⁺ꜝ¹ꜝˡ)),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝ),
    :(-tax_cap + tax◖H◗ + χᵒᵇᶜ⁻ꜝ²ꜝˡ),
    :((-tax_base - tax_rate * y◖H◗) + tax◖H◗ + χᵒᵇᶜ⁻ꜝ²ꜝʳ),
    :(Χᵒᵇᶜ⁻ꜝ²ꜝ - Min(χᵒᵇᶜ⁻ꜝ²ꜝʳ, χᵒᵇᶜ⁻ꜝ²ꜝˡ)),
    :(Χᵒᵇᶜ⁻ꜝ²ꜝ - ϵᵒᵇᶜ⁻ꜝ²ꜝ),
    :(sales_window◖H◗ - 3y◖H◗),
    :(forward_anchor◖H◗ - 0),
    :(inflation_product◖H◗ - inflation◖H◗ ^ 3),
    :(-inflation_bar + inflation◖H◗),
    :((cdf_signal◖H◗ + erfc(0.707106781186547a◖H◗ - 0.707106781186547) / 2) - 1),
    :(pdf_signal◖H◗ - 0.398942280401433 * exp(-((a◖H◗ - 1) ^ 2) / 2)),
    :(logpdf_signal◖H◗ + (a◖H◗ - 1) ^ 2 / 2 + 0.918938533204673),
    :((probability_signal◖H◗ - 0.25 * tanh(a◖H◗ - 1)) - 0.5),
    :(inverse_signal◖H◗ + 1.4142135623731 * erfcinv(2probability_signal◖H◗)),
    :(norminv_signal◖H◗ + 1.4142135623731 * erfcinv(2probability_signal◖H◗)),
    :(qnorm_signal◖H◗ + 1.4142135623731 * erfcinv(2probability_signal◖H◗)),
    :((pnorm_signal◖H◗ + erfc(0.707106781186547a◖H◗ - 0.707106781186547) / 2) - 1),
    :(dnorm_signal◖H◗ - 0.398942280401433 * exp(-((a◖H◗ - 1) ^ 2) / 2)),
    :(steady_gap◖H◗ - 0),
    :(0.5κ◖H◗◖A◗ - λ◖A◗ * (1 - χ◖A◗)),
    :(0.5κ◖H◗◖B◗ - λ◖B◗ * (1 - χ◖B◗)),
    :(-a◖F◗ * foreign_scale * k◖F◗ ^ alpha◖F◗ + y◖F◗),
    :(-consumption_share * y◖F◗ + c◖F◗),
    :(-investment_share * y◖F◗ + i◖F◗),
    :(((-c◖F◗ - g◖F◗) - i◖F◗) + y◖F◗),
    :((-i◖F◗ - k◖F◗ * (1 - delta)) + k◖F◗),
    :((-a◖F◗ * rho + a◖F◗ + rho) - 1),
    :(-rate_floor + r◖F◗ + χᵒᵇᶜ⁺ꜝ³ꜝˡ),
    :((r◖F◗ + χᵒᵇᶜ⁺ꜝ³ꜝʳ) - 1 / beta),
    :(Χᵒᵇᶜ⁺ꜝ³ꜝ - Max(χᵒᵇᶜ⁺ꜝ³ꜝʳ, χᵒᵇᶜ⁺ꜝ³ꜝˡ)),
    :(Χᵒᵇᶜ⁺ꜝ³ꜝ - ϵᵒᵇᶜ⁺ꜝ³ꜝ),
    :(-tax_cap + tax◖F◗ + χᵒᵇᶜ⁻ꜝ⁴ꜝˡ),
    :((-tax_base - tax_rate * y◖F◗) + tax◖F◗ + χᵒᵇᶜ⁻ꜝ⁴ꜝʳ),
    :(Χᵒᵇᶜ⁻ꜝ⁴ꜝ - Min(χᵒᵇᶜ⁻ꜝ⁴ꜝʳ, χᵒᵇᶜ⁻ꜝ⁴ꜝˡ)),
    :(Χᵒᵇᶜ⁻ꜝ⁴ꜝ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝ),
    :(sales_window◖F◗ - 3y◖F◗),
    :(forward_anchor◖F◗ - 0),
    :(inflation_product◖F◗ - inflation◖F◗ ^ 3),
    :(-inflation_bar + inflation◖F◗),
    :((cdf_signal◖F◗ + erfc(0.707106781186547a◖F◗ - 0.707106781186547) / 2) - 1),
    :(pdf_signal◖F◗ - 0.398942280401433 * exp(-((a◖F◗ - 1) ^ 2) / 2)),
    :(logpdf_signal◖F◗ + (a◖F◗ - 1) ^ 2 / 2 + 0.918938533204673),
    :((probability_signal◖F◗ - 0.25 * tanh(a◖F◗ - 1)) - 0.5),
    :(inverse_signal◖F◗ + 1.4142135623731 * erfcinv(2probability_signal◖F◗)),
    :(norminv_signal◖F◗ + 1.4142135623731 * erfcinv(2probability_signal◖F◗)),
    :(qnorm_signal◖F◗ + 1.4142135623731 * erfcinv(2probability_signal◖F◗)),
    :((pnorm_signal◖F◗ + erfc(0.707106781186547a◖F◗ - 0.707106781186547) / 2) - 1),
    :(dnorm_signal◖F◗ - 0.398942280401433 * exp(-((a◖F◗ - 1) ^ 2) / 2)),
    :(steady_gap◖F◗ - 0),
    :(0.5κ◖F◗◖A◗ - λ◖A◗ * (1 - χ◖A◗)),
    :(0.5κ◖F◗◖B◗ - λ◖B◗ * (1 - χ◖B◗)),
    :((world_output - y◖F◗) - y◖H◗),
    :(relative_output - 1),
    :(net_exports◖H◗ - trade_weight◖H◗◖F◗ * (-y◖F◗ + y◖H◗)),
    :(net_exports◖F◗ - trade_weight◖F◗◖H◗ * (y◖F◗ - y◖H◗)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾ - 0),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝ - ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾ - 0),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾ - ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾ - ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾),
    :(-ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾ + ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾),
    :(-ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾ + ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝ - ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾ - 0),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾ - ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾ - ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾ + ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾ + ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾ - 0),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾),
    :(-ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾ + ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾),
    :(-ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾ + ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(y◖H◗ - a◖H◗ * k◖H◗ ^ alpha◖H◗),
    :(c◖H◗ - consumption_share * y◖H◗),
    :(i◖H◗ - investment_share * y◖H◗),
    :(((y◖H◗ - c◖H◗) - i◖H◗) - g◖H◗),
    :(k◖H◗ - ((1 - delta) * k◖H◗ + i◖H◗)),
    :(a◖H◗ - ((1 - rho) + rho * a◖H◗ + sigma * 0 + sigma * 0 + sigma * 0 + sigma * 0)),
    :(χᵒᵇᶜ⁺ꜝ¹ꜝˡ - (rate_floor - r◖H◗)),
    :(χᵒᵇᶜ⁺ꜝ¹ꜝʳ - (c◖H◗ / (beta * c◖H◗) - r◖H◗)),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - max(χᵒᵇᶜ⁺ꜝ¹ꜝˡ, χᵒᵇᶜ⁺ꜝ¹ꜝʳ)),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝ),
    :(χᵒᵇᶜ⁻ꜝ²ꜝˡ - (tax_cap - tax◖H◗)),
    :(χᵒᵇᶜ⁻ꜝ²ꜝʳ - ((tax_base + tax_rate * y◖H◗) - tax◖H◗)),
    :(Χᵒᵇᶜ⁻ꜝ²ꜝ - min(χᵒᵇᶜ⁻ꜝ²ꜝˡ, χᵒᵇᶜ⁻ꜝ²ꜝʳ)),
    :(Χᵒᵇᶜ⁻ꜝ²ꜝ - ϵᵒᵇᶜ⁻ꜝ²ꜝ),
    :(sales_window◖H◗ - (y◖H◗ + y◖H◗ + y◖H◗)),
    :(forward_anchor◖H◗ - (y◖H◗ - y◖H◗)),
    :(inflation_product◖H◗ - inflation◖H◗ * inflation◖H◗ * inflation◖H◗),
    :(inflation◖H◗ - (inflation_bar + inflation_shock * 0)),
    :(cdf_signal◖H◗ - normcdf(a◖H◗ - 1)),
    :(pdf_signal◖H◗ - normpdf(a◖H◗ - 1)),
    :(logpdf_signal◖H◗ - normlogpdf(a◖H◗ - 1)),
    :(probability_signal◖H◗ - (0.5 + 0.25 * tanh(a◖H◗ - 1))),
    :(inverse_signal◖H◗ - norminvcdf(probability_signal◖H◗)),
    :(norminv_signal◖H◗ - norminv(probability_signal◖H◗)),
    :(qnorm_signal◖H◗ - qnorm(probability_signal◖H◗)),
    :(pnorm_signal◖H◗ - pnorm(a◖H◗ - 1)),
    :(dnorm_signal◖H◗ - dnorm(a◖H◗ - 1)),
    :(steady_gap◖H◗ - (y◖H◗ - y◖H◗)),
    :(κ◖H◗◖A◗ - (0.25κ◖H◗◖A◗ + 0.25κ◖H◗◖A◗ + λ◖A◗ * ifelse(ifelse_switch > 0.5, χ◖A◗, 1 - χ◖A◗) + shock_scale * 0 + shock_scale * 0)),
    :(κ◖H◗◖B◗ - (0.25κ◖H◗◖B◗ + 0.25κ◖H◗◖B◗ + λ◖B◗ * ifelse(ifelse_switch > 0.5, χ◖B◗, 1 - χ◖B◗) + shock_scale * 0 + shock_scale * 0)),
    :(y◖F◗ - foreign_scale * a◖F◗ * k◖F◗ ^ alpha◖F◗),
    :(c◖F◗ - consumption_share * y◖F◗),
    :(i◖F◗ - investment_share * y◖F◗),
    :(((y◖F◗ - c◖F◗) - i◖F◗) - g◖F◗),
    :(k◖F◗ - ((1 - delta) * k◖F◗ + i◖F◗)),
    :(a◖F◗ - ((1 - rho) + rho * a◖F◗ + sigma * 0 + sigma * 0 + sigma * 0 + sigma * 0)),
    :(χᵒᵇᶜ⁺ꜝ³ꜝˡ - (rate_floor - r◖F◗)),
    :(χᵒᵇᶜ⁺ꜝ³ꜝʳ - (c◖F◗ / (beta * c◖F◗) - r◖F◗)),
    :(Χᵒᵇᶜ⁺ꜝ³ꜝ - max(χᵒᵇᶜ⁺ꜝ³ꜝˡ, χᵒᵇᶜ⁺ꜝ³ꜝʳ)),
    :(Χᵒᵇᶜ⁺ꜝ³ꜝ - ϵᵒᵇᶜ⁺ꜝ³ꜝ),
    :(χᵒᵇᶜ⁻ꜝ⁴ꜝˡ - (tax_cap - tax◖F◗)),
    :(χᵒᵇᶜ⁻ꜝ⁴ꜝʳ - ((tax_base + tax_rate * y◖F◗) - tax◖F◗)),
    :(Χᵒᵇᶜ⁻ꜝ⁴ꜝ - min(χᵒᵇᶜ⁻ꜝ⁴ꜝˡ, χᵒᵇᶜ⁻ꜝ⁴ꜝʳ)),
    :(Χᵒᵇᶜ⁻ꜝ⁴ꜝ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝ),
    :(sales_window◖F◗ - (y◖F◗ + y◖F◗ + y◖F◗)),
    :(forward_anchor◖F◗ - (y◖F◗ - y◖F◗)),
    :(inflation_product◖F◗ - inflation◖F◗ * inflation◖F◗ * inflation◖F◗),
    :(inflation◖F◗ - (inflation_bar + inflation_shock * 0)),
    :(cdf_signal◖F◗ - normcdf(a◖F◗ - 1)),
    :(pdf_signal◖F◗ - normpdf(a◖F◗ - 1)),
    :(logpdf_signal◖F◗ - normlogpdf(a◖F◗ - 1)),
    :(probability_signal◖F◗ - (0.5 + 0.25 * tanh(a◖F◗ - 1))),
    :(inverse_signal◖F◗ - norminvcdf(probability_signal◖F◗)),
    :(norminv_signal◖F◗ - norminv(probability_signal◖F◗)),
    :(qnorm_signal◖F◗ - qnorm(probability_signal◖F◗)),
    :(pnorm_signal◖F◗ - pnorm(a◖F◗ - 1)),
    :(dnorm_signal◖F◗ - dnorm(a◖F◗ - 1)),
    :(steady_gap◖F◗ - (y◖F◗ - y◖F◗)),
    :(κ◖F◗◖A◗ - (0.25κ◖F◗◖A◗ + 0.25κ◖F◗◖A◗ + λ◖A◗ * ifelse(ifelse_switch > 0.5, χ◖A◗, 1 - χ◖A◗) + shock_scale * 0 + shock_scale * 0)),
    :(κ◖F◗◖B◗ - (0.25κ◖F◗◖B◗ + 0.25κ◖F◗◖B◗ + λ◖B◗ * ifelse(ifelse_switch > 0.5, χ◖B◗, 1 - χ◖B◗) + shock_scale * 0 + shock_scale * 0)),
    :(world_output - (y◖H◗ + y◖F◗)),
    :(relative_output - (y◖H◗ / y◖F◗) / (y◖H◗ / y◖F◗)),
    :(net_exports◖H◗ - trade_weight◖H◗◖F◗ * (y◖H◗ - y◖F◗)),
    :(net_exports◖F◗ - trade_weight◖F◗◖H◗ * (y◖F◗ - y◖H◗)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾ - activeᵒᵇᶜshocks * 0),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝ - ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾ - activeᵒᵇᶜshocks * 0),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾ - (ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾ - (ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾ - (ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾ - (ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝ - ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾ - activeᵒᵇᶜshocks * 0),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾ - (ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾ - (ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾ - (ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾ - (ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾ - activeᵒᵇᶜshocks * 0),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾ - (ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾ - (ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾ - (ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾ + activeᵒᵇᶜshocks * 0)),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾ - (ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾ + activeᵒᵇᶜshocks * 0)),
    :(r◖H◗ - rate_target),
    :(y◖H◗ - output_target),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :(-a◖H◗ * k◖H◗ ^ alpha◖H◗ + y◖H◗),
    :(-consumption_share * y◖H◗ + c◖H◗),
    :(-investment_share * y◖H◗ + i◖H◗),
    :(((-c◖H◗ - g◖H◗) - i◖H◗) + y◖H◗),
    :((-i◖H◗ - k◖H◗ * (1 - delta)) + k◖H◗),
    :((-a◖H◗ * rho + a◖H◗ + rho) - 1),
    :(-rate_floor + r◖H◗ + χᵒᵇᶜ⁺ꜝ¹ꜝˡ),
    :((r◖H◗ + χᵒᵇᶜ⁺ꜝ¹ꜝʳ) - 1 / beta),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - Max(χᵒᵇᶜ⁺ꜝ¹ꜝʳ, χᵒᵇᶜ⁺ꜝ¹ꜝˡ)),
    :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝ),
    :(-tax_cap + tax◖H◗ + χᵒᵇᶜ⁻ꜝ²ꜝˡ),
    :((-tax_base - tax_rate * y◖H◗) + tax◖H◗ + χᵒᵇᶜ⁻ꜝ²ꜝʳ),
    :(Χᵒᵇᶜ⁻ꜝ²ꜝ - Min(χᵒᵇᶜ⁻ꜝ²ꜝʳ, χᵒᵇᶜ⁻ꜝ²ꜝˡ)),
    :(Χᵒᵇᶜ⁻ꜝ²ꜝ - ϵᵒᵇᶜ⁻ꜝ²ꜝ),
    :(sales_window◖H◗ - 3y◖H◗),
    :(forward_anchor◖H◗ - 0),
    :(inflation_product◖H◗ - inflation◖H◗ ^ 3),
    :(-inflation_bar + inflation◖H◗),
    :((cdf_signal◖H◗ + erfc(0.707106781186547a◖H◗ - 0.707106781186547) / 2) - 1),
    :(pdf_signal◖H◗ - 0.398942280401433 * exp(-((a◖H◗ - 1) ^ 2) / 2)),
    :(logpdf_signal◖H◗ + (a◖H◗ - 1) ^ 2 / 2 + 0.918938533204673),
    :((probability_signal◖H◗ - 0.25 * tanh(a◖H◗ - 1)) - 0.5),
    :(inverse_signal◖H◗ + 1.4142135623731 * erfcinv(2probability_signal◖H◗)),
    :(norminv_signal◖H◗ + 1.4142135623731 * erfcinv(2probability_signal◖H◗)),
    :(qnorm_signal◖H◗ + 1.4142135623731 * erfcinv(2probability_signal◖H◗)),
    :((pnorm_signal◖H◗ + erfc(0.707106781186547a◖H◗ - 0.707106781186547) / 2) - 1),
    :(dnorm_signal◖H◗ - 0.398942280401433 * exp(-((a◖H◗ - 1) ^ 2) / 2)),
    :(steady_gap◖H◗ - 0),
    :(0.5κ◖H◗◖A◗ - λ◖A◗ * (1 - χ◖A◗)),
    :(0.5κ◖H◗◖B◗ - λ◖B◗ * (1 - χ◖B◗)),
    :(-a◖F◗ * foreign_scale * k◖F◗ ^ alpha◖F◗ + y◖F◗),
    :(-consumption_share * y◖F◗ + c◖F◗),
    :(-investment_share * y◖F◗ + i◖F◗),
    :(((-c◖F◗ - g◖F◗) - i◖F◗) + y◖F◗),
    :((-i◖F◗ - k◖F◗ * (1 - delta)) + k◖F◗),
    :((-a◖F◗ * rho + a◖F◗ + rho) - 1),
    :(-rate_floor + r◖F◗ + χᵒᵇᶜ⁺ꜝ³ꜝˡ),
    :((r◖F◗ + χᵒᵇᶜ⁺ꜝ³ꜝʳ) - 1 / beta),
    :(Χᵒᵇᶜ⁺ꜝ³ꜝ - Max(χᵒᵇᶜ⁺ꜝ³ꜝʳ, χᵒᵇᶜ⁺ꜝ³ꜝˡ)),
    :(Χᵒᵇᶜ⁺ꜝ³ꜝ - ϵᵒᵇᶜ⁺ꜝ³ꜝ),
    :(-tax_cap + tax◖F◗ + χᵒᵇᶜ⁻ꜝ⁴ꜝˡ),
    :((-tax_base - tax_rate * y◖F◗) + tax◖F◗ + χᵒᵇᶜ⁻ꜝ⁴ꜝʳ),
    :(Χᵒᵇᶜ⁻ꜝ⁴ꜝ - Min(χᵒᵇᶜ⁻ꜝ⁴ꜝʳ, χᵒᵇᶜ⁻ꜝ⁴ꜝˡ)),
    :(Χᵒᵇᶜ⁻ꜝ⁴ꜝ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝ),
    :(sales_window◖F◗ - 3y◖F◗),
    :(forward_anchor◖F◗ - 0),
    :(inflation_product◖F◗ - inflation◖F◗ ^ 3),
    :(-inflation_bar + inflation◖F◗),
    :((cdf_signal◖F◗ + erfc(0.707106781186547a◖F◗ - 0.707106781186547) / 2) - 1),
    :(pdf_signal◖F◗ - 0.398942280401433 * exp(-((a◖F◗ - 1) ^ 2) / 2)),
    :(logpdf_signal◖F◗ + (a◖F◗ - 1) ^ 2 / 2 + 0.918938533204673),
    :((probability_signal◖F◗ - 0.25 * tanh(a◖F◗ - 1)) - 0.5),
    :(inverse_signal◖F◗ + 1.4142135623731 * erfcinv(2probability_signal◖F◗)),
    :(norminv_signal◖F◗ + 1.4142135623731 * erfcinv(2probability_signal◖F◗)),
    :(qnorm_signal◖F◗ + 1.4142135623731 * erfcinv(2probability_signal◖F◗)),
    :((pnorm_signal◖F◗ + erfc(0.707106781186547a◖F◗ - 0.707106781186547) / 2) - 1),
    :(dnorm_signal◖F◗ - 0.398942280401433 * exp(-((a◖F◗ - 1) ^ 2) / 2)),
    :(steady_gap◖F◗ - 0),
    :(0.5κ◖F◗◖A◗ - λ◖A◗ * (1 - χ◖A◗)),
    :(0.5κ◖F◗◖B◗ - λ◖B◗ * (1 - χ◖B◗)),
    :((world_output - y◖F◗) - y◖H◗),
    :(relative_output - 1),
    :(net_exports◖H◗ - trade_weight◖H◗◖F◗ * (-y◖F◗ + y◖H◗)),
    :(net_exports◖F◗ - trade_weight◖F◗◖H◗ * (y◖F◗ - y◖H◗)),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾ - 0),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾ + ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝ - ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾ - 0),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾ - ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾),
    :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾ - ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾),
    :(-ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾ + ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾),
    :(-ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾ + ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝ - ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾ - 0),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾ - ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾),
    :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾ - ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾ + ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾),
    :(-ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾ + ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾ - 0),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾),
    :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾),
    :(-ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾ + ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾),
    :(-ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾ + ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾),
    :(r◖H◗ - rate_target),
    :(y◖H◗ - output_target),
]

const PARAMETER_DEFINITION_NAMES = [
    "foreign_scale",
    "investment_share",
    "rate_target",
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
    "1 - foreign_gap",
    "delta * 3",
    "1 + inflation_target",
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "rho",
    "sigma",
    "consumption_share",
    "foreign_gap",
    "inflation_bar",
    "inflation_shock",
    "rate_floor",
    "tax_cap",
    "tax_base",
    "tax_rate",
    "ifelse_switch",
    "shock_scale",
    "trade_weight{H}{F}",
    "trade_weight{F}{H}",
    "λ{A}",
    "λ{B}",
    "χ{A}",
    "χ{B}",
    "alpha{F}",
    "output_target",
    "inflation_target",
    "delta",
    "activeᵒᵇᶜshocks",
]
const PARAMETER_BOX_LOWER_BOUNDS = Float64[
    1.1920928955078125e-7,
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
    1.1920928955078125e-7,
    -Inf,
    -Inf,
    1.1920928955078125e-7,
    -Inf,
]
const PARAMETER_BOX_UPPER_BOUNDS = Float64[
    0.9999998807907104,
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
    0.9999998807907104,
    Inf,
    Inf,
    0.9999998807907104,
    Inf,
]
const ORIGINAL_BOX_CONSTRAINT_NAMES = [
    "a{F}",
    "a{H}",
    "cdf_signal{F}",
    "cdf_signal{H}",
    "c{F}",
    "c{H}",
    "dnorm_signal{F}",
    "dnorm_signal{H}",
    "forward_anchor{F}",
    "forward_anchor{H}",
    "g{F}",
    "g{H}",
    "inflation_product{F}",
    "inflation_product{H}",
    "inflation{F}",
    "inflation{H}",
    "inverse_signal{F}",
    "inverse_signal{H}",
    "i{F}",
    "i{H}",
    "k{F}",
    "k{H}",
    "logpdf_signal{F}",
    "logpdf_signal{H}",
    "net_exports{F}",
    "net_exports{H}",
    "norminv_signal{F}",
    "norminv_signal{H}",
    "pdf_signal{F}",
    "pdf_signal{H}",
    "pnorm_signal{F}",
    "pnorm_signal{H}",
    "probability_signal{F}",
    "probability_signal{H}",
    "qnorm_signal{F}",
    "qnorm_signal{H}",
    "relative_output",
    "r{F}",
    "r{H}",
    "sales_window{F}",
    "sales_window{H}",
    "steady_gap{F}",
    "steady_gap{H}",
    "tax{F}",
    "tax{H}",
    "world_output",
    "y{F}",
    "y{H}",
    "Χᵒᵇᶜ⁺ꜝ³ꜝ",
    "Χᵒᵇᶜ⁺ꜝ¹ꜝ",
    "Χᵒᵇᶜ⁻ꜝ²ꜝ",
    "Χᵒᵇᶜ⁻ꜝ⁴ꜝ",
    "κ{F}{A}",
    "κ{F}{B}",
    "κ{H}{A}",
    "κ{H}{B}",
    "χᵒᵇᶜ⁺ꜝ³ꜝʳ",
    "χᵒᵇᶜ⁺ꜝ³ꜝˡ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝʳ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝˡ",
    "χᵒᵇᶜ⁻ꜝ²ꜝʳ",
    "χᵒᵇᶜ⁻ꜝ²ꜝˡ",
    "χᵒᵇᶜ⁻ꜝ⁴ꜝʳ",
    "χᵒᵇᶜ⁻ꜝ⁴ꜝˡ",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝ",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝ",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝ",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝ",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾",
    "beta",
    "alpha{H}",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
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
    -1.0e12,
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
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
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
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.1920928955078125e-7,
    1.1920928955078125e-7,
]
const ORIGINAL_BOX_UPPER_BOUNDS = Float64[
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
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
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
    1.0e12,
    Inf,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.9999998807907104,
    0.9999998807907104,
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "a{F}",
    "a{H}",
    "cdf_signal{F}",
    "cdf_signal{H}",
    "c{F}",
    "c{H}",
    "dnorm_signal{F}",
    "dnorm_signal{H}",
    "forward_anchor{F}",
    "forward_anchor{H}",
    "g{F}",
    "g{H}",
    "inflation_product{F}",
    "inflation_product{H}",
    "inflation{F}",
    "inflation{H}",
    "inverse_signal{F}",
    "inverse_signal{H}",
    "i{F}",
    "i{H}",
    "k{F}",
    "k{H}",
    "logpdf_signal{F}",
    "logpdf_signal{H}",
    "net_exports{F}",
    "net_exports{H}",
    "norminv_signal{F}",
    "norminv_signal{H}",
    "pdf_signal{F}",
    "pdf_signal{H}",
    "pnorm_signal{F}",
    "pnorm_signal{H}",
    "probability_signal{F}",
    "probability_signal{H}",
    "qnorm_signal{F}",
    "qnorm_signal{H}",
    "relative_output",
    "r{F}",
    "r{H}",
    "sales_window{F}",
    "sales_window{H}",
    "steady_gap{F}",
    "steady_gap{H}",
    "tax{F}",
    "tax{H}",
    "world_output",
    "y{F}",
    "y{H}",
    "Χᵒᵇᶜ⁺ꜝ³ꜝ",
    "Χᵒᵇᶜ⁺ꜝ¹ꜝ",
    "Χᵒᵇᶜ⁻ꜝ²ꜝ",
    "Χᵒᵇᶜ⁻ꜝ⁴ꜝ",
    "κ{F}{A}",
    "κ{F}{B}",
    "κ{H}{A}",
    "κ{H}{B}",
    "χᵒᵇᶜ⁺ꜝ³ꜝʳ",
    "χᵒᵇᶜ⁺ꜝ³ꜝˡ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝʳ",
    "χᵒᵇᶜ⁺ꜝ¹ꜝˡ",
    "χᵒᵇᶜ⁻ꜝ²ꜝʳ",
    "χᵒᵇᶜ⁻ꜝ²ꜝˡ",
    "χᵒᵇᶜ⁻ꜝ⁴ꜝʳ",
    "χᵒᵇᶜ⁻ꜝ⁴ꜝˡ",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝ",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝ",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝ",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝ",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾",
    "ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾",
    "beta",
    "alpha{H}",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
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
    -1.0e12,
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
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
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
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.1920928955078125e-7,
    1.1920928955078125e-7,
]
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
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
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
    Inf,
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
    1.0e12,
    Inf,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    Inf,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.9999998807907104,
    0.9999998807907104,
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
]
const ALL_AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    2.220446049250313e-16,
    2.220446049250313e-16,
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    2.220446049250313e-16,
    2.220446049250313e-16,
]
const ALL_AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    1.0e12,
    1.0e12,
    600.0,
    600.0,
    1.9999999999999998,
    1.9999999999999998,
    1.9999999999999998,
    1.9999999999999998,
    600.0,
    600.0,
    1.9999999999999998,
    1.9999999999999998,
    1.0e12,
]

const BLOCKS = [
    (
        index = 1,
        solve_order = 82,
        variables = ["alpha{H}"],
        previous_solution_names = ["a{H}", "k{H}", "y{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₃"],
        equation_indices = [1],
        equations = Expr[
            :(-a◖H◗ * ➕₁₃ ^ alpha◖H◗ + y◖H◗),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₃ = min(1.0e12, max(eps(), k◖H◗))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₃ - k◖H◗)),
        ],
        solution_names = ["alpha{H}", "➕₁₃"],
        previous_solution_values = [1.0, 15.600000000000001, 5.2],
        external_solution_values = Float64[],
        solution_values = [0.6001077713277385, 15.600000000000001],
        previous_solution_initial_values = [1.0, 15.600000000000001, 5.2],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.6001077713277385, 15.600000000000001],
        box_lower_bounds = [1.1920928955078125e-7, 2.220446049250313e-16],
        box_upper_bounds = [0.9999998807907104, 1.0e12],
    ),
    (
        index = 2,
        solve_order = 81,
        variables = ["κ{H}{B}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [30],
        equations = Expr[
            :(0.5κ◖H◗◖B◗ - λ◖B◗ * (1 - χ◖B◗)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["κ{H}{B}"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.3],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.3],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 3,
        solve_order = 80,
        variables = ["κ{H}{A}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [29],
        equations = Expr[
            :(0.5κ◖H◗◖A◗ - λ◖A◗ * (1 - χ◖A◗)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["κ{H}{A}"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.1],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.1],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 4,
        solve_order = 79,
        variables = ["κ{F}{B}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [60],
        equations = Expr[
            :(0.5κ◖F◗◖B◗ - λ◖B◗ * (1 - χ◖B◗)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["κ{F}{B}"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.3],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.3],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 5,
        solve_order = 78,
        variables = ["κ{F}{A}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [59],
        equations = Expr[
            :(0.5κ◖F◗◖A◗ - λ◖A◗ * (1 - χ◖A◗)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["κ{F}{A}"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.1],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.1],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 6,
        solve_order = 77,
        variables = ["world_output"],
        previous_solution_names = ["y{F}", "y{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [61],
        equations = Expr[
            :((world_output - y◖F◗) - y◖H◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["world_output"],
        previous_solution_values = [4.570786071618314, 5.2],
        external_solution_values = Float64[],
        solution_values = [9.770786071618314],
        previous_solution_initial_values = [0.0, 5.2],
        external_solution_initial_values = Float64[],
        solution_initial_values = [5.2],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 7,
        solve_order = 76,
        variables = ["tax{H}", "χᵒᵇᶜ⁻ꜝ²ꜝʳ", "χᵒᵇᶜ⁻ꜝ²ꜝˡ"],
        previous_solution_names = ["y{H}", "Χᵒᵇᶜ⁻ꜝ²ꜝ"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [11, 12, 13],
        equations = Expr[
            :(-tax_cap + tax◖H◗ + χᵒᵇᶜ⁻ꜝ²ꜝˡ),
            :((-tax_base - tax_rate * y◖H◗) + tax◖H◗ + χᵒᵇᶜ⁻ꜝ²ꜝʳ),
            :(Χᵒᵇᶜ⁻ꜝ²ꜝ - Min(χᵒᵇᶜ⁻ꜝ²ꜝʳ, χᵒᵇᶜ⁻ꜝ²ꜝˡ)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["tax{H}", "χᵒᵇᶜ⁻ꜝ²ꜝʳ", "χᵒᵇᶜ⁻ꜝ²ꜝˡ"],
        previous_solution_values = [5.2, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.15200000000000002, -3.2590301201736453e-21, 0.24799999999999997],
        previous_solution_initial_values = [5.2, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 0.0, 0.0],
        box_lower_bounds = [-1.0e12, -1.0e12, -1.0e12],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 8,
        solve_order = 75,
        variables = ["Χᵒᵇᶜ⁻ꜝ²ꜝ"],
        previous_solution_names = ["ϵᵒᵇᶜ⁻ꜝ²ꜝ"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [14],
        equations = Expr[
            :(Χᵒᵇᶜ⁻ꜝ²ꜝ - ϵᵒᵇᶜ⁻ꜝ²ꜝ),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["Χᵒᵇᶜ⁻ꜝ²ꜝ"],
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
        index = 9,
        solve_order = 74,
        variables = ["ϵᵒᵇᶜ⁻ꜝ²ꜝ"],
        previous_solution_names = ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [71],
        equations = Expr[
            :(ϵᵒᵇᶜ⁻ꜝ²ꜝ - ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁻ꜝ²ꜝ"],
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
        index = 10,
        solve_order = 73,
        variables = ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [76],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾ + ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾"],
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
        solve_order = 72,
        variables = ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [75],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾ + ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾"],
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
        solve_order = 71,
        variables = ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [74],
        equations = Expr[
            :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾ - ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾"],
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
        solve_order = 70,
        variables = ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [73],
        equations = Expr[
            :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾ - ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾"],
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
        solve_order = 69,
        variables = ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [72],
        equations = Expr[
            :(ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾ - 0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾"],
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
        index = 15,
        solve_order = 68,
        variables = ["tax{F}", "χᵒᵇᶜ⁻ꜝ⁴ꜝʳ", "χᵒᵇᶜ⁻ꜝ⁴ꜝˡ"],
        previous_solution_names = ["y{F}", "Χᵒᵇᶜ⁻ꜝ⁴ꜝ"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [41, 42, 43],
        equations = Expr[
            :(-tax_cap + tax◖F◗ + χᵒᵇᶜ⁻ꜝ⁴ꜝˡ),
            :((-tax_base - tax_rate * y◖F◗) + tax◖F◗ + χᵒᵇᶜ⁻ꜝ⁴ꜝʳ),
            :(Χᵒᵇᶜ⁻ꜝ⁴ꜝ - Min(χᵒᵇᶜ⁻ꜝ⁴ꜝʳ, χᵒᵇᶜ⁻ꜝ⁴ꜝˡ)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["tax{F}", "χᵒᵇᶜ⁻ꜝ⁴ꜝʳ", "χᵒᵇᶜ⁻ꜝ⁴ꜝˡ"],
        previous_solution_values = [4.570786071618314, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.14570786071618316, -3.479470926141543e-21, 0.25429213928381683],
        previous_solution_initial_values = [0.0, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 0.0, 0.0],
        box_lower_bounds = [-1.0e12, -1.0e12, -1.0e12],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 16,
        solve_order = 67,
        variables = ["Χᵒᵇᶜ⁻ꜝ⁴ꜝ"],
        previous_solution_names = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝ"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [44],
        equations = Expr[
            :(Χᵒᵇᶜ⁻ꜝ⁴ꜝ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝ),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["Χᵒᵇᶜ⁻ꜝ⁴ꜝ"],
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
        index = 17,
        solve_order = 66,
        variables = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝ"],
        previous_solution_names = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [83],
        equations = Expr[
            :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝ"],
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
        index = 18,
        solve_order = 65,
        variables = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [88],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾ + ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾"],
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
        solve_order = 64,
        variables = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [87],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾ + ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾"],
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
        solve_order = 63,
        variables = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [86],
        equations = Expr[
            :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾"],
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
        solve_order = 62,
        variables = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [85],
        equations = Expr[
            :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾ - ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾"],
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
        solve_order = 61,
        variables = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [84],
        equations = Expr[
            :(ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾ - 0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾"],
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
        index = 23,
        solve_order = 60,
        variables = ["steady_gap{H}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [28],
        equations = Expr[
            :(steady_gap◖H◗ - 0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["steady_gap{H}"],
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
        index = 24,
        solve_order = 59,
        variables = ["steady_gap{F}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [58],
        equations = Expr[
            :(steady_gap◖F◗ - 0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["steady_gap{F}"],
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
        index = 25,
        solve_order = 58,
        variables = ["sales_window{H}"],
        previous_solution_names = ["y{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [15],
        equations = Expr[
            :(sales_window◖H◗ - 3y◖H◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["sales_window{H}"],
        previous_solution_values = [5.2],
        external_solution_values = Float64[],
        solution_values = [15.600000000000001],
        previous_solution_initial_values = [5.2],
        external_solution_initial_values = Float64[],
        solution_initial_values = [15.600000000000001],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 26,
        solve_order = 57,
        variables = ["sales_window{F}"],
        previous_solution_names = ["y{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [45],
        equations = Expr[
            :(sales_window◖F◗ - 3y◖F◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["sales_window{F}"],
        previous_solution_values = [4.570786071618314],
        external_solution_values = Float64[],
        solution_values = [13.712358214854941],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 27,
        solve_order = 56,
        variables = ["r{F}", "χᵒᵇᶜ⁺ꜝ³ꜝʳ", "χᵒᵇᶜ⁺ꜝ³ꜝˡ"],
        previous_solution_names = ["beta", "Χᵒᵇᶜ⁺ꜝ³ꜝ"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [37, 38, 39],
        equations = Expr[
            :(-rate_floor + r◖F◗ + χᵒᵇᶜ⁺ꜝ³ꜝˡ),
            :((r◖F◗ + χᵒᵇᶜ⁺ꜝ³ꜝʳ) - 1 / beta),
            :(Χᵒᵇᶜ⁺ꜝ³ꜝ - Max(χᵒᵇᶜ⁺ꜝ³ꜝʳ, χᵒᵇᶜ⁺ꜝ³ꜝˡ)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["r{F}", "χᵒᵇᶜ⁺ꜝ³ꜝʳ", "χᵒᵇᶜ⁺ꜝ³ꜝˡ"],
        previous_solution_values = [0.9803921568627451, 0.0],
        external_solution_values = Float64[],
        solution_values = [1.0199999999999998, 1.6431625272369705e-16, -0.06999999999999967],
        previous_solution_initial_values = [0.9803921568627451, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 0.0, 0.0],
        box_lower_bounds = [-1.0e12, -1.0e12, -1.0e12],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 28,
        solve_order = 55,
        variables = ["beta"],
        previous_solution_names = ["r{H}", "χᵒᵇᶜ⁺ꜝ¹ꜝʳ"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [8],
        equations = Expr[
            :((r◖H◗ + χᵒᵇᶜ⁺ꜝ¹ꜝʳ) - 1 / beta),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["beta"],
        previous_solution_values = [1.02, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.9803921568627451],
        previous_solution_initial_values = [1.02, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.9803921568627451],
        box_lower_bounds = [1.1920928955078125e-7],
        box_upper_bounds = [0.9999998807907104],
    ),
    (
        index = 29,
        solve_order = 54,
        variables = ["χᵒᵇᶜ⁺ꜝ¹ꜝʳ"],
        previous_solution_names = ["Χᵒᵇᶜ⁺ꜝ¹ꜝ", "χᵒᵇᶜ⁺ꜝ¹ꜝˡ"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [9],
        equations = Expr[
            :(Χᵒᵇᶜ⁺ꜝ¹ꜝ - Max(χᵒᵇᶜ⁺ꜝ¹ꜝʳ, χᵒᵇᶜ⁺ꜝ¹ꜝˡ)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["χᵒᵇᶜ⁺ꜝ¹ꜝʳ"],
        previous_solution_values = [0.0, -0.07000000000000006],
        external_solution_values = Float64[],
        solution_values = [0.0],
        previous_solution_initial_values = [0.0, -0.07000000000000006],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-1.0e12],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 30,
        solve_order = 53,
        variables = ["χᵒᵇᶜ⁺ꜝ¹ꜝˡ"],
        previous_solution_names = ["r{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [7],
        equations = Expr[
            :(-rate_floor + r◖H◗ + χᵒᵇᶜ⁺ꜝ¹ꜝˡ),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["χᵒᵇᶜ⁺ꜝ¹ꜝˡ"],
        previous_solution_values = [1.02],
        external_solution_values = Float64[],
        solution_values = [-0.07000000000000006],
        previous_solution_initial_values = [1.02],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-0.07000000000000006],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 31,
        solve_order = 52,
        variables = ["Χᵒᵇᶜ⁺ꜝ¹ꜝ"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝ"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [10],
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
        index = 32,
        solve_order = 51,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝ"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [65],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ¹ꜝ - ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾),
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
        index = 33,
        solve_order = 50,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [70],
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
        index = 34,
        solve_order = 49,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [69],
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
        index = 35,
        solve_order = 48,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [68],
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
        index = 36,
        solve_order = 47,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [67],
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
        index = 37,
        solve_order = 46,
        variables = ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [66],
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
        index = 38,
        solve_order = 45,
        variables = ["r{H}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [89],
        equations = Expr[
            :(r◖H◗ - rate_target),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["r{H}"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.02],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.02],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 39,
        solve_order = 44,
        variables = ["Χᵒᵇᶜ⁺ꜝ³ꜝ"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ³ꜝ"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [40],
        equations = Expr[
            :(Χᵒᵇᶜ⁺ꜝ³ꜝ - ϵᵒᵇᶜ⁺ꜝ³ꜝ),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["Χᵒᵇᶜ⁺ꜝ³ꜝ"],
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
        index = 40,
        solve_order = 43,
        variables = ["ϵᵒᵇᶜ⁺ꜝ³ꜝ"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [77],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ³ꜝ - ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ³ꜝ"],
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
        index = 41,
        solve_order = 42,
        variables = ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [82],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾ + ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾"],
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
        solve_order = 41,
        variables = ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [81],
        equations = Expr[
            :(-ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾ + ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾"],
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
        solve_order = 40,
        variables = ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [80],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾ - ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾"],
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
        solve_order = 39,
        variables = ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾"],
        previous_solution_names = ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [79],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾ - ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾"],
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
        solve_order = 38,
        variables = ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [78],
        equations = Expr[
            :(ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾ - 0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾"],
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
        index = 46,
        solve_order = 37,
        variables = ["relative_output"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [62],
        equations = Expr[
            :(relative_output - 1),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["relative_output"],
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
        index = 47,
        solve_order = 36,
        variables = ["qnorm_signal{H}"],
        previous_solution_names = ["probability_signal{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₂"],
        equation_indices = [25],
        equations = Expr[
            :(qnorm_signal◖H◗ + 1.4142135623731 * erfcinv(➕₁₂)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₂ = min(2 - eps(), max(eps(), probability_signal◖H◗ * 2))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₂ - probability_signal◖H◗ * 2)),
        ],
        solution_names = ["qnorm_signal{H}", "➕₁₂"],
        previous_solution_values = [0.5],
        external_solution_values = Float64[],
        solution_values = [0.0, 1.0],
        previous_solution_initial_values = [0.5],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 1.0],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.9999999999999998],
    ),
    (
        index = 48,
        solve_order = 35,
        variables = ["qnorm_signal{F}"],
        previous_solution_names = ["probability_signal{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₁"],
        equation_indices = [55],
        equations = Expr[
            :(qnorm_signal◖F◗ + 1.4142135623731 * erfcinv(➕₁₁)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₁ = min(2 - eps(), max(eps(), probability_signal◖F◗ * 2))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₁ - probability_signal◖F◗ * 2)),
        ],
        solution_names = ["qnorm_signal{F}", "➕₁₁"],
        previous_solution_values = [0.5],
        external_solution_values = Float64[],
        solution_values = [0.0, 1.0],
        previous_solution_initial_values = [0.5],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 1.0],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.9999999999999998],
    ),
    (
        index = 49,
        solve_order = 34,
        variables = ["pnorm_signal{H}"],
        previous_solution_names = ["a{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [26],
        equations = Expr[
            :((pnorm_signal◖H◗ + erfc(0.707106781186547a◖H◗ - 0.707106781186547) / 2) - 1),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["pnorm_signal{H}"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [0.5],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.5],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 50,
        solve_order = 33,
        variables = ["pnorm_signal{F}"],
        previous_solution_names = ["a{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [56],
        equations = Expr[
            :((pnorm_signal◖F◗ + erfc(0.707106781186547a◖F◗ - 0.707106781186547) / 2) - 1),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["pnorm_signal{F}"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [0.5],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.5],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 51,
        solve_order = 32,
        variables = ["pdf_signal{H}"],
        previous_solution_names = ["a{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₀"],
        equation_indices = [20],
        equations = Expr[
            :(pdf_signal◖H◗ - 0.398942280401433 * exp(➕₁₀)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₀ = min(600, max(-1.0e12, -((a◖H◗ - 1) ^ 2) / 2))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₀ - -((a◖H◗ - 1) ^ 2) / 2)),
        ],
        solution_names = ["pdf_signal{H}", "➕₁₀"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [0.398942280401433, -0.0],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.398942280401433, 0.0],
        box_lower_bounds = [-Inf, -1.0e12],
        box_upper_bounds = [Inf, 600.0],
    ),
    (
        index = 52,
        solve_order = 31,
        variables = ["pdf_signal{F}"],
        previous_solution_names = ["a{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₉"],
        equation_indices = [50],
        equations = Expr[
            :(pdf_signal◖F◗ - 0.398942280401433 * exp(➕₉)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₉ = min(600, max(-1.0e12, -((a◖F◗ - 1) ^ 2) / 2))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₉ - -((a◖F◗ - 1) ^ 2) / 2)),
        ],
        solution_names = ["pdf_signal{F}", "➕₉"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [0.398942280401433, -0.0],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.398942280401433, 0.0],
        box_lower_bounds = [-Inf, -1.0e12],
        box_upper_bounds = [Inf, 600.0],
    ),
    (
        index = 53,
        solve_order = 30,
        variables = ["norminv_signal{H}"],
        previous_solution_names = ["probability_signal{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₈"],
        equation_indices = [24],
        equations = Expr[
            :(norminv_signal◖H◗ + 1.4142135623731 * erfcinv(➕₈)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₈ = min(2 - eps(), max(eps(), probability_signal◖H◗ * 2))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₈ - probability_signal◖H◗ * 2)),
        ],
        solution_names = ["norminv_signal{H}", "➕₈"],
        previous_solution_values = [0.5],
        external_solution_values = Float64[],
        solution_values = [0.0, 1.0],
        previous_solution_initial_values = [0.5],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 1.0],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.9999999999999998],
    ),
    (
        index = 54,
        solve_order = 29,
        variables = ["norminv_signal{F}"],
        previous_solution_names = ["probability_signal{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₇"],
        equation_indices = [54],
        equations = Expr[
            :(norminv_signal◖F◗ + 1.4142135623731 * erfcinv(➕₇)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₇ = min(2 - eps(), max(eps(), probability_signal◖F◗ * 2))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₇ - probability_signal◖F◗ * 2)),
        ],
        solution_names = ["norminv_signal{F}", "➕₇"],
        previous_solution_values = [0.5],
        external_solution_values = Float64[],
        solution_values = [0.0, 1.0],
        previous_solution_initial_values = [0.5],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 1.0],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.9999999999999998],
    ),
    (
        index = 55,
        solve_order = 28,
        variables = ["net_exports{H}"],
        previous_solution_names = ["y{F}", "y{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [63],
        equations = Expr[
            :(net_exports◖H◗ - trade_weight◖H◗◖F◗ * (-y◖F◗ + y◖H◗)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["net_exports{H}"],
        previous_solution_values = [4.570786071618314, 5.2],
        external_solution_values = Float64[],
        solution_values = [0.12584278567633725],
        previous_solution_initial_values = [0.0, 5.2],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.04],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 56,
        solve_order = 27,
        variables = ["net_exports{F}"],
        previous_solution_names = ["y{F}", "y{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [64],
        equations = Expr[
            :(net_exports◖F◗ - trade_weight◖F◗◖H◗ * (y◖F◗ - y◖H◗)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["net_exports{F}"],
        previous_solution_values = [4.570786071618314, 5.2],
        external_solution_values = Float64[],
        solution_values = [-0.12584278567633725],
        previous_solution_initial_values = [0.0, 5.2],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-1.04],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 57,
        solve_order = 26,
        variables = ["logpdf_signal{H}"],
        previous_solution_names = ["a{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [21],
        equations = Expr[
            :(logpdf_signal◖H◗ + (a◖H◗ - 1) ^ 2 / 2 + 0.918938533204673),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["logpdf_signal{H}"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [-0.918938533204673],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-0.918938533204673],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 58,
        solve_order = 25,
        variables = ["logpdf_signal{F}"],
        previous_solution_names = ["a{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [51],
        equations = Expr[
            :(logpdf_signal◖F◗ + (a◖F◗ - 1) ^ 2 / 2 + 0.918938533204673),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["logpdf_signal{F}"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [-0.918938533204673],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [-0.918938533204673],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 59,
        solve_order = 24,
        variables = ["k{H}"],
        previous_solution_names = ["i{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [5],
        equations = Expr[
            :((-i◖H◗ - k◖H◗ * (1 - delta)) + k◖H◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["k{H}"],
        previous_solution_values = [0.7800000000000001],
        external_solution_values = Float64[],
        solution_values = [15.600000000000001],
        previous_solution_initial_values = [0.7800000000000001],
        external_solution_initial_values = Float64[],
        solution_initial_values = [15.600000000000001],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 60,
        solve_order = 23,
        variables = ["inverse_signal{H}"],
        previous_solution_names = ["probability_signal{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₆"],
        equation_indices = [23],
        equations = Expr[
            :(inverse_signal◖H◗ + 1.4142135623731 * erfcinv(➕₆)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₆ = min(2 - eps(), max(eps(), probability_signal◖H◗ * 2))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₆ - probability_signal◖H◗ * 2)),
        ],
        solution_names = ["inverse_signal{H}", "➕₆"],
        previous_solution_values = [0.5],
        external_solution_values = Float64[],
        solution_values = [0.0, 1.0],
        previous_solution_initial_values = [0.5],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 1.0],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.9999999999999998],
    ),
    (
        index = 61,
        solve_order = 22,
        variables = ["probability_signal{H}"],
        previous_solution_names = ["a{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [22],
        equations = Expr[
            :((probability_signal◖H◗ - 0.25 * tanh(a◖H◗ - 1)) - 0.5),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["probability_signal{H}"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [0.5],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.5],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 62,
        solve_order = 21,
        variables = ["inverse_signal{F}"],
        previous_solution_names = ["probability_signal{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₅"],
        equation_indices = [53],
        equations = Expr[
            :(inverse_signal◖F◗ + 1.4142135623731 * erfcinv(➕₅)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₅ = min(2 - eps(), max(eps(), probability_signal◖F◗ * 2))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₅ - probability_signal◖F◗ * 2)),
        ],
        solution_names = ["inverse_signal{F}", "➕₅"],
        previous_solution_values = [0.5],
        external_solution_values = Float64[],
        solution_values = [0.0, 1.0],
        previous_solution_initial_values = [0.5],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 1.0],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.9999999999999998],
    ),
    (
        index = 63,
        solve_order = 20,
        variables = ["probability_signal{F}"],
        previous_solution_names = ["a{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [52],
        equations = Expr[
            :((probability_signal◖F◗ - 0.25 * tanh(a◖F◗ - 1)) - 0.5),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["probability_signal{F}"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [0.5],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.5],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 64,
        solve_order = 19,
        variables = ["inflation_product{H}"],
        previous_solution_names = ["inflation{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [17],
        equations = Expr[
            :(inflation_product◖H◗ - inflation◖H◗ ^ 3),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["inflation_product{H}"],
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
        index = 65,
        solve_order = 18,
        variables = ["inflation{H}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [18],
        equations = Expr[
            :(-inflation_bar + inflation◖H◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["inflation{H}"],
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
        index = 66,
        solve_order = 17,
        variables = ["inflation_product{F}"],
        previous_solution_names = ["inflation{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [47],
        equations = Expr[
            :(inflation_product◖F◗ - inflation◖F◗ ^ 3),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["inflation_product{F}"],
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
        index = 67,
        solve_order = 16,
        variables = ["inflation{F}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [48],
        equations = Expr[
            :(-inflation_bar + inflation◖F◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["inflation{F}"],
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
        index = 68,
        solve_order = 15,
        variables = ["g{H}"],
        previous_solution_names = ["c{H}", "i{H}", "y{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [4],
        equations = Expr[
            :(((-c◖H◗ - g◖H◗) - i◖H◗) + y◖H◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["g{H}"],
        previous_solution_values = [3.6399999999999997, 0.7800000000000001, 5.2],
        external_solution_values = Float64[],
        solution_values = [0.7800000000000002],
        previous_solution_initial_values = [3.6399999999999997, 0.7800000000000001, 5.2],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.7800000000000002],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 69,
        solve_order = 14,
        variables = ["i{H}"],
        previous_solution_names = ["y{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [3],
        equations = Expr[
            :(-investment_share * y◖H◗ + i◖H◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["i{H}"],
        previous_solution_values = [5.2],
        external_solution_values = Float64[],
        solution_values = [0.7800000000000001],
        previous_solution_initial_values = [5.2],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.7800000000000001],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 70,
        solve_order = 13,
        variables = ["g{F}"],
        previous_solution_names = ["c{F}", "i{F}", "y{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [34],
        equations = Expr[
            :(((-c◖F◗ - g◖F◗) - i◖F◗) + y◖F◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["g{F}"],
        previous_solution_values = [3.1995502501328197, 0.6856179107427471, 4.570786071618314],
        external_solution_values = Float64[],
        solution_values = [0.6856179107427471],
        previous_solution_initial_values = [0.0, 0.0, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 71,
        solve_order = 12,
        variables = ["forward_anchor{H}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [16],
        equations = Expr[
            :(forward_anchor◖H◗ - 0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["forward_anchor{H}"],
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
        index = 72,
        solve_order = 11,
        variables = ["forward_anchor{F}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [46],
        equations = Expr[
            :(forward_anchor◖F◗ - 0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["forward_anchor{F}"],
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
        index = 73,
        solve_order = 10,
        variables = ["dnorm_signal{H}"],
        previous_solution_names = ["a{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₄"],
        equation_indices = [27],
        equations = Expr[
            :(dnorm_signal◖H◗ - 0.398942280401433 * exp(➕₄)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₄ = min(600, max(-1.0e12, -((a◖H◗ - 1) ^ 2) / 2))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₄ - -((a◖H◗ - 1) ^ 2) / 2)),
        ],
        solution_names = ["dnorm_signal{H}", "➕₄"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [0.398942280401433, -0.0],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.398942280401433, 0.0],
        box_lower_bounds = [-Inf, -1.0e12],
        box_upper_bounds = [Inf, 600.0],
    ),
    (
        index = 74,
        solve_order = 9,
        variables = ["dnorm_signal{F}"],
        previous_solution_names = ["a{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₃"],
        equation_indices = [57],
        equations = Expr[
            :(dnorm_signal◖F◗ - 0.398942280401433 * exp(➕₃)),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₃ = min(600, max(-1.0e12, -((a◖F◗ - 1) ^ 2) / 2))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₃ - -((a◖F◗ - 1) ^ 2) / 2)),
        ],
        solution_names = ["dnorm_signal{F}", "➕₃"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [0.398942280401433, -0.0],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.398942280401433, 0.0],
        box_lower_bounds = [-Inf, -1.0e12],
        box_upper_bounds = [Inf, 600.0],
    ),
    (
        index = 75,
        solve_order = 8,
        variables = ["c{H}"],
        previous_solution_names = ["y{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [2],
        equations = Expr[
            :(-consumption_share * y◖H◗ + c◖H◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["c{H}"],
        previous_solution_values = [5.2],
        external_solution_values = Float64[],
        solution_values = [3.6399999999999997],
        previous_solution_initial_values = [5.2],
        external_solution_initial_values = Float64[],
        solution_initial_values = [3.6399999999999997],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 76,
        solve_order = 7,
        variables = ["y{H}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [90],
        equations = Expr[
            :(y◖H◗ - output_target),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["y{H}"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [5.2],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [5.2],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 77,
        solve_order = 6,
        variables = ["c{F}"],
        previous_solution_names = ["y{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [32],
        equations = Expr[
            :(-consumption_share * y◖F◗ + c◖F◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["c{F}"],
        previous_solution_values = [4.570786071618314],
        external_solution_values = Float64[],
        solution_values = [3.1995502501328197],
        previous_solution_initial_values = [0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 78,
        solve_order = 5,
        variables = ["i{F}", "k{F}", "y{F}"],
        previous_solution_names = ["a{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [33, 35, 31],
        equations = Expr[
            :(-investment_share * y◖F◗ + i◖F◗),
            :((-i◖F◗ - k◖F◗ * (1 - delta)) + k◖F◗),
            :(-a◖F◗ * foreign_scale * k◖F◗ ^ alpha◖F◗ + y◖F◗),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["i{F}", "k{F}", "y{F}"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [0.6856179107427471, 13.71235821485494, 4.570786071618314],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 5.0e11, 0.0],
        box_lower_bounds = [-1.0e12, 2.220446049250313e-16, -1.0e12],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 79,
        solve_order = 4,
        variables = ["cdf_signal{H}"],
        previous_solution_names = ["a{H}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [19],
        equations = Expr[
            :((cdf_signal◖H◗ + erfc(0.707106781186547a◖H◗ - 0.707106781186547) / 2) - 1),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["cdf_signal{H}"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [0.5],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.5],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 80,
        solve_order = 3,
        variables = ["cdf_signal{F}"],
        previous_solution_names = ["a{F}"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [49],
        equations = Expr[
            :((cdf_signal◖F◗ + erfc(0.707106781186547a◖F◗ - 0.707106781186547) / 2) - 1),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["cdf_signal{F}"],
        previous_solution_values = [1.0],
        external_solution_values = Float64[],
        solution_values = [0.5],
        previous_solution_initial_values = [1.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.5],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 81,
        solve_order = 2,
        variables = ["a{H}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [6],
        equations = Expr[
            :((-a◖H◗ * rho + a◖H◗ + rho) - 1),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["a{H}"],
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
        index = 82,
        solve_order = 1,
        variables = ["a{F}"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [36],
        equations = Expr[
            :((-a◖F◗ * rho + a◖F◗ + rho) - 1),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["a{F}"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.0],
        previous_solution_initial_values = Float64[],
        external_solution_initial_values = Float64[],
        solution_initial_values = [1.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
]
const BLOCK_EQUATION_ORDER = [1, 30, 29, 60, 59, 61, 11, 12, 13, 14, 71, 76, 75, 74, 73, 72, 41, 42, 43, 44, 83, 88, 87, 86, 85, 84, 28, 58, 15, 45, 37, 38, 39, 8, 9, 7, 10, 65, 70, 69, 68, 67, 66, 89, 40, 77, 82, 81, 80, 79, 78, 62, 25, 55, 26, 56, 20, 50, 24, 54, 63, 64, 21, 51, 5, 23, 22, 53, 52, 17, 18, 47, 48, 4, 3, 34, 16, 46, 27, 57, 2, 90, 32, 33, 35, 31, 19, 49, 6, 36]
const BLOCK_SOLVE_ORDER = [82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    ["a{H}", "k{H}", "y{H}"],
    String[],
    String[],
    String[],
    String[],
    ["y{F}", "y{H}"],
    ["y{H}", "Χᵒᵇᶜ⁻ꜝ²ꜝ"],
    ["ϵᵒᵇᶜ⁻ꜝ²ꜝ"],
    ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾"],
    String[],
    ["y{F}", "Χᵒᵇᶜ⁻ꜝ⁴ꜝ"],
    ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝ"],
    ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾"],
    String[],
    String[],
    String[],
    ["y{H}"],
    ["y{F}"],
    ["beta", "Χᵒᵇᶜ⁺ꜝ³ꜝ"],
    ["r{H}", "χᵒᵇᶜ⁺ꜝ¹ꜝʳ"],
    ["Χᵒᵇᶜ⁺ꜝ¹ꜝ", "χᵒᵇᶜ⁺ꜝ¹ꜝˡ"],
    ["r{H}"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝ"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾"],
    String[],
    String[],
    ["ϵᵒᵇᶜ⁺ꜝ³ꜝ"],
    ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾"],
    String[],
    String[],
    ["probability_signal{H}"],
    ["probability_signal{F}"],
    ["a{H}"],
    ["a{F}"],
    ["a{H}"],
    ["a{F}"],
    ["probability_signal{H}"],
    ["probability_signal{F}"],
    ["y{F}", "y{H}"],
    ["y{F}", "y{H}"],
    ["a{H}"],
    ["a{F}"],
    ["i{H}"],
    ["probability_signal{H}"],
    ["a{H}"],
    ["probability_signal{F}"],
    ["a{F}"],
    ["inflation{H}"],
    String[],
    ["inflation{F}"],
    String[],
    ["c{H}", "i{H}", "y{H}"],
    ["y{H}"],
    ["c{F}", "i{F}", "y{F}"],
    String[],
    String[],
    ["a{H}"],
    ["a{F}"],
    ["y{H}"],
    String[],
    ["y{F}"],
    ["a{F}"],
    ["a{H}"],
    ["a{F}"],
    String[],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    [1.0, 15.600000000000001, 5.2],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    [4.570786071618314, 5.2],
    [5.2, 0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    Float64[],
    [4.570786071618314, 0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    Float64[],
    Float64[],
    Float64[],
    [5.2],
    [4.570786071618314],
    [0.9803921568627451, 0.0],
    [1.02, 0.0],
    [0.0, -0.07000000000000006],
    [1.02],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    Float64[],
    Float64[],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    Float64[],
    Float64[],
    [0.5],
    [0.5],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [0.5],
    [0.5],
    [4.570786071618314, 5.2],
    [4.570786071618314, 5.2],
    [1.0],
    [1.0],
    [0.7800000000000001],
    [0.5],
    [1.0],
    [0.5],
    [1.0],
    [1.0],
    Float64[],
    [1.0],
    Float64[],
    [3.6399999999999997, 0.7800000000000001, 5.2],
    [5.2],
    [3.1995502501328197, 0.6856179107427471, 4.570786071618314],
    Float64[],
    Float64[],
    [1.0],
    [1.0],
    [5.2],
    Float64[],
    [4.570786071618314],
    [1.0],
    [1.0],
    [1.0],
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
    Float64[],
    Float64[],
    Float64[],
    Float64[],
]
const BLOCK_SOLUTION_NAMES = [
    ["alpha{H}", "➕₁₃"],
    ["κ{H}{B}"],
    ["κ{H}{A}"],
    ["κ{F}{B}"],
    ["κ{F}{A}"],
    ["world_output"],
    ["tax{H}", "χᵒᵇᶜ⁻ꜝ²ꜝʳ", "χᵒᵇᶜ⁻ꜝ²ꜝˡ"],
    ["Χᵒᵇᶜ⁻ꜝ²ꜝ"],
    ["ϵᵒᵇᶜ⁻ꜝ²ꜝ"],
    ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁴⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻³⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻²⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻¹⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ²ꜝᴸ⁽⁻⁰⁾"],
    ["tax{F}", "χᵒᵇᶜ⁻ꜝ⁴ꜝʳ", "χᵒᵇᶜ⁻ꜝ⁴ꜝˡ"],
    ["Χᵒᵇᶜ⁻ꜝ⁴ꜝ"],
    ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝ"],
    ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁴⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻³⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻²⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻¹⁾"],
    ["ϵᵒᵇᶜ⁻ꜝ⁴ꜝᴸ⁽⁻⁰⁾"],
    ["steady_gap{H}"],
    ["steady_gap{F}"],
    ["sales_window{H}"],
    ["sales_window{F}"],
    ["r{F}", "χᵒᵇᶜ⁺ꜝ³ꜝʳ", "χᵒᵇᶜ⁺ꜝ³ꜝˡ"],
    ["beta"],
    ["χᵒᵇᶜ⁺ꜝ¹ꜝʳ"],
    ["χᵒᵇᶜ⁺ꜝ¹ꜝˡ"],
    ["Χᵒᵇᶜ⁺ꜝ¹ꜝ"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝ"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻²⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻¹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁰⁾"],
    ["r{H}"],
    ["Χᵒᵇᶜ⁺ꜝ³ꜝ"],
    ["ϵᵒᵇᶜ⁺ꜝ³ꜝ"],
    ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁴⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻³⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻²⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻¹⁾"],
    ["ϵᵒᵇᶜ⁺ꜝ³ꜝᴸ⁽⁻⁰⁾"],
    ["relative_output"],
    ["qnorm_signal{H}", "➕₁₂"],
    ["qnorm_signal{F}", "➕₁₁"],
    ["pnorm_signal{H}"],
    ["pnorm_signal{F}"],
    ["pdf_signal{H}", "➕₁₀"],
    ["pdf_signal{F}", "➕₉"],
    ["norminv_signal{H}", "➕₈"],
    ["norminv_signal{F}", "➕₇"],
    ["net_exports{H}"],
    ["net_exports{F}"],
    ["logpdf_signal{H}"],
    ["logpdf_signal{F}"],
    ["k{H}"],
    ["inverse_signal{H}", "➕₆"],
    ["probability_signal{H}"],
    ["inverse_signal{F}", "➕₅"],
    ["probability_signal{F}"],
    ["inflation_product{H}"],
    ["inflation{H}"],
    ["inflation_product{F}"],
    ["inflation{F}"],
    ["g{H}"],
    ["i{H}"],
    ["g{F}"],
    ["forward_anchor{H}"],
    ["forward_anchor{F}"],
    ["dnorm_signal{H}", "➕₄"],
    ["dnorm_signal{F}", "➕₃"],
    ["c{H}"],
    ["y{H}"],
    ["c{F}"],
    ["i{F}", "k{F}", "y{F}"],
    ["cdf_signal{H}"],
    ["cdf_signal{F}"],
    ["a{H}"],
    ["a{F}"],
]
const BLOCK_SOLUTION_VALUES = [
    [0.6001077713277385, 15.600000000000001],
    [0.3],
    [0.1],
    [0.3],
    [0.1],
    [9.770786071618314],
    [0.15200000000000002, -3.2590301201736453e-21, 0.24799999999999997],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.14570786071618316, -3.479470926141543e-21, 0.25429213928381683],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [15.600000000000001],
    [13.712358214854941],
    [1.0199999999999998, 1.6431625272369705e-16, -0.06999999999999967],
    [0.9803921568627451],
    [0.0],
    [-0.07000000000000006],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [1.02],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [1.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.5],
    [0.5],
    [0.398942280401433, -0.0],
    [0.398942280401433, -0.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.12584278567633725],
    [-0.12584278567633725],
    [-0.918938533204673],
    [-0.918938533204673],
    [15.600000000000001],
    [0.0, 1.0],
    [0.5],
    [0.0, 1.0],
    [0.5],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [0.7800000000000002],
    [0.7800000000000001],
    [0.6856179107427471],
    [0.0],
    [0.0],
    [0.398942280401433, -0.0],
    [0.398942280401433, -0.0],
    [3.6399999999999997],
    [5.2],
    [3.1995502501328197],
    [0.6856179107427471, 13.71235821485494, 4.570786071618314],
    [0.5],
    [0.5],
    [1.0],
    [1.0],
]
const BLOCK_PREVIOUS_SOLUTION_INITIAL_VALUES = [
    [1.0, 15.600000000000001, 5.2],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    [0.0, 5.2],
    [5.2, 0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    Float64[],
    [0.0, 0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    Float64[],
    Float64[],
    Float64[],
    [5.2],
    [0.0],
    [0.9803921568627451, 0.0],
    [1.02, 0.0],
    [0.0, -0.07000000000000006],
    [1.02],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    Float64[],
    Float64[],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    Float64[],
    Float64[],
    [0.5],
    [0.5],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [0.5],
    [0.5],
    [0.0, 5.2],
    [0.0, 5.2],
    [1.0],
    [1.0],
    [0.7800000000000001],
    [0.5],
    [1.0],
    [0.5],
    [1.0],
    [1.0],
    Float64[],
    [1.0],
    Float64[],
    [3.6399999999999997, 0.7800000000000001, 5.2],
    [5.2],
    [0.0, 0.0, 0.0],
    Float64[],
    Float64[],
    [1.0],
    [1.0],
    [5.2],
    Float64[],
    [0.0],
    [1.0],
    [1.0],
    [1.0],
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
    Float64[],
    Float64[],
    Float64[],
    Float64[],
]
const BLOCK_SOLUTION_INITIAL_VALUES = [
    [0.6001077713277385, 15.600000000000001],
    [0.3],
    [0.1],
    [0.3],
    [0.1],
    [5.2],
    [0.0, 0.0, 0.0],
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
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [15.600000000000001],
    [0.0],
    [0.0, 0.0, 0.0],
    [0.9803921568627451],
    [0.0],
    [-0.07000000000000006],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [1.02],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [1.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.5],
    [0.5],
    [0.398942280401433, 0.0],
    [0.398942280401433, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [1.04],
    [-1.04],
    [-0.918938533204673],
    [-0.918938533204673],
    [15.600000000000001],
    [0.0, 1.0],
    [0.5],
    [0.0, 1.0],
    [0.5],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [0.7800000000000002],
    [0.7800000000000001],
    [0.0],
    [0.0],
    [0.0],
    [0.398942280401433, 0.0],
    [0.398942280401433, 0.0],
    [3.6399999999999997],
    [5.2],
    [0.0],
    [0.0, 5.0e11, 0.0],
    [0.5],
    [0.5],
    [1.0],
    [1.0],
]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[19] = parameters[19]
    complete_parameters[15] = parameters[15]
    complete_parameters[23] = parameters[23]
    complete_parameters[7] = parameters[7]
    complete_parameters[1] = parameters[1]
    complete_parameters[12] = parameters[12]
    complete_parameters[10] = parameters[10]
    complete_parameters[8] = parameters[8]
    complete_parameters[9] = parameters[9]
    complete_parameters[5] = parameters[5]
    complete_parameters[22] = parameters[22]
    complete_parameters[14] = parameters[14]
    complete_parameters[17] = parameters[17]
    complete_parameters[6] = parameters[6]
    complete_parameters[2] = parameters[2]
    complete_parameters[21] = parameters[21]
    complete_parameters[11] = parameters[11]
    complete_parameters[13] = parameters[13]
    complete_parameters[16] = parameters[16]
    complete_parameters[3] = parameters[3]
    complete_parameters[18] = parameters[18]
    complete_parameters[4] = parameters[4]
    complete_parameters[20] = parameters[20]
    complete_parameters[24] = 1 - complete_parameters[4]
    complete_parameters[25] = complete_parameters[22] * 3
    complete_parameters[26] = 1 + complete_parameters[21]
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[48] - solution[2] * solution[22] ^ solution[90],
        solution[6] - complete_parameters[3] * solution[48],
        solution[20] - complete_parameters[25] * solution[48],
        ((solution[48] - solution[6]) - solution[20]) - solution[12],
        solution[22] - ((1 - complete_parameters[22]) * solution[22] + solution[20]),
        solution[2] - ((1 - complete_parameters[1]) + complete_parameters[1] * solution[2] + complete_parameters[2] * 0 + complete_parameters[2] * 0 + complete_parameters[2] * 0 + complete_parameters[2] * 0),
        solution[60] - (complete_parameters[7] - solution[39]),
        solution[59] - (solution[6] / (solution[89] * solution[6]) - solution[39]),
        solution[50] - max(solution[60], solution[59]),
        solution[50] - solution[71],
        solution[62] - (complete_parameters[8] - solution[45]),
        solution[61] - ((complete_parameters[9] + complete_parameters[10] * solution[48]) - solution[45]),
        solution[51] - min(solution[62], solution[61]),
        solution[51] - solution[77],
        solution[41] - (solution[48] + solution[48] + solution[48]),
        solution[10] - (solution[48] - solution[48]),
        solution[14] - solution[16] * solution[16] * solution[16],
        solution[16] - (complete_parameters[5] + complete_parameters[6] * 0),
        solution[4] - normcdf(solution[2] - 1),
        solution[30] - normpdf(solution[2] - 1),
        solution[24] - normlogpdf(solution[2] - 1),
        solution[34] - (0.5 + 0.25 * tanh(solution[2] - 1)),
        solution[18] - norminvcdf(solution[34]),
        solution[28] - norminv(solution[34]),
        solution[36] - qnorm(solution[34]),
        solution[32] - pnorm(solution[2] - 1),
        solution[8] - dnorm(solution[2] - 1),
        solution[43] - (solution[48] - solution[48]),
        solution[55] - (0.25 * solution[55] + 0.25 * solution[55] + complete_parameters[15] * ifelse(complete_parameters[11] > 0.5, complete_parameters[17], 1 - complete_parameters[17]) + complete_parameters[12] * 0 + complete_parameters[12] * 0),
        solution[56] - (0.25 * solution[56] + 0.25 * solution[56] + complete_parameters[16] * ifelse(complete_parameters[11] > 0.5, complete_parameters[18], 1 - complete_parameters[18]) + complete_parameters[12] * 0 + complete_parameters[12] * 0),
        solution[47] - complete_parameters[24] * solution[1] * solution[21] ^ complete_parameters[19],
        solution[5] - complete_parameters[3] * solution[47],
        solution[19] - complete_parameters[25] * solution[47],
        ((solution[47] - solution[5]) - solution[19]) - solution[11],
        solution[21] - ((1 - complete_parameters[22]) * solution[21] + solution[19]),
        solution[1] - ((1 - complete_parameters[1]) + complete_parameters[1] * solution[1] + complete_parameters[2] * 0 + complete_parameters[2] * 0 + complete_parameters[2] * 0 + complete_parameters[2] * 0),
        solution[58] - (complete_parameters[7] - solution[38]),
        solution[57] - (solution[5] / (solution[89] * solution[5]) - solution[38]),
        solution[49] - max(solution[58], solution[57]),
        solution[49] - solution[65],
        solution[64] - (complete_parameters[8] - solution[44]),
        solution[63] - ((complete_parameters[9] + complete_parameters[10] * solution[47]) - solution[44]),
        solution[52] - min(solution[64], solution[63]),
        solution[52] - solution[83],
        solution[40] - (solution[47] + solution[47] + solution[47]),
        solution[9] - (solution[47] - solution[47]),
        solution[13] - solution[15] * solution[15] * solution[15],
        solution[15] - (complete_parameters[5] + complete_parameters[6] * 0),
        solution[3] - normcdf(solution[1] - 1),
        solution[29] - normpdf(solution[1] - 1),
        solution[23] - normlogpdf(solution[1] - 1),
        solution[33] - (0.5 + 0.25 * tanh(solution[1] - 1)),
        solution[17] - norminvcdf(solution[33]),
        solution[27] - norminv(solution[33]),
        solution[35] - qnorm(solution[33]),
        solution[31] - pnorm(solution[1] - 1),
        solution[7] - dnorm(solution[1] - 1),
        solution[42] - (solution[47] - solution[47]),
        solution[53] - (0.25 * solution[53] + 0.25 * solution[53] + complete_parameters[15] * ifelse(complete_parameters[11] > 0.5, complete_parameters[17], 1 - complete_parameters[17]) + complete_parameters[12] * 0 + complete_parameters[12] * 0),
        solution[54] - (0.25 * solution[54] + 0.25 * solution[54] + complete_parameters[16] * ifelse(complete_parameters[11] > 0.5, complete_parameters[18], 1 - complete_parameters[18]) + complete_parameters[12] * 0 + complete_parameters[12] * 0),
        solution[46] - (solution[48] + solution[47]),
        solution[37] - (solution[48] / solution[47]) / (solution[48] / solution[47]),
        solution[26] - complete_parameters[13] * (solution[48] - solution[47]),
        solution[25] - complete_parameters[14] * (solution[47] - solution[48]),
        solution[71] - solution[76],
        solution[75] - complete_parameters[23] * 0,
        solution[74] - (solution[75] + complete_parameters[23] * 0),
        solution[72] - (solution[74] + complete_parameters[23] * 0),
        solution[73] - (solution[72] + complete_parameters[23] * 0),
        solution[76] - (solution[73] + complete_parameters[23] * 0),
        solution[77] - solution[82],
        solution[81] - complete_parameters[23] * 0,
        solution[80] - (solution[81] + complete_parameters[23] * 0),
        solution[78] - (solution[80] + complete_parameters[23] * 0),
        solution[79] - (solution[78] + complete_parameters[23] * 0),
        solution[82] - (solution[79] + complete_parameters[23] * 0),
        solution[65] - solution[70],
        solution[69] - complete_parameters[23] * 0,
        solution[68] - (solution[69] + complete_parameters[23] * 0),
        solution[66] - (solution[68] + complete_parameters[23] * 0),
        solution[67] - (solution[66] + complete_parameters[23] * 0),
        solution[70] - (solution[67] + complete_parameters[23] * 0),
        solution[83] - solution[88],
        solution[87] - complete_parameters[23] * 0,
        solution[86] - (solution[87] + complete_parameters[23] * 0),
        solution[84] - (solution[86] + complete_parameters[23] * 0),
        solution[85] - (solution[84] + complete_parameters[23] * 0),
        solution[88] - (solution[85] + complete_parameters[23] * 0),
        solution[39] - complete_parameters[26],
        solution[48] - complete_parameters[20],
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[2]) * solution[22] ^ solution[90] + solution[48],
        -(complete_parameters[3]) * solution[48] + solution[6],
        -(complete_parameters[25]) * solution[48] + solution[20],
        ((-(solution[6]) - solution[12]) - solution[20]) + solution[48],
        (-(solution[20]) - solution[22] * (1 - complete_parameters[22])) + solution[22],
        (-(solution[2]) * complete_parameters[1] + solution[2] + complete_parameters[1]) - 1,
        -(complete_parameters[7]) + solution[39] + solution[60],
        (solution[39] + solution[59]) - 1 / solution[89],
        solution[50] - Max(solution[59], solution[60]),
        solution[50] - solution[71],
        -(complete_parameters[8]) + solution[45] + solution[62],
        (-(complete_parameters[9]) - complete_parameters[10] * solution[48]) + solution[45] + solution[61],
        solution[51] - Min(solution[61], solution[62]),
        solution[51] - solution[77],
        solution[41] - 3 * solution[48],
        solution[10] - 0,
        solution[14] - solution[16] ^ 3,
        -(complete_parameters[5]) + solution[16],
        (solution[4] + erfc(0.707106781186547 * solution[2] - 0.707106781186547) / 2) - 1,
        solution[30] - 0.398942280401433 * exp(-((solution[2] - 1) ^ 2) / 2),
        solution[24] + (solution[2] - 1) ^ 2 / 2 + 0.918938533204673,
        (solution[34] - 0.25 * tanh(solution[2] - 1)) - 0.5,
        solution[18] + 1.4142135623731 * erfcinv(2 * solution[34]),
        solution[28] + 1.4142135623731 * erfcinv(2 * solution[34]),
        solution[36] + 1.4142135623731 * erfcinv(2 * solution[34]),
        (solution[32] + erfc(0.707106781186547 * solution[2] - 0.707106781186547) / 2) - 1,
        solution[8] - 0.398942280401433 * exp(-((solution[2] - 1) ^ 2) / 2),
        solution[43] - 0,
        0.5 * solution[55] - complete_parameters[15] * (1 - complete_parameters[17]),
        0.5 * solution[56] - complete_parameters[16] * (1 - complete_parameters[18]),
        -(solution[1]) * complete_parameters[24] * solution[21] ^ complete_parameters[19] + solution[47],
        -(complete_parameters[3]) * solution[47] + solution[5],
        -(complete_parameters[25]) * solution[47] + solution[19],
        ((-(solution[5]) - solution[11]) - solution[19]) + solution[47],
        (-(solution[19]) - solution[21] * (1 - complete_parameters[22])) + solution[21],
        (-(solution[1]) * complete_parameters[1] + solution[1] + complete_parameters[1]) - 1,
        -(complete_parameters[7]) + solution[38] + solution[58],
        (solution[38] + solution[57]) - 1 / solution[89],
        solution[49] - Max(solution[57], solution[58]),
        solution[49] - solution[65],
        -(complete_parameters[8]) + solution[44] + solution[64],
        (-(complete_parameters[9]) - complete_parameters[10] * solution[47]) + solution[44] + solution[63],
        solution[52] - Min(solution[63], solution[64]),
        solution[52] - solution[83],
        solution[40] - 3 * solution[47],
        solution[9] - 0,
        solution[13] - solution[15] ^ 3,
        -(complete_parameters[5]) + solution[15],
        (solution[3] + erfc(0.707106781186547 * solution[1] - 0.707106781186547) / 2) - 1,
        solution[29] - 0.398942280401433 * exp(-((solution[1] - 1) ^ 2) / 2),
        solution[23] + (solution[1] - 1) ^ 2 / 2 + 0.918938533204673,
        (solution[33] - 0.25 * tanh(solution[1] - 1)) - 0.5,
        solution[17] + 1.4142135623731 * erfcinv(2 * solution[33]),
        solution[27] + 1.4142135623731 * erfcinv(2 * solution[33]),
        solution[35] + 1.4142135623731 * erfcinv(2 * solution[33]),
        (solution[31] + erfc(0.707106781186547 * solution[1] - 0.707106781186547) / 2) - 1,
        solution[7] - 0.398942280401433 * exp(-((solution[1] - 1) ^ 2) / 2),
        solution[42] - 0,
        0.5 * solution[53] - complete_parameters[15] * (1 - complete_parameters[17]),
        0.5 * solution[54] - complete_parameters[16] * (1 - complete_parameters[18]),
        (solution[46] - solution[47]) - solution[48],
        solution[37] - 1,
        solution[26] - complete_parameters[13] * (-(solution[47]) + solution[48]),
        solution[25] - complete_parameters[14] * (solution[47] - solution[48]),
        solution[71] - solution[76],
        solution[75] - 0,
        solution[74] - solution[75],
        solution[72] - solution[74],
        -(solution[72]) + solution[73],
        -(solution[73]) + solution[76],
        solution[77] - solution[82],
        solution[81] - 0,
        solution[80] - solution[81],
        solution[78] - solution[80],
        -(solution[78]) + solution[79],
        -(solution[79]) + solution[82],
        solution[65] - solution[70],
        solution[69] - 0,
        solution[68] - solution[69],
        solution[66] - solution[68],
        -(solution[66]) + solution[67],
        -(solution[67]) + solution[70],
        solution[83] - solution[88],
        solution[87] - 0,
        solution[86] - solution[87],
        solution[84] - solution[86],
        -(solution[84]) + solution[85],
        -(solution[85]) + solution[88],
        solution[39] - complete_parameters[26],
        solution[48] - complete_parameters[20],
    ]
end

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) * solution[2] ^ solution[1] + previous_solution[3],
        solution[2] - min(1.0e12, max(eps(), previous_solution[2])),
    ]
end

function residuals_block_2(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        0.5 * solution[1] - complete_parameters[16] * (1 - complete_parameters[18]),
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        0.5 * solution[1] - complete_parameters[15] * (1 - complete_parameters[17]),
    ]
end

function residuals_block_4(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        0.5 * solution[1] - complete_parameters[16] * (1 - complete_parameters[18]),
    ]
end

function residuals_block_5(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        0.5 * solution[1] - complete_parameters[15] * (1 - complete_parameters[17]),
    ]
end

function residuals_block_6(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (solution[1] - previous_solution[1]) - previous_solution[2],
    ]
end

function residuals_block_7(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 3
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[8]) + solution[1] + solution[3],
        (-(complete_parameters[9]) - complete_parameters[10] * previous_solution[1]) + solution[1] + solution[2],
        previous_solution[2] - Min(solution[2], solution[3]),
    ]
end

function residuals_block_8(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_9(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
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
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_13(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_14(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0,
    ]
end

function residuals_block_15(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 3
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[8]) + solution[1] + solution[3],
        (-(complete_parameters[9]) - complete_parameters[10] * previous_solution[1]) + solution[1] + solution[2],
        previous_solution[2] - Min(solution[2], solution[3]),
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
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_21(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_22(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0,
    ]
end

function residuals_block_23(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0,
    ]
end

function residuals_block_24(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0,
    ]
end

function residuals_block_25(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 3 * previous_solution[1],
    ]
end

function residuals_block_26(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 3 * previous_solution[1],
    ]
end

function residuals_block_27(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 3
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[7]) + solution[1] + solution[3],
        (solution[1] + solution[2]) - 1 / previous_solution[1],
        previous_solution[2] - Max(solution[2], solution[3]),
    ]
end

function residuals_block_28(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (previous_solution[1] + previous_solution[2]) - 1 / solution[1],
    ]
end

function residuals_block_29(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        previous_solution[1] - Max(solution[1], previous_solution[2]),
    ]
end

function residuals_block_30(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[7]) + previous_solution[1] + solution[1],
    ]
end

function residuals_block_31(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_32(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
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
        solution[1] - previous_solution[1],
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
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0,
    ]
end

function residuals_block_38(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - complete_parameters[26],
    ]
end

function residuals_block_39(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_40(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
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
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_44(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1],
    ]
end

function residuals_block_45(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0,
    ]
end

function residuals_block_46(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 1,
    ]
end

function residuals_block_47(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] + 1.4142135623731 * erfcinv(solution[2]),
        solution[2] - min(2 - eps(), max(eps(), previous_solution[1] * 2)),
    ]
end

function residuals_block_48(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] + 1.4142135623731 * erfcinv(solution[2]),
        solution[2] - min(2 - eps(), max(eps(), previous_solution[1] * 2)),
    ]
end

function residuals_block_49(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (solution[1] + erfc(0.707106781186547 * previous_solution[1] - 0.707106781186547) / 2) - 1,
    ]
end

function residuals_block_50(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (solution[1] + erfc(0.707106781186547 * previous_solution[1] - 0.707106781186547) / 2) - 1,
    ]
end

function residuals_block_51(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0.398942280401433 * exp(solution[2]),
        solution[2] - min(600, max(-1.0e12, -((previous_solution[1] - 1) ^ 2) / 2)),
    ]
end

function residuals_block_52(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0.398942280401433 * exp(solution[2]),
        solution[2] - min(600, max(-1.0e12, -((previous_solution[1] - 1) ^ 2) / 2)),
    ]
end

function residuals_block_53(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] + 1.4142135623731 * erfcinv(solution[2]),
        solution[2] - min(2 - eps(), max(eps(), previous_solution[1] * 2)),
    ]
end

function residuals_block_54(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] + 1.4142135623731 * erfcinv(solution[2]),
        solution[2] - min(2 - eps(), max(eps(), previous_solution[1] * 2)),
    ]
end

function residuals_block_55(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - complete_parameters[13] * (-(previous_solution[1]) + previous_solution[2]),
    ]
end

function residuals_block_56(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - complete_parameters[14] * (previous_solution[1] - previous_solution[2]),
    ]
end

function residuals_block_57(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] + (previous_solution[1] - 1) ^ 2 / 2 + 0.918938533204673,
    ]
end

function residuals_block_58(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] + (previous_solution[1] - 1) ^ 2 / 2 + 0.918938533204673,
    ]
end

function residuals_block_59(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(previous_solution[1]) - solution[1] * (1 - complete_parameters[22])) + solution[1],
    ]
end

function residuals_block_60(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] + 1.4142135623731 * erfcinv(solution[2]),
        solution[2] - min(2 - eps(), max(eps(), previous_solution[1] * 2)),
    ]
end

function residuals_block_61(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (solution[1] - 0.25 * tanh(previous_solution[1] - 1)) - 0.5,
    ]
end

function residuals_block_62(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] + 1.4142135623731 * erfcinv(solution[2]),
        solution[2] - min(2 - eps(), max(eps(), previous_solution[1] * 2)),
    ]
end

function residuals_block_63(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (solution[1] - 0.25 * tanh(previous_solution[1] - 1)) - 0.5,
    ]
end

function residuals_block_64(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1] ^ 3,
    ]
end

function residuals_block_65(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[5]) + solution[1],
    ]
end

function residuals_block_66(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1] ^ 3,
    ]
end

function residuals_block_67(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[5]) + solution[1],
    ]
end

function residuals_block_68(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((-(previous_solution[1]) - solution[1]) - previous_solution[2]) + previous_solution[3],
    ]
end

function residuals_block_69(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[25]) * previous_solution[1] + solution[1],
    ]
end

function residuals_block_70(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 3
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((-(previous_solution[1]) - solution[1]) - previous_solution[2]) + previous_solution[3],
    ]
end

function residuals_block_71(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0,
    ]
end

function residuals_block_72(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0,
    ]
end

function residuals_block_73(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0.398942280401433 * exp(solution[2]),
        solution[2] - min(600, max(-1.0e12, -((previous_solution[1] - 1) ^ 2) / 2)),
    ]
end

function residuals_block_74(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0.398942280401433 * exp(solution[2]),
        solution[2] - min(600, max(-1.0e12, -((previous_solution[1] - 1) ^ 2) / 2)),
    ]
end

function residuals_block_75(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[3]) * previous_solution[1] + solution[1],
    ]
end

function residuals_block_76(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - complete_parameters[20],
    ]
end

function residuals_block_77(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[3]) * previous_solution[1] + solution[1],
    ]
end

function residuals_block_78(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 3
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[25]) * solution[3] + solution[1],
        (-(solution[1]) - solution[2] * (1 - complete_parameters[22])) + solution[2],
        -(previous_solution[1]) * complete_parameters[24] * solution[2] ^ complete_parameters[19] + solution[3],
    ]
end

function residuals_block_79(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (solution[1] + erfc(0.707106781186547 * previous_solution[1] - 0.707106781186547) / 2) - 1,
    ]
end

function residuals_block_80(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (solution[1] + erfc(0.707106781186547 * previous_solution[1] - 0.707106781186547) / 2) - 1,
    ]
end

function residuals_block_81(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[1]) * complete_parameters[1] + solution[1] + complete_parameters[1]) - 1,
    ]
end

function residuals_block_82(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[1]) * complete_parameters[1] + solution[1] + complete_parameters[1]) - 1,
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
        residuals_block_79(parameters, previous_solutions[79], external_solutions[79], solutions[79]),
        residuals_block_80(parameters, previous_solutions[80], external_solutions[80], solutions[80]),
        residuals_block_81(parameters, previous_solutions[81], external_solutions[81], solutions[81]),
        residuals_block_82(parameters, previous_solutions[82], external_solutions[82], solutions[82]),
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
export residuals_block_1, residuals_block_2, residuals_block_3, residuals_block_4, residuals_block_5, residuals_block_6, residuals_block_7, residuals_block_8, residuals_block_9, residuals_block_10, residuals_block_11, residuals_block_12, residuals_block_13, residuals_block_14, residuals_block_15, residuals_block_16, residuals_block_17, residuals_block_18, residuals_block_19, residuals_block_20, residuals_block_21, residuals_block_22, residuals_block_23, residuals_block_24, residuals_block_25, residuals_block_26, residuals_block_27, residuals_block_28, residuals_block_29, residuals_block_30, residuals_block_31, residuals_block_32, residuals_block_33, residuals_block_34, residuals_block_35, residuals_block_36, residuals_block_37, residuals_block_38, residuals_block_39, residuals_block_40, residuals_block_41, residuals_block_42, residuals_block_43, residuals_block_44, residuals_block_45, residuals_block_46, residuals_block_47, residuals_block_48, residuals_block_49, residuals_block_50, residuals_block_51, residuals_block_52, residuals_block_53, residuals_block_54, residuals_block_55, residuals_block_56, residuals_block_57, residuals_block_58, residuals_block_59, residuals_block_60, residuals_block_61, residuals_block_62, residuals_block_63, residuals_block_64, residuals_block_65, residuals_block_66, residuals_block_67, residuals_block_68, residuals_block_69, residuals_block_70, residuals_block_71, residuals_block_72, residuals_block_73, residuals_block_74, residuals_block_75, residuals_block_76, residuals_block_77, residuals_block_78, residuals_block_79, residuals_block_80, residuals_block_81, residuals_block_82
end
