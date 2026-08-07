module Ghironi_Melitz_2005NsssResiduals
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

const MODEL_NAME = "Ghironi_Melitz_2005"
const SOURCE_MODEL_FILE = "models/Ghironi_Melitz_2005.jl"
const NSSS_SOLUTION_ERROR = 4.108051741904708e-15
const NSSS_RESIDUAL_NORM = 1.3546313339843274e-15

const PARAMETER_NAMES = [
    "σᶻ",
    "σᶻ̄",
    "β",
    "γ",
    "δ",
    "θ",
    "k",
    "τ",
    "zmin",
    "zmin̄",
    "fe",
    "fē",
    "L",
    "L̄",
    "ρZ",
    "ρZ̄",
    "fx_share",
]
const PARAMETER_VALUES = Float64[
    0.01,
    0.01,
    0.99,
    2.0,
    0.025,
    3.8,
    3.4,
    1.3,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.9,
    0.9,
    0.235,
]
const COMPLETE_PARAMETER_NAMES = [
    "σᶻ",
    "σᶻ̄",
    "β",
    "γ",
    "δ",
    "θ",
    "k",
    "τ",
    "zmin",
    "zmin̄",
    "fe",
    "fē",
    "L",
    "L̄",
    "ρZ",
    "ρZ̄",
    "fx_share",
    "fx",
    "fx̄",
    "z̃d",
    "z̃d̄",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    0.01,
    0.01,
    0.99,
    2.0,
    0.025,
    3.8,
    3.4,
    1.3,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.9,
    0.9,
    0.235,
    0.008460243460243475,
    0.008460243460243475,
    1.8579995104777778,
    1.8579995104777778,
]
const ORIGINAL_SOLUTION_NAMES = [
    "C",
    "C̄",
    "Nd",
    "Nd̄",
    "Ne",
    "Nx",
    "Nx̄",
    "Nē",
    "Q",
    "Q̃",
    "TOL",
    "Z",
    "Z̄",
    "d̃",
    "d̃d",
    "d̃d̄",
    "d̃x",
    "d̃x̄",
    "d̃̄",
    "r",
    "r̄",
    "w",
    "w̄",
    "zx",
    "zx̄",
    "z̃x",
    "z̃x̄",
    "ρ̃d",
    "ρ̃d̄",
    "ρ̃x",
    "ρ̃x̄",
    "ṽ",
    "ṽ̄",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    3.3868824077313877,
    3.38688240773139,
    7.5069526507064035,
    7.506952650706411,
    0.19248596540272908,
    1.5798796065733554,
    1.5798796065733547,
    0.19248596540272866,
    0.9999999999999999,
    1.0000000000000002,
    1.0,
    1.0,
    1.0,
    0.11313270650973078,
    0.08702172863310761,
    0.08702172863310764,
    0.12406886813900492,
    0.12406886813900493,
    0.11313270650973078,
    0.01010101010101011,
    0.01010101010101011,
    3.1424847470077033,
    3.142484747007704,
    1.5815047299285394,
    1.58150472992854,
    2.938435014025517,
    2.938435014025518,
    2.2953723636801207,
    2.2953723636801207,
    1.8868005996535897,
    1.8868005996535888,
    3.1424847470077033,
    3.142484747007704,
]
const AUXILIARY_SOLUTION_NAMES = [
    "C",
    "C̄",
    "Nd",
    "Nd̄",
    "Ne",
    "Nx",
    "Nx̄",
    "Nē",
    "Q",
    "Q̃",
    "TOL",
    "Z",
    "Z̄",
    "d̃",
    "d̃d",
    "d̃d̄",
    "d̃x",
    "d̃x̄",
    "d̃̄",
    "r",
    "r̄",
    "w",
    "w̄",
    "zx",
    "zx̄",
    "z̃x",
    "z̃x̄",
    "ρ̃d",
    "ρ̃d̄",
    "ρ̃x",
    "ρ̃x̄",
    "ṽ",
    "ṽ̄",
    "➕₁",
    "➕₁₀",
    "➕₁₁",
    "➕₁₂",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
    "➕₆",
    "➕₇",
    "➕₈",
    "➕₉",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    3.3868824077313877,
    3.38688240773139,
    7.5069526507064035,
    7.506952650706411,
    0.19248596540272908,
    1.5798796065733554,
    1.5798796065733547,
    0.19248596540272866,
    0.9999999999999999,
    1.0000000000000002,
    1.0,
    1.0,
    1.0,
    0.11313270650973078,
    0.08702172863310761,
    0.08702172863310764,
    0.124068868139005,
    0.12406886813900504,
    0.11313270650973078,
    0.01010101010101011,
    0.01010101010101011,
    3.1424847470077033,
    3.142484747007704,
    1.5815047299285396,
    1.5815047299285403,
    2.9384350140255173,
    2.9384350140255187,
    2.2953723636801207,
    2.295372363680121,
    1.8868005996535895,
    1.8868005996535888,
    3.1424847470077033,
    3.142484747007704,
    0.3403172080467579,
    20.451494655678342,
    3.142484747007704,
    20.45149465567836,
    5.666666666666666,
    0.34031720804675775,
    0.8220019677454524,
    0.8220019677454521,
    0.9999999999999994,
    0.999999999999999,
    3.1424847470077033,
    1.3571428571428572,
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
]
const ALL_AUXILIARY_VARIABLE_VALUES = Float64[
    0.3403172080467579,
    5.666666666666666,
    0.34031720804675775,
    0.8220019677454524,
    0.8220019677454521,
    0.9999999999999994,
    0.999999999999999,
    3.1424847470077033,
    1.3571428571428572,
    20.451494655678342,
    3.142484747007704,
    20.45149465567836,
    1.3,
    1.3,
    3.3868824077313877,
    3.38688240773139,
]
const DEFAULTED_NSSS_SOLUTION_NAMES = [
]
const CALIBRATION_PARAMETER_NAMES = [
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(1 - (Nd * ρ̃d ^ (1 - θ) + Nx̄ * ρ̃x̄ ^ (1 - θ))),
    :(1 - (Nd̄ * ρ̃d̄ ^ (1 - θ) + Nx * ρ̃x ^ (1 - θ))),
    :(ρ̃d - ((θ / (θ - 1)) * w) / (Z * z̃d)),
    :(ρ̃d̄ - ((θ / (θ - 1)) * w̄) / (Z̄ * z̃d̄)),
    :(ρ̃x - (((θ / (θ - 1)) * τ * w) / (Z * z̃x)) / Q),
    :(ρ̃x̄ - (((Q * θ) / (θ - 1)) * τ * w̄) / (Z̄ * z̃x̄)),
    :(d̃ - (d̃d + (Nx / Nd) * d̃x)),
    :(d̃̄ - (d̃d̄ + (Nx̄ / Nd̄) * d̃x̄)),
    :(d̃d - ((ρ̃d ^ (1 - θ) * 1) / θ) * C),
    :(d̃d̄ - ((ρ̃d̄ ^ (1 - θ) * 1) / θ) * C̄),
    :(ṽ - (w * fe) / Z),
    :(ṽ̄ - (w̄ * fē) / Z̄),
    :(d̃x - (((w * fx) / Z) * (θ - 1)) / (k - (θ - 1))),
    :(d̃x̄ - (((θ - 1) / (k - (θ - 1))) * w̄ * fx̄) / Z̄),
    :(Nx / Nd - (zmin / z̃x) ^ k * (k / (k - (θ - 1))) ^ (k / (θ - 1))),
    :(Nx̄ / Nd̄ - (k / (k - (θ - 1))) ^ (k / (θ - 1)) * (zmin̄ / z̃x̄) ^ k),
    :(Nd - (1 - δ) * (Nd + Ne)),
    :(Nd̄ - (1 - δ) * (Nd̄ + Nē)),
    :(C ^ -γ - β * (1 + r) * C ^ -γ),
    :(C̄ ^ -γ - β * (1 + r̄) * C̄ ^ -γ),
    :(ṽ - (1 - δ) * β * (C / C) ^ -γ * (ṽ + d̃)),
    :(ṽ̄ - (1 - δ) * β * (C̄ / C̄) ^ -γ * (ṽ̄ + d̃̄)),
    :(C - ((w * L + Nd * d̃) - ṽ * Ne)),
    :(C̄ - ((w̄ * L̄ + Nd̄ * d̃̄) - ṽ̄ * Nē)),
    :(Q - (Nx̄ * ρ̃x̄ ^ (1 - θ) * C) / (Nx * ρ̃x ^ (1 - θ) * C̄)),
    :(Q̃ - (((Nd̄ / (Nd̄ + Nx)) * TOL ^ (1 - θ) + (Nx / (Nd̄ + Nx)) * ((τ * z̃d) / z̃x) ^ (1 - θ)) / (Nd / (Nd + Nx̄) + (Nx̄ / (Nd + Nx̄)) * ((τ * TOL * z̃d̄) / z̃x̄) ^ (1 - θ))) ^ (1 / (1 - θ))),
    :(Q̃ - Q * ((Nd + Nx̄) / (Nd̄ + Nx)) ^ (-1 / (θ - 1))),
    :(Z - ((1 - ρZ) * 1.0 + ρZ * Z + σᶻ * 0)),
    :(Z̄ - (1.0 * (1 - ρZ̄) + ρZ̄ * Z̄ + σᶻ̄ * 0)),
    :(z̃x - (θ * fx * (w / Z) ^ θ * (1 + (θ - 1) / (k - (θ - 1))) * Q ^ -θ * τ ^ (θ - 1) * (θ / (θ - 1)) ^ (θ - 1) * C̄ ^ -1) ^ (1 / (θ - 1))),
    :(z̃x̄ - ((θ / (θ - 1)) ^ (θ - 1) * θ * τ ^ (θ - 1) * (1 + (θ - 1) / (k - (θ - 1))) * fx̄ * (w̄ / Z̄) ^ θ * Q ^ θ * C ^ -1) ^ (1 / (θ - 1))),
    :(zx - z̃x / (k / (k - (θ - 1))) ^ (1 / (θ - 1))),
    :(zx̄ - z̃x̄ / (k / (k - (θ - 1))) ^ (1 / (θ - 1))),
]
const CALIBRATION_EQUATIONS = Expr[
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :((-Nd * ρ̃d ^ (1 - θ) - Nx̄ * ρ̃x̄ ^ (1 - θ)) + 1),
    :((-Nd̄ * ρ̃d̄ ^ (1 - θ) - Nx * ρ̃x ^ (1 - θ)) + 1),
    :(ρ̃d - (w * θ) / (Z * z̃d * (θ - 1))),
    :(ρ̃d̄ - (w̄ * θ) / (Z̄ * z̃d̄ * (θ - 1))),
    :(ρ̃x - (w * θ * τ) / (Q * Z * z̃x * (θ - 1))),
    :((-Q * w̄ * θ * τ) / (Z̄ * z̃x̄ * (θ - 1)) + ρ̃x̄),
    :((d̃ - d̃d) - (Nx * d̃x) / Nd),
    :((-d̃d̄ + d̃̄) - (Nx̄ * d̃x̄) / Nd̄),
    :((-C * ρ̃d ^ (1 - θ)) / θ + d̃d),
    :((-C̄ * ρ̃d̄ ^ (1 - θ)) / θ + d̃d̄),
    :(ṽ - (fe * w) / Z),
    :(ṽ̄ - (fē * w̄) / Z̄),
    :(d̃x - (fx * w * (θ - 1)) / (Z * ((k - θ) + 1))),
    :(d̃x̄ - (fx̄ * w̄ * (θ - 1)) / (Z̄ * ((k - θ) + 1))),
    :(➕₁ - zmin / z̃x),
    :(➕₂ - k / ((k - θ) + 1)),
    :(-(➕₁ ^ k) * ➕₂ ^ (k / (θ - 1)) + Nx / Nd),
    :(➕₃ - zmin̄ / z̃x̄),
    :(-(➕₂ ^ (k / (θ - 1))) * ➕₃ ^ k + Nx̄ / Nd̄),
    :(Nd - (1 - δ) * (Nd + Ne)),
    :(Nd̄ - (1 - δ) * (Nd̄ + Nē)),
    :((-β * (r + 1)) / C ^ γ + C ^ -γ),
    :((-β * (r̄ + 1)) / C̄ ^ γ + C̄ ^ -γ),
    :(-β * (1 - δ) * (d̃ + ṽ) + ṽ),
    :(-β * (1 - δ) * (d̃̄ + ṽ̄) + ṽ̄),
    :(((C - L * w) - Nd * d̃) + Ne * ṽ),
    :(((C̄ - L̄ * w̄) - Nd̄ * d̃̄) + Nē * ṽ̄),
    :((-C * Nx̄ * ρ̃x ^ (θ - 1) * ρ̃x̄ ^ (1 - θ)) / (C̄ * Nx) + Q),
    :(➕₄ - (z̃d * τ) / z̃x),
    :(➕₅ - (TOL * z̃d̄ * τ) / z̃x̄),
    :(➕₆ - ((Nd̄ * TOL ^ (1 - θ)) / (Nd̄ + Nx) + (Nx * ➕₄ ^ (1 - θ)) / (Nd̄ + Nx)) / (Nd / (Nd + Nx̄) + (Nx̄ * ➕₅ ^ (1 - θ)) / (Nd + Nx̄))),
    :(Q̃ - ➕₆ ^ (1 / (1 - θ))),
    :(➕₇ - (Nd + Nx̄) / (Nd̄ + Nx)),
    :(-Q / ➕₇ ^ (1 / (θ - 1)) + Q̃),
    :((-Z * ρZ + Z + 1.0ρZ) - 1.0),
    :((-Z̄ * ρZ̄ + Z̄ + 1.0ρZ̄) - 1.0),
    :(➕₈ - w / Z),
    :(➕₉ - θ / (θ - 1)),
    :(➕₁₀ - (fx * θ * τ ^ (θ - 1) * ➕₈ ^ θ * ➕₉ ^ (θ - 1) * ((θ - 1) / ((k - θ) + 1) + 1)) / (C̄ * Q ^ θ)),
    :(z̃x - ➕₁₀ ^ (1 / (θ - 1))),
    :(➕₁₁ - w̄ / Z̄),
    :(➕₁₂ - (Q ^ θ * fx̄ * θ * τ ^ (θ - 1) * ➕₁₁ ^ θ * ➕₉ ^ (θ - 1) * ((θ - 1) / ((k - θ) + 1) + 1)) / C),
    :(z̃x̄ - ➕₁₂ ^ (1 / (θ - 1))),
    :(zx - z̃x / ➕₂ ^ (1 / (θ - 1))),
    :(zx̄ - z̃x̄ / ➕₂ ^ (1 / (θ - 1))),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(1 - (Nd * ρ̃d ^ (1 - θ) + Nx̄ * ρ̃x̄ ^ (1 - θ))),
    :(1 - (Nd̄ * ρ̃d̄ ^ (1 - θ) + Nx * ρ̃x ^ (1 - θ))),
    :(ρ̃d - ((θ / (θ - 1)) * w) / (Z * z̃d)),
    :(ρ̃d̄ - ((θ / (θ - 1)) * w̄) / (Z̄ * z̃d̄)),
    :(ρ̃x - (((θ / (θ - 1)) * τ * w) / (Z * z̃x)) / Q),
    :(ρ̃x̄ - (((Q * θ) / (θ - 1)) * τ * w̄) / (Z̄ * z̃x̄)),
    :(d̃ - (d̃d + (Nx / Nd) * d̃x)),
    :(d̃̄ - (d̃d̄ + (Nx̄ / Nd̄) * d̃x̄)),
    :(d̃d - ((ρ̃d ^ (1 - θ) * 1) / θ) * C),
    :(d̃d̄ - ((ρ̃d̄ ^ (1 - θ) * 1) / θ) * C̄),
    :(ṽ - (w * fe) / Z),
    :(ṽ̄ - (w̄ * fē) / Z̄),
    :(d̃x - (((w * fx) / Z) * (θ - 1)) / (k - (θ - 1))),
    :(d̃x̄ - (((θ - 1) / (k - (θ - 1))) * w̄ * fx̄) / Z̄),
    :(Nx / Nd - (zmin / z̃x) ^ k * (k / (k - (θ - 1))) ^ (k / (θ - 1))),
    :(Nx̄ / Nd̄ - (k / (k - (θ - 1))) ^ (k / (θ - 1)) * (zmin̄ / z̃x̄) ^ k),
    :(Nd - (1 - δ) * (Nd + Ne)),
    :(Nd̄ - (1 - δ) * (Nd̄ + Nē)),
    :(C ^ -γ - β * (1 + r) * C ^ -γ),
    :(C̄ ^ -γ - β * (1 + r̄) * C̄ ^ -γ),
    :(ṽ - (1 - δ) * β * (C / C) ^ -γ * (ṽ + d̃)),
    :(ṽ̄ - (1 - δ) * β * (C̄ / C̄) ^ -γ * (ṽ̄ + d̃̄)),
    :(C - ((w * L + Nd * d̃) - ṽ * Ne)),
    :(C̄ - ((w̄ * L̄ + Nd̄ * d̃̄) - ṽ̄ * Nē)),
    :(Q - (Nx̄ * ρ̃x̄ ^ (1 - θ) * C) / (Nx * ρ̃x ^ (1 - θ) * C̄)),
    :(Q̃ - (((Nd̄ / (Nd̄ + Nx)) * TOL ^ (1 - θ) + (Nx / (Nd̄ + Nx)) * ((τ * z̃d) / z̃x) ^ (1 - θ)) / (Nd / (Nd + Nx̄) + (Nx̄ / (Nd + Nx̄)) * ((τ * TOL * z̃d̄) / z̃x̄) ^ (1 - θ))) ^ (1 / (1 - θ))),
    :(Q̃ - Q * ((Nd + Nx̄) / (Nd̄ + Nx)) ^ (-1 / (θ - 1))),
    :(Z - ((1 - ρZ) * 1.0 + ρZ * Z + σᶻ * 0)),
    :(Z̄ - (1.0 * (1 - ρZ̄) + ρZ̄ * Z̄ + σᶻ̄ * 0)),
    :(z̃x - (θ * fx * (w / Z) ^ θ * (1 + (θ - 1) / (k - (θ - 1))) * Q ^ -θ * τ ^ (θ - 1) * (θ / (θ - 1)) ^ (θ - 1) * C̄ ^ -1) ^ (1 / (θ - 1))),
    :(z̃x̄ - ((θ / (θ - 1)) ^ (θ - 1) * θ * τ ^ (θ - 1) * (1 + (θ - 1) / (k - (θ - 1))) * fx̄ * (w̄ / Z̄) ^ θ * Q ^ θ * C ^ -1) ^ (1 / (θ - 1))),
    :(zx - z̃x / (k / (k - (θ - 1))) ^ (1 / (θ - 1))),
    :(zx̄ - z̃x̄ / (k / (k - (θ - 1))) ^ (1 / (θ - 1))),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :((-Nd * ρ̃d ^ (1 - θ) - Nx̄ * ρ̃x̄ ^ (1 - θ)) + 1),
    :((-Nd̄ * ρ̃d̄ ^ (1 - θ) - Nx * ρ̃x ^ (1 - θ)) + 1),
    :(ρ̃d - (w * θ) / (Z * z̃d * (θ - 1))),
    :(ρ̃d̄ - (w̄ * θ) / (Z̄ * z̃d̄ * (θ - 1))),
    :(ρ̃x - (w * θ * τ) / (Q * Z * z̃x * (θ - 1))),
    :((-Q * w̄ * θ * τ) / (Z̄ * z̃x̄ * (θ - 1)) + ρ̃x̄),
    :((d̃ - d̃d) - (Nx * d̃x) / Nd),
    :((-d̃d̄ + d̃̄) - (Nx̄ * d̃x̄) / Nd̄),
    :((-C * ρ̃d ^ (1 - θ)) / θ + d̃d),
    :((-C̄ * ρ̃d̄ ^ (1 - θ)) / θ + d̃d̄),
    :(ṽ - (fe * w) / Z),
    :(ṽ̄ - (fē * w̄) / Z̄),
    :(d̃x - (fx * w * (θ - 1)) / (Z * ((k - θ) + 1))),
    :(d̃x̄ - (fx̄ * w̄ * (θ - 1)) / (Z̄ * ((k - θ) + 1))),
    :(➕₁ - zmin / z̃x),
    :(➕₂ - k / ((k - θ) + 1)),
    :(-(➕₁ ^ k) * ➕₂ ^ (k / (θ - 1)) + Nx / Nd),
    :(➕₃ - zmin̄ / z̃x̄),
    :(-(➕₂ ^ (k / (θ - 1))) * ➕₃ ^ k + Nx̄ / Nd̄),
    :(Nd - (1 - δ) * (Nd + Ne)),
    :(Nd̄ - (1 - δ) * (Nd̄ + Nē)),
    :((-β * (r + 1)) / C ^ γ + C ^ -γ),
    :((-β * (r̄ + 1)) / C̄ ^ γ + C̄ ^ -γ),
    :(-β * (1 - δ) * (d̃ + ṽ) + ṽ),
    :(-β * (1 - δ) * (d̃̄ + ṽ̄) + ṽ̄),
    :(((C - L * w) - Nd * d̃) + Ne * ṽ),
    :(((C̄ - L̄ * w̄) - Nd̄ * d̃̄) + Nē * ṽ̄),
    :((-C * Nx̄ * ρ̃x ^ (θ - 1) * ρ̃x̄ ^ (1 - θ)) / (C̄ * Nx) + Q),
    :(➕₄ - (z̃d * τ) / z̃x),
    :(➕₅ - (TOL * z̃d̄ * τ) / z̃x̄),
    :(➕₆ - ((Nd̄ * TOL ^ (1 - θ)) / (Nd̄ + Nx) + (Nx * ➕₄ ^ (1 - θ)) / (Nd̄ + Nx)) / (Nd / (Nd + Nx̄) + (Nx̄ * ➕₅ ^ (1 - θ)) / (Nd + Nx̄))),
    :(Q̃ - ➕₆ ^ (1 / (1 - θ))),
    :(➕₇ - (Nd + Nx̄) / (Nd̄ + Nx)),
    :(-Q / ➕₇ ^ (1 / (θ - 1)) + Q̃),
    :((-Z * ρZ + Z + 1.0ρZ) - 1.0),
    :((-Z̄ * ρZ̄ + Z̄ + 1.0ρZ̄) - 1.0),
    :(➕₈ - w / Z),
    :(➕₉ - θ / (θ - 1)),
    :(➕₁₀ - (fx * θ * τ ^ (θ - 1) * ➕₈ ^ θ * ➕₉ ^ (θ - 1) * ((θ - 1) / ((k - θ) + 1) + 1)) / (C̄ * Q ^ θ)),
    :(z̃x - ➕₁₀ ^ (1 / (θ - 1))),
    :(➕₁₁ - w̄ / Z̄),
    :(➕₁₂ - (Q ^ θ * fx̄ * θ * τ ^ (θ - 1) * ➕₁₁ ^ θ * ➕₉ ^ (θ - 1) * ((θ - 1) / ((k - θ) + 1) + 1)) / C),
    :(z̃x̄ - ➕₁₂ ^ (1 / (θ - 1))),
    :(zx - z̃x / ➕₂ ^ (1 / (θ - 1))),
    :(zx̄ - z̃x̄ / ➕₂ ^ (1 / (θ - 1))),
]

const PARAMETER_DEFINITION_NAMES = [
    "fx",
    "fx̄",
    "z̃d",
    "z̃d̄",
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
    "((fx_share * (1 - β * (1 - δ))) / (β * (1 - δ))) * fe",
    "((fx_share * (1 - β * (1 - δ))) / (β * (1 - δ))) * fē",
    "(k / (k - (θ - 1))) ^ (1 / (θ - 1)) * zmin",
    "(k / (k - (θ - 1))) ^ (1 / (θ - 1)) * zmin̄",
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "σᶻ",
    "σᶻ̄",
    "β",
    "γ",
    "δ",
    "θ",
    "k",
    "τ",
    "zmin",
    "zmin̄",
    "fe",
    "fē",
    "L",
    "L̄",
    "ρZ",
    "ρZ̄",
    "fx_share",
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
]
const ORIGINAL_BOX_CONSTRAINT_NAMES = [
    "C",
    "C̄",
    "Nd",
    "Nd̄",
    "Ne",
    "Nx",
    "Nx̄",
    "Nē",
    "Q",
    "Q̃",
    "TOL",
    "Z",
    "Z̄",
    "d̃",
    "d̃d",
    "d̃d̄",
    "d̃x",
    "d̃x̄",
    "d̃̄",
    "r",
    "r̄",
    "w",
    "w̄",
    "zx",
    "zx̄",
    "z̃x",
    "z̃x̄",
    "ρ̃d",
    "ρ̃d̄",
    "ρ̃x",
    "ρ̃x̄",
    "ṽ",
    "ṽ̄",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    -Inf,
    2.220446049250313e-16,
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
    -1.0e12,
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
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
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
    Inf,
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
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "C",
    "C̄",
    "Nd",
    "Nd̄",
    "Ne",
    "Nx",
    "Nx̄",
    "Nē",
    "Q",
    "Q̃",
    "TOL",
    "Z",
    "Z̄",
    "d̃",
    "d̃d",
    "d̃d̄",
    "d̃x",
    "d̃x̄",
    "d̃̄",
    "r",
    "r̄",
    "w",
    "w̄",
    "zx",
    "zx̄",
    "z̃x",
    "z̃x̄",
    "ρ̃d",
    "ρ̃d̄",
    "ρ̃x",
    "ρ̃x̄",
    "ṽ",
    "ṽ̄",
    "➕₁",
    "➕₁₀",
    "➕₁₁",
    "➕₁₂",
    "➕₂",
    "➕₃",
    "➕₄",
    "➕₅",
    "➕₆",
    "➕₇",
    "➕₈",
    "➕₉",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    2.220446049250313e-16,
    -Inf,
    2.220446049250313e-16,
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
    -1.0e12,
    -1.0e12,
    -Inf,
    -Inf,
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
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
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
    "➕₁₃",
    "➕₁₄",
    "➕₁₅",
    "➕₁₆",
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
    1.0e12,
    1.0e12,
    1.0e12,
    1.0e12,
]

const BLOCKS = [
    (
        index = 1,
        solve_order = 14,
        variables = ["zx̄"],
        previous_solution_names = ["z̃x̄", "➕₂"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [45],
        equations = Expr[
            :(zx̄ - z̃x̄ / ➕₂ ^ (1 / (θ - 1))),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["zx̄"],
        previous_solution_values = [2.9384350140255187, 5.666666666666666],
        external_solution_values = Float64[],
        solution_values = [1.5815047299285403],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 2,
        solve_order = 13,
        variables = ["zx"],
        previous_solution_names = ["z̃x", "➕₂"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [44],
        equations = Expr[
            :(zx - z̃x / ➕₂ ^ (1 / (θ - 1))),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["zx"],
        previous_solution_values = [2.9384350140255173, 5.666666666666666],
        external_solution_values = Float64[],
        solution_values = [1.5815047299285396],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 3,
        solve_order = 12,
        variables = ["r̄"],
        previous_solution_names = ["C̄"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₆"],
        equation_indices = [23],
        equations = Expr[
            :((-β * (r̄ + 1)) / ➕₁₆ ^ γ + ➕₁₆ ^ -γ),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₆ = min(1.0e12, max(eps(), C̄))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₆ - C̄)),
        ],
        solution_names = ["r̄", "➕₁₆"],
        previous_solution_values = [3.38688240773139],
        external_solution_values = Float64[],
        solution_values = [0.01010101010101011, 3.38688240773139],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 4,
        solve_order = 11,
        variables = ["r"],
        previous_solution_names = ["C"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₅"],
        equation_indices = [22],
        equations = Expr[
            :((-β * (r + 1)) / ➕₁₅ ^ γ + ➕₁₅ ^ -γ),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₅ = min(1.0e12, max(eps(), C))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₅ - C)),
        ],
        solution_names = ["r", "➕₁₅"],
        previous_solution_values = [3.3868824077313877],
        external_solution_values = Float64[],
        solution_values = [0.01010101010101011, 3.3868824077313877],
        box_lower_bounds = [-Inf, 2.220446049250313e-16],
        box_upper_bounds = [Inf, 1.0e12],
    ),
    (
        index = 5,
        solve_order = 10,
        variables = ["TOL", "➕₅"],
        previous_solution_names = ["Nd", "Nd̄", "Nx", "Nx̄", "z̃x̄", "➕₄", "➕₆"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [30, 31],
        equations = Expr[
            :(➕₅ - (TOL * z̃d̄ * τ) / z̃x̄),
            :(➕₆ - ((Nd̄ * TOL ^ (1 - θ)) / (Nd̄ + Nx) + (Nx * ➕₄ ^ (1 - θ)) / (Nd̄ + Nx)) / (Nd / (Nd + Nx̄) + (Nx̄ * ➕₅ ^ (1 - θ)) / (Nd + Nx̄))),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["TOL", "➕₅"],
        previous_solution_values = [7.5069526507064035, 7.506952650706411, 1.5798796065733554, 1.5798796065733547, 2.9384350140255187, 0.8220019677454524, 0.9999999999999994],
        external_solution_values = Float64[],
        solution_values = [1.0, 0.8220019677454521],
        box_lower_bounds = [2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12],
    ),
    (
        index = 6,
        solve_order = 9,
        variables = ["➕₆"],
        previous_solution_names = ["Q̃"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [32],
        equations = Expr[
            :(Q̃ - ➕₆ ^ (1 / (1 - θ))),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["➕₆"],
        previous_solution_values = [1.0000000000000002],
        external_solution_values = Float64[],
        solution_values = [0.9999999999999994],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 7,
        solve_order = 8,
        variables = ["➕₄"],
        previous_solution_names = ["z̃x"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [29],
        equations = Expr[
            :(➕₄ - (z̃d * τ) / z̃x),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["➕₄"],
        previous_solution_values = [2.9384350140255173],
        external_solution_values = Float64[],
        solution_values = [0.8220019677454524],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 8,
        solve_order = 7,
        variables = ["Q̃"],
        previous_solution_names = ["Q", "➕₇"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [34],
        equations = Expr[
            :(-Q / ➕₇ ^ (1 / (θ - 1)) + Q̃),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["Q̃"],
        previous_solution_values = [0.9999999999999999, 0.999999999999999],
        external_solution_values = Float64[],
        solution_values = [1.0000000000000002],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 9,
        solve_order = 6,
        variables = ["➕₇"],
        previous_solution_names = ["Nd", "Nd̄", "Nx", "Nx̄"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [33],
        equations = Expr[
            :(➕₇ - (Nd + Nx̄) / (Nd̄ + Nx)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["➕₇"],
        previous_solution_values = [7.5069526507064035, 7.506952650706411, 1.5798796065733554, 1.5798796065733547],
        external_solution_values = Float64[],
        solution_values = [0.999999999999999],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 10,
        solve_order = 5,
        variables = ["C", "C̄", "Nd", "Nd̄", "Ne", "Nx", "Nx̄", "Nē", "Q", "d̃", "d̃d", "d̃d̄", "d̃x", "d̃x̄", "d̃̄", "w", "w̄", "z̃x", "z̃x̄", "ρ̃d", "ρ̃d̄", "ρ̃x", "ρ̃x̄", "ṽ", "ṽ̄", "➕₁", "➕₁₀", "➕₁₁", "➕₁₂", "➕₃", "➕₈"],
        previous_solution_names = ["Z", "Z̄", "➕₂", "➕₉"],
        external_solution_names = String[],
        domain_auxiliary_names = ["➕₁₄"],
        equation_indices = [9, 27, 17, 19, 20, 2, 8, 21, 5, 24, 7, 10, 13, 14, 25, 11, 6, 40, 43, 3, 4, 28, 1, 26, 12, 15, 39, 41, 42, 18, 37],
        equations = Expr[
            :((-C * ρ̃d ^ (1 - θ)) / θ + d̃d),
            :(((C̄ - L̄ * w̄) - Nd̄ * d̃̄) + Nē * ṽ̄),
            :(-(➕₁ ^ k) * ➕₂ ^ (k / (θ - 1)) + Nx / Nd),
            :(-(➕₂ ^ (k / (θ - 1))) * ➕₃ ^ k + Nx̄ / Nd̄),
            :(Nd - (1 - δ) * (Nd + Ne)),
            :((-Nd̄ * ρ̃d̄ ^ (1 - θ) - Nx * ρ̃x ^ (1 - θ)) + 1),
            :((-d̃d̄ + d̃̄) - (Nx̄ * d̃x̄) / Nd̄),
            :(Nd̄ - (1 - δ) * (Nd̄ + Nē)),
            :(ρ̃x - (w * θ * τ) / (Q * Z * z̃x * (θ - 1))),
            :(-β * (1 - δ) * (d̃ + ṽ) + ṽ),
            :((d̃ - d̃d) - (Nx * d̃x) / Nd),
            :((-C̄ * ρ̃d̄ ^ (1 - θ)) / θ + d̃d̄),
            :(d̃x - (fx * w * (θ - 1)) / (Z * ((k - θ) + 1))),
            :(d̃x̄ - (fx̄ * w̄ * (θ - 1)) / (Z̄ * ((k - θ) + 1))),
            :(-β * (1 - δ) * (d̃̄ + ṽ̄) + ṽ̄),
            :(ṽ - (fe * w) / Z),
            :((-Q * w̄ * θ * τ) / (Z̄ * z̃x̄ * (θ - 1)) + ρ̃x̄),
            :(z̃x - ➕₁₀ ^ (1 / (θ - 1))),
            :(z̃x̄ - ➕₁₂ ^ (1 / (θ - 1))),
            :(ρ̃d - (w * θ) / (Z * z̃d * (θ - 1))),
            :(ρ̃d̄ - (w̄ * θ) / (Z̄ * z̃d̄ * (θ - 1))),
            :((-C * Nx̄ * ρ̃x ^ (θ - 1) * ρ̃x̄ ^ (1 - θ)) / (C̄ * Nx) + Q),
            :((-Nd * ρ̃d ^ (1 - θ) - Nx̄ * ρ̃x̄ ^ (1 - θ)) + 1),
            :(((C - L * w) - Nd * d̃) + Ne * ṽ),
            :(ṽ̄ - (fē * w̄) / Z̄),
            :(➕₁ - zmin / z̃x),
            :(➕₁₀ - (fx * θ * ➕₁₄ ^ (θ - 1) * ➕₈ ^ θ * ➕₉ ^ (θ - 1) * ((θ - 1) / ((k - θ) + 1) + 1)) / (C̄ * Q ^ θ)),
            :(➕₁₁ - w̄ / Z̄),
            :(➕₁₂ - (Q ^ θ * fx̄ * θ * ➕₁₄ ^ (θ - 1) * ➕₁₁ ^ θ * ➕₉ ^ (θ - 1) * ((θ - 1) / ((k - θ) + 1) + 1)) / C),
            :(➕₃ - zmin̄ / z̃x̄),
            :(➕₈ - w / Z),
        ],
        domain_auxiliary_equations = Expr[
            :(➕₁₄ = min(1.0e12, max(eps(), τ))),
        ],
        domain_auxiliary_error_equations = Expr[
            :(abs(➕₁₄ - τ)),
        ],
        solution_names = ["C", "C̄", "Nd", "Nd̄", "Ne", "Nx", "Nx̄", "Nē", "Q", "d̃", "d̃d", "d̃d̄", "d̃x", "d̃x̄", "d̃̄", "w", "w̄", "z̃x", "z̃x̄", "ρ̃d", "ρ̃d̄", "ρ̃x", "ρ̃x̄", "ṽ", "ṽ̄", "➕₁", "➕₁₀", "➕₁₁", "➕₁₂", "➕₃", "➕₈", "➕₁₄"],
        previous_solution_values = [1.0, 1.0, 5.666666666666666, 1.3571428571428572],
        external_solution_values = Float64[],
        solution_values = [3.3868824077313877, 3.38688240773139, 7.5069526507064035, 7.506952650706411, 0.19248596540272908, 1.5798796065733554, 1.5798796065733547, 0.19248596540272866, 0.9999999999999999, 0.11313270650973078, 0.08702172863310761, 0.08702172863310764, 0.124068868139005, 0.12406886813900504, 0.11313270650973078, 3.1424847470077033, 3.142484747007704, 2.9384350140255173, 2.9384350140255187, 2.2953723636801207, 2.295372363680121, 1.8868005996535895, 1.8868005996535888, 3.1424847470077033, 3.142484747007704, 0.3403172080467579, 20.451494655678342, 3.142484747007704, 20.45149465567836, 0.34031720804675775, 3.1424847470077033, 1.3],
        box_lower_bounds = [-1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 11,
        solve_order = 4,
        variables = ["➕₂"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [16],
        equations = Expr[
            :(➕₂ - k / ((k - θ) + 1)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["➕₂"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [5.666666666666666],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 12,
        solve_order = 3,
        variables = ["➕₉"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [38],
        equations = Expr[
            :(➕₉ - θ / (θ - 1)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["➕₉"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.3571428571428572],
        box_lower_bounds = [2.220446049250313e-16],
        box_upper_bounds = [1.0e12],
    ),
    (
        index = 13,
        solve_order = 2,
        variables = ["Z"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [35],
        equations = Expr[
            :((-Z * ρZ + Z + 1.0ρZ) - 1.0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["Z"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 14,
        solve_order = 1,
        variables = ["Z̄"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [36],
        equations = Expr[
            :((-Z̄ * ρZ̄ + Z̄ + 1.0ρZ̄) - 1.0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["Z̄"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [1.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
]
const BLOCK_EQUATION_ORDER = [45, 44, 23, 22, 30, 31, 32, 29, 34, 33, 9, 27, 17, 19, 20, 2, 8, 21, 5, 24, 7, 10, 13, 14, 25, 11, 6, 40, 43, 3, 4, 28, 1, 26, 12, 15, 39, 41, 42, 18, 37, 16, 38, 35, 36]
const BLOCK_SOLVE_ORDER = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    ["z̃x̄", "➕₂"],
    ["z̃x", "➕₂"],
    ["C̄"],
    ["C"],
    ["Nd", "Nd̄", "Nx", "Nx̄", "z̃x̄", "➕₄", "➕₆"],
    ["Q̃"],
    ["z̃x"],
    ["Q", "➕₇"],
    ["Nd", "Nd̄", "Nx", "Nx̄"],
    ["Z", "Z̄", "➕₂", "➕₉"],
    String[],
    String[],
    String[],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    [2.9384350140255187, 5.666666666666666],
    [2.9384350140255173, 5.666666666666666],
    [3.38688240773139],
    [3.3868824077313877],
    [7.5069526507064035, 7.506952650706411, 1.5798796065733554, 1.5798796065733547, 2.9384350140255187, 0.8220019677454524, 0.9999999999999994],
    [1.0000000000000002],
    [2.9384350140255173],
    [0.9999999999999999, 0.999999999999999],
    [7.5069526507064035, 7.506952650706411, 1.5798796065733554, 1.5798796065733547],
    [1.0, 1.0, 5.666666666666666, 1.3571428571428572],
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
]
const BLOCK_SOLUTION_NAMES = [
    ["zx̄"],
    ["zx"],
    ["r̄", "➕₁₆"],
    ["r", "➕₁₅"],
    ["TOL", "➕₅"],
    ["➕₆"],
    ["➕₄"],
    ["Q̃"],
    ["➕₇"],
    ["C", "C̄", "Nd", "Nd̄", "Ne", "Nx", "Nx̄", "Nē", "Q", "d̃", "d̃d", "d̃d̄", "d̃x", "d̃x̄", "d̃̄", "w", "w̄", "z̃x", "z̃x̄", "ρ̃d", "ρ̃d̄", "ρ̃x", "ρ̃x̄", "ṽ", "ṽ̄", "➕₁", "➕₁₀", "➕₁₁", "➕₁₂", "➕₃", "➕₈", "➕₁₄"],
    ["➕₂"],
    ["➕₉"],
    ["Z"],
    ["Z̄"],
]
const BLOCK_SOLUTION_VALUES = [
    [1.5815047299285403],
    [1.5815047299285396],
    [0.01010101010101011, 3.38688240773139],
    [0.01010101010101011, 3.3868824077313877],
    [1.0, 0.8220019677454521],
    [0.9999999999999994],
    [0.8220019677454524],
    [1.0000000000000002],
    [0.999999999999999],
    [3.3868824077313877, 3.38688240773139, 7.5069526507064035, 7.506952650706411, 0.19248596540272908, 1.5798796065733554, 1.5798796065733547, 0.19248596540272866, 0.9999999999999999, 0.11313270650973078, 0.08702172863310761, 0.08702172863310764, 0.124068868139005, 0.12406886813900504, 0.11313270650973078, 3.1424847470077033, 3.142484747007704, 2.9384350140255173, 2.9384350140255187, 2.2953723636801207, 2.295372363680121, 1.8868005996535895, 1.8868005996535888, 3.1424847470077033, 3.142484747007704, 0.3403172080467579, 20.451494655678342, 3.142484747007704, 20.45149465567836, 0.34031720804675775, 3.1424847470077033, 1.3],
    [5.666666666666666],
    [1.3571428571428572],
    [1.0],
    [1.0],
]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[15] = parameters[15]
    complete_parameters[7] = parameters[7]
    complete_parameters[2] = parameters[2]
    complete_parameters[11] = parameters[11]
    complete_parameters[17] = parameters[17]
    complete_parameters[5] = parameters[5]
    complete_parameters[9] = parameters[9]
    complete_parameters[1] = parameters[1]
    complete_parameters[3] = parameters[3]
    complete_parameters[6] = parameters[6]
    complete_parameters[4] = parameters[4]
    complete_parameters[10] = parameters[10]
    complete_parameters[16] = parameters[16]
    complete_parameters[8] = parameters[8]
    complete_parameters[13] = parameters[13]
    complete_parameters[12] = parameters[12]
    complete_parameters[14] = parameters[14]
    complete_parameters[18] = ((complete_parameters[17] * (1 - complete_parameters[3] * (1 - complete_parameters[5]))) / (complete_parameters[3] * (1 - complete_parameters[5]))) * complete_parameters[11]
    complete_parameters[19] = ((complete_parameters[17] * (1 - complete_parameters[3] * (1 - complete_parameters[5]))) / (complete_parameters[3] * (1 - complete_parameters[5]))) * complete_parameters[12]
    complete_parameters[20] = (complete_parameters[7] / (complete_parameters[7] - (complete_parameters[6] - 1))) ^ (1 / (complete_parameters[6] - 1)) * complete_parameters[9]
    complete_parameters[21] = (complete_parameters[7] / (complete_parameters[7] - (complete_parameters[6] - 1))) ^ (1 / (complete_parameters[6] - 1)) * complete_parameters[10]
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        1 - (solution[3] * solution[28] ^ (1 - complete_parameters[6]) + solution[7] * solution[31] ^ (1 - complete_parameters[6])),
        1 - (solution[4] * solution[29] ^ (1 - complete_parameters[6]) + solution[6] * solution[30] ^ (1 - complete_parameters[6])),
        solution[28] - ((complete_parameters[6] / (complete_parameters[6] - 1)) * solution[22]) / (solution[12] * complete_parameters[20]),
        solution[29] - ((complete_parameters[6] / (complete_parameters[6] - 1)) * solution[23]) / (solution[13] * complete_parameters[21]),
        solution[30] - (((complete_parameters[6] / (complete_parameters[6] - 1)) * complete_parameters[8] * solution[22]) / (solution[12] * solution[26])) / solution[9],
        solution[31] - (((solution[9] * complete_parameters[6]) / (complete_parameters[6] - 1)) * complete_parameters[8] * solution[23]) / (solution[13] * solution[27]),
        solution[14] - (solution[15] + (solution[6] / solution[3]) * solution[17]),
        solution[19] - (solution[16] + (solution[7] / solution[4]) * solution[18]),
        solution[15] - ((solution[28] ^ (1 - complete_parameters[6]) * 1) / complete_parameters[6]) * solution[1],
        solution[16] - ((solution[29] ^ (1 - complete_parameters[6]) * 1) / complete_parameters[6]) * solution[2],
        solution[32] - (solution[22] * complete_parameters[11]) / solution[12],
        solution[33] - (solution[23] * complete_parameters[12]) / solution[13],
        solution[17] - (((solution[22] * complete_parameters[18]) / solution[12]) * (complete_parameters[6] - 1)) / (complete_parameters[7] - (complete_parameters[6] - 1)),
        solution[18] - (((complete_parameters[6] - 1) / (complete_parameters[7] - (complete_parameters[6] - 1))) * solution[23] * complete_parameters[19]) / solution[13],
        solution[6] / solution[3] - (complete_parameters[9] / solution[26]) ^ complete_parameters[7] * (complete_parameters[7] / (complete_parameters[7] - (complete_parameters[6] - 1))) ^ (complete_parameters[7] / (complete_parameters[6] - 1)),
        solution[7] / solution[4] - (complete_parameters[7] / (complete_parameters[7] - (complete_parameters[6] - 1))) ^ (complete_parameters[7] / (complete_parameters[6] - 1)) * (complete_parameters[10] / solution[27]) ^ complete_parameters[7],
        solution[3] - (1 - complete_parameters[5]) * (solution[3] + solution[5]),
        solution[4] - (1 - complete_parameters[5]) * (solution[4] + solution[8]),
        solution[1] ^ -(complete_parameters[4]) - complete_parameters[3] * (1 + solution[20]) * solution[1] ^ -(complete_parameters[4]),
        solution[2] ^ -(complete_parameters[4]) - complete_parameters[3] * (1 + solution[21]) * solution[2] ^ -(complete_parameters[4]),
        solution[32] - (1 - complete_parameters[5]) * complete_parameters[3] * (solution[1] / solution[1]) ^ -(complete_parameters[4]) * (solution[32] + solution[14]),
        solution[33] - (1 - complete_parameters[5]) * complete_parameters[3] * (solution[2] / solution[2]) ^ -(complete_parameters[4]) * (solution[33] + solution[19]),
        solution[1] - ((solution[22] * complete_parameters[13] + solution[3] * solution[14]) - solution[32] * solution[5]),
        solution[2] - ((solution[23] * complete_parameters[14] + solution[4] * solution[19]) - solution[33] * solution[8]),
        solution[9] - (solution[7] * solution[31] ^ (1 - complete_parameters[6]) * solution[1]) / (solution[6] * solution[30] ^ (1 - complete_parameters[6]) * solution[2]),
        solution[10] - (((solution[4] / (solution[4] + solution[6])) * solution[11] ^ (1 - complete_parameters[6]) + (solution[6] / (solution[4] + solution[6])) * ((complete_parameters[8] * complete_parameters[20]) / solution[26]) ^ (1 - complete_parameters[6])) / (solution[3] / (solution[3] + solution[7]) + (solution[7] / (solution[3] + solution[7])) * ((complete_parameters[8] * solution[11] * complete_parameters[21]) / solution[27]) ^ (1 - complete_parameters[6]))) ^ (1 / (1 - complete_parameters[6])),
        solution[10] - solution[9] * ((solution[3] + solution[7]) / (solution[4] + solution[6])) ^ (-1 / (complete_parameters[6] - 1)),
        solution[12] - ((1 - complete_parameters[15]) * 1.0 + complete_parameters[15] * solution[12] + complete_parameters[1] * 0),
        solution[13] - (1.0 * (1 - complete_parameters[16]) + complete_parameters[16] * solution[13] + complete_parameters[2] * 0),
        solution[26] - (complete_parameters[6] * complete_parameters[18] * (solution[22] / solution[12]) ^ complete_parameters[6] * (1 + (complete_parameters[6] - 1) / (complete_parameters[7] - (complete_parameters[6] - 1))) * solution[9] ^ -(complete_parameters[6]) * complete_parameters[8] ^ (complete_parameters[6] - 1) * (complete_parameters[6] / (complete_parameters[6] - 1)) ^ (complete_parameters[6] - 1) * solution[2] ^ -1) ^ (1 / (complete_parameters[6] - 1)),
        solution[27] - ((complete_parameters[6] / (complete_parameters[6] - 1)) ^ (complete_parameters[6] - 1) * complete_parameters[6] * complete_parameters[8] ^ (complete_parameters[6] - 1) * (1 + (complete_parameters[6] - 1) / (complete_parameters[7] - (complete_parameters[6] - 1))) * complete_parameters[19] * (solution[23] / solution[13]) ^ complete_parameters[6] * solution[9] ^ complete_parameters[6] * solution[1] ^ -1) ^ (1 / (complete_parameters[6] - 1)),
        solution[24] - solution[26] / (complete_parameters[7] / (complete_parameters[7] - (complete_parameters[6] - 1))) ^ (1 / (complete_parameters[6] - 1)),
        solution[25] - solution[27] / (complete_parameters[7] / (complete_parameters[7] - (complete_parameters[6] - 1))) ^ (1 / (complete_parameters[6] - 1)),
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[3]) * solution[28] ^ (1 - complete_parameters[6]) - solution[7] * solution[31] ^ (1 - complete_parameters[6])) + 1,
        (-(solution[4]) * solution[29] ^ (1 - complete_parameters[6]) - solution[6] * solution[30] ^ (1 - complete_parameters[6])) + 1,
        solution[28] - (solution[22] * complete_parameters[6]) / (solution[12] * complete_parameters[20] * (complete_parameters[6] - 1)),
        solution[29] - (solution[23] * complete_parameters[6]) / (solution[13] * complete_parameters[21] * (complete_parameters[6] - 1)),
        solution[30] - (solution[22] * complete_parameters[6] * complete_parameters[8]) / (solution[9] * solution[12] * solution[26] * (complete_parameters[6] - 1)),
        (-(solution[9]) * solution[23] * complete_parameters[6] * complete_parameters[8]) / (solution[13] * solution[27] * (complete_parameters[6] - 1)) + solution[31],
        (solution[14] - solution[15]) - (solution[6] * solution[17]) / solution[3],
        (-(solution[16]) + solution[19]) - (solution[7] * solution[18]) / solution[4],
        (-(solution[1]) * solution[28] ^ (1 - complete_parameters[6])) / complete_parameters[6] + solution[15],
        (-(solution[2]) * solution[29] ^ (1 - complete_parameters[6])) / complete_parameters[6] + solution[16],
        solution[32] - (complete_parameters[11] * solution[22]) / solution[12],
        solution[33] - (complete_parameters[12] * solution[23]) / solution[13],
        solution[17] - (complete_parameters[18] * solution[22] * (complete_parameters[6] - 1)) / (solution[12] * ((complete_parameters[7] - complete_parameters[6]) + 1)),
        solution[18] - (complete_parameters[19] * solution[23] * (complete_parameters[6] - 1)) / (solution[13] * ((complete_parameters[7] - complete_parameters[6]) + 1)),
        solution[34] - complete_parameters[9] / solution[26],
        solution[38] - complete_parameters[7] / ((complete_parameters[7] - complete_parameters[6]) + 1),
        -(solution[34] ^ complete_parameters[7]) * solution[38] ^ (complete_parameters[7] / (complete_parameters[6] - 1)) + solution[6] / solution[3],
        solution[39] - complete_parameters[10] / solution[27],
        -(solution[38] ^ (complete_parameters[7] / (complete_parameters[6] - 1))) * solution[39] ^ complete_parameters[7] + solution[7] / solution[4],
        solution[3] - (1 - complete_parameters[5]) * (solution[3] + solution[5]),
        solution[4] - (1 - complete_parameters[5]) * (solution[4] + solution[8]),
        (-(complete_parameters[3]) * (solution[20] + 1)) / solution[1] ^ complete_parameters[4] + solution[1] ^ -(complete_parameters[4]),
        (-(complete_parameters[3]) * (solution[21] + 1)) / solution[2] ^ complete_parameters[4] + solution[2] ^ -(complete_parameters[4]),
        -(complete_parameters[3]) * (1 - complete_parameters[5]) * (solution[14] + solution[32]) + solution[32],
        -(complete_parameters[3]) * (1 - complete_parameters[5]) * (solution[19] + solution[33]) + solution[33],
        ((solution[1] - complete_parameters[13] * solution[22]) - solution[3] * solution[14]) + solution[5] * solution[32],
        ((solution[2] - complete_parameters[14] * solution[23]) - solution[4] * solution[19]) + solution[8] * solution[33],
        (-(solution[1]) * solution[7] * solution[30] ^ (complete_parameters[6] - 1) * solution[31] ^ (1 - complete_parameters[6])) / (solution[2] * solution[6]) + solution[9],
        solution[40] - (complete_parameters[20] * complete_parameters[8]) / solution[26],
        solution[41] - (solution[11] * complete_parameters[21] * complete_parameters[8]) / solution[27],
        solution[42] - ((solution[4] * solution[11] ^ (1 - complete_parameters[6])) / (solution[4] + solution[6]) + (solution[6] * solution[40] ^ (1 - complete_parameters[6])) / (solution[4] + solution[6])) / (solution[3] / (solution[3] + solution[7]) + (solution[7] * solution[41] ^ (1 - complete_parameters[6])) / (solution[3] + solution[7])),
        solution[10] - solution[42] ^ (1 / (1 - complete_parameters[6])),
        solution[43] - (solution[3] + solution[7]) / (solution[4] + solution[6]),
        -(solution[9]) / solution[43] ^ (1 / (complete_parameters[6] - 1)) + solution[10],
        (-(solution[12]) * complete_parameters[15] + solution[12] + 1.0 * complete_parameters[15]) - 1.0,
        (-(solution[13]) * complete_parameters[16] + solution[13] + 1.0 * complete_parameters[16]) - 1.0,
        solution[44] - solution[22] / solution[12],
        solution[45] - complete_parameters[6] / (complete_parameters[6] - 1),
        solution[35] - (complete_parameters[18] * complete_parameters[6] * complete_parameters[8] ^ (complete_parameters[6] - 1) * solution[44] ^ complete_parameters[6] * solution[45] ^ (complete_parameters[6] - 1) * ((complete_parameters[6] - 1) / ((complete_parameters[7] - complete_parameters[6]) + 1) + 1)) / (solution[2] * solution[9] ^ complete_parameters[6]),
        solution[26] - solution[35] ^ (1 / (complete_parameters[6] - 1)),
        solution[36] - solution[23] / solution[13],
        solution[37] - (solution[9] ^ complete_parameters[6] * complete_parameters[19] * complete_parameters[6] * complete_parameters[8] ^ (complete_parameters[6] - 1) * solution[36] ^ complete_parameters[6] * solution[45] ^ (complete_parameters[6] - 1) * ((complete_parameters[6] - 1) / ((complete_parameters[7] - complete_parameters[6]) + 1) + 1)) / solution[1],
        solution[27] - solution[37] ^ (1 / (complete_parameters[6] - 1)),
        solution[24] - solution[26] / solution[38] ^ (1 / (complete_parameters[6] - 1)),
        solution[25] - solution[27] / solution[38] ^ (1 / (complete_parameters[6] - 1)),
    ]
end

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1] / previous_solution[2] ^ (1 / (complete_parameters[6] - 1)),
    ]
end

function residuals_block_2(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - previous_solution[1] / previous_solution[2] ^ (1 / (complete_parameters[6] - 1)),
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(complete_parameters[3]) * (solution[1] + 1)) / solution[2] ^ complete_parameters[4] + solution[2] ^ -(complete_parameters[4]),
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
        (-(complete_parameters[3]) * (solution[1] + 1)) / solution[2] ^ complete_parameters[4] + solution[2] ^ -(complete_parameters[4]),
        solution[2] - min(1.0e12, max(eps(), previous_solution[1])),
    ]
end

function residuals_block_5(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 7
    @assert length(external_solution) == 0
    @assert length(solution) == 2
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[2] - (solution[1] * complete_parameters[21] * complete_parameters[8]) / previous_solution[5],
        previous_solution[7] - ((previous_solution[2] * solution[1] ^ (1 - complete_parameters[6])) / (previous_solution[2] + previous_solution[3]) + (previous_solution[3] * previous_solution[6] ^ (1 - complete_parameters[6])) / (previous_solution[2] + previous_solution[3])) / (previous_solution[1] / (previous_solution[1] + previous_solution[4]) + (previous_solution[4] * solution[2] ^ (1 - complete_parameters[6])) / (previous_solution[1] + previous_solution[4])),
    ]
end

function residuals_block_6(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        previous_solution[1] - solution[1] ^ (1 / (1 - complete_parameters[6])),
    ]
end

function residuals_block_7(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - (complete_parameters[20] * complete_parameters[8]) / previous_solution[1],
    ]
end

function residuals_block_8(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 2
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(previous_solution[1]) / previous_solution[2] ^ (1 / (complete_parameters[6] - 1)) + solution[1],
    ]
end

function residuals_block_9(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 4
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - (previous_solution[1] + previous_solution[4]) / (previous_solution[2] + previous_solution[3]),
    ]
end

function residuals_block_10(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 4
    @assert length(external_solution) == 0
    @assert length(solution) == 32
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[1]) * solution[20] ^ (1 - complete_parameters[6])) / complete_parameters[6] + solution[11],
        ((solution[2] - complete_parameters[14] * solution[17]) - solution[4] * solution[15]) + solution[8] * solution[25],
        -(solution[26] ^ complete_parameters[7]) * previous_solution[3] ^ (complete_parameters[7] / (complete_parameters[6] - 1)) + solution[6] / solution[3],
        -(previous_solution[3] ^ (complete_parameters[7] / (complete_parameters[6] - 1))) * solution[30] ^ complete_parameters[7] + solution[7] / solution[4],
        solution[3] - (1 - complete_parameters[5]) * (solution[3] + solution[5]),
        (-(solution[4]) * solution[21] ^ (1 - complete_parameters[6]) - solution[6] * solution[22] ^ (1 - complete_parameters[6])) + 1,
        (-(solution[12]) + solution[15]) - (solution[7] * solution[14]) / solution[4],
        solution[4] - (1 - complete_parameters[5]) * (solution[4] + solution[8]),
        solution[22] - (solution[16] * complete_parameters[6] * complete_parameters[8]) / (solution[9] * previous_solution[1] * solution[18] * (complete_parameters[6] - 1)),
        -(complete_parameters[3]) * (1 - complete_parameters[5]) * (solution[10] + solution[24]) + solution[24],
        (solution[10] - solution[11]) - (solution[6] * solution[13]) / solution[3],
        (-(solution[2]) * solution[21] ^ (1 - complete_parameters[6])) / complete_parameters[6] + solution[12],
        solution[13] - (complete_parameters[18] * solution[16] * (complete_parameters[6] - 1)) / (previous_solution[1] * ((complete_parameters[7] - complete_parameters[6]) + 1)),
        solution[14] - (complete_parameters[19] * solution[17] * (complete_parameters[6] - 1)) / (previous_solution[2] * ((complete_parameters[7] - complete_parameters[6]) + 1)),
        -(complete_parameters[3]) * (1 - complete_parameters[5]) * (solution[15] + solution[25]) + solution[25],
        solution[24] - (complete_parameters[11] * solution[16]) / previous_solution[1],
        (-(solution[9]) * solution[17] * complete_parameters[6] * complete_parameters[8]) / (previous_solution[2] * solution[19] * (complete_parameters[6] - 1)) + solution[23],
        solution[18] - solution[27] ^ (1 / (complete_parameters[6] - 1)),
        solution[19] - solution[29] ^ (1 / (complete_parameters[6] - 1)),
        solution[20] - (solution[16] * complete_parameters[6]) / (previous_solution[1] * complete_parameters[20] * (complete_parameters[6] - 1)),
        solution[21] - (solution[17] * complete_parameters[6]) / (previous_solution[2] * complete_parameters[21] * (complete_parameters[6] - 1)),
        (-(solution[1]) * solution[7] * solution[22] ^ (complete_parameters[6] - 1) * solution[23] ^ (1 - complete_parameters[6])) / (solution[2] * solution[6]) + solution[9],
        (-(solution[3]) * solution[20] ^ (1 - complete_parameters[6]) - solution[7] * solution[23] ^ (1 - complete_parameters[6])) + 1,
        ((solution[1] - complete_parameters[13] * solution[16]) - solution[3] * solution[10]) + solution[5] * solution[24],
        solution[25] - (complete_parameters[12] * solution[17]) / previous_solution[2],
        solution[26] - complete_parameters[9] / solution[18],
        solution[27] - (complete_parameters[18] * complete_parameters[6] * solution[32] ^ (complete_parameters[6] - 1) * solution[31] ^ complete_parameters[6] * previous_solution[4] ^ (complete_parameters[6] - 1) * ((complete_parameters[6] - 1) / ((complete_parameters[7] - complete_parameters[6]) + 1) + 1)) / (solution[2] * solution[9] ^ complete_parameters[6]),
        solution[28] - solution[17] / previous_solution[2],
        solution[29] - (solution[9] ^ complete_parameters[6] * complete_parameters[19] * complete_parameters[6] * solution[32] ^ (complete_parameters[6] - 1) * solution[28] ^ complete_parameters[6] * previous_solution[4] ^ (complete_parameters[6] - 1) * ((complete_parameters[6] - 1) / ((complete_parameters[7] - complete_parameters[6]) + 1) + 1)) / solution[1],
        solution[30] - complete_parameters[10] / solution[19],
        solution[31] - solution[16] / previous_solution[1],
        solution[32] - min(1.0e12, max(eps(), complete_parameters[8])),
    ]
end

function residuals_block_11(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - complete_parameters[7] / ((complete_parameters[7] - complete_parameters[6]) + 1),
    ]
end

function residuals_block_12(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - complete_parameters[6] / (complete_parameters[6] - 1),
    ]
end

function residuals_block_13(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[1]) * complete_parameters[15] + solution[1] + 1.0 * complete_parameters[15]) - 1.0,
    ]
end

function residuals_block_14(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(solution[1]) * complete_parameters[16] + solution[1] + 1.0 * complete_parameters[16]) - 1.0,
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
export residuals_block_1, residuals_block_2, residuals_block_3, residuals_block_4, residuals_block_5, residuals_block_6, residuals_block_7, residuals_block_8, residuals_block_9, residuals_block_10, residuals_block_11, residuals_block_12, residuals_block_13, residuals_block_14
end
