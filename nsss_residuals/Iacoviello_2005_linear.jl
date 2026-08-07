module Iacoviello_2005_linearNsssResiduals
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

const MODEL_NAME = "Iacoviello_2005_linear"
const SOURCE_MODEL_FILE = "models/Iacoviello_2005_linear.jl"
const NSSS_SOLUTION_ERROR = 0.0
const NSSS_RESIDUAL_NORM = 0.0

const PARAMETER_NAMES = [
    "β",
    "β2",
    "γ",
    "j",
    "η",
    "my",
    "υ",
    "ψ",
    "δ",
    "fie",
    "fih",
    "X",
    "thη",
    "α",
    "m",
    "m2",
    "ρu",
    "ρj",
    "ρA",
    "rR",
    "rpi",
    "rY",
]
const PARAMETER_VALUES = Float64[
    0.99,
    0.95,
    0.98,
    0.1,
    1.01,
    0.3,
    0.03,
    2.0,
    0.03,
    0.0,
    0.0,
    1.05,
    0.75,
    0.64,
    0.89,
    0.55,
    0.59,
    0.85,
    0.03,
    0.73,
    0.27,
    0.13,
]
const COMPLETE_PARAMETER_NAMES = [
    "β",
    "β2",
    "γ",
    "j",
    "η",
    "my",
    "υ",
    "ψ",
    "δ",
    "fie",
    "fih",
    "X",
    "thη",
    "α",
    "m",
    "m2",
    "ρu",
    "ρj",
    "ρA",
    "rR",
    "rpi",
    "rY",
    "ItoY",
    "R",
    "b2toY",
    "btoY",
    "c1toY",
    "c2toY",
    "ctoY",
    "h2toh1",
    "htoh1",
    "qh2toY",
    "qhtoY",
    "s1",
    "s2",
    "γe",
    "γh",
    "ι",
    "ι2",
    "κ",
    "ω",
    "qh1toY",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    0.99,
    0.95,
    0.98,
    0.1,
    1.01,
    0.3,
    0.03,
    2.0,
    0.03,
    0.0,
    0.0,
    1.05,
    0.75,
    0.64,
    0.89,
    0.55,
    0.59,
    0.85,
    0.03,
    0.73,
    0.27,
    0.13,
    0.17004048582995956,
    1.0101010101010102,
    0.43810657993495067,
    2.222594594594589,
    0.4828757694396923,
    0.22528896672504375,
    0.12179477800530432,
    0.1539310979471383,
    0.4825913438957224,
    0.8046034525894398,
    2.5225225225225163,
    0.45599999999999996,
    0.22971428571428568,
    0.9889,
    0.972,
    0.0048259134389572285,
    0.0015393109794713845,
    0.08583333333333336,
    0.938529088913282,
    5.227036403428693,
]
const ORIGINAL_SOLUTION_NAMES = [
    "K̂",
    "R̂",
    "X̂",
    "b2̂",
    "b̂",
    "c1̂",
    "c2̂",
    "h2̂",
    "pî",
    "q̂",
    "rr̂",
    "Â",
    "Î",
    "û",
    "ĉ",
    "ĥ",
    "ĵ",
    "Ŷ",
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
    0.0,
    0.0,
    0.0,
]
const AUXILIARY_SOLUTION_NAMES = [
    "K̂",
    "R̂",
    "X̂",
    "b2̂",
    "b̂",
    "c1̂",
    "c2̂",
    "h2̂",
    "pî",
    "q̂",
    "rr̂",
    "Â",
    "Î",
    "û",
    "ĉ",
    "ĥ",
    "ĵ",
    "Ŷ",
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
    :(Ŷ - (ctoY * ĉ + c1toY * c1̂ + c2toY * c2̂ + ItoY * Î)),
    :(c1̂ - (c1̂ - rr̂)),
    :(Î - (K̂ + γ * (Î - K̂) + ((1 - γ * (1 - δ)) * ((Ŷ - X̂) - K̂)) / ψ + (ĉ - ĉ) / ψ)),
    :(q̂ - ((((γe * q̂ + (1 - γe) * ((Ŷ - X̂) - ĥ)) - rr̂ * m * β) - (1 - m * β) * (ĉ - ĉ)) - fie * ((ĥ - ĥ) - γ * (ĥ - ĥ)))),
    :(q̂ - ((((q̂ * γh + (1 - γh) * (ĵ - h2̂)) - rr̂ * β * m2) + (1 - β * m2) * (c2̂ - ω * c2̂)) - fih * ((h2̂ - h2̂) - β2 * (h2̂ - h2̂)))),
    :(q̂ - (((c1̂ + q̂ * β + ĵ * (1 - β) + ĥ * ι + h2̂ * ι2) - c1̂ * β) + fih * ((((ĥ - ĥ) * htoh1 + (h2̂ - h2̂) * h2toh1) - (ĥ - ĥ) * β * htoh1) - (h2̂ - h2̂) * β * h2toh1))),
    :(b̂ - ((q̂ + ĥ) - rr̂)),
    :(b2̂ - ((q̂ + h2̂) - rr̂)),
    :(Ŷ - ((η * (Â + ĥ * υ + K̂ * my)) / (η - ((1 - υ) - my)) - (((1 - υ) - my) * (X̂ + c1̂ * α + c2̂ * (1 - α))) / (η - ((1 - υ) - my)))),
    :(pî - ((β * pî - X̂ * κ) + û)),
    :(K̂ - (Î * δ + K̂ * (1 - δ))),
    :(b̂ * btoY - ((ItoY * Î + ctoY * ĉ + (ĥ - ĥ) * qhtoY + btoY * R * ((R̂ + b̂) - pî)) - ((1 - s1) - s2) * (Ŷ - X̂))),
    :(b2̂ * b2toY - ((c2toY * c2̂ + (h2̂ - h2̂) * qh2toY + R * b2toY * ((R̂ + b2̂) - pî)) - s2 * (Ŷ - X̂))),
    :(R̂ - ((1 - rR) * (1 + rpi) * pî + (1 - rR) * rY * Ŷ + R̂ * rR + 0)),
    :(rr̂ - (R̂ - pî)),
    :(ĵ - (ρj * ĵ + 0)),
    :(û - (ρu * û + 0)),
    :(Â - (ρA * Â + 0)),
]
const CALIBRATION_EQUATIONS = Expr[
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :((((-ItoY * Î - c1toY * c1̂) - c2toY * c2̂) - ctoY * ĉ) + Ŷ),
    :(rr̂ - 0),
    :(((-K̂ + Î) - γ * (-K̂ + Î)) - ((-γ * (1 - δ) + 1) * ((-K̂ - X̂) + Ŷ)) / ψ),
    :(((m * rr̂ * β - q̂ * γe) + q̂) - (1 - γe) * ((-X̂ - ĥ) + Ŷ)),
    :((((m2 * rr̂ * β - q̂ * γh) + q̂) - (1 - γh) * (-h2̂ + ĵ)) - (-c2̂ * ω + c2̂) * (-m2 * β + 1)),
    :((((((c1̂ * β - c1̂) - h2̂ * ι2) - q̂ * β) + q̂) - ĥ * ι) - ĵ * (1 - β)),
    :(((b̂ - q̂) + rr̂) - ĥ),
    :(((b2̂ - h2̂) - q̂) + rr̂),
    :((Ŷ - (η * (K̂ * my + Â + ĥ * υ)) / ((my + η + υ) - 1)) + ((X̂ + c1̂ * α + c2̂ * (1 - α)) * ((-my - υ) + 1)) / ((my + η + υ) - 1)),
    :(((X̂ * κ - pî * β) + pî) - û),
    :((-K̂ * (1 - δ) + K̂) - Î * δ),
    :((((-ItoY * Î - R * btoY * ((R̂ + b̂) - pî)) + btoY * b̂) - ctoY * ĉ) + (-X̂ + Ŷ) * ((-s1 - s2) + 1)),
    :(((-R * b2toY * ((R̂ + b2̂) - pî) + b2toY * b2̂) - c2toY * c2̂) + s2 * (-X̂ + Ŷ)),
    :(((-R̂ * rR + R̂) - pî * (1 - rR) * (rpi + 1)) - rY * Ŷ * (1 - rR)),
    :(-R̂ + pî + rr̂),
    :(-ĵ * ρj + ĵ),
    :(-û * ρu + û),
    :(-Â * ρA + Â),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(Ŷ - (ctoY * ĉ + c1toY * c1̂ + c2toY * c2̂ + ItoY * Î)),
    :(c1̂ - (c1̂ - rr̂)),
    :(Î - (K̂ + γ * (Î - K̂) + ((1 - γ * (1 - δ)) * ((Ŷ - X̂) - K̂)) / ψ + (ĉ - ĉ) / ψ)),
    :(q̂ - ((((γe * q̂ + (1 - γe) * ((Ŷ - X̂) - ĥ)) - rr̂ * m * β) - (1 - m * β) * (ĉ - ĉ)) - fie * ((ĥ - ĥ) - γ * (ĥ - ĥ)))),
    :(q̂ - ((((q̂ * γh + (1 - γh) * (ĵ - h2̂)) - rr̂ * β * m2) + (1 - β * m2) * (c2̂ - ω * c2̂)) - fih * ((h2̂ - h2̂) - β2 * (h2̂ - h2̂)))),
    :(q̂ - (((c1̂ + q̂ * β + ĵ * (1 - β) + ĥ * ι + h2̂ * ι2) - c1̂ * β) + fih * ((((ĥ - ĥ) * htoh1 + (h2̂ - h2̂) * h2toh1) - (ĥ - ĥ) * β * htoh1) - (h2̂ - h2̂) * β * h2toh1))),
    :(b̂ - ((q̂ + ĥ) - rr̂)),
    :(b2̂ - ((q̂ + h2̂) - rr̂)),
    :(Ŷ - ((η * (Â + ĥ * υ + K̂ * my)) / (η - ((1 - υ) - my)) - (((1 - υ) - my) * (X̂ + c1̂ * α + c2̂ * (1 - α))) / (η - ((1 - υ) - my)))),
    :(pî - ((β * pî - X̂ * κ) + û)),
    :(K̂ - (Î * δ + K̂ * (1 - δ))),
    :(b̂ * btoY - ((ItoY * Î + ctoY * ĉ + (ĥ - ĥ) * qhtoY + btoY * R * ((R̂ + b̂) - pî)) - ((1 - s1) - s2) * (Ŷ - X̂))),
    :(b2̂ * b2toY - ((c2toY * c2̂ + (h2̂ - h2̂) * qh2toY + R * b2toY * ((R̂ + b2̂) - pî)) - s2 * (Ŷ - X̂))),
    :(R̂ - ((1 - rR) * (1 + rpi) * pî + (1 - rR) * rY * Ŷ + R̂ * rR + 0)),
    :(rr̂ - (R̂ - pî)),
    :(ĵ - (ρj * ĵ + 0)),
    :(û - (ρu * û + 0)),
    :(Â - (ρA * Â + 0)),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :((((-ItoY * Î - c1toY * c1̂) - c2toY * c2̂) - ctoY * ĉ) + Ŷ),
    :(rr̂ - 0),
    :(((-K̂ + Î) - γ * (-K̂ + Î)) - ((-γ * (1 - δ) + 1) * ((-K̂ - X̂) + Ŷ)) / ψ),
    :(((m * rr̂ * β - q̂ * γe) + q̂) - (1 - γe) * ((-X̂ - ĥ) + Ŷ)),
    :((((m2 * rr̂ * β - q̂ * γh) + q̂) - (1 - γh) * (-h2̂ + ĵ)) - (-c2̂ * ω + c2̂) * (-m2 * β + 1)),
    :((((((c1̂ * β - c1̂) - h2̂ * ι2) - q̂ * β) + q̂) - ĥ * ι) - ĵ * (1 - β)),
    :(((b̂ - q̂) + rr̂) - ĥ),
    :(((b2̂ - h2̂) - q̂) + rr̂),
    :((Ŷ - (η * (K̂ * my + Â + ĥ * υ)) / ((my + η + υ) - 1)) + ((X̂ + c1̂ * α + c2̂ * (1 - α)) * ((-my - υ) + 1)) / ((my + η + υ) - 1)),
    :(((X̂ * κ - pî * β) + pî) - û),
    :((-K̂ * (1 - δ) + K̂) - Î * δ),
    :((((-ItoY * Î - R * btoY * ((R̂ + b̂) - pî)) + btoY * b̂) - ctoY * ĉ) + (-X̂ + Ŷ) * ((-s1 - s2) + 1)),
    :(((-R * b2toY * ((R̂ + b2̂) - pî) + b2toY * b2̂) - c2toY * c2̂) + s2 * (-X̂ + Ŷ)),
    :(((-R̂ * rR + R̂) - pî * (1 - rR) * (rpi + 1)) - rY * Ŷ * (1 - rR)),
    :(-R̂ + pî + rr̂),
    :(-ĵ * ρj + ĵ),
    :(-û * ρu + û),
    :(-Â * ρA + Â),
]

const PARAMETER_DEFINITION_NAMES = [
    "R",
    "s1",
    "s2",
    "γe",
    "γh",
    "κ",
    "ω",
    "b2toY",
    "c2toY",
    "qh2toY",
    "qhtoY",
    "btoY",
    "c1toY",
    "ctoY",
    "qh1toY",
    "ItoY",
    "h2toh1",
    "htoh1",
    "ι",
    "ι2",
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
    "1 / β",
    "((α * ((1 - my) - υ) + X) - 1) / X",
    "(((1 - my) - υ) * (1 - α)) / X",
    "(1 - m) * γ + m * β",
    "β2 + m2 * (β - β2)",
    "((1 - thη) * (1 - β * thη)) / thη",
    "(β2 - β2 * m2) / (1 - β * m2)",
    "(s2 * m2 * β * j) / (((1 - β2) - m2 * (β - β2)) + (1 - β) * m2 * j)",
    "(s2 * ((1 - β2) - m2 * (β - β2))) / (((1 - β2) - m2 * (β - β2)) + (1 - β) * m2 * j)",
    "(j * s2) / ((1 - β2) - m2 * ((β - β2) - j * (1 - β)))",
    "(γ * υ) / (X * (1 - γe))",
    "m * β * qhtoY",
    "s1 + (1 - β) * (m * qhtoY + m2 * qh2toY)",
    "(((my + υ) - (my * γ * δ) / (1 - γ * (1 - δ))) - qhtoY * X * m * (1 - β)) / X",
    "(j * s1) / (1 - β) + qhtoY * m * j + (m2 * j * s2) / ((1 - β2) - m2 * ((β - β2) - j * (1 - β)))",
    "((1 - ctoY) - c1toY) - c2toY",
    "qh2toY / qh1toY",
    "qhtoY / qh1toY",
    "(1 - β) * htoh1",
    "(1 - β) * h2toh1",
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "β",
    "β2",
    "γ",
    "j",
    "η",
    "my",
    "υ",
    "ψ",
    "δ",
    "fie",
    "fih",
    "X",
    "thη",
    "α",
    "m",
    "m2",
    "ρu",
    "ρj",
    "ρA",
    "rR",
    "rpi",
    "rY",
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
]
const ORIGINAL_BOX_CONSTRAINT_NAMES = [
    "K̂",
    "R̂",
    "X̂",
    "b2̂",
    "b̂",
    "c1̂",
    "c2̂",
    "h2̂",
    "pî",
    "q̂",
    "rr̂",
    "Â",
    "Î",
    "û",
    "ĉ",
    "ĥ",
    "ĵ",
    "Ŷ",
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
    -1.0e12,
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
    -Inf,
    -1.0e12,
    -1.0e12,
    -Inf,
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
    1.0e12,
    Inf,
    Inf,
    1.0e12,
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    1.0e12,
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "K̂",
    "R̂",
    "X̂",
    "b2̂",
    "b̂",
    "c1̂",
    "c2̂",
    "h2̂",
    "pî",
    "q̂",
    "rr̂",
    "Â",
    "Î",
    "û",
    "ĉ",
    "ĥ",
    "ĵ",
    "Ŷ",
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
    -1.0e12,
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
    -Inf,
    -1.0e12,
    -1.0e12,
    -Inf,
    -1.0e12,
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
    1.0e12,
    Inf,
    Inf,
    1.0e12,
    Inf,
    1.0e12,
    1.0e12,
    Inf,
    1.0e12,
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
        solve_order = 5,
        variables = ["K̂", "R̂", "X̂", "b2̂", "b̂", "c1̂", "c2̂", "h2̂", "pî", "q̂", "Î", "ĉ", "ĥ", "Ŷ"],
        previous_solution_names = ["rr̂", "Â", "û", "ĵ"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [3, 15, 10, 8, 7, 1, 5, 6, 14, 4, 11, 12, 9, 13],
        equations = Expr[
            :(((-K̂ + Î) - γ * (-K̂ + Î)) - ((-γ * (1 - δ) + 1) * ((-K̂ - X̂) + Ŷ)) / ψ),
            :(-R̂ + pî + rr̂),
            :(((X̂ * κ - pî * β) + pî) - û),
            :(((b2̂ - h2̂) - q̂) + rr̂),
            :(((b̂ - q̂) + rr̂) - ĥ),
            :((((-ItoY * Î - c1toY * c1̂) - c2toY * c2̂) - ctoY * ĉ) + Ŷ),
            :((((m2 * rr̂ * β - q̂ * γh) + q̂) - (1 - γh) * (-h2̂ + ĵ)) - (-c2̂ * ω + c2̂) * (-m2 * β + 1)),
            :((((((c1̂ * β - c1̂) - h2̂ * ι2) - q̂ * β) + q̂) - ĥ * ι) - ĵ * (1 - β)),
            :(((-R̂ * rR + R̂) - pî * (1 - rR) * (rpi + 1)) - rY * Ŷ * (1 - rR)),
            :(((m * rr̂ * β - q̂ * γe) + q̂) - (1 - γe) * ((-X̂ - ĥ) + Ŷ)),
            :((-K̂ * (1 - δ) + K̂) - Î * δ),
            :((((-ItoY * Î - R * btoY * ((R̂ + b̂) - pî)) + btoY * b̂) - ctoY * ĉ) + (-X̂ + Ŷ) * ((-s1 - s2) + 1)),
            :((Ŷ - (η * (K̂ * my + Â + ĥ * υ)) / ((my + η + υ) - 1)) + ((X̂ + c1̂ * α + c2̂ * (1 - α)) * ((-my - υ) + 1)) / ((my + η + υ) - 1)),
            :(((-R * b2toY * ((R̂ + b2̂) - pî) + b2toY * b2̂) - c2toY * c2̂) + s2 * (-X̂ + Ŷ)),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["K̂", "R̂", "X̂", "b2̂", "b̂", "c1̂", "c2̂", "h2̂", "pî", "q̂", "Î", "ĉ", "ĥ", "Ŷ"],
        previous_solution_values = [0.0, 0.0, 0.0, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        previous_solution_initial_values = [0.0, 0.0, 0.0, 0.0],
        external_solution_initial_values = Float64[],
        solution_initial_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        box_lower_bounds = [-1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 2,
        solve_order = 4,
        variables = ["û"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [17],
        equations = Expr[
            :(-û * ρu + û),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["û"],
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
        index = 3,
        solve_order = 3,
        variables = ["ĵ"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [16],
        equations = Expr[
            :(-ĵ * ρj + ĵ),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ĵ"],
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
        index = 4,
        solve_order = 2,
        variables = ["Â"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [18],
        equations = Expr[
            :(-Â * ρA + Â),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["Â"],
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
        index = 5,
        solve_order = 1,
        variables = ["rr̂"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [2],
        equations = Expr[
            :(rr̂ - 0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["rr̂"],
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
const BLOCK_EQUATION_ORDER = [3, 15, 10, 8, 7, 1, 5, 6, 14, 4, 11, 12, 9, 13, 17, 16, 18, 2]
const BLOCK_SOLVE_ORDER = [5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    ["rr̂", "Â", "û", "ĵ"],
    String[],
    String[],
    String[],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    [0.0, 0.0, 0.0, 0.0],
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
]
const BLOCK_EXTERNAL_SOLUTION_VALUES = [
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
]
const BLOCK_SOLUTION_NAMES = [
    ["K̂", "R̂", "X̂", "b2̂", "b̂", "c1̂", "c2̂", "h2̂", "pî", "q̂", "Î", "ĉ", "ĥ", "Ŷ"],
    ["û"],
    ["ĵ"],
    ["Â"],
    ["rr̂"],
]
const BLOCK_SOLUTION_VALUES = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
]
const BLOCK_PREVIOUS_SOLUTION_INITIAL_VALUES = [
    [0.0, 0.0, 0.0, 0.0],
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
]
const BLOCK_SOLUTION_INITIAL_VALUES = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[14] = parameters[14]
    complete_parameters[6] = parameters[6]
    complete_parameters[4] = parameters[4]
    complete_parameters[22] = parameters[22]
    complete_parameters[17] = parameters[17]
    complete_parameters[18] = parameters[18]
    complete_parameters[20] = parameters[20]
    complete_parameters[21] = parameters[21]
    complete_parameters[7] = parameters[7]
    complete_parameters[13] = parameters[13]
    complete_parameters[12] = parameters[12]
    complete_parameters[5] = parameters[5]
    complete_parameters[19] = parameters[19]
    complete_parameters[16] = parameters[16]
    complete_parameters[10] = parameters[10]
    complete_parameters[11] = parameters[11]
    complete_parameters[9] = parameters[9]
    complete_parameters[1] = parameters[1]
    complete_parameters[3] = parameters[3]
    complete_parameters[15] = parameters[15]
    complete_parameters[8] = parameters[8]
    complete_parameters[2] = parameters[2]
    complete_parameters[24] = 1 / complete_parameters[1]
    complete_parameters[34] = ((complete_parameters[14] * ((1 - complete_parameters[6]) - complete_parameters[7]) + complete_parameters[12]) - 1) / complete_parameters[12]
    complete_parameters[35] = (((1 - complete_parameters[6]) - complete_parameters[7]) * (1 - complete_parameters[14])) / complete_parameters[12]
    complete_parameters[36] = (1 - complete_parameters[15]) * complete_parameters[3] + complete_parameters[15] * complete_parameters[1]
    complete_parameters[37] = complete_parameters[2] + complete_parameters[16] * (complete_parameters[1] - complete_parameters[2])
    complete_parameters[40] = ((1 - complete_parameters[13]) * (1 - complete_parameters[1] * complete_parameters[13])) / complete_parameters[13]
    complete_parameters[41] = (complete_parameters[2] - complete_parameters[2] * complete_parameters[16]) / (1 - complete_parameters[1] * complete_parameters[16])
    complete_parameters[25] = (complete_parameters[35] * complete_parameters[16] * complete_parameters[1] * complete_parameters[4]) / (((1 - complete_parameters[2]) - complete_parameters[16] * (complete_parameters[1] - complete_parameters[2])) + (1 - complete_parameters[1]) * complete_parameters[16] * complete_parameters[4])
    complete_parameters[28] = (complete_parameters[35] * ((1 - complete_parameters[2]) - complete_parameters[16] * (complete_parameters[1] - complete_parameters[2]))) / (((1 - complete_parameters[2]) - complete_parameters[16] * (complete_parameters[1] - complete_parameters[2])) + (1 - complete_parameters[1]) * complete_parameters[16] * complete_parameters[4])
    complete_parameters[32] = (complete_parameters[4] * complete_parameters[35]) / ((1 - complete_parameters[2]) - complete_parameters[16] * ((complete_parameters[1] - complete_parameters[2]) - complete_parameters[4] * (1 - complete_parameters[1])))
    complete_parameters[33] = (complete_parameters[3] * complete_parameters[7]) / (complete_parameters[12] * (1 - complete_parameters[36]))
    complete_parameters[26] = complete_parameters[15] * complete_parameters[1] * complete_parameters[33]
    complete_parameters[27] = complete_parameters[34] + (1 - complete_parameters[1]) * (complete_parameters[15] * complete_parameters[33] + complete_parameters[16] * complete_parameters[32])
    complete_parameters[29] = (((complete_parameters[6] + complete_parameters[7]) - (complete_parameters[6] * complete_parameters[3] * complete_parameters[9]) / (1 - complete_parameters[3] * (1 - complete_parameters[9]))) - complete_parameters[33] * complete_parameters[12] * complete_parameters[15] * (1 - complete_parameters[1])) / complete_parameters[12]
    complete_parameters[42] = (complete_parameters[4] * complete_parameters[34]) / (1 - complete_parameters[1]) + complete_parameters[33] * complete_parameters[15] * complete_parameters[4] + (complete_parameters[16] * complete_parameters[4] * complete_parameters[35]) / ((1 - complete_parameters[2]) - complete_parameters[16] * ((complete_parameters[1] - complete_parameters[2]) - complete_parameters[4] * (1 - complete_parameters[1])))
    complete_parameters[23] = ((1 - complete_parameters[29]) - complete_parameters[27]) - complete_parameters[28]
    complete_parameters[30] = complete_parameters[32] / complete_parameters[42]
    complete_parameters[31] = complete_parameters[33] / complete_parameters[42]
    complete_parameters[38] = (1 - complete_parameters[1]) * complete_parameters[31]
    complete_parameters[39] = (1 - complete_parameters[1]) * complete_parameters[30]
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[18] - (complete_parameters[29] * solution[15] + complete_parameters[27] * solution[6] + complete_parameters[28] * solution[7] + complete_parameters[23] * solution[13]),
        solution[6] - (solution[6] - solution[11]),
        solution[13] - (solution[1] + complete_parameters[3] * (solution[13] - solution[1]) + ((1 - complete_parameters[3] * (1 - complete_parameters[9])) * ((solution[18] - solution[3]) - solution[1])) / complete_parameters[8] + (solution[15] - solution[15]) / complete_parameters[8]),
        solution[10] - ((((complete_parameters[36] * solution[10] + (1 - complete_parameters[36]) * ((solution[18] - solution[3]) - solution[16])) - solution[11] * complete_parameters[15] * complete_parameters[1]) - (1 - complete_parameters[15] * complete_parameters[1]) * (solution[15] - solution[15])) - complete_parameters[10] * ((solution[16] - solution[16]) - complete_parameters[3] * (solution[16] - solution[16]))),
        solution[10] - ((((solution[10] * complete_parameters[37] + (1 - complete_parameters[37]) * (solution[17] - solution[8])) - solution[11] * complete_parameters[1] * complete_parameters[16]) + (1 - complete_parameters[1] * complete_parameters[16]) * (solution[7] - complete_parameters[41] * solution[7])) - complete_parameters[11] * ((solution[8] - solution[8]) - complete_parameters[2] * (solution[8] - solution[8]))),
        solution[10] - (((solution[6] + solution[10] * complete_parameters[1] + solution[17] * (1 - complete_parameters[1]) + solution[16] * complete_parameters[38] + solution[8] * complete_parameters[39]) - solution[6] * complete_parameters[1]) + complete_parameters[11] * ((((solution[16] - solution[16]) * complete_parameters[31] + (solution[8] - solution[8]) * complete_parameters[30]) - (solution[16] - solution[16]) * complete_parameters[1] * complete_parameters[31]) - (solution[8] - solution[8]) * complete_parameters[1] * complete_parameters[30])),
        solution[5] - ((solution[10] + solution[16]) - solution[11]),
        solution[4] - ((solution[10] + solution[8]) - solution[11]),
        solution[18] - ((complete_parameters[5] * (solution[12] + solution[16] * complete_parameters[7] + solution[1] * complete_parameters[6])) / (complete_parameters[5] - ((1 - complete_parameters[7]) - complete_parameters[6])) - (((1 - complete_parameters[7]) - complete_parameters[6]) * (solution[3] + solution[6] * complete_parameters[14] + solution[7] * (1 - complete_parameters[14]))) / (complete_parameters[5] - ((1 - complete_parameters[7]) - complete_parameters[6]))),
        solution[9] - ((complete_parameters[1] * solution[9] - solution[3] * complete_parameters[40]) + solution[14]),
        solution[1] - (solution[13] * complete_parameters[9] + solution[1] * (1 - complete_parameters[9])),
        solution[5] * complete_parameters[26] - ((complete_parameters[23] * solution[13] + complete_parameters[29] * solution[15] + (solution[16] - solution[16]) * complete_parameters[33] + complete_parameters[26] * complete_parameters[24] * ((solution[2] + solution[5]) - solution[9])) - ((1 - complete_parameters[34]) - complete_parameters[35]) * (solution[18] - solution[3])),
        solution[4] * complete_parameters[25] - ((complete_parameters[28] * solution[7] + (solution[8] - solution[8]) * complete_parameters[32] + complete_parameters[24] * complete_parameters[25] * ((solution[2] + solution[4]) - solution[9])) - complete_parameters[35] * (solution[18] - solution[3])),
        solution[2] - ((1 - complete_parameters[20]) * (1 + complete_parameters[21]) * solution[9] + (1 - complete_parameters[20]) * complete_parameters[22] * solution[18] + solution[2] * complete_parameters[20] + 0),
        solution[11] - (solution[2] - solution[9]),
        solution[17] - (complete_parameters[18] * solution[17] + 0),
        solution[14] - (complete_parameters[17] * solution[14] + 0),
        solution[12] - (complete_parameters[19] * solution[12] + 0),
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        (((-(complete_parameters[23]) * solution[13] - complete_parameters[27] * solution[6]) - complete_parameters[28] * solution[7]) - complete_parameters[29] * solution[15]) + solution[18],
        solution[11] - 0,
        ((-(solution[1]) + solution[13]) - complete_parameters[3] * (-(solution[1]) + solution[13])) - ((-(complete_parameters[3]) * (1 - complete_parameters[9]) + 1) * ((-(solution[1]) - solution[3]) + solution[18])) / complete_parameters[8],
        ((complete_parameters[15] * solution[11] * complete_parameters[1] - solution[10] * complete_parameters[36]) + solution[10]) - (1 - complete_parameters[36]) * ((-(solution[3]) - solution[16]) + solution[18]),
        (((complete_parameters[16] * solution[11] * complete_parameters[1] - solution[10] * complete_parameters[37]) + solution[10]) - (1 - complete_parameters[37]) * (-(solution[8]) + solution[17])) - (-(solution[7]) * complete_parameters[41] + solution[7]) * (-(complete_parameters[16]) * complete_parameters[1] + 1),
        (((((solution[6] * complete_parameters[1] - solution[6]) - solution[8] * complete_parameters[39]) - solution[10] * complete_parameters[1]) + solution[10]) - solution[16] * complete_parameters[38]) - solution[17] * (1 - complete_parameters[1]),
        ((solution[5] - solution[10]) + solution[11]) - solution[16],
        ((solution[4] - solution[8]) - solution[10]) + solution[11],
        (solution[18] - (complete_parameters[5] * (solution[1] * complete_parameters[6] + solution[12] + solution[16] * complete_parameters[7])) / ((complete_parameters[6] + complete_parameters[5] + complete_parameters[7]) - 1)) + ((solution[3] + solution[6] * complete_parameters[14] + solution[7] * (1 - complete_parameters[14])) * ((-(complete_parameters[6]) - complete_parameters[7]) + 1)) / ((complete_parameters[6] + complete_parameters[5] + complete_parameters[7]) - 1),
        ((solution[3] * complete_parameters[40] - solution[9] * complete_parameters[1]) + solution[9]) - solution[14],
        (-(solution[1]) * (1 - complete_parameters[9]) + solution[1]) - solution[13] * complete_parameters[9],
        (((-(complete_parameters[23]) * solution[13] - complete_parameters[24] * complete_parameters[26] * ((solution[2] + solution[5]) - solution[9])) + complete_parameters[26] * solution[5]) - complete_parameters[29] * solution[15]) + (-(solution[3]) + solution[18]) * ((-(complete_parameters[34]) - complete_parameters[35]) + 1),
        ((-(complete_parameters[24]) * complete_parameters[25] * ((solution[2] + solution[4]) - solution[9]) + complete_parameters[25] * solution[4]) - complete_parameters[28] * solution[7]) + complete_parameters[35] * (-(solution[3]) + solution[18]),
        ((-(solution[2]) * complete_parameters[20] + solution[2]) - solution[9] * (1 - complete_parameters[20]) * (complete_parameters[21] + 1)) - complete_parameters[22] * solution[18] * (1 - complete_parameters[20]),
        -(solution[2]) + solution[9] + solution[11],
        -(solution[17]) * complete_parameters[18] + solution[17],
        -(solution[14]) * complete_parameters[17] + solution[14],
        -(solution[12]) * complete_parameters[19] + solution[12],
    ]
end

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 4
    @assert length(external_solution) == 0
    @assert length(solution) == 14
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((-(solution[1]) + solution[11]) - complete_parameters[3] * (-(solution[1]) + solution[11])) - ((-(complete_parameters[3]) * (1 - complete_parameters[9]) + 1) * ((-(solution[1]) - solution[3]) + solution[14])) / complete_parameters[8],
        -(solution[2]) + solution[9] + previous_solution[1],
        ((solution[3] * complete_parameters[40] - solution[9] * complete_parameters[1]) + solution[9]) - previous_solution[3],
        ((solution[4] - solution[8]) - solution[10]) + previous_solution[1],
        ((solution[5] - solution[10]) + previous_solution[1]) - solution[13],
        (((-(complete_parameters[23]) * solution[11] - complete_parameters[27] * solution[6]) - complete_parameters[28] * solution[7]) - complete_parameters[29] * solution[12]) + solution[14],
        (((complete_parameters[16] * previous_solution[1] * complete_parameters[1] - solution[10] * complete_parameters[37]) + solution[10]) - (1 - complete_parameters[37]) * (-(solution[8]) + previous_solution[4])) - (-(solution[7]) * complete_parameters[41] + solution[7]) * (-(complete_parameters[16]) * complete_parameters[1] + 1),
        (((((solution[6] * complete_parameters[1] - solution[6]) - solution[8] * complete_parameters[39]) - solution[10] * complete_parameters[1]) + solution[10]) - solution[13] * complete_parameters[38]) - previous_solution[4] * (1 - complete_parameters[1]),
        ((-(solution[2]) * complete_parameters[20] + solution[2]) - solution[9] * (1 - complete_parameters[20]) * (complete_parameters[21] + 1)) - complete_parameters[22] * solution[14] * (1 - complete_parameters[20]),
        ((complete_parameters[15] * previous_solution[1] * complete_parameters[1] - solution[10] * complete_parameters[36]) + solution[10]) - (1 - complete_parameters[36]) * ((-(solution[3]) - solution[13]) + solution[14]),
        (-(solution[1]) * (1 - complete_parameters[9]) + solution[1]) - solution[11] * complete_parameters[9],
        (((-(complete_parameters[23]) * solution[11] - complete_parameters[24] * complete_parameters[26] * ((solution[2] + solution[5]) - solution[9])) + complete_parameters[26] * solution[5]) - complete_parameters[29] * solution[12]) + (-(solution[3]) + solution[14]) * ((-(complete_parameters[34]) - complete_parameters[35]) + 1),
        (solution[14] - (complete_parameters[5] * (solution[1] * complete_parameters[6] + previous_solution[2] + solution[13] * complete_parameters[7])) / ((complete_parameters[6] + complete_parameters[5] + complete_parameters[7]) - 1)) + ((solution[3] + solution[6] * complete_parameters[14] + solution[7] * (1 - complete_parameters[14])) * ((-(complete_parameters[6]) - complete_parameters[7]) + 1)) / ((complete_parameters[6] + complete_parameters[5] + complete_parameters[7]) - 1),
        ((-(complete_parameters[24]) * complete_parameters[25] * ((solution[2] + solution[4]) - solution[9]) + complete_parameters[25] * solution[4]) - complete_parameters[28] * solution[7]) + complete_parameters[35] * (-(solution[3]) + solution[14]),
    ]
end

function residuals_block_2(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) * complete_parameters[17] + solution[1],
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) * complete_parameters[18] + solution[1],
    ]
end

function residuals_block_4(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) * complete_parameters[19] + solution[1],
    ]
end

function residuals_block_5(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0,
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
