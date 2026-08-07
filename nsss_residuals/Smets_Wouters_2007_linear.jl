module Smets_Wouters_2007_linearNsssResiduals
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

const MODEL_NAME = "Smets_Wouters_2007_linear"
const SOURCE_MODEL_FILE = "models/Smets_Wouters_2007_linear.jl"
const NSSS_SOLUTION_ERROR = 0.0
const NSSS_RESIDUAL_NORM = 0.0

const PARAMETER_NAMES = [
    "ctou",
    "clandaw",
    "cg",
    "curvp",
    "curvw",
    "calfa",
    "csigma",
    "cfc",
    "cgy",
    "csadjcost",
    "chabb",
    "cprobw",
    "csigl",
    "cprobp",
    "cindw",
    "cindp",
    "czcap",
    "crpi",
    "crr",
    "cry",
    "crdy",
    "crhoa",
    "crhob",
    "crhog",
    "crhoqs",
    "crhoms",
    "crhopinf",
    "crhow",
    "cmap",
    "cmaw",
    "constelab",
    "constepinf",
    "constebeta",
    "ctrend",
    "z_ea",
    "z_eb",
    "z_eg",
    "z_em",
    "z_ew",
    "z_eqs",
    "z_epinf",
]
const PARAMETER_VALUES = Float64[
    0.025,
    1.5,
    0.18,
    10.0,
    10.0,
    0.24,
    1.5,
    1.5,
    0.51,
    6.0144,
    0.6361,
    0.8087,
    1.9423,
    0.6,
    0.3243,
    0.47,
    0.2696,
    1.488,
    0.8762,
    0.0593,
    0.2347,
    0.9977,
    0.5799,
    0.9957,
    0.7165,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.7,
    0.742,
    0.3982,
    0.4618,
    1.8513,
    0.609,
    0.2397,
    0.2089,
    0.6017,
    0.1455,
]
const COMPLETE_PARAMETER_NAMES = [
    "ctou",
    "clandaw",
    "cg",
    "curvp",
    "curvw",
    "calfa",
    "csigma",
    "cfc",
    "cgy",
    "csadjcost",
    "chabb",
    "cprobw",
    "csigl",
    "cprobp",
    "cindw",
    "cindp",
    "czcap",
    "crpi",
    "crr",
    "cry",
    "crdy",
    "crhoa",
    "crhob",
    "crhog",
    "crhoqs",
    "crhoms",
    "crhopinf",
    "crhow",
    "cmap",
    "cmaw",
    "constelab",
    "constepinf",
    "constebeta",
    "ctrend",
    "z_ea",
    "z_eb",
    "z_eg",
    "z_em",
    "z_ew",
    "z_eqs",
    "z_epinf",
    "cbetabar",
    "ccy",
    "cgamma",
    "cikbar",
    "ciy",
    "conster",
    "crk",
    "crkky",
    "cwhlc",
    "cpie",
    "cbeta",
    "clandap",
    "cr",
    "cw",
    "cik",
    "clk",
    "cky",
]
const COMPLETE_PARAMETER_VALUES = Float64[
    0.025,
    1.5,
    0.18,
    10.0,
    10.0,
    0.24,
    1.5,
    1.5,
    0.51,
    6.0144,
    0.6361,
    0.8087,
    1.9423,
    0.6,
    0.3243,
    0.47,
    0.2696,
    1.488,
    0.8762,
    0.0593,
    0.2347,
    0.9977,
    0.5799,
    0.9957,
    0.7165,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.7,
    0.742,
    0.3982,
    0.4618,
    1.8513,
    0.609,
    0.2397,
    0.2089,
    0.6017,
    0.1455,
    0.9867350192621208,
    0.6390665492639644,
    1.003982,
    0.028867051401319843,
    0.18093345073603553,
    2.0537409073646984,
    0.038443305932122196,
    0.23999999999999996,
    0.7928230123297997,
    1.007,
    0.9926346508903933,
    1.5,
    1.020537409073647,
    0.7948597741356207,
    0.028981999999999897,
    0.1531554865562353,
    6.242959448486515,
]
const ORIGINAL_SOLUTION_NAMES = [
    "a",
    "b",
    "c",
    "cf",
    "dc",
    "dinve",
    "dwobs",
    "dy",
    "epinfma",
    "ewma",
    "g",
    "inve",
    "invef",
    "k",
    "kf",
    "kp",
    "kpf",
    "lab",
    "labf",
    "labobs",
    "mc",
    "ms",
    "pinf",
    "pinfobs",
    "pk",
    "pkf",
    "qs",
    "r",
    "rk",
    "rkf",
    "robs",
    "rrf",
    "spinf",
    "sw",
    "w",
    "wf",
    "y",
    "yf",
    "zcap",
    "zcapf",
]
const ORIGINAL_SOLUTION_VALUES = Float64[
    0.0,
    0.0,
    0.0,
    0.0,
    0.3982,
    0.3982,
    0.3982,
    0.3982,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.7,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.0537409073646984,
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
    "b",
    "c",
    "cf",
    "dc",
    "dinve",
    "dwobs",
    "dy",
    "epinfma",
    "ewma",
    "g",
    "inve",
    "invef",
    "k",
    "kf",
    "kp",
    "kpf",
    "lab",
    "labf",
    "labobs",
    "mc",
    "ms",
    "pinf",
    "pinfobs",
    "pk",
    "pkf",
    "qs",
    "r",
    "rk",
    "rkf",
    "robs",
    "rrf",
    "spinf",
    "sw",
    "w",
    "wf",
    "y",
    "yf",
    "zcap",
    "zcapf",
]
const AUXILIARY_SOLUTION_VALUES = Float64[
    0.0,
    0.0,
    0.0,
    0.0,
    0.3982,
    0.3982,
    0.3982,
    0.3982,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.7,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.0537409073646984,
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
const DEFAULTED_NSSS_SOLUTION_NAMES = [
]
const CALIBRATION_PARAMETER_NAMES = [
]

const ORIGINAL_NSSS_EQUATIONS = Expr[
    :(a - (calfa * rkf + (1 - calfa) * wf)),
    :(zcapf - (rkf * 1) / (czcap / (1 - czcap))),
    :(rkf - ((wf + labf) - kf)),
    :(kf - (zcapf + kpf)),
    :(invef - (qs + (1 / (1 + cgamma * cbetabar)) * ((pkf * 1) / (csadjcost * cgamma ^ 2) + invef + invef * cgamma * cbetabar))),
    :(pkf - ((b * (1 / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)))) - rrf) + rkf * (crk / (crk + (1 - ctou))) + pkf * ((1 - ctou) / (crk + (1 - ctou))))),
    :(cf - ((b + ((cf * chabb) / cgamma) / (1 + chabb / cgamma) + (cf * 1) / (1 + chabb / cgamma) + (labf - labf) * (((csigma - 1) * cwhlc) / (csigma * (1 + chabb / cgamma)))) - (rrf * (1 - chabb / cgamma)) / (csigma * (1 + chabb / cgamma)))),
    :(yf - (g + cf * ccy + invef * ciy + zcapf * crkky)),
    :(yf - cfc * (a + calfa * kf + (1 - calfa) * labf)),
    :(wf - ((labf * csigl + (cf * 1) / (1 - chabb / cgamma)) - ((cf * chabb) / cgamma) / (1 - chabb / cgamma))),
    :(kpf - (kpf * (1 - cikbar) + invef * cikbar + qs * csadjcost * cgamma ^ 2 * cikbar)),
    :(mc - ((calfa * rk + (1 - calfa) * w) - a)),
    :(zcap - (1 / (czcap / (1 - czcap))) * rk),
    :(rk - ((w + lab) - k)),
    :(k - (zcap + kp)),
    :(inve - (qs + (1 / (1 + cgamma * cbetabar)) * ((pk * 1) / (csadjcost * cgamma ^ 2) + inve + inve * cgamma * cbetabar))),
    :(pk - ((pinf - r) + (b * 1) / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))) + rk * (crk / (crk + (1 - ctou))) + pk * ((1 - ctou) / (crk + (1 - ctou))))),
    :(c - ((b + ((c * chabb) / cgamma) / (1 + chabb / cgamma) + (c * 1) / (1 + chabb / cgamma) + (lab - lab) * (((csigma - 1) * cwhlc) / (csigma * (1 + chabb / cgamma)))) - ((r - pinf) * (1 - chabb / cgamma)) / (csigma * (1 + chabb / cgamma)))),
    :(y - (g + c * ccy + inve * ciy + zcap * crkky)),
    :(y - cfc * (a + calfa * k + (1 - calfa) * lab)),
    :(pinf - (spinf + (1 / (1 + cindp * cgamma * cbetabar)) * (cindp * pinf + pinf * cgamma * cbetabar + ((mc * (1 - cprobp) * (1 - cprobp * cgamma * cbetabar)) / cprobp) / (1 + (cfc - 1) * curvp)))),
    :(w - (((sw + (w * 1) / (1 + cgamma * cbetabar) + (w * cgamma * cbetabar) / (1 + cgamma * cbetabar) + (pinf * cindw) / (1 + cgamma * cbetabar)) - (pinf * (1 + cindw * cgamma * cbetabar)) / (1 + cgamma * cbetabar)) + (pinf * cgamma * cbetabar) / (1 + cgamma * cbetabar) + ((((((csigl * lab + (c * 1) / (1 - chabb / cgamma)) - ((c * chabb) / cgamma) / (1 - chabb / cgamma)) - w) * 1) / (1 + (clandaw - 1) * curvw)) * (1 - cprobw) * (1 - cprobw * cgamma * cbetabar)) / (cprobw * (1 + cgamma * cbetabar)))),
    :(r - (pinf * crpi * (1 - crr) + (1 - crr) * cry * (y - yf) + crdy * (((y - yf) - y) + yf) + crr * r + ms)),
    :(a - (crhoa * a + z_ea * 0)),
    :(b - (crhob * b + z_eb * 0)),
    :(g - (crhog * g + z_eg * 0 + z_ea * 0 * cgy)),
    :(qs - (crhoqs * qs + z_eqs * 0)),
    :(ms - (crhoms * ms + z_em * 0)),
    :(spinf - ((crhopinf * spinf + epinfma) - cmap * epinfma)),
    :(epinfma - z_epinf * 0),
    :(sw - ((crhow * sw + ewma) - cmaw * ewma)),
    :(ewma - z_ew * 0),
    :(kp - (kp * (1 - cikbar) + inve * cikbar + qs * csadjcost * cgamma ^ 2 * cikbar)),
    :(dy - ((ctrend + y) - y)),
    :(dc - ((ctrend + c) - c)),
    :(dinve - ((ctrend + inve) - inve)),
    :(pinfobs - (constepinf + pinf)),
    :(robs - (r + conster)),
    :(dwobs - ((ctrend + w) - w)),
    :(labobs - (lab + constelab)),
]
const CALIBRATION_EQUATIONS = Expr[
]
const AUXILIARY_NSSS_EQUATIONS = Expr[
    :((a - calfa * rkf) - wf * (1 - calfa)),
    :(zcapf - (rkf * (1 - czcap)) / czcap),
    :(((kf - labf) + rkf) - wf),
    :((kf - kpf) - zcapf),
    :((invef - qs) - (cbetabar * cgamma * invef + invef + pkf / (cgamma ^ 2 * csadjcost)) / (cbetabar * cgamma + 1)),
    :((((-b * csigma * (1 + chabb / cgamma)) / (1 - chabb / cgamma) - (crk * rkf) / ((crk - ctou) + 1)) - (pkf * (1 - ctou)) / ((crk - ctou) + 1)) + pkf + rrf),
    :((((-b + cf) - cf / (1 + chabb / cgamma)) - (cf * chabb) / (cgamma * (1 + chabb / cgamma))) + (rrf * (1 - chabb / cgamma)) / (csigma * (1 + chabb / cgamma))),
    :((((-ccy * cf - ciy * invef) - crkky * zcapf) - g) + yf),
    :(-cfc * (a + calfa * kf + labf * (1 - calfa)) + yf),
    :(((-cf / (1 - chabb / cgamma) + (cf * chabb) / (cgamma * (1 - chabb / cgamma))) - csigl * labf) + wf),
    :(((-(cgamma ^ 2) * cikbar * csadjcost * qs - cikbar * invef) - kpf * (1 - cikbar)) + kpf),
    :(((a - calfa * rk) + mc) - w * (1 - calfa)),
    :(zcap - (rk * (1 - czcap)) / czcap),
    :(((k - lab) + rk) - w),
    :((k - kp) - zcap),
    :((inve - qs) - (cbetabar * cgamma * inve + inve + pk / (cgamma ^ 2 * csadjcost)) / (cbetabar * cgamma + 1)),
    :(((((-b * csigma * (1 + chabb / cgamma)) / (1 - chabb / cgamma) - (crk * rk) / ((crk - ctou) + 1)) - pinf) - (pk * (1 - ctou)) / ((crk - ctou) + 1)) + pk + r),
    :((((-b + c) - c / (1 + chabb / cgamma)) - (c * chabb) / (cgamma * (1 + chabb / cgamma))) + ((1 - chabb / cgamma) * (-pinf + r)) / (csigma * (1 + chabb / cgamma))),
    :((((-c * ccy - ciy * inve) - crkky * zcap) - g) + y),
    :(-cfc * (a + calfa * k + lab * (1 - calfa)) + y),
    :((pinf - spinf) - (cbetabar * cgamma * pinf + cindp * pinf + (mc * (1 - cprobp) * (-cbetabar * cgamma * cprobp + 1)) / (cprobp * (curvp * (cfc - 1) + 1))) / (cbetabar * cgamma * cindp + 1)),
    :((((((((-cbetabar * cgamma * pinf) / (cbetabar * cgamma + 1) - (cbetabar * cgamma * w) / (cbetabar * cgamma + 1)) - (cindw * pinf) / (cbetabar * cgamma + 1)) + (pinf * (cbetabar * cgamma * cindw + 1)) / (cbetabar * cgamma + 1)) - sw) + w) - w / (cbetabar * cgamma + 1)) - ((1 - cprobw) * (-cbetabar * cgamma * cprobw + 1) * (((c / (1 - chabb / cgamma) - (c * chabb) / (cgamma * (1 - chabb / cgamma))) + csigl * lab) - w)) / (cprobw * (cbetabar * cgamma + 1) * (curvw * (clandaw - 1) + 1))),
    :((((-crpi * pinf * (1 - crr) - crr * r) - cry * (1 - crr) * (y - yf)) - ms) + r),
    :(-a * crhoa + a),
    :(-b * crhob + b),
    :(-crhog * g + g),
    :(-crhoqs * qs + qs),
    :(-crhoms * ms + ms),
    :(((cmap * epinfma - crhopinf * spinf) - epinfma) + spinf),
    :(epinfma - 0),
    :(((cmaw * ewma - crhow * sw) - ewma) + sw),
    :(ewma - 0),
    :(((-(cgamma ^ 2) * cikbar * csadjcost * qs - cikbar * inve) - kp * (1 - cikbar)) + kp),
    :(-ctrend + dy),
    :(-ctrend + dc),
    :(-ctrend + dinve),
    :((-constepinf - pinf) + pinfobs),
    :((-conster - r) + robs),
    :(-ctrend + dwobs),
    :((-constelab - lab) + labobs),
]
const ORIGINAL_RESIDUAL_EQUATIONS = Expr[
    :(a - (calfa * rkf + (1 - calfa) * wf)),
    :(zcapf - (rkf * 1) / (czcap / (1 - czcap))),
    :(rkf - ((wf + labf) - kf)),
    :(kf - (zcapf + kpf)),
    :(invef - (qs + (1 / (1 + cgamma * cbetabar)) * ((pkf * 1) / (csadjcost * cgamma ^ 2) + invef + invef * cgamma * cbetabar))),
    :(pkf - ((b * (1 / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)))) - rrf) + rkf * (crk / (crk + (1 - ctou))) + pkf * ((1 - ctou) / (crk + (1 - ctou))))),
    :(cf - ((b + ((cf * chabb) / cgamma) / (1 + chabb / cgamma) + (cf * 1) / (1 + chabb / cgamma) + (labf - labf) * (((csigma - 1) * cwhlc) / (csigma * (1 + chabb / cgamma)))) - (rrf * (1 - chabb / cgamma)) / (csigma * (1 + chabb / cgamma)))),
    :(yf - (g + cf * ccy + invef * ciy + zcapf * crkky)),
    :(yf - cfc * (a + calfa * kf + (1 - calfa) * labf)),
    :(wf - ((labf * csigl + (cf * 1) / (1 - chabb / cgamma)) - ((cf * chabb) / cgamma) / (1 - chabb / cgamma))),
    :(kpf - (kpf * (1 - cikbar) + invef * cikbar + qs * csadjcost * cgamma ^ 2 * cikbar)),
    :(mc - ((calfa * rk + (1 - calfa) * w) - a)),
    :(zcap - (1 / (czcap / (1 - czcap))) * rk),
    :(rk - ((w + lab) - k)),
    :(k - (zcap + kp)),
    :(inve - (qs + (1 / (1 + cgamma * cbetabar)) * ((pk * 1) / (csadjcost * cgamma ^ 2) + inve + inve * cgamma * cbetabar))),
    :(pk - ((pinf - r) + (b * 1) / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))) + rk * (crk / (crk + (1 - ctou))) + pk * ((1 - ctou) / (crk + (1 - ctou))))),
    :(c - ((b + ((c * chabb) / cgamma) / (1 + chabb / cgamma) + (c * 1) / (1 + chabb / cgamma) + (lab - lab) * (((csigma - 1) * cwhlc) / (csigma * (1 + chabb / cgamma)))) - ((r - pinf) * (1 - chabb / cgamma)) / (csigma * (1 + chabb / cgamma)))),
    :(y - (g + c * ccy + inve * ciy + zcap * crkky)),
    :(y - cfc * (a + calfa * k + (1 - calfa) * lab)),
    :(pinf - (spinf + (1 / (1 + cindp * cgamma * cbetabar)) * (cindp * pinf + pinf * cgamma * cbetabar + ((mc * (1 - cprobp) * (1 - cprobp * cgamma * cbetabar)) / cprobp) / (1 + (cfc - 1) * curvp)))),
    :(w - (((sw + (w * 1) / (1 + cgamma * cbetabar) + (w * cgamma * cbetabar) / (1 + cgamma * cbetabar) + (pinf * cindw) / (1 + cgamma * cbetabar)) - (pinf * (1 + cindw * cgamma * cbetabar)) / (1 + cgamma * cbetabar)) + (pinf * cgamma * cbetabar) / (1 + cgamma * cbetabar) + ((((((csigl * lab + (c * 1) / (1 - chabb / cgamma)) - ((c * chabb) / cgamma) / (1 - chabb / cgamma)) - w) * 1) / (1 + (clandaw - 1) * curvw)) * (1 - cprobw) * (1 - cprobw * cgamma * cbetabar)) / (cprobw * (1 + cgamma * cbetabar)))),
    :(r - (pinf * crpi * (1 - crr) + (1 - crr) * cry * (y - yf) + crdy * (((y - yf) - y) + yf) + crr * r + ms)),
    :(a - (crhoa * a + z_ea * 0)),
    :(b - (crhob * b + z_eb * 0)),
    :(g - (crhog * g + z_eg * 0 + z_ea * 0 * cgy)),
    :(qs - (crhoqs * qs + z_eqs * 0)),
    :(ms - (crhoms * ms + z_em * 0)),
    :(spinf - ((crhopinf * spinf + epinfma) - cmap * epinfma)),
    :(epinfma - z_epinf * 0),
    :(sw - ((crhow * sw + ewma) - cmaw * ewma)),
    :(ewma - z_ew * 0),
    :(kp - (kp * (1 - cikbar) + inve * cikbar + qs * csadjcost * cgamma ^ 2 * cikbar)),
    :(dy - ((ctrend + y) - y)),
    :(dc - ((ctrend + c) - c)),
    :(dinve - ((ctrend + inve) - inve)),
    :(pinfobs - (constepinf + pinf)),
    :(robs - (r + conster)),
    :(dwobs - ((ctrend + w) - w)),
    :(labobs - (lab + constelab)),
]
const AUXILIARY_RESIDUAL_EQUATIONS = Expr[
    :((a - calfa * rkf) - wf * (1 - calfa)),
    :(zcapf - (rkf * (1 - czcap)) / czcap),
    :(((kf - labf) + rkf) - wf),
    :((kf - kpf) - zcapf),
    :((invef - qs) - (cbetabar * cgamma * invef + invef + pkf / (cgamma ^ 2 * csadjcost)) / (cbetabar * cgamma + 1)),
    :((((-b * csigma * (1 + chabb / cgamma)) / (1 - chabb / cgamma) - (crk * rkf) / ((crk - ctou) + 1)) - (pkf * (1 - ctou)) / ((crk - ctou) + 1)) + pkf + rrf),
    :((((-b + cf) - cf / (1 + chabb / cgamma)) - (cf * chabb) / (cgamma * (1 + chabb / cgamma))) + (rrf * (1 - chabb / cgamma)) / (csigma * (1 + chabb / cgamma))),
    :((((-ccy * cf - ciy * invef) - crkky * zcapf) - g) + yf),
    :(-cfc * (a + calfa * kf + labf * (1 - calfa)) + yf),
    :(((-cf / (1 - chabb / cgamma) + (cf * chabb) / (cgamma * (1 - chabb / cgamma))) - csigl * labf) + wf),
    :(((-(cgamma ^ 2) * cikbar * csadjcost * qs - cikbar * invef) - kpf * (1 - cikbar)) + kpf),
    :(((a - calfa * rk) + mc) - w * (1 - calfa)),
    :(zcap - (rk * (1 - czcap)) / czcap),
    :(((k - lab) + rk) - w),
    :((k - kp) - zcap),
    :((inve - qs) - (cbetabar * cgamma * inve + inve + pk / (cgamma ^ 2 * csadjcost)) / (cbetabar * cgamma + 1)),
    :(((((-b * csigma * (1 + chabb / cgamma)) / (1 - chabb / cgamma) - (crk * rk) / ((crk - ctou) + 1)) - pinf) - (pk * (1 - ctou)) / ((crk - ctou) + 1)) + pk + r),
    :((((-b + c) - c / (1 + chabb / cgamma)) - (c * chabb) / (cgamma * (1 + chabb / cgamma))) + ((1 - chabb / cgamma) * (-pinf + r)) / (csigma * (1 + chabb / cgamma))),
    :((((-c * ccy - ciy * inve) - crkky * zcap) - g) + y),
    :(-cfc * (a + calfa * k + lab * (1 - calfa)) + y),
    :((pinf - spinf) - (cbetabar * cgamma * pinf + cindp * pinf + (mc * (1 - cprobp) * (-cbetabar * cgamma * cprobp + 1)) / (cprobp * (curvp * (cfc - 1) + 1))) / (cbetabar * cgamma * cindp + 1)),
    :((((((((-cbetabar * cgamma * pinf) / (cbetabar * cgamma + 1) - (cbetabar * cgamma * w) / (cbetabar * cgamma + 1)) - (cindw * pinf) / (cbetabar * cgamma + 1)) + (pinf * (cbetabar * cgamma * cindw + 1)) / (cbetabar * cgamma + 1)) - sw) + w) - w / (cbetabar * cgamma + 1)) - ((1 - cprobw) * (-cbetabar * cgamma * cprobw + 1) * (((c / (1 - chabb / cgamma) - (c * chabb) / (cgamma * (1 - chabb / cgamma))) + csigl * lab) - w)) / (cprobw * (cbetabar * cgamma + 1) * (curvw * (clandaw - 1) + 1))),
    :((((-crpi * pinf * (1 - crr) - crr * r) - cry * (1 - crr) * (y - yf)) - ms) + r),
    :(-a * crhoa + a),
    :(-b * crhob + b),
    :(-crhog * g + g),
    :(-crhoqs * qs + qs),
    :(-crhoms * ms + ms),
    :(((cmap * epinfma - crhopinf * spinf) - epinfma) + spinf),
    :(epinfma - 0),
    :(((cmaw * ewma - crhow * sw) - ewma) + sw),
    :(ewma - 0),
    :(((-(cgamma ^ 2) * cikbar * csadjcost * qs - cikbar * inve) - kp * (1 - cikbar)) + kp),
    :(-ctrend + dy),
    :(-ctrend + dc),
    :(-ctrend + dinve),
    :((-constepinf - pinf) + pinfobs),
    :((-conster - r) + robs),
    :(-ctrend + dwobs),
    :((-constelab - lab) + labobs),
]

const PARAMETER_DEFINITION_NAMES = [
    "cbeta",
    "cgamma",
    "cikbar",
    "clandap",
    "cpie",
    "cbetabar",
    "cik",
    "cr",
    "crk",
    "cw",
    "clk",
    "conster",
    "cky",
    "crkky",
    "ccy",
    "ciy",
    "cwhlc",
]
const PARAMETER_DEFINITION_EXPRESSIONS = [
    "1 / (1 + constebeta / 100)",
    "1 + ctrend / 100",
    "1 - (1 - ctou) / cgamma",
    "cfc",
    "1 + constepinf / 100",
    "cbeta * cgamma ^ -csigma",
    "cikbar * cgamma",
    "cpie / cbetabar",
    "1 / cbetabar - (1 - ctou)",
    "((calfa ^ calfa * (1 - calfa) ^ (1 - calfa)) / (clandap * crk ^ calfa)) ^ (1 / (1 - calfa))",
    "(((1 - calfa) / calfa) * crk) / cw",
    "(cr - 1) * 100",
    "cfc * clk ^ (calfa - 1)",
    "crk * cky",
    "(1 - cg) - cik * cky",
    "cik * cky",
    "((((1 / clandaw) * (1 - calfa)) / calfa) * crk * cky) / ccy",
]
const PARAMETER_BOX_CONSTRAINT_NAMES = [
    "ctou",
    "clandaw",
    "cg",
    "curvp",
    "curvw",
    "calfa",
    "csigma",
    "cfc",
    "cgy",
    "csadjcost",
    "chabb",
    "cprobw",
    "csigl",
    "cprobp",
    "cindw",
    "cindp",
    "czcap",
    "crpi",
    "crr",
    "cry",
    "crdy",
    "crhoa",
    "crhob",
    "crhog",
    "crhoqs",
    "crhoms",
    "crhopinf",
    "crhow",
    "cmap",
    "cmaw",
    "constelab",
    "constepinf",
    "constebeta",
    "ctrend",
    "z_ea",
    "z_eb",
    "z_eg",
    "z_em",
    "z_ew",
    "z_eqs",
    "z_epinf",
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
    Inf,
]
const ORIGINAL_BOX_CONSTRAINT_NAMES = [
    "a",
    "b",
    "c",
    "cf",
    "dc",
    "dinve",
    "dwobs",
    "dy",
    "epinfma",
    "ewma",
    "g",
    "inve",
    "invef",
    "k",
    "kf",
    "kp",
    "kpf",
    "lab",
    "labf",
    "labobs",
    "mc",
    "ms",
    "pinf",
    "pinfobs",
    "pk",
    "pkf",
    "qs",
    "r",
    "rk",
    "rkf",
    "robs",
    "rrf",
    "spinf",
    "sw",
    "w",
    "wf",
    "y",
    "yf",
    "zcap",
    "zcapf",
]
const ORIGINAL_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
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
    -1.0e12,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
]
const ORIGINAL_BOX_UPPER_BOUNDS = Float64[
    Inf,
    Inf,
    1.0e12,
    1.0e12,
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
    1.0e12,
    Inf,
    1.0e12,
    Inf,
    Inf,
    Inf,
    Inf,
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
]
const AUXILIARY_BOX_CONSTRAINT_NAMES = [
    "a",
    "b",
    "c",
    "cf",
    "dc",
    "dinve",
    "dwobs",
    "dy",
    "epinfma",
    "ewma",
    "g",
    "inve",
    "invef",
    "k",
    "kf",
    "kp",
    "kpf",
    "lab",
    "labf",
    "labobs",
    "mc",
    "ms",
    "pinf",
    "pinfobs",
    "pk",
    "pkf",
    "qs",
    "r",
    "rk",
    "rkf",
    "robs",
    "rrf",
    "spinf",
    "sw",
    "w",
    "wf",
    "y",
    "yf",
    "zcap",
    "zcapf",
]
const AUXILIARY_BOX_LOWER_BOUNDS = Float64[
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
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
    -1.0e12,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -Inf,
    -1.0e12,
    -Inf,
    -Inf,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
    -1.0e12,
]
const AUXILIARY_BOX_UPPER_BOUNDS = Float64[
    Inf,
    Inf,
    1.0e12,
    1.0e12,
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
    1.0e12,
    Inf,
    1.0e12,
    Inf,
    Inf,
    Inf,
    Inf,
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
        solve_order = 18,
        variables = ["robs"],
        previous_solution_names = ["r"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [38],
        equations = Expr[
            :((-conster - r) + robs),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["robs"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [2.0537409073646984],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 2,
        solve_order = 17,
        variables = ["pinfobs"],
        previous_solution_names = ["pinf"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [37],
        equations = Expr[
            :((-constepinf - pinf) + pinfobs),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["pinfobs"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.7],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 3,
        solve_order = 16,
        variables = ["labobs"],
        previous_solution_names = ["lab"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [40],
        equations = Expr[
            :((-constelab - lab) + labobs),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["labobs"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 4,
        solve_order = 15,
        variables = ["dy"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [34],
        equations = Expr[
            :(-ctrend + dy),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["dy"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.3982],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 5,
        solve_order = 14,
        variables = ["dwobs"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [39],
        equations = Expr[
            :(-ctrend + dwobs),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["dwobs"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.3982],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 6,
        solve_order = 13,
        variables = ["dinve"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [36],
        equations = Expr[
            :(-ctrend + dinve),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["dinve"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.3982],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 7,
        solve_order = 12,
        variables = ["dc"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [35],
        equations = Expr[
            :(-ctrend + dc),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["dc"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.3982],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 8,
        solve_order = 11,
        variables = ["c", "inve", "k", "kp", "lab", "mc", "pinf", "pk", "r", "rk", "w", "y", "zcap"],
        previous_solution_names = ["a", "b", "g", "ms", "qs", "spinf", "sw", "yf"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [18, 33, 20, 15, 22, 21, 17, 16, 23, 14, 12, 19, 13],
        equations = Expr[
            :((((-b + c) - c / (1 + chabb / cgamma)) - (c * chabb) / (cgamma * (1 + chabb / cgamma))) + ((1 - chabb / cgamma) * (-pinf + r)) / (csigma * (1 + chabb / cgamma))),
            :(((-(cgamma ^ 2) * cikbar * csadjcost * qs - cikbar * inve) - kp * (1 - cikbar)) + kp),
            :(-cfc * (a + calfa * k + lab * (1 - calfa)) + y),
            :((k - kp) - zcap),
            :((((((((-cbetabar * cgamma * pinf) / (cbetabar * cgamma + 1) - (cbetabar * cgamma * w) / (cbetabar * cgamma + 1)) - (cindw * pinf) / (cbetabar * cgamma + 1)) + (pinf * (cbetabar * cgamma * cindw + 1)) / (cbetabar * cgamma + 1)) - sw) + w) - w / (cbetabar * cgamma + 1)) - ((1 - cprobw) * (-cbetabar * cgamma * cprobw + 1) * (((c / (1 - chabb / cgamma) - (c * chabb) / (cgamma * (1 - chabb / cgamma))) + csigl * lab) - w)) / (cprobw * (cbetabar * cgamma + 1) * (curvw * (clandaw - 1) + 1))),
            :((pinf - spinf) - (cbetabar * cgamma * pinf + cindp * pinf + (mc * (1 - cprobp) * (-cbetabar * cgamma * cprobp + 1)) / (cprobp * (curvp * (cfc - 1) + 1))) / (cbetabar * cgamma * cindp + 1)),
            :(((((-b * csigma * (1 + chabb / cgamma)) / (1 - chabb / cgamma) - (crk * rk) / ((crk - ctou) + 1)) - pinf) - (pk * (1 - ctou)) / ((crk - ctou) + 1)) + pk + r),
            :((inve - qs) - (cbetabar * cgamma * inve + inve + pk / (cgamma ^ 2 * csadjcost)) / (cbetabar * cgamma + 1)),
            :((((-crpi * pinf * (1 - crr) - crr * r) - cry * (1 - crr) * (y - yf)) - ms) + r),
            :(((k - lab) + rk) - w),
            :(((a - calfa * rk) + mc) - w * (1 - calfa)),
            :((((-c * ccy - ciy * inve) - crkky * zcap) - g) + y),
            :(zcap - (rk * (1 - czcap)) / czcap),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["c", "inve", "k", "kp", "lab", "mc", "pinf", "pk", "r", "rk", "w", "y", "zcap"],
        previous_solution_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        box_lower_bounds = [-1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -Inf, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, Inf, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 9,
        solve_order = 10,
        variables = ["cf", "invef", "kf", "kpf", "labf", "pkf", "rkf", "rrf", "wf", "yf", "zcapf"],
        previous_solution_names = ["a", "b", "g", "qs"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [7, 11, 3, 4, 9, 5, 1, 6, 10, 8, 2],
        equations = Expr[
            :((((-b + cf) - cf / (1 + chabb / cgamma)) - (cf * chabb) / (cgamma * (1 + chabb / cgamma))) + (rrf * (1 - chabb / cgamma)) / (csigma * (1 + chabb / cgamma))),
            :(((-(cgamma ^ 2) * cikbar * csadjcost * qs - cikbar * invef) - kpf * (1 - cikbar)) + kpf),
            :(((kf - labf) + rkf) - wf),
            :((kf - kpf) - zcapf),
            :(-cfc * (a + calfa * kf + labf * (1 - calfa)) + yf),
            :((invef - qs) - (cbetabar * cgamma * invef + invef + pkf / (cgamma ^ 2 * csadjcost)) / (cbetabar * cgamma + 1)),
            :((a - calfa * rkf) - wf * (1 - calfa)),
            :((((-b * csigma * (1 + chabb / cgamma)) / (1 - chabb / cgamma) - (crk * rkf) / ((crk - ctou) + 1)) - (pkf * (1 - ctou)) / ((crk - ctou) + 1)) + pkf + rrf),
            :(((-cf / (1 - chabb / cgamma) + (cf * chabb) / (cgamma * (1 - chabb / cgamma))) - csigl * labf) + wf),
            :((((-ccy * cf - ciy * invef) - crkky * zcapf) - g) + yf),
            :(zcapf - (rkf * (1 - czcap)) / czcap),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["cf", "invef", "kf", "kpf", "labf", "pkf", "rkf", "rrf", "wf", "yf", "zcapf"],
        previous_solution_values = [0.0, 0.0, 0.0, 0.0],
        external_solution_values = Float64[],
        solution_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        box_lower_bounds = [-1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -Inf, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12],
        box_upper_bounds = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, Inf, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12],
    ),
    (
        index = 10,
        solve_order = 9,
        variables = ["ms"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [28],
        equations = Expr[
            :(-crhoms * ms + ms),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ms"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 11,
        solve_order = 8,
        variables = ["qs"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [27],
        equations = Expr[
            :(-crhoqs * qs + qs),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["qs"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 12,
        solve_order = 7,
        variables = ["g"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [26],
        equations = Expr[
            :(-crhog * g + g),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["g"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 13,
        solve_order = 6,
        variables = ["spinf"],
        previous_solution_names = ["epinfma"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [29],
        equations = Expr[
            :(((cmap * epinfma - crhopinf * spinf) - epinfma) + spinf),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["spinf"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 14,
        solve_order = 5,
        variables = ["epinfma"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [30],
        equations = Expr[
            :(epinfma - 0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["epinfma"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 15,
        solve_order = 4,
        variables = ["sw"],
        previous_solution_names = ["ewma"],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [31],
        equations = Expr[
            :(((cmaw * ewma - crhow * sw) - ewma) + sw),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["sw"],
        previous_solution_values = [0.0],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 16,
        solve_order = 3,
        variables = ["ewma"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [32],
        equations = Expr[
            :(ewma - 0),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["ewma"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 17,
        solve_order = 2,
        variables = ["b"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [25],
        equations = Expr[
            :(-b * crhob + b),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["b"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
    (
        index = 18,
        solve_order = 1,
        variables = ["a"],
        previous_solution_names = String[],
        external_solution_names = String[],
        domain_auxiliary_names = String[],
        equation_indices = [24],
        equations = Expr[
            :(-a * crhoa + a),
        ],
        domain_auxiliary_equations = Expr[
        ],
        domain_auxiliary_error_equations = Expr[
        ],
        solution_names = ["a"],
        previous_solution_values = Float64[],
        external_solution_values = Float64[],
        solution_values = [0.0],
        box_lower_bounds = [-Inf],
        box_upper_bounds = [Inf],
    ),
]
const BLOCK_EQUATION_ORDER = [38, 37, 40, 34, 39, 36, 35, 18, 33, 20, 15, 22, 21, 17, 16, 23, 14, 12, 19, 13, 7, 11, 3, 4, 9, 5, 1, 6, 10, 8, 2, 28, 27, 26, 29, 30, 31, 32, 25, 24]
const BLOCK_SOLVE_ORDER = [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
const BLOCK_PREVIOUS_SOLUTION_NAMES = [
    ["r"],
    ["pinf"],
    ["lab"],
    String[],
    String[],
    String[],
    String[],
    ["a", "b", "g", "ms", "qs", "spinf", "sw", "yf"],
    ["a", "b", "g", "qs"],
    String[],
    String[],
    String[],
    ["epinfma"],
    String[],
    ["ewma"],
    String[],
    String[],
    String[],
]
const BLOCK_PREVIOUS_SOLUTION_VALUES = [
    [0.0],
    [0.0],
    [0.0],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    Float64[],
    Float64[],
    Float64[],
    [0.0],
    Float64[],
    [0.0],
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
]
const BLOCK_SOLUTION_NAMES = [
    ["robs"],
    ["pinfobs"],
    ["labobs"],
    ["dy"],
    ["dwobs"],
    ["dinve"],
    ["dc"],
    ["c", "inve", "k", "kp", "lab", "mc", "pinf", "pk", "r", "rk", "w", "y", "zcap"],
    ["cf", "invef", "kf", "kpf", "labf", "pkf", "rkf", "rrf", "wf", "yf", "zcapf"],
    ["ms"],
    ["qs"],
    ["g"],
    ["spinf"],
    ["epinfma"],
    ["sw"],
    ["ewma"],
    ["b"],
    ["a"],
]
const BLOCK_SOLUTION_VALUES = [
    [2.0537409073646984],
    [0.7],
    [0.0],
    [0.3982],
    [0.3982],
    [0.3982],
    [0.3982],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
]

function complete_parameter_values(parameters::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))
    complete_parameters[19] = parameters[19]
    complete_parameters[26] = parameters[26]
    complete_parameters[2] = parameters[2]
    complete_parameters[33] = parameters[33]
    complete_parameters[7] = parameters[7]
    complete_parameters[20] = parameters[20]
    complete_parameters[37] = parameters[37]
    complete_parameters[38] = parameters[38]
    complete_parameters[16] = parameters[16]
    complete_parameters[23] = parameters[23]
    complete_parameters[8] = parameters[8]
    complete_parameters[5] = parameters[5]
    complete_parameters[30] = parameters[30]
    complete_parameters[3] = parameters[3]
    complete_parameters[25] = parameters[25]
    complete_parameters[12] = parameters[12]
    complete_parameters[24] = parameters[24]
    complete_parameters[11] = parameters[11]
    complete_parameters[14] = parameters[14]
    complete_parameters[15] = parameters[15]
    complete_parameters[28] = parameters[28]
    complete_parameters[6] = parameters[6]
    complete_parameters[1] = parameters[1]
    complete_parameters[39] = parameters[39]
    complete_parameters[41] = parameters[41]
    complete_parameters[29] = parameters[29]
    complete_parameters[4] = parameters[4]
    complete_parameters[13] = parameters[13]
    complete_parameters[18] = parameters[18]
    complete_parameters[35] = parameters[35]
    complete_parameters[9] = parameters[9]
    complete_parameters[10] = parameters[10]
    complete_parameters[32] = parameters[32]
    complete_parameters[27] = parameters[27]
    complete_parameters[31] = parameters[31]
    complete_parameters[40] = parameters[40]
    complete_parameters[22] = parameters[22]
    complete_parameters[17] = parameters[17]
    complete_parameters[21] = parameters[21]
    complete_parameters[36] = parameters[36]
    complete_parameters[34] = parameters[34]
    complete_parameters[52] = 1 / (1 + complete_parameters[33] / 100)
    complete_parameters[44] = 1 + complete_parameters[34] / 100
    complete_parameters[45] = 1 - (1 - complete_parameters[1]) / complete_parameters[44]
    complete_parameters[53] = complete_parameters[8]
    complete_parameters[51] = 1 + complete_parameters[32] / 100
    complete_parameters[42] = complete_parameters[52] * complete_parameters[44] ^ -(complete_parameters[7])
    complete_parameters[56] = complete_parameters[45] * complete_parameters[44]
    complete_parameters[54] = complete_parameters[51] / complete_parameters[42]
    complete_parameters[48] = 1 / complete_parameters[42] - (1 - complete_parameters[1])
    complete_parameters[55] = ((complete_parameters[6] ^ complete_parameters[6] * (1 - complete_parameters[6]) ^ (1 - complete_parameters[6])) / (complete_parameters[53] * complete_parameters[48] ^ complete_parameters[6])) ^ (1 / (1 - complete_parameters[6]))
    complete_parameters[57] = (((1 - complete_parameters[6]) / complete_parameters[6]) * complete_parameters[48]) / complete_parameters[55]
    complete_parameters[47] = (complete_parameters[54] - 1) * 100
    complete_parameters[58] = complete_parameters[8] * complete_parameters[57] ^ (complete_parameters[6] - 1)
    complete_parameters[49] = complete_parameters[48] * complete_parameters[58]
    complete_parameters[43] = (1 - complete_parameters[3]) - complete_parameters[56] * complete_parameters[58]
    complete_parameters[46] = complete_parameters[56] * complete_parameters[58]
    complete_parameters[50] = ((((1 / complete_parameters[2]) * (1 - complete_parameters[6])) / complete_parameters[6]) * complete_parameters[48] * complete_parameters[58]) / complete_parameters[43]
    return complete_parameters
end

function residuals_original(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(ORIGINAL_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - (complete_parameters[6] * solution[30] + (1 - complete_parameters[6]) * solution[36]),
        solution[40] - (solution[30] * 1) / (complete_parameters[17] / (1 - complete_parameters[17])),
        solution[30] - ((solution[36] + solution[19]) - solution[15]),
        solution[15] - (solution[40] + solution[17]),
        solution[13] - (solution[27] + (1 / (1 + complete_parameters[44] * complete_parameters[42])) * ((solution[26] * 1) / (complete_parameters[10] * complete_parameters[44] ^ 2) + solution[13] + solution[13] * complete_parameters[44] * complete_parameters[42])),
        solution[26] - ((solution[2] * (1 / ((1 - complete_parameters[11] / complete_parameters[44]) / (complete_parameters[7] * (1 + complete_parameters[11] / complete_parameters[44])))) - solution[32]) + solution[30] * (complete_parameters[48] / (complete_parameters[48] + (1 - complete_parameters[1]))) + solution[26] * ((1 - complete_parameters[1]) / (complete_parameters[48] + (1 - complete_parameters[1])))),
        solution[4] - ((solution[2] + ((solution[4] * complete_parameters[11]) / complete_parameters[44]) / (1 + complete_parameters[11] / complete_parameters[44]) + (solution[4] * 1) / (1 + complete_parameters[11] / complete_parameters[44]) + (solution[19] - solution[19]) * (((complete_parameters[7] - 1) * complete_parameters[50]) / (complete_parameters[7] * (1 + complete_parameters[11] / complete_parameters[44])))) - (solution[32] * (1 - complete_parameters[11] / complete_parameters[44])) / (complete_parameters[7] * (1 + complete_parameters[11] / complete_parameters[44]))),
        solution[38] - (solution[11] + solution[4] * complete_parameters[43] + solution[13] * complete_parameters[46] + solution[40] * complete_parameters[49]),
        solution[38] - complete_parameters[8] * (solution[1] + complete_parameters[6] * solution[15] + (1 - complete_parameters[6]) * solution[19]),
        solution[36] - ((solution[19] * complete_parameters[13] + (solution[4] * 1) / (1 - complete_parameters[11] / complete_parameters[44])) - ((solution[4] * complete_parameters[11]) / complete_parameters[44]) / (1 - complete_parameters[11] / complete_parameters[44])),
        solution[17] - (solution[17] * (1 - complete_parameters[45]) + solution[13] * complete_parameters[45] + solution[27] * complete_parameters[10] * complete_parameters[44] ^ 2 * complete_parameters[45]),
        solution[21] - ((complete_parameters[6] * solution[29] + (1 - complete_parameters[6]) * solution[35]) - solution[1]),
        solution[39] - (1 / (complete_parameters[17] / (1 - complete_parameters[17]))) * solution[29],
        solution[29] - ((solution[35] + solution[18]) - solution[14]),
        solution[14] - (solution[39] + solution[16]),
        solution[12] - (solution[27] + (1 / (1 + complete_parameters[44] * complete_parameters[42])) * ((solution[25] * 1) / (complete_parameters[10] * complete_parameters[44] ^ 2) + solution[12] + solution[12] * complete_parameters[44] * complete_parameters[42])),
        solution[25] - ((solution[23] - solution[28]) + (solution[2] * 1) / ((1 - complete_parameters[11] / complete_parameters[44]) / (complete_parameters[7] * (1 + complete_parameters[11] / complete_parameters[44]))) + solution[29] * (complete_parameters[48] / (complete_parameters[48] + (1 - complete_parameters[1]))) + solution[25] * ((1 - complete_parameters[1]) / (complete_parameters[48] + (1 - complete_parameters[1])))),
        solution[3] - ((solution[2] + ((solution[3] * complete_parameters[11]) / complete_parameters[44]) / (1 + complete_parameters[11] / complete_parameters[44]) + (solution[3] * 1) / (1 + complete_parameters[11] / complete_parameters[44]) + (solution[18] - solution[18]) * (((complete_parameters[7] - 1) * complete_parameters[50]) / (complete_parameters[7] * (1 + complete_parameters[11] / complete_parameters[44])))) - ((solution[28] - solution[23]) * (1 - complete_parameters[11] / complete_parameters[44])) / (complete_parameters[7] * (1 + complete_parameters[11] / complete_parameters[44]))),
        solution[37] - (solution[11] + solution[3] * complete_parameters[43] + solution[12] * complete_parameters[46] + solution[39] * complete_parameters[49]),
        solution[37] - complete_parameters[8] * (solution[1] + complete_parameters[6] * solution[14] + (1 - complete_parameters[6]) * solution[18]),
        solution[23] - (solution[33] + (1 / (1 + complete_parameters[16] * complete_parameters[44] * complete_parameters[42])) * (complete_parameters[16] * solution[23] + solution[23] * complete_parameters[44] * complete_parameters[42] + ((solution[21] * (1 - complete_parameters[14]) * (1 - complete_parameters[14] * complete_parameters[44] * complete_parameters[42])) / complete_parameters[14]) / (1 + (complete_parameters[8] - 1) * complete_parameters[4]))),
        solution[35] - (((solution[34] + (solution[35] * 1) / (1 + complete_parameters[44] * complete_parameters[42]) + (solution[35] * complete_parameters[44] * complete_parameters[42]) / (1 + complete_parameters[44] * complete_parameters[42]) + (solution[23] * complete_parameters[15]) / (1 + complete_parameters[44] * complete_parameters[42])) - (solution[23] * (1 + complete_parameters[15] * complete_parameters[44] * complete_parameters[42])) / (1 + complete_parameters[44] * complete_parameters[42])) + (solution[23] * complete_parameters[44] * complete_parameters[42]) / (1 + complete_parameters[44] * complete_parameters[42]) + ((((((complete_parameters[13] * solution[18] + (solution[3] * 1) / (1 - complete_parameters[11] / complete_parameters[44])) - ((solution[3] * complete_parameters[11]) / complete_parameters[44]) / (1 - complete_parameters[11] / complete_parameters[44])) - solution[35]) * 1) / (1 + (complete_parameters[2] - 1) * complete_parameters[5])) * (1 - complete_parameters[12]) * (1 - complete_parameters[12] * complete_parameters[44] * complete_parameters[42])) / (complete_parameters[12] * (1 + complete_parameters[44] * complete_parameters[42]))),
        solution[28] - (solution[23] * complete_parameters[18] * (1 - complete_parameters[19]) + (1 - complete_parameters[19]) * complete_parameters[20] * (solution[37] - solution[38]) + complete_parameters[21] * (((solution[37] - solution[38]) - solution[37]) + solution[38]) + complete_parameters[19] * solution[28] + solution[22]),
        solution[1] - (complete_parameters[22] * solution[1] + complete_parameters[35] * 0),
        solution[2] - (complete_parameters[23] * solution[2] + complete_parameters[36] * 0),
        solution[11] - (complete_parameters[24] * solution[11] + complete_parameters[37] * 0 + complete_parameters[35] * 0 * complete_parameters[9]),
        solution[27] - (complete_parameters[25] * solution[27] + complete_parameters[40] * 0),
        solution[22] - (complete_parameters[26] * solution[22] + complete_parameters[38] * 0),
        solution[33] - ((complete_parameters[27] * solution[33] + solution[9]) - complete_parameters[29] * solution[9]),
        solution[9] - complete_parameters[41] * 0,
        solution[34] - ((complete_parameters[28] * solution[34] + solution[10]) - complete_parameters[30] * solution[10]),
        solution[10] - complete_parameters[39] * 0,
        solution[16] - (solution[16] * (1 - complete_parameters[45]) + solution[12] * complete_parameters[45] + solution[27] * complete_parameters[10] * complete_parameters[44] ^ 2 * complete_parameters[45]),
        solution[8] - ((complete_parameters[34] + solution[37]) - solution[37]),
        solution[5] - ((complete_parameters[34] + solution[3]) - solution[3]),
        solution[6] - ((complete_parameters[34] + solution[12]) - solution[12]),
        solution[24] - (complete_parameters[32] + solution[23]),
        solution[31] - (solution[28] + complete_parameters[47]),
        solution[7] - ((complete_parameters[34] + solution[35]) - solution[35]),
        solution[20] - (solution[18] + complete_parameters[31]),
    ]
end

function residuals_auxiliary(parameters::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(solution) == length(AUXILIARY_SOLUTION_NAMES)
    complete_parameters = complete_parameter_values(parameters)
    return [
        (solution[1] - complete_parameters[6] * solution[30]) - solution[36] * (1 - complete_parameters[6]),
        solution[40] - (solution[30] * (1 - complete_parameters[17])) / complete_parameters[17],
        ((solution[15] - solution[19]) + solution[30]) - solution[36],
        (solution[15] - solution[17]) - solution[40],
        (solution[13] - solution[27]) - (complete_parameters[42] * complete_parameters[44] * solution[13] + solution[13] + solution[26] / (complete_parameters[44] ^ 2 * complete_parameters[10])) / (complete_parameters[42] * complete_parameters[44] + 1),
        (((-(solution[2]) * complete_parameters[7] * (1 + complete_parameters[11] / complete_parameters[44])) / (1 - complete_parameters[11] / complete_parameters[44]) - (complete_parameters[48] * solution[30]) / ((complete_parameters[48] - complete_parameters[1]) + 1)) - (solution[26] * (1 - complete_parameters[1])) / ((complete_parameters[48] - complete_parameters[1]) + 1)) + solution[26] + solution[32],
        (((-(solution[2]) + solution[4]) - solution[4] / (1 + complete_parameters[11] / complete_parameters[44])) - (solution[4] * complete_parameters[11]) / (complete_parameters[44] * (1 + complete_parameters[11] / complete_parameters[44]))) + (solution[32] * (1 - complete_parameters[11] / complete_parameters[44])) / (complete_parameters[7] * (1 + complete_parameters[11] / complete_parameters[44])),
        (((-(complete_parameters[43]) * solution[4] - complete_parameters[46] * solution[13]) - complete_parameters[49] * solution[40]) - solution[11]) + solution[38],
        -(complete_parameters[8]) * (solution[1] + complete_parameters[6] * solution[15] + solution[19] * (1 - complete_parameters[6])) + solution[38],
        ((-(solution[4]) / (1 - complete_parameters[11] / complete_parameters[44]) + (solution[4] * complete_parameters[11]) / (complete_parameters[44] * (1 - complete_parameters[11] / complete_parameters[44]))) - complete_parameters[13] * solution[19]) + solution[36],
        ((-(complete_parameters[44] ^ 2) * complete_parameters[45] * complete_parameters[10] * solution[27] - complete_parameters[45] * solution[13]) - solution[17] * (1 - complete_parameters[45])) + solution[17],
        ((solution[1] - complete_parameters[6] * solution[29]) + solution[21]) - solution[35] * (1 - complete_parameters[6]),
        solution[39] - (solution[29] * (1 - complete_parameters[17])) / complete_parameters[17],
        ((solution[14] - solution[18]) + solution[29]) - solution[35],
        (solution[14] - solution[16]) - solution[39],
        (solution[12] - solution[27]) - (complete_parameters[42] * complete_parameters[44] * solution[12] + solution[12] + solution[25] / (complete_parameters[44] ^ 2 * complete_parameters[10])) / (complete_parameters[42] * complete_parameters[44] + 1),
        ((((-(solution[2]) * complete_parameters[7] * (1 + complete_parameters[11] / complete_parameters[44])) / (1 - complete_parameters[11] / complete_parameters[44]) - (complete_parameters[48] * solution[29]) / ((complete_parameters[48] - complete_parameters[1]) + 1)) - solution[23]) - (solution[25] * (1 - complete_parameters[1])) / ((complete_parameters[48] - complete_parameters[1]) + 1)) + solution[25] + solution[28],
        (((-(solution[2]) + solution[3]) - solution[3] / (1 + complete_parameters[11] / complete_parameters[44])) - (solution[3] * complete_parameters[11]) / (complete_parameters[44] * (1 + complete_parameters[11] / complete_parameters[44]))) + ((1 - complete_parameters[11] / complete_parameters[44]) * (-(solution[23]) + solution[28])) / (complete_parameters[7] * (1 + complete_parameters[11] / complete_parameters[44])),
        (((-(solution[3]) * complete_parameters[43] - complete_parameters[46] * solution[12]) - complete_parameters[49] * solution[39]) - solution[11]) + solution[37],
        -(complete_parameters[8]) * (solution[1] + complete_parameters[6] * solution[14] + solution[18] * (1 - complete_parameters[6])) + solution[37],
        (solution[23] - solution[33]) - (complete_parameters[42] * complete_parameters[44] * solution[23] + complete_parameters[16] * solution[23] + (solution[21] * (1 - complete_parameters[14]) * (-(complete_parameters[42]) * complete_parameters[44] * complete_parameters[14] + 1)) / (complete_parameters[14] * (complete_parameters[4] * (complete_parameters[8] - 1) + 1))) / (complete_parameters[42] * complete_parameters[44] * complete_parameters[16] + 1),
        (((((((-(complete_parameters[42]) * complete_parameters[44] * solution[23]) / (complete_parameters[42] * complete_parameters[44] + 1) - (complete_parameters[42] * complete_parameters[44] * solution[35]) / (complete_parameters[42] * complete_parameters[44] + 1)) - (complete_parameters[15] * solution[23]) / (complete_parameters[42] * complete_parameters[44] + 1)) + (solution[23] * (complete_parameters[42] * complete_parameters[44] * complete_parameters[15] + 1)) / (complete_parameters[42] * complete_parameters[44] + 1)) - solution[34]) + solution[35]) - solution[35] / (complete_parameters[42] * complete_parameters[44] + 1)) - ((1 - complete_parameters[12]) * (-(complete_parameters[42]) * complete_parameters[44] * complete_parameters[12] + 1) * (((solution[3] / (1 - complete_parameters[11] / complete_parameters[44]) - (solution[3] * complete_parameters[11]) / (complete_parameters[44] * (1 - complete_parameters[11] / complete_parameters[44]))) + complete_parameters[13] * solution[18]) - solution[35])) / (complete_parameters[12] * (complete_parameters[42] * complete_parameters[44] + 1) * (complete_parameters[5] * (complete_parameters[2] - 1) + 1)),
        (((-(complete_parameters[18]) * solution[23] * (1 - complete_parameters[19]) - complete_parameters[19] * solution[28]) - complete_parameters[20] * (1 - complete_parameters[19]) * (solution[37] - solution[38])) - solution[22]) + solution[28],
        -(solution[1]) * complete_parameters[22] + solution[1],
        -(solution[2]) * complete_parameters[23] + solution[2],
        -(complete_parameters[24]) * solution[11] + solution[11],
        -(complete_parameters[25]) * solution[27] + solution[27],
        -(complete_parameters[26]) * solution[22] + solution[22],
        ((complete_parameters[29] * solution[9] - complete_parameters[27] * solution[33]) - solution[9]) + solution[33],
        solution[9] - 0,
        ((complete_parameters[30] * solution[10] - complete_parameters[28] * solution[34]) - solution[10]) + solution[34],
        solution[10] - 0,
        ((-(complete_parameters[44] ^ 2) * complete_parameters[45] * complete_parameters[10] * solution[27] - complete_parameters[45] * solution[12]) - solution[16] * (1 - complete_parameters[45])) + solution[16],
        -(complete_parameters[34]) + solution[8],
        -(complete_parameters[34]) + solution[5],
        -(complete_parameters[34]) + solution[6],
        (-(complete_parameters[32]) - solution[23]) + solution[24],
        (-(complete_parameters[47]) - solution[28]) + solution[31],
        -(complete_parameters[34]) + solution[7],
        (-(complete_parameters[31]) - solution[18]) + solution[20],
    ]
end

function residuals_block_1(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(complete_parameters[47]) - previous_solution[1]) + solution[1],
    ]
end

function residuals_block_2(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(complete_parameters[32]) - previous_solution[1]) + solution[1],
    ]
end

function residuals_block_3(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        (-(complete_parameters[31]) - previous_solution[1]) + solution[1],
    ]
end

function residuals_block_4(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[34]) + solution[1],
    ]
end

function residuals_block_5(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[34]) + solution[1],
    ]
end

function residuals_block_6(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[34]) + solution[1],
    ]
end

function residuals_block_7(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[34]) + solution[1],
    ]
end

function residuals_block_8(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 8
    @assert length(external_solution) == 0
    @assert length(solution) == 13
    complete_parameters = complete_parameter_values(parameters)
    return [
        (((-(previous_solution[2]) + solution[1]) - solution[1] / (1 + complete_parameters[11] / complete_parameters[44])) - (solution[1] * complete_parameters[11]) / (complete_parameters[44] * (1 + complete_parameters[11] / complete_parameters[44]))) + ((1 - complete_parameters[11] / complete_parameters[44]) * (-(solution[7]) + solution[9])) / (complete_parameters[7] * (1 + complete_parameters[11] / complete_parameters[44])),
        ((-(complete_parameters[44] ^ 2) * complete_parameters[45] * complete_parameters[10] * previous_solution[5] - complete_parameters[45] * solution[2]) - solution[4] * (1 - complete_parameters[45])) + solution[4],
        -(complete_parameters[8]) * (previous_solution[1] + complete_parameters[6] * solution[3] + solution[5] * (1 - complete_parameters[6])) + solution[12],
        (solution[3] - solution[4]) - solution[13],
        (((((((-(complete_parameters[42]) * complete_parameters[44] * solution[7]) / (complete_parameters[42] * complete_parameters[44] + 1) - (complete_parameters[42] * complete_parameters[44] * solution[11]) / (complete_parameters[42] * complete_parameters[44] + 1)) - (complete_parameters[15] * solution[7]) / (complete_parameters[42] * complete_parameters[44] + 1)) + (solution[7] * (complete_parameters[42] * complete_parameters[44] * complete_parameters[15] + 1)) / (complete_parameters[42] * complete_parameters[44] + 1)) - previous_solution[7]) + solution[11]) - solution[11] / (complete_parameters[42] * complete_parameters[44] + 1)) - ((1 - complete_parameters[12]) * (-(complete_parameters[42]) * complete_parameters[44] * complete_parameters[12] + 1) * (((solution[1] / (1 - complete_parameters[11] / complete_parameters[44]) - (solution[1] * complete_parameters[11]) / (complete_parameters[44] * (1 - complete_parameters[11] / complete_parameters[44]))) + complete_parameters[13] * solution[5]) - solution[11])) / (complete_parameters[12] * (complete_parameters[42] * complete_parameters[44] + 1) * (complete_parameters[5] * (complete_parameters[2] - 1) + 1)),
        (solution[7] - previous_solution[6]) - (complete_parameters[42] * complete_parameters[44] * solution[7] + complete_parameters[16] * solution[7] + (solution[6] * (1 - complete_parameters[14]) * (-(complete_parameters[42]) * complete_parameters[44] * complete_parameters[14] + 1)) / (complete_parameters[14] * (complete_parameters[4] * (complete_parameters[8] - 1) + 1))) / (complete_parameters[42] * complete_parameters[44] * complete_parameters[16] + 1),
        ((((-(previous_solution[2]) * complete_parameters[7] * (1 + complete_parameters[11] / complete_parameters[44])) / (1 - complete_parameters[11] / complete_parameters[44]) - (complete_parameters[48] * solution[10]) / ((complete_parameters[48] - complete_parameters[1]) + 1)) - solution[7]) - (solution[8] * (1 - complete_parameters[1])) / ((complete_parameters[48] - complete_parameters[1]) + 1)) + solution[8] + solution[9],
        (solution[2] - previous_solution[5]) - (complete_parameters[42] * complete_parameters[44] * solution[2] + solution[2] + solution[8] / (complete_parameters[44] ^ 2 * complete_parameters[10])) / (complete_parameters[42] * complete_parameters[44] + 1),
        (((-(complete_parameters[18]) * solution[7] * (1 - complete_parameters[19]) - complete_parameters[19] * solution[9]) - complete_parameters[20] * (1 - complete_parameters[19]) * (solution[12] - previous_solution[8])) - previous_solution[4]) + solution[9],
        ((solution[3] - solution[5]) + solution[10]) - solution[11],
        ((previous_solution[1] - complete_parameters[6] * solution[10]) + solution[6]) - solution[11] * (1 - complete_parameters[6]),
        (((-(solution[1]) * complete_parameters[43] - complete_parameters[46] * solution[2]) - complete_parameters[49] * solution[13]) - previous_solution[3]) + solution[12],
        solution[13] - (solution[10] * (1 - complete_parameters[17])) / complete_parameters[17],
    ]
end

function residuals_block_9(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 4
    @assert length(external_solution) == 0
    @assert length(solution) == 11
    complete_parameters = complete_parameter_values(parameters)
    return [
        (((-(previous_solution[2]) + solution[1]) - solution[1] / (1 + complete_parameters[11] / complete_parameters[44])) - (solution[1] * complete_parameters[11]) / (complete_parameters[44] * (1 + complete_parameters[11] / complete_parameters[44]))) + (solution[8] * (1 - complete_parameters[11] / complete_parameters[44])) / (complete_parameters[7] * (1 + complete_parameters[11] / complete_parameters[44])),
        ((-(complete_parameters[44] ^ 2) * complete_parameters[45] * complete_parameters[10] * previous_solution[4] - complete_parameters[45] * solution[2]) - solution[4] * (1 - complete_parameters[45])) + solution[4],
        ((solution[3] - solution[5]) + solution[7]) - solution[9],
        (solution[3] - solution[4]) - solution[11],
        -(complete_parameters[8]) * (previous_solution[1] + complete_parameters[6] * solution[3] + solution[5] * (1 - complete_parameters[6])) + solution[10],
        (solution[2] - previous_solution[4]) - (complete_parameters[42] * complete_parameters[44] * solution[2] + solution[2] + solution[6] / (complete_parameters[44] ^ 2 * complete_parameters[10])) / (complete_parameters[42] * complete_parameters[44] + 1),
        (previous_solution[1] - complete_parameters[6] * solution[7]) - solution[9] * (1 - complete_parameters[6]),
        (((-(previous_solution[2]) * complete_parameters[7] * (1 + complete_parameters[11] / complete_parameters[44])) / (1 - complete_parameters[11] / complete_parameters[44]) - (complete_parameters[48] * solution[7]) / ((complete_parameters[48] - complete_parameters[1]) + 1)) - (solution[6] * (1 - complete_parameters[1])) / ((complete_parameters[48] - complete_parameters[1]) + 1)) + solution[6] + solution[8],
        ((-(solution[1]) / (1 - complete_parameters[11] / complete_parameters[44]) + (solution[1] * complete_parameters[11]) / (complete_parameters[44] * (1 - complete_parameters[11] / complete_parameters[44]))) - complete_parameters[13] * solution[5]) + solution[9],
        (((-(complete_parameters[43]) * solution[1] - complete_parameters[46] * solution[2]) - complete_parameters[49] * solution[11]) - previous_solution[3]) + solution[10],
        solution[11] - (solution[7] * (1 - complete_parameters[17])) / complete_parameters[17],
    ]
end

function residuals_block_10(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[26]) * solution[1] + solution[1],
    ]
end

function residuals_block_11(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[25]) * solution[1] + solution[1],
    ]
end

function residuals_block_12(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(complete_parameters[24]) * solution[1] + solution[1],
    ]
end

function residuals_block_13(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((complete_parameters[29] * previous_solution[1] - complete_parameters[27] * solution[1]) - previous_solution[1]) + solution[1],
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
    @assert length(previous_solution) == 1
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        ((complete_parameters[30] * previous_solution[1] - complete_parameters[28] * solution[1]) - previous_solution[1]) + solution[1],
    ]
end

function residuals_block_16(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        solution[1] - 0,
    ]
end

function residuals_block_17(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) * complete_parameters[23] + solution[1],
    ]
end

function residuals_block_18(parameters::AbstractVector, previous_solution::AbstractVector, external_solution::AbstractVector, solution::AbstractVector)
    @assert length(parameters) == length(PARAMETER_NAMES)
    @assert length(previous_solution) == 0
    @assert length(external_solution) == 0
    @assert length(solution) == 1
    complete_parameters = complete_parameter_values(parameters)
    return [
        -(solution[1]) * complete_parameters[22] + solution[1],
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
export residuals_block_1, residuals_block_2, residuals_block_3, residuals_block_4, residuals_block_5, residuals_block_6, residuals_block_7, residuals_block_8, residuals_block_9, residuals_block_10, residuals_block_11, residuals_block_12, residuals_block_13, residuals_block_14, residuals_block_15, residuals_block_16, residuals_block_17, residuals_block_18
end
