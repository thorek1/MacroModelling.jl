using ComponentArrays, MacroModelling
include("../test/models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags.jl")
get_solution(m)
𝓂 = m
parameters = 𝓂.parameter_values
SS_and_pars, _ = 𝓂.SS_solve_func(parameters, 𝓂, false, false)

var_past = setdiff(𝓂.var_past,𝓂.nonnegativity_auxilliary_vars)
var_present = setdiff(𝓂.var_present,𝓂.nonnegativity_auxilliary_vars)
var_future = setdiff(𝓂.var_future,𝓂.nonnegativity_auxilliary_vars)

SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]
par = ComponentVector(vcat(parameters,calibrated_parameters),Axis(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)))

past_idx = [indexin(sort([var_past; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_past,𝓂.exo_past))]), sort(union(𝓂.var,𝓂.exo_present)))...]
SS_past =       length(past_idx) > 0 ? SS[past_idx] : zeros(0) #; zeros(length(𝓂.exo_past))...]

present_idx = [indexin(sort([var_present; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_present,𝓂.exo_present))]), sort(union(𝓂.var,𝓂.exo_present)))...]
SS_present =    length(present_idx) > 0 ? SS[present_idx] : zeros(0)#; zeros(length(𝓂.exo_present))...]

future_idx = [indexin(sort([var_future; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_future,𝓂.exo_future))]), sort(union(𝓂.var,𝓂.exo_present)))...]
SS_future =     length(future_idx) > 0 ? SS[future_idx] : zeros(0)#; zeros(length(𝓂.exo_future))...]

shocks_ss = zeros(length(𝓂.exo))

jac = collect(𝓂.model_jacobian([SS_future; SS_present; SS_past; shocks_ss], par, SS))



symbolics = create_symbols_eqs!(𝓂);

dyn_future_list = collect(reduce(union, symbolics.dyn_future_list))
dyn_present_list = collect(reduce(union, symbolics.dyn_present_list))
dyn_past_list = collect(reduce(union, symbolics.dyn_past_list))
dyn_exo_list = collect(reduce(union,symbolics.dyn_exo_list))

future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_future_list))
present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_present_list))
past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_past_list))
exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))

vars = [dyn_future_list[indexin(sort(future),future)]...,
        dyn_present_list[indexin(sort(present),present)]...,
        dyn_past_list[indexin(sort(past),past)]...,
        dyn_exo_list[indexin(sort(exo),exo)]...]



using SymPy, StatsFuns
using MacroTools: postwalk
@vars omegabar sigma nonnegative = true real = true

x = log(omegabar)
x = normpdf((log(omegabar)))
xdiff = diff(x,omegabar) |> string |>Meta.parse

dump(xdiff)
diff_manual = :((-((-1 + (Rk * (G * (1 - mu) + omegabar * (1 - F))) / R)) * (1 - F)) / ((-F - (0.398942280401433 * mu * exp((-((sigma ^ 2 / 2 + log(omegabar))) * (sigma ^ 2 / 2 + conjugate(log(omegabar)))) / (2 * sigma ^ 2))) / sigma) + 1) - (Rk * ((-G - omegabar * (1 - F)) + 1)) / R)
postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, xdiff)
postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, diff_manual)


xdiff.as_real_imag()[1] |> string |>Meta.parse

SymPy.as_real_imag(diff(x,omegabar))