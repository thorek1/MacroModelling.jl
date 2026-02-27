using MacroModelling
import MacroModelling: get_NSSS_and_parameters
include(joinpath(@__DIR__, "..", "models", "Caldara_et_al_2012.jl"))
m = Caldara_et_al_2012
SS_and_pars, _ = get_NSSS_and_parameters(m, m.parameter_values)
∇₁ = calculate_jacobian(m.parameter_values, SS_and_pars, m.caches, m.functions.jacobian)
∇₂ = calculate_hessian(m.parameter_values, SS_and_pars, m.caches, m.functions.hessian)
∇₃ = calculate_third_order_derivatives(m.parameter_values, SS_and_pars, m.caches, m.functions.third_order_derivatives)
println(size(∇₁)," ",size(∇₂)," ",size(∇₃))
