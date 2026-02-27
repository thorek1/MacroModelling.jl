using MacroModelling
using LinearAlgebra
using ChainRulesCore
using SparseArrays
using Random
import MacroModelling: get_NSSS_and_parameters

Random.seed!(1)

include(joinpath(@__DIR__, "..", "test", "models", "Caldara_et_al_2012_estim.jl"))
m = Caldara_et_al_2012_estim

SS_and_pars, _ = get_NSSS_and_parameters(m, m.parameter_values)
∇₁ = calculate_jacobian(m.parameter_values, SS_and_pars, m.caches, m.functions.jacobian)
∇₂ = calculate_hessian(m.parameter_values, SS_and_pars, m.caches, m.functions.hessian)
∇₃ = calculate_third_order_derivatives(m.parameter_values, SS_and_pars, m.caches, m.functions.third_order_derivatives)

S1, _, solved1 = calculate_first_order_solution(∇₁, m.constants, m.workspaces, m.caches)
S2, solved2 = calculate_second_order_solution(∇₁, ∇₂, S1, m.constants, m.workspaces, m.caches)
@assert solved1 && solved2

nPast = m.constants.post_model_macro.nPast_not_future_and_mixed
S1raw = [S1[:,1:nPast] S1[:,nPast+2:end]]
S2raw = sparse(S2 * m.constants.second_order.𝐔₂)

(y, pb) = rrule(calculate_third_order_solution, ∇₁, ∇₂, ∇₃, S1raw, S2raw, m.constants, m.workspaces, m.caches; initial_guess = m.caches.third_order_solution)
S3 = y[1]
@assert y[2]

f0 = norm(S3)
ΔS3 = Matrix(S3) / max(norm(S3), eps())
tangs = pb((ΔS3, NoTangent()))

g1 = tangs[2]
g2 = tangs[3]
g3 = tangs[4]
gS1 = tangs[5]
gS2 = tangs[6]

d1 = randn(size(∇₁)...)
d2 = sprand(Float64, size(∇₂,1), size(∇₂,2), 0.02)
d3 = sprand(Float64, size(∇₃,1), size(∇₃,2), 0.02)
dS1 = randn(size(S1raw)...)
dS2 = sprand(Float64, size(S2raw,1), size(S2raw,2), 0.02)

pred = sum(g1 .* d1) + sum(g2 .* d2) + sum(g3 .* d3) + sum(gS1 .* dS1) + sum(gS2 .* dS2)

h = 1e-6
S3p = calculate_third_order_solution(∇₁ .+ h*d1, ∇₂ .+ h*d2, ∇₃ .+ h*d3, S1raw .+ h*dS1, S2raw .+ h*dS2, m.constants, m.workspaces, m.caches; initial_guess = m.caches.third_order_solution)[1]
S3m = calculate_third_order_solution(∇₁ .- h*d1, ∇₂ .- h*d2, ∇₃ .- h*d3, S1raw .- h*dS1, S2raw .- h*dS2, m.constants, m.workspaces, m.caches; initial_guess = m.caches.third_order_solution)[1]
fd = (norm(S3p) - norm(S3m)) / (2h)

println("f0=", f0)
println("pred=", pred)
println("fd=", fd)
println("abs_diff=", abs(pred-fd))
println("rel_diff=", abs(pred-fd)/max(abs(fd),1e-12))
