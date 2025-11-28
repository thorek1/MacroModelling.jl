# using Pkg; Pkg.offline(true)
using Revise
using MacroModelling

@model RBC_incomplete begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC_incomplete begin
    std_z = 0.01
    ρ = 0.2
    α = 0.3
    β = 0.99
    δ = 0.025
    # Note: α, β, δ are not defined
end

RBC_incomplete.parameters
get_missing_parameters(RBC_incomplete)

get_irf(RBC_incomplete, parameters = [:α => 0.3, :β => 0.99, :δ => 0.025])

RBC_incomplete.SS_solve_func