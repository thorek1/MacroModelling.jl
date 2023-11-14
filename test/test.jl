
using MacroModelling

@model testmax begin
    1  /  c[0] = (β  /  c[1]) * (r[1] + (1 - δ))

    r̂[0] = α * exp(z[0]) * k[-1]^(α - 1)

    r̂[0] = max(r̄, r[0])

    # r̂[0] = r[0] + ϵll[x-3]

    c[0] + k[0] = (1 - δ) * k[-1] + q[0]

    q[0] = exp(z[0]) * k[-1]^α

    z[0] = ρᶻ * z[-1] + σᶻ * ϵᶻ[x]

    # ϵll⁻¹[0] = ϵll⁻²[-1] + ϵll⁻¹[x]

    # ϵll⁻²[0] = ϵll⁻³[-1] + ϵll⁻²[x]

    # ϵll⁻³[0] = ϵll⁻³[x]

end

@parameters testmax begin
    r̄ = 0
    σᶻ= 1#0.01
    ρᶻ= 0.8#2
    δ = 0.02
    α = 0.5
    β = 0.95
end

SS(testmax)


import MacroModelling: match_pattern
match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₁₎")


filter(r -> match(r"₍₁₎", string(r)) !== nothing, string(:χᵒᵇᶜ⁺ꜝ¹ꜝʳ₍₀₎))


occursin(r"₍₁₎|₍₀₎|₍₋₁₎",string(:χᵒᵇᶜ⁺ꜝ¹ꜝʳ₍₀₎))



import MacroModelling: convert_to_ss_equation

convert_to_ss_equation(:(0 = Χᵒᵇᶜ⁺ꜝ¹ꜝ[0] - ϵᵒᵇᶜ⁺ꜝ¹ꜝ[0]))

import MacroTools: postwalk
postwalk(x -> 
x isa Expr ? 
    x.head == :(=) ? 
        Expr(:call,:(-),x.args[1],x.args[2]) :
    x :
x, :(0 = Χᵒᵇᶜ⁺ꜝ¹ꜝ[0] - ϵᵒᵇᶜ⁺ꜝ¹ꜝ[0]))