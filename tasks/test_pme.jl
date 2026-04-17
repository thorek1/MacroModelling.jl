using MacroModelling
block = :(begin
    1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end)
T, eqs, ℂ, 𝓦 = MacroModelling.process_model_equations(block, 40, false)
println("T.nVars = ", T.nVars)
println("dynamic equations: ", length(eqs.dynamic))
println("steady_state eqs: ", length(eqs.steady_state))
println("ss_aux eqs: ", length(eqs.steady_state_aux))
println("original eqs: ", length(eqs.original))
