using MacroModelling
block = :(begin
    1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end)
T, eqs, ℂ, 𝓦 = MacroModelling.process_model_equations(block, 40, false)

param_block = :(begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end)
parsed = MacroModelling.process_parameter_definitions(param_block, T)
println("parameters: ", parsed.parameters)
println("parameter_values: ", parsed.parameter_values)
println("missing_parameters: ", parsed.missing_parameters)
println("calibration: ", parsed.equations.calibration)
println("calibration_original: ", parsed.equations.calibration_original)
println("bounds: ", parsed.bounds)

# try with calibration equation
param_block2 = :(begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    k[ss] / q[ss] = 2.5 | α
    β = 0.95
end)
parsed2 = MacroModelling.process_parameter_definitions(param_block2, T)
println("-- with calib --")
println("parameters: ", parsed2.parameters)
println("parameter_values: ", parsed2.parameter_values)
println("calib_parameters_no_var (should have α? no, α is calib_eq): ", parsed2.calib_parameters_no_var)
println("calibration: ", parsed2.equations.calibration)
println("calibration_parameters: ", parsed2.equations.calibration_parameters)
println("calibration_original: ", parsed2.equations.calibration_original)
