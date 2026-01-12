using MacroModelling

include("test/models/RBC_CME_calibration_equations.jl")

solve!(m)

calibration_equations = get_calibration_equations(m)
calibrated_parameters = get_calibrated_parameters(m; values = true)
steady_state = get_steady_state(m; derivatives = false)

println(calibration_equations)
println(calibrated_parameters)
println(steady_state)
