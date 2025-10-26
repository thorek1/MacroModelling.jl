# Example: Tracking Calibration Equation Changes
#
# This example demonstrates how to use the calibration tracking functionality
# to document and track changes to model calibration over time.

using MacroModelling

# Define a simple RBC model
@model RBC_example begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

# Initial calibration
println("=" ^ 60)
println("Initial Model Calibration")
println("=" ^ 60)

@parameters RBC_example begin
    std_z = 0.01
    ρ = 0.2
    k[ss] / q[ss] = 2.5 | δ
    α = 0.5
    β = 0.95
end

println("\nInitial calibration equations:")
for (param, eq) in zip(get_calibrated_parameters(RBC_example), get_calibration_equations(RBC_example))
    println("  $param: $eq")
end

# Scenario 1: Adjust calibration based on new data
println("\n" * "=" ^ 60)
println("Scenario 1: Adjusting for higher capital intensity")
println("=" ^ 60)

modify_calibration_equations!(RBC_example, 
    [:δ => :(k[ss] / q[ss] - 3.0)],
    "Literature suggests capital-to-output ratio closer to 3.0 for developed economies",
    verbose = true)

# Scenario 2: Further refinement
println("\n" * "=" ^ 60)
println("Scenario 2: Alternative calibration")
println("=" ^ 60)

modify_calibration_equations!(RBC_example, 
    [:δ => :(k[ss] / q[ss] - 2.8)],
    "Compromise value between initial calibration and literature benchmark",
    verbose = false)

# Display complete revision history
println("\n" * "=" ^ 60)
println("Complete Revision History")
println("=" ^ 60)

print_calibration_revision_history(RBC_example)

# Programmatic access to revision history
println("\n" * "=" ^ 60)
println("Programmatic Access to History")
println("=" ^ 60)

history = get_calibration_revision_history(RBC_example)
println("\nTotal number of documented revisions: ", length(history))

println("\nDetailed revision information:")
for (i, (note, equations, parameters)) in enumerate(history)
    println("\nRevision $i:")
    println("  Timestamp and note: $note")
    println("  Parameters modified: ", join(parameters, ", "))
    println("  New equations:")
    for (param, eq) in zip(parameters, equations)
        println("    $param => $eq")
    end
end

println("\n" * "=" ^ 60)
println("Note: To apply any of these changes, re-run @parameters")
println("with the desired calibration equation.")
println("=" ^ 60)
