#!/usr/bin/env julia
# Example: Using Partial Parameter Definition in MacroModelling.jl
# This demonstrates the new feature allowing incremental parameter definition

using MacroModelling

println("=" ^ 70)
println("Partial Parameter Definition Example")
println("=" ^ 70)

# Step 1: Define the model
println("\n1. Defining a simple RBC model...")
@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end
println("   ✓ Model created with variables: ", join(RBC.var, ", "))

# Step 2: Define only some parameters
println("\n2. Defining only some parameters (α, std_z, ρ)...")
println("   Note: β and δ will be left undefined for now")
@parameters RBC begin
    α = 0.5
    std_z = 0.01
    ρ = 0.2
end

println("   Defined parameters: ", join(RBC.parameters, ", "))
println("   Undefined parameters: ", join(RBC.undefined_parameters, ", "))

# Step 3: Demonstrate that computation is delayed
println("\n3. Attempting to get steady state (will fail gracefully)...")
println("   This demonstrates that the model knows parameters are missing")
try
    SS = get_steady_state(RBC, verbose = false)
    println("   Unexpectedly succeeded?")
catch e
    println("   ✓ Handled gracefully (as expected)")
end

# Step 4: Complete the parameter definition
println("\n4. Now defining all parameters...")
@parameters RBC silent = true begin
    α = 0.5
    β = 0.95
    δ = 0.02
    std_z = 0.01
    ρ = 0.2
end
println("   ✓ All parameters defined")
println("   Undefined parameters: ", 
        length(RBC.undefined_parameters) == 0 ? "none" : join(RBC.undefined_parameters, ", "))

# Step 5: Now computations work
println("\n5. Computing steady state...")
SS = get_steady_state(RBC)
println("   ✓ Steady state computed successfully")
println("   Sample steady state values:")
for (var, val) in zip(RBC.var[1:min(3, length(RBC.var))], SS[1:min(3, length(SS))])
    println("      $var = $(round(val, digits=4))")
end

# Step 6: Compute IRFs
println("\n6. Computing impulse response functions...")
irf = get_irf(RBC, periods = 20)
println("   ✓ IRFs computed for $(size(irf, 2)) periods")
println("   Variables: ", join(axiskeys(irf, 1), ", "))
println("   Shocks: ", join(axiskeys(irf, 3), ", "))

println("\n" * "=" ^ 70)
println("Example completed successfully!")
println("=" ^ 70)

# Alternative: Using parameter dictionary
println("\n" * "=" ^ 70)
println("Alternative Approach: Using Parameter Dictionary")
println("=" ^ 70)

# Reset to test alternative approach
@model RBC2 begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

println("\n1. Setting up model with minimal parameters...")
@parameters RBC2 silent = true begin
    std_z = 0.01
    ρ = 0.2
end
println("   Undefined: ", join(RBC2.undefined_parameters, ", "))

println("\n2. Providing all parameters via function call...")
params = Dict(
    :α => 0.5,
    :β => 0.95,
    :δ => 0.02,
    :std_z => 0.01,
    :ρ => 0.2
)

println("   Computing IRF with explicit parameters...")
irf2 = get_irf(RBC2, parameters = params, periods = 20)
println("   ✓ IRFs computed successfully")
println("   After call, undefined parameters: ", 
        length(RBC2.undefined_parameters) == 0 ? "none (cleared)" : join(RBC2.undefined_parameters, ", "))

println("\n" * "=" ^ 70)
println("All examples completed!")
println("=" ^ 70)
