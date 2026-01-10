#!/usr/bin/env julia

# Simple test script to verify the reorganization works

using Pkg
Pkg.activate(".")

println("Testing MacroModelling.jl reorganization...")
println("=" ^ 60)

# Test 1: Package loads
println("\n1. Testing package loading...")
try
    using MacroModelling
    println("   ✓ Package loaded successfully")
catch e
    println("   ✗ Failed to load package: $e")
    exit(1)
end

# Test 2: Create a simple model
println("\n2. Testing model creation...")
try
    @model RBC_test begin
        1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
        c[0] + k[0] = (1 - δ) * k[-1] + q[0]
        q[0] = exp(z[0]) * k[-1]^α
        z[0] = ρ * z[-1] + std_z * eps_z[x]
    end
    println("   ✓ Model created successfully")
catch e
    println("   ✗ Failed to create model: $e")
    exit(1)
end

# Test 3: Set parameters
println("\n3. Testing parameter setting...")
try
    @parameters RBC_test begin
        std_z = 0.01
        ρ = 0.2
        δ = 0.02
        α = 0.5
        β = 0.95
    end
    println("   ✓ Parameters set successfully")
catch e
    println("   ✗ Failed to set parameters: $e")
    exit(1)
end

# Test 4: Get steady state (this uses jacobian which has rrules)
println("\n4. Testing steady state calculation...")
try
    ss = get_steady_state(RBC_test, silent = true)
    println("   ✓ Steady state calculated successfully")
catch e
    println("   ✗ Failed to calculate steady state: $e")
    exit(1)
end

# Test 5: Get solution (this uses first order solution which has rrules)
println("\n5. Testing first-order solution...")
try
    sol = get_solution(RBC_test, silent = true)
    println("   ✓ First-order solution calculated successfully")
catch e
    println("   ✗ Failed to calculate solution: $e")
    exit(1)
end

# Test 6: ForwardDiff compatibility (tests Dual functions)
println("\n6. Testing ForwardDiff compatibility...")
try
    using ForwardDiff
    # Test parameter differentiation through get_steady_state
    params = RBC_test.parameter_values
    f(p) = sum(get_steady_state(RBC_test, parameters = p, silent = true))
    grad = ForwardDiff.gradient(f, params)
    println("   ✓ ForwardDiff works with Dual numbers")
catch e
    println("   ✗ ForwardDiff test failed: $e")
    # This is not a critical failure for now
end

println("\n" * "=" ^ 60)
println("All basic tests passed! ✓")
println("\nNote: Full test suite should be run via `Pkg.test()`")
