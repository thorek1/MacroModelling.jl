# Test for modify_calibration_equations! functionality
println("Testing modify_calibration_equations! functionality...")

# Test 1: Create a simple model with calibration equations
@model RBC_calib_test begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC_calib_test begin
    std_z = 0.01
    ρ = 0.2
    k[ss] / q[ss] = 2.5 | δ
    α = 0.5
    β = 0.95
end

# Test 2: Check initial state
println("\n=== Test 1: Initial calibration equations ===")
initial_calib_eqs = get_calibration_equations(RBC_calib_test)
initial_calib_params = get_calibrated_parameters(RBC_calib_test)
println("Initial calibration equations: ", initial_calib_eqs)
println("Initial calibration parameters: ", initial_calib_params)

@test length(initial_calib_eqs) == 1
@test length(initial_calib_params) == 1
@test initial_calib_params[1] == "δ"

# Test 3: Check revision history is initially empty
println("\n=== Test 2: Initial revision history ===")
initial_history = get_calibration_revision_history(RBC_calib_test)
println("Initial revision history length: ", length(initial_history))

@test length(initial_history) == 0

# Test 4: Document a revision to calibration equation
println("\n=== Test 3: Document calibration equation revision ===")
try
    modify_calibration_equations!(RBC_calib_test, 
        [:δ => :(k[ss] / q[ss] - 3.0)],
        "Updated capital to output ratio target",
        verbose = true)
    println("✓ Successfully documented calibration equation revision")
    
    # Check that revision history now has one entry
    history_after_mod = get_calibration_revision_history(RBC_calib_test)
    println("Revision history length after documentation: ", length(history_after_mod))
    
    @test length(history_after_mod) == 1
    @test occursin("Updated capital to output ratio", history_after_mod[1][1])
    @test length(history_after_mod[1][2]) == 1  # One equation documented
    @test length(history_after_mod[1][3]) == 1  # One parameter documented
    
    println("✓ Revision history updated correctly")
catch e
    println("✗ Error documenting calibration equation revision: ", e)
    rethrow(e)
end

# Test 5: Print revision history
println("\n=== Test 4: Print revision history ===")
try
    print_calibration_revision_history(RBC_calib_test)
    println("✓ Successfully printed revision history")
catch e
    println("✗ Error printing revision history: ", e)
    rethrow(e)
end

# Test 6: Document multiple revisions
println("\n=== Test 5: Multiple revisions ===")
try
    modify_calibration_equations!(RBC_calib_test, 
        [:δ => :(k[ss] / q[ss] - 3.5)],
        "Second update to capital ratio target")
    
    history_after_second_mod = get_calibration_revision_history(RBC_calib_test)
    println("Revision history length after second documentation: ", length(history_after_second_mod))
    
    @test length(history_after_second_mod) == 2
    @test occursin("Second update", history_after_second_mod[2][1])
    
    println("✓ Multiple revisions tracked correctly")
catch e
    println("✗ Error with multiple revisions: ", e)
    rethrow(e)
end

# Test 7: Error handling - invalid parameter
println("\n=== Test 6: Error handling for invalid parameter ===")
try
    modify_calibration_equations!(RBC_calib_test, 
        [:invalid_param => :(k[ss] - 1.0)],
        "This should fail")
    println("✗ Should have raised an error for invalid parameter")
    @test false
catch e
    if occursin("not a calibration parameter", string(e))
        println("✓ Correctly caught invalid parameter error")
    else
        println("✗ Unexpected error: ", e)
        rethrow(e)
    end
end

println("\n=== All tests passed! ===")
