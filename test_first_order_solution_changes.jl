using Test
using MacroModelling

println("Testing calculate_first_order_solution changes...")

# Include a simple model
include("models/RBC_baseline.jl")

@testset "First order solution with cache parameter" begin
    println("Test 1: Basic first order solution (via get_irf)")
    irf = get_irf(RBC_baseline)
    @test !isempty(irf)
    @test all(isfinite, vec(irf))
    println("✓ Test 1 passed")
    
    println("Test 2: First order solution (via get_solution)")
    sol = get_solution(RBC_baseline)
    @test !isempty(sol)
    @test all(isfinite, vec(Array(sol)))
    println("✓ Test 2 passed")
    
    println("Test 3: Variance decomposition (uses first order solution)")
    vd = get_variance_decomposition(RBC_baseline)
    @test !isempty(vd)
    @test all(>=(0), vec(vd))  # Variance decompositions should be non-negative
    @test all(<=(1), vec(vd))  # And sum to 1 (or less)
    println("✓ Test 3 passed")
    
    println("Test 4: Conditional variance decomposition")
    cvd = get_conditional_variance_decomposition(RBC_baseline)
    @test !isempty(cvd)
    @test all(>=(0), vec(cvd))
    println("✓ Test 4 passed")
    
    println("Test 5: First order mean")
    mean_first = get_mean(RBC_baseline; algorithm = :first_order)
    @test !isempty(mean_first)
    @test all(isfinite, vec(mean_first))
    println("✓ Test 5 passed")
    
    println("Test 6: Second order mean (uses first order solution)")
    mean_second = get_mean(RBC_baseline; algorithm = :pruned_second_order)
    @test !isempty(mean_second)
    @test all(isfinite, vec(mean_second))
    println("✓ Test 6 passed")
    
    println("Test 7: Covariance calculation (uses first order solution)")
    cov = get_covariance(RBC_baseline)
    @test !isempty(cov)
    @test all(isfinite, vec(cov))
    println("✓ Test 7 passed")
end

println("\nAll tests passed successfully!")
