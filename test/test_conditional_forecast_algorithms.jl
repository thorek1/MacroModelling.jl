"""
Test conditional forecasts for all perturbation algorithms
Tests that the Lagrange-Newton algorithm works correctly for:
- first_order (fully supported)
- second_order (with known limitations for extended periods)
- third_order (with known limitations for extended periods)
- pruned_second_order (with known limitations for extended periods)
- pruned_third_order (with known limitations for extended periods)
"""

using Test
using MacroModelling
using AxisKeys

@testset "Conditional forecasts - all algorithms" begin
    # Define a simple RBC model for testing
    @model RBC_test begin
        1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
        c[0] + k[0] = (1 - δ) * k[-1] + q[0]
        q[0] = exp(z[0]) * k[-1]^α
        z[0] = ρ * z[-1] + std_z * eps_z[x]
    end

    @parameters RBC_test begin
        std_z = 0.01
        ρ = 0.2
        δ = 0.02
        α = 0.5
        β = 0.95
    end

    # Test first-order with multiple periods (extended to more periods)
    @testset "First-order - multiple periods" begin
        periods = 8  # Extended from 3 to 8 periods
        
        # Define conditions: c is conditioned in periods 1-8
        conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef, 1, periods), 
                               Variables = [:c], 
                               Periods = 1:periods)
        conditions[1,1] = 0.01
        conditions[1,2] = 0.005
        conditions[1,3] = 0.002
        conditions[1,4] = 0.001
        conditions[1,5] = 0.0005
        conditions[1,6] = 0.0002
        conditions[1,7] = 0.0001
        conditions[1,8] = 0.00005

        # Define shocks: all free
        shocks = Matrix{Union{Nothing,Float64}}(undef, 1, periods)
        for p in 1:periods
            shocks[1,p] = nothing
        end

        # Get conditional forecast
        result = get_conditional_forecast(RBC_test, 
                                         conditions, 
                                         shocks = shocks,
                                         conditions_in_levels = false, 
                                         algorithm = :first_order,
                                         periods = periods)
        
        # Find the index of variable c
        c_idx = findfirst(x -> x == :c, axiskeys(result, 1))
        @test !isnothing(c_idx)
        
        # Check that conditioned variable matches target in all periods
        @testset "Period $p" for p in 1:periods
            target = conditions[1, p]
            actual = result[c_idx, p]
            deviation = abs(actual - target)
            
            # Test that deviation is very small (< 1e-10)
            @test deviation < 1e-10
            @test isapprox(actual, target, atol=1e-10)
        end
    end

    # Test higher-order algorithms with same multi-period conditions
    # Now that issues are fixed, these should work with multiple periods
    @testset "Higher-order - multi-period: $algorithm" for algorithm in [:second_order, :third_order, :pruned_second_order, :pruned_third_order]
        periods = 3  # Use same multi-period setup as first-order (but fewer periods)
        
        # Define conditions: c is conditioned in periods 1-3
        conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef, 1, periods), 
                               Variables = [:c], 
                               Periods = 1:periods)
        conditions[1,1] = 0.01
        conditions[1,2] = 0.005
        conditions[1,3] = 0.002

        # Define shocks: all free
        shocks = Matrix{Union{Nothing,Float64}}(undef, 1, periods)
        shocks[1,1] = nothing
        shocks[1,2] = nothing
        shocks[1,3] = nothing

        # Get conditional forecast
        result = get_conditional_forecast(RBC_test, 
                                         conditions, 
                                         shocks = shocks,
                                         conditions_in_levels = false, 
                                         algorithm = algorithm,
                                         periods = periods)
        
        # Find the index of variable c
        c_idx = findfirst(x -> x == :c, axiskeys(result, 1))
        @test !isnothing(c_idx)
        
        # Check that conditioned variable matches target in all periods
        @testset "Period $p" for p in 1:periods
            target = conditions[1, p]
            actual = result[c_idx, p]
            deviation = abs(actual - target)
            
            # Slightly relaxed tolerance for higher-order algorithms
            @test deviation < 1e-8
            @test isapprox(actual, target, atol=1e-8)
        end
    end

    # Test that results have expected structure
    @testset "Result structure" begin
        conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef, 1, 1), 
                               Variables = [:c], 
                               Periods = 1:1)
        conditions[1,1] = 0.01

        shocks = Matrix{Union{Nothing,Float64}}(undef, 1, 1)
        shocks[1,1] = nothing

        result = get_conditional_forecast(RBC_test, 
                                         conditions, 
                                         shocks = shocks,
                                         conditions_in_levels = false, 
                                         algorithm = :first_order,
                                         periods = 1)
        
        # Verify result is a KeyedArray with correct axes
        @test result isa KeyedArray
        @test length(axiskeys(result, 1)) > 0  # Has variables
        @test length(axiskeys(result, 2)) >= 1  # Has at least requested periods
    end
    
    # Test model with multiple shocks (more shocks than conditions)
    # This tests the case where the system is underdetermined
    @testset "Multiple shocks - more shocks than conditions" begin
        # Use FS2000 model which has 2 shocks (e_a and e_m)
        include("../models/FS2000.jl")
        
        # Test all algorithms with multiple shocks
        @testset "Algorithm: $algorithm" for algorithm in [:first_order, :second_order, :third_order, :pruned_second_order, :pruned_third_order]
            periods = 3
            
            # Condition only on y (output) - 1 condition, but have 2 shocks free
            conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef, 1, periods), 
                                   Variables = [:y], 
                                   Periods = 1:periods)
            conditions[1,1] = 0.001
            conditions[1,2] = 0.0005
            conditions[1,3] = 0.0002

            # Both shocks are free (2 shocks > 1 condition per period)
            shocks = Matrix{Union{Nothing,Float64}}(undef, 2, periods)
            for p in 1:periods
                shocks[1,p] = nothing  # e_a is free
                shocks[2,p] = nothing  # e_m is free
            end

            # Get conditional forecast - should find solution minimizing shock magnitude
            result = get_conditional_forecast(FS2000, 
                                             conditions, 
                                             shocks = shocks,
                                             conditions_in_levels = false, 
                                             algorithm = algorithm,
                                             periods = periods)
            
            # Find the index of variable y
            y_idx = findfirst(x -> x == :y, axiskeys(result, 1))
            @test !isnothing(y_idx)
            
            # Check that conditioned variable matches target in all periods
            for p in 1:periods
                target = conditions[1, p]
                actual = result[y_idx, p]
                deviation = abs(actual - target)
                
                # Test that deviation is small
                # Slightly relaxed tolerance for higher-order algorithms
                tol = algorithm == :first_order ? 1e-8 : 1e-6
                @test deviation < tol
            end
        end
    end
end

println("✓ All conditional forecast algorithm tests passed!")
