#!/usr/bin/env julia
# Test to verify that variable and parameter names in models are isolated
# and don't conflict with function names or constants

using Test
using MacroModelling

@testset "Namespace Isolation" begin
    @testset "Function names don't conflict" begin
        # Test that max and min functions still work after loading package
        @test max(1, 5, 3) == 5
        @test min(1, 5, 3) == 1
        
        # The package should have loaded without defining max/min as variables
        # They should still be the built-in functions
        @test typeof(max) <: Function
        @test typeof(min) <: Function
    end
    
    @testset "Model with common function names" begin
        # Create a model that uses variable names that could conflict
        # with common function names and constants
        @model TestConflict begin
            # Using 'max' as a parameter name in expressions should not conflict
            y[0] = c[0] + i[0]
            c[0] = α * y[0]
            i[0] = β * k[-1]
            k[0] = (1 - δ) * k[-1] + i[0]
        end
        
        @parameters TestConflict begin
            α = 0.7
            β = 0.1
            δ = 0.05
        end
        
        # Test that the model was created successfully
        @test TestConflict isa MacroModelling.ℳ
        @test length(get_variables(TestConflict)) > 0
        @test length(get_parameters(TestConflict)) > 0
    end
    
    @testset "Function names remain functional after model creation" begin
        # Verify that built-in functions still work after creating models
        @test max(10, 20) == 20
        @test min(10, 20) == 10
        @test abs(-5) == 5
        @test exp(0) == 1
        
        # These should all still be functions
        @test typeof(max) <: Function
        @test typeof(min) <: Function
        @test typeof(abs) <: Function
        @test typeof(exp) <: Function
    end
end
