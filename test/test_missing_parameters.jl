# Test file for missing parameters functionality

using Test
using MacroModelling

@testset verbose = true "Missing Parameters Functionality" begin

    @testset "Model with missing parameters - basic functionality" begin
        # Define a model with equations that use parameters not yet defined
        @model RBC_missing_params begin
            1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        # Only define some parameters - α, β, δ are missing
        @parameters RBC_missing_params begin
            std_z = 0.01
            ρ = 0.2
        end

        # Test that missing parameters are correctly identified
        @test has_missing_parameters(RBC_missing_params)
        @test sort(get_missing_parameters(RBC_missing_params)) == sort(["α", "β", "δ"])
        @test length(RBC_missing_params.missing_parameters) == 3
        
        # Test that get_irf throws an error when parameters are missing
        @test_throws ErrorException get_irf(RBC_missing_params)
        
        # Test that get_SS throws an error when parameters are missing
        @test_throws ErrorException get_SS(RBC_missing_params)
        
        # Test that simulate throws an error when parameters are missing
        @test_throws ErrorException simulate(RBC_missing_params)
    end

    @testset "Model with missing parameters - providing parameters via keyword" begin
        # Define a model with missing parameters
        @model RBC_missing_provide begin
            1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        # Only define some parameters
        @parameters RBC_missing_provide begin
            std_z = 0.01
            ρ = 0.2
        end

        # Verify parameters are missing before providing them
        @test has_missing_parameters(RBC_missing_provide)
        
        # Provide missing parameters and get IRF
        irf_result = get_irf(RBC_missing_provide, parameters = [:α => 0.5, :β => 0.95, :δ => 0.02])
        
        # After providing parameters, they should no longer be missing
        @test !has_missing_parameters(RBC_missing_provide)
        @test isempty(get_missing_parameters(RBC_missing_provide))
        
        # The IRF should be a valid KeyedArray
        @test irf_result isa KeyedArray
        @test size(irf_result, 2) > 0  # Should have some time periods
    end

    @testset "Model with all parameters defined" begin
        # Define a complete model
        @model RBC_complete begin
            1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        # Define all parameters
        @parameters RBC_complete begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Test that no parameters are missing
        @test !has_missing_parameters(RBC_complete)
        @test isempty(get_missing_parameters(RBC_complete))
        
        # Model should solve without issues
        irf_result = get_irf(RBC_complete)
        @test irf_result isa KeyedArray
    end

    @testset "Model with partial parameter provision" begin
        # Define a model with missing parameters
        @model RBC_partial begin
            1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        # Only define some parameters
        @parameters RBC_partial begin
            std_z = 0.01
            ρ = 0.2
        end

        # Provide only some of the missing parameters - should still fail
        @test_throws ErrorException get_irf(RBC_partial, parameters = [:α => 0.5])
        
        # Verify that α was added but β and δ are still missing
        @test has_missing_parameters(RBC_partial)
        remaining = get_missing_parameters(RBC_partial)
        @test "α" ∉ remaining
        @test "β" ∈ remaining
        @test "δ" ∈ remaining
    end

    @testset "Model with missing parameters - get_SS" begin
        # Define a model with missing parameters
        @model RBC_ss_test begin
            1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_ss_test begin
            std_z = 0.01
            ρ = 0.2
        end

        # Provide missing parameters and get steady state
        ss_result = get_SS(RBC_ss_test, parameters = [:α => 0.5, :β => 0.95, :δ => 0.02])
        
        # Steady state should be computed
        @test ss_result isa KeyedArray
        @test !has_missing_parameters(RBC_ss_test)
    end

    @testset "Model with missing parameters - simulation" begin
        # Define a model with missing parameters
        @model RBC_sim_test begin
            1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_sim_test begin
            std_z = 0.01
            ρ = 0.2
        end

        # Provide missing parameters and simulate
        sim_result = simulate(RBC_sim_test, parameters = [:α => 0.5, :β => 0.95, :δ => 0.02])
        
        # Simulation should work
        @test sim_result isa KeyedArray
        @test !has_missing_parameters(RBC_sim_test)
    end

    @testset "Model display with missing parameters" begin
        # Define a model with missing parameters
        @model RBC_display begin
            1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_display begin
            std_z = 0.01
            ρ = 0.2
        end

        # Capture the model display
        io = IOBuffer()
        show(io, RBC_display)
        display_str = String(take!(io))
        
        # Check that missing parameters are shown
        @test occursin("Missing", display_str)
        # The display shows " Missing:     N" where N is the count
        @test occursin("3", display_str) # 3 missing: α, β, δ
    end
end
