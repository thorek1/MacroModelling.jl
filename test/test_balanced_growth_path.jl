using MacroModelling
using Test

@testset "Balanced Growth Path" begin
    @testset "Model without balanced growth" begin
        @model RBC_no_growth begin
            1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_no_growth begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        @test has_balanced_growth(RBC_no_growth) == false
        
        bg_info = get_balanced_growth_path_info(RBC_no_growth)
        @test bg_info.has_balanced_growth == false
        @test isempty(bg_info.trend_vars)
        @test isempty(bg_info.deflators)
        @test isempty(bg_info.detrended_vars)
    end

    @testset "Deflator parsing in @model" begin
        # Test that deflator option is correctly parsed
        @model RBC_with_deflator deflator = Dict(:c => :A, :k => :A) begin
            A[0] = γ * A[-1]
            1/c[0] = β * (1/c[1]) * (α * A[1]^(1-α) * k[0]^(α-1) + 1-δ)
            c[0] + k[0] = A[0]^(1-α) * k[-1]^α + (1-δ)*k[-1]
            z[0] = ρ * z[-1] + σ * eps[x]
        end

        @parameters RBC_with_deflator trend_var = Dict(:A => :γ) begin
            γ = 1.02
            α = 0.33
            β = 0.99
            δ = 0.025
            ρ = 0.9
            σ = 0.01
        end

        @test has_balanced_growth(RBC_with_deflator) == true
        
        bg_info = get_balanced_growth_path_info(RBC_with_deflator)
        @test bg_info.has_balanced_growth == true
        @test :A ∈ keys(bg_info.trend_vars)
        @test bg_info.trend_vars[:A] == :γ
        @test :c ∈ keys(bg_info.deflators)
        @test :k ∈ keys(bg_info.deflators)
        @test bg_info.deflators[:c] == :A
        @test bg_info.deflators[:k] == :A
        @test :c ∈ bg_info.detrended_vars
        @test :k ∈ bg_info.detrended_vars
    end

    @testset "Trend_var parsing in @parameters" begin
        # Test that trend_var is correctly stored when specified in @parameters
        # Use the same model structure as previous test to avoid analytical solve edge cases
        @model TrendVarTest deflator = Dict(:y => :A) begin
            A[0] = γ * A[-1]
            1/c[0] = β * (1/c[1]) * (α * y[1]/k[0] + 1-δ)
            y[0] = k[-1]^α
            c[0] + k[0] = y[0] + (1-δ)*k[-1]
        end

        @parameters TrendVarTest trend_var = Dict(:A => :γ) begin
            γ = 1.015
            α = 0.3
            β = 0.99
            δ = 0.1
        end

        bg_info = get_balanced_growth_path_info(TrendVarTest)
        @test :A ∈ keys(bg_info.trend_vars)
        @test bg_info.trend_vars[:A] == :γ
    end

    @testset "Empty deflator still works" begin
        # Test that model works when deflator is explicitly empty
        @model RBC_empty_deflator deflator = Dict{Symbol,Symbol}() begin
            1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_empty_deflator begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        @test has_balanced_growth(RBC_empty_deflator) == false
    end
end
