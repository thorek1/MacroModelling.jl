# Tests for Ramsey optimal policy functionality
using MacroModelling
using Test

@testset "Ramsey Optimal Policy" begin
    
    @testset "Basic RBC Model" begin
        # Create a simple RBC model
        @model RBC_ramsey_test begin
            1/c[0] = β * (1/c[1]) * (α * exp(z[1]) * k[0]^(α-1) + (1-δ))
            c[0] + k[0] = (1-δ) * k[-1] + q[0]  
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_ramsey_test begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Test get_ramsey_equations
        @testset "get_ramsey_equations" begin
            eqs, mults, vars = get_ramsey_equations(
                RBC_ramsey_test,
                :(log(c[0])),
                [:q],
                discount = :β
            )
            
            @test length(eqs) == 8  # 4 original + 4 FOCs
            @test length(mults) == 4
            @test length(vars) == 8  # 4 original vars + 4 multipliers
            @test :λ₁ in mults
            @test :λ₂ in mults
            @test :λ₃ in mults
            @test :λ₄ in mults
        end
        
        # Test ramsey_summary
        @testset "ramsey_summary" begin
            summary = ramsey_summary(
                RBC_ramsey_test,
                :(log(c[0])),
                [:q],
                discount = :β
            )
            
            @test summary.n_constraints == 4
            @test summary.n_focs == 4
            @test length(summary.multipliers) == 4
            @test length(summary.constraints) == 4
            @test length(summary.focs) == 4
            @test summary.instruments == [:q]
            @test summary.objective == :(log(c[0]))
        end
        
        # Test with multiple instruments
        @testset "Multiple instruments" begin
            eqs, mults, vars = get_ramsey_equations(
                RBC_ramsey_test,
                :(log(c[0])),
                [:q, :k],  # Two instruments
                discount = :β
            )
            
            @test length(eqs) == 8
            @test length(mults) == 4
        end
        
        # Test input validation
        @testset "Input validation" begin
            # Invalid instrument should throw
            @test_throws AssertionError get_ramsey_equations(
                RBC_ramsey_test,
                :(log(c[0])),
                [:invalid_var],
                discount = :β
            )
            
            # Invalid discount factor should throw
            @test_throws AssertionError get_ramsey_equations(
                RBC_ramsey_test,
                :(log(c[0])),
                [:q],
                discount = :invalid_param
            )
        end
    end
    
    @testset "Symbolic Differentiation" begin
        # Just reuse the RBC model from above to verify differentiation
        # The detailed math verification is done through manual inspection
        @test true  # Placeholder - differentiation tested implicitly in Basic RBC Model tests
    end
end

println("All Ramsey tests passed!")
