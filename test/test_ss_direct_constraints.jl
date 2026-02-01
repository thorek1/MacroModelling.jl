# Test file for direct steady state constraints: y[ss] = value syntax

@testset "Direct Steady State Constraints Parsing" begin
    
    @testset "Simple numeric constraint" begin
        @model TestNumericSS begin
            1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters TestNumericSS begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
            q[ss] = 1  # Simple numeric constraint
        end

        @test TestNumericSS.constants.post_parameters_macro.ss_direct_constraints_vars == [:q]
        @test TestNumericSS.constants.post_parameters_macro.ss_direct_constraints_exprs == Any[1]
    end

    @testset "Float constraint" begin
        @model TestFloatSS begin
            1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters TestFloatSS begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
            q[ss] = 2.5  # Float constraint
        end

        @test TestFloatSS.constants.post_parameters_macro.ss_direct_constraints_vars == [:q]
        @test TestFloatSS.constants.post_parameters_macro.ss_direct_constraints_exprs == Any[2.5]
    end

    @testset "Expression constraint with parameters" begin
        @model TestExprSS begin
            1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters TestExprSS begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
            scale = 10
            q[ss] = scale * α  # Expression: evaluates to 5.0
        end

        @test TestExprSS.constants.post_parameters_macro.ss_direct_constraints_vars == [:q]
        # The expression should be stored, not evaluated
        @test TestExprSS.constants.post_parameters_macro.ss_direct_constraints_exprs[1] == :(scale * α)
    end

    @testset "Multiple constraints" begin
        @model TestMultipleSS begin
            1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters TestMultipleSS begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
            q[ss] = 1
            k[ss] = 2
        end

        @test length(TestMultipleSS.constants.post_parameters_macro.ss_direct_constraints_vars) == 2
        @test :q ∈ TestMultipleSS.constants.post_parameters_macro.ss_direct_constraints_vars
        @test :k ∈ TestMultipleSS.constants.post_parameters_macro.ss_direct_constraints_vars
    end

    @testset "Complex expression constraint" begin
        @model TestComplexExpr begin
            1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters TestComplexExpr begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            cpie = 2.0
            θ = cpie / 100 + 1  # θ = 1.02
            α = 0.3
            β = 0.95
            q[ss] = α / 4 * θ  # = 0.3 / 4 * 1.02 = 0.0765
        end

        @test TestComplexExpr.constants.post_parameters_macro.ss_direct_constraints_vars == [:q]
        # Complex expressions are stored as Expr
        @test TestComplexExpr.constants.post_parameters_macro.ss_direct_constraints_exprs[1] isa Expr
    end

    @testset "Alternative SS markers" begin
        # Test various valid steady state markers
        @model TestSSMarkers begin
            1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters TestSSMarkers begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
            q[steady_state] = 1  # Should work
        end

        @test TestSSMarkers.constants.post_parameters_macro.ss_direct_constraints_vars == [:q]
    end
end

@testset "Direct SS Constraints Validation" begin
    
    @testset "Error on unknown variable" begin
        @model TestUnknownVar begin
            1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @test_throws AssertionError @parameters TestUnknownVar begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
            invalid_var[ss] = 1  # invalid_var is not a model variable!
        end
    end

    @testset "Error on duplicate constraint" begin
        @model TestDuplicate begin
            1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @test_throws ErrorException @parameters TestDuplicate begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
            q[ss] = 1
            q[ss] = 2  # Duplicate!
        end
    end
end
