# Tests for Ramsey optimal policy functionality
using MacroModelling
using Test

@testset "Ramsey Optimal Policy" begin
    
    @testset "parse_ramsey_block" begin
        # Test basic parsing
        block = :(begin
            objective = log(c[0])
            instruments = [q]
            discount = β
        end)
        
        config = MacroModelling.parse_ramsey_block(block)
        
        @test config.objective == :(log(c[0]))
        @test config.instruments == [:q]
        @test config.discount == :β
    end
    
    @testset "parse_ramsey_block with multiple instruments" begin
        block = :(begin
            objective = log(c[0]) - 0.5 * y[0]^2
            instruments = [R, τ]
            discount = β
        end)
        
        config = MacroModelling.parse_ramsey_block(block)
        
        @test config.instruments == [:R, :τ]
    end
    
    @testset "Basic RBC Model Transformation" begin
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

        # Test transformation
        ramsey_config = MacroModelling.parse_ramsey_block(:(begin
            objective = log(c[0])
            instruments = [q]
            discount = β
        end))
        
        new_eqs, new_vars, mults = MacroModelling.transform_equations_for_ramsey(
            RBC_ramsey_test.original_equations,
            RBC_ramsey_test.var,
            RBC_ramsey_test.exo,
            vcat(RBC_ramsey_test.parameters, RBC_ramsey_test.parameters_as_function_of_parameters),
            ramsey_config
        )
        
        @test length(new_eqs) == 8  # 4 original + 4 FOCs
        @test length(mults) == 4
        @test length(new_vars) == 8  # 4 original vars + 4 multipliers
        @test all(m -> startswith(string(m), "Lagr_mult_"), mults)
    end
    
    @testset "@ramsey macro" begin
        # Create base model
        @model RBC_macro_test begin
            1/c[0] = β * (1/c[1]) * (α * exp(z[1]) * k[0]^(α-1) + (1-δ))
            c[0] + k[0] = (1-δ) * k[-1] + q[0]  
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_macro_test begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end
        
        # Test @ramsey macro
        result = @ramsey RBC_macro_test begin
            objective = log(c[0])
            instruments = [q]
            discount = β
        end
        
        @test length(result.equations) == 8
        @test length(result.focs) == 4
        @test length(result.multipliers) == 4
        @test length(result.variables) == 8
        @test result.objective == :(log(c[0]))
        @test result.instruments == [:q]
        @test result.discount == :β
    end
end

println("All Ramsey tests passed!")
