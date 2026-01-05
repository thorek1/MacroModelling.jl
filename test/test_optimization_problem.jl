# Test file for derive_focs functionality
# Tests the automatic derivation of First Order Conditions from optimization problems

using MacroModelling
using Test

@testset "Optimization Problem FOC Derivation" begin
    
    @testset "Simple consumption problem" begin
        # Simple problem: max log(C) s.t. C + K = (1-δ)*K[-1] + Y
        controls = [:C, :K]
        objective = :(U[0] = log(C[0]) + β * U[1])
        constraints = [:(C[0] + K[0] = (1 - δ) * K[-1] + Y[0])]
        
        focs, mults = derive_focs(
            controls = controls,
            objective = objective,
            constraints = constraints,
            discount_factor = :β,
            block_name = "test"
        )
        
        @test length(focs) == 2
        @test length(mults) == 1
        @test mults[1] == :λ_test_1
    end
    
    @testset "Consumer problem with labor" begin
        # Consumer maximizes: U = μ*log(C) + (1-μ)*log(1-L_s) + β*U[1]
        # subject to: I + C = π + r*K[-1] + W*L_s
        #             K = (1-δ)*K[-1] + I
        
        controls = [:K_s, :C, :L_s, :I]
        objective = :(U[0] = μ * log(C[0]) + (1 - μ) * log(1 - L_s[0]) + β * U[1])
        constraints = [
            :(I[0] + C[0] = π[0] + r[0] * K_s[-1] + W[0] * L_s[0]),
            :(K_s[0] = (1 - δ) * K_s[-1] + I[0])
        ]
        
        focs, mults = derive_focs(
            controls = controls,
            objective = objective,
            constraints = constraints,
            discount_factor = :β,
            block_name = "consumer"
        )
        
        @test length(focs) == 4  # One FOC for each control
        @test length(mults) == 2  # One multiplier for each constraint
        @test :λ_consumer_1 in mults
        @test :λ_consumer_2 in mults
    end
    
    @testset "Static firm problem" begin
        # Firm maximizes: π = Y - L_d*W - r*K_d
        # subject to: Y = Z * K_d^α * L_d^(1-α)
        
        controls = [:K_d, :L_d, :Y]
        objective = :(π[0] = Y[0] - L_d[0] * W[0] - r[0] * K_d[0])
        constraints = [
            :(Y[0] = Z[0] * K_d[0]^α * L_d[0]^(1 - α))
        ]
        
        focs, mults = derive_focs(
            controls = controls,
            objective = objective,
            constraints = constraints,
            discount_factor = :β,
            block_name = "firm"
        )
        
        @test length(focs) == 3  # One FOC for each control
        @test length(mults) == 1  # One multiplier for the production constraint
        @test mults[1] == :λ_firm_1
    end
    
    @testset "Multiple terms in instantaneous objective" begin
        # Test that all terms without discount factor are captured
        objective = :(U[0] = log(C[0]) + ψ * log(1 - L[0]) + γ * log(G[0]) + β * U[1])
        
        instant_obj, obj_var = MacroModelling.extract_instantaneous_objective(objective, :β)
        
        @test obj_var == :U
        # The instantaneous objective should contain C, L, and G terms
        obj_str = string(instant_obj)
        @test occursin("C[0]", obj_str)
        @test occursin("L[0]", obj_str)  
        @test occursin("G[0]", obj_str)
    end
    
    @testset "Helper functions" begin
        # Test opt_find_control_occurrences
        expr = :(C[0] + K[0] + β * K[-1])
        @test 0 in MacroModelling.opt_find_control_occurrences(expr, :C)
        @test 0 in MacroModelling.opt_find_control_occurrences(expr, :K)
        @test -1 in MacroModelling.opt_find_control_occurrences(expr, :K)
        
        # Test opt_shift_all_time_indices
        expr = :(C[0] + K[-1])
        shifted = MacroModelling.opt_shift_all_time_indices(expr, 1)
        shifted_str = string(shifted)
        @test occursin("C[1]", shifted_str)
        @test occursin("K[0]", shifted_str)
        
        # Test contains_recursive_term
        @test MacroModelling.contains_recursive_term(:(U[0] = log(C[0]) + β * U[1]), :β)
        @test !MacroModelling.contains_recursive_term(:(π[0] = Y[0] - W[0] * L[0]), :β)
        
        # Test contains_symbol
        @test MacroModelling.contains_symbol(:(a + β * b), :β)
        @test !MacroModelling.contains_symbol(:(a + b), :β)
    end
end

println("All optimization problem tests passed!")
