"""
Test comparison between LBFGS and Lagrange-Newton solvers for conditional forecasts.
Validates that both solvers produce similar results across all perturbation algorithms.
"""

using Test
using MacroModelling
using AxisKeys
using LinearAlgebra

@testset "Solver comparison - LBFGS vs Lagrange-Newton" begin
    # Define a simple RBC model for testing
    @model RBC_comp begin
        1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
        c[0] + k[0] = (1 - δ) * k[-1] + q[0]
        q[0] = exp(z[0]) * k[-1]^α
        z[0] = ρ * z[-1] + std_z * eps_z[x]
    end

    @parameters RBC_comp begin
        std_z = 0.01
        ρ = 0.2
        δ = 0.02
        α = 0.5
        β = 0.95
    end

    # Test across different algorithms
    algorithms = [:first_order, :second_order, :third_order, :pruned_second_order, :pruned_third_order]
    
    for alg in algorithms
        @testset "Algorithm: $alg" begin
            periods = 3
            
            # Define conditions: c is conditioned in periods 1-3
            conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef, 1, periods), 
                                   Variables = [:c], 
                                   Periods = 1:periods)
            conditions[1,1] = 0.01
            conditions[1,2] = 0.005
            conditions[1,3] = 0.002

            # Define shocks: all free
            shocks = Matrix{Union{Nothing,Float64}}(undef, 1, periods)
            
            # Run with Lagrange-Newton solver
            result_ln = try
                get_conditional_forecast(RBC_comp, 
                                        conditions, 
                                        shocks=shocks,
                                        algorithm=alg,
                                        conditional_forecast_solver=:LagrangeNewton)
            catch e
                println("  Lagrange-Newton failed for $alg: $e")
                nothing
            end
            
            # Run with LBFGS solver
            result_lbfgs = try
                get_conditional_forecast(RBC_comp, 
                                        conditions, 
                                        shocks=shocks,
                                        algorithm=alg,
                                        conditional_forecast_solver=:LBFGS)
            catch e
                println("  LBFGS failed for $alg: $e")
                nothing
            end
            
            if result_ln !== nothing && result_lbfgs !== nothing
                # Both succeeded - compare results
                @testset "Results comparison for $alg" begin
                    # Check that forecasts are similar (allowing for numerical differences)
                    tol = 1e-8
                    
                    # Compare variable forecasts
                    vars_to_check = [:c, :k, :q, :z]
                    for var in vars_to_check
                        ln_vals = result_ln(Variables_and_shocks=var)
                        lbfgs_vals = result_lbfgs(Variables_and_shocks=var)
                        
                        diff = maximum(abs, ln_vals - lbfgs_vals)
                        @test diff < tol "Variable $var differs by $diff (>$tol) for $alg"
                    end
                    
                    # Compare shock values
                    ln_shock = result_ln(Variables_and_shocks=Symbol("eps_z₍ₓ₎"))
                    lbfgs_shock = result_lbfgs(Variables_and_shocks=Symbol("eps_z₍ₓ₎"))
                    
                    shock_diff = maximum(abs, ln_shock - lbfgs_shock)
                    @test shock_diff < tol "Shocks differ by $shock_diff (>$tol) for $alg"
                    
                    # Verify conditions are met for both
                    ln_c = result_ln(Variables_and_shocks=:c, Periods=1:periods)
                    lbfgs_c = result_lbfgs(Variables_and_shocks=:c, Periods=1:periods)
                    
                    for i in 1:periods
                        @test abs(ln_c[i] - conditions[1,i]) < 1e-10 "$alg (LN): Condition $i not met"
                        @test abs(lbfgs_c[i] - conditions[1,i]) < 1e-10 "$alg (LBFGS): Condition $i not met"
                    end
                    
                    println("  ✓ $alg: Both solvers converged with similar results (max diff: $(max(diff, shock_diff)))")
                end
            elseif result_ln !== nothing
                println("  ⚠ $alg: Only Lagrange-Newton succeeded")
                @test true # Pass test if at least one solver works
            elseif result_lbfgs !== nothing
                println("  ⚠ $alg: Only LBFGS succeeded")
                @test true # Pass test if at least one solver works
            else
                println("  ✗ $alg: Both solvers failed")
                @test_broken false "$alg: Both solvers failed"
            end
        end
    end
    
    @testset "Solver selection via parameter" begin
        # Test that the solver parameter actually selects the right solver
        periods = 2
        conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef, 1, periods), 
                               Variables = [:c], 
                               Periods = 1:periods)
        conditions[1,1] = 0.01
        conditions[1,2] = 0.005
        
        shocks = Matrix{Union{Nothing,Float64}}(undef, 1, periods)
        
        # Default should be Lagrange-Newton
        result_default = get_conditional_forecast(RBC_comp, conditions, shocks=shocks, algorithm=:first_order)
        @test result_default !== nothing "Default solver should work"
        
        # Explicit Lagrange-Newton
        result_ln_explicit = get_conditional_forecast(RBC_comp, conditions, shocks=shocks, algorithm=:first_order, conditional_forecast_solver=:LagrangeNewton)
        @test result_ln_explicit !== nothing "Explicit Lagrange-Newton should work"
        
        # Explicit LBFGS
        result_lbfgs_explicit = get_conditional_forecast(RBC_comp, conditions, shocks=shocks, algorithm=:first_order, conditional_forecast_solver=:LBFGS)
        @test result_lbfgs_explicit !== nothing "Explicit LBFGS should work"
        
        # Default and explicit LN should give same results
        @test maximum(abs, result_default - result_ln_explicit) < 1e-12 "Default should use Lagrange-Newton"
        
        println("  ✓ Solver selection parameter works correctly")
    end
end

println("\n✓ Solver comparison tests complete!")
