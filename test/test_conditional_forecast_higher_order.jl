# Test script for conditional forecasting with higher order perturbation solutions
# Tests scenarios where there are more shocks than conditions, and same number of shocks as conditions
# Uses both RBC_CME model (2 shocks) and Smets_Wouters_2007 model (7 shocks)

using MacroModelling
using LinearAlgebra
using AxisKeys
using Test

# Define the RBC_CME model with 2 shocks
@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

# Helper function to check if conditions are fulfilled
function check_conditions_fulfilled(forecast, conditions_matrix, var_names; tol=1e-5)
    for period in axes(conditions_matrix, 2)
        for (j, var) in enumerate(var_names)
            val = conditions_matrix[j, period]
            if val !== nothing
                var_idx = findfirst(x -> x == var, axiskeys(forecast, 1))
                if var_idx !== nothing
                    actual = forecast[var_idx, period]
                    if abs(actual - val) > tol
                        return false, "Period $period, Var $var: expected $val, got $actual (diff=$(abs(actual-val)))"
                    end
                end
            end
        end
    end
    return true, "All conditions fulfilled"
end

@testset "Conditional Forecasting with Higher Order Perturbation" begin
    
    @testset "RBC_CME Model (2 shocks)" begin
        conditions_vars = RBC_CME.var
        c_idx = findfirst(==(:c), conditions_vars)
        y_idx = findfirst(==(:y), conditions_vars)
        
        @testset "More shocks (2) than conditions (1) in period 1, same (2=2) in period 2" begin
            conditions = Matrix{Union{Nothing,Float64}}(nothing, length(conditions_vars), 2)
            conditions[c_idx, 1] = 0.01  # 1 condition in period 1 (2 free shocks)
            conditions[c_idx, 2] = 0.005  # 2 conditions in period 2 (2 free shocks = same number)
            conditions[y_idx, 2] = 0.02
            
            for alg in [:second_order, :pruned_second_order, :third_order, :pruned_third_order]
                @testset "$alg" begin
                    forecast = get_conditional_forecast(RBC_CME, conditions, 
                                                        algorithm = alg, 
                                                        conditions_in_levels = false, 
                                                        periods = 10)
                    
                    fulfilled, msg = check_conditions_fulfilled(forecast, conditions, conditions_vars)
                    @test fulfilled
                end
            end
        end
        
        @testset "Single condition comparison: first_order vs second_order" begin
            simple_conditions = Matrix{Union{Nothing,Float64}}(nothing, length(conditions_vars), 1)
            simple_conditions[c_idx, 1] = 0.01
            
            forecast_1st = get_conditional_forecast(RBC_CME, simple_conditions, 
                                                    algorithm = :first_order, 
                                                    conditions_in_levels = false, 
                                                    periods = 10)
            forecast_2nd = get_conditional_forecast(RBC_CME, simple_conditions, 
                                                    algorithm = :second_order, 
                                                    conditions_in_levels = false, 
                                                    periods = 10)
            
            c_forecast_idx = findfirst(==(:c), axiskeys(forecast_1st, 1))
            @test abs(forecast_1st[c_forecast_idx, 1] - 0.01) < 1e-6
            @test abs(forecast_2nd[c_forecast_idx, 1] - 0.01) < 1e-6
        end
        
        @testset "Fixed shocks" begin
            conditions = Matrix{Union{Nothing,Float64}}(nothing, length(conditions_vars), 2)
            conditions[c_idx, 1] = 0.01
            conditions[y_idx, 2] = 0.02
            
            # Fix delta_eps shock in period 1
            shocks = Matrix{Union{Nothing,Float64}}(nothing, 2, 2)
            shocks[1, 1] = 0.0  # Fix delta_eps shock
            
            forecast = get_conditional_forecast(RBC_CME, conditions, 
                                                shocks = shocks,
                                                algorithm = :pruned_second_order, 
                                                conditions_in_levels = false, 
                                                periods = 10)
            
            fulfilled, msg = check_conditions_fulfilled(forecast, conditions, conditions_vars)
            @test fulfilled
            
            # Verify the fixed shock is respected
            # The shock in output is named with subscript: delta_eps₍ₓ₎
            delta_eps_sym = Symbol("delta_eps₍ₓ₎")
            delta_eps_idx = findfirst(==(delta_eps_sym), axiskeys(forecast, 1))
            @test !isnothing(delta_eps_idx)
            @test forecast[delta_eps_idx, 1] ≈ 0.0 atol=1e-10
        end
    end
    
    @testset "Smets_Wouters_2007 Model (7 shocks)" begin
        # Include SW07 model
        include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007.jl"))
        
        sw_vars = Smets_Wouters_2007.var
        y_idx = findfirst(==(:y), sw_vars)
        c_idx = findfirst(==(:c), sw_vars)
        inve_idx = findfirst(==(:inve), sw_vars)
        pinf_idx = findfirst(==(:pinf), sw_vars)
        
        @testset "2 conditions with 7 free shocks (more shocks than conditions)" begin
            conditions = Matrix{Union{Nothing,Float64}}(nothing, length(sw_vars), 2)
            conditions[y_idx, 1] = 0.01
            conditions[c_idx, 2] = 0.005
            
            for alg in [:pruned_second_order, :third_order, :pruned_third_order]
                @testset "$alg" begin
                    forecast = get_conditional_forecast(Smets_Wouters_2007, conditions, 
                                                        algorithm = alg, 
                                                        conditions_in_levels = false, 
                                                        periods = 5)
                    
                    y_forecast_idx = findfirst(==(:y), axiskeys(forecast, 1))
                    c_forecast_idx = findfirst(==(:c), axiskeys(forecast, 1))
                    
                    @test abs(forecast[y_forecast_idx, 1] - 0.01) < 1e-5
                    @test abs(forecast[c_forecast_idx, 2] - 0.005) < 1e-5
                end
            end
        end
        
        @testset "3 conditions with 7 free shocks" begin
            conditions = Matrix{Union{Nothing,Float64}}(nothing, length(sw_vars), 2)
            conditions[y_idx, 1] = 0.01
            conditions[c_idx, 1] = 0.008
            conditions[inve_idx, 2] = 0.02
            
            forecast = get_conditional_forecast(Smets_Wouters_2007, conditions, 
                                                algorithm = :pruned_second_order, 
                                                conditions_in_levels = false, 
                                                periods = 5)
            
            y_forecast_idx = findfirst(==(:y), axiskeys(forecast, 1))
            c_forecast_idx = findfirst(==(:c), axiskeys(forecast, 1))
            inve_forecast_idx = findfirst(==(:inve), axiskeys(forecast, 1))
            
            @test abs(forecast[y_forecast_idx, 1] - 0.01) < 1e-5
            @test abs(forecast[c_forecast_idx, 1] - 0.008) < 1e-5
            @test abs(forecast[inve_forecast_idx, 2] - 0.02) < 1e-5
        end
        
        @testset "2 conditions + 5 fixed shocks = 2 free shocks (same as conditions)" begin
            conditions = Matrix{Union{Nothing,Float64}}(nothing, length(sw_vars), 1)
            conditions[y_idx, 1] = 0.01
            conditions[c_idx, 1] = 0.008
            
            # Fix 5 shocks to 0 in period 1, leaving only 2 free (equal to number of conditions)
            # Use shock indices based on model's exo vector
            sw_shocks = Smets_Wouters_2007.exo
            @test length(sw_shocks) == 7  # Expected 7 shocks in SW07 model
            
            shocks = Matrix{Union{Nothing,Float64}}(nothing, 7, 1)
            # Fix the first 5 shocks (ea, eb, eg, em, epinf)
            for i in 1:5
                shocks[findfirst(==(sw_shocks[i]), sw_shocks), 1] = 0.0
            end
            # Leave eqs and ew free (2 free shocks = 2 conditions)
            
            forecast = get_conditional_forecast(Smets_Wouters_2007, conditions, 
                                                shocks = shocks,
                                                algorithm = :pruned_second_order, 
                                                conditions_in_levels = false, 
                                                periods = 5)
            
            y_forecast_idx = findfirst(==(:y), axiskeys(forecast, 1))
            c_forecast_idx = findfirst(==(:c), axiskeys(forecast, 1))
            
            @test abs(forecast[y_forecast_idx, 1] - 0.01) < 1e-4
            @test abs(forecast[c_forecast_idx, 1] - 0.008) < 1e-4
        end
        
        @testset "5 conditions with 7 free shocks" begin
            conditions = Matrix{Union{Nothing,Float64}}(nothing, length(sw_vars), 2)
            conditions[y_idx, 1] = 0.01
            conditions[c_idx, 1] = 0.008
            conditions[inve_idx, 1] = 0.015
            conditions[y_idx, 2] = 0.005
            conditions[c_idx, 2] = 0.004
            
            forecast = get_conditional_forecast(Smets_Wouters_2007, conditions, 
                                                algorithm = :second_order, 
                                                conditions_in_levels = false, 
                                                periods = 5)
            
            y_forecast_idx = findfirst(==(:y), axiskeys(forecast, 1))
            c_forecast_idx = findfirst(==(:c), axiskeys(forecast, 1))
            inve_forecast_idx = findfirst(==(:inve), axiskeys(forecast, 1))
            
            @test abs(forecast[y_forecast_idx, 1] - 0.01) < 1e-5
            @test abs(forecast[c_forecast_idx, 1] - 0.008) < 1e-5
            @test abs(forecast[inve_forecast_idx, 1] - 0.015) < 1e-5
            @test abs(forecast[y_forecast_idx, 2] - 0.005) < 1e-5
            @test abs(forecast[c_forecast_idx, 2] - 0.004) < 1e-5
        end
        
        @testset "3 conditions across 3 periods with 7 free shocks" begin
            conditions = Matrix{Union{Nothing,Float64}}(nothing, length(sw_vars), 3)
            conditions[y_idx, 1] = 0.01
            conditions[c_idx, 2] = 0.005
            conditions[pinf_idx, 3] = 0.001
            
            forecast = get_conditional_forecast(Smets_Wouters_2007, conditions, 
                                                algorithm = :pruned_third_order, 
                                                conditions_in_levels = false, 
                                                periods = 5)
            
            y_forecast_idx = findfirst(==(:y), axiskeys(forecast, 1))
            c_forecast_idx = findfirst(==(:c), axiskeys(forecast, 1))
            pinf_forecast_idx = findfirst(==(:pinf), axiskeys(forecast, 1))
            
            @test abs(forecast[y_forecast_idx, 1] - 0.01) < 1e-5
            @test abs(forecast[c_forecast_idx, 2] - 0.005) < 1e-5
            @test abs(forecast[pinf_forecast_idx, 3] - 0.001) < 1e-5
        end
    end
end

println("\nAll conditional forecasting tests completed successfully!")
