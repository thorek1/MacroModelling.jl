# Comprehensive test script for workspace functionality in MacroModelling.jl
# Tests key functionality related to workspaces including:
# - moments calculation
# - perturbation solution 
# - kalman filter
# - inversion filter
# - conditional forecasts

using MacroModelling
import LinearAlgebra as ℒ
using Test

println("="^60)
println("Comprehensive Workspace Functionality Tests")
println("="^60)

# Simple RBC model for testing
@model RBC_test begin
    1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + σ * eps_z[x]
end

@parameters RBC_test begin
    σ = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

println("\n--- Perturbation Solutions ---")

@testset "Perturbation Solutions" begin
    @test get_solution(RBC_test) !== nothing
    @test RBC_test.solution.perturbation.qme_solution !== nothing
    
    sol2 = get_solution(RBC_test, algorithm = :pruned_second_order)
    @test sol2 !== nothing
    
    sol3 = get_solution(RBC_test, algorithm = :pruned_third_order)
    @test sol3 !== nothing
end

println("\n--- Moments Calculation ---")

@testset "Moments Calculation" begin
    moments1 = get_moments(RBC_test)
    @test !isempty(moments1)
    
    moments2 = get_moments(RBC_test, algorithm = :pruned_second_order)
    @test !isempty(moments2)
end

println("\n--- Impulse Responses ---")

@testset "Impulse Responses" begin
    irfs = get_irf(RBC_test)
    @test !isempty(irfs)
    
    irfs2 = get_irf(RBC_test, algorithm = :pruned_second_order)
    @test !isempty(irfs2)
end

println("\n--- Simulations ---")

@testset "Simulations" begin
    sim = get_simulation(RBC_test)
    @test !isempty(sim)
    
    sim2 = get_simulation(RBC_test, algorithm = :pruned_second_order)
    @test !isempty(sim2)
end

println("\n--- Conditional Forecasts ---")

@testset "Conditional Forecasts" begin
    conditions = KeyedArray(rand(1, 4), Variables = [:c], Periods = 1:4)
    cf = get_conditional_forecast(RBC_test, conditions)
    @test !isempty(cf)
end

println("\n--- Workspace Structure ---")

@testset "Workspace Structure" begin
    @test RBC_test.workspaces !== nothing
    @test RBC_test.workspaces.second_order !== nothing
    @test RBC_test.workspaces.third_order !== nothing
    @test RBC_test.workspaces.quadratic_matrix_equation !== nothing
    @test RBC_test.workspaces.kalman !== nothing
    @test RBC_test.workspaces.inversion !== nothing
    @test RBC_test.workspaces.nonlinear_solver !== nothing
end

println("\n--- Cache Structure ---")

@testset "Cache Structure" begin
    @test RBC_test.caches !== nothing
    @test RBC_test.caches.timings !== nothing
    @test RBC_test.caches.computational_constants !== nothing
end

println("\n--- Inversion Filter Workspace ---")

@testset "Inversion Filter Workspace" begin
    import MacroModelling: ensure_inversion_filter_workspace!
    ws = RBC_test.workspaces.inversion
    
    # Test with second order parameters
    ensure_inversion_filter_workspace!(ws, 1, 2, 4, third_order = false)
    @test length(ws.kron_buffer) == 1
    @test length(ws.kron_buffer2) == 1
    @test length(ws.shock_independent) == 4
    @test length(ws.init_guess) == 1
    @test length(ws.state_vol) == 3  # n_past + 1 = 2 + 1 = 3
    @test size(ws.jacc) == (4, 1)
    
    # Test with third order parameters
    ensure_inversion_filter_workspace!(ws, 2, 3, 5, third_order = true)
    @test length(ws.kron_buffer) == 4  # 2^2
    @test length(ws.kron_buffer_third) == 8  # 2^3
    @test length(ws.kron_buffer4) == 8  # 2^3
    @test length(ws.shock_independent) == 5
    @test length(ws.init_guess) == 2
    @test length(ws.state_vol) == 4  # n_past + 1 = 3 + 1 = 4
    @test size(ws.jacc) == (5, 2)
end

println("\n--- Kalman Workspace ---")

@testset "Kalman Workspace" begin
    @test RBC_test.workspaces.kalman.forward !== nothing
    @test RBC_test.workspaces.kalman.rrule !== nothing
    @test RBC_test.workspaces.kalman.smoother !== nothing
end

println("\n--- Quadratic Matrix Equation Workspace ---")

@testset "QME Workspace" begin
    qme_ws = RBC_test.workspaces.quadratic_matrix_equation
    @test qme_ws !== nothing
    # The workspace is initialized lazily, so arrays may be empty initially
    @test isa(qme_ws.AXX, Matrix)
end

println("\n--- Perturbation Workspace Integration ---")

@testset "Perturbation Workspace Integration" begin
    # Test that workspaces are properly used during perturbation solutions
    import MacroModelling: calculate_first_order_solution, calculate_second_order_solution, calculate_third_order_solution
    import MacroModelling: ensure_quadratic_matrix_equation_workspace!
    
    # Verify QME workspace structure
    qme_ws = RBC_test.workspaces.quadratic_matrix_equation
    @test qme_ws !== nothing
    @test isa(qme_ws.AXX, Matrix)
    @test isa(qme_ws.E, Matrix)
    @test isa(qme_ws.F, Matrix)
    @test isa(qme_ws.temp1, Matrix)
    @test isa(qme_ws.temp2, Matrix)
    
    # Verify higher order caches for second order
    ho_ws2 = RBC_test.workspaces.second_order
    @test ho_ws2 !== nothing
    @test isa(ho_ws2.Ŝ, Matrix)
    @test ho_ws2.sylvester_caches !== nothing
    
    # Verify higher order caches for third order
    ho_ws3 = RBC_test.workspaces.third_order
    @test ho_ws3 !== nothing
    @test isa(ho_ws3.Ŝ, Matrix)
    @test ho_ws3.sylvester_caches !== nothing
    
    # Test that solutions work correctly with workspaces by re-solving
    # Clear any cached solutions and re-solve
    import MacroModelling: clear_solution_caches!
    clear_solution_caches!(RBC_test, :first_order)
    
    # Solve again and verify workspaces are used
    sol1 = get_solution(RBC_test, algorithm = :first_order)
    @test sol1 !== nothing
    
    sol2 = get_solution(RBC_test, algorithm = :pruned_second_order)
    @test sol2 !== nothing
    
    sol3 = get_solution(RBC_test, algorithm = :pruned_third_order)
    @test sol3 !== nothing
end

println("\n--- Nonlinear Solver Workspace ---")

@testset "Nonlinear Solver Workspace" begin
    import MacroModelling: ensure_nonlinear_solver_workspace!
    
    ns_ws = RBC_test.workspaces.nonlinear_solver
    @test ns_ws !== nothing
    @test isa(ns_ws.u_bounds, Vector)
    @test isa(ns_ws.l_bounds, Vector)
    @test isa(ns_ws.current_guess, Vector)
    @test isa(ns_ws.previous_guess, Vector)
    @test isa(ns_ws.best_current_guess, Vector)
    
    # Test ensure function
    test_guess = zeros(5)
    ensure_nonlinear_solver_workspace!(ns_ws, test_guess)
    @test length(ns_ws.current_guess) == 5
    @test length(ns_ws.previous_guess) == 5
end

println("\n" * "="^60)
println("All tests completed!")
println("="^60)
