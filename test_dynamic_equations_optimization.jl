"""
Test script for solving variables across stochastic shocks using the dynamic equations function.

This script demonstrates:
1. Loading a model and computing steady states
2. Generating Sobol quasi-random sequences for shocks
3. Evaluating dynamic equations across multiple shock realizations
4. Solving for variables (past, present, future) using NonlinearSolve.jl
"""

using MacroModelling
using QuasiMonteCarlo
using NonlinearSolve
using SpecialFunctions
using Statistics

# Define a simple RBC model
@model RBC begin
    1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

println("Solving model...")
solve!(RBC)

# Get steady states
println("Computing steady states...")
SS_non_stochastic = get_steady_state(RBC, stochastic = false)
SS_stochastic = get_steady_state(RBC, stochastic = true)

println("Non-stochastic steady state:")
println(SS_non_stochastic)
println("\nStochastic steady state:")
println(SS_stochastic)

# Get auxiliary indices for the steady state input
aux_idx = RBC.solution.perturbation.auxiliary_indices
SS_for_func = SS_non_stochastic[aux_idx.dyn_ss_idx]

# Number of variables and shocks
n_vars = length(RBC.var)
n_shocks = length(RBC.exo)
n_dyn_eqs = length(RBC.dyn_equations)
n_calib_eqs = length(RBC.calibration_equations)
n_total_eqs = n_dyn_eqs + n_calib_eqs

println("\nModel dimensions:")
println("  Variables: ", n_vars)
println("  Shocks: ", n_shocks)
println("  Dynamic equations: ", n_dyn_eqs)
println("  Calibration equations: ", n_calib_eqs)
println("  Total equations: ", n_total_eqs)

# Generate Sobol sequence for shocks
n_draws = 100
println("\nGenerating ", n_draws, " Sobol draws for shocks...")

# Generate Sobol sequence in [0,1]
shock_draws_uniform = QuasiMonteCarlo.sample(n_draws, n_shocks, SobolSample())

# Convert to normal distribution using inverse error function
normal_draws = @. sqrt(2) * erfinv(2 * shock_draws_uniform - 1)

# Standardize to zero mean and unit variance
normal_draws .-= mean(normal_draws, dims=2)
normal_draws ./= Statistics.std(normal_draws, dims=2)

println("Shock draws shape: ", size(normal_draws))
println("Shock draws mean: ", mean(normal_draws, dims=2))
println("Shock draws std: ", Statistics.std(normal_draws, dims=2))

# Calibration parameters
calib_params = zeros(length(RBC.calibration_equations_parameters))

# Define residual function that returns per-equation average residuals
# This is compatible with NonlinearSolve.jl
function residual_function!(residual_avg, vars_flat, p)
    shock_draws, model, SS_for_func, calib_params = p
    
    n_draws = size(shock_draws, 2)
    n_vars = length(model.var)
    
    # Unpack variables: [past; present; future]
    past = vars_flat[1:n_vars]
    present = vars_flat[n_vars+1:2*n_vars]
    future = vars_flat[2*n_vars+1:3*n_vars]
    
    n_eqs = length(model.dyn_equations) + length(model.calibration_equations)
    residual_temp = zeros(n_eqs)
    
    # Initialize average residuals to zero
    fill!(residual_avg, 0.0)
    
    # Sum residuals across all shock draws
    for i in 1:n_draws
        shocks = shock_draws[:, i]
        
        # Evaluate dynamic equations
        get_dynamic_residuals(residual_temp, model.parameter_values, calib_params, 
                            past, present, future, SS_for_func, shocks, model)
        
        # Accumulate residuals
        residual_avg .+= residual_temp
    end
    
    # Average over draws
    residual_avg ./= n_draws
    
    return nothing
end

# Initial guess: use stochastic steady state for all time periods
println("\nSetting up nonlinear problem...")
initial_vars = vcat(SS_stochastic, SS_stochastic, SS_stochastic)

# Package parameters for the residual function
params = (normal_draws, RBC, SS_for_func, calib_params)

# Test initial residuals
residual_test = zeros(n_total_eqs)
residual_function!(residual_test, initial_vars, params)
println("Initial residual norm: ", norm(residual_test))
println("Initial max|residual|: ", maximum(abs.(residual_test)))

# Set up nonlinear problem
println("\nSolving nonlinear system...")
println("This may take a moment...")

prob = NonlinearProblem(residual_function!, initial_vars, params)

# Solve using Newton-Raphson with automatic differentiation
sol = solve(prob, NewtonRaphson(; autodiff = AutoFiniteDiff()))

println("\nSolution complete!")
println("Solution return code: ", sol.retcode)
println("Final residual norm: ", norm(sol.resid))
println("Final max|residual|: ", maximum(abs.(sol.resid)))

# Extract solved variables
solved_past = sol.u[1:n_vars]
solved_present = sol.u[n_vars+1:2*n_vars]
solved_future = sol.u[2*n_vars+1:3*n_vars]

println("\nSolved variables:")
println("Past:    ", solved_past)
println("Present: ", solved_present)
println("Future:  ", solved_future)

println("\nComparison with steady states:")
for (i, var) in enumerate(RBC.var)
    println("  ", var, ":")
    println("    Non-stochastic SS: ", SS_non_stochastic[i])
    println("    Stochastic SS:     ", SS_stochastic[i])
    println("    Solved (past):     ", solved_past[i])
    println("    Solved (present):  ", solved_present[i])
    println("    Solved (future):   ", solved_future[i])
end

# Evaluate residuals at the solution for a few shock draws
println("\nPer-draw residuals at solution for first 5 shock draws:")
residual = zeros(n_total_eqs)
for i in 1:min(5, n_draws)
    shocks = normal_draws[:, i]
    get_dynamic_residuals(residual, RBC.parameter_values, calib_params, 
                        solved_past, solved_present, solved_future, SS_for_func, shocks, RBC)
    println("Draw ", i, ": max|residual| = ", maximum(abs.(residual)), 
            ", mean|residual| = ", sum(abs.(residual))/length(residual))
end

println("\nScript completed successfully!")

# Evaluate residuals at the optimal point for a few shock draws
println("\nResiduals at optimal point for first 5 shock draws:")
residual = zeros(n_total_eqs)
for i in 1:min(5, n_draws)
    shocks = shock_draws[:, i]
    get_dynamic_residuals(residual, RBC.parameter_values, calib_params, 
                        opt_past, opt_present, opt_future, SS_for_func, shocks, RBC)
    println("Draw ", i, ": max|residual| = ", maximum(abs.(residual)), 
            ", mean|residual| = ", sum(abs.(residual))/length(residual))
end

println("\nScript completed successfully!")
