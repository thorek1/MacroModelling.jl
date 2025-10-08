"""
Test script for optimizing variables across stochastic shocks using the dynamic equations function.

This script demonstrates:
1. Loading a model and computing steady states
2. Generating Sobol quasi-random sequences for shocks
3. Evaluating dynamic equations across multiple shock realizations
4. Optimizing variables (past, present, future) to minimize residuals
"""

using MacroModelling
using QuasiMonteCarlo
using Optimization
using OptimizationOptimJL

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

# Define bounds for shock draws (±3 standard deviations)
shock_std = RBC.parameter_values[findfirst(x -> x == :std_z, RBC.parameters)]
lb = fill(-3 * shock_std, n_shocks)
ub = fill(3 * shock_std, n_shocks)

# Generate Sobol sequence
sampler = SobolSample()
shock_draws = QuasiMonteCarlo.sample(n_draws, lb, ub, sampler)

println("Shock draws shape: ", size(shock_draws))
println("Shock draws range: [", minimum(shock_draws), ", ", maximum(shock_draws), "]")

# Calibration parameters
calib_params = zeros(length(RBC.calibration_equations_parameters))

# Define objective function: sum of squared residuals across all shock draws
function objective(vars_flat, shock_draws, model, SS_for_func, calib_params)
    n_vars = length(model.var)
    n_draws = size(shock_draws, 2)
    n_eqs = length(model.dyn_equations) + length(model.calibration_equations)
    
    # Unpack variables: [past; present; future]
    past = vars_flat[1:n_vars]
    present = vars_flat[n_vars+1:2*n_vars]
    future = vars_flat[2*n_vars+1:3*n_vars]
    
    # Pre-allocate residual vector
    residual = zeros(n_eqs)
    
    # Sum squared residuals across all shock draws
    total_loss = 0.0
    for i in 1:n_draws
        shocks = shock_draws[:, i]
        
        # Evaluate dynamic equations
        get_dynamic_residuals(residual, model.parameter_values, calib_params, 
                            past, present, future, SS_for_func, shocks, model)
        
        # Add squared residuals
        total_loss += sum(residual.^2)
    end
    
    return total_loss / n_draws  # Average loss
end

# Initial guess: use stochastic steady state for all time periods
println("\nSetting up optimization problem...")
initial_vars = vcat(SS_stochastic, SS_stochastic, SS_stochastic)

println("Initial objective value: ", 
      objective(initial_vars, shock_draws, RBC, SS_for_func, calib_params))

# Set up optimization problem
optf = OptimizationFunction((x, p) -> objective(x, shock_draws, RBC, SS_for_func, calib_params))
prob = OptimizationProblem(optf, initial_vars)

println("\nOptimizing variables to minimize residuals across shock draws...")
println("This may take a moment...")

# Solve using BFGS
sol = solve(prob, BFGS())

println("\nOptimization complete!")
println("Final objective value: ", sol.objective)
println("Initial objective value: ", objective(initial_vars, shock_draws, RBC, SS_for_func, calib_params))
println("Improvement: ", (1 - sol.objective / objective(initial_vars, shock_draws, RBC, SS_for_func, calib_params)) * 100, "%")

# Extract optimized variables
opt_past = sol.u[1:n_vars]
opt_present = sol.u[n_vars+1:2*n_vars]
opt_future = sol.u[2*n_vars+1:3*n_vars]

println("\nOptimized variables:")
println("Past:    ", opt_past)
println("Present: ", opt_present)
println("Future:  ", opt_future)

println("\nComparison with steady states:")
for (i, var) in enumerate(RBC.var)
    println("  ", var, ":")
    println("    Non-stochastic SS: ", SS_non_stochastic[i])
    println("    Stochastic SS:     ", SS_stochastic[i])
    println("    Optimized (past):  ", opt_past[i])
    println("    Optimized (pres):  ", opt_present[i])
    println("    Optimized (fut):   ", opt_future[i])
end

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
