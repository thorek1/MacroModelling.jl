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


include("./models/Gali_2015_chapter_3_nonlinear.jl")

m = Gali_2015_chapter_3_nonlinear

# Get steady states
SS_non_stochastic = get_steady_state(m, stochastic = false, derivatives = false) |> collect
SS_stochastic = get_steady_state(m, stochastic = true, derivatives = false) |> collect


# Calibration parameters
calib_params = zeros(length(m.calibration_equations_parameters))

# Define residual function that returns per-equation average residuals
# This is compatible with NonlinearSolve.jl
# function residual_function(vars_flat, p)
function residual_function!(residual_avg, vars_flat, p)
    shock_draws, model, SS_non_stochastic, calib_params = p
    
    n_draws = size(shock_draws, 2)
    # n_vars = length(model.var)
    
    # Unpack variables: [past; present; future]
    past = vars_flat#[1:n_vars]
    present = vars_flat#[n_vars+1:2*n_vars]
    future = vars_flat#[2*n_vars+1:3*n_vars]
    
    n_eqs = length(model.dyn_equations) + length(model.calibration_equations)
    residual_temp = zeros(eltype(vars_flat), n_eqs)
    
    # Initialize average residuals to zero
    fill!(residual_avg, 0.0)
    # residual_avg = zeros(eltype(vars_flat), n_eqs)
    
    # Sum residuals across all shock draws
    for i in 1:n_draws
        shocks = shock_draws[:, i]
        
        # Evaluate dynamic equations
        get_dynamic_residuals(residual_temp, model.parameter_values, calib_params, 
                            past, present, future, SS_non_stochastic, shocks, model)

        # Accumulate residuals
        residual_avg .+= abs.(residual_temp)
    end
    
    # Average over draws
    residual_avg ./= n_draws
    
    return nothing
end


# Number of variables and shocks
n_shocks = length(m.exo)

# Generate Sobol sequence for shocks
n_draws = 1000

# Generate Sobol sequence in [0,1]
shock_draws_uniform = QuasiMonteCarlo.sample(n_draws, n_shocks, SobolSample())

# Convert to normal distribution using inverse error function
normal_draws = @. sqrt(2) * erfinv(2 * shock_draws_uniform - 1)

# Standardize to zero mean and unit variance
normal_draws .-= mean(normal_draws, dims=2)
normal_draws ./= Statistics.std(normal_draws, dims=2)

# Package parameters for the residual function
params = (normal_draws, m, SS_non_stochastic, calib_params)

# Initial guess: use stochastic steady state for all time periods
initial_vars = copy(SS_stochastic)

# Test initial residuals
residual_test = zeros(length(m.dyn_equations) + length(m.calibration_equations))
residual_function!(residual_test, initial_vars, params)
println("Initial residual norm: ", sum(abs2, residual_test))
println("Initial max|residual|: ", maximum(abs.(residual_test)))

# Set up nonlinear problem
prob = NonlinearProblem(residual_function!, initial_vars, params)

# Solve using Newton-Raphson with automatic differentiation
solLM = solve(prob, LevenbergMarquardt(), show_trace = Val(true))
solTR = solve(prob, TrustRegion(), show_trace = Val(true))
# sol = solve(prob, show_trace = Val(true))

sum(abs, solTR.u - SS_stochastic)
sum(abs, solLM.u - SS_stochastic)

sum(abs2, solLM.resid)
sum(abs2, solTR.resid)

residual_function!(residual_test, solLM.u, params)
sum(abs2, residual_test)


println("\nSolution complete!")
println("Solution return code: ", sol.retcode)
println("Final residual norm: ", sum(abs2, sol.resid))
println("Final max|residual|: ", maximum(abs.(sol.resid)))

# Extract solved variables
println("\nSolved variables:", sol.u)

println("\nComparison with steady states:")
for (i, var) in enumerate(m.var)
    println("  ", var, ":")
    println("    Non-stochastic SS: ", SS_non_stochastic[i])
    println("    Stochastic SS:     ", SS_stochastic[i])
    println("    Solved:     ", sol.u[i])
end

n_eqs = length(m.dyn_equations) + length(m.calibration_equations)

# Evaluate residuals at the solution for a few shock draws
println("\nPer-draw residuals at solution for first 5 shock draws:")
residual = zeros(n_eqs)
for i in 1:min(5, n_draws)
    shocks = normal_draws[:, i]
    get_dynamic_residuals(residual, m.parameter_values, calib_params, 
                        sol.u, sol.u, sol.u, SS_non_stochastic, shocks, m)
    println("Draw ", i, ": max|residual| = ", maximum(abs.(residual)), 
            ", mean|residual| = ", sum(abs.(residual))/length(residual))
end


m.dyn_equations_func
