using Revise
using MacroModelling
using QuasiMonteCarlo
using Optimization
# using OptimizationOptimJL
using OptimizationNLopt
import SpecialFunctions: erfinv
using Statistics
# using Zygote

# Define a simple RBC model
# @model RBC begin
#     1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
#     c[0] + k[0] = (1 - δ) * k[-1] + q[0]
#     q[0] = exp(z[0]) * k[-1]^α
#     z[0] = ρ * z[-1] + std_z * eps_z[x]
# end

# @parameters RBC begin
#     std_z = 0.01
#     ρ = 0.2
#     δ = 0.02
#     α = 0.5
#     β = 0.95
# end

include("./models/Gali_2015_chapter_3_nonlinear.jl")

m = Gali_2015_chapter_3_nonlinear
# m.dyn_equations[8]
# Get steady states
SS_non_stochastic = SS(m, derivatives = false) |> collect
SS_stochastic = SSS(m, derivatives = false) |> collect

# SS_for_func = SS_non_stochastic #



# histogram(vec(shock_draws), bins=30, title="Histogram of Uniform [0,1] Sobol Draws", xlabel="Value", ylabel="Frequency")

# histogram(vec(@. sqrt(2) * erfinv(2 * shock_draws - 1)), bins=30, title="Histogram of Normal Sobol Draws", xlabel="Value", ylabel="Frequency")

# Calibration parameters
calib_params = zeros(length(m.calibration_equations_parameters))

# Define objective function: sum of squared residuals across all shock draws
function objective(vars_flat, shock_draws, model, SS_non_stochastic, calib_params)
    # n_vars = length(model.var)
    n_draws = size(shock_draws, 2)

    # Unpack variables: [past; present; future]
    past = vars_flat#[1:n_vars]
    present = vars_flat#[n_vars+1:2*n_vars]
    future = vars_flat#[2*n_vars+1:3*n_vars]
    
    n_eqs = length(model.dyn_equations) + length(model.calibration_equations)

    residual = zeros(eltype(vars_flat), n_eqs)

    # Sum squared residuals across all shock draws
    total_loss = 0.0
    for i in 1:n_draws
        shocks = shock_draws[:, i]
        
        # Evaluate dynamic equations
        get_dynamic_residuals(residual, model.parameter_values, calib_params, 
                            past, present, future, SS_non_stochastic, shocks, model)
        
        # Add squared residuals
        total_loss += sum(residual.^2)
    end
    
    return total_loss / n_draws  # Average loss
end

n_eqs = length(m.dyn_equations) + length(m.calibration_equations)
residual = zeros(n_eqs)

initial_vars = SS_stochastic # vcat(SS_stochastic, SS_stochastic, SS_stochastic)



n_shocks = length(m.exo)

# Generate Sobol sequence for shocks
n_draws = 1000


# Generate Sobol sequence
shock_draws = QuasiMonteCarlo.sample(n_draws, n_shocks, SobolSample())

normal_draws = @. sqrt(2) * erfinv(2 * shock_draws - 1)

normal_draws .-= mean(normal_draws, dims=2)
normal_draws ./= Statistics.std(normal_draws, dims=2)

mean(normal_draws, dims=2)
Statistics.std(normal_draws, dims=2)


objective(initial_vars, normal_draws, m, SS_non_stochastic, calib_params)

# Set up optimization problem
optf = OptimizationFunction((x, p) -> objective(x, normal_draws, m, SS_non_stochastic, calib_params), AutoForwardDiff())
prob = OptimizationProblem(optf, SS_stochastic)

# Solve using BFGS
sol = solve(prob, NLopt.LD_LBFGS())
# solPR = solve(prob, NLopt.LN_PRAXIS())
solNM = solve(prob, NLopt.LN_NELDERMEAD())
# solBO = solve(prob, NLopt.LN_BOBYQA())
# solCO = solve(prob, NLopt.LN_COBYLA())

objective!(residual, sol.u, shock_draws, m, SS_non_stochastic, calib_params)
objective!(residual, initial_vars, shock_draws, m, SS_non_stochastic, calib_params)


ststst = SSS(m, derivatives = false, algorithm = :third_order)

ststst .= sol.u


ststst_orig = SSS(m, derivatives = false, algorithm = :third_order)


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
