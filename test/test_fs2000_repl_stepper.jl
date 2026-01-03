# Interactive REPL script for stepping through FS2000 higher-order conditional forecast
# This script allows you to step through the Lagrange-Newton algorithm iteration by iteration

using MacroModelling, AxisKeys
using LinearAlgebra

# Load FS2000 model
include("../models/FS2000.jl")

println("="^80)
println("FS2000 Higher-Order Conditional Forecast - REPL Stepper")
println("="^80)

# Setup problem
periods = 1
conditions_matrix = KeyedArray(Matrix{Union{Nothing,Float64}}(undef, 1, periods), 
                              Variables = [:y], 
                              Periods = 1:periods)
conditions_matrix[1,1] = 0.001  # Small deviation

shocks_matrix = Matrix{Union{Nothing,Float64}}(undef, 2, periods)
shocks_matrix[1,1] = nothing  # e_a is free
shocks_matrix[2,1] = nothing  # e_m is free

println("\nProblem Setup:")
println("  Model: FS2000 (2 shocks: e_a, e_m)")
println("  Algorithm: second_order")
println("  Conditions: y = $(conditions_matrix[1,1]) (1 condition)")
println("  Free shocks: e_a, e_m (2 shocks)")
println("  Type: UNDERDETERMINED (more shocks than conditions)")
println()

# Get the model solution
𝓂 = FS2000

# Solve model if not already solved
if !haskey(𝓂.solution, :second_order)
    println("Solving model at second order...")
    get_solution(𝓂, algorithm = :second_order)
end

# Extract problem setup like in get_conditional_forecast
nPast_not_future_and_mixed = count(x -> x == 1, Int.(vcat(𝓂.timings.past_not_future_and_mixed_idx...)))
nExo = 𝓂.timings.nExo

# Extract perturbation matrices
𝐒¹ᵉ = 𝓂.solution.perturbation.first_order.solution_matrix[:, nPast_not_future_and_mixed+1:end]

second_order_solution = 𝓂.solution.perturbation.second_order_solution
𝐔₂ = 𝓂.solution.perturbation.second_order.𝐔₂

println("Extracting second-order perturbation matrices...")
if size(second_order_solution, 2) > 0
    𝐒²_full = second_order_solution * 𝐔₂
    e_in_s⁺ = BitVector(vcat(zeros(Bool, nPast_not_future_and_mixed + 1), ones(Bool, nExo)))
    tmp = kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind
    𝐒²ᵉ = 𝐒²_full[:, shock²_idxs]
    println("  𝐒²ᵉ size: $(size(𝐒²ᵉ))")
else
    𝐒²ᵉ = nothing
    println("  𝐒²ᵉ: empty")
end
𝐒³ᵉ = nothing

# Setup for stepping through algorithm
conditions = Float64[conditions_matrix[1,1]]
all_shocks = zeros(2)
cond_var_idx = [findfirst(x -> x == :y, 𝓂.var)]
free_shock_idx = [1, 2]  # Both shocks free
pruning = false

# Create state update function
function state_update(initial_state, shocks)
    return 𝓂.solution.perturbation.second_order.state_update(initial_state, shocks)
end

initial_state = zeros(length(𝓂.var))

println("\nProblem dimensions:")
println("  Variables: $(length(𝓂.var))")
println("  Shocks: $(length(all_shocks))")
println("  Conditioned variables: $(length(cond_var_idx))")
println("  Free shocks: $(length(free_shock_idx))")
println("  𝐒¹ᵉ size: $(size(𝐒¹ᵉ))")

# Initialize algorithm variables
jacobian_init = -𝐒¹ᵉ[cond_var_idx, free_shock_idx]
new_state_init = state_update(initial_state, all_shocks)
cond_vars_init = new_state_init
residual_init = conditions - cond_vars_init[cond_var_idx]

println("\nInitial conditions:")
println("  Target y: $(conditions[1])")
println("  Current y: $(cond_vars_init[cond_var_idx[1]])")
println("  Initial residual: $(residual_init[1])")
println("  Jacobian (∂y/∂shocks): $(jacobian_init)")

# Pseudoinverse initialization
x = pinv(jacobian_init) * residual_init
x = clamp.(x, -5.0, 5.0)
println("\nPseudoinverse initial guess:")
println("  x (free shocks): $x")
println("  ||x||: $(norm(x))")

λ = zeros(length(cond_var_idx))
xλ = vcat(x, λ)

# Pre-allocate buffers
residual = zeros(length(cond_var_idx))
jacobian = zeros(length(cond_var_idx), length(free_shock_idx))
fxλ = zeros(length(xλ))
fxλp = zeros(length(xλ), length(xλ))

lI = -2.0 * I(length(free_shock_idx))

J = Diagonal(ones(Bool, length(all_shocks)))
kron_buffer = zeros(length(all_shocks) * length(all_shocks))
kron_buffer2 = kron(J, zeros(length(all_shocks)))
∂x = zero(𝐒¹ᵉ)

# LM parameters
μ = 1.0
ν = 3.0

println("\nLevenberg-Marquardt parameters:")
println("  Initial damping μ: $μ")
println("  Scaling factor ν: $ν")

println("\n" * "="^80)
println("Ready to step through iterations!")
println("="^80)
println("\nInstructions:")
println("  - Variables are now available in REPL: x, λ, jacobian, residual, μ, etc.")
println("  - Run the code below step by step")
println("  - Inspect variables at each iteration")
println("\nIteration loop template (paste into REPL):")
println("="^80)

println("""
# --- BEGIN ITERATION LOOP ---
max_iter = 100
tol = 1e-13

for iter in 1:max_iter
    println("\\n" * "="^60)
    println("Iteration \$iter")
    println("="^60)
    
    # Update all shocks
    all_shocks[free_shock_idx] .= x
    println("Current x (shocks): \$x")
    println("||x||: \$(norm(x)), μ: \$μ")
    
    # Compute new state
    new_state = state_update(initial_state, all_shocks)
    cond_vars = new_state
    
    # Compute residual
    residual .= conditions - cond_vars[cond_var_idx]
    println("Target y: \$(conditions[1]), Current y: \$(cond_vars[cond_var_idx[1]])")
    println("Residual: \$(residual[1]), ||residual||: \$(norm(residual))")
    
    # Compute Jacobian analytically
    if !isnothing(𝐒²ᵉ)
        # Second-order: ∂x = 𝐒¹ᵉ + 2 * 𝐒²ᵉ * kron(I, all_shocks)
        kron!(kron_buffer2, J, all_shocks)
        mul!(∂x, 𝐒²ᵉ, kron_buffer2)
        axpby!(1, 𝐒¹ᵉ, 2, ∂x)
        jacobian .= -∂x[cond_var_idx, free_shock_idx]
    else
        jacobian .= -𝐒¹ᵉ[cond_var_idx, free_shock_idx]
    end
    println("Jacobian (∂y/∂shocks): \$(jacobian)")
    
    # Check convergence
    residual_norm = norm(residual)
    if residual_norm < tol
        println("\\n✓ CONVERGED! residual_norm = \$residual_norm < \$tol")
        break
    end
    
    # Build KKT system
    fxλ[1:length(x)] .= 2.0 * x + jacobian' * λ
    fxλ[length(x)+1:end] .= residual
    
    fxλp[1:length(x), 1:length(x)] .= lI
    fxλp[1:length(x), length(x)+1:end] .= jacobian'
    fxλp[length(x)+1:end, 1:length(x)] .= jacobian
    fxλp[length(x)+1:end, length(x)+1:end] .= 0.0
    
    # Add LM damping
    for i in 1:length(x)
        fxλp[i, i] -= 2.0 * μ
    end
    
    println("KKT system norm: \$(norm(fxλ))")
    
    # Solve Newton step
    Δxλ = zeros(length(xλ))
    try
        f̂xλp = factorize(fxλp)
        ldiv!(Δxλ, f̂xλp, fxλ)
    catch e
        println("✗ Matrix factorization failed: \$e")
        println("Increasing damping μ *= ν")
        μ *= ν
        if μ > 1e6
            println("✗ Damping too large, giving up")
            break
        end
        continue
    end
    
    if !all(isfinite, Δxλ)
        println("✗ Non-finite Newton step")
        break
    end
    
    println("Newton step norm: \$(norm(Δxλ))")
    println("  Δx: \$(Δxλ[1:length(x)])")
    println("  Δλ: \$(Δxλ[length(x)+1:end])")
    
    # LM adaptive damping
    current_cost = dot(x, x) + dot(residual, residual)
    println("Current cost: \$current_cost")
    
    # Try the step
    xλ_trial = xλ - Δxλ
    x_trial = xλ_trial[1:length(x)]
    
    all_shocks[free_shock_idx] .= x_trial
    new_state_trial = state_update(initial_state, all_shocks)
    cond_vars_trial = new_state_trial
    residual_trial = conditions - cond_vars_trial[cond_var_idx]
    
    trial_cost = dot(x_trial, x_trial) + dot(residual_trial, residual_trial)
    actual_reduction = current_cost - trial_cost
    
    predicted_reduction = -dot(fxλ, Δxλ) - 0.5 * μ * dot(Δxλ[1:length(x)], Δxλ[1:length(x)])
    
    ρ = abs(predicted_reduction) < 1e-20 ? (actual_reduction > 0 ? 1.0 : -1.0) : actual_reduction / predicted_reduction
    
    println("Trial cost: \$trial_cost")
    println("Actual reduction: \$actual_reduction")
    println("Predicted reduction: \$predicted_reduction")
    println("Gain ratio ρ: \$ρ")
    
    # Accept/reject step
    if ρ > 0.0
        println("✓ Step ACCEPTED")
        xλ .= xλ_trial
        x .= x_trial
        λ .= xλ_trial[length(x)+1:end]
        
        # Update damping
        if ρ > 0.75
            μ = max(μ / ν, 1e-12)
            println("  Very good agreement, reducing μ to \$μ")
        elseif ρ > 0.25
            μ = max(μ / 2, 1e-12)
            println("  Good agreement, reducing μ to \$μ")
        elseif ρ < 0.1
            μ = min(μ * ν, 1e6)
            println("  Poor agreement, increasing μ to \$μ")
        end
    else
        println("✗ Step REJECTED")
        μ = min(μ * ν, 1e6)
        println("  Increasing μ to \$μ")
    end
    
    if μ > 1e6
        println("\\n✗ Damping too large (\$μ > 1e6), stopping")
        break
    end
    
    # Pause for inspection
    println("\\nPress Enter to continue to next iteration (or Ctrl+C to stop)...")
    # readline()  # Uncomment to pause at each iteration
end
# --- END ITERATION LOOP ---
""")

println("\nNote: The iteration loop above can be pasted into the REPL.")
println("Uncomment the readline() line to pause at each iteration for inspection.")
println("\n" * "="^80)
