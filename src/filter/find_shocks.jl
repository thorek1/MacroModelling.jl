# Algorithms
# - LagrangeNewton: fast, but no guarantee of convergence to global minimum
# - COBYLA: best known chances of convergence to global minimum; ok speed for third order; lower tol on optimality conditions (1e-7)
# - SLSQP: relatively slow and not guaranteed to converge to global minimum

# Generalized find_shocks for conditional forecasts
# This function finds shocks that minimize their squared magnitude while satisfying 
# conditional forecast constraints (only some variables match target values)
# Uses analytical derivatives from perturbation solution matrices (like find_shocks)

@stable default_mode = "disable" begin
function find_shocks_conditional_forecast(::Val{:LagrangeNewton},
                                         state_update::Function,
                                         initial_state::Union{Vector{Float64}, Vector{Vector{Float64}}},
                                         all_shocks::Vector{Float64},
                                         conditions::Vector{Float64},
                                         cond_var_idx::Vector{Int},
                                         free_shock_idx::Vector{Int},
                                         pruning::Bool,
                                         𝐒¹ᵉ::AbstractMatrix{Float64},  # Shock columns from first-order solution
                                         𝐒²ᵉ::Union{AbstractMatrix{Float64}, Nothing},  # Second-order solution matrix
                                         𝐒³ᵉ::Union{AbstractMatrix{Float64}, Nothing},  # Third-order solution matrix
                                         T::timings;
                                         max_iter::Int = 1000,
                                         tol::Float64 = 1e-13)
    # For underdetermined systems (more shocks than conditions), go straight to LM
    # as it handles these cases better
    if length(free_shock_idx) > length(cond_var_idx)
        # Use fewer iterations for underdetermined systems as each iteration is expensive
        max_iter_adjusted = min(max_iter, 100)  # Cap at 100 iterations
        
        # Try LM directly for underdetermined systems
        x, converged = find_shocks_conditional_forecast_core(
            state_update, initial_state, all_shocks, conditions,
            cond_var_idx, free_shock_idx, pruning,
            𝐒¹ᵉ, 𝐒²ᵉ, 𝐒³ᵉ, T;
            max_iter=max_iter_adjusted, tol=tol, use_globalization=false, use_levenberg_marquardt=true, use_continuation=false)
        
        if !converged
            # Last resort: try with very relaxed tolerance
            x, converged = find_shocks_conditional_forecast_core(
                state_update, initial_state, all_shocks, conditions,
                cond_var_idx, free_shock_idx, pruning,
                𝐒¹ᵉ, 𝐒²ᵉ, 𝐒³ᵉ, T;
                max_iter=max_iter_adjusted, tol=tol*100, use_globalization=false, use_levenberg_marquardt=true, use_continuation=false)
        end
        
        return x, converged
    end
    
    # For determined/overdetermined systems, use standard cascading strategy
    # First try without globalization (faster)
    x, converged = find_shocks_conditional_forecast_core(
        state_update, initial_state, all_shocks, conditions,
        cond_var_idx, free_shock_idx, pruning,
        𝐒¹ᵉ, 𝐒²ᵉ, 𝐒³ᵉ, T;
        max_iter=max_iter, tol=tol, use_globalization=false, use_levenberg_marquardt=false, use_continuation=false)
    
    # If failed, try with line search globalization
    if !converged
        x, converged = find_shocks_conditional_forecast_core(
            state_update, initial_state, all_shocks, conditions,
            cond_var_idx, free_shock_idx, pruning,
            𝐒¹ᵉ, 𝐒²ᵉ, 𝐒³ᵉ, T;
            max_iter=max_iter, tol=tol, use_globalization=true, use_levenberg_marquardt=false, use_continuation=false)
    end
    
    # If still failed, try Levenberg-Marquardt
    if !converged
        x, converged = find_shocks_conditional_forecast_core(
            state_update, initial_state, all_shocks, conditions,
            cond_var_idx, free_shock_idx, pruning,
            𝐒¹ᵉ, 𝐒²ᵉ, 𝐒³ᵉ, T;
            max_iter=max_iter, tol=tol, use_globalization=false, use_levenberg_marquardt=true, use_continuation=false)
    end
    
    return x, converged
end
end # dispatch_doctor


@stable default_mode = "disable" begin
function find_shocks_conditional_forecast_core(
                                         state_update::Function,
                                         initial_state::Union{Vector{Float64}, Vector{Vector{Float64}}},
                                         all_shocks::Vector{Float64},
                                         conditions::Vector{Float64},
                                         cond_var_idx::Vector{Int},
                                         free_shock_idx::Vector{Int},
                                         pruning::Bool,
                                         𝐒¹ᵉ::AbstractMatrix{Float64},  # Shock columns from first-order solution
                                         𝐒²ᵉ::Union{AbstractMatrix{Float64}, Nothing},  # Second-order solution matrix
                                         𝐒³ᵉ::Union{AbstractMatrix{Float64}, Nothing},  # Third-order solution matrix
                                         T::timings;
                                         max_iter::Int = 1000,
                                         tol::Float64 = 1e-13,
                                         use_globalization::Bool = false,
                                         use_levenberg_marquardt::Bool = false,
                                         use_continuation::Bool = false)
    
    # Pure Lagrange-Newton: when no globalization methods are enabled
    pure_newton = !use_globalization && !use_levenberg_marquardt && !use_continuation
    
    # Track improvement for pure Newton
    last_residual_norm = Inf
    stalled_count = 0
    
    # Initialize free shocks
    # For underdetermined systems (more shocks than conditions), use pseudoinverse for better initial guess
    if length(free_shock_idx) > length(cond_var_idx) && !use_levenberg_marquardt && !use_continuation
        # Get initial Jacobian (linear part)
        jacobian_init = -𝐒¹ᵉ[cond_var_idx, free_shock_idx]
        
        # Compute initial state
        new_state_init = state_update(initial_state, all_shocks)
        cond_vars_init = pruning ? sum(new_state_init) : new_state_init
        residual_init = conditions - cond_vars_init[cond_var_idx]
        
        # Use pseudoinverse to get minimum norm solution as initial guess
        # x = J^+ * residual where J^+ is pseudoinverse
        try
            x = ℒ.pinv(jacobian_init) * residual_init
            # Limit initial guess to reasonable range
            x = clamp.(x, -5.0, 5.0)  # Tighter bounds for better stability
        catch
            x = zeros(length(free_shock_idx))
        end
    elseif use_levenberg_marquardt || use_continuation
        # For LM and continuation, use smaller initial shocks for stability
        x = zeros(length(free_shock_idx))
    else
        x = zeros(length(free_shock_idx))
    end
    
    # For continuation method - DISABLED: too slow with recursive calls
    # Left as placeholder for future non-recursive implementation
    if use_continuation
        # Currently disabled - just use zero initialization
        x = zeros(length(free_shock_idx))
    end
    
    # Lagrange multipliers for equality constraints
    λ = zeros(length(cond_var_idx))
    
    xλ = vcat(x, λ)
    Δxλ = copy(xλ)
    
    norm1 = ℒ.norm(conditions)
    norm2 = 1.0
    
    # Pre-allocate buffers
    residual = zeros(length(cond_var_idx))
    jacobian = zeros(length(cond_var_idx), length(free_shock_idx))
    fxλ = zeros(length(xλ))
    fxλp = zeros(length(xλ), length(xλ))
    
    lI = -2.0 * ℒ.I(length(free_shock_idx))
    
    # Buffers for analytical derivative computation  
    J = ℒ.Diagonal(ones(Bool, length(all_shocks)))
    kron_buffer = zeros(length(all_shocks) * length(all_shocks))
    kron_buffer2 = ℒ.kron(J, zeros(length(all_shocks)))  # Initialize with correct dimensions
    kron_buffer3 = ℒ.kron(J, kron_buffer)  # Initialize with correct dimensions for third-order
    ∂x = zero(𝐒¹ᵉ)
    
    # For globalization and Levenberg-Marquardt
    prev_merit = Inf
    if use_globalization
        xλ_temp = copy(xλ)
    end
    
    # Levenberg-Marquardt damping parameter
    # Start with moderate damping for underdetermined systems
    # Larger initial damping for better robustness
    μ = length(free_shock_idx) > length(cond_var_idx) ? 1.0 : 0.1
    ν = 3.0  # Scaling factor for damping updates
    
    # Debug flag
    debug = length(free_shock_idx) > length(cond_var_idx) && use_levenberg_marquardt
    
    @inbounds for iter in 1:max_iter
        if debug && iter <= 5
            println("  LM iter $iter: ||x|| = $(ℒ.norm(x)), μ = $μ")
        end
        
        # Update all shocks with current free shock values
        all_shocks[free_shock_idx] .= x
        
        # Compute new state
        new_state = state_update(initial_state, all_shocks)
        cond_vars = pruning ? sum(new_state) : new_state
        
        # Compute residual: target - actual
        residual .= conditions - cond_vars[cond_var_idx]
        
        # Compute Jacobian analytically using perturbation matrices
        # Following the same pattern as find_shocks
        # ∂y/∂ε = 𝐒¹ᵉ + 2*𝐒²ᵉ*kron(I, ε) + 3*𝐒³ᵉ*kron(I, kron(ε, ε))
        
        if !isnothing(𝐒³ᵉ)
            # Third-order: analytical Jacobian with cubic term
            # ∂x = 𝐒¹ᵉ + 2 * 𝐒²ᵉ * kron(I, all_shocks) + 3 * 𝐒³ᵉ * kron(I, kron(all_shocks, all_shocks))
            ℒ.kron!(kron_buffer, all_shocks, all_shocks)
            ℒ.kron!(kron_buffer2, J, all_shocks)
            ℒ.kron!(kron_buffer3, J, kron_buffer)
            
            copy!(∂x, 𝐒¹ᵉ)
            ℒ.mul!(∂x, 𝐒²ᵉ, kron_buffer2, 2, 1)
            ℒ.mul!(∂x, 𝐒³ᵉ, kron_buffer3, 3, 1)
            
            # Extract rows for conditioned variables and columns for free shocks
            jacobian .= -∂x[cond_var_idx, free_shock_idx]
        elseif !isnothing(𝐒²ᵉ)
            # Second-order: analytical Jacobian with quadratic term
            # ∂x = 𝐒¹ᵉ + 2 * 𝐒²ᵉ * kron(I, all_shocks)
            ℒ.kron!(kron_buffer2, J, all_shocks)
            ℒ.mul!(∂x, 𝐒²ᵉ, kron_buffer2)
            ℒ.axpby!(1, 𝐒¹ᵉ, 2, ∂x)
            
            # Extract rows for conditioned variables and columns for free shocks
            jacobian .= -∂x[cond_var_idx, free_shock_idx]
        else
            # First-order: just use 𝐒¹ᵉ
            jacobian .= -𝐒¹ᵉ[cond_var_idx, free_shock_idx]
        end
        
        # Build KKT system
        # First order optimality: gradient of Lagrangian wrt x
        fxλ[1:length(x)] .= 2.0 * x + jacobian' * λ
        
        # Equality constraints
        fxλ[length(x)+1:end] .= residual
        
        # Build Jacobian of KKT system
        fxλp[1:length(x), 1:length(x)] .= lI
        fxλp[1:length(x), length(x)+1:end] .= jacobian'
        fxλp[length(x)+1:end, 1:length(x)] .= jacobian
        fxλp[length(x)+1:end, length(x)+1:end] .= 0.0
        
        # Apply Levenberg-Marquardt damping if enabled
        if use_levenberg_marquardt
            # Add damping to the Hessian block: (H + μI)
            for i in 1:length(x)
                fxλp[i, i] -= 2.0 * μ  # Subtract 2μ because lI = -2I already
            end
        end
        
        # Solve Newton step
        try
            f̂xλp = ℒ.factorize(fxλp)
            ℒ.ldiv!(Δxλ, f̂xλp, fxλ)
        catch
            if use_levenberg_marquardt && μ < 1e10
                # Try with larger damping
                μ *= ν
                continue
            end
            return x, false
        end
        
        if !all(isfinite, Δxλ)
            break
        end
        
        # Update with Levenberg-Marquardt adaptive damping
        if use_levenberg_marquardt
            # Compute current cost: ||x||^2 + ||residual||^2
            current_cost = ℒ.dot(x, x) + ℒ.dot(residual, residual)
            
            # Try the step
            xλ_trial = xλ - Δxλ
            x_trial = xλ_trial[1:length(x)]
            λ_trial = xλ_trial[length(x)+1:end]
            
            # Compute actual reduction
            all_shocks[free_shock_idx] .= x_trial
            new_state_trial = state_update(initial_state, all_shocks)
            cond_vars_trial = pruning ? sum(new_state_trial) : new_state_trial
            residual_trial = conditions - cond_vars_trial[cond_var_idx]
            
            trial_cost = ℒ.dot(x_trial, x_trial) + ℒ.dot(residual_trial, residual_trial)
            actual_reduction = current_cost - trial_cost
            
            # Predicted reduction from linear model
            # For LM: F(x+h) ≈ F(x) + J*h + 0.5*h'*H*h where H includes damping
            # Here we use simplified predicted reduction
            predicted_reduction = -ℒ.dot(fxλ, Δxλ) - 0.5 * μ * ℒ.dot(Δxλ[1:length(x)], Δxλ[1:length(x)])
            
            # Compute gain ratio
            # Avoid division by very small numbers
            if abs(predicted_reduction) < 1e-20
                ρ = actual_reduction > 0 ? 1.0 : -1.0
            else
                ρ = actual_reduction / predicted_reduction
            end
            
            # More lenient acceptance criterion and better damping strategy
            if ρ > 0.0  # Accept any improvement
                xλ .= xλ_trial
                x .= x_trial
                λ .= λ_trial
                
                # Update damping parameter based on gain ratio
                if ρ > 0.75  # Very good agreement with model
                    μ = max(μ / ν, 1e-12)  # Reduce damping (getting closer to Newton)
                elseif ρ > 0.25  # Reasonable agreement
                    μ = max(μ / 2, 1e-12)  # Moderately reduce damping
                elseif ρ < 0.1  # Poor agreement  
                    μ = min(μ * ν, 1e8)    # Increase damping
                end
                # else: keep μ unchanged for moderate progress
                
            else  # Reject step, increase damping
                μ = min(μ * ν, 1e8)
                if μ > 1e6  # Damping too large, algorithm stuck
                    if debug
                        println("  LM stopped: damping too large (μ = $μ)")
                    end
                    break
                end
                continue  # Don't update x, λ, try again with larger damping
            end
            
        # Update with line search globalization if enabled
        elseif use_globalization
            # Try multiple merit function formulations
            # Merit 1: Standard L2 penalty ||x||^2 + penalty * ||residual||^2
            # Merit 2: L1 penalty ||x||_1 + penalty * ||residual||_1  (more robust to outliers)
            # Merit 3: Fletcher penalty with adaptive weight
            
            # Adaptive penalty based on problem conditioning
            # Higher penalty for underdetermined systems to emphasize constraint satisfaction
            base_penalty = length(free_shock_idx) > length(cond_var_idx) ? 500.0 : 100.0
            
            # Try different merit functions
            best_α = 0.0
            best_merit = Inf
            best_x = copy(x)
            
            for merit_type in [:l2_quadratic, :l1_robust, :adaptive_fletcher]
                # Compute current merit
                if merit_type == :l2_quadratic
                    penalty = base_penalty
                    current_merit = ℒ.dot(x, x) + penalty * ℒ.dot(residual, residual)
                elseif merit_type == :l1_robust
                    penalty = base_penalty
                    current_merit = ℒ.norm(x, 1) + penalty * ℒ.norm(residual, 1)
                else  # adaptive_fletcher
                    # Fletcher's merit: ||x||^2 + λ'*residual + 0.5*penalty*||residual||^2
                    penalty = base_penalty * (1.0 + iter / max_iter)  # Increase penalty over time
                    current_merit = ℒ.dot(x, x) + ℒ.dot(λ, residual) + 0.5 * penalty * ℒ.dot(residual, residual)
                end
                
                # Line search: try step sizes α = 1, 0.5, 0.25, 0.125, ...
                α = 1.0
                xλ_temp .= xλ
                
                for ls_iter in 1:12  # Try up to 12 backtracking steps
                    xλ_temp .= xλ - α * Δxλ
                    x_temp = xλ_temp[1:length(free_shock_idx)]
                    λ_temp = xλ_temp[length(free_shock_idx)+1:end]
                    
                    # Evaluate merit at trial point
                    all_shocks[free_shock_idx] .= x_temp
                    new_state_temp = state_update(initial_state, all_shocks)
                    cond_vars_temp = pruning ? sum(new_state_temp) : new_state_temp
                    residual_temp = conditions - cond_vars_temp[cond_var_idx]
                    
                    if merit_type == :l2_quadratic
                        trial_merit = ℒ.dot(x_temp, x_temp) + penalty * ℒ.dot(residual_temp, residual_temp)
                    elseif merit_type == :l1_robust
                        trial_merit = ℒ.norm(x_temp, 1) + penalty * ℒ.norm(residual_temp, 1)
                    else  # adaptive_fletcher
                        trial_merit = ℒ.dot(x_temp, x_temp) + ℒ.dot(λ_temp, residual_temp) + 0.5 * penalty * ℒ.dot(residual_temp, residual_temp)
                    end
                    
                    # Track best across all merit functions and step sizes
                    if trial_merit < best_merit
                        best_merit = trial_merit
                        best_α = α
                        best_x .= x_temp
                    end
                    
                    # Sufficient decrease condition (Armijo rule with adaptive c)
                    c = merit_type == :l1_robust ? 1e-3 : 1e-4  # More lenient for L1
                    if trial_merit < current_merit - c * α * ℒ.dot(Δxλ, Δxλ)
                        # Found acceptable step for this merit function
                        if best_α == 0.0 || α > best_α
                            best_α = α
                            best_x .= x_temp
                        end
                        break
                    end
                    
                    α *= 0.5
                end
            end
            
            # Use best step found across all merit functions
            if best_α > 0.0
                x .= best_x
                # Update full xλ vector
                xλ[1:length(free_shock_idx)] .= x
                # Recompute λ with current x
                all_shocks[free_shock_idx] .= x
                new_state = state_update(initial_state, all_shocks)
                cond_vars = pruning ? sum(new_state) : new_state
                residual .= conditions - cond_vars[cond_var_idx]
                # Don't update λ here - will be updated in next Newton step
            else
                # No improvement found with any merit function, take very small step
                xλ .-= 0.005 * Δxλ
                x .= xλ[1:length(free_shock_idx)]
            end
        else
            # Standard Newton update without globalization
            xλ .-= Δxλ
        end
        
        x .= xλ[1:length(free_shock_idx)]
        λ .= xλ[length(free_shock_idx)+1:end]
        
        # Check convergence
        norm2 = ℒ.norm(cond_vars[cond_var_idx])
        residual_norm = ℒ.norm(residual) / max(norm1, norm2)
        step_norm = ℒ.norm(Δxλ) / max(ℒ.norm(xλ), 1.0)
        
        if debug && iter <= 5
            println("    residual_norm = $residual_norm, step_norm = $step_norm, tol = $tol")
        end
        
        # For pure Newton: detect stalling
        if pure_newton
            improvement = last_residual_norm - residual_norm
            if improvement < tol * 0.01  # Not making meaningful progress
                stalled_count += 1
            else
                stalled_count = 0  # Reset if made progress
            end
            last_residual_norm = residual_norm
            
            # If stalled for 3 consecutive iterations, apply iterative refinement
            if stalled_count >= 3 || iter == max_iter
                if debug
                    if stalled_count >= 3
                        println("  Pure Newton stalled after $iter iterations")
                    else
                        println("  Pure Newton reached max_iter")
                    end
                    println("  Applying iterative refinement...")
                end
                
                # Apply iterative refinement
                x_refined, improved = iterative_refinement(
                    x, state_update, initial_state, all_shocks, conditions,
                    cond_var_idx, free_shock_idx, pruning,
                    𝐒¹ᵉ, 𝐒²ᵉ, 𝐒³ᵉ, T, jacobian, ∂x, kron_buffer2, kron_buffer3, J;
                    max_refine_iter=10, tol=tol, debug=debug)
                
                # Check if refinement achieved convergence
                all_shocks[free_shock_idx] .= x_refined
                new_state_final = state_update(initial_state, all_shocks)
                cond_vars_final = pruning ? sum(new_state_final) : new_state_final
                residual_final = conditions - cond_vars_final[cond_var_idx]
                norm2_final = ℒ.norm(cond_vars_final[cond_var_idx])
                residual_norm_final = ℒ.norm(residual_final) / max(norm1, norm2_final)
                
                converged_after_refinement = residual_norm_final < tol
                
                if converged_after_refinement && debug
                    println("  ✓ Converged after iterative refinement! (residual_norm = $residual_norm_final)")
                elseif improved && debug
                    println("  Iterative refinement improved solution (residual_norm: $residual_norm → $residual_norm_final)")
                end
                
                return x_refined, converged_after_refinement
            end
        end
        
        if residual_norm < tol && step_norm < sqrt(tol)
            if debug
                println("  Converged in $iter iterations!")
            end
            return x, true
        end
    end
    
    if debug
        println("  Did NOT converge after $max_iter iterations")
    end
    
    # For non-pure Newton methods, also try iterative refinement as final attempt
    if !pure_newton
        if debug
            println("  Attempting iterative refinement as final polish...")
        end
        
        x_refined, improved = iterative_refinement(
            x, state_update, initial_state, all_shocks, conditions,
            cond_var_idx, free_shock_idx, pruning,
            𝐒¹ᵉ, 𝐒²ᵉ, 𝐒³ᵉ, T, jacobian, ∂x, kron_buffer2, kron_buffer3, J;
            max_refine_iter=10, tol=tol, debug=debug)
        
        if improved && debug
            println("  Iterative refinement improved solution")
        end
        
        # Check if refinement achieved convergence
        all_shocks[free_shock_idx] .= x_refined
        new_state_final = state_update(initial_state, all_shocks)
        cond_vars_final = pruning ? sum(new_state_final) : new_state_final
        residual_final = conditions - cond_vars_final[cond_var_idx]
        norm2_final = ℒ.norm(cond_vars_final[cond_var_idx])
        residual_norm_final = ℒ.norm(residual_final) / max(norm1, norm2_final)
        
        converged_after_refinement = residual_norm_final < tol
        
        if converged_after_refinement && debug
            println("  Converged after iterative refinement!")
        end
        
        return x_refined, converged_after_refinement
    end
    
    return x, false
end
end # dispatch_doctor


# Iterative refinement: polish the solution by solving for the residual
# This can improve precision when the main algorithm has stalled
@stable default_mode = "disable" begin
function iterative_refinement(
    x::Vector{Float64},
    state_update::Function,
    initial_state::Union{Vector{Float64}, Vector{Vector{Float64}}},
    all_shocks::Vector{Float64},
    conditions::Vector{Float64},
    cond_var_idx::Vector{Int},
    free_shock_idx::Vector{Int},
    pruning::Bool,
    𝐒¹ᵉ::AbstractMatrix{Float64},
    𝐒²ᵉ::Union{AbstractMatrix{Float64}, Nothing},
    𝐒³ᵉ::Union{AbstractMatrix{Float64}, Nothing},
    T::timings,
    jacobian::Matrix{Float64},
    ∂x::Matrix{Float64},
    kron_buffer2::Matrix{Float64},
    kron_buffer3::Matrix{Float64},
    J::ℒ.Diagonal{Bool, Vector{Bool}};
    max_refine_iter::Int = 5,
    tol::Float64 = 1e-13,
    debug::Bool = false)
    
    x_current = copy(x)
    best_residual_norm = Inf
    improved = false
    
    # Compute initial residual
    all_shocks[free_shock_idx] .= x_current
    new_state = state_update(initial_state, all_shocks)
    cond_vars = pruning ? sum(new_state) : new_state
    residual = conditions - cond_vars[cond_var_idx]
    norm1 = ℒ.norm(conditions)
    norm2 = ℒ.norm(cond_vars[cond_var_idx])
    initial_residual_norm = ℒ.norm(residual) / max(norm1, norm2)
    best_residual_norm = initial_residual_norm
    
    if debug
        println("  Iterative refinement starting with residual_norm = $initial_residual_norm")
    end
    
    # Iterative refinement loop
    for refine_iter in 1:max_refine_iter
        # Compute Jacobian at current point
        if !isnothing(𝐒³ᵉ)
            # Third-order
            kron_buffer = zeros(length(all_shocks) * length(all_shocks))
            ℒ.kron!(kron_buffer, all_shocks, all_shocks)
            ℒ.kron!(kron_buffer2, J, all_shocks)
            ℒ.kron!(kron_buffer3, J, kron_buffer)
            
            copy!(∂x, 𝐒¹ᵉ)
            ℒ.mul!(∂x, 𝐒²ᵉ, kron_buffer2, 2, 1)
            ℒ.mul!(∂x, 𝐒³ᵉ, kron_buffer3, 3, 1)
            
            jacobian .= -∂x[cond_var_idx, free_shock_idx]
        elseif !isnothing(𝐒²ᵉ)
            # Second-order
            ℒ.kron!(kron_buffer2, J, all_shocks)
            ℒ.mul!(∂x, 𝐒²ᵉ, kron_buffer2)
            ℒ.axpby!(1, 𝐒¹ᵉ, 2, ∂x)
            
            jacobian .= -∂x[cond_var_idx, free_shock_idx]
        else
            # First-order
            jacobian .= -𝐒¹ᵉ[cond_var_idx, free_shock_idx]
        end
        
        # Solve for correction: J * δx = residual
        # Use least-squares for robustness
        δx = try
            # Try direct solve first
            jacobian \ residual
        catch
            # Fall back to pseudoinverse if singular
            ℒ.pinv(jacobian) * residual
        end
        
        # Apply damped correction to avoid overshooting
        # Start with full step, reduce if it doesn't improve
        accepted_damping = 0.0
        for damping_factor in [1.0, 0.5, 0.25, 0.1]
            x_trial = x_current + damping_factor * δx
            
            # Clamp to reasonable bounds
            x_trial .= clamp.(x_trial, -10.0, 10.0)
            
            # Evaluate residual at trial point
            all_shocks[free_shock_idx] .= x_trial
            new_state_trial = state_update(initial_state, all_shocks)
            cond_vars_trial = pruning ? sum(new_state_trial) : new_state_trial
            residual_trial = conditions - cond_vars_trial[cond_var_idx]
            
            norm2_trial = ℒ.norm(cond_vars_trial[cond_var_idx])
            residual_norm_trial = ℒ.norm(residual_trial) / max(norm1, norm2_trial)
            
            # Accept if improved
            if residual_norm_trial < best_residual_norm
                x_current .= x_trial
                residual .= residual_trial
                cond_vars .= cond_vars_trial
                best_residual_norm = residual_norm_trial
                improved = true
                accepted_damping = damping_factor
                break
            end
        end
        
        if debug && accepted_damping > 0
            println("    Refine iter $refine_iter: residual_norm = $best_residual_norm (damping = $accepted_damping)")
        end
        
        # Check if we've achieved target tolerance
        if best_residual_norm < tol
            if debug
                println("  Iterative refinement converged to target tolerance!")
            end
            break
        end
        
        # Check if making progress
        if refine_iter > 1 && best_residual_norm > 0.99 * initial_residual_norm
            # Not making meaningful progress, stop
            if debug
                println("  Iterative refinement stopped (no progress)")
            end
            break
        end
        
        # No accepted step, stop
        if accepted_damping == 0.0
            if debug
                println("  Iterative refinement stopped (no acceptable step)")
            end
            break
        end
    end
    
    if debug && improved
        improvement_factor = initial_residual_norm / best_residual_norm
        println("  Iterative refinement improved residual by factor of $improvement_factor")
    end
    
    return x_current, improved
end
end # dispatch_doctor


@stable default_mode = "disable" begin
function find_shocks(::Val{:LagrangeNewton},
                    initial_guess::Vector{Float64},
                    kron_buffer::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    J::ℒ.Diagonal{Bool, Vector{Bool}},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    max_iter::Int = 1000,
                    tol::Float64 = 1e-13) # will fail for higher or lower precision
    x = copy(initial_guess)
    
    λ = zeros(size(𝐒ⁱ, 1))
    
    xλ = [  x
            λ   ]

    Δxλ = copy(xλ)

    norm1 = ℒ.norm(shock_independent) 

    norm2 = 1.0
    
    Δnorm = 1e12

    x̂ = copy(shock_independent)

    x̄ = zeros(size(𝐒ⁱ,2))

    ∂x = zero(𝐒ⁱ)
    
    fxλ = zeros(length(xλ))
    
    fxλp = zeros(length(xλ), length(xλ))

    tmp = zeros(size(𝐒ⁱ, 2) * size(𝐒ⁱ, 2))

    lI = -2 * vec(ℒ.I(size(𝐒ⁱ, 2)))

    @inbounds for i in 1:max_iter
        ℒ.kron!(kron_buffer2, J, x)

        ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2)
        ℒ.axpby!(1, 𝐒ⁱ, 2, ∂x)

        ℒ.mul!(x̄, ∂x', λ)
        
        ℒ.axpy!(-2, x, x̄)

        copyto!(fxλ, 1, x̄, 1, size(𝐒ⁱ,2))
        copyto!(fxλ, size(𝐒ⁱ,2) + 1, x̂, 1, size(shock_independent,1))
        
        # fXλ = [(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * λ - 2 * x
                # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x))]

        ℒ.mul!(tmp, 𝐒ⁱ²ᵉ', λ)
        ℒ.axpby!(1, lI, 2, tmp)

        fxλp[1:size(𝐒ⁱ, 2), 1:size(𝐒ⁱ, 2)] = tmp
        fxλp[1:size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)+1:end] = ∂x'

        ℒ.rmul!(∂x, -1)
        fxλp[size(𝐒ⁱ, 2)+1:end, 1:size(𝐒ⁱ, 2)] = ∂x

        # fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2))  (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))'
        #         -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
        
        # f̂xλp = ℒ.lu(fxλp, check = false)

        # if !ℒ.issuccess(f̂xλp)
        #     return x, false
        # end

        try
            f̂xλp = ℒ.factorize(fxλp)
            ℒ.ldiv!(Δxλ, f̂xλp, fxλ)
        catch
            # ℒ.svd(fxλp)
            # println("factorization fails")
            return x, false
        end
        
        if !all(isfinite,Δxλ) break end
        
        ℒ.axpy!(-1, Δxλ, xλ)
        # xλ -= Δxλ
    
        # x = xλ[1:size(𝐒ⁱ, 2)]
        copyto!(x, 1, xλ, 1, size(𝐒ⁱ,2))

        # λ = xλ[size(𝐒ⁱ, 2)+1:end]
        copyto!(λ, 1, xλ, size(𝐒ⁱ,2) + 1, length(λ))

        ℒ.kron!(kron_buffer, x, x)

        ℒ.mul!(x̂, 𝐒ⁱ²ᵉ, kron_buffer)

        ℒ.mul!(x̂, 𝐒ⁱ, x, 1, 1)

        norm2 = ℒ.norm(x̂)

        ℒ.axpby!(1, shock_independent, -1, x̂)

        if ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol)
            # println("LagrangeNewton: $i, Tol reached, $x")
            break
        end

        # if i > 500 && ℒ.norm(Δxλ) > 1e-11 && ℒ.norm(Δxλ) > Δnorm
        #     # println("LagrangeNewton: $i, Norm increase")
        #     return x, false
        # end
        # # if i == max_iter
        #     println("LagrangeNewton: $i, Max iter reached")
            # println(ℒ.norm(Δxλ) / ℒ.norm(xλ))
        # end
    end

    # println(λ)
    # println("Norm: $(ℒ.norm(x̂) / max(norm1,norm2))")
    # println(ℒ.norm(Δxλ))
    # println(ℒ.norm(Δxλ) / ℒ.norm(xλ))
    # if !(ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol))
    #     println("Find shocks failed. Norm 1: $(ℒ.norm(x̂) / max(norm1,norm2)); Norm 2: $(ℒ.norm(Δxλ) / ℒ.norm(xλ))")
    # end

    return x, ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol)
end

end # dispatch_doctor

function rrule(::typeof(find_shocks),
                ::Val{:LagrangeNewton},
                initial_guess::Vector{Float64},
                kron_buffer::Vector{Float64},
                kron_buffer2::AbstractMatrix{Float64},
                J::ℒ.Diagonal{Bool, Vector{Bool}},
                𝐒ⁱ::AbstractMatrix{Float64},
                𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                shock_independent::Vector{Float64};
                max_iter::Int = 1000,
                tol::Float64 = 1e-13)

    x, matched = find_shocks(Val(:LagrangeNewton),
                            initial_guess,
                            kron_buffer,
                            kron_buffer2,
                            J,
                            𝐒ⁱ,
                            𝐒ⁱ²ᵉ,
                            shock_independent,
                            max_iter = max_iter,
                            tol = tol)

    tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x)

    λ = tmp' \ x * 2

    fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  tmp'
    -tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]

    ℒ.kron!(kron_buffer, x, x)

    xλ = ℒ.kron(x,λ)


    ∂shock_independent = similar(shock_independent)

    # ∂𝐒ⁱ = similar(𝐒ⁱ)

    # ∂𝐒ⁱ²ᵉ = similar(𝐒ⁱ²ᵉ)

    function find_shocks_pullback(∂x)
        ∂x = vcat(∂x[1], zero(λ))

        S = -fXλp' \ ∂x

        copyto!(∂shock_independent, S[length(initial_guess)+1:end])
        
        # copyto!(∂𝐒ⁱ, ℒ.kron(S[1:length(initial_guess)], λ) - ℒ.kron(x, S[length(initial_guess)+1:end]))
        ∂𝐒ⁱ = S[1:length(initial_guess)] * λ' - S[length(initial_guess)+1:end] * x'
        
        # copyto!(∂𝐒ⁱ²ᵉ, 2 * ℒ.kron(S[1:length(initial_guess)], xλ) - ℒ.kron(kron_buffer, S[length(initial_guess)+1:end]))
        ∂𝐒ⁱ²ᵉ = 2 * S[1:length(initial_guess)] * xλ' - S[length(initial_guess)+1:end] * kron_buffer'

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂𝐒ⁱ, ∂𝐒ⁱ²ᵉ, ∂shock_independent, NoTangent(), NoTangent()
    end

    return (x, matched), find_shocks_pullback
end


@stable default_mode = "disable" begin

function find_shocks(::Val{:LagrangeNewton},
                    initial_guess::Vector{Float64},
                    kron_buffer::Vector{Float64},
                    kron_buffer²::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    kron_buffer3::AbstractMatrix{Float64},
                    kron_buffer4::AbstractMatrix{Float64},
                    J::ℒ.Diagonal{Bool, Vector{Bool}},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    max_iter::Int = 1000,
                    tol::Float64 = 1e-13) # will fail for higher or lower precision
    x = copy(initial_guess)

    λ = zeros(size(𝐒ⁱ, 1))
    
    xλ = [  x
            λ   ]

    Δxλ = copy(xλ)

    norm1 = ℒ.norm(shock_independent) 

    norm2 = 1.0
    
    Δnorm = 1e12

    x̂ = copy(shock_independent)

    x̄ = zeros(size(𝐒ⁱ,2))

    ∂x = zero(𝐒ⁱ)

    ∂x̂ = zero(𝐒ⁱ)
    
    fxλ = zeros(length(xλ))
    
    fxλp = zeros(length(xλ), length(xλ))

    tmp = zeros(size(𝐒ⁱ, 2) * size(𝐒ⁱ, 2))

    tmp2 = zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 2) * size(𝐒ⁱ, 2))

    II = sparse(ℒ.I(length(x)^2))

    lI = -2 * vec(ℒ.I(size(𝐒ⁱ, 2)))
    
    @inbounds for i in 1:max_iter
        ℒ.kron!(kron_buffer2, J, x)
        ℒ.kron!(kron_buffer3, J, kron_buffer)

        copy!(∂x, 𝐒ⁱ)
        ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2, 2, 1)

        ℒ.mul!(∂x, 𝐒ⁱ³ᵉ, kron_buffer3, 3, 1)

        ℒ.mul!(x̄, ∂x', λ)
        
        ℒ.axpy!(-2, x, x̄)

        copyto!(fxλ, 1, x̄, 1, size(𝐒ⁱ,2))
        copyto!(fxλ, size(𝐒ⁱ,2) + 1, x̂, 1, size(shock_independent,1))
        # fXλ = [(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
                # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]
        
        x_kron_II!(kron_buffer4, x)
        # ℒ.kron!(kron_buffer4, II, x)
        ℒ.mul!(tmp2, 𝐒ⁱ³ᵉ, kron_buffer4)
        ℒ.mul!(tmp, tmp2', λ)
        ℒ.mul!(tmp, 𝐒ⁱ²ᵉ', λ, 2, 6)
        ℒ.axpy!(1,lI,tmp)

        fxλp[1:size(𝐒ⁱ, 2), 1:size(𝐒ⁱ, 2)] = tmp
        
        fxλp[1:size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)+1:end] = ∂x'

        ℒ.rmul!(∂x, -1)
        fxλp[size(𝐒ⁱ, 2)+1:end, 1:size(𝐒ⁱ, 2)] = ∂x
        # fXλp = [reshape((2 * 𝐒ⁱ²ᵉ + 6 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(ℒ.I(length(x)),x)))' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2))  (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))'
        #         -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
        
        try
            f̂xλp = ℒ.factorize(fxλp)
            ℒ.ldiv!(Δxλ, f̂xλp, fxλ)
        catch
            # ℒ.svd(fxλp)
            # println("factorization fails")
            return x, false
        end
        
        if !all(isfinite,Δxλ) break end
        
        ℒ.axpy!(-1, Δxλ, xλ)
        # xλ -= Δxλ
    
        # x = xλ[1:size(𝐒ⁱ, 2)]
        copyto!(x, 1, xλ, 1, size(𝐒ⁱ,2))

        # λ = xλ[size(𝐒ⁱ, 2)+1:end]
        copyto!(λ, 1, xλ, size(𝐒ⁱ,2) + 1, length(λ))

        ℒ.kron!(kron_buffer, x, x)

        ℒ.kron!(kron_buffer², x, kron_buffer)

        ℒ.mul!(x̂, 𝐒ⁱ, x)

        ℒ.mul!(x̂, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)

        ℒ.mul!(x̂, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)

        norm2 = ℒ.norm(x̂)

        ℒ.axpby!(1, shock_independent, -1, x̂)

        if ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol)
            # println("LagrangeNewton: $i, Tol: $(ℒ.norm(Δxλ) / ℒ.norm(xλ)) reached, x: $x")
            break
        end

        # if i > 500 && ℒ.norm(Δxλ) > 1e-11 && ℒ.norm(Δxλ) > Δnorm
        #     # println(ℒ.norm(Δxλ))
        #     # println(ℒ.norm(x̂) / max(norm1,norm2))
        #     # println("LagrangeNewton: $i, Norm increase")
        #     return x, false
        # end
        # if i == max_iter
        #     println("LagrangeNewton: $i, Max iter reached")
        #     # println(ℒ.norm(Δxλ))
        # end
    end

    # λ = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer))' \ x * 2
    # println("LagrangeNewton: $(ℒ.norm([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
    # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]))")

    # println(ℒ.norm(x))
    # println(x)
    # println(λ)
    # println([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
    # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))])
    # println(fxλp)
    # println(reshape(tmp, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2)))
    # println([reshape((2 * 𝐒ⁱ²ᵉ - 2 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(ℒ.I(length(x)),x)))' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2))  (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))'
    #         -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))])
    # println(fxλp)
    # println("Norm: $(ℒ.norm(x̂) / max(norm1,norm2))")
    # println(ℒ.norm(Δxλ))
    # println(ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < tol)

    # if !(ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol))
    #     println("Find shocks failed. Norm 1: $(ℒ.norm(x̂) / max(norm1,norm2)); Norm 2: $(ℒ.norm(Δxλ) / ℒ.norm(xλ))")
    # end

    return x, ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol)
end


end # dispatch_doctor


function rrule(::typeof(find_shocks),
                ::Val{:LagrangeNewton},
                initial_guess::Vector{Float64},
                kron_buffer::Vector{Float64},
                kron_buffer²::Vector{Float64},
                kron_buffer2::AbstractMatrix{Float64},
                kron_buffer3::AbstractMatrix{Float64},
                kron_buffer4::AbstractMatrix{Float64},
                J::ℒ.Diagonal{Bool, Vector{Bool}},
                𝐒ⁱ::AbstractMatrix{Float64},
                𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                shock_independent::Vector{Float64};
                max_iter::Int = 1000,
                tol::Float64 = 1e-13)

    x, matched = find_shocks(Val(:LagrangeNewton),
                            initial_guess,
                            kron_buffer,
                            kron_buffer²,
                            kron_buffer2,
                            kron_buffer3,
                            kron_buffer4,
                            J,
                            𝐒ⁱ,
                            𝐒ⁱ²ᵉ,
                            𝐒ⁱ³ᵉ,
                            shock_independent,
                            max_iter = max_iter,
                            tol = tol)

    ℒ.kron!(kron_buffer, x, x)

    ℒ.kron!(kron_buffer², x, kron_buffer)

    tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer)

    λ = tmp' \ x * 2

    fXλp = [reshape((2 * 𝐒ⁱ²ᵉ + 6 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(ℒ.I(length(x)),x)))' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  tmp'
    -tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]

    xλ = ℒ.kron(x,λ)

    xxλ = ℒ.kron(x,xλ)

    function find_shocks_pullback(∂x)
        ∂x = vcat(∂x[1], zero(λ))

        S = -fXλp' \ ∂x

        ∂shock_independent = S[length(initial_guess)+1:end]
        
        ∂𝐒ⁱ = ℒ.kron(S[1:length(initial_guess)], λ) - ℒ.kron(x, S[length(initial_guess)+1:end])

        ∂𝐒ⁱ²ᵉ = 2 * ℒ.kron(S[1:length(initial_guess)], xλ) - ℒ.kron(kron_buffer, S[length(initial_guess)+1:end])
        
        ∂𝐒ⁱ³ᵉ = 3 * ℒ.kron(S[1:length(initial_guess)], xxλ) - ℒ.kron(kron_buffer²,S[length(initial_guess)+1:end])

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),  ∂𝐒ⁱ, ∂𝐒ⁱ²ᵉ, ∂𝐒ⁱ³ᵉ, ∂shock_independent, NoTangent(), NoTangent()
    end

    return (x, matched), find_shocks_pullback
end



# @stable default_mode = "disable" begin

# function find_shocks(::Val{:SLSQP},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 500,
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision
#     function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
#         if length(grad) > 0
#             copy!(grad, X)

#             ℒ.rmul!(grad, 2)
#             # grad .= 2 .* X
#         end
        
#         sum(abs2, X)
#     end

#     function constraint_optim(res::Vector{S}, x::Vector{S}, jac::Matrix{S}) where S <: Float64
#         if length(jac) > 0
#             ℒ.kron!(kron_buffer2, J, x)

#             copy!(jac', 𝐒ⁱ)

#             ℒ.mul!(jac', 𝐒ⁱ²ᵉ, kron_buffer2, -2, -1)
#             # jac .= -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))'
#         end

#         ℒ.kron!(kron_buffer, x, x)

#         ℒ.mul!(res, 𝐒ⁱ, x)

#         ℒ.mul!(res, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)

#         ℒ.axpby!(1, shock_independent, -1, res)
#         # res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * ℒ.kron(X,X)
#     end
    
#     opt = NLopt.Opt(NLopt.:LD_SLSQP, size(𝐒ⁱ,2))
                    
#     opt.min_objective = objective_optim_fun

#     # opt.xtol_abs = eps()
#     # opt.ftol_abs = eps()
#     opt.maxeval = max_iter

#     NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

#     (minf,x,ret) = try 
#         NLopt.optimize(opt, initial_guess)
#     catch
#         return initial_guess, false
#     end

#     ℒ.kron!(kron_buffer, x, x)

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron_buffer

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     solved = ret ∈ Symbol.([
#         # NLopt.MAXEVAL_REACHED,
#         NLopt.SUCCESS,
#         NLopt.STOPVAL_REACHED,
#         NLopt.FTOL_REACHED,
#         NLopt.XTOL_REACHED,
#         NLopt.ROUNDOFF_LIMITED,
#     ])

#     # println(ℒ.norm(x))
#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
# end



# function find_shocks(::Val{:SLSQP},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     kron_buffer4::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 500,
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision
#     function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
#         if length(grad) > 0
#             copy!(grad, X)

#             ℒ.rmul!(grad, 2)
#             # grad .= 2 .* X
#         end
        
#         sum(abs2, X)
#     end

#     function constraint_optim(res::Vector{S}, x::Vector{S}, jac::Matrix{S}) where S <: Float64
#         ℒ.kron!(kron_buffer, x, x)

#         ℒ.kron!(kron_buffer², x, kron_buffer)

#         if length(jac) > 0
#             ℒ.kron!(kron_buffer2, J, x)

#             ℒ.kron!(kron_buffer3, J, kron_buffer)

#             copy!(jac', 𝐒ⁱ)

#             ℒ.mul!(jac', 𝐒ⁱ²ᵉ, kron_buffer2, 2, 1)

#             ℒ.mul!(jac', 𝐒ⁱ³ᵉ, kron_buffer3, -3, -1)
#             # jac .= -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(J, x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(J, ℒ.kron(x,x)))'
#         end

#         ℒ.mul!(res, 𝐒ⁱ, x)

#         ℒ.mul!(res, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)

#         ℒ.mul!(res, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)

#         ℒ.axpby!(1, shock_independent, -1, res)
#         # res .= shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * ℒ.kron!(kron_buffer, x, x) - 𝐒ⁱ³ᵉ * ℒ.kron!(kron_buffer², x, kron_buffer)
#     end
    
#     opt = NLopt.Opt(NLopt.:LD_SLSQP, size(𝐒ⁱ,2))
                    
#     opt.min_objective = objective_optim_fun

#     # opt.xtol_abs = eps()
#     # opt.ftol_abs = eps()
#     # opt.constrtol_abs = eps() # doesn't work
#     # opt.xtol_rel = eps()
#     # opt.ftol_rel = eps()
#     opt.maxeval = max_iter

#     NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

#     (minf,x,ret) = try 
#         NLopt.optimize(opt, initial_guess)
#     catch
#         return initial_guess, false
#     end

#     # println("SLSQP - retcode: $ret, nevals: $(opt.numevals)")

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     solved = ret ∈ Symbol.([
#         # NLopt.MAXEVAL_REACHED,
#         NLopt.SUCCESS,
#         NLopt.STOPVAL_REACHED,
#         NLopt.FTOL_REACHED,
#         NLopt.XTOL_REACHED,
#         NLopt.ROUNDOFF_LIMITED,
#     ])

#     # println(ℒ.norm(x))
#     # λ = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer))' \ x * 2
#     # println("SLSQP - $ret: $(ℒ.norm([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
#     # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]))")
#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
# end






# function find_shocks(::Val{:COBYLA},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 10000,
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision
#     function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
#         sum(abs2, X)
#     end

#     function constraint_optim(res::Vector{S}, x::Vector{S}, jac::Matrix{S}) where S <: Float64
#         ℒ.kron!(kron_buffer, x, x)

#         ℒ.mul!(res, 𝐒ⁱ, x)

#         ℒ.mul!(res, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)

#         ℒ.axpby!(1, shock_independent, -1, res)
#         # res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * ℒ.kron(X,X)
#     end

#     opt = NLopt.Opt(NLopt.:LN_COBYLA, size(𝐒ⁱ,2))
                    
#     opt.min_objective = objective_optim_fun

#     # opt.xtol_abs = eps()
#     # opt.ftol_abs = eps()
#     # opt.xtol_rel = eps()
#     # opt.ftol_rel = eps()
#     # opt.constrtol_abs = eps() # doesn't work
#     opt.maxeval = max_iter

#     NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

#     (minf,x,ret) = NLopt.optimize(opt, initial_guess)

#     # println("COBYLA - retcode: $ret, nevals: $(opt.numevals)")

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x)

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     solved = ret ∈ Symbol.([
#         NLopt.SUCCESS,
#         NLopt.STOPVAL_REACHED,
#         NLopt.FTOL_REACHED,
#         NLopt.XTOL_REACHED,
#         NLopt.ROUNDOFF_LIMITED,
#     ])

#     # println("COBYLA: $(opt.numevals)")

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
# end



# function find_shocks(::Val{:COBYLA},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     kron_buffer4::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 10000,
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision
#     function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
#         sum(abs2, X)
#     end

#     function constraint_optim(res::Vector{S}, x::Vector{S}, jac::Matrix{S}) where S <: Float64
#         ℒ.kron!(kron_buffer, x, x)

#         ℒ.kron!(kron_buffer², x, kron_buffer)

#         ℒ.mul!(res, 𝐒ⁱ, x)

#         ℒ.mul!(res, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)

#         ℒ.mul!(res, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)

#         ℒ.axpby!(1, shock_independent, -1, res)
#         # res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * ℒ.kron(X,X) - 𝐒ⁱ³ᵉ * ℒ.kron(X, ℒ.kron(X,X))
#     end

#     opt = NLopt.Opt(NLopt.:LN_COBYLA, size(𝐒ⁱ,2))
                    
#     opt.min_objective = objective_optim_fun

#     # opt.xtol_abs = eps()
#     # opt.ftol_abs = eps()
#     # opt.xtol_rel = eps()
#     # opt.ftol_rel = eps()
#     # opt.constrtol_abs = eps() # doesn't work
#     opt.maxeval = max_iter

#     NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

#     (minf,x,ret) = NLopt.optimize(opt, initial_guess)

#     # println("COBYLA - retcode: $ret, nevals: $(opt.numevals)")

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     solved = ret ∈ Symbol.([
#         NLopt.SUCCESS,
#         NLopt.STOPVAL_REACHED,
#         NLopt.FTOL_REACHED,
#         NLopt.XTOL_REACHED,
#         NLopt.ROUNDOFF_LIMITED,
#     ])
#     # println(ℒ.norm(x))
#     # println(x)
#     # λ = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer))' \ x * 2
#     # println("COBYLA: $(ℒ.norm([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
#     # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]))")
#     # println("COBYLA: $(opt.numevals)")
#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")

#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
# end

# end # dispatch_doctor





# function find_shocks(::Val{:MadNLP},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     kron_buffer4::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 500,
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision
#     model = JuMP.Model(MadNLP.Optimizer)

#     JuMP.set_silent(model)

#     JuMP.set_optimizer_attribute(model, "tol", tol)

#     JuMP.@variable(model, x[1:length(initial_guess)])

#     JuMP.set_start_value.(x, initial_guess)

#     JuMP.@objective(model, Min, sum(abs2,x))

#     JuMP.@constraint(model, 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)) .== shock_independent)

#     JuMP.optimize!(model)

#     x = JuMP.value.(x)

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println(ℒ.norm(y - shock_independent) / max(norm1,norm2))
#     # λ = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer))' \ x * 2
#     # println("SLSQP - $ret: $(ℒ.norm([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
#     # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]))")
#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end




# function find_shocks(::Val{:Ipopt},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     kron_buffer4::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 500,
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision
#     model = JuMP.Model(Ipopt.Optimizer)

#     JuMP.set_silent(model)

#     JuMP.set_optimizer_attribute(model, "tol", tol)

#     JuMP.@variable(model, x[1:length(initial_guess)])

#     JuMP.set_start_value.(x, initial_guess)

#     JuMP.@objective(model, Min, sum(abs2,x))

#     JuMP.@constraint(model, 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)) .== shock_independent)

#     JuMP.optimize!(model)

#     x = JuMP.value.(x)

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println(ℒ.norm(y - shock_independent) / max(norm1,norm2))
#     # λ = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer))' \ x * 2
#     # println("SLSQP - $ret: $(ℒ.norm([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
#     # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]))")
#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end


# function find_shocks(::Val{:newton},
#     kron_buffer::Vector{Float64},
#     kron_buffer2::AbstractMatrix{Float64},
#     J::ℒ.Diagonal{Bool, Vector{Bool}},
#     𝐒ⁱ::AbstractMatrix{Float64},
#     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#     shock_independent::Vector{Float64};
#     tol::Float64 = 1e-13) # will fail for higher or lower precision

#     nExo = Int(sqrt(length(kron_buffer)))

#     x = zeros(nExo)

#     x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

#     x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

#     x̄ = zeros(size(𝐒ⁱ²ᵉ,1))

#     Δx = zeros(nExo)

#     ∂x = zero(𝐒ⁱ)

#     Ĵ = ℒ.I(nExo)*2

#     max_iter = 1000

# 	norm1 = 1

# 	norm2 = ℒ.norm(shock_independent)

#     for i in 1:max_iter
#         ℒ.kron!(kron_buffer, x, x)
#         ℒ.kron!(kron_buffer2, Ĵ, x)
        
#         ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2)
#         ℒ.axpy!(1, 𝐒ⁱ, ∂x)
#         # ∂x = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(nExo), x))

#         ∂x̂ = try 
#             ℒ.factorize(∂x)
#         catch
#             return x, false
#         end 

#         ℒ.mul!(x̂, 𝐒ⁱ²ᵉ, kron_buffer)
#         ℒ.mul!(x̄, 𝐒ⁱ, x)
#         ℒ.axpy!(1, x̄, x̂)
# 				norm1 = ℒ.norm(x̂)
#         ℒ.axpby!(1, shock_independent, -1, x̂)
#         try 
#             ℒ.ldiv!(Δx, ∂x̂, x̂)
#         catch
#             return x, false
#         end
#         # Δx = ∂x̂ \ (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron_buffer)
#         # println(ℒ.norm(Δx))
#         if i > 6 && (ℒ.norm(x̂) / max(norm1,norm2) < tol)
#             # println(i)
#             break
#         end
        
#         ℒ.axpy!(1, Δx, x)
#         # x += Δx

#         if !all(isfinite.(x))
#             return x, false
#         end
#     end

#     return x, ℒ.norm(x̂) / max(norm1,norm2) < tol
# end



# function find_shocks(::Val{:newton},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision

#     nExo = Int(sqrt(length(kron_buffer)))

#     x = zeros(nExo)

#     x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

#     x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

#     x̄ = zeros(size(𝐒ⁱ²ᵉ,1))

#     Δx = zeros(nExo)

#     ∂x = zero(𝐒ⁱ)

#     Ĵ = ℒ.I(nExo)*2

#     max_iter = 1000

#     norm1 = 1

# 	norm2 = ℒ.norm(shock_independent)

#     for i in 1:max_iter
#         ℒ.kron!(kron_buffer, x, x)
#         ℒ.kron!(kron_buffer², x, kron_buffer)
#         ℒ.kron!(kron_buffer2, Ĵ, x)
#         ℒ.kron!(kron_buffer3, Ĵ, kron_buffer)
        
#         ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2)
#         ℒ.mul!(∂x, 𝐒ⁱ³ᵉ, kron_buffer3, 1 ,1)
#         ℒ.axpy!(1, 𝐒ⁱ, ∂x)
#         # ∂x = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(nExo), x) + 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(nExo), ℒ.kron(x,x)))

#         ∂x̂ = try 
#             ℒ.factorize(∂x)
#         catch
#             return x, false
#         end 
							
#         ℒ.mul!(x̂, 𝐒ⁱ²ᵉ, kron_buffer)
#         ℒ.mul!(x̂, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)
#         ℒ.mul!(x̄, 𝐒ⁱ, x)
#         ℒ.axpy!(1, x̄, x̂)
# 				norm1 = ℒ.norm(x̂)
#         ℒ.axpby!(1, shock_independent, -1, x̂)
#         try 
#             ℒ.ldiv!(Δx, ∂x̂, x̂)
#         catch
#             return x, false
#         end
#         # Δx = ∂x̂ \ (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron_buffer)
#         # println(ℒ.norm(Δx))
#         if i > 6 && (ℒ.norm(x̂) / max(norm1,norm2)) < tol
#             # println("Iters: $i Norm: $(ℒ.norm(x̂) / max(norm1,norm2))")
#             break
#         end
        
#         ℒ.axpy!(1, Δx, x)
#         # x += Δx

#         if !all(isfinite.(x))
#             return x, false
#         end
#     end

#     # println("Iters: $max_iter Norm: $(ℒ.norm(x̂) / max(norm1,norm2))")
#     return x, ℒ.norm(x̂) / max(norm1,norm2) < tol
# end



# function find_shocks(::Val{:LBFGS},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-15) # will fail for higher or lower precision

#     function optim_fun(x::Vector{S}, grad::Vector{S}) where S <: Float64
#         if length(grad) > 0
#             grad .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
#         end

#         return sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
#     end
    

#     opt = NLopt.Opt(NLopt.:LD_LBFGS, size(𝐒ⁱ,2))
                    
#     opt.min_objective = optim_fun

#     opt.xtol_abs = eps()
#     opt.ftol_abs = eps()
#     opt.maxeval = 10000

#     (minf,x,ret) = NLopt.optimize(opt, zeros(size(𝐒ⁱ,2)))

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x)

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end




# function find_shocks(::Val{:LBFGS},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = eps()) # will fail for higher or lower precision

#     function optim_fun(x::Vector{S}, grad::Vector{S}) where S <: Float64
#         if length(grad) > 0
#             grad .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * kron(ℒ.I(length(x)),kron(x,x)))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
#         end

#         return sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
#     end

#     opt = NLopt.Opt(NLopt.:LD_LBFGS, size(𝐒ⁱ,2))
                    
#     opt.min_objective = optim_fun

#     # opt.xtol_abs = eps()
#     # opt.ftol_abs = eps()
#     opt.maxeval = 10000

#     (minf,x,ret) = NLopt.optimize(opt, zeros(size(𝐒ⁱ,2)))

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x) + 𝐒ⁱ³ᵉ * kron(x,kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end



# function find_shocks(::Val{:LBFGSjl},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-15) # will fail for higher or lower precision

#     function f(X)
#         sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X) - 𝐒ⁱ³ᵉ * kron(X,kron(X,X))))
#     end

#     function g!(G, x)
#         G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * kron(ℒ.I(length(x)),kron(x,x)))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
#     end

#     sol = Optim.optimize(f,g!,
#         # X -> sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X) - 𝐒ⁱ³ᵉ * kron(X,kron(X,X)))),
#                         zeros(size(𝐒ⁱ,2)), 
#                         Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)))#; 
#                         # autodiff = :forward)

#     x = sol.minimizer

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x) + 𝐒ⁱ³ᵉ * kron(x,kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end



# function find_shocks(::Val{:LBFGSjl},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-15) # will fail for higher or lower precision

#     function f(X)
#         sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X)))
#     end

#     function g!(G, x)
#         G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
#     end

#     sol = Optim.optimize(f,g!,
#     # X -> sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X))),
#                         zeros(size(𝐒ⁱ,2)), 
#                         Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)))#; 
#                         # autodiff = :forward)

#     x = sol.minimizer

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x)

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end


# function find_shocks(::Val{:speedmapping},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-15) # will fail for higher or lower precision

#     function f(X)
#         sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X) - 𝐒ⁱ³ᵉ * kron(X,kron(X,X))))
#     end

#     function g!(G, x)
#         G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * kron(ℒ.I(length(x)),kron(x,x)))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
#     end

#     sol = speedmapping(zeros(size(𝐒ⁱ,2)), f = f, g! = g!, tol = tol, maps_limit = 10000, stabilize = false)
# println(sol)
#     x = sol.minimizer

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x) + 𝐒ⁱ³ᵉ * kron(x,kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end




# function find_shocks(::Val{:speedmapping},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-15) # will fail for higher or lower precision
#     function f(X)
#         sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X)))
#     end

#     function g!(G, x)
#         G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
#     end

#     sol = speedmapping(zeros(size(𝐒ⁱ,2)), f = f, g! = g!, tol = tol, maps_limit = 10000, stabilize = false)

#     x = sol.minimizer
    
#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x)

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end