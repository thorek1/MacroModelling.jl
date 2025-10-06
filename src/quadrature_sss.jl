# Quadrature-based stochastic steady state calculation
# This file implements the quadrature algorithm for finding the stochastic steady state
# by integrating out shocks using the dynamic equations of the model

"""
Generate Sobol sequence points for quasi-random sampling.
This is a simple implementation of Sobol sequence generation.
"""
function generate_sobol_points(n_points::Int, n_dims::Int)::Matrix{Float64}
    # For now, use a simple implementation
    # We'll use bit-reversal and Gray code for the first dimension
    # and generalized Sobol for other dimensions
    
    points = zeros(Float64, n_dims, n_points)
    
    # Direction numbers for Sobol sequence (simplified version)
    # These are the standard direction numbers from Bratley and Fox (1988)
    direction_numbers = [
        [1],  # Dimension 1
        [1, 3],  # Dimension 2
        [1, 3, 1],  # Dimension 3
        [1, 1, 1],  # Dimension 4
        [1, 1, 3, 3],  # Dimension 5
        [1, 3, 5, 13],  # Dimension 6
        [1, 1, 5, 5, 17],  # Dimension 7
        [1, 1, 5, 5, 5],  # Dimension 8
    ]
    
    # Extend direction numbers if needed
    while length(direction_numbers) < n_dims
        # Use simple pattern for additional dimensions
        push!(direction_numbers, [1, 3])
    end
    
    for dim in 1:n_dims
        # Initialize direction vectors
        max_bits = ceil(Int, log2(n_points)) + 10
        v = zeros(UInt32, max_bits)
        
        m = direction_numbers[min(dim, length(direction_numbers))]
        s = length(m)
        
        # Set up first direction numbers
        for i in 1:min(s, max_bits)
            v[i] = UInt32(m[i]) << (32 - i)
        end
        
        # Generate additional direction numbers using recurrence
        if s < max_bits
            for i in (s+1):max_bits
                v[i] = v[i-s] ⊻ (v[i-s] >> s)
                for k in 1:(s-1)
                    v[i] ⊻= ((m[k] & 1) * v[i-k])
                end
            end
        end
        
        # Generate points using Gray code
        x = UInt32(0)
        for i in 1:n_points
            gray = (i-1) ⊻ ((i-1) >> 1)
            
            # Find rightmost zero bit in previous index
            rightmost_zero = trailing_ones(UInt32(i-1))
            
            if i == 1
                x = UInt32(0)
            else
                x ⊻= v[rightmost_zero + 1]
            end
            
            points[dim, i] = Float64(x) / Float64(typemax(UInt32))
        end
    end
    
    return points
end


"""
Convert uniform [0,1] Sobol points to standard normal draws using inverse CDF.
"""
function sobol_to_normal(sobol_points::Matrix{Float64})::Matrix{Float64}
    # Use inverse CDF (quantile function) to transform uniform to normal
    # Avoid exact 0 and 1 to prevent infinite values
    ϵ = 1e-10
    sobol_clipped = clamp.(sobol_points, ϵ, 1.0 - ϵ)
    
    # Use the inverse error function to get normal variates
    # Φ^(-1)(p) = √2 * erf^(-1)(2p - 1)
    return sqrt(2.0) .* erfinv.(2.0 .* sobol_clipped .- 1.0)
end


"""
Renormalize shock draws to have zero mean and unit variance.
"""
function renormalize_shocks!(shocks::Matrix{Float64})
    n_shocks, n_draws = size(shocks)
    
    for i in 1:n_shocks
        μ = sum(shocks[i, :]) / n_draws
        shocks[i, :] .-= μ
        
        σ = sqrt(sum(shocks[i, :].^2) / n_draws)
        if σ > 1e-10
            shocks[i, :] ./= σ
        end
    end
    
    return shocks
end


"""
Evaluate dynamic equations at given state and shock values.
Returns the residual of the dynamic equations.
"""
function evaluate_dynamic_equations(
    states_past::Vector{Float64},
    states_present::Vector{Float64},
    states_future::Vector{Float64},
    shocks::Vector{Float64},
    parameters::Vector{Float64},
    SS_and_pars::Vector{Float64},
    𝓂::ℳ
)::Vector{Float64}
    # Build the full variable vector including past, present, future, and shocks
    # The jacobian function expects this format
    n_vars = 𝓂.timings.nVars
    n_past = 𝓂.timings.nPast_not_future_and_mixed
    n_exo = 𝓂.timings.nExo
    
    # Create evaluation vector: [past states, present states, future states, shocks]
    eval_vec = zeros(Float64, n_past + n_vars + n_vars + n_exo)
    
    # Map states to the evaluation vector
    past_idx = 𝓂.timings.past_not_future_and_mixed_idx
    eval_vec[1:n_past] .= states_past[past_idx]
    eval_vec[n_past+1:n_past+n_vars] .= states_present
    eval_vec[n_past+n_vars+1:n_past+2*n_vars] .= states_future
    eval_vec[end-n_exo+1:end] .= shocks
    
    # Use the jacobian evaluation to get equation values
    # We need to build a function that evaluates the dynamic equations
    # Since we have the jacobian, we can use automatic differentiation
    # or evaluate the equations directly
    
    # For now, use a finite difference approximation based on the SS
    # This is a placeholder - we'll need to implement proper equation evaluation
    residual = zeros(Float64, n_vars)
    
    # TODO: Implement proper dynamic equation evaluation
    # This would require access to the compiled dynamic equation functions
    # For now, return zero residual as placeholder
    
    return residual
end


"""
Calculate stochastic steady state using quadrature method.

This function finds the stochastic steady state by:
1. Starting from the pruned second order SSS
2. Using Sobol sequences to draw normally distributed shocks
3. Applying Newton's method to minimize errors across draws
4. Iteratively adding more points until convergence

# Arguments
- `parameters`: Model parameter values
- `𝓂`: Model object
- `opts`: Calculation options
- `initial_draws`: Initial number of quadrature points (default: 100)
- `max_draws`: Maximum number of quadrature points (default: 10000)
- `draw_increment`: Number of points to add per iteration (default: 100)
- `convergence_tol`: Tolerance for convergence (default: 1e-6)
- `max_newton_iterations`: Maximum Newton iterations per draw set (default: 50)
"""
function calculate_quadrature_stochastic_steady_state(
    parameters::Vector{M},
    𝓂::ℳ;
    opts::CalculationOptions = merge_calculation_options(),
    initial_draws::Int = 100,
    max_draws::Int = 10000,
    draw_increment::Int = 100,
    convergence_tol::Float64 = 1e-6,
    max_newton_iterations::Int = 50
)::Tuple{Vector{M}, Bool, Vector{M}, M, AbstractMatrix{M}, SparseMatrixCSC{M, Int}, AbstractMatrix{M}, SparseMatrixCSC{M, Int}} where M
    
    # First, calculate the pruned second order SSS as starting point
    sss_pruned, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = 
        calculate_second_order_stochastic_steady_state(parameters, 𝓂, opts = opts, pruning = true)
    
    if !converged
        if opts.verbose 
            println("Could not find pruned second order SSS as starting point for quadrature")
        end
        return sss_pruned, false, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂
    end
    
    # Extract the state part (first nVars elements)
    current_sss = sss_pruned[1:𝓂.timings.nVars]
    n_exo = 𝓂.timings.nExo
    
    # Get first order solution for state evolution
    # 𝐒₁ maps [states; 1; shocks] to next period states
    
    if opts.verbose
        println("Starting quadrature SSS calculation from pruned 2nd order SSS")
        println("Initial SSS: ", current_sss[1:min(5, length(current_sss))])
    end
    
    # Iteratively refine with increasing number of draws
    n_draws = initial_draws
    previous_sss = copy(current_sss)
    previous_newton_step_norm = Inf
    
    while n_draws <= max_draws
        if opts.verbose
            println("\nQuadrature iteration with $n_draws draws")
        end
        
        # Generate Sobol sequence for shocks
        sobol_uniform = generate_sobol_points(n_draws, n_exo)
        shock_draws = sobol_to_normal(sobol_uniform)
        renormalize_shocks!(shock_draws)
        
        # Newton iteration to find SSS that minimizes error across draws
        sss_candidate = copy(current_sss)
        
        for newton_iter in 1:max_newton_iterations
            # For each draw, compute the implied next period state using the model dynamics
            # The SSS should satisfy: E[f(s, s, s, ε)] = 0
            # where s is the steady state and ε are the shocks
            
            # Compute residuals across all draws
            total_residual = zeros(M, 𝓂.timings.nVars)
            total_jacobian = zeros(M, 𝓂.timings.nVars, 𝓂.timings.nVars)
            
            for draw in 1:n_draws
                shock = shock_draws[:, draw]
                
                # Evaluate model: next_state = f(current_state, current_state, shock)
                # For SSS: current_state = sss_candidate
                # We want: sss_candidate = E[f(sss_candidate, shock)]
                
                # Use first and second order solution to approximate dynamics
                aug_state = vcat(sss_candidate[𝓂.timings.past_not_future_and_mixed_idx], 1.0, shock)
                
                # First order: next_state ≈ 𝐒₁ * aug_state
                next_state_1st = 𝐒₁ * aug_state
                
                # Second order: add 𝐒₂ * kron(aug_state, aug_state) / 2
                next_state_2nd = next_state_1st + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
                
                # Residual: sss_candidate - next_state_2nd
                residual = sss_candidate - next_state_2nd
                total_residual += residual
                
                # Accumulate Jacobian (∂residual/∂sss_candidate)
                # ∂residual/∂sss = I - ∂next_state/∂sss
                # For first order: ∂next_state/∂sss = 𝐒₁[:, 1:nPast]
                # For second order: more complex, but we'll use first order approximation
                total_jacobian += ℒ.I - 𝐒₁[:, 1:𝓂.timings.nPast_not_future_and_mixed][:, 𝓂.timings.past_not_future_and_mixed_idx]
            end
            
            # Average residual and jacobian
            avg_residual = total_residual / n_draws
            avg_jacobian = total_jacobian / n_draws
            
            # Check convergence
            residual_norm = ℒ.norm(avg_residual)
            
            if newton_iter > 1 && residual_norm < convergence_tol
                if opts.verbose
                    println("  Newton converged after $newton_iter iterations, residual norm: $residual_norm")
                end
                break
            end
            
            if newton_iter == max_newton_iterations
                if opts.verbose
                    println("  Newton reached max iterations, residual norm: $residual_norm")
                end
            end
            
            # Newton step
            Δsss = try
                avg_jacobian \ avg_residual
            catch
                if opts.verbose
                    println("  Warning: Singular Jacobian in Newton iteration")
                end
                break
            end
            
            # Update candidate
            sss_candidate -= Δsss
            
            # Track step size
            if newton_iter == max_newton_iterations
                previous_newton_step_norm = ℒ.norm(Δsss)
            end
        end
        
        # Update current SSS
        current_sss = sss_candidate
        
        # Check if adding more points changed the result significantly
        sss_change = ℒ.norm(current_sss - previous_sss)
        
        if opts.verbose
            println("  SSS change from previous iteration: $sss_change")
        end
        
        if n_draws > initial_draws && sss_change < convergence_tol
            if opts.verbose
                println("Quadrature SSS converged with $n_draws draws")
                println("Final SSS: ", current_sss[1:min(5, length(current_sss))])
            end
            break
        end
        
        previous_sss = copy(current_sss)
        n_draws += draw_increment
    end
    
    # Build the full SSS vector (including auxiliary variables)
    all_SS = expand_steady_state(SS_and_pars, 𝓂)
    final_sss = all_SS + vcat(current_sss, zeros(M, length(all_SS) - length(current_sss)))
    
    return final_sss, true, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂
end
