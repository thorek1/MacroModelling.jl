# Quadrature-based stochastic steady state calculation
# This file implements the quadrature algorithm for finding the stochastic steady state
# by integrating out shocks using the dynamic equations of the model

"""
    write_quadrature_dynamic_function!(𝓂::ℳ; opts...)

Write a function that evaluates the dynamic equations of the model.
This function is similar to `write_ss_check_function!` but uses dynamic equations instead of steady state equations.

The generated function has signature:
    f(residual, parameters, NSSS_variables, state_variables, shocks)

Where:
- residual: output vector for equation residuals
- parameters: model parameters
- NSSS_variables: variables defined in non-stochastic steady state (if applicable)
- state_variables: all model variables (mapped to their respective timings: past, present, future)
- shocks: exogenous shock values

# Arguments
- `𝓂::ℳ`: Model object
- `cse::Bool = true`: Use common subexpression elimination
- `skipzeros::Bool = true`: Skip zero elements in sparse operations
- `density_threshold::Float64 = .1`: Threshold for dense vs sparse representation
- `nnz_parallel_threshold::Int = 1000000`: Threshold for parallel evaluation
- `min_length::Int = 10000`: Minimum length for sparse representation
"""
function write_quadrature_dynamic_function!(𝓂::ℳ;
                                           cse = true,
                                           skipzeros = true,
                                           density_threshold::Float64 = .1,
                                           nnz_parallel_threshold::Int = 1000000,
                                           min_length::Int = 10000)
    # Get dynamic equations
    dyn_equations = 𝓂.dyn_equations
    
    # Extract variables that appear in dynamic equations
    future_varss  = collect(reduce(union, match_pattern.(get_symbols.(dyn_equations), r"₍₁₎$")))
    present_varss = collect(reduce(union, match_pattern.(get_symbols.(dyn_equations), r"₍₀₎$")))
    past_varss    = collect(reduce(union, match_pattern.(get_symbols.(dyn_equations), r"₍₋₁₎$")))
    shock_varss   = collect(reduce(union, match_pattern.(get_symbols.(dyn_equations), r"₍ₓ₎$")))
    ss_varss      = collect(reduce(union, match_pattern.(get_symbols.(dyn_equations), r"₍ₛₛ₎$")))
    
    # Get NSSS variables (variables that need to be provided from steady state)
    nsss_vars = 𝓂.vars_in_ss_equations
    
    # Number of parameters, variables, shocks
    np = length(𝓂.parameters)
    nv = length(𝓂.var)
    ne = length(𝓂.exo)
    n_nsss = length(nsss_vars)
    
    # Create symbolic variables
    Symbolics.@variables 𝔓[1:np] 𝔙ₙₛₛₛ[1:n_nsss] 𝔙ₚₐₛₜ[1:nv] 𝔙ₚᵣₑₛ[1:nv] 𝔙_fₜᵤᵣ[1:nv] 𝔈[1:ne]
    
    # Build replacement dictionaries
    parameter_dict = Dict{Symbol, Any}()
    back_to_array_dict = Dict{Symbolics.Num, Symbolics.Num}()
    
    # Map parameters
    for (i, v) in enumerate(𝓂.parameters)
        push!(parameter_dict, v => :($(Symbol("𝔓_$i"))))
        push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔓_$i"))), @__MODULE__) => 𝔓[i])
    end
    
    # Map NSSS variables
    for (i, v) in enumerate(nsss_vars)
        if v ∈ ss_varss
            # This variable appears with ₍ₛₛ₎ subscript in dynamic equations
            push!(parameter_dict, Symbol(replace(string(v), r"₍ₛₛ₎$" => "")) => :($(Symbol("𝔙ₙₛₛₛ_$i"))))
            push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔙ₙₛₛₛ_$i"))), @__MODULE__) => 𝔙ₙₛₛₛ[i])
        end
    end
    
    # Map past variables
    for (i, v) in enumerate(𝓂.var)
        sym_past = Symbol(string(v) * "₍₋₁₎")
        if sym_past ∈ past_varss
            push!(parameter_dict, sym_past => :($(Symbol("𝔙ₚₐₛₜ_$i"))))
            push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔙ₚₐₛₜ_$i"))), @__MODULE__) => 𝔙ₚₐₛₜ[i])
        end
    end
    
    # Map present variables
    for (i, v) in enumerate(𝓂.var)
        sym_present = Symbol(string(v) * "₍₀₎")
        if sym_present ∈ present_varss
            push!(parameter_dict, sym_present => :($(Symbol("𝔙ₚᵣₑₛ_$i"))))
            push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔙ₚᵣₑₛ_$i"))), @__MODULE__) => 𝔙ₚᵣₑₛ[i])
        end
    end
    
    # Map future variables
    for (i, v) in enumerate(𝓂.var)
        sym_future = Symbol(string(v) * "₍₁₎")
        if sym_future ∈ future_varss
            push!(parameter_dict, sym_future => :($(Symbol("𝔙_fₜᵤᵣ_$i"))))
            push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔙_fₜᵤᵣ_$i"))), @__MODULE__) => 𝔙_fₜᵤᵣ[i])
        end
    end
    
    # Map shocks
    for (i, v) in enumerate(𝓂.exo)
        sym_shock = Symbol(string(v) * "₍ₓ₎")
        if sym_shock ∈ shock_varss
            push!(parameter_dict, sym_shock => :($(Symbol("𝔈_$i"))))
            push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔈_$i"))), @__MODULE__) => 𝔈[i])
        end
    end
    
    # Substitute in dynamic equations
    dyn_equations_sub = dyn_equations |>
        x -> replace_symbols.(x, Ref(parameter_dict)) |>
        x -> Symbolics.parse_expr_to_symbolic.(x, Ref(@__MODULE__)) |>
        x -> Symbolics.substitute.(x, Ref(back_to_array_dict))
    
    lennz = length(dyn_equations_sub)
    
    if lennz > nnz_parallel_threshold
        parallel = Symbolics.ShardedForm(1500, 4)
    else
        parallel = Symbolics.SerialForm()
    end
    
    # Build function
    _, func_exprs = Symbolics.build_function(dyn_equations_sub, 𝔓, 𝔙ₙₛₛₛ, 𝔙ₚₐₛₜ, 𝔙ₚᵣₑₛ, 𝔙_fₜᵤᵣ, 𝔈,
                                            cse = cse,
                                            skipzeros = skipzeros,
                                            parallel = parallel,
                                            expression_module = @__MODULE__,
                                            expression = Val(false))::Tuple{<:Function, <:Function}
    
    # Store in model object
    𝓂.quadrature_dynamic_func = func_exprs
    
    return nothing
end


"""
    renormalize_shocks!(shocks::Matrix{Float64})

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
    generate_shock_samples(n_draws::Int, n_shocks::Int)

Generate quasi-random shock samples using Sobol sequences from QuasiMonteCarlo.jl.
Returns a matrix of standard normal shocks with zero mean and unit variance.
"""
function generate_shock_samples(n_draws::Int, n_shocks::Int)::Matrix{Float64}
    # Generate Sobol sequence points in [0,1]^n_shocks
    lb = zeros(n_shocks)
    ub = ones(n_shocks)
    
    sampler = QuasiMonteCarlo.SobolSample()
    sobol_points = QuasiMonteCarlo.sample(n_draws, lb, ub, sampler)
    
    # Transform to standard normal using inverse CDF
    # Avoid exact 0 and 1 to prevent infinite values
    ϵ = 1e-10
    sobol_clipped = clamp.(sobol_points, ϵ, 1.0 - ϵ)
    
    # Use inverse error function to get normal variates
    # Φ^(-1)(p) = √2 * erf^(-1)(2p - 1)
    normal_shocks = sqrt(2.0) .* erfinv.(2.0 .* sobol_clipped .- 1.0)
    
    # Renormalize to ensure zero mean and unit variance
    renormalize_shocks!(normal_shocks)
    
    return normal_shocks
end


"""
    optimize_quadrature_sss(parameters, nsss_vars, shock_draws, initial_state, 𝓂; opts...)

Optimization routine that finds the state minimizing residuals across shock draws.

# Arguments
- `parameters`: Model parameter values
- `nsss_vars`: Non-stochastic steady state variable values
- `shock_draws`: Matrix of shock samples (n_shocks × n_draws)
- `initial_state`: Initial guess for the state vector
- `𝓂`: Model object
- `opts`: Calculation options
- `max_newton_iterations`: Maximum Newton iterations (default: 50)
- `convergence_tol`: Convergence tolerance (default: 1e-6)

# Returns
- `state`: Optimized state vector
- `converged`: Whether optimization converged
- `residual_norm`: Final residual norm
"""
function optimize_quadrature_sss(
    parameters::Vector{M},
    nsss_vars::Vector{M},
    shock_draws::Matrix{Float64},
    initial_state::Vector{M},
    𝓂::ℳ;
    opts::CalculationOptions = merge_calculation_options(),
    max_newton_iterations::Int = 50,
    convergence_tol::Float64 = 1e-6
)::Tuple{Vector{M}, Bool, M} where M
    
    n_shocks, n_draws = size(shock_draws)
    nv = 𝓂.timings.nVars
    
    # Ensure quadrature dynamic function is available
    if !isdefined(𝓂, :quadrature_dynamic_func) || 𝓂.quadrature_dynamic_func == (x->x)
        write_quadrature_dynamic_function!(𝓂)
    end
    
    state_candidate = copy(initial_state)
    residual_buffer = zeros(M, nv)
    
    for newton_iter in 1:max_newton_iterations
        # Accumulate residuals across all draws
        total_residual = zeros(M, nv)
        
        for draw in 1:n_draws
            shock = shock_draws[:, draw]
            
            # For stochastic steady state: past = present = future = state_candidate
            # Evaluate f(state, state, state, shock) where we want this to equal state
            𝓂.quadrature_dynamic_func(residual_buffer, parameters, nsss_vars, 
                                      state_candidate, state_candidate, state_candidate, shock)
            
            # Residual: state_candidate - implied_next_state
            # If equations are in form: 0 = f(past, present, future, shock)
            # Then implied_next_state satisfies: 0 = f(state, state, implied_next_state, shock)
            # For SSS: state = E[implied_next_state]
            # So residual = state_candidate - (-residual_buffer) = state_candidate + residual_buffer
            total_residual .+= residual_buffer
        end
        
        # Average residual
        avg_residual = total_residual / n_draws
        residual_norm = ℒ.norm(avg_residual)
        
        if opts.verbose && newton_iter % 10 == 0
            println("  Newton iteration $newton_iter, residual norm: $residual_norm")
        end
        
        # Check convergence
        if residual_norm < convergence_tol
            if opts.verbose
                println("  Newton converged after $newton_iter iterations, residual norm: $residual_norm")
            end
            return state_candidate, true, residual_norm
        end
        
        if newton_iter == max_newton_iterations
            if opts.verbose
                println("  Newton reached max iterations, residual norm: $residual_norm")
            end
            return state_candidate, false, residual_norm
        end
        
        # Simple Newton step with identity Jacobian approximation
        # More sophisticated: could compute numerical Jacobian or use the model's jacobian
        step_size = min(1.0, 1.0 / (1.0 + residual_norm))
        state_candidate .-= step_size .* avg_residual
    end
    
    return state_candidate, false, ℒ.norm(total_residual / n_draws)
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
    
    # Get NSSS variables for the dynamic function
    nsss_vars = SS_and_pars[1:length(𝓂.vars_in_ss_equations)]
    
    if opts.verbose
        println("Starting quadrature SSS calculation from pruned 2nd order SSS")
        println("Initial SSS: ", current_sss[1:min(5, length(current_sss))])
    end
    
    # Iteratively refine with increasing number of draws
    n_draws = initial_draws
    previous_sss = copy(current_sss)
    
    while n_draws <= max_draws
        if opts.verbose
            println("\nQuadrature iteration with $n_draws draws")
        end
        
        # Generate quasi-random shock samples using QuasiMonteCarlo.jl
        shock_draws = generate_shock_samples(n_draws, n_exo)
        
        # Find SSS that minimizes error across draws using the optimization routine
        sss_candidate, converged, residual_norm = optimize_quadrature_sss(
            parameters, nsss_vars, shock_draws, current_sss, 𝓂,
            opts = opts,
            max_newton_iterations = max_newton_iterations,
            convergence_tol = convergence_tol
        )
        
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
