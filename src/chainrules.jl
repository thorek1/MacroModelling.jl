# ChainRules rrule definitions for MacroModelling.jl
# This file contains all reverse-mode automatic differentiation rules
# for use with Zygote and other ChainRulesCore-compatible AD systems.

# These rrules enable efficient gradient computation for DSGE model operations
# including perturbation solutions, Kalman filtering, and steady state calculations.

# =============================================================================
# Basic operations
# =============================================================================

function rrule(::typeof(mul_reverse_AD!),
                C::Matrix{S},
                A::AbstractMatrix{M},
                B::AbstractMatrix{N}) where {S <: Real, M <: Real, N <: Real}
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)

    function times_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        dA = @thunk(project_A(Ȳ * B'))
        dB = @thunk(project_B(A' * Ȳ))
        return NoTangent(), NoTangent(), dA, dB
    end

    return ℒ.mul!(C,A,B), times_pullback
end


function rrule(::typeof(sparse_preallocated!), Ŝ::Matrix{T}; ℂ::higher_order_caches{T,F} = Higher_order_caches()) where {T <: Real, F <: AbstractFloat}
    project_Ŝ = ProjectTo(Ŝ)

    function sparse_preallocated_pullback(Ω̄)
        ΔΩ = unthunk(Ω̄)
        ΔŜ = project_Ŝ(ΔΩ)
        return NoTangent(), ΔŜ, NoTangent()
    end

    return sparse_preallocated!(Ŝ, ℂ = ℂ), sparse_preallocated_pullback
end


# ========== src/MacroModelling.jl ==========

# Lines 6450-6524
function rrule(::typeof(calculate_second_order_stochastic_steady_state),
                                                        ::Val{:newton}, 
                                                        𝐒₁::Matrix{Float64}, 
                                                        𝐒₂::AbstractSparseMatrix{Float64}, 
                                                        x::Vector{Float64},
                                                        𝓂::ℳ;
                                                        # timer::TimerOutput = TimerOutput(),
                                                        tol::AbstractFloat = 1e-14)
    # @timeit_debug timer "Calculate SSS - forward" begin
    # @timeit_debug timer "Setup indices" begin

    nᵉ = 𝓂.timings.nExo

    s_in_s⁺ = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))
    s_in_s = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed ), zeros(Bool, nᵉ + 1)))
    
    kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
    
    kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)
    
    A = 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
    B = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
    
    # end # timeit_debug
      
    # @timeit_debug timer "Iterations" begin

    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    for i in 1:max_iters
        ∂x = (A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))

        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            return x, false
        end
        
        Δx = ∂x̂ \ (A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 - x)

        if i > 5 && isapprox(A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    solved = isapprox(A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2, x, rtol = tol)         

    # println(x)

    ∂𝐒₁ =  zero(𝐒₁)
    ∂𝐒₂ =  zero(𝐒₂)

    # end # timeit_debug
    # end # timeit_debug

    function second_order_stochastic_steady_state_pullback(∂x)
        # @timeit_debug timer "Calculate SSS - pullback" begin

        S = -∂x[1]' / (A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))

        ∂𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed] = S' * x'
        
        ∂𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺] = S' * ℒ.kron(vcat(x,1), vcat(x,1))' / 2

        # end # timeit_debug

        return NoTangent(), NoTangent(), ∂𝐒₁, ∂𝐒₂, NoTangent(), NoTangent(), NoTangent()
    end

    return (x, solved), second_order_stochastic_steady_state_pullback
end

# Lines 6801-6868
function rrule(::typeof(calculate_third_order_stochastic_steady_state),
                                                        ::Val{:newton}, 
                                                        𝐒₁::Matrix{Float64}, 
                                                        𝐒₂::AbstractSparseMatrix{Float64}, 
                                                        𝐒₃::AbstractSparseMatrix{Float64},
                                                        x::Vector{Float64},
                                                        𝓂::ℳ;
                                                        tol::AbstractFloat = 1e-14)
    nᵉ = 𝓂.timings.nExo

    s_in_s⁺ = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))
    s_in_s = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed ), zeros(Bool, nᵉ + 1)))
    
    kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
    
    kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)
    
    kron_s⁺_s⁺_s⁺ = ℒ.kron(s_in_s⁺, kron_s⁺_s⁺)
    
    kron_s_s⁺_s⁺ = ℒ.kron(kron_s⁺_s⁺, s_in_s)
    
    A = 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
    B = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
    C = 𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s_s⁺_s⁺]
    Ĉ = 𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺]

    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    for i in 1:max_iters
        ∂x = (A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) + C * ℒ.kron(ℒ.kron(vcat(x,1), vcat(x,1)), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) / 2 - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))
        
        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            return x, false
        end
        
        Δx = ∂x̂ \ (A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 + Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) / 6 - x)

        if i > 5 && isapprox(A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 + Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) / 6, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    solved = isapprox(A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 + Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) / 6, x, rtol = tol)         

    ∂𝐒₁ =  zero(𝐒₁)
    ∂𝐒₂ =  zero(𝐒₂)
    ∂𝐒₃ =  zero(𝐒₃)

    function third_order_stochastic_steady_state_pullback(∂x)
        S = -∂x[1]' / (A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) + C * ℒ.kron(ℒ.kron(vcat(x,1), vcat(x,1)), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) / 2 - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))

        ∂𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed] = S' * x'
        
        ∂𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺] = S' * ℒ.kron(vcat(x,1), vcat(x,1))' / 2

        ∂𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺] = S' * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1)))' / 6

        return NoTangent(), NoTangent(), ∂𝐒₁, ∂𝐒₂, ∂𝐒₃, NoTangent(), NoTangent(), NoTangent()
    end

    return (x, solved), third_order_stochastic_steady_state_pullback
end

# Lines 8566-8591
function rrule(::typeof(calculate_jacobian), 
                parameters, 
                SS_and_pars, 
                𝓂)#;
                # timer::TimerOutput = TimerOutput())
    # @timeit_debug timer "Calculate jacobian - forward" begin

    jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)

    function calculate_jacobian_pullback(∂∇₁)
        # @timeit_debug timer "Calculate jacobian - reverse" begin

        𝓂.jacobian_parameters[2](𝓂.jacobian_parameters[1], parameters, SS_and_pars)
        𝓂.jacobian_SS_and_pars[2](𝓂.jacobian_SS_and_pars[1], parameters, SS_and_pars)

        ∂parameters = 𝓂.jacobian_parameters[1]' * vec(∂∇₁)
        ∂SS_and_pars = 𝓂.jacobian_SS_and_pars[1]' * vec(∂∇₁)

        # end # timeit_debug
        # end # timeit_debug

        return NoTangent(), ∂parameters, ∂SS_and_pars, NoTangent()
    end

    return jacobian, calculate_jacobian_pullback
end

# Lines 8614-8633
function rrule(::typeof(calculate_hessian), parameters, SS_and_pars, 𝓂)
    hessian = calculate_hessian(parameters, SS_and_pars, 𝓂)

    function calculate_hessian_pullback(∂∇₂)
        # @timeit_debug timer "Calculate hessian - reverse" begin

        𝓂.hessian_parameters[2](𝓂.hessian_parameters[1], parameters, SS_and_pars)
        𝓂.hessian_SS_and_pars[2](𝓂.hessian_SS_and_pars[1], parameters, SS_and_pars)

        ∂parameters = 𝓂.hessian_parameters[1]' * vec(∂∇₂)
        ∂SS_and_pars = 𝓂.hessian_SS_and_pars[1]' * vec(∂∇₂)

        # end # timeit_debug
        # end # timeit_debug

        return NoTangent(), ∂parameters, ∂SS_and_pars, NoTangent()
    end

    return hessian, calculate_hessian_pullback
end

# Lines 8658-8679
function rrule(::typeof(calculate_third_order_derivatives), parameters, SS_and_pars, 𝓂) # ;
    # timer::TimerOutput = TimerOutput())
    # @timeit_debug timer "3rd order derivatives - forward" begin
    third_order_derivatives = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂) #, timer = timer)
    # end # timeit_debug

    function calculate_third_order_derivatives_pullback(∂∇₃)
        # @timeit_debug timer "3rd order derivatives - pullback" begin
        𝓂.third_order_derivatives_parameters[2](𝓂.third_order_derivatives_parameters[1], parameters, SS_and_pars)
        𝓂.third_order_derivatives_SS_and_pars[2](𝓂.third_order_derivatives_SS_and_pars[1], parameters, SS_and_pars)

        ∂parameters = 𝓂.third_order_derivatives_parameters[1]' * vec(∂∇₃)
        ∂SS_and_pars = 𝓂.third_order_derivatives_SS_and_pars[1]' * vec(∂∇₃)

        # end # timeit_debug
        # end # timeit_debug

        return NoTangent(), ∂parameters, ∂SS_and_pars, NoTangent()
    end

    return third_order_derivatives, calculate_third_order_derivatives_pullback
end

# Lines 9814-9936
function rrule(::typeof(get_NSSS_and_parameters), 
                𝓂::ℳ, 
                parameter_values::Vector{S}; 
                opts::CalculationOptions = merge_calculation_options(),
                cold_start::Bool = false) where S <: Real
                # timer::TimerOutput = TimerOutput(),
    # @timeit_debug timer "Calculate NSSS - forward" begin

    # Use custom steady state function if available, otherwise use default solver
    if !isnothing(𝓂.custom_steady_state_function)
        SS_and_pars = 𝓂.custom_steady_state_function(parameter_values)

        vars_in_ss_equations = sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))
        expected_length = length(vars_in_ss_equations) + length(𝓂.calibration_equations_parameters)

        if length(SS_and_pars) != expected_length
            throw(ArgumentError("Custom steady state function returned $(length(SS_and_pars)) values, expected $expected_length."))
        end

        residual = zeros(length(𝓂.ss_equations) + length(𝓂.calibration_equations))

        𝓂.SS_check_func(residual, 𝓂.parameter_values, SS_and_pars)

        solution_error = sum(abs, residual)
        
        iters = 0

        if !isfinite(solution_error) || solution_error > opts.tol.NSSS_acceptance_tol
            throw(ArgumentError("Custom steady state function failed steady state check: residual $solution_error > $(opts.tol.NSSS_acceptance_tol)."))
        end
        
        var_idx = indexin([vars_in_ss_equations...], [𝓂.var...,𝓂.calibration_equations_parameters...])

        calib_idx = indexin([𝓂.calibration_equations_parameters...], [𝓂.var...,𝓂.calibration_equations_parameters...])

        SS_and_pars_tmp = zeros(length(𝓂.var) + length(𝓂.calibration_equations_parameters))

        SS_and_pars_tmp[[var_idx..., calib_idx...]] = SS_and_pars

        SS_and_pars = SS_and_pars_tmp
    else
        SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameter_values, 𝓂, opts.tol, opts.verbose, cold_start, 𝓂.solver_parameters)
    end

    # end # timeit_debug

    if solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error)
        return (SS_and_pars, (solution_error, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # @timeit_debug timer "Calculate NSSS - pullback" begin

    SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)
        
    SS_and_pars_names = vcat(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)

    SS_and_pars_names_no_exo = vcat(Symbol.(replace.(string.(sort(setdiff(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)

    # unknowns = union(setdiff(𝓂.vars_in_ss_equations, 𝓂.➕_vars), 𝓂.calibration_equations_parameters)
    unknowns = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))), 𝓂.calibration_equations_parameters))

    ∂ = parameter_values
    C = SS_and_pars[indexin(unique(SS_and_pars_names_no_exo), SS_and_pars_names_lead_lag)] # [dyn_ss_idx])

    if eltype(𝓂.∂SS_equations_∂parameters[1]) != eltype(parameter_values)
        if 𝓂.∂SS_equations_∂parameters[1] isa SparseMatrixCSC
            jac_buffer = similar(𝓂.∂SS_equations_∂parameters[1], eltype(parameter_values))
            jac_buffer.nzval .= 0
        else
            jac_buffer = zeros(eltype(parameter_values), size(𝓂.∂SS_equations_∂parameters[1]))
        end
    else
        jac_buffer = 𝓂.∂SS_equations_∂parameters[1]
    end

    𝓂.∂SS_equations_∂parameters[2](jac_buffer, ∂, C)

    ∂SS_equations_∂parameters = jac_buffer

    
    if eltype(𝓂.∂SS_equations_∂SS_and_pars[1]) != eltype(SS_and_pars)
        if 𝓂.∂SS_equations_∂SS_and_pars[1] isa SparseMatrixCSC
            jac_buffer = similar(𝓂.∂SS_equations_∂SS_and_pars[1], eltype(SS_and_pars))
            jac_buffer.nzval .= 0
        else
            jac_buffer = zeros(eltype(SS_and_pars), size(𝓂.∂SS_equations_∂SS_and_pars[1]))
        end
    else
        jac_buffer = 𝓂.∂SS_equations_∂SS_and_pars[1]
    end

    𝓂.∂SS_equations_∂SS_and_pars[2](jac_buffer, ∂, C)

    ∂SS_equations_∂SS_and_pars = jac_buffer

    ∂SS_equations_∂SS_and_pars_lu = RF.lu(∂SS_equations_∂SS_and_pars, check = false)

    if !ℒ.issuccess(∂SS_equations_∂SS_and_pars_lu)
        return (SS_and_pars, (10.0, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    JVP = -(∂SS_equations_∂SS_and_pars_lu \ ∂SS_equations_∂parameters)#[indexin(SS_and_pars_names, unknowns),:]

    jvp = zeros(length(SS_and_pars_names_lead_lag), length(𝓂.parameters))
    
    for (i,v) in enumerate(SS_and_pars_names)
        if v in unknowns
            jvp[i,:] = JVP[indexin([v], unknowns),:]
        end
    end

    # end # timeit_debug
    # end # timeit_debug

    # try block-gmres here
    function get_non_stochastic_steady_state_pullback(∂SS_and_pars)
        # println(∂SS_and_pars)
        return NoTangent(), NoTangent(), jvp' * ∂SS_and_pars[1], NoTangent()
    end


    return (SS_and_pars, (solution_error, iters)), get_non_stochastic_steady_state_pullback
end

# ========== src/perturbation.jl ==========

# Lines 120-275
function rrule(::typeof(calculate_first_order_solution), 
                ∇₁::Matrix{R};
                T::timings, 
                opts::CalculationOptions = merge_calculation_options(),
                initial_guess::AbstractMatrix{R} = zeros(0,0)) where R <: AbstractFloat
    # Forward pass to compute the output and intermediate values needed for the backward pass
    # @timeit_debug timer "Calculate 1st order solution" begin
    # @timeit_debug timer "Preprocessing" begin

    dynIndex = T.nPresent_only+1:T.nVars

    reverse_dynamic_order = indexin([T.past_not_future_idx; T.future_not_past_and_mixed_idx], T.present_but_not_only_idx)

    comb = union(T.future_not_past_and_mixed_idx, T.past_not_future_idx)
    sort!(comb)

    future_not_past_and_mixed_in_comb = indexin(T.future_not_past_and_mixed_idx, comb)
    past_not_future_and_mixed_in_comb = indexin(T.past_not_future_and_mixed_idx, comb)
    
    Ir = ℒ.I(length(comb))
    
    ∇₊ = ∇₁[:,1:T.nFuture_not_past_and_mixed]
    ∇₀ = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]    
    ∇₋ = ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]
    ∇̂ₑ = ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]
    
    # end # timeit_debug
    # @timeit_debug timer "Invert ∇₀" begin

    Q    = ℒ.qr!(∇₀[:,T.present_only_idx])

    A₊ = Q.Q' * ∇₊
    A₀ = Q.Q' * ∇₀
    A₋ = Q.Q' * ∇₋
    
    # end # timeit_debug
    # @timeit_debug timer "Sort matrices" begin

    Ã₊ = A₊[dynIndex,:] * Ir[future_not_past_and_mixed_in_comb,:]
    Ã₀ = A₀[dynIndex, comb]
    Ã₋ = A₋[dynIndex,:] * Ir[past_not_future_and_mixed_in_comb,:]

    # end # timeit_debug
    # @timeit_debug timer "Quadratic matrix equation solve" begin

    sol, solved = solve_quadratic_matrix_equation(Ã₊, Ã₀, Ã₋, T, 
                                                    initial_guess = initial_guess,
                                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                                    tol = opts.tol.qme_tol,
                                                    acceptance_tol = opts.tol.qme_acceptance_tol,
                                                    verbose = opts.verbose)

    if !solved
        return (zeros(T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false), x -> NoTangent(), NoTangent(), NoTangent()
    end

    # end # timeit_debug
    # @timeit_debug timer "Postprocessing" begin
    # @timeit_debug timer "Setup matrices" begin

    sol_compact = sol[reverse_dynamic_order, past_not_future_and_mixed_in_comb]

    D = sol_compact[end - T.nFuture_not_past_and_mixed + 1:end, :]

    L = sol[indexin(T.past_not_future_and_mixed_idx, T.present_but_not_only_idx), past_not_future_and_mixed_in_comb]

    Ā₀ᵤ  = A₀[1:T.nPresent_only, T.present_only_idx]
    A₊ᵤ  = A₊[1:T.nPresent_only,:]
    Ã₀ᵤ  = A₀[1:T.nPresent_only, T.present_but_not_only_idx]
    A₋ᵤ  = A₋[1:T.nPresent_only,:]

    # end # timeit_debug
    # @timeit_debug timer "Invert Ā₀ᵤ" begin

    Ā̂₀ᵤ = ℒ.lu!(Ā₀ᵤ, check = false)

    if !ℒ.issuccess(Ā̂₀ᵤ)
        return (zeros(T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false), x -> NoTangent(), NoTangent(), NoTangent()
    end

    # A    = vcat(-(Ā̂₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ)), sol)
    if T.nPresent_only > 0
        ℒ.mul!(A₋ᵤ, Ã₀ᵤ, sol[:,past_not_future_and_mixed_in_comb], 1, 1)
        nₚ₋ =  A₊ᵤ * D
        ℒ.mul!(A₋ᵤ, nₚ₋, L, 1, 1)
        ℒ.ldiv!(Ā̂₀ᵤ, A₋ᵤ)
        ℒ.rmul!(A₋ᵤ, -1)
    end

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Exogenous part solution" begin

    expand =   [ℒ.I(T.nVars)[T.future_not_past_and_mixed_idx,:],
                ℒ.I(T.nVars)[T.past_not_future_and_mixed_idx,:]] 

    𝐒ᵗ = vcat(A₋ᵤ, sol_compact)[T.reorder,:]

    𝐒̂ᵗ = 𝐒ᵗ * expand[2]

    ℒ.mul!(∇₀, ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1], 𝐒̂ᵗ, 1, 1)

    C = ℒ.lu!(∇₀, check = false)
    
    if !ℒ.issuccess(C)
        return (zeros(T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false), x -> NoTangent(), NoTangent(), NoTangent()
    end
    
    ℒ.ldiv!(C, ∇̂ₑ)
    ℒ.rmul!(∇̂ₑ, -1)

    # end # timeit_debug
    # end # timeit_debug
    
    M = inv(C)

    tmp2 = -M' * (∇₊ * expand[1])'
    
    ∇₊ = ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    ∇ₑ = ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

    function first_order_solution_pullback(∂𝐒) 
        ∂∇₁ = zero(∇₁)

        ∂𝐒ᵗ = ∂𝐒[1][:,1:T.nPast_not_future_and_mixed]
        ∂𝐒ᵉ = ∂𝐒[1][:,T.nPast_not_future_and_mixed + 1:end]

        ∂∇₁[:,T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1:end] .= -M' * ∂𝐒ᵉ

        ∂∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)] .= M' * ∂𝐒ᵉ * ∇ₑ' * M'

        ∂∇₁[:,1:T.nFuture_not_past_and_mixed] .= (M' * ∂𝐒ᵉ * ∇ₑ' * M' * expand[2]' * 𝐒ᵗ')[:,T.future_not_past_and_mixed_idx]

        ∂𝐒ᵗ .+= ∇₊' * M' * ∂𝐒ᵉ * ∇ₑ' * M' * expand[2]'

        tmp1 = M' * ∂𝐒ᵗ * expand[2]

        ss, solved = solve_sylvester_equation(tmp2, 𝐒̂ᵗ', -tmp1,
                                                sylvester_algorithm = opts.sylvester_algorithm²,
                                                tol = opts.tol.sylvester_tol,
                                                acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                                verbose = opts.verbose)

        if !solved
            NoTangent(), NoTangent(), NoTangent()
        end

        ∂∇₁[:,1:T.nFuture_not_past_and_mixed] .+= (ss * 𝐒̂ᵗ' * 𝐒̂ᵗ')[:,T.future_not_past_and_mixed_idx]
        ∂∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)] .+= ss * 𝐒̂ᵗ'
        ∂∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] .+= ss[:,T.past_not_future_and_mixed_idx]

        return NoTangent(), ∂∇₁, NoTangent()
    end

    return (hcat(𝐒ᵗ, ∇̂ₑ), sol, solved), first_order_solution_pullback
end

# Lines 499-765
function rrule(::typeof(calculate_second_order_solution), 
                    ∇₁::AbstractMatrix{S}, #first order derivatives
                    ∇₂::SparseMatrixCSC{S}, #second order derivatives
                    𝑺₁::AbstractMatrix{S},#first order solution
                    M₂::second_order_auxiliary_matrices,   # aux matrices
                    ℂC::caches;
                    T::timings,
                    initial_guess::AbstractMatrix{R} = zeros(0,0),
                    opts::CalculationOptions = merge_calculation_options()) where {S <: Real, R <: Real}
    if !(eltype(ℂC.second_order_caches.Ŝ) == S)
        ℂC.second_order_caches = Higher_order_caches(T = S)
    end
    ℂ = ℂC.second_order_caches
    # @timeit_debug timer "Second order solution - forward" begin
    # inspired by Levintal

    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx;
    i₋ = T.past_not_future_and_mixed_idx;

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo;
    n  = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    # @timeit_debug timer "Setup matrices" begin

    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]]# |> sparse
    # droptol!(𝐒₁,tol)
    
    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) ℒ.I(nₑ + 1)[1,:] zeros(nₑ + 1, nₑ)]
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 1.0)

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                ℒ.I(nₑ₋)[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]]

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)]

    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.I(n)[i₋,:] - ∇₁[:,range(1,n) .+ n₊]

    # end # timeit_debug
    # @timeit_debug timer "Invert matrix" begin

    ∇₁₊𝐒₁➕∇₁₀lu = ℒ.lu(∇₁₊𝐒₁➕∇₁₀, check = false)

    if !ℒ.issuccess(∇₁₊𝐒₁➕∇₁₀lu)
        if opts.verbose println("Second order solution: inversion failed") end
        return (∇₁₊𝐒₁➕∇₁₀, solved), x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    
    spinv = inv(∇₁₊𝐒₁➕∇₁₀lu)
    spinv = choose_matrix_format(spinv)

    # end # timeit_debug
    # @timeit_debug timer "Setup second order matrices" begin
    # @timeit_debug timer "A" begin

    ∇₁₊ = @views ∇₁[:,1:n₊] * ℒ.I(n)[i₊,:]

    A = spinv * ∇₁₊
    
    # end # timeit_debug
    # @timeit_debug timer "C" begin

    # ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = ∇₂ * (ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔) * M₂.𝐂₂ 
    ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, M₂.𝐂₂) + mat_mult_kron(∇₂, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎, M₂.𝛔 * M₂.𝐂₂)
    
    C = spinv * ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹

    # end # timeit_debug
    # @timeit_debug timer "B" begin

    # 𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0)

    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0)
    B = mat_mult_kron(M₂.𝐔₂, 𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ, M₂.𝐂₂) + M₂.𝐔₂ * M₂.𝛔 * M₂.𝐂₂

    # end # timeit_debug    
    # end # timeit_debug
    # @timeit_debug timer "Solve sylvester equation" begin

    𝐒₂, solved = solve_sylvester_equation(A, B, C,  
                                            initial_guess = initial_guess,
                                            sylvester_algorithm = opts.sylvester_algorithm²,
                                            tol = opts.tol.sylvester_tol,
                                            𝕊ℂ = ℂ.sylvester_caches,
                                            acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                            verbose = opts.verbose) # timer = timer)

    # end # timeit_debug
    # @timeit_debug timer "Post-process" begin

    if !solved
        return (𝐒₂, solved), x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    # end # timeit_debug

    # sp⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t = choose_matrix_format(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋', density_threshold = 1.0)

    # sp𝐒₁₊╱𝟎t = choose_matrix_format(𝐒₁₊╱𝟎', density_threshold = 1.0)

    𝛔t = choose_matrix_format(M₂.𝛔', density_threshold = 1.0)

    𝐔₂t = choose_matrix_format(M₂.𝐔₂', density_threshold = 1.0)

    𝐂₂t = choose_matrix_format(M₂.𝐂₂', density_threshold = 1.0)

    ∇₂t = choose_matrix_format(∇₂', density_threshold = 1.0)

    # end # timeit_debug

    function second_order_solution_pullback(∂𝐒₂_solved) 
        # @timeit_debug timer "Second order solution - pullback" begin
            
        # @timeit_debug timer "Preallocate" begin
        ∂∇₂ = zeros(size(∇₂))
        ∂∇₁ = zero(∇₁)
        ∂𝐒₁ = zero(𝐒₁)
        ∂spinv = zero(∇₁₊𝐒₁➕∇₁₀)
        ∂𝐒₁₋╱𝟏ₑ = zeros(size(𝐒₁₋╱𝟏ₑ))
        ∂𝐒₁₊╱𝟎 = zeros(size(𝐒₁₊╱𝟎))
        ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = zeros(size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋))

        # end # timeit_debug

        ∂𝐒₂ = ∂𝐒₂_solved[1]
        
        # ∂𝐒₂ *= 𝐔₂t

        # @timeit_debug timer "Sylvester" begin
        if ℒ.norm(∂𝐒₂) < opts.tol.sylvester_tol
            return (𝐒₂, false), x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        
        ∂C, solved = solve_sylvester_equation(A', B', ∂𝐒₂,
                                                sylvester_algorithm = opts.sylvester_algorithm²,
                                                tol = opts.tol.sylvester_tol,
                                                𝕊ℂ = ℂ.sylvester_caches,
                                                acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                                verbose = opts.verbose)

        if !solved
            return (𝐒₂, solved), x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        
        # end # timeit_debug

        # @timeit_debug timer "Matmul" begin

        ∂C = choose_matrix_format(∂C) # Dense

        ∂A = ∂C * B' * 𝐒₂' # Dense

        ∂B = 𝐒₂' * A' * ∂C # Dense

        # B = (M₂.𝐔₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + M₂.𝐔₂ * M₂.𝛔) * M₂.𝐂₂
        ∂kron𝐒₁₋╱𝟏ₑ = 𝐔₂t * ∂B * 𝐂₂t

        # end # timeit_debug

        # @timeit_debug timer "Kron adjoint" begin

        fill_kron_adjoint!(∂𝐒₁₋╱𝟏ₑ, ∂𝐒₁₋╱𝟏ₑ, ∂kron𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ)

        # end # timeit_debug

        # @timeit_debug timer "Matmul2" begin

        # A = spinv * ∇₁₊
        ∂∇₁₊ = spinv' * ∂A
        ∂spinv += ∂A * ∇₁₊'
        
        # ∇₁₊ =  sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])
        ∂∇₁[:,1:n₊] += ∂∇₁₊ * ℒ.I(n)[:,i₊]

        # C = spinv * ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹
        ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹𝐂₂ = spinv' * ∂C * 𝐂₂t
        
        ∂spinv += ∂C * ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹'

        # end # timeit_debug

        # @timeit_debug timer "Matmul3" begin

        # ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = ∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) * M₂.𝐂₂  + ∇₂ * ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔 * M₂.𝐂₂
        # kron⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = choose_matrix_format(ℒ.kron(sp⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t, sp⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t), density_threshold = 1.0)

        # 𝛔kron𝐒₁₊╱𝟎 = choose_matrix_format(𝛔t * ℒ.kron(sp𝐒₁₊╱𝟎t, sp𝐒₁₊╱𝟎t), density_threshold = 1.0)

        # ℒ.mul!(∂∇₂, ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹𝐂₂, 𝛔kron𝐒₁₊╱𝟎, 1, 1)
        
        # ℒ.mul!(∂∇₂, ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹𝐂₂, kron⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 1, 1)

        ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹𝐂₂ = choose_matrix_format(∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹𝐂₂, density_threshold = 1.0)

        ∂∇₂ += mat_mult_kron(∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹𝐂₂ * 𝛔t, 𝐒₁₊╱𝟎', 𝐒₁₊╱𝟎')
        
        ∂∇₂ += mat_mult_kron(∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹𝐂₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋', ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋')
        
        # end # timeit_debug

        # @timeit_debug timer "Matmul4" begin

        ∂kron𝐒₁₊╱𝟎 = ∇₂t * ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹𝐂₂ * 𝛔t

        # end # timeit_debug

        # @timeit_debug timer "Kron adjoint 2" begin

        fill_kron_adjoint!(∂𝐒₁₊╱𝟎, ∂𝐒₁₊╱𝟎, ∂kron𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)
        
        # end # timeit_debug

        ∂kron⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = ∇₂t * ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹𝐂₂

        # @timeit_debug timer "Kron adjoint 3" begin

        fill_kron_adjoint!(∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ∂kron⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) # filling dense is much faster

        # end # timeit_debug

        # @timeit_debug timer "Matmul5" begin

        # spinv = sparse(inv(∇₁₊𝐒₁➕∇₁₀))
        ∂∇₁₊𝐒₁➕∇₁₀ = -spinv' * ∂spinv * spinv'

        # ∇₁₊𝐒₁➕∇₁₀ =  -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]
        ∂∇₁[:,1:n₊] -= ∂∇₁₊𝐒₁➕∇₁₀ * ℒ.I(n)[:,i₋] * 𝐒₁[i₊,1:n₋]'
        ∂∇₁[:,range(1,n) .+ n₊] -= ∂∇₁₊𝐒₁➕∇₁₀

        ∂𝐒₁[i₊,1:n₋] -= ∇₁[:,1:n₊]' * ∂∇₁₊𝐒₁➕∇₁₀ * ℒ.I(n)[:,i₋]

        # 𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
        #                 zeros(n₋ + n + nₑ, nₑ₋)];
        ∂𝐒₁[i₊,:] += ∂𝐒₁₊╱𝟎[1:length(i₊),:]

        ###### ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ =  [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
        # ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ =  [ℒ.I(size(𝐒₁,1))[i₊,:] * 𝐒₁ * 𝐒₁₋╱𝟏ₑ
        #                     𝐒₁
        #                     spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];
        ∂𝐒₁ += ℒ.I(size(𝐒₁,1))[:,i₊] * ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[1:length(i₊),:] * 𝐒₁₋╱𝟏ₑ'
        ∂𝐒₁ += ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[length(i₊) .+ (1:size(𝐒₁,1)),:]
        
        ∂𝐒₁₋╱𝟏ₑ += 𝐒₁' * ℒ.I(size(𝐒₁,1))[:,i₊] * ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[1:length(i₊),:]

        # 𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];
        ∂𝐒₁[i₋,:] += ∂𝐒₁₋╱𝟏ₑ[1:length(i₋), :]

        # 𝐒₁ = [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]]
        ∂𝑺₁ = [∂𝐒₁[:,1:n₋] ∂𝐒₁[:,n₋+2:end]]

        # end # timeit_debug

        # end # timeit_debug

        return NoTangent(), ∂∇₁, ∂∇₂, ∂𝑺₁, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    

    # return (sparse(𝐒₂ * M₂.𝐔₂), solved), second_order_solution_pullback
    return (𝐒₂, solved), second_order_solution_pullback
end

# Lines 1015-1583
function rrule(::typeof(calculate_third_order_solution), 
                ∇₁::AbstractMatrix{S}, #first order derivatives
                ∇₂::SparseMatrixCSC{S}, #second order derivatives
                ∇₃::SparseMatrixCSC{S}, #third order derivatives
                𝑺₁::AbstractMatrix{S}, #first order solution
                𝐒₂::SparseMatrixCSC{S}, #second order solution
                M₂::second_order_auxiliary_matrices,  # aux matrices second order
                M₃::third_order_auxiliary_matrices,   # aux matrices third order
                ℂC::caches;
                T::timings,
                initial_guess::AbstractMatrix{Float64} = zeros(0,0),
                opts::CalculationOptions = merge_calculation_options()) where S <: AbstractFloat 
    if !(eltype(ℂC.third_order_caches.Ŝ) == S)
        ℂC.third_order_caches = Higher_order_caches(T = S)
    end
    ℂ = ℂC.third_order_caches

    # @timeit_debug timer "Third order solution - forward" begin
    # inspired by Levintal

    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx;
    i₋ = T.past_not_future_and_mixed_idx;

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo;
    n = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    # @timeit_debug timer "Setup matrices" begin

    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]]# |> sparse
    
    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) ℒ.I(nₑ + 1)[1,:] zeros(nₑ + 1, nₑ)]

    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 1.0, min_length = 10)

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                ℒ.I(nₑ₋)[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]] #|> sparse

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)]# |> sparse
    𝐒₁₊╱𝟎 = choose_matrix_format(𝐒₁₊╱𝟎, density_threshold = 1.0, min_length = 10)

    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.I(n)[i₋,:] - ∇₁[:,range(1,n) .+ n₊]

    # end # timeit_debug
    # @timeit_debug timer "Invert matrix" begin

    ∇₁₊𝐒₁➕∇₁₀lu = ℒ.lu(∇₁₊𝐒₁➕∇₁₀, check = false)

    if !ℒ.issuccess(∇₁₊𝐒₁➕∇₁₀lu)
        if opts.verbose println("Second order solution: inversion failed") end
        return (∇₁₊𝐒₁➕∇₁₀, solved), x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    spinv = inv(∇₁₊𝐒₁➕∇₁₀lu)
    spinv = choose_matrix_format(spinv)

    # end # timeit_debug
    
    ∇₁₊ = @views ∇₁[:,1:n₊] * ℒ.I(n)[i₊,:]

    A = spinv * ∇₁₊

    # tmpkron = ℒ.kron(𝐒₁₋╱𝟏ₑ,M₂.𝛔)
    tmpkron = choose_matrix_format(ℒ.kron(𝐒₁₋╱𝟏ₑ,M₂.𝛔), density_threshold = 1.0, tol = opts.tol.droptol)
    kron𝐒₁₋╱𝟏ₑ = ℒ.kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ)
    
    # @timeit_debug timer "Setup B" begin
    # @timeit_debug timer "Add tmpkron" begin

    B = tmpkron

    # end # timeit_debug
    # @timeit_debug timer "Step 1" begin

    B += M₃.𝐏₁ₗ̄ * tmpkron * M₃.𝐏₁ᵣ̃

    # end # timeit_debug
    # @timeit_debug timer "Step 2" begin

    B += M₃.𝐏₂ₗ̄ * tmpkron * M₃.𝐏₂ᵣ̃

    # end # timeit_debug
    # @timeit_debug timer "Mult" begin

    B *= M₃.𝐂₃
    B = choose_matrix_format(M₃.𝐔₃ * B, tol = opts.tol.droptol, multithreaded = false)

    # end # timeit_debug
    # @timeit_debug timer "3rd Kronecker power" begin

    B += compressed_kron³(𝐒₁₋╱𝟏ₑ, tol = opts.tol.droptol, sparse_preallocation = ℂ.tmp_sparse_prealloc1)#, timer = timer)

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Setup C" begin
    # @timeit_debug timer "Initialise smaller matrices" begin

    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [(𝐒₂ * kron𝐒₁₋╱𝟏ₑ + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
            𝐒₂
            zeros(n₋ + nₑ, nₑ₋^2)];
            
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = choose_matrix_format(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, density_threshold = 0.0, min_length = 10, tol = opts.tol.droptol)
        
    𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:] 
            zeros(n₋ + n + nₑ, nₑ₋^2)];

    aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋

    # end # timeit_debug
    # @timeit_debug timer "∇₃" begin

    # tmpkron0 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)
    # tmpkron22 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, tmpkron0 * M₂.𝛔)

    if length(ℂ.tmpkron0) > 0 && eltype(ℂ.tmpkron0) == S
        ℒ.kron!(ℂ.tmpkron0, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)
    else
        ℂ.tmpkron0 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)
    end

    if length(ℂ.tmpkron22) > 0 && eltype(ℂ.tmpkron22) == S
        ℒ.kron!(ℂ.tmpkron22, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℂ.tmpkron0 * M₂.𝛔)
    else
        ℂ.tmpkron22 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℂ.tmpkron0 * M₂.𝛔)
    end

    𝐔∇₃ = ∇₃ * M₃.𝐔∇₃

    𝐗₃ = 𝐔∇₃ * ℂ.tmpkron22 + 𝐔∇₃ * M₃.𝐏₁ₗ̂ * ℂ.tmpkron22 * M₃.𝐏₁ᵣ̃ + 𝐔∇₃ * M₃.𝐏₂ₗ̂ * ℂ.tmpkron22 * M₃.𝐏₂ᵣ̃
    
    # end # timeit_debug
    # @timeit_debug timer "∇₂ & ∇₁₊" begin

    𝐒₂₊╱𝟎 = choose_matrix_format(𝐒₂₊╱𝟎, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    if length(ℂ.tmpkron1) > 0 && eltype(ℂ.tmpkron1) == S
        ℒ.kron!(ℂ.tmpkron1, 𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎)
    else
        ℂ.tmpkron1 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎)
    end

    if length(ℂ.tmpkron2) > 0 && eltype(ℂ.tmpkron2) == S
        ℒ.kron!(ℂ.tmpkron2, M₂.𝛔, 𝐒₁₋╱𝟏ₑ)
    else
        ℂ.tmpkron2 = ℒ.kron(M₂.𝛔, 𝐒₁₋╱𝟏ₑ)
    end
    
    ∇₁₊ = choose_matrix_format(∇₁₊, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    𝐒₂₋╱𝟎 = [𝐒₂[i₋,:] ; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]

    𝐒₂₋╱𝟎 = choose_matrix_format(𝐒₂₋╱𝟎, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    # @timeit_debug timer "Step 1" begin
    out2 = ∇₂ * ℂ.tmpkron1 * ℂ.tmpkron2 # this help

    # end # timeit_debug
    # @timeit_debug timer "Step 2" begin

    # end # timeit_debug  
    # @timeit_debug timer "Step 3" begin

    out2 += ∇₂ * ℂ.tmpkron1 * M₃.𝐏₁ₗ * ℂ.tmpkron2 * M₃.𝐏₁ᵣ# |> findnz

    # end # timeit_debug
    # @timeit_debug timer "Step 4" begin

    out2 += mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc2)# |> findnz

    # out2 += ∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₂₊╱𝟎 * M₂.𝛔)# |> findnz
    𝐒₂₊╱𝟎𝛔 = 𝐒₂₊╱𝟎 * M₂.𝛔
    
    if length(ℂ.tmpkron11) > 0 && eltype(ℂ.tmpkron11) == S
        ℒ.kron!(ℂ.tmpkron11, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₂₊╱𝟎𝛔)
    else
        ℂ.tmpkron11 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₂₊╱𝟎𝛔)
    end
    out2 += ∇₂ * ℂ.tmpkron11# |> findnz

    # end # timeit_debug
    # @timeit_debug timer "Step 5" begin

    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0, tol = opts.tol.droptol)
    if length(ℂ.tmpkron12) > 0 && eltype(ℂ.tmpkron12) == S
        ℒ.kron!(ℂ.tmpkron12, 𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)
    else
        ℂ.tmpkron12 = ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)
    end
    out2 += ∇₁₊ * 𝐒₂ * ℂ.tmpkron12

    # end # timeit_debug
    # @timeit_debug timer "Mult" begin

    𝐗₃ += out2 * M₃.𝐏

    𝐗₃ *= M₃.𝐂₃

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "3rd Kronecker power aux" begin

    # 𝐗₃ += mat_mult_kron(∇₃, collect(aux), collect(ℒ.kron(aux, aux)), M₃.𝐂₃) # slower than direct compression
    𝐗₃ += ∇₃ * compressed_kron³(aux, rowmask = unique(findnz(∇₃)[2]), tol = opts.tol.droptol, sparse_preallocation = ℂ.tmp_sparse_prealloc3) #, timer = timer)
    𝐗₃ = choose_matrix_format(𝐗₃, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    # end # timeit_debug
    # @timeit_debug timer "Mult 2" begin

    C = spinv * 𝐗₃

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Solve sylvester equation" begin

    𝐒₃, solved = solve_sylvester_equation(A, B, C, 
                                            initial_guess = initial_guess,
                                            sylvester_algorithm = opts.sylvester_algorithm³,
                                            tol = opts.tol.sylvester_tol,
                                            𝕊ℂ = ℂ.sylvester_caches,
                                            acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                            verbose = opts.verbose) # timer = timer)
    
    # end # timeit_debug
    # # @timeit_debug timer "Refine sylvester equation" begin

    # if !solved
    #     𝐒₃, solved = solve_sylvester_equation(A, B, C, 
    #                                             sylvester_algorithm = :doubling, 
    #                                             initial_guess = initial_guess,
    #                                             verbose = verbose,
    #                                             # tol = tol,
    #                                             timer = timer)
    # end

    if !solved
        return (𝐒₃, solved), x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent() 
    end

    𝐒₃ = choose_matrix_format(𝐒₃, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    # # end # timeit_debug

    # @timeit_debug timer "Preallocate for pullback" begin

    # At = choose_matrix_format(A')# , density_threshold = 1.0)

    # Bt = choose_matrix_format(B')# , density_threshold = 1.0)
    
    𝐂₃t = choose_matrix_format(M₃.𝐂₃')# , density_threshold = 1.0)

    𝐔₃t = choose_matrix_format(M₃.𝐔₃')# , density_threshold = 1.0)

    𝐏t = choose_matrix_format(M₃.𝐏')# , density_threshold = 1.0)

    𝐏₁ᵣt = choose_matrix_format(M₃.𝐏₁ᵣ')# , density_threshold = 1.0)
    
    𝐏₁ₗt = choose_matrix_format(M₃.𝐏₁ₗ')# , density_threshold = 1.0)

    M₃𝐔∇₃t = choose_matrix_format(M₃.𝐔∇₃')# , density_threshold = 1.0)
    
    𝐔∇₃t = choose_matrix_format(𝐔∇₃')# , density_threshold = 1.0)
    
    M₃𝐏₂ₗ̂t = choose_matrix_format(M₃.𝐏₂ₗ̂')# , density_threshold = 1.0)
    
    M₃𝐏₂ᵣ̃t = choose_matrix_format(M₃.𝐏₂ᵣ̃')# , density_threshold = 1.0)
    
    M₃𝐏₁ᵣ̃t = choose_matrix_format(M₃.𝐏₁ᵣ̃')# , density_threshold = 1.0)
    
    M₃𝐏₁ₗ̂t = choose_matrix_format(M₃.𝐏₁ₗ̂')# , density_threshold = 1.0)

    𝛔t = choose_matrix_format(M₂.𝛔')# , density_threshold = 1.0)

    ∇₂t = choose_matrix_format(∇₂')# , density_threshold = 1.0)

    tmpkron1t = choose_matrix_format(ℂ.tmpkron1')# , density_threshold = 1.0)
    
    tmpkron2t = choose_matrix_format(ℂ.tmpkron2')# , density_threshold = 1.0)
    
    tmpkron22t = choose_matrix_format(ℂ.tmpkron22')# , density_threshold = 1.0)
    
    tmpkron12t = choose_matrix_format(ℂ.tmpkron12')# , density_threshold = 1.0)
    
    𝐒₂t = choose_matrix_format(𝐒₂', density_threshold = 1.0) # this must be sparse otherwise tests fail
    
    kronaux = ℒ.kron(aux, aux)

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t = choose_matrix_format(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋')
    
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎t = choose_matrix_format(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎')
    
    tmpkron10t = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎t)

    # end # timeit_debug
    # end # timeit_debug
    
    function third_order_solution_pullback(∂𝐒₃_solved) 
        ∂∇₁ = zero(∇₁)
        ∂∇₂ = zero(∇₂)
        # ∂𝐔∇₃ = zero(𝐔∇₃)
        ∂∇₃ = zero(∇₃)
        ∂𝐒₁ = zero(𝐒₁)
        ∂𝐒₂ = zero(𝐒₂)
        ∂spinv = zero(spinv)
        ∂𝐒₁₋╱𝟏ₑ = zero(𝐒₁₋╱𝟏ₑ)
        ∂kron𝐒₁₋╱𝟏ₑ = zero(kron𝐒₁₋╱𝟏ₑ)
        ∂𝐒₁₊╱𝟎 = zero(𝐒₁₊╱𝟎)
        ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = zero(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋)
        ∂tmpkron = zero(tmpkron)
        ∂tmpkron22 = zero(ℂ.tmpkron22)
        ∂kronaux = zero(kronaux)
        ∂aux = zero(aux)
        ∂tmpkron0 = zero(ℂ.tmpkron0)
        ∂⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = zero(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)
        ∂𝐒₂₊╱𝟎 = zero(𝐒₂₊╱𝟎)
        ∂𝐒₂₊╱𝟎𝛔 = zero(𝐒₂₊╱𝟎𝛔)
        ∂∇₁₊ = zero(∇₁₊)
        ∂𝐒₂₋╱𝟎 = zero(𝐒₂₋╱𝟎)

        # @timeit_debug timer "Third order solution - pullback" begin

        # @timeit_debug timer "Solve sylvester equation" begin

        ∂𝐒₃ = ∂𝐒₃_solved[1]

        # ∂𝐒₃ *= 𝐔₃t
        
        ∂C, solved = solve_sylvester_equation(A', B', ∂𝐒₃,
                                                sylvester_algorithm = opts.sylvester_algorithm³,
                                                tol = opts.tol.sylvester_tol,
                                                𝕊ℂ = ℂ.sylvester_caches,
                                                acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                                verbose = opts.verbose)

        if !solved
            return (𝐒₃, solved), x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent() 
        end

        ∂C = choose_matrix_format(∂C, density_threshold = 1.0, min_length = 0)

        # end # timeit_debug
        # @timeit_debug timer "Step 0" begin

        ∂A = ∂C * B' * 𝐒₃'

        # ∂B = 𝐒₃' * A' * ∂C
        ∂B = choose_matrix_format(𝐒₃' * A' * ∂C, density_threshold = 1.0, min_length = 0)

        # end # timeit_debug
        # @timeit_debug timer "Step 1" begin

        # C = spinv * 𝐗₃
        # ∂𝐗₃ = spinv' * ∂C * M₃.𝐂₃'
        ∂𝐗₃ = choose_matrix_format(spinv' * ∂C, density_threshold = 1.0, min_length = 0)

        ∂spinv += ∂C * 𝐗₃'

        # 𝐗₃ = ∇₃ * compressed_kron³(aux, rowmask = unique(findnz(∇₃)[2]))
        # + (𝐔∇₃ * tmpkron22 
        # + 𝐔∇₃ * M₃.𝐏₁ₗ̂ * tmpkron22 * M₃.𝐏₁ᵣ̃ 
        # + 𝐔∇₃ * M₃.𝐏₂ₗ̂ * tmpkron22 * M₃.𝐏₂ᵣ̃
        # + ∇₂ * (tmpkron10 + tmpkron1 * tmpkron2 + tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ + ℂ.tmpkron11) * M₃.𝐏
        # + ∇₁₊ * 𝐒₂ * ℂ.tmpkron12 * M₃.𝐏) * M₃.𝐂₃

        # ∇₁₊ * 𝐒₂ * ℂ.tmpkron12 * M₃.𝐏 * M₃.𝐂₃
        ∂∇₁₊ += ∂𝐗₃ * 𝐂₃t * 𝐏t * tmpkron12t * 𝐒₂t
        ∂𝐒₂ += ∇₁₊' * ∂𝐗₃ * 𝐂₃t * 𝐏t * tmpkron12t
        ∂tmpkron12 = 𝐒₂t * ∇₁₊' * ∂𝐗₃ * 𝐂₃t * 𝐏t

        # ℂ.tmpkron12 = ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)
        fill_kron_adjoint!(∂𝐒₁₋╱𝟏ₑ, ∂𝐒₂₋╱𝟎, ∂tmpkron12, 𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)
        
        # end # timeit_debug
        # @timeit_debug timer "Step 2" begin
        
        # ∇₂ * (tmpkron10 + tmpkron1 * tmpkron2 + tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ + ℂ.tmpkron11) * M₃.𝐏 * M₃.𝐂₃
        #improve this
        # ∂∇₂ += ∂𝐗₃ * 𝐂₃t * 𝐏t * (
        #    tmpkron10
        #  + tmpkron1 * tmpkron2
        #  + tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ
        #  + ℂ.tmpkron11
        #  )'

        ∂∇₂ += ∂𝐗₃ * 𝐂₃t * 𝐏t * tmpkron10t
        # ∂∇₂ += mat_mult_kron(∂𝐗₃ * 𝐂₃t * 𝐏t, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎t)
        # ∂∇₂ += ∂𝐗₃ * 𝐂₃t * 𝐏t * (tmpkron1 * tmpkron2)'
        ∂∇₂ += ∂𝐗₃ * 𝐂₃t * 𝐏t * tmpkron2t * tmpkron1t

        # ∂∇₂ += ∂𝐗₃ * 𝐂₃t * 𝐏t * (tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ)'
        ∂∇₂ += ∂𝐗₃ * 𝐂₃t * 𝐏t * M₃.𝐏₁ᵣ' * tmpkron2t * M₃.𝐏₁ₗ' * tmpkron1t

        ∂∇₂ += ∂𝐗₃ * 𝐂₃t * 𝐏t * ℂ.tmpkron11'

        ∂tmpkron10 = ∇₂t * ∂𝐗₃ * 𝐂₃t * 𝐏t

        # end # timeit_debug
        # @timeit_debug timer "Step 3" begin
        
        # tmpkron10 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)
        fill_kron_adjoint!(∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ∂⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, ∂tmpkron10, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)

        ∂tmpkron11 = ∇₂t * ∂𝐗₃ * 𝐂₃t * 𝐏t

        ∂tmpkron1 = ∂tmpkron11 * tmpkron2t + ∂tmpkron11 * 𝐏₁ᵣt * tmpkron2t * 𝐏₁ₗt

        ∂tmpkron2 = tmpkron1t * ∂tmpkron11

        ∂tmpkron2 += 𝐏₁ₗt * ∂tmpkron2 * 𝐏₁ᵣt

        # ∂tmpkron1 = ∇₂t * ∂𝐗₃ * 𝐂₃t * 𝐏t * tmpkron2t + ∇₂t * ∂𝐗₃ * 𝐂₃t * 𝐏t * 𝐏₁ᵣt * tmpkron2t * 𝐏₁ₗt
        # #improve this
        # ∂tmpkron2 = tmpkron1t * ∇₂t * ∂𝐗₃ * 𝐂₃t * 𝐏t + 𝐏₁ₗt * tmpkron1t * ∇₂t * ∂𝐗₃ * 𝐂₃t * 𝐏t * 𝐏₁ᵣt

        # ∂tmpkron11 = ∇₂t * ∂𝐗₃ * 𝐂₃t * 𝐏t

        # tmpkron1 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎)
        fill_kron_adjoint!(∂𝐒₁₊╱𝟎, ∂𝐒₂₊╱𝟎, ∂tmpkron1, 𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎)

        # tmpkron2 = ℒ.kron(M₂.𝛔, 𝐒₁₋╱𝟏ₑ)
        fill_kron_adjoint_∂B!(∂tmpkron2, ∂𝐒₁₋╱𝟏ₑ, M₂.𝛔)

        # tmpkron11 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₂₊╱𝟎𝛔)
        fill_kron_adjoint!(∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ∂𝐒₂₊╱𝟎𝛔, ∂tmpkron11, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₂₊╱𝟎𝛔)
        
        ∂𝐒₂₊╱𝟎 += ∂𝐒₂₊╱𝟎𝛔 * 𝛔t

        # end # timeit_debug
        # @timeit_debug timer "Step 4" begin

        # out = (𝐔∇₃ * tmpkron22 
        # + 𝐔∇₃ * M₃.𝐏₁ₗ̂ * tmpkron22 * M₃.𝐏₁ᵣ̃ 
        # + 𝐔∇₃ * M₃.𝐏₂ₗ̂ * tmpkron22 * M₃.𝐏₂ᵣ̃ ) * M₃.𝐂₃

        ∂∇₃ += ∂𝐗₃ * 𝐂₃t * tmpkron22t * M₃𝐔∇₃t + ∂𝐗₃ * 𝐂₃t * M₃𝐏₁ᵣ̃t * tmpkron22t * M₃𝐏₁ₗ̂t * M₃𝐔∇₃t + ∂𝐗₃ * 𝐂₃t * M₃𝐏₂ᵣ̃t * tmpkron22t * M₃𝐏₂ₗ̂t * M₃𝐔∇₃t

        ∂tmpkron22 += 𝐔∇₃t * ∂𝐗₃ * 𝐂₃t + M₃𝐏₁ₗ̂t * 𝐔∇₃t * ∂𝐗₃ * 𝐂₃t * M₃𝐏₁ᵣ̃t + M₃𝐏₂ₗ̂t * 𝐔∇₃t * ∂𝐗₃ * 𝐂₃t * M₃𝐏₂ᵣ̃t

        # tmpkron22 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔)
        fill_kron_adjoint!(∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ∂tmpkron0, ∂tmpkron22, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℂ.tmpkron0 * M₂.𝛔)

        ∂kron𝐒₁₊╱𝟎 = ∂tmpkron0 * 𝛔t

        fill_kron_adjoint!(∂𝐒₁₊╱𝟎, ∂𝐒₁₊╱𝟎, ∂kron𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)

        # -∇₃ * ℒ.kron(ℒ.kron(aux, aux), aux)
        # ∂∇₃ += ∂𝐗₃ * ℒ.kron(ℒ.kron(aux', aux'), aux')
        # A_mult_kron_power_3_B!(∂∇₃, ∂𝐗₃, aux') # not a good idea because filling an existing matrix one by one is slow
        # ∂∇₃ += A_mult_kron_power_3_B(∂𝐗₃, aux') # this is slower somehow
        
        # end # timeit_debug
        # @timeit_debug timer "Step 5" begin

        # this is very slow
        ∂∇₃ += ∂𝐗₃ * compressed_kron³(aux', rowmask = unique(findnz(∂𝐗₃)[2]), sparse_preallocation = ℂ.tmp_sparse_prealloc4) # , timer = timer)
        # ∂∇₃ += ∂𝐗₃ * ℒ.kron(aux', aux', aux')
        
        # end # timeit_debug
        # @timeit_debug timer "Step 6" begin

        ∂kronkronaux = 𝐔∇₃t * ∂𝐗₃ * 𝐂₃t

        fill_kron_adjoint!(∂kronaux, ∂aux, ∂kronkronaux, kronaux, aux)

        fill_kron_adjoint!(∂aux, ∂aux, ∂kronaux, aux, aux)

        # end # timeit_debug
        # @timeit_debug timer "Step 7" begin

        # aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋
        ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ += M₃.𝐒𝐏' * ∂aux

        # 𝐒₂₋╱𝟎 = @views [𝐒₂[i₋,:] ; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]
        ∂𝐒₂[i₋,:] += ∂𝐒₂₋╱𝟎[1:length(i₋),:]

        # 𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:] 
        #     zeros(n₋ + n + nₑ, nₑ₋^2)]
        ∂𝐒₂[i₊,:] += ∂𝐒₂₊╱𝟎[1:length(i₊),:]


        # ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = [
            ## (𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
            ## ℒ.diagm(ones(n))[i₊,:] * (𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])
            # ℒ.diagm(ones(n))[i₊,:] * 𝐒₂k𝐒₁₋╱𝟏ₑ
            # 𝐒₂
            # zeros(n₋ + nₑ, nₑ₋^2)
        # ];
        ∂𝐒₂k𝐒₁₋╱𝟏ₑ = ℒ.diagm(ones(n))[i₊,:]' * ∂⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎[1:length(i₊),:]

        ∂𝐒₂ += ∂⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎[length(i₊) .+ (1:size(𝐒₂,1)),:]

        ∂𝐒₂ += ∂𝐒₂k𝐒₁₋╱𝟏ₑ * kron𝐒₁₋╱𝟏ₑ'

        ∂kron𝐒₁₋╱𝟏ₑ += 𝐒₂t * ∂𝐒₂k𝐒₁₋╱𝟏ₑ

        
        # 𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)]
        # 𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐒₁ * 𝐒₂₋╱𝟎
        ∂𝐒₁ += ∂𝐒₂k𝐒₁₋╱𝟏ₑ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)]'
        
        # ∂𝐒₂[i₋,:] += spdiagm(ones(size(𝐒₂,1)))[i₋,:]' * 𝐒₁' * ∂𝐒₂k𝐒₁₋╱𝟏ₑ[1:length(i₋),:]
        ∂𝐒₂╱𝟎 = 𝐒₁' * ∂𝐒₂k𝐒₁₋╱𝟏ₑ
        ∂𝐒₂[i₋,:] += ∂𝐒₂╱𝟎[1:length(i₋),:]

        # end # timeit_debug
        # @timeit_debug timer "Step 8" begin

        ###
        # B = M₃.𝐔₃ * (tmpkron + M₃.𝐏₁ₗ̄ * tmpkron * M₃.𝐏₁ᵣ̃ + M₃.𝐏₂ₗ̄ * tmpkron * M₃.𝐏₂ᵣ̃ + ℒ.kron(𝐒₁₋╱𝟏ₑ, kron𝐒₁₋╱𝟏ₑ)) * M₃.𝐂₃
        ∂tmpkron += 𝐔₃t * ∂B * 𝐂₃t
        ∂tmpkron += M₃.𝐏₁ₗ̄' * 𝐔₃t * ∂B * 𝐂₃t * M₃𝐏₁ᵣ̃t
        ∂tmpkron += M₃.𝐏₂ₗ̄' * 𝐔₃t * ∂B * 𝐂₃t * M₃𝐏₂ᵣ̃t

        ∂kronkron𝐒₁₋╱𝟏ₑ = 𝐔₃t * ∂B * 𝐂₃t

        fill_kron_adjoint!(∂𝐒₁₋╱𝟏ₑ, ∂kron𝐒₁₋╱𝟏ₑ, ∂kronkron𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ, kron𝐒₁₋╱𝟏ₑ)
        
        fill_kron_adjoint!(∂𝐒₁₋╱𝟏ₑ, ∂𝐒₁₋╱𝟏ₑ, ∂kron𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ)

        # tmpkron = ℒ.kron(𝐒₁₋╱𝟏ₑ,M₂.𝛔)
        fill_kron_adjoint_∂A!(∂tmpkron, ∂𝐒₁₋╱𝟏ₑ, M₂.𝛔)
        # A = spinv * ∇₁₊
        ∂∇₁₊ += spinv' * ∂A
        ∂spinv += ∂A * ∇₁₊'
        
        # ∇₁₊ =  sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])
        ∂∇₁[:,1:n₊] += ∂∇₁₊ * ℒ.I(n)[:,i₊]

        # spinv = sparse(inv(∇₁₊𝐒₁➕∇₁₀))
        ∂∇₁₊𝐒₁➕∇₁₀ = -spinv' * ∂spinv * spinv'

        # ∇₁₊𝐒₁➕∇₁₀ =  -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]
        ∂∇₁[:,1:n₊] -= ∂∇₁₊𝐒₁➕∇₁₀ * ℒ.I(n)[:,i₋] * 𝐒₁[i₊,1:n₋]'
        ∂∇₁[:,range(1,n) .+ n₊] -= ∂∇₁₊𝐒₁➕∇₁₀

        ∂𝐒₁[i₊,1:n₋] -= ∇₁[:,1:n₊]' * ∂∇₁₊𝐒₁➕∇₁₀ * ℒ.I(n)[:,i₋]

        # # 𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
        # #                 zeros(n₋ + n + nₑ, nₑ₋)];
        ∂𝐒₁[i₊,:] += ∂𝐒₁₊╱𝟎[1:length(i₊),:]

        # ###### ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ =  [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
        # # ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ =  [ℒ.I(size(𝐒₁,1))[i₊,:] * 𝐒₁ * 𝐒₁₋╱𝟏ₑ
        # #                     𝐒₁
        # #                     spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];
        ∂𝐒₁ += ℒ.I(size(𝐒₁,1))[:,i₊] * ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[1:length(i₊),:] * 𝐒₁₋╱𝟏ₑ'
        ∂𝐒₁ += ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[length(i₊) .+ (1:size(𝐒₁,1)),:]
        
        ∂𝐒₁₋╱𝟏ₑ += 𝐒₁' * ℒ.I(size(𝐒₁,1))[:,i₊] * ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[1:length(i₊),:]

        # 𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];
        ∂𝐒₁[i₋,:] += ∂𝐒₁₋╱𝟏ₑ[1:length(i₋), :]

        # 𝐒₁ = [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]]
        ∂𝑺₁ = [∂𝐒₁[:,1:n₋] ∂𝐒₁[:,n₋+2:end]]

        # end # timeit_debug
        # end # timeit_debug

        return NoTangent(), ∂∇₁, ∂∇₂, ∂∇₃, ∂𝑺₁, ∂𝐒₂, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return (𝐒₃, solved), third_order_solution_pullback
end

# ========== src/algorithms/lyapunov.jl ==========

# Lines 83-109
function rrule(::typeof(solve_lyapunov_equation),
                A::AbstractMatrix{Float64},
                C::AbstractMatrix{Float64};
                lyapunov_algorithm::Symbol = :doubling,
                tol::AbstractFloat = 1e-14,
                acceptance_tol::AbstractFloat = 1e-12,
                # timer::TimerOutput = TimerOutput(),
                verbose::Bool = false)

    P, solved = solve_lyapunov_equation(A, C, lyapunov_algorithm = lyapunov_algorithm, tol = tol, verbose = verbose)

    # pullback 
    # https://arxiv.org/abs/2011.11430  
    function solve_lyapunov_equation_pullback(∂P)
        if ℒ.norm(∂P[1]) < tol return NoTangent(), NoTangent(), NoTangent(), NoTangent() end

        ∂C, slvd = solve_lyapunov_equation(A', ∂P[1], lyapunov_algorithm = lyapunov_algorithm,  tol = tol, verbose = verbose)
    
        solved = solved && slvd

        ∂A = ∂C * A * P' + ∂C' * A * P

        return NoTangent(), ∂A, ∂C, NoTangent()
    end
    
    return (P, solved), solve_lyapunov_equation_pullback
end

# ========== src/algorithms/sylvester.jl ==========

# Lines 239-279
function rrule(::typeof(solve_sylvester_equation),
    A::M,
    B::N,
    C::O;
    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
    sylvester_algorithm::Symbol = :doubling,
    acceptance_tol::AbstractFloat = 1e-10,
    tol::AbstractFloat = 1e-14,
    𝕊ℂ::sylvester_caches = Sylvester_caches(),
    # timer::TimerOutput = TimerOutput(),
    verbose::Bool = false) where {M <: AbstractMatrix{Float64}, N <: AbstractMatrix{Float64}, O <: AbstractMatrix{Float64}}

    P, solved = solve_sylvester_equation(A, B, C, 
                                        sylvester_algorithm = sylvester_algorithm, 
                                        tol = tol, 
                                        𝕊ℂ = 𝕊ℂ,
                                        verbose = verbose, 
                                        initial_guess = initial_guess)

                                        println("C norm: $(ℒ.norm(C))")
    # pullback
    function solve_sylvester_equation_pullback(∂P)
        if ℒ.norm(∂P[1]) < tol return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent() end

        ∂C, slvd = solve_sylvester_equation(A', B', ∂P[1], 
                                            sylvester_algorithm = sylvester_algorithm, 
                                            tol = tol, 
                                            𝕊ℂ = 𝕊ℂ,
                                            verbose = verbose)

        solved = solved && slvd

        ∂A = ∂C * B' * P'

        ∂B = P' * A' * ∂C

        return NoTangent(), ∂A, ∂B, ∂C, NoTangent()
    end

    return (P, solved), solve_sylvester_equation_pullback
end

# ========== src/filter/kalman.jl ==========

# Lines 290-561
function rrule(::typeof(run_kalman_iterations), 
                    A, 
                    𝐁, 
                    C, 
                    P, 
                    data_in_deviations; 
                    presample_periods = 0,
                    on_failure_loglikelihood = -Inf,
                    # timer::TimerOutput = TimerOutput(),
                    verbose::Bool = false)
    # @timeit_debug timer "Calculate Kalman filter - forward" begin
    T = size(data_in_deviations, 2) + 1

    z = zeros(size(data_in_deviations, 1))

    ū = zeros(size(C,2))

    P̄ = deepcopy(P) 

    temp_N_N = similar(P)

    PCtmp = similar(C')

    F = similar(C * C')

    u = [similar(ū) for _ in 1:T] # used in backward pass

    P = [copy(P̄) for _ in 1:T] # used in backward pass

    CP = [zero(C) for _ in 1:T] # used in backward pass

    K = [similar(C') for _ in 1:T] # used in backward pass

    invF = [similar(F) for _ in 1:T] # used in backward pass

    v = [zeros(size(data_in_deviations, 1)) for _ in 1:T] # used in backward pass

    loglik = 0.0

    # @timeit_debug timer "Loop" begin
        
    for t in 2:T
        if !all(isfinite.(z)) 
            if verbose println("KF not finite at step $t") end
            return on_failure_loglikelihood, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent() 
        end

        v[t] .= data_in_deviations[:, t-1] .- z#[t-1]

        # CP[t] .= C * P̄[t-1]
        ℒ.mul!(CP[t], C, P̄)#[t-1])
    
        # F[t] .= CP[t] * C'
        ℒ.mul!(F, CP[t], C')
    
        luF = RF.lu(F, check = false)
    
        if !ℒ.issuccess(luF)
            if verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        Fdet = ℒ.det(luF)

        # Early return if determinant is too small, indicating numerical instability.
        if Fdet < eps(Float64)
            if verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        
        # invF[t] .= inv(luF)
        copy!(invF[t], inv(luF))
        
        if t - 1 > presample_periods
            loglik += log(Fdet) + ℒ.dot(v[t], invF[t], v[t])
        end

        # K[t] .= P̄[t-1] * C' * invF[t]
        ℒ.mul!(PCtmp, P̄, C')
        ℒ.mul!(K[t], PCtmp, invF[t])

        # P[t] .= P̄[t-1] - K[t] * CP[t]
        ℒ.mul!(P[t], K[t], CP[t], -1, 0)
        P[t] .+= P̄
    
        # P̄[t] .= A * P[t] * A' + 𝐁
        ℒ.mul!(temp_N_N, P[t], A')
        ℒ.mul!(P̄, A, temp_N_N)
        P̄ .+= 𝐁

        # u[t] .= K[t] * v[t] + ū[t-1]
        ℒ.mul!(u[t], K[t], v[t])
        u[t] .+= ū
        
        # ū[t] .= A * u[t]
        ℒ.mul!(ū, A, u[t])

        # z[t] .= C * ū[t]
        ℒ.mul!(z, C, ū)
    end

    llh = -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 

    # initialise derivative variables
    ∂A = zero(A)
    ∂F = zero(F)
    ∂Faccum = zero(F)
    ∂P = zero(P̄)
    ∂ū = zero(ū)
    ∂v = zero(v[1])
    ∂𝐁 = zero(𝐁)
    ∂data_in_deviations = zero(data_in_deviations)
    vtmp = zero(v[1])
    Ptmp = zero(P[1])

    # end # timeit_debug
    # end # timeit_debug

    # pullback
    function kalman_pullback(∂llh)
        # @timeit_debug timer "Calculate Kalman filter - reverse" begin
        ℒ.rmul!(∂A, 0)
        ℒ.rmul!(∂Faccum, 0)
        ℒ.rmul!(∂P, 0)
        ℒ.rmul!(∂ū, 0)
        ℒ.rmul!(∂𝐁, 0)

        # @timeit_debug timer "Loop" begin
        for t in T:-1:2
            if t > presample_periods + 1
                # ∂llh∂F
                # loglik += logdet(F[t]) + v[t]' * invF[t] * v[t]
                # ∂F = invF[t]' - invF[t]' * v[t] * v[t]' * invF[t]'
                ℒ.mul!(∂F, v[t], v[t]')
                ℒ.mul!(invF[1], invF[t]', ∂F) # using invF[1] as temporary storage
                ℒ.mul!(∂F, invF[1], invF[t]')
                ℒ.axpby!(1, invF[t]', -1, ∂F)
        
                # ∂llh∂ū
                # loglik += logdet(F[t]) + v[t]' * invF[t] * v[t]
                # z[t] .= C * ū[t]
                # ∂v = (invF[t]' + invF[t]) * v[t]
                copy!(invF[1], invF[t]' .+ invF[t])
                # copy!(invF[1], invF[t]) # using invF[1] as temporary storage
                # ℒ.axpy!(1, invF[t]', invF[1]) # using invF[1] as temporary storage
                ℒ.mul!(∂v, invF[1], v[t])
                # ℒ.mul!(∂ū∂v, C', v[1])
            else
                ℒ.rmul!(∂F, 0)
                ℒ.rmul!(∂v, 0)
            end
        
            # ∂F∂P
            # F[t] .= C * P̄[t-1] * C'
            # ∂P += C' * (∂F + ∂Faccum) * C
            ℒ.axpy!(1, ∂Faccum, ∂F)
            ℒ.mul!(PCtmp, C', ∂F) 
            ℒ.mul!(∂P, PCtmp, C, 1, 1) 
        
            # ∂ū∂P
            # K[t] .= P̄[t-1] * C' * invF[t]
            # u[t] .= K[t] * v[t] + ū[t-1]
            # ū[t] .= A * u[t]
            # ∂P += A' * ∂ū * v[t]' * invF[t]' * C
            ℒ.mul!(CP[1], invF[t]', C) # using CP[1] as temporary storage
            ℒ.mul!(PCtmp, ∂ū , v[t]')
            ℒ.mul!(P[1], PCtmp , CP[1]) # using P[1] as temporary storage
            ℒ.mul!(∂P, A', P[1], 1, 1) 
        
            # ∂ū∂data
            # v[t] .= data_in_deviations[:, t-1] .- z
            # z[t] .= C * ū[t]
            # ∂data_in_deviations[:,t-1] = -C * ∂ū
            ℒ.mul!(u[1], A', ∂ū)
            ℒ.mul!(v[1], K[t]', u[1]) # using v[1] as temporary storage
            ℒ.axpy!(1, ∂v, v[1])
            ∂data_in_deviations[:,t-1] .= v[1]
            # ℒ.mul!(∂data_in_deviations[:,t-1], C, ∂ū, -1, 0) # cannot assign to columns in matrix, must be whole matrix 

            # ∂ū∂ū
            # z[t] .= C * ū[t]
            # v[t] .= data_in_deviations[:, t-1] .- z
            # K[t] .= P̄[t-1] * C' * invF[t]
            # u[t] .= K[t] * v[t] + ū[t-1]
            # ū[t] .= A * u[t]
            # step to next iteration
            # ∂ū = A' * ∂ū - C' * K[t]' * A' * ∂ū
            ℒ.mul!(u[1], A', ∂ū) # using u[1] as temporary storage
            ℒ.mul!(v[1], K[t]', u[1]) # using v[1] as temporary storage
            ℒ.mul!(∂ū, C', v[1])
            ℒ.mul!(u[1], C', v[1], -1, 1)
            copy!(∂ū, u[1])
        
            # ∂llh∂ū
            # loglik += logdet(F[t]) + v[t]' * invF[t] * v[t]
            # v[t] .= data_in_deviations[:, t-1] .- z
            # z[t] .= C * ū[t]
            # ∂ū -= ∂ū∂v
            ℒ.mul!(u[1], C', ∂v) # using u[1] as temporary storage
            ℒ.axpy!(-1, u[1], ∂ū)
        
            if t > 2
                # ∂ū∂A
                # ū[t] .= A * u[t]
                # ∂A += ∂ū * u[t-1]'
                ℒ.mul!(∂A, ∂ū, u[t-1]', 1, 1)
        
                # ∂P̄∂A and ∂P̄∂𝐁
                # P̄[t] .= A * P[t] * A' + 𝐁
                # ∂A += ∂P * A * P[t-1]' + ∂P' * A * P[t-1]
                ℒ.mul!(P[1], A, P[t-1]')
                ℒ.mul!(Ptmp ,∂P, P[1])
                ℒ.mul!(P[1], A, P[t-1])
                ℒ.mul!(Ptmp ,∂P', P[1], 1, 1)
                ℒ.axpy!(1, Ptmp, ∂A)
        
                # ∂𝐁 += ∂P
                ℒ.axpy!(1, ∂P, ∂𝐁)
        
                # ∂P∂P
                # P[t] .= P̄[t-1] - K[t] * C * P̄[t-1]
                # P̄[t] .= A * P[t] * A' + 𝐁
                # step to next iteration
                # ∂P = A' * ∂P * A
                ℒ.mul!(P[1], ∂P, A) # using P[1] as temporary storage
                ℒ.mul!(∂P, A', P[1])
        
                # ∂P̄∂P
                # K[t] .= P̄[t-1] * C' * invF[t]
                # P[t] .= P̄[t-1] - K[t] * CP[t]
                # ∂P -= C' * K[t-1]' * ∂P + ∂P * K[t-1] * C 
                ℒ.mul!(PCtmp, ∂P, K[t-1])
                ℒ.mul!(CP[1], K[t-1]', ∂P) # using CP[1] as temporary storage
                ℒ.mul!(∂P, PCtmp, C, -1, 1)
                ℒ.mul!(∂P, C', CP[1], -1, 1)
        
                # ∂ū∂F
                # K[t] .= P̄[t-1] * C' * invF[t]
                # u[t] .= K[t] * v[t] + ū[t-1]
                # ū[t] .= A * u[t]
                # ∂Faccum = -invF[t-1]' * CP[t-1] * A' * ∂ū * v[t-1]' * invF[t-1]'
                ℒ.mul!(u[1], A', ∂ū) # using u[1] as temporary storage
                ℒ.mul!(v[1], CP[t-1], u[1]) # using v[1] as temporary storage
                ℒ.mul!(vtmp, invF[t-1]', v[1], -1, 0)
                ℒ.mul!(invF[1], vtmp, v[t-1]') # using invF[1] as temporary storage
                ℒ.mul!(∂Faccum, invF[1], invF[t-1]')
        
                # ∂P∂F
                # K[t] .= P̄[t-1] * C' * invF[t]
                # P[t] .= P̄[t-1] - K[t] * CP[t]
                # ∂Faccum -= invF[t-1]' * CP[t-1] * ∂P * CP[t-1]' * invF[t-1]'
                ℒ.mul!(CP[1], invF[t-1]', CP[t-1]) # using CP[1] as temporary storage
                ℒ.mul!(PCtmp, CP[t-1]', invF[t-1]')
                ℒ.mul!(K[1], ∂P, PCtmp) # using K[1] as temporary storage
                ℒ.mul!(∂Faccum, CP[1], K[1], -1, 1)
        
            end
        end
        
        ℒ.rmul!(∂P, -∂llh/2)
        ℒ.rmul!(∂A, -∂llh/2)
        ℒ.rmul!(∂𝐁, -∂llh/2)
        ℒ.rmul!(∂data_in_deviations, -∂llh/2)

        # end # timeit_debug
        # end # timeit_debug

        return NoTangent(), ∂A, ∂𝐁, NoTangent(), ∂P, ∂data_in_deviations, NoTangent()
    end
    
    return llh, kalman_pullback
end

# ========== src/filter/inversion.jl ==========

# Lines 163-331
function rrule(::typeof(calculate_inversion_filter_loglikelihood), 
                ::Val{:first_order}, 
                state::Vector{Vector{Float64}}, 
                𝐒::Matrix{Float64}, 
                data_in_deviations::Matrix{Float64}, 
                observables::Union{Vector{String}, Vector{Symbol}}, 
                T::timings; 
                # timer::TimerOutput = TimerOutput(),
                warmup_iterations::Int = 0, 
                on_failure_loglikelihood = -Inf,
                presample_periods::Int = 0,
                opts::CalculationOptions = merge_calculation_options(),
                filter_algorithm::Symbol = :LagrangeNewton)
    # @timeit_debug timer "Inversion filter - forward" begin    
            
    # first order
    state = copy(state[1])

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    obs_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    t⁻ = T.past_not_future_and_mixed_idx

    shocks² = 0.0
    logabsdets = 0.0

    @assert warmup_iterations == 0 "Warmup iterations not yet implemented for reverse-mode automatic differentiation."

    state = [copy(state) for _ in 1:size(data_in_deviations,2)+1]

    shocks² = 0.0
    logabsdets = 0.0

    y = zeros(length(obs_idx))
    x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]

    jac = 𝐒[obs_idx,end-T.nExo+1:end]

    if T.nExo == length(observables)
        logabsdets = ℒ.logabsdet(jac)[1] #  ./ precision_factor

        jacdecomp = ℒ.lu(jac, check = false)

        if !ℒ.issuccess(jacdecomp)
            if opts.verbose println("Inversion filter failed") end
            return on_failure_loglikelihood, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        invjac = inv(jacdecomp)
    else
        logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(jac)) #' ./ precision_factor
        # jacdecomp = ℒ.svd(jac)
        invjac = ℒ.pinv(jac)
    end

    logabsdets *= size(data_in_deviations,2) - presample_periods

    if !isfinite(logabsdets) 
        return on_failure_loglikelihood, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    @views 𝐒obs = 𝐒[obs_idx,1:end-T.nExo]

    for i in axes(data_in_deviations,2)
        @views ℒ.mul!(y, 𝐒obs, state[i][t⁻])
        @views ℒ.axpby!(1, data_in_deviations[:,i], -1, y)
        ℒ.mul!(x[i],invjac,y)
        # x = 𝐒[obs_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[obs_idx,1:end-T.nExo] * state[t⁻])

        if i > presample_periods
            shocks² += sum(abs2,x[i])
            if !isfinite(shocks²) 
                return on_failure_loglikelihood, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
            end
        end

        ℒ.mul!(state[i+1], 𝐒, vcat(state[i][t⁻], x[i]))
        # state[i+1] =  𝐒 * vcat(state[i][t⁻], x[i])
    end

    llh = -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
    
    if llh < -1e12
        return on_failure_loglikelihood, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    ∂𝐒 = zero(𝐒)
    
    ∂𝐒ᵗ⁻ = copy(∂𝐒[t⁻,:])

    ∂data_in_deviations = zero(data_in_deviations)
    
    ∂data = zeros(length(t⁻), size(data_in_deviations,2) - 1)

    ∂state = zero(state[1])

    # precomputed matrices
    M¹  = 𝐒[obs_idx, 1:end-T.nExo]' * invjac' 
    M²  = 𝐒[t⁻,1:end-T.nExo]' - M¹ * 𝐒[t⁻,end-T.nExo+1:end]'
    M³  = invjac' * 𝐒[t⁻,end-T.nExo+1:end]'

    ∂Stmp = [copy(M¹) for _ in 1:size(data_in_deviations,2)-1]

    for t in 2:size(data_in_deviations,2)-1
        ℒ.mul!(∂Stmp[t], M², ∂Stmp[t-1])
        # ∂Stmp[t] = M² * ∂Stmp[t-1]
    end

    tmp1 = zeros(Float64, T.nExo, length(t⁻) + T.nExo)
    tmp2 = zeros(Float64, length(t⁻), length(t⁻) + T.nExo)
    tmp3 = zeros(Float64, length(t⁻) + T.nExo)

    ∂𝐒t⁻        = copy(tmp2)
    # ∂𝐒obs_idx   = copy(tmp1)

    # end # timeit_debug
    # pullback
    function inversion_pullback(∂llh)
        # @timeit_debug timer "Inversion filter - pullback" begin    
                
        for t in reverse(axes(data_in_deviations,2))
            ∂state[t⁻]                                  .= M² * ∂state[t⁻]

            if t > presample_periods
                ∂state[t⁻]                              += M¹ * x[t]

                ∂data_in_deviations[:,t]                -= invjac' * x[t]

                ∂𝐒[obs_idx, :]                          += invjac' * x[t] * vcat(state[t][t⁻], x[t])'

                if t > 1
                    ∂data[:,t:end]                      .= M² * ∂data[:,t:end]
                    
                    ∂data[:,t-1]                        += M¹ * x[t]
            
                    ∂data_in_deviations[:,t-1]          += M³ * ∂data[:,t-1:end] * ones(size(data_in_deviations,2) - t + 1)

                    for tt in t-1:-1:1
                        for (i,v) in enumerate(t⁻)
                            copyto!(tmp3::Vector{Float64}, i::Int, state[tt]::Vector{Float64}, v::Int, 1)
                        end
                        
                        copyto!(tmp3, length(t⁻) + 1, x[tt], 1, T.nExo)

                        ℒ.mul!(tmp1,  x[t], tmp3')

                        ℒ.mul!(∂𝐒t⁻,  ∂Stmp[t-tt], tmp1, 1, 1)
                        
                    end
                end
            end
        end

        ∂𝐒[t⁻,:]                            += ∂𝐒t⁻
                        
        ∂𝐒[obs_idx, :]                      -= M³ * ∂𝐒t⁻
        
        ∂𝐒[obs_idx,end-T.nExo+1:end] -= (size(data_in_deviations,2) - presample_periods) * invjac' / 2

        # end # timeit_debug

        return NoTangent(), NoTangent(), [∂state * ∂llh], ∂𝐒 * ∂llh, ∂data_in_deviations * ∂llh, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    
    return llh, inversion_pullback
end

# Lines 570-1036
function rrule(::typeof(calculate_inversion_filter_loglikelihood),
                ::Val{:pruned_second_order},
                state::Vector{Vector{Float64}}, 
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                observables::Union{Vector{String}, Vector{Symbol}},
                T::timings; 
                # timer::TimerOutput = TimerOutput(),
                on_failure_loglikelihood = -Inf,
                warmup_iterations::Int = 0,
                presample_periods::Int = 0,
                opts::CalculationOptions = merge_calculation_options(),
                filter_algorithm::Symbol = :LagrangeNewton)# where S <: Real
    # @timeit_debug timer "Inversion filter pruned 2nd - forward" begin
    # @timeit_debug timer "Preallocation" begin
                    
    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))
    
    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind
    
    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind
    
    shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind
    
    tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    var²_idxs = tmp.nzind
    
    𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    𝐒⁻¹ᵉ = 𝐒[1][T.past_not_future_and_mixed_idx,end-T.nExo+1:end]
    𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = nnz(𝐒²⁻)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    state₁ = state[1][T.past_not_future_and_mixed_idx]
    state₂ = state[2][T.past_not_future_and_mixed_idx]

    kronxx = [zeros(T.nExo^2) for _ in 1:size(data_in_deviations,2)]
    
    J = ℒ.I(T.nExo)
    
    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))
    
    kron_buffer3 = ℒ.kron(J, zeros(T.nPast_not_future_and_mixed + 1))

    x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]
    
    state¹⁻ = state₁

    state¹⁻_vol = vcat(state¹⁻, 1)

    state²⁻ = state₂

    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(J, state¹⁻_vol)
   
    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 
    
    aug_state₁ = [copy([state₁; 1; ones(T.nExo)]) for _ in 1:size(data_in_deviations,2)]
    aug_state₂ = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]
    
    tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[1])), x[1])
    
    jacc = [zero(tmp) for _ in 1:size(data_in_deviations,2)]
    
    jacct = copy(tmp')

    λ = [zeros(size(tmp, 1)) for _ in 1:size(data_in_deviations,2)]
    
    λ[1] = copy(tmp' \ x[1] * 2)
    
    fXλp_tmp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[1], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  tmp'
                -tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
    fXλp = [zero(fXλp_tmp) for _ in 1:size(data_in_deviations,2)]
    
    kronxλ_tmp = ℒ.kron(x[1], λ[1])
    
    kronxλ = [zero(kronxλ_tmp) for _ in 1:size(data_in_deviations,2)]
    
    kronstate¹⁻_vol = zeros((T.nPast_not_future_and_mixed + 1)^2)

    kronaug_state₁ = zeros(length(aug_state₁[1])^2)

    shock_independent = zeros(size(data_in_deviations,1))

    init_guess = zeros(size(𝐒ⁱ, 2))

    tmp = zeros(size(𝐒ⁱ, 2) * size(𝐒ⁱ, 2))
    
    lI = -2 * vec(ℒ.I(size(𝐒ⁱ, 2)))
    
    # end # timeit_debug
    # @timeit_debug timer "Main loop" begin

    for i in axes(data_in_deviations,2)
        # state¹⁻ = state₁
    
        # state¹⁻_vol = vcat(state¹⁻, 1)
    
        # state²⁻ = state₂

        copyto!(state¹⁻_vol, 1, state₁, 1)

        copyto!(shock_independent, data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)

        ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)

        ℒ.kron!(kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol, -1/2, 1)
    
        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
        ℒ.kron!(kron_buffer3, J, state¹⁻_vol)

        ℒ.mul!(𝐒ⁱ, 𝐒²⁻ᵉ, kron_buffer3)

        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ)

        init_guess *= 0
    
        # @timeit_debug timer "Find shocks" begin
        x[i], matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kronxx[i],
                                kron_buffer2,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )
        # end # timeit_debug
    
        if !matched
            if opts.verbose println("Inversion filter failed at step $i") end
            return on_failure_loglikelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
        end

        # jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])
        ℒ.kron!(kron_buffer2, J, x[i])

        ℒ.mul!(jacc[i], 𝐒ⁱ²ᵉ, kron_buffer2)

        ℒ.axpby!(1, 𝐒ⁱ, 2, jacc[i])

        copy!(jacct, jacc[i]')

        jacc_fact = try
                        ℒ.factorize(jacct) # otherwise this fails for nshocks > nexo
                    catch
                        if opts.verbose println("Inversion filter failed at step $i") end
                        return on_failure_loglikelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end

        try
            ℒ.ldiv!(λ[i], jacc_fact, x[i])
        catch
            if opts.verbose println("Inversion filter failed at step $i") end
            return on_failure_loglikelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
        end

        ℒ.rmul!(λ[i], 2)
    
        # fXλp[i] = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc[i]'
                    # -jacc[i]  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
        ℒ.mul!(tmp, 𝐒ⁱ²ᵉ', λ[i])
        ℒ.axpby!(1, lI, 2, tmp)

        fXλp[i][1:size(𝐒ⁱ, 2), 1:size(𝐒ⁱ, 2)] = tmp
        fXλp[i][size(𝐒ⁱ, 2)+1:end, 1:size(𝐒ⁱ, 2)] = -jacc[i]
        fXλp[i][1:size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)+1:end] = jacct
    
        ℒ.kron!(kronxx[i], x[i], x[i])
    
        ℒ.kron!(kronxλ[i], x[i], λ[i])
    
        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc_fact)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i]))
            end
    
            shocks² += sum(abs2,x[i])
            
            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_loglikelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
            end
        end
    
        # aug_state₁[i] = [state₁; 1; x[i]]
        # aug_state₂[i] = [state₂; 0; zero(x[1])]
        copyto!(aug_state₁[i], 1, state₁, 1)
        copyto!(aug_state₁[i], length(state₁) + 2, x[i], 1)
        copyto!(aug_state₂[i], 1, state₂, 1)

        # state₁, state₂ = [𝐒⁻¹ * aug_state₁, 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * ℒ.kron(aug_state₁, aug_state₁) / 2] # strictly following Andreasen et al. (2018)
        ℒ.mul!(state₁, 𝐒⁻¹, aug_state₁[i])

        ℒ.mul!(state₂, 𝐒⁻¹, aug_state₂[i])
        ℒ.kron!(kronaug_state₁, aug_state₁[i], aug_state₁[i])
        ℒ.mul!(state₂, 𝐒⁻², kronaug_state₁, 1/2, 1)
    end
    
    # end # timeit_debug
    # end # timeit_debug

    ∂data_in_deviations = similar(data_in_deviations)

    ∂aug_state₁ = zero(aug_state₁[1])

    ∂aug_state₂ = zero(aug_state₂[1])

    ∂kronaug_state₁ = zeros(length(aug_state₁[1])^2)

    ∂kronIx = zero(ℒ.kron(ℒ.I(length(x[1])), x[1]))

    ∂kronIstate¹⁻_vol = zero(ℒ.kron(J, state¹⁻_vol))

    ∂kronstate¹⁻_vol = zero(ℒ.kron(state¹⁻_vol, state¹⁻_vol))

    function inversion_filter_loglikelihood_pullback(∂llh) 
        # @timeit_debug timer "Inversion filter pruned 2nd - pullback" begin
        # @timeit_debug timer "Preallocation" begin
        
        ∂𝐒ⁱ = zero(𝐒ⁱ)
        ∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)

        ∂𝐒¹ᵉ = zero(𝐒¹ᵉ)
        ∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)

        ∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)
        ∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)

        ∂𝐒⁻¹ = zero(𝐒⁻¹)
        ∂𝐒⁻² = zero(𝐒⁻²)

        ∂𝐒¹⁻ = zero(𝐒¹⁻)

        ∂state¹⁻_vol = zero(state¹⁻_vol)
        ∂x = zero(x[1])
        ∂state = [zeros(T.nPast_not_future_and_mixed), zeros(T.nPast_not_future_and_mixed)]

        kronSλ = zeros(length(cond_var_idx) * T.nExo)
        kronxS = zeros(T.nExo * length(cond_var_idx))
        
        # end # timeit_debug
        # @timeit_debug timer "Main loop" begin
        
        for i in reverse(axes(data_in_deviations,2))
            # state₁, state₂ = [𝐒⁻¹ * aug_state₁[i], 𝐒⁻¹ * aug_state₂[i] + 𝐒⁻² * ℒ.kron(aug_state₁[i], aug_state₁[i]) / 2]
            # state₁ = 𝐒⁻¹ * aug_state₁[i]
            # ∂𝐒⁻¹ += ∂state[1] * aug_state₁[i]'
            ℒ.mul!(∂𝐒⁻¹, ∂state[1], aug_state₁[i]', 1, 1)

            # ∂aug_state₁ = 𝐒⁻¹' * ∂state[1]
            ℒ.mul!(∂aug_state₁, 𝐒⁻¹', ∂state[1])

            # state₂ = 𝐒⁻¹ * aug_state₂[i] + 𝐒⁻² * ℒ.kron(aug_state₁[i], aug_state₁[i]) / 2
            # ∂𝐒⁻¹ += ∂state[2] * aug_state₂[i]'
            ℒ.mul!(∂𝐒⁻¹, ∂state[2], aug_state₂[i]', 1, 1)

            # ∂aug_state₂ = 𝐒⁻¹' * ∂state[2]
            ℒ.mul!(∂aug_state₂, 𝐒⁻¹', ∂state[2])

            # ∂𝐒⁻² += ∂state[2] * ℒ.kron(aug_state₁[i], aug_state₁[i])' / 2
            ℒ.kron!(kronaug_state₁, aug_state₁[i], aug_state₁[i])
            ℒ.mul!(∂𝐒⁻², ∂state[2], kronaug_state₁', 1/2, 1)

            # ∂kronaug_state₁ = 𝐒⁻²' * ∂state[2] / 2
            ℒ.mul!(∂kronaug_state₁, 𝐒⁻²', ∂state[2])
            ℒ.rdiv!(∂kronaug_state₁, 2)

            fill_kron_adjoint!(∂aug_state₁, ∂aug_state₁, ∂kronaug_state₁, aug_state₁[i], aug_state₁[i])

            if i > 1 && i < size(data_in_deviations,2)
                ∂state[1] *= 0
                ∂state[2] *= 0
            end
            
            # aug_state₁ = [state₁; 1; x]
            # ∂state[1] += ∂aug_state₁[1:length(∂state[1])]
            ℒ.axpy!(1, ∂aug_state₁[1:length(∂state[1])], ∂state[1])

            ∂x = ∂aug_state₁[T.nPast_not_future_and_mixed+2:end]

            # aug_state₂ = [state₂; 0; zero(x)]
            # ∂state[2] += ∂aug_state₂[1:length(∂state[1])]
            ℒ.axpy!(1, ∂aug_state₂[1:length(∂state[1])], ∂state[2])

            # shocks² += sum(abs2,x[i])
            if i < size(data_in_deviations,2)
                ∂x -= copy(x[i])
            else
                ∂x += copy(x[i])
            end

            # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
            ∂jacc = try if size(jacc[i], 1) == size(jacc[i], 2)
                            inv(jacc[i])'
                        else
                            ℒ.pinv(jacc[i])'
                        end
                    catch
                        return NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end

            # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
            # ∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc
            ℒ.mul!(∂kronIx, 𝐒ⁱ²ᵉ', ∂jacc)

            if i < size(data_in_deviations,2)
                fill_kron_adjoint_∂B!(∂kronIx, ∂x, -J)
            else
                fill_kron_adjoint_∂B!(∂kronIx, ∂x, J)
            end

            # ∂𝐒ⁱ²ᵉ -= ∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'
            ℒ.kron!(kron_buffer2, J, x[i])

            ℒ.mul!(∂𝐒ⁱ²ᵉ, ∂jacc, kron_buffer2', -1, 1)

            # find_shocks
            ∂xλ = vcat(∂x, zero(λ[i]))
            # S = vcat(∂x, zero(λ[i]))

            S = fXλp[i]' \ ∂xλ
            # ℒ.ldiv!(fXλp[i]', S)

            if i < size(data_in_deviations,2)
                S *= -1
            end

            ∂shock_independent = S[T.nExo+1:end] # fine

            # ∂𝐒ⁱ = (S[1:T.nExo] * λ[i]' - S[T.nExo+1:end] * x[i]') # fine
            # ∂𝐒ⁱ -= ∂jacc / 2 # fine
            # copyto!(∂𝐒ⁱ, ℒ.kron(S[1:T.nExo], λ[i]) - ℒ.kron(x[i], S[T.nExo+1:end]))
            ℒ.kron!(kronSλ, S[1:T.nExo], λ[i])
            ℒ.kron!(kronxS, x[i], S[T.nExo+1:end])
            ℒ.axpy!(-1, kronxS, kronSλ)
            copyto!(∂𝐒ⁱ, kronSλ)
            # ∂𝐒ⁱ -= ∂jacc / 2 # fine
            ℒ.axpy!(-1/2, ∂jacc, ∂𝐒ⁱ)
        
            ∂𝐒ⁱ²ᵉ += reshape(2 * ℒ.kron(S[1:T.nExo], ℒ.kron(x[i], λ[i])) - ℒ.kron(kronxx[i], S[T.nExo+1:end]), size(∂𝐒ⁱ²ᵉ))
            # ∂𝐒ⁱ²ᵉ += 2 * S[1:T.nExo] *  kronxλ[i]' - S[T.nExo+1:end] * kronxx[i]'

            # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
            ∂state¹⁻_vol *= 0
            # ∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ
            ℒ.mul!(∂kronIstate¹⁻_vol, 𝐒²⁻ᵉ', ∂𝐒ⁱ)

            fill_kron_adjoint_∂A!(∂kronIstate¹⁻_vol, ∂state¹⁻_vol, J)

            state¹⁻_vol = aug_state₁[i][1:T.nPast_not_future_and_mixed+1]

            # ∂𝐒¹ᵉ += ∂𝐒ⁱ
            ℒ.axpy!(1, ∂𝐒ⁱ, ∂𝐒¹ᵉ)

            # ∂𝐒²⁻ᵉ += ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)'
            ℒ.kron!(∂kronIstate¹⁻_vol, J, state¹⁻_vol)
            ℒ.mul!(∂𝐒²⁻ᵉ, ∂𝐒ⁱ, ∂kronIstate¹⁻_vol', 1, 1)


            # shock_independent = copy(data_in_deviations[:,i])
            ∂data_in_deviations[:,i] = ∂shock_independent

            # ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
            # ∂𝐒¹⁻ᵛ -= ∂shock_independent * state¹⁻_vol'
            ℒ.mul!(∂𝐒¹⁻ᵛ, ∂shock_independent, state¹⁻_vol', -1, 1)

            # ∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent
            ℒ.mul!(∂state¹⁻_vol, 𝐒¹⁻ᵛ', ∂shock_independent, -1, 1)

            # ℒ.mul!(shock_independent, 𝐒¹⁻, state²⁻, -1, 1)
            # ∂𝐒¹⁻ -= ∂shock_independent * aug_state₂[i][1:T.nPast_not_future_and_mixed]'
            ℒ.mul!(∂𝐒¹⁻, ∂shock_independent, aug_state₂[i][1:T.nPast_not_future_and_mixed]', -1, 1)

            # ∂state[2] -= 𝐒¹⁻' * ∂shock_independent
            ℒ.mul!(∂state[2], 𝐒¹⁻', ∂shock_independent, -1, 1)

            # ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
            # ∂𝐒²⁻ᵛ -= ∂shock_independent * ℒ.kron(state¹⁻_vol, state¹⁻_vol)' / 2
            ℒ.kron!(∂kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)
            ℒ.mul!(∂𝐒²⁻ᵛ, ∂shock_independent, ∂kronstate¹⁻_vol', -1/2, 1)
            
            # ∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2
            ℒ.mul!(∂kronstate¹⁻_vol, 𝐒²⁻ᵛ', ∂shock_independent)
            ℒ.rdiv!(∂kronstate¹⁻_vol, -2)

            fill_kron_adjoint!(∂state¹⁻_vol, ∂state¹⁻_vol, ∂kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)

            # state¹⁻_vol = vcat(state¹⁻, 1)
            # ∂state[1] += ∂state¹⁻_vol[1:end-1]
            ℒ.axpy!(1, ∂state¹⁻_vol[1:end-1], ∂state[1])
        end

        # end # timeit_debug
        # @timeit_debug timer "Post allocation" begin

        ∂𝐒 = [zero(𝐒[1]), zeros(size(𝐒[2]))]

        ∂𝐒[1][cond_var_idx,end-T.nExo+1:end] .+= ∂𝐒¹ᵉ
        ∂𝐒[2][cond_var_idx,shockvar²_idxs] .+= ∂𝐒²⁻ᵉ
        ℒ.rdiv!(∂𝐒ⁱ²ᵉ, 2)
        ∂𝐒[2][cond_var_idx,shock²_idxs] .+= ∂𝐒ⁱ²ᵉ# / 2

        ∂𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1] .+= ∂𝐒¹⁻ᵛ
        ∂𝐒[2][cond_var_idx,var_vol²_idxs] .+= ∂𝐒²⁻ᵛ

        ∂𝐒[1][T.past_not_future_and_mixed_idx,:] .+= ∂𝐒⁻¹
        ∂𝐒[2][T.past_not_future_and_mixed_idx,:] .+= ∂𝐒⁻²

        ∂𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed] .+= ∂𝐒¹⁻

        # ∂𝐒[1] *= ∂llh
        # ∂𝐒[2] *= ∂llh
        ℒ.rmul!(∂𝐒[1], ∂llh)
        ℒ.rmul!(∂𝐒[2], ∂llh)

        ℒ.rmul!(∂data_in_deviations, ∂llh)
        
        ∂state[1] = ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state[1] * ∂llh
        ∂state[2] = ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state[2] * ∂llh

        # end # timeit_debug
        # end # timeit_debug

        return NoTangent(), NoTangent(), ∂state, ∂𝐒, ∂data_in_deviations, NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
    end

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    llh = -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2

    return llh, inversion_filter_loglikelihood_pullback
end

# Lines 1267-1703
function rrule(::typeof(calculate_inversion_filter_loglikelihood),
                ::Val{:second_order},
                state::Vector{Float64}, 
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                observables::Union{Vector{String}, Vector{Symbol}},
                T::timings; 
                # timer::TimerOutput = TimerOutput(),
                on_failure_loglikelihood = -Inf,
                warmup_iterations::Int = 0,
                presample_periods::Int = 0,
                opts::CalculationOptions = merge_calculation_options(),
                filter_algorithm::Symbol = :LagrangeNewton)# where S <: Real
    # @timeit_debug timer "Inversion filter 2nd - forward" begin
        
    # @timeit_debug timer "Preallocation" begin

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))
    
    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind
    
    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind
    
    shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind
    
    tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    var²_idxs = tmp.nzind
    
    𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    𝐒⁻¹ᵉ = 𝐒[1][T.past_not_future_and_mixed_idx,end-T.nExo+1:end]
    𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = nnz(𝐒²⁻)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    kronxx = [zeros(T.nExo^2) for _ in 1:size(data_in_deviations,2)]
    
    J = ℒ.I(T.nExo)
    
    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    kron_buffer3 = ℒ.kron(J, zeros(T.nPast_not_future_and_mixed + 1))
    
    x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]
    
    state¹⁻ = state[T.past_not_future_and_mixed_idx]
    
    state¹⁻_vol = vcat(state¹⁻, 1)

    kronstate¹⁻_voltmp = ℒ.kron(state¹⁻_vol, state¹⁻_vol)

    kronstate¹⁻_vol = [kronstate¹⁻_voltmp for _ in 1:size(data_in_deviations,2)]
    
    shock_independent = zeros(size(data_in_deviations,1))

    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(J, state¹⁻_vol)
    
    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

    # aug_state_tmp = [zeros(T.nPast_not_future_and_mixed); 1; zeros(T.nExo)]

    aug_state = [[zeros(T.nPast_not_future_and_mixed); 1; zeros(T.nExo)] for _ in 1:size(data_in_deviations,2)]
    
    kronaug_state = [zeros((T.nPast_not_future_and_mixed + 1 + T.nExo)^2) for _ in 1:size(data_in_deviations,2)]
    
    tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[1])), x[1])
    
    jacc = [zero(tmp) for _ in 1:size(data_in_deviations,2)]

    jacct = copy(tmp')

    λ = [zeros(size(tmp, 1)) for _ in 1:size(data_in_deviations,2)]
    
    λ[1] = tmp' \ x[1] * 2
    
    fXλp_tmp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[1], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  tmp'
                -tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
                
    fXλp = [zero(fXλp_tmp) for _ in 1:size(data_in_deviations,2)]
    
    kronxλ_tmp = ℒ.kron(x[1], λ[1])
    
    kronxλ = [kronxλ_tmp for _ in 1:size(data_in_deviations,2)]
    
    tmp = zeros(size(𝐒ⁱ, 2) * size(𝐒ⁱ, 2))
    
    lI = -2 * vec(ℒ.I(size(𝐒ⁱ, 2)))
    
    init_guess = zeros(size(𝐒ⁱ, 2))

    # end # timeit_debug
    # @timeit_debug timer "Main loop" begin

    @inbounds for i in axes(data_in_deviations,2)
        # aug_state[i][1:T.nPast_not_future_and_mixed] = state¹⁻
        copyto!(aug_state[i], 1, state¹⁻, 1)

        state¹⁻_vol = aug_state[i][1:T.nPast_not_future_and_mixed + 1]
        # copyto!(state¹⁻_vol, 1, aug_state[i], 1, T.nPast_not_future_and_mixed + 1)
        
        copyto!(shock_independent, data_in_deviations[:,i])
    
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)

        ℒ.kron!(kronstate¹⁻_vol[i], state¹⁻_vol, state¹⁻_vol)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol[i], -1/2, 1)
    
        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(J, state¹⁻_vol)
        ℒ.kron!(kron_buffer3, J, state¹⁻_vol)

        ℒ.mul!(𝐒ⁱ, 𝐒²⁻ᵉ, kron_buffer3)

        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ)

        init_guess *= 0
    
        # @timeit_debug timer "Find shocks" begin
        x[i], matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kronxx[i],
                                kron_buffer2,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )
        # end # timeit_debug

        if !matched
            if opts.verbose println("Inversion filter failed at step $i") end
            return on_failure_loglikelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
        end
        
        ℒ.kron!(kron_buffer2, J, x[i])

        ℒ.mul!(jacc[i], 𝐒ⁱ²ᵉ, kron_buffer2)

        ℒ.axpby!(1, 𝐒ⁱ, 2, jacc[i])
        # jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

        copy!(jacct, jacc[i]')

        jacc_fact = try
                        ℒ.factorize(jacct)
                    catch
                        if opts.verbose println("Inversion filter failed at step $i") end
                        return on_failure_loglikelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end

        try
            ℒ.ldiv!(λ[i], jacc_fact, x[i])
        catch
            if opts.verbose println("Inversion filter failed at step $i") end
            return on_failure_loglikelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
        end

        # ℒ.ldiv!(λ[i], jacc_fact', x[i])
        ℒ.rmul!(λ[i], 2)
    
        # fXλp[i] = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc[i]'
                    # -jacc[i]  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
        
        ℒ.mul!(tmp, 𝐒ⁱ²ᵉ', λ[i])
        ℒ.axpby!(1, lI, 2, tmp)

        fXλp[i][1:size(𝐒ⁱ, 2), 1:size(𝐒ⁱ, 2)] = tmp
        fXλp[i][size(𝐒ⁱ, 2)+1:end, 1:size(𝐒ⁱ, 2)] = -jacc[i]
        fXλp[i][1:size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)+1:end] = jacct

        ℒ.kron!(kronxx[i], x[i], x[i])
    
        ℒ.kron!(kronxλ[i], x[i], λ[i])
    
        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc_fact)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i]))
            end
    
            shocks² += sum(abs2, x[i])
            
            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_loglikelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
            end
        end
        
        # aug_state[i] = [state¹⁻; 1; x[i]]
        # aug_state[i][1:T.nPast_not_future_and_mixed] = state¹⁻
        # aug_state[i][end-T.nExo+1:end] = x[i]
        copyto!(aug_state[i], 1, state¹⁻, 1)
        copyto!(aug_state[i], length(state¹⁻) + 2, x[i], 1)
        
        ℒ.kron!(kronaug_state[i], aug_state[i], aug_state[i])
        ℒ.mul!(state¹⁻, 𝐒⁻¹, aug_state[i])
        ℒ.mul!(state¹⁻, 𝐒⁻², kronaug_state[i], 1/2 ,1)
    end
    
    # end # timeit_debug
    # end # timeit_debug

    ∂aug_state = zero(aug_state[1])

    ∂kronaug_state = zero(kronaug_state[1])

    ∂kronstate¹⁻_vol = zero(kronstate¹⁻_vol[1])

    ∂state = similar(state)

    ∂𝐒 = copy(𝐒)

    ∂data_in_deviations = similar(data_in_deviations)

    ∂kronIx = zero(ℒ.kron(ℒ.I(length(x[1])), x[1]))

    function inversion_filter_loglikelihood_pullback(∂llh)
        # @timeit_debug timer "Inversion filter 2nd - pullback" begin

        # @timeit_debug timer "Preallocation" begin

        ∂𝐒ⁱ = zero(𝐒ⁱ)
        ∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)
        ∂𝐒ⁱ²ᵉtmp = zeros(T.nExo, T.nExo * length(λ[1]))    
        ∂𝐒ⁱ²ᵉtmp2 = zeros(length(λ[1]), T.nExo * T.nExo)    

        ∂𝐒¹ᵉ = zero(𝐒¹ᵉ)
        ∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)

        ∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)
        ∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)

        ∂𝐒⁻¹ = zero(𝐒⁻¹)
        ∂𝐒⁻² = zero(𝐒⁻²)

        ∂state¹⁻_vol = zero(state¹⁻_vol)
        # ∂x = zero(x[1])
        ∂state = zeros(T.nPast_not_future_and_mixed)

        ∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

        kronSλ = zeros(length(cond_var_idx) * T.nExo)
        kronxS = zeros(T.nExo * length(cond_var_idx))
        
        # end # timeit_debug
        # @timeit_debug timer "Main loop" begin

        for i in reverse(axes(data_in_deviations,2))
            # stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
            # ∂𝐒⁻¹ += ∂state * aug_state[i]'
            ℒ.mul!(∂𝐒⁻¹, ∂state, aug_state[i]', 1, 1)
            
            # ∂𝐒⁻² += ∂state * kronaug_state[i]' / 2
            ℒ.mul!(∂𝐒⁻², ∂state, kronaug_state[i]', 1/2, 1)

            ℒ.mul!(∂aug_state, 𝐒⁻¹', ∂state)
            # ∂aug_state = 𝐒⁻¹' * ∂state

            ℒ.mul!(∂kronaug_state, 𝐒⁻²', ∂state)
            ℒ.rdiv!(∂kronaug_state, 2)
            # ∂kronaug_state  = 𝐒⁻²' * ∂state / 2

            fill_kron_adjoint!(∂aug_state, ∂aug_state, ∂kronaug_state, aug_state[i], aug_state[i])

            if i > 1 && i < size(data_in_deviations,2)
                ∂state *= 0
            end

            # aug_state[i] = [stt; 1; x[i]]
            ∂state += ∂aug_state[1:length(∂state)]

            # aug_state[i] = [stt; 1; x[i]]
            ∂x = ∂aug_state[T.nPast_not_future_and_mixed+2:end]

            # shocks² += sum(abs2,x[i])
            if i < size(data_in_deviations,2)
                ∂x -= copy(x[i])
            else
                ∂x += copy(x[i])
            end

            # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
            ∂jacc = try if size(jacc[i], 1) == size(jacc[i], 2)
                            inv(jacc[i])'
                        else
                            ℒ.pinv(jacc[i])'
                        end
                    catch
                        return NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end

            # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
            ℒ.mul!(∂kronIx, 𝐒ⁱ²ᵉ', ∂jacc)

            if i < size(data_in_deviations,2)
                fill_kron_adjoint_∂B!(∂kronIx, ∂x, -J)
            else
                fill_kron_adjoint_∂B!(∂kronIx, ∂x, J)
            end

            # ∂𝐒ⁱ²ᵉ -= ∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'
            ℒ.kron!(kron_buffer2, J, x[i])

            ℒ.mul!(∂𝐒ⁱ²ᵉ, ∂jacc, kron_buffer2', -1, 1)

            # find_shocks
            ∂xλ = vcat(∂x, zero(λ[i]))

            S = fXλp[i]' \ ∂xλ

            if i < size(data_in_deviations,2)
                S *= -1
            end

            ∂shock_independent = S[T.nExo+1:end] # fine

            # ℒ.mul!(∂𝐒ⁱ, λ[i], S[1:T.nExo]')
            # ℒ.mul!(∂𝐒ⁱ, S[T.nExo+1:end], x[i]', -1, 1) # fine
            # ℒ.axpy!(-1/2, ∂jacc, ∂𝐒ⁱ)
            # ∂𝐒ⁱ = λ[i] * S[1:T.nExo]' - S[T.nExo+1:end] * x[i]' # fine

            # copyto!(∂𝐒ⁱ, ℒ.kron(S[1:T.nExo], λ[i]) - ℒ.kron(x[i], S[T.nExo+1:end]))
            # ∂𝐒ⁱ -= ∂jacc / 2 # fine
            ℒ.kron!(kronSλ, S[1:T.nExo], λ[i])
            ℒ.kron!(kronxS, x[i], S[T.nExo+1:end])
            ℒ.axpy!(-1, kronxS, kronSλ)
            copyto!(∂𝐒ⁱ, kronSλ)

            ℒ.axpy!(-1/2, ∂jacc, ∂𝐒ⁱ)
        
            ∂𝐒ⁱ²ᵉ += reshape(2 * ℒ.kron(S[1:T.nExo], kronxλ[i]) - ℒ.kron(kronxx[i], S[T.nExo+1:end]), size(∂𝐒ⁱ²ᵉ))
            # ℒ.mul!(∂𝐒ⁱ²ᵉtmp, S[1:T.nExo], kronxλ[i]', 2, 1)
            # ℒ.mul!(∂𝐒ⁱ²ᵉtmp2, S[T.nExo+1:end], kronxx[i]', -1, 1)

            # ℒ.mul!(∂𝐒ⁱ²ᵉ, S[1:T.nExo], kronxλ[i]', 2, 1)
            # ℒ.mul!(∂𝐒ⁱ²ᵉ, S[T.nExo+1:end], kronxx[i]', -1, 1)
            # ∂𝐒ⁱ²ᵉ += 2 * S[1:T.nExo] * kronxλ[i]' - S[T.nExo+1:end] * kronxx[i]'

            # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
            ∂state¹⁻_vol *= 0
            
            ℒ.mul!(∂kronIstate¹⁻_vol, 𝐒²⁻ᵉ', ∂𝐒ⁱ)

            fill_kron_adjoint_∂A!(∂kronIstate¹⁻_vol, ∂state¹⁻_vol, J)

            state¹⁻_vol = aug_state[i][1:T.nPast_not_future_and_mixed + 1]

            ℒ.axpy!(1, ∂𝐒ⁱ, ∂𝐒¹ᵉ)
            # ∂𝐒¹ᵉ += ∂𝐒ⁱ

            ℒ.kron!(kron_buffer3, J, state¹⁻_vol)

            ℒ.mul!(∂𝐒²⁻ᵉ, ∂𝐒ⁱ, kron_buffer3', 1, 1)
            # ∂𝐒²⁻ᵉ += ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)'

            # shock_independent = copy(data_in_deviations[:,i])
            ∂data_in_deviations[:,i] = ∂shock_independent

            # ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
            # ∂𝐒¹⁻ᵛ -= ∂shock_independent * state¹⁻_vol'
            ℒ.mul!(∂𝐒¹⁻ᵛ, ∂shock_independent, state¹⁻_vol', -1 ,1)

            # ∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent
            ℒ.mul!(∂state¹⁻_vol, 𝐒¹⁻ᵛ', ∂shock_independent, -1, 1)

            # ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
            ℒ.kron!(kronstate¹⁻_vol[i], state¹⁻_vol, state¹⁻_vol)
            ℒ.mul!(∂𝐒²⁻ᵛ, ∂shock_independent, kronstate¹⁻_vol[i]', -1/2, 1)
            # ∂𝐒²⁻ᵛ -= ∂shock_independent * ℒ.kron(state¹⁻_vol, state¹⁻_vol)' / 2

            ℒ.mul!(∂kronstate¹⁻_vol, 𝐒²⁻ᵛ', ∂shock_independent)
            ℒ.rdiv!(∂kronstate¹⁻_vol, -2)
            # ∂kronstate¹⁻_vol = 𝐒²⁻ᵛ' * ∂shock_independent / (-2)

            fill_kron_adjoint!(∂state¹⁻_vol, ∂state¹⁻_vol, ∂kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)

            # state¹⁻_vol = vcat(state¹⁻, 1)
            ∂state += ∂state¹⁻_vol[1:end-1]
        end

        # end # timeit_debug
        # @timeit_debug timer "Post allocation" begin

        ∂𝐒 = [copy(𝐒[1]) * 0, copy(𝐒[2]) * 0]

        ∂𝐒[1][cond_var_idx,end-T.nExo+1:end] += ∂𝐒¹ᵉ
        ∂𝐒[2][cond_var_idx,shockvar²_idxs] += ∂𝐒²⁻ᵉ
        ∂𝐒[2][cond_var_idx,shock²_idxs] += ∂𝐒ⁱ²ᵉ / 2
        ∂𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1] += ∂𝐒¹⁻ᵛ
        ∂𝐒[2][cond_var_idx,var_vol²_idxs] += ∂𝐒²⁻ᵛ

        ∂𝐒[1][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻¹
        ∂𝐒[2][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻²

        ∂𝐒[1] *= ∂llh
        ∂𝐒[2] *= ∂llh

        return NoTangent(), NoTangent(),  ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state * ∂llh, ∂𝐒, ∂data_in_deviations * ∂llh, NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
    end

    # end # timeit_debug
    # end # timeit_debug

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    llh = -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2

    return llh, inversion_filter_loglikelihood_pullback
end

# Lines 2136-2682
function rrule(::typeof(calculate_inversion_filter_loglikelihood),
                ::Val{:pruned_third_order},
                state::Vector{Vector{Float64}}, 
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                observables::Union{Vector{String}, Vector{Symbol}},
                T::timings; 
                # timer::TimerOutput = TimerOutput(),
                on_failure_loglikelihood = -Inf,
                warmup_iterations::Int = 0,
                presample_periods::Int = 0,
                opts::CalculationOptions = merge_calculation_options(),
                filter_algorithm::Symbol = :LagrangeNewton)
    # @timeit_debug timer "Inversion filter - forward" begin
    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))

    tmp = ℒ.kron(e_in_s⁺, s_in_s⁺) |> sparse
    shockvar_idxs = tmp.nzind
    
    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind

    tmp = ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺) |> sparse
    shock_idxs2 = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind

    shockvar²_idxs = setdiff(union(shock_idxs), shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind

    tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    var²_idxs = tmp.nzind

    𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²⁻ᵛᵉ = 𝐒[2][cond_var_idx,shockvar_idxs]
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = nnz(𝐒²⁻)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²⁻ᵛᵉ   = nnz(𝐒²⁻ᵛᵉ)   / length(𝐒²⁻ᵛᵉ) > .1 ? collect(𝐒²⁻ᵛᵉ)   : 𝐒²⁻ᵛᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    tmp = ℒ.kron(sv_in_s⁺, ℒ.kron(sv_in_s⁺, sv_in_s⁺)) |> sparse
    var_vol³_idxs = tmp.nzind

    tmp = ℒ.kron(ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1), zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs2 = tmp.nzind

    tmp = ℒ.kron(ℒ.kron(e_in_s⁺, e_in_s⁺), zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs3 = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
    shock³_idxs = tmp.nzind

    tmp = ℒ.kron(zero(e_in_s⁺) .+ 1, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
    shockvar1_idxs = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺)) |> sparse
    shockvar2_idxs = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1)) |> sparse
    shockvar3_idxs = tmp.nzind

    shockvar³2_idxs = setdiff(shock_idxs2, shock³_idxs, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)

    shockvar³_idxs = setdiff(shock_idxs3, shock³_idxs)#, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)

    𝐒³⁻ᵛ = 𝐒[3][cond_var_idx,var_vol³_idxs]
    𝐒³⁻ᵉ² = 𝐒[3][cond_var_idx,shockvar³2_idxs]
    𝐒³⁻ᵉ = 𝐒[3][cond_var_idx,shockvar³_idxs]
    𝐒³ᵉ  = 𝐒[3][cond_var_idx,shock³_idxs]
    𝐒⁻³  = 𝐒[3][T.past_not_future_and_mixed_idx,:]

    𝐒³⁻ᵛ    = nnz(𝐒³⁻ᵛ)    / length(𝐒³⁻ᵛ)  > .1 ? collect(𝐒³⁻ᵛ)    : 𝐒³⁻ᵛ
    𝐒³⁻ᵉ    = nnz(𝐒³⁻ᵉ)    / length(𝐒³⁻ᵉ)  > .1 ? collect(𝐒³⁻ᵉ)    : 𝐒³⁻ᵉ
    𝐒³ᵉ     = nnz(𝐒³ᵉ)     / length(𝐒³ᵉ)   > .1 ? collect(𝐒³ᵉ)     : 𝐒³ᵉ
    𝐒⁻³     = nnz(𝐒⁻³)     / length(𝐒⁻³)   > .1 ? collect(𝐒⁻³)     : 𝐒⁻³

    state₁ = state[1][T.past_not_future_and_mixed_idx]
    state₂ = state[2][T.past_not_future_and_mixed_idx]
    state₃ = state[3][T.past_not_future_and_mixed_idx]

    kronxx = [zeros(T.nExo^2) for _ in 1:size(data_in_deviations,2)]
    
    J = ℒ.I(T.nExo)
    
    II = sparse(ℒ.I(T.nExo^2))

    kronxxx = [zeros(T.nExo^3) for _ in 1:size(data_in_deviations,2)]

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))
    
    kron_buffer3 = ℒ.kron(J, zeros(T.nExo^2))

    kron_buffer4 = ℒ.kron(ℒ.kron(J, J), zeros(T.nExo))

    x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]
    
    state¹⁻ = state₁

    state¹⁻_vol = vcat(state¹⁻, 1)

    state²⁻ = state₂#[T.past_not_future_and_mixed_idx]

    state³⁻ = state₃#[T.past_not_future_and_mixed_idx]
   
    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
    
    𝐒ⁱ²ᵉ = [zero(𝐒²ᵉ) for _ in 1:size(data_in_deviations,2)]

    aug_state₁ = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]
    aug_state₁̂ = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]
    aug_state₂ = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]
    aug_state₃ = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]

    kron_aug_state₁ = [zeros(size(𝐒⁻¹,2)^2) for _ in 1:size(data_in_deviations,2)]

    jacc_tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ[1] * ℒ.kron(ℒ.I(T.nExo), x[1])
    
    jacc = [zero(jacc_tmp) for _ in 1:size(data_in_deviations,2)]
    
    λ = [zeros(size(jacc_tmp, 1)) for _ in 1:size(data_in_deviations,2)]
    
    λ[1] = jacc_tmp' \ x[1] * 2
    
    fXλp_tmp = [reshape(2 * 𝐒ⁱ²ᵉ[1]' * λ[1], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc_tmp'
                -jacc_tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
    fXλp = [zero(fXλp_tmp) for _ in 1:size(data_in_deviations,2)]
    
    kronxλ_tmp = ℒ.kron(x[1], λ[1])
    
    kronxλ = [kronxλ_tmp for _ in 1:size(data_in_deviations,2)]
    
    kronxxλ_tmp = ℒ.kron(x[1], kronxλ_tmp)
    
    kronxxλ = [kronxxλ_tmp for _ in 1:size(data_in_deviations,2)]

    II = sparse(ℒ.I(T.nExo^2))

    lI = 2 * ℒ.I(size(𝐒ⁱ, 2))

    𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

    # @timeit_debug timer "Loop" begin
    for i in axes(data_in_deviations,2)
        state¹⁻ = state₁

        state¹⁻_vol = vcat(state¹⁻, 1)

        state²⁻ = state₂#[T.past_not_future_and_mixed_idx]

        state³⁻ = state₃#[T.past_not_future_and_mixed_idx]

        shock_independent = copy(data_in_deviations[:,i])
    
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒¹⁻, state²⁻, -1, 1)

        ℒ.mul!(shock_independent, 𝐒¹⁻, state³⁻, -1, 1)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
        
        ℒ.mul!(shock_independent, 𝐒²⁻, ℒ.kron(state¹⁻, state²⁻), -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒²⁻ᵛᵉ * ℒ.kron(ℒ.I(T.nExo), state²⁻) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
    
        𝐒ⁱ²ᵉ[i] = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

        𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

        init_guess = zeros(size(𝐒ⁱ, 2))
    
        # @timeit_debug timer "Find shocks" begin
        x[i], matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kronxx[i],
                                kronxxx[i],
                                kron_buffer2,
                                kron_buffer3,
                                kron_buffer4,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ[i],
                                𝐒ⁱ³ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )
        # end # timeit_debug

        if !matched
            if opts.verbose println("Inversion filter failed at step $i") end
            return on_failure_loglikelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
        end 
        
        jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ[i] * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), kronxx[i])
    
        λ[i] = jacc[i]' \ x[i] * 2
        # ℒ.ldiv!(λ[i], tmp', x[i])
        # ℒ.rmul!(λ[i], 2)
        fXλp[i] = [reshape((2 * 𝐒ⁱ²ᵉ[i] + 6 * 𝐒ⁱ³ᵉ * ℒ.kron(II, x[i]))' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - lI  jacc[i]'
                    -jacc[i]  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
        ℒ.kron!(kronxx[i], x[i], x[i])
    
        ℒ.kron!(kronxλ[i], x[i], λ[i])
    
        ℒ.kron!(kronxxλ[i], x[i], kronxλ[i])

        ℒ.kron!(kronxxx[i], x[i], kronxx[i])

        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc[i])[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i]))
            end
    
            shocks² += sum(abs2,x[i])

            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_loglikelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
            end
        end
    
        aug_state₁[i] = [state₁; 1; x[i]]
        aug_state₁̂[i] = [state₁; 0; x[i]]
        aug_state₂[i] = [state₂; 0; zeros(T.nExo)]
        aug_state₃[i] = [state₃; 0; zeros(T.nExo)]

        kron_aug_state₁[i] = ℒ.kron(aug_state₁[i], aug_state₁[i])

        state₁, state₂, state₃ = [𝐒⁻¹ * aug_state₁[i], 𝐒⁻¹ * aug_state₂[i] + 𝐒⁻² * kron_aug_state₁[i] / 2, 𝐒⁻¹ * aug_state₃[i] + 𝐒⁻² * ℒ.kron(aug_state₁̂[i], aug_state₂[i]) + 𝐒⁻³ * ℒ.kron(kron_aug_state₁[i], aug_state₁[i]) / 6]
    end
    # end # timeit_debug

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    llh = -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2

    ∂state = similar(state)

    ∂𝐒 = copy(𝐒)

    ∂data_in_deviations = similar(data_in_deviations)

    # end # timeit_debug

    function inversion_filter_loglikelihood_pullback(∂llh)
        # @timeit_debug timer "Inversion filter - pullback" begin
        ∂𝐒ⁱ = zero(𝐒ⁱ)
        ∂𝐒²ᵉ = zero(𝐒²ᵉ)
        ∂𝐒ⁱ³ᵉ = zero(𝐒ⁱ³ᵉ)

        ∂𝐒¹ᵉ = zero(𝐒¹ᵉ)
        ∂𝐒¹⁻ = zero(𝐒¹⁻)
        ∂𝐒²⁻ = zero(𝐒²⁻)
        ∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)
        ∂𝐒²⁻ᵛᵉ = zero(𝐒²⁻ᵛᵉ)
        ∂𝐒³⁻ᵉ = zero(𝐒³⁻ᵉ)
        ∂𝐒³⁻ᵉ² = zero(𝐒³⁻ᵉ²)

        ∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)
        ∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)
        ∂𝐒³⁻ᵛ = zero(𝐒³⁻ᵛ)
        
        ∂𝐒⁻¹ = zero(𝐒⁻¹)
        ∂𝐒⁻² = zero(𝐒⁻²)
        ∂𝐒⁻³ = zero(𝐒⁻³)

        ∂aug_state₁̂ = zero(aug_state₁̂[1])
        ∂state¹⁻_vol = zero(state¹⁻_vol)
        ∂x = zero(x[1])
        ∂kronxx = zero(kronxx[1])
        ∂kronstate¹⁻_vol = zeros(length(state¹⁻_vol)^2)
        ∂state = [zeros(T.nPast_not_future_and_mixed), zeros(T.nPast_not_future_and_mixed), zeros(T.nPast_not_future_and_mixed)]

        # @timeit_debug timer "Loop" begin
        for i in reverse(axes(data_in_deviations,2))
            # state₁ = 𝐒⁻¹ * aug_state₁[i]
            ∂𝐒⁻¹ += ∂state[1] * aug_state₁[i]'

            ∂aug_state₁ = 𝐒⁻¹' * ∂state[1]

            # state₂ = 𝐒⁻¹ * aug_state₂[i] + 𝐒⁻² * kron_aug_state₁[i] / 2
            ∂𝐒⁻¹ += ∂state[2] * aug_state₂[i]'

            ∂aug_state₂ = 𝐒⁻¹' * ∂state[2]

            ∂𝐒⁻² += ∂state[2] * kron_aug_state₁[i]' / 2

            ∂kronaug_state₁ = 𝐒⁻²' * ∂state[2] / 2

            # state₃ = 𝐒⁻¹ * aug_state₃[i] + 𝐒⁻² * ℒ.kron(aug_state₁̂[i], aug_state₂[i]) + 𝐒⁻³ * ℒ.kron(kron_aug_state₁[i],aug_state₁[i]) / 6
            ∂𝐒⁻¹ += ∂state[3] * aug_state₃[i]'

            ∂aug_state₃ = 𝐒⁻¹' * ∂state[3]

            ∂𝐒⁻² += ∂state[3] * ℒ.kron(aug_state₁̂[i], aug_state₂[i])'

            ∂aug_state₁̂ *= 0

            ∂kronaug_state₁̂₂ = 𝐒⁻²' * ∂state[3]

            fill_kron_adjoint!(∂aug_state₁̂, ∂aug_state₂, ∂kronaug_state₁̂₂, aug_state₁̂[i], aug_state₂[i])

            ∂𝐒⁻³ += ∂state[3] * ℒ.kron(kron_aug_state₁[i],aug_state₁[i])' / 6

            ∂kronkronaug_state₁ = 𝐒⁻³' * ∂state[3] / 6

            fill_kron_adjoint!(∂aug_state₁, ∂kronaug_state₁, ∂kronkronaug_state₁, aug_state₁[i], kron_aug_state₁[i])
    
            # kron_aug_state₁[i] = ℒ.kron(aug_state₁[i], aug_state₁[i])
            fill_kron_adjoint!(∂aug_state₁, ∂aug_state₁, ∂kronaug_state₁, aug_state₁[i], aug_state₁[i])

            if i > 1 && i < size(data_in_deviations,2)
                ∂state[1] *= 0
                ∂state[2] *= 0
                ∂state[3] *= 0
            end

            # aug_state₁[i] = [state₁; 1; x[i]]
            ∂state[1] += ∂aug_state₁[1:length(∂state[1])]

            ∂x = ∂aug_state₁[T.nPast_not_future_and_mixed+2:end]

            # aug_state₁̂[i] = [state₁; 0; x[i]]
            ∂state[1] += ∂aug_state₁̂[1:length(∂state[1])]

            ∂x += ∂aug_state₁̂[T.nPast_not_future_and_mixed+2:end]

            # aug_state₂[i] = [state₂; 0; zeros(T.nExo)]
            ∂state[2] += ∂aug_state₂[1:length(∂state[1])]
            
            # aug_state₃[i] = [state₃; 0; zeros(T.nExo)]
            ∂state[3] += ∂aug_state₃[1:length(∂state[1])]

            # shocks² += sum(abs2,x[i])
            if i < size(data_in_deviations,2)
                ∂x -= copy(x[i])
            else
                ∂x += copy(x[i])
            end

            # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
            ∂jacc = try if size(jacc[i], 1) == size(jacc[i], 2)
                            inv(jacc[i])'
                        else
                            ℒ.pinv(jacc[i])'
                        end
                    catch
                        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end

            # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x, x))
            # ∂𝐒ⁱ = -∂jacc / 2 # fine

            ∂kronIx = 𝐒ⁱ²ᵉ[i]' * ∂jacc

            if i < size(data_in_deviations,2)
                fill_kron_adjoint_∂B!(∂kronIx, ∂x, -ℒ.I(T.nExo))
            else
                fill_kron_adjoint_∂B!(∂kronIx, ∂x, ℒ.I(T.nExo))
            end

            ∂𝐒ⁱ²ᵉ = -∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'

            ∂kronIxx = 𝐒ⁱ³ᵉ' * ∂jacc * 3 / 2

            ∂kronxx *= 0

            if i < size(data_in_deviations,2)
                fill_kron_adjoint_∂B!(∂kronIxx, ∂kronxx, -ℒ.I(T.nExo))
            else
                fill_kron_adjoint_∂B!(∂kronIxx, ∂kronxx, ℒ.I(T.nExo))
            end

            fill_kron_adjoint!(∂x, ∂x, ∂kronxx, x[i], x[i])

            ∂𝐒ⁱ³ᵉ -= ∂jacc * ℒ.kron(ℒ.I(T.nExo), kronxx[i])' * 3 / 2

            # find_shocks
            ∂xλ = vcat(∂x, zero(λ[i]))

            S = fXλp[i]' \ ∂xλ

            if i < size(data_in_deviations,2)
                S *= -1
            end

            ∂shock_independent = S[T.nExo+1:end] # fine

            # ∂𝐒ⁱ += S[1:T.nExo] * λ[i]' - S[T.nExo + 1:end] * x[i]' # fine
            copyto!(∂𝐒ⁱ, ℒ.kron(S[1:T.nExo], λ[i]) - ℒ.kron(x[i], S[T.nExo+1:end]))
            ∂𝐒ⁱ -= ∂jacc / 2 # fine
        
            ∂𝐒ⁱ²ᵉ += reshape(2 * ℒ.kron(S[1:T.nExo], ℒ.kron(x[i], λ[i])) - ℒ.kron(kronxx[i], S[T.nExo+1:end]), size(∂𝐒ⁱ²ᵉ))
            # ∂𝐒ⁱ²ᵉ += 2 * S[1:T.nExo] * kronxλ[i]' - S[T.nExo + 1:end] * kronxx[i]'

            ∂𝐒ⁱ³ᵉ += reshape(3 * ℒ.kron(S[1:T.nExo], ℒ.kron(ℒ.kron(x[i], x[i]), λ[i])) - ℒ.kron(kronxxx[i], S[T.nExo+1:end]), size(∂𝐒ⁱ³ᵉ))
            # ∂𝐒ⁱ³ᵉ += 3 * S[1:T.nExo] * kronxxλ[i]' - S[T.nExo + 1:end] * kronxxx[i]'

            # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒²⁻ᵛᵉ * ℒ.kron(ℒ.I(T.nExo), state²⁻) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
            ∂kronstate¹⁻_vol *= 0

            state¹⁻_vol = [aug_state₁[i][1:T.nPast_not_future_and_mixed];1] # define here as it is used multiple times later
            state¹⁻ = aug_state₁[i][1:T.nPast_not_future_and_mixed]
            state²⁻ = aug_state₂[i][1:T.nPast_not_future_and_mixed]
            state³⁻ = aug_state₃[i][1:T.nPast_not_future_and_mixed]

            ∂𝐒¹ᵉ += ∂𝐒ⁱ

            ∂state¹⁻_vol *= 0

            ∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

            fill_kron_adjoint_∂A!(∂kronIstate¹⁻_vol, ∂state¹⁻_vol, ℒ.I(T.nExo))

            ∂𝐒²⁻ᵉ += ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)'

            ∂kronIstate²⁻ = 𝐒²⁻ᵛᵉ' * ∂𝐒ⁱ

            fill_kron_adjoint_∂A!(∂kronIstate²⁻, ∂state[2], ℒ.I(T.nExo))

            ∂𝐒²⁻ᵛᵉ += ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state²⁻)'

            ∂kronIstate¹⁻_volstate¹⁻_vol = 𝐒³⁻ᵉ²' * ∂𝐒ⁱ / 2

            fill_kron_adjoint_∂A!(∂kronIstate¹⁻_volstate¹⁻_vol, ∂kronstate¹⁻_vol, ℒ.I(T.nExo))

            ∂𝐒³⁻ᵉ² += ∂𝐒ⁱ * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol)' / 2
            
            # 𝐒ⁱ²ᵉ[i] = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2
            ∂𝐒²ᵉ += ∂𝐒ⁱ²ᵉ / 2
            
            ∂𝐒³⁻ᵉ += ∂𝐒ⁱ²ᵉ * ℒ.kron(II, state¹⁻_vol)' / 2
            
            ∂kronIIstate¹⁻_vol = 𝐒³⁻ᵉ' * ∂𝐒ⁱ²ᵉ / 2

            fill_kron_adjoint_∂A!(∂kronIIstate¹⁻_vol, ∂state¹⁻_vol, II)

            # shock_independent = copy(data_in_deviations[:,i])
            ∂data_in_deviations[:,i] = ∂shock_independent

            # ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
            ∂𝐒¹⁻ᵛ -= ∂shock_independent * state¹⁻_vol'

            ∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

            # ℒ.mul!(shock_independent, 𝐒¹⁻, state²⁻, -1, 1)
            ∂𝐒¹⁻ -= ∂shock_independent * state²⁻'

            ∂state[2] -= 𝐒¹⁻' * ∂shock_independent

            # ℒ.mul!(shock_independent, 𝐒¹⁻, state³⁻, -1, 1)
            ∂𝐒¹⁻ -= ∂shock_independent * state³⁻'

            ∂state[3] -= 𝐒¹⁻' * ∂shock_independent

            # ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
            ∂𝐒²⁻ᵛ -= ∂shock_independent * ℒ.kron(state¹⁻_vol, state¹⁻_vol)' / 2

            ∂kronstate¹⁻_vol -= 𝐒²⁻ᵛ' * ∂shock_independent / 2

            # ℒ.mul!(shock_independent, 𝐒²⁻, ℒ.kron(state¹⁻, state²⁻), -1, 1)
            ∂𝐒²⁻ -= ∂shock_independent * ℒ.kron(state¹⁻, state²⁻)'

            ∂kronstate¹⁻²⁻ = -𝐒²⁻' * ∂shock_independent

            fill_kron_adjoint!(∂state[1], ∂state[2], ∂kronstate¹⁻²⁻, state¹⁻, state²⁻)

            # ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   
            ∂𝐒³⁻ᵛ -= ∂shock_independent * ℒ.kron(ℒ.kron(state¹⁻_vol, state¹⁻_vol), state¹⁻_vol)' / 6

            ∂kronstate¹⁻_volstate¹⁻_vol = -𝐒³⁻ᵛ' * ∂shock_independent / 6

            fill_kron_adjoint!(∂kronstate¹⁻_vol, ∂state¹⁻_vol, ∂kronstate¹⁻_volstate¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol), state¹⁻_vol)

            fill_kron_adjoint!(∂state¹⁻_vol, ∂state¹⁻_vol, ∂kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)

            # state¹⁻_vol = vcat(state¹⁻, 1)
            ∂state[1] += ∂state¹⁻_vol[1:end-1]
        end
        # end # timeit_debug

        ∂𝐒 = [copy(𝐒[1]) * 0, copy(𝐒[2]) * 0, copy(𝐒[3]) * 0]

        ∂𝐒[1][cond_var_idx,end-T.nExo+1:end] += ∂𝐒¹ᵉ
        ∂𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed] += ∂𝐒¹⁻
        ∂𝐒[2][cond_var_idx,var²_idxs] += ∂𝐒²⁻
        ∂𝐒[2][cond_var_idx,shockvar²_idxs] += ∂𝐒²⁻ᵉ
        ∂𝐒[2][cond_var_idx,shock²_idxs] += ∂𝐒²ᵉ
        ∂𝐒[2][cond_var_idx,shockvar_idxs] += ∂𝐒²⁻ᵛᵉ
        ∂𝐒[3][cond_var_idx,shockvar³2_idxs] += ∂𝐒³⁻ᵉ²
        ∂𝐒[3][cond_var_idx,shockvar³_idxs] += ∂𝐒³⁻ᵉ
        ∂𝐒[3][cond_var_idx,shock³_idxs] += ∂𝐒ⁱ³ᵉ / 6 # 𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

        ∂𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1] += ∂𝐒¹⁻ᵛ
        ∂𝐒[2][cond_var_idx,var_vol²_idxs] += ∂𝐒²⁻ᵛ
        ∂𝐒[3][cond_var_idx,var_vol³_idxs] += ∂𝐒³⁻ᵛ

        ∂𝐒[1][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻¹
        ∂𝐒[2][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻²
        ∂𝐒[3][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻³

        ∂𝐒[1] *= ∂llh
        ∂𝐒[2] *= ∂llh
        ∂𝐒[3] *= ∂llh

        ∂state[1] = ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state[1] * ∂llh
        ∂state[2] = ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state[2] * ∂llh
        ∂state[3] = ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state[3] * ∂llh

        # end # timeit_debug

        return NoTangent(), NoTangent(), ∂state, ∂𝐒, ∂data_in_deviations * ∂llh, NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
    end

    return llh, inversion_filter_loglikelihood_pullback
end

# Lines 3010-3462
function rrule(::typeof(calculate_inversion_filter_loglikelihood),
                ::Val{:third_order},
                state::Vector{Float64}, 
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                observables::Union{Vector{String}, Vector{Symbol}},
                T::timings; 
                # timer::TimerOutput = TimerOutput(),
                on_failure_loglikelihood = -Inf,
                warmup_iterations::Int = 0,
                presample_periods::Int = 0,
                opts::CalculationOptions = merge_calculation_options(),
                filter_algorithm::Symbol = :LagrangeNewton)
    # @timeit_debug timer "Inversion filter pruned 2nd - forward" begin
    # @timeit_debug timer "Preallocation" begin

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))
    
    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind
    
    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind
    
    shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind
    
    tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    var²_idxs = tmp.nzind
    
    𝐒⁻¹  = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    𝐒⁻¹ᵉ = 𝐒[1][T.past_not_future_and_mixed_idx,end-T.nExo+1:end]
    𝐒¹⁻  = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ  = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    𝐒²⁻  = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²ᵉ  = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻²  = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = nnz(𝐒²⁻)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    tmp = ℒ.kron(sv_in_s⁺, ℒ.kron(sv_in_s⁺, sv_in_s⁺)) |> sparse
    var_vol³_idxs = tmp.nzind

    tmp = ℒ.kron(ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1), zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs2 = tmp.nzind

    tmp = ℒ.kron(ℒ.kron(e_in_s⁺, e_in_s⁺), zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs3 = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
    shock³_idxs = tmp.nzind

    tmp = ℒ.kron(zero(e_in_s⁺) .+ 1, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
    shockvar1_idxs = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺)) |> sparse
    shockvar2_idxs = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1)) |> sparse
    shockvar3_idxs = tmp.nzind

    shockvar³2_idxs = setdiff(shock_idxs2, shock³_idxs, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)

    shockvar³_idxs = setdiff(shock_idxs3, shock³_idxs)#, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)

    𝐒³⁻ᵛ  = 𝐒[3][cond_var_idx,var_vol³_idxs]
    𝐒³⁻ᵉ² = 𝐒[3][cond_var_idx,shockvar³2_idxs]
    𝐒³⁻ᵉ  = 𝐒[3][cond_var_idx,shockvar³_idxs]
    𝐒³ᵉ   = 𝐒[3][cond_var_idx,shock³_idxs]
    𝐒⁻³   = 𝐒[3][T.past_not_future_and_mixed_idx,:]

    𝐒³⁻ᵛ    = nnz(𝐒³⁻ᵛ)    / length(𝐒³⁻ᵛ)  > .1 ? collect(𝐒³⁻ᵛ)    : 𝐒³⁻ᵛ
    𝐒³⁻ᵉ    = nnz(𝐒³⁻ᵉ)    / length(𝐒³⁻ᵉ)  > .1 ? collect(𝐒³⁻ᵉ)    : 𝐒³⁻ᵉ
    𝐒³ᵉ     = nnz(𝐒³ᵉ)     / length(𝐒³ᵉ)   > .1 ? collect(𝐒³ᵉ)     : 𝐒³ᵉ
    𝐒⁻³     = nnz(𝐒⁻³)     / length(𝐒⁻³)   > .1 ? collect(𝐒⁻³)     : 𝐒⁻³

    stt = state[T.past_not_future_and_mixed_idx]

    kronxx = [zeros(T.nExo^2) for _ in 1:size(data_in_deviations,2)]
    
    J = ℒ.I(T.nExo)
    
    kronxxx = [zeros(T.nExo^3) for _ in 1:size(data_in_deviations,2)]

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))
    
    kron_buffer3 = ℒ.kron(J, zeros(T.nExo^2))

    kron_buffer4 = ℒ.kron(ℒ.kron(J, J), zeros(T.nExo))

    x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]
    
    state¹⁻ = stt
    
    state¹⁻_vol = vcat(state¹⁻, 1)
    
    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
    
    𝐒ⁱ²ᵉ = [zero(𝐒²ᵉ) for _ in 1:size(data_in_deviations,2)]

    aug_state = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]
    
    tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ[1] * ℒ.kron(ℒ.I(T.nExo), x[1])
    
    jacc = [zero(tmp) for _ in 1:size(data_in_deviations,2)]
    
    λ = [zeros(size(tmp, 1)) for _ in 1:size(data_in_deviations,2)]
    
    λ[1] = tmp' \ x[1] * 2
    
    fXλp_tmp = [reshape(2 * 𝐒ⁱ²ᵉ[1]' * λ[1], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  tmp'
                -tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
    fXλp = [zero(fXλp_tmp) for _ in 1:size(data_in_deviations,2)]
    
    kronxλ_tmp = ℒ.kron(x[1], λ[1])
    
    kronxλ = [kronxλ_tmp for _ in 1:size(data_in_deviations,2)]
    
    kronxxλ_tmp = ℒ.kron(x[1], kronxλ_tmp)
    
    kronxxλ = [kronxxλ_tmp for _ in 1:size(data_in_deviations,2)]

    II = sparse(ℒ.I(T.nExo^2))

    lI = 2 * ℒ.I(size(𝐒ⁱ, 2))

    𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

    # end # timeit_debug
    # @timeit_debug timer "Main loop" begin

    for i in axes(data_in_deviations,2)
        state¹⁻ = stt
    
        state¹⁻_vol = vcat(state¹⁻, 1)
        
        shock_independent = copy(data_in_deviations[:,i])
    
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   
    
        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
    
        𝐒ⁱ²ᵉ[i] = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

        init_guess = zeros(size(𝐒ⁱ, 2))
    
        # @timeit_debug timer "Find shocks" begin
        x[i], matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kronxx[i],
                                kronxxx[i],
                                kron_buffer2,
                                kron_buffer3,
                                kron_buffer4,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ[i],
                                𝐒ⁱ³ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )
        # end # timeit_debug
    
        if !matched
            if opts.verbose println("Inversion filter failed at step $i") end
            return on_failure_loglikelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
        end

        jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ[i] * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), kronxx[i])
    
        λ[i] = jacc[i]' \ x[i] * 2
        # ℒ.ldiv!(λ[i], tmp', x[i])
        # ℒ.rmul!(λ[i], 2)
        fXλp[i] = [reshape((2 * 𝐒ⁱ²ᵉ[i] + 6 * 𝐒ⁱ³ᵉ * ℒ.kron(II, x[i]))' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - lI  jacc[i]'
                    -jacc[i]  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
        ℒ.kron!(kronxx[i], x[i], x[i])
    
        ℒ.kron!(kronxλ[i], x[i], λ[i])
    
        ℒ.kron!(kronxxλ[i], x[i], kronxλ[i])

        ℒ.kron!(kronxxx[i], x[i], kronxx[i])

        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc[i])[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i]))
            end
    
            shocks² += sum(abs2,x[i])
            
            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_loglikelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
            end
        end
    
        aug_state[i] = [stt; 1; x[i]]
    
        stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6
    end
    
    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    llh = -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2

    # end # timeit_debug
    # end # timeit_debug

    ∂state = similar(state)

    ∂𝐒 = copy(𝐒)

    ∂data_in_deviations = similar(data_in_deviations)

    function inversion_filter_loglikelihood_pullback(∂llh)
        # @timeit_debug timer "Inversion filter pruned 2nd - pullback" begin
        # @timeit_debug timer "Preallocation" begin

        ∂𝐒ⁱ = zero(𝐒ⁱ)
        ∂𝐒²ᵉ = zero(𝐒²ᵉ)
        ∂𝐒ⁱ³ᵉ = zero(𝐒ⁱ³ᵉ)

        ∂𝐒¹ᵉ = zero(𝐒¹ᵉ)
        ∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)
        ∂𝐒³⁻ᵉ = zero(𝐒³⁻ᵉ)
        ∂𝐒³⁻ᵉ² = zero(𝐒³⁻ᵉ²)

        ∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)
        ∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)
        ∂𝐒³⁻ᵛ = zero(𝐒³⁻ᵛ)
        
        ∂𝐒⁻¹ = zero(𝐒⁻¹)
        ∂𝐒⁻² = zero(𝐒⁻²)
        ∂𝐒⁻³ = zero(𝐒⁻³)

        ∂state¹⁻_vol = zero(state¹⁻_vol)
        ∂x = zero(x[1])
        ∂kronxx = zero(kronxx[1])
        ∂kronstate¹⁻_vol = zeros(length(state¹⁻_vol)^2)
        ∂state = zeros(T.nPast_not_future_and_mixed)

        # end # timeit_debug
        # @timeit_debug timer "Main loop" begin
        
        for i in reverse(axes(data_in_deviations,2))
            # stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6
            ∂𝐒⁻¹ += ∂state * aug_state[i]'
            
            ∂𝐒⁻² += ∂state * ℒ.kron(aug_state[i], aug_state[i])' / 2

            ∂𝐒⁻³ += ∂state * ℒ.kron(ℒ.kron(aug_state[i], aug_state[i]), aug_state[i])' / 6
            
            ∂aug_state = 𝐒⁻¹' * ∂state
            ∂kronaug_state = 𝐒⁻²' * ∂state / 2
            ∂kronkronaug_state = 𝐒⁻³' * ∂state / 6
    
            fill_kron_adjoint!(∂aug_state, ∂kronaug_state, ∂kronkronaug_state, aug_state[i], ℒ.kron(aug_state[i], aug_state[i]))
    
            fill_kron_adjoint!(∂aug_state, ∂aug_state, ∂kronaug_state, aug_state[i], aug_state[i])

            if i > 1 && i < size(data_in_deviations,2)
                ∂state *= 0
            end

            # aug_state[i] = [stt; 1; x[i]]
            ∂state += ∂aug_state[1:length(∂state)]

            # aug_state[i] = [stt; 1; x[i]]
            ∂x = ∂aug_state[T.nPast_not_future_and_mixed+2:end]

            # shocks² += sum(abs2,x[i])
            if i < size(data_in_deviations,2)
                ∂x -= copy(x[i])
            else
                ∂x += copy(x[i])
            end

            # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
            ∂jacc = try if size(jacc[i], 1) == size(jacc[i], 2)
                            inv(jacc[i])'
                        else
                            ℒ.pinv(jacc[i])'
                        end
                    catch
                        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end

            # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x, x))
            # ∂𝐒ⁱ = -∂jacc / 2 # fine

            ∂kronIx = 𝐒ⁱ²ᵉ[i]' * ∂jacc

            if i < size(data_in_deviations,2)
                fill_kron_adjoint_∂B!(∂kronIx, ∂x, -ℒ.I(T.nExo))
            else
                fill_kron_adjoint_∂B!(∂kronIx, ∂x, ℒ.I(T.nExo))
            end

            ∂𝐒ⁱ²ᵉ = -∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'

            ∂kronIxx = 𝐒ⁱ³ᵉ' * ∂jacc * 3 / 2
            
            ∂kronxx *= 0

            if i < size(data_in_deviations,2)
                fill_kron_adjoint_∂B!(∂kronIxx, ∂kronxx, -ℒ.I(T.nExo))
            else
                fill_kron_adjoint_∂B!(∂kronIxx, ∂kronxx, ℒ.I(T.nExo))
            end

            fill_kron_adjoint!(∂x, ∂x, ∂kronxx, x[i], x[i])

            ∂𝐒ⁱ³ᵉ -= ∂jacc * ℒ.kron(ℒ.I(T.nExo), kronxx[i])' * 3 / 2

            # find_shocks
            ∂xλ = vcat(∂x, zero(λ[i]))

            S = fXλp[i]' \ ∂xλ

            if i < size(data_in_deviations,2)
                S *= -1
            end

            ∂shock_independent = S[T.nExo+1:end] # fine

            # ∂𝐒ⁱ += S[1:T.nExo] * λ[i]' - S[T.nExo + 1:end] * x[i]' # fine
            copyto!(∂𝐒ⁱ, ℒ.kron(S[1:T.nExo], λ[i]) - ℒ.kron(x[i], S[T.nExo+1:end]))
            ∂𝐒ⁱ -= ∂jacc / 2 # fine
        
            ∂𝐒ⁱ²ᵉ += reshape(2 * ℒ.kron(S[1:T.nExo], ℒ.kron(x[i], λ[i])) - ℒ.kron(kronxx[i], S[T.nExo+1:end]), size(∂𝐒ⁱ²ᵉ))
            # ∂𝐒ⁱ²ᵉ += 2 * S[1:T.nExo] * kronxλ[i]' - S[T.nExo + 1:end] * kronxx[i]'

            ∂𝐒ⁱ³ᵉ += reshape(3 * ℒ.kron(S[1:T.nExo], ℒ.kron(ℒ.kron(x[i], x[i]), λ[i])) - ℒ.kron(kronxxx[i], S[T.nExo+1:end]), size(∂𝐒ⁱ³ᵉ))
            # ∂𝐒ⁱ³ᵉ += 3 * S[1:T.nExo] * kronxxλ[i]' - S[T.nExo + 1:end] * kronxxx[i]'

            # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
            ∂kronstate¹⁻_vol *= 0

            state¹⁻_vol = [aug_state[i][1:T.nPast_not_future_and_mixed];1] # define here as it is used multiple times later

            ∂𝐒¹ᵉ += ∂𝐒ⁱ

            ∂state¹⁻_vol *= 0

            ∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

            fill_kron_adjoint_∂A!(∂kronIstate¹⁻_vol, ∂state¹⁻_vol, ℒ.I(T.nExo))

            ∂𝐒²⁻ᵉ += ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)'

            ∂kronIstate¹⁻_volstate¹⁻_vol = 𝐒³⁻ᵉ²' * ∂𝐒ⁱ / 2

            fill_kron_adjoint_∂A!(∂kronIstate¹⁻_volstate¹⁻_vol, ∂kronstate¹⁻_vol, ℒ.I(T.nExo))

            ∂𝐒³⁻ᵉ² += ∂𝐒ⁱ * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol)' / 2
            

            # 𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2
            ∂𝐒²ᵉ += ∂𝐒ⁱ²ᵉ / 2
            
            ∂𝐒³⁻ᵉ += ∂𝐒ⁱ²ᵉ * ℒ.kron(II, state¹⁻_vol)' / 2
            
            ∂kronIIstate¹⁻_vol = 𝐒³⁻ᵉ' * ∂𝐒ⁱ²ᵉ / 2

            fill_kron_adjoint_∂A!(∂kronIIstate¹⁻_vol, ∂state¹⁻_vol, II)

            # shock_independent = copy(data_in_deviations[:,i])
            ∂data_in_deviations[:,i] = ∂shock_independent


            # ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
            ∂𝐒¹⁻ᵛ -= ∂shock_independent * state¹⁻_vol'

            ∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

            # ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
            ∂𝐒²⁻ᵛ -= ∂shock_independent * ℒ.kron(state¹⁻_vol, state¹⁻_vol)' / 2

            ∂kronstate¹⁻_vol -= 𝐒²⁻ᵛ' * ∂shock_independent / 2

            # ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   
            ∂𝐒³⁻ᵛ -= ∂shock_independent * ℒ.kron(ℒ.kron(state¹⁻_vol, state¹⁻_vol), state¹⁻_vol)' / 6

            ∂kronstate¹⁻_volstate¹⁻_vol = -𝐒³⁻ᵛ' * ∂shock_independent / 6

            fill_kron_adjoint!(∂kronstate¹⁻_vol, ∂state¹⁻_vol, ∂kronstate¹⁻_volstate¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol), state¹⁻_vol)     

            fill_kron_adjoint!(∂state¹⁻_vol, ∂state¹⁻_vol, ∂kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)

            # state¹⁻_vol = vcat(state¹⁻, 1)
            ∂state += ∂state¹⁻_vol[1:end-1]
        end

        # end # timeit_debug
        # @timeit_debug timer "Post allocation" begin

        ∂𝐒 = [copy(𝐒[1]) * 0, copy(𝐒[2]) * 0, copy(𝐒[3]) * 0]

        ∂𝐒[1][cond_var_idx,end-T.nExo+1:end] += ∂𝐒¹ᵉ
        ∂𝐒[2][cond_var_idx,shockvar²_idxs] += ∂𝐒²⁻ᵉ
        ∂𝐒[2][cond_var_idx,shock²_idxs] += ∂𝐒²ᵉ
        ∂𝐒[3][cond_var_idx,shockvar³2_idxs] += ∂𝐒³⁻ᵉ²
        ∂𝐒[3][cond_var_idx,shockvar³_idxs] += ∂𝐒³⁻ᵉ
        ∂𝐒[3][cond_var_idx,shock³_idxs] += ∂𝐒ⁱ³ᵉ / 6 # 𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

        ∂𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1] += ∂𝐒¹⁻ᵛ
        ∂𝐒[2][cond_var_idx,var_vol²_idxs] += ∂𝐒²⁻ᵛ
        ∂𝐒[3][cond_var_idx,var_vol³_idxs] += ∂𝐒³⁻ᵛ

        ∂𝐒[1][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻¹
        ∂𝐒[2][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻²
        ∂𝐒[3][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻³

        ∂𝐒[1] *= ∂llh
        ∂𝐒[2] *= ∂llh
        ∂𝐒[3] *= ∂llh

        return NoTangent(), NoTangent(), ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state * ∂llh, ∂𝐒, ∂data_in_deviations * ∂llh, NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
    end

    # end # timeit_debug
    # end # timeit_debug

    return llh, inversion_filter_loglikelihood_pullback
end

# ========== src/filter/find_shocks.jl ==========

# Lines 1054-1112
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

# Lines 1287-1350
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
