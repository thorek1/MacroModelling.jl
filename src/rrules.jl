# Zygote/ChainRulesCore rrule definitions for reverse-mode automatic differentiation
#
# This file centralizes rrule definitions for computing gradients via reverse-mode AD.
# Each rrule specifies how to propagate gradients backward through custom functions.
#
# Strategy for each rrule:
#   1. Compute the forward pass and store necessary intermediate values
#   2. Return the result and a pullback function
#   3. The pullback computes gradients w.r.t. inputs given upstream gradients
#   4. Use implicit differentiation for iterative solvers and matrix equations
#
# Functions covered:
#   - Basic operations: mat_mult_kron, sparse_preallocated!
#   - Steady states: get_NSSS_and_parameters, calculate_second/third_order_stochastic_steady_state
#   - Derivatives: calculate_jacobian, calculate_hessian, calculate_third_order_derivatives
#   - Solutions: calculate_first/second/third_order_solution
#   - Matrix equations: solve_sylvester_equation, solve_lyapunov_equation
#   - Filters: calculate_loglikelihood, run_kalman_iterations, find_shocks

# clear_solution_caches! is a pure side-effect (cache invalidation) with no
# differentiable outputs, so the pullback is a no-op.
function rrule(::typeof(clear_solution_caches!), 𝓂::ℳ, algorithm::Symbol)
    clear_solution_caches!(𝓂, algorithm)
    return nothing, _ -> (NoTangent(), NoTangent(), NoTangent())
end

function rrule(::typeof(mat_mult_kron),
                                A::AbstractSparseMatrix{R},
                                B::AbstractMatrix{T},
                                C::AbstractMatrix{T},
                                D::AbstractMatrix{S}) where {R <: Real, T <: Real, S <: Real}
    Y = mat_mult_kron(A, B, C, D)

    function mat_mult_kron_pullback(Ȳ)
        Ȳ = unthunk(Ȳ)
        if Ȳ isa AbstractZero
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        Ȳdense = Matrix(Ȳ)

        n_rowB = size(B, 1)
        n_colB = size(B, 2)
        n_rowC = size(C, 1)
        n_colC = size(C, 2)

        G = promote_type(eltype(B), eltype(C), eltype(D), Float64)

        ∂B = zeros(G, size(B))
        ∂C = zeros(G, size(C))
        ∂D = zeros(G, size(D))

        A_csc = A isa SparseMatrixCSC ? A : A.A
        nnzA = nnz(A_csc)
        nz_col = Vector{Int}(undef, nnzA)

        # Linked-list row index: avoids Dict{Int,Vector{Int}} allocation
        n_rows_A = size(A_csc, 1)
        row_head = zeros(Int, n_rows_A)
        row_next = zeros(Int, nnzA)
        @inbounds for col in size(A_csc, 2):-1:1
            for k in A_csc.colptr[col]:(A_csc.colptr[col + 1] - 1)
                nz_col[k] = col
                r = A_csc.rowval[k]
                row_next[k] = row_head[r]
                row_head[r] = k
            end
        end

        ∂A_nz = zeros(G, nnzA)
        Abar_vec = zeros(G, size(A_csc, 2))

        @inbounds for r in 1:n_rows_A
            row_head[r] == 0 && continue

            fill!(Abar_vec, zero(G))
            k = row_head[r]
            while k != 0
                Abar_vec[nz_col[k]] = A_csc.nzval[k]
                k = row_next[k]
            end

            Abar = reshape(Abar_vec, n_rowC, n_rowB)
            AbarB = Abar * B
            CAbarB = C' * AbarB
            vCAbarB = vec(CAbarB)

            g_row = collect(@view Ȳdense[r, :])

            ∂D .+= vCAbarB * g_row'

            vCAbarB̄ = D * g_row
            CAbarB̄ = reshape(vCAbarB̄, n_colC, n_colB)

            ∂C .+= AbarB * CAbarB̄'

            AbarB̄ = C * CAbarB̄
            ∂B .+= Abar' * AbarB̄

            Abar̄ = AbarB̄ * B'
            vecAbar̄ = vec(Abar̄)
            k = row_head[r]
            while k != 0
                ∂A_nz[k] += vecAbar̄[nz_col[k]]
                k = row_next[k]
            end
        end

        ∂A_csc = SparseMatrixCSC(size(A_csc, 1), size(A_csc, 2), copy(A_csc.colptr), copy(A_csc.rowval), ∂A_nz)

        return NoTangent(),
                ProjectTo(A)(∂A_csc),
                ProjectTo(B)(∂B),
                ProjectTo(C)(∂C),
                ProjectTo(D)(∂D)
    end

    return Y, mat_mult_kron_pullback
end



function rrule(::typeof(sparse_preallocated!), Ŝ::Matrix{T}; ℂ::higher_order_workspace{T,F,H} = Higher_order_workspace()) where {T <: Real, F <: AbstractFloat, H <: Real}
    project_Ŝ = ProjectTo(Ŝ)

    function sparse_preallocated_pullback(Ω̄)
        ΔΩ = unthunk(Ω̄)
        ΔŜ = project_Ŝ(ΔΩ)
        return NoTangent(), ΔŜ, NoTangent()
    end

    return sparse_preallocated!(Ŝ, ℂ = ℂ), sparse_preallocated_pullback
end

function rrule(::typeof(solve_stochastic_steady_state_newton),
                                                        ::Val{:second_order}, 
                                                        𝐒₁::Matrix{Float64}, 
                                                        𝐒₂::AbstractSparseMatrix{Float64}, 
                                                        x::Vector{Float64},
                                                        𝓂::ℳ;
                                                        # timer::TimerOutput = TimerOutput(),
                                                        tol::AbstractFloat = 1e-14)
    # @timeit_debug timer "Calculate SSS - forward" begin
    # @timeit_debug timer "Setup indices" begin

    # Get cached computational constants
    constants = initialise_constants!(𝓂)
    so = constants.second_order
    T = constants.post_model_macro
    s_in_s⁺ = so.s_in_s⁺
    s_in_s = so.s_in_s
    I_nPast = T.I_nPast
    
    kron_s⁺_s⁺ = so.kron_s⁺_s⁺
    
    kron_s⁺_s = so.kron_s⁺_s
    
    A = 𝐒₁[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
    B = 𝐒₂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
    
    # end # timeit_debug
      
    # @timeit_debug timer "Iterations" begin

    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    x_aug = Vector{Float64}(undef, length(x) + 1)
    x_aug[end] = 1.0

    ℂ = 𝓂.workspaces.second_order
    nPast = length(x)
    ensure_sss_kron_buffers!(ℂ, nPast; third_order=false)
    kron_x_aug_buf = ℂ.kron_x_aug_xx
    kron_x_aug_I = ℂ.kron_x_aug_I

    for i in 1:max_iters
        copyto!(x_aug, 1, x, 1, nPast)
        ℒ.kron!(kron_x_aug_buf, x_aug, x_aug)

        ℒ.kron!(kron_x_aug_I, x_aug, I_nPast)
        ∂x = (A + B * kron_x_aug_I - I_nPast)

        Δx = (A * x + B̂ * kron_x_aug_buf / 2 - x)
        ensure_dx_lu_buffer!(ℂ, ∂x, Δx)
        sol = 𝒮.solve!(ℂ.dx_lu_buffer)

        if sol.retcode != 𝒮.SciMLBase.ReturnCode.Default && !𝒮.SciMLBase.successful_retcode(sol.retcode)
            return x, false
        end
        copyto!(Δx, sol.u)

        if i > 5 && isapprox(A * x + B̂ * kron_x_aug_buf / 2, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end
    copyto!(x_aug, 1, x, 1, nPast)
    # Local kron for closure capture (workspace buffers may be overwritten before pullback runs)
    kron_x_aug = ℒ.kron(x_aug, x_aug)
    solved = isapprox(A * x + B̂ * kron_x_aug / 2, x, rtol = tol)         

    ∂𝐒₁ =  zero(𝐒₁)
    ∂𝐒₂ =  zero(𝐒₂)

    # end # timeit_debug
    # end # timeit_debug

    function second_order_stochastic_steady_state_pullback(∂x)
        # @timeit_debug timer "Calculate SSS - pullback" begin
        ∂x₁ = unthunk(∂x[1])
        S = -∂x₁' / (A + B * ℒ.kron(x_aug, I_nPast) - I_nPast)

        ∂𝐒₁[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] = S' * x'
        
        ∂𝐒₂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s⁺] = S' * kron_x_aug' / 2
        # end # timeit_debug

        return NoTangent(), NoTangent(), ∂𝐒₁, ∂𝐒₂, NoTangent(), NoTangent(), NoTangent()
    end

    return (x, solved), second_order_stochastic_steady_state_pullback
end


function rrule(::typeof(solve_stochastic_steady_state_newton),
                                                        ::Val{:third_order}, 
                                                        𝐒₁::Matrix{Float64}, 
                                                        𝐒₂::AbstractSparseMatrix{Float64}, 
                                                        𝐒₃::AbstractSparseMatrix{Float64},
                                                        x::Vector{Float64},
                                                        𝓂::ℳ;
                                                        tol::AbstractFloat = 1e-14)
    # Get cached computational constants
    so = ensure_computational_constants!(𝓂.constants)
    T = 𝓂.constants.post_model_macro
    s_in_s⁺ = so.s_in_s⁺
    s_in_s = so.s_in_s
    I_nPast = T.I_nPast
    
    kron_s⁺_s⁺ = so.kron_s⁺_s⁺
    
    kron_s⁺_s = so.kron_s⁺_s
    
    kron_s⁺_s⁺_s⁺ = so.kron_s⁺_s⁺_s⁺
    
    kron_s_s⁺_s⁺ = so.kron_s_s⁺_s⁺
    
    A = 𝐒₁[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed]
    B = 𝐒₂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
    C = 𝐒₃[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s_s⁺_s⁺]
    Ĉ = 𝐒₃[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺]

    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    x_aug = Vector{Float64}(undef, length(x) + 1)
    x_aug[end] = 1.0

    ℂ = 𝓂.workspaces.third_order
    nPast = length(x)
    ensure_sss_kron_buffers!(ℂ, nPast; third_order=true)
    kron_x_aug_buf = ℂ.kron_x_aug_xx
    kron_x_kron_buf = ℂ.kron_x_aug_x_kron
    kron_x_aug_I = ℂ.kron_x_aug_I
    kron_x_kron_I = ℂ.kron_x_kron_I

    for i in 1:max_iters
        copyto!(x_aug, 1, x, 1, nPast)
        ℒ.kron!(kron_x_aug_buf, x_aug, x_aug)
        ℒ.kron!(kron_x_kron_buf, x_aug, kron_x_aug_buf)

        ℒ.kron!(kron_x_aug_I, x_aug, I_nPast)
        ℒ.kron!(kron_x_kron_I, kron_x_aug_buf, I_nPast)
        ∂x = (A + B * kron_x_aug_I + C * kron_x_kron_I / 2 - I_nPast)

        Δx = (A * x + B̂ * kron_x_aug_buf / 2 + Ĉ * kron_x_kron_buf / 6 - x)
        ensure_dx_lu_buffer!(ℂ, ∂x, Δx)
        sol = 𝒮.solve!(ℂ.dx_lu_buffer)

        if sol.retcode != 𝒮.SciMLBase.ReturnCode.Default && !𝒮.SciMLBase.successful_retcode(sol.retcode)
            return x, false
        end
        copyto!(Δx, sol.u)

        if i > 5 && isapprox(A * x + B̂ * kron_x_aug_buf / 2 + Ĉ * kron_x_kron_buf / 6, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    copyto!(x_aug, 1, x, 1, nPast)
    # Local kron for closure capture (workspace buffers may be overwritten before pullback runs)
    kron_x_aug = ℒ.kron(x_aug, x_aug)
    kron_x_kron = ℒ.kron(x_aug, kron_x_aug)
    solved = isapprox(A * x + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6, x, rtol = tol)         

    ∂𝐒₁ =  zero(𝐒₁)
    ∂𝐒₂ =  zero(𝐒₂)
    ∂𝐒₃ =  zero(𝐒₃)

    function third_order_stochastic_steady_state_pullback(∂x)
        ∂x₁ = unthunk(∂x[1])
        S = -∂x₁' / (A + B * ℒ.kron(x_aug, I_nPast) + C * ℒ.kron(kron_x_aug, I_nPast) / 2 - I_nPast)

        ∂𝐒₁[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] = S' * x'
        
        ∂𝐒₂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s⁺] = S' * kron_x_aug' / 2

        ∂𝐒₃[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺] = S' * kron_x_kron' / 6

        return NoTangent(), NoTangent(), ∂𝐒₁, ∂𝐒₂, ∂𝐒₃, NoTangent(), NoTangent(), NoTangent()
    end

    return (x, solved), third_order_stochastic_steady_state_pullback
end


function rrule(::typeof(calculate_jacobian), 
                parameters, 
                SS_and_pars, 
                caches_obj::caches,
                jacobian_funcs::jacobian_functions,
                workspaces::workspaces)
    jacobian = calculate_jacobian(parameters, SS_and_pars, caches_obj, jacobian_funcs, workspaces)
    ∂∇₁_vec = ensure_first_order_cotangent_buffer!(workspaces.first_order, length(jacobian))

    function calculate_jacobian_pullback(∂∇₁)
        if ∂∇₁ isa Union{NoTangent, AbstractZero}
            return NoTangent(), zero(parameters), zero(SS_and_pars), NoTangent(), NoTangent(), NoTangent()
        end

        ∂∇₁u = unthunk(∂∇₁)
        copyto!(∂∇₁_vec, ∂∇₁u)

        jacobian_funcs.f_parameters(caches_obj.jacobian_parameters, parameters, SS_and_pars)
        jacobian_funcs.f_SS_and_pars(caches_obj.jacobian_SS_and_pars, parameters, SS_and_pars)

        ∂parameters = caches_obj.jacobian_parameters * ∂∇₁_vec
        ∂SS_and_pars = caches_obj.jacobian_SS_and_pars * ∂∇₁_vec
        return NoTangent(), ∂parameters, ∂SS_and_pars, NoTangent(), NoTangent(), NoTangent()
    end

    return jacobian, calculate_jacobian_pullback
end


function rrule(::typeof(calculate_hessian), 
                parameters, 
                SS_and_pars, 
                caches_obj::caches,
                hessian_funcs::hessian_functions,
                workspaces::workspaces)
    hessian = calculate_hessian(parameters, SS_and_pars, caches_obj, hessian_funcs, workspaces)
    ∂∇₂_vec = ensure_higher_order_cotangent_buffer!(workspaces.second_order, length(hessian))

    function calculate_hessian_pullback(∂∇₂)
        if ∂∇₂ isa Union{NoTangent, AbstractZero}
            return NoTangent(), zero(parameters), zero(SS_and_pars), NoTangent(), NoTangent(), NoTangent()
        end

        ∂∇₂u = unthunk(∂∇₂)
        copyto!(∂∇₂_vec, ∂∇₂u)

        hessian_funcs.f_parameters(caches_obj.hessian_parameters, parameters, SS_and_pars)
        hessian_funcs.f_SS_and_pars(caches_obj.hessian_SS_and_pars, parameters, SS_and_pars)

        ∂parameters = caches_obj.hessian_parameters * ∂∇₂_vec
        ∂SS_and_pars = caches_obj.hessian_SS_and_pars * ∂∇₂_vec

        return NoTangent(), ∂parameters, ∂SS_and_pars, NoTangent(), NoTangent(), NoTangent()
    end

    return hessian, calculate_hessian_pullback
end


function rrule(::typeof(calculate_third_order_derivatives), 
                parameters, 
                SS_and_pars, 
                caches_obj::caches,
                third_order_derivatives_funcs::third_order_derivatives_functions,
                workspaces::workspaces)
    third_order_derivatives = calculate_third_order_derivatives(parameters, SS_and_pars, caches_obj, third_order_derivatives_funcs, workspaces)
    ∂∇₃_vec = ensure_higher_order_cotangent_buffer!(workspaces.third_order, length(third_order_derivatives))

    function calculate_third_order_derivatives_pullback(∂∇₃)
        if ∂∇₃ isa Union{NoTangent, AbstractZero}
            return NoTangent(), zero(parameters), zero(SS_and_pars), NoTangent(), NoTangent(), NoTangent()
        end

        ∂∇₃u = unthunk(∂∇₃)
        copyto!(∂∇₃_vec, ∂∇₃u)

        third_order_derivatives_funcs.f_parameters(caches_obj.third_order_derivatives_parameters, parameters, SS_and_pars)
        third_order_derivatives_funcs.f_SS_and_pars(caches_obj.third_order_derivatives_SS_and_pars, parameters, SS_and_pars)
        
        ∂parameters = caches_obj.third_order_derivatives_parameters * ∂∇₃_vec
        ∂SS_and_pars = caches_obj.third_order_derivatives_SS_and_pars * ∂∇₃_vec

        return NoTangent(), ∂parameters, ∂SS_and_pars, NoTangent(), NoTangent(), NoTangent()
    end

    return third_order_derivatives, calculate_third_order_derivatives_pullback
end


function incremental_cotangent!(Δ, prev_ref::Base.RefValue)
    if Δ isa Union{NoTangent, AbstractZero}
        return Δ
    end

    Δu = unthunk(Δ)
    prev = prev_ref[]
    prev_ref[] = copy(Δu)

    if prev === nothing
        return Δu
    end

    return Δu .- prev
end

function rrule(::typeof(get_NSSS_and_parameters), 
                𝓂::ℳ, 
                parameter_values::Vector{S}; 
                opts::CalculationOptions = merge_calculation_options(),
                cold_start::Bool = false,
                estimation::Bool = false) where S <: Real
                # timer::TimerOutput = TimerOutput(),
    # @timeit_debug timer "Calculate NSSS - forward" begin
    ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    ms = 𝓂.constants.post_complete_parameters

    # Use custom steady state function if available, otherwise use default solver
    if 𝓂.functions.NSSS_custom isa Function
        vars_in_ss_equations = ms.vars_in_ss_equations
        expected_length = length(vars_in_ss_equations) + length(𝓂.equations.calibration_parameters)

        SS_and_pars_tmp = evaluate_custom_steady_state_function(
            𝓂,
            parameter_values,
            expected_length,
            length(𝓂.constants.post_complete_parameters.parameters),
        )

        residual = zeros(length(𝓂.equations.steady_state) + length(𝓂.equations.calibration))
        
        𝓂.functions.NSSS_check(residual, parameter_values, SS_and_pars_tmp)
        
        solution_error = ℒ.norm(residual)

        iters = 0

        # if !isfinite(solution_error) || solution_error > opts.tol.nsss.acceptance_tol
        #     throw(ArgumentError("Custom steady state function failed steady state check: residual $solution_error > $(opts.tol.nsss.acceptance_tol). Parameters: $(parameter_values). Steady state and parameters returned: $(SS_and_pars_tmp)."))
        # end
        X = ms.custom_ss_expand_matrix
        SS_and_pars = X * SS_and_pars_tmp
    else
        fastest_idx = 𝓂.constants.post_complete_parameters.nsss_fastest_solver_parameter_idx
        preferred_solver_parameter_idx = fastest_idx < 1 || fastest_idx > length(DEFAULT_SOLVER_PARAMETERS) ? 1 : fastest_idx
        SS_and_pars, (solution_error, iters) = solve_nsss_wrapper(parameter_values, 𝓂, opts.tol, opts.verbose, cold_start, DEFAULT_SOLVER_PARAMETERS, preferred_solver_parameter_idx = preferred_solver_parameter_idx)
    end

    # end # timeit_debug

    if solution_error > opts.tol.nsss.acceptance_tol || isnan(solution_error)
        # Update failed counter
        update_ss_counter!(𝓂.counters, false, estimation = estimation)
        return (SS_and_pars, (solution_error, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # Update success counter
    update_ss_counter!(𝓂.counters, true, estimation = estimation)

    # @timeit_debug timer "Calculate NSSS - pullback" begin

    custom_ss_expand_matrix = ms.custom_ss_expand_matrix

    ∂ = parameter_values
    C = SS_and_pars[ms.SS_and_pars_no_exo_idx] # [dyn_ss_idx])

    if eltype(𝓂.caches.NSSS_∂equations_∂parameters) != eltype(parameter_values)
        if 𝓂.caches.NSSS_∂equations_∂parameters isa SparseMatrixCSC
            jac_cache = similar(𝓂.caches.NSSS_∂equations_∂parameters, eltype(parameter_values))
            jac_cache.nzval .= 0
        else
            jac_cache = zeros(eltype(parameter_values), size(𝓂.caches.NSSS_∂equations_∂parameters))
        end
    else
        jac_cache = 𝓂.caches.NSSS_∂equations_∂parameters
    end

    if jac_cache isa SparseMatrixCSC
        jac_cache.nzval .= 0
    else
        fill!(jac_cache, zero(eltype(jac_cache)))
    end

    𝓂.functions.NSSS_∂equations_∂parameters(jac_cache, ∂, C)

    ∂SS_equations_∂parameters = jac_cache

    
    if eltype(𝓂.caches.NSSS_∂equations_∂SS_and_pars) != eltype(SS_and_pars)
        if 𝓂.caches.NSSS_∂equations_∂SS_and_pars isa SparseMatrixCSC
            jac_cache = similar(𝓂.caches.NSSS_∂equations_∂SS_and_pars, eltype(SS_and_pars))
            jac_cache.nzval .= 0
        else
            jac_cache = zeros(eltype(SS_and_pars), size(𝓂.caches.NSSS_∂equations_∂SS_and_pars))
        end
    else
        jac_cache = 𝓂.caches.NSSS_∂equations_∂SS_and_pars
    end

    if jac_cache isa SparseMatrixCSC
        jac_cache.nzval .= 0
    else
        fill!(jac_cache, zero(eltype(jac_cache)))
    end

    𝓂.functions.NSSS_∂equations_∂SS_and_pars(jac_cache, ∂, C)

    ∂SS_equations_∂SS_and_pars = jac_cache
    qme_ws = 𝓂.workspaces.first_order
    if ∂SS_equations_∂SS_and_pars isa SparseMatrixCSC
        rhs_n_rows = size(∂SS_equations_∂SS_and_pars, 1)::Int
        rhs_n_cols = size(∂SS_equations_∂parameters, 2)::Int

        if length(qme_ws.nsss_sparse_rhs) != rhs_n_rows
            qme_ws.nsss_sparse_rhs = zeros(eltype(SS_and_pars), rhs_n_rows)
        end

        if size(qme_ws.nsss_jvp_rhs, 1) != rhs_n_rows || size(qme_ws.nsss_jvp_rhs, 2) != rhs_n_cols
            qme_ws.nsss_jvp_rhs = zeros(eltype(SS_and_pars), rhs_n_rows, rhs_n_cols)
        end

        if size(qme_ws.nsss_sparse_lu_buffer.A, 1) != rhs_n_rows || size(qme_ws.nsss_sparse_lu_buffer.A, 2) != rhs_n_rows
            sparse_prob = 𝒮.LinearProblem(∂SS_equations_∂SS_and_pars, qme_ws.nsss_sparse_rhs)
            qme_ws.nsss_sparse_lu_buffer = 𝒮.init(sparse_prob,
                                                  𝒮.LUFactorization(),
                                                  verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)
        else
            qme_ws.nsss_sparse_lu_buffer.A = ∂SS_equations_∂SS_and_pars
        end

        for j in 1:rhs_n_cols
            @views copyto!(qme_ws.nsss_sparse_rhs, ∂SS_equations_∂parameters[:, j])
            qme_ws.nsss_sparse_lu_buffer.b = qme_ws.nsss_sparse_rhs
            sparse_sol = 𝒮.solve!(qme_ws.nsss_sparse_lu_buffer)

            if sparse_sol.retcode != 𝒮.SciMLBase.ReturnCode.Default && !𝒮.SciMLBase.successful_retcode(sparse_sol.retcode)
                return (SS_and_pars, (10.0, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end

            @views copyto!(qme_ws.nsss_jvp_rhs[:, j], qme_ws.nsss_sparse_lu_buffer.u)
        end

        ℒ.rmul!(qme_ws.nsss_jvp_rhs, -1)
        JVP = qme_ws.nsss_jvp_rhs
    else
        # Old way (≤v0.1.42): nsss_lu = lu(∂SS/∂SS_and_pars)
        qme_ws.fast_lu_ws_nsss, qme_ws.fast_lu_dims_nsss, solved_nsss, nsss_lu = factorize_lu!(Val(:FastLapack), ∂SS_equations_∂SS_and_pars,
                                                                                                 qme_ws.fast_lu_ws_nsss,
                                                                                                 qme_ws.fast_lu_dims_nsss)

        if !solved_nsss
            return (SS_and_pars, (10.0, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end

        rhs_dense = ∂SS_equations_∂parameters isa Matrix ? ∂SS_equations_∂parameters : Matrix(∂SS_equations_∂parameters)

        if size(qme_ws.nsss_jvp_rhs) != size(rhs_dense)
            qme_ws.nsss_jvp_rhs = zeros(eltype(rhs_dense), size(rhs_dense))
        end
        copyto!(qme_ws.nsss_jvp_rhs, rhs_dense)

        # JVP = -(∂SS/∂SS_and_pars \ ∂SS/∂parameters)
        solve_lu_left!(∂SS_equations_∂SS_and_pars,         # rhs ← ∂SS/∂SS_and_pars \ rhs
                       qme_ws.nsss_jvp_rhs,
                       qme_ws.fast_lu_ws_nsss,
                       nsss_lu)

        ℒ.rmul!(qme_ws.nsss_jvp_rhs, -1)                  # JVP = -JVP
        JVP = qme_ws.nsss_jvp_rhs
    end

    jvp_no_exo = custom_ss_expand_matrix * JVP

    # end # timeit_debug
    # end # timeit_debug

    # try block-gmres here
    function get_non_stochastic_steady_state_pullback(∂SS_and_pars)
        ∂SS = unthunk(∂SS_and_pars[1])
        if ∂SS isa Union{NoTangent, AbstractZero}
            return NoTangent(), NoTangent(), zeros(S, size(jvp_no_exo, 2)), NoTangent()
        end
        return NoTangent(), NoTangent(), jvp_no_exo' * ∂SS, NoTangent()
    end


    return (SS_and_pars, (solution_error, iters)), get_non_stochastic_steady_state_pullback
end

function rrule(::typeof(get_relevant_steady_state_and_state_update),
                ::Val{:first_order},
                parameter_values::Vector{S},
                𝓂::ℳ;
                opts::CalculationOptions = merge_calculation_options(),
                estimation::Bool = false) where S <: AbstractFloat
    constants_obj = initialise_constants!(𝓂)

    nsss_out, nsss_pb = rrule(get_NSSS_and_parameters,
                                𝓂,
                                parameter_values;
                                opts = opts,
                                estimation = estimation)

    SS_and_pars = nsss_out[1]
    solution_error = nsss_out[2][1]

    state = zeros(S, 𝓂.constants.post_model_macro.nVars)

    if solution_error > opts.tol.nsss.acceptance_tol
        y = (𝓂.constants, SS_and_pars, zeros(S, 0, 0), [state], false)

        pullback = function (ȳ)
            Δy = unthunk(ȳ)
            if Δy isa NoTangent || Δy isa AbstractZero
                return NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent()
            end

            ΔSS_and_pars = Δy[2]
            nsss_grads = nsss_pb((ΔSS_and_pars, NoTangent()))
            ∂parameter_values = nsss_grads[3]

            return NoTangent(), NoTangent(), ∂parameter_values, NoTangent()
        end

        return y, pullback
    end

    ∇₁, jac_pb = rrule(calculate_jacobian,
                        parameter_values,
                        SS_and_pars,
                        𝓂.caches,
                        𝓂.functions.jacobian,
                        𝓂.workspaces)

    first_out, first_pb = rrule(calculate_first_order_solution,
                                ∇₁,
                                constants_obj,
                                𝓂.workspaces,
                                𝓂.caches;
                                opts = opts,
                                initial_guess = 𝓂.caches.qme_solution,
                                parameter_values = parameter_values)

    𝐒₁ = first_out[1]
    solved = first_out[3]

    update_perturbation_counter!(𝓂.counters, solved, estimation = estimation, order = 1)

    if !solved
        y = (𝓂.constants, SS_and_pars, zeros(S, 0, 0), [state], false)

        pullback = function (ȳ)
            Δy = unthunk(ȳ)
            if Δy isa NoTangent || Δy isa AbstractZero
                return NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent()
            end

            ΔSS_and_pars = Δy[2]

            nsss_grads = nsss_pb((ΔSS_and_pars, NoTangent()))
            ∂parameter_values = nsss_grads[3]

            return NoTangent(), NoTangent(), ∂parameter_values, NoTangent()
        end

        return y, pullback
    end

    y = (𝓂.constants, SS_and_pars, 𝐒₁, [state], true)

    pullback = function (ȳ)
        Δy = unthunk(ȳ)
        if Δy isa NoTangent || Δy isa AbstractZero
            return NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent()
        end

        ΔSS_and_pars = Δy[2]
        Δ𝐒₁ = Δy[3]

        # When the caller passes NoTangent for the solution matrix cotangent
        # (e.g. filter failure), skip the first-order solution pullback and
        # only propagate through the steady-state.
        if Δ𝐒₁ isa Union{NoTangent, AbstractZero}
            nsss_grads = nsss_pb((ΔSS_and_pars, NoTangent()))
            return NoTangent(), NoTangent(), nsss_grads[3], NoTangent()
        end

        first_grads = first_pb((Δ𝐒₁, NoTangent(), NoTangent()))
        ∂∇₁ = first_grads[2]

        jac_grads = jac_pb(∂∇₁)
        ∂parameter_values = jac_grads[2]
        ∂SS_and_pars_from_jac = jac_grads[3]

        nsss_grads = nsss_pb((ΔSS_and_pars + ∂SS_and_pars_from_jac, NoTangent()))
        ∂parameter_values .+= nsss_grads[3]

        return NoTangent(), NoTangent(), ∂parameter_values, NoTangent()
    end

    return y, pullback
end

function rrule(::typeof(prepare_stochastic_steady_state_base_terms),
                parameters::Vector{Float64},
                𝓂::ℳ;
                opts::CalculationOptions = merge_calculation_options(),
                estimation::Bool = false)
    constants = initialise_constants!(𝓂)
    T = constants.post_model_macro
    nVars = T.nVars
    nPast = T.nPast_not_future_and_mixed
    nExo = T.nExo
    past_idx = T.past_not_future_and_mixed_idx

    (SS_and_pars, (solution_error, iters)), nsss_pullback =
        rrule(get_NSSS_and_parameters, 𝓂, parameters, opts = opts, estimation = estimation)

    if solution_error > opts.tol.nsss.acceptance_tol || isnan(solution_error)
        common = (false,
                  zeros(Float64, nVars),
                  SS_and_pars,
                  solution_error,
                  zeros(Float64,0,0),
                  spzeros(Float64,0,0),
                  zeros(Float64,0,0),
                  spzeros(Float64,0,0),
                  zeros(Float64,0),
                  constants)
        pullback = function (Δcommon)
            return NoTangent(), zeros(Float64, length(parameters)), NoTangent()
        end
        return common, pullback
    end

    ensure_model_structure_constants!(constants, 𝓂.equations.calibration_parameters)
    ms = constants.post_complete_parameters
    all_SS = expand_steady_state(SS_and_pars, ms)

    ∇₁, jacobian_pullback =
        rrule(calculate_jacobian, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces)

    (𝐒₁_raw, qme_sol, solved), first_order_pullback =
        rrule(calculate_first_order_solution, ∇₁, constants, 𝓂.workspaces, 𝓂.caches;
              opts = opts, initial_guess = 𝓂.caches.qme_solution,
              parameter_values = parameters)

    update_perturbation_counter!(𝓂.counters, solved, estimation = estimation, order = 1)

    if !solved
        common = (false,
                  all_SS,
                  SS_and_pars,
                  solution_error,
                  zeros(Float64,0,0),
                  spzeros(Float64,0,0),
                  zeros(Float64,0,0),
                  spzeros(Float64,0,0),
                  zeros(Float64,0),
                  constants)
        pullback = function (Δcommon)
            return NoTangent(), zeros(Float64, length(parameters)), NoTangent()
        end
        return common, pullback
    end

    ∇₂, hessian_pullback =
        rrule(calculate_hessian, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian, 𝓂.workspaces)

    (𝐒₂_raw, solved2), second_order_pullback =
        rrule(calculate_second_order_solution, ∇₁, ∇₂, 𝐒₁_raw, 𝓂.constants, 𝓂.workspaces, 𝓂.caches;
              initial_guess = 𝓂.caches.second_order_solution, opts = opts,
              parameter_values = parameters)

    update_perturbation_counter!(𝓂.counters, solved2, estimation = estimation, order = 2)

    if !solved2
        common = (false,
                  all_SS,
                  SS_and_pars,
                  solution_error,
                  zeros(Float64,0,0),
                  spzeros(Float64,0,0),
                  zeros(Float64,0,0),
                  spzeros(Float64,0,0),
                  zeros(Float64,0),
                  constants)
        pullback = function (Δcommon)
            return NoTangent(), zeros(Float64, length(parameters)), NoTangent()
        end
        return common, pullback
    end

    𝐔₂ = 𝓂.constants.second_order.𝐔₂
    𝐒₂ = (sparse(𝐒₂_raw) * 𝐔₂)::SparseMatrixCSC{Float64, Int}  # was: dense_to_sparse

    𝐒₁ = [𝐒₁_raw[:, 1:nPast] zeros(nVars) 𝐒₁_raw[:, nPast+1:end]]
    aug_state₁ = sparse([zeros(nPast); 1; zeros(nExo)])
    kron_aug1 = ℒ.kron(aug_state₁, aug_state₁)

    tmp = collect(T.I_nPast - 𝐒₁[past_idx, 1:nPast])
    rhs = collect((𝐒₂ * kron_aug1 / 2)[past_idx])
    tmp_for_pullback = copy(tmp)

    ensure_sss_tmp_lu_buffer!(𝓂.workspaces.second_order, tmp, rhs)
    tmp_sol = 𝒮.solve!(𝓂.workspaces.second_order.sss_tmp_lu_buffer)

    if tmp_sol.retcode != 𝒮.SciMLBase.ReturnCode.Default && !𝒮.SciMLBase.successful_retcode(tmp_sol.retcode)
        common = (false,
                  all_SS,
                  SS_and_pars,
                  solution_error,
                  zeros(Float64,0,0),
                  spzeros(Float64,0,0),
                  zeros(Float64,0,0),
                  spzeros(Float64,0,0),
                  zeros(Float64,0),
                  constants)
        pullback = function (Δcommon)
            return NoTangent(), zeros(Float64, length(parameters)), NoTangent()
        end
        return common, pullback
    end

    SSSstates = collect(tmp_sol.u)
    tmp_pb_lu_ws, tmp_pb_lu_dims = ensure_sss_pullback_fast_lu_workspace!(𝓂.workspaces.second_order, tmp_for_pullback)
    tmp_pb_lu_ws, tmp_pb_lu_dims, solved_tmp_pb_lu, tmp_pb_lu = factorize_lu!(Val(:FastLapack), tmp_for_pullback, tmp_pb_lu_ws, tmp_pb_lu_dims)
    𝓂.workspaces.second_order.fast_lu_ws_sss_pullback = tmp_pb_lu_ws
    𝓂.workspaces.second_order.fast_lu_dims_sss_pullback = tmp_pb_lu_dims
    use_fastlapack_tmp_pb = solved_tmp_pb_lu
    if !solved_tmp_pb_lu
        tmp_pb_lu_ws, tmp_pb_lu_dims, solved_tmp_pb_lu, tmp_pb_lu =
            factorize_lu!(Val(:Julia), tmp_for_pullback, tmp_pb_lu_ws, tmp_pb_lu_dims)
        @assert solved_tmp_pb_lu "Could not factorize preserved stochastic steady-state pullback matrix."
        use_fastlapack_tmp_pb = false
    end
    ∂rhs_buffer = zeros(Float64, length(SSSstates))

    common = (true,
              all_SS,
              SS_and_pars,
              solution_error,
              ∇₁,
              ∇₂,
              𝐒₁,
              𝐒₂_raw,
              SSSstates,
              constants)

    pullback = function (Δcommon)
        ∂all_SS = zeros(Float64, length(all_SS))
        ∂SS_and_pars_direct = zeros(Float64, length(SS_and_pars))
        ∂∇₁_direct = zeros(Float64, size(∇₁))
        ∂∇₂_direct = zeros(Float64, size(∇₂))
        ∂𝐒₁_aug = zeros(Float64, size(𝐒₁))
        ∂𝐒₂_raw_total = zeros(Float64, size(𝐒₂_raw))
        ∂SSSstates = zeros(Float64, length(SSSstates))

        if !(Δcommon isa Union{NoTangent, AbstractZero})
            v2 = Δcommon[2]
            v3 = Δcommon[3]
            v5 = Δcommon[5]
            v6 = Δcommon[6]
            v7 = Δcommon[7]
            v8 = Δcommon[8]
            v9 = Δcommon[9]
            ∂all_SS = v2 isa Union{NoTangent, AbstractZero} ? ∂all_SS : v2
            ∂SS_and_pars_direct = v3 isa Union{NoTangent, AbstractZero} ? ∂SS_and_pars_direct : v3
            ∂∇₁_direct = v5 isa Union{NoTangent, AbstractZero} ? ∂∇₁_direct : v5
            ∂∇₂_direct = v6 isa Union{NoTangent, AbstractZero} ? ∂∇₂_direct : v6
            ∂𝐒₁_aug = v7 isa Union{NoTangent, AbstractZero} ? ∂𝐒₁_aug : v7
            ∂𝐒₂_raw_total = v8 isa Union{NoTangent, AbstractZero} ? ∂𝐒₂_raw_total : v8
            ∂SSSstates = v9 isa Union{NoTangent, AbstractZero} ? ∂SSSstates : v9
        end

        if !isempty(∂SSSstates)
            copyto!(∂rhs_buffer, ∂SSSstates)
            solve_lu_left_transpose!(tmp_for_pullback, ∂rhs_buffer, tmp_pb_lu_ws, tmp_pb_lu;
                                     use_fastlapack_lu = use_fastlapack_tmp_pb)
            ∂tmp = -∂rhs_buffer * SSSstates'
            ∂𝐒₁_aug[past_idx, 1:nPast] .-= ∂tmp
            ∂𝐒₂_from_rhs = spzeros(Float64, size(𝐒₂)...)
            ∂𝐒₂_from_rhs[past_idx, :] += ∂rhs_buffer * kron_aug1' / 2
            ∂𝐒₂_raw_total += ∂𝐒₂_from_rhs * 𝐔₂'
        end

        X = ms.steady_state_expand_matrix
        ∂SS_and_pars_from_allSS = X' * ∂all_SS

        ∂𝐒₁_raw = hcat(∂𝐒₁_aug[:, 1:nPast], ∂𝐒₁_aug[:, nPast+2:end])

        so2_tangents = second_order_pullback((∂𝐒₂_raw_total, NoTangent()))
        ∂∇₁_from_so2 = so2_tangents[2]
        ∂∇₂_from_so2 = so2_tangents[3]
        ∂𝐒₁_raw_from_so2 = so2_tangents[4]

        ∂∇₂_total = ∂∇₂_from_so2 + ∂∇₂_direct
        hess_tangents = hessian_pullback(∂∇₂_total)
        ∂params_from_hess = hess_tangents[2]
        ∂SS_and_pars_from_hess = hess_tangents[3]

        ∂𝐒₁_raw_total = ∂𝐒₁_raw + ∂𝐒₁_raw_from_so2
        fo_tangents = first_order_pullback((∂𝐒₁_raw_total, NoTangent(), NoTangent()))
        ∂∇₁_from_fo = fo_tangents[2]

        ∂∇₁_total = ∂∇₁_from_so2 + ∂∇₁_from_fo + ∂∇₁_direct
        jac_tangents = jacobian_pullback(∂∇₁_total)
        ∂params_from_jac = jac_tangents[2]
        ∂SS_and_pars_from_jac = jac_tangents[3]

        ∂SS_and_pars_total = ∂SS_and_pars_from_allSS + ∂SS_and_pars_from_hess + ∂SS_and_pars_from_jac + ∂SS_and_pars_direct
        nsss_tangents = nsss_pullback((∂SS_and_pars_total, NoTangent()))
        ∂params_from_nsss = nsss_tangents[3]

        ∂parameters = ∂params_from_nsss + ∂params_from_jac + ∂params_from_hess

        return NoTangent(), ∂parameters, NoTangent()
    end

    return common, pullback
end

function rrule(::typeof(calculate_stochastic_steady_state),
                ::Val{:second_order},
                parameters::Vector{Float64},
                𝓂::ℳ;
                opts::CalculationOptions = merge_calculation_options(),
                estimation::Bool = false)
    common, common_pullback = rrule(prepare_stochastic_steady_state_base_terms,
                                    parameters,
                                    𝓂;
                                    opts = opts,
                                    estimation = estimation)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂_raw, SSSstates, _ = common

    if !ok
        result = (all_SS, false, SS_and_pars, solution_error,
                  zeros(Float64,0,0), spzeros(Float64,0,0), zeros(Float64,0,0), spzeros(Float64,0,0))
        pullback = function (Δresult)
            Δ = unthunk(Δresult)
            Δsss = zeros(Float64, length(all_SS))
            ΔSS_and_pars = zeros(Float64, length(SS_and_pars))
            if !(Δ isa Union{NoTangent, AbstractZero}) && hasmethod(getindex, Tuple{typeof(Δ), Int})
                v1 = Δ[1]
                v3 = Δ[3]
                Δsss = v1 isa Union{NoTangent, AbstractZero} ? Δsss : v1
                ΔSS_and_pars = v3 isa Union{NoTangent, AbstractZero} ? ΔSS_and_pars : v3
            end
            common_tangents = common_pullback((NoTangent(), Δsss, ΔSS_and_pars, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()))
            return NoTangent(), NoTangent(), common_tangents[2], NoTangent()
        end
        return result, pullback
    end

    # Expand compressed 𝐒₂_raw to full for stochastic SS computation
    𝐔₂ = 𝓂.constants.second_order.𝐔₂
    𝐒₂ = (sparse(𝐒₂_raw) * 𝐔₂)::SparseMatrixCSC{Float64, Int}  # was: dense_to_sparse

    so = 𝓂.constants.second_order
    nPast = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed
    kron_s⁺_s⁺ = so.kron_s⁺_s⁺
    A = 𝐒₁[:,1:nPast]
    B̂ = 𝐒₂[:,kron_s⁺_s⁺]

    newton_result, newton_pullback =
        rrule(solve_stochastic_steady_state_newton, Val(:second_order), 𝐒₁, 𝐒₂, collect(SSSstates), 𝓂)
    SSSstates_final, converged::Bool = newton_result

    if !converged
        result = (all_SS, false, SS_and_pars, solution_error,
                  zeros(Float64,0,0), spzeros(Float64,0,0), zeros(Float64,0,0), spzeros(Float64,0,0))
        pullback = function (Δresult)
            Δ = unthunk(Δresult)
            Δsss = zeros(Float64, length(all_SS))
            ΔSS_and_pars = zeros(Float64, length(SS_and_pars))
            if !(Δ isa Union{NoTangent, AbstractZero}) && hasmethod(getindex, Tuple{typeof(Δ), Int})
                v1 = Δ[1]
                v3 = Δ[3]
                Δsss = v1 isa Union{NoTangent, AbstractZero} ? Δsss : v1
                ΔSS_and_pars = v3 isa Union{NoTangent, AbstractZero} ? ΔSS_and_pars : v3
            end
            common_tangents = common_pullback((NoTangent(), Δsss, ΔSS_and_pars, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()))
            return NoTangent(), NoTangent(), common_tangents[2], NoTangent()
        end
        return result, pullback
    end

    state = A * SSSstates_final + B̂ * ℒ.kron(vcat(SSSstates_final,1), vcat(SSSstates_final,1)) / 2
    sss = all_SS + vec(state)
    result = (sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂)

    pullback = function (Δresult)
        Δ = unthunk(Δresult)
        Δsss = zeros(Float64, length(sss))
        ΔSS_and_pars = zeros(Float64, length(SS_and_pars))
        Δ∇₁ = zeros(Float64, size(∇₁))
        Δ∇₂ = zeros(Float64, size(∇₂))
        Δ𝐒₁ = zeros(Float64, size(𝐒₁))
        Δ𝐒₂ = spzeros(Float64, size(𝐒₂)...)
        if !(Δ isa Union{NoTangent, AbstractZero}) && hasmethod(getindex, Tuple{typeof(Δ), Int})
            v1 = Δ[1]
            v3 = Δ[3]
            v5 = Δ[5]
            v6 = Δ[6]
            v7 = Δ[7]
            v8 = Δ[8]
            Δsss = v1 isa Union{NoTangent, AbstractZero} ? Δsss : v1
            ΔSS_and_pars = v3 isa Union{NoTangent, AbstractZero} ? ΔSS_and_pars : v3
            Δ∇₁ = v5 isa Union{NoTangent, AbstractZero} ? Δ∇₁ : v5
            Δ∇₂ = v6 isa Union{NoTangent, AbstractZero} ? Δ∇₂ : v6
            Δ𝐒₁ = v7 isa Union{NoTangent, AbstractZero} ? Δ𝐒₁ : v7
            Δ𝐒₂ = v8 isa Union{NoTangent, AbstractZero} ? Δ𝐒₂ : v8
        end

        ∂state_vec = Δsss
        aug_sss = vcat(SSSstates_final, 1)
        kron_aug = ℒ.kron(aug_sss, aug_sss)

        ∂𝐒₁_from_state = zeros(Float64, size(𝐒₁))
        ∂𝐒₁_from_state[:, 1:nPast] += ∂state_vec * SSSstates_final'

        ∂𝐒₂_from_state = spzeros(Float64, size(𝐒₂)...)
        ∂𝐒₂_from_state[:, kron_s⁺_s⁺] += ∂state_vec * kron_aug' / 2

        ∂SSSstates_from_state = A' * ∂state_vec
        n_aug = length(aug_sss)
        I_aug = Matrix{Float64}(ℒ.I, n_aug, n_aug)
        pad = vcat(Matrix{Float64}(ℒ.I, nPast, nPast), zeros(1, nPast))
        dkron_dx = ℒ.kron(I_aug, aug_sss) * pad + ℒ.kron(aug_sss, I_aug) * pad
        ∂SSSstates_from_state += (B̂' * ∂state_vec)' * dkron_dx / 2 |> vec

        newton_tangents = newton_pullback((∂SSSstates_from_state, NoTangent()))
        ∂𝐒₁_newton = newton_tangents[3]
        ∂𝐒₂_newton = newton_tangents[4]

        # Convert full-space ∂𝐒₂ to compressed for common_pullback
        ∂𝐒₂_raw_total = (∂𝐒₂_from_state + ∂𝐒₂_newton + Δ𝐒₂) * 𝐔₂'

        common_tangents = common_pullback((NoTangent(),
                                           Δsss,
                                           ΔSS_and_pars,
                                           NoTangent(),
                                           Δ∇₁,
                                           Δ∇₂,
                                           ∂𝐒₁_from_state + ∂𝐒₁_newton + Δ𝐒₁,
                                           ∂𝐒₂_raw_total,
                                           NoTangent(),
                                           NoTangent()))

        return NoTangent(), NoTangent(), common_tangents[2], NoTangent()
    end

    return result, pullback
end

function rrule(::typeof(calculate_stochastic_steady_state),
                ::Val{:pruned_second_order},
                parameters::Vector{Float64},
                𝓂::ℳ;
                opts::CalculationOptions = merge_calculation_options(),
                estimation::Bool = false)
    common, common_pullback = rrule(prepare_stochastic_steady_state_base_terms,
                                    parameters,
                                    𝓂;
                                    opts = opts,
                                    estimation = estimation)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂_raw, SSSstates, _ = common

    if !ok
        result = (all_SS, false, SS_and_pars, solution_error,
                  zeros(Float64,0,0), spzeros(Float64,0,0), zeros(Float64,0,0), spzeros(Float64,0,0))
        pullback = function (Δresult)
            Δ = unthunk(Δresult)
            Δsss = zeros(Float64, length(all_SS))
            ΔSS_and_pars = zeros(Float64, length(SS_and_pars))
            if !(Δ isa Union{NoTangent, AbstractZero}) && hasmethod(getindex, Tuple{typeof(Δ), Int})
                v1 = Δ[1]
                v3 = Δ[3]
                Δsss = v1 isa Union{NoTangent, AbstractZero} ? Δsss : v1
                ΔSS_and_pars = v3 isa Union{NoTangent, AbstractZero} ? ΔSS_and_pars : v3
            end
            common_tangents = common_pullback((NoTangent(), Δsss, ΔSS_and_pars, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()))
            return NoTangent(), NoTangent(), common_tangents[2], NoTangent()
        end
        return result, pullback
    end

    # Expand compressed 𝐒₂_raw to full for stochastic SS computation
    𝐔₂ = 𝓂.constants.second_order.𝐔₂
    𝐒₂ = (sparse(𝐒₂_raw) * 𝐔₂)::SparseMatrixCSC{Float64, Int}  # was: dense_to_sparse

    T = 𝓂.constants.post_model_macro
    nPast = T.nPast_not_future_and_mixed
    aug_state₁ = sparse([zeros(nPast); 1; zeros(T.nExo)])
    kron_aug1 = ℒ.kron(aug_state₁, aug_state₁)

    state = 𝐒₁[:,1:nPast] * SSSstates + 𝐒₂ * kron_aug1 / 2
    sss = all_SS + vec(state)
    result = (sss, true, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂)

    pullback = function (Δresult)
        Δ = unthunk(Δresult)
        Δsss = zeros(Float64, length(sss))
        ΔSS_and_pars = zeros(Float64, length(SS_and_pars))
        Δ∇₁ = zeros(Float64, size(∇₁))
        Δ∇₂ = zeros(Float64, size(∇₂))
        Δ𝐒₁ = zeros(Float64, size(𝐒₁))
        Δ𝐒₂ = spzeros(Float64, size(𝐒₂)...)
        if !(Δ isa Union{NoTangent, AbstractZero}) && hasmethod(getindex, Tuple{typeof(Δ), Int})
            v1 = Δ[1]
            v3 = Δ[3]
            v5 = Δ[5]
            v6 = Δ[6]
            v7 = Δ[7]
            v8 = Δ[8]
            Δsss = v1 isa Union{NoTangent, AbstractZero} ? Δsss : v1
            ΔSS_and_pars = v3 isa Union{NoTangent, AbstractZero} ? ΔSS_and_pars : v3
            Δ∇₁ = v5 isa Union{NoTangent, AbstractZero} ? Δ∇₁ : v5
            Δ∇₂ = v6 isa Union{NoTangent, AbstractZero} ? Δ∇₂ : v6
            Δ𝐒₁ = v7 isa Union{NoTangent, AbstractZero} ? Δ𝐒₁ : v7
            Δ𝐒₂ = v8 isa Union{NoTangent, AbstractZero} ? Δ𝐒₂ : v8
        end

        ∂state_vec = Δsss
        ∂𝐒₁_from_state = zeros(Float64, size(𝐒₁))
        ∂𝐒₁_from_state[:, 1:nPast] += ∂state_vec * SSSstates'
        ∂𝐒₂_from_state = spzeros(Float64, size(𝐒₂)...)
        ∂𝐒₂_from_state += ∂state_vec * kron_aug1' / 2
        ∂SSSstates = 𝐒₁[:,1:nPast]' * ∂state_vec

        # Convert full-space ∂𝐒₂ to compressed for common_pullback
        ∂𝐒₂_raw_total = (∂𝐒₂_from_state + Δ𝐒₂) * 𝐔₂'

        common_tangents = common_pullback((NoTangent(),
                                           Δsss,
                                           ΔSS_and_pars,
                                           NoTangent(),
                                           Δ∇₁,
                                           Δ∇₂,
                                           ∂𝐒₁_from_state + Δ𝐒₁,
                                           ∂𝐒₂_raw_total,
                                           ∂SSSstates,
                                           NoTangent()))

        return NoTangent(), NoTangent(), common_tangents[2], NoTangent()
    end

    return result, pullback
end

function rrule(::typeof(calculate_stochastic_steady_state),
                ::Val{:third_order},
                parameters::Vector{Float64},
                𝓂::ℳ;
                opts::CalculationOptions = merge_calculation_options(),
                estimation::Bool = false)
    common, common_pullback = rrule(prepare_stochastic_steady_state_base_terms,
                                    parameters,
                                    𝓂;
                                    opts = opts,
                                    estimation = estimation)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂_raw, SSSstates, _ = common

    if !ok
        result = (all_SS, false, SS_and_pars, solution_error,
                  zeros(Float64,0,0), spzeros(Float64,0,0), spzeros(Float64,0,0), zeros(Float64,0,0), spzeros(Float64,0,0), spzeros(Float64,0,0))
        pullback = function (Δresult)
            Δ = unthunk(Δresult)
            Δsss = zeros(Float64, length(all_SS))
            ΔSS_and_pars = zeros(Float64, length(SS_and_pars))
            if !(Δ isa Union{NoTangent, AbstractZero}) && hasmethod(getindex, Tuple{typeof(Δ), Int})
                v1 = Δ[1]
                v3 = Δ[3]
                Δsss = v1 isa Union{NoTangent, AbstractZero} ? Δsss : v1
                ΔSS_and_pars = v3 isa Union{NoTangent, AbstractZero} ? ΔSS_and_pars : v3
            end
            common_tangents = common_pullback((NoTangent(), Δsss, ΔSS_and_pars, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()))
            return NoTangent(), NoTangent(), common_tangents[2], NoTangent()
        end
        return result, pullback
    end

    𝐔₂ = 𝓂.constants.second_order.𝐔₂
    𝐒₂ = (sparse(𝐒₂_raw) * 𝐔₂)::SparseMatrixCSC{Float64, Int}  # was: dense_to_sparse

    ∇₃, third_derivatives_pullback =
        rrule(calculate_third_order_derivatives, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives, 𝓂.workspaces)
    nPast = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed
    𝐒₁_raw = [𝐒₁[:, 1:nPast] 𝐒₁[:, nPast+2:end]]

    (𝐒₃, solved3), third_order_solution_pullback =
        rrule(calculate_third_order_solution, ∇₁, ∇₂, ∇₃, 𝐒₁_raw, 𝐒₂_raw,
              𝓂.constants,
              𝓂.workspaces,
              𝓂.caches;
              initial_guess = 𝓂.caches.third_order_solution,
              opts = opts,
              parameter_values = parameters)

    if !solved3
        result = (all_SS, false, SS_and_pars, solution_error,
                  zeros(Float64,0,0), spzeros(Float64,0,0), spzeros(Float64,0,0), zeros(Float64,0,0), spzeros(Float64,0,0), spzeros(Float64,0,0))
        pullback = function (Δresult)
            Δ = unthunk(Δresult)
            Δsss = zeros(Float64, length(all_SS))
            ΔSS_and_pars = zeros(Float64, length(SS_and_pars))
            if !(Δ isa Union{NoTangent, AbstractZero}) && hasmethod(getindex, Tuple{typeof(Δ), Int})
                v1 = Δ[1]
                v3 = Δ[3]
                Δsss = v1 isa Union{NoTangent, AbstractZero} ? Δsss : v1
                ΔSS_and_pars = v3 isa Union{NoTangent, AbstractZero} ? ΔSS_and_pars : v3
            end
            common_tangents = common_pullback((NoTangent(), Δsss, ΔSS_and_pars, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()))
            return NoTangent(), NoTangent(), common_tangents[2], NoTangent()
        end
        return result, pullback
    end

    𝐔₃ = 𝓂.constants.third_order.𝐔₃
    𝐒₃̂ = sparse(𝐒₃) * 𝐔₃  # was: dense_to_sparse

    so = 𝓂.constants.second_order
    nPast = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed
    kron_s⁺_s⁺ = so.kron_s⁺_s⁺
    kron_s⁺_s⁺_s⁺ = so.kron_s⁺_s⁺_s⁺

    A = 𝐒₁[:,1:nPast]
    B̂ = 𝐒₂[:,kron_s⁺_s⁺]
    Ĉ = 𝐒₃̂[:,kron_s⁺_s⁺_s⁺]

    newton_result, newton_pullback =
        rrule(solve_stochastic_steady_state_newton, Val(:third_order), 𝐒₁, 𝐒₂, 𝐒₃̂, collect(SSSstates), 𝓂)
    SSSstates_final, converged::Bool = newton_result

    if !converged
        result = (all_SS, false, SS_and_pars, solution_error,
                  zeros(Float64,0,0), spzeros(Float64,0,0), spzeros(Float64,0,0), zeros(Float64,0,0), spzeros(Float64,0,0), spzeros(Float64,0,0))
        pullback = function (Δresult)
            Δ = unthunk(Δresult)
            Δsss = zeros(Float64, length(all_SS))
            ΔSS_and_pars = zeros(Float64, length(SS_and_pars))
            if !(Δ isa Union{NoTangent, AbstractZero}) && hasmethod(getindex, Tuple{typeof(Δ), Int})
                v1 = Δ[1]
                v3 = Δ[3]
                Δsss = v1 isa Union{NoTangent, AbstractZero} ? Δsss : v1
                ΔSS_and_pars = v3 isa Union{NoTangent, AbstractZero} ? ΔSS_and_pars : v3
            end
            common_tangents = common_pullback((NoTangent(), Δsss, ΔSS_and_pars, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()))
            return NoTangent(), NoTangent(), common_tangents[2], NoTangent()
        end
        return result, pullback
    end

    aug_sss = vcat(SSSstates_final, 1)
    kron_aug = ℒ.kron(aug_sss, aug_sss)
    kron_aug3 = ℒ.kron(aug_sss, kron_aug)

    state = A * SSSstates_final + B̂ * kron_aug / 2 + Ĉ * kron_aug3 / 6
    sss = all_SS + vec(state)
    result = (sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃̂)

    pullback = function (Δresult)
        Δ = unthunk(Δresult)
        Δsss = zeros(Float64, length(sss))
        ΔSS_and_pars = zeros(Float64, length(SS_and_pars))
        Δ∇₁ = zeros(Float64, size(∇₁))
        Δ∇₂ = zeros(Float64, size(∇₂))
        Δ∇₃ = spzeros(Float64, size(∇₃)...)
        Δ𝐒₁ = zeros(Float64, size(𝐒₁))
        Δ𝐒₂ = spzeros(Float64, size(𝐒₂)...)
        Δ𝐒₃̂ = spzeros(Float64, size(𝐒₃̂)...)
        if !(Δ isa Union{NoTangent, AbstractZero}) && hasmethod(getindex, Tuple{typeof(Δ), Int})
            v1 = Δ[1]
            v3 = Δ[3]
            v5 = Δ[5]
            v6 = Δ[6]
            v7 = Δ[7]
            v8 = Δ[8]
            v9 = Δ[9]
            v10 = Δ[10]
            Δsss = v1 isa Union{NoTangent, AbstractZero} ? Δsss : v1
            ΔSS_and_pars = v3 isa Union{NoTangent, AbstractZero} ? ΔSS_and_pars : v3
            Δ∇₁ = v5 isa Union{NoTangent, AbstractZero} ? Δ∇₁ : v5
            Δ∇₂ = v6 isa Union{NoTangent, AbstractZero} ? Δ∇₂ : v6
            Δ∇₃ = v7 isa Union{NoTangent, AbstractZero} ? Δ∇₃ : v7
            Δ𝐒₁ = v8 isa Union{NoTangent, AbstractZero} ? Δ𝐒₁ : v8
            Δ𝐒₂ = v9 isa Union{NoTangent, AbstractZero} ? Δ𝐒₂ : v9
            Δ𝐒₃̂ = v10 isa Union{NoTangent, AbstractZero} ? Δ𝐒₃̂ : v10
        end

        ∂state_vec = Δsss

        ∂𝐒₁_from_state = zeros(Float64, size(𝐒₁))
        ∂𝐒₁_from_state[:, 1:nPast] += ∂state_vec * SSSstates_final'

        ∂𝐒₂_from_state = spzeros(Float64, size(𝐒₂)...)
        ∂𝐒₂_from_state[:, kron_s⁺_s⁺] += ∂state_vec * kron_aug' / 2

        ∂𝐒₃̂_from_state = spzeros(Float64, size(𝐒₃̂)...)
        ∂𝐒₃̂_from_state[:, kron_s⁺_s⁺_s⁺] += ∂state_vec * kron_aug3' / 6

        ∂SSSstates_from_state = A' * ∂state_vec
        n_aug = length(aug_sss)
        I_aug = Matrix{Float64}(ℒ.I, n_aug, n_aug)
        pad = vcat(Matrix{Float64}(ℒ.I, nPast, nPast), zeros(1, nPast))
        dkron_dx = ℒ.kron(I_aug, aug_sss) * pad + ℒ.kron(aug_sss, I_aug) * pad
        ∂SSSstates_from_state += (B̂' * ∂state_vec)' * dkron_dx / 2 |> vec

        dkron3_dx = ℒ.kron(pad, ℒ.kron(aug_sss, aug_sss)) +
                    ℒ.kron(aug_sss, ℒ.kron(pad, aug_sss)) +
                    ℒ.kron(aug_sss, ℒ.kron(aug_sss, pad))
        ∂SSSstates_from_state += (Ĉ' * ∂state_vec)' * dkron3_dx / 6 |> vec

        newton_tangents = newton_pullback((∂SSSstates_from_state, NoTangent()))
        ∂𝐒₁_newton = newton_tangents[3]
        ∂𝐒₂_newton = newton_tangents[4]
        ∂𝐒₃̂_newton = newton_tangents[5]

        ∂𝐒₃̂_total = ∂𝐒₃̂_from_state + ∂𝐒₃̂_newton + Δ𝐒₃̂
        ∂𝐒₃_raw = Matrix(∂𝐒₃̂_total) * 𝐔₃' 

        so3_tangents = third_order_solution_pullback((∂𝐒₃_raw, NoTangent()))
        ∂∇₁_from_so3 = so3_tangents[2] isa Union{NoTangent, AbstractZero} ? zero(∇₁) : so3_tangents[2]
        ∂∇₂_from_so3 = so3_tangents[3] isa Union{NoTangent, AbstractZero} ? zero(∇₂) : so3_tangents[3]
        ∂∇₃_from_so3 = so3_tangents[4] isa Union{NoTangent, AbstractZero} ? zero(∇₃) : so3_tangents[4]
        ∂𝐒₁_raw_from_so3 = so3_tangents[5] isa Union{NoTangent, AbstractZero} ? zero(𝐒₁_raw) : so3_tangents[5]
        ∂𝐒₂_raw_from_so3 = so3_tangents[6] isa Union{NoTangent, AbstractZero} ? zero(𝐒₂_raw) : so3_tangents[6]

        ∂𝐒₁_from_so3 = zeros(Float64, size(𝐒₁))
        ∂𝐒₁_from_so3[:, 1:nPast] = ∂𝐒₁_raw_from_so3[:, 1:nPast]
        ∂𝐒₁_from_so3[:, nPast+2:end] = ∂𝐒₁_raw_from_so3[:, nPast+1:end]

        ∂∇₃_total = Δ∇₃ + ∂∇₃_from_so3
        third_derivatives_tangents = third_derivatives_pullback(∂∇₃_total)
        ∂params_from_∇₃ = third_derivatives_tangents[2]
        ∂SS_and_pars_from_∇₃ = third_derivatives_tangents[3]

        # Convert full-space ∂𝐒₂ terms to compressed, then accumulate with compressed ∂𝐒₂_raw_from_so3
        ∂𝐒₂_raw_for_common = ∂𝐒₂_raw_from_so3 + (∂𝐒₂_from_state + ∂𝐒₂_newton + Δ𝐒₂) * 𝐔₂'

        common_tangents = common_pullback((NoTangent(),
                                           Δsss,
                                           ΔSS_and_pars + ∂SS_and_pars_from_∇₃,
                                           NoTangent(),
                                           Δ∇₁ + ∂∇₁_from_so3,
                                           Δ∇₂ + ∂∇₂_from_so3,
                                           ∂𝐒₁_from_state + ∂𝐒₁_newton + Δ𝐒₁ + ∂𝐒₁_from_so3,
                                           ∂𝐒₂_raw_for_common,
                                           NoTangent(),
                                           NoTangent()))

        ∂parameters = common_tangents[2] + ∂params_from_∇₃
        return NoTangent(), NoTangent(), ∂parameters, NoTangent()
    end

    return result, pullback
end

function rrule(::typeof(calculate_stochastic_steady_state),
                ::Val{:pruned_third_order},
                parameters::Vector{Float64},
                𝓂::ℳ;
                opts::CalculationOptions = merge_calculation_options(),
                estimation::Bool = false)
    common, common_pullback = rrule(prepare_stochastic_steady_state_base_terms,
                                    parameters,
                                    𝓂;
                                    opts = opts,
                                    estimation = estimation)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂_raw, SSSstates, _ = common

    if !ok
        result = (all_SS, false, SS_and_pars, solution_error,
                  zeros(Float64,0,0), spzeros(Float64,0,0), spzeros(Float64,0,0), zeros(Float64,0,0), spzeros(Float64,0,0), spzeros(Float64,0,0))
        pullback = function (Δresult)
            Δ = unthunk(Δresult)
            Δsss = zeros(Float64, length(all_SS))
            ΔSS_and_pars = zeros(Float64, length(SS_and_pars))
            if !(Δ isa Union{NoTangent, AbstractZero}) && hasmethod(getindex, Tuple{typeof(Δ), Int})
                v1 = Δ[1]
                v3 = Δ[3]
                Δsss = v1 isa Union{NoTangent, AbstractZero} ? Δsss : v1
                ΔSS_and_pars = v3 isa Union{NoTangent, AbstractZero} ? ΔSS_and_pars : v3
            end
            common_tangents = common_pullback((NoTangent(), Δsss, ΔSS_and_pars, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()))
            return NoTangent(), NoTangent(), common_tangents[2], NoTangent()
        end
        return result, pullback
    end

    𝐔₂ = 𝓂.constants.second_order.𝐔₂
    𝐒₂ = (sparse(𝐒₂_raw) * 𝐔₂)::SparseMatrixCSC{Float64, Int}  # was: dense_to_sparse

    ∇₃, third_derivatives_pullback =
        rrule(calculate_third_order_derivatives, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives, 𝓂.workspaces)
    nPast = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed
    𝐒₁_raw = [𝐒₁[:, 1:nPast] 𝐒₁[:, nPast+2:end]]

    (𝐒₃, solved3), third_order_solution_pullback =
        rrule(calculate_third_order_solution, ∇₁, ∇₂, ∇₃, 𝐒₁_raw, 𝐒₂_raw,
              𝓂.constants,
              𝓂.workspaces,
              𝓂.caches;
              initial_guess = 𝓂.caches.third_order_solution,
              opts = opts,
              parameter_values = parameters)

    if !solved3
        result = (all_SS, false, SS_and_pars, solution_error,
                  zeros(Float64,0,0), spzeros(Float64,0,0), spzeros(Float64,0,0), zeros(Float64,0,0), spzeros(Float64,0,0), spzeros(Float64,0,0))
        pullback = function (Δresult)
            Δ = unthunk(Δresult)
            Δsss = zeros(Float64, length(all_SS))
            ΔSS_and_pars = zeros(Float64, length(SS_and_pars))
            if !(Δ isa Union{NoTangent, AbstractZero}) && hasmethod(getindex, Tuple{typeof(Δ), Int})
                v1 = Δ[1]
                v3 = Δ[3]
                Δsss = v1 isa Union{NoTangent, AbstractZero} ? Δsss : v1
                ΔSS_and_pars = v3 isa Union{NoTangent, AbstractZero} ? ΔSS_and_pars : v3
            end
            common_tangents = common_pullback((NoTangent(), Δsss, ΔSS_and_pars, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()))
            return NoTangent(), NoTangent(), common_tangents[2], NoTangent()
        end
        return result, pullback
    end

    𝐔₃ = 𝓂.constants.third_order.𝐔₃
    𝐒₃̂ = sparse(𝐒₃) * 𝐔₃  # was: dense_to_sparse

    T = 𝓂.constants.post_model_macro
    nPast = T.nPast_not_future_and_mixed
    aug_state₁ = sparse([zeros(nPast); 1; zeros(T.nExo)])
    kron_aug1 = ℒ.kron(aug_state₁, aug_state₁)

    state = 𝐒₁[:,1:nPast] * SSSstates + 𝐒₂ * kron_aug1 / 2
    sss = all_SS + vec(state)
    result = (sss, true, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃̂)

    pullback = function (Δresult)
        Δ = unthunk(Δresult)
        Δsss = zeros(Float64, length(sss))
        ΔSS_and_pars = zeros(Float64, length(SS_and_pars))
        Δ∇₁ = zeros(Float64, size(∇₁))
        Δ∇₂ = zeros(Float64, size(∇₂))
        Δ∇₃ = spzeros(Float64, size(∇₃)...)
        Δ𝐒₁ = zeros(Float64, size(𝐒₁))
        Δ𝐒₂ = spzeros(Float64, size(𝐒₂)...)
        Δ𝐒₃̂ = spzeros(Float64, size(𝐒₃̂)...)
        if !(Δ isa Union{NoTangent, AbstractZero}) && hasmethod(getindex, Tuple{typeof(Δ), Int})
            v1 = Δ[1]
            v3 = Δ[3]
            v5 = Δ[5]
            v6 = Δ[6]
            v7 = Δ[7]
            v8 = Δ[8]
            v9 = Δ[9]
            v10 = Δ[10]
            Δsss = v1 isa Union{NoTangent, AbstractZero} ? Δsss : v1
            ΔSS_and_pars = v3 isa Union{NoTangent, AbstractZero} ? ΔSS_and_pars : v3
            Δ∇₁ = v5 isa Union{NoTangent, AbstractZero} ? Δ∇₁ : v5
            Δ∇₂ = v6 isa Union{NoTangent, AbstractZero} ? Δ∇₂ : v6
            Δ∇₃ = v7 isa Union{NoTangent, AbstractZero} ? Δ∇₃ : v7
            Δ𝐒₁ = v8 isa Union{NoTangent, AbstractZero} ? Δ𝐒₁ : v8
            Δ𝐒₂ = v9 isa Union{NoTangent, AbstractZero} ? Δ𝐒₂ : v9
            Δ𝐒₃̂ = v10 isa Union{NoTangent, AbstractZero} ? Δ𝐒₃̂ : v10
        end

        ∂state_vec = Δsss
        ∂𝐒₁_from_state = zeros(Float64, size(𝐒₁))
        ∂𝐒₁_from_state[:, 1:nPast] += ∂state_vec * SSSstates'
        ∂𝐒₂_from_state = spzeros(Float64, size(𝐒₂)...)
        ∂𝐒₂_from_state += ∂state_vec * kron_aug1' / 2
        ∂SSSstates = 𝐒₁[:,1:nPast]' * ∂state_vec

        ∂𝐒₃_raw = Matrix(Δ𝐒₃̂) * 𝐔₃'
        so3_tangents = third_order_solution_pullback((∂𝐒₃_raw, NoTangent()))
        ∂∇₁_from_so3 = so3_tangents[2] isa Union{NoTangent, AbstractZero} ? zero(∇₁) : so3_tangents[2]
        ∂∇₂_from_so3 = so3_tangents[3] isa Union{NoTangent, AbstractZero} ? zero(∇₂) : so3_tangents[3]
        ∂∇₃_from_so3 = so3_tangents[4] isa Union{NoTangent, AbstractZero} ? zero(∇₃) : so3_tangents[4]
        ∂𝐒₁_raw_from_so3 = so3_tangents[5] isa Union{NoTangent, AbstractZero} ? zero(𝐒₁_raw) : so3_tangents[5]
        ∂𝐒₂_raw_from_so3 = so3_tangents[6] isa Union{NoTangent, AbstractZero} ? zero(𝐒₂_raw) : so3_tangents[6]

        ∂𝐒₁_from_so3 = zeros(Float64, size(𝐒₁))
        ∂𝐒₁_from_so3[:, 1:nPast] = ∂𝐒₁_raw_from_so3[:, 1:nPast]
        ∂𝐒₁_from_so3[:, nPast+2:end] = ∂𝐒₁_raw_from_so3[:, nPast+1:end]

        ∂∇₃_total = Δ∇₃ + ∂∇₃_from_so3
        third_derivatives_tangents = third_derivatives_pullback(∂∇₃_total)
        ∂params_from_∇₃ = third_derivatives_tangents[2]
        ∂SS_and_pars_from_∇₃ = third_derivatives_tangents[3]

        # Convert full-space ∂𝐒₂ terms to compressed, then accumulate with compressed ∂𝐒₂_raw_from_so3
        ∂𝐒₂_raw_for_common = ∂𝐒₂_raw_from_so3 + (∂𝐒₂_from_state + Δ𝐒₂) * 𝐔₂'

        common_tangents = common_pullback((NoTangent(),
                                           Δsss,
                                           ΔSS_and_pars + ∂SS_and_pars_from_∇₃,
                                           NoTangent(),
                                           Δ∇₁ + ∂∇₁_from_so3,
                                           Δ∇₂ + ∂∇₂_from_so3,
                                           ∂𝐒₁_from_state + Δ𝐒₁ + ∂𝐒₁_from_so3,
                                           ∂𝐒₂_raw_for_common,
                                           ∂SSSstates,
                                           NoTangent()))

        ∂parameters = common_tangents[2] + ∂params_from_∇₃
        return NoTangent(), NoTangent(), ∂parameters, NoTangent()
    end

    return result, pullback
end


function rrule(::typeof(get_relevant_steady_state_and_state_update),
                ::Val{:second_order},
                parameter_values::Vector{S},
                𝓂::ℳ;
                opts::CalculationOptions = merge_calculation_options(),
                estimation::Bool = false) where S <: AbstractFloat
    # Call inner rrule in the forward pass to capture pullback (avoids re-computing in backward)
    ss_rrule = rrule(calculate_stochastic_steady_state,
                        Val(:second_order),
                        parameter_values,
                        𝓂;
                        opts = opts,
                        estimation = estimation)

    if ss_rrule === nothing
        y = get_relevant_steady_state_and_state_update(Val(:second_order), parameter_values, 𝓂, opts = opts, estimation = estimation)
        return y, _ -> (NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent())
    end

    ss_out, ss_pb = ss_rrule
    sss = ss_out[1]
    converged = ss_out[2]
    SS_and_pars = ss_out[3]
    solution_error = ss_out[4]
    𝐒₁ = ss_out[7]
    𝐒₂ = ss_out[8]

    if !converged || solution_error > opts.tol.nsss.acceptance_tol
        y = (𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂], collect(sss), converged)
        return y, _ -> (NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent())
    end

    ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    ms = 𝓂.constants.post_complete_parameters
    all_SS = expand_steady_state(SS_and_pars, ms)
    state = collect(sss) - all_SS

    y = (𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂], state, converged)

    pullback = function (ȳ)
        Δy = unthunk(ȳ)
        if Δy isa NoTangent || Δy isa AbstractZero
            return NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent()
        end

        ΔSS_and_pars = Δy[2]
        Δ𝐒 = Δy[3]
        Δstate = Δy[4]

        # Guard against NoTangent cotangents from filter failure
        Δstate_val = Δstate isa Union{NoTangent, AbstractZero} ? zeros(S, length(state)) : Δstate
        Δ𝐒₁ = Δ𝐒 isa Union{NoTangent, AbstractZero} ? zeros(S, size(𝐒₁)) : Δ𝐒[1]
        Δ𝐒₂ = Δ𝐒 isa Union{NoTangent, AbstractZero} ? zeros(S, size(𝐒₂)) : Δ𝐒[2]

        Δsss = Δstate_val
        E = ms.steady_state_expand_matrix
        ΔSS_and_pars = ΔSS_and_pars - E' * Δstate_val

        ss_grads = ss_pb((Δsss,
                            NoTangent(),
                            ΔSS_and_pars,
                            NoTangent(),
                            NoTangent(),
                            NoTangent(),
                            Δ𝐒₁,
                            Δ𝐒₂))

        return NoTangent(), NoTangent(), ss_grads[3], NoTangent()
    end

    return y, pullback
end

function rrule(::typeof(get_relevant_steady_state_and_state_update),
                ::Val{:pruned_second_order},
                parameter_values::Vector{S},
                𝓂::ℳ;
                opts::CalculationOptions = merge_calculation_options(),
                estimation::Bool = false) where S <: AbstractFloat
    # Call inner rrule in the forward pass to capture pullback (avoids re-computing in backward)
    ss_rrule = rrule(calculate_stochastic_steady_state,
                        Val(:pruned_second_order),
                        parameter_values,
                        𝓂;
                        opts = opts,
                        estimation = estimation)

    if ss_rrule === nothing
        y = get_relevant_steady_state_and_state_update(Val(:pruned_second_order), parameter_values, 𝓂, opts = opts, estimation = estimation)
        return y, _ -> (NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent())
    end

    ss_out, ss_pb = ss_rrule
    sss = ss_out[1]
    converged = ss_out[2]
    SS_and_pars = ss_out[3]
    solution_error = ss_out[4]
    𝐒₁ = ss_out[7]
    𝐒₂ = ss_out[8]
    nVars = 𝓂.constants.post_model_macro.nVars

    if !converged || solution_error > opts.tol.nsss.acceptance_tol
        y = (𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂], [zeros(S, nVars), zeros(S, nVars)], converged)
        return y, _ -> (NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent())
    end

    ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    ms = 𝓂.constants.post_complete_parameters
    all_SS = expand_steady_state(SS_and_pars, ms)
    state = [zeros(S, nVars), collect(sss) - all_SS]

    y = (𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂], state, converged)

    pullback = function (ȳ)
        Δy = unthunk(ȳ)
        if Δy isa NoTangent || Δy isa AbstractZero
            return NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent()
        end

        ΔSS_and_pars = Δy[2]
        Δ𝐒 = Δy[3]
        Δstate = Δy[4]

        E = ms.steady_state_expand_matrix
        # Guard against NoTangent cotangents from filter failure
        Δstate_val = Δstate isa Union{NoTangent, AbstractZero} ? [zeros(S, nVars), zeros(S, nVars)] : Δstate
        Δ𝐒₁ = Δ𝐒 isa Union{NoTangent, AbstractZero} ? zeros(S, size(𝐒₁)) : Δ𝐒[1]
        Δ𝐒₂ = Δ𝐒 isa Union{NoTangent, AbstractZero} ? zeros(S, size(𝐒₂)) : Δ𝐒[2]

        Δsss = Δstate_val[2]
        ΔSS_and_pars = ΔSS_and_pars - E' * Δstate_val[2]

        ss_grads = ss_pb((Δsss,
                            NoTangent(),
                            ΔSS_and_pars,
                            NoTangent(),
                            NoTangent(),
                            NoTangent(),
                            Δ𝐒₁,
                            Δ𝐒₂))

        return NoTangent(), NoTangent(), ss_grads[3], NoTangent()
    end

    return y, pullback
end

function rrule(::typeof(get_relevant_steady_state_and_state_update),
                ::Val{:third_order},
                parameter_values::Vector{S},
                𝓂::ℳ;
                opts::CalculationOptions = merge_calculation_options(),
                estimation::Bool = false) where S <: AbstractFloat
    # Call inner rrule in the forward pass to capture pullback (avoids re-computing in backward)
    ss_rrule = rrule(calculate_stochastic_steady_state,
                        Val(:third_order),
                        parameter_values,
                        𝓂;
                        opts = opts,
                        estimation = estimation)

    if ss_rrule === nothing
        y = get_relevant_steady_state_and_state_update(Val(:third_order), parameter_values, 𝓂, opts = opts, estimation = estimation)
        return y, _ -> (NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent())
    end

    ss_out, ss_pb = ss_rrule
    sss = ss_out[1]
    converged = ss_out[2]
    SS_and_pars = ss_out[3]
    solution_error = ss_out[4]
    𝐒₁ = ss_out[8]
    𝐒₂ = ss_out[9]
    𝐒₃ = ss_out[10]

    if !converged || solution_error > opts.tol.nsss.acceptance_tol
        y = (𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], collect(sss), converged)
        return y, _ -> (NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent())
    end

    ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    ms = 𝓂.constants.post_complete_parameters
    all_SS = expand_steady_state(SS_and_pars, ms)
    state = collect(sss) - all_SS

    y = (𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], state, converged)

    pullback = function (ȳ)
        Δy = unthunk(ȳ)
        if Δy isa NoTangent || Δy isa AbstractZero
            return NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent()
        end

        ΔSS_and_pars = Δy[2]
        Δ𝐒 = Δy[3]
        Δstate = Δy[4]
        ΔSS_and_pars = ΔSS_and_pars isa Union{NoTangent, AbstractZero} ? zero(SS_and_pars) : ΔSS_and_pars

        # Guard against NoTangent cotangents from filter failure
        Δstate_val = Δstate isa Union{NoTangent, AbstractZero} ? zeros(S, length(state)) : Δstate
        Δ𝐒₁ = Δ𝐒 isa Union{NoTangent, AbstractZero} ? zero(𝐒₁) : Δ𝐒[1]
        Δ𝐒₂ = Δ𝐒 isa Union{NoTangent, AbstractZero} ? zero(𝐒₂) : Δ𝐒[2]
        Δ𝐒₃ = Δ𝐒 isa Union{NoTangent, AbstractZero} ? zero(𝐒₃) : Δ𝐒[3]

        Δsss = Δstate_val
        E = ms.steady_state_expand_matrix
        ΔSS_and_pars = ΔSS_and_pars - E' * Δstate_val

        ss_grads = ss_pb((Δsss,
                            NoTangent(),
                            ΔSS_and_pars,
                            NoTangent(),
                            NoTangent(),
                            NoTangent(),
                            NoTangent(),
                            Δ𝐒₁,
                            Δ𝐒₂,
                            Δ𝐒₃))

        return NoTangent(), NoTangent(), ss_grads[3], NoTangent()
    end
    return y, pullback
end

function rrule(::typeof(get_relevant_steady_state_and_state_update),
                ::Val{:pruned_third_order},
                parameter_values::Vector{S},
                𝓂::ℳ;
                opts::CalculationOptions = merge_calculation_options(),
                estimation::Bool = false) where S <: AbstractFloat
    # Call inner rrule in the forward pass to capture pullback (avoids re-computing in backward)
    ss_rrule = rrule(calculate_stochastic_steady_state,
                        Val(:pruned_third_order),
                        parameter_values,
                        𝓂;
                        opts = opts,
                        estimation = estimation)

    if ss_rrule === nothing
        y = get_relevant_steady_state_and_state_update(Val(:pruned_third_order), parameter_values, 𝓂, opts = opts, estimation = estimation)
        return y, _ -> (NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent())
    end

    ss_out, ss_pb = ss_rrule
    sss = ss_out[1]
    converged = ss_out[2]
    SS_and_pars = ss_out[3]
    solution_error = ss_out[4]
    𝐒₁ = ss_out[8]
    𝐒₂ = ss_out[9]
    𝐒₃ = ss_out[10]
    nVars = 𝓂.constants.post_model_macro.nVars

    if !converged || solution_error > opts.tol.nsss.acceptance_tol
        y = (𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], [zeros(S, nVars), zeros(S, nVars), zeros(S, nVars)], converged)
        return y, _ -> (NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent())
    end

    ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    ms = 𝓂.constants.post_complete_parameters
    all_SS = expand_steady_state(SS_and_pars, ms)
    state = [zeros(S, nVars), collect(sss) - all_SS, zeros(S, nVars)]

    y = (𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], state, converged)

    pullback = function (ȳ)
        Δy = unthunk(ȳ)
        if Δy isa NoTangent || Δy isa AbstractZero
            return NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent()
        end

        ΔSS_and_pars = Δy[2]
        Δ𝐒 = Δy[3]
        Δstate = Δy[4]

        E = ms.steady_state_expand_matrix
        # Guard against NoTangent cotangents from filter failure
        Δstate_val = Δstate isa Union{NoTangent, AbstractZero} ? [zeros(S, nVars), zeros(S, nVars), zeros(S, nVars)] : Δstate
        Δ𝐒₁ = Δ𝐒 isa Union{NoTangent, AbstractZero} ? zeros(S, size(𝐒₁)) : Δ𝐒[1]
        Δ𝐒₂ = Δ𝐒 isa Union{NoTangent, AbstractZero} ? zeros(S, size(𝐒₂)) : Δ𝐒[2]
        Δ𝐒₃ = Δ𝐒 isa Union{NoTangent, AbstractZero} ? zeros(S, size(𝐒₃)) : Δ𝐒[3]

        Δsss = Δstate_val[2]
        ΔSS_and_pars = ΔSS_and_pars - E' * Δstate_val[2]

        ss_grads = ss_pb((Δsss,
                            NoTangent(),
                            ΔSS_and_pars,
                            NoTangent(),
                            NoTangent(),
                            NoTangent(),
                            NoTangent(),
                            Δ𝐒₁,
                            Δ𝐒₂,
                            Δ𝐒₃))

        return NoTangent(), NoTangent(), ss_grads[3], NoTangent()
    end

    return y, pullback
end

function rrule(::typeof(get_loglikelihood),
                𝓂::ℳ,
                data::KeyedArray,
                parameter_values::Vector{S};
                steady_state_function::SteadyStateFunctionType = missing,
                algorithm::Symbol = DEFAULT_ALGORITHM,
                filter::Symbol = DEFAULT_FILTER_SELECTOR(algorithm),
                on_failure_loglikelihood::U = -Inf,
                warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS,
                presample_periods::Int = DEFAULT_PRESAMPLE_PERIODS,
                initial_covariance::Symbol = :theoretical,
                filter_algorithm::Symbol = :LagrangeNewton,
                tol::Tolerances = Tolerances(),
                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                verbose::Bool = DEFAULT_VERBOSE) where {S <: Real, U <: AbstractFloat}

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                            sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                            sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                            lyapunov_algorithm = lyapunov_algorithm)

    estimation = true

    filter, _, algorithm, _, _, warmup_iterations = normalize_filtering_options(filter, false, algorithm, false, warmup_iterations)

    observables = get_and_check_observables(𝓂.constants.post_model_macro, data)

    solve!(𝓂, opts = opts, steady_state_function = steady_state_function, algorithm = algorithm)

    bounds_violated = check_bounds(parameter_values, 𝓂)

    if bounds_violated
        llh = S(on_failure_loglikelihood)
        return llh, _ -> (NoTangent(), NoTangent(), NoTangent(), zeros(S, length(parameter_values)))
    end

    obs_indices = convert(Vector{Int}, indexin(observables, 𝓂.constants.post_complete_parameters.SS_and_pars_names))

    # ── step 1: get_relevant_steady_state_and_state_update ──
    ss_rrule = rrule(get_relevant_steady_state_and_state_update,
                     Val(algorithm), parameter_values, 𝓂;
                     opts = opts, estimation = estimation)

    if ss_rrule === nothing
        # fall back to primal-only when no rrule is available
        constants_obj, SS_and_pars, 𝐒, state, solved = get_relevant_steady_state_and_state_update(
            Val(algorithm), parameter_values, 𝓂, opts = opts, estimation = estimation)
        ss_pb = nothing
    else
        (constants_obj, SS_and_pars, 𝐒, state, solved), ss_pb = ss_rrule
    end

    if !solved
        llh = S(on_failure_loglikelihood)
        return llh, _ -> (NoTangent(), NoTangent(), NoTangent(), zeros(S, length(parameter_values)))
    end

    # ── step 2: data_in_deviations = dt .- SS_and_pars[obs_indices] ──
    dt = if collect(axiskeys(data, 1)) isa Vector{String}
        collect(rekey(data, 1 => axiskeys(data, 1) .|> Meta.parse .|> replace_indices)(observables))
    else
        collect(data(observables))
    end

    data_in_deviations = missing_data_to_nan(dt) .- SS_and_pars[obs_indices]

    obs_idx_per_t, has_missing = build_obs_index(data_in_deviations)

    # ── step 3: calculate_loglikelihood ──
    llh_rrule = if has_missing
        rrule(calculate_loglikelihood_with_missing,
              Val(filter), Val(algorithm), obs_indices,
              𝐒, data_in_deviations, constants_obj, state, 𝓂.workspaces, obs_idx_per_t;
              warmup_iterations = warmup_iterations,
              presample_periods = presample_periods,
              initial_covariance = initial_covariance,
              filter_algorithm = filter_algorithm,
              opts = opts,
              on_failure_loglikelihood = on_failure_loglikelihood)
    else
        rrule(calculate_loglikelihood,
              Val(filter), Val(algorithm), obs_indices,
              𝐒, data_in_deviations, constants_obj, state, 𝓂.workspaces;
              warmup_iterations = warmup_iterations,
              presample_periods = presample_periods,
              initial_covariance = initial_covariance,
              filter_algorithm = filter_algorithm,
              opts = opts,
              on_failure_loglikelihood = on_failure_loglikelihood)
    end

    if llh_rrule === nothing
        # When the inner rrule cannot run, we cannot supply a gradient anyway, so
        # short-circuit to the failure path instead of recomputing the primal.
        return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), zeros(S, length(parameter_values)))
    end

    llh, llh_pb = llh_rrule

    # ── pullback ──
    pullback = function (∂llh_bar)
        ∂llh = unthunk(∂llh_bar)

        # backprop through calculate_loglikelihood
        # returns: (_, _, _, _, ∂𝐒, ∂data_in_deviations, _, ∂state, _)
        llh_grads = llh_pb(∂llh)
        ∂𝐒              = llh_grads[5]
        ∂data_in_devs    = llh_grads[6]
        ∂state           = llh_grads[8]

        # When the filter forward pass fails (non-finite states, factorisation
        # failure, etc.) the filter rrule returns on_failure_loglikelihood with
        # an all-NoTangent pullback.  The loglikelihood is then a constant, so
        # the parameter gradient is exactly zero.
        if ∂𝐒 isa Union{NoTangent, AbstractZero}
            return NoTangent(), NoTangent(), NoTangent(), zeros(S, length(parameter_values))
        end

        # backprop through data_in_deviations = dt .- SS_and_pars[obs_indices]
        ∂SS_and_pars = zeros(S, length(SS_and_pars))
        if !(∂data_in_devs isa Union{NoTangent, AbstractZero})
            ∂SS_and_pars[obs_indices] .-= vec(sum(∂data_in_devs, dims = 2))
        end

        if ss_pb === nothing
            return NoTangent(), NoTangent(), NoTangent(), zeros(S, length(parameter_values))
        end

        # backprop through get_relevant_steady_state_and_state_update
        # cotangent: (Δconstants, ΔSS_and_pars, Δ𝐒, Δstate, Δsolved)
        ss_grads = ss_pb((NoTangent(), ∂SS_and_pars, ∂𝐒, ∂state, NoTangent()))
        ∂parameter_values = ss_grads[3]

        return NoTangent(), NoTangent(), NoTangent(), ∂parameter_values
    end

    if !isfinite(llh)
        return S(on_failure_loglikelihood), _ -> (NoTangent(), NoTangent(), NoTangent(), zeros(S, length(parameter_values)))
    end

    return llh, pullback
end



# ── get_irf rrule BPTT helpers ──

# Backpropagation-through-time (BPTT) pullback: returns
# (∂𝐒_list, ∂state_init, ∂SS_and_pars_from_init)
# These helpers replay the saved IRF simulation backward in time. Each
# algorithm-specific method mirrors the corresponding forward state update while
# accumulating cotangents for the perturbation solution objects, the effective
# IRF initial state, and the steady-state offset induced by a user-supplied
# initial condition.
function irf_bptt(::Val{:first_order},
        ∂Y_all::Array{S,3}, states_store, shocks_store,
        nShocks, periods, past_idx, nPast, nVars, nExo,
        𝐒, initial_state, nVar_len) where S
    sol_mat = 𝐒
    ∂sol_mat = zeros(S, size(sol_mat))
    ∂state_init = zeros(S, nVars)
    ∂SS_from_init = zeros(S, nVar_len)

    for si in 1:nShocks
        ∂y_accum = zeros(S, nVars)

        for t in periods:-1:1
            ∂y_t = ∂Y_all[:, t, si] + ∂y_accum

            prev_st = states_store[si, t]
            shock_t = shocks_store[si, t]
            input_t = [prev_st[past_idx]; shock_t]

            ∂sol_mat .+= ∂y_t * input_t'
            ∂input_t = sol_mat' * ∂y_t

            ∂y_accum = zeros(S, nVars)
            ∂y_accum[past_idx] .+= ∂input_t[1:nPast]
        end

        ∂state_init .+= ∂y_accum
        if initial_state != [0.0]
            ∂SS_from_init[1:nVar_len] .-= ∂y_accum[1:nVar_len]
        end
    end

    return [∂sol_mat], ∂state_init, ∂SS_from_init
end

function irf_bptt(::Val{:pruned_second_order},
        ∂Y_all::Array{S,3}, states_store, shocks_store,
        nShocks, periods, past_idx, nPast, nVars, nExo,
        𝐒, initial_state, nVar_len) where S
    𝐒₁, 𝐒₂ = 𝐒
    ∂𝐒₁ = zeros(S, size(𝐒₁))
    ∂𝐒₂ = zeros(S, size(𝐒₂))
    ∂state_init = [zeros(S, nVars), zeros(S, nVars)]
    ∂SS_from_init = zeros(S, nVar_len)
    n_aug = nPast + 1 + nExo
    # Preallocated kron buffer reused across all (si, t) iterations
    kaug₁ = Vector{S}(undef, n_aug^2)

    for si in 1:nShocks
        ∂y₁_accum = zeros(S, nVars)
        ∂δ_accum  = zeros(S, nVars)

        for t in periods:-1:1
            ∂out_t = ∂Y_all[:, t, si]
            ∂y₁_t = ∂out_t + ∂y₁_accum
            ∂δ_t  = ∂out_t + ∂δ_accum

            prev_st = states_store[si, t]
            shock_t = shocks_store[si, t]

            aug₁ = [prev_st[1][past_idx]; one(S); shock_t]
            aug₂ = [prev_st[2][past_idx]; zero(S); zero(shock_t)]
            ℒ.kron!(kaug₁, aug₁, aug₁)

            # y₁_new = 𝐒₁ * aug₁
            ∂𝐒₁ .+= ∂y₁_t * aug₁'
            ∂aug₁ = 𝐒₁' * ∂y₁_t

            # δ_new = 𝐒₁ * aug₂ + 𝐒₂ * kron(aug₁,aug₁) / 2
            ∂𝐒₁ .+= ∂δ_t * aug₂'
            ∂aug₂ = 𝐒₁' * ∂δ_t
            ∂𝐒₂ .+= ∂δ_t * kaug₁' / 2
            ∂kaug₁ = 𝐒₂' * ∂δ_t / 2
            ∂kaug₁_mat = reshape(∂kaug₁, n_aug, n_aug)
            ∂aug₁ .+= ∂kaug₁_mat' * aug₁ + ∂kaug₁_mat * aug₁

            ∂y₁_accum = zeros(S, nVars)
            ∂δ_accum  = zeros(S, nVars)
            ∂y₁_accum[past_idx] .+= ∂aug₁[1:nPast]
            ∂δ_accum[past_idx]  .+= ∂aug₂[1:nPast]
        end

        ∂state_init[1] .+= ∂y₁_accum
        ∂state_init[2] .+= ∂δ_accum
        if initial_state != [0.0]
            ∂SS_from_init[1:nVar_len] .-= ∂y₁_accum[1:nVar_len]
        end
    end

    return [∂𝐒₁, ∂𝐒₂], ∂state_init, ∂SS_from_init
end

function irf_bptt(::Val{:pruned_third_order},
        ∂Y_all::Array{S,3}, states_store, shocks_store,
        nShocks, periods, past_idx, nPast, nVars, nExo,
        𝐒, initial_state, nVar_len) where S
    𝐒₁, 𝐒₂, 𝐒₃ = 𝐒
    ∂𝐒₁ = zeros(S, size(𝐒₁))
    ∂𝐒₂ = zeros(S, size(𝐒₂))
    ∂𝐒₃ = zeros(S, size(𝐒₃))
    ∂state_init = [zeros(S, nVars), zeros(S, nVars), zeros(S, nVars)]
    ∂SS_from_init = zeros(S, nVar_len)
    n_aug = nPast + 1 + nExo
    # Preallocated kron buffers reused across all (si, t) iterations
    kaug₁ = Vector{S}(undef, n_aug^2)
    kaug₁₁ = Vector{S}(undef, n_aug^3)
    k_aug₁̂_aug₂ = Vector{S}(undef, n_aug^2)

    for si in 1:nShocks
        ∂y₁_accum = zeros(S, nVars)
        ∂δ_accum  = zeros(S, nVars)
        ∂ξ_accum  = zeros(S, nVars)

        for t in periods:-1:1
            ∂out_t = ∂Y_all[:, t, si]
            ∂y₁_t = ∂out_t + ∂y₁_accum
            ∂δ_t  = ∂out_t + ∂δ_accum
            ∂ξ_t  = ∂out_t + ∂ξ_accum

            prev_st = states_store[si, t]
            shock_t = shocks_store[si, t]

            aug₁  = [prev_st[1][past_idx]; one(S); shock_t]
            aug₁̂  = [prev_st[1][past_idx]; zero(S); shock_t]
            aug₂  = [prev_st[2][past_idx]; zero(S); zero(shock_t)]
            aug₃  = [prev_st[3][past_idx]; zero(S); zero(shock_t)]
            ℒ.kron!(kaug₁, aug₁, aug₁)
            ℒ.kron!(kaug₁₁, kaug₁, aug₁)

            # y₁_new = 𝐒₁ * aug₁
            ∂𝐒₁ .+= ∂y₁_t * aug₁'
            ∂aug₁ = 𝐒₁' * ∂y₁_t

            # δ_new = 𝐒₁ * aug₂ + 𝐒₂ * kron(aug₁,aug₁) / 2
            ∂𝐒₁ .+= ∂δ_t * aug₂'
            ∂aug₂ = 𝐒₁' * ∂δ_t
            ∂𝐒₂ .+= ∂δ_t * kaug₁' / 2
            ∂kaug₁_from_δ = 𝐒₂' * ∂δ_t / 2
            ∂kaug₁_mat = reshape(∂kaug₁_from_δ, n_aug, n_aug)
            ∂aug₁ .+= ∂kaug₁_mat' * aug₁ + ∂kaug₁_mat * aug₁

            # ξ_new = 𝐒₁ * aug₃ + 𝐒₂ * kron(aug₁̂, aug₂) + 𝐒₃ * kron(kaug₁, aug₁) / 6
            ∂𝐒₁ .+= ∂ξ_t * aug₃'
            ∂aug₃ = 𝐒₁' * ∂ξ_t

            ℒ.kron!(k_aug₁̂_aug₂, aug₁̂, aug₂)
            ∂𝐒₂ .+= ∂ξ_t * k_aug₁̂_aug₂'
            ∂k12 = 𝐒₂' * ∂ξ_t
            ∂k12_mat = reshape(∂k12, n_aug, n_aug)
            ∂aug₁̂ = ∂k12_mat * aug₂
            ∂aug₂ .+= ∂k12_mat' * aug₁̂

            ∂𝐒₃ .+= ∂ξ_t * kaug₁₁' / 6
            ∂kaug₁₁ = 𝐒₃' * ∂ξ_t / 6
            n_aug2 = n_aug * n_aug
            ∂kaug₁₁_mat = reshape(∂kaug₁₁, n_aug2, n_aug)
            ∂kaug₁_from_ξ = ∂kaug₁₁_mat * aug₁
            ∂aug₁ .+= ∂kaug₁₁_mat' * kaug₁
            ∂kaug₁_mat2 = reshape(∂kaug₁_from_ξ, n_aug, n_aug)
            ∂aug₁ .+= ∂kaug₁_mat2' * aug₁ + ∂kaug₁_mat2 * aug₁

            # aug₁̂ shares past_idx and shock with aug₁
            ∂aug₁[1:nPast] .+= ∂aug₁̂[1:nPast]
            ∂aug₁[nPast+2:end] .+= ∂aug₁̂[nPast+2:end]

            ∂y₁_accum = zeros(S, nVars)
            ∂δ_accum  = zeros(S, nVars)
            ∂ξ_accum  = zeros(S, nVars)
            ∂y₁_accum[past_idx] .+= ∂aug₁[1:nPast]
            ∂δ_accum[past_idx]  .+= ∂aug₂[1:nPast]
            ∂ξ_accum[past_idx]  .+= ∂aug₃[1:nPast]
        end

        ∂state_init[1] .+= ∂y₁_accum
        ∂state_init[2] .+= ∂δ_accum
        ∂state_init[3] .+= ∂ξ_accum
        if initial_state != [0.0]
            ∂SS_from_init[1:nVar_len] .-= ∂y₁_accum[1:nVar_len]
        end
    end

    return [∂𝐒₁, ∂𝐒₂, ∂𝐒₃], ∂state_init, ∂SS_from_init
end

function irf_bptt(::Val{:second_order},
        ∂Y_all::Array{S,3}, states_store, shocks_store,
        nShocks, periods, past_idx, nPast, nVars, nExo,
        𝐒, initial_state, nVar_len) where S
    𝐒₁, 𝐒₂ = 𝐒
    ∂𝐒₁ = zeros(S, size(𝐒₁))
    ∂𝐒₂ = zeros(S, size(𝐒₂))
    ∂state_init = zeros(S, nVars)
    ∂SS_from_init = zeros(S, nVar_len)
    n_aug = nPast + 1 + nExo
    kaug = Vector{S}(undef, n_aug^2)

    for si in 1:nShocks
        ∂y_accum = zeros(S, nVars)

        for t in periods:-1:1
            ∂y_t = ∂Y_all[:, t, si] + ∂y_accum

            prev_st = states_store[si, t]
            shock_t = shocks_store[si, t]
            aug = [prev_st[past_idx]; one(S); shock_t]
            ℒ.kron!(kaug, aug, aug)

            ∂𝐒₁ .+= ∂y_t * aug'
            ∂aug = 𝐒₁' * ∂y_t
            ∂𝐒₂ .+= ∂y_t * kaug' / 2
            ∂kaug = 𝐒₂' * ∂y_t / 2
            ∂kaug_mat = reshape(∂kaug, n_aug, n_aug)
            ∂aug .+= ∂kaug_mat' * aug + ∂kaug_mat * aug

            ∂y_accum = zeros(S, nVars)
            ∂y_accum[past_idx] .+= ∂aug[1:nPast]
        end

        ∂state_init .+= ∂y_accum
        if initial_state != [0.0]
            ∂SS_from_init[1:nVar_len] .-= ∂y_accum[1:nVar_len]
        end
    end

    return [∂𝐒₁, ∂𝐒₂], ∂state_init, ∂SS_from_init
end

function irf_bptt(::Val{:third_order},
        ∂Y_all::Array{S,3}, states_store, shocks_store,
        nShocks, periods, past_idx, nPast, nVars, nExo,
        𝐒, initial_state, nVar_len) where S
    𝐒₁, 𝐒₂, 𝐒₃ = 𝐒
    ∂𝐒₁ = zeros(S, size(𝐒₁))
    ∂𝐒₂ = zeros(S, size(𝐒₂))
    ∂𝐒₃ = zeros(S, size(𝐒₃))
    ∂state_init = zeros(S, nVars)
    ∂SS_from_init = zeros(S, nVar_len)
    n_aug = nPast + 1 + nExo
    kaug = Vector{S}(undef, n_aug^2)
    kaug3 = Vector{S}(undef, n_aug^3)

    for si in 1:nShocks
        ∂y_accum = zeros(S, nVars)

        for t in periods:-1:1
            ∂y_t = ∂Y_all[:, t, si] + ∂y_accum

            prev_st = states_store[si, t]
            shock_t = shocks_store[si, t]
            aug = [prev_st[past_idx]; one(S); shock_t]
            ℒ.kron!(kaug, aug, aug)
            ℒ.kron!(kaug3, kaug, aug)

            ∂𝐒₁ .+= ∂y_t * aug'
            ∂aug = 𝐒₁' * ∂y_t
            ∂𝐒₂ .+= ∂y_t * kaug' / 2
            ∂kaug = 𝐒₂' * ∂y_t / 2
            ∂𝐒₃ .+= ∂y_t * kaug3' / 6
            ∂kaug3 = 𝐒₃' * ∂y_t / 6

            n_aug2 = n_aug * n_aug
            ∂kaug3_mat = reshape(∂kaug3, n_aug2, n_aug)
            ∂kaug .+= ∂kaug3_mat * aug
            ∂aug .+= ∂kaug3_mat' * kaug

            ∂kaug_mat = reshape(∂kaug, n_aug, n_aug)
            ∂aug .+= ∂kaug_mat' * aug + ∂kaug_mat * aug

            ∂y_accum = zeros(S, nVars)
            ∂y_accum[past_idx] .+= ∂aug[1:nPast]
        end

        ∂state_init .+= ∂y_accum
        if initial_state != [0.0]
            ∂SS_from_init[1:nVar_len] .-= ∂y_accum[1:nVar_len]
        end
    end

    return [∂𝐒₁, ∂𝐒₂, ∂𝐒₃], ∂state_init, ∂SS_from_init
end


# ── Dispatched rrule chain helpers for get_irf ───────────────────────────────────

# Forward chain: set up the rrule sub-calls and return standardized output.
# Returns (𝐒, SS_and_pars, state, solved, chain_ctx) or nothing on failure.
function irf_rrule_forward_chain(::Val{:first_order}, parameters::Vector{S},
        𝓂, constants_obj, opts, tol) where S
    nsss_out, nsss_pb = rrule(get_NSSS_and_parameters, 𝓂, parameters;
                                opts = opts, estimation = true)
    reference_steady_state = nsss_out[1]
    solution_error = nsss_out[2][1]

    if (solution_error > tol.nsss.acceptance_tol) || isnan(solution_error)
        return nothing
    end

    ∇₁, jac_pb = rrule(calculate_jacobian, parameters, reference_steady_state,
                        𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces)

    first_out, first_pb = rrule(calculate_first_order_solution, ∇₁, constants_obj,
                                𝓂.workspaces, 𝓂.caches;
                                opts = opts, initial_guess = 𝓂.caches.qme_solution,
                                parameter_values = parameters)

    sol_mat = first_out[1]
    solved  = first_out[3]
    update_perturbation_counter!(𝓂.counters, solved, estimation = true, order = 1)

    return (sol_mat, reference_steady_state, nothing, solved, (nsss_pb, jac_pb, first_pb))
end

function irf_rrule_forward_chain(val_alg::Val, parameters::Vector{S},
        𝓂, constants_obj, opts, tol) where S
    ss_rrule = rrule(get_relevant_steady_state_and_state_update,
                     val_alg, parameters, 𝓂;
                     opts = opts, estimation = true)

    if ss_rrule === nothing
        return nothing
    end

    ss_out, ss_pb = ss_rrule
    SS_and_pars = ss_out[2]
    𝐒           = ss_out[3]
    state       = ss_out[4]
    solved      = ss_out[5]

    return (𝐒, SS_and_pars, state, solved, (ss_pb,))
end

# Backward chain: propagate gradients through sub-rrule pullbacks.
# Returns ∂parameters vector.
function irf_rrule_backward_chain(::Val{:first_order}, ∂SS_and_pars, ∂𝐒_list, ∂state_init, chain_ctx)
    nsss_pb, jac_pb, first_pb = chain_ctx
    ∂sol_mat = ∂𝐒_list[1]

    first_grads = first_pb((∂sol_mat, NoTangent(), NoTangent()))
    ∂∇₁ = first_grads[2]

    jac_grads = jac_pb(∂∇₁)
    ∂parameters_from_jac = jac_grads[2]
    ∂SS_from_jac = jac_grads[3]

    ∂SS_and_pars .+= ∂SS_from_jac

    nsss_grads = nsss_pb((∂SS_and_pars, NoTangent()))
    ∂parameters_from_nsss = nsss_grads[3]

    return ∂parameters_from_jac .+ ∂parameters_from_nsss
end

function irf_rrule_backward_chain(::Val, ∂SS_and_pars, ∂𝐒_list, ∂state_init, chain_ctx)
    ss_pb = chain_ctx[1]
    ss_grads = ss_pb((NoTangent(), ∂SS_and_pars, ∂𝐒_list, ∂state_init, NoTangent()))
    return ss_grads[3]
end


function rrule(::typeof(get_irf),
                𝓂::ℳ,
                parameters::Vector{S};
                steady_state_function::SteadyStateFunctionType = missing,
                periods::Int = DEFAULT_PERIODS,
                algorithm::Symbol = :first_order,
                variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_OBC,
                shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = DEFAULT_SHOCK_SELECTION,
                negative_shock::Bool = DEFAULT_NEGATIVE_SHOCK,
                initial_state::Vector{Float64} = DEFAULT_INITIAL_STATE,
                levels::Bool = false,
                verbose::Bool = DEFAULT_VERBOSE,
                tol::Tolerances = Tolerances(),
                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                caching::Bool = DEFAULT_CACHING,
                use_workspaces::Bool = DEFAULT_USE_WORKSPACES) where S <: Real

    val_alg = Val(algorithm)

    # Construct calculation options (sylvester/lyapunov fields ignored for first-order)
    nPnExo = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo
    syl_sum = sum(k * (k + 1) ÷ 2 for k in 1:nPnExo)
    opts = merge_calculation_options(tol = tol, verbose = verbose,
        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
        sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
        sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ?
            syl_sum > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM :
            sylvester_algorithm[2],
        lyapunov_algorithm = lyapunov_algorithm)

    # Pre-solve setup
    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    constants_obj = initialise_constants!(𝓂)

    solve!(𝓂,
           steady_state_function = steady_state_function,
           opts = opts,
           algorithm = algorithm)

    # Common shock/variable processing
    shocks = 𝓂.constants.post_model_macro.nExo == 0 ? :none : shocks
    shocks, negative_shock, _, periods, shock_idx, shock_history = process_shocks_input(shocks, negative_shock, 1.0, periods, 𝓂)
    var_idx = parse_variables_input_to_index(variables, 𝓂) |> sort

    nVars    = 𝓂.constants.post_model_macro.nVars
    past_idx = 𝓂.constants.post_model_macro.past_not_future_and_mixed_idx
    nPast    = length(past_idx)
    nExo     = 𝓂.constants.post_model_macro.nExo
    nShocks  = shocks == :none ? 1 : length(shock_idx)
    nVar_len = length(𝓂.constants.post_model_macro.var)

    zero_result() = zeros(S, length(var_idx), periods, nShocks)
    zero_pb(_) = (NoTangent(), NoTangent(), zeros(S, length(parameters)))

    # Dispatched rrule chain forward
    chain_result = irf_rrule_forward_chain(val_alg, parameters, 𝓂, constants_obj, opts, tol)

    if chain_result === nothing
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return zero_result(), zero_pb
    end

    𝐒, SS_and_pars, state, solved, chain_ctx = chain_result

    if !solved
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return zero_result(), zero_pb
    end

    reference_ss = SS_and_pars[1:nVars]

    # Forward simulation (already dispatched via irf_initial_state / irf_forward_simulate!)
    init_st = irf_initial_state(val_alg, state, SS_and_pars, initial_state, nVars, S)

    Y_all = zeros(S, nVars, periods, nShocks)
    states_store = Array{Any}(undef, nShocks, periods + 1)
    shocks_store = Array{Vector{S}}(undef, nShocks, periods)

    irf_forward_simulate!(val_alg, Y_all, states_store, shocks_store,
        init_st, shock_idx, shocks, negative_shock, shock_history,
        nExo, periods, past_idx, nVars, 𝐒)

    # Assemble output
    deviations = Y_all[var_idx, :, :]
    result = levels ? deviations .+ reference_ss[var_idx] : deviations

    if !use_workspaces; 𝓂.workspaces = orig_ws; end

    # Pullback (common structure, algorithm-specific parts dispatched)
    function get_irf_pullback(∂result_bar)
        ∂result = unthunk(∂result_bar)

        if ∂result isa Union{NoTangent, AbstractZero}
            return NoTangent(), NoTangent(), zeros(S, length(parameters))
        end

        ∂Y_all = zeros(S, nVars, periods, nShocks)
        ∂Y_all[var_idx, :, :] .= ∂result

        ∂SS_and_pars = zeros(S, length(SS_and_pars))
        if levels
            ∂SS_and_pars[var_idx] .+= dropdims(sum(∂result, dims = (2, 3)), dims = (2, 3))
        end

        # Dispatched BPTT
        ∂𝐒_list, ∂state_init, ∂SS_from_init = irf_bptt(val_alg,
            ∂Y_all, states_store, shocks_store,
            nShocks, periods, past_idx, nPast, nVars, nExo,
            𝐒, initial_state, nVar_len)

        ∂SS_and_pars[1:nVar_len] .+= ∂SS_from_init

        # Dispatched backward chain through sub-rrule pullbacks
        ∂parameters = irf_rrule_backward_chain(val_alg, ∂SS_and_pars, ∂𝐒_list, ∂state_init, chain_ctx)

        return NoTangent(), NoTangent(), ∂parameters
    end

    return result, get_irf_pullback
end

# ── calculate_covariance rrule ──────────────────────────────────────────────────
function rrule(::typeof(calculate_covariance),
                parameters::Vector{S},
                𝓂::ℳ;
                opts::CalculationOptions = merge_calculation_options()) where S <: Real

    # ── Non-differentiable setup ──
    constants_obj = initialise_constants!(𝓂)
    idx_constants = constants_obj.post_complete_parameters
    T = constants_obj.post_model_macro
    nPast = T.nPast_not_future_and_mixed
    past_idx = T.past_not_future_and_mixed_idx
    P = idx_constants.diag_nVars[past_idx, :]  # (nPast, nVars) constant selection matrix

    zero_result() = (zeros(S, 0, 0), zeros(S, 0, 0), zeros(S, 0, 0), zeros(S, 0), false)
    zero_pb(_) = (NoTangent(), zeros(S, length(parameters)), NoTangent())

    # ── Step 1: NSSS ──
    nsss_out, nsss_pb = rrule(get_NSSS_and_parameters, 𝓂, parameters; opts = opts)
    SS_and_pars = nsss_out[1]
    solution_error = nsss_out[2][1]

    if solution_error > opts.tol.nsss.acceptance_tol
        return (zeros(S, 0, 0), zeros(S, 0, 0), zeros(S, 0, 0), SS_and_pars, false), zero_pb
    end

    # ── Step 2: Jacobian ──
    ∇₁, jac_pb = rrule(calculate_jacobian, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces)

    # ── Step 3: First-order solution ──
    first_out, first_pb = rrule(calculate_first_order_solution,
                                ∇₁,
                                constants_obj,
                                𝓂.workspaces,
                                𝓂.caches;
                                initial_guess = 𝓂.caches.qme_solution,
                                opts = opts,
                                parameter_values = parameters)
    sol = first_out[1]
    solved_first = first_out[3]

    update_perturbation_counter!(𝓂.counters, solved_first, order = 1)

    # ── Step 4: A, C, CC (mutation-free) ──
    A = sol[:, 1:nPast] * P
    C = sol[:, nPast+1:end]
    CC = C * C'

    if !solved_first
        return (CC, sol, ∇₁, SS_and_pars, solved_first), zero_pb
    end

    # ── Step 5: Lyapunov ──
    lyap_ws = ensure_lyapunov_workspace!(𝓂.workspaces, T.nVars, :first_order)

    lyap_out, lyap_pb = rrule(solve_lyapunov_equation, A, CC, lyap_ws;
                                initial_guess = 𝓂.caches.covariance_first_order,
                                lyapunov_algorithm = opts.lyapunov_algorithm,
                                tol = opts.tol.first_order.ad.lyapunov,
                                verbose = opts.verbose,
                                has_unit_roots = 𝓂.caches.has_unit_roots)
    covar_raw = lyap_out[1]
    solved_lyap = lyap_out[2]

    # Cache the Lyapunov result for reuse
    if solved_lyap
        if size(𝓂.caches.covariance_first_order) != size(covar_raw)
            𝓂.caches.covariance_first_order = Matrix{Float64}(undef, size(covar_raw)...)
        end
        copyto!(𝓂.caches.covariance_first_order, covar_raw)
        𝓂.caches.valid_for.covariance_first_order = Float64.(parameters)
    end

    solved = solved_first && solved_lyap

    result = (covar_raw, sol, ∇₁, SS_and_pars, solved)

    # ── Pullback ──
    function calculate_covariance_pullback(Δout)
        Δcovar, Δsol_ret, Δ∇₁_ret, ΔSS_ret, _ = Δout

        # Materialise any InplaceableThunk / Thunk wrappers
        Δcovar   = unthunk(Δcovar)
        Δsol_ret = unthunk(Δsol_ret)
        Δ∇₁_ret  = unthunk(Δ∇₁_ret)
        ΔSS_ret  = unthunk(ΔSS_ret)

        # Accumulators
        ∂sol_total = zeros(S, size(sol))
        ∂∇₁_total = zeros(S, size(∇₁))
        ∂SS_total  = zeros(S, length(SS_and_pars))

        # Direct cotangents passed through the tuple
        if !(Δsol_ret isa AbstractZero)
            ∂sol_total .+= Δsol_ret
        end
        if !(Δ∇₁_ret isa AbstractZero)
            ∂∇₁_total .+= Δ∇₁_ret
        end
        if !(ΔSS_ret isa AbstractZero)
            ∂SS_total .+= ΔSS_ret
        end

        # Backprop through Lyapunov equation
        if !(Δcovar isa AbstractZero)
            lyap_grad = lyap_pb((Δcovar, NoTangent()))
            ΔA  = lyap_grad[2]   # ∂A
            ΔCC = lyap_grad[3]   # ∂CC

            # CC = C * C'  →  ∂C = (∂CC + ∂CC') * C
            ΔC = (ΔCC + ΔCC') * C

            # A = sol[:, 1:nPast] * P  →  ∂sol[:, 1:nPast] += ∂A * P'
            ∂sol_total[:, 1:nPast] .+= ΔA * P'

            # C = sol[:, nPast+1:end]
            ∂sol_total[:, nPast+1:end] .+= ΔC
        end

        # Backprop through first-order solution
        first_grad = first_pb((∂sol_total, NoTangent(), NoTangent()))
        ∂∇₁_total .+= first_grad[2]

        # Backprop through Jacobian
        jac_grad = jac_pb(∂∇₁_total)
        ∂parameters_from_jac = jac_grad[2]
        ∂SS_from_jac = jac_grad[3]
        ∂SS_total .+= ∂SS_from_jac

        # Backprop through NSSS
        nsss_grad = nsss_pb((∂SS_total, NoTangent()))
        ∂parameters_from_nsss = nsss_grad[3]

        ∂parameters_total = ∂parameters_from_jac .+ ∂parameters_from_nsss

        return NoTangent(), ∂parameters_total, NoTangent()
    end

    return result, calculate_covariance_pullback
end


# ── Helper: VJP of kron(A, B) ───────────────────────────────────────────────────
# Given C = kron(A, B) and cotangent ∂C, returns (∂A, ∂B).
function kron_vjp_helper(∂C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    m, n = size(A)
    p, q = size(B)
    S = eltype(∂C)
    ∂A = zeros(S, m, n)
    ∂B = zeros(S, p, q)
    @inbounds for j in 1:n
        for i in 1:m
            blk = @view ∂C[(i-1)*p+1:i*p, (j-1)*q+1:j*q]
            ∂A[i,j] = ℒ.dot(blk, B)
            if !iszero(A[i,j])
                ∂B .+= A[i,j] .* blk
            end
        end
    end
    return ∂A, ∂B
end


# ── calculate_mean rrule ────────────────────────────────────────────────────────
function rrule(::typeof(calculate_mean),
                parameters::Vector{S},
                𝓂::ℳ;
                algorithm = :pruned_second_order,
                opts::CalculationOptions = merge_calculation_options()) where S <: Real

    @assert algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] "Theoretical mean available only for first order, pruned second and pruned third order perturbation solutions."

    # ── Non-differentiable setup ──
    constants_obj = initialise_constants!(𝓂)
    T_pm = constants_obj.post_model_macro
    nVars = T_pm.nVars
    np = length(parameters)

    zero_pb(_) = (NoTangent(), zeros(S, np), NoTangent())

    # ── Step 1: NSSS ──
    nsss_out, nsss_pb = rrule(get_NSSS_and_parameters, 𝓂, parameters; opts = opts)
    SS_and_pars = nsss_out[1]
    solution_error = nsss_out[2][1]

    # ── First-order path (mean = steady state) ──
    if algorithm == :first_order
        solved = solution_error < opts.tol.nsss.acceptance_tol
        mean_of_variables = SS_and_pars[1:nVars]

        function first_order_mean_pullback(∂out)
            ∂mean = unthunk(∂out[1])
            if ∂mean isa AbstractZero
                return NoTangent(), zeros(S, np), NoTangent()
            end
            ∂SS = zeros(S, length(SS_and_pars))
            ∂SS[1:nVars] .= ∂mean
            nsss_grad = nsss_pb((∂SS, NoTangent()))
            ∂params = nsss_grad[3] isa AbstractZero ? zeros(S, np) : nsss_grad[3]
            return NoTangent(), ∂params, NoTangent()
        end

        return (mean_of_variables, solved), first_order_mean_pullback
    end

    # ── Higher-order path: early exit on NSSS failure ──
    if solution_error > opts.tol.nsss.acceptance_tol
        return (SS_and_pars[1:nVars], false), zero_pb
    end

    ensure_moments_constants!(constants_obj)
    so = constants_obj.second_order

    nᵉ = T_pm.nExo
    nˢ = T_pm.nPast_not_future_and_mixed
    iˢ = T_pm.past_not_future_and_mixed_idx
    𝐔₂ = 𝓂.constants.second_order.𝐔₂
    vec_Iₑ = so.vec_Iₑ

    # ── Step 2: Jacobian ──
    ∇₁, jac_pb = rrule(calculate_jacobian, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces)

    # ── Step 3: First-order solution ──
    first_out, first_pb = rrule(calculate_first_order_solution,
                                ∇₁,
                                constants_obj,
                                𝓂.workspaces,
                                𝓂.caches;
                                initial_guess = 𝓂.caches.qme_solution,
                                opts = opts,
                                parameter_values = parameters)
    𝐒₁ = first_out[1]
    solved_first = first_out[3]

    update_perturbation_counter!(𝓂.counters, solved_first, order = 1)

    if !solved_first
        return (SS_and_pars[1:nVars], false), zero_pb
    end

    # ── Step 4: Hessian ──
    ∇₂, hess_pb = rrule(calculate_hessian, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian, 𝓂.workspaces)

    # ── Step 5: Second-order solution ──
    so2_out, so2_pb = rrule(calculate_second_order_solution, ∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches; opts = opts, parameter_values = parameters)
    𝐒₂_raw = so2_out[1]
    solved2 = so2_out[2]

    update_perturbation_counter!(𝓂.counters, solved2, order = 2)

    if !solved2
        return (SS_and_pars[1:nVars], false), zero_pb
    end

    # ── Step 6: Decompress S₂ ──
    𝐒₂_full = 𝐒₂_raw * 𝐔₂

    # ── Step 7: Slicing and mean computation ──
    kron_s_s = so.kron_states
    kron_e_e = so.kron_e_e
    kron_v_v = so.kron_v_v

    # First-order slices
    s_to_y₁ = 𝐒₁[:, 1:nˢ]
    s_to_s₁ = 𝐒₁[iˢ, 1:nˢ]
    e_to_s₁ = 𝐒₁[iˢ, (nˢ+1):end]

    # Second-order slices (dense)
    s_s_to_y₂ = Matrix(𝐒₂_full[:, kron_s_s])
    e_e_to_y₂ = Matrix(𝐒₂_full[:, kron_e_e])
    v_v_to_y₂_v = vec(𝐒₂_full[:, kron_v_v])
    s_s_to_s₂ = Matrix(𝐒₂_full[iˢ, kron_s_s])
    e_e_to_s₂ = Matrix(𝐒₂_full[iˢ, kron_e_e])
    v_v_to_s₂_v = vec(𝐒₂_full[iˢ, kron_v_v])

    # Kronecker products
    s₁_kron_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
    e₁_kron_e₁ = ℒ.kron(e_to_s₁, e_to_s₁) |> collect

    # Block transition matrix
    ŝ_to_ŝ₂ = [ s_to_s₁              zeros(S, nˢ, nˢ + nˢ^2)
                 zeros(S, nˢ, nˢ)     s_to_s₁              s_s_to_s₂ / 2
                 zeros(S, nˢ^2, 2*nˢ)                       s₁_kron_s₁        ]

    ŝ_to_y₂ = [s_to_y₁  s_to_y₁  s_s_to_y₂ / 2]

    ŝv₂ = vcat(zeros(S, nˢ),
               v_v_to_s₂_v / 2 + e_e_to_s₂ * vec_Iₑ / 2,
               e₁_kron_e₁ * vec_Iₑ)

    yv₂ = (v_v_to_y₂_v + e_e_to_y₂ * vec_Iₑ) / 2

    # Mean solve
    A_mean = collect(ℒ.I(size(ŝ_to_ŝ₂, 1))) - ŝ_to_ŝ₂
    μˢ⁺₂ = A_mean \ ŝv₂

    mean_of_variables = SS_and_pars[1:nVars] + ŝ_to_y₂ * μˢ⁺₂ + yv₂

    slvd = solved_first && solved2

    result = (mean_of_variables, slvd)

    # ── Pullback ──
    function calculate_mean_pullback(∂out)
        ∂mean_in = unthunk(∂out[1])

        if ∂mean_in isa AbstractZero
            return NoTangent(), zeros(S, np), NoTangent()
        end

        # Accumulators
        ∂𝐒₁_acc = zeros(S, size(𝐒₁))
        ∂S2f     = zeros(S, size(𝐒₂_full))
        ∂SS_acc  = zeros(S, length(SS_and_pars))

        ∂μʸ = ∂mean_in

        # ── Backprop through mean_of_variables ──
        # mean_of_variables = SS[1:n] + ŝ_to_y₂ * μˢ⁺₂ + yv₂
        ∂SS_acc[1:nVars] .+= ∂μʸ
        ∂ŝ_to_y₂ = ∂μʸ * μˢ⁺₂'
        ∂μˢ⁺₂ = ŝ_to_y₂' * ∂μʸ
        ∂yv₂ = copy(∂μʸ)

        # ── Backprop through (I - ŝ_to_ŝ₂) \ ŝv₂ ──
        λ = A_mean' \ ∂μˢ⁺₂
        ∂ŝv₂ = copy(λ)
        ∂ŝ_to_ŝ₂ = λ * μˢ⁺₂'   # from -(I - A): sign is +

        # ── yv₂ = (v_v_to_y₂_v + e_e_to_y₂ * vec_Iₑ) / 2 ──
        ∂S2f[:, kron_v_v] .+= reshape(∂yv₂ / 2, :, 1)
        ∂S2f[:, kron_e_e] .+= (∂yv₂ / 2) * vec_Iₑ'

        # ── ŝv₂ = [0; v_v/2 + e_e·v/2; e₁⊗e₁·v] ──
        ∂ŝv₂_mid = ∂ŝv₂[nˢ+1:2nˢ]
        ∂ŝv₂_bot = ∂ŝv₂[2nˢ+1:end]

        ∂S2f[iˢ, kron_v_v] .+= reshape(∂ŝv₂_mid / 2, :, 1)
        ∂S2f[iˢ, kron_e_e] .+= (∂ŝv₂_mid / 2) * vec_Iₑ'
        ∂e₁ke₁ = ∂ŝv₂_bot * vec_Iₑ'

        # ── ŝ_to_y₂ = [s_to_y₁  s_to_y₁  s_s_to_y₂/2] ──
        ∂𝐒₁_acc[:, 1:nˢ] .+= ∂ŝ_to_y₂[:, 1:nˢ] .+ ∂ŝ_to_y₂[:, nˢ+1:2nˢ]
        ∂S2f[:, kron_s_s]  .+= ∂ŝ_to_y₂[:, 2nˢ+1:end] / 2

        # ── ŝ_to_ŝ₂ block adjoints ──
        ∂s₁_from_ŝŝ  = ∂ŝ_to_ŝ₂[1:nˢ, 1:nˢ] + ∂ŝ_to_ŝ₂[nˢ+1:2nˢ, nˢ+1:2nˢ]
        ∂ss2_from_ŝŝ = ∂ŝ_to_ŝ₂[nˢ+1:2nˢ, 2nˢ+1:end] / 2
        ∂s₁ks₁       = ∂ŝ_to_ŝ₂[2nˢ+1:end, 2nˢ+1:end]

        # ── Kron VJPs ──
        ∂s₁_L, ∂s₁_R = kron_vjp_helper(∂s₁ks₁, s_to_s₁, s_to_s₁)
        ∂e₁_L, ∂e₁_R = kron_vjp_helper(∂e₁ke₁, e_to_s₁, e_to_s₁)

        # Aggregate into 𝐒₁
        ∂𝐒₁_acc[iˢ, 1:nˢ]      .+= ∂s₁_from_ŝŝ .+ ∂s₁_L .+ ∂s₁_R
        ∂𝐒₁_acc[iˢ, nˢ+1:end]  .+= ∂e₁_L .+ ∂e₁_R

        # Aggregate into S₂_full
        ∂S2f[iˢ, kron_s_s] .+= ∂ss2_from_ŝŝ

        # ── S₂_full → S₂_raw via 𝐔₂ ──
        ∂S2_raw = ∂S2f * 𝐔₂'

        # ── Chain through sub-rrule pullbacks (reverse order) ──
        # Second-order solution
        so2_grad = so2_pb((∂S2_raw, NoTangent()))
        ∂∇₁_acc  = so2_grad[2] isa AbstractZero ? zeros(S, size(∇₁)) : collect(S, so2_grad[2])
        ∂∇₂_total = so2_grad[3] isa AbstractZero ? zeros(S, size(∇₂)) : so2_grad[3]
        ∂𝐒₁_from_so2 = so2_grad[4] isa AbstractZero ? zeros(S, size(𝐒₁)) : collect(S, so2_grad[4])
        ∂𝐒₁_acc .+= ∂𝐒₁_from_so2

        # Hessian
        hess_grad = hess_pb(∂∇₂_total)
        ∂params_hess = hess_grad[2] isa AbstractZero ? zeros(S, np) : hess_grad[2]
        ∂SS_from_hess = hess_grad[3] isa AbstractZero ? zeros(S, length(SS_and_pars)) : hess_grad[3]
        ∂SS_acc .+= ∂SS_from_hess

        # First-order solution
        first_grad = first_pb((∂𝐒₁_acc, NoTangent(), NoTangent()))
        ∂∇₁_from_first = first_grad[2] isa AbstractZero ? zeros(S, size(∇₁)) : first_grad[2]
        ∂∇₁_acc .+= ∂∇₁_from_first

        # Jacobian
        jac_grad = jac_pb(∂∇₁_acc)
        ∂params_jac = jac_grad[2] isa AbstractZero ? zeros(S, np) : jac_grad[2]
        ∂SS_from_jac = jac_grad[3] isa AbstractZero ? zeros(S, length(SS_and_pars)) : jac_grad[3]
        ∂SS_acc .+= ∂SS_from_jac

        # NSSS
        nsss_grad = nsss_pb((∂SS_acc, NoTangent()))
        ∂params_nsss = nsss_grad[3] isa AbstractZero ? zeros(S, np) : nsss_grad[3]

        ∂parameters_total = ∂params_hess .+ ∂params_jac .+ ∂params_nsss

        return NoTangent(), ∂parameters_total, NoTangent()
    end

    return result, calculate_mean_pullback
end


# ── calculate_second_order_moments rrule ────────────────────────────────────────
function rrule(::typeof(calculate_second_order_moments),
                parameters::Vector{S},
                𝓂::ℳ;
                opts::CalculationOptions = merge_calculation_options()) where S <: Real

    # ── Non-differentiable setup ──
    constants_obj = initialise_constants!(𝓂)
    ensure_moments_constants!(constants_obj)
    so = constants_obj.second_order
    T_pm = constants_obj.post_model_macro
    nᵉ = T_pm.nExo
    nˢ = T_pm.nPast_not_future_and_mixed
    nVars = T_pm.nVars
    iˢ = T_pm.past_not_future_and_mixed_idx
    𝐔₂ = 𝓂.constants.second_order.𝐔₂
    vec_Iₑ = so.vec_Iₑ

    zero_10() = (zeros(S,0), zeros(S,0), zeros(S,0,0), zeros(S,0,0),
                 zeros(S,0), zeros(S,0,0), zeros(S,0,0), spzeros(S,0,0), spzeros(S,0,0), false)
    zero_pb(_) = (NoTangent(), zeros(S, length(parameters)), NoTangent())

    # ── Step 1: Covariance ──
    cov_out, cov_pb = rrule(calculate_covariance, parameters, 𝓂; opts = opts)
    Σʸ₁, 𝐒₁, ∇₁, SS_and_pars, solved = cov_out

    if !solved
        return zero_10(), zero_pb
    end

    Σᶻ₁ = Σʸ₁[iˢ, iˢ]

    # ── Step 2: Hessian ──
    ∇₂, hess_pb = rrule(calculate_hessian, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian, 𝓂.workspaces)

    # ── Step 3: Second-order solution ──
    so2_out, so2_pb = rrule(calculate_second_order_solution, ∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches; opts = opts, parameter_values = parameters)
    𝐒₂_raw = so2_out[1]
    solved2 = so2_out[2]

    update_perturbation_counter!(𝓂.counters, solved2, order = 2)

    if !solved2
        return (zeros(S,0), zeros(S,0), Σʸ₁, zeros(S,0,0), SS_and_pars, 𝐒₁, ∇₁, spzeros(S,0,0), ∇₂, solved2), zero_pb
    end

    # ── Step 4: Decompress S₂ (mutation-free) ──
    𝐒₂_full = 𝐒₂_raw * 𝐔₂

    # ── Step 5: Slicing and mean computation ──
    kron_s_s = so.kron_states
    kron_e_e = so.kron_e_e
    kron_v_v = so.kron_v_v

    # First-order slices
    s_to_y₁ = 𝐒₁[:, 1:nˢ]
    s_to_s₁ = 𝐒₁[iˢ, 1:nˢ]
    e_to_s₁ = 𝐒₁[iˢ, (nˢ+1):end]

    # Second-order slices (dense)
    s_s_to_y₂ = Matrix(𝐒₂_full[:, kron_s_s])
    e_e_to_y₂ = Matrix(𝐒₂_full[:, kron_e_e])
    v_v_to_y₂_v = vec(𝐒₂_full[:, kron_v_v])
    s_s_to_s₂ = Matrix(𝐒₂_full[iˢ, kron_s_s])
    e_e_to_s₂ = Matrix(𝐒₂_full[iˢ, kron_e_e])
    v_v_to_s₂_v = vec(𝐒₂_full[iˢ, kron_v_v])

    # Kronecker products
    s₁_kron_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
    e₁_kron_e₁ = ℒ.kron(e_to_s₁, e_to_s₁) |> collect

    # Block matrices
    ŝ_to_ŝ₂ = [ s_to_s₁             zeros(S, nˢ, nˢ + nˢ^2)
                 zeros(S, nˢ, nˢ)    s_to_s₁             s_s_to_s₂ / 2
                 zeros(S, nˢ^2, 2*nˢ) s₁_kron_s₁                       ]

    ŝ_to_y₂ = [s_to_y₁  s_to_y₁  s_s_to_y₂ / 2]

    ŝv₂ = vcat(zeros(S, nˢ),
               v_v_to_s₂_v / 2 + e_e_to_s₂ * vec_Iₑ / 2,
               e₁_kron_e₁ * vec_Iₑ)

    yv₂ = (v_v_to_y₂_v + e_e_to_y₂ * vec_Iₑ) / 2

    # Mean solve
    A_mean = collect(ℒ.I(size(ŝ_to_ŝ₂, 1))) - ŝ_to_ŝ₂
    μˢ⁺₂ = A_mean \ ŝv₂

    A_Δ = collect(ℒ.I(nˢ)) - s_to_s₁
    rhs_Δ = s_s_to_s₂ * vec(Σᶻ₁) / 2 + (v_v_to_s₂_v + e_e_to_s₂ * vec_Iₑ) / 2
    Δμˢ₂ = vec(A_Δ \ rhs_Δ)

    μʸ₂ = SS_and_pars[1:nVars] + ŝ_to_y₂ * μˢ⁺₂ + yv₂

    slvd = solved && solved2
    𝐒₂_sp = sparse(𝐒₂_full)  # was: dense_to_sparse

    result = (μʸ₂, Δμˢ₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂_sp, ∇₂, slvd)

    # ── Pullback ──
    function calculate_second_order_moments_pullback(∂out)
        ∂μʸ₂_in, ∂Δμˢ₂_in, ∂Σʸ₁_pass, ∂Σᶻ₁_pass, ∂SS_pass,
            ∂𝐒₁_pass, ∂∇₁_pass, ∂𝐒₂_pass, ∂∇₂_pass, _ = ∂out

        # Materialise any InplaceableThunk / Thunk wrappers
        ∂μʸ₂_in   = unthunk(∂μʸ₂_in)
        ∂Δμˢ₂_in  = unthunk(∂Δμˢ₂_in)
        ∂Σʸ₁_pass = unthunk(∂Σʸ₁_pass)
        ∂Σᶻ₁_pass = unthunk(∂Σᶻ₁_pass)
        ∂SS_pass   = unthunk(∂SS_pass)
        ∂𝐒₁_pass   = unthunk(∂𝐒₁_pass)
        ∂∇₁_pass   = unthunk(∂∇₁_pass)
        ∂𝐒₂_pass   = unthunk(∂𝐒₂_pass)
        ∂∇₂_pass   = unthunk(∂∇₂_pass)

        # Accumulators
        ∂𝐒₁_acc = zeros(S, size(𝐒₁))
        ∂S2f     = zeros(S, size(𝐒₂_full))
        ∂SS_acc  = zeros(S, length(SS_and_pars))
        ∂∇₁_acc  = zeros(S, size(∇₁))
        ∂Σᶻ₁_acc = zeros(S, nˢ, nˢ)

        # Pass-through cotangents
        if !(∂𝐒₁_pass isa AbstractZero);  ∂𝐒₁_acc .+= ∂𝐒₁_pass;  end
        if !(∂SS_pass  isa AbstractZero);  ∂SS_acc  .+= ∂SS_pass;   end
        if !(∂𝐒₂_pass  isa AbstractZero);  ∂S2f     .+= ∂𝐒₂_pass;   end
        if !(∂∇₁_pass  isa AbstractZero);  ∂∇₁_acc  .+= ∂∇₁_pass;   end
        if !(∂Σᶻ₁_pass isa AbstractZero);  ∂Σᶻ₁_acc .+= ∂Σᶻ₁_pass;  end

        # ──── Backprop through μʸ₂ ────
        if !(∂μʸ₂_in isa AbstractZero)
            ∂μʸ₂ = ∂μʸ₂_in
            # μʸ₂ = SS[1:n] + ŝ_to_y₂ * μˢ⁺₂ + yv₂
            ∂SS_acc[1:nVars] .+= ∂μʸ₂
            ∂ŝ_to_y₂ = ∂μʸ₂ * μˢ⁺₂'
            ∂μˢ⁺₂ = ŝ_to_y₂' * ∂μʸ₂
            ∂yv₂ = copy(∂μʸ₂)

            # μˢ⁺₂ = A_mean \ ŝv₂  →  λ = A_mean' \ ∂μˢ⁺₂
            λ = A_mean' \ ∂μˢ⁺₂
            ∂ŝv₂ = copy(λ)
            ∂ŝ_to_ŝ₂ = λ * μˢ⁺₂'  # from (I - ŝ_to_ŝ₂)

            # ── yv₂ = (v_v_to_y₂_v + e_e_to_y₂ * vec_Iₑ) / 2 ──
            ∂S2f[:, kron_v_v] .+= reshape(∂yv₂ / 2, :, 1)
            ∂S2f[:, kron_e_e] .+= (∂yv₂ / 2) * vec_Iₑ'

            # ── ŝv₂ = [0; v_v/2 + e_e·v/2; e₁⊗e₁·v] ──
            ∂ŝv₂_mid = ∂ŝv₂[nˢ+1:2nˢ]
            ∂ŝv₂_bot = ∂ŝv₂[2nˢ+1:end]

            ∂S2f[iˢ, kron_v_v] .+= reshape(∂ŝv₂_mid / 2, :, 1)
            ∂S2f[iˢ, kron_e_e] .+= (∂ŝv₂_mid / 2) * vec_Iₑ'
            ∂e₁ke₁ = ∂ŝv₂_bot * vec_Iₑ'

            # ── ŝ_to_y₂ = [s_to_y₁  s_to_y₁  s_s_to_y₂/2] ──
            ∂𝐒₁_acc[:, 1:nˢ] .+= ∂ŝ_to_y₂[:, 1:nˢ] .+ ∂ŝ_to_y₂[:, nˢ+1:2nˢ]
            ∂S2f[:, kron_s_s]  .+= ∂ŝ_to_y₂[:, 2nˢ+1:end] / 2

            # ── ŝ_to_ŝ₂ blocks ──
            ∂s₁_from_ŝŝ  = ∂ŝ_to_ŝ₂[1:nˢ, 1:nˢ] + ∂ŝ_to_ŝ₂[nˢ+1:2nˢ, nˢ+1:2nˢ]
            ∂ss2_from_ŝŝ = ∂ŝ_to_ŝ₂[nˢ+1:2nˢ, 2nˢ+1:end] / 2
            ∂s₁ks₁       = ∂ŝ_to_ŝ₂[2nˢ+1:end, 2nˢ+1:end]

            # ── Kron VJPs ──
            ∂s₁_L, ∂s₁_R = kron_vjp_helper(∂s₁ks₁, s_to_s₁, s_to_s₁)
            ∂e₁_L, ∂e₁_R = kron_vjp_helper(∂e₁ke₁, e_to_s₁, e_to_s₁)

            # Aggregate into 𝐒₁
            ∂𝐒₁_acc[iˢ, 1:nˢ]      .+= ∂s₁_from_ŝŝ .+ ∂s₁_L .+ ∂s₁_R
            ∂𝐒₁_acc[iˢ, nˢ+1:end]  .+= ∂e₁_L .+ ∂e₁_R

            # Aggregate into S₂_full
            ∂S2f[iˢ, kron_s_s] .+= ∂ss2_from_ŝŝ
        end

        # ──── Backprop through Δμˢ₂ ────
        if !(∂Δμˢ₂_in isa AbstractZero)
            ∂Δμˢ₂ = ∂Δμˢ₂_in
            # Δμˢ₂ = A_Δ \ rhs_Δ
            λ_Δ = A_Δ' \ ∂Δμˢ₂
            # ∂(I - s_to_s₁) → ∂s_to_s₁
            ∂𝐒₁_acc[iˢ, 1:nˢ] .+= λ_Δ * Δμˢ₂'
            # rhs_Δ = s_s_to_s₂ * vec(Σᶻ₁)/2 + (v_v_to_s₂_v + e_e_to_s₂*vec_Iₑ)/2
            ∂S2f[iˢ, kron_s_s]  .+= λ_Δ * vec(Σᶻ₁)' / 2
            ∂Σᶻ₁_acc .+= reshape(s_s_to_s₂' * λ_Δ / 2, nˢ, nˢ)
            ∂S2f[iˢ, kron_v_v]  .+= reshape(λ_Δ / 2, :, 1)
            ∂S2f[iˢ, kron_e_e]  .+= (λ_Δ / 2) * vec_Iₑ'
        end

        # ── Σᶻ₁ → Σʸ₁ ──
        ∂Σʸ₁ = zeros(S, size(Σʸ₁))
        ∂Σʸ₁[iˢ, iˢ] .= ∂Σᶻ₁_acc
        if !(∂Σʸ₁_pass isa AbstractZero)
            ∂Σʸ₁ .+= ∂Σʸ₁_pass
        end

        # ── S₂_full → S₂_raw via 𝐔₂ ──
        ∂S2_raw = ∂S2f * 𝐔₂'

        # ── Chain through sub-rrule pullbacks ──
        # Second-order solution
        so2_grad = so2_pb((∂S2_raw, NoTangent()))
        # Coerce AbstractZero returns to typed zeros
        ∂∇₁_from_so2 = so2_grad[2] isa AbstractZero ? zeros(S, size(∇₁)) : so2_grad[2]
        ∂∇₂_total    = so2_grad[3] isa AbstractZero ? zeros(S, size(∇₂)) : so2_grad[3]
        ∂𝐒₁_from_so2 = so2_grad[4] isa AbstractZero ? zeros(S, size(𝐒₁)) : so2_grad[4]
        ∂∇₁_acc .+= ∂∇₁_from_so2
        ∂𝐒₁_acc .+= ∂𝐒₁_from_so2

        if !(∂∇₂_pass isa AbstractZero)
            ∂∇₂_total = ∂∇₂_total .+ ∂∇₂_pass
        end

        # Hessian
        hess_grad = hess_pb(∂∇₂_total)
        ∂params_hess = hess_grad[2] isa AbstractZero ? zeros(S, length(parameters)) : hess_grad[2]
        ∂SS_from_hess = hess_grad[3] isa AbstractZero ? zeros(S, length(SS_and_pars)) : hess_grad[3]
        ∂SS_acc .+= ∂SS_from_hess

        # Covariance (chains through NSSS → Jacobian → 1st sol → Lyapunov)
        cov_grad = cov_pb((∂Σʸ₁, ∂𝐒₁_acc, ∂∇₁_acc, ∂SS_acc, NoTangent()))
        ∂params_cov = cov_grad[2] isa AbstractZero ? zeros(S, length(parameters)) : cov_grad[2]

        ∂parameters_total = ∂params_hess .+ ∂params_cov

        return NoTangent(), ∂parameters_total, NoTangent()
    end

    return result, calculate_second_order_moments_pullback
end


# ── calculate_second_order_moments_with_covariance rrule ────────────────────────
function rrule(::typeof(calculate_second_order_moments_with_covariance),
                parameters::Vector{S},
                𝓂::ℳ;
                opts::CalculationOptions = merge_calculation_options()) where S <: Real

    # ── Non-differentiable setup ──
    constants_obj = initialise_constants!(𝓂)
    ensure_moments_constants!(constants_obj)
    so = constants_obj.second_order
    T_pm = constants_obj.post_model_macro
    nᵉ = T_pm.nExo
    nˢ = T_pm.nPast_not_future_and_mixed
    nVars = T_pm.nVars
    iˢ = T_pm.past_not_future_and_mixed_idx
    𝐔₂ = 𝓂.constants.second_order.𝐔₂
    vec_Iₑ = so.vec_Iₑ
    I_plus_s_s = so.I_plus_s_s
    e4_minus = so.e4_minus_vecIₑ_outer
    Iₑ = collect(S, ℒ.I(nᵉ))

    np = length(parameters)
    zero_15() = (zeros(S,0,0), zeros(S,0,0), zeros(S,0), zeros(S,0),
                 zeros(S,0,0), zeros(S,0,0), zeros(S,0,0),
                 zeros(S,0,0), zeros(S,0,0), zeros(S,0),
                 zeros(S,0,0), zeros(S,0,0), spzeros(S,0,0), spzeros(S,0,0), false)
    zero_pb(_) = (NoTangent(), zeros(S, np), NoTangent())

    # ── Step 1: Covariance ──
    cov_out, cov_pb = rrule(calculate_covariance, parameters, 𝓂; opts = opts)
    Σʸ₁, 𝐒₁, ∇₁, SS_and_pars, solved = cov_out

    if !solved; return zero_15(), zero_pb; end

    Σᶻ₁ = Σʸ₁[iˢ, iˢ]

    # ── Step 2: Hessian ──
    ∇₂, hess_pb = rrule(calculate_hessian, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian, 𝓂.workspaces)

    # ── Step 3: Second-order solution ──
    so2_out, so2_pb = rrule(calculate_second_order_solution, ∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches; opts = opts, parameter_values = parameters)
    𝐒₂_raw, solved2 = so2_out

    update_perturbation_counter!(𝓂.counters, solved2, order = 2)

    if !solved2; return zero_15(), zero_pb; end

    # ── Step 4: Decompress S₂ ──
    𝐒₂_full = 𝐒₂_raw * 𝐔₂

    # ── Step 5: Slicing ──
    kron_s_s = so.kron_states
    kron_e_e = so.kron_e_e
    kron_v_v = so.kron_v_v
    kron_s_e = so.kron_s_e

    s_to_y₁ = 𝐒₁[:, 1:nˢ]
    e_to_y₁ = 𝐒₁[:, (nˢ+1):end]
    s_to_s₁ = 𝐒₁[iˢ, 1:nˢ]
    e_to_s₁ = 𝐒₁[iˢ, (nˢ+1):end]

    s_s_to_y₂ = Matrix(𝐒₂_full[:, kron_s_s])
    e_e_to_y₂ = Matrix(𝐒₂_full[:, kron_e_e])
    v_v_to_y₂_v = vec(𝐒₂_full[:, kron_v_v])
    s_e_to_y₂ = Matrix(𝐒₂_full[:, kron_s_e])

    s_s_to_s₂ = Matrix(𝐒₂_full[iˢ, kron_s_s])
    e_e_to_s₂ = Matrix(𝐒₂_full[iˢ, kron_e_e])
    v_v_to_s₂_v = vec(𝐒₂_full[iˢ, kron_v_v])
    s_e_to_s₂ = Matrix(𝐒₂_full[iˢ, kron_s_e])

    # Kronecker products
    s₁_kron_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
    e₁_kron_e₁ = ℒ.kron(e_to_s₁, e_to_s₁) |> collect
    s₁_kron_e₁ = ℒ.kron(s_to_s₁, e_to_s₁) |> collect

    # ── Block matrices ──
    ŝ_to_ŝ₂ = [ s_to_s₁             zeros(S, nˢ, nˢ + nˢ^2)
                 zeros(S, nˢ, nˢ)    s_to_s₁             s_s_to_s₂ / 2
                 zeros(S, nˢ^2, 2*nˢ) s₁_kron_s₁                       ]

    ê_to_ŝ₂ = [ e_to_s₁         zeros(S, nˢ, nᵉ^2 + nᵉ * nˢ)
                 zeros(S, nˢ, nᵉ)    e_e_to_s₂ / 2       s_e_to_s₂
                 zeros(S, nˢ^2, nᵉ)  e₁_kron_e₁  I_plus_s_s * s₁_kron_e₁ ]

    ŝ_to_y₂ = [s_to_y₁  s_to_y₁  s_s_to_y₂ / 2]

    ê_to_y₂ = [e_to_y₁  e_e_to_y₂ / 2   s_e_to_y₂]

    ŝv₂ = vcat(zeros(S, nˢ),
               v_v_to_s₂_v / 2 + e_e_to_s₂ * vec_Iₑ / 2,
               e₁_kron_e₁ * vec_Iₑ)

    yv₂ = (v_v_to_y₂_v + e_e_to_y₂ * vec_Iₑ) / 2

    # Mean solve
    A_mean = collect(ℒ.I(size(ŝ_to_ŝ₂, 1))) - ŝ_to_ŝ₂
    μˢ⁺₂ = A_mean \ ŝv₂

    A_Δ = collect(ℒ.I(nˢ)) - s_to_s₁
    rhs_Δ = s_s_to_s₂ * vec(Σᶻ₁) / 2 + (v_v_to_s₂_v + e_e_to_s₂ * vec_Iₑ) / 2
    Δμˢ₂ = vec(A_Δ \ rhs_Δ)

    μʸ₂ = SS_and_pars[1:nVars] + ŝ_to_y₂ * μˢ⁺₂ + yv₂

    # ── Step 6: Pruned covariance ──
    kron_Σᶻ₁_Iₑ = ℒ.kron(Σᶻ₁, Iₑ)

    Γ₂ = [ Iₑ              zeros(S, nᵉ, nᵉ^2 + nᵉ * nˢ)
            zeros(S, nᵉ^2, nᵉ)    e4_minus     zeros(S, nᵉ^2, nᵉ * nˢ)
            zeros(S, nˢ * nᵉ, nᵉ + nᵉ^2)    kron_Σᶻ₁_Iₑ ]

    CC = ê_to_ŝ₂ * Γ₂ * ê_to_ŝ₂'

    lyap_ws_2nd = ensure_lyapunov_workspace!(𝓂.workspaces, size(ŝ_to_ŝ₂, 1), :second_order)

    lyap_out, lyap_pb = rrule(solve_lyapunov_equation,
                              Float64.(ŝ_to_ŝ₂), Float64.(CC), lyap_ws_2nd;
                              initial_guess = 𝓂.caches.covariance_second_order,
                              lyapunov_algorithm = opts.lyapunov_algorithm,
                              tol = opts.tol.second_order.ad.lyapunov,
                              verbose = opts.verbose,
                              has_unit_roots = 𝓂.caches.has_unit_roots)
    Σᶻ₂ = lyap_out[1]
    info = lyap_out[2]

    # Cache the 2nd-order Lyapunov result for reuse
    if info
        if size(𝓂.caches.covariance_second_order) != size(Σᶻ₂)
            𝓂.caches.covariance_second_order = Matrix{Float64}(undef, size(Σᶻ₂)...)
        end
        copyto!(𝓂.caches.covariance_second_order, Σᶻ₂)
        𝓂.caches.valid_for.covariance_second_order = Float64.(parameters)
    end

    if !info; return zero_15(), zero_pb; end

    Σʸ₂ = ŝ_to_y₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_y₂ * Γ₂ * ê_to_y₂'
    autocorr_tmp = ŝ_to_ŝ₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_ŝ₂ * Γ₂ * ê_to_y₂'

    slvd = solved && solved2 && info

    result = (Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂_raw, ∇₂, slvd)

    # ── Pullback ──
    function calculate_second_order_moments_with_covariance_pullback(∂out)
        ∂Σʸ₂_in, ∂Σᶻ₂_pass, ∂μʸ₂_in, ∂Δμˢ₂_in, ∂at_in,
            ∂ŝŝ₂_pass, ∂ŝy₂_pass, ∂Σʸ₁_pass, ∂Σᶻ₁_pass, ∂SS_pass,
            ∂𝐒₁_pass, ∂∇₁_pass, ∂𝐒₂_pass, ∂∇₂_pass, _ = ∂out

        # Materialise any InplaceableThunk / Thunk wrappers
        ∂Σʸ₂_in   = unthunk(∂Σʸ₂_in)
        ∂Σᶻ₂_pass = unthunk(∂Σᶻ₂_pass)
        ∂μʸ₂_in   = unthunk(∂μʸ₂_in)
        ∂Δμˢ₂_in  = unthunk(∂Δμˢ₂_in)
        ∂at_in    = unthunk(∂at_in)
        ∂ŝŝ₂_pass = unthunk(∂ŝŝ₂_pass)
        ∂ŝy₂_pass = unthunk(∂ŝy₂_pass)
        ∂Σʸ₁_pass = unthunk(∂Σʸ₁_pass)
        ∂Σᶻ₁_pass = unthunk(∂Σᶻ₁_pass)
        ∂SS_pass   = unthunk(∂SS_pass)
        ∂𝐒₁_pass   = unthunk(∂𝐒₁_pass)
        ∂∇₁_pass   = unthunk(∂∇₁_pass)
        ∂𝐒₂_pass   = unthunk(∂𝐒₂_pass)
        ∂∇₂_pass   = unthunk(∂∇₂_pass)

        # Accumulators
        ∂𝐒₁_acc = zeros(S, size(𝐒₁))
        ∂S2f     = zeros(S, size(𝐒₂_full))
        ∂SS_acc  = zeros(S, length(SS_and_pars))
        ∂∇₁_acc  = zeros(S, size(∇₁))
        ∂Σᶻ₁_acc = zeros(S, nˢ, nˢ)

        ∂ŝ_to_ŝ₂_acc = zeros(S, size(ŝ_to_ŝ₂))
        ∂ŝ_to_y₂_acc = zeros(S, size(ŝ_to_y₂))
        ∂ê_to_ŝ₂_acc = zeros(S, size(ê_to_ŝ₂))
        ∂ê_to_y₂_acc = zeros(S, size(ê_to_y₂))
        ∂Γ₂_acc      = zeros(S, size(Γ₂))
        ∂Σᶻ₂_acc     = zeros(S, size(Σᶻ₂))

        # Pass-through cotangents
        if !(∂𝐒₁_pass  isa AbstractZero); ∂𝐒₁_acc .+= ∂𝐒₁_pass;  end
        if !(∂SS_pass   isa AbstractZero); ∂SS_acc  .+= ∂SS_pass;   end
        # ∂𝐒₂_pass is now compressed — accumulate after ∂S2f * 𝐔₂' conversion below
        if !(∂∇₁_pass   isa AbstractZero); ∂∇₁_acc  .+= ∂∇₁_pass;   end
        if !(∂Σᶻ₁_pass  isa AbstractZero); ∂Σᶻ₁_acc .+= ∂Σᶻ₁_pass;  end
        if !(∂Σᶻ₂_pass  isa AbstractZero); ∂Σᶻ₂_acc .+= ∂Σᶻ₂_pass;  end
        if !(∂ŝŝ₂_pass  isa AbstractZero); ∂ŝ_to_ŝ₂_acc .+= ∂ŝŝ₂_pass; end
        if !(∂ŝy₂_pass  isa AbstractZero); ∂ŝ_to_y₂_acc .+= ∂ŝy₂_pass; end

        # ──── Backprop through Σʸ₂ ────
        # Σʸ₂ = ŝ_to_y₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_y₂ * Γ₂ * ê_to_y₂'
        if !(∂Σʸ₂_in isa AbstractZero)
            ∂Σʸ₂_sym = ∂Σʸ₂_in + ∂Σʸ₂_in'
            ∂ŝ_to_y₂_acc .+= ∂Σʸ₂_sym * ŝ_to_y₂ * Σᶻ₂
            ∂Σᶻ₂_acc     .+= ŝ_to_y₂' * ∂Σʸ₂_in * ŝ_to_y₂
            ∂ê_to_y₂_acc .+= ∂Σʸ₂_sym * ê_to_y₂ * Γ₂
            ∂Γ₂_acc      .+= ê_to_y₂' * ∂Σʸ₂_in * ê_to_y₂
        end

        # ──── Backprop through autocorr_tmp ────
        # autocorr_tmp = ŝ_to_ŝ₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_ŝ₂ * Γ₂ * ê_to_y₂'
        # For C = A*X*B': ∂A = ∂C*B*X', ∂X = A'*∂C*B, ∂B = ∂C'*A*X
        if !(∂at_in isa AbstractZero)
            ∂at = ∂at_in
            ∂ŝ_to_ŝ₂_acc .+= ∂at * ŝ_to_y₂ * Σᶻ₂
            ∂Σᶻ₂_acc     .+= ŝ_to_ŝ₂' * ∂at * ŝ_to_y₂
            ∂ŝ_to_y₂_acc .+= ∂at' * ŝ_to_ŝ₂ * Σᶻ₂
            ∂ê_to_ŝ₂_acc .+= ∂at * ê_to_y₂ * Γ₂
            ∂Γ₂_acc      .+= ê_to_ŝ₂' * ∂at * ê_to_y₂
            ∂ê_to_y₂_acc .+= ∂at' * ê_to_ŝ₂ * Γ₂
        end

        # ──── Backprop through Lyapunov: Σᶻ₂ = lyap(ŝ_to_ŝ₂, CC) ────
        lyap_grad = lyap_pb((∂Σᶻ₂_acc, NoTangent()))
        ∂ŝ_to_ŝ₂_lyap = lyap_grad[2] isa AbstractZero ? zeros(S, size(ŝ_to_ŝ₂)) : S.(lyap_grad[2])
        ∂CC            = lyap_grad[3] isa AbstractZero ? zeros(S, size(CC))         : S.(lyap_grad[3])
        ∂ŝ_to_ŝ₂_acc .+= ∂ŝ_to_ŝ₂_lyap

        # ──── Backprop through CC = ê_to_ŝ₂ * Γ₂ * ê_to_ŝ₂' ────
        ∂CC_sym = ∂CC + ∂CC'
        ∂ê_to_ŝ₂_acc .+= ∂CC_sym * ê_to_ŝ₂ * Γ₂
        ∂Γ₂_acc      .+= ê_to_ŝ₂' * ∂CC * ê_to_ŝ₂

        # ──── Backprop through Γ₂ → ∂Σᶻ₁ ────
        # Only the bottom-right block kron(Σᶻ₁, Iₑ) depends on parameters
        br_row = nᵉ + nᵉ^2
        ∂Γ₂_br = ∂Γ₂_acc[br_row+1:end, br_row+1:end]
        ∂Σᶻ₁_from_Γ₂, _ = kron_vjp_helper(∂Γ₂_br, Σᶻ₁, Iₑ)
        ∂Σᶻ₁_acc .+= ∂Σᶻ₁_from_Γ₂

        # ──── Backprop through μʸ₂ (same as base) ────
        if !(∂μʸ₂_in isa AbstractZero)
            ∂μʸ₂ = ∂μʸ₂_in
            ∂SS_acc[1:nVars] .+= ∂μʸ₂
            ∂ŝ_to_y₂_acc .+= ∂μʸ₂ * μˢ⁺₂'
            ∂μˢ⁺₂ = ŝ_to_y₂' * ∂μʸ₂
            ∂yv₂ = copy(∂μʸ₂)

            λ = A_mean' \ ∂μˢ⁺₂
            ∂ŝv₂ = copy(λ)
            ∂ŝ_to_ŝ₂_acc .+= λ * μˢ⁺₂'

            # yv₂
            ∂S2f[:, kron_v_v] .+= reshape(∂yv₂ / 2, :, 1)
            ∂S2f[:, kron_e_e] .+= (∂yv₂ / 2) * vec_Iₑ'

            # ŝv₂
            ∂ŝv₂_mid = ∂ŝv₂[nˢ+1:2nˢ]
            ∂ŝv₂_bot = ∂ŝv₂[2nˢ+1:end]
            ∂S2f[iˢ, kron_v_v] .+= reshape(∂ŝv₂_mid / 2, :, 1)
            ∂S2f[iˢ, kron_e_e] .+= (∂ŝv₂_mid / 2) * vec_Iₑ'
            ∂e₁ke₁_from_ŝv = ∂ŝv₂_bot * vec_Iₑ'
        else
            ∂e₁ke₁_from_ŝv = zeros(S, size(e₁_kron_e₁))
        end

        # ──── Backprop through Δμˢ₂ ────
        if !(∂Δμˢ₂_in isa AbstractZero)
            λ_Δ = A_Δ' \ ∂Δμˢ₂_in
            ∂𝐒₁_acc[iˢ, 1:nˢ] .+= λ_Δ * Δμˢ₂'
            ∂S2f[iˢ, kron_s_s]  .+= λ_Δ * vec(Σᶻ₁)' / 2
            ∂Σᶻ₁_acc .+= reshape(s_s_to_s₂' * λ_Δ / 2, nˢ, nˢ)
            ∂S2f[iˢ, kron_v_v]  .+= reshape(λ_Δ / 2, :, 1)
            ∂S2f[iˢ, kron_e_e]  .+= (λ_Δ / 2) * vec_Iₑ'
        end

        # ──── Distribute block matrix grads to slice grads ────
        # ŝ_to_y₂ = [s_to_y₁  s_to_y₁  s_s_to_y₂/2]
        ∂𝐒₁_acc[:, 1:nˢ]    .+= ∂ŝ_to_y₂_acc[:, 1:nˢ] .+ ∂ŝ_to_y₂_acc[:, nˢ+1:2nˢ]
        ∂S2f[:, kron_s_s]    .+= ∂ŝ_to_y₂_acc[:, 2nˢ+1:end] / 2

        # ê_to_y₂ = [e_to_y₁  e_e_to_y₂/2  s_e_to_y₂]
        ∂𝐒₁_acc[:, nˢ+1:end] .+= ∂ê_to_y₂_acc[:, 1:nᵉ]
        ∂S2f[:, kron_e_e]     .+= ∂ê_to_y₂_acc[:, nᵉ+1:nᵉ+nᵉ^2] / 2
        ∂S2f[:, kron_s_e]     .+= ∂ê_to_y₂_acc[:, nᵉ+nᵉ^2+1:end]

        # ŝ_to_ŝ₂ blocks
        ∂s₁_from_ŝŝ  = ∂ŝ_to_ŝ₂_acc[1:nˢ, 1:nˢ] + ∂ŝ_to_ŝ₂_acc[nˢ+1:2nˢ, nˢ+1:2nˢ]
        ∂ss2_from_ŝŝ = ∂ŝ_to_ŝ₂_acc[nˢ+1:2nˢ, 2nˢ+1:end] / 2
        ∂s₁ks₁_from_ŝŝ = ∂ŝ_to_ŝ₂_acc[2nˢ+1:end, 2nˢ+1:end]

        # ê_to_ŝ₂ blocks
        ∂𝐒₁_acc[iˢ, nˢ+1:end] .+= ∂ê_to_ŝ₂_acc[1:nˢ, 1:nᵉ]      # e_to_s₁
        ∂S2f[iˢ, kron_e_e]     .+= ∂ê_to_ŝ₂_acc[nˢ+1:2nˢ, nᵉ+1:nᵉ+nᵉ^2] / 2  # e_e_to_s₂
        ∂S2f[iˢ, kron_s_e]     .+= ∂ê_to_ŝ₂_acc[nˢ+1:2nˢ, nᵉ+nᵉ^2+1:end]       # s_e_to_s₂
        ∂e₁ke₁_from_ê = ∂ê_to_ŝ₂_acc[2nˢ+1:end, nᵉ+1:nᵉ+nᵉ^2]
        ∂Ips_s₁ke₁   = ∂ê_to_ŝ₂_acc[2nˢ+1:end, nᵉ+nᵉ^2+1:end]
        # I_plus_s_s * s₁_kron_e₁ → ∂s₁_kron_e₁ += I_plus_s_s' * ∂Ips_s₁ke₁
        ∂s₁ke₁_from_ê = I_plus_s_s' * ∂Ips_s₁ke₁

        # ──── Kron VJPs ────
        ∂s₁_L, ∂s₁_R = kron_vjp_helper(∂s₁ks₁_from_ŝŝ, s_to_s₁, s_to_s₁)
        ∂e₁ke₁_total = ∂e₁ke₁_from_ŝv .+ ∂e₁ke₁_from_ê
        ∂e₁_L, ∂e₁_R = kron_vjp_helper(∂e₁ke₁_total, e_to_s₁, e_to_s₁)
        ∂s₁_se_L, ∂e₁_se_R = kron_vjp_helper(∂s₁ke₁_from_ê, s_to_s₁, e_to_s₁)

        # Aggregate into 𝐒₁
        ∂𝐒₁_acc[iˢ, 1:nˢ]     .+= ∂s₁_from_ŝŝ .+ ∂s₁_L .+ ∂s₁_R .+ ∂s₁_se_L
        ∂𝐒₁_acc[iˢ, nˢ+1:end] .+= ∂e₁_L .+ ∂e₁_R .+ ∂e₁_se_R
        ∂S2f[iˢ, kron_s_s]    .+= ∂ss2_from_ŝŝ

        # ── Σᶻ₁ → Σʸ₁ ──
        ∂Σʸ₁ = zeros(S, size(Σʸ₁))
        ∂Σʸ₁[iˢ, iˢ] .= ∂Σᶻ₁_acc
        if !(∂Σʸ₁_pass isa AbstractZero); ∂Σʸ₁ .+= ∂Σʸ₁_pass; end

        # ── S₂_full → S₂_raw (compressed) ──
        ∂S2_raw = ∂S2f * 𝐔₂'
        # Add compressed pass-through from callers (position 13 now holds compressed 𝐒₂_raw)
        if !(∂𝐒₂_pass isa AbstractZero); ∂S2_raw .+= ∂𝐒₂_pass; end

        # ── Chain through sub-rrule pullbacks ──
        so2_grad = so2_pb((∂S2_raw, NoTangent()))
        ∂∇₁_from_so2 = so2_grad[2] isa AbstractZero ? zeros(S, size(∇₁)) : so2_grad[2]
        ∂∇₂_total    = so2_grad[3] isa AbstractZero ? zeros(S, size(∇₂)) : so2_grad[3]
        ∂𝐒₁_from_so2 = so2_grad[4] isa AbstractZero ? zeros(S, size(𝐒₁)) : so2_grad[4]
        ∂∇₁_acc .+= ∂∇₁_from_so2
        ∂𝐒₁_acc .+= ∂𝐒₁_from_so2

        if !(∂∇₂_pass isa AbstractZero); ∂∇₂_total = ∂∇₂_total .+ ∂∇₂_pass; end

        hess_grad = hess_pb(∂∇₂_total)
        ∂params_hess = hess_grad[2] isa AbstractZero ? zeros(S, np) : hess_grad[2]
        ∂SS_from_hess = hess_grad[3] isa AbstractZero ? zeros(S, length(SS_and_pars)) : hess_grad[3]
        ∂SS_acc .+= ∂SS_from_hess

        cov_grad = cov_pb((∂Σʸ₁, ∂𝐒₁_acc, ∂∇₁_acc, ∂SS_acc, NoTangent()))
        ∂params_cov = cov_grad[2] isa AbstractZero ? zeros(S, np) : cov_grad[2]

        ∂parameters_total = ∂params_hess .+ ∂params_cov

        return NoTangent(), ∂parameters_total, NoTangent()
    end

    return result, calculate_second_order_moments_with_covariance_pullback
end


# ── calculate_third_order_moments rrule ────────────────────────────────────────
function rrule(::typeof(calculate_third_order_moments),
                parameters::Vector{T},
                observables::Union{Symbol_input,String_input},
                𝓂::ℳ;
                covariance::Union{Symbol_input,String_input} = Symbol[],
                opts::CalculationOptions = merge_calculation_options()) where T <: Real

    # ── Non-differentiable constants ──
    ensure_moments_constants!(𝓂.constants)
    so = 𝓂.constants.second_order
    to = 𝓂.constants.third_order
    T_pm = 𝓂.constants.post_model_macro
    np = length(parameters)
    nᵉ = T_pm.nExo

    zero_4() = (zeros(T,0,0), zeros(T,0), zeros(T,0), false)
    zero_pb(_) = (NoTangent(), zeros(T, np), NoTangent(), NoTangent())

    # ── Step 1: Second-order moments with covariance ──
    som2_out, som2_pb = rrule(calculate_second_order_moments_with_covariance, parameters, 𝓂; opts = opts)
    Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr_tmp_2, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂_raw, ∇₂, solved = som2_out

    if !solved; return zero_4(), zero_pb; end

    # Expand compressed 𝐒₂_raw to full for moments computation
    𝐔₂ = 𝓂.constants.second_order.𝐔₂
    𝐒₂ = (sparse(𝐒₂_raw) * 𝐔₂)::SparseMatrixCSC{T, Int}  # was: dense_to_sparse

    # ── Step 2: Third-order derivatives ──
    ∇₃, ∇₃_pb = rrule(calculate_third_order_derivatives, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives, 𝓂.workspaces)

    # ── Step 3: Third-order solution (pass compressed 𝐒₂_raw) ──
    so3_out, so3_pb = rrule(calculate_third_order_solution, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂_raw,
                            𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                            initial_guess = 𝓂.caches.third_order_solution,
                            opts = opts,
                            parameter_values = parameters)
    𝐒₃, solved3 = so3_out

    update_perturbation_counter!(𝓂.counters, solved3, order = 3)

    if !solved3; return zero_4(), zero_pb; end

    # ── Step 4: Decompress S₃ ──
    𝐔₃ = 𝓂.constants.third_order.𝐔₃
    𝐒₃_full = 𝐒₃ * 𝐔₃

    𝐒₃_full = sparse(𝐒₃_full)

    # ── Step 5: Determine iteration groups ──
    orders = determine_efficient_order(𝐒₁, 𝐒₂, 𝐒₃_full, 𝓂.constants, observables,
                                       covariance = covariance, tol = opts.tol.third_order.dependencies_tol)

    kron_e_e = so.kron_e_e
    kron_v_v = so.kron_v_v
    kron_e_v = to.kron_e_v
    e_in_s⁺ = so.e_in_s⁺
    v_in_s⁺ = so.v_in_s⁺
    vec_Iₑ = so.vec_Iₑ
    e4_nᵉ²_nᵉ² = so.e4_nᵉ²_nᵉ²
    e4_nᵉ_nᵉ³ = so.e4_nᵉ_nᵉ³
    e4_minus_vecIₑ_outer = so.e4_minus_vecIₑ_outer
    e6_nᵉ³_nᵉ³ = to.e6_nᵉ³_nᵉ³

    Σʸ₃ = zeros(T, size(Σʸ₂))
    solved_lyapunov = true

    # Per-iteration storage for pullback
    n_iters = length(orders)
    iter_data = Vector{Any}(undef, n_iters)

    for (iter_idx, ords) in enumerate(orders)
        variance_observable, dependencies_all_vars = ords

        sort!(variance_observable)
        sort!(dependencies_all_vars)

        dependencies = intersect(T_pm.past_not_future_and_mixed, dependencies_all_vars)

        obs_in_y = indexin(variance_observable, T_pm.var)

        dependencies_in_states_idx = indexin(dependencies, T_pm.past_not_future_and_mixed)

        dependencies_in_var_idx = Int.(indexin(dependencies, T_pm.var))

        nˢ = length(dependencies)

        iˢ = dependencies_in_var_idx

        Σ̂ᶻ₁ = Σʸ₁[iˢ, iˢ]

        dependencies_extended_idx = vcat(dependencies_in_states_idx,
                dependencies_in_states_idx .+ T_pm.nPast_not_future_and_mixed,
                findall(ℒ.kron(T_pm.past_not_future_and_mixed .∈ (intersect(T_pm.past_not_future_and_mixed,dependencies),),
                               T_pm.past_not_future_and_mixed .∈ (intersect(T_pm.past_not_future_and_mixed,dependencies),))) .+ 2*T_pm.nPast_not_future_and_mixed)

        Σ̂ᶻ₂ = Σᶻ₂[dependencies_extended_idx, dependencies_extended_idx]

        Δ̂μˢ₂ = Δμˢ₂[dependencies_in_states_idx]

        s_in_s⁺ = BitVector(vcat(T_pm.past_not_future_and_mixed .∈ (dependencies,), zeros(Bool, nᵉ + 1)))

        substate_indices = ensure_moments_substate_indices!(𝓂, nˢ)
        I_plus_s_s = substate_indices.I_plus_s_s
        e_es = substate_indices.e_es
        e_ss = substate_indices.e_ss
        ss_s = substate_indices.ss_s
        s_s = substate_indices.s_s
        D₂ˢ = substate_indices.D₂ˢ
        L₂ˢ = substate_indices.L₂ˢ
        D₃ˢ = substate_indices.D₃ˢ
        L₃ˢ = substate_indices.L₃ˢ
        n₂ˢ = size(D₂ˢ, 2)
        n₃ˢ = size(D₃ˢ, 2)

        # first order slices
        s_to_y₁ = 𝐒₁[obs_in_y,:][:,dependencies_in_states_idx]
        e_to_y₁ = 𝐒₁[obs_in_y,:][:, (T_pm.nPast_not_future_and_mixed + 1):end]

        s_to_s₁ = 𝐒₁[iˢ, dependencies_in_states_idx]
        e_to_s₁ = 𝐒₁[iˢ, (T_pm.nPast_not_future_and_mixed + 1):end]

        # second order slices
        dep_kron = ensure_moments_dependency_kron_indices!(𝓂, dependencies, s_in_s⁺)
        kron_s_s = dep_kron.kron_s_s
        kron_s_e = dep_kron.kron_s_e

        s_s_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_s]
        e_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_e_e]
        s_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_e]

        s_s_to_s₂ = 𝐒₂[iˢ, kron_s_s] |> collect
        e_e_to_s₂ = 𝐒₂[iˢ, kron_e_e]
        v_v_to_s₂ = 𝐒₂[iˢ, kron_v_v] |> collect
        s_e_to_s₂ = 𝐒₂[iˢ, kron_s_e]

        s_to_s₁_by_s_to_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
        e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
        s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)
        s_to_s₁_by_s_to_s₁_c = L₂ˢ * s_to_s₁_by_s_to_s₁ * D₂ˢ

        # third order slices
        kron_s_v = dep_kron.kron_s_v

        kron_s_s_s = ℒ.kron(kron_s_s, s_in_s⁺)
        kron_s_s_e = ℒ.kron(kron_s_s, e_in_s⁺)
        kron_s_e_e = ℒ.kron(kron_s_e, e_in_s⁺)
        kron_e_e_e = ℒ.kron(kron_e_e, e_in_s⁺)
        kron_s_v_v = ℒ.kron(kron_s_v, v_in_s⁺)
        kron_e_v_v = ℒ.kron(kron_e_v, v_in_s⁺)

        s_s_s_to_y₃ = 𝐒₃_full[obs_in_y,:][:, kron_s_s_s]
        s_s_e_to_y₃ = 𝐒₃_full[obs_in_y,:][:, kron_s_s_e]
        s_e_e_to_y₃ = 𝐒₃_full[obs_in_y,:][:, kron_s_e_e]
        e_e_e_to_y₃ = 𝐒₃_full[obs_in_y,:][:, kron_e_e_e]
        s_v_v_to_y₃ = 𝐒₃_full[obs_in_y,:][:, kron_s_v_v]
        e_v_v_to_y₃ = 𝐒₃_full[obs_in_y,:][:, kron_e_v_v]

        s_s_s_to_s₃ = 𝐒₃_full[iˢ, kron_s_s_s]
        s_s_e_to_s₃ = 𝐒₃_full[iˢ, kron_s_s_e]
        s_e_e_to_s₃ = 𝐒₃_full[iˢ, kron_s_e_e]
        e_e_e_to_s₃ = 𝐒₃_full[iˢ, kron_e_e_e]
        s_v_v_to_s₃ = 𝐒₃_full[iˢ, kron_s_v_v]
        e_v_v_to_s₃ = 𝐒₃_full[iˢ, kron_e_v_v]

        # Set up pruned state transition sub-blocks (compressed)
        N_upper = 2 * nˢ + n₂ˢ
        N_lower = nˢ + nˢ^2 + n₃ˢ

        A_UU = [s_to_s₁                spzeros(nˢ, nˢ + n₂ˢ)
                spzeros(nˢ, nˢ) s_to_s₁   s_s_to_s₂ / 2 * D₂ˢ
                spzeros(n₂ˢ, 2 * nˢ)               s_to_s₁_by_s_to_s₁_c]

        A_LU = [s_v_v_to_s₃ / 2                    spzeros(nˢ, nˢ + n₂ˢ)
                ℒ.kron(s_to_s₁,v_v_to_s₂ / 2)    spzeros(nˢ^2, nˢ + n₂ˢ)
                spzeros(n₃ˢ, 2 * nˢ + n₂ˢ)]

        A_LL = [s_to_s₁           s_s_to_s₂             s_s_s_to_s₃ / 6 * D₃ˢ
                spzeros(nˢ^2, nˢ) s_to_s₁_by_s_to_s₁  ℒ.kron(s_to_s₁,s_s_to_s₂ / 2) * D₃ˢ
                spzeros(n₃ˢ, nˢ + nˢ^2)               L₃ˢ * ℒ.kron(s_to_s₁,s_to_s₁_by_s_to_s₁) * D₃ˢ]

        ê_to_ŝ₃ = [ e_to_s₁   zeros(nˢ,nᵉ^2 + 2*nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        zeros(nˢ,nᵉ)  e_e_to_s₂ / 2   s_e_to_s₂   zeros(nˢ,nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        zeros(n₂ˢ,nᵉ)  L₂ˢ * e_to_s₁_by_e_to_s₁  L₂ˢ * I_plus_s_s * s_to_s₁_by_e_to_s₁  zeros(n₂ˢ, nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        e_v_v_to_s₃ / 2    zeros(nˢ,nᵉ^2 + nᵉ * nˢ)  s_e_to_s₂    s_s_e_to_s₃ / 2    s_e_e_to_s₃ / 2    e_e_e_to_s₃ / 6
                                        ℒ.kron(e_to_s₁, v_v_to_s₂ / 2)    zeros(nˢ^2, nᵉ^2 + nᵉ * nˢ)      s_s * s_to_s₁_by_e_to_s₁    ℒ.kron(s_to_s₁, s_e_to_s₂) + s_s * ℒ.kron(s_s_to_s₂ / 2, e_to_s₁)  ℒ.kron(s_to_s₁, e_e_to_s₂ / 2) + s_s * ℒ.kron(s_e_to_s₂, e_to_s₁)  ℒ.kron(e_to_s₁, e_e_to_s₂ / 2)
                                        zeros(n₃ˢ, nᵉ + nᵉ^2 + 2*nᵉ * nˢ) L₃ˢ * (ℒ.kron(s_to_s₁_by_s_to_s₁,e_to_s₁) + ℒ.kron(s_to_s₁, s_s * s_to_s₁_by_e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_s_to_s₁) * e_ss)   L₃ˢ * (ℒ.kron(s_to_s₁_by_e_to_s₁,e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_e_to_s₁) * e_es + ℒ.kron(e_to_s₁, s_s * s_to_s₁_by_e_to_s₁) * e_es)  L₃ˢ * ℒ.kron(e_to_s₁,e_to_s₁_by_e_to_s₁)]

        ŝ_to_y₃ = [s_to_y₁ + s_v_v_to_y₃ / 2  s_to_y₁  s_s_to_y₂ / 2 * D₂ˢ   s_to_y₁    s_s_to_y₂     s_s_s_to_y₃ / 6 * D₃ˢ]

        ê_to_y₃ = [e_to_y₁ + e_v_v_to_y₃ / 2  e_e_to_y₂ / 2  s_e_to_y₂   s_e_to_y₂     s_s_e_to_y₃ / 2    s_e_e_to_y₃ / 2    e_e_e_to_y₃ / 6]

        μˢ₃δμˢ₁ = reshape((ℒ.I(size(s_to_s₁_by_s_to_s₁, 1)) - s_to_s₁_by_s_to_s₁) \ vec( 
                                    (s_s_to_s₂  * reshape(ss_s * vec(Σ̂ᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σ̂ᶻ₁) * Δ̂μˢ₂'),nˢ^2, nˢ) +
                                    s_s_s_to_s₃ * reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end , 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ^3, nˢ) / 6 +
                                    s_e_e_to_s₃ * ℒ.kron(Σ̂ᶻ₁, vec_Iₑ) / 2 +
                                    s_v_v_to_s₃ * Σ̂ᶻ₁ / 2) * s_to_s₁' +
                                    (s_e_to_s₂  * ℒ.kron(Δ̂μˢ₂,ℒ.I(nᵉ)) +
                                    e_e_e_to_s₃ * e4_nᵉ_nᵉ³' / 6 +
                                    s_s_e_to_s₃ * ℒ.kron(vec(Σ̂ᶻ₁), ℒ.I(nᵉ)) / 2 +
                                    e_v_v_to_s₃ * ℒ.I(nᵉ) / 2) * e_to_s₁'
                                    ), nˢ, nˢ)

        Γ₃ = [ ℒ.I(nᵉ)             spzeros(nᵉ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δ̂μˢ₂', ℒ.I(nᵉ))  ℒ.kron(vec(Σ̂ᶻ₁)', ℒ.I(nᵉ)) spzeros(nᵉ, nˢ * nᵉ^2)    e4_nᵉ_nᵉ³
                spzeros(nᵉ^2, nᵉ)    e4_minus_vecIₑ_outer     spzeros(nᵉ^2, 2*nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
                spzeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σ̂ᶻ₁, ℒ.I(nᵉ))   spzeros(nˢ * nᵉ, nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
                ℒ.kron(Δ̂μˢ₂,ℒ.I(nᵉ))    spzeros(nᵉ * nˢ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σ̂ᶻ₂[nˢ + 1:2*nˢ,nˢ + 1:2*nˢ] + Δ̂μˢ₂ * Δ̂μˢ₂',ℒ.I(nᵉ)) ℒ.kron(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)',ℒ.I(nᵉ))   spzeros(nᵉ * nˢ, nˢ * nᵉ^2) ℒ.kron(Δ̂μˢ₂, e4_nᵉ_nᵉ³)
                ℒ.kron(vec(Σ̂ᶻ₁), ℒ.I(nᵉ))  spzeros(nᵉ * nˢ^2, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σ̂ᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σ̂ᶻ₁) * Δ̂μˢ₂', ℒ.I(nᵉ))  ℒ.kron(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', ℒ.I(nᵉ))   spzeros(nᵉ * nˢ^2, nˢ * nᵉ^2)  ℒ.kron(vec(Σ̂ᶻ₁), e4_nᵉ_nᵉ³)
                spzeros(nˢ*nᵉ^2, nᵉ + nᵉ^2 + 2*nᵉ * nˢ + nˢ^2*nᵉ)   ℒ.kron(Σ̂ᶻ₁, e4_nᵉ²_nᵉ²)    spzeros(nˢ*nᵉ^2,nᵉ^3)
                e4_nᵉ_nᵉ³'  spzeros(nᵉ^3, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δ̂μˢ₂', e4_nᵉ_nᵉ³')     ℒ.kron(vec(Σ̂ᶻ₁)', e4_nᵉ_nᵉ³')  spzeros(nᵉ^3, nˢ*nᵉ^2)     e6_nᵉ³_nᵉ³]


        Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + n₂ˢ + nˢ^2 + n₃ˢ)
                ℒ.kron(Σ̂ᶻ₁,vec_Iₑ)   zeros(nˢ*nᵉ^2, nˢ + n₂ˢ)  ℒ.kron(μˢ₃δμˢ₁',vec_Iₑ)    ℒ.kron(reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec_Iₑ)  ℒ.kron(reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3) * L₃ˢ', vec_Iₑ)
                spzeros(nᵉ^3, 3*nˢ + n₂ˢ + nˢ^2 + n₃ˢ)]

        droptol!(A_UU, eps())
        droptol!(A_LU, eps())
        droptol!(A_LL, eps())
        droptol!(ê_to_ŝ₃, eps())
        droptol!(Eᴸᶻ, eps())
        droptol!(Γ₃, eps())

        # ── Standard Lyapunov solve (compressed) ──
        ŝ_to_ŝ₃ = [A_UU spzeros(N_upper, N_lower); A_LU A_LL]

        A_cross = Matrix{Float64}(ê_to_ŝ₃ * Eᴸᶻ) * ŝ_to_ŝ₃'
        C_dense = Matrix{Float64}(sparse_ABAt(ê_to_ŝ₃, Γ₃)) + A_cross + A_cross'

        N_total = N_upper + N_lower
        lyap_ws_3rd = Lyapunov_workspace(N_total)
        lyap_out, lyap_pb_iter = rrule(solve_lyapunov_equation,
                                    ŝ_to_ŝ₃, C_dense, lyap_ws_3rd,
                                    lyapunov_algorithm = opts.lyapunov_algorithm,
                                    tol = opts.tol.third_order.ad.lyapunov,
                                    verbose = opts.verbose)
        Σᶻ₃ = lyap_out[1]
        info = lyap_out[2]

        if !info
            return zero_4(), zero_pb
        end

        solved_lyapunov = solved_lyapunov && info

        Σʸ₃tmp = ŝ_to_y₃ * Σᶻ₃ * ŝ_to_y₃' + sparse_ABAt(ê_to_y₃, Γ₃) + ê_to_y₃ * Eᴸᶻ * ŝ_to_y₃' + ŝ_to_y₃ * Eᴸᶻ' * ê_to_y₃'

        for obs in variance_observable
            Σʸ₃[indexin([obs], T_pm.var), indexin(variance_observable, T_pm.var)] = Σʸ₃tmp[indexin([obs], variance_observable), :]
        end

        # Store per-iteration data for pullback
        iter_data[iter_idx] = (
            variance_observable = variance_observable,
            obs_in_y = obs_in_y,
            iˢ = iˢ,
            nˢ = nˢ,
            dependencies_in_states_idx = dependencies_in_states_idx,
            dependencies_extended_idx = dependencies_extended_idx,
            Σ̂ᶻ₁ = Σ̂ᶻ₁,
            Σ̂ᶻ₂ = Σ̂ᶻ₂,
            Δ̂μˢ₂ = Δ̂μˢ₂,
            s_in_s⁺ = s_in_s⁺,
            s_to_y₁ = s_to_y₁,
            e_to_y₁ = e_to_y₁,
            s_to_s₁ = s_to_s₁,
            e_to_s₁ = e_to_s₁,
            kron_s_s = kron_s_s,
            kron_s_e = kron_s_e,
            kron_s_v = kron_s_v,
            kron_s_s_s = kron_s_s_s,
            kron_s_s_e = kron_s_s_e,
            kron_s_e_e = kron_s_e_e,
            kron_e_e_e = kron_e_e_e,
            kron_s_v_v = kron_s_v_v,
            kron_e_v_v = kron_e_v_v,
            s_s_to_y₂ = s_s_to_y₂,
            e_e_to_y₂ = e_e_to_y₂,
            s_e_to_y₂ = s_e_to_y₂,
            s_s_to_s₂ = s_s_to_s₂,
            e_e_to_s₂ = e_e_to_s₂,
            v_v_to_s₂ = v_v_to_s₂,
            s_e_to_s₂ = s_e_to_s₂,
            s_to_s₁_by_s_to_s₁ = s_to_s₁_by_s_to_s₁,
            e_to_s₁_by_e_to_s₁ = e_to_s₁_by_e_to_s₁,
            s_to_s₁_by_e_to_s₁ = s_to_s₁_by_e_to_s₁,
            s_s_s_to_y₃ = s_s_s_to_y₃,
            s_s_e_to_y₃ = s_s_e_to_y₃,
            s_e_e_to_y₃ = s_e_e_to_y₃,
            e_e_e_to_y₃ = e_e_e_to_y₃,
            s_v_v_to_y₃ = s_v_v_to_y₃,
            e_v_v_to_y₃ = e_v_v_to_y₃,
            s_s_s_to_s₃ = s_s_s_to_s₃,
            s_s_e_to_s₃ = s_s_e_to_s₃,
            s_e_e_to_s₃ = s_e_e_to_s₃,
            e_e_e_to_s₃ = e_e_e_to_s₃,
            s_v_v_to_s₃ = s_v_v_to_s₃,
            e_v_v_to_s₃ = e_v_v_to_s₃,
            ê_to_ŝ₃ = ê_to_ŝ₃,
            ŝ_to_y₃ = ŝ_to_y₃,
            ê_to_y₃ = ê_to_y₃,
            Γ₃ = Γ₃,
            Eᴸᶻ = Eᴸᶻ,
            ŝ_to_ŝ₃ = ŝ_to_ŝ₃,
            Σᶻ₃ = Σᶻ₃,
            Σʸ₃tmp = Σʸ₃tmp,
            μˢ₃δμˢ₁ = μˢ₃δμˢ₁,
            lyap_pb = lyap_pb_iter,
            N_upper = N_upper,
            N_lower = N_lower,
            D₂ˢ = D₂ˢ,
            L₂ˢ = L₂ˢ,
            D₃ˢ = D₃ˢ,
            L₃ˢ = L₃ˢ,
            n₂ˢ = n₂ˢ,
            n₃ˢ = n₃ˢ,
            s_to_s₁_by_s_to_s₁_c = s_to_s₁_by_s_to_s₁_c,
            I_plus_s_s = I_plus_s_s,
            ss_s = ss_s,
            s_s = s_s,
            e_es = e_es,
            e_ss = e_ss,
        )
    end

    # Cache the 3rd-order covariance for reuse
    all_solved_3rd = solved && solved3 && solved_lyapunov
    if all_solved_3rd
        if size(𝓂.caches.covariance_third_order) != size(Σʸ₃)
            𝓂.caches.covariance_third_order = Matrix{Float64}(undef, size(Σʸ₃)...)
        end
        copyto!(𝓂.caches.covariance_third_order, Σʸ₃)
        𝓂.caches.valid_for.covariance_third_order = Float64.(parameters)
        nVars_rrule = T_pm.nVars
        obs_key_rrule = if observables == :full_covar
            collect(1:nVars_rrule)
        else
            obs_idx = parse_variables_input_to_index(observables, 𝓂.constants) |> sort
            if covariance == Symbol[]
                collect(obs_idx)
            else
                covar_idx = parse_variables_input_to_index(covariance, 𝓂.constants) |> sort
                sort(union(obs_idx, covar_idx))
            end
        end
        𝓂.caches.valid_for.covariance_third_order_obs_key = obs_key_rrule
    end

    result = (Σʸ₃, μʸ₂, SS_and_pars, all_solved_3rd)

    # ── Pullback ──
    function calculate_third_order_moments_pullback(∂out)
        ∂Σʸ₃_in, ∂μʸ₂_in, ∂SS_in, _ = ∂out

        ∂Σʸ₃_in = unthunk(∂Σʸ₃_in)
        ∂μʸ₂_in = unthunk(∂μʸ₂_in)
        ∂SS_in  = unthunk(∂SS_in)

        n₋ = T_pm.nPast_not_future_and_mixed

        # Accumulators for cotangents flowing to sub-rrule inputs
        ∂Σʸ₁_acc  = zeros(T, size(Σʸ₁))
        ∂Σᶻ₂_acc  = zeros(T, size(Σᶻ₂))
        ∂Δμˢ₂_acc = zeros(T, length(Δμˢ₂))
        ∂𝐒₁_acc   = zeros(T, size(𝐒₁))
        ∂S2f_acc   = zeros(T, size(𝐒₂))
        ∂S3f_acc   = zeros(T, size(𝐒₃_full))
        ∂SS_acc    = zeros(T, length(SS_and_pars))
        ∂∇₁_acc   = zeros(T, size(∇₁))
        ∂∇₂_acc   = zeros(T, size(∇₂))
        ∂∇₃_acc   = zeros(T, size(∇₃))

        if !(∂SS_in isa AbstractZero); ∂SS_acc .+= ∂SS_in; end

        # ──── Reverse loop over iterations ────
        for iter_idx in n_iters:-1:1
            d = iter_data[iter_idx]
            nˢ_i = d.nˢ
            n₂ˢ_i = d.n₂ˢ
            n₃ˢ_i = d.n₃ˢ

            # ── Gather ∂Σʸ₃tmp from ∂Σʸ₃ (reverse of scatter) ──
            nObs_iter = length(d.variance_observable)
            ∂Σʸ₃tmp = zeros(T, nObs_iter, nObs_iter)

            if !(∂Σʸ₃_in isa AbstractZero)
                ∂Σʸ₃tmp .= ∂Σʸ₃_in[d.obs_in_y, indexin(d.variance_observable, T_pm.var)]
            end

            if ℒ.norm(∂Σʸ₃tmp) < eps(T); continue; end

            ∂Σʸ₃tmp_sym = ∂Σʸ₃tmp + ∂Σʸ₃tmp'

            # ── Σʸ₃tmp = ŝ_y * Σᶻ₃ * ŝ_y' + ê_y * Γ₃ * ê_y' + ê_y * Eᴸᶻ * ŝ_y' + ŝ_y * Eᴸᶻ' * ê_y' ──
            # Terms 1+2 are AXA' forms; terms 3+4 form M + M' where M = ê_y * Eᴸᶻ * ŝ_y'.
            # Effective cotangent for M+M' is G_eff = ∂ + ∂' = ∂Σʸ₃tmp_sym.

            ∂ŝ_to_y₃ = ∂Σʸ₃tmp_sym * (d.ŝ_to_y₃ * d.Σᶻ₃ + d.ê_to_y₃ * Matrix(d.Eᴸᶻ))
            ∂ê_to_y₃ = ∂Σʸ₃tmp_sym * (d.ê_to_y₃ * d.Γ₃  + d.ŝ_to_y₃ * Matrix(d.Eᴸᶻ'))
            ∂Σᶻ₃      = d.ŝ_to_y₃' * ∂Σʸ₃tmp * d.ŝ_to_y₃
            ∂Γ₃_iter   = d.ê_to_y₃' * ∂Σʸ₃tmp * d.ê_to_y₃
            ∂Eᴸᶻ_iter  = d.ê_to_y₃' * ∂Σʸ₃tmp_sym * d.ŝ_to_y₃

            # ── Standard Lyapunov adjoint ──
            Nu = d.N_upper;  Nl = d.N_lower
            ru_i = 1:Nu;  rl_i = (Nu+1):(Nu+Nl)

            lyap_grad = d.lyap_pb((∂Σᶻ₃, NoTangent()))
            ∂ŝ_to_ŝ₃ = lyap_grad[2] isa AbstractZero ? zeros(T, size(d.ŝ_to_ŝ₃)) : Matrix{T}(lyap_grad[2])
            ∂C_lyap   = lyap_grad[3] isa AbstractZero ? zeros(T, size(d.ŝ_to_ŝ₃)) : Matrix{T}(lyap_grad[3])

            # Backprop through C = ê * Γ₃ * ê' + M + M' where M = ê * Eᴸᶻ * ŝ'
            ∂C_sym = ∂C_lyap + ∂C_lyap'
            ê_d = Matrix{T}(d.ê_to_ŝ₃)
            ŝ_d = Matrix{T}(d.ŝ_to_ŝ₃)
            EL_d = Matrix{T}(d.Eᴸᶻ)
            Γ₃_d = Matrix{T}(d.Γ₃)

            # Term 1: ê * Γ₃ * ê'
            ∂Γ₃_iter .+= ê_d' * ∂C_lyap * ê_d
            ∂ê_to_ŝ₃ = ∂C_sym * ê_d * Γ₃_d

            # Terms 2+3: M + M' where M = ê * Eᴸᶻ * ŝ'
            ∂ê_to_ŝ₃ .+= ∂C_sym * ŝ_d * EL_d'
            ∂Eᴸᶻ_iter .+= ê_d' * ∂C_sym * ŝ_d
            ∂ŝ_to_ŝ₃ .+= ∂C_sym' * ê_d * EL_d

            # Extract ∂A_UU, ∂A_LU, ∂A_LL from ∂ŝ_to_ŝ₃
            ∂A_UU = ∂ŝ_to_ŝ₃[ru_i, ru_i]
            ∂A_LU = ∂ŝ_to_ŝ₃[rl_i, ru_i]
            ∂A_LL = ∂ŝ_to_ŝ₃[rl_i, rl_i]


            # ── Disaggregate ŝ_to_y₃ → ∂𝐒₁, ∂𝐒₂, ∂𝐒₃ ──
            # ŝ_to_y₃ = [s_to_y₁+svv/2 | s_to_y₁ | ss_to_y₂/2 | s_to_y₁ | ss_to_y₂ | sss_to_y₃/6]
            c = 0
            ∂blk1 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i];      c += nˢ_i
            ∂blk2 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i];      c += nˢ_i
            ∂blk3 = ∂ŝ_to_y₃[:, c+1:c+n₂ˢ_i];     c += n₂ˢ_i   # compressed
            ∂blk4 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i];      c += nˢ_i
            ∂blk5 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i^2];    c += nˢ_i^2
            ∂blk6 = ∂ŝ_to_y₃[:, c+1:end]

            ∂𝐒₁_acc[d.obs_in_y, d.dependencies_in_states_idx] .+= ∂blk1 .+ ∂blk2 .+ ∂blk4     # ∂s_to_y₁
            ∂S2f_acc[d.obs_in_y, d.kron_s_s]                  .+= (∂blk3 * Matrix(d.D₂ˢ)') ./ 2 .+ ∂blk5           # ∂s_s_to_y₂ (decompress blk3)
            ∂S3f_acc[d.obs_in_y, d.kron_s_v_v]                .+= ∂blk1 ./ 2                     # ∂s_v_v_to_y₃
            ∂S3f_acc[d.obs_in_y, d.kron_s_s_s]                .+= (∂blk6 * Matrix(d.D₃ˢ)') ./ 6  # ∂s_s_s_to_y₃ (decompress blk6)

            # ── Disaggregate ê_to_y₃ → ∂𝐒₁, ∂𝐒₂, ∂𝐒₃ ──
            # ê_to_y₃ = [e_to_y₁+evv/2 | ee_to_y₂/2 | se_to_y₂ | se_to_y₂ | sse_to_y₃/2 | see_to_y₃/2 | eee_to_y₃/6]
            c = 0
            ∂eblk1 = ∂ê_to_y₃[:, c+1:c+nᵉ];          c += nᵉ
            ∂eblk2 = ∂ê_to_y₃[:, c+1:c+nᵉ^2];        c += nᵉ^2
            ∂eblk3 = ∂ê_to_y₃[:, c+1:c+nˢ_i*nᵉ];     c += nˢ_i*nᵉ
            ∂eblk4 = ∂ê_to_y₃[:, c+1:c+nˢ_i*nᵉ];     c += nˢ_i*nᵉ
            ∂eblk5 = ∂ê_to_y₃[:, c+1:c+nˢ_i^2*nᵉ];   c += nˢ_i^2*nᵉ
            ∂eblk6 = ∂ê_to_y₃[:, c+1:c+nˢ_i*nᵉ^2];   c += nˢ_i*nᵉ^2
            ∂eblk7 = ∂ê_to_y₃[:, c+1:end]

            ∂𝐒₁_acc[d.obs_in_y, n₋+1:end]    .+= ∂eblk1                  # ∂e_to_y₁
            ∂S2f_acc[d.obs_in_y, kron_e_e]     .+= ∂eblk2 ./ 2            # ∂e_e_to_y₂
            ∂S2f_acc[d.obs_in_y, d.kron_s_e]   .+= ∂eblk3 .+ ∂eblk4      # ∂s_e_to_y₂
            ∂S3f_acc[d.obs_in_y, d.kron_e_v_v] .+= ∂eblk1 ./ 2            # ∂e_v_v_to_y₃
            ∂S3f_acc[d.obs_in_y, d.kron_s_s_e] .+= ∂eblk5 ./ 2            # ∂s_s_e_to_y₃
            ∂S3f_acc[d.obs_in_y, d.kron_s_e_e] .+= ∂eblk6 ./ 2            # ∂s_e_e_to_y₃
            ∂S3f_acc[d.obs_in_y, d.kron_e_e_e] .+= ∂eblk7 ./ 6            # ∂e_e_e_to_y₃

            # ════════════════════════════════════════════════════════════════════
            # Stage 2+3: Disaggregate block matrices → slice & data cotangents
            # ════════════════════════════════════════════════════════════════════
            n = nˢ_i;  ne = nᵉ
            Ine = Matrix{T}(ℒ.I(ne))
            vec_Ie_col = reshape(T.(vec_Iₑ), :, 1)

            # Dense copies of frequently used slices
            s₁  = Matrix{T}(d.s_to_s₁)
            e₁  = Matrix{T}(d.e_to_s₁)
            s₁² = Matrix{T}(d.s_to_s₁_by_s_to_s₁)
            e₁² = Matrix{T}(d.e_to_s₁_by_e_to_s₁)
            s₁e₁ = Matrix{T}(d.s_to_s₁_by_e_to_s₁)
            ss₂  = Matrix{T}(d.s_s_to_s₂)
            ee₂  = Matrix{T}(d.e_e_to_s₂)
            se₂  = Matrix{T}(d.s_e_to_s₂)
            vv₂  = Matrix{T}(d.v_v_to_s₂)

            # Local slice cotangent accumulators
            ∂s₁_l  = zeros(T, n, n)
            ∂e₁_l  = zeros(T, n, ne)
            ∂ss₂_l = zeros(T, n, n^2)
            ∂ee₂_l = zeros(T, n, ne^2)
            ∂se₂_l = zeros(T, n, n * ne)
            ∂vv₂_l = zeros(T, size(vv₂))
            ∂Σ̂ᶻ₁  = zeros(T, n, n)
            ∂Σ̂ᶻ₂  = zeros(T, size(d.Σ̂ᶻ₂))
            ∂Δ̂μˢ₂_l = zeros(T, n)

            # Block boundary arrays
            sb = cumsum([0, n, n, n₂ˢ_i, n, n^2, n₃ˢ_i])          # ŝ_to_ŝ₃ row/col (compressed)
            eb = cumsum([0, ne, ne^2, n*ne, n*ne, n^2*ne, n*ne^2, ne^3])  # ê_to_ŝ₃ cols
            gb = eb  # Γ₃ row/col (same block sizes)

            vvh = vv₂ ./ 2;  ssh = ss₂ ./ 2;  eeh = ee₂ ./ 2

            # Reusable buffers for in-place kron adjoint operations
            ∂s₁²_buf = zeros(T, n^2, n^2)
            ∂e₁²_buf = zeros(T, n^2, ne^2)
            ∂kron_buf = zeros(T, n^2, n * ne)
            ∂vvh_buf = zeros(T, size(vvh))
            ∂ssh_buf = zeros(T, size(ssh))
            ∂eeh_buf = zeros(T, size(eeh))

            # ── 2a: A_UU, A_LU, A_LL disaggregation ──
            # Block boundaries within sub-matrices
            bu = cumsum([0, n, n, n₂ˢ_i])                 # A_UU row/col blocks
            bl = cumsum([0, n, n^2, n₃ˢ_i])               # A_LL row/col blocks (also A_LU rows)

            # ── From ∂A_UU ──
            # (1,1) s₁, (2,2) s₁
            ∂s₁_l .+= ∂A_UU[bu[1]+1:bu[2], bu[1]+1:bu[2]] .+
                       ∂A_UU[bu[2]+1:bu[3], bu[2]+1:bu[3]]
            # (2,3) ss₂/2 * D₂ˢ — decompress cols
            ∂ss₂_l .+= ∂A_UU[bu[2]+1:bu[3], bu[3]+1:bu[4]] * Matrix(d.D₂ˢ)' ./ 2
            # (3,3) L₂ˢ * kron(s₁,s₁) * D₂ˢ — decompress then kron_vjp
            ∂inner33 = Matrix(d.L₂ˢ)' * Matrix(∂A_UU[bu[3]+1:bu[4], bu[3]+1:bu[4]]) * Matrix(d.D₂ˢ)'
            fill_kron_adjoint!(∂s₁_l, ∂s₁_l, ∂inner33, s₁, s₁)

            # ── From ∂A_LU ──
            # (1,1) s_vv₃/2
            ∂S3f_acc[d.iˢ, d.kron_s_v_v] .+= ∂A_LU[bl[1]+1:bl[2], bu[1]+1:bu[2]] ./ 2
            # (2,1) kron(s₁, vv₂/2)
            ∂vvh_buf .= 0
            fill_kron_adjoint!(∂vvh_buf, ∂s₁_l, Matrix(∂A_LU[bl[2]+1:bl[3], bu[1]+1:bu[2]]), vvh, s₁)
            ∂vv₂_l .+= ∂vvh_buf ./ 2

            # ── From ∂A_LL ──
            # (1,1) s₁
            ∂s₁_l .+= ∂A_LL[bl[1]+1:bl[2], bl[1]+1:bl[2]]
            # (1,2) ss₂
            ∂ss₂_l .+= ∂A_LL[bl[1]+1:bl[2], bl[2]+1:bl[3]]
            # (1,3) sss₃/6 * D₃ˢ — decompress cols
            ∂S3f_acc[d.iˢ, d.kron_s_s_s] .+= ∂A_LL[bl[1]+1:bl[2], bl[3]+1:bl[4]] * Matrix(d.D₃ˢ)' ./ 6
            # (2,2) kron(s₁,s₁)
            fill_kron_adjoint!(∂s₁_l, ∂s₁_l, Matrix(∂A_LL[bl[2]+1:bl[3], bl[2]+1:bl[3]]), s₁, s₁)
            # (2,3) kron(s₁, ss₂/2) * D₃ˢ — decompress cols then kron_vjp
            ∂inner56 = Matrix(∂A_LL[bl[2]+1:bl[3], bl[3]+1:bl[4]]) * Matrix(d.D₃ˢ)'
            ∂ssh_buf .= 0
            fill_kron_adjoint!(∂ssh_buf, ∂s₁_l, ∂inner56, ssh, s₁)
            ∂ss₂_l .+= ∂ssh_buf ./ 2
            # (3,3) L₃ˢ * kron(s₁, kron(s₁,s₁)) * D₃ˢ — decompress then kron_vjp
            ∂inner66 = Matrix(d.L₃ˢ)' * Matrix(∂A_LL[bl[3]+1:bl[4], bl[3]+1:bl[4]]) * Matrix(d.D₃ˢ)'
            ∂s₁²_buf .= 0
            fill_kron_adjoint!(∂s₁²_buf, ∂s₁_l, ∂inner66, s₁², s₁)
            fill_kron_adjoint!(∂s₁_l, ∂s₁_l, ∂s₁²_buf, s₁, s₁)


            # ── 2b: ê_to_ŝ₃ disaggregation ──
            ∂ê₃ = Matrix{T}(∂ê_to_ŝ₃)
            ss_s1e1 = Matrix(d.s_s) * s₁e₁   # pre-compute

            # Row 1: (1,1) e₁
            ∂e₁_l .+= ∂ê₃[sb[1]+1:sb[2], eb[1]+1:eb[2]]
            # Row 2: (2,2) ee₂/2; (2,3) se₂
            ∂ee₂_l .+= ∂ê₃[sb[2]+1:sb[3], eb[2]+1:eb[3]] ./ 2
            ∂se₂_l .+= ∂ê₃[sb[2]+1:sb[3], eb[3]+1:eb[4]]
            # Row 3: (3,2) L₂ˢ * kron(e₁,e₁) — decompress rows
            fill_kron_adjoint!(∂e₁_l, ∂e₁_l, Matrix(d.L₂ˢ)' * Matrix(∂ê₃[sb[3]+1:sb[4], eb[2]+1:eb[3]]), e₁, e₁)
            # (3,3) L₂ˢ * I_plus_s_s * kron(s₁,e₁) — decompress rows
            ∂k33 = Matrix(d.I_plus_s_s') * Matrix(d.L₂ˢ)' * Matrix(∂ê₃[sb[3]+1:sb[4], eb[3]+1:eb[4]])
            fill_kron_adjoint!(∂e₁_l, ∂s₁_l, ∂k33, e₁, s₁)
            # Row 4: direct S₃ slices
            ∂S3f_acc[d.iˢ, d.kron_e_v_v] .+= ∂ê₃[sb[4]+1:sb[5], eb[1]+1:eb[2]] ./ 2
            ∂se₂_l .+= ∂ê₃[sb[4]+1:sb[5], eb[4]+1:eb[5]]
            ∂S3f_acc[d.iˢ, d.kron_s_s_e] .+= ∂ê₃[sb[4]+1:sb[5], eb[5]+1:eb[6]] ./ 2
            ∂S3f_acc[d.iˢ, d.kron_s_e_e] .+= ∂ê₃[sb[4]+1:sb[5], eb[6]+1:eb[7]] ./ 2
            ∂S3f_acc[d.iˢ, d.kron_e_e_e] .+= ∂ê₃[sb[4]+1:sb[5], eb[7]+1:eb[8]] ./ 6
            # Row 5: (5,1) kron(e₁,vv₂/2)
            ∂vvh_buf .= 0
            fill_kron_adjoint!(∂vvh_buf, ∂e₁_l, Matrix(∂ê₃[sb[5]+1:sb[6], eb[1]+1:eb[2]]), vvh, e₁)
            ∂vv₂_l .+= ∂vvh_buf ./ 2
            # (5,4) s_s * kron(s₁,e₁)
            ∂k54 = Matrix(d.s_s') * Matrix(∂ê₃[sb[5]+1:sb[6], eb[4]+1:eb[5]])
            fill_kron_adjoint!(∂e₁_l, ∂s₁_l, ∂k54, e₁, s₁)
            # (5,5) kron(s₁,se₂) + s_s * kron(ss₂/2, e₁)
            ∂b55 = Matrix(∂ê₃[sb[5]+1:sb[6], eb[5]+1:eb[6]])
            fill_kron_adjoint!(∂se₂_l, ∂s₁_l, ∂b55, se₂, s₁)
            ∂k55b = Matrix(d.s_s') * ∂b55
            ∂ssh_buf .= 0
            fill_kron_adjoint!(∂e₁_l, ∂ssh_buf, ∂k55b, e₁, ssh)
            ∂ss₂_l .+= ∂ssh_buf ./ 2
            # (5,6) kron(s₁,ee₂/2) + s_s * kron(se₂, e₁)
            ∂b56 = Matrix(∂ê₃[sb[5]+1:sb[6], eb[6]+1:eb[7]])
            ∂eeh_buf .= 0
            fill_kron_adjoint!(∂eeh_buf, ∂s₁_l, ∂b56, eeh, s₁)
            ∂ee₂_l .+= ∂eeh_buf ./ 2
            ∂k56b = Matrix(d.s_s') * ∂b56
            fill_kron_adjoint!(∂e₁_l, ∂se₂_l, ∂k56b, e₁, se₂)
            # (5,7) kron(e₁, ee₂/2)
            ∂eeh_buf .= 0
            fill_kron_adjoint!(∂eeh_buf, ∂e₁_l, Matrix(∂ê₃[sb[5]+1:sb[6], eb[7]+1:eb[8]]), eeh, e₁)
            ∂ee₂_l .+= ∂eeh_buf ./ 2
            # Row 6: (6,5) L₃ˢ * (kron(s₁²,e₁) + kron(s₁,s_s*s₁e₁) + kron(e₁,s₁²)*e_ss) — decompress rows
            ∂b65 = Matrix(d.L₃ˢ)' * Matrix(∂ê₃[sb[6]+1:sb[7], eb[5]+1:eb[6]])
            ∂s₁²_buf .= 0                                            # Term 1: kron(s₁², e₁)
            fill_kron_adjoint!(∂e₁_l, ∂s₁²_buf, ∂b65, e₁, s₁²)
            fill_kron_adjoint!(∂s₁_l, ∂s₁_l, ∂s₁²_buf, s₁, s₁)
            ∂kron_buf .= 0                                            # Term 2: kron(s₁, ss_s1e1)
            fill_kron_adjoint!(∂kron_buf, ∂s₁_l, ∂b65, ss_s1e1, s₁)
            tmpC = Matrix(d.s_s') * ∂kron_buf
            fill_kron_adjoint!(∂e₁_l, ∂s₁_l, tmpC, e₁, s₁)
            ∂k65c = ∂b65 * Matrix(d.e_ss')                           # Term 3: kron(e₁, s₁²) * e_ss
            ∂s₁²_buf .= 0
            fill_kron_adjoint!(∂s₁²_buf, ∂e₁_l, ∂k65c, s₁², e₁)
            fill_kron_adjoint!(∂s₁_l, ∂s₁_l, ∂s₁²_buf, s₁, s₁)
            # (6,6) L₃ˢ * (kron(s₁e₁,e₁) + kron(e₁,s₁e₁)*e_es + kron(e₁,s_s*s₁e₁)*e_es) — decompress rows
            ∂b66 = Matrix(d.L₃ˢ)' * Matrix(∂ê₃[sb[6]+1:sb[7], eb[6]+1:eb[7]])
            ∂kron_buf .= 0                                            # Term 1: kron(s₁e₁, e₁)
            fill_kron_adjoint!(∂e₁_l, ∂kron_buf, ∂b66, e₁, s₁e₁)
            fill_kron_adjoint!(∂e₁_l, ∂s₁_l, ∂kron_buf, e₁, s₁)
            ∂pre = ∂b66 * Matrix(d.e_es')                            # shared for Terms 2+3
            ∂kron_buf .= 0                                            # Term 2: kron(e₁, s₁e₁)
            fill_kron_adjoint!(∂kron_buf, ∂e₁_l, ∂pre, s₁e₁, e₁)
            fill_kron_adjoint!(∂e₁_l, ∂s₁_l, ∂kron_buf, e₁, s₁)
            ∂kron_buf .= 0                                            # Term 3: kron(e₁, ss_s1e1)
            fill_kron_adjoint!(∂kron_buf, ∂e₁_l, ∂pre, ss_s1e1, e₁)
            tmpC = Matrix(d.s_s') * ∂kron_buf
            fill_kron_adjoint!(∂e₁_l, ∂s₁_l, tmpC, e₁, s₁)
            # (6,7) L₃ˢ * kron(e₁, e₁²) — decompress rows
            ∂e₁²_buf .= 0
            fill_kron_adjoint!(∂e₁²_buf, ∂e₁_l, Matrix(d.L₃ˢ)' * Matrix(∂ê₃[sb[6]+1:sb[7], eb[7]+1:eb[8]]), e₁², e₁)
            fill_kron_adjoint!(∂e₁_l, ∂e₁_l, ∂e₁²_buf, e₁, e₁)

            # ── 3a: Γ₃ disaggregation → ∂Σ̂ᶻ₁, ∂Σ̂ᶻ₂, ∂Δ̂μˢ₂ ──
            ∂Γ = Matrix{T}(∂Γ₃_iter)
            vΣ = vec(d.Σ̂ᶻ₁)

            # Row 1: (1,4) kron(Δ̂μˢ₂',Ine)
            ∂tmp14 = kron_vjp_helper(∂Γ[gb[1]+1:gb[2], gb[4]+1:gb[5]], reshape(d.Δ̂μˢ₂, 1, :), Ine)[1]
            ∂Δ̂μˢ₂_l .+= vec(∂tmp14')
            # (1,5) kron(vec(Σ̂ᶻ₁)',Ine)
            ∂tmp15 = kron_vjp_helper(∂Γ[gb[1]+1:gb[2], gb[5]+1:gb[6]], reshape(vΣ, 1, :), Ine)[1]
            ∂Σ̂ᶻ₁ .+= reshape(vec(∂tmp15'), n, n)
            # Row 3: (3,3) kron(Σ̂ᶻ₁,Ine)
            ∂Σ̂ᶻ₁ .+= kron_vjp_helper(∂Γ[gb[3]+1:gb[4], gb[3]+1:gb[4]], Matrix(d.Σ̂ᶻ₁), Ine)[1]
            # Row 4: (4,1) kron(Δ̂μˢ₂,Ine)
            ∂Δ̂μˢ₂_l .+= vec(kron_vjp_helper(∂Γ[gb[4]+1:gb[5], gb[1]+1:gb[2]], reshape(d.Δ̂μˢ₂, :, 1), Ine)[1])
            # (4,4) kron(Σ̂ᶻ₂_22 + Δ*Δ', Ine)
            M44 = d.Σ̂ᶻ₂[n+1:2n, n+1:2n] + d.Δ̂μˢ₂ * d.Δ̂μˢ₂'
            ∂M44 = kron_vjp_helper(∂Γ[gb[4]+1:gb[5], gb[4]+1:gb[5]], Matrix(M44), Ine)[1]
            ∂Σ̂ᶻ₂[n+1:2n, n+1:2n] .+= ∂M44
            ∂Δ̂μˢ₂_l .+= (∂M44 + ∂M44') * d.Δ̂μˢ₂
            # (4,5) kron(Σ̂ᶻ₂_23 + Δ*vΣ', Ine)
            M45 = d.Σ̂ᶻ₂[n+1:2n, 2n+1:end] + d.Δ̂μˢ₂ * vΣ'
            ∂M45 = kron_vjp_helper(∂Γ[gb[4]+1:gb[5], gb[5]+1:gb[6]], Matrix(M45), Ine)[1]
            ∂Σ̂ᶻ₂[n+1:2n, 2n+1:end] .+= ∂M45
            ∂Δ̂μˢ₂_l .+= ∂M45 * vΣ
            ∂Σ̂ᶻ₁ .+= reshape(∂M45' * d.Δ̂μˢ₂, n, n)
            # (4,7) kron(Δ̂μˢ₂, e4_nᵉ_nᵉ³)
            ∂Δ̂μˢ₂_l .+= vec(kron_vjp_helper(∂Γ[gb[4]+1:gb[5], gb[7]+1:gb[8]], reshape(d.Δ̂μˢ₂, :, 1), Matrix(e4_nᵉ_nᵉ³))[1])
            # Row 5: (5,1) kron(vΣ, Ine)
            ∂Σ̂ᶻ₁ .+= reshape(kron_vjp_helper(∂Γ[gb[5]+1:gb[6], gb[1]+1:gb[2]], reshape(vΣ, :, 1), Ine)[1], n, n)
            # (5,4) kron(Σ̂ᶻ₂_32 + vΣ*Δ', Ine)
            M54 = d.Σ̂ᶻ₂[2n+1:end, n+1:2n] + vΣ * d.Δ̂μˢ₂'
            ∂M54 = kron_vjp_helper(∂Γ[gb[5]+1:gb[6], gb[4]+1:gb[5]], Matrix(M54), Ine)[1]
            ∂Σ̂ᶻ₂[2n+1:end, n+1:2n] .+= ∂M54
            ∂Σ̂ᶻ₁ .+= reshape(∂M54 * d.Δ̂μˢ₂, n, n)
            ∂Δ̂μˢ₂_l .+= ∂M54' * vΣ
            # (5,5) kron(Σ̂ᶻ₂_33 + vΣ*vΣ', Ine)
            M55 = d.Σ̂ᶻ₂[2n+1:end, 2n+1:end] + vΣ * vΣ'
            ∂M55 = kron_vjp_helper(∂Γ[gb[5]+1:gb[6], gb[5]+1:gb[6]], Matrix(M55), Ine)[1]
            ∂Σ̂ᶻ₂[2n+1:end, 2n+1:end] .+= ∂M55
            ∂Σ̂ᶻ₁ .+= reshape((∂M55 + ∂M55') * vΣ, n, n)
            # (5,7) kron(vΣ, e4_nᵉ_nᵉ³)
            ∂Σ̂ᶻ₁ .+= reshape(kron_vjp_helper(∂Γ[gb[5]+1:gb[6], gb[7]+1:gb[8]], reshape(vΣ, :, 1), Matrix(e4_nᵉ_nᵉ³))[1], n, n)
            # Row 6: (6,6) kron(Σ̂ᶻ₁, e4_nᵉ²_nᵉ²)
            ∂Σ̂ᶻ₁ .+= kron_vjp_helper(∂Γ[gb[6]+1:gb[7], gb[6]+1:gb[7]], Matrix(d.Σ̂ᶻ₁), Matrix(e4_nᵉ²_nᵉ²))[1]
            # Row 7: (7,4) kron(Δ̂μˢ₂', e4')
            ∂tmp74 = kron_vjp_helper(∂Γ[gb[7]+1:gb[8], gb[4]+1:gb[5]], reshape(d.Δ̂μˢ₂, 1, :), Matrix(e4_nᵉ_nᵉ³'))[1]
            ∂Δ̂μˢ₂_l .+= vec(∂tmp74')
            # (7,5) kron(vΣ', e4')
            ∂tmp75 = kron_vjp_helper(∂Γ[gb[7]+1:gb[8], gb[5]+1:gb[6]], reshape(vΣ, 1, :), Matrix(e4_nᵉ_nᵉ³'))[1]
            ∂Σ̂ᶻ₁ .+= reshape(vec(∂tmp75'), n, n)

            # ── 3b: Eᴸᶻ disaggregation ──
            ∂EL = Matrix{T}(∂Eᴸᶻ_iter)
            # Only row block 6 is data-dependent
            ∂EL6 = ∂EL[gb[6]+1:gb[7], :]
            # Col 1: kron(Σ̂ᶻ₁, vec_Ie)
            ∂Σ̂ᶻ₁ .+= kron_vjp_helper(∂EL6[:, sb[1]+1:sb[2]], Matrix(d.Σ̂ᶻ₁), vec_Ie_col)[1]
            # Col 4: kron(μˢ₃δμˢ₁', vec_Ie)
            ∂μ_T = kron_vjp_helper(∂EL6[:, sb[4]+1:sb[5]], Matrix(d.μˢ₃δμˢ₁'), vec_Ie_col)[1]
            ∂μˢ₃δμˢ₁ = Matrix(∂μ_T')   # n×n
            # Col 5: kron(C₄, vec_Ie)
            inner_C4 = d.Σ̂ᶻ₂[n+1:2n, 2n+1:end] + d.Δ̂μˢ₂ * vΣ'
            ss_s_M = Matrix(d.ss_s)
            C4m = reshape(ss_s_M * vec(inner_C4), n, n^2)
            ∂C4 = kron_vjp_helper(∂EL6[:, sb[5]+1:sb[6]], C4m, vec_Ie_col)[1]
            ∂iC4 = reshape(ss_s_M' * vec(∂C4), n, n^2)
            ∂Σ̂ᶻ₂[n+1:2n, 2n+1:end] .+= ∂iC4
            ∂Δ̂μˢ₂_l .+= ∂iC4 * vΣ
            ∂Σ̂ᶻ₁ .+= reshape(∂iC4' * d.Δ̂μˢ₂, n, n)
            # Col 6: kron(C₅ * L₃ˢ', vec_Ie) — compress C₅ cols
            inner_C5 = d.Σ̂ᶻ₂[2n+1:end, 2n+1:end] + vΣ * vΣ'
            C5m = reshape(Matrix(inner_C5), n, n^3)
            C5m_c = C5m * Matrix(d.L₃ˢ)'
            ∂C5_c = kron_vjp_helper(∂EL6[:, sb[6]+1:sb[7]], C5m_c, vec_Ie_col)[1]
            ∂C5 = ∂C5_c * Matrix(d.L₃ˢ)
            ∂iC5 = reshape(∂C5, n^2, n^2)
            ∂Σ̂ᶻ₂[2n+1:end, 2n+1:end] .+= ∂iC5
            ∂Σ̂ᶻ₁ .+= reshape((∂iC5 + ∂iC5') * vΣ, n, n)

            # ── 3c: μˢ₃δμˢ₁ adjoint ──
            # μˢ₃δμˢ₁ = reshape((I - s₁²) \ vec(RHS), n, n)
            ∂x_μ = vec(∂μˢ₃δμˢ₁)
            I_m_s₁² = Matrix{T}(ℒ.I(n^2)) - s₁²
            ∂b_μ = I_m_s₁²' \ ∂x_μ
            # ∂(kron(s₁,s₁)) = ∂b * vec(μ)'
            ∂s₁²_from_μ = ∂b_μ * vec(d.μˢ₃δμˢ₁)'
            fill_kron_adjoint!(∂s₁_l, ∂s₁_l, ∂s₁²_from_μ, s₁, s₁)

            # Decompose ∂RHS:  RHS = L₁ * s₁' + L₂ * e₁'
            ∂RHS = reshape(∂b_μ, n, n)

            # Reconstruct L₁ and L₂
            inner_M1 = d.Σ̂ᶻ₂[2n+1:end, n+1:2n] + vΣ * d.Δ̂μˢ₂'
            M1 = reshape(ss_s_M * vec(inner_M1), n^2, n)
            inner_M2 = d.Σ̂ᶻ₂[2n+1:end, 2n+1:end] + vΣ * vΣ'
            M2 = reshape(Matrix(inner_M2), n^3, n)
            M3 = ℒ.kron(Matrix(d.Σ̂ᶻ₁), vec_Ie_col)

            L₁ = ss₂ * M1 + Matrix(d.s_s_s_to_s₃) * M2 / 6 +
                 Matrix(d.s_e_e_to_s₃) * M3 / 2 + Matrix(d.s_v_v_to_s₃) * Matrix(d.Σ̂ᶻ₁) / 2

            M4 = ℒ.kron(reshape(d.Δ̂μˢ₂, :, 1), Ine)
            M5 = Matrix(e4_nᵉ_nᵉ³')
            M6 = ℒ.kron(reshape(vΣ, :, 1), Ine)

            L₂ = se₂ * M4 + Matrix(d.e_e_e_to_s₃) * M5 / 6 +
                 Matrix(d.s_s_e_to_s₃) * M6 / 2 + Matrix(d.e_v_v_to_s₃) * Ine / 2

            ∂L₁ = ∂RHS * s₁;    ∂s₁_l .+= ∂RHS' * L₁
            ∂L₂ = ∂RHS * e₁;    ∂e₁_l .+= ∂RHS' * L₂

            # Decompose ∂L₁
            ∂ss₂_l .+= ∂L₁ * M1'
            ∂M1_raw = ss₂' * ∂L₁
            ∂S3f_acc[d.iˢ, d.kron_s_s_s] .+= ∂L₁ * M2' ./ 6
            ∂M2_raw = Matrix(d.s_s_s_to_s₃)' * ∂L₁ ./ 6
            ∂S3f_acc[d.iˢ, d.kron_s_e_e] .+= ∂L₁ * M3' ./ 2
            ∂M3_raw = Matrix(d.s_e_e_to_s₃)' * ∂L₁ ./ 2
            ∂S3f_acc[d.iˢ, d.kron_s_v_v] .+= ∂L₁ * Matrix(d.Σ̂ᶻ₁)' ./ 2
            ∂Σ̂ᶻ₁ .+= Matrix(d.s_v_v_to_s₃)' * ∂L₁ ./ 2

            # Decompose ∂L₂
            ∂se₂_l .+= ∂L₂ * M4'
            ∂M4_raw = se₂' * ∂L₂
            ∂S3f_acc[d.iˢ, d.kron_e_e_e] .+= ∂L₂ * M5' ./ 6
            ∂S3f_acc[d.iˢ, d.kron_s_s_e] .+= ∂L₂ * M6' ./ 2
            ∂M6_raw = Matrix(d.s_s_e_to_s₃)' * ∂L₂ ./ 2
            ∂S3f_acc[d.iˢ, d.kron_e_v_v] .+= ∂L₂ ./ 2

            # Decompose ∂M1 → ∂Σ̂ᶻ₂, ∂Σ̂ᶻ₁, ∂Δ̂μˢ₂
            ∂iM1 = reshape(ss_s_M' * vec(∂M1_raw), n^2, n)
            ∂Σ̂ᶻ₂[2n+1:end, n+1:2n] .+= ∂iM1
            ∂Σ̂ᶻ₁ .+= reshape(∂iM1 * d.Δ̂μˢ₂, n, n)
            ∂Δ̂μˢ₂_l .+= ∂iM1' * vΣ
            # Decompose ∂M2 → ∂Σ̂ᶻ₂, ∂Σ̂ᶻ₁
            ∂iM2 = reshape(∂M2_raw, n^2, n^2)
            ∂Σ̂ᶻ₂[2n+1:end, 2n+1:end] .+= ∂iM2
            ∂Σ̂ᶻ₁ .+= reshape((∂iM2 + ∂iM2') * vΣ, n, n)
            # Decompose ∂M3 → ∂Σ̂ᶻ₁
            ∂Σ̂ᶻ₁ .+= kron_vjp_helper(∂M3_raw, Matrix(d.Σ̂ᶻ₁), vec_Ie_col)[1]
            # Decompose ∂M4 → ∂Δ̂μˢ₂
            ∂Δ̂μˢ₂_l .+= vec(kron_vjp_helper(∂M4_raw, reshape(d.Δ̂μˢ₂, :, 1), Ine)[1])
            # Decompose ∂M6 → ∂Σ̂ᶻ₁
            ∂Σ̂ᶻ₁ .+= reshape(kron_vjp_helper(∂M6_raw, reshape(vΣ, :, 1), Ine)[1], n, n)

            # ── 4: Scatter local cotangents to global accumulators ──
            ∂𝐒₁_acc[d.iˢ, d.dependencies_in_states_idx] .+= ∂s₁_l
            ∂𝐒₁_acc[d.iˢ, n₋+1:size(∂𝐒₁_acc, 2)] .+= ∂e₁_l
            ∂S2f_acc[d.iˢ, d.kron_s_s]  .+= ∂ss₂_l
            ∂S2f_acc[d.iˢ, kron_e_e]    .+= ∂ee₂_l
            ∂S2f_acc[d.iˢ, d.kron_s_e]  .+= ∂se₂_l
            ∂S2f_acc[d.iˢ, kron_v_v]    .+= ∂vv₂_l
            ∂Σʸ₁_acc[d.iˢ, d.iˢ]       .+= ∂Σ̂ᶻ₁
            ∂Σᶻ₂_acc[d.dependencies_extended_idx, d.dependencies_extended_idx] .+= ∂Σ̂ᶻ₂
            ∂Δμˢ₂_acc[d.dependencies_in_states_idx] .+= ∂Δ̂μˢ₂_l
        end

        # ── Sub-rrule pullback chain ──

        # S₃_full = S₃ * 𝐔₃  →  ∂S₃ = ∂S₃_full * 𝐔₃'
        ∂𝐒₃_compressed = ∂S3f_acc * 𝐔₃'

        # Third-order solution pullback: returns (NoTangent, ∂∇₁, ∂∇₂, ∂∇₃, ∂𝑺₁, ∂𝐒₂, NT, NT, NT)
        so3_grad = so3_pb((∂𝐒₃_compressed, NoTangent()))
        if !(so3_grad[2] isa AbstractZero); ∂∇₁_acc .+= so3_grad[2]; end
        if !(so3_grad[3] isa AbstractZero); ∂∇₂_acc .+= so3_grad[3]; end
        if !(so3_grad[4] isa AbstractZero); ∂∇₃_acc .+= so3_grad[4]; end
        if !(so3_grad[5] isa AbstractZero); ∂𝐒₁_acc .+= so3_grad[5]; end
        # so3_grad[6] is now compressed ∂𝐒₂_raw — kept separate

        # Third-order derivatives pullback: returns (NoTangent, ∂params, ∂SS, NT, NT)
        ∇₃_grad = ∇₃_pb(∂∇₃_acc)
        ∂params_∇₃  = ∇₃_grad[2] isa AbstractZero ? zeros(T, np) : ∇₃_grad[2]
        if !(∇₃_grad[3] isa AbstractZero); ∂SS_acc .+= ∇₃_grad[3]; end

        # Convert full-space ∂S2f_acc to compressed and add compressed so3 gradient
        ∂S2_raw_acc = ∂S2f_acc * 𝐔₂'
        if !(so3_grad[6] isa AbstractZero); ∂S2_raw_acc .+= so3_grad[6]; end

        # Second-order moments pullback: cotangent tuple for 15-element output
        # (Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr, ŝŝ₂, ŝy₂, Σʸ₁, Σᶻ₁, SS, 𝐒₁, ∇₁, 𝐒₂, ∇₂, slvd)
        ∂som2 = (
            NoTangent(),             # ∂Σʸ₂ (not used by third-order)
            ∂Σᶻ₂_acc,               # ∂Σᶻ₂
            ∂μʸ₂_in isa AbstractZero ? NoTangent() : ∂μʸ₂_in,  # ∂μʸ₂
            ∂Δμˢ₂_acc,              # ∂Δμˢ₂
            NoTangent(),             # ∂autocorr (not used)
            NoTangent(),             # ∂ŝ_to_ŝ₂ (not used)
            NoTangent(),             # ∂ŝ_to_y₂ (not used)
            ∂Σʸ₁_acc,               # ∂Σʸ₁
            NoTangent(),             # ∂Σᶻ₁
            ∂SS_acc,                 # ∂SS_and_pars
            ∂𝐒₁_acc,                # ∂𝐒₁
            ∂∇₁_acc,                # ∂∇₁
            ∂S2_raw_acc,             # ∂𝐒₂ (compressed)
            ∂∇₂_acc,                # ∂∇₂
            NoTangent(),             # ∂slvd
        )

        som2_grad = som2_pb(∂som2)
        ∂params_som2 = som2_grad[2] isa AbstractZero ? zeros(T, np) : som2_grad[2]

        ∂parameters_total = ∂params_som2 .+ ∂params_∇₃

        return NoTangent(), ∂parameters_total, NoTangent(), NoTangent()
    end

    return result, calculate_third_order_moments_pullback
end

# ── calculate_third_order_moments_with_autocorrelation rrule ───────────────────
function rrule(::typeof(calculate_third_order_moments_with_autocorrelation),
                parameters::Vector{T},
                observables::Union{Symbol_input,String_input},
                𝓂::ℳ;
                autocorrelation_periods::U = 1:5,
                covariance::Union{Symbol_input,String_input} = Symbol[],
                opts::CalculationOptions = merge_calculation_options()) where {U, T <: Real}

    # ── Non-differentiable constants ──
    ensure_moments_constants!(𝓂.constants)
    so = 𝓂.constants.second_order
    to = 𝓂.constants.third_order
    T_pm = 𝓂.constants.post_model_macro
    np = length(parameters)
    nᵉ = T_pm.nExo
    n_ac = length(autocorrelation_periods)

    zero_5() = (zeros(T,0,0), zeros(T,0), zeros(T,0,0), zeros(T,0), false)
    zero_pb(_) = (NoTangent(), zeros(T, np), NoTangent(), NoTangent())

    # ── Step 1: Second-order moments with covariance ──
    som2_out, som2_pb = rrule(calculate_second_order_moments_with_covariance, parameters, 𝓂; opts = opts)
    Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr_tmp_2, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂_raw, ∇₂, solved = som2_out

    if !solved; return zero_5(), zero_pb; end

    # Expand compressed 𝐒₂_raw to full for moments computation
    𝐔₂ = 𝓂.constants.second_order.𝐔₂
    𝐒₂ = (sparse(𝐒₂_raw) * 𝐔₂)::SparseMatrixCSC{T, Int}  # was: dense_to_sparse

    # ── Step 2: Third-order derivatives ──
    ∇₃, ∇₃_pb = rrule(calculate_third_order_derivatives, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives, 𝓂.workspaces)

    # ── Step 3: Third-order solution (pass compressed 𝐒₂_raw) ──
    so3_out, so3_pb = rrule(calculate_third_order_solution, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂_raw,
                            𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                            initial_guess = 𝓂.caches.third_order_solution,
                            opts = opts,
                            parameter_values = parameters)
    𝐒₃, solved3 = so3_out

    update_perturbation_counter!(𝓂.counters, solved3, order = 3)

    if !solved3; return zero_5(), zero_pb; end

    # ── Step 4: Decompress S₃ ──
    𝐔₃ = 𝓂.constants.third_order.𝐔₃
    𝐒₃_full = 𝐒₃ * 𝐔₃

    𝐒₃_full = sparse(𝐒₃_full)

    # ── Step 5: Determine iteration groups ──
    orders = determine_efficient_order(𝐒₁, 𝐒₂, 𝐒₃_full, 𝓂.constants, observables,
                                       covariance = covariance, tol = opts.tol.third_order.dependencies_tol)

    kron_e_e = so.kron_e_e
    kron_v_v = so.kron_v_v
    kron_e_v = to.kron_e_v
    e_in_s⁺ = so.e_in_s⁺
    v_in_s⁺ = so.v_in_s⁺
    vec_Iₑ = so.vec_Iₑ
    e4_nᵉ²_nᵉ² = so.e4_nᵉ²_nᵉ²
    e4_nᵉ_nᵉ³ = so.e4_nᵉ_nᵉ³
    e4_minus_vecIₑ_outer = so.e4_minus_vecIₑ_outer
    e6_nᵉ³_nᵉ³ = to.e6_nᵉ³_nᵉ³

    Σʸ₃ = zeros(T, size(Σʸ₂))
    autocorr = zeros(T, size(Σʸ₂, 1), n_ac)
    solved_lyapunov = true

    # Per-iteration storage for pullback
    n_iters = length(orders)
    iter_data = Vector{Any}(undef, n_iters)

    for (iter_idx, ords) in enumerate(orders)
        variance_observable, dependencies_all_vars = ords

        sort!(variance_observable)
        sort!(dependencies_all_vars)

        dependencies = intersect(T_pm.past_not_future_and_mixed, dependencies_all_vars)

        obs_in_y = indexin(variance_observable, T_pm.var)

        dependencies_in_states_idx = indexin(dependencies, T_pm.past_not_future_and_mixed)

        dependencies_in_var_idx = Int.(indexin(dependencies, T_pm.var))

        nˢ = length(dependencies)

        iˢ = dependencies_in_var_idx

        Σ̂ᶻ₁ = Σʸ₁[iˢ, iˢ]

        dependencies_extended_idx = vcat(dependencies_in_states_idx,
                dependencies_in_states_idx .+ T_pm.nPast_not_future_and_mixed,
                findall(ℒ.kron(T_pm.past_not_future_and_mixed .∈ (intersect(T_pm.past_not_future_and_mixed,dependencies),),
                               T_pm.past_not_future_and_mixed .∈ (intersect(T_pm.past_not_future_and_mixed,dependencies),))) .+ 2*T_pm.nPast_not_future_and_mixed)

        Σ̂ᶻ₂ = Σᶻ₂[dependencies_extended_idx, dependencies_extended_idx]

        Δ̂μˢ₂ = Δμˢ₂[dependencies_in_states_idx]

        s_in_s⁺ = BitVector(vcat(T_pm.past_not_future_and_mixed .∈ (dependencies,), zeros(Bool, nᵉ + 1)))

        substate_indices = ensure_moments_substate_indices!(𝓂, nˢ)
        I_plus_s_s = substate_indices.I_plus_s_s
        e_es = substate_indices.e_es
        e_ss = substate_indices.e_ss
        ss_s = substate_indices.ss_s
        s_s = substate_indices.s_s

        # first order slices
        s_to_y₁ = 𝐒₁[obs_in_y,:][:,dependencies_in_states_idx]
        e_to_y₁ = 𝐒₁[obs_in_y,:][:, (T_pm.nPast_not_future_and_mixed + 1):end]

        s_to_s₁ = 𝐒₁[iˢ, dependencies_in_states_idx]
        e_to_s₁ = 𝐒₁[iˢ, (T_pm.nPast_not_future_and_mixed + 1):end]

        # second order slices
        dep_kron = ensure_moments_dependency_kron_indices!(𝓂, dependencies, s_in_s⁺)
        kron_s_s = dep_kron.kron_s_s
        kron_s_e = dep_kron.kron_s_e

        s_s_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_s]
        e_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_e_e]
        s_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_e]

        s_s_to_s₂ = 𝐒₂[iˢ, kron_s_s] |> collect
        e_e_to_s₂ = 𝐒₂[iˢ, kron_e_e]
        v_v_to_s₂ = 𝐒₂[iˢ, kron_v_v] |> collect
        s_e_to_s₂ = 𝐒₂[iˢ, kron_s_e]

        s_to_s₁_by_s_to_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
        e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
        s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)

        # third order slices
        kron_s_v = dep_kron.kron_s_v

        kron_s_s_s = ℒ.kron(kron_s_s, s_in_s⁺)
        kron_s_s_e = ℒ.kron(kron_s_s, e_in_s⁺)
        kron_s_e_e = ℒ.kron(kron_s_e, e_in_s⁺)
        kron_e_e_e = ℒ.kron(kron_e_e, e_in_s⁺)
        kron_s_v_v = ℒ.kron(kron_s_v, v_in_s⁺)
        kron_e_v_v = ℒ.kron(kron_e_v, v_in_s⁺)

        s_s_s_to_y₃ = 𝐒₃_full[obs_in_y,:][:, kron_s_s_s]
        s_s_e_to_y₃ = 𝐒₃_full[obs_in_y,:][:, kron_s_s_e]
        s_e_e_to_y₃ = 𝐒₃_full[obs_in_y,:][:, kron_s_e_e]
        e_e_e_to_y₃ = 𝐒₃_full[obs_in_y,:][:, kron_e_e_e]
        s_v_v_to_y₃ = 𝐒₃_full[obs_in_y,:][:, kron_s_v_v]
        e_v_v_to_y₃ = 𝐒₃_full[obs_in_y,:][:, kron_e_v_v]

        s_s_s_to_s₃ = 𝐒₃_full[iˢ, kron_s_s_s]
        s_s_e_to_s₃ = 𝐒₃_full[iˢ, kron_s_s_e]
        s_e_e_to_s₃ = 𝐒₃_full[iˢ, kron_s_e_e]
        e_e_e_to_s₃ = 𝐒₃_full[iˢ, kron_e_e_e]
        s_v_v_to_s₃ = 𝐒₃_full[iˢ, kron_s_v_v]
        e_v_v_to_s₃ = 𝐒₃_full[iˢ, kron_e_v_v]

        # Set up compression matrices
        D₂ˢ = substate_indices.D₂ˢ
        L₂ˢ = substate_indices.L₂ˢ
        D₃ˢ = substate_indices.D₃ˢ
        L₃ˢ = substate_indices.L₃ˢ
        n₂ˢ = size(D₂ˢ, 2)
        n₃ˢ = size(D₃ˢ, 2)
        s_to_s₁_by_s_to_s₁_c = L₂ˢ * s_to_s₁_by_s_to_s₁ * D₂ˢ

        # Set up pruned state transition sub-blocks (compressed)
        N_upper = 2 * nˢ + n₂ˢ
        N_lower = nˢ + nˢ^2 + n₃ˢ

        A_UU = [s_to_s₁                spzeros(nˢ, nˢ + n₂ˢ)
                spzeros(nˢ, nˢ) s_to_s₁   s_s_to_s₂ / 2 * D₂ˢ
                spzeros(n₂ˢ, 2 * nˢ)               s_to_s₁_by_s_to_s₁_c]

        A_LU = [s_v_v_to_s₃ / 2                    spzeros(nˢ, nˢ + n₂ˢ)
                ℒ.kron(s_to_s₁,v_v_to_s₂ / 2)    spzeros(nˢ^2, nˢ + n₂ˢ)
                spzeros(n₃ˢ, 2 * nˢ + n₂ˢ)]

        A_LL = [s_to_s₁           s_s_to_s₂             s_s_s_to_s₃ / 6 * D₃ˢ
                spzeros(nˢ^2, nˢ) s_to_s₁_by_s_to_s₁  ℒ.kron(s_to_s₁,s_s_to_s₂ / 2) * D₃ˢ
                spzeros(n₃ˢ, nˢ + nˢ^2)               L₃ˢ * ℒ.kron(s_to_s₁,s_to_s₁_by_s_to_s₁) * D₃ˢ]

        ê_to_ŝ₃ = [ e_to_s₁   zeros(nˢ,nᵉ^2 + 2*nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        zeros(nˢ,nᵉ)  e_e_to_s₂ / 2   s_e_to_s₂   zeros(nˢ,nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        zeros(n₂ˢ,nᵉ)  L₂ˢ * e_to_s₁_by_e_to_s₁  L₂ˢ * I_plus_s_s * s_to_s₁_by_e_to_s₁  zeros(n₂ˢ, nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        e_v_v_to_s₃ / 2    zeros(nˢ,nᵉ^2 + nᵉ * nˢ)  s_e_to_s₂    s_s_e_to_s₃ / 2    s_e_e_to_s₃ / 2    e_e_e_to_s₃ / 6
                                        ℒ.kron(e_to_s₁, v_v_to_s₂ / 2)    zeros(nˢ^2, nᵉ^2 + nᵉ * nˢ)      s_s * s_to_s₁_by_e_to_s₁    ℒ.kron(s_to_s₁, s_e_to_s₂) + s_s * ℒ.kron(s_s_to_s₂ / 2, e_to_s₁)  ℒ.kron(s_to_s₁, e_e_to_s₂ / 2) + s_s * ℒ.kron(s_e_to_s₂, e_to_s₁)  ℒ.kron(e_to_s₁, e_e_to_s₂ / 2)
                                        zeros(n₃ˢ, nᵉ + nᵉ^2 + 2*nᵉ * nˢ) L₃ˢ * (ℒ.kron(s_to_s₁_by_s_to_s₁,e_to_s₁) + ℒ.kron(s_to_s₁, s_s * s_to_s₁_by_e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_s_to_s₁) * e_ss)   L₃ˢ * (ℒ.kron(s_to_s₁_by_e_to_s₁,e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_e_to_s₁) * e_es + ℒ.kron(e_to_s₁, s_s * s_to_s₁_by_e_to_s₁) * e_es)  L₃ˢ * ℒ.kron(e_to_s₁,e_to_s₁_by_e_to_s₁)]

        ŝ_to_y₃ = [s_to_y₁ + s_v_v_to_y₃ / 2  s_to_y₁  s_s_to_y₂ / 2 * D₂ˢ   s_to_y₁    s_s_to_y₂     s_s_s_to_y₃ / 6 * D₃ˢ]

        ê_to_y₃ = [e_to_y₁ + e_v_v_to_y₃ / 2  e_e_to_y₂ / 2  s_e_to_y₂   s_e_to_y₂     s_s_e_to_y₃ / 2    s_e_e_to_y₃ / 2    e_e_e_to_y₃ / 6]

        μˢ₃δμˢ₁ = reshape((ℒ.I(size(s_to_s₁_by_s_to_s₁, 1)) - s_to_s₁_by_s_to_s₁) \ vec( 
                                    (s_s_to_s₂  * reshape(ss_s * vec(Σ̂ᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σ̂ᶻ₁) * Δ̂μˢ₂'),nˢ^2, nˢ) +
                                    s_s_s_to_s₃ * reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end , 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ^3, nˢ) / 6 +
                                    s_e_e_to_s₃ * ℒ.kron(Σ̂ᶻ₁, vec_Iₑ) / 2 +
                                    s_v_v_to_s₃ * Σ̂ᶻ₁ / 2) * s_to_s₁' +
                                    (s_e_to_s₂  * ℒ.kron(Δ̂μˢ₂,ℒ.I(nᵉ)) +
                                    e_e_e_to_s₃ * e4_nᵉ_nᵉ³' / 6 +
                                    s_s_e_to_s₃ * ℒ.kron(vec(Σ̂ᶻ₁), ℒ.I(nᵉ)) / 2 +
                                    e_v_v_to_s₃ * ℒ.I(nᵉ) / 2) * e_to_s₁'
                                    ), nˢ, nˢ)

        Γ₃ = [ ℒ.I(nᵉ)             spzeros(nᵉ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δ̂μˢ₂', ℒ.I(nᵉ))  ℒ.kron(vec(Σ̂ᶻ₁)', ℒ.I(nᵉ)) spzeros(nᵉ, nˢ * nᵉ^2)    e4_nᵉ_nᵉ³
                spzeros(nᵉ^2, nᵉ)    e4_minus_vecIₑ_outer     spzeros(nᵉ^2, 2*nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
                spzeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σ̂ᶻ₁, ℒ.I(nᵉ))   spzeros(nˢ * nᵉ, nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
                ℒ.kron(Δ̂μˢ₂,ℒ.I(nᵉ))    spzeros(nᵉ * nˢ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σ̂ᶻ₂[nˢ + 1:2*nˢ,nˢ + 1:2*nˢ] + Δ̂μˢ₂ * Δ̂μˢ₂',ℒ.I(nᵉ)) ℒ.kron(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)',ℒ.I(nᵉ))   spzeros(nᵉ * nˢ, nˢ * nᵉ^2) ℒ.kron(Δ̂μˢ₂, e4_nᵉ_nᵉ³)
                ℒ.kron(vec(Σ̂ᶻ₁), ℒ.I(nᵉ))  spzeros(nᵉ * nˢ^2, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σ̂ᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σ̂ᶻ₁) * Δ̂μˢ₂', ℒ.I(nᵉ))  ℒ.kron(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', ℒ.I(nᵉ))   spzeros(nᵉ * nˢ^2, nˢ * nᵉ^2)  ℒ.kron(vec(Σ̂ᶻ₁), e4_nᵉ_nᵉ³)
                spzeros(nˢ*nᵉ^2, nᵉ + nᵉ^2 + 2*nᵉ * nˢ + nˢ^2*nᵉ)   ℒ.kron(Σ̂ᶻ₁, e4_nᵉ²_nᵉ²)    spzeros(nˢ*nᵉ^2,nᵉ^3)
                e4_nᵉ_nᵉ³'  spzeros(nᵉ^3, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δ̂μˢ₂', e4_nᵉ_nᵉ³')     ℒ.kron(vec(Σ̂ᶻ₁)', e4_nᵉ_nᵉ³')  spzeros(nᵉ^3, nˢ*nᵉ^2)     e6_nᵉ³_nᵉ³]


        Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + n₂ˢ + nˢ^2 + n₃ˢ)
                ℒ.kron(Σ̂ᶻ₁,vec_Iₑ)   zeros(nˢ*nᵉ^2, nˢ + n₂ˢ)  ℒ.kron(μˢ₃δμˢ₁',vec_Iₑ)    ℒ.kron(reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec_Iₑ)  ℒ.kron(reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3) * L₃ˢ', vec_Iₑ)
                spzeros(nᵉ^3, 3*nˢ + n₂ˢ + nˢ^2 + n₃ˢ)]

        droptol!(A_UU, eps())
        droptol!(A_LU, eps())
        droptol!(A_LL, eps())
        droptol!(ê_to_ŝ₃, eps())
        droptol!(Eᴸᶻ, eps())
        droptol!(Γ₃, eps())

        # ── Standard Lyapunov solve (compressed) ──
        N_total = N_upper + N_lower
        ŝ_to_ŝ₃ = [A_UU spzeros(N_upper, N_lower); A_LU A_LL]
        A_cross = Matrix{Float64}(ê_to_ŝ₃ * Eᴸᶻ) * ŝ_to_ŝ₃'
        C_dense = Matrix{Float64}(sparse_ABAt(ê_to_ŝ₃, Γ₃)) + A_cross + A_cross'

        lyap_ws_3rd = Lyapunov_workspace(N_total)
        lyap_out, lyap_pb_iter = rrule(solve_lyapunov_equation,
                                       ŝ_to_ŝ₃, C_dense, lyap_ws_3rd,
                                       lyapunov_algorithm = opts.lyapunov_algorithm,
                                       tol = opts.tol.third_order.ad.lyapunov,
                                       verbose = opts.verbose)
        Σᶻ₃ = lyap_out[1]
        info = lyap_out[2]

        if !info
            return zero_5(), zero_pb
        end

        solved_lyapunov = solved_lyapunov && info

        Σʸ₃tmp = ŝ_to_y₃ * Σᶻ₃ * ŝ_to_y₃' + sparse_ABAt(ê_to_y₃, Γ₃) + ê_to_y₃ * Eᴸᶻ * ŝ_to_y₃' + ŝ_to_y₃ * Eᴸᶻ' * ê_to_y₃'

        for obs in variance_observable
            Σʸ₃[indexin([obs], T_pm.var), indexin(variance_observable, T_pm.var)] = Σʸ₃tmp[indexin([obs], variance_observable), :]
        end

        # ── Autocorrelation forward pass ──
        Eᴸᶻ_orig = Eᴸᶻ   # save original for pullback

        autocorr_tmp_ac = ŝ_to_ŝ₃ * Eᴸᶻ' * ê_to_y₃' + ê_to_ŝ₃ * Γ₃ * ê_to_y₃'

        s_to_s₁ⁱ = Matrix{T}(ℒ.I(nˢ))
        ŝ_to_ŝ₃ⁱ = Matrix{T}(ℒ.I(size(Σᶻ₃, 1)))
        Σᶻ₃ⁱ = copy(Matrix{T}(Σᶻ₃))

        norm_diag = max.(ℒ.diag(Σʸ₃tmp), eps(Float64))

        per_period = Vector{Any}(undef, n_ac)
        Eᴸᶻ_cur = Eᴸᶻ_orig   # tracks current Eᴸᶻ for step 1

        for (pi, i) in enumerate(autocorrelation_periods)
            # Snapshot before step 1
            Σᶻ₃ⁱ_prev = copy(Σᶻ₃ⁱ)
            Eᴸᶻ_used = Eᴸᶻ_cur  # Eᴸᶻ used in step 1

            # Step 1: Σᶻ₃ⁱ update
            Σᶻ₃ⁱ .= Matrix(ŝ_to_ŝ₃) * Σᶻ₃ⁱ + Matrix(ê_to_ŝ₃) * Matrix(Eᴸᶻ_cur)

            # Step 2: s_to_s₁ⁱ update (snapshot before)
            s_to_s₁ⁱ_prev = copy(s_to_s₁ⁱ)
            s_to_s₁ⁱ = s_to_s₁ⁱ * Matrix{T}(s_to_s₁)

            # Step 3: rebuild Eᴸᶻ with s_to_s₁ⁱ prefix
            Eᴸᶻⁱ = [ spzeros(T, nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + n₂ˢ + nˢ^2 + n₃ˢ)
                ℒ.kron(s_to_s₁ⁱ * Σ̂ᶻ₁, vec_Iₑ)   zeros(T, nˢ*nᵉ^2, nˢ + n₂ˢ)  ℒ.kron(s_to_s₁ⁱ * μˢ₃δμˢ₁', vec_Iₑ)    ℒ.kron(s_to_s₁ⁱ * reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ, 2*nˢ + 1:end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec_Iₑ)  ℒ.kron(s_to_s₁ⁱ * reshape(Σ̂ᶻ₂[2*nˢ + 1:end, 2*nˢ + 1:end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3) * L₃ˢ', vec_Iₑ)
                spzeros(T, nᵉ^3, 3*nˢ + n₂ˢ + nˢ^2 + n₃ˢ)]
            Eᴸᶻ_cur = Eᴸᶻⁱ

            # Step 4: compute autocorrelation
            ŝ_to_ŝ₃ⁱ_snap = copy(ŝ_to_ŝ₃ⁱ)  # snapshot before step 5
            num_mat = Matrix(ŝ_to_y₃) * Σᶻ₃ⁱ * Matrix(ŝ_to_y₃)' + Matrix(ŝ_to_y₃) * ŝ_to_ŝ₃ⁱ * Matrix(autocorr_tmp_ac) + Matrix(ê_to_y₃) * Matrix(Eᴸᶻⁱ) * Matrix(ŝ_to_y₃)'
            num_diag_i = ℒ.diag(num_mat)
            ac_val = num_diag_i ./ norm_diag
            diag_Σ = ℒ.diag(Σʸ₃tmp)
            zero_mask_i = diag_Σ .< opts.tol.third_order.ad.lyapunov.acceptance_tol
            ac_val[zero_mask_i] .= 0

            for obs in variance_observable
                autocorr[indexin([obs], T_pm.var), i] .= ac_val[indexin([obs], variance_observable)]
            end

            per_period[pi] = (
                Σᶻ₃ⁱ_prev = Σᶻ₃ⁱ_prev,
                Eᴸᶻ_used = Eᴸᶻ_used,
                s_to_s₁ⁱ = copy(s_to_s₁ⁱ),       # after step 2
                s_to_s₁ⁱ_prev = s_to_s₁ⁱ_prev,
                Eᴸᶻⁱ = Eᴸᶻⁱ,                      # after step 3
                ŝ_to_ŝ₃ⁱ = ŝ_to_ŝ₃ⁱ_snap,         # before step 5
                Σᶻ₃ⁱ = copy(Σᶻ₃ⁱ),                # after step 1
                num_diag = num_diag_i,
                zero_mask = zero_mask_i,
                period_index = i,
            )

            # Step 5: ŝ_to_ŝ₃ⁱ update
            ŝ_to_ŝ₃ⁱ = ŝ_to_ŝ₃ⁱ * Matrix{T}(ŝ_to_ŝ₃)
        end

        # Store per-iteration data for pullback
        iter_data[iter_idx] = (
            variance_observable = variance_observable,
            obs_in_y = obs_in_y,
            iˢ = iˢ,
            nˢ = nˢ,
            dependencies_in_states_idx = dependencies_in_states_idx,
            dependencies_extended_idx = dependencies_extended_idx,
            Σ̂ᶻ₁ = Σ̂ᶻ₁,
            Σ̂ᶻ₂ = Σ̂ᶻ₂,
            Δ̂μˢ₂ = Δ̂μˢ₂,
            s_in_s⁺ = s_in_s⁺,
            s_to_y₁ = s_to_y₁,
            e_to_y₁ = e_to_y₁,
            s_to_s₁ = s_to_s₁,
            e_to_s₁ = e_to_s₁,
            kron_s_s = kron_s_s,
            kron_s_e = kron_s_e,
            kron_s_v = kron_s_v,
            kron_s_s_s = kron_s_s_s,
            kron_s_s_e = kron_s_s_e,
            kron_s_e_e = kron_s_e_e,
            kron_e_e_e = kron_e_e_e,
            kron_s_v_v = kron_s_v_v,
            kron_e_v_v = kron_e_v_v,
            s_s_to_y₂ = s_s_to_y₂,
            e_e_to_y₂ = e_e_to_y₂,
            s_e_to_y₂ = s_e_to_y₂,
            s_s_to_s₂ = s_s_to_s₂,
            e_e_to_s₂ = e_e_to_s₂,
            v_v_to_s₂ = v_v_to_s₂,
            s_e_to_s₂ = s_e_to_s₂,
            s_to_s₁_by_s_to_s₁ = s_to_s₁_by_s_to_s₁,
            e_to_s₁_by_e_to_s₁ = e_to_s₁_by_e_to_s₁,
            s_to_s₁_by_e_to_s₁ = s_to_s₁_by_e_to_s₁,
            s_s_s_to_y₃ = s_s_s_to_y₃,
            s_s_e_to_y₃ = s_s_e_to_y₃,
            s_e_e_to_y₃ = s_e_e_to_y₃,
            e_e_e_to_y₃ = e_e_e_to_y₃,
            s_v_v_to_y₃ = s_v_v_to_y₃,
            e_v_v_to_y₃ = e_v_v_to_y₃,
            s_s_s_to_s₃ = s_s_s_to_s₃,
            s_s_e_to_s₃ = s_s_e_to_s₃,
            s_e_e_to_s₃ = s_e_e_to_s₃,
            e_e_e_to_s₃ = e_e_e_to_s₃,
            s_v_v_to_s₃ = s_v_v_to_s₃,
            e_v_v_to_s₃ = e_v_v_to_s₃,
            ŝ_to_ŝ₃ = ŝ_to_ŝ₃,
            ê_to_ŝ₃ = ê_to_ŝ₃,
            ŝ_to_y₃ = ŝ_to_y₃,
            ê_to_y₃ = ê_to_y₃,
            Γ₃ = Γ₃,
            Eᴸᶻ = Eᴸᶻ_orig,
            N_upper = N_upper,
            N_lower = N_lower,
            lyap_pb = lyap_pb_iter,
            D₂ˢ = D₂ˢ,
            L₂ˢ = L₂ˢ,
            D₃ˢ = D₃ˢ,
            L₃ˢ = L₃ˢ,
            n₂ˢ = n₂ˢ,
            n₃ˢ = n₃ˢ,
            s_to_s₁_by_s_to_s₁_c = s_to_s₁_by_s_to_s₁_c,
            Σᶻ₃ = Σᶻ₃,
            Σʸ₃tmp = Σʸ₃tmp,
            μˢ₃δμˢ₁ = μˢ₃δμˢ₁,
            I_plus_s_s = I_plus_s_s,
            ss_s = ss_s,
            s_s = s_s,
            e_es = e_es,
            e_ss = e_ss,
            # Autocorrelation-specific
            autocorr_tmp_ac = autocorr_tmp_ac,
            norm_diag = norm_diag,
            per_period = per_period,
        )
    end

    # Cache the 3rd-order covariance for reuse
    all_solved_3rd = solved && solved3 && solved_lyapunov
    if all_solved_3rd
        if size(𝓂.caches.covariance_third_order) != size(Σʸ₃)
            𝓂.caches.covariance_third_order = Matrix{Float64}(undef, size(Σʸ₃)...)
        end
        copyto!(𝓂.caches.covariance_third_order, Σʸ₃)
        𝓂.caches.valid_for.covariance_third_order = Float64.(parameters)
        nVars_rrule = T_pm.nVars
        obs_key_rrule = if observables == :full_covar
            collect(1:nVars_rrule)
        else
            obs_idx = parse_variables_input_to_index(observables, 𝓂.constants) |> sort
            if covariance == Symbol[]
                collect(obs_idx)
            else
                covar_idx = parse_variables_input_to_index(covariance, 𝓂.constants) |> sort
                sort(union(obs_idx, covar_idx))
            end
        end
        𝓂.caches.valid_for.covariance_third_order_obs_key = obs_key_rrule

        # Cache autocorrelation
        if size(𝓂.caches.covariance_third_order_autocorr) != size(autocorr)
            𝓂.caches.covariance_third_order_autocorr = Matrix{Float64}(undef, size(autocorr)...)
        end
        copyto!(𝓂.caches.covariance_third_order_autocorr, autocorr)
        𝓂.caches.valid_for.covariance_third_order_autocorr = Float64.(parameters)
        𝓂.caches.valid_for.covariance_third_order_autocorr_obs_key = obs_key_rrule
        𝓂.caches.valid_for.covariance_third_order_autocorr_periods = collect(Int, autocorrelation_periods)
    end

    result = (Σʸ₃, μʸ₂, autocorr, SS_and_pars, all_solved_3rd)

    # ── Pullback ──
    function calculate_third_order_moments_with_autocorrelation_pullback(∂out)
        ∂Σʸ₃_in, ∂μʸ₂_in, ∂autocorr_in, ∂SS_in, _ = ∂out

        ∂Σʸ₃_in = unthunk(∂Σʸ₃_in)
        ∂μʸ₂_in = unthunk(∂μʸ₂_in)
        ∂autocorr_in = unthunk(∂autocorr_in)
        ∂SS_in  = unthunk(∂SS_in)

        n₋ = T_pm.nPast_not_future_and_mixed

        # Accumulators for cotangents flowing to sub-rrule inputs
        ∂Σʸ₁_acc  = zeros(T, size(Σʸ₁))
        ∂Σᶻ₂_acc  = zeros(T, size(Σᶻ₂))
        ∂Δμˢ₂_acc = zeros(T, length(Δμˢ₂))
        ∂𝐒₁_acc   = zeros(T, size(𝐒₁))
        ∂S2f_acc   = zeros(T, size(𝐒₂))
        ∂S3f_acc   = zeros(T, size(𝐒₃_full))
        ∂SS_acc    = zeros(T, length(SS_and_pars))
        ∂∇₁_acc   = zeros(T, size(∇₁))
        ∂∇₂_acc   = zeros(T, size(∇₂))
        ∂∇₃_acc   = zeros(T, size(∇₃))

        if !(∂SS_in isa AbstractZero); ∂SS_acc .+= ∂SS_in; end

        # ──── Reverse loop over iterations ────
        for iter_idx in n_iters:-1:1
            d = iter_data[iter_idx]
            nˢ_i = d.nˢ

            # ═══════════════════════════════════════════════════════════════════
            # Stage 0: Autocorrelation reverse loop
            # ═══════════════════════════════════════════════════════════════════
            nObs_iter = length(d.variance_observable)

            # Initialize cotangents that accumulate through autocorrelation loop
            ∂ŝ_to_y₃_ac = zeros(T, size(d.ŝ_to_y₃))
            ∂ê_to_y₃_ac = zeros(T, size(d.ê_to_y₃))
            ∂Σᶻ₃ⁱ_co   = zeros(T, size(d.Σᶻ₃))   # cotangent for Σᶻ₃ⁱ state
            ∂ŝ_to_ŝ₃_ac = zeros(T, size(d.ŝ_to_ŝ₃))
            ∂ê_to_ŝ₃_ac = zeros(T, size(d.ê_to_ŝ₃))
            ∂Eᴸᶻ_ac = zeros(T, size(d.Eᴸᶻ))     # cotangent for original Eᴸᶻ
            ∂Γ₃_ac  = zeros(T, size(d.Γ₃))
            ∂autocorr_tmp_co = zeros(T, size(d.autocorr_tmp_ac))
            ∂s₁_ac  = zeros(T, nˢ_i, nˢ_i)        # cotangent for s_to_s₁
            ∂Σʸ₃tmp_ac = zeros(T, nObs_iter, nObs_iter) # cotangent from norm_diag
            ∂ŝ_to_ŝ₃ⁱ_co = zeros(T, size(d.Σᶻ₃))  # cotangent for ŝ_to_ŝ₃ⁱ state
            ∂s_to_s₁ⁱ_co = zeros(T, nˢ_i, nˢ_i)   # cotangent for s_to_s₁ⁱ state
            # Data cotangents from Eᴸᶻⁱ disaggregation
            ∂Σ̂ᶻ₁_ac = zeros(T, nˢ_i, nˢ_i)
            ∂Σ̂ᶻ₂_ac = zeros(T, size(d.Σ̂ᶻ₂))
            ∂Δ̂μˢ₂_ac = zeros(T, nˢ_i)
            ∂μˢ₃δμˢ₁_ac = zeros(T, nˢ_i, nˢ_i)

            ŝ_y = Matrix{T}(d.ŝ_to_y₃)
            ê_y = Matrix{T}(d.ê_to_y₃)
            ŝ_ŝ = Matrix{T}(d.ŝ_to_ŝ₃)
            ê_ŝ = Matrix{T}(d.ê_to_ŝ₃)
            vec_Ie_col = reshape(T.(vec_Iₑ), :, 1)
            ss_s_M = Matrix(d.ss_s)
            vΣ_ac = vec(d.Σ̂ᶻ₁)
            n = nˢ_i; ne = nᵉ
            sb_ac = cumsum([0, n, n, d.n₂ˢ, n, n^2, d.n₃ˢ])
            eb_ac = cumsum([0, ne, ne^2, n*ne, n*ne, n^2*ne, n*ne^2, ne^3])

            # Reverse loop over autocorrelation periods
            for pi in n_ac:-1:1
                pp = d.per_period[pi]

                # ── Step 5 reverse: ŝ_to_ŝ₃ⁱ_after = ŝ_to_ŝ₃ⁱ_before * ŝ_to_ŝ₃ ──
                ∂ŝ_to_ŝ₃_ac .+= pp.ŝ_to_ŝ₃ⁱ' * ∂ŝ_to_ŝ₃ⁱ_co
                ∂ŝ_to_ŝ₃ⁱ_co .= ∂ŝ_to_ŝ₃ⁱ_co * ŝ_ŝ'

                # ── Step 4 reverse: autocorrelation output ──
                # Gather ∂autocorr for this period
                ∂ac = zeros(T, nObs_iter)
                if !(∂autocorr_in isa AbstractZero)
                    for obs in d.variance_observable
                        obs_local = indexin([obs], d.variance_observable)
                        obs_global = indexin([obs], T_pm.var)
                        ∂ac[obs_local] .+= ∂autocorr_in[obs_global, pp.period_index]
                    end
                end

                # Apply zero mask
                ∂ac[pp.zero_mask] .= 0

                if ℒ.norm(∂ac) > eps(T)
                    # Division adjoint: ac = num_diag / norm_diag
                    ∂num_diag = ∂ac ./ d.norm_diag
                    ∂norm_diag_from_ac = -∂ac .* pp.num_diag ./ (d.norm_diag .^ 2)
                    # norm_diag = max.(diag(Σʸ₃tmp), eps()) → adjoint only where diag >= eps
                    norm_mask = ℒ.diag(d.Σʸ₃tmp) .>= eps(Float64)
                    ∂Σʸ₃tmp_ac .+= ℒ.Diagonal(∂norm_diag_from_ac .* norm_mask)

                    # Numerator: N = ŝ_y * Σᶻ₃ⁱ * ŝ_y' + ŝ_y * ŝ_ŝ₃ⁱ * ac_tmp + ê_y * Eᴸᶻⁱ * ŝ_y'
                    # Adjoint of diag extraction: ∂D = Diagonal(∂num_diag)
                    ∂D = ℒ.Diagonal(∂num_diag)

                    Σᶻ₃ⁱ_i = pp.Σᶻ₃ⁱ
                    ŝ_ŝ₃ⁱ_i = pp.ŝ_to_ŝ₃ⁱ
                    ELⁱ = Matrix{T}(pp.Eᴸᶻⁱ)
                    ac_tmp = Matrix{T}(d.autocorr_tmp_ac)

                    # Term 1: diag(ŝ_y * Σᶻ₃ⁱ * ŝ_y')
                    ∂ŝ_to_y₃_ac .+= ∂D * ŝ_y * (Σᶻ₃ⁱ_i + Σᶻ₃ⁱ_i')
                    ∂Σᶻ₃ⁱ_co   .+= ŝ_y' * ∂D * ŝ_y

                    # Term 2: diag(ŝ_y * ŝ_ŝ₃ⁱ * ac_tmp)
                    ∂ŝ_to_y₃_ac   .+= ∂D * ac_tmp' * ŝ_ŝ₃ⁱ_i'
                    ∂ŝ_to_ŝ₃ⁱ_co  .+= ŝ_y' * ∂D * ac_tmp'
                    ∂autocorr_tmp_co .+= ŝ_ŝ₃ⁱ_i' * ŝ_y' * ∂D

                    # Term 3: diag(ê_y * Eᴸᶻⁱ * ŝ_y')
                    ∂ê_to_y₃_ac .+= ∂D * ŝ_y * ELⁱ'
                    ∂ŝ_to_y₃_ac .+= ∂D * ê_y * ELⁱ
                    ∂Eᴸᶻⁱ = ê_y' * ∂D * ŝ_y

                    # ── Eᴸᶻⁱ disaggregation: only row block 6 has s_to_s₁ⁱ prefix ──
                    ∂ELⁱ6 = ∂Eᴸᶻⁱ[eb_ac[6]+1:eb_ac[7], :]

                    s₁ⁱ = pp.s_to_s₁ⁱ  # s₁^i (after step 2)

                    # Col 1: kron(s₁ⁱ * Σ̂ᶻ₁, vec_Ie)
                    A_c1 = s₁ⁱ * Matrix{T}(d.Σ̂ᶻ₁)
                    ∂A_c1 = kron_vjp_helper(∂ELⁱ6[:, sb_ac[1]+1:sb_ac[2]], A_c1, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_c1 * Matrix{T}(d.Σ̂ᶻ₁)'
                    ∂Σ̂ᶻ₁_ac .+= s₁ⁱ' * ∂A_c1

                    # Col 4: kron(s₁ⁱ * μˢ₃δμˢ₁', vec_Ie)
                    A_c4 = s₁ⁱ * Matrix{T}(d.μˢ₃δμˢ₁')
                    ∂A_c4 = kron_vjp_helper(∂ELⁱ6[:, sb_ac[4]+1:sb_ac[5]], A_c4, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_c4 * Matrix{T}(d.μˢ₃δμˢ₁)
                    ∂μˢ₃δμˢ₁_ac .+= ∂A_c4' * s₁ⁱ

                    # Col 5: kron(s₁ⁱ * C4m, vec_Ie)
                    inner_C4 = d.Σ̂ᶻ₂[n+1:2n, 2n+1:end] + d.Δ̂μˢ₂ * vΣ_ac'
                    C4m = reshape(ss_s_M * vec(inner_C4), n, n^2)
                    A_c5 = s₁ⁱ * C4m
                    ∂A_c5 = kron_vjp_helper(∂ELⁱ6[:, sb_ac[5]+1:sb_ac[6]], A_c5, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_c5 * C4m'
                    ∂C4_i = s₁ⁱ' * ∂A_c5
                    ∂iC4_i = reshape(ss_s_M' * vec(∂C4_i), n, n^2)
                    ∂Σ̂ᶻ₂_ac[n+1:2n, 2n+1:end] .+= ∂iC4_i
                    ∂Δ̂μˢ₂_ac .+= ∂iC4_i * vΣ_ac
                    ∂Σ̂ᶻ₁_ac .+= reshape(∂iC4_i' * d.Δ̂μˢ₂, n, n)

                    # Col 6: kron(s₁ⁱ * C5m * L₃ˢ', vec_Ie)
                    inner_C5 = d.Σ̂ᶻ₂[2n+1:end, 2n+1:end] + vΣ_ac * vΣ_ac'
                    C5m = reshape(Matrix{T}(inner_C5), n, n^3)
                    C5m_c = C5m * Matrix(d.L₃ˢ)'
                    A_c6 = s₁ⁱ * C5m_c
                    ∂A_c6 = kron_vjp_helper(∂ELⁱ6[:, sb_ac[6]+1:sb_ac[7]], A_c6, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_c6 * C5m_c'
                    ∂C5m_c_i = s₁ⁱ' * ∂A_c6
                    ∂C5_i = ∂C5m_c_i * Matrix(d.L₃ˢ)
                    ∂iC5_i = reshape(∂C5_i, n^2, n^2)
                    ∂Σ̂ᶻ₂_ac[2n+1:end, 2n+1:end] .+= ∂iC5_i
                    ∂Σ̂ᶻ₁_ac .+= reshape((∂iC5_i + ∂iC5_i') * vΣ_ac, n, n)
                end  # norm(∂ac) check

                # ── Step 2 reverse: s_to_s₁ⁱ_after = s_to_s₁ⁱ_prev * s_to_s₁ ──
                s₁_d = Matrix{T}(d.s_to_s₁)
                ∂s₁_ac .+= pp.s_to_s₁ⁱ_prev' * ∂s_to_s₁ⁱ_co
                ∂s_to_s₁ⁱ_co .= ∂s_to_s₁ⁱ_co * s₁_d'

                # ── Step 1 reverse: Σᶻ₃ⁱ = ŝ_ŝ * Σᶻ₃ⁱ_prev + ê_ŝ * Eᴸᶻ_used ──
                EL_used = Matrix{T}(pp.Eᴸᶻ_used)
                ∂ŝ_to_ŝ₃_ac .+= ∂Σᶻ₃ⁱ_co * pp.Σᶻ₃ⁱ_prev'
                ∂ê_to_ŝ₃_ac .+= ∂Σᶻ₃ⁱ_co * EL_used'
                # ∂Eᴸᶻ_used: this flows to the previous period's Eᴸᶻⁱ or to the original Eᴸᶻ
                ∂Eᴸᶻ_used = ê_ŝ' * ∂Σᶻ₃ⁱ_co
                if pi == 1
                    ∂Eᴸᶻ_ac .+= ∂Eᴸᶻ_used
                else
                    # Flows to previous period's Eᴸᶻⁱ — need to disaggregate
                    # The previous Eᴸᶻⁱ has s_to_s₁ⁱ prefix from period pi-1
                    pp_prev = d.per_period[pi-1]
                    s₁ⁱ_prev = pp_prev.s_to_s₁ⁱ
                    ∂ELprev6 = ∂Eᴸᶻ_used[eb_ac[6]+1:eb_ac[7], :]

                    # Col 1
                    A_pc1 = s₁ⁱ_prev * Matrix{T}(d.Σ̂ᶻ₁)
                    ∂A_pc1 = kron_vjp_helper(∂ELprev6[:, sb_ac[1]+1:sb_ac[2]], A_pc1, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_pc1 * Matrix{T}(d.Σ̂ᶻ₁)'
                    ∂Σ̂ᶻ₁_ac .+= s₁ⁱ_prev' * ∂A_pc1

                    # Col 4
                    A_pc4 = s₁ⁱ_prev * Matrix{T}(d.μˢ₃δμˢ₁')
                    ∂A_pc4 = kron_vjp_helper(∂ELprev6[:, sb_ac[4]+1:sb_ac[5]], A_pc4, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_pc4 * Matrix{T}(d.μˢ₃δμˢ₁)
                    ∂μˢ₃δμˢ₁_ac .+= ∂A_pc4' * s₁ⁱ_prev

                    # Col 5
                    inner_C4p = d.Σ̂ᶻ₂[n+1:2n, 2n+1:end] + d.Δ̂μˢ₂ * vΣ_ac'
                    C4mp = reshape(ss_s_M * vec(inner_C4p), n, n^2)
                    A_pc5 = s₁ⁱ_prev * C4mp
                    ∂A_pc5 = kron_vjp_helper(∂ELprev6[:, sb_ac[5]+1:sb_ac[6]], A_pc5, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_pc5 * C4mp'
                    ∂C4p = s₁ⁱ_prev' * ∂A_pc5
                    ∂iC4p = reshape(ss_s_M' * vec(∂C4p), n, n^2)
                    ∂Σ̂ᶻ₂_ac[n+1:2n, 2n+1:end] .+= ∂iC4p
                    ∂Δ̂μˢ₂_ac .+= ∂iC4p * vΣ_ac
                    ∂Σ̂ᶻ₁_ac .+= reshape(∂iC4p' * d.Δ̂μˢ₂, n, n)

                    # Col 6
                    inner_C5p = d.Σ̂ᶻ₂[2n+1:end, 2n+1:end] + vΣ_ac * vΣ_ac'
                    C5mp = reshape(Matrix{T}(inner_C5p), n, n^3)
                    C5mp_c = C5mp * Matrix(d.L₃ˢ)'
                    A_pc6 = s₁ⁱ_prev * C5mp_c
                    ∂A_pc6 = kron_vjp_helper(∂ELprev6[:, sb_ac[6]+1:sb_ac[7]], A_pc6, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_pc6 * C5mp_c'
                    ∂C5m_c_p = s₁ⁱ_prev' * ∂A_pc6
                    ∂C5p = ∂C5m_c_p * Matrix(d.L₃ˢ)
                    ∂iC5p = reshape(∂C5p, n^2, n^2)
                    ∂Σ̂ᶻ₂_ac[2n+1:end, 2n+1:end] .+= ∂iC5p
                    ∂Σ̂ᶻ₁_ac .+= reshape((∂iC5p + ∂iC5p') * vΣ_ac, n, n)

                    # The remaining rows (1-5 and 7) of ∂Eᴸᶻ_used are zero (spzeros in forward)
                end

                # Propagate ∂Σᶻ₃ⁱ to previous state
                ∂Σᶻ₃ⁱ_co .= ŝ_ŝ' * ∂Σᶻ₃ⁱ_co
            end  # end autocorrelation reverse loop

            # ── autocorr_tmp adjoint ──
            # autocorr_tmp = ŝ_ŝ * Eᴸᶻ' * ê_y' + ê_ŝ * Γ₃ * ê_y'
            ∂act = Matrix{T}(∂autocorr_tmp_co)
            EL_orig = Matrix{T}(d.Eᴸᶻ)
            Γ₃_d = Matrix{T}(d.Γ₃)

            # Term 1: ŝ_ŝ * Eᴸᶻ' * ê_y'
            ∂ŝ_to_ŝ₃_ac .+= ∂act * ê_y * EL_orig
            ∂Eᴸᶻ_ac    .+= ê_y' * ∂act' * ŝ_ŝ
            ∂ê_to_y₃_ac .+= ∂act' * ŝ_ŝ * EL_orig'

            # Term 2: ê_ŝ * Γ₃ * ê_y'
            ∂ê_to_ŝ₃_ac .+= ∂act * ê_y * Γ₃_d'
            ∂Γ₃_ac      .+= ê_ŝ' * ∂act * ê_y
            ∂ê_to_y₃_ac .+= ∂act' * ê_ŝ * Γ₃_d

            # Σᶻ₃ⁱ_co now holds the cotangent at the initial state (Σᶻ₃ⁱ₀ = Σᶻ₃)
            # This adds to ∂Σᶻ₃ from the Lyapunov path

            # ═══════════════════════════════════════════════════════════════════
            # Stage 1: Output mapping (variance) — same as existing rrule
            # ═══════════════════════════════════════════════════════════════════

            # ── Gather ∂Σʸ₃tmp from ∂Σʸ₃ (reverse of scatter) ──
            ∂Σʸ₃tmp = zeros(T, nObs_iter, nObs_iter)

            if !(∂Σʸ₃_in isa AbstractZero)
                ∂Σʸ₃tmp .= ∂Σʸ₃_in[d.obs_in_y, indexin(d.variance_observable, T_pm.var)]
            end

            # Add autocorrelation contribution to ∂Σʸ₃tmp (from norm_diag)
            ∂Σʸ₃tmp .+= ∂Σʸ₃tmp_ac

            if ℒ.norm(∂Σʸ₃tmp) + ℒ.norm(∂ŝ_to_y₃_ac) + ℒ.norm(∂ê_to_y₃_ac) + ℒ.norm(∂Σᶻ₃ⁱ_co) + ℒ.norm(∂ŝ_to_ŝ₃_ac) + ℒ.norm(∂ê_to_ŝ₃_ac) + ℒ.norm(∂Eᴸᶻ_ac) + ℒ.norm(∂Γ₃_ac) < eps(T); continue; end

            ∂Σʸ₃tmp_sym = ∂Σʸ₃tmp + ∂Σʸ₃tmp'

            # ── Σʸ₃tmp = ŝ_y * Σᶻ₃ * ŝ_y' + ê_y * Γ₃ * ê_y' + ê_y * Eᴸᶻ * ŝ_y' + ŝ_y * Eᴸᶻ' * ê_y' ──
            ∂ŝ_to_y₃ = ∂ŝ_to_y₃_ac .+ ∂Σʸ₃tmp_sym * (d.ŝ_to_y₃ * d.Σᶻ₃ + d.ê_to_y₃ * Matrix(d.Eᴸᶻ))
            ∂ê_to_y₃ = ∂ê_to_y₃_ac .+ ∂Σʸ₃tmp_sym * (d.ê_to_y₃ * d.Γ₃  + d.ŝ_to_y₃ * Matrix(d.Eᴸᶻ'))
            ∂Σᶻ₃      = ∂Σᶻ₃ⁱ_co .+ d.ŝ_to_y₃' * ∂Σʸ₃tmp * d.ŝ_to_y₃
            ∂Γ₃_iter   = ∂Γ₃_ac  .+ d.ê_to_y₃' * ∂Σʸ₃tmp * d.ê_to_y₃
            ∂Eᴸᶻ_iter  = ∂Eᴸᶻ_ac .+ d.ê_to_y₃' * ∂Σʸ₃tmp_sym * d.ŝ_to_y₃

            # ── Standard Lyapunov adjoint ──
            Nu = d.N_upper;  Nl = d.N_lower
            ru_i = 1:Nu;  rl_i = (Nu+1):(Nu+Nl)

            lyap_grad = d.lyap_pb((∂Σᶻ₃, NoTangent()))
            ∂ŝ_to_ŝ₃ = lyap_grad[2] isa AbstractZero ? zeros(T, size(d.ŝ_to_ŝ₃)) : Matrix{T}(lyap_grad[2])
            ∂C_lyap   = lyap_grad[3] isa AbstractZero ? zeros(T, size(d.ŝ_to_ŝ₃)) : Matrix{T}(lyap_grad[3])

            # Backprop through C = ê * Γ₃ * ê' + M + M' where M = ê * Eᴸᶻ * ŝ'
            ∂C_sym = ∂C_lyap + ∂C_lyap'
            ê_d = Matrix{T}(d.ê_to_ŝ₃)
            ŝ_d = Matrix{T}(d.ŝ_to_ŝ₃)
            EL_d = Matrix{T}(d.Eᴸᶻ)
            Γ₃_d = Matrix{T}(d.Γ₃)

            # Term 1: ê * Γ₃ * ê'
            ∂Γ₃_iter .+= ê_d' * ∂C_lyap * ê_d
            ∂ê_to_ŝ₃ = ∂ê_to_ŝ₃_ac .+ ∂C_sym * ê_d * Γ₃_d

            # Terms 2+3: M + M' where M = ê * Eᴸᶻ * ŝ'
            ∂ê_to_ŝ₃ .+= ∂C_sym * ŝ_d * EL_d'
            ∂Eᴸᶻ_iter .+= ê_d' * ∂C_sym * ŝ_d
            ∂ŝ_to_ŝ₃ .+= ∂C_sym' * ê_d * EL_d

            # Add autocorrelation contributions
            ∂ŝ_to_ŝ₃ .+= ∂ŝ_to_ŝ₃_ac

            # Extract ∂A_UU, ∂A_LU, ∂A_LL from ∂ŝ_to_ŝ₃
            ∂A_UU = ∂ŝ_to_ŝ₃[ru_i, ru_i]
            ∂A_LU = ∂ŝ_to_ŝ₃[rl_i, ru_i]
            ∂A_LL = ∂ŝ_to_ŝ₃[rl_i, rl_i]

            # ── Disaggregate ŝ_to_y₃ → ∂𝐒₁, ∂𝐒₂, ∂𝐒₃ ──
            n₂ˢ_i = d.n₂ˢ;  n₃ˢ_i = d.n₃ˢ
            c = 0
            ∂blk1 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i];      c += nˢ_i
            ∂blk2 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i];      c += nˢ_i
            ∂blk3 = ∂ŝ_to_y₃[:, c+1:c+n₂ˢ_i];     c += n₂ˢ_i   # compressed
            ∂blk4 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i];      c += nˢ_i
            ∂blk5 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i^2];    c += nˢ_i^2
            ∂blk6 = ∂ŝ_to_y₃[:, c+1:end]

            ∂𝐒₁_acc[d.obs_in_y, d.dependencies_in_states_idx] .+= ∂blk1 .+ ∂blk2 .+ ∂blk4
            ∂S2f_acc[d.obs_in_y, d.kron_s_s]                  .+= (∂blk3 * Matrix(d.D₂ˢ)') ./ 2 .+ ∂blk5  # decompress blk3
            ∂S3f_acc[d.obs_in_y, d.kron_s_v_v]                .+= ∂blk1 ./ 2
            ∂S3f_acc[d.obs_in_y, d.kron_s_s_s]                .+= (∂blk6 * Matrix(d.D₃ˢ)') ./ 6  # decompress blk6

            # ── Disaggregate ê_to_y₃ → ∂𝐒₁, ∂𝐒₂, ∂𝐒₃ ──
            c = 0
            ∂eblk1 = ∂ê_to_y₃[:, c+1:c+nᵉ];          c += nᵉ
            ∂eblk2 = ∂ê_to_y₃[:, c+1:c+nᵉ^2];        c += nᵉ^2
            ∂eblk3 = ∂ê_to_y₃[:, c+1:c+nˢ_i*nᵉ];     c += nˢ_i*nᵉ
            ∂eblk4 = ∂ê_to_y₃[:, c+1:c+nˢ_i*nᵉ];     c += nˢ_i*nᵉ
            ∂eblk5 = ∂ê_to_y₃[:, c+1:c+nˢ_i^2*nᵉ];   c += nˢ_i^2*nᵉ
            ∂eblk6 = ∂ê_to_y₃[:, c+1:c+nˢ_i*nᵉ^2];   c += nˢ_i*nᵉ^2
            ∂eblk7 = ∂ê_to_y₃[:, c+1:end]

            ∂𝐒₁_acc[d.obs_in_y, n₋+1:end]    .+= ∂eblk1
            ∂S2f_acc[d.obs_in_y, kron_e_e]     .+= ∂eblk2 ./ 2
            ∂S2f_acc[d.obs_in_y, d.kron_s_e]   .+= ∂eblk3 .+ ∂eblk4
            ∂S3f_acc[d.obs_in_y, d.kron_e_v_v] .+= ∂eblk1 ./ 2
            ∂S3f_acc[d.obs_in_y, d.kron_s_s_e] .+= ∂eblk5 ./ 2
            ∂S3f_acc[d.obs_in_y, d.kron_s_e_e] .+= ∂eblk6 ./ 2
            ∂S3f_acc[d.obs_in_y, d.kron_e_e_e] .+= ∂eblk7 ./ 6

            # ════════════════════════════════════════════════════════════════════
            # Stage 2+3: Disaggregate block matrices → slice & data cotangents
            # ════════════════════════════════════════════════════════════════════
            Ine = Matrix{T}(ℒ.I(ne))

            # Dense copies of frequently used slices
            s₁  = Matrix{T}(d.s_to_s₁)
            e₁  = Matrix{T}(d.e_to_s₁)
            s₁² = Matrix{T}(d.s_to_s₁_by_s_to_s₁)
            e₁² = Matrix{T}(d.e_to_s₁_by_e_to_s₁)
            s₁e₁ = Matrix{T}(d.s_to_s₁_by_e_to_s₁)
            ss₂  = Matrix{T}(d.s_s_to_s₂)
            ee₂  = Matrix{T}(d.e_e_to_s₂)
            se₂  = Matrix{T}(d.s_e_to_s₂)
            vv₂  = Matrix{T}(d.v_v_to_s₂)

            # Local slice cotangent accumulators
            ∂s₁_l  = ∂s₁_ac    # start with autocorrelation contribution
            ∂e₁_l  = zeros(T, n, ne)
            ∂ss₂_l = zeros(T, n, n^2)
            ∂ee₂_l = zeros(T, n, ne^2)
            ∂se₂_l = zeros(T, n, n * ne)
            ∂vv₂_l = zeros(T, size(vv₂))
            ∂Σ̂ᶻ₁  = ∂Σ̂ᶻ₁_ac    # start with autocorrelation contribution
            ∂Σ̂ᶻ₂  = ∂Σ̂ᶻ₂_ac    # start with autocorrelation contribution
            ∂Δ̂μˢ₂_l = ∂Δ̂μˢ₂_ac  # start with autocorrelation contribution

            # Block boundary arrays
            sb = cumsum([0, n, n, n₂ˢ_i, n, n^2, n₃ˢ_i])          # ŝ_to_ŝ₃ row/col (compressed)
            eb = cumsum([0, ne, ne^2, n*ne, n*ne, n^2*ne, n*ne^2, ne^3])
            gb = eb

            vvh = vv₂ ./ 2;  ssh = ss₂ ./ 2;  eeh = ee₂ ./ 2

            # ── 2a: A_UU, A_LU, A_LL disaggregation ──
            # Block boundaries within sub-matrices
            bu = cumsum([0, n, n, n₂ˢ_i])                 # A_UU row/col blocks
            bl = cumsum([0, n, n^2, n₃ˢ_i])               # A_LL row/col blocks (also A_LU rows)

            # ── From ∂A_UU ──
            # (1,1) s₁, (2,2) s₁
            ∂s₁_l .+= ∂A_UU[bu[1]+1:bu[2], bu[1]+1:bu[2]] .+
                       ∂A_UU[bu[2]+1:bu[3], bu[2]+1:bu[3]]
            # (2,3) ss₂/2 * D₂ˢ — decompress cols
            ∂ss₂_l .+= ∂A_UU[bu[2]+1:bu[3], bu[3]+1:bu[4]] * Matrix(d.D₂ˢ)' ./ 2
            # (3,3) L₂ˢ * kron(s₁,s₁) * D₂ˢ — decompress then kron_vjp
            ∂inner33 = Matrix(d.L₂ˢ)' * Matrix(∂A_UU[bu[3]+1:bu[4], bu[3]+1:bu[4]]) * Matrix(d.D₂ˢ)'
            tmpL, tmpR = kron_vjp_helper(∂inner33, s₁, s₁)
            ∂s₁_l .+= tmpL .+ tmpR

            # ── From ∂A_LU ──
            # (1,1) s_vv₃/2
            ∂S3f_acc[d.iˢ, d.kron_s_v_v] .+= ∂A_LU[bl[1]+1:bl[2], bu[1]+1:bu[2]] ./ 2
            # (2,1) kron(s₁, vv₂/2)
            tmpA, tmpB = kron_vjp_helper(Matrix(∂A_LU[bl[2]+1:bl[3], bu[1]+1:bu[2]]), s₁, vvh)
            ∂s₁_l .+= tmpA;  ∂vv₂_l .+= tmpB ./ 2

            # ── From ∂A_LL ──
            # (1,1) s₁
            ∂s₁_l .+= ∂A_LL[bl[1]+1:bl[2], bl[1]+1:bl[2]]
            # (1,2) ss₂
            ∂ss₂_l .+= ∂A_LL[bl[1]+1:bl[2], bl[2]+1:bl[3]]
            # (1,3) sss₃/6 * D₃ˢ — decompress cols
            ∂S3f_acc[d.iˢ, d.kron_s_s_s] .+= ∂A_LL[bl[1]+1:bl[2], bl[3]+1:bl[4]] * Matrix(d.D₃ˢ)' ./ 6
            # (2,2) kron(s₁,s₁)
            tmpL, tmpR = kron_vjp_helper(Matrix(∂A_LL[bl[2]+1:bl[3], bl[2]+1:bl[3]]), s₁, s₁)
            ∂s₁_l .+= tmpL .+ tmpR
            # (2,3) kron(s₁, ss₂/2) * D₃ˢ — decompress cols then kron_vjp
            ∂inner56 = Matrix(∂A_LL[bl[2]+1:bl[3], bl[3]+1:bl[4]]) * Matrix(d.D₃ˢ)'
            tmpA, tmpB = kron_vjp_helper(∂inner56, s₁, ssh)
            ∂s₁_l .+= tmpA;  ∂ss₂_l .+= tmpB ./ 2
            # (3,3) L₃ˢ * kron(s₁, kron(s₁,s₁)) * D₃ˢ — decompress then kron_vjp
            ∂inner66 = Matrix(d.L₃ˢ)' * Matrix(∂A_LL[bl[3]+1:bl[4], bl[3]+1:bl[4]]) * Matrix(d.D₃ˢ)'
            tmpA, tmpB = kron_vjp_helper(∂inner66, s₁, s₁²)
            ∂s₁_l .+= tmpA
            tmpL, tmpR = kron_vjp_helper(tmpB, s₁, s₁)
            ∂s₁_l .+= tmpL .+ tmpR

            # ── 2b: ê_to_ŝ₃ disaggregation ──
            ∂ê₃ = Matrix{T}(∂ê_to_ŝ₃)
            ss_s1e1 = Matrix(d.s_s) * s₁e₁

            # Row 1: (1,1) e₁
            ∂e₁_l .+= ∂ê₃[sb[1]+1:sb[2], eb[1]+1:eb[2]]
            # Row 2: (2,2) ee₂/2; (2,3) se₂
            ∂ee₂_l .+= ∂ê₃[sb[2]+1:sb[3], eb[2]+1:eb[3]] ./ 2
            ∂se₂_l .+= ∂ê₃[sb[2]+1:sb[3], eb[3]+1:eb[4]]
            # Row 3: (3,2) L₂ˢ * kron(e₁,e₁) — decompress rows
            tmpL, tmpR = kron_vjp_helper(Matrix(d.L₂ˢ)' * Matrix(∂ê₃[sb[3]+1:sb[4], eb[2]+1:eb[3]]), e₁, e₁)
            ∂e₁_l .+= tmpL .+ tmpR
            # (3,3) L₂ˢ * I_plus_s_s * kron(s₁,e₁) — decompress rows
            ∂k33 = Matrix(d.I_plus_s_s') * Matrix(d.L₂ˢ)' * Matrix(∂ê₃[sb[3]+1:sb[4], eb[3]+1:eb[4]])
            tmpA, tmpB = kron_vjp_helper(∂k33, s₁, e₁)
            ∂s₁_l .+= tmpA;  ∂e₁_l .+= tmpB
            # Row 4: direct S₃ slices
            ∂S3f_acc[d.iˢ, d.kron_e_v_v] .+= ∂ê₃[sb[4]+1:sb[5], eb[1]+1:eb[2]] ./ 2
            ∂se₂_l .+= ∂ê₃[sb[4]+1:sb[5], eb[4]+1:eb[5]]
            ∂S3f_acc[d.iˢ, d.kron_s_s_e] .+= ∂ê₃[sb[4]+1:sb[5], eb[5]+1:eb[6]] ./ 2
            ∂S3f_acc[d.iˢ, d.kron_s_e_e] .+= ∂ê₃[sb[4]+1:sb[5], eb[6]+1:eb[7]] ./ 2
            ∂S3f_acc[d.iˢ, d.kron_e_e_e] .+= ∂ê₃[sb[4]+1:sb[5], eb[7]+1:eb[8]] ./ 6
            # Row 5: (5,1) kron(e₁,vv₂/2)
            tmpA, tmpB = kron_vjp_helper(Matrix(∂ê₃[sb[5]+1:sb[6], eb[1]+1:eb[2]]), e₁, vvh)
            ∂e₁_l .+= tmpA;  ∂vv₂_l .+= tmpB ./ 2
            # (5,4) s_s * kron(s₁,e₁)
            ∂k54 = Matrix(d.s_s') * Matrix(∂ê₃[sb[5]+1:sb[6], eb[4]+1:eb[5]])
            tmpA, tmpB = kron_vjp_helper(∂k54, s₁, e₁)
            ∂s₁_l .+= tmpA;  ∂e₁_l .+= tmpB
            # (5,5) kron(s₁,se₂) + s_s * kron(ss₂/2, e₁)
            ∂b55 = Matrix(∂ê₃[sb[5]+1:sb[6], eb[5]+1:eb[6]])
            tmpA, tmpB = kron_vjp_helper(∂b55, s₁, se₂)
            ∂s₁_l .+= tmpA;  ∂se₂_l .+= tmpB
            ∂k55b = Matrix(d.s_s') * ∂b55
            tmpA, tmpB = kron_vjp_helper(∂k55b, ssh, e₁)
            ∂ss₂_l .+= tmpA ./ 2;  ∂e₁_l .+= tmpB
            # (5,6) kron(s₁,ee₂/2) + s_s * kron(se₂, e₁)
            ∂b56 = Matrix(∂ê₃[sb[5]+1:sb[6], eb[6]+1:eb[7]])
            tmpA, tmpB = kron_vjp_helper(∂b56, s₁, eeh)
            ∂s₁_l .+= tmpA;  ∂ee₂_l .+= tmpB ./ 2
            ∂k56b = Matrix(d.s_s') * ∂b56
            tmpA, tmpB = kron_vjp_helper(∂k56b, se₂, e₁)
            ∂se₂_l .+= tmpA;  ∂e₁_l .+= tmpB
            # (5,7) kron(e₁, ee₂/2)
            tmpA, tmpB = kron_vjp_helper(Matrix(∂ê₃[sb[5]+1:sb[6], eb[7]+1:eb[8]]), e₁, eeh)
            ∂e₁_l .+= tmpA;  ∂ee₂_l .+= tmpB ./ 2
            # Row 6: (6,5) L₃ˢ * (kron(s₁²,e₁) + kron(s₁,s_s*s₁e₁) + kron(e₁,s₁²)*e_ss) — decompress rows
            ∂b65 = Matrix(d.L₃ˢ)' * Matrix(∂ê₃[sb[6]+1:sb[7], eb[5]+1:eb[6]])
            tmpA, tmpB = kron_vjp_helper(∂b65, s₁², e₁)
            ∂e₁_l .+= tmpB
            tmpL, tmpR = kron_vjp_helper(tmpA, s₁, s₁);  ∂s₁_l .+= tmpL .+ tmpR
            tmpA, tmpB = kron_vjp_helper(∂b65, s₁, ss_s1e1)
            ∂s₁_l .+= tmpA
            tmpC = Matrix(d.s_s') * tmpB
            tmpL, tmpR = kron_vjp_helper(tmpC, s₁, e₁);  ∂s₁_l .+= tmpL;  ∂e₁_l .+= tmpR
            ∂k65c = ∂b65 * Matrix(d.e_ss')
            tmpA, tmpB = kron_vjp_helper(∂k65c, e₁, s₁²)
            ∂e₁_l .+= tmpA
            tmpL, tmpR = kron_vjp_helper(tmpB, s₁, s₁);  ∂s₁_l .+= tmpL .+ tmpR
            # (6,6) L₃ˢ * (kron(s₁e₁,e₁) + kron(e₁,s₁e₁)*e_es + kron(e₁,s_s*s₁e₁)*e_es) — decompress rows
            ∂b66 = Matrix(d.L₃ˢ)' * Matrix(∂ê₃[sb[6]+1:sb[7], eb[6]+1:eb[7]])
            tmpA, tmpB = kron_vjp_helper(∂b66, s₁e₁, e₁)
            ∂e₁_l .+= tmpB
            tmpL, tmpR = kron_vjp_helper(tmpA, s₁, e₁);  ∂s₁_l .+= tmpL;  ∂e₁_l .+= tmpR
            ∂pre = ∂b66 * Matrix(d.e_es')
            tmpA, tmpB = kron_vjp_helper(∂pre, e₁, s₁e₁)
            ∂e₁_l .+= tmpA
            tmpL, tmpR = kron_vjp_helper(tmpB, s₁, e₁);  ∂s₁_l .+= tmpL;  ∂e₁_l .+= tmpR
            tmpA, tmpB = kron_vjp_helper(∂pre, e₁, ss_s1e1)
            ∂e₁_l .+= tmpA
            tmpC = Matrix(d.s_s') * tmpB
            tmpL, tmpR = kron_vjp_helper(tmpC, s₁, e₁);  ∂s₁_l .+= tmpL;  ∂e₁_l .+= tmpR
            # (6,7) L₃ˢ * kron(e₁, e₁²) — decompress rows
            tmpA, tmpB = kron_vjp_helper(Matrix(d.L₃ˢ)' * Matrix(∂ê₃[sb[6]+1:sb[7], eb[7]+1:eb[8]]), e₁, e₁²)
            ∂e₁_l .+= tmpA
            tmpL, tmpR = kron_vjp_helper(tmpB, e₁, e₁);  ∂e₁_l .+= tmpL .+ tmpR

            # ── 3a: Γ₃ disaggregation → ∂Σ̂ᶻ₁, ∂Σ̂ᶻ₂, ∂Δ̂μˢ₂ ──
            ∂Γ = Matrix{T}(∂Γ₃_iter)
            vΣ = vec(d.Σ̂ᶻ₁)

            # Row 1: (1,4) kron(Δ̂μˢ₂',Ine)
            ∂tmp14 = kron_vjp_helper(∂Γ[gb[1]+1:gb[2], gb[4]+1:gb[5]], reshape(d.Δ̂μˢ₂, 1, :), Ine)[1]
            ∂Δ̂μˢ₂_l .+= vec(∂tmp14')
            # (1,5) kron(vec(Σ̂ᶻ₁)',Ine)
            ∂tmp15 = kron_vjp_helper(∂Γ[gb[1]+1:gb[2], gb[5]+1:gb[6]], reshape(vΣ, 1, :), Ine)[1]
            ∂Σ̂ᶻ₁ .+= reshape(vec(∂tmp15'), n, n)
            # Row 3: (3,3) kron(Σ̂ᶻ₁,Ine)
            ∂Σ̂ᶻ₁ .+= kron_vjp_helper(∂Γ[gb[3]+1:gb[4], gb[3]+1:gb[4]], Matrix(d.Σ̂ᶻ₁), Ine)[1]
            # Row 4: (4,1) kron(Δ̂μˢ₂,Ine)
            ∂Δ̂μˢ₂_l .+= vec(kron_vjp_helper(∂Γ[gb[4]+1:gb[5], gb[1]+1:gb[2]], reshape(d.Δ̂μˢ₂, :, 1), Ine)[1])
            # (4,4) kron(Σ̂ᶻ₂_22 + Δ*Δ', Ine)
            M44 = d.Σ̂ᶻ₂[n+1:2n, n+1:2n] + d.Δ̂μˢ₂ * d.Δ̂μˢ₂'
            ∂M44 = kron_vjp_helper(∂Γ[gb[4]+1:gb[5], gb[4]+1:gb[5]], Matrix(M44), Ine)[1]
            ∂Σ̂ᶻ₂[n+1:2n, n+1:2n] .+= ∂M44
            ∂Δ̂μˢ₂_l .+= (∂M44 + ∂M44') * d.Δ̂μˢ₂
            # (4,5) kron(Σ̂ᶻ₂_23 + Δ*vΣ', Ine)
            M45 = d.Σ̂ᶻ₂[n+1:2n, 2n+1:end] + d.Δ̂μˢ₂ * vΣ'
            ∂M45 = kron_vjp_helper(∂Γ[gb[4]+1:gb[5], gb[5]+1:gb[6]], Matrix(M45), Ine)[1]
            ∂Σ̂ᶻ₂[n+1:2n, 2n+1:end] .+= ∂M45
            ∂Δ̂μˢ₂_l .+= ∂M45 * vΣ
            ∂Σ̂ᶻ₁ .+= reshape(∂M45' * d.Δ̂μˢ₂, n, n)
            # (4,7) kron(Δ̂μˢ₂, e4_nᵉ_nᵉ³)
            ∂Δ̂μˢ₂_l .+= vec(kron_vjp_helper(∂Γ[gb[4]+1:gb[5], gb[7]+1:gb[8]], reshape(d.Δ̂μˢ₂, :, 1), Matrix(e4_nᵉ_nᵉ³))[1])
            # Row 5: (5,1) kron(vΣ, Ine)
            ∂Σ̂ᶻ₁ .+= reshape(kron_vjp_helper(∂Γ[gb[5]+1:gb[6], gb[1]+1:gb[2]], reshape(vΣ, :, 1), Ine)[1], n, n)
            # (5,4) kron(Σ̂ᶻ₂_32 + vΣ*Δ', Ine)
            M54 = d.Σ̂ᶻ₂[2n+1:end, n+1:2n] + vΣ * d.Δ̂μˢ₂'
            ∂M54 = kron_vjp_helper(∂Γ[gb[5]+1:gb[6], gb[4]+1:gb[5]], Matrix(M54), Ine)[1]
            ∂Σ̂ᶻ₂[2n+1:end, n+1:2n] .+= ∂M54
            ∂Σ̂ᶻ₁ .+= reshape(∂M54 * d.Δ̂μˢ₂, n, n)
            ∂Δ̂μˢ₂_l .+= ∂M54' * vΣ
            # (5,5) kron(Σ̂ᶻ₂_33 + vΣ*vΣ', Ine)
            M55 = d.Σ̂ᶻ₂[2n+1:end, 2n+1:end] + vΣ * vΣ'
            ∂M55 = kron_vjp_helper(∂Γ[gb[5]+1:gb[6], gb[5]+1:gb[6]], Matrix(M55), Ine)[1]
            ∂Σ̂ᶻ₂[2n+1:end, 2n+1:end] .+= ∂M55
            ∂Σ̂ᶻ₁ .+= reshape((∂M55 + ∂M55') * vΣ, n, n)
            # (5,7) kron(vΣ, e4_nᵉ_nᵉ³)
            ∂Σ̂ᶻ₁ .+= reshape(kron_vjp_helper(∂Γ[gb[5]+1:gb[6], gb[7]+1:gb[8]], reshape(vΣ, :, 1), Matrix(e4_nᵉ_nᵉ³))[1], n, n)
            # Row 6: (6,6) kron(Σ̂ᶻ₁, e4_nᵉ²_nᵉ²)
            ∂Σ̂ᶻ₁ .+= kron_vjp_helper(∂Γ[gb[6]+1:gb[7], gb[6]+1:gb[7]], Matrix(d.Σ̂ᶻ₁), Matrix(e4_nᵉ²_nᵉ²))[1]
            # Row 7: (7,4) kron(Δ̂μˢ₂', e4')
            ∂tmp74 = kron_vjp_helper(∂Γ[gb[7]+1:gb[8], gb[4]+1:gb[5]], reshape(d.Δ̂μˢ₂, 1, :), Matrix(e4_nᵉ_nᵉ³'))[1]
            ∂Δ̂μˢ₂_l .+= vec(∂tmp74')
            # (7,5) kron(vΣ', e4')
            ∂tmp75 = kron_vjp_helper(∂Γ[gb[7]+1:gb[8], gb[5]+1:gb[6]], reshape(vΣ, 1, :), Matrix(e4_nᵉ_nᵉ³'))[1]
            ∂Σ̂ᶻ₁ .+= reshape(vec(∂tmp75'), n, n)

            # ── 3b: Eᴸᶻ disaggregation ──
            ∂EL = Matrix{T}(∂Eᴸᶻ_iter)
            # Only row block 6 is data-dependent
            ∂EL6 = ∂EL[gb[6]+1:gb[7], :]
            # Col 1: kron(Σ̂ᶻ₁, vec_Ie)
            ∂Σ̂ᶻ₁ .+= kron_vjp_helper(∂EL6[:, sb[1]+1:sb[2]], Matrix(d.Σ̂ᶻ₁), vec_Ie_col)[1]
            # Col 4: kron(μˢ₃δμˢ₁', vec_Ie)
            ∂μ_T = kron_vjp_helper(∂EL6[:, sb[4]+1:sb[5]], Matrix(d.μˢ₃δμˢ₁'), vec_Ie_col)[1]
            ∂μˢ₃δμˢ₁ = ∂μˢ₃δμˢ₁_ac .+ Matrix(∂μ_T')
            # Col 5: kron(C₄, vec_Ie)
            inner_C4 = d.Σ̂ᶻ₂[n+1:2n, 2n+1:end] + d.Δ̂μˢ₂ * vΣ'
            C4m = reshape(ss_s_M * vec(inner_C4), n, n^2)
            ∂C4 = kron_vjp_helper(∂EL6[:, sb[5]+1:sb[6]], C4m, vec_Ie_col)[1]
            ∂iC4 = reshape(ss_s_M' * vec(∂C4), n, n^2)
            ∂Σ̂ᶻ₂[n+1:2n, 2n+1:end] .+= ∂iC4
            ∂Δ̂μˢ₂_l .+= ∂iC4 * vΣ
            ∂Σ̂ᶻ₁ .+= reshape(∂iC4' * d.Δ̂μˢ₂, n, n)
            # Col 6: kron(C₅ * L₃ˢ', vec_Ie) — compress col 6
            inner_C5 = d.Σ̂ᶻ₂[2n+1:end, 2n+1:end] + vΣ * vΣ'
            C5m = reshape(Matrix(inner_C5), n, n^3)
            C5m_c = C5m * Matrix(d.L₃ˢ)'
            ∂C5_c = kron_vjp_helper(∂EL6[:, sb[6]+1:sb[7]], C5m_c, vec_Ie_col)[1]
            ∂C5 = ∂C5_c * Matrix(d.L₃ˢ)
            ∂iC5 = reshape(∂C5, n^2, n^2)
            ∂Σ̂ᶻ₂[2n+1:end, 2n+1:end] .+= ∂iC5
            ∂Σ̂ᶻ₁ .+= reshape((∂iC5 + ∂iC5') * vΣ, n, n)

            # ── 3c: μˢ₃δμˢ₁ adjoint ──
            ∂x_μ = vec(∂μˢ₃δμˢ₁)
            I_m_s₁² = Matrix{T}(ℒ.I(n^2)) - s₁²
            ∂b_μ = I_m_s₁²' \ ∂x_μ
            ∂s₁²_from_μ = ∂b_μ * vec(d.μˢ₃δμˢ₁)'
            tmpL, tmpR = kron_vjp_helper(∂s₁²_from_μ, s₁, s₁);  ∂s₁_l .+= tmpL .+ tmpR

            ∂RHS = reshape(∂b_μ, n, n)

            inner_M1 = d.Σ̂ᶻ₂[2n+1:end, n+1:2n] + vΣ * d.Δ̂μˢ₂'
            M1 = reshape(ss_s_M * vec(inner_M1), n^2, n)
            inner_M2 = d.Σ̂ᶻ₂[2n+1:end, 2n+1:end] + vΣ * vΣ'
            M2 = reshape(Matrix(inner_M2), n^3, n)
            M3 = ℒ.kron(Matrix(d.Σ̂ᶻ₁), vec_Ie_col)

            L₁ = ss₂ * M1 + Matrix(d.s_s_s_to_s₃) * M2 / 6 +
                 Matrix(d.s_e_e_to_s₃) * M3 / 2 + Matrix(d.s_v_v_to_s₃) * Matrix(d.Σ̂ᶻ₁) / 2

            M4 = ℒ.kron(reshape(d.Δ̂μˢ₂, :, 1), Ine)
            M5 = Matrix(e4_nᵉ_nᵉ³')
            M6 = ℒ.kron(reshape(vΣ, :, 1), Ine)

            L₂ = se₂ * M4 + Matrix(d.e_e_e_to_s₃) * M5 / 6 +
                 Matrix(d.s_s_e_to_s₃) * M6 / 2 + Matrix(d.e_v_v_to_s₃) * Ine / 2

            ∂L₁ = ∂RHS * s₁;    ∂s₁_l .+= ∂RHS' * L₁
            ∂L₂ = ∂RHS * e₁;    ∂e₁_l .+= ∂RHS' * L₂

            # Decompose ∂L₁
            ∂ss₂_l .+= ∂L₁ * M1'
            ∂M1_raw = ss₂' * ∂L₁
            ∂S3f_acc[d.iˢ, d.kron_s_s_s] .+= ∂L₁ * M2' ./ 6
            ∂M2_raw = Matrix(d.s_s_s_to_s₃)' * ∂L₁ ./ 6
            ∂S3f_acc[d.iˢ, d.kron_s_e_e] .+= ∂L₁ * M3' ./ 2
            ∂M3_raw = Matrix(d.s_e_e_to_s₃)' * ∂L₁ ./ 2
            ∂S3f_acc[d.iˢ, d.kron_s_v_v] .+= ∂L₁ * Matrix(d.Σ̂ᶻ₁)' ./ 2
            ∂Σ̂ᶻ₁ .+= Matrix(d.s_v_v_to_s₃)' * ∂L₁ ./ 2

            # Decompose ∂L₂
            ∂se₂_l .+= ∂L₂ * M4'
            ∂M4_raw = se₂' * ∂L₂
            ∂S3f_acc[d.iˢ, d.kron_e_e_e] .+= ∂L₂ * M5' ./ 6
            ∂S3f_acc[d.iˢ, d.kron_s_s_e] .+= ∂L₂ * M6' ./ 2
            ∂M6_raw = Matrix(d.s_s_e_to_s₃)' * ∂L₂ ./ 2
            ∂S3f_acc[d.iˢ, d.kron_e_v_v] .+= ∂L₂ ./ 2

            # Decompose ∂M1 → ∂Σ̂ᶻ₂, ∂Σ̂ᶻ₁, ∂Δ̂μˢ₂
            ∂iM1 = reshape(ss_s_M' * vec(∂M1_raw), n^2, n)
            ∂Σ̂ᶻ₂[2n+1:end, n+1:2n] .+= ∂iM1
            ∂Σ̂ᶻ₁ .+= reshape(∂iM1 * d.Δ̂μˢ₂, n, n)
            ∂Δ̂μˢ₂_l .+= ∂iM1' * vΣ
            # Decompose ∂M2 → ∂Σ̂ᶻ₂, ∂Σ̂ᶻ₁
            ∂iM2 = reshape(∂M2_raw, n^2, n^2)
            ∂Σ̂ᶻ₂[2n+1:end, 2n+1:end] .+= ∂iM2
            ∂Σ̂ᶻ₁ .+= reshape((∂iM2 + ∂iM2') * vΣ, n, n)
            # Decompose ∂M3 → ∂Σ̂ᶻ₁
            ∂Σ̂ᶻ₁ .+= kron_vjp_helper(∂M3_raw, Matrix(d.Σ̂ᶻ₁), vec_Ie_col)[1]
            # Decompose ∂M4 → ∂Δ̂μˢ₂
            ∂Δ̂μˢ₂_l .+= vec(kron_vjp_helper(∂M4_raw, reshape(d.Δ̂μˢ₂, :, 1), Ine)[1])
            # Decompose ∂M6 → ∂Σ̂ᶻ₁
            ∂Σ̂ᶻ₁ .+= reshape(kron_vjp_helper(∂M6_raw, reshape(vΣ, :, 1), Ine)[1], n, n)

            # ── 4: Scatter local cotangents to global accumulators ──
            ∂𝐒₁_acc[d.iˢ, d.dependencies_in_states_idx] .+= ∂s₁_l
            ∂𝐒₁_acc[d.iˢ, n₋+1:size(∂𝐒₁_acc, 2)] .+= ∂e₁_l
            ∂S2f_acc[d.iˢ, d.kron_s_s]  .+= ∂ss₂_l
            ∂S2f_acc[d.iˢ, kron_e_e]    .+= ∂ee₂_l
            ∂S2f_acc[d.iˢ, d.kron_s_e]  .+= ∂se₂_l
            ∂S2f_acc[d.iˢ, kron_v_v]    .+= ∂vv₂_l
            ∂Σʸ₁_acc[d.iˢ, d.iˢ]       .+= ∂Σ̂ᶻ₁
            ∂Σᶻ₂_acc[d.dependencies_extended_idx, d.dependencies_extended_idx] .+= ∂Σ̂ᶻ₂
            ∂Δμˢ₂_acc[d.dependencies_in_states_idx] .+= ∂Δ̂μˢ₂_l
        end

        # ── Sub-rrule pullback chain ──

        # S₃_full = S₃ * 𝐔₃  →  ∂S₃ = ∂S₃_full * 𝐔₃'
        ∂𝐒₃_compressed = ∂S3f_acc * 𝐔₃'

        # Third-order solution pullback
        so3_grad = so3_pb((∂𝐒₃_compressed, NoTangent()))
        if !(so3_grad[2] isa AbstractZero); ∂∇₁_acc .+= so3_grad[2]; end
        if !(so3_grad[3] isa AbstractZero); ∂∇₂_acc .+= so3_grad[3]; end
        if !(so3_grad[4] isa AbstractZero); ∂∇₃_acc .+= so3_grad[4]; end
        if !(so3_grad[5] isa AbstractZero); ∂𝐒₁_acc .+= so3_grad[5]; end
        # so3_grad[6] is now compressed ∂𝐒₂_raw — kept separate

        # Third-order derivatives pullback
        ∇₃_grad = ∇₃_pb(∂∇₃_acc)
        ∂params_∇₃  = ∇₃_grad[2] isa AbstractZero ? zeros(T, np) : ∇₃_grad[2]
        if !(∇₃_grad[3] isa AbstractZero); ∂SS_acc .+= ∇₃_grad[3]; end

        # Convert full-space ∂S2f_acc to compressed and add compressed so3 gradient
        ∂S2_raw_acc = ∂S2f_acc * 𝐔₂'
        if !(so3_grad[6] isa AbstractZero); ∂S2_raw_acc .+= so3_grad[6]; end

        # Second-order moments pullback
        ∂som2 = (
            NoTangent(),             # ∂Σʸ₂
            ∂Σᶻ₂_acc,               # ∂Σᶻ₂
            ∂μʸ₂_in isa AbstractZero ? NoTangent() : ∂μʸ₂_in,  # ∂μʸ₂
            ∂Δμˢ₂_acc,              # ∂Δμˢ₂
            NoTangent(),             # ∂autocorr (not used)
            NoTangent(),             # ∂ŝ_to_ŝ₂ (not used)
            NoTangent(),             # ∂ŝ_to_y₂ (not used)
            ∂Σʸ₁_acc,               # ∂Σʸ₁
            NoTangent(),             # ∂Σᶻ₁
            ∂SS_acc,                 # ∂SS_and_pars
            ∂𝐒₁_acc,                # ∂𝐒₁
            ∂∇₁_acc,                # ∂∇₁
            ∂S2_raw_acc,             # ∂𝐒₂ (compressed)
            ∂∇₂_acc,                # ∂∇₂
            NoTangent(),             # ∂slvd
        )

        som2_grad = som2_pb(∂som2)
        ∂params_som2 = som2_grad[2] isa AbstractZero ? zeros(T, np) : som2_grad[2]

        ∂parameters_total = ∂params_som2 .+ ∂params_∇₃

        return NoTangent(), ∂parameters_total, NoTangent(), NoTangent()
    end

    return result, calculate_third_order_moments_with_autocorrelation_pullback
end


function rrule(::typeof(calculate_first_order_solution), 
                ∇₁::Matrix{R},
                constants::constants,
                workspaces::workspaces,
                cache::caches;
                opts::CalculationOptions = merge_calculation_options(),
                use_fastlapack_qr::Bool = true,
                use_fastlapack_lu::Bool = true,
                initial_guess::AbstractMatrix{R} = zeros(0,0),
                parameter_values::AbstractVector{<:Real} = Float64[],
                caching::Bool = true) where {R <: AbstractFloat}
    # Forward pass to compute the output and intermediate values needed for the backward pass
    # @timeit_debug timer "Calculate 1st order solution" begin
    # @timeit_debug timer "Preprocessing" begin

    T = constants.post_model_macro
    ensure_first_order_constants!(constants)
    idx_constants = constants.post_complete_parameters

    dynIndex = idx_constants.dyn_index
    reverse_dynamic_order = idx_constants.reverse_dynamic_order
    comb = idx_constants.comb
    future_not_past_and_mixed_in_comb = idx_constants.future_not_past_and_mixed_in_comb
    past_not_future_and_mixed_in_comb = idx_constants.past_not_future_and_mixed_in_comb
    past_not_future_and_mixed_in_present_but_not_only = idx_constants.past_not_future_and_mixed_in_present_but_not_only
    Ir = idx_constants.Ir

    qme_ws = workspaces.first_order
    sylv_ws = workspaces.sylvester_1st_order
    ensure_sylvester_krylov_buffers!(qme_ws.sylvester, T.nVars, T.nVars)
    ensure_sylvester_doubling_buffers!(qme_ws.sylvester, T.nVars, T.nVars)

    ensure_first_order_workspace_buffers!(qme_ws, T, length(dynIndex), length(comb))
    
    ∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
    ∇₀ = qme_ws.∇₀
    copyto!(∇₀, @view(∇₁[:,idx_constants.nabla_zero_cols]))
    ∇₋ = @view ∇₁[:,idx_constants.nabla_minus_cols]
    ∇̂ₑ = qme_ws.∇ₑ
    copyto!(∇̂ₑ, @view(∇₁[:,idx_constants.nabla_e_start:end]))
    
    # end # timeit_debug
    # @timeit_debug timer "Invert ∇₀" begin

    A₊ = qme_ws.𝐀₊
    A₀ = qme_ws.𝐀₀
    A₋ = qme_ws.𝐀₋
    ∇₀_present = @view ∇₀[:, T.present_only_idx]
    # Old way (≤v0.1.42):
    #   Q = qr(∇₀[:, present_only_idx])
    #   A₊ = Q' * ∇₊;  A₀ = Q' * ∇₀;  A₋ = Q' * ∇₋
    # Current code reuses QR workspaces to avoid allocations.
    qr_factors, qr_ws = ensure_first_order_fast_qr_workspace!(qme_ws, ∇₀_present)
    Q = factorize_qr!((use_fastlapack_qr ? Val(:FastLapack) : Val(:Julia)), ∇₀_present, qr_factors, qr_ws)                 # Q = qr(∇₀_present)

    qme_ws.fast_qr_orm_ws_plus, qme_ws.fast_qr_orm_dims_plus = apply_qr_transpose_left!(A₊, ∇₊, Q,           # A₊ = Q' * ∇₊
                                                                                        qme_ws.fast_qr_orm_ws_plus,
                                                                                        qme_ws.fast_qr_orm_dims_plus,
                                                                                        qr_ws;
                                                                                        use_fastlapack_qr = use_fastlapack_qr)
    qme_ws.fast_qr_orm_ws_zero, qme_ws.fast_qr_orm_dims_zero = apply_qr_transpose_left!(A₀, ∇₀, Q,           # A₀ = Q' * ∇₀
                                                                                        qme_ws.fast_qr_orm_ws_zero,
                                                                                        qme_ws.fast_qr_orm_dims_zero,
                                                                                        qr_ws;
                                                                                        use_fastlapack_qr = use_fastlapack_qr)
    qme_ws.fast_qr_orm_ws_minus, qme_ws.fast_qr_orm_dims_minus = apply_qr_transpose_left!(A₋, ∇₋, Q,         # A₋ = Q' * ∇₋
                                                                                           qme_ws.fast_qr_orm_ws_minus,
                                                                                           qme_ws.fast_qr_orm_dims_minus,
                                                                                           qr_ws;
                                                                                           use_fastlapack_qr = use_fastlapack_qr)
    
    # end # timeit_debug
    # @timeit_debug timer "Sort matrices" begin

    Ã₊ = qme_ws.𝐀̃₊
    ℒ.mul!(Ã₊, @view(A₊[dynIndex,:]), Ir[future_not_past_and_mixed_in_comb,:])  # Ã₊ = A₊[dynIndex,:] * Ir

    Ã₀ = qme_ws.𝐀̃₀
    copyto!(Ã₀, @view(A₀[dynIndex, comb]))

    Ã₋ = qme_ws.𝐀̃₋
    ℒ.mul!(Ã₋, @view(A₋[dynIndex,:]), Ir[past_not_future_and_mixed_in_comb,:])  # Ã₋ = A₋[dynIndex,:] * Ir

    # end # timeit_debug
    # @timeit_debug timer "Quadratic matrix equation solve" begin

    sol, solved = solve_quadratic_matrix_equation(Ã₊, Ã₀, Ã₋, constants, workspaces, cache;
                                                    initial_guess = initial_guess,
                                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                                    tol = opts.tol.first_order.ad.qme,
                                                    verbose = opts.verbose,
                                                    caching = caching)

    if !solved
        return (fill(NaN, T.nVars, T.nPast_not_future_and_mixed + T.nExo), sol, false), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # end # timeit_debug
    # @timeit_debug timer "Postprocessing" begin
    # @timeit_debug timer "Setup matrices" begin

    sol_compact = @view sol[reverse_dynamic_order, past_not_future_and_mixed_in_comb]

    D = @view sol_compact[end - T.nFuture_not_past_and_mixed + 1:end, :]

    L = @view sol[past_not_future_and_mixed_in_present_but_not_only, past_not_future_and_mixed_in_comb]

    Ā₀ᵤ = qme_ws.𝐀̄₀ᵤ
    copyto!(Ā₀ᵤ, @view(A₀[1:T.nPresent_only, T.present_only_idx]))

    A₊ᵤ = qme_ws.𝐀₊ᵤ
    copyto!(A₊ᵤ, @view(A₊[1:T.nPresent_only,:]))

    Ã₀ᵤ = qme_ws.𝐀̃₀ᵤ
    copyto!(Ã₀ᵤ, @view(A₀[1:T.nPresent_only, T.present_but_not_only_idx]))

    A₋ᵤ = qme_ws.𝐀₋ᵤ
    copyto!(A₋ᵤ, @view(A₋[1:T.nPresent_only,:]))

    # end # timeit_debug
    # @timeit_debug timer "Invert Ā₀ᵤ" begin

    # Old way (≤v0.1.42): Ā̂₀ᵤ = lu(Ā₀ᵤ)
    qme_ws.fast_lu_ws_a0u, qme_ws.fast_lu_dims_a0u, solved_Ā₀ᵤ, Ā̂₀ᵤ = factorize_lu!((use_fastlapack_lu ? Val(:FastLapack) : Val(:Julia)), Ā₀ᵤ,
                                                                                       qme_ws.fast_lu_ws_a0u,
                                                                                       qme_ws.fast_lu_dims_a0u)

    if !solved_Ā₀ᵤ
        return (zeros(T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # A    = vcat(-(Ā̂₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ)), sol)
    if T.nPresent_only > 0
        ℒ.mul!(A₋ᵤ, Ã₀ᵤ, @view(sol[:,past_not_future_and_mixed_in_comb]), 1, 1)  # A₋ᵤ = A₋ᵤ + Ã₀ᵤ * sol
        nₚ₋ = qme_ws.𝐧ₚ₋
        ℒ.mul!(nₚ₋, A₊ᵤ, D)  # nₚ₋ = A₊ᵤ * D
        ℒ.mul!(A₋ᵤ, nₚ₋, L, 1, 1)  # A₋ᵤ = A₋ᵤ + nₚ₋ * L
        solve_lu_left!(Ā₀ᵤ, A₋ᵤ, qme_ws.fast_lu_ws_a0u, Ā̂₀ᵤ;
                       use_fastlapack_lu = use_fastlapack_lu)
        ℒ.rmul!(A₋ᵤ, -1)  # A₋ᵤ = -A₋ᵤ
    end

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Exogenous part solution" begin

    expand_future = idx_constants.expand_future
    expand_past = idx_constants.expand_past

    𝐒ᵗ = qme_ws.𝐀

    for i in 1:T.nVars
        src = T.reorder[i]
        if src <= T.nPresent_only
            @views copyto!(𝐒ᵗ[i, :], A₋ᵤ[src, :])
        else
            src_idx = src - T.nPresent_only
            @views copyto!(𝐒ᵗ[i, :], sol_compact[src_idx, :])
        end
    end
    
    𝐒̂ᵗ = qme_ws.sylvester.tmp
    ℒ.mul!(𝐒̂ᵗ, 𝐒ᵗ, expand_past)  # Ŝᵗ = Sᵗ * expand_past  # Ŝᵗ = Sᵗ * expand_past

    ∇₊ = qme_ws.sylvester.𝐀
    ℒ.mul!(∇₊, @view(∇₁[:,1:T.nFuture_not_past_and_mixed]), expand_future)  # ∇₊ = ∇₁[:, future_cols] * expand_future

    ℒ.mul!(∇₀, ∇₊, 𝐒̂ᵗ, 1, 1)  # ∇₀ = ∇₊ * Ŝᵗ + ∇₀  # ∇₀ = ∇₊ * Ŝᵗ + ∇₀

    # Old way (≤v0.1.42): C = lu(∇₀)
    # Old way (≤v0.1.42): C = lu(∇₀)
    qme_ws.fast_lu_ws_nabla0, qme_ws.fast_lu_dims_nabla0, solved_∇₀, C = factorize_lu!((use_fastlapack_lu ? Val(:FastLapack) : Val(:Julia)), ∇₀,
                                                                                         qme_ws.fast_lu_ws_nabla0,
                                                                                         qme_ws.fast_lu_dims_nabla0)

    if !solved_∇₀
        return (zeros(T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # Old way (≤v0.1.42): ∇ₑ = -(∇₀ \ ∇ₑ)
    solve_lu_left!(∇₀, ∇̂ₑ, qme_ws.fast_lu_ws_nabla0, C;
                   use_fastlapack_lu = use_fastlapack_lu)
    ℒ.rmul!(∇̂ₑ, -1)

    # end # timeit_debug
    # end # timeit_debug
    
    M = qme_ws.sylvester.𝐀¹
    fill!(M, zero(R))
    @inbounds for i in axes(M, 1)
        M[i, i] = one(R)
    end
    # Old way (≤v0.1.42): M = ∇₀ \ I  (i.e. inv(∇₀))
    solve_lu_left!(∇₀, M, qme_ws.fast_lu_ws_nabla0, C;
                   use_fastlapack_lu = use_fastlapack_lu)

    tmp2 = qme_ws.sylvester.𝐁
    ℒ.mul!(tmp2, M', ∇₊')  # tmp2 = M' * ∇₊'
    ℒ.rmul!(tmp2, -1)  # tmp2 = -tmp2

    ∇ₑ = @view ∇₁[:,idx_constants.nabla_e_start:end]

    function first_order_solution_pullback(∂𝐒) 
        # Guard: if the cotangent for the solution matrix is NoTangent
        # (e.g. because a downstream filter failure returned all-NoTangent),
        # return zero gradients immediately.
        ∂𝐒_mat = unthunk(∂𝐒[1])
        if ∂𝐒_mat isa Union{NoTangent, AbstractZero}
            return NoTangent(), zero(∇₁), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        ∂∇₁ = zero(∇₁)

        ∂𝐒ᵗ = ∂𝐒_mat[:,1:T.nPast_not_future_and_mixed]
        ∂𝐒ᵉ = ∂𝐒_mat[:,T.nPast_not_future_and_mixed + 1:end]

        # Shared sub-expression: W = M' * ∂𝐒ᵉ * ∇ₑ' * M'
        # Use workspace buffers to avoid repeated intermediate allocations.
        # t1 = M' * ∂𝐒ᵉ  (nVars × nExo)
        t1 = M' * ∂𝐒ᵉ  # one alloc for nVars×nExo

        # ∂∇₁[:,nabla_e_start:end] = -t1
        @views ∂∇₁[:,idx_constants.nabla_e_start:end] .= .-t1

        # t2 = t1 * ∇ₑ'  (nVars × nVars) → store in 𝐗 workspace
        t2 = qme_ws.sylvester.𝐗
        ℒ.mul!(t2, t1, ∇ₑ')

        # W = t2 * M'  (nVars × nVars) → store in 𝐂_dbl workspace
        W = qme_ws.sylvester.𝐂_dbl
        ℒ.mul!(W, t2, M')

        @views ∂∇₁[:,idx_constants.nabla_zero_cols] .= W

        # Wp = W * expand_past'  (nVars × nPast) → store in view of 𝐂¹ workspace (nVars×nVars)
        Wp = @view qme_ws.sylvester.𝐂¹[:, 1:T.nPast_not_future_and_mixed]
        ℒ.mul!(Wp, W, expand_past')

        # ∂∇₁[:,1:nFuture] = (Wp * 𝐒ᵗ')[:,future_idx]
        # WpSt = Wp * 𝐒ᵗ'  (nVars × nVars) → store in 𝐂B workspace
        WpSt = qme_ws.sylvester.𝐂B
        ℒ.mul!(WpSt, Wp, 𝐒ᵗ')
        @views ∂∇₁[:,1:T.nFuture_not_past_and_mixed] .= WpSt[:,T.future_not_past_and_mixed_idx]

        # ∂𝐒ᵗ += ∇₊' * Wp  (nVars × nPast, ∇₊ is nVars×nVars, Wp is nVars×nPast)
        ℒ.mul!(∂𝐒ᵗ, ∇₊', Wp, 1, 1)

        tmp1 = qme_ws.sylvester.𝐂
        # tmp1 = M' * ∂𝐒ᵗ * expand_past  (nVars × nVars)
        # t_ms = M' * ∂𝐒ᵗ  (nVars × nPast) → reuse Wp (view of 𝐂¹, same dims)
        ℒ.mul!(Wp, M', ∂𝐒ᵗ)
        ℒ.mul!(tmp1, Wp, expand_past)
        ℒ.lmul!(-1, tmp1)

        ss, solved = solve_sylvester_equation(tmp2, 𝐒̂ᵗ', tmp1, sylv_ws,
                                                sylvester_algorithm = opts.sylvester_algorithm²,
                                                preconditioner = opts.sylvester_preconditioner,
                                                tol = opts.tol.first_order.ad.sylvester,
                                                verbose = opts.verbose)

        if !solved
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        # ss_Sht = ss * 𝐒̂ᵗ'  (nVars × nVars) → reuse t2
        ℒ.mul!(t2, ss, 𝐒̂ᵗ')
        @views ∂∇₁[:,idx_constants.nabla_zero_cols] .+= t2

        # ss_Sht_Sht = t2 * 𝐒̂ᵗ'  (nVars × nVars) → reuse W
        ℒ.mul!(W, t2, 𝐒̂ᵗ')
        @views ∂∇₁[:,1:T.nFuture_not_past_and_mixed] .+= W[:,T.future_not_past_and_mixed_idx]

        @views ∂∇₁[:,idx_constants.nabla_minus_cols] .+= ss[:,T.past_not_future_and_mixed_idx]

        return NoTangent(), ∂∇₁, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    n_rows = size(𝐒ᵗ, 1)
    n_cols_A = size(𝐒ᵗ, 2)
    n_cols_ϵ = size(∇̂ₑ, 2)
    total_cols = n_cols_A + n_cols_ϵ

    S₁_existing = cache.first_order_solution_matrix
    if S₁_existing isa Matrix{R} && size(S₁_existing) == (n_rows, total_cols)
        copyto!(@view(S₁_existing[:, 1:n_cols_A]), 𝐒ᵗ)
        copyto!(@view(S₁_existing[:, n_cols_A+1:total_cols]), ∇̂ₑ)
        𝐒₁ = S₁_existing
    else
        𝐒₁ = hcat(𝐒ᵗ, ∇̂ₑ)
        cache.first_order_solution_matrix = 𝐒₁
    end

    if !isempty(parameter_values)
        cache.valid_for.first_order_solution = Float64.(parameter_values)
    end

    return (𝐒₁, sol, solved), first_order_solution_pullback
end

function rrule(::typeof(calculate_second_order_solution), 
                    ∇₁::AbstractMatrix{S}, #first order derivatives
                    ∇₂::SparseMatrixCSC{S}, #second order derivatives
                    𝑺₁::AbstractMatrix{S},#first order solution
                    constants::constants,
                    workspaces::workspaces,
                    cache::caches;
                    initial_guess::AbstractMatrix{R} = zeros(0,0),
                    opts::CalculationOptions = merge_calculation_options(),
                    parameter_values::AbstractVector{<:Real} = Float64[],
                    caching::Bool = true) where {S <: Real, R <: Real}
    if !(eltype(workspaces.second_order.Ŝ) == S)
        workspaces.second_order = Higher_order_workspace(S)
    end
    ℂ = workspaces.second_order
    M₂ = constants.second_order
    T = constants.post_model_macro

    # Expand compressed hessian to full space for internal computation
    ∇₂_full = ∇₂ * M₂.𝐔∇₂

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

    ensure_higher_order_solution_buffers!(ℂ, n, nₑ₋)

    # @timeit_debug timer "Setup matrices" begin

    # 1st order solution
    𝐒₁ = ℂ.𝐒₁::Matrix{S}
    copyto!(@view(𝐒₁[:,1:n₋]), @view(𝑺₁[:,1:n₋]))
    fill!(@view(𝐒₁[:,n₋+1]), zero(S))
    copyto!(@view(𝐒₁[:,n₋+2:end]), @view(𝑺₁[:,n₋+1:end]))
    # droptol!(𝐒₁,tol)
    
    𝐒₁₋╱𝟏ₑ = ℂ.𝐒₁₋╱𝟏ₑ::Matrix{S}
    copyto!(@view(𝐒₁₋╱𝟏ₑ[1:n₋,:]), @view(𝐒₁[i₋,:]))
    fill!(@view(𝐒₁₋╱𝟏ₑ[n₋+1:end,:]), zero(S))
    @inbounds 𝐒₁₋╱𝟏ₑ[n₋+1,n₋+1] = one(S)
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 1.0)

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                ℒ.I(nₑ₋)[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]]

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)]

    ∇₁₊𝐒₁➕∇₁₀ = collect(@views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * M₂.𝐈ₙ₋ - ∇₁[:,range(1,n) .+ n₊])

    # end # timeit_debug
    # @timeit_debug timer "Invert matrix" begin

    qme_ws = workspaces.first_order

    if S === Float64
        qme_ws.fast_lu_ws_nabla0, qme_ws.fast_lu_dims_nabla0, solved_∇lu, lu_handle =
            factorize_lu!(Val(:FastLapack), ∇₁₊𝐒₁➕∇₁₀, qme_ws.fast_lu_ws_nabla0, qme_ws.fast_lu_dims_nabla0)

        if !solved_∇lu
            if opts.verbose println("Second order solution: inversion failed") end
            return (∇₁₊𝐒₁➕∇₁₀, false), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end

        spinv = Matrix{S}(ℒ.I, size(∇₁₊𝐒₁➕∇₁₀))
        solve_lu_left!(∇₁₊𝐒₁➕∇₁₀, spinv, qme_ws.fast_lu_ws_nabla0, lu_handle)
    else
        ∇₁₊𝐒₁➕∇₁₀lu = ℒ.lu(∇₁₊𝐒₁➕∇₁₀, check = false)

        if !ℒ.issuccess(∇₁₊𝐒₁➕∇₁₀lu)
            if opts.verbose println("Second order solution: inversion failed") end
            return (∇₁₊𝐒₁➕∇₁₀, false), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end

        spinv = inv(∇₁₊𝐒₁➕∇₁₀lu)
    end
    spinv = choose_matrix_format(spinv)

    # end # timeit_debug
    # @timeit_debug timer "Setup second order matrices" begin
    # @timeit_debug timer "A" begin

    ∇₁₊ = @views ∇₁[:,1:n₊] * M₂.𝐈ₙ₊

    A = spinv * ∇₁₊
    
    # end # timeit_debug
    # @timeit_debug timer "C" begin

    kron_compressed = compressed_kron²(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,
                                        rowmask = M₂.∇₂_nonempty_col_as_kron_rowmask,
                                        sparse_preallocation = ℂ.tmp_sparse_prealloc2)

    term1 = ∇₂ * kron_compressed

    kron_sigma_compressed = compressed_kron²(𝐒₁₊╱𝟎,
                                            rowmask = M₂.∇₂_nonempty_col_as_kron_rowmask,
                                            colmask = M₂.𝛔𝐂₂_nonempty_row_as_kron_colmask,
                                            sparse_preallocation = ℂ.tmp_sparse_prealloc3)

    term2 = (∇₂ * kron_sigma_compressed) * M₂.𝛔c₂

    ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = term1 + term2
    
    C = spinv * ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹

    # end # timeit_debug
    # @timeit_debug timer "B" begin

    # 𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0)

    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0)
    B = compressed_kron²(𝐒₁₋╱𝟏ₑ, sparse_preallocation = ℂ.tmp_sparse_prealloc1) + M₂.𝛔c₂

    # end # timeit_debug    
    # end # timeit_debug
    # @timeit_debug timer "Solve sylvester equation" begin

    # Doubling power-cache: enable capture so the pullback's adjoint solve can
    # reuse A^(2^k), B^(2^k) from this forward pass.
    cache_eligible_2nd = opts.sylvester_algorithm² == :doubling
    if cache_eligible_2nd
        ℂ.sylvester_workspace.pow_iters = 0
        ℂ.sylvester_workspace.pow_capture = true
        ℂ.sylvester_workspace.pow_transposed = true
    end
    𝐒₂, solved = solve_sylvester_equation(A, B, C, ℂ.sylvester_workspace,
                                        initial_guess = initial_guess,
                                        sylvester_algorithm = opts.sylvester_algorithm²,
                                        preconditioner = opts.sylvester_preconditioner,
                                        tol = opts.tol.second_order.ad.sylvester,
                                        verbose = opts.verbose)
    ℂ.sylvester_workspace.pow_capture = false
    pow_iters_captured_2nd = ℂ.sylvester_workspace.pow_iters
    ℂ.sylvester_workspace.pow_iters = 0
    𝐒₂_stable = copy(𝐒₂)

    # end # timeit_debug
    # @timeit_debug timer "Post-process" begin

    if !solved
        return (𝐒₂_stable, solved), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # end # timeit_debug

    # sp⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t = choose_matrix_format(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋', density_threshold = 1.0)

    # sp𝐒₁₊╱𝟎t = choose_matrix_format(𝐒₁₊╱𝟎', density_threshold = 1.0)

    𝛔t = M₂.𝛔ᵀ

    𝐔₂t = M₂.𝐔₂ᵀ

    𝐂₂t = M₂.𝐂₂ᵀ

    Bt = choose_matrix_format(B', density_threshold = 1.0)
    At = choose_matrix_format(A', density_threshold = 1.0)
    𝐒₂_stable_t = choose_matrix_format(𝐒₂_stable', density_threshold = 1.0)

    ∇₂t = choose_matrix_format(∇₂', density_threshold = 1.0)

    # end # timeit_debug

    # Ensure pullback workspaces are properly sized
    if size(ℂ.∂∇₂) != size(∇₂)
        ℂ.∂∇₂ = zeros(S, size(∇₂))
    end
    if size(ℂ.∂∇₁) != size(∇₁)
        ℂ.∂∇₁ = zeros(S, size(∇₁))
    end
    if size(ℂ.∂𝐒₁) != size(𝐒₁)
        ℂ.∂𝐒₁ = zeros(S, size(𝐒₁))
    end
    if size(ℂ.∂spinv) != size(∇₁₊𝐒₁➕∇₁₀)
        ℂ.∂spinv = zeros(S, size(∇₁₊𝐒₁➕∇₁₀))
    end
    if size(ℂ.∂𝐒₁₋╱𝟏ₑ) != size(𝐒₁₋╱𝟏ₑ)
        ℂ.∂𝐒₁₋╱𝟏ₑ = zeros(S, size(𝐒₁₋╱𝟏ₑ))
    end
    if size(ℂ.∂𝐒₁₊╱𝟎) != size(𝐒₁₊╱𝟎)
        ℂ.∂𝐒₁₊╱𝟎 = zeros(S, size(𝐒₁₊╱𝟎))
    end
    if size(ℂ.∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) != size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋)
        ℂ.∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = zeros(S, size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋))
    end

    function second_order_solution_pullback(∂𝐒₂_solved) 
        # @timeit_debug timer "Second order solution - pullback" begin
            
        # @timeit_debug timer "Preallocate" begin
        # Use workspaces and fill with zeros instead of allocating new arrays
        ∂∇₂ = ℂ.∂∇₂; fill!(∂∇₂, zero(S))
        ∂∇₁ = ℂ.∂∇₁; fill!(∂∇₁, zero(S))
        ∂𝐒₁ = ℂ.∂𝐒₁; fill!(∂𝐒₁, zero(S))
        ∂spinv = ℂ.∂spinv; fill!(∂spinv, zero(S))
        ∂𝐒₁₋╱𝟏ₑ = ℂ.∂𝐒₁₋╱𝟏ₑ; fill!(∂𝐒₁₋╱𝟏ₑ, zero(S))
        ∂𝐒₁₊╱𝟎 = ℂ.∂𝐒₁₊╱𝟎; fill!(∂𝐒₁₊╱𝟎, zero(S))
        ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = ℂ.∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋; fill!(∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, zero(S))

        # end # timeit_debug

        ∂𝐒₂ = unthunk(∂𝐒₂_solved[1])

        if size(∂𝐒₂, 2) == size(𝐒₂_stable, 2)
            nothing
        elseif size(∂𝐒₂, 2) == size(M₂.𝐔₂, 2)
            ∂𝐒₂ = ∂𝐒₂ * 𝐔₂t
        else
            throw(DimensionMismatch("second_order_solution_pullback: expected ∂𝐒₂ to have $(size(𝐒₂_stable, 2)) (compressed) or $(size(M₂.𝐔₂, 2)) (full) columns, got $(size(∂𝐒₂, 2))."))
        end

        # @timeit_debug timer "Sylvester" begin
        if ℒ.norm(∂𝐒₂) < opts.tol.second_order.ad.sylvester.acceptance_tol
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end
        
        ws = ℂ.sylvester_workspace
        cache_valid = cache_eligible_2nd &&
                      pow_iters_captured_2nd >= 1 &&
                      ws.pow_transposed
        saved_capture = ws.pow_capture
        if cache_valid
            ws.pow_iters = pow_iters_captured_2nd
            ws.pow_capture = false
        end
        ∂C, solved = solve_sylvester_equation(At, Bt, ∂𝐒₂, ws,
                                              sylvester_algorithm = opts.sylvester_algorithm²,
                                              preconditioner = opts.sylvester_preconditioner,
                                              tol = opts.tol.second_order.ad.sylvester,
                                              verbose = opts.verbose)
        ws.pow_capture = saved_capture
        ws.pow_iters = 0

        if !solved
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end
        
        # end # timeit_debug

        # @timeit_debug timer "Matmul" begin

        ∂C = choose_matrix_format(∂C) # Dense

        ∂A = ∂C * Bt * 𝐒₂_stable_t

        ∂B = 𝐒₂_stable_t * At * ∂C

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
        ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = spinv' * ∂C
        
        ∂spinv += ∂C * ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹'

        # end # timeit_debug

        # @timeit_debug timer "Matmul3" begin

        ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = choose_matrix_format(∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹, density_threshold = 1.0)

        ∂term2 = ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ * M₂.𝛔c₂'

        ∂∇₂ += ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ * kron_compressed'
        ∂∇₂ += ∂term2 * kron_sigma_compressed'
        
        # end # timeit_debug

        # @timeit_debug timer "Matmul4" begin

        ∂kron𝐒₁₊╱𝟎 = ∇₂t * ∂term2

        # end # timeit_debug

        # @timeit_debug timer "Kron adjoint 2" begin

        compressed_kron²_pullback!(∂𝐒₁₊╱𝟎, ∂kron𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎,
                        tol = opts.tol.second_order.droptol, rowmask = M₂.∇₂_nonempty_col_as_kron_rowmask,
                        colmask = M₂.𝛔𝐂₂_nonempty_row_as_kron_colmask)
        
        # end # timeit_debug

        ∂kron⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = ∇₂t * ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹

        # @timeit_debug timer "Kron adjoint 3" begin

        compressed_kron²_pullback!(∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ∂kron⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,
                        tol = opts.tol.second_order.droptol, rowmask = M₂.∇₂_nonempty_col_as_kron_rowmask)

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

        return NoTangent(), ∂∇₁, ∂∇₂, ∂𝑺₁, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    

    if solved
        if 𝐒₂ isa Matrix{S} && cache.second_order_solution isa Matrix{S} && size(cache.second_order_solution) == size(𝐒₂)
            copyto!(cache.second_order_solution, 𝐒₂)
        elseif 𝐒₂ isa SparseMatrixCSC{S, Int} && cache.second_order_solution isa SparseMatrixCSC{S, Int} &&
               size(cache.second_order_solution) == size(𝐒₂) &&
               cache.second_order_solution.colptr == 𝐒₂.colptr &&
               cache.second_order_solution.rowval == 𝐒₂.rowval
            copyto!(cache.second_order_solution.nzval, 𝐒₂.nzval)
        else
            cache.second_order_solution = 𝐒₂
        end
        if !isempty(parameter_values)
            cache.valid_for.second_order_solution = Float64.(parameter_values)
        end
        empty!(cache.valid_for.pruned_second_order_solution)
    end

    # return (sparse(𝐒₂ * M₂.𝐔₂), solved), second_order_solution_pullback  # was: dense_to_sparse
    return (𝐒₂_stable, solved), second_order_solution_pullback
end


# ═══════════════════════════════════════════════════════════════════════════════
#  Kron-adjoint helper kernels (fill_kron_adjoint!, mul_fill_kron_adjoint!, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

function fill_kron_adjoint!(∂A::AbstractMatrix{R}, 
                            ∂B::AbstractMatrix{R}, 
                            ∂X::AbstractSparseMatrix{R}, 
                            A::AbstractMatrix{TA}, 
                            B::AbstractMatrix{TB}) where {R <: Real, TA <: Real, TB <: Real}
    @assert size(∂A) == size(A)
    @assert size(∂B) == size(B)
    @assert length(∂X) == length(B) * length(A) "∂X must have the same length as kron(B,A)"
    
    n1, m1 = size(B)
    n2, m2 = size(A)

    # Access the sparse matrix internal representation
    if ∂X isa SparseMatrixCSC
        colptr = ∂X.colptr  # Column pointers
        rowval = ∂X.rowval  # Row indices of non-zeros
        nzval  = ∂X.nzval   # Non-zero values
    else
        colptr = ∂X.A.colptr  # Column pointers
        rowval = ∂X.A.rowval  # Row indices of non-zeros
        nzval  = ∂X.A.nzval   # Non-zero values
    end
    
    # Iterate over columns of ∂X
    for col in 1:size(∂X, 2)
        # Iterate over the non-zeros in this column
        for idx in colptr[col]:(colptr[col + 1] - 1)
            row = rowval[idx]
            val = nzval[idx]

            @inbounds begin
                i = (row - 1) ÷ n2 + 1
                k = (row - 1) % n2 + 1
                j = (col - 1) ÷ m2 + 1
                l = (col - 1) % m2 + 1
                
                # Update ∂B and ∂A
                ∂A[k,l] += B[i,j] * val
                ∂B[i,j] += A[k,l] * val
            end
        end
    end
end


function fill_kron_adjoint!(∂A::AbstractMatrix{R}, 
                            ∂B::AbstractMatrix{R}, 
                            ∂X::DenseMatrix{R}, 
                            A::AbstractMatrix{TA}, 
                            B::AbstractMatrix{TB}) where {R <: Real, TA <: Real, TB <: Real}
    @assert size(∂A) == size(A)
    @assert size(∂B) == size(B)
    @assert length(∂X) == length(B) * length(A) "∂X must have the same length as kron(B,A)"
    
    re∂X = reshape(∂X, 
                    size(A,1), 
                    size(B,1), 
                    size(A,2), 
                    size(B,2))

    ei = 1
    for e in eachslice(re∂X; dims = (1,3))
        @inbounds ∂A[ei] += ℒ.dot(B,e)
        ei += 1
    end

    ei = 1
    for e in eachslice(re∂X; dims = (2,4))
        @inbounds ∂B[ei] += ℒ.dot(A,e)
        ei += 1
    end
end


function fill_kron_adjoint!(∂A::AbstractMatrix{R},
                            ∂B::AbstractMatrix{R},
                            ∂X::DenseMatrix{R},
                            A::SparseMatrixCSC{TA, Int},
                            B::SparseMatrixCSC{TB, Int}) where {R <: Real, TA <: Real, TB <: Real}
    @assert size(∂A) == size(A)
    @assert size(∂B) == size(B)
    @assert length(∂X) == length(B) * length(A) "∂X must have the same length as kron(B,A)"

    n1, m1 = size(B)
    n2, m2 = size(A)

    A_colptr = A.colptr
    A_rowval = A.rowval
    A_nzval = A.nzval

    B_colptr = B.colptr
    B_rowval = B.rowval
    B_nzval = B.nzval

    # ∂A[k,l] += Σ_{i,j} B[i,j] * ∂X[(i-1)n2 + k, (j-1)m2 + l]
    @inbounds for l in 1:m2
        base_col_l = l
        for k in 1:n2
            acc = zero(R)
            for j in 1:m1
                b_start = B_colptr[j]
                b_stop = B_colptr[j + 1] - 1
                col_idx = (j - 1) * m2 + base_col_l
                for bidx in b_start:b_stop
                    i = B_rowval[bidx]
                    row_idx = (i - 1) * n2 + k
                    acc += R(B_nzval[bidx]) * ∂X[row_idx, col_idx]
                end
            end
            ∂A[k, l] += acc
        end
    end

    # ∂B[i,j] += Σ_{k,l} A[k,l] * ∂X[(i-1)n2 + k, (j-1)m2 + l]
    @inbounds for j in 1:m1
        b_start = B_colptr[j]
        b_stop = B_colptr[j + 1] - 1
        for bidx in b_start:b_stop
            i = B_rowval[bidx]
            row_base = (i - 1) * n2
            col_base = (j - 1) * m2
            acc = zero(R)
            for l in 1:m2
                a_start = A_colptr[l]
                a_stop = A_colptr[l + 1] - 1
                col_idx = col_base + l
                for aidx in a_start:a_stop
                    k = A_rowval[aidx]
                    row_idx = row_base + k
                    acc += R(A_nzval[aidx]) * ∂X[row_idx, col_idx]
                end
            end
            ∂B[i, j] += acc
        end
    end
end



function fill_kron_adjoint!(∂A::V, ∂B::V, ∂X::V, A::V, B::V) where V <: Vector{<: Real}
    @assert size(∂A) == size(A)
    @assert size(∂B) == size(B)
    @assert length(∂X) == length(B) * length(A) "∂X must have the same length as kron(B,A)"
    
    re∂X = reshape(∂X, 
                    length(A), 
                    length(B))

    ei = 1
    for e in eachslice(re∂X; dims = 1)
        @inbounds ∂A[ei] += ℒ.dot(B,e)
        ei += 1
    end

    ei = 1
    for e in eachslice(re∂X; dims = 2)
        @inbounds ∂B[ei] += ℒ.dot(A,e)
        ei += 1
    end
end


function fill_kron_adjoint_∂B!(∂X::AbstractSparseMatrix{R}, ∂B::AbstractArray{S}, A::AbstractMatrix{T}) where {R <: Real, S <: Real, T <: Real}
    @assert length(∂X) == length(∂B) * length(A) "∂X must have the same length as kron(B,A)"
    
    n1, m1 = size(∂B)
    n2, m2 = size(A)
    
    # Access the sparse matrix internal representation
    colptr = ∂X.colptr  # Column pointers
    rowval = ∂X.rowval  # Row indices of non-zeros
    nzval  = ∂X.nzval   # Non-zero values
    
    # Iterate over columns of ∂X
    for col in 1:size(∂X, 2)
        # Iterate over the non-zeros in this column
        for idx in colptr[col]:(colptr[col + 1] - 1)
            row = rowval[idx]
            val = nzval[idx]

            @inbounds begin
                i = (row - 1) ÷ n2 + 1
                k = (row - 1) % n2 + 1
                j = (col - 1) ÷ m2 + 1
                l = (col - 1) % m2 + 1
                
                # Update ∂B and ∂A
                ∂B[i,j] += A[k,l] * val
            end
        end
    end
end



function fill_kron_adjoint_∂B!(∂X::AbstractSparseMatrix{R}, ∂B::Vector{S}, A::AbstractMatrix{T}) where {R <: Real, S <: Real, T <: Real}
    @assert length(∂X) == length(∂B) * length(A) "∂X must have the same length as kron(B,A)"
    
    n1 = length(∂B)
    n2 = size(A,1)
    # println("hello")
    # Precompute constants
    const_n1n2 = n1 * n2
    
    # Access the sparse matrix internal representation
    colptr = ∂X.colptr  # Column pointers
    rowval = ∂X.rowval  # Row indices of non-zeros
    nzval  = ∂X.nzval   # Non-zero values
    
    # Iterate over columns of ∂X
    for col in 1:size(∂X, 2)
        # Iterate over the non-zeros in this column
        for idx in colptr[col]:(colptr[col + 1] - 1)
            row = rowval[idx]
            val = nzval[idx]

            linear_idx = (col - 1) * size(∂X, 1) + row

            @inbounds begin
                i = (linear_idx - 1) % n1 + 1
                k = ((linear_idx - 1) ÷ n1) % n2 + 1
                l = ((linear_idx - 1) ÷ const_n1n2) + 1
                
                # Update ∂B and ∂A
                ∂B[i] += A[k,l] * val
            end
        end
    end
end



function fill_kron_adjoint_∂B!(∂X::DenseMatrix{R}, ∂B::Vector{S}, A::AbstractMatrix{T}) where {R <: Real, S <: Real, T <: Real}
    @assert length(∂X) == length(∂B) * length(A) "∂X must have the same length as kron(B,A)"
        
    re∂X = reshape(∂X, 
                    size(A,1), 
                    length(∂B), 
                    size(A,2))

    ei = 1
    for e in eachslice(re∂X; dims = 2)
        @inbounds ∂B[ei] += ℒ.dot(A,e)
        ei += 1
    end
end


function fill_kron_adjoint_∂A!(∂X::DenseMatrix{R}, ∂A::Vector{S}, B::AbstractMatrix{T}) where {R <: Real, S <: Real, T <: Real}
    @assert length(∂X) == length(∂A) * length(B) "∂X must have the same length as kron(B,A)"
        
    re∂X = reshape(∂X, 
                    length(∂A), 
                    size(B,1), 
                    size(B,2))

    ei = 1
    for e in eachslice(re∂X; dims = 1)
        @inbounds ∂A[ei] += ℒ.dot(B,e)
        ei += 1
    end
end


function fill_kron_adjoint_∂A!(∂X::AbstractSparseMatrix{R}, ∂A::AbstractVector{S}, B::AbstractMatrix{T}) where {R <: Real, S <: Real, T <: Real}
    @assert length(∂X) == length(B) * length(∂A) "∂X must have the same length as kron(B,A)"

    n1, m1 = size(B)
    n2 = length(∂A)

    const_n1n2 = n1 * n2

    colptr = ∂X.colptr
    rowval = ∂X.rowval
    nzval  = ∂X.nzval

    for col in 1:size(∂X, 2)
        for idx in colptr[col]:(colptr[col + 1] - 1)
            row = rowval[idx]
            val = nzval[idx]

            linear_idx = (col - 1) * size(∂X, 1) + row

            @inbounds begin
                i = (linear_idx - 1) % n1 + 1
                k = ((linear_idx - 1) ÷ n1) % n2 + 1
                j = ((linear_idx - 1) ÷ const_n1n2) % m1 + 1

                ∂A[k] += B[i, j] * val
            end
        end
    end
end


function fill_kron_adjoint_∂A!(∂X::AbstractSparseMatrix{R}, ∂A::AbstractMatrix{S}, B::AbstractMatrix{T}) where {R <: Real, S <: Real, T <: Real}
    @assert length(∂X) == length(B) * length(∂A) "∂X must have the same length as kron(B,A)"
    
    n1, m1 = size(B)
    n2 = size(∂A,1)
    
    # Precompute constants
    const_n1n2 = n1 * n2
    const_n1n2m1 = n1 * n2 * m1
    
    # Access the sparse matrix internal representation
    colptr = ∂X.colptr  # Column pointers
    rowval = ∂X.rowval  # Row indices of non-zeros
    nzval  = ∂X.nzval   # Non-zero values
    
    # Iterate over columns of ∂X
    for col in 1:size(∂X, 2)
        # Iterate over the non-zeros in this column
        for idx in colptr[col]:(colptr[col + 1] - 1)
            row = rowval[idx]
            val = nzval[idx]

            linear_idx = (col - 1) * size(∂X, 1) + row

            @inbounds begin
                i = (linear_idx - 1) % n1 + 1
                k = ((linear_idx - 1) ÷ n1) % n2 + 1
                j = ((linear_idx - 1) ÷ const_n1n2) % m1 + 1
                l = ((linear_idx - 1) ÷ const_n1n2m1) + 1
                
                # Update ∂B and ∂A
                ∂A[k,l] += B[i,j] * val
            end
        end
    end
end


# Fused operation: computes fill_kron_adjoint!(∂A, ∂B, M1*M2, A, B)
# without materializing the full product M1*M2.
# 
# M1*M2 has shape (n1*n2, m1*m2) where kron(B,A) has the same shape,
# B is (n1,m1) and A is (n2,m2).
#
# Processes column-blocks of M1*M2 to keep memory usage at O(n1*n2*block_size)
# instead of O(n1*n2*m1*m2).
function mul_fill_kron_adjoint!(∂A::AbstractMatrix{R},
                                ∂B::AbstractMatrix{R},
                                M1::AbstractMatrix,
                                M2::AbstractMatrix,
                                A::AbstractMatrix{TA},
                                B::AbstractMatrix{TB};
                                tol::Real = 0.0,
                                block::AbstractMatrix{R} = Matrix{R}(undef, size(M1, 1), 0)) where {R <: Real, TA <: Real, TB <: Real}
    n2, m2 = size(A)
    n1, m1 = size(B)

    @assert size(M1, 1) == n1 * n2 "M1 rows ($(size(M1,1))) must equal n1*n2 ($(n1*n2))"
    @assert size(M2, 2) == m1 * m2 "M2 cols ($(size(M2,2))) must equal m1*m2 ($(m1*m2))"
    @assert size(M1, 2) == size(M2, 1) "M1 cols ($(size(M1,2))) must equal M2 rows ($(size(M2,1)))"

    nrows = n1 * n2

    # Process one j-block at a time: columns (j-1)*m2+1 : j*m2
    # Each block produces a (nrows × m2) matrix, reshaped to (n2, n1, m2)
    if size(block, 1) == nrows && size(block, 2) >= m2
        blk = view(block, :, 1:m2)
    else
        blk = Matrix{R}(undef, nrows, m2)
    end

    @inbounds for j in 1:m1
        col_start = (j - 1) * m2 + 1
        col_end   = j * m2
        # blk = M1 * M2[:, col_start:col_end]  — shape (n1*n2, m2)
        ℒ.mul!(blk, M1, view(M2, :, col_start:col_end))

        # Reshape blk to (n2, n1, m2) for accumulation
        re_blk = reshape(blk, n2, n1, m2)

        # ∂A[:,l] += re_blk[:,i,l] * B[i,j] for all i → ∂A[:,l] += Σ_i B[i,j]*re_blk[:,i,l]
        # = re_blk[:,:,l] * B[:,j]
        for l in 1:m2
            slice_l = view(re_blk, :, :, l)  # (n2, n1)
            for i in 1:n1
                bij = B[i, j]
                if abs(bij) > tol
                    for k in 1:n2
                        ∂A[k, l] += bij * slice_l[k, i]
                    end
                end
            end
        end

        # ∂B[i,j] += Σ_{k,l} A[k,l] * re_blk[k,i,l] = Σ_l dot(A[:,l], re_blk[:,i,l])
        for i in 1:n1
            acc = zero(R)
            for l in 1:m2
                for k in 1:n2
                    acc += A[k, l] * re_blk[k, i, l]
                end
            end
            ∂B[i, j] += acc
        end
    end
end


# Sparse-factor variant: when A and B are sparse, exploit nzrange for dot products
function mul_fill_kron_adjoint!(∂A::AbstractMatrix{R},
                                ∂B::AbstractMatrix{R},
                                M1::AbstractMatrix,
                                M2::AbstractMatrix,
                                A::SparseMatrixCSC{TA, Int},
                                B::SparseMatrixCSC{TB, Int};
                                tol::Real = 0.0,
                                block::AbstractMatrix{R} = Matrix{R}(undef, size(M1, 1), 0)) where {R <: Real, TA <: Real, TB <: Real}
    n2, m2 = size(A)
    n1, m1 = size(B)

    @assert size(M1, 1) == n1 * n2
    @assert size(M2, 2) == m1 * m2
    @assert size(M1, 2) == size(M2, 1)

    nrows = n1 * n2

    if size(block, 1) == nrows && size(block, 2) >= m2
        blk = view(block, :, 1:m2)
    else
        blk = Matrix{R}(undef, nrows, m2)
    end

    B_colptr = B.colptr
    B_rowval = B.rowval
    B_nzval  = B.nzval
    A_colptr = A.colptr
    A_rowval = A.rowval
    A_nzval  = A.nzval

    @inbounds for j in 1:m1
        col_start = (j - 1) * m2 + 1
        col_end   = j * m2
        ℒ.mul!(blk, M1, view(M2, :, col_start:col_end))

        re_blk = reshape(blk, n2, n1, m2)

        # ∂A[k,l] += B[i,j] * re_blk[k,i,l] — only iterate nonzero B[i,j]
        b_start = B_colptr[j]
        b_stop  = B_colptr[j + 1] - 1
        for l in 1:m2
            for bidx in b_start:b_stop
                i   = B_rowval[bidx]
                bij = R(B_nzval[bidx])
                for k in 1:n2
                    ∂A[k, l] += bij * re_blk[k, i, l]
                end
            end
        end

        # ∂B[i,j] += Σ_{k,l} A[k,l] * re_blk[k,i,l] — only iterate nonzero A[k,l]
        for bidx in b_start:b_stop
            i = B_rowval[bidx]
            acc = zero(R)
            for l in 1:m2
                for aidx in A_colptr[l]:(A_colptr[l + 1] - 1)
                    k = A_rowval[aidx]
                    acc += R(A_nzval[aidx]) * re_blk[k, i, l]
                end
            end
            ∂B[i, j] += acc
        end
    end
end


# Mixed-sparsity variant: A is sparse, B is dense
function mul_fill_kron_adjoint!(∂A::AbstractMatrix{R},
                                ∂B::AbstractMatrix{R},
                                M1::AbstractMatrix,
                                M2::AbstractMatrix,
                                A::SparseMatrixCSC{TA, Int},
                                B::AbstractMatrix{TB};
                                tol::Real = 0.0,
                                block::AbstractMatrix{R} = Matrix{R}(undef, size(M1, 1), 0)) where {R <: Real, TA <: Real, TB <: Real}
    n2, m2 = size(A)
    n1, m1 = size(B)

    @assert size(M1, 1) == n1 * n2
    @assert size(M2, 2) == m1 * m2
    @assert size(M1, 2) == size(M2, 1)

    nrows = n1 * n2

    if size(block, 1) == nrows && size(block, 2) >= m2
        blk = view(block, :, 1:m2)
    else
        blk = Matrix{R}(undef, nrows, m2)
    end

    A_colptr = A.colptr
    A_rowval = A.rowval
    A_nzval  = A.nzval

    @inbounds for j in 1:m1
        col_start = (j - 1) * m2 + 1
        col_end   = j * m2
        ℒ.mul!(blk, M1, view(M2, :, col_start:col_end))

        re_blk = reshape(blk, n2, n1, m2)

        # ∂A[k,l] += B[i,j] * re_blk[k,i,l] — B is dense, use iszero guard
        for l in 1:m2
            for i in 1:n1
                bij = B[i, j]
                if abs(bij) > tol
                    for k in 1:n2
                        ∂A[k, l] += bij * re_blk[k, i, l]
                    end
                end
            end
        end

        # ∂B[i,j] += Σ_{k,l} A[k,l] * re_blk[k,i,l] — A is sparse, use nzrange
        for i in 1:n1
            acc = zero(R)
            for l in 1:m2
                for aidx in A_colptr[l]:(A_colptr[l + 1] - 1)
                    k = A_rowval[aidx]
                    acc += R(A_nzval[aidx]) * re_blk[k, i, l]
                end
            end
            ∂B[i, j] += acc
        end
    end
end


# Mixed-sparsity variant: A is dense, B is sparse
function mul_fill_kron_adjoint!(∂A::AbstractMatrix{R},
                                ∂B::AbstractMatrix{R},
                                M1::AbstractMatrix,
                                M2::AbstractMatrix,
                                A::AbstractMatrix{TA},
                                B::SparseMatrixCSC{TB, Int};
                                tol::Real = 0.0,
                                block::AbstractMatrix{R} = Matrix{R}(undef, size(M1, 1), 0)) where {R <: Real, TA <: Real, TB <: Real}
    n2, m2 = size(A)
    n1, m1 = size(B)

    @assert size(M1, 1) == n1 * n2
    @assert size(M2, 2) == m1 * m2
    @assert size(M1, 2) == size(M2, 1)

    nrows = n1 * n2

    if size(block, 1) == nrows && size(block, 2) >= m2
        blk = view(block, :, 1:m2)
    else
        blk = Matrix{R}(undef, nrows, m2)
    end

    B_colptr = B.colptr
    B_rowval = B.rowval
    B_nzval  = B.nzval

    @inbounds for j in 1:m1
        col_start = (j - 1) * m2 + 1
        col_end   = j * m2
        ℒ.mul!(blk, M1, view(M2, :, col_start:col_end))

        re_blk = reshape(blk, n2, n1, m2)

        # ∂A[k,l] += B[i,j] * re_blk[k,i,l] — B is sparse, only iterate nonzero B[i,j]
        b_start = B_colptr[j]
        b_stop  = B_colptr[j + 1] - 1
        for l in 1:m2
            for bidx in b_start:b_stop
                i   = B_rowval[bidx]
                bij = R(B_nzval[bidx])
                for k in 1:n2
                    ∂A[k, l] += bij * re_blk[k, i, l]
                end
            end
        end

        # ∂B[i,j] += Σ_{k,l} A[k,l] * re_blk[k,i,l] — iterate all i (∂B is dense)
        for i in 1:n1
            acc = zero(R)
            for l in 1:m2
                for k in 1:n2
                    akl = A[k, l]
                    if abs(akl) > tol
                        acc += akl * re_blk[k, i, l]
                    end
                end
            end
            ∂B[i, j] += acc
        end
    end
end


# Variant that computes fill_kron_adjoint_∂A! for both the identity and a permuted
# version of ∂X in a single sparse iteration pass.
#
# Equivalent to:
#   fill_kron_adjoint_∂A!(∂X, ∂A, B)
#   fill_kron_adjoint_∂A!(Pₗ * ∂X * Pᵣ, ∂A, B)
# but avoids materializing the permuted matrix.
#
# perm_row and perm_col are integer vectors representing the row/column permutations
# such that (Pₗ * ∂X * Pᵣ)[perm_row[row], perm_col[col]] = ∂X[row, col].
# Accumulates the ∂A adjoint from ∂X + P₁ₗ * ∂X * P₁ᵣ where ∂X is the cotangent
# of kron(B, A) and P₁ is the (2,1,3) tensor-axis swap on the d³ row/column space
# (d = n_A = size(∂A,1)).  The permutation is baked in — no external vectors needed.
#
# Requires n_B = n_A² and m_B = m_A² (i.e. B is the d²×d² outer factor).
function fill_kron_adjoint_∂A_with_perm!(∂X::AbstractSparseMatrix{R},
                                          ∂A::AbstractMatrix{S},
                                          B::AbstractMatrix{T}) where {R <: Real, S <: Real, T <: Real}
    @assert length(∂X) == length(∂A) * length(B) "∂X must have the same length as kron(B,A)"

    # Convention: kron(B, A) — A is inner (fastest-varying), B is outer
    # Same decomposition as fill_kron_adjoint! reshape(∂X, n_A, n_B, m_A, m_B)
    n_A = size(∂A, 1)
    n_B = size(B, 1)
    m_A = size(∂A, 2)

    @assert n_B == n_A * n_A "fill_kron_adjoint_∂A_with_perm! requires n_B == n_A² for the (2,1,3) axis swap"
    @assert size(B, 2) == m_A * m_A "fill_kron_adjoint_∂A_with_perm! requires m_B == m_A² for the (2,1,3) axis swap"

    const_nAnB   = n_A * n_B
    const_nAnBmA = n_A * n_B * m_A
    nrows = size(∂X, 1)

    colptr = ∂X.colptr
    rowval = ∂X.rowval
    nzval  = ∂X.nzval

    @inbounds for col in 1:size(∂X, 2)
        for idx in colptr[col]:(colptr[col + 1] - 1)
            row = rowval[idx]
            val = nzval[idx]

            # --- Identity contribution (linear-index decomposition) ---
            L = (col - 1) * nrows + row - 1
            i_A = L % n_A + 1
            i_B = (L ÷ n_A) % n_B + 1
            j_A = (L ÷ const_nAnB) % m_A + 1
            j_B = (L ÷ const_nAnBmA) + 1
            ∂A[i_A, j_A] += B[i_B, j_B] * val

            # --- (2,1,3) axis-swap contribution ---
            # The outer index i_B (1-based) encodes two sub-axes of size n_A:
            #   k₂ = (i_B-1) % n_A,  k₃ = (i_B-1) ÷ n_A
            # Swapping axis 1 (i_A) with axis 2 (k₂) gives:
            i_Ap = (i_B - 1) % n_A + 1
            i_Bp = (i_A - 1) + ((i_B - 1) ÷ n_A) * n_A + 1
            j_Ap = (j_B - 1) % m_A + 1
            j_Bp = (j_A - 1) + ((j_B - 1) ÷ m_A) * m_A + 1
            ∂A[i_Ap, j_Ap] += B[i_Bp, j_Bp] * val
        end
    end
end


# Fused variant of fill_kron_adjoint_∂A_with_perm! that processes M1 * M2
# in column blocks without materializing the full product.
#
# Equivalent to:
#   fill_kron_adjoint_∂A_with_perm!(sparse(M1 * M2), ∂A, B)
# but avoids allocating the (n_A³ × m_A³) intermediate.
#
# Requires n_B = n_A² and m_B = m_A² (same as fill_kron_adjoint_∂A_with_perm!).
function mul_fill_kron_adjoint_∂A_with_perm!(M1::AbstractMatrix,
                                              M2::AbstractMatrix,
                                              ∂A::AbstractMatrix{S},
                                              B::AbstractMatrix{T};
                                              block::AbstractMatrix{S} = Matrix{S}(undef, size(M1, 1), 0)) where {S <: Real, T <: Real}
    n_A = size(∂A, 1)
    m_A = size(∂A, 2)
    n_B = size(B, 1)
    m_B = size(B, 2)

    @assert n_B == n_A * n_A "mul_fill_kron_adjoint_∂A_with_perm! requires n_B == n_A²"
    @assert m_B == m_A * m_A "mul_fill_kron_adjoint_∂A_with_perm! requires m_B == m_A²"
    @assert size(M1, 1) == n_A * n_B "M1 rows ($(size(M1,1))) must equal n_A * n_B ($(n_A * n_B))"
    @assert size(M2, 2) == m_A * m_B "M2 cols ($(size(M2,2))) must equal m_A * m_B ($(m_A * m_B))"
    @assert size(M1, 2) == size(M2, 1) "M1 cols ($(size(M1,2))) must equal M2 rows ($(size(M2,1)))"

    nrows = n_A * n_B  # = n_A³

    if size(block, 1) == nrows && size(block, 2) >= m_A
        blk = view(block, :, 1:m_A)
    else
        blk = Matrix{S}(undef, nrows, m_A)
    end

    @inbounds for j in 1:m_B  # j = j_B (outer column index of B)
        col_start = (j - 1) * m_A + 1
        col_end   = j * m_A
        ℒ.mul!(blk, M1, view(M2, :, col_start:col_end))

        # Pre-compute the fixed permuted column index for j_B = j
        # (2,1,3) axis swap: j_Ap depends only on j, not on j_A
        j_Ap_fixed = (j - 1) % m_A + 1

        for j_A in 1:m_A
            # (2,1,3) axis swap: j_Bp depends on both j_A and j
            j_Bp = (j_A - 1) + ((j - 1) ÷ m_A) * m_A + 1

            for row in 1:nrows
                val = blk[row, j_A]

                # Decompose row into (i_A, i_B) for kron(B, A) convention
                i_A = (row - 1) % n_A + 1
                i_B = (row - 1) ÷ n_A + 1

                # Identity contribution
                ∂A[i_A, j_A] += B[i_B, j] * val

                # (2,1,3) axis-swap contribution
                i_Ap = (i_B - 1) % n_A + 1
                i_Bp = (i_A - 1) + ((i_B - 1) ÷ n_A) * n_A + 1
                ∂A[i_Ap, j_Ap_fixed] += B[i_Bp, j_Bp] * val
            end
        end
    end
end


# Sparse-B variant of mul_fill_kron_adjoint_∂A_with_perm! that exploits B's sparsity.
# When B is ultra-sparse (e.g. σ with ~nₑ nonzeros in nₑ₋² × nₑ₋²),
# this skips ~99.7% of work by iterating only nzrange columns.
function mul_fill_kron_adjoint_∂A_with_perm!(M1::AbstractMatrix,
                                              M2::AbstractMatrix,
                                              ∂A::AbstractMatrix{S},
                                              B::SparseMatrixCSC{TB, Int};
                                              block::AbstractMatrix{S} = Matrix{S}(undef, size(M1, 1), 0)) where {S <: Real, TB <: Real}
    n_A = size(∂A, 1)
    m_A = size(∂A, 2)
    n_B = size(B, 1)
    m_B = size(B, 2)

    @assert n_B == n_A * n_A "mul_fill_kron_adjoint_∂A_with_perm! requires n_B == n_A²"
    @assert m_B == m_A * m_A "mul_fill_kron_adjoint_∂A_with_perm! requires m_B == m_A²"
    @assert size(M1, 1) == n_A * n_B "M1 rows ($(size(M1,1))) must equal n_A * n_B ($(n_A * n_B))"
    @assert size(M2, 2) == m_A * m_B "M2 cols ($(size(M2,2))) must equal m_A * m_B ($(m_A * m_B))"
    @assert size(M1, 2) == size(M2, 1) "M1 cols ($(size(M1,2))) must equal M2 rows ($(size(M2,1)))"

    nrows = n_A * n_B  # = n_A³

    B_colptr = B.colptr
    B_rowval = SparseArrays.rowvals(B)
    B_nzval  = nonzeros(B)

    # Precompute which B columns have nonzeros for fast skip checks
    has_nz = falses(m_B)
    @inbounds for col in 1:m_B
        has_nz[col] = B_colptr[col] < B_colptr[col + 1]
    end

    if size(block, 1) == nrows && size(block, 2) >= m_A
        blk = view(block, :, 1:m_A)
    else
        blk = Matrix{S}(undef, nrows, m_A)
    end

    @inbounds for j in 1:m_B  # j = j_B (outer column index of B)
        # Check if this j contributes anything:
        # Identity path: B[:,j] has nonzeros
        # Perm path: for each j_A, B[:, j_Bp(j_A, j)] has nonzeros
        need_blk = has_nz[j]
        if !need_blk
            j_div = (j - 1) ÷ m_A
            for j_A in 1:m_A
                j_Bp = (j_A - 1) + j_div * m_A + 1
                if has_nz[j_Bp]
                    need_blk = true
                    break
                end
            end
        end
        need_blk || continue

        col_start = (j - 1) * m_A + 1
        col_end   = j * m_A
        ℒ.mul!(blk, M1, view(M2, :, col_start:col_end))

        # Pre-compute for (2,1,3) axis swap
        j_Ap_fixed = (j - 1) % m_A + 1
        j_div = (j - 1) ÷ m_A

        # Identity contribution: iterate nonzeros of B[:, j]
        for bidx in B_colptr[j]:(B_colptr[j + 1] - 1)
            i_B = B_rowval[bidx]
            b_val = S(B_nzval[bidx])
            # i_A = (row-1) % n_A + 1 for row = (i_B-1)*n_A + 1 : i_B*n_A
            row_start = (i_B - 1) * n_A
            for j_A in 1:m_A
                for i_A in 1:n_A
                    ∂A[i_A, j_A] += b_val * blk[row_start + i_A, j_A]
                end
            end
        end

        # (2,1,3) axis-swap contribution: for each j_A, iterate nonzeros of B[:, j_Bp]
        for j_A in 1:m_A
            j_Bp = (j_A - 1) + j_div * m_A + 1
            for bidx in B_colptr[j_Bp]:(B_colptr[j_Bp + 1] - 1)
                i_Bp = B_rowval[bidx]
                b_val = S(B_nzval[bidx])
                # Reverse-map: i_Ap = (i_B-1) % n_A + 1, but here i_Bp encodes
                # i_Bp = (i_A-1) + ((i_B-1) ÷ n_A) * n_A + 1
                # So: i_A = (i_Bp-1) % n_A + 1, block_offset = (i_Bp-1) ÷ n_A
                i_A = (i_Bp - 1) % n_A + 1
                block_k3 = (i_Bp - 1) ÷ n_A  # = (i_B-1) ÷ n_A = k₃ - 1

                # The identity i_Ap = (i_B-1) % n_A + 1 = k₂
                # and row = (i_B-1)*n_A + i_A  where i_B = k₂ + k₃*n_A + 1
                # We need to iterate over all k₂ (= i_Ap's corresponding i_B values)
                # For a given i_Bp, we have i_A and block_k3 fixed.
                # i_Ap = k₂ + 1 ranges over 1:n_A, with i_B = k₂ + block_k3*n_A + 1
                # and row = (i_B-1)*n_A + i_A = (k₂ + block_k3*n_A)*n_A + i_A
                for k2 in 0:(n_A - 1)
                    i_Ap = k2 + 1
                    row = (k2 + block_k3 * n_A) * n_A + i_A
                    ∂A[i_Ap, j_Ap_fixed] += b_val * blk[row, j_A]
                end
            end
        end
    end
end

# Dead code: compressed_kron_pullback_2arg! — never called anywhere (2-arg compressed_kron is also dead)
#=
# Helper: adjoint of compressed_kron(A, σ; tol) w.r.t. A and σ.
# Forward contribution for each sorted output column triple (α≥β≥γ) is:
#   Y[row,col] += A[i,α] * σ[(j-1)*nᵣ+k, (β-1)*nᶜ+γ]
# where row is obtained by sorting (i,j,k) into i₁≥j₁≥k₁.
function compressed_kron_pullback_2arg!(∂A::AbstractMatrix{T},
                                        ∂σ::AbstractMatrix{T},
                                        ∂Y::AbstractMatrix{T},
                                        A::AbstractMatrix{TA},
                                        σ::AbstractMatrix{Tσ};
                                        tol::AbstractFloat = eps()) where {T <: Real, TA <: Real, Tσ <: Real}

    nᵣ, nᶜ = size(A)
    size(σ) == (nᵣ^2, nᶜ^2) || throw(DimensionMismatch("σ must be $(nᵣ^2)×$(nᶜ^2), got $(size(σ))"))

    As = A isa SparseMatrixCSC ? A : sparse(A)
    σs = σ isa SparseMatrixCSC ? σ : sparse(σ)

    rv_A = SparseArrays.rowvals(As)
    nzv_A = nonzeros(As)
    rv_σ = SparseArrays.rowvals(σs)
    nzv_σ = nonzeros(σs)

    ranges_A = Vector{UnitRange{Int}}(undef, nᶜ)
    ranges_σ = Vector{UnitRange{Int}}(undef, nᶜ^2)
    @inbounds for col in 1:nᶜ
        ranges_A[col] = SparseArrays.nzrange(As, col)
    end
    @inbounds for col in 1:(nᶜ^2)
        ranges_σ[col] = SparseArrays.nzrange(σs, col)
    end

    @inbounds for α in 1:nᶜ
        rng_A = ranges_A[α]
        isempty(rng_A) && continue

        for β in 1:α
            for γ in 1:β
                σ_col = (β - 1) * nᶜ + γ
                rng_σ = ranges_σ[σ_col]
                isempty(rng_σ) && continue

                col = (α - 1) * α * (α + 1) ÷ 6 + (β - 1) * β ÷ 2 + γ

                for pA in rng_A
                    i = rv_A[pA]
                    a_val = nzv_A[pA]

                    for pσ in rng_σ
                        s = rv_σ[pσ]
                        σ_val = nzv_σ[pσ]

                        val = a_val * σ_val
                        abs(val) > tol || continue

                        j = (s - 1) ÷ nᵣ + 1
                        k = (s - 1) % nᵣ + 1

                        i₁ = i; j₁ = j; k₁ = k
                        if i₁ < j₁; i₁, j₁ = j₁, i₁; end
                        if j₁ < k₁; j₁, k₁ = k₁, j₁; end
                        if i₁ < j₁; i₁, j₁ = j₁, i₁; end

                        row = (i₁ - 1) * i₁ * (i₁ + 1) ÷ 6 + (j₁ - 1) * j₁ ÷ 2 + k₁
                        g = ∂Y[row, col]
                        abs(g) <= tol && continue

                        ∂A[i, α] += g * σ_val
                        ∂σ[s, σ_col] += g * a_val
                    end
                end
            end
        end
    end

    return
end
=#

# Helper: adjoint of compressed_permuted_mixed_kron(A, σ; tol) w.r.t. A and σ.
function compressed_permuted_mixed_kron_pullback!(∂A::AbstractMatrix{T},
                                                  ∂σ::AbstractMatrix{T},
                                                  ∂Y::AbstractMatrix{T},
                                                  A::AbstractMatrix{TA},
                                                  σ::AbstractMatrix{Tσ};
                                                  tol::AbstractFloat = eps()) where {T <: Real, TA <: Real, Tσ <: Real}

    nr, nc = size(A)
    size(σ) == (nr^2, nc^2) || throw(DimensionMismatch("σ must be $(nr^2)×$(nc^2), got $(size(σ))"))

    As = A isa SparseMatrixCSC ? A : sparse(A)
    σs = σ isa SparseMatrixCSC ? σ : sparse(σ)

    rv_A = SparseArrays.rowvals(As)
    nzv_A = nonzeros(As)
    rv_σ = SparseArrays.rowvals(σs)
    nzv_σ = nonzeros(σs)

    ranges_A = Vector{UnitRange{Int}}(undef, nc)
    ranges_σ = Vector{UnitRange{Int}}(undef, nc^2)
    @inbounds for col in 1:nc
        ranges_A[col] = SparseArrays.nzrange(As, col)
    end
    @inbounds for col in 1:(nc^2)
        ranges_σ[col] = SparseArrays.nzrange(σs, col)
    end

    G = Matrix(∂Y)

    @inbounds for α in 1:nc
        rng_Aα = ranges_A[α]
        for β in 1:α
            rng_Aβ = ranges_A[β]
            for γ in 1:β
                rng_Aγ = ranges_A[γ]

                σ_col_βγ = (β - 1) * nc + γ
                σ_col_αγ = (α - 1) * nc + γ
                σ_col_αβ = (α - 1) * nc + β

                rng_σβγ = ranges_σ[σ_col_βγ]
                rng_σαγ = ranges_σ[σ_col_αγ]
                rng_σαβ = ranges_σ[σ_col_αβ]

                has_t1 = !isempty(rng_Aα) && !isempty(rng_σβγ)
                has_t2 = !isempty(rng_Aβ) && !isempty(rng_σαγ)
                has_t3 = !isempty(rng_Aγ) && !isempty(rng_σαβ)

                (has_t1 || has_t2 || has_t3) || continue

                col = (α - 1) * α * (α + 1) ÷ 6 + (β - 1) * β ÷ 2 + γ

                if has_t1
                    for ia in rng_Aα
                        p = rv_A[ia]
                        a_val = nzv_A[ia]
                        for is in rng_σβγ
                            qr = rv_σ[is]
                            q = (qr - 1) ÷ nr + 1
                            r = qr - (q - 1) * nr

                            i1 = p
                            j1 = q
                            k1 = r
                            if i1 < j1
                                i1, j1 = j1, i1
                            end
                            if j1 < k1
                                j1, k1 = k1, j1
                            end
                            if i1 < j1
                                i1, j1 = j1, i1
                            end

                            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1
                            g = G[row, col]
                            abs(g) <= tol && continue

                            σ_val = nzv_σ[is]
                            ∂A[p, α] += g * σ_val
                            ∂σ[qr, σ_col_βγ] += g * a_val
                        end
                    end
                end

                if has_t2
                    for ia in rng_Aβ
                        q = rv_A[ia]
                        a_val = nzv_A[ia]
                        for is in rng_σαγ
                            pr = rv_σ[is]
                            p = (pr - 1) ÷ nr + 1
                            r = pr - (p - 1) * nr

                            i1 = p
                            j1 = q
                            k1 = r
                            if i1 < j1
                                i1, j1 = j1, i1
                            end
                            if j1 < k1
                                j1, k1 = k1, j1
                            end
                            if i1 < j1
                                i1, j1 = j1, i1
                            end

                            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1
                            g = G[row, col]
                            abs(g) <= tol && continue

                            σ_val = nzv_σ[is]
                            ∂A[q, β] += g * σ_val
                            ∂σ[pr, σ_col_αγ] += g * a_val
                        end
                    end
                end

                if has_t3
                    for ia in rng_Aγ
                        r = rv_A[ia]
                        a_val = nzv_A[ia]
                        for is in rng_σαβ
                            pq = rv_σ[is]
                            p = (pq - 1) ÷ nr + 1
                            q = pq - (p - 1) * nr

                            i1 = p
                            j1 = q
                            k1 = r
                            if i1 < j1
                                i1, j1 = j1, i1
                            end
                            if j1 < k1
                                j1, k1 = k1, j1
                            end
                            if i1 < j1
                                i1, j1 = j1, i1
                            end

                            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1
                            g = G[row, col]
                            abs(g) <= tol && continue

                            σ_val = nzv_σ[is]
                            ∂A[r, γ] += g * σ_val
                            ∂σ[pq, σ_col_αβ] += g * a_val
                        end
                    end
                end
            end
        end
    end

    return
end


# ∂A-only variant: skips ∂σ accumulation (matches fill_kron_adjoint_∂A! pattern).
# Use when the ∂σ output is discarded (e.g. B-pullback for Sylvester).
function compressed_permuted_mixed_kron_pullback_∂A!(∂A::AbstractMatrix{T},
                                                     ∂Y::AbstractMatrix{T},
                                                     A::AbstractMatrix{TA},
                                                     σ::AbstractMatrix{Tσ};
                                                     tol::AbstractFloat = eps()) where {T <: Real, TA <: Real, Tσ <: Real}

    nr, nc = size(A)
    size(σ) == (nr^2, nc^2) || throw(DimensionMismatch("σ must be $(nr^2)×$(nc^2), got $(size(σ))"))

    As = A isa SparseMatrixCSC ? A : sparse(A)
    σs = σ isa SparseMatrixCSC ? σ : sparse(σ)

    rv_A = SparseArrays.rowvals(As)
    nzv_A = nonzeros(As)
    rv_σ = SparseArrays.rowvals(σs)
    nzv_σ = nonzeros(σs)

    ranges_A = Vector{UnitRange{Int}}(undef, nc)
    ranges_σ = Vector{UnitRange{Int}}(undef, nc^2)
    @inbounds for col in 1:nc
        ranges_A[col] = SparseArrays.nzrange(As, col)
    end
    @inbounds for col in 1:(nc^2)
        ranges_σ[col] = SparseArrays.nzrange(σs, col)
    end

    G = Matrix(∂Y)

    @inbounds for α in 1:nc
        rng_Aα = ranges_A[α]
        for β in 1:α
            rng_Aβ = ranges_A[β]
            for γ in 1:β
                rng_Aγ = ranges_A[γ]

                σ_col_βγ = (β - 1) * nc + γ
                σ_col_αγ = (α - 1) * nc + γ
                σ_col_αβ = (α - 1) * nc + β

                rng_σβγ = ranges_σ[σ_col_βγ]
                rng_σαγ = ranges_σ[σ_col_αγ]
                rng_σαβ = ranges_σ[σ_col_αβ]

                has_t1 = !isempty(rng_Aα) && !isempty(rng_σβγ)
                has_t2 = !isempty(rng_Aβ) && !isempty(rng_σαγ)
                has_t3 = !isempty(rng_Aγ) && !isempty(rng_σαβ)

                (has_t1 || has_t2 || has_t3) || continue

                col = (α - 1) * α * (α + 1) ÷ 6 + (β - 1) * β ÷ 2 + γ

                if has_t1
                    for ia in rng_Aα
                        p = rv_A[ia]
                        for is in rng_σβγ
                            qr = rv_σ[is]
                            q = (qr - 1) ÷ nr + 1
                            r = qr - (q - 1) * nr

                            i1 = p
                            j1 = q
                            k1 = r
                            if i1 < j1
                                i1, j1 = j1, i1
                            end
                            if j1 < k1
                                j1, k1 = k1, j1
                            end
                            if i1 < j1
                                i1, j1 = j1, i1
                            end

                            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1
                            g = G[row, col]
                            abs(g) <= tol && continue

                            σ_val = nzv_σ[is]
                            ∂A[p, α] += g * σ_val
                        end
                    end
                end

                if has_t2
                    for ia in rng_Aβ
                        q = rv_A[ia]
                        for is in rng_σαγ
                            pr = rv_σ[is]
                            p = (pr - 1) ÷ nr + 1
                            r = pr - (p - 1) * nr

                            i1 = p
                            j1 = q
                            k1 = r
                            if i1 < j1
                                i1, j1 = j1, i1
                            end
                            if j1 < k1
                                j1, k1 = k1, j1
                            end
                            if i1 < j1
                                i1, j1 = j1, i1
                            end

                            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1
                            g = G[row, col]
                            abs(g) <= tol && continue

                            σ_val = nzv_σ[is]
                            ∂A[q, β] += g * σ_val
                        end
                    end
                end

                if has_t3
                    for ia in rng_Aγ
                        r = rv_A[ia]
                        for is in rng_σαβ
                            pq = rv_σ[is]
                            p = (pq - 1) ÷ nr + 1
                            q = pq - (p - 1) * nr

                            i1 = p
                            j1 = q
                            k1 = r
                            if i1 < j1
                                i1, j1 = j1, i1
                            end
                            if j1 < k1
                                j1, k1 = k1, j1
                            end
                            if i1 < j1
                                i1, j1 = j1, i1
                            end

                            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1
                            g = G[row, col]
                            abs(g) <= tol && continue

                            σ_val = nzv_σ[is]
                            ∂A[r, γ] += g * σ_val
                        end
                    end
                end
            end
        end
    end

    return
end


# Fused variant: computes g_col = M1 * M2[:, col] lazily per (α,β,γ) triple
# instead of materializing the full ∂Y = M1 * M2 matrix.
# Equivalent to:
#   compressed_permuted_mixed_kron_pullback!(∂A, ∂σ, M1 * M2, A, σ; tol)
# but avoids the n_compressed³ × n_compressed³ allocation.
function mul_compressed_permuted_mixed_kron_pullback!(∂A::AbstractMatrix{T},
                                                      ∂σ::AbstractMatrix{T},
                                                      M1::AbstractMatrix,
                                                      M2::AbstractMatrix,
                                                      A::AbstractMatrix{TA},
                                                      σ::AbstractMatrix{Tσ};
                                                      tol::AbstractFloat = eps()) where {T <: Real, TA <: Real, Tσ <: Real}

    nr, nc = size(A)
    size(σ) == (nr^2, nc^2) || throw(DimensionMismatch("σ must be $(nr^2)×$(nc^2), got $(size(σ))"))

    As = A isa SparseMatrixCSC ? A : sparse(A)
    σs = σ isa SparseMatrixCSC ? σ : sparse(σ)

    rv_A = SparseArrays.rowvals(As)
    nzv_A = nonzeros(As)
    rv_σ = SparseArrays.rowvals(σs)
    nzv_σ = nonzeros(σs)

    ranges_A = Vector{UnitRange{Int}}(undef, nc)
    ranges_σ = Vector{UnitRange{Int}}(undef, nc^2)
    @inbounds for col in 1:nc
        ranges_A[col] = SparseArrays.nzrange(As, col)
    end
    @inbounds for col in 1:(nc^2)
        ranges_σ[col] = SparseArrays.nzrange(σs, col)
    end

    g_col = Vector{T}(undef, size(M1, 1))

    @inbounds for α in 1:nc
        rng_Aα = ranges_A[α]
        for β in 1:α
            rng_Aβ = ranges_A[β]
            for γ in 1:β
                rng_Aγ = ranges_A[γ]

                σ_col_βγ = (β - 1) * nc + γ
                σ_col_αγ = (α - 1) * nc + γ
                σ_col_αβ = (α - 1) * nc + β

                rng_σβγ = ranges_σ[σ_col_βγ]
                rng_σαγ = ranges_σ[σ_col_αγ]
                rng_σαβ = ranges_σ[σ_col_αβ]

                has_t1 = !isempty(rng_Aα) && !isempty(rng_σβγ)
                has_t2 = !isempty(rng_Aβ) && !isempty(rng_σαγ)
                has_t3 = !isempty(rng_Aγ) && !isempty(rng_σαβ)

                (has_t1 || has_t2 || has_t3) || continue

                col = (α - 1) * α * (α + 1) ÷ 6 + (β - 1) * β ÷ 2 + γ

                # Compute g_col = M1 * M2[:, col] lazily for this triple
                ℒ.mul!(g_col, M1, view(M2, :, col))

                if has_t1
                    for ia in rng_Aα
                        p = rv_A[ia]
                        a_val = nzv_A[ia]
                        for is in rng_σβγ
                            qr = rv_σ[is]
                            q = (qr - 1) ÷ nr + 1
                            r = qr - (q - 1) * nr

                            i1 = p
                            j1 = q
                            k1 = r
                            if i1 < j1
                                i1, j1 = j1, i1
                            end
                            if j1 < k1
                                j1, k1 = k1, j1
                            end
                            if i1 < j1
                                i1, j1 = j1, i1
                            end

                            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1
                            g = g_col[row]
                            abs(g) <= tol && continue

                            σ_val = nzv_σ[is]
                            ∂A[p, α] += g * σ_val
                            ∂σ[qr, σ_col_βγ] += g * a_val
                        end
                    end
                end

                if has_t2
                    for ia in rng_Aβ
                        q = rv_A[ia]
                        a_val = nzv_A[ia]
                        for is in rng_σαγ
                            pr = rv_σ[is]
                            p = (pr - 1) ÷ nr + 1
                            r = pr - (p - 1) * nr

                            i1 = p
                            j1 = q
                            k1 = r
                            if i1 < j1
                                i1, j1 = j1, i1
                            end
                            if j1 < k1
                                j1, k1 = k1, j1
                            end
                            if i1 < j1
                                i1, j1 = j1, i1
                            end

                            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1
                            g = g_col[row]
                            abs(g) <= tol && continue

                            σ_val = nzv_σ[is]
                            ∂A[q, β] += g * σ_val
                            ∂σ[pr, σ_col_αγ] += g * a_val
                        end
                    end
                end

                if has_t3
                    for ia in rng_Aγ
                        r = rv_A[ia]
                        a_val = nzv_A[ia]
                        for is in rng_σαβ
                            pq = rv_σ[is]
                            p = (pq - 1) ÷ nr + 1
                            q = pq - (p - 1) * nr

                            i1 = p
                            j1 = q
                            k1 = r
                            if i1 < j1
                                i1, j1 = j1, i1
                            end
                            if j1 < k1
                                j1, k1 = k1, j1
                            end
                            if i1 < j1
                                i1, j1 = j1, i1
                            end

                            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1
                            g = g_col[row]
                            abs(g) <= tol && continue

                            σ_val = nzv_σ[is]
                            ∂A[r, γ] += g * σ_val
                            ∂σ[pq, σ_col_αβ] += g * a_val
                        end
                    end
                end
            end
        end
    end

    return
end


# Helper: adjoint of compressed_kron²(X; rowmask, colmask) w.r.t. X.
# Forward value at (row(i1,j1), col(i2,j2)): (X[i1,i2]*X[j1,j2] + X[i1,j2]*X[j1,i2]) / divisor,
# where divisor = 2 if i1 == j1 else 1, and only masked rows/cols are materialized.
function compressed_kron²_pullback!(∂X::AbstractMatrix{T},
                                    ∂Y::AbstractMatrix{T},
                                    X::AbstractMatrix{T};
                                    tol::Real = 0.0,
                                    rowmask::Vector{Int} = Int[],
                                    colmask::Vector{Int} = Int[]) where T <: Real
    Xd = X isa DenseMatrix ? X : collect(X)
    n_rows, n_cols = size(Xd)

    m2_rows = n_rows * (n_rows + 1) ÷ 2
    m2_cols = n_cols * (n_cols + 1) ÷ 2

    if rowmask == Int[0] || colmask == Int[0]
        return
    end

    norowmask = length(rowmask) == 0
    nocolmask = length(colmask) == 0
    rowmask_lookup = norowmask ? BitVector() : falses(m2_rows)
    colmask_lookup = nocolmask ? BitVector() : falses(m2_cols)

    if !norowmask
        @inbounds for r in rowmask
            if 1 <= r <= m2_rows
                rowmask_lookup[r] = true
            end
        end
    end

    if !nocolmask
        @inbounds for c in colmask
            if 1 <= c <= m2_cols
                colmask_lookup[c] = true
            end
        end
    end

    for i1 in 1:n_rows, j1 in 1:i1
        row = (i1 - 1) * i1 ÷ 2 + j1
        (norowmask || rowmask_lookup[row]) || continue
        divisor = i1 == j1 ? 2 : 1

        for i2 in 1:n_cols, j2 in 1:i2
            col = (i2 - 1) * i2 ÷ 2 + j2
            (nocolmask || colmask_lookup[col]) || continue

            g = ∂Y[row, col]
            abs(g) <= tol && continue
            g_d = g / divisor

            @inbounds aii = Xd[i1, i2]
            @inbounds aij = Xd[i1, j2]
            @inbounds aji = Xd[j1, i2]
            @inbounds ajj = Xd[j1, j2]

            ∂X[i1, i2] += g_d * ajj
            ∂X[j1, j2] += g_d * aii
            ∂X[i1, j2] += g_d * aji
            ∂X[j1, i2] += g_d * aij
        end
    end
end


# Helper: adjoint of compressed_kron³(X) w.r.t. X.
# Forward: out[row,col] = (aii*(ajj*akk + ajk*akj) + aij*(aji*akk + ajk*aki) + aik*(aji*akj + ajj*aki)) / divisor
# where row ↔ (i1≥j1≥k1) and col ↔ (i2≥j2≥k2) and a_pq = X[p,q].
function compressed_kron³_pullback!(∂X::AbstractMatrix{T}, ∂Y::AbstractMatrix{T}, X::AbstractMatrix{T}; tol::Real = 0.0) where T <: Real
    Xd = X isa DenseMatrix ? X : collect(X)
    n_rows, n_cols = size(Xd)
    # Unlike the forward pass, the pullback must iterate over ALL row/column
    # indices, not just nonzero ones.  The gradient at a zero entry X[r,c] can
    # be non-zero because  ∂(X[i]*X[j]*X[k])/∂X[i] = X[j]*X[k]  which is
    # generically non-zero even when X[i]=0.
    # However, we can skip columns that have no stored entries in sparse ∂Y.
    sparse_cols = if ∂Y isa SparseMatrixCSC
        colmask = falses(size(∂Y, 2))
        @inbounds for col in 1:size(∂Y, 2)
            colmask[col] = ∂Y.colptr[col] < ∂Y.colptr[col + 1]
        end
        colmask
    else
        trues(size(∂Y, 2))
    end
    for i2 in 1:n_cols, j2 in 1:i2
        for k2 in 1:j2
            col = (i2 - 1) * i2 * (i2 + 1) ÷ 6 + (j2 - 1) * j2 ÷ 2 + k2
            sparse_cols[col] || continue
            for i1 in 1:n_rows
                # Hoist i1-dependent reads (column indices fixed by outer loop)
                @inbounds aii = Xd[i1, i2]; aij = Xd[i1, j2]; aik = Xd[i1, k2]
                for j1 in 1:i1
                    # Hoist j1-dependent reads
                    @inbounds aji = Xd[j1, i2]; ajj = Xd[j1, j2]; ajk = Xd[j1, k2]
                    # Precompute sub-expressions for ∂X[k1, ...] updates
                    q_i2 = aij * ajk + aik * ajj
                    q_j2 = aik * aji + aii * ajk
                    q_k2 = aii * ajj + aij * aji
                    @inbounds for k1 in 1:j1
                        row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1
                        g = ∂Y[row, col]
                        abs(g) <= tol && continue
                        if i1 == j1
                            divisor = (j1 == k1) ? 6 : 2
                        else
                            divisor = (j1 == k1 || i1 == k1) ? 2 : 1
                        end
                        g_d = g / divisor
                        aki = Xd[k1, i2]; akj = Xd[k1, j2]; akk = Xd[k1, k2]
                        ∂X[i1, i2] += g_d * (ajj * akk + ajk * akj)
                        ∂X[i1, j2] += g_d * (aji * akk + ajk * aki)
                        ∂X[i1, k2] += g_d * (aji * akj + ajj * aki)
                        ∂X[j1, i2] += g_d * (aij * akk + aik * akj)
                        ∂X[j1, j2] += g_d * (aii * akk + aik * aki)
                        ∂X[j1, k2] += g_d * (aij * aki + aii * akj)
                        ∂X[k1, i2] += g_d * q_i2
                        ∂X[k1, j2] += g_d * q_j2
                        ∂X[k1, k2] += g_d * q_k2
                    end
                end
            end
        end
    end
end

# Fused variant: computes g_col = M1 * M2[:, col] lazily per (i2,j2,k2) triple
# instead of materializing the full ∂Y = M1 * M2 matrix.
# Equivalent to:
#   compressed_kron³_pullback!(∂X, M1 * M2, X)
# but avoids the n_compressed³ × n_compressed³ allocation.
function mul_compressed_kron³_pullback!(∂X::AbstractMatrix{T},
                                        M1::AbstractMatrix,
                                        M2::AbstractMatrix,
                                        X::AbstractMatrix{T};
                                        tol::Real = 0.0) where T <: Real
    Xd = X isa DenseMatrix ? X : collect(X)
    n_rows, n_cols = size(Xd)

    g_col = Vector{T}(undef, size(M1, 1))

    for i2 in 1:n_cols, j2 in 1:i2
        for k2 in 1:j2
            col = (i2 - 1) * i2 * (i2 + 1) ÷ 6 + (j2 - 1) * j2 ÷ 2 + k2

            # Compute g_col = M1 * M2[:, col] lazily for this triple
            ℒ.mul!(g_col, M1, view(M2, :, col))

            for i1 in 1:n_rows
                # Hoist i1-dependent reads
                @inbounds aii = Xd[i1, i2]; aij = Xd[i1, j2]; aik = Xd[i1, k2]
                for j1 in 1:i1
                    # Hoist j1-dependent reads
                    @inbounds aji = Xd[j1, i2]; ajj = Xd[j1, j2]; ajk = Xd[j1, k2]
                    # Precompute sub-expressions for ∂X[k1, ...] updates
                    q_i2 = aij * ajk + aik * ajj
                    q_j2 = aik * aji + aii * ajk
                    q_k2 = aii * ajj + aij * aji
                    @inbounds for k1 in 1:j1
                        row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1
                        g = g_col[row]
                        abs(g) <= tol && continue
                        if i1 == j1
                            divisor = (j1 == k1) ? 6 : 2
                        else
                            divisor = (j1 == k1 || i1 == k1) ? 2 : 1
                        end
                        g_d = g / divisor
                        aki = Xd[k1, i2]; akj = Xd[k1, j2]; akk = Xd[k1, k2]
                        ∂X[i1, i2] += g_d * (ajj * akk + ajk * akj)
                        ∂X[i1, j2] += g_d * (aji * akk + ajk * aki)
                        ∂X[i1, k2] += g_d * (aji * akj + ajj * aki)
                        ∂X[j1, i2] += g_d * (aij * akk + aik * akj)
                        ∂X[j1, j2] += g_d * (aii * akk + aik * aki)
                        ∂X[j1, k2] += g_d * (aij * aki + aii * akj)
                        ∂X[k1, i2] += g_d * q_i2
                        ∂X[k1, j2] += g_d * q_j2
                        ∂X[k1, k2] += g_d * q_k2
                    end
                end
            end
        end
    end
end

# =====================================================================================
# Third-order solution rrule  (correctness-first, allocating version)
# =====================================================================================

function rrule(::typeof(calculate_third_order_solution),
                    ∇₁::AbstractMatrix{S},
                    ∇₂::SparseMatrixCSC{S},
                    ∇₃::SparseMatrixCSC{S},
                    𝑺₁::AbstractMatrix{S},
                    𝐒₂::AbstractMatrix{S},
                    constants::constants,
                    workspaces::workspaces,
                    cache::caches;
                    initial_guess::AbstractMatrix{R} = zeros(0,0),
                    opts::CalculationOptions = merge_calculation_options(),
                    parameter_values::AbstractVector{<:Real} = Float64[],
                    caching::Bool = true) where {S <: Real, R <: Real}

    # --- workspace / constants ---------------------------------------------------
    if !(eltype(workspaces.third_order.Ŝ) == S)
        workspaces.third_order = Higher_order_workspace(S)
    end
    ℂ = workspaces.third_order
    M₂ = constants.second_order
    M₃ = constants.third_order
    T = constants.post_model_macro

    # Expand compressed inputs to full space for internal computation
    ∇₂ = ∇₂ * M₂.𝐔∇₂
    𝐒₂ = sparse(𝐒₂ * M₂.𝐔₂)::SparseMatrixCSC{S, Int}  # was: dense_to_sparse

    i₊ = T.future_not_past_and_mixed_idx
    i₋ = T.past_not_future_and_mixed_idx
    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo
    n  = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    ensure_higher_order_solution_buffers!(ℂ, n, nₑ₋)

    initial_guess_sylv = if length(initial_guess) == 0
        zeros(S, 0, 0)
    elseif eltype(initial_guess) <: AbstractFloat
        initial_guess isa Matrix{S} ? initial_guess : Matrix{S}(initial_guess)
    else
        zeros(S, 0, 0)
    end

    # --- forward pass (mirrors the primal, but stores intermediates) ---------------

    # 1st-order solution with zero-column
    𝐒₁ = ℂ.𝐒₁::Matrix{S}
    copyto!(@view(𝐒₁[:,1:n₋]), @view(𝑺₁[:,1:n₋]))
    fill!(@view(𝐒₁[:,n₋+1]), zero(S))
    copyto!(@view(𝐒₁[:,n₋+2:end]), @view(𝑺₁[:,n₋+1:end]))

    𝐒₁₋╱𝟏ₑ = ℂ.𝐒₁₋╱𝟏ₑ::Matrix{S}
    copyto!(@view(𝐒₁₋╱𝟏ₑ[1:n₋,:]), @view(𝐒₁[i₋,:]))
    fill!(@view(𝐒₁₋╱𝟏ₑ[n₋+1:end,:]), zero(S))
    @inbounds 𝐒₁₋╱𝟏ₑ[n₋+1,n₋+1] = one(S)
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 1.0, min_length = 10, tol = opts.tol.third_order.droptol)

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                ℒ.I(nₑ₋)[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]]

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]; zeros(n₋ + n + nₑ, nₑ₋)]
    𝐒₁₊╱𝟎 = choose_matrix_format(𝐒₁₊╱𝟎, density_threshold = 1.0, min_length = 10, tol = opts.tol.third_order.droptol)

    ∇₁₊𝐒₁➕∇₁₀ = collect(@views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * M₂.𝐈ₙ₋ - ∇₁[:,range(1,n) .+ n₊])

    qme_ws = workspaces.first_order

    if S === Float64
        qme_ws.fast_lu_ws_nabla0, qme_ws.fast_lu_dims_nabla0, solved_∇lu, lu_handle =
            factorize_lu!(Val(:FastLapack), ∇₁₊𝐒₁➕∇₁₀, qme_ws.fast_lu_ws_nabla0, qme_ws.fast_lu_dims_nabla0)

        if !solved_∇lu
            return (∇₁₊𝐒₁➕∇₁₀, false), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end

        spinv = Matrix{S}(ℒ.I, size(∇₁₊𝐒₁➕∇₁₀))
        solve_lu_left!(∇₁₊𝐒₁➕∇₁₀, spinv, qme_ws.fast_lu_ws_nabla0, lu_handle)
    else
        ∇₁₊𝐒₁➕∇₁₀lu = ℒ.lu(∇₁₊𝐒₁➕∇₁₀, check = false)

        if !ℒ.issuccess(∇₁₊𝐒₁➕∇₁₀lu)
            return (∇₁₊𝐒₁➕∇₁₀, false), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end

        spinv = inv(∇₁₊𝐒₁➕∇₁₀lu)
    end
    spinv = choose_matrix_format(spinv)

    ∇₁₊ = @views ∇₁[:,1:n₊] * M₂.𝐈ₙ₊

    A = spinv * ∇₁₊

    # --- B matrix -----------------------------------------------------------------
    kron𝐒₁₋╱𝟏ₑ = ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ)

    B = compressed_permuted_mixed_kron(𝐒₁₋╱𝟏ₑ, M₂.𝛔,
                                       sparse_preallocation = ℂ.tmp_sparse_prealloc7)

    B += compressed_kron³(𝐒₁₋╱𝟏ₑ, tol = opts.tol.third_order.droptol, sparse_preallocation = ℂ.tmp_sparse_prealloc1)

    # --- 𝐗₃ (C-matrix ingredients) -----------------------------------------------
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [(𝐒₂ * kron𝐒₁₋╱𝟏ₑ + 𝐒₁ * [𝐒₂[i₋,:]; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
                                          𝐒₂
                                          zeros(n₋ + nₑ, nₑ₋^2)]
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = choose_matrix_format(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, density_threshold = 0.0, min_length = 10, tol = opts.tol.third_order.droptol)

    𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:]; zeros(n₋ + n + nₑ, nₑ₋^2)]

    aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋

    S1p0_kron_sigma = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔
    tmpkron22 = compressed_permuted_mixed_kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,
                                               S1p0_kron_sigma,
                                               sparse_preallocation = ℂ.tmp_sparse_prealloc6)

    𝐒₂₊╱𝟎 = choose_matrix_format(𝐒₂₊╱𝟎, density_threshold = 1.0, min_length = 10, tol = opts.tol.third_order.droptol)

    ∇₁₊ = choose_matrix_format(∇₁₊, density_threshold = 1.0, min_length = 10, tol = opts.tol.third_order.droptol)

    𝐒₂₋╱𝟎 = [𝐒₂[i₋,:]; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]

    # Terms (a)+(b): ∇₂ * kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎) * [tmpkron2 + 𝐏₁ₗ * tmpkron2 * 𝐏₁ᵣ] * 𝐏𝐂₃
    tmpkron2 = ℒ.kron(M₂.𝛔, choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0, tol = opts.tol.third_order.droptol))
    D_ab = (tmpkron2 + M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ) * M₃.𝐏𝐂₃
    𝐗₃ = mat_mult_kron(∇₂, collect(𝐒₁₊╱𝟎), collect(𝐒₂₊╱𝟎), D_ab, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc2)

    # Term (c): ∇₂ * kron(⎸𝐒₁..⎹, ⎸𝐒₂k..⎹) * 𝐏𝐂₃
    𝐗₃ += mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, M₃.𝐏𝐂₃, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc3)

    # Term (d): ∇₂ * kron(⎸𝐒₁..⎹, 𝐒₂₊╱𝟎*𝛔) * 𝐏𝐂₃
    S2p0_sigma = 𝐒₂₊╱𝟎 * M₂.𝛔
    𝐗₃ += mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, collect(S2p0_sigma), M₃.𝐏𝐂₃, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc4)

    # Term (e): ∇₁₊ * 𝐒₂ * kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎) * 𝐏𝐂₃
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0, tol = opts.tol.third_order.droptol)
    mm_𝐒₂_kron = mat_mult_kron(𝐒₂, 𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc4)
    𝐗₃ += ∇₁₊ * mm_𝐒₂_kron * M₃.𝐏𝐂₃

    𝐗₃ += ∇₃ * tmpkron22

    # Compute compressed_kron³(aux) WITHOUT rowmask: the pullback needs ∂∇₃ at ALL
    # positions (including currently-zero columns of ∇₃) so that gradients flow
    # correctly through calculate_third_order_derivatives back to parameters.
    ck3_aux_mat = compressed_kron³(aux, rowmask = M₃.∇₃_rowmask, tol = opts.tol.third_order.droptol, sparse_preallocation = ℂ.tmp_sparse_prealloc5)
    ck3_aux = ∇₃ * ck3_aux_mat
    𝐗₃ += ck3_aux
    
    C = spinv * 𝐗₃

    # --- solve Sylvester  A·𝐒₃·B + C = 𝐒₃ ----------------------------------------
    cache_eligible_3rd = opts.sylvester_algorithm³ == :doubling
    if cache_eligible_3rd
        ℂ.sylvester_workspace.pow_iters = 0
        ℂ.sylvester_workspace.pow_capture = true
        ℂ.sylvester_workspace.pow_transposed = true
    end
    𝐒₃, solved = solve_sylvester_equation(A, B, C, ℂ.sylvester_workspace,
                                        initial_guess = initial_guess_sylv,
                                        sylvester_algorithm = opts.sylvester_algorithm³,
                                        preconditioner = opts.sylvester_preconditioner,
                                        tol = opts.tol.third_order.ad.sylvester,
                                        verbose = opts.verbose)
    ℂ.sylvester_workspace.pow_capture = false
    pow_iters_captured_3rd = ℂ.sylvester_workspace.pow_iters
    ℂ.sylvester_workspace.pow_iters = 0

    𝐒₃ = choose_matrix_format(𝐒₃, multithreaded = false, tol = opts.tol.third_order.droptol)
    𝐒₃_stable = copy(𝐒₃)

    if !solved
        return (𝐒₃_stable, solved), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # cache update (same as primal)
    if 𝐒₃_stable isa Matrix{S} && cache.third_order_solution isa Matrix{S} && size(cache.third_order_solution) == size(𝐒₃_stable)
        copyto!(cache.third_order_solution, 𝐒₃_stable)
    elseif 𝐒₃_stable isa SparseMatrixCSC{S, Int} && cache.third_order_solution isa SparseMatrixCSC{S, Int} &&
           size(cache.third_order_solution) == size(𝐒₃_stable) &&
           cache.third_order_solution.colptr == 𝐒₃_stable.colptr &&
           cache.third_order_solution.rowval == 𝐒₃_stable.rowval
        copyto!(cache.third_order_solution.nzval, 𝐒₃_stable.nzval)
    else
        cache.third_order_solution = 𝐒₃_stable
    end
    if !isempty(parameter_values)
        cache.valid_for.third_order_solution = Float64.(parameter_values)
    end
    empty!(cache.valid_for.pruned_third_order_solution)

    # --- precompute transposed constants for pullback -----------------------------
    # Use pre-cached transposes from constants (computed once at model compile time)
    𝐏𝐂₃t = M₃.𝐏𝐂₃ᵀ
    𝛔t  = M₂.𝛔ᵀ
    𝐔∇₂t = M₂.𝐔∇₂ᵀ
    𝐔₂t  = M₂.𝐔₂ᵀ

    # Materialized transposes of forward-pass intermediates
    At = choose_matrix_format(A')
    Bt = choose_matrix_format(B')
    ∇₂t = choose_matrix_format(∇₂')
    ∇₃t = choose_matrix_format(∇₃')
    D_ab_t = choose_matrix_format(D_ab')
    tmpkron22_t = choose_matrix_format(tmpkron22')
    ck3_aux_mat_t = choose_matrix_format(ck3_aux_mat')
    𝐒₂t = choose_matrix_format(𝐒₂', density_threshold = 1.0)
    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t = choose_matrix_format(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋')
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎t = choose_matrix_format(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎')
    S2p0_sigma_t = choose_matrix_format(S2p0_sigma')

    mm_𝐒₂_kron_t = choose_matrix_format(mm_𝐒₂_kron')

    # Precompute (∇₁₊ · 𝐒₂)ᵀ for term 8 fused kron adjoint
    ∇₁₊_𝐒₂_t = choose_matrix_format((∇₁₊ * 𝐒₂)')

    # Precompute (∇₂ · kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎))ᵀ for fused terms a+b pullback
    nabla2_kron_S1S2_t = collect(mat_mult_kron(collect(∇₂), collect(𝐒₁₊╱𝟎), collect(𝐒₂₊╱𝟎))')

    # Sparse σ for fill_kron_adjoint_∂A_with_perm! (ultra-sparse: ~nₑ nonzeros in nₑ₋² × nₑ₋²)
    σ_sparse = M₂.𝛔 isa SparseMatrixCSC ? M₂.𝛔 : sparse(M₂.𝛔)

    # --- ensure pullback workspace buffers ---
    ensure_third_order_pullback_workspaces!(ℂ, S, T, M₂, M₃)

    tmpkron22_ck3_aux_mat_t = choose_matrix_format(tmpkron22_t + ck3_aux_mat_t)
    # =========================================================================
    #   PULLBACK
    # =========================================================================
    function third_order_solution_pullback(∂𝐒₃_solved)
        ∂𝐒₃ = choose_matrix_format(unthunk(∂𝐒₃_solved[1]))

        if ℒ.norm(∂𝐒₃) < opts.tol.third_order.ad.sylvester.acceptance_tol
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end

        # --- adjoint Sylvester:  Aᵀ ∂C_adj Bᵀ + ∂𝐒₃ = ∂C_adj --------------------
        ws = ℂ.sylvester_workspace
        cache_valid = cache_eligible_3rd &&
                      pow_iters_captured_3rd >= 1 &&
                      ws.pow_transposed
        saved_capture = ws.pow_capture
        if cache_valid
            ws.pow_iters = pow_iters_captured_3rd
            ws.pow_capture = false
        end
        ∂C_adj, slvd = solve_sylvester_equation(At, Bt, ∂𝐒₃, ws,
                                              sylvester_algorithm = opts.sylvester_algorithm³,
                                              preconditioner = opts.sylvester_preconditioner,
                                              tol = opts.tol.third_order.ad.sylvester,
                                              verbose = opts.verbose)
        ws.pow_capture = saved_capture
        ws.pow_iters = 0
        if !slvd
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end

        ∂C_adj = choose_matrix_format(∂C_adj)

        # --- Initialize all gradient accumulators ---
        # Dense workspace temporaries (overwritten by mul! each call)
        ∂𝐗₃           = ℂ.∂𝐗₃_3rd
        ∂A             = ℂ.∂A_3rd
        ∂B_from_sylv   = ℂ.∂B_sylv_3rd
        ∂out2          = ℂ.∂out2_3rd
        mul_tmp        = ℂ.mul_tmp_3rd
        ∂∇₁₊𝐒₁➕∇₁₀   = ℂ.∂∇₁₊𝐒₁➕∇₁₀_3rd

        # Dense workspace accumulators (need zeroing)
        ∂spinv         = ℂ.∂spinv_3rd
        ∂∇₁            = ℂ.∂∇₁_3rd;  fill!(∂∇₁, zero(S))
        ∂𝐒₁₃           = ℂ.∂𝐒₁_3rd;  fill!(∂𝐒₁₃, zero(S))

        # Sparse-preserving gradient accumulators (reuse workspace buffers)
        ∂𝐒₂            = zero(𝐒₂)  # sparse — must stay fresh

        ∂𝐒₁₊╱𝟎_tmp    = ℂ.∂𝐒₁₊╱𝟎_tmp_3rd;  fill!(∂𝐒₁₊╱𝟎_tmp, zero(S))
        ∂𝐒₂₊╱𝟎        = ℂ.∂𝐒₂₊╱𝟎_3rd;       fill!(∂𝐒₂₊╱𝟎, zero(S))
        ∂L_c           = ℂ.∂L_c_3rd;          fill!(∂L_c, zero(S))
        ∂R_c           = ℂ.∂R_c_3rd;          fill!(∂R_c, zero(S))
        ∂L_d           = ℂ.∂L_d_3rd;          fill!(∂L_d, zero(S))
        ∂R_d           = ℂ.∂R_d_3rd;          fill!(∂R_d, zero(S))
        ∂𝐒₁₋╱𝟏ₑ_t8   = ℂ.∂𝐒₁₋╱𝟏ₑ_t8_3rd;  fill!(∂𝐒₁₋╱𝟏ₑ_t8, zero(S))
        ∂𝐒₂₋╱𝟎        = ℂ.∂𝐒₂₋╱𝟎_3rd;       fill!(∂𝐒₂₋╱𝟎, zero(S))
        ∂𝐒₁₋╱𝟏ₑ₃     = ℂ.∂𝐒₁₋╱𝟏ₑ_3rd;     fill!(∂𝐒₁₋╱𝟏ₑ₃, zero(S))
        ∂𝐒₁₊╱𝟎₃      = ℂ.∂𝐒₁₊╱𝟎_3rd;       fill!(∂𝐒₁₊╱𝟎₃, zero(S))
        ∂S1S1_stack    = ℂ.∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋_3rd; fill!(∂S1S1_stack, zero(S))
        ∂aux           = ℂ.∂aux_3rd;          fill!(∂aux, zero(S))

        # --- gradient of A, B, C from 𝐒₃ = A·𝐒₃·B + C ---------------------------
        # ∂A = ∂C_adj * B' * 𝐒₃_stable' — use ∂𝐗₃ as temp for intermediate
        ℒ.mul!(∂𝐗₃, ∂C_adj, Bt)
        ℒ.mul!(∂A, ∂𝐗₃, 𝐒₃_stable')
        # ∂B_from_sylv = 𝐒₃_stable' * A' * ∂C_adj — reuse ∂𝐗₃ as temp
        ℒ.mul!(∂𝐗₃, At, ∂C_adj)
        ℒ.mul!(∂B_from_sylv, 𝐒₃_stable', ∂𝐗₃)
        # ∂B_from_sylv = sparse(𝐒₃_stable' * ∂𝐗₃)
        # ∂𝐗₃ = spinv' * ∂C_adj (overwrite temp with real value)
        # ℒ.mul!(∂𝐗₃, sxpinv', ∂C_adj)
        ∂𝐗₃ = choose_matrix_format(spinv' * ∂C_adj)

        # C = spinv * 𝐗₃  →  ∂spinv
        # A = spinv * ∇₁₊  →  ∂spinv accumulation
        ℒ.mul!(∂spinv, ∂C_adj, 𝐗₃')
        ℒ.mul!(∂spinv, ∂A, ∇₁₊', 1, 1)

        # =====================================================================
        #  ∂∇₃  (linear: ∇₃ appears in two additive terms of 𝐗₃)
        # =====================================================================
        # 𝐗₃ = out2 * 𝐏𝐂₃ + ∇₃ * tmpkron22 + ∇₃ * ck3_aux_mat
        # ∇₃ has two direct linear terms; out2 maps through 𝐏𝐂₃.
        ∂∇₃ = ∂𝐗₃ * tmpkron22_ck3_aux_mat_t
        # =====================================================================
        #  ∂∇₂  (∇₂ is linear in out2 → 𝐗₃_pre → 𝐗₃)
        # =====================================================================
        # out2 enters 𝐗₃ as: 𝐗₃ = out2 · 𝐏𝐂₃ + ...
        # ∂out2 = ∂𝐗₃ · (𝐏𝐂₃)ᵀ
        ℒ.mul!(∂out2, ∂𝐗₃, 𝐏𝐂₃t)

        # 𝐗₃ = ∇₂ * kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎) * D_ab                               (terms a+b)
        #     + ∇₂ * kron(⎸𝐒₁..⎹, ⎸𝐒₂k..⎹) * 𝐏𝐂₃                             (term c)
        #     + ∇₂ * kron(⎸𝐒₁..⎹, 𝐒₂₊╱𝟎·𝛔) * 𝐏𝐂₃                             (term d)
        #   (term e = ∇₁₊ · 𝐒₂ · kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎) · 𝐏𝐂₃ does not involve ∇₂.)

        # ∂∇₂ via mat_mult_kron (avoids materializing cubic kron transposes)
        ∂mid_ab = choose_matrix_format(∂𝐗₃ * D_ab_t)                                   # n × nₑ₋³
        ∂∇₂ = mat_mult_kron(∂mid_ab, collect(𝐒₁₊╱𝟎'), collect(𝐒₂₊╱𝟎'))               # terms a+b
        ∂∇₂ = ∂∇₂ + mat_mult_kron(∂out2, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎t) # term c
        ∂∇₂ = ∂∇₂ + mat_mult_kron(∂out2, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t, S2p0_sigma_t)        # term d


        # =====================================================================
        #  ∂𝐒₂  (𝐒₂ enters out2 via several stacking matrices)
        # =====================================================================
        # 𝐒₂ does NOT affect A, B, or the ∇₃ terms — only out2.
        # We already have ∂out2 from the 𝐗₃ = out2 * 𝐏𝐂₃ adjoint.
        #
        # out2 terms that depend on 𝐒₂:
        #   (a) ∇₂ · tmpkron1 · tmpkron2           — tmpkron1 = kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎)
        #   (b) ∇₂ · tmpkron1 · 𝐏₁ₗ · tmpkron2 · 𝐏₁ᵣ  — same tmpkron1
        #   (c) ∇₂ · kron(⎸𝐒₁..⎹, ⎸𝐒₂k..⎹)       — second factor depends on 𝐒₂
        #   (d) ∇₂ · kron(⎸𝐒₁..⎹, 𝐒₂₊╱𝟎·𝛔)       — second factor depends on 𝐒₂
        #   (8) ∇₁₊ · 𝐒₂ · kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)  — both 𝐒₂ and 𝐒₂₋╱𝟎 depend on 𝐒₂

        # --- terms (a) and (b): through kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎) via D_ab ---
        # ∂kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎) = ∇₂ᵀ * ∂𝐗₃ * D_ab' (combines terms a+b)
        ∂tmpkron1 = (∇₂t * ∂mid_ab)
        # ∂tmpkron1 = sparse(∇₂t * ∂mid_ab)

        # Force only the cotangent argument onto the dense fill_kron_adjoint! path here
        # and in the analogous calls below. The primal factors may stay sparse/abstract,
        # but the sparse ∂X overload only iterates stored cotangent entries.
        # kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎) pullback → ∂𝐒₂₊╱𝟎 via fill_kron_adjoint!
        fill_kron_adjoint!(∂𝐒₂₊╱𝟎, ∂𝐒₁₊╱𝟎_tmp, ∂tmpkron1, 𝐒₂₊╱𝟎, 𝐒₁₊╱𝟎)

        # 𝐒₂₊╱𝟎 = [𝐒₂[i₊,:]; 0]  →  ∂𝐒₂[i₊,:] += ∂𝐒₂₊╱𝟎[1:length(i₊),:]
        @views ∂𝐒₂[i₊,:] .+= ∂𝐒₂₊╱𝟎[1:length(i₊),:]

        # --- term (c): through ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 ---
        # Fused: ∇₂ᵀ * ∂out2 with fill_kron_adjoint! — avoids materializing ∇₂t_∂out2
        mul_fill_kron_adjoint!(∂R_c, ∂L_c, ∇₂t, ∂out2, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, tol = opts.tol.third_order.droptol)

        # ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = [ (𝐒₂·kron𝐒₁₋╱𝟏ₑ + 𝐒₁·[𝐒₂[i₋,:];0])[i₊,:] ; 𝐒₂ ; 0 ]
        # Top block (rows 1:n₊): depends on 𝐒₂ through 𝐒₂·kron𝐒₁₋╱𝟏ₑ and 𝐒₁·[𝐒₂[i₋,:];0]
        n₊_len = length(i₊)
        ∂top_block = ∂R_c[1:n₊_len, :]
        # From 𝐒₂·kron𝐒₁₋╱𝟏ₑ:
        @views ∂𝐒₂[i₊,:] .+= ∂top_block * kron𝐒₁₋╱𝟏ₑ'
        # From 𝐒₁·[𝐒₂[i₋,:];0] → ∂𝐒₂[i₋,:] += 𝐒₁' * I[:,i₊] * ∂top_block
        #   (since [𝐒₂[i₋,:];0] pads with zeros, only i₋ rows of 𝐒₂ contribute)
        ∂𝐒₂_padded = 𝐒₁' * ℒ.I(n)[:,i₊] * ∂top_block   # TODO: In general check if there are more optimizations that can be carried over from the non-AD call. # n₋+1+nₑ × nₑ₋²
        @views ∂𝐒₂[i₋,:] .+= ∂𝐒₂_padded[1:n₋, :]

        # Middle block (rows n₊_len+1 : n₊_len+n): directly 𝐒₂
        @views ∂𝐒₂ .+= ∂R_c[n₊_len .+ (1:n), :]

        # Bottom block is zeros

        # --- term (d): through kron(⎸𝐒₁..⎹, 𝐒₂₊╱𝟎·𝛔) ---
        # Fused: ∇₂ᵀ * ∂out2 with fill_kron_adjoint! — same pattern, different kron factors
        mul_fill_kron_adjoint!(∂R_d, ∂L_d, ∇₂t, ∂out2, S2p0_sigma, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, tol = opts.tol.third_order.droptol)

        # 𝐒₂₊╱𝟎·𝛔  →  ∂𝐒₂₊╱𝟎_d = ∂R_d · 𝛔ᵀ
        ∂𝐒₂₊╱𝟎_d = ∂R_d * 𝛔t
        @views ∂𝐒₂[i₊,:] .+= ∂𝐒₂₊╱𝟎_d[1:length(i₊),:]

        # --- term (8): ∇₁₊ · 𝐒₂ · kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎) ---
        # out2_term8 = ∇₁₊ · 𝐒₂ · kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)
        # ∂(∇₁₊·𝐒₂·K) w.r.t. 𝐒₂ = ∇₁₊ᵀ · ∂out2 · Kᵀ
        tmp_t8 = ∇₁₊' * ∂out2
        ∂𝐒₂ = ∂𝐒₂ + mat_mult_kron(tmp_t8, collect(𝐒₁₋╱𝟏ₑ'), collect(𝐒₂₋╱𝟎'))

        # ∂(∇₁₊·𝐒₂·kron(𝐒₁₋╱𝟏ₑ,𝐒₂₋╱𝟎)) w.r.t. 𝐒₂₋╱𝟎  (through the kron)
        # Fused: (∇₁₊·𝐒₂)ᵀ · ∂out2 with fill_kron_adjoint! in one pass
        mul_fill_kron_adjoint!(∂𝐒₂₋╱𝟎, ∂𝐒₁₋╱𝟏ₑ_t8, ∇₁₊_𝐒₂_t, ∂out2, 𝐒₂₋╱𝟎, 𝐒₁₋╱𝟏ₑ, tol = opts.tol.third_order.droptol)

        # 𝐒₂₋╱𝟎 = [𝐒₂[i₋,:]; 0]  →  ∂𝐒₂[i₋,:] += ∂𝐒₂₋╱𝟎[1:n₋,:]
        @views ∂𝐒₂[i₋,:] .+= ∂𝐒₂₋╱𝟎[1:n₋,:]

        # =====================================================================
        #  ∂∇₁
        # =====================================================================
        # ∇₁ enters through:
        #   1. ∇₁₊𝐒₁➕∇₁₀ = -∇₁[:,1:n₊]·𝐒₁[i₊,1:n₋]·I[i₋,:] - ∇₁[:,n₊+1:n₊+n]
        #      → spinv = inv(∇₁₊𝐒₁➕∇₁₀)  →  used in A and C
        #   2. ∇₁₊ = ∇₁[:,1:n₊] · I(n)[i₊,:]
        #      → A = spinv·∇₁₊   and   out2 += ∇₁₊ · mm_𝐒₂_kron

        # step 1: ∂ through inv(∇₁₊𝐒₁➕∇₁₀)  (∂spinv already accumulated)
        ℒ.mul!(mul_tmp, spinv', ∂spinv)
        ℒ.mul!(∂∇₁₊𝐒₁➕∇₁₀, mul_tmp, spinv')
        ℒ.rmul!(∂∇₁₊𝐒₁➕∇₁₀, -1)

        ∂∇₁[:,1:n₊] -= ∂∇₁₊𝐒₁➕∇₁₀ * ℒ.I(n)[:,i₋] * 𝐒₁[i₊,1:n₋]'
        ∂∇₁[:,range(1,n) .+ n₊] -= ∂∇₁₊𝐒₁➕∇₁₀

        # step 2: ∂ through ∇₁₊
        ∂∇₁₊ = ℂ.∂∇₁₊_3rd
        ℒ.mul!(∂∇₁₊, spinv', ∂A)  # from A = spinv · ∇₁₊
        ℒ.mul!(∂∇₁₊, ∂out2, mm_𝐒₂_kron_t, 1, 1)  # from out2 += ∇₁₊ · mm_𝐒₂_kron

        ∂∇₁[:,1:n₊] += ∂∇₁₊ * ℒ.I(n)[:,i₊]

        # =====================================================================
        #  ∂𝑺₁  (𝑺₁ enters through 𝐒₁, affecting A,B,C,out2 via many paths)
        # =====================================================================
        # --- ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ : from out2 terms c,d (kron outer factors) ---
        ℒ.axpy!(1, ∂L_c, ∂S1S1_stack)
        ℒ.axpy!(1, ∂L_d, ∂S1S1_stack)

        # --- ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ + ∂𝐒₁₊╱𝟎 : from ∇₃ * compressed_kron(...) ---
        # Fused: compute g_col = ∇₃ᵀ * ∂𝐗₃[:, col] lazily per (α,β,γ) triple
        # instead of materializing the full ∂tmpkron22 = ∇₃ᵀ * ∂𝐗₃ matrix.
        ∂S1S1_from_ck = ℂ.∂S1S1_from_ck_3rd
        fill!(∂S1S1_from_ck, zero(S))
        ∂S1p0_kron_sigma = ℂ.∂S1p0_kron_sigma_3rd
        fill!(∂S1p0_kron_sigma, zero(S))
        mul_compressed_permuted_mixed_kron_pullback!(∂S1S1_from_ck,
                             ∂S1p0_kron_sigma,
                             ∇₃t, ∂𝐗₃,
                             ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,
                             S1p0_kron_sigma;
                             tol = opts.tol.third_order.droptol)

        # Sparsify ∂S1p0_kron_sigma: structurally bounded by σ's support, so very sparse.
        # sparse × sparse matmul avoids dense intermediate; downstream fill_kron_adjoint!
        # uses the sparse overload that iterates only nonzero cotangent entries.
        ∂S1p0_kron = choose_matrix_format(sparse(∂S1p0_kron_sigma) * 𝛔t)  # was: dense_to_sparse
        ∂S1p0_left = ℂ.∂S1p0_left_3rd
        fill!(∂S1p0_left, zero(S))
        ∂S1p0_right = ℂ.∂S1p0_right_3rd
        fill!(∂S1p0_right, zero(S))
        fill_kron_adjoint!(∂S1p0_left, ∂S1p0_right, ∂S1p0_kron, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)

        ℒ.axpy!(1, ∂S1S1_from_ck, ∂S1S1_stack)
        ℒ.axpy!(1, ∂S1p0_left, ∂𝐒₁₊╱𝟎₃)
        ℒ.axpy!(1, ∂S1p0_right, ∂𝐒₁₊╱𝟎₃)

        # --- ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ : from compressed_kron³(aux) → 𝐗₃ ---
        # Fused: compute g_col = ∇₃ᵀ * ∂𝐗₃[:, col] lazily per (i2,j2,k2) triple
        mul_compressed_kron³_pullback!(∂aux, ∇₃t, ∂𝐗₃, aux; tol = opts.tol.third_order.droptol)
        ℒ.mul!(∂S1S1_stack, M₃.𝐒𝐏', ∂aux, 1, 1)

        # --- ∂𝐒₁₊╱𝟎 : from tmpkron1 (already computed for ∂𝐒₂) ---
        ℒ.axpy!(1, ∂𝐒₁₊╱𝟎_tmp, ∂𝐒₁₊╱𝟎₃)

        # --- ∂𝐒₁₋╱𝟏ₑ : from B via compressed_permuted_mixed_kron(𝐒₁₋╱𝟏ₑ, 𝛔) ---
        compressed_permuted_mixed_kron_pullback_∂A!(∂𝐒₁₋╱𝟏ₑ₃, ∂B_from_sylv, 𝐒₁₋╱𝟏ₑ, M₂.𝛔; tol = opts.tol.third_order.droptol)

        # --- ∂𝐒₁₋╱𝟏ₑ : from B via compressed_kron³(𝐒₁₋╱𝟏ₑ) ---
        compressed_kron³_pullback!(∂𝐒₁₋╱𝟏ₑ₃, ∂B_from_sylv, 𝐒₁₋╱𝟏ₑ; tol = opts.tol.third_order.droptol)

        # --- ∂𝐒₁₋╱𝟏ₑ : from out2 terms a,b via tmpkron2 = kron(B=𝛔, A=𝐒₁₋╱𝟏ₑ) ---
        # Fused: nabla2_kron_S1S2_t * ∂out2 in blocks + identity/(2,1,3) permuted ∂A
        # Avoids materializing both ∇₂t_∂out2 (n_∇₂ × n_out2_c) and tmp_a (nₑ₋³ × nₑ₋³)
        mul_fill_kron_adjoint_∂A_with_perm!(nabla2_kron_S1S2_t, ∂out2, ∂𝐒₁₋╱𝟏ₑ₃, σ_sparse)

        # --- ∂𝐒₁₋╱𝟏ₑ : from term 8 kron (already computed for ∂𝐒₂) ---
        ℒ.axpy!(1, ∂𝐒₁₋╱𝟏ₑ_t8, ∂𝐒₁₋╱𝟏ₑ₃)

        # --- ∂𝐒₁₋╱𝟏ₑ : from kron𝐒₁₋╱𝟏ₑ in ⎸𝐒₂k..⎹ top block ---
        # ∂kron𝐒₁₋╱𝟏ₑ₃ = sparse(𝐒₂t * ℒ.I(n)[:,i₊] * ∂top_block)
        ∂kron𝐒₁₋╱𝟏ₑ₃ = (𝐒₂t * ℒ.I(n)[:,i₊] * ∂top_block)
        fill_kron_adjoint!(∂𝐒₁₋╱𝟏ₑ₃, ∂𝐒₁₋╱𝟏ₑ₃, ∂kron𝐒₁₋╱𝟏ₑ₃, 𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ)

        # --- ∂𝐒₁ : from 𝐒₁·[𝐒₂[i₋,:];0] in ⎸𝐒₂k..⎹ top block ---
        S2_padded = [𝐒₂[i₋,:]; zeros(S, nₑ + 1, nₑ₋^2)]
        @views ∂𝐒₁₃[i₊,:] .+= ∂top_block * S2_padded'

        # === Convert ∂S1S1_stack → ∂𝐒₁ and ∂𝐒₁₋╱𝟏ₑ ===
        n₊l = length(i₊)
        ∂top_S1S1 = ∂S1S1_stack[1:n₊l, :]
        @views ∂𝐒₁₃[i₊,:] .+= ∂top_S1S1 * 𝐒₁₋╱𝟏ₑ'
        ∂𝐒₁₋╱𝟏ₑ₃ .+= 𝐒₁' * ℒ.I(n)[:,i₊] * ∂top_S1S1
        @views ∂𝐒₁₃ .+= ∂S1S1_stack[n₊l .+ (1:n), :]

        # === Convert ∂𝐒₁₊╱𝟎ₓ → ∂𝐒₁ ===
        @views ∂𝐒₁₃[i₊,:] .+= ∂𝐒₁₊╱𝟎₃[1:n₊l,:]

        # === Convert ∂𝐒₁₋╱𝟏ₑ → ∂𝐒₁ ===
        @views ∂𝐒₁₃[i₋,:] .+= ∂𝐒₁₋╱𝟏ₑ₃[1:length(i₋),:]

        # === ∂𝐒₁ from ∇₁₊𝐒₁➕∇₁₀ (spinv) ===
        ∂𝐒₁₃[i₊,1:n₋] -= ∇₁[:,1:n₊]' * ∂∇₁₊𝐒₁➕∇₁₀ * ℒ.I(n)[:,i₋]

        # === 𝐒₁ = [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] → ∂𝑺₁ ===
        ∂𝑺₁ = [∂𝐒₁₃[:,1:n₋] ∂𝐒₁₃[:,n₋+2:end]]

        # Map ∂∇₂ and ∂𝐒₂ back to compressed space
        # (adjoint of ∇₂_full = ∇₂_compressed * 𝐔∇₂ and 𝐒₂_full = 𝐒₂_compressed * 𝐔₂)
        ∂∇₂ = ∂∇₂ * 𝐔∇₂t
        ∂𝐒₂ = ∂𝐒₂ * 𝐔₂t

        return (NoTangent(), ∂∇₁, ∂∇₂, ∂∇₃, ∂𝑺₁, ∂𝐒₂, NoTangent(), NoTangent(), NoTangent())
    end

    return (𝐒₃_stable, solved), third_order_solution_pullback
end


function rrule(::typeof(solve_sylvester_equation),
    A::M,
    B::N,
    C::O,
    𝕊ℂ::sylvester_workspace;
    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
    sylvester_algorithm::Symbol = :doubling,
    preconditioner::Symbol = :ilu,
    tol::SolverTolerances = SolverTolerances(),
    # timer::TimerOutput = TimerOutput(),
    verbose::Bool = false) where {M <: AbstractMatrix{Float64}, N <: AbstractMatrix{Float64}, O <: AbstractMatrix{Float64}}

    # Enable doubling-power capture only for the dense-dense :doubling path
    # (the dense-dense overload of solve_sylvester_equation populates 𝐀_pow/𝐁_pow).
    # Enable doubling-power capture for the :doubling algorithm path.
    # The solver overloads populate 𝕊ℂ.𝐀_pow / 𝐁_pow during forward iteration so
    # the pullback can skip squaring. With pow_transposed=true, powers are stored
    # in transposed form directly, saving a post-hoc transpose pass.
    cache_eligible = sylvester_algorithm == :doubling
    if cache_eligible
        𝕊ℂ.pow_iters = 0
        𝕊ℂ.pow_capture = true
        𝕊ℂ.pow_transposed = true
    end
    P, solved = solve_sylvester_equation(A, B, C, 𝕊ℂ,
                                    sylvester_algorithm = sylvester_algorithm,
                                    preconditioner = preconditioner,
                                    tol = tol,
                                    verbose = verbose,
                                    initial_guess = initial_guess)
    𝕊ℂ.pow_capture = false
    pow_iters_captured = 𝕊ℂ.pow_iters
    𝕊ℂ.pow_iters = 0

    if size(𝕊ℂ.P) != size(P)
        𝕊ℂ.P = zeros(eltype(P), size(P)...)
    end
    copyto!(𝕊ℂ.P, P)
    P_cached = 𝕊ℂ.P

    ensure_sylvester_doubling_buffers!(𝕊ℂ, size(A, 1), size(B, 1))

    # Precompute transposes once outside the pullback closure: needed for both
    # the matmul forming ∂A/∂B (every call) and the fallback adjoint solve
    # (when the doubling power cache is unavailable).
    At = A'
    Bt = B'

    # pullback
    function solve_sylvester_equation_pullback(∂P)
        ∂P₁ = unthunk(∂P[1])
        if ℒ.norm(∂P₁) < tol.rtol
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        cache_valid = cache_eligible &&
                      pow_iters_captured >= 1 &&
                      𝕊ℂ.pow_transposed
        saved_capture = 𝕊ℂ.pow_capture
        if cache_valid
            𝕊ℂ.pow_iters = pow_iters_captured
            𝕊ℂ.pow_capture = false
        end
        ∂C, slvd = solve_sylvester_equation(At, Bt, ∂P₁, 𝕊ℂ,
                                            sylvester_algorithm = sylvester_algorithm,
                                            preconditioner = preconditioner,
                                            tol = tol,
                                            verbose = verbose)
        𝕊ℂ.pow_capture = saved_capture
        𝕊ℂ.pow_iters = 0

        solved = solved && slvd

        if !slvd
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        # ∂C is n×m, B' is m×m, P_cached is n×m, A is n×n
        # Intermediate products are n×m and m×n — not n×n or m×m,
        # so workspace buffers 𝐀 (n×n) / 𝐁 (m×m) are wrong shape when n ≠ m.
        ∂A = (∂C * Bt) * P_cached'
        ∂B = (P_cached' * At) * ∂C

        return NoTangent(), ∂A, ∂B, ∂C, NoTangent()
    end

    return (P_cached, solved), solve_sylvester_equation_pullback
end

function rrule(::typeof(solve_lyapunov_equation),
                A::AbstractMatrix{Float64},
                C::AbstractMatrix{Float64},
                workspace::lyapunov_workspace;
                initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                lyapunov_algorithm::Symbol = :doubling,
                tol::SolverTolerances = SolverTolerances(atol = 1e-14,
                                                                                rtol = 1e-14,
                                                          initial_guess_acceptance_tol = 1e-12,
                                                          acceptance_tol = 1e-12),
                # timer::TimerOutput = TimerOutput(),
                verbose::Bool = false,
                has_unit_roots::Bool = false)

    # Enable doubling-power capture for the :doubling algorithm path.
    # With pow_transposed=true, powers are stored in transposed form directly.
    if lyapunov_algorithm == :doubling
        workspace.pow_iters = 0
        workspace.pow_capture = true
        workspace.pow_transposed = true
    end
    P, solved = solve_lyapunov_equation(A, C, workspace,
                            initial_guess = initial_guess,
                            lyapunov_algorithm = lyapunov_algorithm,
                            tol = tol,
                            verbose = verbose,
                            has_unit_roots = has_unit_roots)
    workspace.pow_capture = false
    pow_iters_captured = workspace.pow_iters
    workspace.pow_iters = 0
    if size(workspace.P) != size(P)
        workspace.P = zeros(eltype(P), size(P)...)
    end
    copyto!(workspace.P, P)
    P_cached = workspace.P
    ensure_lyapunov_doubling_buffers!(workspace)
    A_dense = collect(A)
    # Precompute Aᵀ once outside the pullback closure: needed by the matmul
    # forming ∂A (every call) and by the fallback adjoint solve.
    At = A_dense'
    ∂A_buf = zeros(eltype(A), size(A))

    # pullback 
    # https://arxiv.org/abs/2011.11430  
    function solve_lyapunov_equation_pullback(∂P)
        ∂P₁ = unthunk(∂P[1])
        if ℒ.norm(∂P₁) < tol.rtol
            return NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        # Adjoint Lyapunov: ∂P is generally not symmetric, so issymmetric will route to full-space.
        # Prefer the forward dense doubling solver in replay mode against the
        # transposed power cache when the forward pass populated workspace.𝐀_pow;
        # otherwise fall back to the legacy solver call.
        cache_valid = lyapunov_algorithm == :doubling &&
                      pow_iters_captured >= 1 &&
                      workspace.pow_transposed
        saved_capture = workspace.pow_capture
        if cache_valid
            workspace.pow_iters = pow_iters_captured
            workspace.pow_capture = false
        end
        ∂C, slvd = solve_lyapunov_equation(At, Matrix{Float64}(∂P₁), workspace,
                                           lyapunov_algorithm = lyapunov_algorithm,
                                           tol = tol,
                                           verbose = verbose)
        workspace.pow_capture = saved_capture
        workspace.pow_iters = 0
    
        solved = solved && slvd

        if !slvd
            return NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        tmp_n1 = workspace.𝐂A
        tmp_n2 = workspace.𝐀²
        ∂A = ∂A_buf
        fill!(∂A, 0)

        ℒ.mul!(tmp_n1, ∂C, A_dense)
        ℒ.mul!(∂A, tmp_n1, P_cached')

        ℒ.mul!(tmp_n2, ∂C', A_dense)
        ℒ.mul!(∂A, tmp_n2, P_cached, 1, 1)

        return NoTangent(), ∂A, ∂C, NoTangent()
    end
    
    return (P_cached, solved), solve_lyapunov_equation_pullback
end

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
        ∂x = vcat(unthunk(∂x[1]), zero(λ))

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
        ∂x = vcat(unthunk(∂x[1]), zero(λ))

        S = -fXλp' \ ∂x

        ∂shock_independent = S[length(initial_guess)+1:end]
        
        ∂𝐒ⁱ = ℒ.kron(S[1:length(initial_guess)], λ) - ℒ.kron(x, S[length(initial_guess)+1:end])

        ∂𝐒ⁱ²ᵉ = 2 * ℒ.kron(S[1:length(initial_guess)], xλ) - ℒ.kron(kron_buffer, S[length(initial_guess)+1:end])
        
        ∂𝐒ⁱ³ᵉ = 3 * ℒ.kron(S[1:length(initial_guess)], xxλ) - ℒ.kron(kron_buffer²,S[length(initial_guess)+1:end])

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),  ∂𝐒ⁱ, ∂𝐒ⁱ²ᵉ, ∂𝐒ⁱ³ᵉ, ∂shock_independent, NoTangent(), NoTangent()
    end

    return (x, matched), find_shocks_pullback
end


function rrule(::typeof(calculate_loglikelihood_with_missing), ::Val{:inversion}, ::Val{:first_order},
                                       observables_index::Vector{Int},
                                              𝐒::Matrix{Float64},
                                              data_in_deviations::Matrix{Float64},
                                              constants::constants,
                                              state::Vector{Vector{Float64}},
                                              workspaces::workspaces,
                                              obs_idx_per_t::Vector{Vector{Int}};
                                              warmup_iterations::Int = 0,
                                              on_failure_loglikelihood = -Inf,
                                              presample_periods::Int = 0,
                                              initial_covariance::Symbol = :theoretical,
                                              opts::CalculationOptions = merge_calculation_options(),
                                              filter_algorithm::Symbol = :LagrangeNewton)
    Tcc = constants.post_model_macro
    n_exo = Tcc.nExo
    obs_idx_full = observables_index
    n_obs_full = length(obs_idx_full)
    t⁻ = Tcc.past_not_future_and_mixed_idx
    n_past = length(t⁻)
    Tt = size(data_in_deviations, 2)

    eff_presample = presample_periods + warmup_iterations

    ws = workspaces.inversion
    ensure_inversion_buffers!(ws, n_exo, n_past)
    ensure_inversion_estimation_buffers!(ws, n_exo, n_obs_full)
    ensure_inversion_rrule_buffers!(ws, n_exo, n_past, n_obs_full, Tt; order = :first_order)

    state₀ = copy(state[1])
    state_seq = ws.state_seq_rrule
    @inbounds for t in 1:Tt+1
        if !isassigned(state_seq, t) || length(state_seq[t]) != length(state₀)
            state_seq[t] = copy(state₀)
        else
            copyto!(state_seq[t], state₀)
        end
    end

    # full observation Jacobian (constant across periods)
    jac_full = 𝐒[obs_idx_full, end-n_exo+1:end]
    𝐒obs_past_full = 𝐒[obs_idx_full, 1:end-n_exo]

    # per-period storage (cached in workspace; see ensure_inversion_rrule_buffers!)
    x_seq = ws.x_seq_rrule
    idx_seq = obs_idx_per_t
    invjac_v_seq = ws.invjac_v_seq_rrule  # m_t × m_t inverse for square; pinv-T for wide
    G_seq = ws.G_seq_rrule                # (jac_v jac_v')^{-1}, m_t × m_t
    n_obs_total = 0
    shocks² = 0.0
    logabsdets = 0.0
    concat_buf = ws.state_concat

    for t in 1:Tt
        idx = idx_seq[t]
        m = length(idx)
        # y_full = data[:,t] - 𝐒obs_past_full * state[t][t⁻]
        y_full_view = view(ws.obs_sub_buf, 1:n_obs_full)
        copyto!(y_full_view, view(data_in_deviations, :, t))
        ℒ.mul!(y_full_view, 𝐒obs_past_full, view(state_seq[t], t⁻), -1.0, 1.0)
        if m == 0
            fill!(x_seq[t], 0.0)
            invjac_v_seq[t] = zeros(0, 0)
            G_seq[t] = zeros(0, 0)
        else
            jac_v_buf = view(ws.jacc_v_buf, 1:m, 1:n_exo)
            @inbounds for j in 1:n_exo, i in 1:m
                jac_v_buf[i, j] = jac_full[idx[i], j]
            end
            jac_v = jac_v_buf
            y_v   = view(y_full_view, idx)
            logabsdet_t = 0.0
            if m == n_exo
                jac_v_lu = ℒ.lu(jac_v, check = false)
                if !ℒ.issuccess(jac_v_lu)
                    if opts.verbose println("Inversion filter rrule (missing) failed at step $t") end
                    return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
                end
                xv = jac_v_lu \ y_v
                invjac_v = inv(jac_v_lu)
                G = invjac_v' * invjac_v  # = (jac_v jac_v')^{-1}
                logabsdet_t = ℒ.logabsdet(jac_v_lu)[1]
            else
                # m < n_exo (or > n_exo handled below)
                if m > n_exo
                    if opts.verbose println("Inversion filter rrule (missing) failed at step $t: m=$m > n_exo=$n_exo") end
                    return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
                end
                JJt = jac_v * jac_v'
                JJt_lu = ℒ.lu(JJt, check = false)
                if !ℒ.issuccess(JJt_lu)
                    if opts.verbose println("Inversion filter rrule (missing) failed at step $t (rank-deficient row block)") end
                    return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
                end
                G = inv(JJt_lu)
                # x = jac_v' * G * y_v (min-norm solution)
                xv = jac_v' * (G * y_v)
                invjac_v = G * jac_v  # this is pinv(jac_v)' (m × n_exo); useful below
                logabsdet_t = ℒ.logabsdet(JJt_lu)[1] / 2
            end
            x_seq[t] .= xv
            invjac_v_seq[t] = invjac_v
            G_seq[t] = G

            if t > eff_presample
                shocks² += sum(abs2, xv)
                logabsdets += logabsdet_t
                n_obs_total += m
                if !isfinite(shocks²) || !isfinite(logabsdets)
                    return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
                end
            end
        end
        # state[t+1] = 𝐒 * vcat(state[t][t⁻], x_seq[t])
        copyto!(concat_buf, 1, view(state_seq[t], t⁻), 1, n_past)
        copyto!(concat_buf, n_past + 1, x_seq[t], 1, n_exo)
        ℒ.mul!(state_seq[t+1], 𝐒, concat_buf)
    end

    llh = -(logabsdets + shocks² + n_obs_total * log(2π)) / 2
    if llh < -1e12 || !isfinite(llh)
        return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    n_vars = size(𝐒, 1)
    n_cols = size(𝐒, 2)

    # Pre-allocate adjoint buffers outside the pullback closure (mirrors the
    # dense rrule pattern). The closure resets them with fill! on each call.
    ∂𝐒 = zero(𝐒)
    ∂data_in_deviations = zero(data_in_deviations)
    ∂state_t⁻ = zeros(n_past)
    v_buf = zeros(n_cols)
    ∂state_full_next = zeros(n_vars)
    ∂v = zeros(n_cols)
    ∂state₀_full = zeros(size(state₀))

    function inversion_pullback_missing(∂llh)
        fill!(∂𝐒, 0)
        fill!(∂data_in_deviations, 0)
        fill!(∂state_t⁻, 0)
        fill!(v_buf, 0)
        fill!(∂state_full_next, 0)
        fill!(∂v, 0)
        fill!(∂state₀_full, 0)

        # Backward pass over periods
        for t in Tt:-1:1
            # state[t+1] = 𝐒 * v where v = vcat(state[t][t⁻], x_seq[t])
            copyto!(v_buf, 1, view(state_seq[t], t⁻), 1, n_past)
            copyto!(v_buf, n_past + 1, x_seq[t], 1, n_exo)
            # Cotangent on state[t+1] enters via the state recursion.
            # ∂state[t+1] is the accumulated cotangent on state[t+1] from later steps;
            # we represent it in its t⁻ projection only for the next iteration.
            # Build full ∂state[t+1] by lifting ∂state_t⁻ into the n_vars-sized vector:
            fill!(∂state_full_next, 0.0)
            ∂state_full_next[t⁻] .= ∂state_t⁻

            # ∂𝐒 += ∂state_full_next * v'
            ℒ.mul!(∂𝐒, ∂state_full_next, v_buf', 1.0, 1.0)

            # ∂v = 𝐒' * ∂state_full_next
            ℒ.mul!(∂v, 𝐒', ∂state_full_next)
            copyto!(∂state_t⁻, view(∂v, 1:n_past))
            ∂x_t = view(∂v, n_past+1:n_cols)

            idx = idx_seq[t]
            m = length(idx)

            if m > 0 && t > eff_presample
                # shocks² adds: ∂x_t += -∂llh * x_seq[t]
                ∂x_t = ∂x_t .+ (-∂llh) .* x_seq[t]
            end

            if m > 0
                # x_seq[t] = pinv(jac_v) * y_v, where y_v = data[idx,t] - (𝐒obs_past)[idx,:] * state[t][t⁻]
                # ∂y_v = (jac_v^+)' * ∂x_t
                # For square m == n_exo: invjac_v_seq[t] = jac_v^{-1}; (jac_v^+)' = invjac_v'
                # For wide m < n_exo:    invjac_v_seq[t] = pinv(jac_v)' = G * jac_v; so (pinv)' (above name) is invjac_v'... careful.
                if m == n_exo
                    invjac_v = invjac_v_seq[t]   # = jac_v^{-1}, m × m
                    # ∂y_v = invjac_v' * ∂x_t
                    ∂y_v = invjac_v' * ∂x_t
                    # ∂jac_v from x = invjac * y: ∂jac_v += -invjac' * ∂x * x'  → here x = x_seq[t]
                    ∂jac_v = -(invjac_v' * ∂x_t) * x_seq[t]'
                else
                    G = G_seq[t]
                    jac_v = jac_full[idx, :]
                    pinvA = jac_v' * G            # n_exo × m  (= pinv(jac_v))
                    pinvA_T = G * jac_v           # m × n_exo  (= pinv(jac_v)')
                    ∂y_v = pinvA_T * ∂x_t
                    # ∂A = -(pinvA)' * ∂x_t * x_seq[t]'  + (I - pinvA * jac_v) * ∂x_t * g'
                    # since x_seq[t] ∈ row(jac_v), the second term contribution to shocks² part
                    # vanishes; but ∂x_t includes the state-recursion contribution which is NOT
                    # guaranteed to lie in the same projection. So we keep both terms.
                    g_t = G * (data_in_deviations[idx, t] - 𝐒obs_past_full[idx, :] * state_seq[t][t⁻])
                    P_perp = ℒ.I(n_exo) - pinvA * jac_v
                    ∂jac_v = -(pinvA' * ∂x_t) * x_seq[t]' + g_t * (P_perp * ∂x_t)'
                end

                # logabsdet[t] term (only if t > eff_presample): ∂jac_v += -∂llh/2 * pinv(jac_v)'
                if t > eff_presample
                    if m == n_exo
                        invjac_v = invjac_v_seq[t]
                        ∂jac_v = ∂jac_v .+ (-∂llh / 2) .* invjac_v'
                    else
                        G = G_seq[t]
                        jac_v = jac_full[idx, :]
                        pinvA_T = G * jac_v
                        ∂jac_v = ∂jac_v .+ (-∂llh / 2) .* pinvA_T
                    end
                end

                # Scatter ∂jac_v into ∂𝐒[obs_idx_full[idx], end-n_exo+1:end]
                rows = obs_idx_full[idx]
                col_off = size(𝐒, 2) - n_exo
                @inbounds for j in 1:n_exo
                    c = col_off + j
                    for i in 1:m
                        ∂𝐒[rows[i], c] += ∂jac_v[i, j]
                    end
                end

                # ∂y_v contributes to ∂data_in_deviations[idx, t] (data is indexed by
                # observable position, NOT full variable position) and to ∂state[t][t⁻]
                # via -𝐒obs_past_full[idx, :]
                @inbounds for i in 1:m
                    ∂data_in_deviations[idx[i], t] += ∂y_v[i]
                end
                # ∂(𝐒obs_past_full[idx,:]) += -∂y_v * state[t][t⁻]'  → goes into ∂𝐒[rows_full, 1:end-n_exo]
                stm = state_seq[t][t⁻]
                @inbounds for j in 1:n_past
                    sj = stm[j]
                    for i in 1:m
                        ∂𝐒[rows[i], j] += -∂y_v[i] * sj
                    end
                end
                # ∂state[t][t⁻] += -𝐒obs_past_full[idx,:]' * ∂y_v
                ∂state_t⁻ = ∂state_t⁻ .+ (-𝐒obs_past_full[idx, :]' * ∂y_v)
            end
            # for m == 0: nothing else; state[t+1] depended only on state[t][t⁻] and x=0
        end

        # Initial state cotangent: ∂state[1] is zero except at t⁻ entries.
        ∂state₀_full[t⁻] .= ∂state_t⁻

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂𝐒, ∂data_in_deviations, NoTangent(), [∂state₀_full], NoTangent()
    end

    return llh, inversion_pullback_missing
end


# Per-period O(Tt) backward sweep of the dense first-order inversion-filter
# pullback. Extracted into a standalone function so the compiler concretely
# specialises on every argument type. When this body lived directly inside the
# closure, scalar indexing of the captured ∂𝐒/state inferred as `Any`, costing
# ~600 allocs per period (~100k allocs total on SW07-class problems).
function dense_first_order_inv_pullback_loop!(
        ∂𝐒::Matrix{Float64},
        ∂data_in_deviations::Matrix{Float64},
        ∂state_t⁻::Vector{Float64},
        v_buf::Vector{Float64},
        ∂state_full_next::Vector{Float64},
        ∂v::Vector{Float64},
        ∂y::Vector{Float64},
        state::Vector{Vector{Float64}},
        x::Vector{Vector{Float64}},
        data_in_deviations::Matrix{Float64},
        𝐒::Matrix{Float64},
        𝐒obs_v::SubArray{Float64,2,Matrix{Float64}},
        invjac::Matrix{Float64},
        jac::Matrix{Float64},
        G_fat::Matrix{Float64},
        obs_idx::Vector{Int},
        t⁻::Vector{Int},
        Tt::Int,
        n_pnf::Int,
        n_cols::Int,
        n_obs_loc::Int,
        nExo::Int,
        presample_periods::Int,
        square_case::Bool,
    )
    𝐒T       = 𝐒'
    𝐒obs_v_T = 𝐒obs_v'
    invjac_T = invjac'
    v_buf_T  = v_buf'
    col_off  = n_cols - nExo

    @inbounds for t in Tt:-1:1
        fill!(∂state_full_next, 0.0)
        for k in 1:n_pnf
            ∂state_full_next[t⁻[k]] = ∂state_t⁻[k]
        end

        st_t = state[t]
        x_t  = x[t]
        for k in 1:n_pnf
            v_buf[k] = st_t[t⁻[k]]
        end
        copyto!(v_buf, n_pnf + 1, x_t, 1, nExo)

        ℒ.mul!(∂𝐒, ∂state_full_next, v_buf_T, 1.0, 1.0)
        ℒ.mul!(∂v, 𝐒T, ∂state_full_next)
        for k in 1:n_pnf
            ∂state_t⁻[k] = ∂v[k]
        end

        if t > presample_periods
            for k in 1:nExo
                ∂v[n_pnf + k] -= x_t[k]
            end
        end

        ∂x_view = view(∂v, n_pnf + 1 : n_pnf + nExo)

        if square_case
            ℒ.mul!(∂y, invjac_T, ∂x_view)
        else
            jac_∂x = jac * ∂x_view
            ℒ.mul!(∂y, G_fat, jac_∂x)
        end

        for i in 1:n_obs_loc
            ∂data_in_deviations[i, t] += ∂y[i]
        end

        for j in 1:n_pnf
            sj = st_t[t⁻[j]]
            for i in 1:n_obs_loc
                ∂𝐒[obs_idx[i], j] -= ∂y[i] * sj
            end
        end

        ℒ.mul!(∂state_t⁻, 𝐒obs_v_T, ∂y, -1.0, 1.0)

        if square_case
            for j in 1:nExo
                xtj = x_t[j]
                c = col_off + j
                for i in 1:n_obs_loc
                    ∂𝐒[obs_idx[i], c] -= ∂y[i] * xtj
                end
            end
        else
            y_t = data_in_deviations[:, t] - 𝐒obs_v * view(st_t, t⁻)
            g_t = G_fat * y_t
            Jdx    = jac * ∂x_view
            GJdx   = G_fat * Jdx
            JtGJdx = jac' * GJdx
            for j in 1:nExo
                xtj   = x_t[j]
                pterm = ∂x_view[j] - JtGJdx[j]
                c     = col_off + j
                for i in 1:n_obs_loc
                    ∂𝐒[obs_idx[i], c] += -∂y[i] * xtj + g_t[i] * pterm
                end
            end
        end
    end
    return nothing
end


function rrule(::typeof(calculate_loglikelihood), 
                ::Val{:inversion},
                ::Val{:first_order}, 
                observables_index::Vector{Int},
                𝐒::Matrix{Float64}, 
                data_in_deviations::Matrix{Float64}, 
                constants::constants,
                state::Vector{Vector{Float64}}, 
                workspaces::workspaces; 
                # timer::TimerOutput = TimerOutput(),
                warmup_iterations::Int = 0, 
                on_failure_loglikelihood = -Inf,
                presample_periods::Int = 0,
                initial_covariance::Symbol = :theoretical,
                opts::CalculationOptions = merge_calculation_options(),
                filter_algorithm::Symbol = :LagrangeNewton)
    T = constants.post_model_macro
    ws = workspaces.inversion
    # @timeit_debug timer "Inversion filter - forward" begin    

    # first order
    state = copy(state[1])

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    obs_idx = observables_index

    t⁻ = T.past_not_future_and_mixed_idx

    shocks² = 0.0
    logabsdets = 0.0

    # Warmup forward pass.  When `warmup_iterations > 0` we build a
    # block-concatenated jacobian, solve a min-norm linear system to recover
    # `warmup_iterations` worth of shocks, propagate the state through the
    # warmup window, and add the corresponding contributions to `logabsdets`
    # and `shocks²`.  Intermediates are captured so the pullback can backprop
    # through the linear solve, the state propagation and the jacobian
    # construction.
    warmup_jac           = zeros(0, 0)
    warmup_x             = zeros(0)
    warmup_y             = zeros(0)              # = inv(JJt) * data[:,1]   (fat case only)
    warmup_state_history = Vector{Vector{Float64}}()
    warmup_Sᵉ_powers     = Matrix{Float64}[]     # [I, Sᵉ, Sᵉ², …, Sᵉ^(N-2)]
    warmup_data_first    = zeros(length(obs_idx))

    if warmup_iterations > 0
        warmup_data_first = collect(data_in_deviations[:,1])

        warmup_jac = 𝐒[obs_idx, end-T.nExo+1:end]

        if warmup_iterations >= 2
            warmup_jac = hcat(𝐒[obs_idx, 1:T.nPast_not_future_and_mixed] *
                              𝐒[t⁻, end-T.nExo+1:end], warmup_jac)
            push!(warmup_Sᵉ_powers, Matrix{Float64}(ℒ.I, T.nPast_not_future_and_mixed,
                                                     T.nPast_not_future_and_mixed))   # Sᵉ^0
            if warmup_iterations >= 3
                Sᵉ_pow = 𝐒[t⁻, 1:T.nPast_not_future_and_mixed]
                push!(warmup_Sᵉ_powers, copy(Sᵉ_pow))                                  # Sᵉ^1
                for e in 1:warmup_iterations-2
                    warmup_jac = hcat(𝐒[obs_idx, 1:T.nPast_not_future_and_mixed] *
                                      Sᵉ_pow * 𝐒[t⁻, end-T.nExo+1:end], warmup_jac)
                    if e < warmup_iterations - 2
                        Sᵉ_pow = Sᵉ_pow * 𝐒[t⁻, 1:T.nPast_not_future_and_mixed]
                        push!(warmup_Sᵉ_powers, copy(Sᵉ_pow))
                    end
                end
            end
        end

        # Solve linear system
        if size(warmup_jac, 1) == size(warmup_jac, 2)
            warmup_lu = ℒ.lu(warmup_jac, check = false)
            if !ℒ.issuccess(warmup_lu)
                if opts.verbose println("Inversion filter failed (warmup, rrule)") end
                return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end
            warmup_x = warmup_lu \ warmup_data_first
        else
            JJt_w    = warmup_jac * warmup_jac'
            JJt_w_lu = ℒ.lu(JJt_w, check = false)
            if !ℒ.issuccess(JJt_w_lu)
                if opts.verbose println("Inversion filter failed (warmup, rrule)") end
                return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end
            warmup_y = JJt_w_lu \ warmup_data_first
            warmup_x = warmup_jac' * warmup_y
        end

        warmup_shocks_mat = reshape(warmup_x, T.nExo, warmup_iterations)

        # State propagation across warmup window
        st_local = copy(state)
        push!(warmup_state_history, copy(st_local))
        for i in 1:warmup_iterations-1
            st_concat = vcat(st_local[t⁻], warmup_shocks_mat[:,i])
            st_local  = 𝐒 * st_concat
            push!(warmup_state_history, copy(st_local))
        end
        state = st_local

        # NOTE: We deliberately do NOT add per-block logabsdets here.  The
        # primal in `src/filter/inversion.jl` accumulates them at lines 90-97
        # but then unconditionally overwrites `logabsdets` at lines 119/133/145
        # before the main loop scales it by `(n_obs - presample)`.  As a result,
        # warmup logabsdets contributions never enter `llh`, so the rrule must
        # not produce gradients for them either.

        shocks² += sum(abs2, warmup_x)
    end

    state = [copy(state) for _ in 1:size(data_in_deviations,2)+1]

    y = zeros(length(obs_idx))
    x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]

    jac = 𝐒[obs_idx,end-T.nExo+1:end]

    if T.nExo == length(observables_index)
        lu_ws = FastLapackInterface.LUWs(jac)
        lu_ws, _, ok, lu_handle = factorize_lu!(Val(:FastLapack), jac, lu_ws, size(jac))

        if !ok
            if opts.verbose println("Inversion filter failed") end
            return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end

        logabsdets = 0.0
        @inbounds for k in 1:size(jac,1)
            logabsdets += log(abs(jac[k,k]))
        end
        invjac = Matrix{Float64}(ℒ.I, size(jac))
        solve_lu_left!(jac, invjac, lu_ws, lu_handle)
    else
        logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(jac)) #' ./ precision_factor
        # jacdecomp = ℒ.svd(jac)
        invjac = ℒ.pinv(jac)
    end

    logabsdets *= size(data_in_deviations,2) - presample_periods

    if !isfinite(logabsdets) 
        return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    @views 𝐒obs = 𝐒[obs_idx,1:end-T.nExo]
    # Pre-slice 𝐒 to past_not_future_and_mixed_idx — the only rows of state[i+1]
    # that are ever read downstream (state[i][t⁻] and ∂state[t⁻]). Matches the
    # primal at lines ~80/219 of src/filter/inversion.jl.
    n_pnf_local = length(t⁻)
    𝐒past = 𝐒[t⁻, :]
    state_past_buf  = zeros(Float64, n_pnf_local)
    concat_buf_fwd  = zeros(Float64, n_pnf_local + T.nExo)

    for i in axes(data_in_deviations,2)
        @views ℒ.mul!(y, 𝐒obs, state[i][t⁻])
        @views ℒ.axpby!(1, data_in_deviations[:,i], -1, y)
        ℒ.mul!(x[i],invjac,y)
        # x = 𝐒[obs_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[obs_idx,1:end-T.nExo] * state[t⁻])

        if i > presample_periods
            shocks² += sum(abs2,x[i])
            if !isfinite(shocks²) 
                return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end
        end

        @inbounds for k in 1:n_pnf_local
            concat_buf_fwd[k] = state[i][t⁻[k]]
        end
        @inbounds for k in eachindex(x[i])
            concat_buf_fwd[n_pnf_local + k] = x[i][k]
        end
        ℒ.mul!(state_past_buf, 𝐒past, concat_buf_fwd)
        @inbounds for k in 1:n_pnf_local
            state[i+1][t⁻[k]] = state_past_buf[k]
        end
        # state[i+1] =  𝐒 * vcat(state[i][t⁻], x[i])  (only t⁻ rows are ever read)
    end

    llh = -(logabsdets + shocks² + (length(observables_index) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
    
    if llh < -1e12
        return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    ∂𝐒 = zero(𝐒)
    ∂data_in_deviations = zero(data_in_deviations)
    ∂state = zero(state[1])

    n_obs_loc = length(obs_idx)
    n_pnf     = length(t⁻)
    n_cols    = size(𝐒, 2)
    n_vars    = size(𝐒, 1)
    Tt        = size(data_in_deviations, 2)

    # For the fat case (m < n_exo) we need G = (jac*jac')^{-1} so the pullback
    # can apply the analytic pseudoinverse adjoint (matches the missing-rrule
    # formulation). Precomputed once because jac is constant across periods.
    # Always typed as Matrix{Float64} (empty in the square case) to keep the
    # pullback closure type-stable.
    G_fat = zeros(0, 0)
    if T.nExo != n_obs_loc
        JJt_p = jac * jac'
        JJt_p_lu = ℒ.lu(JJt_p, check = false)
        if ℒ.issuccess(JJt_p_lu)
            G_fat = inv(JJt_p_lu)
        end
    end

    # Pre-allocate pullback buffers outside the closure; the closure resets
    # them with fill! on each invocation.
    ∂state_t⁻        = zeros(n_pnf)
    v_buf            = zeros(n_cols)
    ∂state_full_next = zeros(n_vars)
    ∂v               = zeros(n_cols)
    ∂y               = zeros(n_obs_loc)

    # end # timeit_debug
    # pullback — O(Tt) backward recursion mirroring the missing rrule's
    # pattern.  Replaces an earlier O(Tt²) explicit-unrolling implementation
    # that built powers of M² = 𝐒past' - 𝐒obs' * invjac' * 𝐒past_v' and
    # accumulated ∂𝐒t⁻ in a nested t/tt loop.  See PR #295 review for context.
    function inversion_pullback(∂llh)
        # @timeit_debug timer "Inversion filter - pullback" begin

        fill!(∂state_t⁻, 0)
        fill!(v_buf, 0)
        fill!(∂state_full_next, 0)
        fill!(∂v, 0)
        fill!(∂y, 0)

        𝐒obs_v = view(𝐒, obs_idx, 1:n_cols - T.nExo)
        square_case = (T.nExo == n_obs_loc)

        # Hot loop extracted into a function — see dense_first_order_inv_pullback_loop!
        dense_first_order_inv_pullback_loop!(
            ∂𝐒, ∂data_in_deviations, ∂state_t⁻, v_buf, ∂state_full_next, ∂v, ∂y,
            state, x, data_in_deviations, 𝐒, 𝐒obs_v, invjac, jac, G_fat,
            obs_idx, t⁻, Tt, n_pnf, n_cols, n_obs_loc, T.nExo,
            presample_periods, square_case,
        )

        col_off = n_cols - T.nExo

        # Constant-jac logabsdet contribution (scales with Tt - presample).
        # Square: ∂jac += -(Tt - p)/2 * invjac'
        # Fat   : ∂jac += -(Tt - p)/2 * (G * jac)    (since d log|det(JJt)|/2 / d jac = G * jac)
        factor = -(Tt - presample_periods) / 2
        if T.nExo == n_obs_loc
            invjac_T = invjac'
            for j in 1:T.nExo
                c = col_off + j
                for i in 1:n_obs_loc
                    ∂𝐒[obs_idx[i], c] += factor * invjac_T[i, j]
                end
            end
        else
            Gjac = G_fat * jac
            for j in 1:T.nExo
                c = col_off + j
                for i in 1:n_obs_loc
                    ∂𝐒[obs_idx[i], c] += factor * Gjac[i, j]
                end
            end
        end

        # Lift accumulated ∂state_t⁻ into the n_vars-sized ∂state (t⁻ slots).
        for k in 1:n_pnf
            ∂state[t⁻[k]] = ∂state_t⁻[k]
        end

        # ----- Warmup pullback ------------------------------------------------
        # Backprop through the warmup forward.  At this point ∂state holds the
        # adjoint of the state at the start of the main loop, which equals the
        # state at the end of the warmup window (i.e. ∂state_after_warmup).
        # We propagate it back through (i) state propagation across the warmup
        # window, (ii) the linear-solve recovery of the warmup shocks, and
        # (iii) the block-concatenated jacobian construction.
        if warmup_iterations > 0
            N    = warmup_iterations
            nExo = T.nExo
            n_pnf = T.nPast_not_future_and_mixed

            # ∂x_warmup gets contributions from (a) shocks² += sum(abs2, x_warmup)
            # and (b) the state-propagation backward sweep.
            ∂x_warmup = -copy(warmup_x)        # from shocks² (∂llh*-1/2 implicit)

            # Backprop state propagation (warmup_iterations-1 evolution steps).
            ∂state_local = copy(∂state)        # = ∂state_after_warmup
            for i in (N-1):-1:1
                state_concat_i = vcat(warmup_state_history[i][t⁻],
                                       warmup_x[(i-1)*nExo+1 : i*nExo])
                ∂𝐒 .+= ∂state_local * state_concat_i'
                ∂state_concat = 𝐒' * ∂state_local
                # ∂warmup_shocks[:,i] contribution
                ∂x_warmup[(i-1)*nExo+1 : i*nExo] .+= ∂state_concat[n_pnf+1:end]
                # Reset ∂state and inject t⁻ slots for previous step
                ∂state_local = zero(∂state_local)
                ∂state_local[t⁻] .= ∂state_concat[1:n_pnf]
            end
            # After the loop, ∂state_local is the gradient wrt state_initial,
            # supported only on t⁻ slots.  Override the ∂state we'll return.
            ∂state .= ∂state_local

            # ∂jac_concat collects contributions from the linear-solve adjoint
            # only.  We do NOT add per-block logabsdets contributions because
            # the primal silently overwrites the accumulated warmup logabsdets
            # before scaling (see note in the forward pass), so they don't
            # enter llh and must not enter the gradient.
            ∂jac_concat = zeros(size(warmup_jac))

            # Backprop the linear solve to recover warmup shocks.
            ∂data_first = zeros(length(obs_idx))
            if size(warmup_jac, 1) == size(warmup_jac, 2)
                # x = jac \ data;  ∂data = jac' \ ∂x;  ∂jac = -∂data * x'
                ∂data_first = warmup_jac' \ ∂x_warmup
                ∂jac_concat .-= ∂data_first * warmup_x'
            else
                # x = jac' * inv(JJt) * data,  JJt = jac*jac', y = inv(JJt)*data
                # ∂data = inv(JJt) * jac * ∂x
                # ∂jac  += y * ∂x' - ∂data * x' - y * (jac' * ∂data)'
                JJt_w   = warmup_jac * warmup_jac'
                ∂data_first = JJt_w \ (warmup_jac * ∂x_warmup)
                ∂jac_concat .+= warmup_y * ∂x_warmup'
                ∂jac_concat .-= ∂data_first * warmup_x'
                ∂jac_concat .-= warmup_y * (warmup_jac' * ∂data_first)'
            end
            ∂data_in_deviations[:,1] .+= ∂data_first

            # Map ∂jac_concat → ∂𝐒.
            # Block N is C = 𝐒[obs_idx, end-nExo+1:end].
            ∂𝐒[obs_idx, end-nExo+1:end] .+= ∂jac_concat[:, (N-1)*nExo+1 : N*nExo]
            # Blocks 1..N-1 are A * Sᵉ^(N-1-k) * B.
            if N >= 2
                A  = 𝐒[obs_idx, 1:n_pnf]
                B  = 𝐒[t⁻,   end-nExo+1:end]
                Sᵉ = 𝐒[t⁻,   1:n_pnf]
                ∂A  = zeros(size(A))
                ∂B  = zeros(size(B))
                ∂Sᵉ = zeros(size(Sᵉ))
                for k in 1:(N-1)
                    p     = N - 1 - k                         # power of Sᵉ
                    M     = warmup_Sᵉ_powers[p+1]             # Sᵉ^p (1-indexed)
                    ∂blk  = ∂jac_concat[:, (k-1)*nExo+1 : k*nExo]
                    ∂A   .+= ∂blk * (M * B)'
                    ∂B   .+= (A * M)' * ∂blk
                    if p >= 1
                        ∂M = A' * ∂blk * B'
                        for j in 0:p-1
                            Sj  = warmup_Sᵉ_powers[j+1]
                            Spj = warmup_Sᵉ_powers[p-j]       # Sᵉ^(p-1-j) → index p-j
                            ∂Sᵉ .+= Sj' * ∂M * Spj'
                        end
                    end
                end
                ∂𝐒[obs_idx, 1:n_pnf]      .+= ∂A
                ∂𝐒[t⁻,    end-nExo+1:end] .+= ∂B
                ∂𝐒[t⁻,    1:n_pnf]        .+= ∂Sᵉ
            end
        end
        # ----- end warmup pullback --------------------------------------------

        # end # timeit_debug

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂𝐒 * ∂llh, ∂data_in_deviations * ∂llh, NoTangent(), [∂state * ∂llh], NoTangent()
    end
    
    return llh, inversion_pullback
end


function rrule(::typeof(calculate_loglikelihood_with_missing), ::Val{:inversion}, ::Val{:pruned_second_order},
                                       observables_index::Vector{Int},
                                                     𝐒::Vector{AbstractMatrix{Float64}},
                                                     data_in_deviations::Matrix{Float64},
                                                     constants::constants,
                                                     state::Vector{Vector{Float64}},
                                                     workspaces::workspaces,
                                                     obs_idx_per_t::Vector{Vector{Int}};
                                                     warmup_iterations::Int = 0,
                                                     on_failure_loglikelihood = -Inf,
                                                     presample_periods::Int = 0,
                                                     initial_covariance::Symbol = :theoretical,
                                                     opts::CalculationOptions = merge_calculation_options(),
                                                     filter_algorithm::Symbol = :LagrangeNewton)
    Tcc = constants.post_model_macro
    n_exo  = Tcc.nExo
    n_past = Tcc.nPast_not_future_and_mixed
    cond_var_idx = observables_index
    n_cond = length(cond_var_idx)
    Tt = size(data_in_deviations, 2)

    eff_presample = presample_periods + warmup_iterations

    ws = workspaces.inversion
    ensure_inversion_buffers!(ws, n_exo, n_past)
    ensure_inversion_estimation_buffers!(ws, n_exo, n_cond)
    ensure_inversion_rrule_buffers!(ws, n_exo, n_past, n_cond, Tt; order = :pruned_second_order)

    cc = ensure_conditional_forecast_constants!(constants)
    shock_idxs     = cc.shock_idxs
    shock²_idxs    = cc.shock²_idxs
    shockvar²_idxs = cc.shockvar²_idxs
    var_vol²_idxs  = cc.var_vol²_idxs
    var²_idxs      = cc.var²_idxs

    𝐒⁻¹  = 𝐒[1][Tcc.past_not_future_and_mixed_idx, :]
    𝐒¹⁻  = 𝐒[1][cond_var_idx, 1:n_past]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:n_past+1]
    𝐒¹ᵉ  = 𝐒[1][cond_var_idx, end-n_exo+1:end]
    𝐒²⁻ᵛ = collect(𝐒[2][cond_var_idx, var_vol²_idxs])
    𝐒²⁻  = collect(𝐒[2][cond_var_idx, var²_idxs])
    𝐒²⁻ᵉ = collect(𝐒[2][cond_var_idx, shockvar²_idxs])
    𝐒²ᵉ  = collect(𝐒[2][cond_var_idx, shock²_idxs])
    𝐒⁻²  = collect(𝐒[2][Tcc.past_not_future_and_mixed_idx, :])

    𝐒ⁱ²ᵉ = 𝐒²ᵉ ./ 2
    J = ℒ.I(n_exo)

    state₁ = copy(state[1][Tcc.past_not_future_and_mixed_idx])
    state₂ = copy(state[2][Tcc.past_not_future_and_mixed_idx])

    # Per-period storage (cached in workspace; see ensure_inversion_rrule_buffers!)
    state₁_seq      = ws.state_seq_rrule
    state₂_seq      = ws.state₂_seq_rrule
    state₁_seq[1] .= state₁
    state₂_seq[1] .= state₂
    state¹⁻_vol_seq = ws.state¹⁻_vol_seq_rrule
    aug_state₁_seq  = ws.aug_state₁_seq_rrule
    aug_state₂_seq  = ws.aug_state₂_seq_rrule
    x_seq           = ws.x_seq_rrule
    𝐒ⁱ_full_seq    = ws.𝐒ⁱ_full_seq_rrule

    shocks² = 0.0
    logabsdets = 0.0
    n_obs_total = 0

    state¹⁻_vol       = ws.state_vol
    shock_independent = ws.shock_independent
    kronstate¹⁻_vol   = ws.kronstate_vol
    kron_buffer3      = ws.kron_buffer_state
    𝐒ⁱ_full          = ws.Si_buffer
    kronaug_state₁    = ws.kronaug_state
    init_guess        = ws.init_guess
    kron_buffer       = ws.kron_buffer
    kron_buffer2      = ws.kron_buffer2

    for t in 1:Tt
        idx = obs_idx_per_t[t]
        m = length(idx)

        copyto!(state¹⁻_vol, 1, state₁, 1)
        state¹⁻_vol[end] = 1.0

        copyto!(shock_independent, view(data_in_deviations, :, t))
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)
        ℒ.kron!(kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol, -1/2, 1)

        ℒ.kron!(kron_buffer3, J, state¹⁻_vol)
        ℒ.mul!(𝐒ⁱ_full, 𝐒²⁻ᵉ, kron_buffer3)
        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ_full)

        copyto!(state¹⁻_vol_seq[t], state¹⁻_vol)
        𝐒ⁱ_full_seq[t] .= 𝐒ⁱ_full

        if m == 0
            fill!(init_guess, 0.0)
            x = init_guess
        else
            if m > n_exo
                if opts.verbose println("Inversion filter rrule (pruned 2nd, missing) failed at step $t: m=$m > n_exo=$n_exo") end
                return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end
            𝐒ⁱ_v   = 𝐒ⁱ_full[idx, :]
            𝐒ⁱ²ᵉ_v = 𝐒ⁱ²ᵉ[idx, :]
            si_v   = shock_independent[idx]
            fill!(init_guess, 0.0)
            x, matched = find_shocks(Val(filter_algorithm),
                                     init_guess, kron_buffer, kron_buffer2, J,
                                     𝐒ⁱ_v, 𝐒ⁱ²ᵉ_v, si_v)
            if !matched
                if opts.verbose println("Inversion filter rrule (pruned 2nd, missing) failed at step $t") end
                return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end
            if t > eff_presample
                jac_v = similar(𝐒ⁱ_v)
                ℒ.kron!(kron_buffer2, J, x)
                ℒ.mul!(jac_v, 𝐒ⁱ²ᵉ_v, kron_buffer2)
                ℒ.axpby!(1, 𝐒ⁱ_v, 2, jac_v)
                logabsdets += m == n_exo ? ℒ.logabsdet(jac_v)[1] : ℒ.logabsdet(jac_v * jac_v')[1] / 2
                shocks² += sum(abs2, x)
                n_obs_total += m
                if !isfinite(logabsdets) || !isfinite(shocks²)
                    return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
                end
            end
        end
        x_seq[t] .= x

        # aug states
        copyto!(aug_state₁_seq[t], 1, state₁, 1)
        aug_state₁_seq[t][n_past + 1] = 1.0
        copyto!(aug_state₁_seq[t], n_past + 2, x, 1)
        copyto!(aug_state₂_seq[t], 1, state₂, 1)

        ℒ.mul!(state₁, 𝐒⁻¹, aug_state₁_seq[t])
        ℒ.mul!(state₂, 𝐒⁻¹, aug_state₂_seq[t])
        ℒ.kron!(kronaug_state₁, aug_state₁_seq[t], aug_state₁_seq[t])
        ℒ.mul!(state₂, 𝐒⁻², kronaug_state₁, 1/2, 1)
        copyto!(state₁_seq[t+1], state₁)
        copyto!(state₂_seq[t+1], state₂)
    end

    llh = -(logabsdets + shocks² + n_obs_total * log(2 * 3.141592653589793)) / 2

    if !isfinite(llh) || llh < -1e12
        return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # Pre-allocate adjoint buffers outside the pullback closure (mirrors the
    # dense non-missing rrule pattern at L9778+). The closure resets them with
    # fill! on each invocation.
    ∂𝐒_1 = zeros(size(𝐒[1]))
    ∂𝐒_2 = zeros(size(𝐒[2]))
    ∂𝐒⁻¹  = zero(𝐒⁻¹)
    ∂𝐒⁻²  = zero(𝐒⁻²)
    ∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)
    ∂𝐒¹⁻  = zero(𝐒¹⁻)
    ∂𝐒¹ᵉ  = zero(𝐒¹ᵉ)
    ∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)
    ∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)
    ∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)
    ∂data_in_deviations = zeros(size(data_in_deviations))

    ∂state₁_next = zeros(n_past)
    ∂state₂_next = zeros(n_past)

    kronaug_buf = zeros((n_past + 1 + n_exo)^2)
    ∂kronaug    = zeros((n_past + 1 + n_exo)^2)
    ∂aug_state₁ = zeros(n_past + 1 + n_exo)
    ∂aug_state₂ = zeros(n_past + 1 + n_exo)
    ∂kronstate  = zeros((n_past + 1)^2)
    ∂state¹⁻_vol = zeros(n_past + 1)
    ∂kronIstate = zeros(n_exo * (n_past + 1))
    # Hoisted per-period pullback buffers (max-size, used via views when m varies)
    ∂𝐒ⁱ_full_buf       = zeros(length(cond_var_idx), n_exo)
    ∂shock_independent  = zeros(length(cond_var_idx))
    ∂jac_v_buf          = zeros(length(cond_var_idx), n_exo)
    kron_Isv_buf        = zeros(n_exo * (n_past + 1), n_exo)
    ∂state₂_contrib     = zeros(n_past)
    ∂kron_sv            = zeros((n_past + 1)^2)
    kron_sv             = zeros((n_past + 1)^2)
    ∂kronIstate_local   = zeros(n_exo * (n_past + 1), n_exo)

    function pruned2_missing_pullback(∂llh)
        fill!(∂𝐒_1, 0); fill!(∂𝐒_2, 0)
        fill!(∂𝐒⁻¹, 0); fill!(∂𝐒⁻², 0)
        fill!(∂𝐒¹⁻ᵛ, 0); fill!(∂𝐒¹⁻, 0); fill!(∂𝐒¹ᵉ, 0)
        fill!(∂𝐒²⁻ᵛ, 0); fill!(∂𝐒²⁻ᵉ, 0); fill!(∂𝐒ⁱ²ᵉ, 0)
        fill!(∂data_in_deviations, 0)
        fill!(∂state₁_next, 0); fill!(∂state₂_next, 0)
        fill!(kronaug_buf, 0); fill!(∂kronaug, 0)
        fill!(∂aug_state₁, 0); fill!(∂aug_state₂, 0)
        fill!(∂kronstate, 0); fill!(∂state¹⁻_vol, 0); fill!(∂kronIstate, 0)
        fill!(∂𝐒ⁱ_full_buf, 0); fill!(∂shock_independent, 0)
        fill!(∂jac_v_buf, 0)
        fill!(kron_Isv_buf, 0); fill!(∂state₂_contrib, 0)
        fill!(∂kron_sv, 0); fill!(kron_sv, 0); fill!(∂kronIstate_local, 0)

        for t in Tt:-1:1
            aug_state₁ = aug_state₁_seq[t]
            aug_state₂ = aug_state₂_seq[t]
            state¹⁻_vol = state¹⁻_vol_seq[t]
            stm = state₁_seq[t]
            stm2 = state₂_seq[t]
            x = x_seq[t]
            idx = obs_idx_per_t[t]
            m = length(idx)

            # state₁_next = 𝐒⁻¹ * aug_state₁
            ℒ.mul!(∂𝐒⁻¹, ∂state₁_next, aug_state₁', 1, 1)
            ℒ.mul!(∂aug_state₁, 𝐒⁻¹', ∂state₁_next)

            # state₂_next = 𝐒⁻¹ * aug_state₂ + 0.5 * 𝐒⁻² * kron(aug_state₁, aug_state₁)
            ℒ.mul!(∂𝐒⁻¹, ∂state₂_next, aug_state₂', 1, 1)
            ℒ.mul!(∂aug_state₂, 𝐒⁻¹', ∂state₂_next)
            ℒ.kron!(kronaug_buf, aug_state₁, aug_state₁)
            ℒ.mul!(∂𝐒⁻², ∂state₂_next, kronaug_buf', 1/2, 1)
            ℒ.mul!(∂kronaug, 𝐒⁻²', ∂state₂_next)
            ℒ.rdiv!(∂kronaug, 2)
            fill_kron_adjoint!(∂aug_state₁, ∂aug_state₁, ∂kronaug, aug_state₁, aug_state₁)

            fill!(∂state₁_next, 0)
            fill!(∂state₂_next, 0)

            # split aug_state contributions
            ∂state₁_now = ∂aug_state₁[1:n_past]
            ∂x = ∂aug_state₁[n_past+2:end]
            ∂state₂_now = ∂aug_state₂[1:n_past]

            # accumulate into ∂state[t] which equals ∂state₁_next, ∂state₂_next for the prior period
            @inbounds for j in 1:n_past
                ∂state₁_next[j] += ∂state₁_now[j]
                ∂state₂_next[j] += ∂state₂_now[j]
            end

            # shocks² and logabsdet contributions (only if t > presample and m > 0)
            ∂jac_v = view(∂jac_v_buf, 1:m, :); fill!(∂jac_v, 0)
            jac_v_local = zeros(m, n_exo)
            𝐒ⁱ²ᵉ_v_local = zeros(m, n_exo^2)
            if m > 0
                𝐒ⁱ_v_local = 𝐒ⁱ_full_seq[t][idx, :]
                𝐒ⁱ²ᵉ_v_local = 𝐒ⁱ²ᵉ[idx, :]
                jac_v_local = 𝐒ⁱ_v_local + 2 * 𝐒ⁱ²ᵉ_v_local * ℒ.kron(J, x)
            end
            if m > 0 && t > eff_presample
                # ∂shocks² = -1/2 (from llh wrt shocks²); ∂x_k += -x_k
                @inbounds for k in 1:n_exo
                    ∂x[k] += -x[k]
                end
                # logabsdet pullback: ∂jac_v += -1/2 * pinv(jac_v)'
                if m == n_exo
                    invjac_v = inv(jac_v_local)
                    ∂jac_v .+= (-0.5) .* invjac_v'
                else
                    G = inv(jac_v_local * jac_v_local')
                    ∂jac_v .+= (-0.5) .* (G * jac_v_local)
                end
                # Add ∂jac_v's contribution to ∂x via the (I⊗x) term in jac_v.
                # d jac_v[i,r] / d x_l = 2 𝐒ⁱ²ᵉ_v[i, (r-1)n_exo + l]
                @inbounds for l in 1:n_exo
                    s = 0.0
                    for r in 1:n_exo
                        col = (r-1) * n_exo + l
                        for i_local in 1:m
                            s += ∂jac_v[i_local, r] * 𝐒ⁱ²ᵉ_v_local[i_local, col]
                        end
                    end
                    ∂x[l] += 2 * s
                end
            end

            fill!(∂shock_independent, 0)

            if m > 0
                𝐒ⁱ_v   = 𝐒ⁱ_full_seq[t][idx, :]
                𝐒ⁱ²ᵉ_v = 𝐒ⁱ²ᵉ_v_local
                jac_v  = jac_v_local

                # Lagrange multiplier from KKT (2x = jac_v' λ)
                local λ
                if m == n_exo
                    λ = 2 * (jac_v' \ x)
                else
                    Gloc = inv(jac_v * jac_v')
                    λ = 2 * (Gloc * (jac_v * x))
                end

                # KKT system: G(y; θ) = [2x - jac_v(x)' λ; F(x; θ)] = 0.
                # dG/dy = [2I - 2 reshape(𝐒ⁱ²ᵉ_v' λ, n, n)   -jac_v']
                #         [jac_v                              0      ]
                M = reshape(𝐒ⁱ²ᵉ_v' * λ, n_exo, n_exo)
                topL = 2 * ℒ.I(n_exo) - 2 * M
                fXλp = [topL          -jac_v'
                        jac_v          zeros(m, m)]

                # Adjoint: S = (dG/dy)^{-T} * vcat(∂x, 0); ∂θ contrib = -S^T * dG/dθ
                rhs = vcat(∂x, zeros(m))
                S = fXλp' \ rhs
                Sx = S[1:n_exo]
                Sλ = S[n_exo+1:end]

                # ∂v_v: dG/dv_v has -I in lower block. ∂v_v = -(-Sλ) = +Sλ.
                ∂v_v = Sλ

                # ∂𝐒ⁱ_v from KKT:
                #   dG_top[r]/d𝐒ⁱ_v[i,j] = -δ_{rj} λ[i] → contrib = -Sx[j] * (-λ[i]) = +λ[i] Sx[j]
                #   dG_F[i']/d𝐒ⁱ_v[i,j]  = δ_{ii'} x[j]  → contrib = -Sλ[i] * x[j]
                ∂𝐒ⁱ_v = λ * Sx' - Sλ * x'

                # ∂𝐒ⁱ²ᵉ_v from KKT:
                #   dG_top[r]/d𝐒ⁱ²ᵉ_v[i, (p-1)n_exo+q] = -2 δ_{rp} x_q λ[i] → contrib = +2 λ[i] Sx[p] x_q
                #   dG_F[i']/d𝐒ⁱ²ᵉ_v[i, (p-1)n_exo+q] = δ_{ii'} x_p x_q       → contrib = -Sλ[i] x_p x_q
                xSx = x * Sx'      # xSx[q,p] = x_q * Sx[p]
                xx_outer = x * x'  # symmetric
                ∂𝐒ⁱ²ᵉ_v_top = 2 * λ * vec(xSx)'
                ∂𝐒ⁱ²ᵉ_v_F   = -Sλ * vec(xx_outer)'
                ∂𝐒ⁱ²ᵉ_v_kkt = ∂𝐒ⁱ²ᵉ_v_top + ∂𝐒ⁱ²ᵉ_v_F

                # Add direct ∂jac_v contributions:
                #   ∂𝐒ⁱ_v    += ∂jac_v
                #   ∂𝐒ⁱ²ᵉ_v += 2 * ∂jac_v * kron(I, x)'
                if t > eff_presample
                    ∂𝐒ⁱ_v_total = ∂𝐒ⁱ_v + ∂jac_v
                    ∂𝐒ⁱ²ᵉ_v_total = ∂𝐒ⁱ²ᵉ_v_kkt + 2 * ∂jac_v * ℒ.kron(J, x)'
                else
                    ∂𝐒ⁱ_v_total = ∂𝐒ⁱ_v
                    ∂𝐒ⁱ²ᵉ_v_total = ∂𝐒ⁱ²ᵉ_v_kkt
                end

                # Scatter into ∂𝐒ⁱ_full and ∂𝐒ⁱ²ᵉ
                ∂𝐒ⁱ_full = ∂𝐒ⁱ_full_buf
                fill!(∂𝐒ⁱ_full, 0)
                @inbounds for j in 1:n_exo
                    for i_local in 1:m
                        ∂𝐒ⁱ_full[idx[i_local], j] = ∂𝐒ⁱ_v_total[i_local, j]
                    end
                end
                @inbounds for j in 1:n_exo^2
                    for i_local in 1:m
                        ∂𝐒ⁱ²ᵉ[idx[i_local], j] += ∂𝐒ⁱ²ᵉ_v_total[i_local, j]
                    end
                end

                # ∂shock_independent[idx] += ∂v_v
                @inbounds for i_local in 1:m
                    ∂shock_independent[idx[i_local]] += ∂v_v[i_local]
                end

                # Propagate ∂𝐒ⁱ_full back through 𝐒ⁱ_full = 𝐒¹ᵉ + 𝐒²⁻ᵉ (I⊗state¹⁻_vol)
                ∂𝐒¹ᵉ .+= ∂𝐒ⁱ_full
                ℒ.kron!(kron_Isv_buf, J, state¹⁻_vol)
                ℒ.mul!(∂𝐒²⁻ᵉ, ∂𝐒ⁱ_full, kron_Isv_buf', 1, 1)
                ℒ.mul!(∂kronIstate_local, 𝐒²⁻ᵉ', ∂𝐒ⁱ_full)
                @inbounds for p in 1:(n_past + 1)
                    s = 0.0
                    for j in 1:n_exo
                        s += ∂kronIstate_local[(j-1)*(n_past+1) + p, j]
                    end
                    ∂state¹⁻_vol[p] = s
                end
            else
                fill!(∂state¹⁻_vol, 0)
            end

            # Now propagate shock_independent dependencies
            # shock_independent = data[:,t] - 𝐒¹⁻ᵛ * state¹⁻_vol - 𝐒¹⁻ * state₂ - 0.5 * 𝐒²⁻ᵛ * (state¹⁻_vol ⊗ state¹⁻_vol)
            # ∂data[:,t] += ∂shock_independent
            @inbounds for i in 1:length(cond_var_idx)
                ∂data_in_deviations[i, t] += ∂shock_independent[i]
            end
            # ∂𝐒¹⁻ᵛ -= ∂shock_independent * state¹⁻_vol'
            ℒ.mul!(∂𝐒¹⁻ᵛ, ∂shock_independent, state¹⁻_vol', -1, 1)
            # ∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent
            ℒ.mul!(∂state¹⁻_vol, 𝐒¹⁻ᵛ', ∂shock_independent, -1, 1)
            # ∂𝐒¹⁻ -= ∂shock_independent * state₂'  (note: 𝐒¹⁻ * state₂ uses state₂ which is stm2_seq... actually it's the CURRENT state₂ at start of period: state₂_seq[t])
            ℒ.mul!(∂𝐒¹⁻, ∂shock_independent, stm2', -1, 1)
            # ∂state₂_now (== ∂state₂_next for prior period via accumulation) -= 𝐒¹⁻' * ∂shock_independent
            ∂state₂_contrib = 𝐒¹⁻' * ∂shock_independent
            @inbounds for j in 1:n_past
                ∂state₂_next[j] += -∂state₂_contrib[j]
            end
            # ∂𝐒²⁻ᵛ -= 0.5 * ∂shock_independent * kron(s¹⁻_vol, s¹⁻_vol)'
            kron_sv = ℒ.kron(state¹⁻_vol, state¹⁻_vol)
            ℒ.mul!(∂𝐒²⁻ᵛ, ∂shock_independent, kron_sv', -1/2, 1)
            # ∂kron_sv = -0.5 * 𝐒²⁻ᵛ' * ∂shock_independent
            ∂kron_sv = -(𝐒²⁻ᵛ' * ∂shock_independent) ./ 2
            fill!(∂kronstate, 0)
            ∂kronstate .+= ∂kron_sv
            # fill_kron_adjoint!(∂A, ∂B, ∂X, A, B) for vectors; A = B = state¹⁻_vol; ∂A = ∂B = ∂state¹⁻_vol
            fill_kron_adjoint!(∂state¹⁻_vol, ∂state¹⁻_vol, ∂kronstate, state¹⁻_vol, state¹⁻_vol)

            # state¹⁻_vol = vcat(state₁, 1) → ∂state₁ += ∂state¹⁻_vol[1:n_past]
            @inbounds for j in 1:n_past
                ∂state₁_next[j] += ∂state¹⁻_vol[j]
            end
        end

        # Apply ∂llh scaling and assemble ∂𝐒
        ∂𝐒_1[Tcc.past_not_future_and_mixed_idx, :]                 .+= ∂𝐒⁻¹
        ∂𝐒_2[Tcc.past_not_future_and_mixed_idx, :]                 .+= ∂𝐒⁻²
        ∂𝐒_1[cond_var_idx, 1:n_past+1]                              .+= ∂𝐒¹⁻ᵛ
        ∂𝐒_1[cond_var_idx, 1:n_past]                                .+= ∂𝐒¹⁻
        ∂𝐒_1[cond_var_idx, end-n_exo+1:end]                         .+= ∂𝐒¹ᵉ
        ∂𝐒_2[cond_var_idx, var_vol²_idxs]                           .+= ∂𝐒²⁻ᵛ
        ∂𝐒_2[cond_var_idx, shockvar²_idxs]                          .+= ∂𝐒²⁻ᵉ
        # 𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 → ∂𝐒²ᵉ = ∂𝐒ⁱ²ᵉ / 2
        ∂𝐒_2[cond_var_idx, shock²_idxs]                             .+= ∂𝐒ⁱ²ᵉ ./ 2

        ℒ.rmul!(∂𝐒_1, ∂llh)
        ℒ.rmul!(∂𝐒_2, ∂llh)
        ℒ.rmul!(∂data_in_deviations, ∂llh)

        ∂state₀_full_1 = zeros(size(state[1]))
        ∂state₀_full_2 = zeros(size(state[2]))
        ∂state₀_full_1[Tcc.past_not_future_and_mixed_idx] .= ∂state₁_next .* ∂llh
        ∂state₀_full_2[Tcc.past_not_future_and_mixed_idx] .= ∂state₂_next .* ∂llh

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), [∂𝐒_1, ∂𝐒_2], ∂data_in_deviations, NoTangent(), [∂state₀_full_1, ∂state₀_full_2], NoTangent()
    end

    return llh, pruned2_missing_pullback
end


function rrule(::typeof(calculate_loglikelihood),
                ::Val{:inversion},
                ::Val{:pruned_second_order},
                observables_index::Vector{Int},
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                constants::constants,
                state::Vector{Vector{Float64}}, 
                workspaces::workspaces; 
                # timer::TimerOutput = TimerOutput(),
                on_failure_loglikelihood = -Inf,
                warmup_iterations::Int = 0,
                presample_periods::Int = 0,
                initial_covariance::Symbol = :theoretical,
                opts::CalculationOptions = merge_calculation_options(),
                filter_algorithm::Symbol = :LagrangeNewton)# where S <: Real
    T = constants.post_model_macro
    ws = workspaces.inversion

    # @timeit_debug timer "Inversion filter pruned 2nd - forward" begin
    # @timeit_debug timer "Preallocation" begin
                    
    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = observables_index

    shocks² = 0.0
    logabsdets = 0.0

    cc = ensure_conditional_forecast_constants!(constants)
    shock_idxs = cc.shock_idxs
    shock²_idxs = cc.shock²_idxs
    shockvar²_idxs = cc.shockvar²_idxs
    var_vol²_idxs = cc.var_vol²_idxs
    var²_idxs = cc.var²_idxs
    
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
            return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
        end

        # jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])
        ℒ.kron!(kron_buffer2, J, x[i])

        ℒ.mul!(jacc[i], 𝐒ⁱ²ᵉ, kron_buffer2)

        ℒ.axpby!(1, 𝐒ⁱ, 2, jacc[i])

        copy!(jacct, jacc[i]')

        if size(jacct, 1) == size(jacct, 2)
            jacc_fact = ℒ.lu(jacct, check = false)
            if !ℒ.issuccess(jacc_fact)
                if opts.verbose println("Inversion filter failed at step $i") end
                return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
            end
        else
            jacc_fact = ℒ.qr(jacct)
            R = jacc_fact.R
            if any(k -> R[k,k] == 0, axes(R, 1))
                if opts.verbose println("Inversion filter failed at step $i") end
                return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
            end
        end

        ℒ.ldiv!(λ[i], jacc_fact, x[i])

        if !all(isfinite, λ[i])
            if opts.verbose println("Inversion filter failed at step $i") end
            return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
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
            if T.nExo == length(observables_index)
                logabsdets += ℒ.logabsdet(jacc_fact)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i]))
            end
    
            shocks² += sum(abs2,x[i])
            
            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
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

    ∂𝐒 = [zero(𝐒[1]), zeros(size(𝐒[2]))]

    # Pre-allocated per-period buffers (formerly created fresh each iteration).
    ∂jacc_buf  = zero(jacc[1])
    ∂xλ_buf    = zeros(T.nExo + size(jacc[1], 1))
    S_buf      = zeros(T.nExo + size(jacc[1], 1))
    kron_xλ    = zeros(T.nExo * length(λ[1]))         # ℒ.kron(x[i], λ[i])
    kron_S1_xλ = zeros(T.nExo * length(kron_xλ))      # ℒ.kron(S[1:T.nExo], kron(x, λ))
    kron_xx_S2 = zeros(length(kronxx[1]) * size(jacc[1], 1))  # ℒ.kron(kronxx[i], S[T.nExo+1:end])

    function inversion_filter_loglikelihood_pullback(∂llh) 
        # @timeit_debug timer "Inversion filter pruned 2nd - pullback" begin
        # @timeit_debug timer "Preallocation" begin
        
        fill!(∂𝐒ⁱ, 0)
        fill!(∂𝐒ⁱ²ᵉ, 0)

        fill!(∂𝐒¹ᵉ, 0)
        fill!(∂𝐒²⁻ᵉ, 0)

        fill!(∂𝐒¹⁻ᵛ, 0)
        fill!(∂𝐒²⁻ᵛ, 0)

        fill!(∂𝐒⁻¹, 0)
        fill!(∂𝐒⁻², 0)

        fill!(∂𝐒¹⁻, 0)

        fill!(∂state¹⁻_vol, 0)
        fill!(∂x, 0)
        fill!(∂state[1], 0)
        fill!(∂state[2], 0)

        fill!(kronSλ, 0)
        fill!(kronxS, 0)
        
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

            if i < size(data_in_deviations,2)
                fill!(∂state[1], 0)
                fill!(∂state[2], 0)
            end
            
            # aug_state₁ = [state₁; 1; x]
            # ∂state[1] += ∂aug_state₁[1:length(∂state[1])]
            @views ℒ.axpy!(1, ∂aug_state₁[1:length(∂state[1])], ∂state[1])

            @views copyto!(∂x, ∂aug_state₁[T.nPast_not_future_and_mixed+2:end])

            # aug_state₂ = [state₂; 0; zero(x)]
            # ∂state[2] += ∂aug_state₂[1:length(∂state[1])]
            @views ℒ.axpy!(1, ∂aug_state₂[1:length(∂state[1])], ∂state[2])

            # shocks² += sum(abs2,x[i]) — only for i > presample_periods
            if i > presample_periods
                if i < size(data_in_deviations,2)
                    @inbounds @simd for k in eachindex(∂x)
                        ∂x[k] -= x[i][k]
                    end
                else
                    @inbounds @simd for k in eachindex(∂x)
                        ∂x[k] += x[i][k]
                    end
                end
            end

            # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1] — only for i > presample_periods
            if i > presample_periods
                if size(jacc[i], 1) == size(jacc[i], 2)
                    jacc_lu = ℒ.lu(jacc[i], check = false)
                    if !ℒ.issuccess(jacc_lu)
                        return NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end
                    copyto!(∂jacc_buf, inv(jacc_lu)')
                    ∂jacc = ∂jacc_buf
                else
                    ∂jacc = ℒ.pinv(jacc[i])'
                    if !all(isfinite, ∂jacc)
                        return NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end
                end
            else
                fill!(∂jacc_buf, 0)
                ∂jacc = ∂jacc_buf
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
            # ∂xλ = vcat(∂x, zero(λ[i]))
            copyto!(∂xλ_buf, 1, ∂x, 1, length(∂x))
            fill!(view(∂xλ_buf, length(∂x)+1:length(∂xλ_buf)), 0)

            S_solved = fXλp[i]' \ ∂xλ_buf
            copyto!(S_buf, S_solved)
            S = S_buf

            if i < size(data_in_deviations,2)
                ℒ.rmul!(S, -1)
            end

            S1 = view(S, 1:T.nExo)
            S2 = view(S, T.nExo+1:length(S))
            ∂shock_independent = S2

            # ∂𝐒ⁱ = (S[1:T.nExo] * λ[i]' - S[T.nExo+1:end] * x[i]')
            ℒ.kron!(kronSλ, S1, λ[i])
            ℒ.kron!(kronxS, x[i], S2)
            ℒ.axpy!(-1, kronxS, kronSλ)
            copyto!(∂𝐒ⁱ, kronSλ)
            # ∂𝐒ⁱ -= ∂jacc / 2
            ℒ.axpy!(-1/2, ∂jacc, ∂𝐒ⁱ)

            # ∂𝐒ⁱ²ᵉ += reshape(2 * ℒ.kron(S[1:T.nExo], ℒ.kron(x[i], λ[i])) - ℒ.kron(kronxx[i], S[T.nExo+1:end]), size(∂𝐒ⁱ²ᵉ))
            ℒ.kron!(kron_xλ,    x[i], λ[i])
            ℒ.kron!(kron_S1_xλ, S1,   kron_xλ)
            ℒ.kron!(kron_xx_S2, kronxx[i], S2)
            ℒ.axpby!(-1, kron_xx_S2, 2, kron_S1_xλ)
            ∂𝐒ⁱ²ᵉ .+= reshape(kron_S1_xλ, size(∂𝐒ⁱ²ᵉ))

            # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
            fill!(∂state¹⁻_vol, 0)
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
            @inbounds for k in eachindex(∂shock_independent)
                ∂data_in_deviations[k, i] = ∂shock_independent[k]
            end

            # ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
            # ∂𝐒¹⁻ᵛ -= ∂shock_independent * state¹⁻_vol'
            ℒ.mul!(∂𝐒¹⁻ᵛ, ∂shock_independent, state¹⁻_vol', -1, 1)

            # ∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent
            ℒ.mul!(∂state¹⁻_vol, 𝐒¹⁻ᵛ', ∂shock_independent, -1, 1)

            # ℒ.mul!(shock_independent, 𝐒¹⁻, state²⁻, -1, 1)
            # ∂𝐒¹⁻ -= ∂shock_independent * aug_state₂[i][1:T.nPast_not_future_and_mixed]'
            @views ℒ.mul!(∂𝐒¹⁻, ∂shock_independent, aug_state₂[i][1:T.nPast_not_future_and_mixed]', -1, 1)

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
            @views ℒ.axpy!(1, ∂state¹⁻_vol[1:end-1], ∂state[1])
        end

        # end # timeit_debug
        # @timeit_debug timer "Post allocation" begin

        fill!(∂𝐒[1], 0)
        fill!(∂𝐒[2], 0)

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

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂𝐒, ∂data_in_deviations, NoTangent(), ∂state, NoTangent()
    end

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    llh = -(logabsdets + shocks² + (length(observables_index) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2

    return llh, inversion_filter_loglikelihood_pullback
end

function rrule(::typeof(calculate_loglikelihood_with_missing), ::Val{:inversion}, ::Val{:second_order},
                                       observables_index::Vector{Int},
                                              𝐒::Vector{AbstractMatrix{Float64}},
                                              data_in_deviations::Matrix{Float64},
                                              constants::constants,
                                              state::Vector{Float64},
                                              workspaces::workspaces,
                                              obs_idx_per_t::Vector{Vector{Int}};
                                              warmup_iterations::Int = 0,
                                              on_failure_loglikelihood = -Inf,
                                              presample_periods::Int = 0,
                                              initial_covariance::Symbol = :theoretical,
                                              opts::CalculationOptions = merge_calculation_options(),
                                              filter_algorithm::Symbol = :LagrangeNewton)
    Tcc = constants.post_model_macro
    n_exo  = Tcc.nExo
    n_past = Tcc.nPast_not_future_and_mixed
    cond_var_idx = observables_index
    n_cond = length(cond_var_idx)
    Tt = size(data_in_deviations, 2)

    eff_presample = presample_periods + warmup_iterations

    ws = workspaces.inversion
    ensure_inversion_buffers!(ws, n_exo, n_past)
    ensure_inversion_estimation_buffers!(ws, n_exo, n_cond)
    ensure_inversion_rrule_buffers!(ws, n_exo, n_past, n_cond, Tt; order = :second_order)

    cc = ensure_conditional_forecast_constants!(constants)
    shock_idxs     = cc.shock_idxs
    shock²_idxs    = cc.shock²_idxs
    shockvar²_idxs = cc.shockvar²_idxs
    var_vol²_idxs  = cc.var_vol²_idxs
    var²_idxs      = cc.var²_idxs

    𝐒⁻¹  = 𝐒[1][Tcc.past_not_future_and_mixed_idx, :]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:n_past+1]
    𝐒¹ᵉ  = 𝐒[1][cond_var_idx, end-n_exo+1:end]
    𝐒²⁻ᵛ = collect(𝐒[2][cond_var_idx, var_vol²_idxs])
    𝐒²⁻ᵉ = collect(𝐒[2][cond_var_idx, shockvar²_idxs])
    𝐒²ᵉ  = collect(𝐒[2][cond_var_idx, shock²_idxs])
    𝐒⁻²  = collect(𝐒[2][Tcc.past_not_future_and_mixed_idx, :])

    𝐒ⁱ²ᵉ = 𝐒²ᵉ ./ 2
    J = ℒ.I(n_exo)

    st = copy(state[Tcc.past_not_future_and_mixed_idx])

    # Per-period storage (cached in workspace; see ensure_inversion_rrule_buffers!)
    st_seq          = ws.state_seq_rrule
    st_seq[1]      .= st
    state¹⁻_vol_seq = ws.state¹⁻_vol_seq_rrule
    aug_state_seq   = ws.aug_state_seq_rrule
    x_seq           = ws.x_seq_rrule
    𝐒ⁱ_full_seq    = ws.𝐒ⁱ_full_seq_rrule

    shocks² = 0.0
    logabsdets = 0.0
    n_obs_total = 0

    state¹⁻_vol      = ws.state_vol
    shock_independent = ws.shock_independent
    kronstate¹⁻_vol  = ws.kronstate_vol
    kron_buffer3     = ws.kron_buffer_state
    𝐒ⁱ_full         = ws.Si_buffer
    kronaug_state    = ws.kronaug_state
    init_guess       = ws.init_guess
    kron_buffer      = ws.kron_buffer
    kron_buffer2     = ws.kron_buffer2

    for t in 1:Tt
        idx = obs_idx_per_t[t]
        m = length(idx)

        copyto!(state¹⁻_vol, 1, st, 1)
        state¹⁻_vol[end] = 1.0

        copyto!(shock_independent, view(data_in_deviations, :, t))
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        ℒ.kron!(kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol, -1/2, 1)

        ℒ.kron!(kron_buffer3, J, state¹⁻_vol)
        ℒ.mul!(𝐒ⁱ_full, 𝐒²⁻ᵉ, kron_buffer3)
        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ_full)

        copyto!(state¹⁻_vol_seq[t], state¹⁻_vol)
        𝐒ⁱ_full_seq[t] .= 𝐒ⁱ_full

        if m == 0
            fill!(init_guess, 0.0)
            x = init_guess
        else
            if m > n_exo
                if opts.verbose println("Inversion filter rrule (2nd, missing) failed at step $t: m=$m > n_exo=$n_exo") end
                return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end
            𝐒ⁱ_v   = 𝐒ⁱ_full[idx, :]
            𝐒ⁱ²ᵉ_v = 𝐒ⁱ²ᵉ[idx, :]
            si_v   = shock_independent[idx]
            fill!(init_guess, 0.0)
            x, matched = find_shocks(Val(filter_algorithm),
                                     init_guess, kron_buffer, kron_buffer2, J,
                                     𝐒ⁱ_v, 𝐒ⁱ²ᵉ_v, si_v)
            if !matched
                if opts.verbose println("Inversion filter rrule (2nd, missing) failed at step $t") end
                return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end
            if t > eff_presample
                jac_v = similar(𝐒ⁱ_v)
                ℒ.kron!(kron_buffer2, J, x)
                ℒ.mul!(jac_v, 𝐒ⁱ²ᵉ_v, kron_buffer2)
                ℒ.axpby!(1, 𝐒ⁱ_v, 2, jac_v)
                logabsdets += m == n_exo ? ℒ.logabsdet(jac_v)[1] : ℒ.logabsdet(jac_v * jac_v')[1] / 2
                shocks² += sum(abs2, x)
                n_obs_total += m
                if !isfinite(logabsdets) || !isfinite(shocks²)
                    return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
                end
            end
        end
        x_seq[t] .= x

        # aug_state = vcat(st, 1, x)
        copyto!(aug_state_seq[t], 1, st, 1)
        aug_state_seq[t][n_past + 1] = 1.0
        copyto!(aug_state_seq[t], n_past + 2, x, 1)

        # state ← 𝐒⁻¹ aug + 0.5 𝐒⁻² kron(aug, aug)
        ℒ.mul!(st, 𝐒⁻¹, aug_state_seq[t])
        ℒ.kron!(kronaug_state, aug_state_seq[t], aug_state_seq[t])
        ℒ.mul!(st, 𝐒⁻², kronaug_state, 1/2, 1)
        copyto!(st_seq[t+1], st)
    end

    llh = -(logabsdets + shocks² + n_obs_total * log(2 * 3.141592653589793)) / 2

    if !isfinite(llh) || llh < -1e12
        return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # Pre-allocate adjoint buffers outside the pullback closure.
    ∂𝐒_1 = zeros(size(𝐒[1]))
    ∂𝐒_2 = zeros(size(𝐒[2]))
    ∂𝐒⁻¹  = zero(𝐒⁻¹)
    ∂𝐒⁻²  = zero(𝐒⁻²)
    ∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)
    ∂𝐒¹ᵉ  = zero(𝐒¹ᵉ)
    ∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)
    ∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)
    ∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)
    ∂data_in_deviations = zeros(size(data_in_deviations))
    ∂st_next = zeros(n_past)
    kronaug_buf = zeros((n_past + 1 + n_exo)^2)
    ∂kronaug    = zeros((n_past + 1 + n_exo)^2)
    ∂aug_state  = zeros(n_past + 1 + n_exo)
    ∂kronstate  = zeros((n_past + 1)^2)
    ∂state¹⁻_vol = zeros(n_past + 1)
    ∂𝐒ⁱ_full_buf      = zeros(length(cond_var_idx), n_exo)
    ∂shock_independent = zeros(length(cond_var_idx))
    ∂jac_v_buf         = zeros(length(cond_var_idx), n_exo)
    kron_Isv_buf       = zeros(n_exo * (n_past + 1), n_exo)
    ∂kronIstate_local  = zeros(n_exo * (n_past + 1), n_exo)

    function second_missing_pullback(∂llh)
        fill!(∂𝐒_1, 0); fill!(∂𝐒_2, 0)
        fill!(∂𝐒⁻¹, 0); fill!(∂𝐒⁻², 0)
        fill!(∂𝐒¹⁻ᵛ, 0); fill!(∂𝐒¹ᵉ, 0)
        fill!(∂𝐒²⁻ᵛ, 0); fill!(∂𝐒²⁻ᵉ, 0); fill!(∂𝐒ⁱ²ᵉ, 0)
        fill!(∂data_in_deviations, 0)
        fill!(∂st_next, 0)
        fill!(kronaug_buf, 0); fill!(∂kronaug, 0)
        fill!(∂aug_state, 0); fill!(∂kronstate, 0); fill!(∂state¹⁻_vol, 0)
        fill!(∂𝐒ⁱ_full_buf, 0); fill!(∂shock_independent, 0)
        fill!(kron_Isv_buf, 0); fill!(∂kronIstate_local, 0)
        fill!(∂jac_v_buf, 0)

        for t in Tt:-1:1
            aug_state = aug_state_seq[t]
            state¹⁻_vol = state¹⁻_vol_seq[t]
            x = x_seq[t]
            idx = obs_idx_per_t[t]
            m = length(idx)

            # st_next = 𝐒⁻¹ * aug + 0.5 * 𝐒⁻² * kron(aug, aug)
            ℒ.mul!(∂𝐒⁻¹, ∂st_next, aug_state', 1, 1)
            ℒ.mul!(∂aug_state, 𝐒⁻¹', ∂st_next)
            ℒ.kron!(kronaug_buf, aug_state, aug_state)
            ℒ.mul!(∂𝐒⁻², ∂st_next, kronaug_buf', 1/2, 1)
            ℒ.mul!(∂kronaug, 𝐒⁻²', ∂st_next)
            ℒ.rdiv!(∂kronaug, 2)
            fill_kron_adjoint!(∂aug_state, ∂aug_state, ∂kronaug, aug_state, aug_state)

            fill!(∂st_next, 0)

            # split aug_state contributions: aug_state = vcat(st, 1, x)
            ∂st_now = ∂aug_state[1:n_past]
            ∂x = ∂aug_state[n_past+2:end]

            # accumulate ∂st_now into ∂st_next (which becomes the prior period's downstream)
            @inbounds for j in 1:n_past
                ∂st_next[j] += ∂st_now[j]
            end

            # shocks² and logabsdet contributions (only if t > presample and m > 0)
            ∂jac_v = view(∂jac_v_buf, 1:m, :); fill!(∂jac_v, 0)
            jac_v_local = zeros(m, n_exo)
            𝐒ⁱ²ᵉ_v_local = zeros(m, n_exo^2)
            if m > 0
                𝐒ⁱ_v_local = 𝐒ⁱ_full_seq[t][idx, :]
                𝐒ⁱ²ᵉ_v_local = 𝐒ⁱ²ᵉ[idx, :]
                jac_v_local = 𝐒ⁱ_v_local + 2 * 𝐒ⁱ²ᵉ_v_local * ℒ.kron(J, x)
            end
            if m > 0 && t > eff_presample
                @inbounds for k in 1:n_exo
                    ∂x[k] += -x[k]
                end
                if m == n_exo
                    invjac_v = inv(jac_v_local)
                    ∂jac_v .+= (-0.5) .* invjac_v'
                else
                    G = inv(jac_v_local * jac_v_local')
                    ∂jac_v .+= (-0.5) .* (G * jac_v_local)
                end
                # Indirect channel: ∂jac_v → x via the (I⊗x) term in jac_v.
                # d jac_v[i,r] / d x_l = 2 𝐒ⁱ²ᵉ_v[i, (r-1)n_exo + l]
                @inbounds for l in 1:n_exo
                    s = 0.0
                    for r in 1:n_exo
                        col = (r-1) * n_exo + l
                        for i_local in 1:m
                            s += ∂jac_v[i_local, r] * 𝐒ⁱ²ᵉ_v_local[i_local, col]
                        end
                    end
                    ∂x[l] += 2 * s
                end
            end

            fill!(∂shock_independent, 0)

            if m > 0
                𝐒ⁱ_v   = 𝐒ⁱ_full_seq[t][idx, :]
                𝐒ⁱ²ᵉ_v = 𝐒ⁱ²ᵉ_v_local
                jac_v  = jac_v_local

                local λ
                if m == n_exo
                    λ = 2 * (jac_v' \ x)
                else
                    Gloc = inv(jac_v * jac_v')
                    λ = 2 * (Gloc * (jac_v * x))
                end

                M = reshape(𝐒ⁱ²ᵉ_v' * λ, n_exo, n_exo)
                topL = 2 * ℒ.I(n_exo) - 2 * M
                fXλp = [topL          -jac_v'
                        jac_v          zeros(m, m)]

                rhs = vcat(∂x, zeros(m))
                S = fXλp' \ rhs
                Sx = S[1:n_exo]
                Sλ = S[n_exo+1:end]

                ∂v_v = Sλ
                ∂𝐒ⁱ_v = λ * Sx' - Sλ * x'

                xSx = x * Sx'
                xx_outer = x * x'
                ∂𝐒ⁱ²ᵉ_v_top = 2 * λ * vec(xSx)'
                ∂𝐒ⁱ²ᵉ_v_F   = -Sλ * vec(xx_outer)'
                ∂𝐒ⁱ²ᵉ_v_kkt = ∂𝐒ⁱ²ᵉ_v_top + ∂𝐒ⁱ²ᵉ_v_F

                if t > eff_presample
                    ∂𝐒ⁱ_v_total = ∂𝐒ⁱ_v + ∂jac_v
                    ∂𝐒ⁱ²ᵉ_v_total = ∂𝐒ⁱ²ᵉ_v_kkt + 2 * ∂jac_v * ℒ.kron(J, x)'
                else
                    ∂𝐒ⁱ_v_total = ∂𝐒ⁱ_v
                    ∂𝐒ⁱ²ᵉ_v_total = ∂𝐒ⁱ²ᵉ_v_kkt
                end

                ∂𝐒ⁱ_full = ∂𝐒ⁱ_full_buf
                fill!(∂𝐒ⁱ_full, 0)
                @inbounds for j in 1:n_exo
                    for i_local in 1:m
                        ∂𝐒ⁱ_full[idx[i_local], j] = ∂𝐒ⁱ_v_total[i_local, j]
                    end
                end
                @inbounds for j in 1:n_exo^2
                    for i_local in 1:m
                        ∂𝐒ⁱ²ᵉ[idx[i_local], j] += ∂𝐒ⁱ²ᵉ_v_total[i_local, j]
                    end
                end

                @inbounds for i_local in 1:m
                    ∂shock_independent[idx[i_local]] += ∂v_v[i_local]
                end

                ∂𝐒¹ᵉ .+= ∂𝐒ⁱ_full
                ℒ.kron!(kron_Isv_buf, J, state¹⁻_vol)
                ℒ.mul!(∂𝐒²⁻ᵉ, ∂𝐒ⁱ_full, kron_Isv_buf', 1, 1)
                ℒ.mul!(∂kronIstate_local, 𝐒²⁻ᵉ', ∂𝐒ⁱ_full)
                @inbounds for p in 1:(n_past + 1)
                    s = 0.0
                    for j in 1:n_exo
                        s += ∂kronIstate_local[(j-1)*(n_past+1) + p, j]
                    end
                    ∂state¹⁻_vol[p] = s
                end
            else
                fill!(∂state¹⁻_vol, 0)
            end

            # shock_independent = data[:,t] - 𝐒¹⁻ᵛ * state¹⁻_vol - 0.5 * 𝐒²⁻ᵛ * (state¹⁻_vol ⊗ state¹⁻_vol)
            @inbounds for i in 1:length(cond_var_idx)
                ∂data_in_deviations[i, t] += ∂shock_independent[i]
            end
            ℒ.mul!(∂𝐒¹⁻ᵛ, ∂shock_independent, state¹⁻_vol', -1, 1)
            ℒ.mul!(∂state¹⁻_vol, 𝐒¹⁻ᵛ', ∂shock_independent, -1, 1)
            kron_sv = ℒ.kron(state¹⁻_vol, state¹⁻_vol)
            ℒ.mul!(∂𝐒²⁻ᵛ, ∂shock_independent, kron_sv', -1/2, 1)
            ∂kron_sv = -(𝐒²⁻ᵛ' * ∂shock_independent) ./ 2
            fill!(∂kronstate, 0)
            ∂kronstate .+= ∂kron_sv
            fill_kron_adjoint!(∂state¹⁻_vol, ∂state¹⁻_vol, ∂kronstate, state¹⁻_vol, state¹⁻_vol)

            # state¹⁻_vol = vcat(st, 1) → ∂st += ∂state¹⁻_vol[1:n_past]
            @inbounds for j in 1:n_past
                ∂st_next[j] += ∂state¹⁻_vol[j]
            end
        end

        ∂𝐒_1[Tcc.past_not_future_and_mixed_idx, :]                 .+= ∂𝐒⁻¹
        ∂𝐒_2[Tcc.past_not_future_and_mixed_idx, :]                 .+= ∂𝐒⁻²
        ∂𝐒_1[cond_var_idx, 1:n_past+1]                              .+= ∂𝐒¹⁻ᵛ
        ∂𝐒_1[cond_var_idx, end-n_exo+1:end]                         .+= ∂𝐒¹ᵉ
        ∂𝐒_2[cond_var_idx, var_vol²_idxs]                           .+= ∂𝐒²⁻ᵛ
        ∂𝐒_2[cond_var_idx, shockvar²_idxs]                          .+= ∂𝐒²⁻ᵉ
        ∂𝐒_2[cond_var_idx, shock²_idxs]                             .+= ∂𝐒ⁱ²ᵉ ./ 2

        ℒ.rmul!(∂𝐒_1, ∂llh)
        ℒ.rmul!(∂𝐒_2, ∂llh)
        ℒ.rmul!(∂data_in_deviations, ∂llh)

        ∂state₀_full = zeros(size(state))
        ∂state₀_full[Tcc.past_not_future_and_mixed_idx] .= ∂st_next .* ∂llh

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), [∂𝐒_1, ∂𝐒_2], ∂data_in_deviations, NoTangent(), ∂state₀_full, NoTangent()
    end

    return llh, second_missing_pullback
end


function rrule(::typeof(calculate_loglikelihood),
                ::Val{:inversion},
                ::Val{:second_order},
                observables_index::Vector{Int},
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                constants::constants,
                state::Vector{Float64}, 
                workspaces::workspaces; 
                # timer::TimerOutput = TimerOutput(),
                on_failure_loglikelihood = -Inf,
                warmup_iterations::Int = 0,
                presample_periods::Int = 0,
                initial_covariance::Symbol = :theoretical,
                opts::CalculationOptions = merge_calculation_options(),
                filter_algorithm::Symbol = :LagrangeNewton)# where S <: Real
    T = constants.post_model_macro
    ws = workspaces.inversion

    # @timeit_debug timer "Inversion filter 2nd - forward" begin
        
    # @timeit_debug timer "Preallocation" begin

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = observables_index

    shocks² = 0.0
    logabsdets = 0.0

    cc = ensure_conditional_forecast_constants!(constants)
    shock_idxs = cc.shock_idxs
    shock²_idxs = cc.shock²_idxs
    shockvar²_idxs = cc.shockvar²_idxs
    var_vol²_idxs = cc.var_vol²_idxs
    var²_idxs = cc.var²_idxs
    
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
            return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
        end
        
        ℒ.kron!(kron_buffer2, J, x[i])

        ℒ.mul!(jacc[i], 𝐒ⁱ²ᵉ, kron_buffer2)

        ℒ.axpby!(1, 𝐒ⁱ, 2, jacc[i])
        # jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

        copy!(jacct, jacc[i]')

        if size(jacct, 1) == size(jacct, 2)
            jacc_fact = ℒ.lu(jacct, check = false)
            if !ℒ.issuccess(jacc_fact)
                if opts.verbose println("Inversion filter failed at step $i") end
                return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
            end
        else
            jacc_fact = ℒ.qr(jacct)
            R = jacc_fact.R
            if any(k -> R[k,k] == 0, axes(R, 1))
                if opts.verbose println("Inversion filter failed at step $i") end
                return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
            end
        end

        ℒ.ldiv!(λ[i], jacc_fact, x[i])

        if !all(isfinite, λ[i])
            if opts.verbose println("Inversion filter failed at step $i") end
            return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
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
            if T.nExo == length(observables_index)
                logabsdets += ℒ.logabsdet(jacc_fact)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i]))
            end
    
            shocks² += sum(abs2, x[i])
            
            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
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


    ∂𝐒 = [zero(𝐒[1]), zero(𝐒[2])]

    ∂data_in_deviations = similar(data_in_deviations)

    ∂kronIx = zero(ℒ.kron(ℒ.I(length(x[1])), x[1]))

    ∂𝐒ⁱ = zero(𝐒ⁱ)

    ∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)

    ∂𝐒¹ᵉ = zero(𝐒¹ᵉ)

    ∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)

    ∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)

    ∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)

    ∂𝐒⁻¹ = zero(𝐒⁻¹)

    ∂𝐒⁻² = zero(𝐒⁻²)

    ∂state¹⁻_vol = zero(state¹⁻_vol)

    ∂state = zeros(T.nPast_not_future_and_mixed)

    # Pre-allocated per-period buffers (formerly created fresh each iteration).
    ∂x         = zero(x[1])
    ∂jacc_buf  = zero(jacc[1])
    ∂xλ_buf    = zeros(T.nExo + size(jacc[1], 1))
    S_buf      = zeros(T.nExo + size(jacc[1], 1))
    kron_S1_kxλ = zeros(T.nExo * length(kronxλ[1]))
    kron_xx_S2  = zeros(length(kronxx[1]) * size(jacc[1], 1))

    function inversion_filter_loglikelihood_pullback(∂llh)
        # @timeit_debug timer "Inversion filter 2nd - pullback" begin

        # @timeit_debug timer "Preallocation" begin

        fill!(∂𝐒ⁱ, 0)
        fill!(∂𝐒ⁱ²ᵉ, 0)
        
        # Allocate or reuse workspaces for pullback temps
        if size(ws.∂𝐒ⁱ²ᵉtmp) != (T.nExo, T.nExo * length(λ[1]))
            ws.∂𝐒ⁱ²ᵉtmp = zeros(T.nExo, T.nExo * length(λ[1]))
        else
            fill!(ws.∂𝐒ⁱ²ᵉtmp, zero(eltype(ws.∂𝐒ⁱ²ᵉtmp)))
        end
        ∂𝐒ⁱ²ᵉtmp = ws.∂𝐒ⁱ²ᵉtmp
        
        if size(ws.∂𝐒ⁱ²ᵉtmp2) != (length(λ[1]), T.nExo * T.nExo)
            ws.∂𝐒ⁱ²ᵉtmp2 = zeros(length(λ[1]), T.nExo * T.nExo)
        else
            fill!(ws.∂𝐒ⁱ²ᵉtmp2, zero(eltype(ws.∂𝐒ⁱ²ᵉtmp2)))
        end
        ∂𝐒ⁱ²ᵉtmp2 = ws.∂𝐒ⁱ²ᵉtmp2

        fill!(∂𝐒¹ᵉ, 0)
        fill!(∂𝐒²⁻ᵉ, 0)

        fill!(∂𝐒¹⁻ᵛ, 0)
        fill!(∂𝐒²⁻ᵛ, 0)

        fill!(∂𝐒⁻¹, 0)
        fill!(∂𝐒⁻², 0)

        fill!(∂state¹⁻_vol, 0)
        # ∂x = zero(x[1])
        fill!(∂state, 0)

        ∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

        # Allocate or reuse workspaces for kron products
        if length(ws.kronSλ) != length(cond_var_idx) * T.nExo
            ws.kronSλ = zeros(length(cond_var_idx) * T.nExo)
        else
            fill!(ws.kronSλ, zero(eltype(ws.kronSλ)))
        end
        kronSλ = ws.kronSλ
        
        if length(ws.kronxS) != T.nExo * length(cond_var_idx)
            ws.kronxS = zeros(T.nExo * length(cond_var_idx))
        else
            fill!(ws.kronxS, zero(eltype(ws.kronxS)))
        end
        kronxS = ws.kronxS
        
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

            if i < size(data_in_deviations,2)
                fill!(∂state, 0)
            end

            # aug_state[i] = [stt; 1; x[i]]
            @views ℒ.axpy!(1, ∂aug_state[1:length(∂state)], ∂state)

            # aug_state[i] = [stt; 1; x[i]]
            @views copyto!(∂x, ∂aug_state[T.nPast_not_future_and_mixed+2:end])

            # shocks² += sum(abs2,x[i]) — only contributes for i > presample_periods
            if i > presample_periods
                if i < size(data_in_deviations,2)
                    @inbounds @simd for k in eachindex(∂x)
                        ∂x[k] -= x[i][k]
                    end
                else
                    @inbounds @simd for k in eachindex(∂x)
                        ∂x[k] += x[i][k]
                    end
                end
            end

            # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1] — only for i > presample_periods
            if i > presample_periods
                if size(jacc[i], 1) == size(jacc[i], 2)
                    jacc_lu = ℒ.lu(jacc[i], check = false)
                    if !ℒ.issuccess(jacc_lu)
                        return NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end
                    copyto!(∂jacc_buf, inv(jacc_lu)')
                    ∂jacc = ∂jacc_buf
                else
                    ∂jacc = ℒ.pinv(jacc[i])'
                    if !all(isfinite, ∂jacc)
                        return NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end
                end
            else
                fill!(∂jacc_buf, 0)
                ∂jacc = ∂jacc_buf
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
            # ∂xλ = vcat(∂x, zero(λ[i]))
            copyto!(∂xλ_buf, 1, ∂x, 1, length(∂x))
            fill!(view(∂xλ_buf, length(∂x)+1:length(∂xλ_buf)), 0)

            S_solved = fXλp[i]' \ ∂xλ_buf
            copyto!(S_buf, S_solved)
            S = S_buf

            if i < size(data_in_deviations,2)
                ℒ.rmul!(S, -1)
            end

            S1 = view(S, 1:T.nExo)
            S2 = view(S, T.nExo+1:length(S))
            ∂shock_independent = S2

            # ∂𝐒ⁱ = ℒ.kron(S[1:T.nExo], λ[i]) - ℒ.kron(x[i], S[T.nExo+1:end])
            ℒ.kron!(kronSλ, S1, λ[i])
            ℒ.kron!(kronxS, x[i], S2)
            ℒ.axpy!(-1, kronxS, kronSλ)
            copyto!(∂𝐒ⁱ, kronSλ)

            ℒ.axpy!(-1/2, ∂jacc, ∂𝐒ⁱ)

            # ∂𝐒ⁱ²ᵉ += reshape(2 * ℒ.kron(S[1:T.nExo], kronxλ[i]) - ℒ.kron(kronxx[i], S[T.nExo+1:end]), size(∂𝐒ⁱ²ᵉ))
            ℒ.kron!(kron_S1_kxλ, S1, kronxλ[i])
            ℒ.kron!(kron_xx_S2, kronxx[i], S2)
            ℒ.axpby!(-1, kron_xx_S2, 2, kron_S1_kxλ)
            ∂𝐒ⁱ²ᵉ .+= reshape(kron_S1_kxλ, size(∂𝐒ⁱ²ᵉ))

            # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
            fill!(∂state¹⁻_vol, 0)
            
            ℒ.mul!(∂kronIstate¹⁻_vol, 𝐒²⁻ᵉ', ∂𝐒ⁱ)

            fill_kron_adjoint_∂A!(∂kronIstate¹⁻_vol, ∂state¹⁻_vol, J)

            state¹⁻_vol = aug_state[i][1:T.nPast_not_future_and_mixed + 1]

            ℒ.axpy!(1, ∂𝐒ⁱ, ∂𝐒¹ᵉ)
            # ∂𝐒¹ᵉ += ∂𝐒ⁱ

            ℒ.kron!(kron_buffer3, J, state¹⁻_vol)

            ℒ.mul!(∂𝐒²⁻ᵉ, ∂𝐒ⁱ, kron_buffer3', 1, 1)
            # ∂𝐒²⁻ᵉ += ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)'

            # shock_independent = copy(data_in_deviations[:,i])
            @inbounds @simd for k in eachindex(∂shock_independent)
                ∂data_in_deviations[k, i] = ∂shock_independent[k]
            end

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

        fill!(∂𝐒[1], 0)
        fill!(∂𝐒[2], 0)

        ∂𝐒[1][cond_var_idx,end-T.nExo+1:end] += ∂𝐒¹ᵉ
        ∂𝐒[2][cond_var_idx,shockvar²_idxs] += ∂𝐒²⁻ᵉ
        ∂𝐒[2][cond_var_idx,shock²_idxs] += ∂𝐒ⁱ²ᵉ / 2
        ∂𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1] += ∂𝐒¹⁻ᵛ
        ∂𝐒[2][cond_var_idx,var_vol²_idxs] += ∂𝐒²⁻ᵛ

        ∂𝐒[1][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻¹
        ∂𝐒[2][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻²

        ∂𝐒[1] *= ∂llh
        ∂𝐒[2] *= ∂llh

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂𝐒, ∂data_in_deviations * ∂llh, NoTangent(), ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state * ∂llh, NoTangent()
    end

    # end # timeit_debug
    # end # timeit_debug

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    llh = -(logabsdets + shocks² + (length(observables_index) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2

    return llh, inversion_filter_loglikelihood_pullback
end

function rrule(::typeof(calculate_loglikelihood_with_missing), ::Val{:inversion}, ::Val{:pruned_third_order},
                                       observables_index::Vector{Int},
                                                    𝐒::Vector{AbstractMatrix{Float64}},
                                                    data_in_deviations::Matrix{Float64},
                                                    constants::constants,
                                                    state::Vector{Vector{Float64}},
                                                    workspaces::workspaces,
                                                    obs_idx_per_t::Vector{Vector{Int}};
                                                    warmup_iterations::Int = 0,
                                                    on_failure_loglikelihood = -Inf,
                                                    presample_periods::Int = 0,
                                                    initial_covariance::Symbol = :theoretical,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)
    Tcc = constants.post_model_macro
    n_exo  = Tcc.nExo
    n_past = Tcc.nPast_not_future_and_mixed
    cond_var_idx = observables_index
    n_cond = length(cond_var_idx)
    Tt = size(data_in_deviations, 2)

    eff_presample = presample_periods + warmup_iterations

    ws = workspaces.inversion
    ensure_inversion_buffers!(ws, n_exo, n_past; third_order = true)
    ensure_inversion_estimation_buffers!(ws, n_exo, n_cond; third_order = true)
    ensure_inversion_rrule_buffers!(ws, n_exo, n_past, n_cond, Tt; order = :pruned_third_order)

    cc = ensure_conditional_forecast_constants!(constants; third_order = true)
    tc = constants.third_order
    shockvar_idxs   = cc.shockvar_idxs
    shock_idxs      = cc.shock_idxs
    shock²_idxs     = cc.shock²_idxs
    shockvar²_idxs  = cc.shockvar²_idxs
    var_vol²_idxs   = cc.var_vol²_idxs
    var²_idxs       = cc.var²_idxs
    var_vol³_idxs   = tc.var_vol³_idxs
    shock³_idxs     = tc.shock³_idxs
    shockvar³2_idxs = tc.shockvar³2_idxs
    shockvar³_idxs  = tc.shockvar³_idxs

    𝐒⁻¹   = 𝐒[1][Tcc.past_not_future_and_mixed_idx, :]
    𝐒¹⁻   = 𝐒[1][cond_var_idx, 1:n_past]
    𝐒¹⁻ᵛ  = 𝐒[1][cond_var_idx, 1:n_past+1]
    𝐒¹ᵉ   = 𝐒[1][cond_var_idx, end-n_exo+1:end]
    𝐒²⁻ᵛ  = collect(𝐒[2][cond_var_idx, var_vol²_idxs])
    𝐒²⁻   = collect(𝐒[2][cond_var_idx, var²_idxs])
    𝐒²⁻ᵉ  = collect(𝐒[2][cond_var_idx, shockvar²_idxs])
    𝐒²⁻ᵛᵉ = collect(𝐒[2][cond_var_idx, shockvar_idxs])
    𝐒²ᵉ   = collect(𝐒[2][cond_var_idx, shock²_idxs])
    𝐒⁻²   = collect(𝐒[2][Tcc.past_not_future_and_mixed_idx, :])
    𝐒³⁻ᵛ  = collect(𝐒[3][cond_var_idx, var_vol³_idxs])
    𝐒³⁻ᵉ² = collect(𝐒[3][cond_var_idx, shockvar³2_idxs])
    𝐒³⁻ᵉ  = collect(𝐒[3][cond_var_idx, shockvar³_idxs])
    𝐒³ᵉ   = collect(𝐒[3][cond_var_idx, shock³_idxs])
    𝐒⁻³   = collect(𝐒[3][Tcc.past_not_future_and_mixed_idx, :])

    𝐒ⁱ³ᵉ = 𝐒³ᵉ ./ 6
    J  = ℒ.I(n_exo)
    II = sparse(ℒ.I(n_exo^2))

    state₁ = copy(state[1][Tcc.past_not_future_and_mixed_idx])
    state₂ = copy(state[2][Tcc.past_not_future_and_mixed_idx])
    state₃ = copy(state[3][Tcc.past_not_future_and_mixed_idx])

    # Per-period storage (cached in workspace; see ensure_inversion_rrule_buffers!)
    state₁_seq      = ws.state_seq_rrule
    state₂_seq      = ws.state₂_seq_rrule
    state₃_seq      = ws.state₃_seq_rrule
    state₁_seq[1] .= state₁
    state₂_seq[1] .= state₂
    state₃_seq[1] .= state₃
    state¹⁻_vol_seq = ws.state¹⁻_vol_seq_rrule
    aug_state₁_seq  = ws.aug_state₁_seq_rrule
    aug_state₁̂_seq = ws.aug_state₁̂_seq_rrule
    aug_state₂_seq  = ws.aug_state₂_seq_rrule
    aug_state₃_seq  = ws.aug_state₃_seq_rrule
    x_seq           = ws.x_seq_rrule
    𝐒ⁱ_full_seq    = ws.𝐒ⁱ_full_seq_rrule
    𝐒ⁱ²ᵉ_full_seq  = ws.𝐒ⁱ²ᵉ_full_seq_rrule

    shocks² = 0.0
    logabsdets = 0.0
    n_obs_total = 0

    state¹⁻_vol       = ws.state_vol
    state²⁻_vol       = ws.state²⁻_vol
    shock_independent = ws.shock_independent
    kronstate¹⁻_vol   = ws.kronstate_vol
    kron_kron_state¹⁻_vol = ws.kronstate_vol³
    𝐒ⁱ_full          = ws.Si_buffer
    𝐒ⁱ²ᵉ_full        = ws.Si2e_buffer
    kron_buffer3sv = zeros(n_exo * (n_past + 1)^2, n_exo)
    kron_buffer4sv = zeros(n_exo^2 * (n_past + 1), n_exo^2)
    kron_aug_state₁      = ws.kronaug_state
    kron_kron_aug_state₁ = ws.kron_kron_aug_state
    init_guess = ws.init_guess
    kb1 = ws.kron_buffer
    kb2 = ws.kron_buffer²
    kb3 = ws.kron_buffer2
    kb4 = ws.kron_buffer3
    kb5 = ws.kron_buffer4

    for t in 1:Tt
        idx = obs_idx_per_t[t]
        m = length(idx)

        copyto!(state¹⁻_vol, 1, state₁, 1)
        state¹⁻_vol[end] = 1.0

        copyto!(state²⁻_vol, 1, state₂, 1)
        state²⁻_vol[end] = 0.0

        # shock_independent = data - 𝐒¹⁻ᵛ s¹v - 𝐒¹⁻ s2 - 𝐒¹⁻ s3 - 0.5 𝐒²⁻ᵛ k(s¹v,s¹v) - 𝐒²⁻ k(s1,s2) - (1/6) 𝐒³⁻ᵛ k(s¹v,k(s¹v,s¹v))
        copyto!(shock_independent, view(data_in_deviations, :, t))
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)
        ℒ.mul!(shock_independent, 𝐒¹⁻, state₃, -1, 1)
        ℒ.kron!(kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol, -1/2, 1)
        kron_state₁_state₂ = ℒ.kron(state₁, state₂)
        ℒ.mul!(shock_independent, 𝐒²⁻, kron_state₁_state₂, -1, 1)
        ℒ.kron!(kron_kron_state¹⁻_vol, kronstate¹⁻_vol, state¹⁻_vol)
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, kron_kron_state¹⁻_vol, -1/6, 1)

        # 𝐒ⁱ_full = 𝐒¹ᵉ + 𝐒²⁻ᵉ k(I,s¹v) + 𝐒²⁻ᵛᵉ k(I,s²v) + 0.5 𝐒³⁻ᵉ² k(k(I,s¹v),s¹v)
        kron_J_s1v  = ℒ.kron(J, state¹⁻_vol)
        kron_J_s2v  = ℒ.kron(J, state²⁻_vol)
        copyto!(𝐒ⁱ_full, 𝐒¹ᵉ)
        ℒ.mul!(𝐒ⁱ_full, 𝐒²⁻ᵉ, kron_J_s1v, 1, 1)
        ℒ.mul!(𝐒ⁱ_full, 𝐒²⁻ᵛᵉ, kron_J_s2v, 1, 1)
        ℒ.kron!(kron_buffer3sv, kron_J_s1v, state¹⁻_vol)
        ℒ.mul!(𝐒ⁱ_full, 𝐒³⁻ᵉ², kron_buffer3sv, 1/2, 1)

        # 𝐒ⁱ²ᵉ_full = 𝐒²ᵉ/2 + 𝐒³⁻ᵉ k(II, s¹v) / 2
        x_kron_II!(kron_buffer4sv, state¹⁻_vol)
        copyto!(𝐒ⁱ²ᵉ_full, 𝐒²ᵉ); ℒ.rdiv!(𝐒ⁱ²ᵉ_full, 2)
        ℒ.mul!(𝐒ⁱ²ᵉ_full, 𝐒³⁻ᵉ, kron_buffer4sv, 1/2, 1)

        copyto!(state¹⁻_vol_seq[t], state¹⁻_vol)
        𝐒ⁱ_full_seq[t]   .= 𝐒ⁱ_full
        𝐒ⁱ²ᵉ_full_seq[t] .= 𝐒ⁱ²ᵉ_full

        if m == 0
            fill!(init_guess, 0.0)
            x = init_guess
        else
            if m > n_exo
                if opts.verbose println("Inversion filter rrule (pruned 3rd, missing) failed at step $t: m=$m > n_exo=$n_exo") end
                return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end
            𝐒ⁱ_v   = 𝐒ⁱ_full[idx, :]
            𝐒ⁱ²ᵉ_v = 𝐒ⁱ²ᵉ_full[idx, :]
            𝐒ⁱ³ᵉ_v = 𝐒ⁱ³ᵉ[idx, :]
            si_v   = shock_independent[idx]
            fill!(init_guess, 0.0)
            x, matched = find_shocks(Val(filter_algorithm),
                                     init_guess, kb1, kb2, kb3, kb4, kb5, J,
                                     𝐒ⁱ_v, 𝐒ⁱ²ᵉ_v, 𝐒ⁱ³ᵉ_v, si_v)
            if !matched
                if opts.verbose println("Inversion filter rrule (pruned 3rd, missing) failed at step $t") end
                return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end
            if t > eff_presample
                kron_J_x = ℒ.kron(J, x)
                kron_xx  = ℒ.kron(x, x)
                kron_J_xx = ℒ.kron(J, kron_xx)
                jac_v = 𝐒ⁱ_v + 2 * 𝐒ⁱ²ᵉ_v * kron_J_x + 3 * 𝐒ⁱ³ᵉ_v * kron_J_xx
                logabsdets += m == n_exo ? ℒ.logabsdet(jac_v)[1] : ℒ.logabsdet(jac_v * jac_v')[1] / 2
                shocks² += sum(abs2, x)
                n_obs_total += m
                if !isfinite(logabsdets) || !isfinite(shocks²)
                    return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
                end
            end
        end
        x_seq[t] .= x

        # aug states
        copyto!(aug_state₁_seq[t], 1, state₁, 1); aug_state₁_seq[t][n_past+1] = 1.0; copyto!(aug_state₁_seq[t], n_past+2, x, 1, n_exo)
        copyto!(aug_state₁̂_seq[t], 1, state₁, 1); aug_state₁̂_seq[t][n_past+1] = 0.0; copyto!(aug_state₁̂_seq[t], n_past+2, x, 1, n_exo)
        copyto!(aug_state₂_seq[t], 1, state₂, 1); aug_state₂_seq[t][n_past+1] = 0.0
        copyto!(aug_state₃_seq[t], 1, state₃, 1); aug_state₃_seq[t][n_past+1] = 0.0

        ℒ.kron!(kron_aug_state₁, aug_state₁_seq[t], aug_state₁_seq[t])
        ℒ.kron!(kron_kron_aug_state₁, kron_aug_state₁, aug_state₁_seq[t])

        ℒ.mul!(state₁, 𝐒⁻¹, aug_state₁_seq[t])
        ℒ.mul!(state₂, 𝐒⁻¹, aug_state₂_seq[t]); ℒ.mul!(state₂, 𝐒⁻², kron_aug_state₁, 1/2, 1)
        ℒ.mul!(state₃, 𝐒⁻¹, aug_state₃_seq[t])
        kron_aug₁̂_aug₂ = ℒ.kron(aug_state₁̂_seq[t], aug_state₂_seq[t])
        ℒ.mul!(state₃, 𝐒⁻², kron_aug₁̂_aug₂, 1, 1)
        ℒ.mul!(state₃, 𝐒⁻³, kron_kron_aug_state₁, 1/6, 1)

        copyto!(state₁_seq[t+1], state₁)
        copyto!(state₂_seq[t+1], state₂)
        copyto!(state₃_seq[t+1], state₃)
    end

    llh = -(logabsdets + shocks² + n_obs_total * log(2 * 3.141592653589793)) / 2

    if !isfinite(llh) || llh < -1e12
        return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # Pre-allocate adjoint buffers outside the pullback closure.
    ∂𝐒_1 = zeros(size(𝐒[1]))
    ∂𝐒_2 = zeros(size(𝐒[2]))
    ∂𝐒_3 = zeros(size(𝐒[3]))
    ∂𝐒⁻¹  = zero(𝐒⁻¹)
    ∂𝐒⁻²  = zero(𝐒⁻²)
    ∂𝐒⁻³  = zero(𝐒⁻³)
    ∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)
    ∂𝐒¹⁻  = zero(𝐒¹⁻)
    ∂𝐒¹ᵉ  = zero(𝐒¹ᵉ)
    ∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)
    ∂𝐒²⁻  = zero(𝐒²⁻)
    ∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)
    ∂𝐒²⁻ᵛᵉ = zero(𝐒²⁻ᵛᵉ)
    ∂𝐒²ᵉ  = zero(𝐒²ᵉ)
    ∂𝐒³⁻ᵛ = zero(𝐒³⁻ᵛ)
    ∂𝐒³⁻ᵉ² = zero(𝐒³⁻ᵉ²)
    ∂𝐒³⁻ᵉ = zero(𝐒³⁻ᵉ)
    ∂𝐒ⁱ³ᵉ = zero(𝐒ⁱ³ᵉ)
    ∂data_in_deviations = zeros(size(data_in_deviations))
    ∂state₁_next = zeros(n_past)
    ∂state₂_next = zeros(n_past)
    ∂state₃_next = zeros(n_past)
    kronaug_buf = zeros((n_past + 1 + n_exo)^2)
    ∂kronaug    = zeros((n_past + 1 + n_exo)^2)
    ∂aug_state₁ = zeros(n_past + 1 + n_exo)
    ∂aug_state₁̂ = zeros(n_past + 1 + n_exo)
    ∂aug_state₂ = zeros(n_past + 1 + n_exo)
    ∂aug_state₃ = zeros(n_past + 1 + n_exo)
    ∂kronstate  = zeros((n_past + 1)^2)
    ∂state¹⁻_vol = zeros(n_past + 1)
    ∂𝐒ⁱ_full_buf       = zeros(n_cond, n_exo)
    ∂𝐒ⁱ²ᵉ_full_buf     = zeros(n_cond, n_exo^2)
    ∂shock_independent  = zeros(n_cond)
    ∂kronaug_for3       = zeros((n_past + 1 + n_exo)^2)
    ∂jac_v_buf          = zeros(n_cond, n_exo)

    function pruned3_missing_pullback(∂llh)
        fill!(∂𝐒_1, 0); fill!(∂𝐒_2, 0); fill!(∂𝐒_3, 0)
        fill!(∂𝐒⁻¹, 0); fill!(∂𝐒⁻², 0); fill!(∂𝐒⁻³, 0)
        fill!(∂𝐒¹⁻ᵛ, 0); fill!(∂𝐒¹⁻, 0); fill!(∂𝐒¹ᵉ, 0)
        fill!(∂𝐒²⁻ᵛ, 0); fill!(∂𝐒²⁻, 0); fill!(∂𝐒²⁻ᵉ, 0); fill!(∂𝐒²⁻ᵛᵉ, 0); fill!(∂𝐒²ᵉ, 0)
        fill!(∂𝐒³⁻ᵛ, 0); fill!(∂𝐒³⁻ᵉ², 0); fill!(∂𝐒³⁻ᵉ, 0); fill!(∂𝐒ⁱ³ᵉ, 0)
        fill!(∂data_in_deviations, 0)
        fill!(∂state₁_next, 0); fill!(∂state₂_next, 0); fill!(∂state₃_next, 0)
        fill!(kronaug_buf, 0); fill!(∂kronaug, 0)
        fill!(∂aug_state₁, 0); fill!(∂aug_state₁̂, 0); fill!(∂aug_state₂, 0); fill!(∂aug_state₃, 0)
        fill!(∂kronstate, 0); fill!(∂state¹⁻_vol, 0)
        fill!(∂𝐒ⁱ_full_buf, 0); fill!(∂𝐒ⁱ²ᵉ_full_buf, 0); fill!(∂shock_independent, 0)
        fill!(∂kronaug_for3, 0)
        fill!(∂jac_v_buf, 0)

        for t in Tt:-1:1
            aug_state₁  = aug_state₁_seq[t]
            aug_state₁̂ = aug_state₁̂_seq[t]
            aug_state₂  = aug_state₂_seq[t]
            aug_state₃  = aug_state₃_seq[t]
            state¹⁻_vol = state¹⁻_vol_seq[t]
            stm1  = state₁_seq[t]
            stm2  = state₂_seq[t]
            stm3  = state₃_seq[t]
            x = x_seq[t]
            idx = obs_idx_per_t[t]
            m = length(idx)

            # state¹⁻ = stm1, state²⁻ = stm2, state³⁻ = stm3 (current period start states)

            # state₃_next = 𝐒⁻¹ aug₃ + 𝐒⁻² kron(aug₁̂, aug₂) + (1/6) 𝐒⁻³ kron(kron(aug₁,aug₁), aug₁)
            ℒ.mul!(∂𝐒⁻¹, ∂state₃_next, aug_state₃', 1, 1)
            ℒ.mul!(∂aug_state₃, 𝐒⁻¹', ∂state₃_next)
            kron_aug₁̂_aug₂ = ℒ.kron(aug_state₁̂, aug_state₂)
            ℒ.mul!(∂𝐒⁻², ∂state₃_next, kron_aug₁̂_aug₂', 1, 1)
            ∂kronaug₁̂₂ = 𝐒⁻²' * ∂state₃_next
            fill!(∂aug_state₁̂, 0)
            fill_kron_adjoint!(∂aug_state₁̂, ∂aug_state₂, ∂kronaug₁̂₂, aug_state₁̂, aug_state₂)
            ℒ.kron!(kronaug_buf, aug_state₁, aug_state₁)
            kron_kron_aug₁ = ℒ.kron(kronaug_buf, aug_state₁)
            ℒ.mul!(∂𝐒⁻³, ∂state₃_next, kron_kron_aug₁', 1/6, 1)
            ∂kronkronaug₁ = (𝐒⁻³' * ∂state₃_next) ./ 6
            fill!(∂aug_state₁, 0)
            fill!(∂kronaug_for3, 0)
            fill_kron_adjoint!(∂aug_state₁, ∂kronaug_for3, ∂kronkronaug₁, aug_state₁, kronaug_buf)

            # state₂_next = 𝐒⁻¹ aug₂ + 0.5 𝐒⁻² kron(aug₁, aug₁)
            ℒ.mul!(∂𝐒⁻¹, ∂state₂_next, aug_state₂', 1, 1)
            ℒ.mul!(∂aug_state₂, 𝐒⁻¹', ∂state₂_next, 1, 1)
            ℒ.mul!(∂𝐒⁻², ∂state₂_next, kronaug_buf', 1/2, 1)
            ∂kronaug2 = (𝐒⁻²' * ∂state₂_next) ./ 2
            # Combine kron(aug₁, aug₁) contributions from state₃ (∂kronaug_for3) and state₂ (∂kronaug2)
            fill!(∂kronaug, 0)
            ∂kronaug .+= ∂kronaug_for3
            ∂kronaug .+= ∂kronaug2
            fill_kron_adjoint!(∂aug_state₁, ∂aug_state₁, ∂kronaug, aug_state₁, aug_state₁)

            # state₁_next = 𝐒⁻¹ aug₁
            ℒ.mul!(∂𝐒⁻¹, ∂state₁_next, aug_state₁', 1, 1)
            ℒ.mul!(∂aug_state₁, 𝐒⁻¹', ∂state₁_next, 1, 1)

            fill!(∂state₁_next, 0)
            fill!(∂state₂_next, 0)
            fill!(∂state₃_next, 0)

            # split aug_state contributions
            ∂x = ∂aug_state₁[n_past+2:end] .+ ∂aug_state₁̂[n_past+2:end]
            ∂state₁_now = ∂aug_state₁[1:n_past] .+ ∂aug_state₁̂[1:n_past]
            ∂state₂_now = ∂aug_state₂[1:n_past]
            ∂state₃_now = ∂aug_state₃[1:n_past]

            @inbounds for j in 1:n_past
                ∂state₁_next[j] += ∂state₁_now[j]
                ∂state₂_next[j] += ∂state₂_now[j]
                ∂state₃_next[j] += ∂state₃_now[j]
            end

            # zero ∂aug_state buffers for next iteration
            fill!(∂aug_state₁, 0)
            fill!(∂aug_state₁̂, 0)
            fill!(∂aug_state₂, 0)
            fill!(∂aug_state₃, 0)

            # shocks² and logabsdet contributions (only if t > presample and m > 0)
            ∂jac_v = view(∂jac_v_buf, 1:m, :); fill!(∂jac_v, 0)
            jac_v_local = zeros(m, n_exo)
            𝐒ⁱ²ᵉ_v_local = zeros(m, n_exo^2)
            𝐒ⁱ³ᵉ_v_local = zeros(m, n_exo^3)
            if m > 0
                𝐒ⁱ_v_local = 𝐒ⁱ_full_seq[t][idx, :]
                𝐒ⁱ²ᵉ_v_local = 𝐒ⁱ²ᵉ_full_seq[t][idx, :]
                𝐒ⁱ³ᵉ_v_local = 𝐒ⁱ³ᵉ[idx, :]
                kron_J_x_local = ℒ.kron(J, x)
                kron_xx_local  = ℒ.kron(x, x)
                kron_J_xx_local = ℒ.kron(J, kron_xx_local)
                jac_v_local = 𝐒ⁱ_v_local + 2 * 𝐒ⁱ²ᵉ_v_local * kron_J_x_local + 3 * 𝐒ⁱ³ᵉ_v_local * kron_J_xx_local
            end
            if m > 0 && t > eff_presample
                @inbounds for k in 1:n_exo
                    ∂x[k] += -x[k]
                end
                if m == n_exo
                    invjac_v = inv(jac_v_local)
                    ∂jac_v .+= (-0.5) .* invjac_v'
                else
                    G = inv(jac_v_local * jac_v_local')
                    ∂jac_v .+= (-0.5) .* (G * jac_v_local)
                end
                # Indirect channel: ∂jac_v → ∂x via the (J⊗x) and (J⊗ kron(x,x)) terms in jac_v.
                # d jac_v[i,r]/dx_l = 2 𝐒ⁱ²ᵉ_v[i,(r-1)n+l] + 3 (Σ_q 𝐒ⁱ³ᵉ_v[i,(r-1)n²+(l-1)n+q] x_q + Σ_p 𝐒ⁱ³ᵉ_v[i,(r-1)n²+(p-1)n+l] x_p)
                @inbounds for l in 1:n_exo
                    s = 0.0
                    for r in 1:n_exo
                        # 2nd order channel
                        for i_local in 1:m
                            s += 2 * ∂jac_v[i_local, r] * 𝐒ⁱ²ᵉ_v_local[i_local, (r-1)*n_exo + l]
                        end
                        # 3rd order channel — symmetric in p,q so two terms
                        for q in 1:n_exo
                            col = (r-1)*n_exo^2 + (l-1)*n_exo + q
                            for i_local in 1:m
                                s += 3 * ∂jac_v[i_local, r] * 𝐒ⁱ³ᵉ_v_local[i_local, col] * x[q]
                            end
                        end
                        for p in 1:n_exo
                            col = (r-1)*n_exo^2 + (p-1)*n_exo + l
                            for i_local in 1:m
                                s += 3 * ∂jac_v[i_local, r] * 𝐒ⁱ³ᵉ_v_local[i_local, col] * x[p]
                            end
                        end
                    end
                    ∂x[l] += s
                end
            end

            fill!(∂shock_independent, 0)

            if m > 0
                𝐒ⁱ_v   = 𝐒ⁱ_full_seq[t][idx, :]
                𝐒ⁱ²ᵉ_v = 𝐒ⁱ²ᵉ_v_local
                𝐒ⁱ³ᵉ_v = 𝐒ⁱ³ᵉ_v_local
                jac_v  = jac_v_local

                local λ
                if m == n_exo
                    λ = 2 * (jac_v' \ x)
                else
                    Gloc = inv(jac_v * jac_v')
                    λ = 2 * (Gloc * (jac_v * x))
                end

                # KKT topL = 2I - 2 M, M = reshape((𝐒ⁱ²ᵉ_v + 3 𝐒ⁱ³ᵉ_v kron(II, x))' λ, n, n)
                kron_II_x = ℒ.kron(II, x)
                M = reshape((𝐒ⁱ²ᵉ_v + 3 * 𝐒ⁱ³ᵉ_v * kron_II_x)' * λ, n_exo, n_exo)
                topL = 2 * ℒ.I(n_exo) - 2 * M
                fXλp = [topL          -jac_v'
                        jac_v          zeros(m, m)]

                rhs = vcat(∂x, zeros(m))
                S = fXλp' \ rhs
                Sx = S[1:n_exo]
                Sλ = S[n_exo+1:end]

                ∂v_v = Sλ
                ∂𝐒ⁱ_v = λ * Sx' - Sλ * x'

                # ∂𝐒ⁱ²ᵉ_v_kkt: same structure as 2nd order
                xSx = x * Sx'
                xx_outer = x * x'
                ∂𝐒ⁱ²ᵉ_v_top = 2 * λ * vec(xSx)'
                ∂𝐒ⁱ²ᵉ_v_F   = -Sλ * vec(xx_outer)'
                ∂𝐒ⁱ²ᵉ_v_kkt = ∂𝐒ⁱ²ᵉ_v_top + ∂𝐒ⁱ²ᵉ_v_F

                # ∂𝐒ⁱ³ᵉ_v_kkt:
                #   Top: [i, (r-1)n²+(p-1)n+q] = 3 λ[i] Sx[r] x_p x_q  →  3 λ * vec(Sx_outer_with_xx)'
                #     with kron(Sx, kron(x,x))[(r-1)n²+(p-1)n+q] = Sx[r] x_p x_q
                #   F  : [i, k] = -Sλ[i] kron(x, kron(x,x))[k] (which has entry x_r x_p x_q with index (r-1)n²+(p-1)n+q)
                kron_Sx_xx = ℒ.kron(Sx, kron_xx_local)        # length n³
                kron_x_xx  = ℒ.kron(x,  kron_xx_local)        # length n³ (= kron(x,x,x))
                ∂𝐒ⁱ³ᵉ_v_top = 3 * λ * kron_Sx_xx'
                ∂𝐒ⁱ³ᵉ_v_F   = -Sλ * kron_x_xx'
                ∂𝐒ⁱ³ᵉ_v_kkt = ∂𝐒ⁱ³ᵉ_v_top + ∂𝐒ⁱ³ᵉ_v_F

                # Add direct ∂jac_v contributions for periods past presample
                if t > eff_presample
                    ∂𝐒ⁱ_v_total    = ∂𝐒ⁱ_v + ∂jac_v
                    ∂𝐒ⁱ²ᵉ_v_total  = ∂𝐒ⁱ²ᵉ_v_kkt + 2 * ∂jac_v * ℒ.kron(J, x)'
                    ∂𝐒ⁱ³ᵉ_v_total  = ∂𝐒ⁱ³ᵉ_v_kkt + 3 * ∂jac_v * ℒ.kron(J, kron_xx_local)'
                else
                    ∂𝐒ⁱ_v_total    = ∂𝐒ⁱ_v
                    ∂𝐒ⁱ²ᵉ_v_total  = ∂𝐒ⁱ²ᵉ_v_kkt
                    ∂𝐒ⁱ³ᵉ_v_total  = ∂𝐒ⁱ³ᵉ_v_kkt
                end

                # Scatter into ∂𝐒ⁱ_full and ∂𝐒ⁱ²ᵉ_full and ∂𝐒ⁱ³ᵉ
                ∂𝐒ⁱ_full = ∂𝐒ⁱ_full_buf
                ∂𝐒ⁱ²ᵉ_full = ∂𝐒ⁱ²ᵉ_full_buf
                fill!(∂𝐒ⁱ_full, 0)
                fill!(∂𝐒ⁱ²ᵉ_full, 0)
                @inbounds for j in 1:n_exo
                    for i_local in 1:m
                        ∂𝐒ⁱ_full[idx[i_local], j] = ∂𝐒ⁱ_v_total[i_local, j]
                    end
                end
                @inbounds for j in 1:n_exo^2
                    for i_local in 1:m
                        ∂𝐒ⁱ²ᵉ_full[idx[i_local], j] = ∂𝐒ⁱ²ᵉ_v_total[i_local, j]
                    end
                end
                @inbounds for j in 1:n_exo^3
                    for i_local in 1:m
                        ∂𝐒ⁱ³ᵉ[idx[i_local], j] += ∂𝐒ⁱ³ᵉ_v_total[i_local, j]
                    end
                end

                @inbounds for i_local in 1:m
                    ∂shock_independent[idx[i_local]] += ∂v_v[i_local]
                end

                # Propagate ∂𝐒ⁱ_full back through:
                #   𝐒ⁱ_full = 𝐒¹ᵉ + 𝐒²⁻ᵉ k(I,s¹v) + 𝐒²⁻ᵛᵉ k(I,s²v) + 0.5 𝐒³⁻ᵉ² k(k(I,s¹v),s¹v)
                ∂𝐒¹ᵉ .+= ∂𝐒ⁱ_full
                kron_J_s1v = ℒ.kron(J, state¹⁻_vol)
                state²⁻_vol_local = vcat(stm2, 0.0)
                kron_J_s2v = ℒ.kron(J, state²⁻_vol_local)
                ℒ.mul!(∂𝐒²⁻ᵉ, ∂𝐒ⁱ_full, kron_J_s1v', 1, 1)
                ℒ.mul!(∂𝐒²⁻ᵛᵉ, ∂𝐒ⁱ_full, kron_J_s2v', 1, 1)
                kron_kron_J_s1v_s1v = ℒ.kron(kron_J_s1v, state¹⁻_vol)
                ℒ.mul!(∂𝐒³⁻ᵉ², ∂𝐒ⁱ_full, kron_kron_J_s1v_s1v', 1/2, 1)

                # Propagate to ∂state¹⁻_vol via (I⊗s¹v) and via (k(I,s¹v),s¹v)
                ∂kronIs1v_a = 𝐒²⁻ᵉ' * ∂𝐒ⁱ_full   # n_exo*(n_past+1) × n_exo
                fill!(∂state¹⁻_vol, 0)
                @inbounds for p in 1:(n_past + 1)
                    s = 0.0
                    for j in 1:n_exo
                        s += ∂kronIs1v_a[(j-1)*(n_past+1) + p, j]
                    end
                    ∂state¹⁻_vol[p] += s
                end
                # 𝐒³⁻ᵉ² contribution: (1/2) 𝐒³⁻ᵉ² k(k(I,s¹v), s¹v).
                # Use full kron-adjoint: ∂(k(k(I,s¹v),s¹v)) = 0.5 𝐒³⁻ᵉ²' ∂𝐒ⁱ_full (a (n_exo·(n_past+1)²) × n_exo matrix).
                # Decompose: u := k(I, s¹v) (shape n_exo·(n_past+1) × n_exo), then kron(u, s¹v).
                ∂kron_u_s1v = (𝐒³⁻ᵉ²' * ∂𝐒ⁱ_full) ./ 2
                # Apply Kronecker adjoint: ∂u = sum_{q,j} ∂kron_u_s1v[(q-1)·(n_past+1)+r, j]·s1v[r] etc.
                u_mat = ℒ.kron(J, state¹⁻_vol)   # (n_exo·(n_past+1)) × n_exo
                ∂u_mat = zeros(size(u_mat))
                @inbounds for j in 1:n_exo
                    for q in 1:(n_exo*(n_past+1))
                        for r in 1:(n_past+1)
                            ∂u_mat[q, j] += ∂kron_u_s1v[(q-1)*(n_past+1) + r, j] * state¹⁻_vol[r]
                            ∂state¹⁻_vol[r] += ∂kron_u_s1v[(q-1)*(n_past+1) + r, j] * u_mat[q, j]
                        end
                    end
                end
                # Now propagate ∂u_mat through u_mat = kron(J, s¹v) (J fixed I_n)
                @inbounds for p in 1:(n_past + 1)
                    s = 0.0
                    for j in 1:n_exo
                        s += ∂u_mat[(j-1)*(n_past+1) + p, j]
                    end
                    ∂state¹⁻_vol[p] += s
                end

                # Propagate to ∂state²⁻_vol via (I⊗s²v) and then to ∂state₂_now (since state²⁻_vol = vcat(state₂, 0))
                ∂kronIs2v = 𝐒²⁻ᵛᵉ' * ∂𝐒ⁱ_full
                ∂state²⁻_vol = zeros(n_past + 1)
                @inbounds for p in 1:(n_past + 1)
                    s = 0.0
                    for j in 1:n_exo
                        s += ∂kronIs2v[(j-1)*(n_past+1) + p, j]
                    end
                    ∂state²⁻_vol[p] = s
                end
                # state²⁻_vol = vcat(stm2, 0) → ∂stm2 += ∂state²⁻_vol[1:n_past]
                @inbounds for j in 1:n_past
                    ∂state₂_next[j] += ∂state²⁻_vol[j]
                end

                # Propagate ∂𝐒ⁱ²ᵉ_full back through 𝐒ⁱ²ᵉ_full = 𝐒²ᵉ/2 + 𝐒³⁻ᵉ k(II, s¹v)/2
                ∂𝐒²ᵉ .+= ∂𝐒ⁱ²ᵉ_full ./ 2
                # 𝐒³⁻ᵉ contribution: 0.5 𝐒³⁻ᵉ k(II, s¹v). The kron(II, s¹v) is (n²·(n_past+1)) × n².
                kron_II_s1v = ℒ.kron(II, state¹⁻_vol)  # n²·(n_past+1) × n²
                ℒ.mul!(∂𝐒³⁻ᵉ, ∂𝐒ⁱ²ᵉ_full, kron_II_s1v', 1/2, 1)
                ∂kronIIs1v = (𝐒³⁻ᵉ' * ∂𝐒ⁱ²ᵉ_full) ./ 2  # (n²·(n_past+1)) × n²
                # kron(II, s¹v)[(j-1)(n_past+1)+p, k] = II[j,k] * s¹v[p] = δ_{jk} s¹v[p]
                @inbounds for p in 1:(n_past + 1)
                    s = 0.0
                    for j in 1:n_exo^2
                        s += ∂kronIIs1v[(j-1)*(n_past+1) + p, j]
                    end
                    ∂state¹⁻_vol[p] += s
                end
            else
                fill!(∂state¹⁻_vol, 0)
            end

            # Now propagate shock_independent dependencies
            @inbounds for i in 1:n_cond
                ∂data_in_deviations[i, t] += ∂shock_independent[i]
            end
            ℒ.mul!(∂𝐒¹⁻ᵛ, ∂shock_independent, state¹⁻_vol', -1, 1)
            ℒ.mul!(∂state¹⁻_vol, 𝐒¹⁻ᵛ', ∂shock_independent, -1, 1)
            # 𝐒¹⁻ * stm2 and 𝐒¹⁻ * stm3
            ℒ.mul!(∂𝐒¹⁻, ∂shock_independent, stm2', -1, 1)
            ∂stm2_contrib = 𝐒¹⁻' * ∂shock_independent
            @inbounds for j in 1:n_past
                ∂state₂_next[j] += -∂stm2_contrib[j]
            end
            ℒ.mul!(∂𝐒¹⁻, ∂shock_independent, stm3', -1, 1)
            ∂stm3_contrib = 𝐒¹⁻' * ∂shock_independent
            @inbounds for j in 1:n_past
                ∂state₃_next[j] += -∂stm3_contrib[j]
            end
            # 0.5 𝐒²⁻ᵛ k(s¹v, s¹v)
            kron_sv = ℒ.kron(state¹⁻_vol, state¹⁻_vol)
            ℒ.mul!(∂𝐒²⁻ᵛ, ∂shock_independent, kron_sv', -1/2, 1)
            ∂kron_sv = -(𝐒²⁻ᵛ' * ∂shock_independent) ./ 2
            fill!(∂kronstate, 0)
            ∂kronstate .+= ∂kron_sv
            # 𝐒²⁻ k(stm1, stm2)
            kron_s1_s2 = ℒ.kron(stm1, stm2)
            ℒ.mul!(∂𝐒²⁻, ∂shock_independent, kron_s1_s2', -1, 1)
            ∂kron_s1_s2 = -(𝐒²⁻' * ∂shock_independent)
            ∂stm1_s12 = zeros(n_past)
            ∂stm2_s12 = zeros(n_past)
            fill_kron_adjoint!(∂stm1_s12, ∂stm2_s12, ∂kron_s1_s2, stm1, stm2)
            @inbounds for j in 1:n_past
                ∂state₁_next[j] += ∂stm1_s12[j]
                ∂state₂_next[j] += ∂stm2_s12[j]
            end
            # (1/6) 𝐒³⁻ᵛ k(s¹v, k(s¹v, s¹v))
            kron_s1v_3 = ℒ.kron(state¹⁻_vol, kron_sv)
            ℒ.mul!(∂𝐒³⁻ᵛ, ∂shock_independent, kron_s1v_3', -1/6, 1)
            ∂kron_s1v_3 = -(𝐒³⁻ᵛ' * ∂shock_independent) ./ 6
            # Decompose kron(s¹v, kron(s¹v, s¹v)): use chain - first kron(a, b) with a=s¹v, b=kron(s¹v, s¹v)
            ∂a_outer = zeros(n_past + 1)
            ∂b_outer = zeros((n_past + 1)^2)
            fill_kron_adjoint!(∂a_outer, ∂b_outer, ∂kron_s1v_3, state¹⁻_vol, kron_sv)
            ∂state¹⁻_vol .+= ∂a_outer
            ∂kronstate   .+= ∂b_outer
            # Now ∂kron_sv = ∂kronstate (both contributions accumulated)
            fill_kron_adjoint!(∂state¹⁻_vol, ∂state¹⁻_vol, ∂kronstate, state¹⁻_vol, state¹⁻_vol)

            # state¹⁻_vol = vcat(state₁, 1) → ∂state₁_next += ∂state¹⁻_vol[1:n_past]
            @inbounds for j in 1:n_past
                ∂state₁_next[j] += ∂state¹⁻_vol[j]
            end
        end

        # Apply ∂llh scaling and assemble ∂𝐒
        ∂𝐒_1[Tcc.past_not_future_and_mixed_idx, :]                 .+= ∂𝐒⁻¹
        ∂𝐒_2[Tcc.past_not_future_and_mixed_idx, :]                 .+= ∂𝐒⁻²
        ∂𝐒_3[Tcc.past_not_future_and_mixed_idx, :]                 .+= ∂𝐒⁻³
        ∂𝐒_1[cond_var_idx, 1:n_past+1]                              .+= ∂𝐒¹⁻ᵛ
        ∂𝐒_1[cond_var_idx, 1:n_past]                                .+= ∂𝐒¹⁻
        ∂𝐒_1[cond_var_idx, end-n_exo+1:end]                         .+= ∂𝐒¹ᵉ
        ∂𝐒_2[cond_var_idx, var_vol²_idxs]                           .+= ∂𝐒²⁻ᵛ
        ∂𝐒_2[cond_var_idx, var²_idxs]                               .+= ∂𝐒²⁻
        ∂𝐒_2[cond_var_idx, shockvar²_idxs]                          .+= ∂𝐒²⁻ᵉ
        ∂𝐒_2[cond_var_idx, shockvar_idxs]                           .+= ∂𝐒²⁻ᵛᵉ
        ∂𝐒_2[cond_var_idx, shock²_idxs]                             .+= ∂𝐒²ᵉ
        ∂𝐒_3[cond_var_idx, var_vol³_idxs]                           .+= ∂𝐒³⁻ᵛ
        ∂𝐒_3[cond_var_idx, shockvar³2_idxs]                         .+= ∂𝐒³⁻ᵉ²
        ∂𝐒_3[cond_var_idx, shockvar³_idxs]                          .+= ∂𝐒³⁻ᵉ
        # 𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6 → ∂𝐒³ᵉ = ∂𝐒ⁱ³ᵉ / 6
        ∂𝐒_3[cond_var_idx, shock³_idxs]                             .+= ∂𝐒ⁱ³ᵉ ./ 6

        ℒ.rmul!(∂𝐒_1, ∂llh)
        ℒ.rmul!(∂𝐒_2, ∂llh)
        ℒ.rmul!(∂𝐒_3, ∂llh)
        ℒ.rmul!(∂data_in_deviations, ∂llh)

        ∂state₀_full_1 = zeros(size(state[1]))
        ∂state₀_full_2 = zeros(size(state[2]))
        ∂state₀_full_3 = zeros(size(state[3]))
        ∂state₀_full_1[Tcc.past_not_future_and_mixed_idx] .= ∂state₁_next .* ∂llh
        ∂state₀_full_2[Tcc.past_not_future_and_mixed_idx] .= ∂state₂_next .* ∂llh
        ∂state₀_full_3[Tcc.past_not_future_and_mixed_idx] .= ∂state₃_next .* ∂llh

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), [∂𝐒_1, ∂𝐒_2, ∂𝐒_3], ∂data_in_deviations, NoTangent(), [∂state₀_full_1, ∂state₀_full_2, ∂state₀_full_3], NoTangent()
    end

    return llh, pruned3_missing_pullback
end


function rrule(::typeof(calculate_loglikelihood),
                ::Val{:inversion},
                ::Val{:pruned_third_order},
                observables_index::Vector{Int},
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                constants::constants,
                state::Vector{Vector{Float64}}, 
                workspaces::workspaces; 
                # timer::TimerOutput = TimerOutput(),
                on_failure_loglikelihood = -Inf,
                warmup_iterations::Int = 0,
                presample_periods::Int = 0,
                initial_covariance::Symbol = :theoretical,
                opts::CalculationOptions = merge_calculation_options(),
                filter_algorithm::Symbol = :LagrangeNewton)
    T = constants.post_model_macro
    ws = workspaces.inversion

    # @timeit_debug timer "Inversion filter - forward" begin
    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = observables_index

    shocks² = 0.0
    logabsdets = 0.0

    cc = ensure_conditional_forecast_constants!(constants; third_order = true)
    tc = constants.third_order
    shockvar_idxs = cc.shockvar_idxs
    shock_idxs = cc.shock_idxs
    shock²_idxs = cc.shock²_idxs
    shockvar²_idxs = cc.shockvar²_idxs
    var_vol²_idxs = cc.var_vol²_idxs
    var²_idxs = cc.var²_idxs
    var_vol³_idxs = tc.var_vol³_idxs
    shock_idxs2 = tc.shock_idxs2
    shock_idxs3 = tc.shock_idxs3
    shock³_idxs = tc.shock³_idxs
    shockvar1_idxs = tc.shockvar1_idxs
    shockvar2_idxs = tc.shockvar2_idxs
    shockvar3_idxs = tc.shockvar3_idxs
    shockvar³2_idxs = tc.shockvar³2_idxs
    shockvar³_idxs = tc.shockvar³_idxs

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
        state²⁻_vol = vcat(state²⁻, 0)

        state³⁻ = state₃#[T.past_not_future_and_mixed_idx]

        shock_independent = copy(data_in_deviations[:,i])
    
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒¹⁻, state²⁻, -1, 1)

        ℒ.mul!(shock_independent, 𝐒¹⁻, state³⁻, -1, 1)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
        
        ℒ.mul!(shock_independent, 𝐒²⁻, ℒ.kron(state¹⁻, state²⁻), -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒²⁻ᵛᵉ * ℒ.kron(ℒ.I(T.nExo), state²⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
    
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
            return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
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
            if T.nExo == length(observables_index)
                logabsdets += ℒ.logabsdet(jacc[i])[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i]))
            end
    
            shocks² += sum(abs2,x[i])

            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
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
    llh = -(logabsdets + shocks² + (length(observables_index) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2


    ∂𝐒 = [zero(𝐒[1]), zero(𝐒[2]), zero(𝐒[3])]

    ∂data_in_deviations = similar(data_in_deviations)

    # end # timeit_debug

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

    # Pre-allocated per-period buffers (formerly created fresh each iteration).
    ∂jacc_buf   = zero(jacc[1])
    ∂xλ_buf     = zeros(T.nExo + size(jacc[1], 1))
    S_buf       = zeros(T.nExo + size(jacc[1], 1))
    kronSλ      = zeros(length(cond_var_idx) * T.nExo)
    kronxS      = zeros(T.nExo * length(cond_var_idx))
    kron_S1_kxλ = zeros(T.nExo * length(kronxλ[1]))
    kron_xx_S2  = zeros(length(kronxx[1]) * size(jacc[1], 1))
    kron_S1_kxxλ = zeros(T.nExo * length(kronxxλ[1]))
    kron_xxx_S2  = zeros(length(kronxxx[1]) * size(jacc[1], 1))
    kron_xλ      = zero(kronxλ[1])
    kron_xxλ     = zero(kronxxλ[1])
    kron_Ix      = zero(ℒ.kron(ℒ.I(T.nExo), x[1]))
    kron_Ixx     = zero(ℒ.kron(ℒ.I(T.nExo), kronxx[1]))
    ∂𝐒ⁱ²ᵉ_tmp   = zero(𝐒ⁱ²ᵉ[1])

    function inversion_filter_loglikelihood_pullback(∂llh)
        # @timeit_debug timer "Inversion filter - pullback" begin
        fill!(∂𝐒ⁱ, 0)
        fill!(∂𝐒²ᵉ, 0)
        fill!(∂𝐒ⁱ³ᵉ, 0)

        fill!(∂𝐒¹ᵉ, 0)
        fill!(∂𝐒¹⁻, 0)
        fill!(∂𝐒²⁻, 0)
        fill!(∂𝐒²⁻ᵉ, 0)
        fill!(∂𝐒²⁻ᵛᵉ, 0)
        fill!(∂𝐒³⁻ᵉ, 0)
        fill!(∂𝐒³⁻ᵉ², 0)

        fill!(∂𝐒¹⁻ᵛ, 0)
        fill!(∂𝐒²⁻ᵛ, 0)
        fill!(∂𝐒³⁻ᵛ, 0)
        
        fill!(∂𝐒⁻¹, 0)
        fill!(∂𝐒⁻², 0)
        fill!(∂𝐒⁻³, 0)

        fill!(∂aug_state₁̂, 0)
        fill!(∂state¹⁻_vol, 0)
        fill!(∂x, 0)
        fill!(∂kronxx, 0)
        fill!(∂kronstate¹⁻_vol, 0)
        fill!(∂state[1], 0)
        fill!(∂state[2], 0)
        fill!(∂state[3], 0)

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

            if i < size(data_in_deviations,2)
                ∂state[1] *= 0
                ∂state[2] *= 0
                ∂state[3] *= 0
            end

            # aug_state₁[i] = [state₁; 1; x[i]]
            ∂state[1] .+= @view ∂aug_state₁[1:length(∂state[1])]

            @views copyto!(∂x, ∂aug_state₁[T.nPast_not_future_and_mixed+2:end])

            # aug_state₁̂[i] = [state₁; 0; x[i]]
            ∂state[1] .+= @view ∂aug_state₁̂[1:length(∂state[1])]

            @views ℒ.axpy!(1, ∂aug_state₁̂[T.nPast_not_future_and_mixed+2:end], ∂x)

            # aug_state₂[i] = [state₂; 0; zeros(T.nExo)]
            ∂state[2] .+= @view ∂aug_state₂[1:length(∂state[1])]
            
            # aug_state₃[i] = [state₃; 0; zeros(T.nExo)]
            ∂state[3] .+= @view ∂aug_state₃[1:length(∂state[1])]

            # shocks² += sum(abs2,x[i]) — only for i > presample_periods
            if i > presample_periods
                if i < size(data_in_deviations,2)
                    @inbounds @simd for k in eachindex(∂x)
                        ∂x[k] -= x[i][k]
                    end
                else
                    @inbounds @simd for k in eachindex(∂x)
                        ∂x[k] += x[i][k]
                    end
                end
            end

            # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1] — only for i > presample_periods
            if i > presample_periods
                if size(jacc[i], 1) == size(jacc[i], 2)
                    jacc_lu = ℒ.lu(jacc[i], check = false)
                    if !ℒ.issuccess(jacc_lu)
                        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end
                    copyto!(∂jacc_buf, inv(jacc_lu)')
                    ∂jacc = ∂jacc_buf
                else
                    ∂jacc = ℒ.pinv(jacc[i])'
                    if !all(isfinite, ∂jacc)
                        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end
                end
            else
                fill!(∂jacc_buf, 0)
                ∂jacc = ∂jacc_buf
            end

            # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x, x))
            # ∂𝐒ⁱ = -∂jacc / 2 # fine

            ℒ.mul!(kron_Ix, 𝐒ⁱ²ᵉ[i]', ∂jacc)
            ∂kronIx = kron_Ix

            if i < size(data_in_deviations,2)
                fill_kron_adjoint_∂B!(∂kronIx, ∂x, -ℒ.I(T.nExo))
            else
                fill_kron_adjoint_∂B!(∂kronIx, ∂x, ℒ.I(T.nExo))
            end

            ℒ.kron!(kron_Ix, ℒ.I(T.nExo), x[i])
            ℒ.mul!(∂𝐒ⁱ²ᵉ_tmp, ∂jacc, kron_Ix', -1, 0)

            ℒ.mul!(kron_Ixx, 𝐒ⁱ³ᵉ', ∂jacc, 3/2, 0)
            ∂kronIxx = kron_Ixx

            fill!(∂kronxx, 0)

            if i < size(data_in_deviations,2)
                fill_kron_adjoint_∂B!(∂kronIxx, ∂kronxx, -ℒ.I(T.nExo))
            else
                fill_kron_adjoint_∂B!(∂kronIxx, ∂kronxx, ℒ.I(T.nExo))
            end

            fill_kron_adjoint!(∂x, ∂x, ∂kronxx, x[i], x[i])

            ℒ.kron!(kron_Ixx, ℒ.I(T.nExo), kronxx[i])
            ℒ.mul!(∂𝐒ⁱ³ᵉ, ∂jacc, kron_Ixx', -3/2, 1)

            # find_shocks
            # ∂xλ = vcat(∂x, zero(λ[i]))
            copyto!(∂xλ_buf, 1, ∂x, 1, length(∂x))
            fill!(view(∂xλ_buf, length(∂x)+1:length(∂xλ_buf)), 0)

            S_solved = fXλp[i]' \ ∂xλ_buf
            copyto!(S_buf, S_solved)
            S = S_buf

            if i < size(data_in_deviations,2)
                ℒ.rmul!(S, -1)
            end

            S1 = view(S, 1:T.nExo)
            S2 = view(S, T.nExo+1:length(S))
            ∂shock_independent = S2

            # ∂𝐒ⁱ = ℒ.kron(S[1:T.nExo], λ[i]) - ℒ.kron(x[i], S[T.nExo+1:end])
            ℒ.kron!(kronSλ, S1, λ[i])
            ℒ.kron!(kronxS, x[i], S2)
            ℒ.axpy!(-1, kronxS, kronSλ)
            copyto!(∂𝐒ⁱ, kronSλ)
            ℒ.axpy!(-1/2, ∂jacc, ∂𝐒ⁱ)

            # ∂𝐒ⁱ²ᵉ += reshape(2 * ℒ.kron(S[1:T.nExo], ℒ.kron(x[i], λ[i])) - ℒ.kron(kronxx[i], S[T.nExo+1:end]), size(∂𝐒ⁱ²ᵉ))
            ℒ.kron!(kron_xλ, x[i], λ[i])
            ℒ.kron!(kron_S1_kxλ, S1, kron_xλ)
            ℒ.kron!(kron_xx_S2, kronxx[i], S2)
            ℒ.axpby!(-1, kron_xx_S2, 2, kron_S1_kxλ)
            ∂𝐒ⁱ²ᵉ_tmp .+= reshape(kron_S1_kxλ, size(∂𝐒ⁱ²ᵉ_tmp))
            ∂𝐒ⁱ²ᵉ = ∂𝐒ⁱ²ᵉ_tmp

            # ∂𝐒ⁱ³ᵉ += reshape(3 * ℒ.kron(S[1:T.nExo], ℒ.kron(ℒ.kron(x[i], x[i]), λ[i])) - ℒ.kron(kronxxx[i], S[T.nExo+1:end]), size(∂𝐒ⁱ³ᵉ))
            ℒ.kron!(kron_xxλ, kronxx[i], λ[i])
            ℒ.kron!(kron_S1_kxxλ, S1, kron_xxλ)
            ℒ.kron!(kron_xxx_S2, kronxxx[i], S2)
            ℒ.axpby!(-1, kron_xxx_S2, 3, kron_S1_kxxλ)
            ∂𝐒ⁱ³ᵉ .+= reshape(kron_S1_kxxλ, size(∂𝐒ⁱ³ᵉ))

            # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒²⁻ᵛᵉ * ℒ.kron(ℒ.I(T.nExo), state²⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
            ∂kronstate¹⁻_vol *= 0

            state¹⁻_vol = [aug_state₁[i][1:T.nPast_not_future_and_mixed];1] # define here as it is used multiple times later
            state¹⁻ = aug_state₁[i][1:T.nPast_not_future_and_mixed]
            state²⁻ = aug_state₂[i][1:T.nPast_not_future_and_mixed]
            state²⁻_vol = [state²⁻; 0]
            state³⁻ = aug_state₃[i][1:T.nPast_not_future_and_mixed]

            ∂𝐒¹ᵉ += ∂𝐒ⁱ

            ∂state¹⁻_vol *= 0

            ∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

            fill_kron_adjoint_∂A!(∂kronIstate¹⁻_vol, ∂state¹⁻_vol, ℒ.I(T.nExo))

            ∂𝐒²⁻ᵉ += ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)'

            ∂kronIstate²⁻ = 𝐒²⁻ᵛᵉ' * ∂𝐒ⁱ

            ∂state²⁻_vol = zeros(length(state²⁻_vol))

            fill_kron_adjoint_∂A!(∂kronIstate²⁻, ∂state²⁻_vol, ℒ.I(T.nExo))

            ∂𝐒²⁻ᵛᵉ += ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state²⁻_vol)'

            ∂state[2] += ∂state²⁻_vol[1:end-1]

            ∂kronIstate¹⁻_volstate¹⁻_vol = 𝐒³⁻ᵉ²' * ∂𝐒ⁱ / 2

            fill_kron_adjoint_∂A!(∂kronIstate¹⁻_volstate¹⁻_vol, ∂kronstate¹⁻_vol, ℒ.I(T.nExo))

            ∂𝐒³⁻ᵉ² += ∂𝐒ⁱ * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol)' / 2
            
            # 𝐒ⁱ²ᵉ[i] = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2
            ∂𝐒²ᵉ += ∂𝐒ⁱ²ᵉ / 2
            
            ∂𝐒³⁻ᵉ += ∂𝐒ⁱ²ᵉ * ℒ.kron(II, state¹⁻_vol)' / 2
            
            ∂kronIIstate¹⁻_vol = 𝐒³⁻ᵉ' * ∂𝐒ⁱ²ᵉ / 2

            fill_kron_adjoint_∂A!(∂kronIIstate¹⁻_vol, ∂state¹⁻_vol, II)

            # shock_independent = copy(data_in_deviations[:,i])
            @inbounds @simd for k in eachindex(∂shock_independent); ∂data_in_deviations[k, i] = ∂shock_independent[k]; end

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

        fill!(∂𝐒[1], 0)
        fill!(∂𝐒[2], 0)
        fill!(∂𝐒[3], 0)

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

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂𝐒, ∂data_in_deviations * ∂llh, NoTangent(), ∂state, NoTangent()
    end

    return llh, inversion_filter_loglikelihood_pullback
end

function rrule(::typeof(calculate_loglikelihood_with_missing), ::Val{:inversion}, ::Val{:third_order},
                                       observables_index::Vector{Int},
                                             𝐒::Vector{AbstractMatrix{Float64}},
                                             data_in_deviations::Matrix{Float64},
                                             constants::constants,
                                             state::Vector{Float64},
                                             workspaces::workspaces,
                                             obs_idx_per_t::Vector{Vector{Int}};
                                             warmup_iterations::Int = 0,
                                             on_failure_loglikelihood = -Inf,
                                             presample_periods::Int = 0,
                                             initial_covariance::Symbol = :theoretical,
                                             opts::CalculationOptions = merge_calculation_options(),
                                             filter_algorithm::Symbol = :LagrangeNewton)
    Tcc = constants.post_model_macro
    n_exo  = Tcc.nExo
    n_past = Tcc.nPast_not_future_and_mixed
    cond_var_idx = observables_index
    n_cond = length(cond_var_idx)
    Tt = size(data_in_deviations, 2)

    eff_presample = presample_periods + warmup_iterations

    ws = workspaces.inversion
    ensure_inversion_buffers!(ws, n_exo, n_past; third_order = true)
    ensure_inversion_estimation_buffers!(ws, n_exo, n_cond; third_order = true)

    cc = ensure_conditional_forecast_constants!(constants; third_order = true)
    tc = constants.third_order
    shock²_idxs     = cc.shock²_idxs
    shockvar²_idxs  = cc.shockvar²_idxs
    var_vol²_idxs   = cc.var_vol²_idxs
    var_vol³_idxs   = tc.var_vol³_idxs
    shock³_idxs     = tc.shock³_idxs
    shockvar³2_idxs = tc.shockvar³2_idxs
    shockvar³_idxs  = tc.shockvar³_idxs

    𝐒⁻¹   = 𝐒[1][Tcc.past_not_future_and_mixed_idx, :]
    𝐒¹⁻ᵛ  = 𝐒[1][cond_var_idx, 1:n_past+1]
    𝐒¹ᵉ   = 𝐒[1][cond_var_idx, end-n_exo+1:end]
    𝐒²⁻ᵛ  = collect(𝐒[2][cond_var_idx, var_vol²_idxs])
    𝐒²⁻ᵉ  = collect(𝐒[2][cond_var_idx, shockvar²_idxs])
    𝐒²ᵉ   = collect(𝐒[2][cond_var_idx, shock²_idxs])
    𝐒⁻²   = collect(𝐒[2][Tcc.past_not_future_and_mixed_idx, :])
    𝐒³⁻ᵛ  = collect(𝐒[3][cond_var_idx, var_vol³_idxs])
    𝐒³⁻ᵉ² = collect(𝐒[3][cond_var_idx, shockvar³2_idxs])
    𝐒³⁻ᵉ  = collect(𝐒[3][cond_var_idx, shockvar³_idxs])
    𝐒³ᵉ   = collect(𝐒[3][cond_var_idx, shock³_idxs])
    𝐒⁻³   = collect(𝐒[3][Tcc.past_not_future_and_mixed_idx, :])

    𝐒ⁱ³ᵉ = 𝐒³ᵉ ./ 6
    J  = ℒ.I(n_exo)
    II = sparse(ℒ.I(n_exo^2))

    st = copy(state[Tcc.past_not_future_and_mixed_idx])

    ensure_inversion_rrule_buffers!(ws, n_exo, n_past, n_cond, Tt; order = :third_order)

    # Per-period storage (cached in workspace; see ensure_inversion_rrule_buffers!)
    st_seq          = ws.state_seq_rrule
    st_seq[1]      .= st
    state¹⁻_vol_seq = ws.state¹⁻_vol_seq_rrule
    aug_state_seq   = ws.aug_state_seq_rrule
    x_seq           = ws.x_seq_rrule
    𝐒ⁱ_full_seq    = ws.𝐒ⁱ_full_seq_rrule
    𝐒ⁱ²ᵉ_full_seq  = ws.𝐒ⁱ²ᵉ_full_seq_rrule

    shocks² = 0.0
    logabsdets = 0.0
    n_obs_total = 0

    state¹⁻_vol           = ws.state_vol
    shock_independent     = ws.shock_independent
    kronstate¹⁻_vol       = ws.kronstate_vol
    kron_kron_state¹⁻_vol = ws.kronstate_vol³
    𝐒ⁱ_full              = ws.Si_buffer
    𝐒ⁱ²ᵉ_full            = ws.Si2e_buffer
    kron_buffer3sv = zeros(n_exo * (n_past + 1)^2, n_exo)
    kron_buffer4sv = zeros(n_exo^2 * (n_past + 1), n_exo^2)
    kron_aug_state      = ws.kronaug_state
    kron_kron_aug_state = ws.kron_kron_aug_state
    init_guess = ws.init_guess
    kb1 = ws.kron_buffer
    kb2 = ws.kron_buffer²
    kb3 = ws.kron_buffer2
    kb4 = ws.kron_buffer3
    kb5 = ws.kron_buffer4

    for t in 1:Tt
        idx = obs_idx_per_t[t]
        m = length(idx)

        copyto!(state¹⁻_vol, 1, st, 1)
        state¹⁻_vol[end] = 1.0

        # shock_independent = data - 𝐒¹⁻ᵛ s¹v - 0.5 𝐒²⁻ᵛ k(s¹v,s¹v) - (1/6) 𝐒³⁻ᵛ k(s¹v,k(s¹v,s¹v))
        copyto!(shock_independent, view(data_in_deviations, :, t))
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        ℒ.kron!(kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol, -1/2, 1)
        ℒ.kron!(kron_kron_state¹⁻_vol, kronstate¹⁻_vol, state¹⁻_vol)
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, kron_kron_state¹⁻_vol, -1/6, 1)

        # 𝐒ⁱ_full = 𝐒¹ᵉ + 𝐒²⁻ᵉ k(I,s¹v) + 0.5 𝐒³⁻ᵉ² k(k(I,s¹v),s¹v)
        kron_J_s1v = ℒ.kron(J, state¹⁻_vol)
        copyto!(𝐒ⁱ_full, 𝐒¹ᵉ)
        ℒ.mul!(𝐒ⁱ_full, 𝐒²⁻ᵉ, kron_J_s1v, 1, 1)
        ℒ.kron!(kron_buffer3sv, kron_J_s1v, state¹⁻_vol)
        ℒ.mul!(𝐒ⁱ_full, 𝐒³⁻ᵉ², kron_buffer3sv, 1/2, 1)

        # 𝐒ⁱ²ᵉ_full = 𝐒²ᵉ/2 + 𝐒³⁻ᵉ k(II, s¹v)/2
        x_kron_II!(kron_buffer4sv, state¹⁻_vol)
        copyto!(𝐒ⁱ²ᵉ_full, 𝐒²ᵉ); ℒ.rdiv!(𝐒ⁱ²ᵉ_full, 2)
        ℒ.mul!(𝐒ⁱ²ᵉ_full, 𝐒³⁻ᵉ, kron_buffer4sv, 1/2, 1)

        copyto!(state¹⁻_vol_seq[t], state¹⁻_vol)
        𝐒ⁱ_full_seq[t]   .= 𝐒ⁱ_full
        𝐒ⁱ²ᵉ_full_seq[t] .= 𝐒ⁱ²ᵉ_full

        if m == 0
            x = zeros(n_exo)
        else
            if m > n_exo
                if opts.verbose println("Inversion filter rrule (3rd, missing) failed at step $t: m=$m > n_exo=$n_exo") end
                return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end
            𝐒ⁱ_v   = 𝐒ⁱ_full[idx, :]
            𝐒ⁱ²ᵉ_v = 𝐒ⁱ²ᵉ_full[idx, :]
            𝐒ⁱ³ᵉ_v = 𝐒ⁱ³ᵉ[idx, :]
            si_v   = shock_independent[idx]
            fill!(init_guess, 0.0)
            x, matched = find_shocks(Val(filter_algorithm),
                                     init_guess, kb1, kb2, kb3, kb4, kb5, J,
                                     𝐒ⁱ_v, 𝐒ⁱ²ᵉ_v, 𝐒ⁱ³ᵉ_v, si_v)
            if !matched
                if opts.verbose println("Inversion filter rrule (3rd, missing) failed at step $t") end
                return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end
            if t > eff_presample
                kron_J_x = ℒ.kron(J, x)
                kron_xx  = ℒ.kron(x, x)
                kron_J_xx = ℒ.kron(J, kron_xx)
                jac_v = 𝐒ⁱ_v + 2 * 𝐒ⁱ²ᵉ_v * kron_J_x + 3 * 𝐒ⁱ³ᵉ_v * kron_J_xx
                logabsdets += m == n_exo ? ℒ.logabsdet(jac_v)[1] : ℒ.logabsdet(jac_v * jac_v')[1] / 2
                shocks² += sum(abs2, x)
                n_obs_total += m
                if !isfinite(logabsdets) || !isfinite(shocks²)
                    return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
                end
            end
        end
        x_seq[t] .= x

        copyto!(aug_state_seq[t], 1, st, 1); aug_state_seq[t][n_past+1] = 1.0; copyto!(aug_state_seq[t], n_past+2, x, 1, n_exo)
        ℒ.kron!(kron_aug_state, aug_state_seq[t], aug_state_seq[t])
        ℒ.kron!(kron_kron_aug_state, kron_aug_state, aug_state_seq[t])
        ℒ.mul!(st, 𝐒⁻¹, aug_state_seq[t])
        ℒ.mul!(st, 𝐒⁻², kron_aug_state, 1/2, 1)
        ℒ.mul!(st, 𝐒⁻³, kron_kron_aug_state, 1/6, 1)
        copyto!(st_seq[t+1], st)
    end

    llh = -(logabsdets + shocks² + n_obs_total * log(2 * 3.141592653589793)) / 2

    if !isfinite(llh) || llh < -1e12
        return on_failure_loglikelihood, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # Pre-allocate adjoint buffers outside the pullback closure.
    ∂𝐒_1 = zeros(size(𝐒[1]))
    ∂𝐒_2 = zeros(size(𝐒[2]))
    ∂𝐒_3 = zeros(size(𝐒[3]))
    ∂𝐒⁻¹  = zero(𝐒⁻¹)
    ∂𝐒⁻²  = zero(𝐒⁻²)
    ∂𝐒⁻³  = zero(𝐒⁻³)
    ∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)
    ∂𝐒¹ᵉ  = zero(𝐒¹ᵉ)
    ∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)
    ∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)
    ∂𝐒²ᵉ  = zero(𝐒²ᵉ)
    ∂𝐒³⁻ᵛ = zero(𝐒³⁻ᵛ)
    ∂𝐒³⁻ᵉ² = zero(𝐒³⁻ᵉ²)
    ∂𝐒³⁻ᵉ = zero(𝐒³⁻ᵉ)
    ∂𝐒ⁱ³ᵉ = zero(𝐒ⁱ³ᵉ)
    ∂data_in_deviations = zeros(size(data_in_deviations))
    ∂st_next = zeros(n_past)
    kronaug_buf = zeros((n_past + 1 + n_exo)^2)
    ∂kronaug    = zeros((n_past + 1 + n_exo)^2)
    ∂aug_state  = zeros(n_past + 1 + n_exo)
    ∂kronstate  = zeros((n_past + 1)^2)
    ∂state¹⁻_vol = zeros(n_past + 1)
    ∂𝐒ⁱ_full_buf      = zeros(length(cond_var_idx), n_exo)
    ∂𝐒ⁱ²ᵉ_full_buf    = zeros(length(cond_var_idx), n_exo^2)
    ∂shock_independent = zeros(length(cond_var_idx))
    kron_Isv_buf       = zeros(n_exo * (n_past + 1), n_exo)
    ∂kronIstate_local  = zeros(n_exo * (n_past + 1), n_exo)
    ∂kronaug_for3      = zeros((n_past + 1 + n_exo)^2)
    ∂u_mat             = zeros(n_exo * (n_past + 1), n_exo)
    ∂jac_v_buf         = zeros(length(cond_var_idx), n_exo)

    function third_order_missing_pullback(∂llh)
        fill!(∂𝐒_1, 0); fill!(∂𝐒_2, 0); fill!(∂𝐒_3, 0)
        fill!(∂𝐒⁻¹, 0); fill!(∂𝐒⁻², 0); fill!(∂𝐒⁻³, 0)
        fill!(∂𝐒¹⁻ᵛ, 0); fill!(∂𝐒¹ᵉ, 0)
        fill!(∂𝐒²⁻ᵛ, 0); fill!(∂𝐒²⁻ᵉ, 0); fill!(∂𝐒²ᵉ, 0)
        fill!(∂𝐒³⁻ᵛ, 0); fill!(∂𝐒³⁻ᵉ², 0); fill!(∂𝐒³⁻ᵉ, 0); fill!(∂𝐒ⁱ³ᵉ, 0)
        fill!(∂data_in_deviations, 0)
        fill!(∂st_next, 0)
        fill!(kronaug_buf, 0); fill!(∂kronaug, 0)
        fill!(∂aug_state, 0); fill!(∂kronstate, 0); fill!(∂state¹⁻_vol, 0)
        fill!(∂𝐒ⁱ_full_buf, 0); fill!(∂𝐒ⁱ²ᵉ_full_buf, 0); fill!(∂shock_independent, 0)
        fill!(kron_Isv_buf, 0); fill!(∂kronIstate_local, 0)
        fill!(∂kronaug_for3, 0); fill!(∂u_mat, 0)
        fill!(∂jac_v_buf, 0)

        for t in Tt:-1:1
            aug_state   = aug_state_seq[t]
            state¹⁻_vol = state¹⁻_vol_seq[t]
            x           = x_seq[t]
            idx         = obs_idx_per_t[t]
            m           = length(idx)

            # State recursion: st_next = 𝐒⁻¹ aug + 0.5 𝐒⁻² kron(aug,aug) + (1/6) 𝐒⁻³ kron(aug, kron(aug,aug))
            ℒ.mul!(∂𝐒⁻¹, ∂st_next, aug_state', 1, 1)
            ℒ.mul!(∂aug_state, 𝐒⁻¹', ∂st_next)
            ℒ.kron!(kronaug_buf, aug_state, aug_state)
            ℒ.mul!(∂𝐒⁻², ∂st_next, kronaug_buf', 1/2, 1)
            ∂kronaug2 = (𝐒⁻²' * ∂st_next) ./ 2
            kron_kron_aug = ℒ.kron(kronaug_buf, aug_state)
            ℒ.mul!(∂𝐒⁻³, ∂st_next, kron_kron_aug', 1/6, 1)
            ∂kronkronaug = (𝐒⁻³' * ∂st_next) ./ 6
            fill!(∂kronaug_for3, 0)
            fill_kron_adjoint!(∂aug_state, ∂kronaug_for3, ∂kronkronaug, aug_state, kronaug_buf)
            fill!(∂kronaug, 0)
            ∂kronaug .+= ∂kronaug_for3
            ∂kronaug .+= ∂kronaug2
            fill_kron_adjoint!(∂aug_state, ∂aug_state, ∂kronaug, aug_state, aug_state)

            fill!(∂st_next, 0)
            ∂x = ∂aug_state[n_past+2:end]
            ∂state_now = ∂aug_state[1:n_past]
            @inbounds for j in 1:n_past
                ∂st_next[j] += ∂state_now[j]
            end
            fill!(∂aug_state, 0)

            # shocks² and logabsdet contributions (only if t > presample and m > 0)
            ∂jac_v = view(∂jac_v_buf, 1:m, :); fill!(∂jac_v, 0)
            jac_v_local = zeros(m, n_exo)
            𝐒ⁱ²ᵉ_v_local = zeros(m, n_exo^2)
            𝐒ⁱ³ᵉ_v_local = zeros(m, n_exo^3)
            if m > 0
                𝐒ⁱ_v_local = 𝐒ⁱ_full_seq[t][idx, :]
                𝐒ⁱ²ᵉ_v_local = 𝐒ⁱ²ᵉ_full_seq[t][idx, :]
                𝐒ⁱ³ᵉ_v_local = 𝐒ⁱ³ᵉ[idx, :]
                kron_J_x_local = ℒ.kron(J, x)
                kron_xx_local  = ℒ.kron(x, x)
                kron_J_xx_local = ℒ.kron(J, kron_xx_local)
                jac_v_local = 𝐒ⁱ_v_local + 2 * 𝐒ⁱ²ᵉ_v_local * kron_J_x_local + 3 * 𝐒ⁱ³ᵉ_v_local * kron_J_xx_local
            end
            if m > 0 && t > eff_presample
                @inbounds for k in 1:n_exo
                    ∂x[k] += -x[k]
                end
                if m == n_exo
                    invjac_v = inv(jac_v_local)
                    ∂jac_v .+= (-0.5) .* invjac_v'
                else
                    G = inv(jac_v_local * jac_v_local')
                    ∂jac_v .+= (-0.5) .* (G * jac_v_local)
                end
                # Indirect channel: ∂jac_v → ∂x
                @inbounds for l in 1:n_exo
                    s = 0.0
                    for r in 1:n_exo
                        for i_local in 1:m
                            s += 2 * ∂jac_v[i_local, r] * 𝐒ⁱ²ᵉ_v_local[i_local, (r-1)*n_exo + l]
                        end
                        for q in 1:n_exo
                            col = (r-1)*n_exo^2 + (l-1)*n_exo + q
                            for i_local in 1:m
                                s += 3 * ∂jac_v[i_local, r] * 𝐒ⁱ³ᵉ_v_local[i_local, col] * x[q]
                            end
                        end
                        for p in 1:n_exo
                            col = (r-1)*n_exo^2 + (p-1)*n_exo + l
                            for i_local in 1:m
                                s += 3 * ∂jac_v[i_local, r] * 𝐒ⁱ³ᵉ_v_local[i_local, col] * x[p]
                            end
                        end
                    end
                    ∂x[l] += s
                end
            end

            fill!(∂shock_independent, 0)

            if m > 0
                𝐒ⁱ_v   = 𝐒ⁱ_full_seq[t][idx, :]
                𝐒ⁱ²ᵉ_v = 𝐒ⁱ²ᵉ_v_local
                𝐒ⁱ³ᵉ_v = 𝐒ⁱ³ᵉ_v_local
                jac_v  = jac_v_local

                local λ
                if m == n_exo
                    λ = 2 * (jac_v' \ x)
                else
                    Gloc = inv(jac_v * jac_v')
                    λ = 2 * (Gloc * (jac_v * x))
                end

                kron_II_x = ℒ.kron(II, x)
                M = reshape((𝐒ⁱ²ᵉ_v + 3 * 𝐒ⁱ³ᵉ_v * kron_II_x)' * λ, n_exo, n_exo)
                topL = 2 * ℒ.I(n_exo) - 2 * M
                fXλp = [topL          -jac_v'
                        jac_v          zeros(m, m)]

                rhs = vcat(∂x, zeros(m))
                S = fXλp' \ rhs
                Sx = S[1:n_exo]
                Sλ = S[n_exo+1:end]

                ∂v_v = Sλ
                ∂𝐒ⁱ_v = λ * Sx' - Sλ * x'

                xSx = x * Sx'
                xx_outer = x * x'
                ∂𝐒ⁱ²ᵉ_v_kkt = 2 * λ * vec(xSx)' - Sλ * vec(xx_outer)'

                kron_Sx_xx = ℒ.kron(Sx, kron_xx_local)
                kron_x_xx  = ℒ.kron(x,  kron_xx_local)
                ∂𝐒ⁱ³ᵉ_v_kkt = 3 * λ * kron_Sx_xx' - Sλ * kron_x_xx'

                if t > eff_presample
                    ∂𝐒ⁱ_v_total    = ∂𝐒ⁱ_v + ∂jac_v
                    ∂𝐒ⁱ²ᵉ_v_total  = ∂𝐒ⁱ²ᵉ_v_kkt + 2 * ∂jac_v * ℒ.kron(J, x)'
                    ∂𝐒ⁱ³ᵉ_v_total  = ∂𝐒ⁱ³ᵉ_v_kkt + 3 * ∂jac_v * ℒ.kron(J, kron_xx_local)'
                else
                    ∂𝐒ⁱ_v_total    = ∂𝐒ⁱ_v
                    ∂𝐒ⁱ²ᵉ_v_total  = ∂𝐒ⁱ²ᵉ_v_kkt
                    ∂𝐒ⁱ³ᵉ_v_total  = ∂𝐒ⁱ³ᵉ_v_kkt
                end

                ∂𝐒ⁱ_full = ∂𝐒ⁱ_full_buf
                ∂𝐒ⁱ²ᵉ_full = ∂𝐒ⁱ²ᵉ_full_buf
                fill!(∂𝐒ⁱ_full, 0)
                fill!(∂𝐒ⁱ²ᵉ_full, 0)
                @inbounds for j in 1:n_exo
                    for i_local in 1:m
                        ∂𝐒ⁱ_full[idx[i_local], j] = ∂𝐒ⁱ_v_total[i_local, j]
                    end
                end
                @inbounds for j in 1:n_exo^2
                    for i_local in 1:m
                        ∂𝐒ⁱ²ᵉ_full[idx[i_local], j] = ∂𝐒ⁱ²ᵉ_v_total[i_local, j]
                    end
                end
                @inbounds for j in 1:n_exo^3
                    for i_local in 1:m
                        ∂𝐒ⁱ³ᵉ[idx[i_local], j] += ∂𝐒ⁱ³ᵉ_v_total[i_local, j]
                    end
                end

                @inbounds for i_local in 1:m
                    ∂shock_independent[idx[i_local]] += ∂v_v[i_local]
                end

                # Propagate ∂𝐒ⁱ_full back: 𝐒ⁱ_full = 𝐒¹ᵉ + 𝐒²⁻ᵉ k(I,s¹v) + 0.5 𝐒³⁻ᵉ² k(k(I,s¹v),s¹v)
                ∂𝐒¹ᵉ .+= ∂𝐒ⁱ_full
                kron_J_s1v = ℒ.kron(J, state¹⁻_vol)
                ℒ.mul!(∂𝐒²⁻ᵉ, ∂𝐒ⁱ_full, kron_J_s1v', 1, 1)
                kron_kron_J_s1v_s1v = ℒ.kron(kron_J_s1v, state¹⁻_vol)
                ℒ.mul!(∂𝐒³⁻ᵉ², ∂𝐒ⁱ_full, kron_kron_J_s1v_s1v', 1/2, 1)

                ∂kronIs1v_a = 𝐒²⁻ᵉ' * ∂𝐒ⁱ_full
                fill!(∂state¹⁻_vol, 0)
                @inbounds for p in 1:(n_past + 1)
                    s = 0.0
                    for j in 1:n_exo
                        s += ∂kronIs1v_a[(j-1)*(n_past+1) + p, j]
                    end
                    ∂state¹⁻_vol[p] += s
                end
                ∂kron_u_s1v = (𝐒³⁻ᵉ²' * ∂𝐒ⁱ_full) ./ 2
                u_mat = ℒ.kron(J, state¹⁻_vol)
                fill!(∂u_mat, 0)
                @inbounds for j in 1:n_exo
                    for q in 1:(n_exo*(n_past+1))
                        for r in 1:(n_past+1)
                            ∂u_mat[q, j] += ∂kron_u_s1v[(q-1)*(n_past+1) + r, j] * state¹⁻_vol[r]
                            ∂state¹⁻_vol[r] += ∂kron_u_s1v[(q-1)*(n_past+1) + r, j] * u_mat[q, j]
                        end
                    end
                end
                @inbounds for p in 1:(n_past + 1)
                    s = 0.0
                    for j in 1:n_exo
                        s += ∂u_mat[(j-1)*(n_past+1) + p, j]
                    end
                    ∂state¹⁻_vol[p] += s
                end

                # Propagate ∂𝐒ⁱ²ᵉ_full back: 𝐒ⁱ²ᵉ_full = 𝐒²ᵉ/2 + 𝐒³⁻ᵉ k(II, s¹v)/2
                ∂𝐒²ᵉ .+= ∂𝐒ⁱ²ᵉ_full ./ 2
                kron_II_s1v = ℒ.kron(II, state¹⁻_vol)
                ℒ.mul!(∂𝐒³⁻ᵉ, ∂𝐒ⁱ²ᵉ_full, kron_II_s1v', 1/2, 1)
                ∂kronIIs1v = (𝐒³⁻ᵉ' * ∂𝐒ⁱ²ᵉ_full) ./ 2
                @inbounds for p in 1:(n_past + 1)
                    s = 0.0
                    for j in 1:n_exo^2
                        s += ∂kronIIs1v[(j-1)*(n_past+1) + p, j]
                    end
                    ∂state¹⁻_vol[p] += s
                end
            else
                fill!(∂state¹⁻_vol, 0)
            end

            # Propagate shock_independent dependencies
            @inbounds for i in 1:n_cond
                ∂data_in_deviations[i, t] += ∂shock_independent[i]
            end
            ℒ.mul!(∂𝐒¹⁻ᵛ, ∂shock_independent, state¹⁻_vol', -1, 1)
            ℒ.mul!(∂state¹⁻_vol, 𝐒¹⁻ᵛ', ∂shock_independent, -1, 1)
            kron_sv = ℒ.kron(state¹⁻_vol, state¹⁻_vol)
            ℒ.mul!(∂𝐒²⁻ᵛ, ∂shock_independent, kron_sv', -1/2, 1)
            ∂kron_sv = -(𝐒²⁻ᵛ' * ∂shock_independent) ./ 2
            fill!(∂kronstate, 0)
            ∂kronstate .+= ∂kron_sv
            kron_s1v_3 = ℒ.kron(state¹⁻_vol, kron_sv)
            ℒ.mul!(∂𝐒³⁻ᵛ, ∂shock_independent, kron_s1v_3', -1/6, 1)
            ∂kron_s1v_3 = -(𝐒³⁻ᵛ' * ∂shock_independent) ./ 6
            ∂a_outer = zeros(n_past + 1)
            ∂b_outer = zeros((n_past + 1)^2)
            fill_kron_adjoint!(∂a_outer, ∂b_outer, ∂kron_s1v_3, state¹⁻_vol, kron_sv)
            ∂state¹⁻_vol .+= ∂a_outer
            ∂kronstate   .+= ∂b_outer
            fill_kron_adjoint!(∂state¹⁻_vol, ∂state¹⁻_vol, ∂kronstate, state¹⁻_vol, state¹⁻_vol)

            # state¹⁻_vol = vcat(st, 1) → ∂st_next += ∂state¹⁻_vol[1:n_past]
            @inbounds for j in 1:n_past
                ∂st_next[j] += ∂state¹⁻_vol[j]
            end
        end

        ∂𝐒_1[Tcc.past_not_future_and_mixed_idx, :]                 .+= ∂𝐒⁻¹
        ∂𝐒_2[Tcc.past_not_future_and_mixed_idx, :]                 .+= ∂𝐒⁻²
        ∂𝐒_3[Tcc.past_not_future_and_mixed_idx, :]                 .+= ∂𝐒⁻³
        ∂𝐒_1[cond_var_idx, 1:n_past+1]                              .+= ∂𝐒¹⁻ᵛ
        ∂𝐒_1[cond_var_idx, end-n_exo+1:end]                         .+= ∂𝐒¹ᵉ
        ∂𝐒_2[cond_var_idx, var_vol²_idxs]                           .+= ∂𝐒²⁻ᵛ
        ∂𝐒_2[cond_var_idx, shockvar²_idxs]                          .+= ∂𝐒²⁻ᵉ
        ∂𝐒_2[cond_var_idx, shock²_idxs]                             .+= ∂𝐒²ᵉ
        ∂𝐒_3[cond_var_idx, var_vol³_idxs]                           .+= ∂𝐒³⁻ᵛ
        ∂𝐒_3[cond_var_idx, shockvar³2_idxs]                         .+= ∂𝐒³⁻ᵉ²
        ∂𝐒_3[cond_var_idx, shockvar³_idxs]                          .+= ∂𝐒³⁻ᵉ
        ∂𝐒_3[cond_var_idx, shock³_idxs]                             .+= ∂𝐒ⁱ³ᵉ ./ 6

        ℒ.rmul!(∂𝐒_1, ∂llh)
        ℒ.rmul!(∂𝐒_2, ∂llh)
        ℒ.rmul!(∂𝐒_3, ∂llh)
        ℒ.rmul!(∂data_in_deviations, ∂llh)

        ∂state₀_full = zeros(size(state))
        ∂state₀_full[Tcc.past_not_future_and_mixed_idx] .= ∂st_next .* ∂llh

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), [∂𝐒_1, ∂𝐒_2, ∂𝐒_3], ∂data_in_deviations, NoTangent(), ∂state₀_full, NoTangent()
    end

    return llh, third_order_missing_pullback
end


function rrule(::typeof(calculate_loglikelihood),
                ::Val{:inversion},
                ::Val{:third_order},
                observables_index::Vector{Int},
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                constants::constants,
                state::Vector{Float64}, 
                workspaces::workspaces; 
                # timer::TimerOutput = TimerOutput(),
                on_failure_loglikelihood = -Inf,
                warmup_iterations::Int = 0,
                presample_periods::Int = 0,
                initial_covariance::Symbol = :theoretical,
                opts::CalculationOptions = merge_calculation_options(),
                filter_algorithm::Symbol = :LagrangeNewton)
    T = constants.post_model_macro
    ws = workspaces.inversion

    # @timeit_debug timer "Inversion filter pruned 2nd - forward" begin
    # @timeit_debug timer "Preallocation" begin

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = observables_index

    shocks² = 0.0
    logabsdets = 0.0

    cc = ensure_conditional_forecast_constants!(constants; third_order = true)
    tc = constants.third_order
    shock_idxs = cc.shock_idxs
    shock²_idxs = cc.shock²_idxs
    shockvar²_idxs = cc.shockvar²_idxs
    var_vol²_idxs = cc.var_vol²_idxs
    var²_idxs = cc.var²_idxs
    var_vol³_idxs = tc.var_vol³_idxs
    shock_idxs2 = tc.shock_idxs2
    shock_idxs3 = tc.shock_idxs3
    shock³_idxs = tc.shock³_idxs
    shockvar1_idxs = tc.shockvar1_idxs
    shockvar2_idxs = tc.shockvar2_idxs
    shockvar3_idxs = tc.shockvar3_idxs
    shockvar³2_idxs = tc.shockvar³2_idxs
    shockvar³_idxs = tc.shockvar³_idxs
    
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
            return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
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
            if T.nExo == length(observables_index)
                logabsdets += ℒ.logabsdet(jacc[i])[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i]))
            end
    
            shocks² += sum(abs2,x[i])
            
            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
            end
        end
    
        aug_state[i] = [stt; 1; x[i]]
    
        stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6
    end
    
    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    llh = -(logabsdets + shocks² + (length(observables_index) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2

    # end # timeit_debug
    # end # timeit_debug


    ∂𝐒 = [zero(𝐒[1]), zero(𝐒[2]), zero(𝐒[3])]

    ∂data_in_deviations = similar(data_in_deviations)

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

    # Pre-allocated per-period buffers (formerly created fresh each iteration).
    ∂jacc_buf    = zero(jacc[1])
    ∂xλ_buf      = zeros(T.nExo + size(jacc[1], 1))
    S_buf        = zeros(T.nExo + size(jacc[1], 1))
    kronSλ       = zeros(length(cond_var_idx) * T.nExo)
    kronxS       = zeros(T.nExo * length(cond_var_idx))
    kron_S1_kxλ  = zeros(T.nExo * length(kronxλ[1]))
    kron_xx_S2   = zeros(length(kronxx[1]) * size(jacc[1], 1))
    kron_S1_kxxλ = zeros(T.nExo * length(kronxxλ[1]))
    kron_xxx_S2  = zeros(length(kronxxx[1]) * size(jacc[1], 1))
    kron_xλ      = zero(kronxλ[1])
    kron_xxλ     = zero(kronxxλ[1])
    kron_Ix      = zero(ℒ.kron(ℒ.I(T.nExo), x[1]))
    kron_Ixx     = zero(ℒ.kron(ℒ.I(T.nExo), kronxx[1]))
    ∂𝐒ⁱ²ᵉ_tmp    = zero(𝐒ⁱ²ᵉ[1])

    function inversion_filter_loglikelihood_pullback(∂llh)
        # @timeit_debug timer "Inversion filter pruned 2nd - pullback" begin
        # @timeit_debug timer "Preallocation" begin

        fill!(∂𝐒ⁱ, 0)
        fill!(∂𝐒²ᵉ, 0)
        fill!(∂𝐒ⁱ³ᵉ, 0)

        fill!(∂𝐒¹ᵉ, 0)
        fill!(∂𝐒²⁻ᵉ, 0)
        fill!(∂𝐒³⁻ᵉ, 0)
        fill!(∂𝐒³⁻ᵉ², 0)

        fill!(∂𝐒¹⁻ᵛ, 0)
        fill!(∂𝐒²⁻ᵛ, 0)
        fill!(∂𝐒³⁻ᵛ, 0)
        
        fill!(∂𝐒⁻¹, 0)
        fill!(∂𝐒⁻², 0)
        fill!(∂𝐒⁻³, 0)

        fill!(∂state¹⁻_vol, 0)
        fill!(∂x, 0)
        fill!(∂kronxx, 0)
        fill!(∂kronstate¹⁻_vol, 0)
        fill!(∂state, 0)

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

            if i < size(data_in_deviations,2)
                fill!(∂state, 0)
            end

            # aug_state[i] = [stt; 1; x[i]]
            @views ℒ.axpy!(1, ∂aug_state[1:length(∂state)], ∂state)

            # aug_state[i] = [stt; 1; x[i]]
            @views copyto!(∂x, ∂aug_state[T.nPast_not_future_and_mixed+2:end])

            # shocks² += sum(abs2,x[i]) — only for i > presample_periods
            if i > presample_periods
                if i < size(data_in_deviations,2)
                    @inbounds @simd for k in eachindex(∂x)
                        ∂x[k] -= x[i][k]
                    end
                else
                    @inbounds @simd for k in eachindex(∂x)
                        ∂x[k] += x[i][k]
                    end
                end
            end

            # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1] — only for i > presample_periods
            if i > presample_periods
                if size(jacc[i], 1) == size(jacc[i], 2)
                    jacc_lu = ℒ.lu(jacc[i], check = false)
                    if !ℒ.issuccess(jacc_lu)
                        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end
                    copyto!(∂jacc_buf, inv(jacc_lu)')
                    ∂jacc = ∂jacc_buf
                else
                    ∂jacc = ℒ.pinv(jacc[i])'
                    if !all(isfinite, ∂jacc)
                        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end
                end
            else
                fill!(∂jacc_buf, 0)
                ∂jacc = ∂jacc_buf
            end

            # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x, x))
            # ∂𝐒ⁱ = -∂jacc / 2 # fine

            ℒ.mul!(kron_Ix, 𝐒ⁱ²ᵉ[i]', ∂jacc)
            ∂kronIx = kron_Ix

            if i < size(data_in_deviations,2)
                fill_kron_adjoint_∂B!(∂kronIx, ∂x, -ℒ.I(T.nExo))
            else
                fill_kron_adjoint_∂B!(∂kronIx, ∂x, ℒ.I(T.nExo))
            end

            ℒ.kron!(kron_Ix, ℒ.I(T.nExo), x[i])
            ℒ.mul!(∂𝐒ⁱ²ᵉ_tmp, ∂jacc, kron_Ix', -1, 0)

            ℒ.mul!(kron_Ixx, 𝐒ⁱ³ᵉ', ∂jacc, 3/2, 0)
            ∂kronIxx = kron_Ixx
            
            fill!(∂kronxx, 0)

            if i < size(data_in_deviations,2)
                fill_kron_adjoint_∂B!(∂kronIxx, ∂kronxx, -ℒ.I(T.nExo))
            else
                fill_kron_adjoint_∂B!(∂kronIxx, ∂kronxx, ℒ.I(T.nExo))
            end

            fill_kron_adjoint!(∂x, ∂x, ∂kronxx, x[i], x[i])

            ℒ.kron!(kron_Ixx, ℒ.I(T.nExo), kronxx[i])
            ℒ.mul!(∂𝐒ⁱ³ᵉ, ∂jacc, kron_Ixx', -3/2, 1)

            # find_shocks
            # ∂xλ = vcat(∂x, zero(λ[i]))
            copyto!(∂xλ_buf, 1, ∂x, 1, length(∂x))
            fill!(view(∂xλ_buf, length(∂x)+1:length(∂xλ_buf)), 0)

            S_solved = fXλp[i]' \ ∂xλ_buf
            copyto!(S_buf, S_solved)
            S = S_buf

            if i < size(data_in_deviations,2)
                ℒ.rmul!(S, -1)
            end

            S1 = view(S, 1:T.nExo)
            S2 = view(S, T.nExo+1:length(S))
            ∂shock_independent = S2

            # ∂𝐒ⁱ = ℒ.kron(S[1:T.nExo], λ[i]) - ℒ.kron(x[i], S[T.nExo+1:end])
            ℒ.kron!(kronSλ, S1, λ[i])
            ℒ.kron!(kronxS, x[i], S2)
            ℒ.axpy!(-1, kronxS, kronSλ)
            copyto!(∂𝐒ⁱ, kronSλ)
            ℒ.axpy!(-1/2, ∂jacc, ∂𝐒ⁱ)

            # ∂𝐒ⁱ²ᵉ += reshape(2 * ℒ.kron(S[1:T.nExo], ℒ.kron(x[i], λ[i])) - ℒ.kron(kronxx[i], S[T.nExo+1:end]), size(∂𝐒ⁱ²ᵉ))
            ℒ.kron!(kron_xλ, x[i], λ[i])
            ℒ.kron!(kron_S1_kxλ, S1, kron_xλ)
            ℒ.kron!(kron_xx_S2, kronxx[i], S2)
            ℒ.axpby!(-1, kron_xx_S2, 2, kron_S1_kxλ)
            ∂𝐒ⁱ²ᵉ_tmp .+= reshape(kron_S1_kxλ, size(∂𝐒ⁱ²ᵉ_tmp))
            ∂𝐒ⁱ²ᵉ = ∂𝐒ⁱ²ᵉ_tmp

            # ∂𝐒ⁱ³ᵉ += reshape(3 * ℒ.kron(S[1:T.nExo], ℒ.kron(ℒ.kron(x[i], x[i]), λ[i])) - ℒ.kron(kronxxx[i], S[T.nExo+1:end]), size(∂𝐒ⁱ³ᵉ))
            ℒ.kron!(kron_xxλ, kronxx[i], λ[i])
            ℒ.kron!(kron_S1_kxxλ, S1, kron_xxλ)
            ℒ.kron!(kron_xxx_S2, kronxxx[i], S2)
            ℒ.axpby!(-1, kron_xxx_S2, 3, kron_S1_kxxλ)
            ∂𝐒ⁱ³ᵉ .+= reshape(kron_S1_kxxλ, size(∂𝐒ⁱ³ᵉ))

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
            @inbounds @simd for k in eachindex(∂shock_independent); ∂data_in_deviations[k, i] = ∂shock_independent[k]; end


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

        fill!(∂𝐒[1], 0)
        fill!(∂𝐒[2], 0)
        fill!(∂𝐒[3], 0)

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

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂𝐒, ∂data_in_deviations * ∂llh, NoTangent(), ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state * ∂llh, NoTangent()
    end

    # end # timeit_debug
    # end # timeit_debug

    return llh, inversion_filter_loglikelihood_pullback
end

function rrule(::typeof(calculate_loglikelihood),
                ::Val{:kalman},
                ::Val,
                observables_index::Vector{Int},
                𝐒::AbstractMatrix{Float64},
                data_in_deviations::Matrix{Float64},
                constants::constants,
                state,
                workspaces::workspaces;
                warmup_iterations::Int = 0,
                presample_periods::Int = 0,
                initial_covariance::Symbol = :theoretical,
                filter_algorithm::Symbol = :LagrangeNewton,
                lyapunov_algorithm::Symbol = :doubling,
                on_failure_loglikelihood::U = -Inf,
                opts::CalculationOptions = merge_calculation_options()) where {U <: AbstractFloat}
                
    T = constants.post_model_macro
    idx_constants = constants.post_complete_parameters
    lyap_ws = ensure_lyapunov_workspace!(workspaces, T.nVars, :first_order)
    observables_and_states = sort(union(T.past_not_future_and_mixed_idx, observables_index))
    observables_sorted = sort(observables_index)
    I_nVars = idx_constants.diag_nVars

    A_map = @views I_nVars[T.past_not_future_and_mixed_idx, observables_and_states]

    A = @views 𝐒[observables_and_states,1:T.nPast_not_future_and_mixed] * A_map
    B = @views 𝐒[observables_and_states,T.nPast_not_future_and_mixed+1:end]

    C = @views I_nVars[observables_sorted, observables_and_states]

    kalman_ws = ensure_kalman_workspaces!(workspaces, size(C, 1), size(C, 2))
    𝐁 = kalman_ws.𝐁
    ℒ.mul!(𝐁, B, B')

    lyap_pullback = nothing
    lyap_solved = true
    P = if initial_covariance == :theoretical
        lyap_rrule_result, lyap_pullback_local = rrule(solve_lyapunov_equation,
                                                        A,
                                                        𝐁,
                                                        lyap_ws,
                                                        lyapunov_algorithm = opts.lyapunov_algorithm,
                                                        tol = opts.tol.first_order.ad.lyapunov,
                                                        verbose = opts.verbose)
        lyap_pullback = lyap_pullback_local
        lyap_solved = lyap_rrule_result[2]
        lyap_rrule_result[1]
    else
        get_initial_covariance(Val(initial_covariance), A, 𝐁, lyap_ws, opts = opts)
    end

    if !lyap_solved
        if opts.verbose println("KF initial Lyapunov solve failed") end
        return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    Tt = size(data_in_deviations, 2) + 1

    z = zeros(size(data_in_deviations, 1))
    ū = zeros(size(C,2))
    P̄ = deepcopy(P)

    temp_N_N = similar(P)
    PCtmp = similar(P, size(P, 1), size(C, 1))
    F = similar(P, size(C, 1), size(C, 1))

    # Per-period sequence buffers (cached in workspace; see ensure_kalman_rrule_buffers!)
    ensure_kalman_rrule_buffers!(kalman_ws, size(C, 1), size(C, 2), Tt)
    u     = kalman_ws.u_seq_rrule
    P_seq = kalman_ws.P_seq_rrule
    CP    = kalman_ws.CP_seq_rrule
    K     = kalman_ws.K_seq_rrule
    invF  = kalman_ws.invF_seq_rrule
    v     = kalman_ws.v_seq_rrule
    @inbounds for t in 1:Tt
        copyto!(P_seq[t], P̄)
    end

    # Missing-value support: per-period observable indices.  When all entries
    # of data_in_deviations are finite, obs_idx_per_t[t] == 1:n_obs_full and
    # the m == n_obs_full branch reduces to the original dense fast path
    # (zero overhead).  When missing values are present, the forward step
    # computes the reduced (m × m) innovation/F/K/invF on sub-views, then
    # scatters them back into the full-size buffers with zeros outside the
    # observed rows/columns.  Because every product in the analytical
    # pullback that touches a "missing" row of v[t]/invF[t]/K[t]/CP[t] then
    # sees a zero, the existing pullback math is correct unchanged.
    obs_idx_per_t, has_missing = build_obs_index(data_in_deviations)
    n_obs_full = size(C, 1)
    n_obs_total = 0  # observed scalars contributing to the loglik normaliser

    loglik = 0.0

    for t in 2:Tt
        if !all(isfinite.(z))
            if opts.verbose println("KF not finite at step $t") end
            return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end

        idx = obs_idx_per_t[t-1]
        m   = length(idx)

        # Always zero the full-size storage so missing rows/columns stay zero.
        fill!(v[t], 0.0)
        fill!(K[t], 0.0)
        fill!(CP[t], 0.0)
        fill!(invF[t], 0.0)

        if m == 0
            # Pure predict step: no innovation, no Kalman gain.
            copyto!(P_seq[t], P̄)            # used by pullback at step t+1
            copyto!(u[t], ū)                 # u[t] = u_predict (no update)

            ℒ.mul!(ū, A, u[t])
            ℒ.mul!(z, C, ū)

            ℒ.mul!(temp_N_N, P_seq[t], A')
            ℒ.mul!(P̄, A, temp_N_N)
            P̄ .+= 𝐁
            continue
        end

        if m == n_obs_full
            # Original dense fast path.
            v[t] .= data_in_deviations[:, t-1] .- z

            ℒ.mul!(CP[t], C, P̄)
            ℒ.mul!(F, CP[t], C')

            kalman_ws.fast_lu_ws_f, kalman_ws.fast_lu_dims_f, solved_F, luF = factorize_lu!(Val(:FastLapack), F,
                                                                                                kalman_ws.fast_lu_ws_f,
                                                                                                kalman_ws.fast_lu_dims_f)

            if !solved_F
                if opts.verbose println("KF factorisation failed step $t") end
                return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end

            logabsdetF = 0.0
            signF = isodd(count(i -> kalman_ws.fast_lu_ws_f.ipiv[i] != i, eachindex(kalman_ws.fast_lu_ws_f.ipiv))) ? -1.0 : 1.0
            @inbounds for i in 1:size(F, 1)
                di = F[i, i]
                if di == 0
                    if opts.verbose println("KF factorisation failed step $t") end
                    return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
                end
                logabsdetF += log(abs(di))
                signF *= sign(di)
            end

            if signF <= 0 || logabsdetF < log(eps(Float64))
                if opts.verbose println("KF factorisation failed step $t") end
                return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end

            @inbounds for i in 1:size(invF[t], 1)
                invF[t][i, i] = 1.0
            end
            solve_lu_left!(F, invF[t], kalman_ws.fast_lu_ws_f, luF)

            if t - 1 > presample_periods
                loglik += logabsdetF + ℒ.dot(v[t], invF[t], v[t])
                n_obs_total += m
            end

            ℒ.mul!(PCtmp, P̄, C')
            copyto!(K[t], PCtmp)
            solve_lu_right!(F, K[t], kalman_ws.fast_lu_ws_f, luF, kalman_ws.fast_lu_rhs_t_k)

            ℒ.mul!(P_seq[t], K[t], CP[t], -1, 0)
            P_seq[t] .+= P̄

            ℒ.mul!(temp_N_N, P_seq[t], A')
            ℒ.mul!(P̄, A, temp_N_N)
            P̄ .+= 𝐁

            ℒ.mul!(u[t], K[t], v[t])
            u[t] .+= ū

            ℒ.mul!(ū, A, u[t])
            ℒ.mul!(z, C, ū)
        else
            # Partial-missing step: compute reduced m × m quantities on
            # sub-views, then scatter back into full-size storage.
            Cv  = view(C, idx, :)                    # m × n_state
            Fv  = view(F, 1:m, 1:m)                  # m × m
            CPv = view(CP[t], idx, :)                # m × n_state (target)

            ℒ.mul!(CPv, Cv, P̄)
            ℒ.mul!(Fv, CPv, Cv')

            kalman_ws.fast_lu_ws_f, kalman_ws.fast_lu_dims_f, solved_F, luF = factorize_lu!(Val(:Julia), Fv,
                                                                                                kalman_ws.fast_lu_ws_f,
                                                                                                kalman_ws.fast_lu_dims_f)

            if !solved_F
                if opts.verbose println("KF factorisation failed step $t") end
                return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end

            logabsdetF = 0.0
            signF = isodd(count(i -> luF.ipiv[i] != i, 1:m)) ? -1.0 : 1.0
            @inbounds for i in 1:m
                di = Fv[i, i]
                if di == 0
                    if opts.verbose println("KF factorisation failed step $t") end
                    return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
                end
                logabsdetF += log(abs(di))
                signF *= sign(di)
            end

            if signF <= 0 || logabsdetF < log(eps(Float64))
                if opts.verbose println("KF factorisation failed step $t") end
                return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
            end

            # Solve invF_m = Fv \ I, then scatter into invF[t][idx, idx].
            invF_scratch = view(temp_N_N, 1:m, 1:m)
            fill!(invF_scratch, 0.0)
            @inbounds for i in 1:m
                invF_scratch[i, i] = 1.0
            end
            solve_lu_left!(Fv, invF_scratch, kalman_ws.fast_lu_ws_f, luF; use_fastlapack_lu = false)
            @inbounds for j in 1:m, i in 1:m
                invF[t][idx[i], idx[j]] = invF_scratch[i, j]
            end

            # innovation v_m = data[idx, t-1] - C[idx,:] * u_predict
            vv = view(v[t], idx)
            @inbounds for i in 1:m
                acc = 0.0
                for k in 1:size(C, 2)
                    acc += C[idx[i], k] * ū[k]
                end
                vv[i] = data_in_deviations[idx[i], t-1] - acc
            end

            if t - 1 > presample_periods
                loglik += logabsdetF + ℒ.dot(vv, invF_scratch, vv)
                n_obs_total += m
            end

            # K_m = P̄ * C[idx,:]' / Fv  →  scatter into K[t][:, idx]
            Kv_scratch = view(PCtmp, :, 1:m)
            ℒ.mul!(Kv_scratch, P̄, Cv')
            rhs_t_kv = view(kalman_ws.fast_lu_rhs_t_k, 1:m, :)
            solve_lu_right!(Fv, Kv_scratch, kalman_ws.fast_lu_ws_f, luF, rhs_t_kv; use_fastlapack_lu = false)
            @inbounds for j in 1:m, i in 1:size(K[t], 1)
                K[t][i, idx[j]] = Kv_scratch[i, j]
            end

            # P_seq[t] = P̄ - K[t] * CP[t]   (zero outside idx in K and CP).
            ℒ.mul!(P_seq[t], K[t], CP[t], -1, 0)
            P_seq[t] .+= P̄

            ℒ.mul!(temp_N_N, P_seq[t], A')
            ℒ.mul!(P̄, A, temp_N_N)
            P̄ .+= 𝐁

            ℒ.mul!(u[t], K[t], v[t])
            u[t] .+= ū

            ℒ.mul!(ū, A, u[t])
            ℒ.mul!(z, C, ū)
        end
    end

    llh = -(loglik + (has_missing ?
                        n_obs_total :
                        ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1))) *
            log(2 * 3.141592653589793)) / 2

    ∂F = zero(F)
    ∂Faccum = zero(F)
    ∂P = zero(P̄)
    ∂ū = zero(ū)
    ∂v = zero(v[1])
    ∂data_in_deviations = zero(data_in_deviations)
    vtmp = zero(v[1])
    Ptmp = zero(P_seq[1])
    ∂A_kf = zero(A)
    ∂𝐁_kf = zero(𝐁)
    ∂A_buf = zero(A)
    ∂𝐁_buf = zero(𝐁)

    function calculate_loglikelihood_pullback(∂llh)
        ℒ.rmul!(∂A_kf, 0)
        ℒ.rmul!(∂Faccum, 0)
        ℒ.rmul!(∂P, 0)
        ℒ.rmul!(∂ū, 0)
        ℒ.rmul!(∂𝐁_kf, 0)

        for t in Tt:-1:2
            if t > presample_periods + 1
                ℒ.mul!(∂F, v[t], v[t]')
                ℒ.mul!(invF[1], invF[t]', ∂F)
                ℒ.mul!(∂F, invF[1], invF[t]')
                ℒ.axpby!(1, invF[t]', -1, ∂F)

                copy!(invF[1], invF[t]' .+ invF[t])
                ℒ.mul!(∂v, invF[1], v[t])
            else
                ℒ.rmul!(∂F, 0)
                ℒ.rmul!(∂v, 0)
            end

            ℒ.axpy!(1, ∂Faccum, ∂F)
            ℒ.mul!(PCtmp, C', ∂F)
            ℒ.mul!(∂P, PCtmp, C, 1, 1)

            ℒ.mul!(CP[1], invF[t]', C)
            ℒ.mul!(PCtmp, ∂ū, v[t]')
            ℒ.mul!(P_seq[1], PCtmp, CP[1])
            ℒ.mul!(∂P, A', P_seq[1], 1, 1)

            ℒ.mul!(u[1], A', ∂ū)
            ℒ.mul!(v[1], K[t]', u[1])
            ℒ.axpy!(1, ∂v, v[1])
            ∂data_in_deviations[:,t-1] .= v[1]

            ℒ.mul!(u[1], A', ∂ū)
            ℒ.mul!(v[1], K[t]', u[1])
            ℒ.mul!(∂ū, C', v[1])
            ℒ.mul!(u[1], C', v[1], -1, 1)
            copy!(∂ū, u[1])

            ℒ.mul!(u[1], C', ∂v)
            ℒ.axpy!(-1, u[1], ∂ū)

            if t > 2
                ℒ.mul!(∂A_kf, ∂ū, u[t-1]', 1, 1)

                ℒ.mul!(P_seq[1], A, P_seq[t-1]')
                ℒ.mul!(Ptmp, ∂P, P_seq[1])
                ℒ.mul!(P_seq[1], A, P_seq[t-1])
                ℒ.mul!(Ptmp, ∂P', P_seq[1], 1, 1)
                ℒ.axpy!(1, Ptmp, ∂A_kf)

                ℒ.axpy!(1, ∂P, ∂𝐁_kf)

                ℒ.mul!(P_seq[1], ∂P, A)
                ℒ.mul!(∂P, A', P_seq[1])

                ℒ.mul!(PCtmp, ∂P, K[t-1])
                ℒ.mul!(CP[1], K[t-1]', ∂P)
                ℒ.mul!(∂P, PCtmp, C, -1, 1)
                ℒ.mul!(∂P, C', CP[1], -1, 1)

                ℒ.mul!(u[1], A', ∂ū)
                ℒ.mul!(v[1], CP[t-1], u[1])
                ℒ.mul!(vtmp, invF[t-1]', v[1], -1, 0)
                ℒ.mul!(invF[1], vtmp, v[t-1]')
                ℒ.mul!(∂Faccum, invF[1], invF[t-1]')

                ℒ.mul!(CP[1], invF[t-1]', CP[t-1])
                ℒ.mul!(PCtmp, CP[t-1]', invF[t-1]')
                ℒ.mul!(K[1], ∂P, PCtmp)
                ℒ.mul!(∂Faccum, CP[1], K[1], -1, 1)
            end
        end

        ℒ.rmul!(∂P, -∂llh/2)
        ℒ.rmul!(∂A_kf, -∂llh/2)
        ℒ.rmul!(∂𝐁_kf, -∂llh/2)
        ℒ.rmul!(∂data_in_deviations, -∂llh/2)

        ∂A = ∂A_buf; copyto!(∂A, ∂A_kf)
        ∂𝐁 = ∂𝐁_buf; copyto!(∂𝐁, ∂𝐁_kf)

        if !isnothing(lyap_pullback)
            lyap_grads = lyap_pullback((∂P, NoTangent()))
            if !(lyap_grads[2] isa AbstractZero)
                ℒ.axpy!(1, lyap_grads[2], ∂A)
            end
            if !(lyap_grads[3] isa AbstractZero)
                ℒ.axpy!(1, lyap_grads[3], ∂𝐁)
            end
        end

        ∂B = (∂𝐁 + ∂𝐁') * B

        ∂𝐒 = zero(𝐒)
        @views ∂𝐒[observables_and_states, 1:T.nPast_not_future_and_mixed] .+= ∂A * A_map'
        @views ∂𝐒[observables_and_states, T.nPast_not_future_and_mixed+1:end] .+= ∂B

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂𝐒, ∂data_in_deviations, NoTangent(), NoTangent(), NoTangent()
    end

    return llh, calculate_loglikelihood_pullback
end


# Kalman missing-data path delegates to the dense Kalman rrule because the
# dense rrule already slices per-period via `build_obs_index` internally and
# yields a correct gradient for missing observations. The pullback gains one
# extra NoTangent slot for the trailing `obs_idx_per_t` argument.
function rrule(::typeof(calculate_loglikelihood_with_missing),
                ::Val{:kalman},
                val_algo::Val,
                observables_index::Vector{Int},
                𝐒::AbstractMatrix{Float64},
                data_in_deviations::Matrix{Float64},
                constants::constants,
                state,
                workspaces::workspaces,
                obs_idx_per_t::Vector{Vector{Int}};
                kwargs...)
    llh, dense_pb = rrule(calculate_loglikelihood, Val(:kalman), val_algo,
                          observables_index, 𝐒, data_in_deviations,
                          constants, state, workspaces; kwargs...)
    pullback = ∂llh -> (dense_pb(∂llh)..., NoTangent())
    return llh, pullback
end


function get_statistics_cotangent_helper(Δret, key::Symbol)
    Δ = unthunk(Δret)
    if Δ isa Union{NoTangent, AbstractZero}
        return NoTangent()
    end

    if Δ isa AbstractDict
        return get(Δ, key, NoTangent())
    end

    if Δ isa NamedTuple
        return get(Δ, key, NoTangent())
    end

    if hasproperty(Δ, key)
        return getproperty(Δ, key)
    end

    if hasmethod(haskey, Tuple{typeof(Δ), Symbol}) && haskey(Δ, key)
        return Δ[key]
    end

    if hasmethod(pairs, Tuple{typeof(Δ)})
        for (k, v) in pairs(Δ)
            if k == key
                return v
            end
        end
    end

    if hasproperty(Δ, :pairs)
        pairs_obj = getproperty(Δ, :pairs)
        if pairs_obj isa AbstractDict
            return get(pairs_obj, key, NoTangent())
        elseif pairs_obj isa NamedTuple
            return get(pairs_obj, key, NoTangent())
        elseif hasmethod(pairs, Tuple{typeof(pairs_obj)})
            for (k, v) in pairs(pairs_obj)
                if k == key
                    return v
                end
            end
        end
    end

    return NoTangent()
end


function rrule(::typeof(get_statistics),
                𝓂::ℳ,
                parameter_values::Vector{T};
                parameters::Union{Vector{Symbol},Vector{String}} = 𝓂.constants.post_complete_parameters.parameters,
                steady_state_function::SteadyStateFunctionType = missing,
                non_stochastic_steady_state::Union{Symbol_input,String_input} = Symbol[],
                mean::Union{Symbol_input,String_input} = Symbol[],
                standard_deviation::Union{Symbol_input,String_input} = Symbol[],
                variance::Union{Symbol_input,String_input} = Symbol[],
                covariance::Union{Symbol_input,String_input, Vector{Vector{Symbol}},Vector{Tuple{Symbol,Vararg{Symbol}}},Vector{Vector{Symbol}},Tuple{Tuple{Symbol,Vararg{Symbol}},Vararg{Tuple{Symbol,Vararg{Symbol}}}}, Vector{Vector{String}},Vector{Tuple{String,Vararg{String}}},Vector{Vector{String}},Tuple{Tuple{String,Vararg{String}},Vararg{Tuple{String,Vararg{String}}}}} = Symbol[],
                correlation::Union{Symbol_input,String_input, Vector{Vector{Symbol}},Vector{Tuple{Symbol,Vararg{Symbol}}},Vector{Vector{Symbol}},Tuple{Tuple{Symbol,Vararg{Symbol}},Vararg{Tuple{Symbol,Vararg{Symbol}}}}, Vector{Vector{String}},Vector{Tuple{String,Vararg{String}}},Vector{Vector{String}},Tuple{Tuple{String,Vararg{String}},Vararg{Tuple{String,Vararg{String}}}}} = Symbol[],
                autocorrelation::Union{Symbol_input,String_input} = Symbol[],
                autocorrelation_periods::UnitRange{Int} = DEFAULT_AUTOCORRELATION_PERIODS,
                algorithm::Symbol = DEFAULT_ALGORITHM,
                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                verbose::Bool = DEFAULT_VERBOSE,
                tol::Tolerances = Tolerances()) where T

    opts = merge_calculation_options(tol = tol,
                                    verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                                    lyapunov_algorithm = lyapunov_algorithm)

    @assert length(parameter_values) == length(parameters) "Vector of `parameters` must correspond to `parameter_values` in length and order. Define the parameter names in the `parameters` keyword argument."

    @assert algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] || !(!(standard_deviation == Symbol[]) || !(mean == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(correlation == Symbol[]) || !(autocorrelation == Symbol[])) "Statistics can only be provided for first order perturbation or second and third order pruned perturbation solutions."

    @assert !(non_stochastic_steady_state == Symbol[]) || !(standard_deviation == Symbol[]) || !(mean == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(correlation == Symbol[]) || !(autocorrelation == Symbol[]) "Provide variables for at least one output."

    SS_var_idx = parse_variables_input_to_index(non_stochastic_steady_state, 𝓂)
    mean_var_idx = parse_variables_input_to_index(mean, 𝓂)
    std_var_idx = parse_variables_input_to_index(standard_deviation, 𝓂)
    var_var_idx = parse_variables_input_to_index(variance, 𝓂)
    covar_var_idx = parse_variables_input_to_index(covariance, 𝓂)
    covar_groups = is_grouped_covariance_input(covariance) ? parse_covariance_groups(covariance, 𝓂.constants) : nothing
    corr_var_idx = parse_variables_input_to_index(correlation, 𝓂)
    corr_groups = is_grouped_covariance_input(correlation) ? parse_covariance_groups(correlation, 𝓂.constants) : nothing
    autocorr_var_idx = parse_variables_input_to_index(autocorrelation, 𝓂)

    other_parameter_values = 𝓂.parameter_values[indexin(setdiff(𝓂.constants.post_complete_parameters.parameters, parameters), 𝓂.constants.post_complete_parameters.parameters)]
    sort_idx = sortperm(vcat(indexin(setdiff(𝓂.constants.post_complete_parameters.parameters, parameters), 𝓂.constants.post_complete_parameters.parameters), indexin(parameters, 𝓂.constants.post_complete_parameters.parameters)))

    all_parameters = vcat(other_parameter_values, parameter_values)[sort_idx]
    n_other = length(other_parameter_values)
    inv_sort = invperm(sort_idx)

    run_algorithm = algorithm
    if run_algorithm == :pruned_third_order && !(!(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(correlation == Symbol[]) || !(autocorrelation == Symbol[]))
        run_algorithm = :pruned_second_order
    end

        solve!(𝓂,
            algorithm = run_algorithm,
            steady_state_function = steady_state_function,
            opts = opts)

    nVars = length(𝓂.constants.post_model_macro.var)

    nsss_only = !(non_stochastic_steady_state == Symbol[]) && (standard_deviation == Symbol[]) && (variance == Symbol[]) && (covariance == Symbol[]) && (correlation == Symbol[]) && (autocorrelation == Symbol[])

    nsss_pb = nothing
    cov_pb = nothing
    som_pb = nothing
    somc_pb = nothing
    tom_pb = nothing
    toma_pb = nothing

    solved = true
    SS_and_pars = zeros(T, 0)
    SS = zeros(T, 0)
    state_μ = zeros(T, 0)

    covar_dcmp = zeros(T, 0, 0)
    sol = zeros(T, 0, 0)

    Σᶻ₂ = zeros(T, 0, 0)
    Δμˢ₂ = zeros(T, 0)
    autocorr_tmp = zeros(T, 0, 0)
    ŝ_to_ŝ₂ = zeros(T, 0, 0)
    ŝ_to_y₂ = zeros(T, 0, 0)

    autocorr = zeros(T, 0, 0)
    first_order_A = zeros(T, 0, 0)
    first_order_P = zeros(T, 0, 0)
    first_order_R_seq = Matrix{T}[]
    first_order_d = zeros(T, 0)
    first_order_mask = BitVector()

    second_order_P_seq = Matrix{T}[]
    second_order_M_seq = Matrix{T}[]
    second_order_d = zeros(T, 0)
    second_order_mask = BitVector()

    st_dev = zeros(T, 0)
    varrs = zeros(T, 0)
    diag_covar = zeros(T, 0)
    diag_gate = falses(0)

    covar_dcmp_sp = zeros(T, 0, 0)
    covar_group_pairs = NTuple{4,Int}[]

    corr_full_mat = zeros(T, 0, 0)
    diag_C_corr = zeros(T, 0)
    s_corr = zeros(T, 0)
    corr_group_pairs = NTuple{4,Int}[]

    if nsss_only
        prev_Δnsss = Ref{Any}(nothing)

        nsss_out, nsss_pb_local = rrule(get_NSSS_and_parameters, 𝓂, all_parameters; opts = opts)
        nsss_pb = nsss_pb_local

        SS_and_pars = nsss_out[1]
        solution_error = nsss_out[2][1]
        SS = SS_and_pars[1:end - length(𝓂.equations.calibration)]

        ret = Dict{Symbol,AbstractArray{T}}()
        ret[:non_stochastic_steady_state] = solution_error < opts.tol.nsss.acceptance_tol ? SS[SS_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(SS_var_idx) ? 0 : length(SS_var_idx))

        function nsss_only_pullback(Δret)
            Δnsss = incremental_cotangent!(get_statistics_cotangent_helper(Δret, :non_stochastic_steady_state), prev_Δnsss)
            if Δnsss isa Union{NoTangent, AbstractZero}
                return NoTangent(), NoTangent(), zeros(T, length(parameter_values))
            end

            ∂SS = zeros(T, length(SS))
            ∂SS[SS_var_idx] .+= unthunk(Δnsss)

            ∂SS_and_pars = zeros(T, length(SS_and_pars))
            ∂SS_and_pars[1:length(SS)] .+= ∂SS

            nsss_grads = nsss_pb((∂SS_and_pars, NoTangent()))
            ∂all_parameters = nsss_grads[3] isa AbstractZero ? zeros(T, length(all_parameters)) : nsss_grads[3]

            ∂concat = ∂all_parameters[inv_sort]
            ∂parameter_values = ∂concat[(n_other + 1):end]

            return NoTangent(), NoTangent(), ∂parameter_values
        end

        return ret, nsss_only_pullback
    end

    if run_algorithm == :pruned_third_order
        if !(autocorrelation == Symbol[])
            second_mom_third_order = union(autocorr_var_idx, std_var_idx, var_var_idx, corr_var_idx)
            toma_out, toma_pb_local = rrule(calculate_third_order_moments_with_autocorrelation,
                                            all_parameters,
                                            𝓂.constants.post_model_macro.var[second_mom_third_order],
                                            𝓂;
                                            covariance = 𝓂.constants.post_model_macro.var[union(covar_var_idx, corr_var_idx)],
                                            opts = opts,
                                            autocorrelation_periods = autocorrelation_periods)
            toma_pb = toma_pb_local

            covar_dcmp = toma_out[1]
            state_μ = toma_out[2]
            autocorr = toma_out[3]
            SS_and_pars = toma_out[4]
            solved = toma_out[5]
        elseif !(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(correlation == Symbol[])
            tom_out, tom_pb_local = rrule(calculate_third_order_moments,
                                        all_parameters,
                                        𝓂.constants.post_model_macro.var[union(std_var_idx, var_var_idx, corr_var_idx)],
                                        𝓂;
                                        covariance = 𝓂.constants.post_model_macro.var[union(covar_var_idx, corr_var_idx)],
                                        opts = opts)
            tom_pb = tom_pb_local

            covar_dcmp = tom_out[1]
            state_μ = tom_out[2]
            SS_and_pars = tom_out[3]
            solved = tom_out[4]
        end
    elseif run_algorithm == :pruned_second_order
        if !(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(correlation == Symbol[]) || !(autocorrelation == Symbol[])
            somc_out, somc_pb_local = rrule(calculate_second_order_moments_with_covariance, all_parameters, 𝓂; opts = opts)
            somc_pb = somc_pb_local

            covar_dcmp = somc_out[1]
            Σᶻ₂ = somc_out[2]
            state_μ = somc_out[3]
            Δμˢ₂ = somc_out[4]
            autocorr_tmp = somc_out[5]
            ŝ_to_ŝ₂ = somc_out[6]
            ŝ_to_y₂ = somc_out[7]
            SS_and_pars = somc_out[10]
            solved = somc_out[15]
        else
            som_out, som_pb_local = rrule(calculate_second_order_moments, all_parameters, 𝓂; opts = opts)
            som_pb = som_pb_local

            state_μ = som_out[1]
            Δμˢ₂ = som_out[2]
            SS_and_pars = som_out[5]
            solved = som_out[10]
        end
    else
        cov_out, cov_pb_local = rrule(calculate_covariance, all_parameters, 𝓂; opts = opts)
        cov_pb = cov_pb_local

        covar_dcmp = cov_out[1]
        sol = cov_out[2]
        SS_and_pars = cov_out[4]
        solved = cov_out[5]
    end

    SS = SS_and_pars[1:end - length(𝓂.equations.calibration)]

    if !(variance == Symbol[]) || !(standard_deviation == Symbol[])
        diag_covar = convert(Vector{T}, ℒ.diag(covar_dcmp))
        diag_max = max.(diag_covar, eps(Float64))
        diag_gate = diag_covar .> eps(Float64)
        if !(variance == Symbol[])
            varrs = convert(Vector{T}, diag_max)
        end
        if !(standard_deviation == Symbol[])
            st_dev = sqrt.(abs.(convert(Vector{T}, diag_max)))
        end
    end

    if !(autocorrelation == Symbol[])
        if run_algorithm == :pruned_second_order
            P_i = Matrix{T}(ℒ.I(size(ŝ_to_ŝ₂, 1)))
            autocorr = zeros(T, size(covar_dcmp, 1), length(autocorrelation_periods))
            second_order_P_seq = [zeros(T, 0, 0) for _ in 1:maximum(autocorrelation_periods)]
            second_order_M_seq = [zeros(T, 0, 0) for _ in 1:maximum(autocorrelation_periods)]
            second_order_d = max.(convert(Vector{T}, ℒ.diag(covar_dcmp)), eps(Float64))

            for i in autocorrelation_periods
                second_order_P_seq[i] = copy(P_i)
                M_i = ŝ_to_y₂ * P_i * autocorr_tmp
                second_order_M_seq[i] = M_i
                autocorr[:, i] .= ℒ.diag(M_i) ./ second_order_d
                P_i = P_i * ŝ_to_ŝ₂
            end

            second_order_mask = ℒ.diag(covar_dcmp) .< opts.tol.second_order.lyapunov.acceptance_tol
            autocorr[second_order_mask, :] .= 0
        elseif !(run_algorithm == :pruned_third_order)
            first_order_P = ℒ.diagm(ones(T, 𝓂.constants.post_model_macro.nVars))[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx, :]
            first_order_A = @views sol[:, 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] * first_order_P
            first_order_d = max.(convert(Vector{T}, ℒ.diag(covar_dcmp)), eps(Float64))
            d_inv = 1 ./ first_order_d

            autocorr = zeros(T, size(covar_dcmp, 1), length(autocorrelation_periods))
            first_order_R_seq = [zeros(T, 0, 0) for _ in 1:maximum(autocorrelation_periods)]

            R = Matrix(covar_dcmp)
            for i in 1:maximum(autocorrelation_periods)
                R = first_order_A * R
                first_order_R_seq[i] = copy(R)
            end

            for i in autocorrelation_periods
                autocorr[:, i] .= ℒ.diag(first_order_R_seq[i]) .* d_inv
            end

            first_order_mask = ℒ.diag(covar_dcmp) .< opts.tol.first_order.lyapunov.acceptance_tol
            autocorr[first_order_mask, :] .= 0
        end
    end

    if !(covariance == Symbol[])
        covar_dcmp_sp = ℒ.triu(covar_dcmp)

        if !isnothing(covar_groups)
            for group in covar_groups
                for i in group
                    i_pos = findfirst(==(i), covar_var_idx)
                    isnothing(i_pos) && continue
                    for j in group
                        j_pos = findfirst(==(j), covar_var_idx)
                        isnothing(j_pos) && continue
                        push!(covar_group_pairs, (i_pos, j_pos, i, j))
                    end
                end
            end
        end
    end

    if !(correlation == Symbol[])
        if size(covar_dcmp, 1) > 0
            corr_full_mat, _, diag_C_corr, s_corr = covariance_to_correlation(covar_dcmp)
        end

        if !isnothing(corr_groups)
            for group in corr_groups
                for i in group
                    i_pos = findfirst(==(i), corr_var_idx)
                    isnothing(i_pos) && continue
                    for j in group
                        j_pos = findfirst(==(j), corr_var_idx)
                        isnothing(j_pos) && continue
                        push!(corr_group_pairs, (i_pos, j_pos, i, j))
                    end
                end
            end
        end
    end

    ret = Dict{Symbol,AbstractArray{T}}()

    if !(non_stochastic_steady_state == Symbol[])
        ret[:non_stochastic_steady_state] = solved ? SS[SS_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(SS_var_idx) ? 0 : length(SS_var_idx))
    end
    if !(mean == Symbol[])
        if run_algorithm ∉ [:pruned_second_order,:pruned_third_order]
            ret[:mean] = solved ? SS[mean_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(mean_var_idx) ? 0 : length(mean_var_idx))
        else
            ret[:mean] = solved ? state_μ[mean_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(mean_var_idx) ? 0 : length(mean_var_idx))
        end
    end
    if !(standard_deviation == Symbol[])
        ret[:standard_deviation] = solved ? st_dev[std_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(std_var_idx) ? 0 : length(std_var_idx))
    end
    if !(variance == Symbol[])
        ret[:variance] = solved ? varrs[var_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(var_var_idx) ? 0 : length(var_var_idx))
    end
    if !(covariance == Symbol[])
        if !isnothing(covar_groups)
            if solved
                covar_result = zeros(T, length(covar_var_idx), length(covar_var_idx))
                for (i_pos, j_pos, i, j) in covar_group_pairs
                    covar_result[i_pos, j_pos] = covar_dcmp_sp[i, j]
                end
                ret[:covariance] = covar_result
            else
                ret[:covariance] = fill(Inf * sum(abs2,parameter_values), length(covar_var_idx), length(covar_var_idx))
            end
        else
            ret[:covariance] = solved ? covar_dcmp_sp[covar_var_idx, covar_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(covar_var_idx) ? 0 : length(covar_var_idx), isnothing(covar_var_idx) ? 0 : length(covar_var_idx))
        end
    end
    if !(correlation == Symbol[])
        if solved
            if !isnothing(corr_groups)
                corr_result = zeros(T, length(corr_var_idx), length(corr_var_idx))
                for (i_pos, j_pos, i, j) in corr_group_pairs
                    corr_result[i_pos, j_pos] = corr_full_mat[i, j]
                end
                ret[:correlation] = corr_result
            else
                ret[:correlation] = corr_full_mat[corr_var_idx, corr_var_idx]
            end
        else
            ret[:correlation] = fill(Inf * sum(abs2,parameter_values), isnothing(corr_var_idx) ? 0 : length(corr_var_idx), isnothing(corr_var_idx) ? 0 : length(corr_var_idx))
        end
    end
    if !(autocorrelation == Symbol[])
        ret[:autocorrelation] = solved ? autocorr[autocorr_var_idx, :] : fill(Inf * sum(abs2,parameter_values), isnothing(autocorr_var_idx) ? 0 : length(autocorr_var_idx), isnothing(autocorrelation_periods) ? 0 : length(autocorrelation_periods))
    end

    prev_Δnsss = Ref{Any}(nothing)
    prev_Δmean = Ref{Any}(nothing)
    prev_Δstd = Ref{Any}(nothing)
    prev_Δvar = Ref{Any}(nothing)
    prev_Δcov = Ref{Any}(nothing)
    prev_Δcorr = Ref{Any}(nothing)
    prev_Δautocorr = Ref{Any}(nothing)

    function get_statistics_pullback(Δret)
        if !solved
            return NoTangent(), NoTangent(), zeros(T, length(parameter_values))
        end

        Δnsss = incremental_cotangent!(get_statistics_cotangent_helper(Δret, :non_stochastic_steady_state), prev_Δnsss)
        Δmean = incremental_cotangent!(get_statistics_cotangent_helper(Δret, :mean), prev_Δmean)
        Δstd = incremental_cotangent!(get_statistics_cotangent_helper(Δret, :standard_deviation), prev_Δstd)
        Δvar = incremental_cotangent!(get_statistics_cotangent_helper(Δret, :variance), prev_Δvar)
        Δcov = incremental_cotangent!(get_statistics_cotangent_helper(Δret, :covariance), prev_Δcov)
        Δcorr = incremental_cotangent!(get_statistics_cotangent_helper(Δret, :correlation), prev_Δcorr)
        Δautocorr = incremental_cotangent!(get_statistics_cotangent_helper(Δret, :autocorrelation), prev_Δautocorr)

        ∂SS_and_pars = zeros(T, length(SS_and_pars))
        ∂state_μ = length(state_μ) == 0 ? zeros(T, 0) : zeros(T, length(state_μ))
        ∂covar_dcmp = size(covar_dcmp, 1) == 0 ? zeros(T, 0, 0) : zeros(T, size(covar_dcmp))
        ∂sol = size(sol, 1) == 0 ? zeros(T, 0, 0) : zeros(T, size(sol))
        ∂autocorr_tmp = size(autocorr_tmp, 1) == 0 ? zeros(T, 0, 0) : zeros(T, size(autocorr_tmp))
        ∂ŝ_to_ŝ₂ = size(ŝ_to_ŝ₂, 1) == 0 ? zeros(T, 0, 0) : zeros(T, size(ŝ_to_ŝ₂))
        ∂ŝ_to_y₂ = size(ŝ_to_y₂, 1) == 0 ? zeros(T, 0, 0) : zeros(T, size(ŝ_to_y₂))

        if !(Δnsss isa Union{NoTangent, AbstractZero})
            ∂SS_and_pars[SS_var_idx] .+= Δnsss
        end

        if !(Δmean isa Union{NoTangent, AbstractZero})
            if run_algorithm ∉ [:pruned_second_order,:pruned_third_order]
                ∂SS_and_pars[mean_var_idx] .+= Δmean
            else
                ∂state_μ[mean_var_idx] .+= Δmean
            end
        end

        if !(Δvar isa Union{NoTangent, AbstractZero})
            ∂var_full = zeros(T, length(diag_covar))
            ∂var_full[var_var_idx] .+= Δvar
            @inbounds for i in eachindex(diag_covar)
                if diag_gate[i]
                    ∂covar_dcmp[i, i] += ∂var_full[i]
                end
            end
        end

        if !(Δstd isa Union{NoTangent, AbstractZero})
            ∂std_full = zeros(T, length(diag_covar))
            ∂std_full[std_var_idx] .+= Δstd
            @inbounds for i in eachindex(diag_covar)
                if diag_gate[i]
                    ∂covar_dcmp[i, i] += ∂std_full[i] / (2 * st_dev[i])
                end
            end
        end

        if !(Δcov isa Union{NoTangent, AbstractZero})
            ∂covar_dcmp_sp = zeros(T, size(covar_dcmp))

            if !isnothing(covar_groups)
                for (i_pos, j_pos, i, j) in covar_group_pairs
                    ∂covar_dcmp_sp[i, j] += Δcov[i_pos, j_pos]
                end
            else
                ∂covar_dcmp_sp[covar_var_idx, covar_var_idx] .+= Δcov
            end

            ∂covar_dcmp .+= ℒ.triu(∂covar_dcmp_sp)
        end

        if !(Δcorr isa Union{NoTangent, AbstractZero}) && !(correlation == Symbol[])
            Δcorr_full = zeros(T, length(corr_var_idx), length(corr_var_idx))
            if !isnothing(corr_groups)
                for (i_pos, j_pos, i, j) in corr_group_pairs
                    Δcorr_full[i_pos, j_pos] += Δcorr[i_pos, j_pos]
                end
            else
                Δcorr_full .+= Δcorr
            end

            @inbounds for a_pos in eachindex(corr_var_idx)
                a = corr_var_idx[a_pos]
                sa = s_corr[a]
                isnan(sa) && continue
                for b_pos in eachindex(corr_var_idx)
                    b = corr_var_idx[b_pos]
                    g = Δcorr_full[a_pos, b_pos]
                    g == 0 && continue
                    sb = s_corr[b]
                    isnan(sb) && continue
                    sasb = sa * sb
                    sasb == 0 && continue
                    corr_ab = corr_full_mat[a, b]
                    src_a = min(a, b)
                    src_b = max(a, b)
                    ∂covar_dcmp[src_a, src_b] += g / sasb
                    ∂covar_dcmp[a, a] += -g * corr_ab / (2 * diag_C_corr[a])
                    ∂covar_dcmp[b, b] += -g * corr_ab / (2 * diag_C_corr[b])
                end
            end
        end

        if !(Δautocorr isa Union{NoTangent, AbstractZero}) && !(autocorrelation == Symbol[])
            if run_algorithm == :pruned_second_order
                ∂autocorr_full = zeros(T, size(covar_dcmp, 1), length(autocorrelation_periods))
                ∂autocorr_full[autocorr_var_idx, :] .= Δautocorr
                ∂autocorr_full[second_order_mask, :] .= 0

                ∂d = zeros(T, length(second_order_d))
                ∂P = [zeros(T, size(second_order_P_seq[i])) for i in 1:length(second_order_P_seq)]

                for i in reverse(collect(autocorrelation_periods))
                    g = view(∂autocorr_full, :, i)
                    M_i = second_order_M_seq[i]
                    P_i = second_order_P_seq[i]

                    ∂M_i = zeros(T, size(M_i))
                    @inbounds for j in 1:size(M_i, 1)
                        ∂M_i[j, j] += g[j] / second_order_d[j]
                        ∂d[j] -= g[j] * M_i[j, j] / (second_order_d[j]^2)
                    end

                    P_aut = P_i * autocorr_tmp
                    ∂ŝ_to_y₂ .+= ∂M_i * P_aut'

                    ∂Paut = ŝ_to_y₂' * ∂M_i
                    ∂P[i] .+= ∂Paut * autocorr_tmp'
                    ∂autocorr_tmp .+= P_i' * ∂Paut
                end

                if length(second_order_P_seq) >= 2
                    for i in reverse(1:(length(second_order_P_seq) - 1))
                        ∂ŝ_to_ŝ₂ .+= second_order_P_seq[i]' * ∂P[i + 1]
                        ∂P[i] .+= ∂P[i + 1] * ŝ_to_ŝ₂'
                    end
                end

                diag_raw = convert(Vector{T}, ℒ.diag(covar_dcmp))
                @inbounds for i in eachindex(∂d)
                    if diag_raw[i] > eps(Float64)
                        ∂covar_dcmp[i, i] += ∂d[i]
                    end
                end

                ∂state_μ .+= zero(∂state_μ)
            elseif run_algorithm != :pruned_third_order
                ∂autocorr_full = zeros(T, size(covar_dcmp, 1), length(autocorrelation_periods))
                ∂autocorr_full[autocorr_var_idx, :] .= Δautocorr
                ∂autocorr_full[first_order_mask, :] .= 0

                d_inv = 1 ./ first_order_d
                ∂d = zeros(T, length(first_order_d))
                max_p = maximum(autocorrelation_periods)
                ∂R = [zeros(T, size(covar_dcmp)) for _ in 1:max_p]
                ∂A = zeros(T, size(first_order_A))

                for i in reverse(collect(autocorrelation_periods))
                    g = view(∂autocorr_full, :, i)
                    Ri = first_order_R_seq[i]
                    @inbounds for j in 1:length(g)
                        ∂R[i][j, j] += g[j] * d_inv[j]
                        ∂d[j] -= g[j] * Ri[j, j] / (first_order_d[j]^2)
                    end
                end

                for i in reverse(1:max_p)
                    if i < max_p
                        ∂R[i] .+= first_order_A' * ∂R[i + 1]
                    end
                    R_prev = (i == 1) ? Matrix(covar_dcmp) : first_order_R_seq[i - 1]
                    ∂A .+= ∂R[i] * R_prev'
                end

                if max_p >= 1
                    ∂covar_dcmp .+= first_order_A' * ∂R[1]
                end

                diag_raw = convert(Vector{T}, ℒ.diag(covar_dcmp))
                @inbounds for i in eachindex(∂d)
                    if diag_raw[i] > eps(Float64)
                        ∂covar_dcmp[i, i] += ∂d[i]
                    end
                end

                ∂sol[:, 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] .+= ∂A * first_order_P'
            end
        end

        ∂all_parameters = zeros(T, length(all_parameters))

        if nsss_only
            nsss_grads = nsss_pb((∂SS_and_pars, NoTangent()))
            ∂all_parameters .+= (nsss_grads[3] isa AbstractZero ? zeros(T, length(all_parameters)) : nsss_grads[3])
        elseif run_algorithm == :first_order
            cov_grads = cov_pb((∂covar_dcmp, ∂sol, NoTangent(), ∂SS_and_pars, NoTangent()))
            ∂all_parameters .+= (cov_grads[2] isa AbstractZero ? zeros(T, length(all_parameters)) : cov_grads[2])
        elseif run_algorithm == :pruned_second_order
            if som_pb !== nothing
                som_grads = som_pb((∂state_μ, NoTangent(), NoTangent(), NoTangent(), ∂SS_and_pars, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()))
                ∂all_parameters .+= (som_grads[2] isa AbstractZero ? zeros(T, length(all_parameters)) : som_grads[2])
            else
                somc_grads = somc_pb((∂covar_dcmp,
                                    NoTangent(),
                                    ∂state_μ,
                                    NoTangent(),
                                    run_algorithm == :pruned_second_order && !(autocorrelation == Symbol[]) ? ∂autocorr_tmp : NoTangent(),
                                    run_algorithm == :pruned_second_order && !(autocorrelation == Symbol[]) ? ∂ŝ_to_ŝ₂ : NoTangent(),
                                    run_algorithm == :pruned_second_order && !(autocorrelation == Symbol[]) ? ∂ŝ_to_y₂ : NoTangent(),
                                    NoTangent(),
                                    NoTangent(),
                                    ∂SS_and_pars,
                                    NoTangent(),
                                    NoTangent(),
                                    NoTangent(),
                                    NoTangent(),
                                    NoTangent()))
                ∂all_parameters .+= (somc_grads[2] isa AbstractZero ? zeros(T, length(all_parameters)) : somc_grads[2])
            end
        elseif run_algorithm == :pruned_third_order
            if toma_pb !== nothing
                ∂autocorr_full = zeros(T, size(autocorr))
                if !(Δautocorr isa Union{NoTangent, AbstractZero})
                    ∂autocorr_full[autocorr_var_idx, :] .= Δautocorr
                end
                toma_grads = toma_pb((∂covar_dcmp, ∂state_μ, ∂autocorr_full, ∂SS_and_pars, NoTangent()))
                ∂all_parameters .+= (toma_grads[2] isa AbstractZero ? zeros(T, length(all_parameters)) : toma_grads[2])
            elseif tom_pb !== nothing
                tom_grads = tom_pb((∂covar_dcmp, ∂state_μ, ∂SS_and_pars, NoTangent()))
                ∂all_parameters .+= (tom_grads[2] isa AbstractZero ? zeros(T, length(all_parameters)) : tom_grads[2])
            end
        end

        ∂concat = ∂all_parameters[inv_sort]
        ∂parameter_values = ∂concat[(n_other + 1):end]

        return NoTangent(), NoTangent(), ∂parameter_values
    end

    return ret, get_statistics_pullback
end


# ── get_solution rrule ──────────────────────────────────────────────────────────
# Custom rrule for get_solution(𝓂, parameters; ...) that chains existing
# sub-rrules without using AD inside the pullback.
# Supports first_order, second_order/pruned_second_order,
# and third_order/pruned_third_order algorithms.

function rrule(::typeof(get_solution),
                𝓂::ℳ,
                parameters::Vector{S};
                steady_state_function::SteadyStateFunctionType = missing,
                algorithm::Symbol = DEFAULT_ALGORITHM,
                verbose::Bool = DEFAULT_VERBOSE,
                tol::Tolerances = Tolerances(),
                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂)) where S <: Real

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? :bicgstab : sylvester_algorithm[2])

    estimation = true

    constants_obj = initialise_constants!(𝓂)

    solve!(𝓂,
           opts = opts,
           steady_state_function = steady_state_function,
           algorithm = algorithm)

    nVar = length(𝓂.constants.post_model_macro.var)

    zero_pullback(_) = (NoTangent(), NoTangent(), zeros(S, length(parameters)))

    # ── Check parameter bounds ──
    if check_bounds(parameters, 𝓂)
        return get_solution_fail(algorithm, fill(S(-Inf), nVar), nVar, S), zero_pullback
    end

    # ── Step 1: NSSS ──
    nsss_out, nsss_pb = rrule(get_NSSS_and_parameters,
                              𝓂,
                              parameters;
                              opts = opts,
                              estimation = estimation)

    SS_and_pars = nsss_out[1]
    solution_error = nsss_out[2][1]

    if solution_error > tol.nsss.acceptance_tol || isnan(solution_error)
        result = get_solution_fail(algorithm, SS_and_pars[1:nVar], nVar, S)
        return result, zero_pullback
    end

    # ── Step 2: Jacobian ──
    ∇₁, jac_pb = rrule(calculate_jacobian,
                        parameters,
                        SS_and_pars,
                        𝓂.caches,
                        𝓂.functions.jacobian,
                        𝓂.workspaces)

    # ── Step 3: First-order solution ──
    first_out, first_pb = rrule(calculate_first_order_solution,
                                ∇₁,
                                constants_obj,
                                𝓂.workspaces,
                                𝓂.caches;
                                opts = opts,
                                initial_guess = 𝓂.caches.qme_solution,
                                parameter_values = parameters)

    𝐒₁ = first_out[1]
    solved = first_out[3]

    update_perturbation_counter!(𝓂.counters, solved, estimation = estimation, order = 1)

    if !solved
        result = get_solution_fail(algorithm, SS_and_pars[1:nVar], nVar, S, 𝐒₁)
        return result, zero_pullback
    end

    # ── Branch by algorithm ──
    if algorithm in [:second_order, :pruned_second_order]
        # ── Step 4: Hessian ──
        ∇₂, hess_pb = rrule(calculate_hessian,
                             parameters,
                             SS_and_pars,
                             𝓂.caches,
                            𝓂.functions.hessian,
                            𝓂.workspaces)

        # ── Step 5: Second-order solution ──
        second_out, second_pb = rrule(calculate_second_order_solution,
                                      ∇₁, ∇₂, 𝐒₁,
                                      𝓂.constants,
                                      𝓂.workspaces,
                                      𝓂.caches;
                                      initial_guess = 𝓂.caches.second_order_solution,
                                      opts = opts,
                                      parameter_values = parameters)

        𝐒₂_raw = second_out[1]
        solved2 = second_out[2]

        update_perturbation_counter!(𝓂.counters, solved2, estimation = estimation, order = 2)

        # Return: (NSSS, [𝐒₁, 𝐒₂], solved)
        result = (SS_and_pars[1:nVar], AbstractMatrix{S}[𝐒₁, 𝐒₂_raw], true)

        pullback_2nd = function (∂result_bar)
            Δ = unthunk(∂result_bar)

            if Δ isa Union{NoTangent, AbstractZero}
                return NoTangent(), NoTangent(), zeros(S, length(parameters))
            end

            ∂NSSS    = Δ[1]
            ∂mats    = unthunk(Δ[2])  # cotangent for Vector{AbstractMatrix{S}}
            # Δ[3] is ∂solved — not differentiable

            # Extract per-matrix cotangents defensively
            ∂𝐒₁_ext = if ∂mats isa Union{NoTangent, AbstractZero}
                NoTangent()
            else
                m = unthunk(∂mats[1])
                m isa Union{NoTangent, AbstractZero} ? NoTangent() : m
            end
            ∂𝐒₂_ext = if ∂mats isa Union{NoTangent, AbstractZero}
                NoTangent()
            else
                m = unthunk(∂mats[2])
                m isa Union{NoTangent, AbstractZero} ? NoTangent() : m
            end

            # ── Accumulate ∂SS_and_pars (zero-pad to full length) ──
            ∂SS_and_pars = zeros(S, length(SS_and_pars))
            if !(∂NSSS isa Union{NoTangent, AbstractZero})
                ∂SS_and_pars[1:nVar] .+= ∂NSSS
            end

            ∂parameters = zeros(S, length(parameters))

            # ── 𝐒₂ is already in compressed space — no 𝐔₂ adjoint needed ──
            ∂𝐒₂_raw = if ∂𝐒₂_ext isa Union{NoTangent, AbstractZero}
                zeros(S, size(𝐒₂_raw))
            else
                Matrix{S}(∂𝐒₂_ext)
            end

            # ── second_pb: (∂𝐒₂_raw, ∂solved2) ──
            second_grads = second_pb((∂𝐒₂_raw, NoTangent()))
            ∂∇₁_from_2nd  = second_grads[2]
            ∂∇₂_from_2nd  = second_grads[3]
            ∂𝑺₁_from_2nd  = second_grads[4]

            # ── ∇₂ is internal-only; gradient comes from second-order solution path ──
            ∂∇₂_total = ∂∇₂_from_2nd

            # ── hess_pb ──
            hess_grads = hess_pb(∂∇₂_total)
            ∂parameters  .+= hess_grads[2]
            ∂SS_and_pars .+= hess_grads[3]

            # ── Accumulate ∂𝐒₁ ──
            ∂𝐒₁_total = if ∂𝐒₁_ext isa Union{NoTangent, AbstractZero}
                ∂𝑺₁_from_2nd
            else
                ∂𝐒₁_ext + ∂𝑺₁_from_2nd
            end

            # ── first_pb ──
            first_grads = first_pb((∂𝐒₁_total, NoTangent(), NoTangent()))
            
            ∂∇₁_total = ∂∇₁_from_2nd + first_grads[2]

            # ── jac_pb ──
            jac_grads = jac_pb(∂∇₁_total)
            ∂parameters  .+= jac_grads[2]
            ∂SS_and_pars .+= jac_grads[3]

            # ── nsss_pb ──
            nsss_grads = nsss_pb((∂SS_and_pars, NoTangent()))
            ∂parameters .+= nsss_grads[3]

            return NoTangent(), NoTangent(), ∂parameters
        end

        return result, pullback_2nd

    elseif algorithm in [:third_order, :pruned_third_order]
        # ── Step 4: Hessian ──
        ∇₂, hess_pb = rrule(calculate_hessian,
                             parameters,
                             SS_and_pars,
                             𝓂.caches,
                            𝓂.functions.hessian,
                            𝓂.workspaces)

        # ── Step 5: Second-order solution ──
        second_out, second_pb = rrule(calculate_second_order_solution,
                                      ∇₁, ∇₂, 𝐒₁,
                                      𝓂.constants,
                                      𝓂.workspaces,
                                      𝓂.caches;
                                      initial_guess = 𝓂.caches.second_order_solution,
                                      opts = opts,
                                      parameter_values = parameters)

        𝐒₂_raw = second_out[1]
        solved2 = second_out[2]

        update_perturbation_counter!(𝓂.counters, solved2, estimation = estimation, order = 2)

        # ── Step 6: Third-order derivatives ──
        ∇₃, third_deriv_pb = rrule(calculate_third_order_derivatives,
                                    parameters,
                                    SS_and_pars,
                                    𝓂.caches,
                                    𝓂.functions.third_order_derivatives,
                                    𝓂.workspaces)

        # ── Step 7: Third-order solution ──
        # calculate_third_order_solution now receives compressed 𝐒₂ and compressed ∇₂
        third_out, third_pb = rrule(calculate_third_order_solution,
                                    ∇₁, ∇₂, ∇₃,
                                    𝐒₁, 𝐒₂_raw,
                                    𝓂.constants,
                                    𝓂.workspaces,
                                    𝓂.caches;
                                    initial_guess = 𝓂.caches.third_order_solution,
                                    opts = opts,
                                    parameter_values = parameters)

        𝐒₃_raw = third_out[1]
        solved3 = third_out[2]

        update_perturbation_counter!(𝓂.counters, solved3, estimation = estimation, order = 3)

        # Return: (NSSS, [𝐒₁, 𝐒₂, 𝐒₃], solved)
        result = (SS_and_pars[1:nVar], AbstractMatrix{S}[𝐒₁, 𝐒₂_raw, 𝐒₃_raw], true)

        pullback_3rd = function (∂result_bar)
            Δ = unthunk(∂result_bar)

            if Δ isa Union{NoTangent, AbstractZero}
                return NoTangent(), NoTangent(), zeros(S, length(parameters))
            end

            ∂NSSS    = Δ[1]
            ∂mats    = unthunk(Δ[2])  # cotangent for Vector{AbstractMatrix{S}}
            # Δ[3] is ∂solved — not differentiable

            # Extract per-matrix cotangents defensively
            ∂𝐒₁_ext = if ∂mats isa Union{NoTangent, AbstractZero}
                NoTangent()
            else
                m = unthunk(∂mats[1])
                m isa Union{NoTangent, AbstractZero} ? NoTangent() : m
            end
            ∂𝐒₂_ext = if ∂mats isa Union{NoTangent, AbstractZero}
                NoTangent()
            else
                m = unthunk(∂mats[2])
                m isa Union{NoTangent, AbstractZero} ? NoTangent() : m
            end
            ∂𝐒₃_ext = if ∂mats isa Union{NoTangent, AbstractZero}
                NoTangent()
            else
                m = unthunk(∂mats[3])
                m isa Union{NoTangent, AbstractZero} ? NoTangent() : m
            end

            # ── Accumulate ∂SS_and_pars (zero-pad to full length) ──
            ∂SS_and_pars = zeros(S, length(SS_and_pars))
            if !(∂NSSS isa Union{NoTangent, AbstractZero})
                ∂SS_and_pars[1:nVar] .+= ∂NSSS
            end

            ∂parameters = zeros(S, length(parameters))

            # ── 𝐒₃ is already in compressed space — no 𝐔₃ adjoint needed ──
            ∂𝐒₃_raw = if ∂𝐒₃_ext isa Union{NoTangent, AbstractZero}
                zeros(S, size(𝐒₃_raw))
            else
                Matrix{S}(∂𝐒₃_ext)
            end

            # ── third_pb: (∂𝐒₃_raw, ∂solved3) ──
            # Returns (NT, ∂∇₁, ∂∇₂, ∂∇₃, ∂𝑺₁, ∂𝐒₂, NT, NT, NT)
            third_grads = third_pb((∂𝐒₃_raw, NoTangent()))
            ∂∇₁_from_3rd  = third_grads[2]
            ∂∇₂_from_3rd  = third_grads[3]
            ∂∇₃_from_3rd  = third_grads[4]
            ∂𝑺₁_from_3rd  = third_grads[5]
            ∂𝐒₂_from_3rd  = third_grads[6]  # w.r.t. compressed 𝐒₂

            # ── ∇₃ is internal-only; gradient comes from third-order solution path ──
            ∂∇₃_total = ∂∇₃_from_3rd
            third_deriv_grads = third_deriv_pb(∂∇₃_total)
            ∂parameters  .+= third_deriv_grads[2]
            ∂SS_and_pars .+= third_deriv_grads[3]

            # ── Accumulate ∂𝐒₂ (compressed) from external + third-order ──
            ∂𝐒₂_total = if ∂𝐒₂_ext isa Union{NoTangent, AbstractZero}
                ∂𝐒₂_from_3rd isa Union{NoTangent, AbstractZero} ? zeros(S, size(𝐒₂_raw)) : Matrix{S}(∂𝐒₂_from_3rd)
            else
                ∂𝐒₂_from_3rd isa Union{NoTangent, AbstractZero} ? Matrix{S}(∂𝐒₂_ext) : Matrix{S}(∂𝐒₂_ext) + Matrix{S}(∂𝐒₂_from_3rd)
            end

            # ── second_pb: (∂𝐒₂_raw, ∂solved2) ──
            second_grads = second_pb((∂𝐒₂_total, NoTangent()))
            ∂∇₁_from_2nd  = second_grads[2]
            ∂∇₂_from_2nd  = second_grads[3]
            ∂𝑺₁_from_2nd  = second_grads[4]

            # ── hess_pb (accumulate ∂∇₂ from 2nd and 3rd order paths) ──
            ∂∇₂_total = ∂∇₂_from_3rd + ∂∇₂_from_2nd
            hess_grads = hess_pb(∂∇₂_total)
            ∂parameters  .+= hess_grads[2]
            ∂SS_and_pars .+= hess_grads[3]

            # ── Accumulate ∂𝐒₁ from external + 2nd + 3rd order ──
            ∂𝐒₁_total = if ∂𝐒₁_ext isa Union{NoTangent, AbstractZero}
                ∂𝑺₁_from_2nd + ∂𝑺₁_from_3rd
            else
                ∂𝐒₁_ext + ∂𝑺₁_from_2nd + ∂𝑺₁_from_3rd
            end

            # ── first_pb ──
            first_grads = first_pb((∂𝐒₁_total, NoTangent(), NoTangent()))
            ∂∇₁_total = ∂∇₁_from_3rd + ∂∇₁_from_2nd + first_grads[2]

            # ── jac_pb ──
            jac_grads = jac_pb(∂∇₁_total)
            ∂parameters  .+= jac_grads[2]
            ∂SS_and_pars .+= jac_grads[3]

            # ── nsss_pb ──
            nsss_grads = nsss_pb((∂SS_and_pars, NoTangent()))
            ∂parameters .+= nsss_grads[3]

            return NoTangent(), NoTangent(), ∂parameters
        end

        return result, pullback_3rd

    else
        # ── First order ──
        result = (SS_and_pars[1:nVar], AbstractMatrix{S}[𝐒₁], true)

        pullback_1st = function (∂result_bar)
            Δ = unthunk(∂result_bar)

            if Δ isa Union{NoTangent, AbstractZero}
                return NoTangent(), NoTangent(), zeros(S, length(parameters))
            end

            ∂NSSS    = Δ[1]
            ∂mats    = unthunk(Δ[2])  # cotangent for Vector{AbstractMatrix{S}}
            # Δ[3] is ∂solved — not differentiable

            # Extract ∂𝐒₁ defensively
            ∂𝐒₁_ext = if ∂mats isa Union{NoTangent, AbstractZero}
                NoTangent()
            else
                m = unthunk(∂mats[1])
                m isa Union{NoTangent, AbstractZero} ? NoTangent() : m
            end

            # ── Accumulate ∂SS_and_pars (zero-pad to full length) ──
            ∂SS_and_pars = zeros(S, length(SS_and_pars))
            if !(∂NSSS isa Union{NoTangent, AbstractZero})
                ∂SS_and_pars[1:nVar] .+= ∂NSSS
            end

            # Short-circuit when solution matrix cotangent is absent
            if ∂𝐒₁_ext isa Union{NoTangent, AbstractZero}
                nsss_grads = nsss_pb((∂SS_and_pars, NoTangent()))
                return NoTangent(), NoTangent(), nsss_grads[3]
            end

            # ── first_pb: (∂𝐒₁, ∂qme_sol, ∂solved) ──
            # Returns (NT, ∂∇₁, NT, NT, NT, NT)
            first_grads = first_pb((∂𝐒₁_ext, NoTangent(), NoTangent()))
            ∂∇₁ = first_grads[2]

            # ── jac_pb ──
            # Returns (NT, ∂parameters, ∂SS_and_pars, NT, NT)
            jac_grads = jac_pb(∂∇₁)
            ∂parameters  = copy(jac_grads[2])
            ∂SS_and_pars .+= jac_grads[3]

            # ── nsss_pb ──
            # Returns (NT, NT, ∂parameter_values, NT)
            nsss_grads = nsss_pb((∂SS_and_pars, NoTangent()))
            ∂parameters .+= nsss_grads[3]

            return NoTangent(), NoTangent(), ∂parameters
        end

        return result, pullback_1st
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# rrule for get_filter_free_loglikelihood
#
# Joint sampling of parameters + latent shocks per Childers, Fernández-Villaverde,
# Perla, Rackauckas & Wu (2025). The forward pass solves the model, then runs
# a deterministic forward simulation under user-supplied shocks. The pullback
# is analytical: backward through the per-period quadratic recursion gives the
# adjoints for shocks, me_std, S1, S2, the initial state, and SS_and_pars; the
# captured pullback of get_relevant_steady_state_and_state_update converts the
# matrix/state cotangents back to parameter cotangents.
# ─────────────────────────────────────────────────────────────────────────────

# Adjoint of Y = 𝐒₂ · kron(aug, aug) / 2 wrt aug, given d_new_state.
# Returns (d_aug_contribution, d_𝐒₂_contribution).
@inline function quad_adjoint(𝐒₂, aug::AbstractVector, d_new_state::AbstractVector)
    n_aug = length(aug)
    g = 𝐒₂' * d_new_state                       # length n_aug²
    G = reshape(Vector(g), n_aug, n_aug) ./ 2   # halve to fold in the /2 factor
    d_aug = G * aug .+ G' * aug
    kaa   = kron(aug, aug)
    d_𝐒₂  = (d_new_state * kaa') ./ 2
    return d_aug, d_𝐒₂
end

# Split a length-(npast+1+nExo) augmented adjoint back into past-state /
# constant / shock contributions.
@inline function split_aug_adjoint(d_aug::AbstractVector, npast::Int, nExo::Int)
    d_past  = d_aug[1:npast]
    d_shock = d_aug[npast+2:npast+1+nExo]
    return d_past, d_shock
end

function filter_free_pullback_2nd(
        Δllh::Real, intermediates, 𝐒₁, 𝐒₂, past_idx, obs_indices,
        nVars::Int, npast::Int, nExo::Int, nT::Int,
        me_std,
    )
    me_std_is_vec = me_std isa AbstractVector
    d_𝐒₁  = zeros(eltype(𝐒₁), size(𝐒₁))
    d_𝐒₂  = zeros(eltype(𝐒₂), size(𝐒₂))
    d_shocks = zeros(eltype(intermediates[1].aug), nExo, nT)
    d_SS_obs = zeros(eltype(intermediates[1].aug), length(obs_indices))
    d_me_std = me_std_is_vec ? zero(me_std) : zero(eltype(me_std))
    d_cur_state_next = zeros(eltype(intermediates[1].aug), nVars)

    @inbounds for t in nT:-1:1
        it       = intermediates[t]
        residual = it.residual
        aug      = it.aug
        n        = length(residual)
        # Logpdf adjoints
        if me_std_is_vec
            σ²       = me_std .^ 2
            d_residual = (.-residual ./ σ²) .* Δllh
            d_me_std .+= ((.-one(eltype(me_std)) ./ me_std) .+ (residual .^ 2) ./ (me_std .^ 3)) .* Δllh
        else
            σ²         = me_std^2
            d_residual = (-residual ./ σ²) .* Δllh
            d_me_std  += (-n/me_std + sum(abs2, residual) / me_std^3) * Δllh
        end
        # residual = data_dev - new_state[obs_indices]
        d_obs_dev   = -d_residual
        # data_dev = data - SS_and_pars[obs_indices]
        d_SS_obs  .-= d_residual
        # Scatter into d_new_state (length nVars)
        d_new_state = copy(d_cur_state_next)
        @inbounds for k in eachindex(obs_indices)
            d_new_state[obs_indices[k]] += d_obs_dev[k]
        end
        # Linear part: 𝐒₁ * aug
        d_𝐒₁ .+= d_new_state * aug'
        d_aug_lin = 𝐒₁' * d_new_state
        # Quadratic part: 𝐒₂ * kron(aug, aug) / 2
        d_aug_quad, d_𝐒₂_t = quad_adjoint(𝐒₂, aug, d_new_state)
        d_𝐒₂ .+= d_𝐒₂_t
        d_aug = d_aug_lin .+ d_aug_quad
        d_past, d_shock = split_aug_adjoint(d_aug, npast, nExo)
        d_shocks[:, t] .= d_shock
        # Feed past-state adjoint back as next iteration's d_cur_state_next
        d_cur_state_next = zeros(eltype(d_aug), nVars)
        @inbounds for k in eachindex(past_idx)
            d_cur_state_next[past_idx[k]] += d_past[k]
        end
    end
    return d_𝐒₁, d_𝐒₂, d_cur_state_next, d_SS_obs, d_shocks, d_me_std
end

function filter_free_pullback_pruned2nd(
        Δllh::Real, intermediates, 𝐒₁, 𝐒₂, past_idx, obs_indices,
        nVars::Int, npast::Int, nExo::Int, nT::Int,
        me_std,
    )
    me_std_is_vec = me_std isa AbstractVector
    d_𝐒₁  = zeros(eltype(𝐒₁), size(𝐒₁))
    d_𝐒₂  = zeros(eltype(𝐒₂), size(𝐒₂))
    d_shocks = zeros(eltype(intermediates[1].aug₁), nExo, nT)
    d_SS_obs = zeros(eltype(intermediates[1].aug₁), length(obs_indices))
    d_me_std = me_std_is_vec ? zero(me_std) : zero(eltype(me_std))
    d_cur_state_next = [zeros(eltype(intermediates[1].aug₁), nVars),
                        zeros(eltype(intermediates[1].aug₁), nVars)]

    @inbounds for t in nT:-1:1
        it       = intermediates[t]
        residual = it.residual
        aug₁     = it.aug₁
        aug₂     = it.aug₂
        n        = length(residual)
        if me_std_is_vec
            σ²       = me_std .^ 2
            d_residual = (.-residual ./ σ²) .* Δllh
            d_me_std .+= ((.-one(eltype(me_std)) ./ me_std) .+ (residual .^ 2) ./ (me_std .^ 3)) .* Δllh
        else
            σ²         = me_std^2
            d_residual = (-residual ./ σ²) .* Δllh
            d_me_std  += (-n/me_std + sum(abs2, residual) / me_std^3) * Δllh
        end
        d_obs_dev = -d_residual
        d_SS_obs .-= d_residual    # obs_dev = new[1][obs] + new[2][obs]
        # Scatter into d_new[1] and d_new[2]
        d_new_1 = copy(d_cur_state_next[1])
        d_new_2 = copy(d_cur_state_next[2])
        @inbounds for k in eachindex(obs_indices)
            d_new_1[obs_indices[k]] += d_obs_dev[k]
            d_new_2[obs_indices[k]] += d_obs_dev[k]
        end
        # Component 2: 𝐒₁·aug₂ + 𝐒₂·kron(aug₁,aug₁)/2
        d_𝐒₁ .+= d_new_2 * aug₂'
        d_aug₂  = 𝐒₁' * d_new_2
        d_aug₁_from_quad, d_𝐒₂_t = quad_adjoint(𝐒₂, aug₁, d_new_2)
        d_𝐒₂ .+= d_𝐒₂_t
        # Component 1: 𝐒₁·aug₁
        d_𝐒₁ .+= d_new_1 * aug₁'
        d_aug₁_from_lin = 𝐒₁' * d_new_1
        d_aug₁ = d_aug₁_from_lin .+ d_aug₁_from_quad
        d_past₁, d_shock = split_aug_adjoint(d_aug₁, npast, nExo)
        d_past₂, _       = split_aug_adjoint(d_aug₂, npast, nExo)  # shock part has zero primal dep
        d_shocks[:, t] .= d_shock
        d_cur_state_next = [zeros(eltype(d_aug₁), nVars), zeros(eltype(d_aug₁), nVars)]
        @inbounds for k in eachindex(past_idx)
            d_cur_state_next[1][past_idx[k]] += d_past₁[k]
            d_cur_state_next[2][past_idx[k]] += d_past₂[k]
        end
    end
    return d_𝐒₁, d_𝐒₂, d_cur_state_next, d_SS_obs, d_shocks, d_me_std
end


function filter_free_pullback_1st(
        Δllh::Real, intermediates, 𝐒₁, past_idx, obs_indices,
        nVars::Int, npast::Int, nExo::Int, nT::Int,
        me_std,
    )
    me_std_is_vec = me_std isa AbstractVector
    d_𝐒₁  = zeros(eltype(𝐒₁), size(𝐒₁))
    d_shocks = zeros(eltype(intermediates[1].aug), nExo, nT)
    d_SS_obs = zeros(eltype(intermediates[1].aug), length(obs_indices))
    d_me_std = me_std_is_vec ? zero(me_std) : zero(eltype(me_std))
    d_cur_state_next = zeros(eltype(intermediates[1].aug), nVars)

    @inbounds for t in nT:-1:1
        it       = intermediates[t]
        residual = it.residual
        aug      = it.aug
        n        = length(residual)
        if me_std_is_vec
            σ²       = me_std .^ 2
            d_residual = (.-residual ./ σ²) .* Δllh
            d_me_std .+= ((.-one(eltype(me_std)) ./ me_std) .+ (residual .^ 2) ./ (me_std .^ 3)) .* Δllh
        else
            σ²         = me_std^2
            d_residual = (-residual ./ σ²) .* Δllh
            d_me_std  += (-n/me_std + sum(abs2, residual) / me_std^3) * Δllh
        end
        d_obs_dev   = -d_residual
        d_SS_obs  .-= d_residual
        d_new_state = copy(d_cur_state_next)
        @inbounds for k in eachindex(obs_indices)
            d_new_state[obs_indices[k]] += d_obs_dev[k]
        end
        # new_state = 𝐒₁ * aug, aug = [past_state; ϵ]
        d_𝐒₁ .+= d_new_state * aug'
        d_aug = 𝐒₁' * d_new_state
        d_past  = d_aug[1:npast]
        d_shock = d_aug[npast+1:npast+nExo]
        d_shocks[:, t] .= d_shock
        d_cur_state_next = zeros(eltype(d_aug), nVars)
        @inbounds for k in eachindex(past_idx)
            d_cur_state_next[past_idx[k]] += d_past[k]
        end
    end
    return d_𝐒₁, d_cur_state_next, d_SS_obs, d_shocks, d_me_std
end

function filter_free_pullback_3rd(
        Δllh::Real, intermediates, 𝐒₁, 𝐒₂, 𝐒₃, past_idx, obs_indices,
        nVars::Int, npast::Int, nExo::Int, nT::Int,
        me_std,
    )
    me_std_is_vec = me_std isa AbstractVector
    d_𝐒₁  = zeros(eltype(𝐒₁), size(𝐒₁))
    d_𝐒₂  = zeros(eltype(𝐒₂), size(𝐒₂))
    d_𝐒₃  = zeros(eltype(𝐒₃), size(𝐒₃))
    n_aug = npast + 1 + nExo
    d_shocks = zeros(eltype(intermediates[1].aug), nExo, nT)
    d_SS_obs = zeros(eltype(intermediates[1].aug), length(obs_indices))
    d_me_std = me_std_is_vec ? zero(me_std) : zero(eltype(me_std))
    d_cur_state_next = zeros(eltype(intermediates[1].aug), nVars)

    @inbounds for t in nT:-1:1
        it       = intermediates[t]
        residual = it.residual
        aug      = it.aug
        kaug     = it.kaug
        n        = length(residual)
        if me_std_is_vec
            σ²       = me_std .^ 2
            d_residual = (.-residual ./ σ²) .* Δllh
            d_me_std .+= ((.-one(eltype(me_std)) ./ me_std) .+ (residual .^ 2) ./ (me_std .^ 3)) .* Δllh
        else
            σ²         = me_std^2
            d_residual = (-residual ./ σ²) .* Δllh
            d_me_std  += (-n/me_std + sum(abs2, residual) / me_std^3) * Δllh
        end
        d_obs_dev   = -d_residual
        d_SS_obs  .-= d_residual
        d_new_state = copy(d_cur_state_next)
        @inbounds for k in eachindex(obs_indices)
            d_new_state[obs_indices[k]] += d_obs_dev[k]
        end
        # Linear: 𝐒₁ * aug
        d_𝐒₁ .+= d_new_state * aug'
        d_aug = 𝐒₁' * d_new_state
        # Quadratic: 𝐒₂ * kaug / 2
        d_𝐒₂ .+= (d_new_state * kaug') ./ 2
        d_kaug = (𝐒₂' * d_new_state) ./ 2
        # Cubic: 𝐒₃ * kron(kaug, aug) / 6
        d_𝐒₃ .+= (d_new_state * kron(kaug, aug)') ./ 6
        d_kaug3 = (𝐒₃' * d_new_state) ./ 6
        # ∂kron(kaug, aug) → ∂kaug and ∂aug
        # Using convention: kron(A,B)[(i-1)*nB+j] = A[i]*B[j];
        # reshape(d_k, nB, nA)[j,i] gives the gradient; d_A = mat' * B, d_B = mat * A.
        d_kaug3_mat = reshape(d_kaug3, n_aug, n_aug^2)   # nB=n_aug, nA=n_aug^2
        d_kaug .+= d_kaug3_mat' * aug                    # gradient wrt outer A=kaug
        d_aug  .+= d_kaug3_mat * kaug                    # gradient wrt inner B=aug
        # ∂kron(aug, aug) → ∂aug (×2 via symmetric outer product)
        G = reshape(d_kaug, n_aug, n_aug)
        d_aug .+= (G + G') * aug
        # Split aug back
        d_past, d_shock = split_aug_adjoint(d_aug, npast, nExo)
        d_shocks[:, t] .= d_shock
        d_cur_state_next = zeros(eltype(d_aug), nVars)
        @inbounds for k in eachindex(past_idx)
            d_cur_state_next[past_idx[k]] += d_past[k]
        end
    end
    return d_𝐒₁, d_𝐒₂, d_𝐒₃, d_cur_state_next, d_SS_obs, d_shocks, d_me_std
end

function filter_free_pullback_pruned3rd(
        Δllh::Real, intermediates, 𝐒₁, 𝐒₂, 𝐒₃, past_idx, obs_indices,
        nVars::Int, npast::Int, nExo::Int, nT::Int,
        me_std,
    )
    me_std_is_vec = me_std isa AbstractVector
    d_𝐒₁  = zeros(eltype(𝐒₁), size(𝐒₁))
    d_𝐒₂  = zeros(eltype(𝐒₂), size(𝐒₂))
    d_𝐒₃  = zeros(eltype(𝐒₃), size(𝐒₃))
    n_aug = npast + 1 + nExo
    d_shocks = zeros(eltype(intermediates[1].aug₁), nExo, nT)
    d_SS_obs = zeros(eltype(intermediates[1].aug₁), length(obs_indices))
    d_me_std = me_std_is_vec ? zero(me_std) : zero(eltype(me_std))
    d_cur_state_next = [zeros(eltype(intermediates[1].aug₁), nVars),
                        zeros(eltype(intermediates[1].aug₁), nVars),
                        zeros(eltype(intermediates[1].aug₁), nVars)]

    @inbounds for t in nT:-1:1
        it       = intermediates[t]
        residual = it.residual
        aug₁     = it.aug₁
        aug₁̂    = it.aug₁̂
        aug₂     = it.aug₂
        aug₃     = it.aug₃
        kaug₁    = it.kaug₁
        n        = length(residual)
        if me_std_is_vec
            σ²       = me_std .^ 2
            d_residual = (.-residual ./ σ²) .* Δllh
            d_me_std .+= ((.-one(eltype(me_std)) ./ me_std) .+ (residual .^ 2) ./ (me_std .^ 3)) .* Δllh
        else
            σ²         = me_std^2
            d_residual = (-residual ./ σ²) .* Δllh
            d_me_std  += (-n/me_std + sum(abs2, residual) / me_std^3) * Δllh
        end
        d_obs_dev = -d_residual
        d_SS_obs .-= d_residual
        d_new_1 = copy(d_cur_state_next[1])
        d_new_2 = copy(d_cur_state_next[2])
        d_new_3 = copy(d_cur_state_next[3])
        @inbounds for k in eachindex(obs_indices)
            d_new_1[obs_indices[k]] += d_obs_dev[k]
            d_new_2[obs_indices[k]] += d_obs_dev[k]
            d_new_3[obs_indices[k]] += d_obs_dev[k]
        end
        # Component 1: y_new = 𝐒₁ * aug₁
        d_𝐒₁ .+= d_new_1 * aug₁'
        d_aug₁ = 𝐒₁' * d_new_1
        # Component 2: δ_new = 𝐒₁ * aug₂ + 𝐒₂ * kron(aug₁, aug₁) / 2
        d_𝐒₁ .+= d_new_2 * aug₂'
        d_aug₂ = 𝐒₁' * d_new_2
        d_aug₁_from_2quad, d_𝐒₂_t = quad_adjoint(𝐒₂, aug₁, d_new_2)
        d_𝐒₂ .+= d_𝐒₂_t
        d_aug₁ .+= d_aug₁_from_2quad
        # Component 3: ξ_new = 𝐒₁ * aug₃ + 𝐒₂ * kron(aug₁̂, aug₂) + 𝐒₃ * kron(kaug₁, aug₁) / 6
        d_𝐒₁ .+= d_new_3 * aug₃'
        d_aug₃ = 𝐒₁' * d_new_3
        # 𝐒₂ * kron(aug₁̂, aug₂)  (no /2 factor here)
        k12 = kron(aug₁̂, aug₂)
        d_𝐒₂ .+= d_new_3 * k12'
        d_k12 = 𝐒₂' * d_new_3
        # reshape(d_k12, len(B)=n_aug, len(A)=n_aug); d_A=mat'*B, d_B=mat*A
        d_k12_mat = reshape(d_k12, n_aug, n_aug)
        d_aug₁̂  = d_k12_mat' * aug₂
        d_aug₂ .+= d_k12_mat * aug₁̂
        # 𝐒₃ * kron(kaug₁, aug₁) / 6
        kaug3 = kron(kaug₁, aug₁)
        d_𝐒₃ .+= (d_new_3 * kaug3') ./ 6
        d_kaug3 = (𝐒₃' * d_new_3) ./ 6
        # kron(kaug₁, aug₁): A=kaug₁ (n²), B=aug₁ (n); reshape (n, n²)
        d_kaug3_mat = reshape(d_kaug3, n_aug, n_aug^2)
        d_kaug₁_from_3 = d_kaug3_mat' * aug₁
        d_aug₁_from_3 = d_kaug3_mat * kaug₁
        d_aug₁ .+= d_aug₁_from_3
        # ∂kron(aug₁, aug₁) → ∂aug₁ (symmetric)
        G1 = reshape(d_kaug₁_from_3, n_aug, n_aug)
        d_aug₁ .+= (G1 + G1') * aug₁
        # Combine aug₁̂ into aug₁: aug₁̂ shares past_idx and shock with aug₁, constant slot is 0
        d_aug₁[1:npast] .+= d_aug₁̂[1:npast]
        d_aug₁[npast+2:npast+1+nExo] .+= d_aug₁̂[npast+2:npast+1+nExo]
        # Split each augmented adjoint
        d_past₁, d_shock = split_aug_adjoint(d_aug₁, npast, nExo)
        d_past₂, _       = split_aug_adjoint(d_aug₂, npast, nExo)
        d_past₃, _       = split_aug_adjoint(d_aug₃, npast, nExo)
        d_shocks[:, t] .= d_shock
        d_cur_state_next = [zeros(eltype(d_aug₁), nVars),
                            zeros(eltype(d_aug₁), nVars),
                            zeros(eltype(d_aug₁), nVars)]
        @inbounds for k in eachindex(past_idx)
            d_cur_state_next[1][past_idx[k]] += d_past₁[k]
            d_cur_state_next[2][past_idx[k]] += d_past₂[k]
            d_cur_state_next[3][past_idx[k]] += d_past₃[k]
        end
    end
    return d_𝐒₁, d_𝐒₂, d_𝐒₃, d_cur_state_next, d_SS_obs, d_shocks, d_me_std
end


function rrule(::typeof(get_filter_free_loglikelihood),
                𝓂::ℳ,
                data::KeyedArray{Float64},
                parameter_values::Vector{S},
                shocks::AbstractMatrix{T},
                measurement_error_std::Union{T, AbstractVector{T}};
                steady_state_function::SteadyStateFunctionType = missing,
                algorithm::Symbol = :second_order,
                on_failure_loglikelihood::U = -Inf,
                tol::Tolerances = Tolerances(),
                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_SELECTOR(𝓂),
                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                verbose::Bool = DEFAULT_VERBOSE,
                caching::Bool = DEFAULT_CACHING,
                use_workspaces::Bool = DEFAULT_USE_WORKSPACES) where {S <: Real, T <: Real, U <: AbstractFloat}

    @assert algorithm ∈ [:first_order, :second_order, :pruned_second_order, :third_order, :pruned_third_order] "rrule for `get_filter_free_loglikelihood` supports `:first_order`, `:second_order`, `:pruned_second_order`, `:third_order`, `:pruned_third_order`."

    R = promote_type(S, T)

    nP = length(parameter_values)
    on_failure = (
        convert(R, on_failure_loglikelihood),
        _ -> (NoTangent(), NoTangent(), NoTangent(), zeros(S, nP), zero(shocks),
              measurement_error_std isa AbstractVector ? zero(measurement_error_std) : zero(T))
    )

    if !caching; invalidate_cache_validity!(𝓂); end
    orig_ws = 𝓂.workspaces
    if !use_workspaces; 𝓂.workspaces = fresh_workspaces(orig_ws); end

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                            sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                            sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                            lyapunov_algorithm = lyapunov_algorithm)

    observables = get_and_check_observables(𝓂.constants.post_model_macro, data)

    solve!(𝓂, opts = opts, steady_state_function = steady_state_function, algorithm = algorithm)

    if check_bounds(parameter_values, 𝓂)
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return on_failure
    end

    me_std_vec = measurement_error_std isa AbstractVector ? measurement_error_std : nothing
    if me_std_vec !== nothing
        if any(x -> !isfinite(x) || x <= zero(T), me_std_vec)
            if !use_workspaces; 𝓂.workspaces = orig_ws; end
            return on_failure
        end
    else
        if !isfinite(measurement_error_std) || measurement_error_std <= zero(T)
            if !use_workspaces; 𝓂.workspaces = orig_ws; end
            return on_failure
        end
    end

    # Capture rrule of solve so we can re-use its pullback later.
    ss_y, ss_pb = rrule(get_relevant_steady_state_and_state_update,
                        Val(algorithm), parameter_values, 𝓂;
                        opts = opts, estimation = true)
    _, SS_and_pars, 𝐒, state, solved = ss_y

    if !solved
        if !use_workspaces; 𝓂.workspaces = orig_ws; end
        return on_failure
    end

    if collect(axiskeys(data,1)) isa Vector{String}
        data = rekey(data, 1 => axiskeys(data,1) .|> Meta.parse .|> replace_indices)
    end

    SS_and_pars_names = 𝓂.constants.post_complete_parameters.SS_and_pars_names
    obs_indices       = convert(Vector{Int}, indexin(observables, SS_and_pars_names))
    dt                = collect(data(observables))
    data_in_deviations = dt .- SS_and_pars[obs_indices]

    nExo  = 𝓂.constants.post_model_macro.nExo
    past_idx = 𝓂.constants.post_model_macro.past_not_future_and_mixed_idx
    npast = length(past_idx)
    nT    = size(data_in_deviations, 2)

    @assert size(shocks, 1) == nExo
    @assert size(shocks, 2) == nT

    # Keep only the rows of the policy functions strictly required to
    # propagate the state (past_idx slots) and to form the residual
    # (observable rows). Everything else is discarded for the forward
    # recursion and re-inserted as zero cotangents before the captured
    # solve-pullback is invoked.
    needed = sort(unique(vcat(past_idx, obs_indices)))
    past_in_needed = convert(Vector{Int}, indexin(past_idx, needed))
    obs_in_needed  = convert(Vector{Int}, indexin(obs_indices, needed))
    nNeeded = length(needed)

    llh = zero(R)

    if algorithm == :first_order
        𝐒₁_full = Matrix(𝐒)
        nVars_full = size(𝐒₁_full, 1)
        ncols₁ = size(𝐒₁_full, 2)
        𝐒₁_mat = 𝐒₁_full[needed, :]
        intermediates = Vector{NamedTuple{(:aug, :new_state, :residual),
                                          Tuple{Vector{R}, Vector{R}, Vector{R}}}}(undef, nT)
        cur_state = convert(Vector{R}, state[1])[needed]
        @inbounds for t in 1:nT
            aug = vcat(cur_state[past_in_needed], Vector{R}(shocks[:, t]))
            new_state = 𝐒₁_mat * aug
            residual  = data_in_deviations[:, t] - new_state[obs_in_needed]
            llh += filter_free_obs_logpdf(residual, measurement_error_std)
            intermediates[t] = (; aug = aug, new_state = new_state, residual = residual)
            cur_state = new_state
        end

        if !use_workspaces; 𝓂.workspaces = orig_ws; end

        pullback = function (Δ)
            Δllh = unthunk(Δ)
            if Δllh isa AbstractZero
                return NoTangent(), NoTangent(), NoTangent(), zeros(S, nP), zero(shocks),
                       measurement_error_std isa AbstractVector ? zero(measurement_error_std) : zero(T)
            end
            d_𝐒₁_red, d_state_red, d_SS_obs, d_shocks, d_me_std =
                filter_free_pullback_1st(Δllh, intermediates, 𝐒₁_mat,
                                          past_in_needed, obs_in_needed,
                                          nNeeded, npast, nExo, nT,
                                          measurement_error_std)
            d_𝐒₁_full_cot = zeros(eltype(d_𝐒₁_red), nVars_full, ncols₁)
            @inbounds d_𝐒₁_full_cot[needed, :] .= d_𝐒₁_red
            d_SS_and_pars = zeros(eltype(d_SS_obs), length(SS_and_pars))
            @inbounds for k in eachindex(obs_indices)
                d_SS_and_pars[obs_indices[k]] += d_SS_obs[k]
            end
            # first_order ss rrule expects bare 𝐒₁ cotangent and ignores Δstate
            ss_grads = ss_pb((NoTangent(), d_SS_and_pars, d_𝐒₁_full_cot, NoTangent()))
            d_params = ss_grads[3]
            return NoTangent(), NoTangent(), NoTangent(), d_params, d_shocks, d_me_std
        end
        return isfinite(llh) ? (llh, pullback) : on_failure

    elseif algorithm == :second_order
        𝐒₁_full = Matrix(𝐒[1])
        𝐒₂_full = Matrix(𝐒[2])
        nVars_full = size(𝐒₁_full, 1)
        ncols₁ = size(𝐒₁_full, 2)
        ncols₂ = size(𝐒₂_full, 2)
        𝐒₁ = 𝐒₁_full[needed, :]
        𝐒₂ = 𝐒₂_full[needed, :]
        intermediates = Vector{NamedTuple{(:aug, :new_state, :residual),
                                          Tuple{Vector{R}, Vector{R}, Vector{R}}}}(undef, nT)
        cur_state = convert(Vector{R}, state)[needed]
        @inbounds for t in 1:nT
            aug = vcat(cur_state[past_in_needed], one(R), Vector{R}(shocks[:, t]))
            new_state = 𝐒₁ * aug + (𝐒₂ * kron(aug, aug)) ./ R(2)
            residual  = data_in_deviations[:, t] - new_state[obs_in_needed]
            llh += filter_free_obs_logpdf(residual, measurement_error_std)
            intermediates[t] = (; aug = aug, new_state = new_state, residual = residual)
            cur_state = new_state
        end

        if !use_workspaces; 𝓂.workspaces = orig_ws; end

        pullback = function (Δ)
            Δllh = unthunk(Δ)
            if Δllh isa AbstractZero
                return NoTangent(), NoTangent(), NoTangent(), zeros(S, nP), zero(shocks),
                       measurement_error_std isa AbstractVector ? zero(measurement_error_std) : zero(T)
            end
            d_𝐒₁_red, d_𝐒₂_red, d_state_red, d_SS_obs, d_shocks, d_me_std =
                filter_free_pullback_2nd(Δllh, intermediates, 𝐒₁, 𝐒₂,
                                          past_in_needed, obs_in_needed,
                                          nNeeded, npast, nExo, nT,
                                          measurement_error_std)
            d_𝐒₁ = zeros(eltype(d_𝐒₁_red), nVars_full, ncols₁); @inbounds d_𝐒₁[needed, :] .= d_𝐒₁_red
            d_𝐒₂ = zeros(eltype(d_𝐒₂_red), nVars_full, ncols₂); @inbounds d_𝐒₂[needed, :] .= d_𝐒₂_red
            d_state = zeros(eltype(d_state_red), nVars_full); @inbounds d_state[needed] .= d_state_red
            d_SS_and_pars = zeros(eltype(d_SS_obs), length(SS_and_pars))
            @inbounds for k in eachindex(obs_indices)
                d_SS_and_pars[obs_indices[k]] += d_SS_obs[k]
            end
            ss_grads = ss_pb((NoTangent(), d_SS_and_pars, [d_𝐒₁, d_𝐒₂], d_state, NoTangent()))
            d_params = ss_grads[3]
            return NoTangent(), NoTangent(), NoTangent(), d_params, d_shocks, d_me_std
        end
        return isfinite(llh) ? (llh, pullback) : on_failure

    elseif algorithm == :pruned_second_order
        𝐒₁_full = Matrix(𝐒[1])
        𝐒₂_full = Matrix(𝐒[2])
        nVars_full = size(𝐒₁_full, 1)
        ncols₁ = size(𝐒₁_full, 2)
        ncols₂ = size(𝐒₂_full, 2)
        𝐒₁ = 𝐒₁_full[needed, :]
        𝐒₂ = 𝐒₂_full[needed, :]
        intermediates = Vector{NamedTuple{(:aug₁, :aug₂, :new_state, :residual),
                                          Tuple{Vector{R}, Vector{R}, Vector{Vector{R}}, Vector{R}}}}(undef, nT)
        cur_state = [convert(Vector{R}, state[1])[needed], convert(Vector{R}, state[2])[needed]]
        @inbounds for t in 1:nT
            ϵ = Vector{R}(shocks[:, t])
            aug₁ = vcat(cur_state[1][past_in_needed], one(R), ϵ)
            aug₂ = vcat(cur_state[2][past_in_needed], zero(R), zeros(R, nExo))
            new1 = 𝐒₁ * aug₁
            new2 = 𝐒₁ * aug₂ + (𝐒₂ * kron(aug₁, aug₁)) ./ R(2)
            new_state = [new1, new2]
            residual  = data_in_deviations[:, t] - (new1[obs_in_needed] + new2[obs_in_needed])
            llh += filter_free_obs_logpdf(residual, measurement_error_std)
            intermediates[t] = (; aug₁ = aug₁, aug₂ = aug₂, new_state = new_state, residual = residual)
            cur_state = new_state
        end

        if !use_workspaces; 𝓂.workspaces = orig_ws; end

        pullback = function (Δ)
            Δllh = unthunk(Δ)
            if Δllh isa AbstractZero
                return NoTangent(), NoTangent(), NoTangent(), zeros(S, nP), zero(shocks),
                       measurement_error_std isa AbstractVector ? zero(measurement_error_std) : zero(T)
            end
            d_𝐒₁_red, d_𝐒₂_red, d_state_red, d_SS_obs, d_shocks, d_me_std =
                filter_free_pullback_pruned2nd(Δllh, intermediates, 𝐒₁, 𝐒₂,
                                                past_in_needed, obs_in_needed,
                                                nNeeded, npast, nExo, nT,
                                                measurement_error_std)
            d_𝐒₁ = zeros(eltype(d_𝐒₁_red), nVars_full, ncols₁); @inbounds d_𝐒₁[needed, :] .= d_𝐒₁_red
            d_𝐒₂ = zeros(eltype(d_𝐒₂_red), nVars_full, ncols₂); @inbounds d_𝐒₂[needed, :] .= d_𝐒₂_red
            d_state = [zeros(eltype(d_state_red[1]), nVars_full),
                       zeros(eltype(d_state_red[2]), nVars_full)]
            @inbounds d_state[1][needed] .= d_state_red[1]
            @inbounds d_state[2][needed] .= d_state_red[2]
            d_SS_and_pars = zeros(eltype(d_SS_obs), length(SS_and_pars))
            @inbounds for k in eachindex(obs_indices)
                d_SS_and_pars[obs_indices[k]] += d_SS_obs[k]
            end
            ss_grads = ss_pb((NoTangent(), d_SS_and_pars, [d_𝐒₁, d_𝐒₂], d_state, NoTangent()))
            d_params = ss_grads[3]
            return NoTangent(), NoTangent(), NoTangent(), d_params, d_shocks, d_me_std
        end
        return isfinite(llh) ? (llh, pullback) : on_failure

    elseif algorithm == :third_order
        𝐒₁_full = Matrix(𝐒[1])
        𝐒₂_full = Matrix(𝐒[2])
        𝐒₃_full = Matrix(𝐒[3])
        nVars_full = size(𝐒₁_full, 1)
        ncols₁ = size(𝐒₁_full, 2)
        ncols₂ = size(𝐒₂_full, 2)
        ncols₃ = size(𝐒₃_full, 2)
        𝐒₁ = 𝐒₁_full[needed, :]
        𝐒₂ = 𝐒₂_full[needed, :]
        𝐒₃ = 𝐒₃_full[needed, :]
        intermediates = Vector{NamedTuple{(:aug, :kaug, :new_state, :residual),
                                          Tuple{Vector{R}, Vector{R}, Vector{R}, Vector{R}}}}(undef, nT)
        cur_state = convert(Vector{R}, state)[needed]
        @inbounds for t in 1:nT
            aug = vcat(cur_state[past_in_needed], one(R), Vector{R}(shocks[:, t]))
            kaug = kron(aug, aug)
            new_state = 𝐒₁ * aug + (𝐒₂ * kaug) ./ R(2) + (𝐒₃ * kron(kaug, aug)) ./ R(6)
            residual  = data_in_deviations[:, t] - new_state[obs_in_needed]
            llh += filter_free_obs_logpdf(residual, measurement_error_std)
            intermediates[t] = (; aug = aug, kaug = kaug, new_state = new_state, residual = residual)
            cur_state = new_state
        end

        if !use_workspaces; 𝓂.workspaces = orig_ws; end

        pullback = function (Δ)
            Δllh = unthunk(Δ)
            if Δllh isa AbstractZero
                return NoTangent(), NoTangent(), NoTangent(), zeros(S, nP), zero(shocks),
                       measurement_error_std isa AbstractVector ? zero(measurement_error_std) : zero(T)
            end
            d_𝐒₁_red, d_𝐒₂_red, d_𝐒₃_red, d_state_red, d_SS_obs, d_shocks, d_me_std =
                filter_free_pullback_3rd(Δllh, intermediates, 𝐒₁, 𝐒₂, 𝐒₃,
                                          past_in_needed, obs_in_needed,
                                          nNeeded, npast, nExo, nT,
                                          measurement_error_std)
            d_𝐒₁ = zeros(eltype(d_𝐒₁_red), nVars_full, ncols₁); @inbounds d_𝐒₁[needed, :] .= d_𝐒₁_red
            d_𝐒₂ = zeros(eltype(d_𝐒₂_red), nVars_full, ncols₂); @inbounds d_𝐒₂[needed, :] .= d_𝐒₂_red
            d_𝐒₃ = zeros(eltype(d_𝐒₃_red), nVars_full, ncols₃); @inbounds d_𝐒₃[needed, :] .= d_𝐒₃_red
            d_state = zeros(eltype(d_state_red), nVars_full); @inbounds d_state[needed] .= d_state_red
            d_SS_and_pars = zeros(eltype(d_SS_obs), length(SS_and_pars))
            @inbounds for k in eachindex(obs_indices)
                d_SS_and_pars[obs_indices[k]] += d_SS_obs[k]
            end
            ss_grads = ss_pb((NoTangent(), d_SS_and_pars, [d_𝐒₁, d_𝐒₂, d_𝐒₃], d_state, NoTangent()))
            d_params = ss_grads[3]
            return NoTangent(), NoTangent(), NoTangent(), d_params, d_shocks, d_me_std
        end
        return isfinite(llh) ? (llh, pullback) : on_failure

    else  # :pruned_third_order
        𝐒₁_full = Matrix(𝐒[1])
        𝐒₂_full = Matrix(𝐒[2])
        𝐒₃_full = Matrix(𝐒[3])
        nVars_full = size(𝐒₁_full, 1)
        ncols₁ = size(𝐒₁_full, 2)
        ncols₂ = size(𝐒₂_full, 2)
        ncols₃ = size(𝐒₃_full, 2)
        𝐒₁ = 𝐒₁_full[needed, :]
        𝐒₂ = 𝐒₂_full[needed, :]
        𝐒₃ = 𝐒₃_full[needed, :]
        intermediates = Vector{NamedTuple{(:aug₁, :aug₁̂, :aug₂, :aug₃, :kaug₁, :new_state, :residual),
                                          Tuple{Vector{R}, Vector{R}, Vector{R}, Vector{R}, Vector{R}, Vector{Vector{R}}, Vector{R}}}}(undef, nT)
        cur_state = [convert(Vector{R}, state[1])[needed],
                     convert(Vector{R}, state[2])[needed],
                     convert(Vector{R}, state[3])[needed]]
        @inbounds for t in 1:nT
            ϵ = Vector{R}(shocks[:, t])
            aug₁  = vcat(cur_state[1][past_in_needed], one(R), ϵ)
            aug₁̂ = vcat(cur_state[1][past_in_needed], zero(R), ϵ)
            aug₂  = vcat(cur_state[2][past_in_needed], zero(R), zeros(R, nExo))
            aug₃  = vcat(cur_state[3][past_in_needed], zero(R), zeros(R, nExo))
            kaug₁ = kron(aug₁, aug₁)
            new1 = 𝐒₁ * aug₁
            new2 = 𝐒₁ * aug₂ + (𝐒₂ * kaug₁) ./ R(2)
            new3 = 𝐒₁ * aug₃ + 𝐒₂ * kron(aug₁̂, aug₂) + (𝐒₃ * kron(kaug₁, aug₁)) ./ R(6)
            new_state = [new1, new2, new3]
            residual  = data_in_deviations[:, t] - (new1[obs_in_needed] + new2[obs_in_needed] + new3[obs_in_needed])
            llh += filter_free_obs_logpdf(residual, measurement_error_std)
            intermediates[t] = (; aug₁ = aug₁, aug₁̂ = aug₁̂, aug₂ = aug₂, aug₃ = aug₃,
                                  kaug₁ = kaug₁, new_state = new_state, residual = residual)
            cur_state = new_state
        end

        if !use_workspaces; 𝓂.workspaces = orig_ws; end

        pullback = function (Δ)
            Δllh = unthunk(Δ)
            if Δllh isa AbstractZero
                return NoTangent(), NoTangent(), NoTangent(), zeros(S, nP), zero(shocks),
                       measurement_error_std isa AbstractVector ? zero(measurement_error_std) : zero(T)
            end
            d_𝐒₁_red, d_𝐒₂_red, d_𝐒₃_red, d_state_red, d_SS_obs, d_shocks, d_me_std =
                filter_free_pullback_pruned3rd(Δllh, intermediates, 𝐒₁, 𝐒₂, 𝐒₃,
                                                past_in_needed, obs_in_needed,
                                                nNeeded, npast, nExo, nT,
                                                measurement_error_std)
            d_𝐒₁ = zeros(eltype(d_𝐒₁_red), nVars_full, ncols₁); @inbounds d_𝐒₁[needed, :] .= d_𝐒₁_red
            d_𝐒₂ = zeros(eltype(d_𝐒₂_red), nVars_full, ncols₂); @inbounds d_𝐒₂[needed, :] .= d_𝐒₂_red
            d_𝐒₃ = zeros(eltype(d_𝐒₃_red), nVars_full, ncols₃); @inbounds d_𝐒₃[needed, :] .= d_𝐒₃_red
            d_state = [zeros(eltype(d_state_red[1]), nVars_full),
                       zeros(eltype(d_state_red[2]), nVars_full),
                       zeros(eltype(d_state_red[3]), nVars_full)]
            @inbounds d_state[1][needed] .= d_state_red[1]
            @inbounds d_state[2][needed] .= d_state_red[2]
            @inbounds d_state[3][needed] .= d_state_red[3]
            d_SS_and_pars = zeros(eltype(d_SS_obs), length(SS_and_pars))
            @inbounds for k in eachindex(obs_indices)
                d_SS_and_pars[obs_indices[k]] += d_SS_obs[k]
            end
            ss_grads = ss_pb((NoTangent(), d_SS_and_pars, [d_𝐒₁, d_𝐒₂, d_𝐒₃], d_state, NoTangent()))
            d_params = ss_grads[3]
            return NoTangent(), NoTangent(), NoTangent(), d_params, d_shocks, d_me_std
        end
        return isfinite(llh) ? (llh, pullback) : on_failure
    end
end
