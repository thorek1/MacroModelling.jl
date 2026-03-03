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
#   - Basic operations: mul_reverse_AD!, mat_mult_kron, sparse_preallocated!
#   - Steady states: get_NSSS_and_parameters, calculate_second/third_order_stochastic_steady_state
#   - Derivatives: calculate_jacobian, calculate_hessian, calculate_third_order_derivatives
#   - Solutions: calculate_first/second/third_order_solution
#   - Matrix equations: solve_sylvester_equation, solve_lyapunov_equation
#   - Filters: calculate_loglikelihood, run_kalman_iterations, find_shocks

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
        return (NoTangent(), NoTangent(), dA, dB)
    end

    return ℒ.mul!(C,A,B), times_pullback
end

function rrule(::typeof(mat_mult_kron),
                                A::AbstractSparseMatrix{R},
                                B::AbstractMatrix{T},
                                C::AbstractMatrix{T},
                                D::AbstractMatrix{S}) where {R <: Real, T <: Real, S <: Real}
    Y = mat_mult_kron(A, B, C, D)

    function mat_mult_kron_pullback(Ȳ)
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
        row_to_nzinds = Dict{Int, Vector{Int}}()

        for col in 1:size(A_csc, 2)
            for k in A_csc.colptr[col]:(A_csc.colptr[col + 1] - 1)
                nz_col[k] = col
                r = A_csc.rowval[k]
                push!(get!(row_to_nzinds, r, Int[]), k)
            end
        end

        ∂A_nz = zeros(G, nnzA)
        Abar_vec = zeros(G, size(A_csc, 2))

        for (r, ks) in row_to_nzinds
            fill!(Abar_vec, zero(G))
            @inbounds for k in ks
                Abar_vec[nz_col[k]] = A_csc.nzval[k]
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
            @inbounds for k in ks
                ∂A_nz[k] += vecAbar̄[nz_col[k]]
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

    for i in 1:max_iters
        copyto!(x_aug, 1, x, 1, length(x))
        kron_x_aug = ℒ.kron(x_aug, x_aug)

        ∂x = (A + B * ℒ.kron(x_aug, I_nPast) - I_nPast)

        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            return x, false
        end
        
        Δx = ∂x̂ \ (A * x + B̂ * kron_x_aug / 2 - x)

        if i > 5 && isapprox(A * x + B̂ * kron_x_aug / 2, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    copyto!(x_aug, 1, x, 1, length(x))
    kron_x_aug = ℒ.kron(x_aug, x_aug)
    solved = isapprox(A * x + B̂ * kron_x_aug / 2, x, rtol = tol)         

    # println(x)

    ∂𝐒₁ =  zero(𝐒₁)
    ∂𝐒₂ =  zero(𝐒₂)

    # end # timeit_debug
    # end # timeit_debug

    function second_order_stochastic_steady_state_pullback(∂x)
        # @timeit_debug timer "Calculate SSS - pullback" begin

        S = -∂x[1]' / (A + B * ℒ.kron(x_aug, I_nPast) - I_nPast)

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

    for i in 1:max_iters
        copyto!(x_aug, 1, x, 1, length(x))
        kron_x_aug = ℒ.kron(x_aug, x_aug)
        kron_x_kron = ℒ.kron(x_aug, kron_x_aug)

        ∂x = (A + B * ℒ.kron(x_aug, I_nPast) + C * ℒ.kron(kron_x_aug, I_nPast) / 2 - I_nPast)
        
        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            return x, false
        end
        
        Δx = ∂x̂ \ (A * x + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6 - x)

        if i > 5 && isapprox(A * x + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    copyto!(x_aug, 1, x, 1, length(x))
    kron_x_aug = ℒ.kron(x_aug, x_aug)
    kron_x_kron = ℒ.kron(x_aug, kron_x_aug)
    solved = isapprox(A * x + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6, x, rtol = tol)         

    ∂𝐒₁ =  zero(𝐒₁)
    ∂𝐒₂ =  zero(𝐒₂)
    ∂𝐒₃ =  zero(𝐒₃)

    function third_order_stochastic_steady_state_pullback(∂x)
        S = -∂x[1]' / (A + B * ℒ.kron(x_aug, I_nPast) + C * ℒ.kron(kron_x_aug, I_nPast) / 2 - I_nPast)

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
                jacobian_funcs::jacobian_functions)
    jacobian = calculate_jacobian(parameters, SS_and_pars, caches_obj, jacobian_funcs)

    function calculate_jacobian_pullback(∂∇₁)
        if ∂∇₁ isa Union{NoTangent, AbstractZero}
            return NoTangent(), zero(parameters), zero(SS_and_pars), NoTangent(), NoTangent()
        end

        ∂∇₁u = unthunk(∂∇₁)

        jacobian_funcs.f_parameters(caches_obj.jacobian_parameters, parameters, SS_and_pars)
        jacobian_funcs.f_SS_and_pars(caches_obj.jacobian_SS_and_pars, parameters, SS_and_pars)

        ∂parameters = caches_obj.jacobian_parameters' * vec(∂∇₁u)
        ∂SS_and_pars = caches_obj.jacobian_SS_and_pars' * vec(∂∇₁u)
        return NoTangent(), ∂parameters, ∂SS_and_pars, NoTangent(), NoTangent()
    end

    return jacobian, calculate_jacobian_pullback
end


function rrule(::typeof(calculate_hessian), 
                parameters, 
                SS_and_pars, 
                caches_obj::caches,
                hessian_funcs::hessian_functions)
    hessian = calculate_hessian(parameters, SS_and_pars, caches_obj, hessian_funcs)

    function calculate_hessian_pullback(∂∇₂)
        if ∂∇₂ isa Union{NoTangent, AbstractZero}
            return NoTangent(), zero(parameters), zero(SS_and_pars), NoTangent(), NoTangent()
        end

        ∂∇₂u = unthunk(∂∇₂)

        hessian_funcs.f_parameters(caches_obj.hessian_parameters, parameters, SS_and_pars)
        hessian_funcs.f_SS_and_pars(caches_obj.hessian_SS_and_pars, parameters, SS_and_pars)

        ∂parameters = caches_obj.hessian_parameters' * vec(∂∇₂u)
        ∂SS_and_pars = caches_obj.hessian_SS_and_pars' * vec(∂∇₂u)

        return NoTangent(), ∂parameters, ∂SS_and_pars, NoTangent(), NoTangent()
    end

    return hessian, calculate_hessian_pullback
end


function rrule(::typeof(calculate_third_order_derivatives), 
                parameters, 
                SS_and_pars, 
                caches_obj::caches,
                third_order_derivatives_funcs::third_order_derivatives_functions)
    third_order_derivatives = calculate_third_order_derivatives(parameters, SS_and_pars, caches_obj, third_order_derivatives_funcs)

    function calculate_third_order_derivatives_pullback(∂∇₃)
        if ∂∇₃ isa Union{NoTangent, AbstractZero}
            return NoTangent(), zero(parameters), zero(SS_and_pars), NoTangent(), NoTangent()
        end

        ∂∇₃u = unthunk(∂∇₃)

        third_order_derivatives_funcs.f_parameters(caches_obj.third_order_derivatives_parameters, parameters, SS_and_pars)
        third_order_derivatives_funcs.f_SS_and_pars(caches_obj.third_order_derivatives_SS_and_pars, parameters, SS_and_pars)

        ∂parameters = caches_obj.third_order_derivatives_parameters' * vec(∂∇₃u)
        ∂SS_and_pars = caches_obj.third_order_derivatives_SS_and_pars' * vec(∂∇₃u)

        return NoTangent(), ∂parameters, ∂SS_and_pars, NoTangent(), NoTangent()
    end

    return third_order_derivatives, calculate_third_order_derivatives_pullback
end


function _incremental_cotangent!(Δ, prev_ref::Base.RefValue)
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
    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)

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

        # if !isfinite(solution_error) || solution_error > opts.tol.NSSS_acceptance_tol
        #     throw(ArgumentError("Custom steady state function failed steady state check: residual $solution_error > $(opts.tol.NSSS_acceptance_tol). Parameters: $(parameter_values). Steady state and parameters returned: $(SS_and_pars_tmp)."))
        # end
        X = ms.custom_ss_expand_matrix
        SS_and_pars = X * SS_and_pars_tmp
    else
        fastest_idx = 𝓂.constants.post_complete_parameters.nsss_fastest_solver_parameter_idx
        preferred_solver_parameter_idx = fastest_idx < 1 || fastest_idx > length(DEFAULT_SOLVER_PARAMETERS) ? 1 : fastest_idx
        SS_and_pars, (solution_error, iters) = solve_nsss_wrapper(parameter_values, 𝓂, opts.tol, opts.verbose, cold_start, DEFAULT_SOLVER_PARAMETERS, preferred_solver_parameter_idx = preferred_solver_parameter_idx)
    end

    # end # timeit_debug

    if solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error)
        # Update failed counter
        update_ss_counter!(𝓂.counters, false, estimation = estimation)
        return (SS_and_pars, (solution_error, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # Update success counter
    update_ss_counter!(𝓂.counters, true, estimation = estimation)

    # @timeit_debug timer "Calculate NSSS - pullback" begin

    SS_and_pars_names = ms.SS_and_pars_names
    SS_and_pars_names_lead_lag = ms.SS_and_pars_names_lead_lag

    # unknowns = union(setdiff(𝓂.vars_in_ss_equations, 𝓂.constants.post_model_macro.➕_vars), 𝓂.calibration_equations_parameters)
    unknowns = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.equations.steady_state_aux)),union(𝓂.constants.post_model_macro.parameters_in_equations,𝓂.constants.post_model_macro.➕_vars))))), 𝓂.equations.calibration_parameters))

    ∂ = parameter_values
    C = SS_and_pars[ms.SS_and_pars_no_exo_idx] # [dyn_ss_idx])

    if eltype(𝓂.caches.∂equations_∂parameters) != eltype(parameter_values)
        if 𝓂.caches.∂equations_∂parameters isa SparseMatrixCSC
            jac_buffer = similar(𝓂.caches.∂equations_∂parameters, eltype(parameter_values))
            jac_buffer.nzval .= 0
        else
            jac_buffer = zeros(eltype(parameter_values), size(𝓂.caches.∂equations_∂parameters))
        end
    else
        jac_buffer = 𝓂.caches.∂equations_∂parameters
    end

    𝓂.functions.NSSS_∂equations_∂parameters(jac_buffer, ∂, C)

    ∂SS_equations_∂parameters = jac_buffer

    
    if eltype(𝓂.caches.∂equations_∂SS_and_pars) != eltype(SS_and_pars)
        if 𝓂.caches.∂equations_∂SS_and_pars isa SparseMatrixCSC
            jac_buffer = similar(𝓂.caches.∂equations_∂SS_and_pars, eltype(SS_and_pars))
            jac_buffer.nzval .= 0
        else
            jac_buffer = zeros(eltype(SS_and_pars), size(𝓂.caches.∂equations_∂SS_and_pars))
        end
    else
        jac_buffer = 𝓂.caches.∂equations_∂SS_and_pars
    end

    𝓂.functions.NSSS_∂equations_∂SS_and_pars(jac_buffer, ∂, C)

    ∂SS_equations_∂SS_and_pars = jac_buffer

    ∂SS_equations_∂SS_and_pars_lu = RF.lu(∂SS_equations_∂SS_and_pars, check = false)

    if !ℒ.issuccess(∂SS_equations_∂SS_and_pars_lu)
        return (SS_and_pars, (10.0, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    JVP = -(∂SS_equations_∂SS_and_pars_lu \ ∂SS_equations_∂parameters)#[indexin(SS_and_pars_names, unknowns),:]

    jvp = zeros(length(SS_and_pars_names_lead_lag), length(𝓂.constants.post_complete_parameters.parameters))
    
    for (i,v) in enumerate(SS_and_pars_names)
        if v in unknowns
            jvp[i,:] = JVP[indexin([v], unknowns),:]
        end
    end

    # end # timeit_debug
    # end # timeit_debug

    # try block-gmres here
    function get_non_stochastic_steady_state_pullback(∂SS_and_pars)
        return NoTangent(), NoTangent(), jvp' * ∂SS_and_pars[1], NoTangent()
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

    if solution_error > opts.tol.NSSS_acceptance_tol
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
                        𝓂.functions.jacobian)

    first_out, first_pb = rrule(calculate_first_order_solution,
                                ∇₁,
                                constants_obj,
                                𝓂.workspaces,
                                𝓂.caches;
                                opts = opts,
                                initial_guess = 𝓂.caches.qme_solution)

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

function rrule(::typeof(_prepare_stochastic_steady_state_base_terms),
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

    if solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error)
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

    ms = ensure_model_structure_constants!(constants, 𝓂.equations.calibration_parameters)
    all_SS = expand_steady_state(SS_and_pars, ms)

    ∇₁, jacobian_pullback =
        rrule(calculate_jacobian, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian)

    (𝐒₁_raw, qme_sol, solved), first_order_pullback =
        rrule(calculate_first_order_solution, ∇₁, constants, 𝓂.workspaces, 𝓂.caches;
              opts = opts, initial_guess = 𝓂.caches.qme_solution)

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
        rrule(calculate_hessian, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian)

    (𝐒₂_raw, solved2), second_order_pullback =
        rrule(calculate_second_order_solution, ∇₁, ∇₂, 𝐒₁_raw, 𝓂.constants, 𝓂.workspaces, 𝓂.caches;
              initial_guess = 𝓂.caches.second_order_solution, opts = opts)

    update_perturbation_counter!(𝓂.counters, solved2, estimation = estimation, order = 2)

    𝐔₂ = 𝓂.constants.second_order.𝐔₂
    𝐒₂ = sparse(𝐒₂_raw * 𝐔₂)::SparseMatrixCSC{Float64, Int}

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

    𝐒₁ = [𝐒₁_raw[:, 1:nPast] zeros(nVars) 𝐒₁_raw[:, nPast+1:end]]
    aug_state₁ = sparse([zeros(nPast); 1; zeros(nExo)])
    kron_aug1 = ℒ.kron(aug_state₁, aug_state₁)

    tmp = (T.I_nPast - 𝐒₁[past_idx, 1:nPast])
    tmp̄_lu = ℒ.lu(tmp, check = false)

    if !ℒ.issuccess(tmp̄_lu)
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

    SSSstates = collect(tmp̄_lu \ (𝐒₂ * kron_aug1 / 2)[past_idx])

    common = (true,
              all_SS,
              SS_and_pars,
              solution_error,
              ∇₁,
              ∇₂,
              𝐒₁,
              𝐒₂,
              SSSstates,
              constants)

    pullback = function (Δcommon)
        ∂all_SS = zeros(Float64, length(all_SS))
        ∂SS_and_pars_direct = zeros(Float64, length(SS_and_pars))
        ∂∇₁_direct = zeros(Float64, size(∇₁))
        ∂∇₂_direct = zeros(Float64, size(∇₂))
        ∂𝐒₁_aug = zeros(Float64, size(𝐒₁))
        ∂𝐒₂_total = spzeros(Float64, size(𝐒₂)...)
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
            ∂𝐒₂_total = v8 isa Union{NoTangent, AbstractZero} ? ∂𝐒₂_total : v8
            ∂SSSstates = v9 isa Union{NoTangent, AbstractZero} ? ∂SSSstates : v9
        end

        if !isempty(∂SSSstates)
            ∂rhs = tmp̄_lu' \ ∂SSSstates
            ∂tmp = -(tmp̄_lu' \ ∂SSSstates) * SSSstates'
            ∂𝐒₁_aug[past_idx, 1:nPast] .-= ∂tmp
            ∂𝐒₂_from_rhs = spzeros(Float64, size(𝐒₂)...)
            ∂𝐒₂_from_rhs[past_idx, :] += ∂rhs * kron_aug1' / 2
            ∂𝐒₂_total += ∂𝐒₂_from_rhs
        end

        X = ms.steady_state_expand_matrix
        ∂SS_and_pars_from_allSS = X' * ∂all_SS

        ∂𝐒₁_raw = hcat(∂𝐒₁_aug[:, 1:nPast], ∂𝐒₁_aug[:, nPast+2:end])
        ∂𝐒₂_raw = ∂𝐒₂_total * 𝐔₂'

        so2_tangents = second_order_pullback((∂𝐒₂_raw, NoTangent()))
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
    common, common_pullback = rrule(_prepare_stochastic_steady_state_base_terms,
                                    parameters,
                                    𝓂;
                                    opts = opts,
                                    estimation = estimation)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂, SSSstates, _ = common

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

    so = 𝓂.constants.second_order
    nPast = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed
    kron_s⁺_s⁺ = so.kron_s⁺_s⁺
    A = 𝐒₁[:,1:nPast]
    B̂ = 𝐒₂[:,kron_s⁺_s⁺]

    (SSSstates_final, converged), newton_pullback =
        rrule(solve_stochastic_steady_state_newton, Val(:second_order), 𝐒₁, 𝐒₂, collect(SSSstates), 𝓂)

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
    sss = all_SS + Vector{Float64}(state)
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

        common_tangents = common_pullback((NoTangent(),
                                           Δsss,
                                           ΔSS_and_pars,
                                           NoTangent(),
                                           Δ∇₁,
                                           Δ∇₂,
                                           ∂𝐒₁_from_state + ∂𝐒₁_newton + Δ𝐒₁,
                                           ∂𝐒₂_from_state + ∂𝐒₂_newton + Δ𝐒₂,
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
    common, common_pullback = rrule(_prepare_stochastic_steady_state_base_terms,
                                    parameters,
                                    𝓂;
                                    opts = opts,
                                    estimation = estimation)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂, SSSstates, _ = common

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

    T = 𝓂.constants.post_model_macro
    nPast = T.nPast_not_future_and_mixed
    aug_state₁ = sparse([zeros(nPast); 1; zeros(T.nExo)])
    kron_aug1 = ℒ.kron(aug_state₁, aug_state₁)

    state = 𝐒₁[:,1:nPast] * SSSstates + 𝐒₂ * kron_aug1 / 2
    sss = all_SS + Vector{Float64}(state)
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

        common_tangents = common_pullback((NoTangent(),
                                           Δsss,
                                           ΔSS_and_pars,
                                           NoTangent(),
                                           Δ∇₁,
                                           Δ∇₂,
                                           ∂𝐒₁_from_state + Δ𝐒₁,
                                           ∂𝐒₂_from_state + Δ𝐒₂,
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
    common, common_pullback = rrule(_prepare_stochastic_steady_state_base_terms,
                                    parameters,
                                    𝓂;
                                    opts = opts,
                                    estimation = estimation)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂, SSSstates, _ = common

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

    ∇₃, third_derivatives_pullback =
        rrule(calculate_third_order_derivatives, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives)
    nPast = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed
    𝐒₁_raw = [𝐒₁[:, 1:nPast] 𝐒₁[:, nPast+2:end]]

    (𝐒₃, solved3), third_order_solution_pullback =
        rrule(calculate_third_order_solution, ∇₁, ∇₂, ∇₃, 𝐒₁_raw, 𝐒₂,
              𝓂.constants,
              𝓂.workspaces,
              𝓂.caches;
              initial_guess = 𝓂.caches.third_order_solution,
              opts = opts)

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
    𝐒₃̂ = sparse(𝐒₃ * 𝐔₃)

    so = 𝓂.constants.second_order
    nPast = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed
    kron_s⁺_s⁺ = so.kron_s⁺_s⁺
    kron_s⁺_s⁺_s⁺ = so.kron_s⁺_s⁺_s⁺

    A = 𝐒₁[:,1:nPast]
    B̂ = 𝐒₂[:,kron_s⁺_s⁺]
    Ĉ = 𝐒₃̂[:,kron_s⁺_s⁺_s⁺]

    (SSSstates_final, converged), newton_pullback =
        rrule(solve_stochastic_steady_state_newton, Val(:third_order), 𝐒₁, 𝐒₂, 𝐒₃̂, collect(SSSstates), 𝓂)

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
    sss = all_SS + Vector{Float64}(state)
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
        ∂𝐒₂_from_so3 = so3_tangents[6] isa Union{NoTangent, AbstractZero} ? zero(𝐒₂) : so3_tangents[6]

        ∂𝐒₁_from_so3 = zeros(Float64, size(𝐒₁))
        ∂𝐒₁_from_so3[:, 1:nPast] = ∂𝐒₁_raw_from_so3[:, 1:nPast]
        ∂𝐒₁_from_so3[:, nPast+2:end] = ∂𝐒₁_raw_from_so3[:, nPast+1:end]

        ∂∇₃_total = Δ∇₃ + ∂∇₃_from_so3
        third_derivatives_tangents = third_derivatives_pullback(∂∇₃_total)
        ∂params_from_∇₃ = third_derivatives_tangents[2]
        ∂SS_and_pars_from_∇₃ = third_derivatives_tangents[3]

        common_tangents = common_pullback((NoTangent(),
                                           Δsss,
                                           ΔSS_and_pars + ∂SS_and_pars_from_∇₃,
                                           NoTangent(),
                                           Δ∇₁ + ∂∇₁_from_so3,
                                           Δ∇₂ + ∂∇₂_from_so3,
                                           ∂𝐒₁_from_state + ∂𝐒₁_newton + Δ𝐒₁ + ∂𝐒₁_from_so3,
                                           ∂𝐒₂_from_state + ∂𝐒₂_newton + Δ𝐒₂ + ∂𝐒₂_from_so3,
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
    common, common_pullback = rrule(_prepare_stochastic_steady_state_base_terms,
                                    parameters,
                                    𝓂;
                                    opts = opts,
                                    estimation = estimation)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂, SSSstates, _ = common

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

    ∇₃, third_derivatives_pullback =
        rrule(calculate_third_order_derivatives, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives)
    nPast = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed
    𝐒₁_raw = [𝐒₁[:, 1:nPast] 𝐒₁[:, nPast+2:end]]

    (𝐒₃, solved3), third_order_solution_pullback =
        rrule(calculate_third_order_solution, ∇₁, ∇₂, ∇₃, 𝐒₁_raw, 𝐒₂,
              𝓂.constants,
              𝓂.workspaces,
              𝓂.caches;
              initial_guess = 𝓂.caches.third_order_solution,
              opts = opts)

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
    𝐒₃̂ = sparse(𝐒₃ * 𝐔₃)

    T = 𝓂.constants.post_model_macro
    nPast = T.nPast_not_future_and_mixed
    aug_state₁ = sparse([zeros(nPast); 1; zeros(T.nExo)])
    kron_aug1 = ℒ.kron(aug_state₁, aug_state₁)

    state = 𝐒₁[:,1:nPast] * SSSstates + 𝐒₂ * kron_aug1 / 2
    sss = all_SS + Vector{Float64}(state)
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
        ∂𝐒₂_from_so3 = so3_tangents[6] isa Union{NoTangent, AbstractZero} ? zero(𝐒₂) : so3_tangents[6]

        ∂𝐒₁_from_so3 = zeros(Float64, size(𝐒₁))
        ∂𝐒₁_from_so3[:, 1:nPast] = ∂𝐒₁_raw_from_so3[:, 1:nPast]
        ∂𝐒₁_from_so3[:, nPast+2:end] = ∂𝐒₁_raw_from_so3[:, nPast+1:end]

        ∂∇₃_total = Δ∇₃ + ∂∇₃_from_so3
        third_derivatives_tangents = third_derivatives_pullback(∂∇₃_total)
        ∂params_from_∇₃ = third_derivatives_tangents[2]
        ∂SS_and_pars_from_∇₃ = third_derivatives_tangents[3]

        common_tangents = common_pullback((NoTangent(),
                                           Δsss,
                                           ΔSS_and_pars + ∂SS_and_pars_from_∇₃,
                                           NoTangent(),
                                           Δ∇₁ + ∂∇₁_from_so3,
                                           Δ∇₂ + ∂∇₂_from_so3,
                                           ∂𝐒₁_from_state + Δ𝐒₁ + ∂𝐒₁_from_so3,
                                           ∂𝐒₂_from_state + Δ𝐒₂ + ∂𝐒₂_from_so3,
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

    if !converged || solution_error > opts.tol.NSSS_acceptance_tol
        y = (𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂], collect(sss), converged)
        return y, _ -> (NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent())
    end

    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
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

    if !converged || solution_error > opts.tol.NSSS_acceptance_tol
        y = (𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂], [zeros(S, nVars), zeros(S, nVars)], converged)
        return y, _ -> (NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent())
    end

    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
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

    if !converged || solution_error > opts.tol.NSSS_acceptance_tol
        y = (𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], collect(sss), converged)
        return y, _ -> (NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent())
    end

    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
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

    if !converged || solution_error > opts.tol.NSSS_acceptance_tol
        y = (𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], [zeros(S, nVars), zeros(S, nVars), zeros(S, nVars)], converged)
        return y, _ -> (NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent())
    end

    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
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
                data::KeyedArray{Float64},
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
                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
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

    data_in_deviations = dt .- SS_and_pars[obs_indices]

    # ── step 3: calculate_loglikelihood ──
    llh_rrule = rrule(calculate_loglikelihood,
                      Val(filter), Val(algorithm), obs_indices,
                      𝐒, data_in_deviations, constants_obj, state, 𝓂.workspaces;
                      warmup_iterations = warmup_iterations,
                      presample_periods = presample_periods,
                      initial_covariance = initial_covariance,
                      filter_algorithm = filter_algorithm,
                      opts = opts,
                      on_failure_loglikelihood = on_failure_loglikelihood)

    if llh_rrule === nothing
        llh = calculate_loglikelihood(Val(filter), Val(algorithm), obs_indices,
                    𝐒, data_in_deviations, constants_obj, state, 𝓂.workspaces;
                    warmup_iterations = warmup_iterations,
                    presample_periods = presample_periods,
                    initial_covariance = initial_covariance,
                    filter_algorithm = filter_algorithm,
                    opts = opts,
                    on_failure_loglikelihood = on_failure_loglikelihood)

        return llh, _ -> (NoTangent(), NoTangent(), NoTangent(), zeros(S, length(parameter_values)))
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

    return llh, pullback
end

function rrule(::typeof(get_irf),
                𝓂::ℳ,
                parameters::Vector{S};
                steady_state_function::SteadyStateFunctionType = missing,
                periods::Int = DEFAULT_PERIODS,
                variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_OBC,
                shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = DEFAULT_SHOCK_SELECTION,
                negative_shock::Bool = DEFAULT_NEGATIVE_SHOCK,
                initial_state::Vector{Float64} = DEFAULT_INITIAL_STATE,
                levels::Bool = false,
                verbose::Bool = DEFAULT_VERBOSE,
                tol::Tolerances = Tolerances(),
                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM) where S <: Real

    opts = merge_calculation_options(tol = tol, verbose = verbose,
        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm)

    estimation = true

    constants_obj = initialise_constants!(𝓂)

    solve!(𝓂,
            steady_state_function = steady_state_function,
            opts = opts)

    shocks = 𝓂.constants.post_model_macro.nExo == 0 ? :none : shocks

    shocks, negative_shock, _, periods, shock_idx, shock_history = process_shocks_input(shocks, negative_shock, 1.0, periods, 𝓂)

    var_idx = parse_variables_input_to_index(variables, 𝓂) |> sort

    nVars = 𝓂.constants.post_model_macro.nVars
    nExo  = 𝓂.constants.post_model_macro.nExo
    past_idx = 𝓂.constants.post_model_macro.past_not_future_and_mixed_idx
    nPast = length(past_idx)
    nShocks = shocks == :none ? 1 : length(shock_idx)

    zero_result() = zeros(S, length(var_idx), periods, nShocks)
    zero_pullback(_) = (NoTangent(), NoTangent(), zeros(S, length(parameters)))

    # ── step 1: NSSS ──
    nsss_out, nsss_pb = rrule(get_NSSS_and_parameters,
                                𝓂,
                                parameters;
                                opts = opts,
                                estimation = estimation)

    reference_steady_state = nsss_out[1]
    solution_error = nsss_out[2][1]

    if (solution_error > tol.NSSS_acceptance_tol) || isnan(solution_error)
        return zero_result(), zero_pullback
    end

    # ── step 2: Jacobian ──
    ∇₁, jac_pb = rrule(calculate_jacobian,
                        parameters,
                        reference_steady_state,
                        𝓂.caches,
                        𝓂.functions.jacobian)

    # ── step 3: First-order solution ──
    first_out, first_pb = rrule(calculate_first_order_solution,
                                ∇₁,
                                constants_obj,
                                𝓂.workspaces,
                                𝓂.caches;
                                opts = opts,
                                initial_guess = 𝓂.caches.qme_solution)

    sol_mat = first_out[1]
    solved  = first_out[3]

    update_perturbation_counter!(𝓂.counters, solved, estimation = estimation, order = 1)

    if !solved
        return zero_result(), zero_pullback
    end

    # ── step 4: Forward simulation (mutation-free, storing inputs for pullback) ──
    init_state = initial_state == [0.0] ? zeros(S, nVars) : initial_state - reference_steady_state[1:length(𝓂.constants.post_model_macro.var)]

    # Pre-allocate output and input storage
    Y_all = zeros(S, nVars, periods, nShocks)
    # Store the input vectors [state[past_idx]; shock] for each (shock_i, t) — needed for pullback
    inputs_all = Array{Vector{S}}(undef, nShocks, periods)

    for (si, ii) in enumerate(shock_idx)
        # Build shock history for this shock index
        if shocks isa Union{Symbol_input,String_input}
            shock_hist = zeros(nExo, periods)
            if shocks ≠ :none
                shock_hist[ii, 1] = negative_shock ? -1.0 : 1.0
            end
        else
            shock_hist = shock_history
        end

        # t = 1
        prev_state = init_state
        input_vec = vcat(prev_state[past_idx], shock_hist[:, 1])
        y_t = sol_mat * input_vec
        inputs_all[si, 1] = input_vec
        Y_all[:, 1, si] = y_t

        # t = 2:periods
        for t in 2:periods
            input_vec = vcat(y_t[past_idx], shock_hist[:, t])
            y_t = sol_mat * input_vec
            inputs_all[si, t] = input_vec
            Y_all[:, t, si] = y_t
        end
    end

    # ── step 5: Assemble output ──
    deviations = Y_all[var_idx, :, :]

    result = if levels
        deviations .+ reference_steady_state[var_idx]
    else
        deviations
    end

    # ── step 6: Pullback ──
    pullback = function (∂result_bar)
        ∂result = unthunk(∂result_bar)

        if ∂result isa Union{NoTangent, AbstractZero}
            return NoTangent(), NoTangent(), zeros(S, length(parameters))
        end

        # Scatter var_idx back to full nVars dimension
        ∂Y_all = zeros(S, nVars, periods, nShocks)
        ∂Y_all[var_idx, :, :] .= ∂result

        # SS gradient from levels mode
        ∂SS_and_pars = zeros(S, length(reference_steady_state))
        if levels
            ∂SS_and_pars[var_idx] .+= dropdims(sum(∂result, dims = (2, 3)), dims = (2, 3))
        end

        # BPTT through the linear simulation to get ∂sol_mat
        ∂sol_mat = zeros(S, size(sol_mat))

        for si in 1:nShocks
            # Accumulated gradient flowing backward through states
            ∂y_accum = zeros(S, nVars)

            for t in periods:-1:1
                # Total gradient at time t = direct gradient + propagated from t+1
                ∂y_t = ∂Y_all[:, t, si] .+ ∂y_accum

                # ∂sol_mat += ∂y_t * input_t'
                input_t = inputs_all[si, t]
                ∂sol_mat .+= ∂y_t * input_t'

                # Propagate gradient to previous state through sol_mat
                # input_t = [y_{t-1}[past_idx]; shock_t]
                # ∂input_t = sol_mat' * ∂y_t
                ∂input_t = sol_mat' * ∂y_t

                # Only the first nPast entries of ∂input_t flow to ∂y_{t-1}[past_idx]
                ∂y_accum = zeros(S, nVars)
                ∂y_accum[past_idx] .+= ∂input_t[1:nPast]
            end

            # After BPTT for this shock, ∂y_accum is the gradient w.r.t. init_state.
            # When init_state = initial_state - reference_steady_state[1:nVar],
            # propagate gradient to reference_steady_state with negative sign.
            if initial_state != [0.0]
                nVar_len = length(𝓂.constants.post_model_macro.var)
                ∂SS_and_pars[1:nVar_len] .-= ∂y_accum[1:nVar_len]
            end
        end

        # ── Chain backward through sub-pullbacks ──
        # first_pb expects cotangent tuple: (∂sol_mat, ∂qme_sol, ∂solved)
        first_grads = first_pb((∂sol_mat, NoTangent(), NoTangent()))
        ∂∇₁ = first_grads[2]

        jac_grads = jac_pb(∂∇₁)
        ∂parameters_from_jac = jac_grads[2]
        ∂SS_from_jac = jac_grads[3]

        ∂SS_and_pars .+= ∂SS_from_jac

        nsss_grads = nsss_pb((∂SS_and_pars, NoTangent()))
        ∂parameters_from_nsss = nsss_grads[3]

        ∂parameters_total = ∂parameters_from_jac .+ ∂parameters_from_nsss

        return NoTangent(), NoTangent(), ∂parameters_total
    end

    return result, pullback
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

    if solution_error > opts.tol.NSSS_acceptance_tol
        return (zeros(S, 0, 0), zeros(S, 0, 0), zeros(S, 0, 0), SS_and_pars, false), zero_pb
    end

    # ── Step 2: Jacobian ──
    ∇₁, jac_pb = rrule(calculate_jacobian, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian)

    # ── Step 3: First-order solution ──
    first_out, first_pb = rrule(calculate_first_order_solution,
                                ∇₁,
                                constants_obj,
                                𝓂.workspaces,
                                𝓂.caches;
                                initial_guess = 𝓂.caches.qme_solution,
                                opts = opts)
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
                                lyapunov_algorithm = opts.lyapunov_algorithm,
                                tol = opts.tol.lyapunov_tol,
                                acceptance_tol = opts.tol.lyapunov_acceptance_tol,
                                verbose = opts.verbose)
    covar_raw = lyap_out[1]
    solved_lyap = lyap_out[2]

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
function _kron_vjp(∂C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
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
        solved = solution_error < opts.tol.NSSS_acceptance_tol
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
    if solution_error > opts.tol.NSSS_acceptance_tol
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
    ∇₁, jac_pb = rrule(calculate_jacobian, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian)

    # ── Step 3: First-order solution ──
    first_out, first_pb = rrule(calculate_first_order_solution,
                                ∇₁,
                                constants_obj,
                                𝓂.workspaces,
                                𝓂.caches;
                                initial_guess = 𝓂.caches.qme_solution,
                                opts = opts)
    𝐒₁ = first_out[1]
    solved_first = first_out[3]

    update_perturbation_counter!(𝓂.counters, solved_first, order = 1)

    if !solved_first
        return (SS_and_pars[1:nVars], false), zero_pb
    end

    # ── Step 4: Hessian ──
    ∇₂, hess_pb = rrule(calculate_hessian, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian)

    # ── Step 5: Second-order solution ──
    so2_out, so2_pb = rrule(calculate_second_order_solution, ∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches; opts = opts)
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
        ∂s₁_L, ∂s₁_R = _kron_vjp(∂s₁ks₁, s_to_s₁, s_to_s₁)
        ∂e₁_L, ∂e₁_R = _kron_vjp(∂e₁ke₁, e_to_s₁, e_to_s₁)

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
    ∇₂, hess_pb = rrule(calculate_hessian, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian)

    # ── Step 3: Second-order solution ──
    so2_out, so2_pb = rrule(calculate_second_order_solution, ∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches; opts = opts)
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
    𝐒₂_sp = sparse(𝐒₂_full)

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
            ∂s₁_L, ∂s₁_R = _kron_vjp(∂s₁ks₁, s_to_s₁, s_to_s₁)
            ∂e₁_L, ∂e₁_R = _kron_vjp(∂e₁ke₁, e_to_s₁, e_to_s₁)

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
    ∇₂, hess_pb = rrule(calculate_hessian, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian)

    # ── Step 3: Second-order solution ──
    so2_out, so2_pb = rrule(calculate_second_order_solution, ∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches; opts = opts)
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
                              lyapunov_algorithm = opts.lyapunov_algorithm,
                              tol = opts.tol.lyapunov_tol,
                              acceptance_tol = opts.tol.lyapunov_acceptance_tol,
                              verbose = opts.verbose)
    Σᶻ₂ = lyap_out[1]
    info = lyap_out[2]

    if !info; return zero_15(), zero_pb; end

    Σʸ₂ = ŝ_to_y₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_y₂ * Γ₂ * ê_to_y₂'
    autocorr_tmp = ŝ_to_ŝ₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_ŝ₂ * Γ₂ * ê_to_y₂'

    slvd = solved && solved2 && info
    𝐒₂_sp = sparse(𝐒₂_full)

    result = (Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂_sp, ∇₂, slvd)

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
        if !(∂𝐒₂_pass   isa AbstractZero); ∂S2f     .+= ∂𝐒₂_pass;   end
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
        ∂Σᶻ₁_from_Γ₂, _ = _kron_vjp(∂Γ₂_br, Σᶻ₁, Iₑ)
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
        ∂s₁_L, ∂s₁_R = _kron_vjp(∂s₁ks₁_from_ŝŝ, s_to_s₁, s_to_s₁)
        ∂e₁ke₁_total = ∂e₁ke₁_from_ŝv .+ ∂e₁ke₁_from_ê
        ∂e₁_L, ∂e₁_R = _kron_vjp(∂e₁ke₁_total, e_to_s₁, e_to_s₁)
        ∂s₁_se_L, ∂e₁_se_R = _kron_vjp(∂s₁ke₁_from_ê, s_to_s₁, e_to_s₁)

        # Aggregate into 𝐒₁
        ∂𝐒₁_acc[iˢ, 1:nˢ]     .+= ∂s₁_from_ŝŝ .+ ∂s₁_L .+ ∂s₁_R .+ ∂s₁_se_L
        ∂𝐒₁_acc[iˢ, nˢ+1:end] .+= ∂e₁_L .+ ∂e₁_R .+ ∂e₁_se_R
        ∂S2f[iˢ, kron_s_s]    .+= ∂ss2_from_ŝŝ

        # ── Σᶻ₁ → Σʸ₁ ──
        ∂Σʸ₁ = zeros(S, size(Σʸ₁))
        ∂Σʸ₁[iˢ, iˢ] .= ∂Σᶻ₁_acc
        if !(∂Σʸ₁_pass isa AbstractZero); ∂Σʸ₁ .+= ∂Σʸ₁_pass; end

        # ── S₂_full → S₂_raw ──
        ∂S2_raw = ∂S2f * 𝐔₂'

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
    Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr_tmp_2, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = som2_out

    if !solved; return zero_4(), zero_pb; end

    # ── Step 2: Third-order derivatives ──
    ∇₃, ∇₃_pb = rrule(calculate_third_order_derivatives, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives)

    # ── Step 3: Third-order solution ──
    so3_out, so3_pb = rrule(calculate_third_order_solution, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂,
                            𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                            initial_guess = 𝓂.caches.third_order_solution,
                            opts = opts)
    𝐒₃, solved3 = so3_out

    update_perturbation_counter!(𝓂.counters, solved3, order = 3)

    if !solved3; return zero_4(), zero_pb; end

    # ── Step 4: Decompress S₃ ──
    𝐔₃ = 𝓂.constants.third_order.𝐔₃
    𝐒₃_full = 𝐒₃ * 𝐔₃

    if !(typeof(𝐒₃_full) <: AbstractSparseMatrix)
        𝐒₃_full = sparse(𝐒₃_full)
    end

    # ── Step 5: Determine iteration groups ──
    orders = determine_efficient_order(𝐒₁, 𝐒₂, 𝐒₃_full, 𝓂.constants, observables,
                                       covariance = covariance, tol = opts.tol.dependencies_tol)

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

        # Set up pruned state transition matrices
        ŝ_to_ŝ₃ = [  s_to_s₁                zeros(nˢ, 2*nˢ + 2*nˢ^2 + nˢ^3)
                                            zeros(nˢ, nˢ) s_to_s₁   s_s_to_s₂ / 2   zeros(nˢ, nˢ + nˢ^2 + nˢ^3)
                                            zeros(nˢ^2, 2 * nˢ)               s_to_s₁_by_s_to_s₁  zeros(nˢ^2, nˢ + nˢ^2 + nˢ^3)
                                            s_v_v_to_s₃ / 2    zeros(nˢ, nˢ + nˢ^2)      s_to_s₁       s_s_to_s₂    s_s_s_to_s₃ / 6
                                            ℒ.kron(s_to_s₁,v_v_to_s₂ / 2)    zeros(nˢ^2, 2*nˢ + nˢ^2)     s_to_s₁_by_s_to_s₁  ℒ.kron(s_to_s₁,s_s_to_s₂ / 2)    
                                            zeros(nˢ^3, 3*nˢ + 2*nˢ^2)   ℒ.kron(s_to_s₁,s_to_s₁_by_s_to_s₁)]

        ê_to_ŝ₃ = [ e_to_s₁   zeros(nˢ,nᵉ^2 + 2*nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        zeros(nˢ,nᵉ)  e_e_to_s₂ / 2   s_e_to_s₂   zeros(nˢ,nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        zeros(nˢ^2,nᵉ)  e_to_s₁_by_e_to_s₁  I_plus_s_s * s_to_s₁_by_e_to_s₁  zeros(nˢ^2, nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        e_v_v_to_s₃ / 2    zeros(nˢ,nᵉ^2 + nᵉ * nˢ)  s_e_to_s₂    s_s_e_to_s₃ / 2    s_e_e_to_s₃ / 2    e_e_e_to_s₃ / 6
                                        ℒ.kron(e_to_s₁, v_v_to_s₂ / 2)    zeros(nˢ^2, nᵉ^2 + nᵉ * nˢ)      s_s * s_to_s₁_by_e_to_s₁    ℒ.kron(s_to_s₁, s_e_to_s₂) + s_s * ℒ.kron(s_s_to_s₂ / 2, e_to_s₁)  ℒ.kron(s_to_s₁, e_e_to_s₂ / 2) + s_s * ℒ.kron(s_e_to_s₂, e_to_s₁)  ℒ.kron(e_to_s₁, e_e_to_s₂ / 2)
                                        zeros(nˢ^3, nᵉ + nᵉ^2 + 2*nᵉ * nˢ) ℒ.kron(s_to_s₁_by_s_to_s₁,e_to_s₁) + ℒ.kron(s_to_s₁, s_s * s_to_s₁_by_e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_s_to_s₁) * e_ss   ℒ.kron(s_to_s₁_by_e_to_s₁,e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_e_to_s₁) * e_es + ℒ.kron(e_to_s₁, s_s * s_to_s₁_by_e_to_s₁) * e_es  ℒ.kron(e_to_s₁,e_to_s₁_by_e_to_s₁)]

        ŝ_to_y₃ = [s_to_y₁ + s_v_v_to_y₃ / 2  s_to_y₁  s_s_to_y₂ / 2   s_to_y₁    s_s_to_y₂     s_s_s_to_y₃ / 6]

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


        Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + 2*nˢ^2 +nˢ^3)
                ℒ.kron(Σ̂ᶻ₁,vec_Iₑ)   zeros(nˢ*nᵉ^2, nˢ + nˢ^2)  ℒ.kron(μˢ₃δμˢ₁',vec_Iₑ)    ℒ.kron(reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec_Iₑ)  ℒ.kron(reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3), vec_Iₑ)
                spzeros(nᵉ^3, 3*nˢ + 2*nˢ^2 +nˢ^3)]

        droptol!(ŝ_to_ŝ₃, eps())
        droptol!(ê_to_ŝ₃, eps())
        droptol!(Eᴸᶻ, eps())
        droptol!(Γ₃, eps())

        A_mat = ê_to_ŝ₃ * Eᴸᶻ * ŝ_to_ŝ₃'
        droptol!(A_mat, eps())

        C_mat = ê_to_ŝ₃ * Γ₃ * ê_to_ŝ₃' + A_mat + A_mat'
        droptol!(C_mat, eps())

        # Ensure third-order lyapunov workspace and solve
        lyap_ws_3rd = ensure_lyapunov_workspace!(𝓂.workspaces, size(ŝ_to_ŝ₃, 1), :third_order)

        lyap_out, lyap_pb_iter = rrule(solve_lyapunov_equation,
                                    Float64.(ŝ_to_ŝ₃), Float64.(C_mat), lyap_ws_3rd,
                                    lyapunov_algorithm = opts.lyapunov_algorithm,
                                    tol = opts.tol.lyapunov_tol,
                                    acceptance_tol = opts.tol.lyapunov_acceptance_tol,
                                    verbose = opts.verbose)
        Σᶻ₃ = lyap_out[1]
        info = lyap_out[2]

        if !info
            return zero_4(), zero_pb
        end

        solved_lyapunov = solved_lyapunov && info

        Σʸ₃tmp = ŝ_to_y₃ * Σᶻ₃ * ŝ_to_y₃' + ê_to_y₃ * Γ₃ * ê_to_y₃' + ê_to_y₃ * Eᴸᶻ * ŝ_to_y₃' + ŝ_to_y₃ * Eᴸᶻ' * ê_to_y₃'

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
            ŝ_to_ŝ₃ = ŝ_to_ŝ₃,
            ê_to_ŝ₃ = ê_to_ŝ₃,
            ŝ_to_y₃ = ŝ_to_y₃,
            ê_to_y₃ = ê_to_y₃,
            Γ₃ = Γ₃,
            Eᴸᶻ = Eᴸᶻ,
            A_mat = A_mat,
            C_mat = C_mat,
            Σᶻ₃ = Σᶻ₃,
            Σʸ₃tmp = Σʸ₃tmp,
            μˢ₃δμˢ₁ = μˢ₃δμˢ₁,
            lyap_pb = lyap_pb_iter,
            I_plus_s_s = I_plus_s_s,
            ss_s = ss_s,
            s_s = s_s,
            e_es = e_es,
            e_ss = e_ss,
        )
    end

    result = (Σʸ₃, μʸ₂, SS_and_pars, solved && solved3 && solved_lyapunov)

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

            # ── Lyapunov adjoint: Σᶻ₃ = lyap(ŝ_to_ŝ₃, C_mat) ──
            lyap_grad = d.lyap_pb((∂Σᶻ₃, NoTangent()))
            ∂ŝ_to_ŝ₃ = lyap_grad[2] isa AbstractZero ? zeros(T, size(d.ŝ_to_ŝ₃)) : Matrix{T}(lyap_grad[2])
            ∂C_mat    = lyap_grad[3] isa AbstractZero ? zeros(T, size(d.C_mat))     : Matrix{T}(lyap_grad[3])

            # ── C_mat = ê_s * Γ₃ * ê_s' + A + A'  where A = ê_s * Eᴸᶻ * ŝ_s' ──
            # ê_s * Γ₃ * ê_s' is AXA': ∂ê += (∂C+∂C') * ê * Γ₃,  ∂Γ₃ += ê' * ∂C * ê
            # A + A' with cotangent ∂C: ∂A = ∂C + ∂C'
            ∂C_sym = ∂C_mat + ∂C_mat'

            ∂ê_to_ŝ₃  = ∂C_sym * (d.ê_to_ŝ₃ * d.Γ₃ + d.ŝ_to_ŝ₃ * Matrix(d.Eᴸᶻ'))
            ∂Γ₃_iter  .+= d.ê_to_ŝ₃' * ∂C_mat * d.ê_to_ŝ₃
            ∂Eᴸᶻ_iter .+= d.ê_to_ŝ₃' * ∂C_sym * d.ŝ_to_ŝ₃
            ∂ŝ_to_ŝ₃  .+= ∂C_sym * d.ê_to_ŝ₃ * Matrix(d.Eᴸᶻ)

            # ── Disaggregate ŝ_to_y₃ → ∂𝐒₁, ∂𝐒₂, ∂𝐒₃ ──
            # ŝ_to_y₃ = [s_to_y₁+svv/2 | s_to_y₁ | ss_to_y₂/2 | s_to_y₁ | ss_to_y₂ | sss_to_y₃/6]
            c = 0
            ∂blk1 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i];      c += nˢ_i
            ∂blk2 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i];      c += nˢ_i
            ∂blk3 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i^2];    c += nˢ_i^2
            ∂blk4 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i];      c += nˢ_i
            ∂blk5 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i^2];    c += nˢ_i^2
            ∂blk6 = ∂ŝ_to_y₃[:, c+1:end]

            ∂𝐒₁_acc[d.obs_in_y, d.dependencies_in_states_idx] .+= ∂blk1 .+ ∂blk2 .+ ∂blk4     # ∂s_to_y₁
            ∂S2f_acc[d.obs_in_y, d.kron_s_s]                  .+= ∂blk3 ./ 2 .+ ∂blk5           # ∂s_s_to_y₂
            ∂S3f_acc[d.obs_in_y, d.kron_s_v_v]                .+= ∂blk1 ./ 2                     # ∂s_v_v_to_y₃
            ∂S3f_acc[d.obs_in_y, d.kron_s_s_s]                .+= ∂blk6 ./ 6                     # ∂s_s_s_to_y₃

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
            sb = cumsum([0, n, n, n^2, n, n^2, n^3])          # ŝ_to_ŝ₃ row/col
            eb = cumsum([0, ne, ne^2, n*ne, n*ne, n^2*ne, n*ne^2, ne^3])  # ê_to_ŝ₃ cols
            gb = eb  # Γ₃ row/col (same block sizes)

            vvh = vv₂ ./ 2;  ssh = ss₂ ./ 2;  eeh = ee₂ ./ 2

            # ── 2a: ŝ_to_ŝ₃ disaggregation ──
            ∂ŝ₃ = ∂ŝ_to_ŝ₃   # already dense Matrix{T}

            # Direct s₁ blocks: (1,1), (2,2), (4,4)
            ∂s₁_l .+= ∂ŝ₃[sb[1]+1:sb[2], sb[1]+1:sb[2]] .+
                       ∂ŝ₃[sb[2]+1:sb[3], sb[2]+1:sb[3]] .+
                       ∂ŝ₃[sb[4]+1:sb[5], sb[4]+1:sb[5]]
            # (2,3) ss₂/2
            ∂ss₂_l .+= ∂ŝ₃[sb[2]+1:sb[3], sb[3]+1:sb[4]] ./ 2
            # (4,5) ss₂
            ∂ss₂_l .+= ∂ŝ₃[sb[4]+1:sb[5], sb[5]+1:sb[6]]
            # (4,1) s_vv₃/2
            ∂S3f_acc[d.iˢ, d.kron_s_v_v] .+= ∂ŝ₃[sb[4]+1:sb[5], sb[1]+1:sb[2]] ./ 2
            # (4,6) sss₃/6
            ∂S3f_acc[d.iˢ, d.kron_s_s_s] .+= ∂ŝ₃[sb[4]+1:sb[5], sb[6]+1:sb[7]] ./ 6
            # (3,3) kron(s₁,s₁)
            tmpL, tmpR = _kron_vjp(Matrix(∂ŝ₃[sb[3]+1:sb[4], sb[3]+1:sb[4]]), s₁, s₁)
            ∂s₁_l .+= tmpL .+ tmpR
            # (5,1) kron(s₁, vv₂/2)
            tmpA, tmpB = _kron_vjp(Matrix(∂ŝ₃[sb[5]+1:sb[6], sb[1]+1:sb[2]]), s₁, vvh)
            ∂s₁_l .+= tmpA;  ∂vv₂_l .+= tmpB ./ 2
            # (5,5) kron(s₁,s₁)
            tmpL, tmpR = _kron_vjp(Matrix(∂ŝ₃[sb[5]+1:sb[6], sb[5]+1:sb[6]]), s₁, s₁)
            ∂s₁_l .+= tmpL .+ tmpR
            # (5,6) kron(s₁, ss₂/2)
            tmpA, tmpB = _kron_vjp(Matrix(∂ŝ₃[sb[5]+1:sb[6], sb[6]+1:sb[7]]), s₁, ssh)
            ∂s₁_l .+= tmpA;  ∂ss₂_l .+= tmpB ./ 2
            # (6,6) kron(s₁, kron(s₁,s₁))
            tmpA, tmpB = _kron_vjp(Matrix(∂ŝ₃[sb[6]+1:sb[7], sb[6]+1:sb[7]]), s₁, s₁²)
            ∂s₁_l .+= tmpA
            tmpL, tmpR = _kron_vjp(tmpB, s₁, s₁)
            ∂s₁_l .+= tmpL .+ tmpR

            # ── 2b: ê_to_ŝ₃ disaggregation ──
            ∂ê₃ = Matrix{T}(∂ê_to_ŝ₃)
            ss_s1e1 = Matrix(d.s_s) * s₁e₁   # pre-compute

            # Row 1: (1,1) e₁
            ∂e₁_l .+= ∂ê₃[sb[1]+1:sb[2], eb[1]+1:eb[2]]
            # Row 2: (2,2) ee₂/2; (2,3) se₂
            ∂ee₂_l .+= ∂ê₃[sb[2]+1:sb[3], eb[2]+1:eb[3]] ./ 2
            ∂se₂_l .+= ∂ê₃[sb[2]+1:sb[3], eb[3]+1:eb[4]]
            # Row 3: (3,2) kron(e₁,e₁)
            tmpL, tmpR = _kron_vjp(Matrix(∂ê₃[sb[3]+1:sb[4], eb[2]+1:eb[3]]), e₁, e₁)
            ∂e₁_l .+= tmpL .+ tmpR
            # (3,3) I_plus_s_s * kron(s₁,e₁)
            ∂k33 = Matrix(d.I_plus_s_s') * Matrix(∂ê₃[sb[3]+1:sb[4], eb[3]+1:eb[4]])
            tmpA, tmpB = _kron_vjp(∂k33, s₁, e₁)
            ∂s₁_l .+= tmpA;  ∂e₁_l .+= tmpB
            # Row 4: direct S₃ slices
            ∂S3f_acc[d.iˢ, d.kron_e_v_v] .+= ∂ê₃[sb[4]+1:sb[5], eb[1]+1:eb[2]] ./ 2
            ∂se₂_l .+= ∂ê₃[sb[4]+1:sb[5], eb[4]+1:eb[5]]
            ∂S3f_acc[d.iˢ, d.kron_s_s_e] .+= ∂ê₃[sb[4]+1:sb[5], eb[5]+1:eb[6]] ./ 2
            ∂S3f_acc[d.iˢ, d.kron_s_e_e] .+= ∂ê₃[sb[4]+1:sb[5], eb[6]+1:eb[7]] ./ 2
            ∂S3f_acc[d.iˢ, d.kron_e_e_e] .+= ∂ê₃[sb[4]+1:sb[5], eb[7]+1:eb[8]] ./ 6
            # Row 5: (5,1) kron(e₁,vv₂/2)
            tmpA, tmpB = _kron_vjp(Matrix(∂ê₃[sb[5]+1:sb[6], eb[1]+1:eb[2]]), e₁, vvh)
            ∂e₁_l .+= tmpA;  ∂vv₂_l .+= tmpB ./ 2
            # (5,4) s_s * kron(s₁,e₁)
            ∂k54 = Matrix(d.s_s') * Matrix(∂ê₃[sb[5]+1:sb[6], eb[4]+1:eb[5]])
            tmpA, tmpB = _kron_vjp(∂k54, s₁, e₁)
            ∂s₁_l .+= tmpA;  ∂e₁_l .+= tmpB
            # (5,5) kron(s₁,se₂) + s_s * kron(ss₂/2, e₁)
            ∂b55 = Matrix(∂ê₃[sb[5]+1:sb[6], eb[5]+1:eb[6]])
            tmpA, tmpB = _kron_vjp(∂b55, s₁, se₂)
            ∂s₁_l .+= tmpA;  ∂se₂_l .+= tmpB
            ∂k55b = Matrix(d.s_s') * ∂b55
            tmpA, tmpB = _kron_vjp(∂k55b, ssh, e₁)
            ∂ss₂_l .+= tmpA ./ 2;  ∂e₁_l .+= tmpB
            # (5,6) kron(s₁,ee₂/2) + s_s * kron(se₂, e₁)
            ∂b56 = Matrix(∂ê₃[sb[5]+1:sb[6], eb[6]+1:eb[7]])
            tmpA, tmpB = _kron_vjp(∂b56, s₁, eeh)
            ∂s₁_l .+= tmpA;  ∂ee₂_l .+= tmpB ./ 2
            ∂k56b = Matrix(d.s_s') * ∂b56
            tmpA, tmpB = _kron_vjp(∂k56b, se₂, e₁)
            ∂se₂_l .+= tmpA;  ∂e₁_l .+= tmpB
            # (5,7) kron(e₁, ee₂/2)
            tmpA, tmpB = _kron_vjp(Matrix(∂ê₃[sb[5]+1:sb[6], eb[7]+1:eb[8]]), e₁, eeh)
            ∂e₁_l .+= tmpA;  ∂ee₂_l .+= tmpB ./ 2
            # Row 6: (6,5) kron(s₁²,e₁) + kron(s₁,s_s*s₁e₁) + kron(e₁,s₁²)*e_ss
            ∂b65 = Matrix(∂ê₃[sb[6]+1:sb[7], eb[5]+1:eb[6]])
            tmpA, tmpB = _kron_vjp(∂b65, s₁², e₁)                    # Term 1
            ∂e₁_l .+= tmpB
            tmpL, tmpR = _kron_vjp(tmpA, s₁, s₁);  ∂s₁_l .+= tmpL .+ tmpR
            tmpA, tmpB = _kron_vjp(∂b65, s₁, ss_s1e1)                # Term 2
            ∂s₁_l .+= tmpA
            tmpC = Matrix(d.s_s') * tmpB
            tmpL, tmpR = _kron_vjp(tmpC, s₁, e₁);  ∂s₁_l .+= tmpL;  ∂e₁_l .+= tmpR
            ∂k65c = ∂b65 * Matrix(d.e_ss')                           # Term 3
            tmpA, tmpB = _kron_vjp(∂k65c, e₁, s₁²)
            ∂e₁_l .+= tmpA
            tmpL, tmpR = _kron_vjp(tmpB, s₁, s₁);  ∂s₁_l .+= tmpL .+ tmpR
            # (6,6) kron(s₁e₁,e₁) + kron(e₁,s₁e₁)*e_es + kron(e₁,s_s*s₁e₁)*e_es
            ∂b66 = Matrix(∂ê₃[sb[6]+1:sb[7], eb[6]+1:eb[7]])
            tmpA, tmpB = _kron_vjp(∂b66, s₁e₁, e₁)                  # Term 1
            ∂e₁_l .+= tmpB
            tmpL, tmpR = _kron_vjp(tmpA, s₁, e₁);  ∂s₁_l .+= tmpL;  ∂e₁_l .+= tmpR
            ∂pre = ∂b66 * Matrix(d.e_es')                            # shared for Terms 2+3
            tmpA, tmpB = _kron_vjp(∂pre, e₁, s₁e₁)                  # Term 2
            ∂e₁_l .+= tmpA
            tmpL, tmpR = _kron_vjp(tmpB, s₁, e₁);  ∂s₁_l .+= tmpL;  ∂e₁_l .+= tmpR
            tmpA, tmpB = _kron_vjp(∂pre, e₁, ss_s1e1)                # Term 3
            ∂e₁_l .+= tmpA
            tmpC = Matrix(d.s_s') * tmpB
            tmpL, tmpR = _kron_vjp(tmpC, s₁, e₁);  ∂s₁_l .+= tmpL;  ∂e₁_l .+= tmpR
            # (6,7) kron(e₁, e₁²)
            tmpA, tmpB = _kron_vjp(Matrix(∂ê₃[sb[6]+1:sb[7], eb[7]+1:eb[8]]), e₁, e₁²)
            ∂e₁_l .+= tmpA
            tmpL, tmpR = _kron_vjp(tmpB, e₁, e₁);  ∂e₁_l .+= tmpL .+ tmpR

            # ── 3a: Γ₃ disaggregation → ∂Σ̂ᶻ₁, ∂Σ̂ᶻ₂, ∂Δ̂μˢ₂ ──
            ∂Γ = Matrix{T}(∂Γ₃_iter)
            vΣ = vec(d.Σ̂ᶻ₁)

            # Row 1: (1,4) kron(Δ̂μˢ₂',Ine)
            ∂tmp14 = _kron_vjp(∂Γ[gb[1]+1:gb[2], gb[4]+1:gb[5]], reshape(d.Δ̂μˢ₂, 1, :), Ine)[1]
            ∂Δ̂μˢ₂_l .+= vec(∂tmp14')
            # (1,5) kron(vec(Σ̂ᶻ₁)',Ine)
            ∂tmp15 = _kron_vjp(∂Γ[gb[1]+1:gb[2], gb[5]+1:gb[6]], reshape(vΣ, 1, :), Ine)[1]
            ∂Σ̂ᶻ₁ .+= reshape(vec(∂tmp15'), n, n)
            # Row 3: (3,3) kron(Σ̂ᶻ₁,Ine)
            ∂Σ̂ᶻ₁ .+= _kron_vjp(∂Γ[gb[3]+1:gb[4], gb[3]+1:gb[4]], Matrix(d.Σ̂ᶻ₁), Ine)[1]
            # Row 4: (4,1) kron(Δ̂μˢ₂,Ine)
            ∂Δ̂μˢ₂_l .+= vec(_kron_vjp(∂Γ[gb[4]+1:gb[5], gb[1]+1:gb[2]], reshape(d.Δ̂μˢ₂, :, 1), Ine)[1])
            # (4,4) kron(Σ̂ᶻ₂_22 + Δ*Δ', Ine)
            M44 = d.Σ̂ᶻ₂[n+1:2n, n+1:2n] + d.Δ̂μˢ₂ * d.Δ̂μˢ₂'
            ∂M44 = _kron_vjp(∂Γ[gb[4]+1:gb[5], gb[4]+1:gb[5]], Matrix(M44), Ine)[1]
            ∂Σ̂ᶻ₂[n+1:2n, n+1:2n] .+= ∂M44
            ∂Δ̂μˢ₂_l .+= (∂M44 + ∂M44') * d.Δ̂μˢ₂
            # (4,5) kron(Σ̂ᶻ₂_23 + Δ*vΣ', Ine)
            M45 = d.Σ̂ᶻ₂[n+1:2n, 2n+1:end] + d.Δ̂μˢ₂ * vΣ'
            ∂M45 = _kron_vjp(∂Γ[gb[4]+1:gb[5], gb[5]+1:gb[6]], Matrix(M45), Ine)[1]
            ∂Σ̂ᶻ₂[n+1:2n, 2n+1:end] .+= ∂M45
            ∂Δ̂μˢ₂_l .+= ∂M45 * vΣ
            ∂Σ̂ᶻ₁ .+= reshape(∂M45' * d.Δ̂μˢ₂, n, n)
            # (4,7) kron(Δ̂μˢ₂, e4_nᵉ_nᵉ³)
            ∂Δ̂μˢ₂_l .+= vec(_kron_vjp(∂Γ[gb[4]+1:gb[5], gb[7]+1:gb[8]], reshape(d.Δ̂μˢ₂, :, 1), Matrix(e4_nᵉ_nᵉ³))[1])
            # Row 5: (5,1) kron(vΣ, Ine)
            ∂Σ̂ᶻ₁ .+= reshape(_kron_vjp(∂Γ[gb[5]+1:gb[6], gb[1]+1:gb[2]], reshape(vΣ, :, 1), Ine)[1], n, n)
            # (5,4) kron(Σ̂ᶻ₂_32 + vΣ*Δ', Ine)
            M54 = d.Σ̂ᶻ₂[2n+1:end, n+1:2n] + vΣ * d.Δ̂μˢ₂'
            ∂M54 = _kron_vjp(∂Γ[gb[5]+1:gb[6], gb[4]+1:gb[5]], Matrix(M54), Ine)[1]
            ∂Σ̂ᶻ₂[2n+1:end, n+1:2n] .+= ∂M54
            ∂Σ̂ᶻ₁ .+= reshape(∂M54 * d.Δ̂μˢ₂, n, n)
            ∂Δ̂μˢ₂_l .+= ∂M54' * vΣ
            # (5,5) kron(Σ̂ᶻ₂_33 + vΣ*vΣ', Ine)
            M55 = d.Σ̂ᶻ₂[2n+1:end, 2n+1:end] + vΣ * vΣ'
            ∂M55 = _kron_vjp(∂Γ[gb[5]+1:gb[6], gb[5]+1:gb[6]], Matrix(M55), Ine)[1]
            ∂Σ̂ᶻ₂[2n+1:end, 2n+1:end] .+= ∂M55
            ∂Σ̂ᶻ₁ .+= reshape((∂M55 + ∂M55') * vΣ, n, n)
            # (5,7) kron(vΣ, e4_nᵉ_nᵉ³)
            ∂Σ̂ᶻ₁ .+= reshape(_kron_vjp(∂Γ[gb[5]+1:gb[6], gb[7]+1:gb[8]], reshape(vΣ, :, 1), Matrix(e4_nᵉ_nᵉ³))[1], n, n)
            # Row 6: (6,6) kron(Σ̂ᶻ₁, e4_nᵉ²_nᵉ²)
            ∂Σ̂ᶻ₁ .+= _kron_vjp(∂Γ[gb[6]+1:gb[7], gb[6]+1:gb[7]], Matrix(d.Σ̂ᶻ₁), Matrix(e4_nᵉ²_nᵉ²))[1]
            # Row 7: (7,4) kron(Δ̂μˢ₂', e4')
            ∂tmp74 = _kron_vjp(∂Γ[gb[7]+1:gb[8], gb[4]+1:gb[5]], reshape(d.Δ̂μˢ₂, 1, :), Matrix(e4_nᵉ_nᵉ³'))[1]
            ∂Δ̂μˢ₂_l .+= vec(∂tmp74')
            # (7,5) kron(vΣ', e4')
            ∂tmp75 = _kron_vjp(∂Γ[gb[7]+1:gb[8], gb[5]+1:gb[6]], reshape(vΣ, 1, :), Matrix(e4_nᵉ_nᵉ³'))[1]
            ∂Σ̂ᶻ₁ .+= reshape(vec(∂tmp75'), n, n)

            # ── 3b: Eᴸᶻ disaggregation ──
            ∂EL = Matrix{T}(∂Eᴸᶻ_iter)
            # Only row block 6 is data-dependent
            ∂EL6 = ∂EL[gb[6]+1:gb[7], :]
            # Col 1: kron(Σ̂ᶻ₁, vec_Ie)
            ∂Σ̂ᶻ₁ .+= _kron_vjp(∂EL6[:, sb[1]+1:sb[2]], Matrix(d.Σ̂ᶻ₁), vec_Ie_col)[1]
            # Col 4: kron(μˢ₃δμˢ₁', vec_Ie)
            ∂μ_T = _kron_vjp(∂EL6[:, sb[4]+1:sb[5]], Matrix(d.μˢ₃δμˢ₁'), vec_Ie_col)[1]
            ∂μˢ₃δμˢ₁ = Matrix(∂μ_T')   # n×n
            # Col 5: kron(C₄, vec_Ie)
            inner_C4 = d.Σ̂ᶻ₂[n+1:2n, 2n+1:end] + d.Δ̂μˢ₂ * vΣ'
            ss_s_M = Matrix(d.ss_s)
            C4m = reshape(ss_s_M * vec(inner_C4), n, n^2)
            ∂C4 = _kron_vjp(∂EL6[:, sb[5]+1:sb[6]], C4m, vec_Ie_col)[1]
            ∂iC4 = reshape(ss_s_M' * vec(∂C4), n, n^2)
            ∂Σ̂ᶻ₂[n+1:2n, 2n+1:end] .+= ∂iC4
            ∂Δ̂μˢ₂_l .+= ∂iC4 * vΣ
            ∂Σ̂ᶻ₁ .+= reshape(∂iC4' * d.Δ̂μˢ₂, n, n)
            # Col 6: kron(C₅, vec_Ie)
            inner_C5 = d.Σ̂ᶻ₂[2n+1:end, 2n+1:end] + vΣ * vΣ'
            C5m = reshape(Matrix(inner_C5), n, n^3)
            ∂C5 = _kron_vjp(∂EL6[:, sb[6]+1:sb[7]], C5m, vec_Ie_col)[1]
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
            tmpL, tmpR = _kron_vjp(∂s₁²_from_μ, s₁, s₁);  ∂s₁_l .+= tmpL .+ tmpR

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
            ∂Σ̂ᶻ₁ .+= _kron_vjp(∂M3_raw, Matrix(d.Σ̂ᶻ₁), vec_Ie_col)[1]
            # Decompose ∂M4 → ∂Δ̂μˢ₂
            ∂Δ̂μˢ₂_l .+= vec(_kron_vjp(∂M4_raw, reshape(d.Δ̂μˢ₂, :, 1), Ine)[1])
            # Decompose ∂M6 → ∂Σ̂ᶻ₁
            ∂Σ̂ᶻ₁ .+= reshape(_kron_vjp(∂M6_raw, reshape(vΣ, :, 1), Ine)[1], n, n)

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
        if !(so3_grad[6] isa AbstractZero); ∂S2f_acc .+= so3_grad[6]; end

        # Third-order derivatives pullback: returns (NoTangent, ∂params, ∂SS, NT, NT)
        ∇₃_grad = ∇₃_pb(∂∇₃_acc)
        ∂params_∇₃  = ∇₃_grad[2] isa AbstractZero ? zeros(T, np) : ∇₃_grad[2]
        if !(∇₃_grad[3] isa AbstractZero); ∂SS_acc .+= ∇₃_grad[3]; end

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
            ∂S2f_acc,                # ∂𝐒₂
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
    Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr_tmp_2, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = som2_out

    if !solved; return zero_5(), zero_pb; end

    # ── Step 2: Third-order derivatives ──
    ∇₃, ∇₃_pb = rrule(calculate_third_order_derivatives, parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives)

    # ── Step 3: Third-order solution ──
    so3_out, so3_pb = rrule(calculate_third_order_solution, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂,
                            𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                            initial_guess = 𝓂.caches.third_order_solution,
                            opts = opts)
    𝐒₃, solved3 = so3_out

    update_perturbation_counter!(𝓂.counters, solved3, order = 3)

    if !solved3; return zero_5(), zero_pb; end

    # ── Step 4: Decompress S₃ ──
    𝐔₃ = 𝓂.constants.third_order.𝐔₃
    𝐒₃_full = 𝐒₃ * 𝐔₃

    if !(typeof(𝐒₃_full) <: AbstractSparseMatrix)
        𝐒₃_full = sparse(𝐒₃_full)
    end

    # ── Step 5: Determine iteration groups ──
    orders = determine_efficient_order(𝐒₁, 𝐒₂, 𝐒₃_full, 𝓂.constants, observables,
                                       covariance = covariance, tol = opts.tol.dependencies_tol)

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

        # Set up pruned state transition matrices
        ŝ_to_ŝ₃ = [  s_to_s₁                zeros(nˢ, 2*nˢ + 2*nˢ^2 + nˢ^3)
                                            zeros(nˢ, nˢ) s_to_s₁   s_s_to_s₂ / 2   zeros(nˢ, nˢ + nˢ^2 + nˢ^3)
                                            zeros(nˢ^2, 2 * nˢ)               s_to_s₁_by_s_to_s₁  zeros(nˢ^2, nˢ + nˢ^2 + nˢ^3)
                                            s_v_v_to_s₃ / 2    zeros(nˢ, nˢ + nˢ^2)      s_to_s₁       s_s_to_s₂    s_s_s_to_s₃ / 6
                                            ℒ.kron(s_to_s₁,v_v_to_s₂ / 2)    zeros(nˢ^2, 2*nˢ + nˢ^2)     s_to_s₁_by_s_to_s₁  ℒ.kron(s_to_s₁,s_s_to_s₂ / 2)    
                                            zeros(nˢ^3, 3*nˢ + 2*nˢ^2)   ℒ.kron(s_to_s₁,s_to_s₁_by_s_to_s₁)]

        ê_to_ŝ₃ = [ e_to_s₁   zeros(nˢ,nᵉ^2 + 2*nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        zeros(nˢ,nᵉ)  e_e_to_s₂ / 2   s_e_to_s₂   zeros(nˢ,nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        zeros(nˢ^2,nᵉ)  e_to_s₁_by_e_to_s₁  I_plus_s_s * s_to_s₁_by_e_to_s₁  zeros(nˢ^2, nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        e_v_v_to_s₃ / 2    zeros(nˢ,nᵉ^2 + nᵉ * nˢ)  s_e_to_s₂    s_s_e_to_s₃ / 2    s_e_e_to_s₃ / 2    e_e_e_to_s₃ / 6
                                        ℒ.kron(e_to_s₁, v_v_to_s₂ / 2)    zeros(nˢ^2, nᵉ^2 + nᵉ * nˢ)      s_s * s_to_s₁_by_e_to_s₁    ℒ.kron(s_to_s₁, s_e_to_s₂) + s_s * ℒ.kron(s_s_to_s₂ / 2, e_to_s₁)  ℒ.kron(s_to_s₁, e_e_to_s₂ / 2) + s_s * ℒ.kron(s_e_to_s₂, e_to_s₁)  ℒ.kron(e_to_s₁, e_e_to_s₂ / 2)
                                        zeros(nˢ^3, nᵉ + nᵉ^2 + 2*nᵉ * nˢ) ℒ.kron(s_to_s₁_by_s_to_s₁,e_to_s₁) + ℒ.kron(s_to_s₁, s_s * s_to_s₁_by_e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_s_to_s₁) * e_ss   ℒ.kron(s_to_s₁_by_e_to_s₁,e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_e_to_s₁) * e_es + ℒ.kron(e_to_s₁, s_s * s_to_s₁_by_e_to_s₁) * e_es  ℒ.kron(e_to_s₁,e_to_s₁_by_e_to_s₁)]

        ŝ_to_y₃ = [s_to_y₁ + s_v_v_to_y₃ / 2  s_to_y₁  s_s_to_y₂ / 2   s_to_y₁    s_s_to_y₂     s_s_s_to_y₃ / 6]

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


        Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + 2*nˢ^2 +nˢ^3)
                ℒ.kron(Σ̂ᶻ₁,vec_Iₑ)   zeros(nˢ*nᵉ^2, nˢ + nˢ^2)  ℒ.kron(μˢ₃δμˢ₁',vec_Iₑ)    ℒ.kron(reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec_Iₑ)  ℒ.kron(reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3), vec_Iₑ)
                spzeros(nᵉ^3, 3*nˢ + 2*nˢ^2 +nˢ^3)]

        droptol!(ŝ_to_ŝ₃, eps())
        droptol!(ê_to_ŝ₃, eps())
        droptol!(Eᴸᶻ, eps())
        droptol!(Γ₃, eps())

        A_mat = ê_to_ŝ₃ * Eᴸᶻ * ŝ_to_ŝ₃'
        droptol!(A_mat, eps())

        C_mat = ê_to_ŝ₃ * Γ₃ * ê_to_ŝ₃' + A_mat + A_mat'
        droptol!(C_mat, eps())

        # Ensure third-order lyapunov workspace and solve
        lyap_ws_3rd = ensure_lyapunov_workspace!(𝓂.workspaces, size(ŝ_to_ŝ₃, 1), :third_order)

        lyap_out, lyap_pb_iter = rrule(solve_lyapunov_equation,
                                    Float64.(ŝ_to_ŝ₃), Float64.(C_mat), lyap_ws_3rd,
                                    lyapunov_algorithm = opts.lyapunov_algorithm,
                                    tol = opts.tol.lyapunov_tol,
                                    acceptance_tol = opts.tol.lyapunov_acceptance_tol,
                                    verbose = opts.verbose)
        Σᶻ₃ = lyap_out[1]
        info = lyap_out[2]

        if !info
            return zero_5(), zero_pb
        end

        solved_lyapunov = solved_lyapunov && info

        Σʸ₃tmp = ŝ_to_y₃ * Σᶻ₃ * ŝ_to_y₃' + ê_to_y₃ * Γ₃ * ê_to_y₃' + ê_to_y₃ * Eᴸᶻ * ŝ_to_y₃' + ŝ_to_y₃ * Eᴸᶻ' * ê_to_y₃'

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
            Eᴸᶻⁱ = [ spzeros(T, nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + 2*nˢ^2 + nˢ^3)
                ℒ.kron(s_to_s₁ⁱ * Σ̂ᶻ₁, vec_Iₑ)   zeros(T, nˢ*nᵉ^2, nˢ + nˢ^2)  ℒ.kron(s_to_s₁ⁱ * μˢ₃δμˢ₁', vec_Iₑ)    ℒ.kron(s_to_s₁ⁱ * reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ, 2*nˢ + 1:end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec_Iₑ)  ℒ.kron(s_to_s₁ⁱ * reshape(Σ̂ᶻ₂[2*nˢ + 1:end, 2*nˢ + 1:end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3), vec_Iₑ)
                spzeros(T, nᵉ^3, 3*nˢ + 2*nˢ^2 + nˢ^3)]
            Eᴸᶻ_cur = Eᴸᶻⁱ

            # Step 4: compute autocorrelation
            ŝ_to_ŝ₃ⁱ_snap = copy(ŝ_to_ŝ₃ⁱ)  # snapshot before step 5
            num_mat = Matrix(ŝ_to_y₃) * Σᶻ₃ⁱ * Matrix(ŝ_to_y₃)' + Matrix(ŝ_to_y₃) * ŝ_to_ŝ₃ⁱ * Matrix(autocorr_tmp_ac) + Matrix(ê_to_y₃) * Matrix(Eᴸᶻⁱ) * Matrix(ŝ_to_y₃)'
            num_diag_i = ℒ.diag(num_mat)
            ac_val = num_diag_i ./ norm_diag
            diag_Σ = ℒ.diag(Σʸ₃tmp)
            zero_mask_i = diag_Σ .< opts.tol.lyapunov_acceptance_tol
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
            A_mat = A_mat,
            C_mat = C_mat,
            Σᶻ₃ = Σᶻ₃,
            Σʸ₃tmp = Σʸ₃tmp,
            μˢ₃δμˢ₁ = μˢ₃δμˢ₁,
            lyap_pb = lyap_pb_iter,
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

    result = (Σʸ₃, μʸ₂, autocorr, SS_and_pars, solved && solved3 && solved_lyapunov)

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
            sb_ac = cumsum([0, n, n, n^2, n, n^2, n^3])
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
                    ∂A_c1 = _kron_vjp(∂ELⁱ6[:, sb_ac[1]+1:sb_ac[2]], A_c1, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_c1 * Matrix{T}(d.Σ̂ᶻ₁)'
                    ∂Σ̂ᶻ₁_ac .+= s₁ⁱ' * ∂A_c1

                    # Col 4: kron(s₁ⁱ * μˢ₃δμˢ₁', vec_Ie)
                    A_c4 = s₁ⁱ * Matrix{T}(d.μˢ₃δμˢ₁')
                    ∂A_c4 = _kron_vjp(∂ELⁱ6[:, sb_ac[4]+1:sb_ac[5]], A_c4, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_c4 * Matrix{T}(d.μˢ₃δμˢ₁)
                    ∂μˢ₃δμˢ₁_ac .+= ∂A_c4' * s₁ⁱ

                    # Col 5: kron(s₁ⁱ * C4m, vec_Ie)
                    inner_C4 = d.Σ̂ᶻ₂[n+1:2n, 2n+1:end] + d.Δ̂μˢ₂ * vΣ_ac'
                    C4m = reshape(ss_s_M * vec(inner_C4), n, n^2)
                    A_c5 = s₁ⁱ * C4m
                    ∂A_c5 = _kron_vjp(∂ELⁱ6[:, sb_ac[5]+1:sb_ac[6]], A_c5, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_c5 * C4m'
                    ∂C4_i = s₁ⁱ' * ∂A_c5
                    ∂iC4_i = reshape(ss_s_M' * vec(∂C4_i), n, n^2)
                    ∂Σ̂ᶻ₂_ac[n+1:2n, 2n+1:end] .+= ∂iC4_i
                    ∂Δ̂μˢ₂_ac .+= ∂iC4_i * vΣ_ac
                    ∂Σ̂ᶻ₁_ac .+= reshape(∂iC4_i' * d.Δ̂μˢ₂, n, n)

                    # Col 6: kron(s₁ⁱ * C5m, vec_Ie)
                    inner_C5 = d.Σ̂ᶻ₂[2n+1:end, 2n+1:end] + vΣ_ac * vΣ_ac'
                    C5m = reshape(Matrix{T}(inner_C5), n, n^3)
                    A_c6 = s₁ⁱ * C5m
                    ∂A_c6 = _kron_vjp(∂ELⁱ6[:, sb_ac[6]+1:sb_ac[7]], A_c6, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_c6 * C5m'
                    ∂C5_i = s₁ⁱ' * ∂A_c6
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
                    ∂A_pc1 = _kron_vjp(∂ELprev6[:, sb_ac[1]+1:sb_ac[2]], A_pc1, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_pc1 * Matrix{T}(d.Σ̂ᶻ₁)'
                    ∂Σ̂ᶻ₁_ac .+= s₁ⁱ_prev' * ∂A_pc1

                    # Col 4
                    A_pc4 = s₁ⁱ_prev * Matrix{T}(d.μˢ₃δμˢ₁')
                    ∂A_pc4 = _kron_vjp(∂ELprev6[:, sb_ac[4]+1:sb_ac[5]], A_pc4, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_pc4 * Matrix{T}(d.μˢ₃δμˢ₁)
                    ∂μˢ₃δμˢ₁_ac .+= ∂A_pc4' * s₁ⁱ_prev

                    # Col 5
                    inner_C4p = d.Σ̂ᶻ₂[n+1:2n, 2n+1:end] + d.Δ̂μˢ₂ * vΣ_ac'
                    C4mp = reshape(ss_s_M * vec(inner_C4p), n, n^2)
                    A_pc5 = s₁ⁱ_prev * C4mp
                    ∂A_pc5 = _kron_vjp(∂ELprev6[:, sb_ac[5]+1:sb_ac[6]], A_pc5, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_pc5 * C4mp'
                    ∂C4p = s₁ⁱ_prev' * ∂A_pc5
                    ∂iC4p = reshape(ss_s_M' * vec(∂C4p), n, n^2)
                    ∂Σ̂ᶻ₂_ac[n+1:2n, 2n+1:end] .+= ∂iC4p
                    ∂Δ̂μˢ₂_ac .+= ∂iC4p * vΣ_ac
                    ∂Σ̂ᶻ₁_ac .+= reshape(∂iC4p' * d.Δ̂μˢ₂, n, n)

                    # Col 6
                    inner_C5p = d.Σ̂ᶻ₂[2n+1:end, 2n+1:end] + vΣ_ac * vΣ_ac'
                    C5mp = reshape(Matrix{T}(inner_C5p), n, n^3)
                    A_pc6 = s₁ⁱ_prev * C5mp
                    ∂A_pc6 = _kron_vjp(∂ELprev6[:, sb_ac[6]+1:sb_ac[7]], A_pc6, vec_Ie_col)[1]
                    ∂s_to_s₁ⁱ_co .+= ∂A_pc6 * C5mp'
                    ∂C5p = s₁ⁱ_prev' * ∂A_pc6
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

            # ── Lyapunov adjoint: Σᶻ₃ = lyap(ŝ_to_ŝ₃, C_mat) ──
            lyap_grad = d.lyap_pb((∂Σᶻ₃, NoTangent()))
            ∂ŝ_to_ŝ₃ = ∂ŝ_to_ŝ₃_ac .+ (lyap_grad[2] isa AbstractZero ? zeros(T, size(d.ŝ_to_ŝ₃)) : Matrix{T}(lyap_grad[2]))
            ∂C_mat    = lyap_grad[3] isa AbstractZero ? zeros(T, size(d.C_mat))     : Matrix{T}(lyap_grad[3])

            # ── C_mat = ê_s * Γ₃ * ê_s' + A + A'  where A = ê_s * Eᴸᶻ * ŝ_s' ──
            ∂C_sym = ∂C_mat + ∂C_mat'

            ∂ê_to_ŝ₃  = ∂ê_to_ŝ₃_ac .+ ∂C_sym * (d.ê_to_ŝ₃ * d.Γ₃ + d.ŝ_to_ŝ₃ * Matrix(d.Eᴸᶻ'))
            ∂Γ₃_iter  .+= d.ê_to_ŝ₃' * ∂C_mat * d.ê_to_ŝ₃
            ∂Eᴸᶻ_iter .+= d.ê_to_ŝ₃' * ∂C_sym * d.ŝ_to_ŝ₃
            ∂ŝ_to_ŝ₃  .+= ∂C_sym * d.ê_to_ŝ₃ * Matrix(d.Eᴸᶻ)

            # ── Disaggregate ŝ_to_y₃ → ∂𝐒₁, ∂𝐒₂, ∂𝐒₃ ──
            c = 0
            ∂blk1 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i];      c += nˢ_i
            ∂blk2 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i];      c += nˢ_i
            ∂blk3 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i^2];    c += nˢ_i^2
            ∂blk4 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i];      c += nˢ_i
            ∂blk5 = ∂ŝ_to_y₃[:, c+1:c+nˢ_i^2];    c += nˢ_i^2
            ∂blk6 = ∂ŝ_to_y₃[:, c+1:end]

            ∂𝐒₁_acc[d.obs_in_y, d.dependencies_in_states_idx] .+= ∂blk1 .+ ∂blk2 .+ ∂blk4
            ∂S2f_acc[d.obs_in_y, d.kron_s_s]                  .+= ∂blk3 ./ 2 .+ ∂blk5
            ∂S3f_acc[d.obs_in_y, d.kron_s_v_v]                .+= ∂blk1 ./ 2
            ∂S3f_acc[d.obs_in_y, d.kron_s_s_s]                .+= ∂blk6 ./ 6

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
            sb = cumsum([0, n, n, n^2, n, n^2, n^3])
            eb = cumsum([0, ne, ne^2, n*ne, n*ne, n^2*ne, n*ne^2, ne^3])
            gb = eb

            vvh = vv₂ ./ 2;  ssh = ss₂ ./ 2;  eeh = ee₂ ./ 2

            # ── 2a: ŝ_to_ŝ₃ disaggregation ──
            ∂ŝ₃ = ∂ŝ_to_ŝ₃

            # Direct s₁ blocks: (1,1), (2,2), (4,4)
            ∂s₁_l .+= ∂ŝ₃[sb[1]+1:sb[2], sb[1]+1:sb[2]] .+
                       ∂ŝ₃[sb[2]+1:sb[3], sb[2]+1:sb[3]] .+
                       ∂ŝ₃[sb[4]+1:sb[5], sb[4]+1:sb[5]]
            # (2,3) ss₂/2
            ∂ss₂_l .+= ∂ŝ₃[sb[2]+1:sb[3], sb[3]+1:sb[4]] ./ 2
            # (4,5) ss₂
            ∂ss₂_l .+= ∂ŝ₃[sb[4]+1:sb[5], sb[5]+1:sb[6]]
            # (4,1) s_vv₃/2
            ∂S3f_acc[d.iˢ, d.kron_s_v_v] .+= ∂ŝ₃[sb[4]+1:sb[5], sb[1]+1:sb[2]] ./ 2
            # (4,6) sss₃/6
            ∂S3f_acc[d.iˢ, d.kron_s_s_s] .+= ∂ŝ₃[sb[4]+1:sb[5], sb[6]+1:sb[7]] ./ 6
            # (3,3) kron(s₁,s₁)
            tmpL, tmpR = _kron_vjp(Matrix(∂ŝ₃[sb[3]+1:sb[4], sb[3]+1:sb[4]]), s₁, s₁)
            ∂s₁_l .+= tmpL .+ tmpR
            # (5,1) kron(s₁, vv₂/2)
            tmpA, tmpB = _kron_vjp(Matrix(∂ŝ₃[sb[5]+1:sb[6], sb[1]+1:sb[2]]), s₁, vvh)
            ∂s₁_l .+= tmpA;  ∂vv₂_l .+= tmpB ./ 2
            # (5,5) kron(s₁,s₁)
            tmpL, tmpR = _kron_vjp(Matrix(∂ŝ₃[sb[5]+1:sb[6], sb[5]+1:sb[6]]), s₁, s₁)
            ∂s₁_l .+= tmpL .+ tmpR
            # (5,6) kron(s₁, ss₂/2)
            tmpA, tmpB = _kron_vjp(Matrix(∂ŝ₃[sb[5]+1:sb[6], sb[6]+1:sb[7]]), s₁, ssh)
            ∂s₁_l .+= tmpA;  ∂ss₂_l .+= tmpB ./ 2
            # (6,6) kron(s₁, kron(s₁,s₁))
            tmpA, tmpB = _kron_vjp(Matrix(∂ŝ₃[sb[6]+1:sb[7], sb[6]+1:sb[7]]), s₁, s₁²)
            ∂s₁_l .+= tmpA
            tmpL, tmpR = _kron_vjp(tmpB, s₁, s₁)
            ∂s₁_l .+= tmpL .+ tmpR

            # ── 2b: ê_to_ŝ₃ disaggregation ──
            ∂ê₃ = Matrix{T}(∂ê_to_ŝ₃)
            ss_s1e1 = Matrix(d.s_s) * s₁e₁

            # Row 1: (1,1) e₁
            ∂e₁_l .+= ∂ê₃[sb[1]+1:sb[2], eb[1]+1:eb[2]]
            # Row 2: (2,2) ee₂/2; (2,3) se₂
            ∂ee₂_l .+= ∂ê₃[sb[2]+1:sb[3], eb[2]+1:eb[3]] ./ 2
            ∂se₂_l .+= ∂ê₃[sb[2]+1:sb[3], eb[3]+1:eb[4]]
            # Row 3: (3,2) kron(e₁,e₁)
            tmpL, tmpR = _kron_vjp(Matrix(∂ê₃[sb[3]+1:sb[4], eb[2]+1:eb[3]]), e₁, e₁)
            ∂e₁_l .+= tmpL .+ tmpR
            # (3,3) I_plus_s_s * kron(s₁,e₁)
            ∂k33 = Matrix(d.I_plus_s_s') * Matrix(∂ê₃[sb[3]+1:sb[4], eb[3]+1:eb[4]])
            tmpA, tmpB = _kron_vjp(∂k33, s₁, e₁)
            ∂s₁_l .+= tmpA;  ∂e₁_l .+= tmpB
            # Row 4: direct S₃ slices
            ∂S3f_acc[d.iˢ, d.kron_e_v_v] .+= ∂ê₃[sb[4]+1:sb[5], eb[1]+1:eb[2]] ./ 2
            ∂se₂_l .+= ∂ê₃[sb[4]+1:sb[5], eb[4]+1:eb[5]]
            ∂S3f_acc[d.iˢ, d.kron_s_s_e] .+= ∂ê₃[sb[4]+1:sb[5], eb[5]+1:eb[6]] ./ 2
            ∂S3f_acc[d.iˢ, d.kron_s_e_e] .+= ∂ê₃[sb[4]+1:sb[5], eb[6]+1:eb[7]] ./ 2
            ∂S3f_acc[d.iˢ, d.kron_e_e_e] .+= ∂ê₃[sb[4]+1:sb[5], eb[7]+1:eb[8]] ./ 6
            # Row 5: (5,1) kron(e₁,vv₂/2)
            tmpA, tmpB = _kron_vjp(Matrix(∂ê₃[sb[5]+1:sb[6], eb[1]+1:eb[2]]), e₁, vvh)
            ∂e₁_l .+= tmpA;  ∂vv₂_l .+= tmpB ./ 2
            # (5,4) s_s * kron(s₁,e₁)
            ∂k54 = Matrix(d.s_s') * Matrix(∂ê₃[sb[5]+1:sb[6], eb[4]+1:eb[5]])
            tmpA, tmpB = _kron_vjp(∂k54, s₁, e₁)
            ∂s₁_l .+= tmpA;  ∂e₁_l .+= tmpB
            # (5,5) kron(s₁,se₂) + s_s * kron(ss₂/2, e₁)
            ∂b55 = Matrix(∂ê₃[sb[5]+1:sb[6], eb[5]+1:eb[6]])
            tmpA, tmpB = _kron_vjp(∂b55, s₁, se₂)
            ∂s₁_l .+= tmpA;  ∂se₂_l .+= tmpB
            ∂k55b = Matrix(d.s_s') * ∂b55
            tmpA, tmpB = _kron_vjp(∂k55b, ssh, e₁)
            ∂ss₂_l .+= tmpA ./ 2;  ∂e₁_l .+= tmpB
            # (5,6) kron(s₁,ee₂/2) + s_s * kron(se₂, e₁)
            ∂b56 = Matrix(∂ê₃[sb[5]+1:sb[6], eb[6]+1:eb[7]])
            tmpA, tmpB = _kron_vjp(∂b56, s₁, eeh)
            ∂s₁_l .+= tmpA;  ∂ee₂_l .+= tmpB ./ 2
            ∂k56b = Matrix(d.s_s') * ∂b56
            tmpA, tmpB = _kron_vjp(∂k56b, se₂, e₁)
            ∂se₂_l .+= tmpA;  ∂e₁_l .+= tmpB
            # (5,7) kron(e₁, ee₂/2)
            tmpA, tmpB = _kron_vjp(Matrix(∂ê₃[sb[5]+1:sb[6], eb[7]+1:eb[8]]), e₁, eeh)
            ∂e₁_l .+= tmpA;  ∂ee₂_l .+= tmpB ./ 2
            # Row 6: (6,5) kron(s₁²,e₁) + kron(s₁,s_s*s₁e₁) + kron(e₁,s₁²)*e_ss
            ∂b65 = Matrix(∂ê₃[sb[6]+1:sb[7], eb[5]+1:eb[6]])
            tmpA, tmpB = _kron_vjp(∂b65, s₁², e₁)
            ∂e₁_l .+= tmpB
            tmpL, tmpR = _kron_vjp(tmpA, s₁, s₁);  ∂s₁_l .+= tmpL .+ tmpR
            tmpA, tmpB = _kron_vjp(∂b65, s₁, ss_s1e1)
            ∂s₁_l .+= tmpA
            tmpC = Matrix(d.s_s') * tmpB
            tmpL, tmpR = _kron_vjp(tmpC, s₁, e₁);  ∂s₁_l .+= tmpL;  ∂e₁_l .+= tmpR
            ∂k65c = ∂b65 * Matrix(d.e_ss')
            tmpA, tmpB = _kron_vjp(∂k65c, e₁, s₁²)
            ∂e₁_l .+= tmpA
            tmpL, tmpR = _kron_vjp(tmpB, s₁, s₁);  ∂s₁_l .+= tmpL .+ tmpR
            # (6,6) kron(s₁e₁,e₁) + kron(e₁,s₁e₁)*e_es + kron(e₁,s_s*s₁e₁)*e_es
            ∂b66 = Matrix(∂ê₃[sb[6]+1:sb[7], eb[6]+1:eb[7]])
            tmpA, tmpB = _kron_vjp(∂b66, s₁e₁, e₁)
            ∂e₁_l .+= tmpB
            tmpL, tmpR = _kron_vjp(tmpA, s₁, e₁);  ∂s₁_l .+= tmpL;  ∂e₁_l .+= tmpR
            ∂pre = ∂b66 * Matrix(d.e_es')
            tmpA, tmpB = _kron_vjp(∂pre, e₁, s₁e₁)
            ∂e₁_l .+= tmpA
            tmpL, tmpR = _kron_vjp(tmpB, s₁, e₁);  ∂s₁_l .+= tmpL;  ∂e₁_l .+= tmpR
            tmpA, tmpB = _kron_vjp(∂pre, e₁, ss_s1e1)
            ∂e₁_l .+= tmpA
            tmpC = Matrix(d.s_s') * tmpB
            tmpL, tmpR = _kron_vjp(tmpC, s₁, e₁);  ∂s₁_l .+= tmpL;  ∂e₁_l .+= tmpR
            # (6,7) kron(e₁, e₁²)
            tmpA, tmpB = _kron_vjp(Matrix(∂ê₃[sb[6]+1:sb[7], eb[7]+1:eb[8]]), e₁, e₁²)
            ∂e₁_l .+= tmpA
            tmpL, tmpR = _kron_vjp(tmpB, e₁, e₁);  ∂e₁_l .+= tmpL .+ tmpR

            # ── 3a: Γ₃ disaggregation → ∂Σ̂ᶻ₁, ∂Σ̂ᶻ₂, ∂Δ̂μˢ₂ ──
            ∂Γ = Matrix{T}(∂Γ₃_iter)
            vΣ = vec(d.Σ̂ᶻ₁)

            # Row 1: (1,4) kron(Δ̂μˢ₂',Ine)
            ∂tmp14 = _kron_vjp(∂Γ[gb[1]+1:gb[2], gb[4]+1:gb[5]], reshape(d.Δ̂μˢ₂, 1, :), Ine)[1]
            ∂Δ̂μˢ₂_l .+= vec(∂tmp14')
            # (1,5) kron(vec(Σ̂ᶻ₁)',Ine)
            ∂tmp15 = _kron_vjp(∂Γ[gb[1]+1:gb[2], gb[5]+1:gb[6]], reshape(vΣ, 1, :), Ine)[1]
            ∂Σ̂ᶻ₁ .+= reshape(vec(∂tmp15'), n, n)
            # Row 3: (3,3) kron(Σ̂ᶻ₁,Ine)
            ∂Σ̂ᶻ₁ .+= _kron_vjp(∂Γ[gb[3]+1:gb[4], gb[3]+1:gb[4]], Matrix(d.Σ̂ᶻ₁), Ine)[1]
            # Row 4: (4,1) kron(Δ̂μˢ₂,Ine)
            ∂Δ̂μˢ₂_l .+= vec(_kron_vjp(∂Γ[gb[4]+1:gb[5], gb[1]+1:gb[2]], reshape(d.Δ̂μˢ₂, :, 1), Ine)[1])
            # (4,4) kron(Σ̂ᶻ₂_22 + Δ*Δ', Ine)
            M44 = d.Σ̂ᶻ₂[n+1:2n, n+1:2n] + d.Δ̂μˢ₂ * d.Δ̂μˢ₂'
            ∂M44 = _kron_vjp(∂Γ[gb[4]+1:gb[5], gb[4]+1:gb[5]], Matrix(M44), Ine)[1]
            ∂Σ̂ᶻ₂[n+1:2n, n+1:2n] .+= ∂M44
            ∂Δ̂μˢ₂_l .+= (∂M44 + ∂M44') * d.Δ̂μˢ₂
            # (4,5) kron(Σ̂ᶻ₂_23 + Δ*vΣ', Ine)
            M45 = d.Σ̂ᶻ₂[n+1:2n, 2n+1:end] + d.Δ̂μˢ₂ * vΣ'
            ∂M45 = _kron_vjp(∂Γ[gb[4]+1:gb[5], gb[5]+1:gb[6]], Matrix(M45), Ine)[1]
            ∂Σ̂ᶻ₂[n+1:2n, 2n+1:end] .+= ∂M45
            ∂Δ̂μˢ₂_l .+= ∂M45 * vΣ
            ∂Σ̂ᶻ₁ .+= reshape(∂M45' * d.Δ̂μˢ₂, n, n)
            # (4,7) kron(Δ̂μˢ₂, e4_nᵉ_nᵉ³)
            ∂Δ̂μˢ₂_l .+= vec(_kron_vjp(∂Γ[gb[4]+1:gb[5], gb[7]+1:gb[8]], reshape(d.Δ̂μˢ₂, :, 1), Matrix(e4_nᵉ_nᵉ³))[1])
            # Row 5: (5,1) kron(vΣ, Ine)
            ∂Σ̂ᶻ₁ .+= reshape(_kron_vjp(∂Γ[gb[5]+1:gb[6], gb[1]+1:gb[2]], reshape(vΣ, :, 1), Ine)[1], n, n)
            # (5,4) kron(Σ̂ᶻ₂_32 + vΣ*Δ', Ine)
            M54 = d.Σ̂ᶻ₂[2n+1:end, n+1:2n] + vΣ * d.Δ̂μˢ₂'
            ∂M54 = _kron_vjp(∂Γ[gb[5]+1:gb[6], gb[4]+1:gb[5]], Matrix(M54), Ine)[1]
            ∂Σ̂ᶻ₂[2n+1:end, n+1:2n] .+= ∂M54
            ∂Σ̂ᶻ₁ .+= reshape(∂M54 * d.Δ̂μˢ₂, n, n)
            ∂Δ̂μˢ₂_l .+= ∂M54' * vΣ
            # (5,5) kron(Σ̂ᶻ₂_33 + vΣ*vΣ', Ine)
            M55 = d.Σ̂ᶻ₂[2n+1:end, 2n+1:end] + vΣ * vΣ'
            ∂M55 = _kron_vjp(∂Γ[gb[5]+1:gb[6], gb[5]+1:gb[6]], Matrix(M55), Ine)[1]
            ∂Σ̂ᶻ₂[2n+1:end, 2n+1:end] .+= ∂M55
            ∂Σ̂ᶻ₁ .+= reshape((∂M55 + ∂M55') * vΣ, n, n)
            # (5,7) kron(vΣ, e4_nᵉ_nᵉ³)
            ∂Σ̂ᶻ₁ .+= reshape(_kron_vjp(∂Γ[gb[5]+1:gb[6], gb[7]+1:gb[8]], reshape(vΣ, :, 1), Matrix(e4_nᵉ_nᵉ³))[1], n, n)
            # Row 6: (6,6) kron(Σ̂ᶻ₁, e4_nᵉ²_nᵉ²)
            ∂Σ̂ᶻ₁ .+= _kron_vjp(∂Γ[gb[6]+1:gb[7], gb[6]+1:gb[7]], Matrix(d.Σ̂ᶻ₁), Matrix(e4_nᵉ²_nᵉ²))[1]
            # Row 7: (7,4) kron(Δ̂μˢ₂', e4')
            ∂tmp74 = _kron_vjp(∂Γ[gb[7]+1:gb[8], gb[4]+1:gb[5]], reshape(d.Δ̂μˢ₂, 1, :), Matrix(e4_nᵉ_nᵉ³'))[1]
            ∂Δ̂μˢ₂_l .+= vec(∂tmp74')
            # (7,5) kron(vΣ', e4')
            ∂tmp75 = _kron_vjp(∂Γ[gb[7]+1:gb[8], gb[5]+1:gb[6]], reshape(vΣ, 1, :), Matrix(e4_nᵉ_nᵉ³'))[1]
            ∂Σ̂ᶻ₁ .+= reshape(vec(∂tmp75'), n, n)

            # ── 3b: Eᴸᶻ disaggregation ──
            ∂EL = Matrix{T}(∂Eᴸᶻ_iter)
            # Only row block 6 is data-dependent
            ∂EL6 = ∂EL[gb[6]+1:gb[7], :]
            # Col 1: kron(Σ̂ᶻ₁, vec_Ie)
            ∂Σ̂ᶻ₁ .+= _kron_vjp(∂EL6[:, sb[1]+1:sb[2]], Matrix(d.Σ̂ᶻ₁), vec_Ie_col)[1]
            # Col 4: kron(μˢ₃δμˢ₁', vec_Ie)
            ∂μ_T = _kron_vjp(∂EL6[:, sb[4]+1:sb[5]], Matrix(d.μˢ₃δμˢ₁'), vec_Ie_col)[1]
            ∂μˢ₃δμˢ₁ = ∂μˢ₃δμˢ₁_ac .+ Matrix(∂μ_T')
            # Col 5: kron(C₄, vec_Ie)
            inner_C4 = d.Σ̂ᶻ₂[n+1:2n, 2n+1:end] + d.Δ̂μˢ₂ * vΣ'
            C4m = reshape(ss_s_M * vec(inner_C4), n, n^2)
            ∂C4 = _kron_vjp(∂EL6[:, sb[5]+1:sb[6]], C4m, vec_Ie_col)[1]
            ∂iC4 = reshape(ss_s_M' * vec(∂C4), n, n^2)
            ∂Σ̂ᶻ₂[n+1:2n, 2n+1:end] .+= ∂iC4
            ∂Δ̂μˢ₂_l .+= ∂iC4 * vΣ
            ∂Σ̂ᶻ₁ .+= reshape(∂iC4' * d.Δ̂μˢ₂, n, n)
            # Col 6: kron(C₅, vec_Ie)
            inner_C5 = d.Σ̂ᶻ₂[2n+1:end, 2n+1:end] + vΣ * vΣ'
            C5m = reshape(Matrix(inner_C5), n, n^3)
            ∂C5 = _kron_vjp(∂EL6[:, sb[6]+1:sb[7]], C5m, vec_Ie_col)[1]
            ∂iC5 = reshape(∂C5, n^2, n^2)
            ∂Σ̂ᶻ₂[2n+1:end, 2n+1:end] .+= ∂iC5
            ∂Σ̂ᶻ₁ .+= reshape((∂iC5 + ∂iC5') * vΣ, n, n)

            # ── 3c: μˢ₃δμˢ₁ adjoint ──
            ∂x_μ = vec(∂μˢ₃δμˢ₁)
            I_m_s₁² = Matrix{T}(ℒ.I(n^2)) - s₁²
            ∂b_μ = I_m_s₁²' \ ∂x_μ
            ∂s₁²_from_μ = ∂b_μ * vec(d.μˢ₃δμˢ₁)'
            tmpL, tmpR = _kron_vjp(∂s₁²_from_μ, s₁, s₁);  ∂s₁_l .+= tmpL .+ tmpR

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
            ∂Σ̂ᶻ₁ .+= _kron_vjp(∂M3_raw, Matrix(d.Σ̂ᶻ₁), vec_Ie_col)[1]
            # Decompose ∂M4 → ∂Δ̂μˢ₂
            ∂Δ̂μˢ₂_l .+= vec(_kron_vjp(∂M4_raw, reshape(d.Δ̂μˢ₂, :, 1), Ine)[1])
            # Decompose ∂M6 → ∂Σ̂ᶻ₁
            ∂Σ̂ᶻ₁ .+= reshape(_kron_vjp(∂M6_raw, reshape(vΣ, :, 1), Ine)[1], n, n)

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
        if !(so3_grad[6] isa AbstractZero); ∂S2f_acc .+= so3_grad[6]; end

        # Third-order derivatives pullback
        ∇₃_grad = ∇₃_pb(∂∇₃_acc)
        ∂params_∇₃  = ∇₃_grad[2] isa AbstractZero ? zeros(T, np) : ∇₃_grad[2]
        if !(∇₃_grad[3] isa AbstractZero); ∂SS_acc .+= ∇₃_grad[3]; end

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
            ∂S2f_acc,                # ∂𝐒₂
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
                initial_guess::AbstractMatrix{R} = zeros(0,0)) where {R <: AbstractFloat}
    # Forward pass to compute the output and intermediate values needed for the backward pass
    # @timeit_debug timer "Calculate 1st order solution" begin
    # @timeit_debug timer "Preprocessing" begin

    T = constants.post_model_macro
    idx_constants = ensure_first_order_constants!(constants)

    dynIndex = idx_constants.dyn_index
    reverse_dynamic_order = idx_constants.reverse_dynamic_order
    comb = idx_constants.comb
    future_not_past_and_mixed_in_comb = idx_constants.future_not_past_and_mixed_in_comb
    past_not_future_and_mixed_in_comb = idx_constants.past_not_future_and_mixed_in_comb
    past_not_future_and_mixed_in_present_but_not_only = idx_constants.past_not_future_and_mixed_in_present_but_not_only
    Ir = idx_constants.Ir

    qme_ws = workspaces.first_order
    sylv_ws = workspaces.sylvester_1st_order
    ensure_sylvester_krylov_buffers!(qme_ws.sylvester_ws, T.nVars, T.nVars)
    ensure_sylvester_doubling_buffers!(qme_ws.sylvester_ws, T.nVars, T.nVars)

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
    # Legacy readable flow mirrored from primal first-order solver:
    #   Q = qr!(∇₀[:, T.present_only_idx])
    #   A₊ = Q.Q' * ∇₊;  A₀ = Q.Q' * ∇₀;  A₋ = Q.Q' * ∇₋
    # The current implementation keeps the same algebra while reusing QR workspaces.
    qr_factors, qr_ws = ensure_first_order_fast_qr_workspace!(qme_ws, ∇₀_present)
    Q = factorize_qr!(∇₀_present, qr_factors, qr_ws;
                        use_fastlapack_qr = use_fastlapack_qr)

    qme_ws.fast_qr_orm_ws_plus, qme_ws.fast_qr_orm_dims_plus = apply_qr_transpose_left!(A₊, ∇₊, Q,
                                                                                        qme_ws.fast_qr_orm_ws_plus,
                                                                                        qme_ws.fast_qr_orm_dims_plus,
                                                                                        qr_ws;
                                                                                        use_fastlapack_qr = use_fastlapack_qr)
    qme_ws.fast_qr_orm_ws_zero, qme_ws.fast_qr_orm_dims_zero = apply_qr_transpose_left!(A₀, ∇₀, Q,
                                                                                        qme_ws.fast_qr_orm_ws_zero,
                                                                                        qme_ws.fast_qr_orm_dims_zero,
                                                                                        qr_ws;
                                                                                        use_fastlapack_qr = use_fastlapack_qr)
    qme_ws.fast_qr_orm_ws_minus, qme_ws.fast_qr_orm_dims_minus = apply_qr_transpose_left!(A₋, ∇₋, Q,
                                                                                           qme_ws.fast_qr_orm_ws_minus,
                                                                                           qme_ws.fast_qr_orm_dims_minus,
                                                                                           qr_ws;
                                                                                           use_fastlapack_qr = use_fastlapack_qr)
    
    # end # timeit_debug
    # @timeit_debug timer "Sort matrices" begin

    Ã₊ = qme_ws.𝐀̃₊
    ℒ.mul!(Ã₊, @view(A₊[dynIndex,:]), Ir[future_not_past_and_mixed_in_comb,:])

    Ã₀ = qme_ws.𝐀̃₀
    copyto!(Ã₀, @view(A₀[dynIndex, comb]))

    Ã₋ = qme_ws.𝐀̃₋
    ℒ.mul!(Ã₋, @view(A₋[dynIndex,:]), Ir[past_not_future_and_mixed_in_comb,:])

    # end # timeit_debug
    # @timeit_debug timer "Quadratic matrix equation solve" begin

    sol, solved = solve_quadratic_matrix_equation(Ã₊, Ã₀, Ã₋, constants, workspaces, cache;
                                                    initial_guess = initial_guess,
                                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                                    tol = opts.tol.qme_tol,
                                                    acceptance_tol = opts.tol.qme_acceptance_tol,
                                                    verbose = opts.verbose)

    if !solved
        return (zeros(T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
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

    qme_ws.fast_lu_ws_a0u, qme_ws.fast_lu_dims_a0u, solved_Ā₀ᵤ, Ā̂₀ᵤ = factorize_lu!(Ā₀ᵤ,
                                                                                       qme_ws.fast_lu_ws_a0u,
                                                                                       qme_ws.fast_lu_dims_a0u;
                                                                                       use_fastlapack_lu = use_fastlapack_lu)

    if !solved_Ā₀ᵤ
        return (zeros(T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # A    = vcat(-(Ā̂₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ)), sol)
    if T.nPresent_only > 0
        ℒ.mul!(A₋ᵤ, Ã₀ᵤ, @view(sol[:,past_not_future_and_mixed_in_comb]), 1, 1)
        nₚ₋ = qme_ws.𝐧ₚ₋
        ℒ.mul!(nₚ₋, A₊ᵤ, D)
        ℒ.mul!(A₋ᵤ, nₚ₋, L, 1, 1)
        solve_lu_left!(Ā₀ᵤ, A₋ᵤ, qme_ws.fast_lu_ws_a0u, Ā̂₀ᵤ;
                       use_fastlapack_lu = use_fastlapack_lu)
        ℒ.rmul!(A₋ᵤ, -1)
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
    
    𝐒̂ᵗ = qme_ws.sylvester_ws.tmp
    ℒ.mul!(𝐒̂ᵗ, 𝐒ᵗ, expand_past)

    ∇₊ = qme_ws.sylvester_ws.𝐀
    ℒ.mul!(∇₊, @view(∇₁[:,1:T.nFuture_not_past_and_mixed]), expand_future)

    ℒ.mul!(∇₀, ∇₊, 𝐒̂ᵗ, 1, 1)

    qme_ws.fast_lu_ws_nabla0, qme_ws.fast_lu_dims_nabla0, solved_∇₀, C = factorize_lu!(∇₀,
                                                                                         qme_ws.fast_lu_ws_nabla0,
                                                                                         qme_ws.fast_lu_dims_nabla0;
                                                                                         use_fastlapack_lu = use_fastlapack_lu)

    if !solved_∇₀
        return (zeros(T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    solve_lu_left!(∇₀, ∇̂ₑ, qme_ws.fast_lu_ws_nabla0, C;
                   use_fastlapack_lu = use_fastlapack_lu)
    ℒ.rmul!(∇̂ₑ, -1)

    # end # timeit_debug
    # end # timeit_debug
    
    M = qme_ws.sylvester_ws.𝐀¹
    fill!(M, zero(R))
    @inbounds for i in axes(M, 1)
        M[i, i] = one(R)
    end
    solve_lu_left!(∇₀, M, qme_ws.fast_lu_ws_nabla0, C;
                   use_fastlapack_lu = use_fastlapack_lu)

    tmp2 = qme_ws.sylvester_ws.𝐁
    ℒ.mul!(tmp2, M', ∇₊')
    ℒ.rmul!(tmp2, -1)

    ∇ₑ = @view ∇₁[:,idx_constants.nabla_e_start:end]

    function first_order_solution_pullback(∂𝐒) 
        # Guard: if the cotangent for the solution matrix is NoTangent
        # (e.g. because a downstream filter failure returned all-NoTangent),
        # return zero gradients immediately.
        if ∂𝐒[1] isa Union{NoTangent, AbstractZero}
            return NoTangent(), zero(∇₁), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        ∂∇₁ = zero(∇₁)

        ∂𝐒ᵗ = ∂𝐒[1][:,1:T.nPast_not_future_and_mixed]
        ∂𝐒ᵉ = ∂𝐒[1][:,T.nPast_not_future_and_mixed + 1:end]

        # Shared sub-expression: W = M' * ∂𝐒ᵉ * ∇ₑ' * M'
        # Use workspace buffers to avoid repeated intermediate allocations.
        # t1 = M' * ∂𝐒ᵉ  (nVars × nExo)
        t1 = M' * ∂𝐒ᵉ  # one alloc for nVars×nExo

        # ∂∇₁[:,nabla_e_start:end] = -t1
        @views ∂∇₁[:,idx_constants.nabla_e_start:end] .= .-t1

        # t2 = t1 * ∇ₑ'  (nVars × nVars) → store in 𝐗 workspace
        t2 = qme_ws.sylvester_ws.𝐗
        ℒ.mul!(t2, t1, ∇ₑ')

        # W = t2 * M'  (nVars × nVars) → store in 𝐂_dbl workspace
        W = qme_ws.sylvester_ws.𝐂_dbl
        ℒ.mul!(W, t2, M')

        @views ∂∇₁[:,idx_constants.nabla_zero_cols] .= W

        # Wp = W * expand_past'  (nVars × nPast) → store in view of 𝐂¹ workspace (nVars×nVars)
        Wp = @view qme_ws.sylvester_ws.𝐂¹[:, 1:T.nPast_not_future_and_mixed]
        ℒ.mul!(Wp, W, expand_past')

        # ∂∇₁[:,1:nFuture] = (Wp * 𝐒ᵗ')[:,future_idx]
        # WpSt = Wp * 𝐒ᵗ'  (nVars × nVars) → store in 𝐂B workspace
        WpSt = qme_ws.sylvester_ws.𝐂B
        ℒ.mul!(WpSt, Wp, 𝐒ᵗ')
        @views ∂∇₁[:,1:T.nFuture_not_past_and_mixed] .= WpSt[:,T.future_not_past_and_mixed_idx]

        # ∂𝐒ᵗ += ∇₊' * Wp  (nVars × nPast, ∇₊ is nVars×nVars, Wp is nVars×nPast)
        ℒ.mul!(∂𝐒ᵗ, ∇₊', Wp, 1, 1)

        tmp1 = qme_ws.sylvester_ws.𝐂
        # tmp1 = M' * ∂𝐒ᵗ * expand_past  (nVars × nVars)
        # t_ms = M' * ∂𝐒ᵗ  (nVars × nPast) → reuse Wp (view of 𝐂¹, same dims)
        ℒ.mul!(Wp, M', ∂𝐒ᵗ)
        ℒ.mul!(tmp1, Wp, expand_past)

        ss, solved = solve_sylvester_equation(tmp2, 𝐒̂ᵗ', -tmp1, sylv_ws,
                                                sylvester_algorithm = opts.sylvester_algorithm²,
                                                tol = opts.tol.sylvester_tol,
                                                acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                                verbose = opts.verbose)

        if !solved
            NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
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
                    opts::CalculationOptions = merge_calculation_options()) where {S <: Real, R <: Real}
    if !(eltype(workspaces.second_order.Ŝ) == S)
        workspaces.second_order = Higher_order_workspace(T = S)
    end
    ℂ = workspaces.second_order
    M₂ = constants.second_order
    T = constants.post_model_macro
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
        return (∇₁₊𝐒₁➕∇₁₀, false), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
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

    𝐒₂, solved = solve_sylvester_equation(A, B, C, ℂ.sylvester_workspace,
                                            initial_guess = initial_guess,
                                            sylvester_algorithm = opts.sylvester_algorithm²,
                                            tol = opts.tol.sylvester_tol,
                                            acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                            verbose = opts.verbose)

    # end # timeit_debug
    # @timeit_debug timer "Post-process" begin

    if !solved
        return (𝐒₂, solved), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # end # timeit_debug

    # sp⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t = choose_matrix_format(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋', density_threshold = 1.0)

    # sp𝐒₁₊╱𝟎t = choose_matrix_format(𝐒₁₊╱𝟎', density_threshold = 1.0)

    𝛔t = choose_matrix_format(M₂.𝛔', density_threshold = 1.0)

    𝐔₂t = choose_matrix_format(M₂.𝐔₂', density_threshold = 1.0)

    𝐂₂t = choose_matrix_format(M₂.𝐂₂', density_threshold = 1.0)

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

        ∂𝐒₂ = ∂𝐒₂_solved[1]
        
        # ∂𝐒₂ *= 𝐔₂t

        # @timeit_debug timer "Sylvester" begin
        if ℒ.norm(∂𝐒₂) < opts.tol.sylvester_tol
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end
        
        ∂C, solved = solve_sylvester_equation(A', B', ∂𝐒₂, ℂ.sylvester_workspace,
                                                sylvester_algorithm = opts.sylvester_algorithm²,
                                                tol = opts.tol.sylvester_tol,
                                                acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                                verbose = opts.verbose)

        if !solved
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
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
    end

    # return (sparse(𝐒₂ * M₂.𝐔₂), solved), second_order_solution_pullback
    return (𝐒₂, solved), second_order_solution_pullback
end


# Helper: adjoint of compressed_kron³(X) w.r.t. X.
# Forward: out[row,col] = (aii*(ajj*akk + ajk*akj) + aij*(aji*akk + ajk*aki) + aik*(aji*akj + ajj*aki)) / divisor
# where row ↔ (i1≥j1≥k1) and col ↔ (i2≥j2≥k2) and a_pq = X[p,q].
function compressed_kron³_pullback!(∂X::AbstractMatrix{T}, ∂Y::AbstractMatrix{T}, X::AbstractMatrix{T}) where T <: Real
    Xd = X isa DenseMatrix ? X : collect(X)
    n_rows, n_cols = size(Xd)
    # Unlike the forward pass, the pullback must iterate over ALL row/column
    # indices, not just nonzero ones.  The gradient at a zero entry X[r,c] can
    # be non-zero because  ∂(X[i]*X[j]*X[k])/∂X[i] = X[j]*X[k]  which is
    # generically non-zero even when X[i]=0.
    for i1 in 1:n_rows, j1 in 1:n_rows
        j1 ≤ i1 || continue
        for k1 in 1:n_rows
            k1 ≤ j1 || continue
            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1
            # divisor for row symmetry
            if i1 == j1
                divisor = (j1 == k1) ? 6 : 2
            else
                divisor = (j1 == k1 || i1 == k1) ? 2 : 1
            end
            for i2 in 1:n_cols, j2 in 1:n_cols
                j2 ≤ i2 || continue
                for k2 in 1:n_cols
                    k2 ≤ j2 || continue
                    col = (i2 - 1) * i2 * (i2 + 1) ÷ 6 + (j2 - 1) * j2 ÷ 2 + k2
                    g = ∂Y[row, col]
                    iszero(g) && continue
                    g_d = g / divisor
                    @inbounds aii = Xd[i1, i2]; aij = Xd[i1, j2]; aik = Xd[i1, k2]
                    @inbounds aji = Xd[j1, i2]; ajj = Xd[j1, j2]; ajk = Xd[j1, k2]
                    @inbounds aki = Xd[k1, i2]; akj = Xd[k1, j2]; akk = Xd[k1, k2]
                    ∂X[i1, i2] += g_d * (ajj * akk + ajk * akj)
                    ∂X[i1, j2] += g_d * (aji * akk + ajk * aki)
                    ∂X[i1, k2] += g_d * (aji * akj + ajj * aki)
                    ∂X[j1, i2] += g_d * (aij * akk + aik * akj)
                    ∂X[j1, j2] += g_d * (aii * akk + aik * aki)
                    ∂X[j1, k2] += g_d * (aij * aki + aii * akj)
                    ∂X[k1, i2] += g_d * (aij * ajk + aik * ajj)
                    ∂X[k1, j2] += g_d * (aik * aji + aii * ajk)
                    ∂X[k1, k2] += g_d * (aii * ajj + aij * aji)
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
                    𝐒₂::SparseMatrixCSC{S},
                    constants::constants,
                    workspaces::workspaces,
                    cache::caches;
                    initial_guess::AbstractMatrix{R} = zeros(0,0),
                    opts::CalculationOptions = merge_calculation_options()) where {S <: Real, R <: Real}

    # --- workspace / constants ---------------------------------------------------
    if !(eltype(workspaces.third_order.Ŝ) == S)
        workspaces.third_order = Higher_order_workspace(T = S)
    end
    ℂ = workspaces.third_order
    M₂ = constants.second_order
    M₃ = constants.third_order
    T = constants.post_model_macro

    i₊ = T.future_not_past_and_mixed_idx
    i₋ = T.past_not_future_and_mixed_idx
    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo
    n  = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    initial_guess_sylv = if length(initial_guess) == 0
        zeros(S, 0, 0)
    elseif eltype(initial_guess) <: AbstractFloat
        initial_guess isa Matrix{S} ? initial_guess : Matrix{S}(initial_guess)
    else
        zeros(S, 0, 0)
    end

    # --- forward pass (mirrors the primal, but stores intermediates) ---------------

    # 1st-order solution with zero-column
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]]

    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) ℒ.I(nₑ + 1)[1,:] zeros(nₑ + 1, nₑ)]
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                ℒ.I(nₑ₋)[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]]

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]; zeros(n₋ + n + nₑ, nₑ₋)]
    𝐒₁₊╱𝟎 = choose_matrix_format(𝐒₁₊╱𝟎, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.I(n)[i₋,:] - ∇₁[:,range(1,n) .+ n₊]

    ∇₁₊𝐒₁➕∇₁₀lu = ℒ.lu(∇₁₊𝐒₁➕∇₁₀, check = false)

    if !ℒ.issuccess(∇₁₊𝐒₁➕∇₁₀lu)
        return (∇₁₊𝐒₁➕∇₁₀, false), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    spinv = inv(∇₁₊𝐒₁➕∇₁₀lu)
    spinv = choose_matrix_format(spinv)

    ∇₁₊ = @views ∇₁[:,1:n₊] * ℒ.I(n)[i₊,:]

    A = spinv * ∇₁₊

    # --- B matrix -----------------------------------------------------------------
    tmpkron_σ = ℒ.kron(𝐒₁₋╱𝟏ₑ, M₂.𝛔)
    kron𝐒₁₋╱𝟏ₑ = ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ)

    B_pre = tmpkron_σ + M₃.𝐏₁ₗ̄ * tmpkron_σ * M₃.𝐏₁ᵣ̃ + M₃.𝐏₂ₗ̄ * tmpkron_σ * M₃.𝐏₂ᵣ̃
    B_pre *= M₃.𝐂₃
    B = choose_matrix_format(M₃.𝐔₃ * B_pre, tol = opts.tol.droptol, multithreaded = false)

    ck3_𝐒₁₋╱𝟏ₑ = compressed_kron³(𝐒₁₋╱𝟏ₑ, tol = opts.tol.droptol, sparse_preallocation = ℂ.tmp_sparse_prealloc1)
    B += ck3_𝐒₁₋╱𝟏ₑ

    # --- 𝐗₃ (C-matrix ingredients) -----------------------------------------------
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [(𝐒₂ * kron𝐒₁₋╱𝟏ₑ + 𝐒₁ * [𝐒₂[i₋,:]; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
                                          𝐒₂
                                          zeros(n₋ + nₑ, nₑ₋^2)]
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = choose_matrix_format(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, density_threshold = 0.0, min_length = 10, tol = opts.tol.droptol)

    𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:]; zeros(n₋ + n + nₑ, nₑ₋^2)]

    aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋

    # tmpkron0 = kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)
    tmpkron0 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)
    # tmpkron22 = kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, tmpkron0 * 𝛔)
    tmpkron22 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, tmpkron0 * M₂.𝛔)

    𝐔∇₃ = ∇₃ * M₃.𝐔∇₃

    K22_sum = tmpkron22 + M₃.𝐏₁ₗ̂ * tmpkron22 * M₃.𝐏₁ᵣ̃ + M₃.𝐏₂ₗ̂ * tmpkron22 * M₃.𝐏₂ᵣ̃

    𝐗₃_∇₃_term = 𝐔∇₃ * K22_sum   # the ∇₃-dependent part (before 𝐂₃ and ck3)

    𝐒₂₊╱𝟎 = choose_matrix_format(𝐒₂₊╱𝟎, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    tmpkron1 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎)
    tmpkron2 = ℒ.kron(M₂.𝛔, 𝐒₁₋╱𝟏ₑ)

    ∇₁₊ = choose_matrix_format(∇₁₊, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    𝐒₂₋╱𝟎 = [𝐒₂[i₋,:]; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]

    out2  = ∇₂ * tmpkron1 * tmpkron2
    out2 += ∇₂ * tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ
    out2 += mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc2)
    out2 += mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, collect(𝐒₂₊╱𝟎 * M₂.𝛔), sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc3)

    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0, tol = opts.tol.droptol)
    mm_𝐒₂_kron = mat_mult_kron(𝐒₂, 𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc4)
    out2 += ∇₁₊ * mm_𝐒₂_kron

    𝐗₃_pre = 𝐗₃_∇₃_term + out2 * M₃.𝐏    # before 𝐂₃ compression

    𝐗₃ = 𝐗₃_pre * M₃.𝐂₃

    # Compute compressed_kron³(aux) WITHOUT rowmask: the pullback needs ∂∇₃ at ALL
    # positions (including currently-zero columns of ∇₃) so that gradients flow
    # correctly through calculate_third_order_derivatives back to parameters.
    ck3_aux_mat = compressed_kron³(aux, tol = opts.tol.droptol, sparse_preallocation = ℂ.tmp_sparse_prealloc5)
    ck3_aux = ∇₃ * ck3_aux_mat
    𝐗₃ += ck3_aux

    C = spinv * 𝐗₃

    # --- solve Sylvester  A·𝐒₃·B + C = 𝐒₃ ----------------------------------------
    𝐒₃, solved = solve_sylvester_equation(A, B, C, ℂ.sylvester_workspace,
                                            initial_guess = initial_guess_sylv,
                                            sylvester_algorithm = opts.sylvester_algorithm³,
                                            tol = opts.tol.sylvester_tol,
                                            acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                            verbose = opts.verbose)

    𝐒₃ = choose_matrix_format(𝐒₃, multithreaded = false, tol = opts.tol.droptol)

    if !solved
        return (𝐒₃, solved), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # cache update (same as primal)
    if 𝐒₃ isa Matrix{S} && cache.third_order_solution isa Matrix{S} && size(cache.third_order_solution) == size(𝐒₃)
        copyto!(cache.third_order_solution, 𝐒₃)
    elseif 𝐒₃ isa SparseMatrixCSC{S, Int} && cache.third_order_solution isa SparseMatrixCSC{S, Int} &&
           size(cache.third_order_solution) == size(𝐒₃) &&
           cache.third_order_solution.colptr == 𝐒₃.colptr &&
           cache.third_order_solution.rowval == 𝐒₃.rowval
        copyto!(cache.third_order_solution.nzval, 𝐒₃.nzval)
    else
        cache.third_order_solution = 𝐒₃
    end

    # --- precompute transposed constants for pullback -----------------------------
    𝐂₃t = choose_matrix_format(M₃.𝐂₃', density_threshold = 1.0)
    𝐔₃t = choose_matrix_format(M₃.𝐔₃', density_threshold = 1.0)
    𝐏t  = choose_matrix_format(M₃.𝐏',  density_threshold = 1.0)
    𝐔∇₃t = choose_matrix_format(M₃.𝐔∇₃', density_threshold = 1.0)
    𝛔t  = choose_matrix_format(M₂.𝛔', density_threshold = 1.0)

    # ck3_aux_mat already computed above (without rowmask) — reuse for pullback

    # =========================================================================
    #   PULLBACK
    # =========================================================================
    function third_order_solution_pullback(∂𝐒₃_solved)
        ∂𝐒₃ = ∂𝐒₃_solved[1]

        if ℒ.norm(∂𝐒₃) < opts.tol.sylvester_tol
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end

        # --- adjoint Sylvester:  Aᵀ ∂C_adj Bᵀ + ∂C_adj = ∂𝐒₃ --------------------
        ∂C_adj, slvd = solve_sylvester_equation(A', B', Matrix{Float64}(∂𝐒₃), ℂ.sylvester_workspace,
                                                  sylvester_algorithm = opts.sylvester_algorithm³,
                                                  tol = opts.tol.sylvester_tol,
                                                  acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                                  verbose = opts.verbose)
        if !slvd
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end

        ∂C_adj = choose_matrix_format(∂C_adj)

        # --- gradient of A, B, C from 𝐒₃ = A·𝐒₃·B + C ---------------------------
        ∂A = ∂C_adj * B' * 𝐒₃'
        ∂B_from_sylv = 𝐒₃' * A' * ∂C_adj

        # C = spinv * 𝐗₃
        ∂𝐗₃   = spinv' * ∂C_adj
        ∂spinv = ∂C_adj * 𝐗₃'

        # A = spinv * ∇₁₊
        ∂spinv += ∂A * ∇₁₊'

        # =====================================================================
        #  ∂∇₃  (linear: ∇₃ appears in two additive terms of 𝐗₃)
        # =====================================================================
        # Term 1:  𝐗₃ contains (∇₃·𝐔∇₃)·K22_sum  (goes through ·𝐂₃ then ·spinv⁻¹)
        #   i.e.  𝐗₃_pre_part1 = ∇₃ · 𝐔∇₃ · K22_sum  →  𝐗₃ += 𝐗₃_pre_part1 · 𝐂₃
        #   ∂∇₃_term1 = ∂𝐗₃ · 𝐂₃ᵀ · K22_sumᵀ · 𝐔∇₃ᵀ  (but that's = ∂𝐗₃_pre · K22_sumᵀ · 𝐔∇₃ᵀ)
        # Term 2:  𝐗₃ += ∇₃ · ck3_aux_mat
        #   ∂∇₃_term2 = ∂𝐗₃ · ck3_aux_matᵀ

        ∂𝐗₃_pre = ∂𝐗₃ * 𝐂₃t   # adjoint of 𝐗₃ = 𝐗₃_pre * 𝐂₃ + ck3_aux

        ∂∇₃ = ∂𝐗₃_pre * K22_sum' * 𝐔∇₃t + ∂𝐗₃ * ck3_aux_mat'

        # =====================================================================
        #  ∂∇₂  (∇₂ is linear in out2 → 𝐗₃_pre → 𝐗₃)
        # =====================================================================
        # out2 enters 𝐗₃_pre as:  𝐗₃_pre = ... + out2 · 𝐏
        # ∂out2 = ∂𝐗₃_pre · 𝐏ᵀ
        ∂out2 = ∂𝐗₃_pre * 𝐏t

        # out2  = ∇₂ · tmpkron1 · tmpkron2                                      (term a)
        #       + ∇₂ · tmpkron1 · 𝐏₁ₗ · tmpkron2 · 𝐏₁ᵣ                        (term b)
        #       + ∇₂ · kron(⎸𝐒₁..⎹, ⎸𝐒₂..⎹)                                   (term c)
        #       + ∇₂ · kron(⎸𝐒₁..⎹, 𝐒₂₊╱𝟎·𝛔)                                  (term d)
        #   (term 8 = ∇₁₊ · mm_𝐒₂_kron does not involve ∇₂.)

        # For correctness-first: materialize kron products
        R_a = tmpkron1 * tmpkron2                                       # term a right factor
        R_b = tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ                  # term b right factor
        R_c = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)  # term c right factor
        R_d = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, collect(𝐒₂₊╱𝟎 * M₂.𝛔))   # term d right factor

        ∂∇₂ = ∂out2 * R_a' + ∂out2 * R_b' + ∂out2 * R_c' + ∂out2 * R_d'


        # =====================================================================
        #  ∂𝐒₂  (𝐒₂ enters out2 via several stacking matrices)
        # =====================================================================
        # 𝐒₂ does NOT affect A, B, or the ∇₃ terms — only out2.
        # We already have ∂out2 = ∂𝐗₃_pre · 𝐏ᵀ from the ∂∇₂ section above.
        #
        # out2 terms that depend on 𝐒₂:
        #   (a) ∇₂ · tmpkron1 · tmpkron2           — tmpkron1 = kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎)
        #   (b) ∇₂ · tmpkron1 · 𝐏₁ₗ · tmpkron2 · 𝐏₁ᵣ  — same tmpkron1
        #   (c) ∇₂ · kron(⎸𝐒₁..⎹, ⎸𝐒₂k..⎹)       — second factor depends on 𝐒₂
        #   (d) ∇₂ · kron(⎸𝐒₁..⎹, 𝐒₂₊╱𝟎·𝛔)       — second factor depends on 𝐒₂
        #   (8) ∇₁₊ · 𝐒₂ · kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)  — both 𝐒₂ and 𝐒₂₋╱𝟎 depend on 𝐒₂

        ∂𝐒₂ = zeros(S, size(𝐒₂))

        # --- terms (a) and (b):  through tmpkron1 = kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎) ---
        # ∂(∇₂·tmpkron1·R) w.r.t. tmpkron1 = ∇₂ᵀ·∂out2·Rᵀ
        ∂tmpkron1  = ∇₂' * ∂out2 * tmpkron2'                            # from (a)
        ∂tmpkron1 += ∇₂' * ∂out2 * (M₃.𝐏₁ᵣ' * tmpkron2' * M₃.𝐏₁ₗ')    # from (b)

        # kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎) pullback → ∂𝐒₂₊╱𝟎 via fill_kron_adjoint!
        ∂𝐒₁₊╱𝟎_tmp = zeros(S, size(𝐒₁₊╱𝟎))
        ∂𝐒₂₊╱𝟎 = zeros(S, size(𝐒₂₊╱𝟎))
        fill_kron_adjoint!(∂𝐒₂₊╱𝟎, ∂𝐒₁₊╱𝟎_tmp, Matrix{S}(∂tmpkron1), Matrix{S}(𝐒₂₊╱𝟎), 𝐒₁₊╱𝟎)

        # 𝐒₂₊╱𝟎 = [𝐒₂[i₊,:]; 0]  →  ∂𝐒₂[i₊,:] += ∂𝐒₂₊╱𝟎[1:length(i₊),:]
        ∂𝐒₂[i₊,:] += ∂𝐒₂₊╱𝟎[1:length(i₊),:]

        # --- term (c): through ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 ---
        # ∇₂ · kron(⎸𝐒₁..⎹, ⎸𝐒₂..⎹)  →  ∂kron_c = ∇₂ᵀ · ∂out2
        ∂kron_c = ∇₂' * ∂out2
        # kron(L, R) pullback  where L = ⎸𝐒₁..⎹, R = ⎸𝐒₂k..⎹
        ∂L_c = zeros(S, size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋))
        ∂R_c = zeros(S, size(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎))
        fill_kron_adjoint!(∂R_c, ∂L_c, Matrix{S}(∂kron_c), Matrix{S}(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎), Matrix{S}(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋))

        # ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = [ (𝐒₂·kron𝐒₁₋╱𝟏ₑ + 𝐒₁·[𝐒₂[i₋,:];0])[i₊,:] ; 𝐒₂ ; 0 ]
        # Top block (rows 1:n₊): depends on 𝐒₂ through 𝐒₂·kron𝐒₁₋╱𝟏ₑ and 𝐒₁·[𝐒₂[i₋,:];0]
        n₊_len = length(i₊)
        ∂top_block = ∂R_c[1:n₊_len, :]
        # From 𝐒₂·kron𝐒₁₋╱𝟏ₑ:
        ∂𝐒₂ += ℒ.I(n)[:,i₊] * ∂top_block * kron𝐒₁₋╱𝟏ₑ'
        # From 𝐒₁·[𝐒₂[i₋,:];0] → ∂𝐒₂[i₋,:] += 𝐒₁' * I[:,i₊] * ∂top_block
        #   (since [𝐒₂[i₋,:];0] pads with zeros, only i₋ rows of 𝐒₂ contribute)
        ∂𝐒₂_padded = 𝐒₁' * ℒ.I(n)[:,i₊] * ∂top_block   # n₋+1+nₑ × nₑ₋²
        ∂𝐒₂[i₋,:] += ∂𝐒₂_padded[1:n₋, :]

        # Middle block (rows n₊_len+1 : n₊_len+n): directly 𝐒₂
        ∂𝐒₂ += ∂R_c[n₊_len .+ (1:n), :]

        # Bottom block is zeros

        # --- term (d): through kron(⎸𝐒₁..⎹, 𝐒₂₊╱𝟎·𝛔) ---
        # ∇₂ · kron(⎸𝐒₁..⎹, 𝐒₂₊╱𝟎·𝛔)  →  ∂kron_d = ∇₂ᵀ · ∂out2
        # (same ∂kron_d = ∂kron_c since ∂out2 is the total adjoint — but we need
        #  the Kron adjoint for the actual kron pair (L, 𝐒₂₊╱𝟎·𝛔) )
        ∂L_d = zeros(S, size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋))
        S2p0_sigma = collect(𝐒₂₊╱𝟎 * M₂.𝛔)
        ∂R_d = zeros(S, size(S2p0_sigma))
        fill_kron_adjoint!(∂R_d, ∂L_d, Matrix{S}(∂kron_c), Matrix{S}(S2p0_sigma), Matrix{S}(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋))

        # 𝐒₂₊╱𝟎·𝛔  →  ∂𝐒₂₊╱𝟎_d = ∂R_d · 𝛔ᵀ
        ∂𝐒₂₊╱𝟎_d = ∂R_d * 𝛔t
        ∂𝐒₂[i₊,:] += ∂𝐒₂₊╱𝟎_d[1:length(i₊),:]

        # --- term (8): ∇₁₊ · 𝐒₂ · kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎) ---
        # out2_term8 = ∇₁₊ · 𝐒₂ · kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)
        # ∂(∇₁₊·𝐒₂·K) w.r.t. 𝐒₂ = ∇₁₊ᵀ · ∂out2 · Kᵀ
        kron_s1_s2 = ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)
        ∂𝐒₂ += ∇₁₊' * ∂out2 * kron_s1_s2'

        # ∂(∇₁₊·𝐒₂·kron(𝐒₁₋╱𝟏ₑ,𝐒₂₋╱𝟎)) w.r.t. 𝐒₂₋╱𝟎  (through the kron)
        # ∂kron_term8 = (∇₁₊·𝐒₂)ᵀ · ∂out2
        ∂kron_term8 = (∇₁₊ * 𝐒₂)' * ∂out2
        ∂𝐒₁₋╱𝟏ₑ_t8 = zeros(S, size(𝐒₁₋╱𝟏ₑ))
        ∂𝐒₂₋╱𝟎 = zeros(S, size(𝐒₂₋╱𝟎))
        fill_kron_adjoint!(∂𝐒₂₋╱𝟎, ∂𝐒₁₋╱𝟏ₑ_t8, Matrix{S}(∂kron_term8), Matrix{S}(𝐒₂₋╱𝟎), Matrix{S}(𝐒₁₋╱𝟏ₑ))

        # 𝐒₂₋╱𝟎 = [𝐒₂[i₋,:]; 0]  →  ∂𝐒₂[i₋,:] += ∂𝐒₂₋╱𝟎[1:n₋,:]
        ∂𝐒₂[i₋,:] += ∂𝐒₂₋╱𝟎[1:n₋,:]

        # =====================================================================
        #  ∂∇₁
        # =====================================================================
        # ∇₁ enters through:
        #   1. ∇₁₊𝐒₁➕∇₁₀ = -∇₁[:,1:n₊]·𝐒₁[i₊,1:n₋]·I[i₋,:] - ∇₁[:,n₊+1:n₊+n]
        #      → spinv = inv(∇₁₊𝐒₁➕∇₁₀)  →  used in A and C
        #   2. ∇₁₊ = ∇₁[:,1:n₊] · I(n)[i₊,:]
        #      → A = spinv·∇₁₊   and   out2 += ∇₁₊ · mm_𝐒₂_kron

        # step 1: ∂ through inv(∇₁₊𝐒₁➕∇₁₀)  (∂spinv already accumulated)
        ∂∇₁₊𝐒₁➕∇₁₀ = -spinv' * ∂spinv * spinv'

        ∂∇₁ = zeros(S, size(∇₁))
        ∂∇₁[:,1:n₊] -= ∂∇₁₊𝐒₁➕∇₁₀ * ℒ.I(n)[:,i₋] * 𝐒₁[i₊,1:n₋]'
        ∂∇₁[:,range(1,n) .+ n₊] -= ∂∇₁₊𝐒₁➕∇₁₀

        # step 2: ∂ through ∇₁₊
        ∂∇₁₊ = spinv' * ∂A             # from A = spinv · ∇₁₊
        ∂∇₁₊ += ∂out2 * mm_𝐒₂_kron'    # from out2 += ∇₁₊ · mm_𝐒₂_kron

        ∂∇₁[:,1:n₊] += ∂∇₁₊ * ℒ.I(n)[:,i₊]

        # =====================================================================
        #  ∂𝑺₁  (𝑺₁ enters through 𝐒₁, affecting A,B,C,out2 via many paths)
        # =====================================================================
        ∂𝐒₁₋╱𝟏ₑ₃ = zeros(S, size(𝐒₁₋╱𝟏ₑ))
        ∂𝐒₁₊╱𝟎₃ = zeros(S, size(𝐒₁₊╱𝟎))
        ∂S1S1_stack = zeros(S, size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋))
        ∂𝐒₁₃ = zeros(S, n, nₑ₋)

        # --- ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ : from out2 terms c,d (kron outer factors) ---
        ∂S1S1_stack .+= ∂L_c .+ ∂L_d

        # --- ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ + ∂𝐒₁₊╱𝟎 : from K22_sum → tmpkron22 ---
        ∂K22_sum = 𝐔∇₃' * ∂𝐗₃_pre
        ∂tmpkron22 = ∂K22_sum + M₃.𝐏₁ₗ̂' * ∂K22_sum * M₃.𝐏₁ᵣ̃' + M₃.𝐏₂ₗ̂' * ∂K22_sum * M₃.𝐏₂ᵣ̃'
        tmpkron0_σ = collect(tmpkron0 * M₂.𝛔)
        ∂tmpkron0_σ = zeros(S, size(tmpkron0_σ))
        ∂S1S1_from22 = zeros(S, size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋))
        fill_kron_adjoint!(∂tmpkron0_σ, ∂S1S1_from22, Matrix{S}(∂tmpkron22), Matrix{S}(tmpkron0_σ), Matrix{S}(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋))
        ∂S1S1_stack .+= ∂S1S1_from22
        ∂tmpkron0 = ∂tmpkron0_σ * 𝛔t
        ∂𝐒₁₊╱𝟎_tk0 = zeros(S, size(𝐒₁₊╱𝟎))
        fill_kron_adjoint!(∂𝐒₁₊╱𝟎_tk0, ∂𝐒₁₊╱𝟎_tk0, Matrix{S}(∂tmpkron0), Matrix{S}(𝐒₁₊╱𝟎), Matrix{S}(𝐒₁₊╱𝟎))
        ∂𝐒₁₊╱𝟎₃ .+= ∂𝐒₁₊╱𝟎_tk0

        # --- ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ : from compressed_kron³(aux) → 𝐗₃ ---
        ∂ck3_aux = ∇₃' * ∂𝐗₃
        ∂aux = zeros(S, size(aux))
        compressed_kron³_pullback!(∂aux, Matrix{S}(∂ck3_aux), Matrix{S}(aux))
        ∂S1S1_stack .+= M₃.𝐒𝐏' * ∂aux

        # --- ∂𝐒₁₊╱𝟎 : from tmpkron1 (already computed for ∂𝐒₂) ---
        ∂𝐒₁₊╱𝟎₃ .+= ∂𝐒₁₊╱𝟎_tmp

        # --- ∂𝐒₁₋╱𝟏ₑ : from B via tmpkron_σ = kron(B=𝐒₁₋╱𝟏ₑ, A=𝛔) ---
        ∂B_pre = 𝐔₃t * ∂B_from_sylv
        ∂B_pre_raw = ∂B_pre * 𝐂₃t
        ∂tmpkron_σ₃ = ∂B_pre_raw + M₃.𝐏₁ₗ̄' * ∂B_pre_raw * M₃.𝐏₁ᵣ̃' + M₃.𝐏₂ₗ̄' * ∂B_pre_raw * M₃.𝐏₂ᵣ̃'
        ∂𝛔_discard = zeros(S, size(M₂.𝛔))
        fill_kron_adjoint!(∂𝛔_discard, ∂𝐒₁₋╱𝟏ₑ₃, Matrix{S}(∂tmpkron_σ₃), Matrix{S}(M₂.𝛔), Matrix{S}(𝐒₁₋╱𝟏ₑ))

        # --- ∂𝐒₁₋╱𝟏ₑ : from B via compressed_kron³(𝐒₁₋╱𝟏ₑ) ---
        compressed_kron³_pullback!(∂𝐒₁₋╱𝟏ₑ₃, Matrix{S}(∂B_from_sylv), Matrix{S}(𝐒₁₋╱𝟏ₑ))

        # --- ∂𝐒₁₋╱𝟏ₑ : from out2 terms a,b via tmpkron2 = kron(B=𝛔, A=𝐒₁₋╱𝟏ₑ) ---
        tmp_a = tmpkron1' * ∇₂' * ∂out2
        ∂tmpkron2 = tmp_a + M₃.𝐏₁ₗ' * tmp_a * M₃.𝐏₁ᵣ'
        ∂𝛔_discard2 = zeros(S, size(M₂.𝛔))
        fill_kron_adjoint!(∂𝐒₁₋╱𝟏ₑ₃, ∂𝛔_discard2, Matrix{S}(∂tmpkron2), Matrix{S}(𝐒₁₋╱𝟏ₑ), Matrix{S}(M₂.𝛔))

        # --- ∂𝐒₁₋╱𝟏ₑ : from term 8 kron (already computed for ∂𝐒₂) ---
        ∂𝐒₁₋╱𝟏ₑ₃ .+= ∂𝐒₁₋╱𝟏ₑ_t8

        # --- ∂𝐒₁₋╱𝟏ₑ : from kron𝐒₁₋╱𝟏ₑ in ⎸𝐒₂k..⎹ top block ---
        ∂kron𝐒₁₋╱𝟏ₑ₃ = Matrix{S}(𝐒₂' * ℒ.I(n)[:,i₊] * ∂top_block)
        fill_kron_adjoint!(∂𝐒₁₋╱𝟏ₑ₃, ∂𝐒₁₋╱𝟏ₑ₃, ∂kron𝐒₁₋╱𝟏ₑ₃, Matrix{S}(𝐒₁₋╱𝟏ₑ), Matrix{S}(𝐒₁₋╱𝟏ₑ))

        # --- ∂𝐒₁ : from 𝐒₁·[𝐒₂[i₋,:];0] in ⎸𝐒₂k..⎹ top block ---
        S2_padded = [𝐒₂[i₋,:]; zeros(S, nₑ + 1, nₑ₋^2)]
        ∂𝐒₁₃ += ℒ.I(n)[:,i₊] * ∂top_block * S2_padded'

        # === Convert ∂S1S1_stack → ∂𝐒₁ and ∂𝐒₁₋╱𝟏ₑ ===
        n₊l = length(i₊)
        ∂top_S1S1 = ∂S1S1_stack[1:n₊l, :]
        ∂𝐒₁₃ += ℒ.I(n)[:,i₊] * ∂top_S1S1 * 𝐒₁₋╱𝟏ₑ'
        ∂𝐒₁₋╱𝟏ₑ₃ += 𝐒₁' * ℒ.I(n)[:,i₊] * ∂top_S1S1
        ∂𝐒₁₃ += ∂S1S1_stack[n₊l .+ (1:n), :]

        # === Convert ∂𝐒₁₊╱𝟎ₓ → ∂𝐒₁ ===
        ∂𝐒₁₃[i₊,:] += ∂𝐒₁₊╱𝟎₃[1:n₊l,:]

        # === Convert ∂𝐒₁₋╱𝟏ₑ → ∂𝐒₁ ===
        ∂𝐒₁₃[i₋,:] += ∂𝐒₁₋╱𝟏ₑ₃[1:length(i₋),:]

        # === ∂𝐒₁ from ∇₁₊𝐒₁➕∇₁₀ (spinv) ===
        ∂𝐒₁₃[i₊,1:n₋] -= ∇₁[:,1:n₊]' * ∂∇₁₊𝐒₁➕∇₁₀ * ℒ.I(n)[:,i₋]

        # === 𝐒₁ = [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] → ∂𝑺₁ ===
        ∂𝑺₁ = [∂𝐒₁₃[:,1:n₋] ∂𝐒₁₃[:,n₋+2:end]]

        return (NoTangent(), ∂∇₁, ∂∇₂, ∂∇₃, ∂𝑺₁, ∂𝐒₂, NoTangent(), NoTangent(), NoTangent())
    end

    return (𝐒₃, solved), third_order_solution_pullback
end


function rrule(::typeof(solve_sylvester_equation),
    A::M,
    B::N,
    C::O,
    𝕊ℂ::sylvester_workspace;
    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
    sylvester_algorithm::Symbol = :doubling,
    acceptance_tol::AbstractFloat = 1e-10,
    tol::AbstractFloat = 1e-14,
    # timer::TimerOutput = TimerOutput(),
    verbose::Bool = false) where {M <: AbstractMatrix{Float64}, N <: AbstractMatrix{Float64}, O <: AbstractMatrix{Float64}}

    P, solved = solve_sylvester_equation(A, B, C, 𝕊ℂ,
                                        sylvester_algorithm = sylvester_algorithm, 
                                        tol = tol, 
                                        verbose = verbose, 
                                        initial_guess = initial_guess)

    ensure_sylvester_doubling_buffers!(𝕊ℂ, size(A, 1), size(B, 1))

    # pullback
    function solve_sylvester_equation_pullback(∂P)
        if ℒ.norm(∂P[1]) < tol return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent() end

        ∂C, slvd = solve_sylvester_equation(A', B', ∂P[1], 𝕊ℂ,
                                            sylvester_algorithm = sylvester_algorithm, 
                                            tol = tol, 
                                            verbose = verbose)

        solved = solved && slvd

        tmp_n = 𝕊ℂ.𝐀
        tmp_m = 𝕊ℂ.𝐁

        ℒ.mul!(tmp_n, ∂C, B')
        ∂A = tmp_n * P'

        ℒ.mul!(tmp_m, P', A')
        ∂B = tmp_m * ∂C

        return NoTangent(), ∂A, ∂B, ∂C, NoTangent()
    end

    return (P, solved), solve_sylvester_equation_pullback
end

function rrule(::typeof(solve_lyapunov_equation),
                A::AbstractMatrix{Float64},
                C::AbstractMatrix{Float64},
                workspace::lyapunov_workspace;
                lyapunov_algorithm::Symbol = :doubling,
                tol::AbstractFloat = 1e-14,
                acceptance_tol::AbstractFloat = 1e-12,
                # timer::TimerOutput = TimerOutput(),
                verbose::Bool = false)

    P, solved = solve_lyapunov_equation(A, C, workspace, lyapunov_algorithm = lyapunov_algorithm, tol = tol, verbose = verbose)
    ensure_lyapunov_doubling_buffers!(workspace)

    # pullback 
    # https://arxiv.org/abs/2011.11430  
    function solve_lyapunov_equation_pullback(∂P)
        if ℒ.norm(∂P[1]) < tol return NoTangent(), NoTangent(), NoTangent(), NoTangent() end

        ∂C, slvd = solve_lyapunov_equation(A', ∂P[1], workspace, lyapunov_algorithm = lyapunov_algorithm,  tol = tol, verbose = verbose)
    
        solved = solved && slvd

        tmp_n1 = workspace.𝐂A
        tmp_n2 = workspace.𝐀²
        ∂A = zero(A)

        ℒ.mul!(tmp_n1, ∂C, A)
        ℒ.mul!(∂A, tmp_n1, P')

        ℒ.mul!(tmp_n2, ∂C', A)
        ℒ.mul!(∂A, tmp_n2, P, 1, 1)

        return NoTangent(), ∂A, ∂C, NoTangent()
    end
    
    return (P, solved), solve_lyapunov_equation_pullback
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

    @assert warmup_iterations == 0 "Warmup iterations not yet implemented for reverse-mode automatic differentiation."

    state = [copy(state) for _ in 1:size(data_in_deviations,2)+1]

    shocks² = 0.0
    logabsdets = 0.0

    y = zeros(length(obs_idx))
    x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]

    jac = 𝐒[obs_idx,end-T.nExo+1:end]

    if T.nExo == length(observables_index)
        logabsdets = ℒ.logabsdet(jac)[1] #  ./ precision_factor

        jacdecomp = ℒ.lu(jac, check = false)

        if !ℒ.issuccess(jacdecomp)
            if opts.verbose println("Inversion filter failed") end
            return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end

        invjac = inv(jacdecomp)
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

        ℒ.mul!(state[i+1], 𝐒, vcat(state[i][t⁻], x[i]))
        # state[i+1] =  𝐒 * vcat(state[i][t⁻], x[i])
    end

    llh = -(logabsdets + shocks² + (length(observables_index) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
    
    if llh < -1e12
        return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    ∂𝐒 = zero(𝐒)
    
    ∂𝐒ᵗ⁻ = copy(∂𝐒[t⁻,:])

    ∂data_in_deviations = zero(data_in_deviations)
    
    # Allocate or reuse workspaces for pullback
    n_periods = size(data_in_deviations,2) - 1
    if size(ws.∂data) != (length(t⁻), n_periods)
        ws.∂data = zeros(length(t⁻), n_periods)
    else
        fill!(ws.∂data, zero(eltype(ws.∂data)))
    end
    ∂data = ws.∂data

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

    # Allocate or reuse workspaces for temporary matrices
    if size(ws.∂_tmp1) != (T.nExo, length(t⁻) + T.nExo)
        ws.∂_tmp1 = zeros(Float64, T.nExo, length(t⁻) + T.nExo)
    else
        fill!(ws.∂_tmp1, zero(Float64))
    end
    tmp1 = ws.∂_tmp1
    
    if size(ws.∂_tmp2) != (length(t⁻), length(t⁻) + T.nExo)
        ws.∂_tmp2 = zeros(Float64, length(t⁻), length(t⁻) + T.nExo)
    else
        fill!(ws.∂_tmp2, zero(Float64))
    end
    tmp2 = ws.∂_tmp2
    
    if size(ws.∂_tmp3) != (length(t⁻) + T.nExo,)
        ws.∂_tmp3 = zeros(Float64, length(t⁻) + T.nExo)
    else
        fill!(ws.∂_tmp3, zero(Float64))
    end
    tmp3 = ws.∂_tmp3

    if size(ws.∂𝐒t⁻) != size(tmp2)
        ws.∂𝐒t⁻ = copy(tmp2)
    else
        fill!(ws.∂𝐒t⁻, zero(Float64))
    end
    ∂𝐒t⁻ = ws.∂𝐒t⁻
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

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂𝐒 * ∂llh, ∂data_in_deviations * ∂llh, NoTangent(), [∂state * ∂llh], NoTangent()
    end
    
    return llh, inversion_pullback
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

        jacc_fact = try
                        ℒ.factorize(jacct) # otherwise this fails for nshocks > nexo
                    catch
                        if opts.verbose println("Inversion filter failed at step $i") end
                        return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
                    end

        try
            ℒ.ldiv!(λ[i], jacc_fact, x[i])
        catch
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

        jacc_fact = try
                        ℒ.factorize(jacct)
                    catch
                        if opts.verbose println("Inversion filter failed at step $i") end
                        return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent())
                    end

        try
            ℒ.ldiv!(λ[i], jacc_fact, x[i])
        catch
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
    # pruned variant needs kron(e, s_in_s) (no vol), not the cached kron(e, s_in_s⁺)
    shockvar_idxs = sparse(ℒ.kron(cc.e_in_s⁺, cc.s_in_s)).nzind
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
    P = if initial_covariance == :theoretical
        lyap_rrule_result, lyap_pullback_local = rrule(solve_lyapunov_equation,
                                                        A,
                                                        𝐁,
                                                        lyap_ws,
                                                        lyapunov_algorithm = opts.lyapunov_algorithm,
                                                        tol = opts.tol.lyapunov_tol,
                                                        acceptance_tol = opts.tol.lyapunov_acceptance_tol,
                                                        verbose = opts.verbose)
        lyap_pullback = lyap_pullback_local
        lyap_rrule_result[1]
    else
        get_initial_covariance(Val(initial_covariance), A, 𝐁, lyap_ws, opts = opts)
    end

    Tt = size(data_in_deviations, 2) + 1

    z = zeros(size(data_in_deviations, 1))
    ū = zeros(size(C,2))
    P̄ = deepcopy(P)

    temp_N_N = similar(P)
    PCtmp = similar(P, size(P, 1), size(C, 1))
    F = similar(P, size(C, 1), size(C, 1))

    u = [similar(ū) for _ in 1:Tt]
    P_seq = [copy(P̄) for _ in 1:Tt]
    CP = [zeros(eltype(P), size(C, 1), size(P, 2)) for _ in 1:Tt]
    K = [similar(P, size(P, 1), size(C, 1)) for _ in 1:Tt]
    invF = [similar(F) for _ in 1:Tt]
    v = [zeros(size(data_in_deviations, 1)) for _ in 1:Tt]

    loglik = 0.0

    for t in 2:Tt
        if !all(isfinite.(z))
            if opts.verbose println("KF not finite at step $t") end
            return on_failure_loglikelihood, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end

        v[t] .= data_in_deviations[:, t-1] .- z

        ℒ.mul!(CP[t], C, P̄)
        ℒ.mul!(F, CP[t], C')

        kalman_ws.fast_lu_ws_f, kalman_ws.fast_lu_dims_f, solved_F, luF = factorize_lu!(F,
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

        fill!(invF[t], 0.0)
        @inbounds for i in 1:size(invF[t], 1)
            invF[t][i, i] = 1.0
        end
        solve_lu_left!(F, invF[t], kalman_ws.fast_lu_ws_f, luF)

        if t - 1 > presample_periods
            loglik += logabsdetF + ℒ.dot(v[t], invF[t], v[t])
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
        u[t] .+= ū

        ℒ.mul!(ū, A, u[t])
        ℒ.mul!(z, C, ū)
    end

    llh = -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2

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

        ∂A = copy(∂A_kf)
        ∂𝐁 = copy(∂𝐁_kf)

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


function _get_statistics_cotangent(Δret, key::Symbol)
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
                autocorrelation::Union{Symbol_input,String_input} = Symbol[],
                autocorrelation_periods::UnitRange{Int} = DEFAULT_AUTOCORRELATION_PERIODS,
                algorithm::Symbol = DEFAULT_ALGORITHM,
                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
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

    @assert algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] || !(!(standard_deviation == Symbol[]) || !(mean == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(autocorrelation == Symbol[])) "Statistics can only be provided for first order perturbation or second and third order pruned perturbation solutions."

    @assert !(non_stochastic_steady_state == Symbol[]) || !(standard_deviation == Symbol[]) || !(mean == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(autocorrelation == Symbol[]) "Provide variables for at least one output."

    SS_var_idx = parse_variables_input_to_index(non_stochastic_steady_state, 𝓂)
    mean_var_idx = parse_variables_input_to_index(mean, 𝓂)
    std_var_idx = parse_variables_input_to_index(standard_deviation, 𝓂)
    var_var_idx = parse_variables_input_to_index(variance, 𝓂)
    covar_var_idx = parse_variables_input_to_index(covariance, 𝓂)
    covar_groups = is_grouped_covariance_input(covariance) ? parse_covariance_groups(covariance, 𝓂.constants) : nothing
    autocorr_var_idx = parse_variables_input_to_index(autocorrelation, 𝓂)

    other_parameter_values = 𝓂.parameter_values[indexin(setdiff(𝓂.constants.post_complete_parameters.parameters, parameters), 𝓂.constants.post_complete_parameters.parameters)]
    sort_idx = sortperm(vcat(indexin(setdiff(𝓂.constants.post_complete_parameters.parameters, parameters), 𝓂.constants.post_complete_parameters.parameters), indexin(parameters, 𝓂.constants.post_complete_parameters.parameters)))

    all_parameters = vcat(other_parameter_values, parameter_values)[sort_idx]
    n_other = length(other_parameter_values)
    inv_sort = invperm(sort_idx)

    run_algorithm = algorithm
    if run_algorithm == :pruned_third_order && !(!(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(autocorrelation == Symbol[]))
        run_algorithm = :pruned_second_order
    end

        solve!(𝓂,
            algorithm = run_algorithm,
            steady_state_function = steady_state_function,
            opts = opts)

    nVars = length(𝓂.constants.post_model_macro.var)

    nsss_only = !(non_stochastic_steady_state == Symbol[]) && (standard_deviation == Symbol[]) && (variance == Symbol[]) && (covariance == Symbol[]) && (autocorrelation == Symbol[])

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

    if nsss_only
        prev_Δnsss = Ref{Any}(nothing)

        nsss_out, nsss_pb_local = rrule(get_NSSS_and_parameters, 𝓂, all_parameters; opts = opts)
        nsss_pb = nsss_pb_local

        SS_and_pars = nsss_out[1]
        solution_error = nsss_out[2][1]
        SS = SS_and_pars[1:end - length(𝓂.equations.calibration)]

        ret = Dict{Symbol,AbstractArray{T}}()
        ret[:non_stochastic_steady_state] = solution_error < opts.tol.NSSS_acceptance_tol ? SS[SS_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(SS_var_idx) ? 0 : length(SS_var_idx))

        function nsss_only_pullback(Δret)
            Δnsss = _incremental_cotangent!(_get_statistics_cotangent(Δret, :non_stochastic_steady_state), prev_Δnsss)
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
            second_mom_third_order = union(autocorr_var_idx, std_var_idx, var_var_idx)
            toma_out, toma_pb_local = rrule(calculate_third_order_moments_with_autocorrelation,
                                            all_parameters,
                                            𝓂.constants.post_model_macro.var[second_mom_third_order],
                                            𝓂;
                                            covariance = 𝓂.constants.post_model_macro.var[covar_var_idx],
                                            opts = opts,
                                            autocorrelation_periods = autocorrelation_periods)
            toma_pb = toma_pb_local

            covar_dcmp = toma_out[1]
            state_μ = toma_out[2]
            autocorr = toma_out[3]
            SS_and_pars = toma_out[4]
            solved = toma_out[5]
        elseif !(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[])
            tom_out, tom_pb_local = rrule(calculate_third_order_moments,
                                        all_parameters,
                                        𝓂.constants.post_model_macro.var[union(std_var_idx, var_var_idx)],
                                        𝓂;
                                        covariance = 𝓂.constants.post_model_macro.var[covar_var_idx],
                                        opts = opts)
            tom_pb = tom_pb_local

            covar_dcmp = tom_out[1]
            state_μ = tom_out[2]
            SS_and_pars = tom_out[3]
            solved = tom_out[4]
        end
    elseif run_algorithm == :pruned_second_order
        if !(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(autocorrelation == Symbol[])
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

            second_order_mask = ℒ.diag(covar_dcmp) .< opts.tol.lyapunov_acceptance_tol
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

            first_order_mask = ℒ.diag(covar_dcmp) .< opts.tol.lyapunov_acceptance_tol
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
    if !(autocorrelation == Symbol[])
        ret[:autocorrelation] = solved ? autocorr[autocorr_var_idx, :] : fill(Inf * sum(abs2,parameter_values), isnothing(autocorr_var_idx) ? 0 : length(autocorr_var_idx), isnothing(autocorrelation_periods) ? 0 : length(autocorrelation_periods))
    end

    prev_Δnsss = Ref{Any}(nothing)
    prev_Δmean = Ref{Any}(nothing)
    prev_Δstd = Ref{Any}(nothing)
    prev_Δvar = Ref{Any}(nothing)
    prev_Δcov = Ref{Any}(nothing)
    prev_Δautocorr = Ref{Any}(nothing)

    function get_statistics_pullback(Δret)
        if !solved
            return NoTangent(), NoTangent(), zeros(T, length(parameter_values))
        end

        Δnsss = _incremental_cotangent!(_get_statistics_cotangent(Δret, :non_stochastic_steady_state), prev_Δnsss)
        Δmean = _incremental_cotangent!(_get_statistics_cotangent(Δret, :mean), prev_Δmean)
        Δstd = _incremental_cotangent!(_get_statistics_cotangent(Δret, :standard_deviation), prev_Δstd)
        Δvar = _incremental_cotangent!(_get_statistics_cotangent(Δret, :variance), prev_Δvar)
        Δcov = _incremental_cotangent!(_get_statistics_cotangent(Δret, :covariance), prev_Δcov)
        Δautocorr = _incremental_cotangent!(_get_statistics_cotangent(Δret, :autocorrelation), prev_Δautocorr)

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
                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
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
    if length(𝓂.constants.post_parameters_macro.bounds) > 0
        for (k, v) in 𝓂.constants.post_parameters_macro.bounds
            if k ∈ 𝓂.constants.post_complete_parameters.parameters
                idx = indexin([k], 𝓂.constants.post_complete_parameters.parameters)[1]
                if min(max(parameters[idx], v[1]), v[2]) != parameters[idx]
                    return -Inf, zero_pullback
                end
            end
        end
    end

    # ── Step 1: NSSS ──
    nsss_out, nsss_pb = rrule(get_NSSS_and_parameters,
                              𝓂,
                              parameters;
                              opts = opts,
                              estimation = estimation)

    SS_and_pars = nsss_out[1]
    solution_error = nsss_out[2][1]

    if solution_error > tol.NSSS_acceptance_tol || isnan(solution_error)
        if algorithm in [:second_order, :pruned_second_order]
            result = (SS_and_pars[1:nVar], zeros(nVar, 2), spzeros(nVar, 2), false)
        elseif algorithm in [:third_order, :pruned_third_order]
            result = (SS_and_pars[1:nVar], zeros(nVar, 2), spzeros(nVar, 2), spzeros(nVar, 2), false)
        else
            result = (SS_and_pars[1:nVar], zeros(nVar, 2), false)
        end
        return result, zero_pullback
    end

    # ── Step 2: Jacobian ──
    ∇₁, jac_pb = rrule(calculate_jacobian,
                        parameters,
                        SS_and_pars,
                        𝓂.caches,
                        𝓂.functions.jacobian)

    # ── Step 3: First-order solution ──
    first_out, first_pb = rrule(calculate_first_order_solution,
                                ∇₁,
                                constants_obj,
                                𝓂.workspaces,
                                𝓂.caches;
                                opts = opts,
                                initial_guess = 𝓂.caches.qme_solution)

    𝐒₁ = first_out[1]
    solved = first_out[3]

    update_perturbation_counter!(𝓂.counters, solved, estimation = estimation, order = 1)

    if !solved
        if algorithm in [:second_order, :pruned_second_order]
            result = (SS_and_pars[1:nVar], 𝐒₁, spzeros(nVar, 2), false)
        elseif algorithm in [:third_order, :pruned_third_order]
            result = (SS_and_pars[1:nVar], 𝐒₁, spzeros(nVar, 2), spzeros(nVar, 2), false)
        else
            result = (SS_and_pars[1:nVar], 𝐒₁, false)
        end
        return result, zero_pullback
    end

    # ── Branch by algorithm ──
    if algorithm in [:second_order, :pruned_second_order]
        # ── Step 4: Hessian ──
        ∇₂, hess_pb = rrule(calculate_hessian,
                             parameters,
                             SS_and_pars,
                             𝓂.caches,
                             𝓂.functions.hessian)

        # ── Step 5: Second-order solution ──
        second_out, second_pb = rrule(calculate_second_order_solution,
                                      ∇₁, ∇₂, 𝐒₁,
                                      𝓂.constants,
                                      𝓂.workspaces,
                                      𝓂.caches;
                                      initial_guess = 𝓂.caches.second_order_solution,
                                      opts = opts)

        𝐒₂_raw = second_out[1]
        solved2 = second_out[2]

        update_perturbation_counter!(𝓂.counters, solved2, estimation = estimation, order = 2)

        𝐔₂ = 𝓂.constants.second_order.𝐔₂
        𝐒₂ = 𝐒₂_raw * 𝐔₂

        if !(typeof(𝐒₂) <: AbstractSparseMatrix)
            𝐒₂ = sparse(𝐒₂)
        end

        result = (SS_and_pars[1:nVar], 𝐒₁, 𝐒₂, true)

        pullback_2nd = function (∂result_bar)
            Δ = unthunk(∂result_bar)

            if Δ isa Union{NoTangent, AbstractZero}
                return NoTangent(), NoTangent(), zeros(S, length(parameters))
            end

            ∂NSSS    = Δ[1]
            ∂𝐒₁_ext = Δ[2]
            ∂𝐒₂_ext = Δ[3]
            # Δ[4] is ∂solved — not differentiable

            # ── Accumulate ∂SS_and_pars (zero-pad to full length) ──
            ∂SS_and_pars = zeros(S, length(SS_and_pars))
            if !(∂NSSS isa Union{NoTangent, AbstractZero})
                ∂SS_and_pars[1:nVar] .+= ∂NSSS
            end

            ∂parameters = zeros(S, length(parameters))

            # ── Adjoint of 𝐒₂ = 𝐒₂_raw * 𝐔₂ ──
            if ∂𝐒₂_ext isa Union{NoTangent, AbstractZero}
                ∂𝐒₂_raw = zeros(S, size(𝐒₂_raw))
            else
                ∂𝐒₂_raw = Matrix{S}(∂𝐒₂_ext) * 𝐔₂'
            end

            # ── second_pb: (∂𝐒₂_raw, ∂solved2) ──
            # Returns (NT, ∂∇₁, ∂∇₂, ∂𝑺₁, NT, NT, NT, NT, NT, NT)
            second_grads = second_pb((∂𝐒₂_raw, NoTangent()))
            ∂∇₁_from_2nd  = second_grads[2]
            ∂∇₂_from_2nd  = second_grads[3]
            ∂𝑺₁_from_2nd  = second_grads[4]

            # ── hess_pb ──
            # Returns (NT, ∂parameters, ∂SS_and_pars, NT, NT)
            hess_grads = hess_pb(∂∇₂_from_2nd)
            ∂parameters  .+= hess_grads[2]
            ∂SS_and_pars .+= hess_grads[3]

            # ── Accumulate ∂𝐒₁ ──
            ∂𝐒₁_total = if ∂𝐒₁_ext isa Union{NoTangent, AbstractZero}
                ∂𝑺₁_from_2nd
            else
                ∂𝐒₁_ext + ∂𝑺₁_from_2nd
            end

            # ── first_pb: (∂𝐒₁, ∂qme_sol, ∂solved) ──
            # Returns (NT, ∂∇₁, NT, NT, NT, NT)
            first_grads = first_pb((∂𝐒₁_total, NoTangent(), NoTangent()))
            ∂∇₁_total = ∂∇₁_from_2nd + first_grads[2]

            # ── jac_pb ──
            # Returns (NT, ∂parameters, ∂SS_and_pars, NT, NT)
            jac_grads = jac_pb(∂∇₁_total)
            ∂parameters  .+= jac_grads[2]
            ∂SS_and_pars .+= jac_grads[3]

            # ── nsss_pb ──
            # Returns (NT, NT, ∂parameter_values, NT)
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
                             𝓂.functions.hessian)

        # ── Step 5: Second-order solution ──
        second_out, second_pb = rrule(calculate_second_order_solution,
                                      ∇₁, ∇₂, 𝐒₁,
                                      𝓂.constants,
                                      𝓂.workspaces,
                                      𝓂.caches;
                                      initial_guess = 𝓂.caches.second_order_solution,
                                      opts = opts)

        𝐒₂_raw = second_out[1]
        solved2 = second_out[2]

        update_perturbation_counter!(𝓂.counters, solved2, estimation = estimation, order = 2)

        𝐔₂ = 𝓂.constants.second_order.𝐔₂
        𝐒₂ = 𝐒₂_raw * 𝐔₂

        if !(typeof(𝐒₂) <: AbstractSparseMatrix)
            𝐒₂ = sparse(𝐒₂)
        end

        # ── Step 6: Third-order derivatives ──
        ∇₃, third_deriv_pb = rrule(calculate_third_order_derivatives,
                                    parameters,
                                    SS_and_pars,
                                    𝓂.caches,
                                    𝓂.functions.third_order_derivatives)

        # ── Step 7: Third-order solution ──
        # calculate_third_order_solution receives 𝐒₂ after 𝐔₂ multiplication
        third_out, third_pb = rrule(calculate_third_order_solution,
                                    ∇₁, ∇₂, ∇₃,
                                    𝐒₁, 𝐒₂,
                                    𝓂.constants,
                                    𝓂.workspaces,
                                    𝓂.caches;
                                    initial_guess = 𝓂.caches.third_order_solution,
                                    opts = opts)

        𝐒₃_raw = third_out[1]
        solved3 = third_out[2]

        update_perturbation_counter!(𝓂.counters, solved3, estimation = estimation, order = 3)

        𝐔₃ = 𝓂.constants.third_order.𝐔₃
        𝐒₃ = 𝐒₃_raw * 𝐔₃

        if !(typeof(𝐒₃) <: AbstractSparseMatrix)
            𝐒₃ = sparse(𝐒₃)
        end

        result = (SS_and_pars[1:nVar], 𝐒₁, 𝐒₂, 𝐒₃, true)

        pullback_3rd = function (∂result_bar)
            Δ = unthunk(∂result_bar)

            if Δ isa Union{NoTangent, AbstractZero}
                return NoTangent(), NoTangent(), zeros(S, length(parameters))
            end

            ∂NSSS    = Δ[1]
            ∂𝐒₁_ext = Δ[2]
            ∂𝐒₂_ext = Δ[3]
            ∂𝐒₃_ext = Δ[4]
            # Δ[5] is ∂solved — not differentiable

            # ── Accumulate ∂SS_and_pars (zero-pad to full length) ──
            ∂SS_and_pars = zeros(S, length(SS_and_pars))
            if !(∂NSSS isa Union{NoTangent, AbstractZero})
                ∂SS_and_pars[1:nVar] .+= ∂NSSS
            end

            ∂parameters = zeros(S, length(parameters))

            # ── Adjoint of 𝐒₃ = 𝐒₃_raw * 𝐔₃ ──
            if ∂𝐒₃_ext isa Union{NoTangent, AbstractZero}
                ∂𝐒₃_raw = zeros(S, size(𝐒₃_raw))
            else
                ∂𝐒₃_raw = Matrix{S}(∂𝐒₃_ext) * 𝐔₃'
            end

            # ── third_pb: (∂𝐒₃_raw, ∂solved3) ──
            # Returns (NT, ∂∇₁, ∂∇₂, ∂∇₃, ∂𝑺₁, ∂𝐒₂, NT, NT, NT)
            third_grads = third_pb((∂𝐒₃_raw, NoTangent()))
            ∂∇₁_from_3rd  = third_grads[2]
            ∂∇₂_from_3rd  = third_grads[3]
            ∂∇₃_from_3rd  = third_grads[4]
            ∂𝑺₁_from_3rd  = third_grads[5]
            ∂𝐒₂_from_3rd  = third_grads[6]  # w.r.t. post-𝐔₂ version

            # ── third_deriv_pb ──
            # Returns (NT, ∂parameters, ∂SS_and_pars, NT, NT)
            third_deriv_grads = third_deriv_pb(∂∇₃_from_3rd)
            ∂parameters  .+= third_deriv_grads[2]
            ∂SS_and_pars .+= third_deriv_grads[3]

            # ── Accumulate ∂𝐒₂ (post-𝐔₂) from external + third-order ──
            ∂𝐒₂_post = if ∂𝐒₂_ext isa Union{NoTangent, AbstractZero}
                ∂𝐒₂_from_3rd isa Union{NoTangent, AbstractZero} ? zeros(S, size(𝐒₂)) : Matrix{S}(∂𝐒₂_from_3rd)
            else
                ∂𝐒₂_from_3rd isa Union{NoTangent, AbstractZero} ? Matrix{S}(∂𝐒₂_ext) : Matrix{S}(∂𝐒₂_ext) + Matrix{S}(∂𝐒₂_from_3rd)
            end

            # ── Adjoint of 𝐒₂ = 𝐒₂_raw * 𝐔₂ ──
            ∂𝐒₂_raw = ∂𝐒₂_post * 𝐔₂'

            # ── second_pb: (∂𝐒₂_raw, ∂solved2) ──
            # Returns (NT, ∂∇₁, ∂∇₂, ∂𝑺₁, NT, NT, NT, NT, NT, NT)
            second_grads = second_pb((∂𝐒₂_raw, NoTangent()))
            ∂∇₁_from_2nd  = second_grads[2]
            ∂∇₂_from_2nd  = second_grads[3]
            ∂𝑺₁_from_2nd  = second_grads[4]

            # ── hess_pb (accumulate ∂∇₂ from 2nd and 3rd order) ──
            # Returns (NT, ∂parameters, ∂SS_and_pars, NT, NT)
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

            # ── first_pb: (∂𝐒₁, ∂qme_sol, ∂solved) ──
            # Returns (NT, ∂∇₁, NT, NT, NT, NT)
            first_grads = first_pb((∂𝐒₁_total, NoTangent(), NoTangent()))
            ∂∇₁_total = ∂∇₁_from_3rd + ∂∇₁_from_2nd + first_grads[2]

            # ── jac_pb ──
            # Returns (NT, ∂parameters, ∂SS_and_pars, NT, NT)
            jac_grads = jac_pb(∂∇₁_total)
            ∂parameters  .+= jac_grads[2]
            ∂SS_and_pars .+= jac_grads[3]

            # ── nsss_pb ──
            # Returns (NT, NT, ∂parameter_values, NT)
            nsss_grads = nsss_pb((∂SS_and_pars, NoTangent()))
            ∂parameters .+= nsss_grads[3]

            return NoTangent(), NoTangent(), ∂parameters
        end

        return result, pullback_3rd

    else
        # ── First order ──
        result = (SS_and_pars[1:nVar], 𝐒₁, true)

        pullback_1st = function (∂result_bar)
            Δ = unthunk(∂result_bar)

            if Δ isa Union{NoTangent, AbstractZero}
                return NoTangent(), NoTangent(), zeros(S, length(parameters))
            end

            ∂NSSS    = Δ[1]
            ∂𝐒₁_ext = Δ[2]
            # Δ[3] is ∂solved — not differentiable

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
