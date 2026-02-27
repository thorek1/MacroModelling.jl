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
        jacobian_funcs.f_parameters(caches_obj.jacobian_parameters, parameters, SS_and_pars)
        jacobian_funcs.f_SS_and_pars(caches_obj.jacobian_SS_and_pars, parameters, SS_and_pars)

        ∂parameters = caches_obj.jacobian_parameters' * vec(∂∇₁)
        ∂SS_and_pars = caches_obj.jacobian_SS_and_pars' * vec(∂∇₁)
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
        hessian_funcs.f_parameters(caches_obj.hessian_parameters, parameters, SS_and_pars)
        hessian_funcs.f_SS_and_pars(caches_obj.hessian_SS_and_pars, parameters, SS_and_pars)

        ∂parameters = caches_obj.hessian_parameters' * vec(∂∇₂)
        ∂SS_and_pars = caches_obj.hessian_SS_and_pars' * vec(∂∇₂)

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
        third_order_derivatives_funcs.f_parameters(caches_obj.third_order_derivatives_parameters, parameters, SS_and_pars)
        third_order_derivatives_funcs.f_SS_and_pars(caches_obj.third_order_derivatives_SS_and_pars, parameters, SS_and_pars)

        ∂parameters = caches_obj.third_order_derivatives_parameters' * vec(∂∇₃)
        ∂SS_and_pars = caches_obj.third_order_derivatives_SS_and_pars' * vec(∂∇₃)

        return NoTangent(), ∂parameters, ∂SS_and_pars, NoTangent(), NoTangent()
    end

    return third_order_derivatives, calculate_third_order_derivatives_pullback
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
        # println(∂SS_and_pars)
        return NoTangent(), NoTangent(), jvp' * ∂SS_and_pars[1], NoTangent()
    end


    return (SS_and_pars, (solution_error, iters)), get_non_stochastic_steady_state_pullback
end

# function rrule(::typeof(get_relevant_steady_state_and_state_update),
#                 ::Val{:first_order},
#                 parameter_values::Vector{S},
#                 𝓂::ℳ;
#                 opts::CalculationOptions = merge_calculation_options(),
#                 estimation::Bool = false) where S <: AbstractFloat
#     constants_obj = initialise_constants!(𝓂)

#     nsss_out, nsss_pb = rrule(get_NSSS_and_parameters,
#                                 𝓂,
#                                 parameter_values;
#                                 opts = opts,
#                                 estimation = estimation)

#     SS_and_pars = nsss_out[1]
#     solution_error = nsss_out[2][1]

#     state = zeros(S, 𝓂.constants.post_model_macro.nVars)

#     if solution_error > opts.tol.NSSS_acceptance_tol
#         y = (𝓂.constants, SS_and_pars, zeros(S, 0, 0), [state], false)

#         pullback = function (ȳ)
#             Δy = unthunk(ȳ)
#             if Δy isa NoTangent || Δy isa AbstractZero
#                 return NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent()
#             end

#             ΔSS_and_pars = Δy[2]
#             nsss_grads = nsss_pb((ΔSS_and_pars, NoTangent()))
#             ∂parameter_values = nsss_grads[3]

#             return NoTangent(), NoTangent(), ∂parameter_values, NoTangent()
#         end

#         return y, pullback
#     end

#     ∇₁, jac_pb = rrule(calculate_jacobian,
#                         parameter_values,
#                         SS_and_pars,
#                         𝓂.caches,
#                         𝓂.functions.jacobian)

#     first_out, first_pb = rrule(calculate_first_order_solution,
#                                 ∇₁,
#                                 constants_obj,
#                                 𝓂.workspaces,
#                                 𝓂.caches;
#                                 opts = opts,
#                                 initial_guess = 𝓂.caches.qme_solution)

#     𝐒₁ = first_out[1]
#     solved = first_out[3]

#     update_perturbation_counter!(𝓂.counters, solved, estimation = estimation, order = 1)

#     if !solved
#         y = (𝓂.constants, SS_and_pars, zeros(S, 0, 0), [state], false)

#         pullback = function (ȳ)
#             Δy = unthunk(ȳ)
#             if Δy isa NoTangent || Δy isa AbstractZero
#                 return NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent()
#             end

#             ΔSS_and_pars = Δy[2]

#             nsss_grads = nsss_pb((ΔSS_and_pars, NoTangent()))
#             ∂parameter_values = nsss_grads[3]

#             return NoTangent(), NoTangent(), ∂parameter_values, NoTangent()
#         end

#         return y, pullback
#     end

#     y = (𝓂.constants, SS_and_pars, 𝐒₁, [state], true)

#     pullback = function (ȳ)
#         Δy = unthunk(ȳ)
#         if Δy isa NoTangent || Δy isa AbstractZero
#             return NoTangent(), NoTangent(), zeros(S, length(parameter_values)), NoTangent()
#         end

#         ΔSS_and_pars = Δy[2]
#         Δ𝐒₁ = Δy[3]

#         # When the caller passes NoTangent for the solution matrix cotangent
#         # (e.g. filter failure), skip the first-order solution pullback and
#         # only propagate through the steady-state.
#         if Δ𝐒₁ isa Union{NoTangent, AbstractZero}
#             nsss_grads = nsss_pb((ΔSS_and_pars, NoTangent()))
#             return NoTangent(), NoTangent(), nsss_grads[3], NoTangent()
#         end

#         first_grads = first_pb((Δ𝐒₁, NoTangent(), NoTangent()))
#         ∂∇₁ = first_grads[2]

#         jac_grads = jac_pb(∂∇₁)
#         ∂parameter_values = jac_grads[2]
#         ∂SS_and_pars_from_jac = jac_grads[3]

#         nsss_grads = nsss_pb((ΔSS_and_pars + ∂SS_and_pars_from_jac, NoTangent()))
#         ∂parameter_values .+= nsss_grads[3]

#         return NoTangent(), NoTangent(), ∂parameter_values, NoTangent()
#     end

#     return y, pullback
# end

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

    @ignore_derivatives update_perturbation_counter!(𝓂.counters, solved, estimation = estimation, order = 1)

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

    @ignore_derivatives update_perturbation_counter!(𝓂.counters, solved2, estimation = estimation, order = 2)

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
        ∂∇₁_from_so3 = so3_tangents[2]
        ∂∇₂_from_so3 = so3_tangents[3]
        ∂∇₃_from_so3 = so3_tangents[4]
        ∂𝐒₁_raw_from_so3 = so3_tangents[5]
        ∂𝐒₂_from_so3 = so3_tangents[6]

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
        ∂∇₁_from_so3 = so3_tangents[2]
        ∂∇₂_from_so3 = so3_tangents[3]
        ∂∇₃_from_so3 = so3_tangents[4]
        ∂𝐒₁_raw_from_so3 = so3_tangents[5]
        ∂𝐒₂_from_so3 = so3_tangents[6]

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

        # Guard against NoTangent cotangents from filter failure
        Δstate_val = Δstate isa Union{NoTangent, AbstractZero} ? zeros(S, length(state)) : Δstate
        Δ𝐒₁ = Δ𝐒 isa Union{NoTangent, AbstractZero} ? zeros(S, size(𝐒₁)) : Δ𝐒[1]
        Δ𝐒₂ = Δ𝐒 isa Union{NoTangent, AbstractZero} ? zeros(S, size(𝐒₂)) : Δ𝐒[2]
        Δ𝐒₃ = Δ𝐒 isa Union{NoTangent, AbstractZero} ? zeros(S, size(𝐒₃)) : Δ𝐒[3]

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

function rrule(::typeof(calculate_third_order_solution), 
                ∇₁::AbstractMatrix{S}, #first order derivatives
                ∇₂::SparseMatrixCSC{S}, #second order derivatives
                ∇₃::SparseMatrixCSC{S}, #third order derivatives
                𝑺₁::AbstractMatrix{S}, #first order solution
                𝐒₂::SparseMatrixCSC{S}, #second order solution
                constants::constants,
                workspaces::workspaces,
                cache::caches;
                initial_guess::AbstractMatrix{Float64} = zeros(0,0),
                opts::CalculationOptions = merge_calculation_options()) where S <: AbstractFloat 
    if !(eltype(workspaces.third_order.Ŝ) == S)
        workspaces.third_order = Higher_order_workspace(T = S)
    end
    ℂ = workspaces.third_order
    M₂ = constants.second_order
    M₃ = constants.third_order
    T = constants.post_model_macro

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
        return (∇₁₊𝐒₁➕∇₁₀, solved), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
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

    𝐒₃, solved = solve_sylvester_equation(A, B, C, ℂ.sylvester_workspace,
                                            initial_guess = initial_guess,
                                            sylvester_algorithm = opts.sylvester_algorithm³,
                                            tol = opts.tol.sylvester_tol,
                                            acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                            verbose = opts.verbose)
    
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
        return (𝐒₃, solved), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
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
    
    # Ensure pullback workspaces are properly sized (for dense matrices only)
    if size(ℂ.∂∇₁_3rd) != size(∇₁)
        ℂ.∂∇₁_3rd = zeros(S, size(∇₁))
    end
    if size(ℂ.∂𝐒₁_3rd) != size(𝐒₁)
        ℂ.∂𝐒₁_3rd = zeros(S, size(𝐒₁))
    end
    if size(ℂ.∂spinv_3rd) != size(spinv)
        ℂ.∂spinv_3rd = zeros(S, size(spinv))
    end

    function third_order_solution_pullback(∂𝐒₃_solved) 
        # Use workspaces for dense matrices, zero() for sparse
        ∂∇₁ = ℂ.∂∇₁_3rd; fill!(∂∇₁, zero(S))
        ∂∇₂ = zero(∇₂)  # sparse
        # ∂𝐔∇₃ = zero(𝐔∇₃)
        ∂∇₃ = zero(∇₃)  # sparse
        ∂𝐒₁ = ℂ.∂𝐒₁_3rd; fill!(∂𝐒₁, zero(S))
        ∂𝐒₂ = zero(𝐒₂)  # sparse
        ∂spinv = ℂ.∂spinv_3rd; fill!(∂spinv, zero(S))
        ∂𝐒₁₋╱𝟏ₑ = zero(𝐒₁₋╱𝟏ₑ)  # may be sparse
        ∂kron𝐒₁₋╱𝟏ₑ = zero(kron𝐒₁₋╱𝟏ₑ)  # may be sparse
        ∂𝐒₁₊╱𝟎 = zero(𝐒₁₊╱𝟎)  # may be sparse
        ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = zero(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋)  # may be sparse
        ∂tmpkron = zero(tmpkron)  # sparse
        ∂tmpkron22 = zero(ℂ.tmpkron22)  # sparse
        ∂kronaux = zero(kronaux)  # kron product
        ∂aux = zero(aux)
        ∂tmpkron0 = zero(ℂ.tmpkron0)  # sparse
        ∂⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = zero(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)  # may be sparse
        ∂𝐒₂₊╱𝟎 = zero(𝐒₂₊╱𝟎)  # may be sparse
        ∂𝐒₂₊╱𝟎𝛔 = zero(𝐒₂₊╱𝟎𝛔)  # may be sparse
        ∂∇₁₊ = zero(∇₁₊)  # may be sparse
        ∂𝐒₂₋╱𝟎 = zero(𝐒₂₋╱𝟎)  # may be sparse

        # @timeit_debug timer "Third order solution - pullback" begin

        # @timeit_debug timer "Solve sylvester equation" begin

        ∂𝐒₃ = ∂𝐒₃_solved[1]

        # ∂𝐒₃ *= 𝐔₃t
        
        ∂C, solved = solve_sylvester_equation(A', B', ∂𝐒₃, ℂ.sylvester_workspace,
                                                sylvester_algorithm = opts.sylvester_algorithm³,
                                                tol = opts.tol.sylvester_tol,
                                                acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                                verbose = opts.verbose)

        if !solved
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
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

        return NoTangent(), ∂∇₁, ∂∇₂, ∂∇₃, ∂𝑺₁, ∂𝐒₂, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    if solved
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
