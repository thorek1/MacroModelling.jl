# ForwardDiff Dual number specializations for forward-mode automatic differentiation
#
# This file centralizes method specializations for ForwardDiff.Dual types, enabling
# forward-mode AD through the model solution pipeline.
#
# Strategy for each function:
#   1. Extract Float64 values from Dual numbers using ℱ.value.(...)
#   2. Compute the function result on Float64 values
#   3. Compute partials using implicit differentiation or chain rule
#   4. Reconstruct Dual numbers by combining values and partials
#
# Functions covered:
#   - sparse_preallocated!
#   - calculate_second/third_order_stochastic_steady_state  
#   - separate_values_and_partials_from_sparsevec_dual
#   - get_NSSS_and_parameters
#   - calculate_first_order_solution
#   - solve_quadratic_matrix_equation
#   - solve_sylvester_equation
#   - solve_lyapunov_equation


function sparse_preallocated!(Ŝ::Matrix{ℱ.Dual{Z,S,N}}; ℂ::higher_order_workspace = Higher_order_workspace()) where {Z,S,N}
    sparse(Ŝ)
end

function solve_stochastic_steady_state_newton(::Val{:second_order}, 
                                              𝐒₁::Matrix{ℱ.Dual{Z,S,N}}, 
                                              𝐒₂::AbstractSparseMatrix{ℱ.Dual{Z,S,N}}, 
                                              x::Vector{ℱ.Dual{Z,S,N}},
                                              𝓂::ℳ;
                                              tol::AbstractFloat = 1e-14)::Tuple{Vector{ℱ.Dual{Z,S,N}}, Bool} where {Z,S,N}

    𝐒₁̂ = ℱ.value.(𝐒₁)
    𝐒₂̂ = ℱ.value.(𝐒₂)
    x̂ = ℱ.value.(x)
    
    # Get cached computational constants
    constants = initialise_constants!(𝓂)
    so = constants.second_order
    ℂ = 𝓂.workspaces.second_order
    T = constants.post_model_macro
    s_in_s⁺ = so.s_in_s⁺
    s_in_s = so.s_in_s
    I_nPast = T.I_nPast
    
    kron_s⁺_s⁺ = so.kron_s⁺_s⁺
    
    kron_s⁺_s = so.kron_s⁺_s
    
    A = 𝐒₁̂[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
    B = 𝐒₂̂[T.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂̂[T.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
 
    # Allocate or reuse workspace for partials
    if size(ℂ.∂x_second_order) != (length(x̂), N)
        ℂ.∂x_second_order = zeros(S, length(x̂), N)
    else
        fill!(ℂ.∂x_second_order, zero(S))
    end
    ∂x̄ = ℂ.∂x_second_order
    
    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    for i in 1:max_iters
        ∂x = (A + B * ℒ.kron(vcat(x̂,1), I_nPast) - I_nPast)

        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            break
        end
        
        Δx = ∂x̂ \ (A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 - x̂)

        if i > 5 && isapprox(A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2, x̂, rtol = tol)
            break
        end
        
        # x̂ += Δx
        ℒ.axpy!(-1, Δx, x̂)
    end

    solved = isapprox(A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2, x̂, rtol = tol)

    if solved
        for i in 1:N
            ∂𝐒₁ = ℱ.partials.(𝐒₁, i)
            ∂𝐒₂ = ℱ.partials.(𝐒₂, i)

            ∂A = ∂𝐒₁[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed]
            ∂B̂ = ∂𝐒₂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s⁺]

            tmp = ∂A * x̂ + ∂B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2

            TMP = A + B * ℒ.kron(vcat(x̂,1), I_nPast) - I_nPast

            ∂x̄[:,i] = -TMP \ tmp
        end
    end
    
    return reshape(map(x̂, eachrow(∂x̄)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(x̂)), solved
end

function solve_stochastic_steady_state_newton(::Val{:third_order}, 
                                              𝐒₁::Matrix{ℱ.Dual{Z,S,N}}, 
                                              𝐒₂::AbstractSparseMatrix{ℱ.Dual{Z,S,N}}, 
                                              𝐒₃::AbstractSparseMatrix{ℱ.Dual{Z,S,N}},
                                              x::Vector{ℱ.Dual{Z,S,N}},
                                              𝓂::ℳ;
                                              tol::AbstractFloat = 1e-14)::Tuple{Vector{ℱ.Dual{Z,S,N}}, Bool} where {Z,S,N}
    𝐒₁̂ = ℱ.value.(𝐒₁)
    𝐒₂̂ = ℱ.value.(𝐒₂)
    𝐒₃̂ = ℱ.value.(𝐒₃)
    x̂ = ℱ.value.(x)
    
    # Get cached computational constants
    so = ensure_computational_constants!(𝓂.constants)
    T = 𝓂.constants.post_model_macro
    ℂ = 𝓂.workspaces.third_order
    s_in_s⁺ = so.s_in_s⁺
    s_in_s = so.s_in_s
    I_nPast = T.I_nPast
    
    kron_s⁺_s⁺ = so.kron_s⁺_s⁺
    
    kron_s⁺_s = so.kron_s⁺_s
    
    kron_s⁺_s⁺_s⁺ = so.kron_s⁺_s⁺_s⁺
    
    kron_s_s⁺_s⁺ = so.kron_s_s⁺_s⁺
    
    A = 𝐒₁̂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed]
    B = 𝐒₂̂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂̂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
    C = 𝐒₃̂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s_s⁺_s⁺]
    Ĉ = 𝐒₃̂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺]

    # Allocate or reuse workspace for partials
    if size(ℂ.∂x_third_order) != (length(x̂), N)
        ℂ.∂x_third_order = zeros(S, length(x̂), N)
    else
        fill!(ℂ.∂x_third_order, zero(S))
    end
    ∂x̄ = ℂ.∂x_third_order
    
    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    for i in 1:max_iters
        ∂x = (A + B * ℒ.kron(vcat(x̂,1), I_nPast) + C * ℒ.kron(ℒ.kron(vcat(x̂,1), vcat(x̂,1)), I_nPast) / 2 - I_nPast)

        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            break
        end
        
        Δx = ∂x̂ \ (A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 + Ĉ * ℒ.kron(vcat(x̂,1), ℒ.kron(vcat(x̂,1), vcat(x̂,1))) / 6 - x̂)

        if i > 5 && isapprox(A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 + Ĉ * ℒ.kron(vcat(x̂,1), ℒ.kron(vcat(x̂,1), vcat(x̂,1))) / 6, x̂, rtol = tol)
            break
        end
        
        # x̂ += Δx
        ℒ.axpy!(-1, Δx, x̂)
    end

    solved = isapprox(A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 + Ĉ * ℒ.kron(vcat(x̂,1), ℒ.kron(vcat(x̂,1), vcat(x̂,1))) / 6, x̂, rtol = tol)
    
    if solved
        for i in 1:N
            ∂𝐒₁ = ℱ.partials.(𝐒₁, i)
            ∂𝐒₂ = ℱ.partials.(𝐒₂, i)
            ∂𝐒₃ = ℱ.partials.(𝐒₃, i)

            ∂A = ∂𝐒₁[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed]
            ∂B̂ = ∂𝐒₂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
            ∂Ĉ = ∂𝐒₃[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺]

            tmp = ∂A * x̂ + ∂B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 + ∂Ĉ * ℒ.kron(vcat(x̂,1), ℒ.kron(vcat(x̂,1), vcat(x̂,1))) / 6

            TMP = A + B * ℒ.kron(vcat(x̂,1), I_nPast) + C * ℒ.kron(ℒ.kron(vcat(x̂,1), vcat(x̂,1)), I_nPast) / 2 - I_nPast

            ∂x̄[:,i] = -TMP \ tmp
        end
    end
    
    return reshape(map(x̂, eachrow(∂x̄)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(x̂)), solved
end

function separate_values_and_partials_from_sparsevec_dual(V::SparseVector{ℱ.Dual{Z,S,N}}; tol::AbstractFloat = eps()) where {Z,S,N}
    nrows = length(V)
    ncols = length(V.nzval[1].partials)

    rows = Int[]
    cols = Int[]

    prtls = Float64[]

    for (i,v) in enumerate(V.nzind)
        for (k,w) in enumerate(V.nzval[i].partials)
            if abs(w) > tol
                push!(rows,v)
                push!(cols,k)
                push!(prtls,w)
            end
        end
    end

    vvals = sparsevec(V.nzind,[i.value for i in V.nzval],nrows)
    ps = sparse(rows,cols,prtls,nrows,ncols)

    return vvals, ps
end

function get_NSSS_and_parameters(𝓂::ℳ, 
                                parameter_values_dual::Vector{ℱ.Dual{Z,S,N}}; 
                                opts::CalculationOptions = merge_calculation_options(),
                                cold_start::Bool = false,
                                estimation::Bool = false)::Tuple{Vector{ℱ.Dual{Z,S,N}}, Tuple{S, Int}} where {Z, S <: AbstractFloat, N}
                                # timer::TimerOutput = TimerOutput(),
    parameter_values = ℱ.value.(parameter_values_dual)
    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    T = 𝓂.constants.post_model_macro
    qme_ws = 𝓂.workspaces.first_order

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
    
    # Allocate or reuse workspace for partials
    if size(qme_ws.∂SS_and_pars) != (length(SS_and_pars), N)
        qme_ws.∂SS_and_pars = zeros(S, length(SS_and_pars), N)
    else
        fill!(qme_ws.∂SS_and_pars, zero(S))
    end
    ∂SS_and_pars = qme_ws.∂SS_and_pars

    if solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error)
        if opts.verbose println("Failed to find NSSS") end

        # Update failed counter
        update_ss_counter!(𝓂.counters, false, estimation = estimation)

        solution_error = S(10.0)
    else
        # Update success counter
        update_ss_counter!(𝓂.counters, true, estimation = estimation)

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
            if jac_cache isa SparseMatrixCSC
                jac_cache.nzval .= 0
            else
                fill!(jac_cache, zero(eltype(jac_cache)))
            end
        end

        𝓂.functions.NSSS_∂equations_∂parameters(jac_cache, ∂, C)

        ∂SS_equations_∂parameters = jac_cache

        
        if eltype(𝓂.caches.NSSS_∂equations_∂SS_and_pars) != eltype(parameter_values)
            if 𝓂.caches.NSSS_∂equations_∂SS_and_pars isa SparseMatrixCSC
                jac_cache = similar(𝓂.caches.NSSS_∂equations_∂SS_and_pars, eltype(SS_and_pars))
                jac_cache.nzval .= 0
            else
                jac_cache = zeros(eltype(SS_and_pars), size(𝓂.caches.NSSS_∂equations_∂SS_and_pars))
            end
        else
            jac_cache = 𝓂.caches.NSSS_∂equations_∂SS_and_pars
            if jac_cache isa SparseMatrixCSC
                jac_cache.nzval .= 0
            else
                fill!(jac_cache, zero(eltype(jac_cache)))
            end
        end

        𝓂.functions.NSSS_∂equations_∂SS_and_pars(jac_cache, ∂, C)

        ∂SS_equations_∂SS_and_pars = jac_cache

        if ∂SS_equations_∂SS_and_pars isa SparseMatrixCSC
            rhs_n_rows = size(∂SS_equations_∂SS_and_pars, 1)
            rhs_n_cols = size(∂SS_equations_∂parameters, 2)

            if length(qme_ws.nsss_sparse_rhs) != rhs_n_rows
                qme_ws.nsss_sparse_rhs = zeros(eltype(SS_and_pars), rhs_n_rows)
            end

            if size(qme_ws.nsss_jvp_rhs) != (rhs_n_rows, rhs_n_cols)
                qme_ws.nsss_jvp_rhs = zeros(eltype(SS_and_pars), rhs_n_rows, rhs_n_cols)
            end

            if size(qme_ws.nsss_sparse_lu_buffer.A) != (rhs_n_rows, rhs_n_rows)
                sparse_prob = 𝒮.LinearProblem(∂SS_equations_∂SS_and_pars, qme_ws.nsss_sparse_rhs)
                qme_ws.nsss_sparse_lu_buffer = 𝒮.init(sparse_prob,
                                                      𝒮.LUFactorization(),
                                                      verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)
            else
                qme_ws.nsss_sparse_lu_buffer.A = ∂SS_equations_∂SS_and_pars
            end

            sparse_solved = true
            for j in 1:rhs_n_cols
                @views copyto!(qme_ws.nsss_sparse_rhs, ∂SS_equations_∂parameters[:, j])
                qme_ws.nsss_sparse_lu_buffer.b = qme_ws.nsss_sparse_rhs
                sparse_sol = 𝒮.solve!(qme_ws.nsss_sparse_lu_buffer)

                if sparse_sol.retcode != 𝒮.SciMLBase.ReturnCode.Default && !𝒮.SciMLBase.successful_retcode(sparse_sol.retcode)
                    sparse_solved = false
                    break
                end

                @views copyto!(qme_ws.nsss_jvp_rhs[:, j], qme_ws.nsss_sparse_lu_buffer.u)
            end

            if !sparse_solved
                if opts.verbose println("Failed to calculate implicit derivative of NSSS") end
                solution_error = S(10.0)
            else
                ℒ.rmul!(qme_ws.nsss_jvp_rhs, -1)
                jvp_no_exo = custom_ss_expand_matrix * qme_ws.nsss_jvp_rhs
                for i in 1:N
                    parameter_values_partials = ℱ.partials.(parameter_values_dual, i)
                    @view(∂SS_and_pars[:,i]) .= jvp_no_exo * parameter_values_partials
                end
            end
        else
            qme_ws.fast_lu_ws_nsss, qme_ws.fast_lu_dims_nsss, solved_nsss, nsss_lu = factorize_lu!(∂SS_equations_∂SS_and_pars,
                                                                                                     qme_ws.fast_lu_ws_nsss,
                                                                                                     qme_ws.fast_lu_dims_nsss)

            if !solved_nsss
                if opts.verbose println("Failed to calculate implicit derivative of NSSS") end
                solution_error = S(10.0)
            else
                rhs_dense = ∂SS_equations_∂parameters isa Matrix ? ∂SS_equations_∂parameters : Matrix(∂SS_equations_∂parameters)

                if size(qme_ws.nsss_jvp_rhs) != size(rhs_dense)
                    qme_ws.nsss_jvp_rhs = zeros(eltype(rhs_dense), size(rhs_dense))
                end
                copyto!(qme_ws.nsss_jvp_rhs, rhs_dense)

                solve_lu_left!(∂SS_equations_∂SS_and_pars,
                               qme_ws.nsss_jvp_rhs,
                               qme_ws.fast_lu_ws_nsss,
                               nsss_lu)

                ℒ.rmul!(qme_ws.nsss_jvp_rhs, -1)
                jvp_no_exo = custom_ss_expand_matrix * qme_ws.nsss_jvp_rhs
                for i in 1:N
                    parameter_values_partials = ℱ.partials.(parameter_values_dual, i)
                    @view(∂SS_and_pars[:,i]) .= jvp_no_exo * parameter_values_partials
                end
            end
        end
    end
    
    return reshape(map(SS_and_pars, eachrow(∂SS_and_pars)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(SS_and_pars)), (solution_error, iters)
end

function calculate_first_order_solution(∇₁::Matrix{ℱ.Dual{Z,S,N}},
                                        constants::constants,
                                        workspaces::workspaces,
                                        cache::caches;
                                        opts::CalculationOptions = merge_calculation_options(),
                                        use_fastlapack_lu::Bool = true,
                                        initial_guess::AbstractMatrix{<:Real} = zeros(0,0))::Tuple{Matrix{ℱ.Dual{Z,S,N}}, Matrix{Float64}, Bool} where {Z,S,N}
    T = constants.post_model_macro
    idx_constants = ensure_first_order_constants!(constants)
    qme_ws = workspaces.first_order
    sylv_ws = workspaces.sylvester_1st_order
    ensure_first_order_workspace_buffers!(qme_ws, T, length(idx_constants.dyn_index), length(idx_constants.comb))
    ensure_sylvester_krylov_buffers!(qme_ws.sylvester, T.nVars, T.nVars)
    ensure_sylvester_doubling_buffers!(qme_ws.sylvester, T.nVars, T.nVars)

    if size(qme_ws.p_tmp) != size(∇₁)
        qme_ws.p_tmp = zeros(S, size(∇₁, 1), size(∇₁, 2))
    end
    ∇̂₁ = qme_ws.p_tmp
    @inbounds for j in axes(∇₁, 2), i in axes(∇₁, 1)
        ∇̂₁[i, j] = ℱ.value(∇₁[i, j])
    end

    expand_future = idx_constants.expand_future
    expand_past = idx_constants.expand_past

    A = qme_ws.𝐀₀
    B = qme_ws.∇₀
    X = qme_ws.sylvester.tmp
    AXB = qme_ws.sylvester.𝐗
    AA = qme_ws.sylvester.𝐂
    X² = qme_ws.sylvester.𝐀
    dA = qme_ws.sylvester.𝐀¹
    dB = qme_ws.sylvester.𝐁
    dC = qme_ws.sylvester.𝐁¹
    CC = qme_ws.sylvester.𝐂_dbl
    tmp = qme_ws.sylvester.𝐂¹
    B_sylv = qme_ws.sylvester.𝐂B

    # Legacy readable path (before workspace reuse):
    #   ∇̂₁ = value.(∇₁)
    #   A = ∇̂₁[:, 1:T.nFuture_not_past_and_mixed] * expand_future
    #   B = ∇̂₁[:, idx_constants.nabla_zero_cols]
    #   X = 𝐒₁[:, 1:end-T.nExo] * expand_past
    #   AXB = A * X + B
    #   AA = inv(AXB) * A
    # Current code computes the same objects via `mul!`/`copyto!`/LU solves in reusable buffers.

    initial_guess_value = if length(initial_guess) == 0
        zeros(eltype(∇̂₁), 0, 0)
    elseif eltype(initial_guess) <: AbstractFloat
        initial_guess isa Matrix{eltype(∇̂₁)} ? initial_guess : Matrix{eltype(∇̂₁)}(initial_guess)
    else
        ℱ.value.(initial_guess)
    end

    𝐒₁, qme_sol, solved = calculate_first_order_solution(∇̂₁, constants, workspaces, cache; opts = opts, initial_guess = initial_guess_value)

    if !solved 
        return ∇₁, qme_sol, false
    end

    ℒ.mul!(A, @view(∇̂₁[:,1:T.nFuture_not_past_and_mixed]), expand_future)
    copyto!(B, @view(∇̂₁[:,idx_constants.nabla_zero_cols]))

    ℒ.mul!(X, @view(𝐒₁[:,1:end-T.nExo]), expand_past)

    copyto!(AXB, B)
    ℒ.mul!(AXB, A, X, 1, 1)

    qme_ws.fast_lu_ws_nabla0, qme_ws.fast_lu_dims_nabla0, solved_AXB, AXBfact = factorize_lu!(AXB,
                                                                                                 qme_ws.fast_lu_ws_nabla0,
                                                                                                 qme_ws.fast_lu_dims_nabla0;
                                                                                                 use_fastlapack_lu = use_fastlapack_lu)

    if !solved_AXB
        return ∇₁, qme_sol, false
    end

    copyto!(AA, A)
    solve_lu_left!(AXB, AA, qme_ws.fast_lu_ws_nabla0, AXBfact;
                   use_fastlapack_lu = use_fastlapack_lu)

    ℒ.mul!(X², X, X)

    # Allocate or reuse workspace for partials (from first_order_workspace)
    if size(qme_ws.X̃_first_order) != (length(𝐒₁[:,1:end-T.nExo]), N)
        qme_ws.X̃_first_order = zeros(length(𝐒₁[:,1:end-T.nExo]), N)
    else
        fill!(qme_ws.X̃_first_order, zero(eltype(qme_ws.X̃_first_order)))
    end
    X̃ = qme_ws.X̃_first_order

    p = ∇̂₁

    copyto!(B_sylv, X)
    ℒ.rmul!(B_sylv, -1)

    initial_guess = zeros(eltype(X), size(X, 1), size(X, 2))

    # https://arxiv.org/abs/2011.11430  
    for i in 1:N
        p .= ℱ.partials.(∇₁, i)

        ℒ.mul!(dA, @view(p[:,1:T.nFuture_not_past_and_mixed]), expand_future)
        copyto!(dB, @view(p[:,idx_constants.nabla_zero_cols]))
        ℒ.mul!(dC, @view(p[:,idx_constants.nabla_minus_cols]), expand_past)

        copyto!(CC, dC)
        ℒ.mul!(tmp, dA, X²)
        CC .+= tmp
        ℒ.mul!(tmp, dB, X)
        CC .+= tmp

        # Legacy readable equivalent:
        #   CC = inv(AXB) * (dA * X² + dC + dB * X)
        # followed by Sylvester solve with (-X, -CC).
        # Here, `solve_lu_left!` replaces explicit inverse multiplication,
        # and `B_sylv`/sign flip encode the same Sylvester system.

        solve_lu_left!(AXB, CC, qme_ws.fast_lu_ws_nabla0, AXBfact;
                       use_fastlapack_lu = use_fastlapack_lu)

        if ℒ.norm(CC) < eps() continue end

        ℒ.rmul!(CC, -1)

        dX, solved = solve_sylvester_equation(AA, B_sylv, CC, sylv_ws,
                                                initial_guess = initial_guess,
                                                sylvester_algorithm = opts.sylvester_algorithm²,
                                                tol = opts.tol.sylvester_tol,
                                                acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                                verbose = opts.verbose)

        # if !solved
        #     dX, solved = solve_sylvester_equation(AA, -X, -CC, 
        #                                             sylvester_algorithm = :bicgstab, # more robust than sylvester
        #                                             initial_guess = initial_guess, 
        #                                             verbose = verbose)

        #     if !solved
        #         return ∇₁, qme_sol, false
        #     end
        # end
    
        initial_guess = dX

        X̃[:,i] = vec(dX[:,T.past_not_future_and_mixed_idx])
    end

    x = reshape(map(𝐒₁[:,1:end-T.nExo], eachrow(X̃)) do v, p
            ℱ.Dual{Z}(v, p...) # Z is the tag
        end, size(𝐒₁[:,1:end-T.nExo]))

    Jm = expand_past
    
    ∇₊ = ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand_future
    ∇₀ = ∇₁[:,idx_constants.nabla_zero_cols]
    ∇ₑ = ∇₁[:,idx_constants.nabla_e_start:end]

    B = -((∇₊ * x * Jm + ∇₀) \ ∇ₑ)

    S₁ = hcat(x, B)

    S₁_value = ℱ.value.(S₁)
    S₁_existing = cache.first_order_solution_matrix
    if S₁_existing isa Matrix{S} && size(S₁_existing) == size(S₁_value)
        copyto!(S₁_existing, S₁_value)
    else
        cache.first_order_solution_matrix = S₁_value
    end

    return S₁, qme_sol, solved
end

function solve_quadratic_matrix_equation(A::AbstractMatrix{ℱ.Dual{Z,S,N}}, 
                                        B::AbstractMatrix{ℱ.Dual{Z,S,N}}, 
                                        C::AbstractMatrix{ℱ.Dual{Z,S,N}}, 
                                        constants::constants,
                                        workspaces::workspaces,
                                        cache::caches;
                                        initial_guess::AbstractMatrix{<:Real} = zeros(0,0),
                                        tol::AbstractFloat = 1e-8, 
                                        quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                                        verbose::Bool = false) where {Z,S,N}
    T = constants.post_model_macro
    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    B̂ = ℱ.value.(B)
    Ĉ = ℱ.value.(C)

    initial_guess_value = if length(initial_guess) == 0
        zeros(eltype(Â), 0, 0)
    elseif eltype(initial_guess) <: AbstractFloat
        initial_guess isa Matrix{eltype(Â)} ? initial_guess : Matrix{eltype(Â)}(initial_guess)
    else
        ℱ.value.(initial_guess)
    end

    qme_ws = ensure_qme_doubling_workspace!(workspaces,
                                            T.nVars - T.nPresent_only)

    X, solved = solve_quadratic_matrix_equation(Â, B̂, Ĉ,
                                                constants,
                                                workspaces,
                                                cache;
                                                tol = tol,
                                                initial_guess = initial_guess_value,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                verbose = verbose)

    AXB = Â * X + B̂
    
    AXBfact = ℒ.lu(AXB, check = false)

    if !ℒ.issuccess(AXBfact)
        AXBfact = ℒ.svd(AXB)
    end

    invAXB = inv(AXBfact)

    AA = invAXB * Â

    X² = X * X

    # Allocate or reuse workspace for partials (from qme_doubling_workspace)
    if size(qme_ws.X̃) != (length(X), N)
        qme_ws.X̃ = zeros(length(X), N)
    else
        fill!(qme_ws.X̃, zero(eltype(qme_ws.X̃)))
    end
    X̃ = qme_ws.X̃

    # https://arxiv.org/abs/2011.11430  
    for i in 1:N
        dA = ℱ.partials.(A, i)
        dB = ℱ.partials.(B, i)
        dC = ℱ.partials.(C, i)
    
        CC = invAXB * (dA * X² + dB * X + dC)

        if ℒ.norm(CC) < eps() continue end
    
        dX, slvd = solve_sylvester_equation(AA, -X, -CC, qme_ws.sylvester, sylvester_algorithm = :doubling)

        solved = Bool(solved) && Bool(slvd)

        X̃[:,i] = vec(dX)
    end
    
    return reshape(map(X, eachrow(X̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(X)), solved
end

function solve_sylvester_equation(  A::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    B::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    C::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    𝕊ℂ::sylvester_workspace;
                                    initial_guess::AbstractMatrix{<:Real} = zeros(0,0),
                                    sylvester_algorithm::Symbol = :doubling,
                                    acceptance_tol::AbstractFloat = 1e-10,
                                    tol::AbstractFloat = 1e-14,
                                    verbose::Bool = false)::Tuple{Matrix{ℱ.Dual{Z,S,N}}, Bool} where {Z,S,N}
    # Extract Float64 values from Dual numbers
    Â = ℱ.value.(A)
    B̂ = ℱ.value.(B)
    Ĉ = ℱ.value.(C)

    initial_guess_value = if length(initial_guess) == 0
        zeros(eltype(Â), 0, 0)
    elseif eltype(initial_guess) <: AbstractFloat
        initial_guess isa Matrix{eltype(Â)} ? initial_guess : Matrix{eltype(Â)}(initial_guess)
    else
        ℱ.value.(initial_guess)
    end

    P̂, solved = solve_sylvester_equation(Â, B̂, Ĉ, 𝕊ℂ,
                                        sylvester_algorithm = sylvester_algorithm, 
                                        tol = tol, 
                                        verbose = verbose, 
                                        initial_guess = initial_guess_value)

    if size(𝕊ℂ.P) != size(P̂)
        𝕊ℂ.P = zeros(eltype(P̂), size(P̂)...)
    end
    copyto!(𝕊ℂ.P, P̂)
    P̂_stable = 𝕊ℂ.P

    # Allocate or reuse workspaces for temporary copies
    if size(𝕊ℂ.Ã_fd) != size(Â)
        𝕊ℂ.Ã_fd = copy(Â)
    else
        copyto!(𝕊ℂ.Ã_fd, Â)
    end
    Ã = 𝕊ℂ.Ã_fd
    
    if size(𝕊ℂ.B̃_fd) != size(B̂)
        𝕊ℂ.B̃_fd = copy(B̂)
    else
        copyto!(𝕊ℂ.B̃_fd, B̂)
    end
    B̃ = 𝕊ℂ.B̃_fd
    
    if size(𝕊ℂ.C̃_fd) != size(Ĉ)
        𝕊ℂ.C̃_fd = copy(Ĉ)
    else
        copyto!(𝕊ℂ.C̃_fd, Ĉ)
    end
    C̃ = 𝕊ℂ.C̃_fd
    
    # Allocate or reuse workspace for partials
    if size(𝕊ℂ.P̃) != (length(P̂), N)
        𝕊ℂ.P̃ = zeros(S, length(P̂), N)
    else
        fill!(𝕊ℂ.P̃, zero(S))
    end
    P̃ = 𝕊ℂ.P̃
    
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        B̃ .= ℱ.partials.(B, i)
        C̃ .= ℱ.partials.(C, i)

        X = Ã * P̂_stable * B̂ + Â * P̂_stable * B̃ + C̃
        
        if ℒ.norm(X) < eps() continue end

        P, slvd = solve_sylvester_equation(Â, B̂, X, 𝕊ℂ,
                                            sylvester_algorithm = sylvester_algorithm, 
                                            tol = tol, 
                                            verbose = verbose)

        solved = solved && slvd

        P̃[:,i] = vec(P)
    end
    
    return reshape(map(P̂_stable, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂_stable)), solved
end

function solve_lyapunov_equation(  A::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    C::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    workspace::lyapunov_workspace;
                                    lyapunov_algorithm::Symbol = :doubling,
                                    tol::AbstractFloat = 1e-14,
                                    acceptance_tol::AbstractFloat = 1e-12,
                                    verbose::Bool = false)::Tuple{Matrix{ℱ.Dual{Z,S,N}}, Bool} where {Z,S,N}
    # Extract Float64 values from Dual numbers
    Â = ℱ.value.(A)
    Ĉ = ℱ.value.(C)

    P̂, solved = solve_lyapunov_equation(Â, Ĉ, workspace, lyapunov_algorithm = lyapunov_algorithm, tol = tol, verbose = verbose)

    if size(workspace.P) != size(P̂)
        workspace.P = zeros(eltype(P̂), size(P̂)...)
    end
    copyto!(workspace.P, P̂)
    P̂_stable = workspace.P

    # Allocate or reuse workspaces for temporary copies (from lyapunov_workspace)
    if size(workspace.Ã_fd) != size(Â)
        workspace.Ã_fd = copy(Â)
    else
        copyto!(workspace.Ã_fd, Â)
    end
    Ã = workspace.Ã_fd
    
    if size(workspace.C̃_fd) != size(Ĉ)
        workspace.C̃_fd = copy(Ĉ)
    else
        copyto!(workspace.C̃_fd, Ĉ)
    end
    C̃ = workspace.C̃_fd
    
    # Allocate or reuse workspace for partials (from lyapunov_workspace)
    if size(workspace.P̃) != (length(P̂), N)
        workspace.P̃ = zeros(length(P̂), N)
    else
        fill!(workspace.P̃, zero(eltype(workspace.P̃)))
    end
    P̃ = workspace.P̃
    
    # https://arxiv.org/abs/2011.11430  
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        C̃ .= ℱ.partials.(C, i)

        X = Ã * P̂_stable * Â' + Â * P̂_stable * Ã' + C̃

        if ℒ.norm(X) < eps() continue end

        P, slvd = solve_lyapunov_equation(Â, X, workspace, lyapunov_algorithm = lyapunov_algorithm, tol = tol, verbose = verbose)
        
        solved = solved && slvd

        P̃[:,i] = vec(P)
    end
    
    return reshape(map(P̂_stable, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂_stable)), solved
end


function calculate_loglikelihood(::Val{:kalman},
                                ::Val,
                                observables_index::Vector{Int},
                                𝐒::Union{Matrix{ℱ.Dual{Z,S,N}},Vector{AbstractMatrix{ℱ.Dual{Z,S,N}}}},
                                data_in_deviations::Matrix{R},
                                constants::constants,
                                state,
                                workspaces::workspaces;
                                warmup_iterations::Int = 0,
                                presample_periods::Int = 0,
                                initial_covariance::Symbol = :theoretical,
                                filter_algorithm::Symbol = :LagrangeNewton,
                                lyapunov_algorithm::Symbol = :doubling,
                                on_failure_loglikelihood::U = -Inf,
                                opts::CalculationOptions = merge_calculation_options())::ℱ.Dual{Z,S,N} where {Z,S,N,R <: Real, U <: AbstractFloat}
                                                
    T = constants.post_model_macro
    idx_constants = constants.post_complete_parameters
    lyap_ws = ensure_lyapunov_workspace!(workspaces, T.nVars, :first_order)
    kalman_ws = workspaces.kalman

    observables_and_states = sort(union(T.past_not_future_and_mixed_idx, observables_index))
    observables_sorted = sort(observables_index)
    I_nVars = idx_constants.diag_nVars

    A = @views 𝐒[observables_and_states,1:T.nPast_not_future_and_mixed] * I_nVars[T.past_not_future_and_mixed_idx, observables_and_states]
    B = @views 𝐒[observables_and_states,T.nPast_not_future_and_mixed+1:end]

    C = @views I_nVars[observables_sorted, observables_and_states]
    𝐁 = B * B'

    P = get_initial_covariance(Val(initial_covariance), A, 𝐁, lyap_ws, opts = opts)

    if !(eltype(P) <: ℱ.Dual)
        dual_zero = zero(A[1])
        P_float = P
        P = similar(A, size(P_float, 1), size(P_float, 2))
        @inbounds for i in eachindex(P)
            P[i] = dual_zero + S(P_float[i])
        end
    end

    u = zeros(eltype(A), size(C, 2))
    z = C * u
    loglik = zero(eltype(A))

    for t in 1:size(data_in_deviations, 2)
        if !all(isfinite.(z))
            if opts.verbose println("KF not finite at step $t") end
            return on_failure_loglikelihood
        end

        v = data_in_deviations[:, t] - z
        F = C * P * C'

        luF = ℒ.lu(F, check = false)
        if !ℒ.issuccess(luF)
            if opts.verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood
        end

        Fdet = ℒ.det(luF)
        if Fdet < eps(Float64)
            if opts.verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood
        end

        invF = inv(luF)

        if t > presample_periods
            loglik += log(Fdet) + ℒ.dot(v, invF, v)
        end

        K = P * C' * invF
        P = A * (P - K * C * P) * A' + 𝐁
        u = A * (u + K * v)
        z = C * u
    end

    return -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2
end
