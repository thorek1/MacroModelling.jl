function _prepare_stochastic_steady_state_base_terms(parameters::Vector{M},
                                                     𝓂::ℳ;
                                                     opts::CalculationOptions = merge_calculation_options(),
                                                     estimation::Bool = false,
                                                     caching::Bool = true) where M
    constants = initialise_constants!(𝓂)
    T = constants.post_model_macro

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, opts = opts, estimation = estimation, caching = caching)

    if solution_error > opts.tol.nsss.acceptance_tol || isnan(solution_error)
        return (false,
            zeros(T.nVars),
            SS_and_pars,
            solution_error,
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0),
            constants)
    end

    ms = ensure_model_structure_constants!(constants, 𝓂.equations.calibration_parameters)
    all_SS = expand_steady_state(SS_and_pars, ms)

    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces, caching = caching)

    𝐒₁, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                         constants,
                                                         𝓂.workspaces,
                                                         𝓂.caches;
                                                         opts = opts,
                                                         initial_guess = 𝓂.caches.qme_solution,
                                                         parameter_values = parameters,
                                                         caching = caching)

    update_perturbation_counter!(𝓂.counters, solved, estimation = estimation, order = 1)

    if !solved
        if opts.verbose println("1st order solution not found") end
        return (false,
            all_SS,
            SS_and_pars,
            solution_error,
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0),
            constants)
    end

    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian, 𝓂.workspaces, caching = caching)

    𝐒₂_raw, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                                                  initial_guess = 𝓂.caches.second_order_solution,
                                                  opts = opts,
                                                  parameter_values = parameters,
                                                  caching = caching)

    update_perturbation_counter!(𝓂.counters, solved2, estimation = estimation, order = 2)

    if !solved2
        if opts.verbose println("2nd order solution not found") end
        return (false,
            all_SS,
            SS_and_pars,
            solution_error,
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0),
            constants)
    end

    𝐒₂ = sparse(𝐒₂_raw * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{M, Int}

    𝐒₁ = [𝐒₁[:,1:T.nPast_not_future_and_mixed] zeros(T.nVars) 𝐒₁[:,T.nPast_not_future_and_mixed+1:end]]

    aug_state₁ = sparse([zeros(T.nPast_not_future_and_mixed); 1; zeros(T.nExo)])
    tmp = (T.I_nPast - 𝐒₁[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed])
    tmp̄ = ℒ.lu(tmp, check = false)

    if !ℒ.issuccess(tmp̄)
        if opts.verbose println("SSS not found") end
        return (false,
            all_SS,
            SS_and_pars,
            solution_error,
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0),
            constants)
    end

    SSSstates = collect(tmp \ (𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2)[T.past_not_future_and_mixed_idx])

        return (true,
            all_SS,
            SS_and_pars,
            solution_error,
            ∇₁,
            ∇₂,
            𝐒₁,
            𝐒₂_raw,
            SSSstates,
            constants)
end

function calculate_stochastic_steady_state(::Val{:second_order},
                                           parameters::Vector{M},
                                           𝓂::ℳ;
                                           opts::CalculationOptions = merge_calculation_options(),
                                           estimation::Bool = false,
                                           caching::Bool = true) where M
    # Cache hit: return cached SSS if valid for current parameters
    if caching && M === Float64 && !isempty(parameters) &&
       cache_valid_for_parameters(𝓂.caches.valid_for.second_order_stochastic_steady_state, parameters)
        cached_sss = 𝓂.caches.second_order_stochastic_steady_state::Vector{M}
        if !isempty(cached_sss)
            T = 𝓂.constants.post_model_macro
            SS_and_pars = 𝓂.caches.non_stochastic_steady_state::Vector{M}
            ∇₁ = Matrix(𝓂.caches.jacobian)::Matrix{M}
            ∇₂ = sparse(𝓂.caches.hessian)::SparseMatrixCSC{M, Int}
            𝐒₁_raw = Matrix(𝓂.caches.first_order_solution_matrix)::Matrix{M}
            𝐒₁ = [𝐒₁_raw[:,1:T.nPast_not_future_and_mixed] zeros(M, T.nVars) 𝐒₁_raw[:,T.nPast_not_future_and_mixed+1:end]]
            𝐒₂ = sparse(𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{M, Int}
            return cached_sss, true, SS_and_pars, zero(M), ∇₁, ∇₂, 𝐒₁, 𝐒₂
        end
    end

    common = _prepare_stochastic_steady_state_base_terms(parameters, 𝓂, opts = opts, estimation = estimation, caching = caching)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂_raw, SSSstates, _ = common

    if !ok
        if caching && M === Float64 𝓂.caches.second_order_stochastic_steady_state = all_SS end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0)
    end

    # Expand compressed 𝐒₂_raw to full
    𝐒₂ = sparse(𝐒₂_raw * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{M, Int}

    so = 𝓂.constants.second_order
    kron_s⁺_s⁺ = so.kron_s⁺_s⁺
    A = 𝐒₁[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed]
    B̂ = 𝐒₂[:,kron_s⁺_s⁺]

    SSSstates, converged = solve_stochastic_steady_state_newton(Val(:second_order), 𝐒₁, 𝐒₂, collect(SSSstates), 𝓂)

    if !converged
        if opts.verbose println("SSS not found") end
        if caching && M === Float64 𝓂.caches.second_order_stochastic_steady_state = all_SS end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0)
    end

    state = A * SSSstates + B̂ * ℒ.kron(vcat(SSSstates,1), vcat(SSSstates,1)) / 2
    result = all_SS + Vector{M}(state)

    if caching && M === Float64
        𝓂.caches.second_order_stochastic_steady_state = result
        𝓂.caches.valid_for.second_order_stochastic_steady_state = Float64.(parameters)
    end

    return result, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂
end

function calculate_stochastic_steady_state(::Val{:pruned_second_order},
                                           parameters::Vector{M},
                                           𝓂::ℳ;
                                           opts::CalculationOptions = merge_calculation_options(),
                                           estimation::Bool = false,
                                           caching::Bool = true) where M
    # Cache hit: return cached pruned SSS if valid for current parameters
    if caching && M === Float64 && !isempty(parameters) &&
       cache_valid_for_parameters(𝓂.caches.valid_for.pruned_second_order_stochastic_steady_state, parameters)
        cached_sss = 𝓂.caches.pruned_second_order_stochastic_steady_state::Vector{M}
        if !isempty(cached_sss)
            T = 𝓂.constants.post_model_macro
            SS_and_pars = 𝓂.caches.non_stochastic_steady_state::Vector{M}
            ∇₁ = Matrix(𝓂.caches.jacobian)::Matrix{M}
            ∇₂ = sparse(𝓂.caches.hessian)::SparseMatrixCSC{M, Int}
            𝐒₁_raw = Matrix(𝓂.caches.first_order_solution_matrix)::Matrix{M}
            𝐒₁ = [𝐒₁_raw[:,1:T.nPast_not_future_and_mixed] zeros(M, T.nVars) 𝐒₁_raw[:,T.nPast_not_future_and_mixed+1:end]]
            𝐒₂ = sparse(𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{M, Int}
            return cached_sss, true, SS_and_pars, zero(M), ∇₁, ∇₂, 𝐒₁, 𝐒₂
        end
    end

    common = _prepare_stochastic_steady_state_base_terms(parameters, 𝓂, opts = opts, estimation = estimation, caching = caching)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂_raw, SSSstates, _ = common

    if !ok
        if caching && M === Float64 𝓂.caches.pruned_second_order_stochastic_steady_state = all_SS end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0)
    end

    # Expand compressed 𝐒₂_raw to full
    𝐒₂ = sparse(𝐒₂_raw * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{M, Int}

    state = 𝐒₁[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] * SSSstates +
            𝐒₂ * ℒ.kron(sparse([zeros(𝓂.constants.post_model_macro.nPast_not_future_and_mixed); 1; zeros(𝓂.constants.post_model_macro.nExo)]), sparse([zeros(𝓂.constants.post_model_macro.nPast_not_future_and_mixed); 1; zeros(𝓂.constants.post_model_macro.nExo)])) / 2

    result = all_SS + Vector{M}(state)

    if caching && M === Float64
        𝓂.caches.pruned_second_order_stochastic_steady_state = result
        𝓂.caches.valid_for.pruned_second_order_stochastic_steady_state = Float64.(parameters)
    end

    return result, true, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂
end



function solve_stochastic_steady_state_newton(::Val{:second_order}, 
                                              𝐒₁::Matrix{R}, 
                                              𝐒₂::AbstractSparseMatrix{R}, 
                                              x::Vector{R},
                                              𝓂::ℳ;
                                              tol::AbstractFloat = 1e-14) where R <: AbstractFloat
    # @timeit_debug timer "Setup matrices" begin

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
    B = 𝐒₂[T.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂[T.past_not_future_and_mixed_idx,kron_s⁺_s⁺]

    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    
    # Pre-allocate augmented state vector [x; 1]
    x_aug = Vector{R}(undef, length(x) + 1)
    x_aug[end] = one(R)

    # end # timeit_debug
      
    # @timeit_debug timer "Iterations" begin

    for i in 1:max_iters
        copyto!(x_aug, 1, x, 1, length(x))

        ∂x = (A + B * ℒ.kron(x_aug, I_nPast) - I_nPast)

        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            return x, false
        end

        x̂ = A * x + B̂ * ℒ.kron(x_aug, x_aug) / 2

        Δx = ∂x̂ \ (x̂ - x)
        
        if i > 3 && isapprox(x̂, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    # end # timeit_debug

    copyto!(x_aug, 1, x, 1, length(x))
    return x, isapprox(A * x + B̂ * ℒ.kron(x_aug, x_aug) / 2, x, rtol = tol)
end





function calculate_stochastic_steady_state(::Val{:third_order},
                                           parameters::Vector{M},
                                           𝓂::ℳ;
                                           opts::CalculationOptions = merge_calculation_options(),
                                           estimation::Bool = false,
                                           caching::Bool = true) where M <: Real
    # Cache hit: return cached SSS if valid for current parameters
    if caching && M === Float64 && !isempty(parameters) &&
       cache_valid_for_parameters(𝓂.caches.valid_for.third_order_stochastic_steady_state, parameters)
        cached_sss = 𝓂.caches.third_order_stochastic_steady_state::Vector{M}
        if !isempty(cached_sss)
            T = 𝓂.constants.post_model_macro
            SS_and_pars = 𝓂.caches.non_stochastic_steady_state::Vector{M}
            ∇₁ = Matrix(𝓂.caches.jacobian)::Matrix{M}
            ∇₂ = sparse(𝓂.caches.hessian)::SparseMatrixCSC{M, Int}
            ∇₃ = sparse(𝓂.caches.third_order_derivatives)::SparseMatrixCSC{M, Int}
            𝐒₁_raw = Matrix(𝓂.caches.first_order_solution_matrix)::Matrix{M}
            𝐒₁ = [𝐒₁_raw[:,1:T.nPast_not_future_and_mixed] zeros(M, T.nVars) 𝐒₁_raw[:,T.nPast_not_future_and_mixed+1:end]]
            𝐒₂ = sparse(𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{M, Int}
            𝐒̂₃ = sparse(𝓂.caches.third_order_solution * 𝓂.constants.third_order.𝐔₃)::SparseMatrixCSC{M, Int}
            return cached_sss, true, SS_and_pars, zero(M), ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒̂₃
        end
    end

    common = _prepare_stochastic_steady_state_base_terms(parameters, 𝓂, opts = opts, estimation = estimation, caching = caching)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂_raw, SSSstates, _ = common

    if !ok
        if caching && M === Float64 𝓂.caches.third_order_stochastic_steady_state = all_SS end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0)
    end

    # Expand compressed 𝐒₂_raw to full
    𝐒₂ = sparse(𝐒₂_raw * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{M, Int}

    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives, 𝓂.workspaces, caching = caching)
    nPast = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed
    𝐒₁_raw = [𝐒₁[:, 1:nPast] 𝐒₁[:, nPast+2:end]]

    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁_raw, 𝐒₂_raw,
                                                 𝓂.constants,
                                                 𝓂.workspaces,
                                                 𝓂.caches;
                                                 initial_guess = 𝓂.caches.third_order_solution,
                                                 opts = opts,
                                                 parameter_values = parameters,
                                                 caching = caching)

    update_perturbation_counter!(𝓂.counters, solved3, estimation = estimation, order = 3)

    if !solved3
        if opts.verbose println("3rd order solution not found") end
        if caching && M === Float64 𝓂.caches.third_order_stochastic_steady_state = all_SS end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0)
    end

    if length(𝓂.workspaces.third_order.Ŝ) == 0 || !(eltype(𝐒₃) == eltype(𝓂.workspaces.third_order.Ŝ))
        𝓂.workspaces.third_order.Ŝ = 𝐒₃ * 𝓂.constants.third_order.𝐔₃
    else
        ℒ.mul!(𝓂.workspaces.third_order.Ŝ, 𝐒₃, 𝓂.constants.third_order.𝐔₃)
    end

    Ŝ = 𝓂.workspaces.third_order.Ŝ
    𝐒₃̂ = sparse_preallocated!(Ŝ, ℂ = 𝓂.workspaces.third_order)::SparseMatrixCSC{M, Int}

    so = 𝓂.constants.second_order
    kron_s⁺_s⁺ = so.kron_s⁺_s⁺
    kron_s⁺_s⁺_s⁺ = so.kron_s⁺_s⁺_s⁺

    A = 𝐒₁[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed]
    B̂ = 𝐒₂[:,kron_s⁺_s⁺]
    Ĉ = 𝐒₃̂[:,kron_s⁺_s⁺_s⁺]

    SSSstates, converged = solve_stochastic_steady_state_newton(Val(:third_order), 𝐒₁, 𝐒₂, 𝐒₃̂, collect(SSSstates), 𝓂)

    if !converged
        if opts.verbose println("SSS not found") end
        if caching && M === Float64 𝓂.caches.third_order_stochastic_steady_state = all_SS end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0)
    end

    state = A * SSSstates + B̂ * ℒ.kron(vcat(SSSstates,1), vcat(SSSstates,1)) / 2 + Ĉ * ℒ.kron(vcat(SSSstates,1), ℒ.kron(vcat(SSSstates,1), vcat(SSSstates,1))) / 6


    result = all_SS + Vector{M}(state)

    if caching && M === Float64
        𝓂.caches.third_order_stochastic_steady_state = result
        𝓂.caches.valid_for.third_order_stochastic_steady_state = Float64.(parameters)
    end

    return result, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃̂
end

function calculate_stochastic_steady_state(::Val{:pruned_third_order},
                                           parameters::Vector{M},
                                           𝓂::ℳ;
                                           opts::CalculationOptions = merge_calculation_options(),
                                           estimation::Bool = false,
                                           caching::Bool = true) where M <: Real
    # Cache hit: return cached pruned SSS if valid for current parameters
    if caching && M === Float64 && !isempty(parameters) &&
       cache_valid_for_parameters(𝓂.caches.valid_for.pruned_third_order_stochastic_steady_state, parameters)
        cached_sss = 𝓂.caches.pruned_third_order_stochastic_steady_state::Vector{M}
        if !isempty(cached_sss)
            T = 𝓂.constants.post_model_macro
            SS_and_pars = 𝓂.caches.non_stochastic_steady_state::Vector{M}
            ∇₁ = Matrix(𝓂.caches.jacobian)::Matrix{M}
            ∇₂ = sparse(𝓂.caches.hessian)::SparseMatrixCSC{M, Int}
            ∇₃ = sparse(𝓂.caches.third_order_derivatives)::SparseMatrixCSC{M, Int}
            𝐒₁_raw = Matrix(𝓂.caches.first_order_solution_matrix)::Matrix{M}
            𝐒₁ = [𝐒₁_raw[:,1:T.nPast_not_future_and_mixed] zeros(M, T.nVars) 𝐒₁_raw[:,T.nPast_not_future_and_mixed+1:end]]
            𝐒₂ = sparse(𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{M, Int}
            𝐒̂₃ = sparse(𝓂.caches.third_order_solution * 𝓂.constants.third_order.𝐔₃)::SparseMatrixCSC{M, Int}
            return cached_sss, true, SS_and_pars, zero(M), ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒̂₃
        end
    end

    common = _prepare_stochastic_steady_state_base_terms(parameters, 𝓂, opts = opts, estimation = estimation, caching = caching)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂_raw, SSSstates, _ = common

    if !ok
        if caching && M === Float64 𝓂.caches.pruned_third_order_stochastic_steady_state = all_SS end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0)
    end

    # Expand compressed 𝐒₂_raw to full
    𝐒₂ = sparse(𝐒₂_raw * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{M, Int}

    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives, 𝓂.workspaces, caching = caching)
    nPast = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed
    𝐒₁_raw = [𝐒₁[:, 1:nPast] 𝐒₁[:, nPast+2:end]]

    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁_raw, 𝐒₂_raw,
                                                 𝓂.constants,
                                                 𝓂.workspaces,
                                                 𝓂.caches;
                                                 initial_guess = 𝓂.caches.third_order_solution,
                                                 opts = opts, parameter_values = parameters, caching = caching)

    update_perturbation_counter!(𝓂.counters, solved3, estimation = estimation, order = 3)

    if !solved3
        if opts.verbose println("3rd order solution not found") end
        if caching && M === Float64 𝓂.caches.pruned_third_order_stochastic_steady_state = all_SS end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0)
    end

    if length(𝓂.workspaces.third_order.Ŝ) == 0 || !(eltype(𝐒₃) == eltype(𝓂.workspaces.third_order.Ŝ))
        𝓂.workspaces.third_order.Ŝ = 𝐒₃ * 𝓂.constants.third_order.𝐔₃
    else
        ℒ.mul!(𝓂.workspaces.third_order.Ŝ, 𝐒₃, 𝓂.constants.third_order.𝐔₃)
    end

    Ŝ = 𝓂.workspaces.third_order.Ŝ
    𝐒₃̂ = sparse_preallocated!(Ŝ, ℂ = 𝓂.workspaces.third_order)::SparseMatrixCSC{M, Int}

    aug_state₁ = sparse([zeros(𝓂.constants.post_model_macro.nPast_not_future_and_mixed); 1; zeros(𝓂.constants.post_model_macro.nExo)])
    state = 𝐒₁[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] * SSSstates + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2

    result = all_SS + Vector{M}(state)

    if caching && M === Float64
        𝓂.caches.pruned_third_order_stochastic_steady_state = result
        𝓂.caches.valid_for.pruned_third_order_stochastic_steady_state = Float64.(parameters)
    end

    return result, true, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃̂
end


function solve_stochastic_steady_state_newton(::Val{:third_order}, 
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
    Ĉ = 𝐒₃[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺]

    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6

    # Pre-allocate augmented state vector [x; 1]
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
        
        Δx = ∂x̂ \ (A * x + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6 - x)

        if i > 5 && isapprox(A * x + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    copyto!(x_aug, 1, x, 1, length(x))
    kron_x_aug = ℒ.kron(x_aug, x_aug)
    kron_x_kron = ℒ.kron(x_aug, kron_x_aug)
    return x, isapprox(A * x + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6, x, rtol = tol)
end
