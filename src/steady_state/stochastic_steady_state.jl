@stable default_mode = "disable" begin


function prepare_stochastic_steady_state_base_terms(parameters::Vector{M},
                                                     𝓂::ℳ;
                                                     opts::CalculationOptions = merge_calculation_options(),
                                                     estimation::Bool = false,
                                                     caching::Bool = true)::Tuple{Bool, Vector{M}, Vector{M}, M, Matrix{M}, SparseMatrixCSC{M, Int}, Matrix{M}, SparseMatrixCSC{M, Int}, Vector{M}, constants} where M
    C = initialise_constants!(𝓂)
    T = C.post_model_macro

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, opts = opts, estimation = estimation, caching = caching)

    if solution_error > opts.tol.nsss.acceptance_tol || isnan(solution_error)
        return (false,
            zeros(M, T.nVars),
            SS_and_pars,
            solution_error,
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0),
            C)
    end

    ensure_model_structure_constants!(C, 𝓂.equations.calibration_parameters)
    ms = C.post_complete_parameters
    all_SS = expand_steady_state(SS_and_pars, ms)

    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces, caching = caching)

    𝐒₁, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                         C,
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
            C)
    end

    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian, 𝓂.workspaces, caching = caching)

    𝐒₂_raw_untyped, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                                                  initial_guess = 𝓂.caches.second_order_solution,
                                                  opts = opts,
                                                  parameter_values = parameters,
                                                  caching = caching)

    𝐒₂_raw = sparse(𝐒₂_raw_untyped)::SparseMatrixCSC{M, Int}  # was: dense_to_sparse

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
            C)
    end

    𝐒₁ = [𝐒₁[:,1:T.nPast_not_future_and_mixed] zeros(M, T.nVars) 𝐒₁[:,T.nPast_not_future_and_mixed+1:end]]

    aug_state₁ = [zeros(M, T.nPast_not_future_and_mixed); one(M); zeros(M, T.nExo)]
    tmp = collect(T.I_nPast - 𝐒₁[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed])
    rhs = collect((𝐒₂_raw * compressed_kron²_power(aug_state₁) / 2)[T.past_not_future_and_mixed_idx])

    if M === Float64
        ensure_sss_tmp_lu_buffer!(𝓂.workspaces.second_order, tmp, rhs)
        tmp_sol = 𝒮.solve!(𝓂.workspaces.second_order.sss_tmp_lu_buffer)

        if tmp_sol.retcode != 𝒮.SciMLBase.ReturnCode.Default && !𝒮.SciMLBase.successful_retcode(tmp_sol.retcode)
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
                C)
        end

        SSSstates = collect(tmp_sol.u)
    else
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
                C)
        end

        SSSstates = collect(tmp̄ \ rhs)
    end

    SSSstates = SSSstates::Vector{M}

        return (true,
            all_SS,
            SS_and_pars,
            solution_error,
            ∇₁,
            ∇₂,
            𝐒₁,
            𝐒₂_raw,
            SSSstates,
            C)
end

function calculate_stochastic_steady_state(::Val{:second_order},
                                           parameters::Vector{M},
                                           𝓂::ℳ;
                                           opts::CalculationOptions = merge_calculation_options(),
                                           estimation::Bool = false,
                                           caching::Bool = true)::Tuple{Vector{M}, Bool, Vector{M}, M, Matrix{M}, SparseMatrixCSC{M, Int}, Matrix{M}, SparseMatrixCSC{M, Int}} where M
    # Cache hit: return cached SSS if valid for current parameters
    if caching && M === Float64 && !isempty(parameters) &&
       cache_valid_for_parameters(𝓂.caches.valid_for.second_order_stochastic_steady_state, parameters)
        cached_sss = 𝓂.caches.second_order_stochastic_steady_state::Vector{M}
        if !isempty(cached_sss)
            T = 𝓂.constants.post_model_macro
            SS_and_pars = 𝓂.caches.non_stochastic_steady_state::Vector{M}
            ∇₁ = Matrix(𝓂.caches.jacobian)::Matrix{M}
            ∇₂ = sparse(𝓂.caches.hessian)::SparseMatrixCSC{M, Int}  # was: dense_to_sparse
            𝐒₁_raw = Matrix(𝓂.caches.first_order_solution_matrix)::Matrix{M}
            𝐒₁ = [𝐒₁_raw[:,1:T.nPast_not_future_and_mixed] zeros(M, T.nVars) 𝐒₁_raw[:,T.nPast_not_future_and_mixed+1:end]]
            𝐒₂ = sparse(𝓂.caches.second_order_solution)::SparseMatrixCSC{M, Int}
            return cached_sss, true, SS_and_pars, zero(M), ∇₁, ∇₂, 𝐒₁, 𝐒₂
        end
    end

    common = prepare_stochastic_steady_state_base_terms(parameters, 𝓂, opts = opts, estimation = estimation, caching = caching)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂_raw, SSSstates, _ = common

    if !ok
        if caching && M === Float64; 𝓂.caches.second_order_stochastic_steady_state = all_SS; end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0)
    end

    𝐒₂ = 𝐒₂_raw

    T = 𝓂.constants.post_model_macro
    A = 𝐒₁[:,1:T.nPast_not_future_and_mixed]
    n_state_aug = T.nPast_not_future_and_mixed + 1
    n_state_pair = n_state_aug * (n_state_aug + 1) ÷ 2
    # Newton only uses past/mixed rows and the state/constant prefix of the
    # compressed policy coefficients; shock columns are not unknowns here.
    A_sss = 𝐒₁[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
    B_sss = 𝐒₂[T.past_not_future_and_mixed_idx,1:n_state_pair]

    SSSstates, converged = solve_stochastic_steady_state_newton(
        Val(:second_order), A_sss, B_sss, collect(SSSstates), 𝓂; filtered = true)

    if !converged
        if opts.verbose println("SSS not found") end
        if caching && M === Float64; 𝓂.caches.second_order_stochastic_steady_state = all_SS; end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0)
    end

    aug_sss = [SSSstates; one(M); zeros(M, 𝓂.constants.post_model_macro.nExo)]
    state = A * SSSstates + (𝐒₂ * compressed_kron²_power(aug_sss) / 2)
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
                                           caching::Bool = true)::Tuple{Vector{M}, Bool, Vector{M}, M, Matrix{M}, SparseMatrixCSC{M, Int}, Matrix{M}, SparseMatrixCSC{M, Int}} where M
    # Cache hit: return cached pruned SSS if valid for current parameters
    if caching && M === Float64 && !isempty(parameters) &&
       cache_valid_for_parameters(𝓂.caches.valid_for.pruned_second_order_stochastic_steady_state, parameters)
        cached_sss = 𝓂.caches.pruned_second_order_stochastic_steady_state::Vector{M}
        if !isempty(cached_sss)
            T = 𝓂.constants.post_model_macro
            SS_and_pars = 𝓂.caches.non_stochastic_steady_state::Vector{M}
            ∇₁ = Matrix(𝓂.caches.jacobian)::Matrix{M}
            ∇₂ = sparse(𝓂.caches.hessian)::SparseMatrixCSC{M, Int}  # was: dense_to_sparse
            𝐒₁_raw = Matrix(𝓂.caches.first_order_solution_matrix)::Matrix{M}
            𝐒₁ = [𝐒₁_raw[:,1:T.nPast_not_future_and_mixed] zeros(M, T.nVars) 𝐒₁_raw[:,T.nPast_not_future_and_mixed+1:end]]
            𝐒₂ = sparse(𝓂.caches.second_order_solution)::SparseMatrixCSC{M, Int}
            return cached_sss, true, SS_and_pars, zero(M), ∇₁, ∇₂, 𝐒₁, 𝐒₂
        end
    end

    common = prepare_stochastic_steady_state_base_terms(parameters, 𝓂, opts = opts, estimation = estimation, caching = caching)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂_raw, SSSstates, _ = common

    if !ok
        if caching && M === Float64; 𝓂.caches.pruned_second_order_stochastic_steady_state = all_SS; end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0)
    end

    𝐒₂ = 𝐒₂_raw

    T = 𝓂.constants.post_model_macro
    aug_state₁ = [zeros(M, T.nPast_not_future_and_mixed); one(M); zeros(M, T.nExo)]
    state = 𝐒₁[:,1:T.nPast_not_future_and_mixed] * SSSstates +
            𝐒₂ * compressed_kron²_power(aug_state₁) / 2

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
                                              tol::AbstractFloat = 1e-14,
                                              filtered::Bool = false)::Tuple{Vector{R}, Bool} where R <: AbstractFloat
    # @timeit_debug timer "Setup matrices" begin

    # Get cached computational constants
    constants = initialise_constants!(𝓂)
    so = ensure_computational_constants!(constants)
    T = constants.post_model_macro
    I_nPast = T.I_nPast

    A = filtered ? 𝐒₁ : 𝐒₁[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
    n_state_aug = T.nPast_not_future_and_mixed + 1
    n_state_pair = n_state_aug * (n_state_aug + 1) ÷ 2
    B = filtered ? 𝐒₂ : 𝐒₂[T.past_not_future_and_mixed_idx, 1:n_state_pair]
    B̂ = B

    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6

    ℂ = 𝓂.workspaces.second_order
    nPast = length(x)
    state_identity = @view so.I_state_vol[:, 1:nPast]
    ensure_sss_kron_buffers!(ℂ, nPast; third_order=false)
    x_aug = ℂ.x_aug_buf
    x_aug[end] = one(R)
    kron_x_aug_xx = ℂ.kron_x_aug_xx
    kron_x_aug_I = ℂ.kron_x_aug_I

    for i in 1:max_iters
        copyto!(x_aug, 1, x, 1, nPast)

        compressed_kron²!(kron_x_aug_I, x_aug, state_identity)
        ∂x = (A + B * kron_x_aug_I - I_nPast)

        compressed_kron²_power!(kron_x_aug_xx, x_aug)
        x̂ = A * x + B̂ * kron_x_aug_xx / 2

        Δx = x̂ - x
        ensure_dx_lu_buffer!(ℂ, ∂x, Δx)
        sol = 𝒮.solve!(ℂ.dx_lu_buffer)

        if sol.retcode != 𝒮.SciMLBase.ReturnCode.Default && !𝒮.SciMLBase.successful_retcode(sol.retcode)
            return x, false
        end
        copyto!(Δx, sol.u)

        if i > 3 && isapprox(x̂, x, rtol = tol)
            break
        end

        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    # end # timeit_debug

    copyto!(x_aug, 1, x, 1, nPast)
    compressed_kron²_power!(kron_x_aug_xx, x_aug)
    return x, isapprox(A * x + B̂ * kron_x_aug_xx / 2, x, rtol = tol)
end





function calculate_stochastic_steady_state(::Val{:third_order},
                                           parameters::Vector{M},
                                           𝓂::ℳ;
                                           opts::CalculationOptions = merge_calculation_options(),
                                           estimation::Bool = false,
                                           caching::Bool = true)::Tuple{Vector{M}, Bool, Vector{M}, M, Matrix{M}, SparseMatrixCSC{M, Int}, SparseMatrixCSC{M, Int}, Matrix{M}, SparseMatrixCSC{M, Int}, SparseMatrixCSC{M, Int}} where M <: Real
    # Cache hit: return cached SSS if valid for current parameters
    if caching && M === Float64 && !isempty(parameters) &&
       cache_valid_for_parameters(𝓂.caches.valid_for.third_order_stochastic_steady_state, parameters)
        cached_sss = 𝓂.caches.third_order_stochastic_steady_state::Vector{M}
        if !isempty(cached_sss)
            T = 𝓂.constants.post_model_macro
            SS_and_pars = 𝓂.caches.non_stochastic_steady_state::Vector{M}
            ∇₁ = Matrix(𝓂.caches.jacobian)::Matrix{M}
            ∇₂ = sparse(𝓂.caches.hessian)::SparseMatrixCSC{M, Int}  # was: dense_to_sparse
            ∇₃ = sparse(𝓂.caches.third_order_derivatives)::SparseMatrixCSC{M, Int}  # was: dense_to_sparse
            𝐒₁_raw = Matrix(𝓂.caches.first_order_solution_matrix)::Matrix{M}
            𝐒₁ = [𝐒₁_raw[:,1:T.nPast_not_future_and_mixed] zeros(M, T.nVars) 𝐒₁_raw[:,T.nPast_not_future_and_mixed+1:end]]
            𝐒₂ = sparse(𝓂.caches.second_order_solution)::SparseMatrixCSC{M, Int}
            𝐒₃ = sparse(𝓂.caches.third_order_solution)::SparseMatrixCSC{M, Int}
            return cached_sss, true, SS_and_pars, zero(M), ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃
        end
    end

    common = prepare_stochastic_steady_state_base_terms(parameters, 𝓂, opts = opts, estimation = estimation, caching = caching)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂_raw, SSSstates, _ = common

    if !ok
        if caching && M === Float64; 𝓂.caches.third_order_stochastic_steady_state = all_SS; end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0)
    end

    𝐒₂ = 𝐒₂_raw

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
        if caching && M === Float64; 𝓂.caches.third_order_stochastic_steady_state = all_SS; end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0)
    end

    T = 𝓂.constants.post_model_macro
    A = 𝐒₁[:,1:T.nPast_not_future_and_mixed]
    𝐒₃ = sparse(𝐒₃)::SparseMatrixCSC{M, Int}
    n_state_aug = T.nPast_not_future_and_mixed + 1
    n_state_pair = n_state_aug * (n_state_aug + 1) ÷ 2
    n_state_triple = n_state_aug * (n_state_aug + 1) * (n_state_aug + 2) ÷ 6
    # Keep the Newton system restricted to the state/constant prefix of the
    # compressed coefficients. The full compressed matrices remain available
    # for the returned solution and final state evaluation below.
    A_sss = 𝐒₁[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
    B_sss = 𝐒₂[T.past_not_future_and_mixed_idx,1:n_state_pair]
    C_sss = 𝐒₃[T.past_not_future_and_mixed_idx,1:n_state_triple]
    SSSstates, converged = solve_stochastic_steady_state_newton(
        Val(:third_order), A_sss, B_sss, C_sss, collect(SSSstates), 𝓂; filtered = true)

    if !converged
        if opts.verbose println("SSS not found") end
        if caching && M === Float64; 𝓂.caches.third_order_stochastic_steady_state = all_SS; end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0)
    end

    aug_sss = [SSSstates; one(M); zeros(M, 𝓂.constants.post_model_macro.nExo)]
    state = A * SSSstates + 𝐒₂ * compressed_kron²_power(aug_sss) / 2 + 𝐒₃ * compressed_kron³_power(aug_sss) / 6


    result = all_SS + Vector{M}(state)

    if caching && M === Float64
        𝓂.caches.third_order_stochastic_steady_state = result
        𝓂.caches.valid_for.third_order_stochastic_steady_state = Float64.(parameters)
    end

    return result, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃
end

function calculate_stochastic_steady_state(::Val{:pruned_third_order},
                                           parameters::Vector{M},
                                           𝓂::ℳ;
                                           opts::CalculationOptions = merge_calculation_options(),
                                           estimation::Bool = false,
                                           caching::Bool = true)::Tuple{Vector{M}, Bool, Vector{M}, M, Matrix{M}, SparseMatrixCSC{M, Int}, SparseMatrixCSC{M, Int}, Matrix{M}, SparseMatrixCSC{M, Int}, SparseMatrixCSC{M, Int}} where M <: Real
    # Cache hit: return cached pruned SSS if valid for current parameters
    if caching && M === Float64 && !isempty(parameters) &&
       cache_valid_for_parameters(𝓂.caches.valid_for.pruned_third_order_stochastic_steady_state, parameters)
        cached_sss = 𝓂.caches.pruned_third_order_stochastic_steady_state::Vector{M}
        if !isempty(cached_sss)
            T = 𝓂.constants.post_model_macro
            SS_and_pars = 𝓂.caches.non_stochastic_steady_state::Vector{M}
            ∇₁ = Matrix(𝓂.caches.jacobian)::Matrix{M}
            ∇₂ = sparse(𝓂.caches.hessian)::SparseMatrixCSC{M, Int}  # was: dense_to_sparse
            ∇₃ = sparse(𝓂.caches.third_order_derivatives)::SparseMatrixCSC{M, Int}  # was: dense_to_sparse
            𝐒₁_raw = Matrix(𝓂.caches.first_order_solution_matrix)::Matrix{M}
            𝐒₁ = [𝐒₁_raw[:,1:T.nPast_not_future_and_mixed] zeros(M, T.nVars) 𝐒₁_raw[:,T.nPast_not_future_and_mixed+1:end]]
            𝐒₂ = sparse(𝓂.caches.second_order_solution)::SparseMatrixCSC{M, Int}
            𝐒₃ = sparse(𝓂.caches.third_order_solution)::SparseMatrixCSC{M, Int}
            return cached_sss, true, SS_and_pars, zero(M), ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃
        end
    end

    common = prepare_stochastic_steady_state_base_terms(parameters, 𝓂, opts = opts, estimation = estimation, caching = caching)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂_raw, SSSstates, _ = common

    if !ok
        if caching && M === Float64; 𝓂.caches.pruned_third_order_stochastic_steady_state = all_SS; end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0)
    end

    𝐒₂ = 𝐒₂_raw

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
        if caching && M === Float64; 𝓂.caches.pruned_third_order_stochastic_steady_state = all_SS; end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0)
    end

    T = 𝓂.constants.post_model_macro
    aug_state₁ = [zeros(M, T.nPast_not_future_and_mixed); one(M); zeros(M, T.nExo)]
    state = 𝐒₁[:,1:T.nPast_not_future_and_mixed] * SSSstates + 𝐒₂ * compressed_kron²_power(aug_state₁) / 2

    result = all_SS + Vector{M}(state)

    if caching && M === Float64
        𝓂.caches.pruned_third_order_stochastic_steady_state = result
        𝓂.caches.valid_for.pruned_third_order_stochastic_steady_state = Float64.(parameters)
    end

    return result, true, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃
end


function solve_stochastic_steady_state_newton(::Val{:third_order}, 
                                              𝐒₁::Matrix{Float64}, 
                                              𝐒₂::AbstractMatrix{Float64},
                                              𝐒₃::AbstractMatrix{Float64},
                                              x::Vector{Float64},
                                              𝓂::ℳ;
                                              tol::AbstractFloat = 1e-14,
                                              filtered::Bool = false)::Tuple{Vector{Float64}, Bool}
    # Get cached computational constants
    T = 𝓂.constants.post_model_macro
    I_nPast = T.I_nPast
    so = ensure_computational_constants!(𝓂.constants)
    
    A = filtered ? 𝐒₁ : 𝐒₁[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed]
    n_state_aug = T.nPast_not_future_and_mixed + 1
    n_state_pair = n_state_aug * (n_state_aug + 1) ÷ 2
    n_state_triple = n_state_aug * (n_state_aug + 1) * (n_state_aug + 2) ÷ 6
    B = filtered ? 𝐒₂ : 𝐒₂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx, 1:n_state_pair]
    B̂ = B
    C = filtered ? 𝐒₃ : 𝐒₃[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx, 1:n_state_triple]
    Ĉ = C

    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6

    ℂ = 𝓂.workspaces.third_order
    nPast = length(x)
    state_identity = @view so.I_state_vol[:, 1:nPast]
    ensure_sss_kron_buffers!(ℂ, nPast; third_order=true)
    x_aug = ℂ.x_aug_buf
    x_aug[end] = 1.0
    kron_x_aug = ℂ.kron_x_aug_xx
    kron_x_kron = ℂ.kron_x_aug_x_kron
    kron_x_aug_I = ℂ.kron_x_aug_I
    kron_x_kron_I = ℂ.kron_x_kron_I

    for i in 1:max_iters
        copyto!(x_aug, 1, x, 1, nPast)
        compressed_kron²_power!(kron_x_aug, x_aug)
        compressed_kron³_power!(kron_x_kron, x_aug)

        compressed_kron²!(kron_x_aug_I, x_aug, state_identity)
        compressed_kron³!(kron_x_kron_I, x_aug, x_aug, state_identity)
        ∂x = (A + B * kron_x_aug_I + C * kron_x_kron_I / 2 - I_nPast)

        Δx = (A * x + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6 - x)
        ensure_dx_lu_buffer!(ℂ, ∂x, Δx)
        sol = 𝒮.solve!(ℂ.dx_lu_buffer)

        if sol.retcode != 𝒮.SciMLBase.ReturnCode.Default && !𝒮.SciMLBase.successful_retcode(sol.retcode)
            return x, false
        end
        copyto!(Δx, sol.u)

        if i > 5 && isapprox(A * x + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    copyto!(x_aug, 1, x, 1, nPast)
    compressed_kron²_power!(kron_x_aug, x_aug)
    compressed_kron³_power!(kron_x_kron, x_aug)
    return x, isapprox(A * x + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6, x, rtol = tol)
end


end # @stable
