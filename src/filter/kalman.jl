
function calculate_loglikelihood(::Val{:kalman},
                                ::Val,
                                observables_index::Vector{Int}, 
                                                𝐒::Union{Matrix{S},Vector{AbstractMatrix{S}}}, 
                                                data_in_deviations::Matrix{S},
                                                constants::constants,
                                                state,
                                                workspaces::workspaces; 
                                                # timer::TimerOutput = TimerOutput(), 
                                                warmup_iterations::Int = 0,
                                                presample_periods::Int = 0,
                                                initial_covariance::Symbol = :theoretical,
                                                filter_algorithm::Symbol = :LagrangeNewton,
                                                lyapunov_algorithm::Symbol = :doubling,
                                                on_failure_loglikelihood::U = -Inf,
                                                opts::CalculationOptions = merge_calculation_options())::S where {S <: Real, U <: AbstractFloat}
    T = constants.post_model_macro
    idx_constants = constants.post_complete_parameters
    lyap_ws = ensure_lyapunov_workspace!(workspaces, T.nVars, :first_order)

    observables_and_states = sort(union(T.past_not_future_and_mixed_idx,observables_index))
    observables_sorted = sort(observables_index)
    I_nVars = idx_constants.diag_nVars

    A = @views 𝐒[observables_and_states,1:T.nPast_not_future_and_mixed] * I_nVars[T.past_not_future_and_mixed_idx, observables_and_states]
    B = @views 𝐒[observables_and_states,T.nPast_not_future_and_mixed+1:end]

    C = @views I_nVars[observables_sorted, observables_and_states]

    kalman_ws = ensure_kalman_workspaces!(workspaces, size(C, 1), size(C, 2))

    𝐁 = kalman_ws.𝐁
    ℒ.mul!(𝐁, B, B')

    # Gaussian Prior
    P = get_initial_covariance(Val(initial_covariance), A, 𝐁, lyap_ws, opts = opts)
    # timer = timer, 

    return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, kalman_ws, presample_periods = presample_periods, verbose = opts.verbose, on_failure_loglikelihood = on_failure_loglikelihood)
    # timer = timer, 
end

# Specialization for :theoretical
function get_initial_covariance(::Val{:theoretical}, 
                                A::AbstractMatrix{S}, 
                                B::AbstractMatrix{S},
                                lyap_ws::lyapunov_workspace; 
                                opts::CalculationOptions = merge_calculation_options())::Matrix{S} where S <: Real
                                # timer::TimerOutput = TimerOutput(), 
    P, _ = solve_lyapunov_equation(A, B, lyap_ws,
                                    lyapunov_algorithm = opts.lyapunov_algorithm, 
                                    tol = opts.tol.first_order.lyapunov,
                                    verbose = opts.verbose) # timer = timer, 

    return copy(P)
end


# Specialization for :diagonal
function get_initial_covariance(::Val{:diagonal}, 
                                A::AbstractMatrix{S}, 
                                B::AbstractMatrix{S},
                                lyap_ws::lyapunov_workspace; 
                                opts::CalculationOptions = merge_calculation_options())::Matrix{S} where S <: Real
                                # timer::TimerOutput = TimerOutput(), 
    P = collect(ℒ.I(size(A, 1)) * 10.0)
    return P
end


function run_kalman_iterations(A::Matrix{S}, 
                                𝐁::Matrix{S},
                                C::AbstractMatrix{R}, 
                                P::Matrix{S}, 
                                data_in_deviations::Matrix{S},
                                ws::kalman_workspace; 
                                presample_periods::Int = 0,
                                on_failure_loglikelihood::U = -Inf,
                                # timer::TimerOutput = TimerOutput(),
                                verbose::Bool = false)::S where {S <: Float64, R <: Real, U <: AbstractFloat}
    # @timeit_debug timer "Calculate Kalman filter" begin

    # Use workspaces
    u = ws.u
    z = ws.z
    ztmp = ws.ztmp
    utmp = ws.utmp
    Ctmp = ws.Ctmp
    F = ws.F
    K = ws.K
    tmp = ws.tmp
    Ptmp = ws.Ptmp
    
    # Initialize state estimate to zero
    fill!(u, zero(S))
    ℒ.mul!(z, C, u)                          # z = C * u

    loglik = S(0.0)

    # @timeit_debug timer "Loop" begin
    for t in 1:size(data_in_deviations, 2)
        if any(!isfinite, z)
            if verbose println("KF not finite at step $t") end
            return on_failure_loglikelihood 
        end

        ℒ.axpby!(1, @view(data_in_deviations[:, t]), -1, z)   # z = data[:,t] - z  (innovation v)

        ℒ.mul!(Ctmp, C, P)                                     # Ctmp = C * P
        ℒ.mul!(F, Ctmp, C')                                    # F = C * P * C'

        # Old way (≤v0.1.42): luF = lu(F)  — allocates new LU each step
        ws.fast_lu_ws_f, ws.fast_lu_dims_f, solved_F, luF = factorize_lu!(F,
                                                                           ws.fast_lu_ws_f,
                                                                           ws.fast_lu_dims_f)

        if !solved_F
            if verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood
        end

        # Old way (≤v0.1.42): Fdet = det(luF); loglik += log(Fdet) + v' * inv(F) * v
        # Current code computes log|det(F)| from the LU diagonal and pivot signs.
        logabsdetF = zero(S)
        signF = isodd(count(i -> ws.fast_lu_ws_f.ipiv[i] != i, eachindex(ws.fast_lu_ws_f.ipiv))) ? -one(S) : one(S)
        @inbounds for i in 1:size(F, 1)
            di = F[i, i]
            if di == 0
                if verbose println("KF factorisation failed step $t") end
                return on_failure_loglikelihood
            end
            logabsdetF += log(abs(di))
            signF *= sign(di)
        end

        # Early return if determinant is too small, indicating numerical instability.
        if signF <= 0 || logabsdetF < log(eps(Float64))
            if verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood
        end

        # Old way (≤v0.1.42): loglik += log(det(F)) + v' * inv(F) * v
        if t > presample_periods
            copyto!(ztmp, z)
            solve_lu_left!(F, ztmp, ws.fast_lu_ws_f, luF)      # ztmp = F \ z
            loglik += logabsdetF + ℒ.dot(z', ztmp)             # loglik += log|det(F)| + z' * (F \ z)
        end

        # Old way (≤v0.1.42): K = P * C' / F  — Kalman gain
        ℒ.mul!(K, P, C')                                       # K = P * C'
        solve_lu_right!(F, K, ws.fast_lu_ws_f, luF, ws.fast_lu_rhs_t_k)  # K = K / F

        # end # timeit_debug
        # @timeit_debug timer "Matmul" begin

        # P = A * (P - K * C * P) * A' + B
        ℒ.mul!(tmp, K, C)                                      # tmp = K * C
        ℒ.mul!(Ptmp, tmp, P)                                   # Ptmp = K * C * P
        ℒ.axpy!(-1, Ptmp, P)                                   # P = P - K * C * P

        ℒ.mul!(Ptmp, A, P)                                     # Ptmp = A * P
        ℒ.mul!(P, Ptmp, A')                                    # P = A * P * A'
        ℒ.axpy!(1, 𝐁, P)                                      # P = P + B

        # u = A * (u + K * v)
        ℒ.mul!(u, K, z, 1, 1)                                  # u = u + K * v
        ℒ.mul!(utmp, A, u)                                     # utmp = A * u
        u .= utmp                                              # u = A * (u + K * v)

        ℒ.mul!(z, C, u)                                        # z = C * u

        # end # timeit_debug
    end

    # end # timeit_debug
    # end # timeit_debug

    return -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 
end


@unstable function filter_data_with_model(𝓂::ℳ,
    data_in_deviations::KeyedArray{Float64},
    ::Val{:first_order}, # algo
    ::Val{:kalman}; # filter,
    warmup_iterations::Int = 0,
    opts::CalculationOptions = merge_calculation_options(),
    smooth::Bool = true)

    obs_axis = collect(axiskeys(data_in_deviations,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    filtered_and_smoothed = filter_and_smooth(𝓂, data_in_deviations, obs_symbols; opts = opts)

    variables           = filtered_and_smoothed[smooth ? 1 : 5]
    standard_deviations = filtered_and_smoothed[smooth ? 2 : 6]
    shocks              = filtered_and_smoothed[smooth ? 3 : 7]
    decomposition       = filtered_and_smoothed[smooth ? 4 : 8]

    return variables, shocks, standard_deviations, decomposition
end



function filter_and_smooth(𝓂::ℳ, 
                            data_in_deviations::AbstractArray{Float64}, 
                            observables::Vector{Symbol};
                            opts::CalculationOptions = merge_calculation_options())
    # Based on Durbin and Koopman (2012)
    # https://jrnold.github.io/ssmodels-in-stan/filtering-and-smoothing.html#smoothing

    @assert length(observables) == size(data_in_deviations)[1] "Data columns and number of observables are not identical. Make sure the data contains only the selected observables."
    @assert length(observables) <= 𝓂.constants.post_model_macro.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."

    sort!(observables)

    solve!(𝓂, opts = opts)
    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)
    idx_constants = constants.post_complete_parameters
    T = constants.post_model_macro

    parameters = 𝓂.parameter_values

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, opts = opts)
    
    @assert solution_error < opts.tol.nsss.acceptance_tol "Could not solve non-stochastic steady state." 

    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces)# |> Matrix

    sol, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                            constants,
                                                            𝓂.workspaces,
                                                            𝓂.caches;
                                                            opts = opts,
                                                            parameter_values = parameters)
    
    update_perturbation_counter!(𝓂.counters, solved, order = 1)

    # Direct constants access
    A = @views sol[:,1:T.nPast_not_future_and_mixed] * idx_constants.diag_nVars[T.past_not_future_and_mixed_idx,:]

    B = @views sol[:,T.nPast_not_future_and_mixed+1:end]

    C = @views ℒ.diagm(ones(T.nVars))[sort(indexin(observables, sort(union(T.aux, T.var, T.exo_present)))),:]

    𝐁 = B * B'

    P̄ = calculate_covariance(𝓂.parameter_values, 𝓂, opts = opts)[1]

    n_obs = size(data_in_deviations,2)

    n_obs_C = size(C,1)
    n_states = size(A,1)
    kalman_ws = ensure_kalman_workspaces!(𝓂.workspaces, n_obs_C, n_states)

    v = zeros(n_obs_C, n_obs)
    μ = zeros(n_states, n_obs+1) # filtered_states
    P = zeros(n_states, n_states, n_obs+1) # filtered_covariances
    σ = zeros(n_states, n_obs) # filtered_standard_deviations
    iF= zeros(n_obs_C, n_obs_C, n_obs)
    L = zeros(n_states, n_states, n_obs)
    ϵ = zeros(size(B,2), n_obs) # filtered_shocks

    P[:, :, 1] = P̄

    F_buf = kalman_ws.F

    # Kalman Filter
    for t in axes(data_in_deviations,2)
        v[:, t]     .= data_in_deviations[:, t] - C * μ[:, t]

        @views F_buf .= C * P[:, :, t] * C'
        @views iF_t = iF[:, :, t]
        fill!(iF_t, 0.0)
        @inbounds for i in 1:n_obs_C
            iF_t[i, i] = 1.0
        end

        kalman_ws.fast_lu_ws_f, kalman_ws.fast_lu_dims_f, solved_F, _ =
            factorize_lu!(F_buf, kalman_ws.fast_lu_ws_f, kalman_ws.fast_lu_dims_f)

        if !solved_F
            @warn "Kalman filter stopped in period $t due to numerical stabiltiy issues."
            break
        end

        solve_lu_left!(F_buf, iF_t, kalman_ws.fast_lu_ws_f, nothing) # iF_t = F̄ \ I
        PCiF         = P[:, :, t] * C' * iF_t
        L[:, :, t]  .= A - A * PCiF * C
        P[:, :, t+1].= A * P[:, :, t] * L[:, :, t]' + 𝐁
        σ[:, t]     .= sqrt.(abs.(ℒ.diag(P[:, :, t+1]))) # small numerical errors in this computation
        μ[:, t+1]   .= A * (μ[:, t] + PCiF * v[:, t])
        ϵ[:, t]     .= B' * C' * iF_t * v[:, t]
    end


    # Historical shock decompositionm (filter)
    filter_decomposition = zeros(size(A,1), size(B,2)+2, n_obs)

    filter_decomposition[:,end,:] .= μ[:, 2:end]
    filter_decomposition[:,1:end-2,1] .= B .* repeat(ϵ[:, 1]', size(A,1))
    filter_decomposition[:,end-1,1] .= filter_decomposition[:,end,1] - sum(filter_decomposition[:,1:end-2,1],dims=2)

    for i in 2:size(data_in_deviations,2)
        filter_decomposition[:,1:end-2,i] .= A * filter_decomposition[:,1:end-2,i-1]
        filter_decomposition[:,1:end-2,i] .+= B .* repeat(ϵ[:, i]', size(A,1))
        filter_decomposition[:,end-1,i] .= filter_decomposition[:,end,i] - sum(filter_decomposition[:,1:end-2,i],dims=2)
    end
    
    μ̄ = zeros(size(A,1), n_obs) # smoothed_states
    σ̄ = zeros(size(A,1), n_obs) # smoothed_standard_deviations
    ϵ̄ = zeros(size(B,2), n_obs) # smoothed_shocks

    r = zeros(size(A,1))
    N = zeros(size(A,1), size(A,1))

    # Kalman Smoother
    for t in n_obs:-1:1
        r       .= C' * iF[:, :, t] * v[:, t] + L[:, :, t]' * r
        μ̄[:, t] .= μ[:, t] + P[:, :, t] * r
        N       .= C' * iF[:, :, t] * C + L[:, :, t]' * N * L[:, :, t]
        σ̄[:, t] .= sqrt.(abs.(ℒ.diag(P[:, :, t] - P[:, :, t] * N * P[:, :, t]'))) # can go negative
        ϵ̄[:, t] .= B' * r
    end

    # Historical shock decompositionm (smoother)
    smooth_decomposition = zeros(size(A,1), size(B,2)+2, n_obs)

    smooth_decomposition[:,end,:] .= μ̄
    smooth_decomposition[:,1:end-2,1] .= B .* repeat(ϵ̄[:, 1]', size(A,1))
    smooth_decomposition[:,end-1,1] .= smooth_decomposition[:,end,1] - sum(smooth_decomposition[:,1:end-2,1],dims=2)

    for i in 2:size(data_in_deviations,2)
        smooth_decomposition[:,1:end-2,i] .= A * smooth_decomposition[:,1:end-2,i-1]
        smooth_decomposition[:,1:end-2,i] .+= B .* repeat(ϵ̄[:, i]', size(A,1))
        smooth_decomposition[:,end-1,i] .= smooth_decomposition[:,end,i] - sum(smooth_decomposition[:,1:end-2,i],dims=2)
    end

    return μ̄, σ̄, ϵ̄, smooth_decomposition, μ[:, 2:end], σ, ϵ, filter_decomposition
end

