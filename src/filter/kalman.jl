@stable default_mode = "disable" begin


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
                                                initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
                                                filter_algorithm::Symbol = :LagrangeNewton,
                                                lyapunov_algorithm::Symbol = :doubling,
                                                on_failure_loglikelihood::U = -Inf,
                                                measurement_error::Union{Nothing,AbstractVector{<:Real},AbstractMatrix{<:Real}} = nothing,
                                                opts::CalculationOptions = merge_calculation_options())::S where {S <: Real, U <: AbstractFloat}
    presample_periods = normalize_presample_periods(presample_periods, size(data_in_deviations, 2))
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

    # When S === Float64 (typical hot path) reuse the workspace `𝐁` buffer and
    # the LAPACK-backed Lyapunov solver. When S is non-Float64 (e.g. Dual from
    # ForwardDiff'ing parameters or `initial_state`) the workspace buffers and
    # `lyap_ws` are Float64-only and would strip the wider eltype; allocate a
    # fresh `𝐁` and use the generic Lyapunov path instead.
    if S === Float64
        𝐁 = kalman_ws.𝐁
        ℒ.mul!(𝐁, B, B')

        P = initial_covariance isa AbstractMatrix ?
            convert(Matrix{S}, initial_covariance) :
            get_initial_covariance(Val(initial_covariance), A, 𝐁, lyap_ws, opts = opts)
    else
        𝐁 = B * B'

        P = initial_covariance isa AbstractMatrix ?
            convert(Matrix{S}, initial_covariance) :
            get_initial_covariance(Val(initial_covariance), A, 𝐁, lyap_ws, opts = opts)
    end

    # Initial mean for the Kalman recursion. `state` is the deviation from
    # the steady state at which the filter is initialised; pre-edit this was
    # implicitly zero, and that remains the default (state[1] = zeros(nVars)
    # for :first_order). A non-zero state[1] reflects a user-supplied
    # initial_state at the get_loglikelihood level.
    u₀ = state[1][observables_and_states]

    return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, kalman_ws, u₀, presample_periods = presample_periods, verbose = opts.verbose, on_failure_loglikelihood = on_failure_loglikelihood, measurement_error = measurement_error)
    # timer = timer,
end


function calculate_loglikelihood_with_missing(::Val{:kalman},
                                ::Val,
                                observables_index::Vector{Int},
                                                𝐒::Union{Matrix{S},Vector{AbstractMatrix{S}}},
                                                data_in_deviations::Matrix{S},
                                                constants::constants,
                                                state,
                                                workspaces::workspaces,
                                                obs_idx_per_t::Vector{Vector{Int}};
                                                warmup_iterations::Int = 0,
                                                presample_periods::Int = 0,
                                                initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
                                                filter_algorithm::Symbol = :LagrangeNewton,
                                                lyapunov_algorithm::Symbol = :doubling,
                                                on_failure_loglikelihood::U = -Inf,
                                                measurement_error::Union{Nothing,AbstractVector{<:Real},AbstractMatrix{<:Real}} = nothing,
                                                opts::CalculationOptions = merge_calculation_options())::S where {S <: Real, U <: AbstractFloat}
    presample_periods = normalize_presample_periods(presample_periods, size(data_in_deviations, 2))
    T = constants.post_model_macro
    idx_constants = constants.post_complete_parameters
    lyap_ws = ensure_lyapunov_workspace!(workspaces, T.nVars, :first_order)

    observables_and_states = sort(union(T.past_not_future_and_mixed_idx, observables_index))
    observables_sorted = sort(observables_index)
    I_nVars = idx_constants.diag_nVars

    A = @views 𝐒[observables_and_states, 1:T.nPast_not_future_and_mixed] * I_nVars[T.past_not_future_and_mixed_idx, observables_and_states]
    B = @views 𝐒[observables_and_states, T.nPast_not_future_and_mixed+1:end]

    C = @views I_nVars[observables_sorted, observables_and_states]

    kalman_ws = ensure_kalman_workspaces!(workspaces, size(C, 1), size(C, 2))

    𝐁 = kalman_ws.𝐁
    ℒ.mul!(𝐁, B, B')

    P = initial_covariance isa AbstractMatrix ?
        convert(Matrix{S}, initial_covariance) :
        get_initial_covariance(Val(initial_covariance), A, 𝐁, lyap_ws, opts = opts)

    u₀ = state[1][observables_and_states]

    return run_kalman_iterations_missing(A, 𝐁, C, P, data_in_deviations,
                                          obs_idx_per_t, kalman_ws, u₀,
                                          presample_periods = presample_periods,
                                          verbose = opts.verbose,
                                          on_failure_loglikelihood = on_failure_loglikelihood,
                                          measurement_error = measurement_error)
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


# `measurement_error` is the covariance H of the Gaussian measurement error in
# yₜ = C xₜ + ηₜ, ηₜ ~ N(0, H). It is *not* a standard deviation: a vector is read
# as the per-observable variances (the diagonal of H), a matrix as the full
# covariance H. `nothing` means no measurement error. It enters the filter only
# through the innovation covariance, F = C P C' + H.
function run_kalman_iterations(A::Matrix{S},
                                𝐁::Matrix{S},
                                C::AbstractMatrix{R},
                                P::Matrix{S},
                                data_in_deviations::Matrix{S},
                                ws::kalman_workspace,
                                u₀::AbstractVector{V};
                                presample_periods::Int = 0,
                                on_failure_loglikelihood::U = -Inf,
                                measurement_error::Union{Nothing,AbstractVector{<:Real},AbstractMatrix{<:Real}} = nothing,
                                # timer::TimerOutput = TimerOutput(),
                                verbose::Bool = false) where {S <: Real, R <: Real, V <: Real, U <: AbstractFloat}
    presample_periods = normalize_presample_periods(presample_periods, size(data_in_deviations, 2))

    # Promoted working eltype. For the Float64 hot path (S = V = R = Float64)
    # this is Float64 and the function reuses the Float64-typed `ws` and the
    # LAPACK-backed `factorize_lu!` / `solve_lu_*!` routines. For non-Float64
    # eltypes (e.g. ForwardDiff `Dual`s introduced by differentiating w.r.t.
    # `initial_state` or parameters) the function falls back to fresh
    # allocations and Base.lu, since the workspace buffers and FastLapack
    # routines are Float64-only. The branches are guarded by `T === Float64`
    # which is constant-folded per specialization, so the active path has no
    # runtime overhead.
    T = promote_type(S, V, R)
    n_state = size(A, 1)
    n_obs   = size(C, 1)

    if T === Float64
        u    = ws.u
        z    = ws.z
        ztmp = ws.ztmp
        utmp = ws.utmp
        Ctmp = ws.Ctmp
        F    = ws.F
        K    = ws.K
        tmp  = ws.tmp
        Ptmp = ws.Ptmp
        Pwork = P
        copyto!(u, u₀)
    else
        u    = collect(T, u₀)
        z    = Vector{T}(undef, n_obs)
        ztmp = Vector{T}(undef, n_obs)
        utmp = Vector{T}(undef, n_state)
        Ctmp = Matrix{T}(undef, n_obs, n_state)
        F    = Matrix{T}(undef, n_obs, n_obs)
        K    = Matrix{T}(undef, n_state, n_obs)
        tmp  = Matrix{T}(undef, n_state, n_state)
        Ptmp = Matrix{T}(undef, n_state, n_state)
        Pwork = collect(T, P)
    end

    ℒ.mul!(z, C, u)                          # z = C * u

    loglik = zero(T)

    for t in 1:size(data_in_deviations, 2)
        if any(!isfinite, z)
            if verbose println("KF not finite at step $t") end
            return T(on_failure_loglikelihood)
        end

        ℒ.axpby!(one(T), @view(data_in_deviations[:, t]), -one(T), z)   # z = data[:,t] - z  (innovation v)

        ℒ.mul!(Ctmp, C, Pwork)                                  # Ctmp = C * P
        ℒ.mul!(F, Ctmp, C')                                     # F = C * P * C'

        # Add the measurement-error covariance H: F = C P C' + H. `H` may be a
        # vector of per-observable variances (diagonal H, the common case) or a
        # full covariance matrix; both are in the innovation (data-row) order,
        # which matches F's rows and columns.
        if measurement_error !== nothing
            if measurement_error isa AbstractMatrix
                @inbounds for j in 1:n_obs, i in 1:n_obs
                    F[i, j] += measurement_error[i, j]
                end
            else
                @inbounds for i in 1:n_obs
                    F[i, i] += measurement_error[i]
                end
            end
        end

        if T === Float64
            ws.fast_lu_ws_f, ws.fast_lu_dims_f, solved_F, luF = factorize_lu!(Val(:FastLapack), F,
                                                                               ws.fast_lu_ws_f,
                                                                               ws.fast_lu_dims_f)
            if !solved_F
                if verbose println("KF factorisation failed step $t") end
                return T(on_failure_loglikelihood)
            end

            logabsdetF = zero(T)
            signF = isodd(count(i -> ws.fast_lu_ws_f.ipiv[i] != i, eachindex(ws.fast_lu_ws_f.ipiv))) ? -one(T) : one(T)
            @inbounds for i in 1:size(F, 1)
                di = F[i, i]
                if di == 0
                    if verbose println("KF factorisation failed step $t") end
                    return T(on_failure_loglikelihood)
                end
                logabsdetF += log(abs(di))
                signF *= sign(di)
            end
            if signF <= 0 || logabsdetF < log(eps(Float64))
                if verbose println("KF factorisation failed step $t") end
                return T(on_failure_loglikelihood)
            end

            if t > presample_periods
                copyto!(ztmp, z)
                solve_lu_left!(F, ztmp, ws.fast_lu_ws_f, luF)   # ztmp = F \ z
                loglik += logabsdetF + ℒ.dot(z, ztmp)
            end

            # K = P * C' / F
            ℒ.mul!(K, Pwork, C')
            solve_lu_right!(F, K, ws.fast_lu_ws_f, luF, ws.fast_lu_rhs_t_k)
        else
            Flu = ℒ.lu(F, check = false)
            if !ℒ.issuccess(Flu)
                if verbose println("KF factorisation failed step $t") end
                return T(on_failure_loglikelihood)
            end
            logabsdetF, signF = ℒ.logabsdet(Flu)

            if signF <= 0 || logabsdetF < log(eps(Float64))
                if verbose println("KF factorisation failed step $t") end
                return T(on_failure_loglikelihood)
            end

            if t > presample_periods
                copyto!(ztmp, z)
                ℒ.ldiv!(Flu, ztmp)                         # ztmp = F⁻¹ * z
                loglik += logabsdetF + ℒ.dot(z, ztmp)
            end

            # K = P * C' / F
            ℒ.mul!(K, Pwork, C')
            ℒ.rdiv!(K, Flu)                                  # K = K / F
        end

        # P = A * (P - K * C * P) * A' + 𝐁
        ℒ.mul!(tmp, K, C)                                       # tmp = K * C
        ℒ.mul!(Ptmp, tmp, Pwork)                                # Ptmp = K * C * P
        ℒ.axpy!(-one(T), Ptmp, Pwork)                           # P = P - K * C * P

        ℒ.mul!(Ptmp, A, Pwork)                                  # Ptmp = A * P
        ℒ.mul!(Pwork, Ptmp, A')                                 # P = A * P * A'
        ℒ.axpy!(one(T), 𝐁, Pwork)                              # P = P + 𝐁

        # u = A * (u + K * v)
        ℒ.mul!(u, K, z, one(T), one(T))                         # u = u + K * v
        ℒ.mul!(utmp, A, u)                                      # utmp = A * u
        u .= utmp                                               # u = A * (u + K * v)

        ℒ.mul!(z, C, u)                                         # z = C * u
    end

    return -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2π)) / 2
end


# Missing-data variant of run_kalman_iterations.
# Uses the same workspace buffers but takes per-period sub-views of size m_t
# (= number of observed variables in period t). Periods with m_t == 0 become
# pure predict steps (no update, no likelihood contribution).
# `measurement_error` carries the same meaning as in `run_kalman_iterations`: the
# covariance H (vector ⇒ per-observable variances, matrix ⇒ full covariance), not
# a standard deviation. Here it is subset to the observed rows of period t.
function run_kalman_iterations_missing(A::Matrix{S}, 
                                𝐁::Matrix{S},
                                C::AbstractMatrix{R}, 
                                P::Matrix{S}, 
                                data_in_deviations::Matrix{S},
                                obs_idx_per_t::Vector{Vector{Int}},
                                ws::kalman_workspace,
                                u₀::AbstractVector{<:Real};
                                presample_periods::Int = 0,
                                on_failure_loglikelihood::U = -Inf,
                                measurement_error::Union{Nothing,AbstractVector{<:Real},AbstractMatrix{<:Real}} = nothing,
                                verbose::Bool = false)::S where {S <: Float64, R <: Real, U <: AbstractFloat}

    n_obs   = size(C, 1)
    n_state = size(C, 2)
    n_steps = size(data_in_deviations, 2)
    presample_periods = normalize_presample_periods(presample_periods, n_steps)

    u    = ws.u
    z    = ws.z
    ztmp = ws.ztmp
    utmp = ws.utmp
    Ctmp = ws.Ctmp
    F    = ws.F
    K    = ws.K
    tmp  = ws.tmp
    Ptmp = ws.Ptmp

    copyto!(u, u₀)

    loglik = S(0.0)
    n_obs_total = 0

    for t in 1:n_steps
        idx = obs_idx_per_t[t]
        m   = length(idx)

        if any(!isfinite, u)
            if verbose println("KF not finite at step $t") end
            return on_failure_loglikelihood
        end

        if m == 0
            # Pure predict step.
            ℒ.mul!(utmp, A, u)
            u .= utmp

            ℒ.mul!(Ptmp, A, P)
            ℒ.mul!(P, Ptmp, A')
            ℒ.axpy!(1, 𝐁, P)

            continue
        end

        Cv  = view(C, idx, :)                  # m × n_state
        dv  = view(data_in_deviations, idx, t) # m
        zv  = view(z, 1:m)                     # m  (innovation buffer)
        ztv = view(ztmp, 1:m)
        Ctv = view(Ctmp, 1:m, :)               # m × n_state
        Fv  = view(F, 1:m, 1:m)                # m × m
        Kv  = view(K, :, 1:m)                  # n_state × m
        rhs_t_kv = view(ws.fast_lu_rhs_t_k, 1:m, :)  # m × n_state

        # innovation v = data[idx, t] - C[idx,:] * u   (stored into zv)
        ℒ.mul!(zv, Cv, u)
        @inbounds for i in 1:m
            zv[i] = dv[i] - zv[i]
        end

        ℒ.mul!(Ctv, Cv, P)        # Ctv = C[idx,:] * P
        ℒ.mul!(Fv, Ctv, Cv')      # Fv = C[idx,:] * P * C[idx,:]'

        # Add the measurement-error covariance restricted to the observed rows
        # (the conditional block H[idx, idx] of a full covariance matrix).
        if measurement_error !== nothing
            if measurement_error isa AbstractMatrix
                @inbounds for j in 1:m, i in 1:m
                    Fv[i, j] += measurement_error[idx[i], idx[j]]
                end
            else
                @inbounds for i in 1:m
                    Fv[i, i] += measurement_error[idx[i]]
                end
            end
        end

        ws.fast_lu_ws_f, ws.fast_lu_dims_f, solved_F, luF = factorize_lu!(Val(:Julia), Fv,
                                                                            ws.fast_lu_ws_f,
                                                                            ws.fast_lu_dims_f)

        if !solved_F
            if verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood
        end

        logabsdetF = zero(S)
        signF = isodd(count(i -> luF.ipiv[i] != i, 1:m)) ? -one(S) : one(S)
        @inbounds for i in 1:m
            di = Fv[i, i]
            if di == 0
                if verbose println("KF factorisation failed step $t") end
                return on_failure_loglikelihood
            end
            logabsdetF += log(abs(di))
            signF *= sign(di)
        end

        if signF <= 0 || logabsdetF < log(eps(Float64))
            if verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood
        end

        if t > presample_periods
            copyto!(ztv, zv)
            solve_lu_left!(Fv, ztv, ws.fast_lu_ws_f, luF; use_fastlapack_lu = false)
            loglik += logabsdetF + ℒ.dot(zv, ztv)
            n_obs_total += m
        end

        ℒ.mul!(Kv, P, Cv')                   # K = P * C[idx,:]'
        solve_lu_right!(Fv, Kv, ws.fast_lu_ws_f, luF, rhs_t_kv; use_fastlapack_lu = false)  # K = K / F

        # P = A * (P - K * C[idx,:] * P) * A' + 𝐁
        ℒ.mul!(tmp, Kv, Cv)                  # tmp = K * C[idx,:]
        ℒ.mul!(Ptmp, tmp, P)                 # Ptmp = K * C[idx,:] * P
        ℒ.axpy!(-1, Ptmp, P)                 # P -= Ptmp

        ℒ.mul!(Ptmp, A, P)
        ℒ.mul!(P, Ptmp, A')
        ℒ.axpy!(1, 𝐁, P)

        # u = A * (u + K * v)
        ℒ.mul!(u, Kv, zv, 1, 1)
        ℒ.mul!(utmp, A, u)
        u .= utmp
    end

    return -(loglik + n_obs_total * log(2π)) / 2
end


@unstable function filter_data_with_model(𝓂::ℳ,
    data_in_deviations::KeyedArray{Float64},
    ::Val{:first_order}, # algo
    ::Val{:kalman}; # filter,
    warmup_iterations::Int = 0,
    opts::CalculationOptions = merge_calculation_options(),
    initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
    smooth::Bool = true)

    obs_axis = collect(axiskeys(data_in_deviations,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    filtered_and_smoothed = filter_and_smooth(𝓂, data_in_deviations, obs_symbols; opts = opts, initial_covariance = initial_covariance)

    variables           = filtered_and_smoothed[smooth ? 1 : 5]
    standard_deviations = filtered_and_smoothed[smooth ? 2 : 6]
    shocks              = filtered_and_smoothed[smooth ? 3 : 7]
    decomposition       = filtered_and_smoothed[smooth ? 4 : 8]

    return variables, shocks, standard_deviations, decomposition
end



function filter_and_smooth(𝓂::ℳ, 
                            data_in_deviations::AbstractArray, 
                            observables::Vector{Symbol};
                            initial_covariance::Union{Symbol,AbstractMatrix{<:Real}} = :theoretical,
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

    # Prior on the state at the start of the sample. `:theoretical` is the ergodic
    # covariance (the historical behaviour and the default); `:diagonal` starts
    # diffuse; a matrix is used as given. Supplying B B' reproduces the inversion
    # filter's implicit prior — see the Filters page.
    P̄ = if initial_covariance isa AbstractMatrix
        Matrix{Float64}(initial_covariance)
    elseif initial_covariance == :diagonal
        Matrix{Float64}(10.0 * ℒ.I(size(A, 1)))
    else
        calculate_covariance(𝓂.parameter_values, 𝓂, opts = opts)[1]
    end

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

    # Convert any Missing entries to NaN sentinels and build per-period
    # observable index. Periods with no observed rows are handled by a
    # predict-only Kalman step; partially-missing periods compute reduced
    # m × m innovation/F/iF/K on sub-views and scatter the results back into
    # the full-size storage. Because every row/column of the scattered v[:,t],
    # iF[:,:,t], L[:,:,t] that touches a "missing" position is zero, the
    # downstream smoother equations work unchanged.
    data_in_deviations = missing_data_to_nan(data_in_deviations)
    obs_idx_per_t, has_missing = build_obs_index(data_in_deviations)

    P[:, :, 1] = P̄

    F_buf = kalman_ws.F

    # Kalman Filter
    for t in axes(data_in_deviations,2)
        idx = obs_idx_per_t[t]
        m   = length(idx)

        if m == 0
            # Predict-only step: no innovation, no Kalman update.
            # v[:, t], iF[:, :, t] stay zero; L_t = A; ϵ_t = 0.
            L[:, :, t] .= A
            μ[:, t+1]  .= A * μ[:, t]
            P[:, :, t+1] .= A * P[:, :, t] * A' .+ 𝐁
            σ[:, t]    .= sqrt.(abs.(ℒ.diag(P[:, :, t+1])))
            continue
        end

        if m == n_obs_C
            # Dense fast path.
            v[:, t]     .= data_in_deviations[:, t] - C * μ[:, t]

            @views F_buf .= C * P[:, :, t] * C'
            @views iF_t = iF[:, :, t]
            fill!(iF_t, 0.0)
            @inbounds for i in 1:n_obs_C
                iF_t[i, i] = 1.0
            end

            kalman_ws.fast_lu_ws_f, kalman_ws.fast_lu_dims_f, solved_F, _ =
                factorize_lu!(Val(:FastLapack), F_buf, kalman_ws.fast_lu_ws_f, kalman_ws.fast_lu_dims_f)

            if !solved_F
                @warn "Kalman filter stopped in period $t due to numerical stabiltiy issues."
                break
            end

            solve_lu_left!(F_buf, iF_t, kalman_ws.fast_lu_ws_f, nothing) # iF_t = F̄ \ I
            PCiF         = P[:, :, t] * C' * iF_t
            L[:, :, t]  .= A - A * PCiF * C
            P[:, :, t+1].= A * P[:, :, t] * L[:, :, t]' + 𝐁
            σ[:, t]     .= sqrt.(abs.(ℒ.diag(P[:, :, t+1])))
            μ[:, t+1]   .= A * (μ[:, t] + PCiF * v[:, t])
            ϵ[:, t]     .= B' * C' * iF_t * v[:, t]
        else
            # Partial-missing step: reduced m × m operations, scatter back.
            Cv = view(C, idx, :)                        # m × n_states
            Fv = view(F_buf, 1:m, 1:m)                  # m × m
            Fv .= Cv * P[:, :, t] * Cv'

            kalman_ws.fast_lu_ws_f, kalman_ws.fast_lu_dims_f, solved_F, luF_v =
                factorize_lu!(Val(:Julia), Fv, kalman_ws.fast_lu_ws_f, kalman_ws.fast_lu_dims_f)

            if !solved_F
                @warn "Kalman filter stopped in period $t due to numerical stabiltiy issues."
                break
            end

            iF_m = Matrix{Float64}(ℒ.I, m, m)
            solve_lu_left!(Fv, iF_m, kalman_ws.fast_lu_ws_f, luF_v; use_fastlapack_lu = false)

            v_m = data_in_deviations[idx, t] .- Cv * μ[:, t]

            PCiF = P[:, :, t] * Cv' * iF_m              # n_states × m
            L[:, :, t]  .= A - A * PCiF * Cv
            P[:, :, t+1].= A * P[:, :, t] * L[:, :, t]' + 𝐁
            σ[:, t]     .= sqrt.(abs.(ℒ.diag(P[:, :, t+1])))
            μ[:, t+1]   .= A * (μ[:, t] + PCiF * v_m)
            ϵ[:, t]     .= B' * Cv' * iF_m * v_m

            # Scatter v_m and iF_m into the full-size buffers (zero elsewhere).
            fill!(view(v, :, t), 0.0)
            @inbounds for i in 1:m
                v[idx[i], t] = v_m[i]
            end
            @views fill!(iF[:, :, t], 0.0)
            @inbounds for j in 1:m, i in 1:m
                iF[idx[i], idx[j], t] = iF_m[i, j]
            end
        end
    end


    # Historical shock decompositionm (filter)
    filter_decomposition = zeros(size(A,1), size(B,2)+2, n_obs)

    filter_decomposition[:,end,:] .= μ[:, 2:end]
    @inbounds for j in axes(B, 2), i in axes(B, 1)
        filter_decomposition[i, j, 1] = B[i, j] * ϵ[j, 1]
    end
    filter_decomposition[:,end-1,1] .= filter_decomposition[:,end,1] - sum(filter_decomposition[:,1:end-2,1],dims=2)

    for i in 2:size(data_in_deviations,2)
        filter_decomposition[:,1:end-2,i] .= A * filter_decomposition[:,1:end-2,i-1]
        @inbounds for j in axes(B, 2), k in axes(B, 1)
            filter_decomposition[k, j, i] += B[k, j] * ϵ[j, i]
        end
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
    @inbounds for j in axes(B, 2), i in axes(B, 1)
        smooth_decomposition[i, j, 1] = B[i, j] * ϵ̄[j, 1]
    end
    smooth_decomposition[:,end-1,1] .= smooth_decomposition[:,end,1] - sum(smooth_decomposition[:,1:end-2,1],dims=2)

    for i in 2:size(data_in_deviations,2)
        smooth_decomposition[:,1:end-2,i] .= A * smooth_decomposition[:,1:end-2,i-1]
        @inbounds for j in axes(B, 2), k in axes(B, 1)
            smooth_decomposition[k, j, i] += B[k, j] * ϵ̄[j, i]
        end
        smooth_decomposition[:,end-1,i] .= smooth_decomposition[:,end,i] - sum(smooth_decomposition[:,1:end-2,i],dims=2)
    end

    return μ̄, σ̄, ϵ̄, smooth_decomposition, μ[:, 2:end], σ, ϵ, filter_decomposition
end


end # @stable
