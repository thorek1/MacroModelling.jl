@stable default_mode = "disable" begin

# Specialization for :kalman filter
function calculate_loglikelihood(::Val{:kalman}, 
                                algorithm, 
                                observables, 
                                𝐒, 
                                data_in_deviations, 
                                TT, 
                                presample_periods, 
                                initial_covariance, 
                                state, 
                                warmup_iterations, 
                                filter_algorithm, 
                                opts,
                                on_failure_loglikelihood) #; 
                                # timer::TimerOutput = TimerOutput())
    return calculate_kalman_filter_loglikelihood(observables, 
                                                𝐒, 
                                                data_in_deviations, 
                                                TT, 
                                                presample_periods = presample_periods, 
                                                initial_covariance = initial_covariance, 
                                                # timer = timer, 
                                                opts = opts,
                                                on_failure_loglikelihood = on_failure_loglikelihood)
end

function calculate_kalman_filter_loglikelihood(observables::Vector{Symbol}, 
                                                𝐒::Union{Matrix{S},Vector{AbstractMatrix{S}}}, 
                                                data_in_deviations::Matrix{S},
                                                T::timings; 
                                                # timer::TimerOutput = TimerOutput(), 
                                                on_failure_loglikelihood::U = -Inf,
                                                presample_periods::Int = 0, 
                                                initial_covariance::Symbol = :theoretical,
                                                opts::CalculationOptions = merge_calculation_options())::S where {S <: Real, U <: AbstractFloat}
    obs_idx = @ignore_derivatives convert(Vector{Int},indexin(observables,sort(union(T.aux,T.var,T.exo_present))))

    calculate_kalman_filter_loglikelihood(obs_idx, 𝐒, data_in_deviations, T, presample_periods = presample_periods, initial_covariance = initial_covariance, opts = opts, on_failure_loglikelihood = on_failure_loglikelihood)
    # timer = timer, 
end

function calculate_kalman_filter_loglikelihood(observables::Vector{String}, 
                                                𝐒::Union{Matrix{S},Vector{AbstractMatrix{S}}}, 
                                                data_in_deviations::Matrix{S},
                                                T::timings; 
                                                # timer::TimerOutput = TimerOutput(), 
                                                presample_periods::Int = 0, 
                                                on_failure_loglikelihood::U = -Inf,
                                                initial_covariance::Symbol = :theoretical,
                                                opts::CalculationOptions = merge_calculation_options())::S where {S <: Real, U <: AbstractFloat}
    obs_idx = @ignore_derivatives convert(Vector{Int},indexin(observables,sort(union(T.aux,T.var,T.exo_present))))

    calculate_kalman_filter_loglikelihood(obs_idx, 𝐒, data_in_deviations, T, presample_periods = presample_periods, initial_covariance = initial_covariance, opts = opts, on_failure_loglikelihood = on_failure_loglikelihood)
    # timer = timer, 
end

function calculate_kalman_filter_loglikelihood(observables_index::Vector{Int}, 
                                                𝐒::Union{Matrix{S},Vector{AbstractMatrix{S}}}, 
                                                data_in_deviations::Matrix{S},
                                                T::timings; 
                                                # timer::TimerOutput = TimerOutput(), 
                                                presample_periods::Int = 0,
                                                initial_covariance::Symbol = :theoretical,
                                                lyapunov_algorithm::Symbol = :doubling,
                                                on_failure_loglikelihood::U = -Inf,
                                                opts::CalculationOptions = merge_calculation_options())::S where {S <: Real, U <: AbstractFloat}
    observables_and_states = @ignore_derivatives sort(union(T.past_not_future_and_mixed_idx,observables_index))

    A = 𝐒[observables_and_states,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(S, length(observables_and_states)))[@ignore_derivatives(indexin(T.past_not_future_and_mixed_idx,observables_and_states)),:]
    B = 𝐒[observables_and_states,T.nPast_not_future_and_mixed+1:end]

    C = ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(observables_index), observables_and_states)),:]

    𝐁 = B * B'

    # Gaussian Prior
    P = get_initial_covariance(Val(initial_covariance), A, 𝐁, opts = opts)
    # timer = timer, 

    return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods, verbose = opts.verbose, on_failure_loglikelihood = on_failure_loglikelihood)
    # timer = timer, 
end

# Specialization for :theoretical
function get_initial_covariance(::Val{:theoretical}, 
                                A::AbstractMatrix{S}, 
                                B::AbstractMatrix{S}; 
                                opts::CalculationOptions = merge_calculation_options())::Matrix{S} where S <: Real
                                # timer::TimerOutput = TimerOutput(), 
    P, _ = solve_lyapunov_equation(A, B, 
                                    lyapunov_algorithm = opts.lyapunov_algorithm, 
                                    tol = opts.tol.lyapunov_tol,
                                    acceptance_tol = opts.tol.lyapunov_acceptance_tol,
                                    verbose = opts.verbose) # timer = timer, 

    return P
end


# Specialization for :diagonal
function get_initial_covariance(::Val{:diagonal}, 
                                A::AbstractMatrix{S}, 
                                B::AbstractMatrix{S}; 
                                opts::CalculationOptions = merge_calculation_options())::Matrix{S} where S <: Real
                                # timer::TimerOutput = TimerOutput(), 
    P = @ignore_derivatives collect(ℒ.I(size(A, 1)) * 10.0)
    return P
end


function run_kalman_iterations(A::Matrix{S}, 
                                𝐁::Matrix{S},
                                C::Matrix{Float64}, 
                                P::Matrix{S}, 
                                data_in_deviations::Matrix{S}; 
                                presample_periods::Int = 0,
                                on_failure_loglikelihood::U = -Inf,
                                # timer::TimerOutput = TimerOutput(),
                                verbose::Bool = false)::S where {S <: Float64, U <: AbstractFloat}
    # @timeit_debug timer "Calculate Kalman filter" begin

    u = zeros(S, size(C,2))

    z = C * u

    ztmp = similar(z)

    loglik = S(0.0)

    utmp = similar(u)

    Ctmp = similar(C)

    F = similar(C * C')

    K = similar(C')
    # Ktmp = similar(C')

    tmp = similar(P)
    Ptmp = similar(P)

    # @timeit_debug timer "Loop" begin
    for t in 1:size(data_in_deviations, 2)
        if !all(isfinite.(z)) 
            if verbose println("KF not finite at step $t") end
            return on_failure_loglikelihood 
        end

        ℒ.axpby!(1, data_in_deviations[:, t], -1, z)
        # v = data_in_deviations[:, t] - z

        ℒ.mul!(Ctmp, C, P) # use Octavian.jl
        ℒ.mul!(F, Ctmp, C')
        # F = C * P * C'

        # @timeit_debug timer "LU factorisation" begin
        luF = RF.lu!(F, check = false) ### has to be LU since F will always be symmetric and positive semi-definite but not positive definite (due to linear dependencies)
        # end # timeit_debug

        if !ℒ.issuccess(luF)
            if verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood
        end

        Fdet = ℒ.det(luF)

        # Early return if determinant is too small, indicating numerical instability.
        if Fdet < eps(Float64)
            if verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood
        end

        # invF = inv(luF) ###

        # @timeit_debug timer "LU div" begin
        if t > presample_periods
            ℒ.ldiv!(ztmp, luF, z)
            loglik += log(Fdet) + ℒ.dot(z', ztmp) ###
            # loglik += log(Fdet) + z' * invF * z###
            # loglik += log(Fdet) + v' * invF * v###
        end

        # ℒ.mul!(Ktmp, P, C')
        # ℒ.mul!(K, Ktmp, invF)
        ℒ.mul!(K, P, C')
        ℒ.rdiv!(K, luF)
        # K = P * Ct / luF
        # K = P * C' * invF

        # end # timeit_debug
        # @timeit_debug timer "Matmul" begin

        ℒ.mul!(tmp, K, C)
        ℒ.mul!(Ptmp, tmp, P)
        ℒ.axpy!(-1, Ptmp, P)

        ℒ.mul!(Ptmp, A, P)
        ℒ.mul!(P, Ptmp, A')
        ℒ.axpy!(1, 𝐁, P)
        # P = A * (P - K * C * P) * A' + 𝐁

        ℒ.mul!(u, K, z, 1, 1)
        ℒ.mul!(utmp, A, u)
        u .= utmp
        # u = A * (u + K * v)

        ℒ.mul!(z, C, u)
        # z = C * u

        # end # timeit_debug
    end

    # end # timeit_debug
    # end # timeit_debug

    return -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 
end



end # dispatch_doctor

@stable default_mode = "disable" begin

function filter_data_with_model(𝓂::ℳ,
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
    @assert length(observables) <= 𝓂.timings.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."

    sort!(observables)

    solve!(𝓂, opts = opts)

    parameters = 𝓂.parameter_values

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, opts = opts)
    
    @assert solution_error < opts.tol.NSSS_acceptance_tol "Could not solve non-stochastic steady state." 

	∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)# |> Matrix

    sol, qme_sol, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings, opts = opts)

    if solved 𝓂.solution.perturbation.qme_solution = qme_sol end

    A = @views sol[:,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]

    B = @views sol[:,𝓂.timings.nPast_not_future_and_mixed+1:end]

    C = @views ℒ.diagm(ones(𝓂.timings.nVars))[sort(indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))),:]

    𝐁 = B * B'

    P̄ = calculate_covariance(𝓂.parameter_values, 𝓂, opts = opts)[1]

    n_obs = size(data_in_deviations,2)

    v = zeros(size(C,1), n_obs)
    μ = zeros(size(A,1), n_obs+1) # filtered_states
    P = zeros(size(A,1), size(A,1), n_obs+1) # filtered_covariances
    σ = zeros(size(A,1), n_obs) # filtered_standard_deviations
    iF= zeros(size(C,1), size(C,1), n_obs)
    L = zeros(size(A,1), size(A,1), n_obs)
    ϵ = zeros(size(B,2), n_obs) # filtered_shocks

    P[:, :, 1] = P̄

    # Kalman Filter
    for t in axes(data_in_deviations,2)
        v[:, t]     .= data_in_deviations[:, t] - C * μ[:, t]

        F̄ = ℒ.lu(C * P[:, :, t] * C', check = false)

        if !ℒ.issuccess(F̄) 
            @warn "Kalman filter stopped in period $t due to numerical stabiltiy issues."
            break
        end

        iF[:, :, t] .= inv(F̄)
        PCiF         = P[:, :, t] * C' * iF[:, :, t]
        L[:, :, t]  .= A - A * PCiF * C
        P[:, :, t+1].= A * P[:, :, t] * L[:, :, t]' + 𝐁
        σ[:, t]     .= sqrt.(abs.(ℒ.diag(P[:, :, t+1]))) # small numerical errors in this computation
        μ[:, t+1]   .= A * (μ[:, t] + PCiF * v[:, t])
        ϵ[:, t]     .= B' * C' * iF[:, :, t] * v[:, t]
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

end # dispatch_doctor