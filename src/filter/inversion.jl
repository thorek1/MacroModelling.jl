@stable default_mode = "disable" begin

# Specialization for :inversion filter
function calculate_loglikelihood(::Val{:inversion}, 
                                algorithm, observables, 
                                𝐒, 
                                data_in_deviations, 
                                TT, 
                                presample_periods, 
                                initial_covariance, 
                                state, 
                                warmup_iterations, 
                                filter_algorithm, 
                                opts,
                                on_failure_likelihood) #; 
                                # timer::TimerOutput = TimerOutput())
    return calculate_inversion_filter_loglikelihood(Val(algorithm), 
                                                    state, 
                                                    𝐒, 
                                                    data_in_deviations, 
                                                    observables, 
                                                    TT, 
                                                    warmup_iterations = warmup_iterations, 
                                                    presample_periods = presample_periods, 
                                                    filter_algorithm = filter_algorithm, 
                                                    # timer = timer, 
                                                    opts = opts,
                                                    on_failure_likelihood = on_failure_likelihood)
end


function calculate_inversion_filter_loglikelihood(::Val{:first_order},
                                                    state::Vector{Vector{R}}, 
                                                    𝐒::Matrix{R}, 
                                                    data_in_deviations::Matrix{R}, 
                                                    observables::Union{Vector{String}, Vector{Symbol}},
                                                    T::timings; 
                                                    # timer::TimerOutput = TimerOutput(),
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    on_failure_likelihood::U = -Inf,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)::R where {R <: AbstractFloat,U <: AbstractFloat}
    # @timeit_debug timer "Inversion filter" begin    
    # first order
    state = copy(state[1])

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0
    
    if warmup_iterations > 0
        if warmup_iterations >= 1
            jac = 𝐒[cond_var_idx,end-T.nExo+1:end]
            if warmup_iterations >= 2
                jac = hcat(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], jac)
                if warmup_iterations >= 3
                    Sᵉ = 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
                    for e in 1:warmup_iterations-2
                        jac = hcat(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * Sᵉ * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], jac)
                        Sᵉ *= 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
                    end
                end
            end
        end
    
        jacdecomp = ℒ.svd(jac)

        x = jacdecomp \ data_in_deviations[:,1]
    
        warmup_shocks = reshape(x, T.nExo, warmup_iterations)
    
        for i in 1:warmup_iterations-1
            ℒ.mul!(state, 𝐒, vcat(state[T.past_not_future_and_mixed_idx], warmup_shocks[:,i]))
            # state = state_update(state, warmup_shocks[:,i])
        end

        for i in 1:warmup_iterations
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jac[:,(i - 1) * T.nExo+1:i*T.nExo] ./ precision_factor)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jac[:,(i - 1) * T.nExo+1:i*T.nExo] ./ precision_factor))
            end
        end
    
        shocks² += sum(abs2,x)
    end

    y = zeros(length(cond_var_idx))
    x = zeros(T.nExo)
    jac = 𝐒[cond_var_idx,end-T.nExo+1:end]

    if T.nExo == length(observables)
        jacdecomp = ℒ.lu(jac, check = false)

        if !ℒ.issuccess(jacdecomp)
            if opts.verbose println("Inversion filter failed") end
            return on_failure_likelihood
        end

        logabsdets = ℒ.logabsdet(jac)[1]
        invjac = inv(jacdecomp)
    else
        jacdecomp = try ℒ.svd(jac)
        catch
            if opts.verbose println("Inversion filter failed") end
            return on_failure_likelihood
        end
        
        logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(jac))
        invjac = try inv(jacdecomp)
        catch
            if opts.verbose println("Inversion filter failed") end
            return on_failure_likelihood
        end
    end

    logabsdets *= size(data_in_deviations,2) - presample_periods
    
    if !isfinite(logabsdets) return on_failure_likelihood end

    𝐒obs = 𝐒[cond_var_idx,1:end-T.nExo]

    # @timeit_debug timer "Loop" begin    
    for i in axes(data_in_deviations,2)
        @views ℒ.mul!(y, 𝐒obs, state[T.past_not_future_and_mixed_idx])
        @views ℒ.axpby!(1, data_in_deviations[:,i], -1, y)
        ℒ.mul!(x, invjac, y)

        # x = invjac * (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])

        if i > presample_periods
            shocks² += sum(abs2,x)
            if !isfinite(shocks²) return on_failure_likelihood end
        end

        ℒ.mul!(state, 𝐒, vcat(state[T.past_not_future_and_mixed_idx], x))
        # state = 𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x)
    end

    # end # timeit_debug
    # end # timeit_debug

    return -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
    # return -(logabsdets + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end

end # dispatch_doctor

function rrule(::typeof(calculate_inversion_filter_loglikelihood), 
                ::Val{:first_order}, 
                state::Vector{Vector{Float64}}, 
                𝐒::Matrix{Float64}, 
                data_in_deviations::Matrix{Float64}, 
                observables::Union{Vector{String}, Vector{Symbol}}, 
                T::timings; 
                # timer::TimerOutput = TimerOutput(),
                warmup_iterations::Int = 0, 
                on_failure_likelihood = -Inf,
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
            return on_failure_likelihood, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        invjac = inv(jacdecomp)
    else
        logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(jac)) #' ./ precision_factor
        jacdecomp = ℒ.svd(jac)
        invjac = inv(jacdecomp)
    end

    logabsdets *= size(data_in_deviations,2) - presample_periods

    if !isfinite(logabsdets) 
        return on_failure_likelihood, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
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
                return on_failure_likelihood, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
            end
        end

        ℒ.mul!(state[i+1], 𝐒, vcat(state[i][t⁻], x[i]))
        # state[i+1] =  𝐒 * vcat(state[i][t⁻], x[i])
    end

    llh = -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
    
    if llh < -1e12
        return on_failure_likelihood, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
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


@stable default_mode = "disable" begin

function calculate_inversion_filter_loglikelihood(::Val{:pruned_second_order},
                                                    state::Vector{Vector{R}}, 
                                                    𝐒::Vector{AbstractMatrix{R}}, 
                                                    data_in_deviations::Matrix{R}, 
                                                    observables::Union{Vector{String}, Vector{Symbol}},
                                                    T::timings; 
                                                    # timer::TimerOutput = TimerOutput(),
                                                    warmup_iterations::Int = 0,
                                                    on_failure_likelihood::U = -Inf,
                                                    presample_periods::Int = 0,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)::R where {R <: AbstractFloat,U <: AbstractFloat}
    # @timeit_debug timer "Pruned 2nd - Inversion filter" begin
    # @timeit_debug timer "Preallocation" begin
             
    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = @ignore_derivatives indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    s_in_s⁺  = @ignore_derivatives BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = @ignore_derivatives BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺  = @ignore_derivatives BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))
    
    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind
    
    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind
    
    shockvar²_idxs = @ignore_derivatives setdiff(shock_idxs, shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind
    
    tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    var²_idxs = tmp.nzind
    
    𝐒⁻¹  = 𝐒[1][T.past_not_future_and_mixed_idx, :]
    𝐒¹⁻  = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ  = 𝐒[1][cond_var_idx, end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx, var_vol²_idxs]
    𝐒²⁻  = 𝐒[2][cond_var_idx, var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx, shockvar²_idxs]
    𝐒²ᵉ  = 𝐒[2][cond_var_idx, shock²_idxs]
    𝐒⁻²  = 𝐒[2][T.past_not_future_and_mixed_idx, :]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = nnz(𝐒²⁻)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    state₁ = state[1][T.past_not_future_and_mixed_idx]
    state₂ = state[2][T.past_not_future_and_mixed_idx]

    state¹⁻_vol = vcat(state₁, 1)

    aug_state₁ = [state₁; 1; ones(T.nExo)]
    aug_state₂ = [state₂; 0; zeros(T.nExo)]

    kronaug_state₁ = zeros(length(aug_state₁)^2)

    J = ℒ.I(T.nExo)

    kron_buffer = zeros(T.nExo^2)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    kron_buffer3 = ℒ.kron(J, zeros(T.nPast_not_future_and_mixed + 1))

    kronstate¹⁻_vol = zeros((T.nPast_not_future_and_mixed + 1)^2)

    shock_independent = zeros(size(data_in_deviations,1))

    𝐒ⁱ = copy(𝐒¹ᵉ)

    jacc = copy(𝐒¹ᵉ)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 
        
    init_guess = zeros(size(𝐒ⁱ, 2))

    # end # timeit_debug
    # @timeit_debug timer "Loop" begin

    for i in axes(data_in_deviations, 2)
        # state¹⁻ = state₁
        # state¹⁻_vol = vcat(state¹⁻, 1)
        # state²⁻ = state₂#[T.past_not_future_and_mixed_idx]

        copyto!(state¹⁻_vol, 1, state₁, 1)

        # shock_independent = data_in_deviations[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒¹⁻ * state²⁻ + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)
        copyto!(shock_independent, data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)

        ℒ.kron!(kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol, -1/2, 1)

        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)  
        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron_buffer3
        ℒ.kron!(kron_buffer3, J, state¹⁻_vol)

        ℒ.mul!(𝐒ⁱ, 𝐒²⁻ᵉ, kron_buffer3)

        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ)

        init_guess *= 0

        # @timeit_debug timer "Find shocks" begin
        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer2,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )
        # end # timeit_debug
                     
        # if matched println("$filter_algorithm: $matched; current x: $x") end      
        # if !matched
        #     x, matched = find_shocks(Val(:COBYLA), 
        #                             zeros(size(𝐒ⁱ, 2)),
        #                             kron_buffer,
        #                             kron_buffer2,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             shock_independent,
        #                             # max_iter = 500
        #                             )
            # println("COBYLA: $matched; current x: $x")
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer2,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             shock_independent)
                if !matched
                    if opts.verbose println("Inversion filter failed at step $i") end
                    return on_failure_likelihood # it can happen that there is no solution. think of a = bx + cx² where a is negative, b is zero and c is positive 
                end 
            # end
        # end

        # x2, mat = find_shocks(Val(:SLSQP), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
            
        # x3, mat2 = find_shocks(Val(:COBYLA), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # if mat
        #     println("SLSQP: $(ℒ.norm(x2-x) / max(ℒ.norm(x2), ℒ.norm(x)))")
        # elseif mat2
        #     println("COBYLA: $(ℒ.norm(x3-x) / max(ℒ.norm(x3), ℒ.norm(x)))")
        # end

        # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x)
        ℒ.kron!(kron_buffer2, J, x)

        ℒ.mul!(jacc, 𝐒ⁱ²ᵉ, kron_buffer2)

        ℒ.axpby!(1, 𝐒ⁱ, 2, jacc)

        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc))
            end

            shocks² += sum(abs2,x)
            
            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_likelihood
            end
        end

        # aug_state₁ = [state₁; 1; x]
        # aug_state₂ = [state₂; 0; zero(x)]
        copyto!(aug_state₁, 1, state₁, 1)
        copyto!(aug_state₁, length(state₁) + 2, x, 1)
        copyto!(aug_state₂, 1, state₂, 1)

        # state₁, state₂ = [𝐒⁻¹ * aug_state₁, 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * ℒ.kron(aug_state₁, aug_state₁) / 2] # strictly following Andreasen et al. (2018)
        ℒ.mul!(state₁, 𝐒⁻¹, aug_state₁)

        ℒ.mul!(state₂, 𝐒⁻¹, aug_state₂)
        ℒ.kron!(kronaug_state₁, aug_state₁, aug_state₁)
        ℒ.mul!(state₂, 𝐒⁻², kronaug_state₁, 1/2, 1)
    end

    # end # timeit_debug
    # end # timeit_debug

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf and Fair and Taylor (1983)
    return -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end

end # dispatch_doctor

function rrule(::typeof(calculate_inversion_filter_loglikelihood),
                ::Val{:pruned_second_order},
                state::Vector{Vector{Float64}}, 
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                observables::Union{Vector{String}, Vector{Symbol}},
                T::timings; 
                # timer::TimerOutput = TimerOutput(),
                on_failure_likelihood = -Inf,
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
            return on_failure_likelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
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
                        return on_failure_likelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end

        try
            ℒ.ldiv!(λ[i], jacc_fact, x[i])
        catch
            if opts.verbose println("Inversion filter failed at step $i") end
            return on_failure_likelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
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
                return on_failure_likelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
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
                            inv(ℒ.svd(jacc[i]))'
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

@stable default_mode = "disable" begin


function calculate_inversion_filter_loglikelihood(::Val{:second_order},
                                                    state::Vector{R}, 
                                                    𝐒::Vector{AbstractMatrix{R}}, 
                                                    data_in_deviations::Matrix{R}, 
                                                    observables::Union{Vector{String}, Vector{Symbol}},
                                                    T::timings; 
                                                    # timer::TimerOutput = TimerOutput(),
                                                    on_failure_likelihood::U = -Inf,
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)::R where {R <: AbstractFloat, U <: AbstractFloat}
    # @timeit_debug timer "2nd - Inversion filter" begin
    # @timeit_debug timer "Preallocation" begin

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    # s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))
    
    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind
    
    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind
    
    shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind
    
    # tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    # var²_idxs = tmp.nzind
    
    𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    # 𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    # 𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    # 𝐒²⁻     = length(𝐒²⁻.nzval)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    state = state[T.past_not_future_and_mixed_idx]

    state¹⁻_vol = vcat(state, 1)

    aug_state = [zeros(T.nPast_not_future_and_mixed); 1; zeros(T.nExo)]

    kronaug_state = zeros((T.nPast_not_future_and_mixed + 1 + T.nExo)^2)

    kron_buffer = zeros(T.nExo^2)

    J = ℒ.I(T.nExo)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    kron_buffer3 = ℒ.kron(J, zeros(T.nPast_not_future_and_mixed + 1))

    shock_independent = zeros(size(data_in_deviations,1))

    kronstate¹⁻_vol = zeros((T.nPast_not_future_and_mixed + 1)^2)

    𝐒ⁱ = copy(𝐒¹ᵉ)

    jacc = copy(𝐒¹ᵉ)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

    init_guess = zeros(size(𝐒ⁱ, 2))

    # end # timeit_debug
    # @timeit_debug timer "Loop" begin

    for i in axes(data_in_deviations,2)
        # state¹⁻ = state#[T.past_not_future_and_mixed_idx]
        # state¹⁻_vol = vcat(state¹⁻, 1)
        
        copyto!(state¹⁻_vol, 1, state, 1)

        copyto!(shock_independent, data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)

        ℒ.kron!(kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol, -1/2, 1)
        # shock_independent = data_in_deviations[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)
        ℒ.kron!(kron_buffer3, J, state¹⁻_vol)

        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron_buffer3
        ℒ.mul!(𝐒ⁱ, 𝐒²⁻ᵉ, kron_buffer3)

        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ)

        init_guess *= 0

        # @timeit_debug timer "Find shocks" begin
        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer2,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )  
        # end # timeit_debug

        # if !matched
        #     x, matched = find_shocks(Val(:COBYLA), 
        #                             zeros(size(𝐒ⁱ, 2)),
        #                             kron_buffer,
        #                             kron_buffer2,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             shock_independent,
        #                             # max_iter = 500
        #                             )
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer2,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             shock_independent)
                if !matched
                    if opts.verbose println("Inversion filter failed at step $i") end
                    return on_failure_likelihood # it can happen that there is no solution. think of a = bx + cx² where a is negative, b is zero and c is positive 
                end 
            # end
        # end

        # x2, mat = find_shocks(Val(:SLSQP), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
            
        # x3, mat2 = find_shocks(Val(:COBYLA), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # if mat
        #     println("SLSQP: $(ℒ.norm(x2-x) / max(ℒ.norm(x2), ℒ.norm(x)))")
        # elseif mat2
        #     println("COBYLA: $(ℒ.norm(x3-x) / max(ℒ.norm(x3), ℒ.norm(x)))")
        # end

        # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x)
        ℒ.kron!(kron_buffer2, J, x)

        ℒ.mul!(jacc, 𝐒ⁱ²ᵉ, kron_buffer2)

        ℒ.axpby!(1, 𝐒ⁱ, 2, jacc)

        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc)[1] # ./ precision_factor
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc)) # ./ precision_factor
            end

            shocks² += sum(abs2,x)

            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_likelihood
            end
        end

        # aug_state = [state; 1; x]
        # aug_state[1:T.nPast_not_future_and_mixed] = state
        # aug_state[end-T.nExo+1:end] = x
        copyto!(aug_state, 1, state, 1)
        copyto!(aug_state, length(state) + 2, x, 1)

        # res = 𝐒[1][cond_var_idx, :] * aug_state + 𝐒[2][cond_var_idx, :] * ℒ.kron(aug_state, aug_state) / 2 - data_in_deviations[:,i]
        # println("Match with data: $res")

        # state = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
        ℒ.kron!(kronaug_state, aug_state, aug_state)
        ℒ.mul!(state, 𝐒⁻¹, aug_state)
        ℒ.mul!(state, 𝐒⁻², kronaug_state, 1/2 ,1)
    end

    # end # timeit_debug
    # end # timeit_debug

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    return -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end

end # dispatch_doctor

function rrule(::typeof(calculate_inversion_filter_loglikelihood),
                ::Val{:second_order},
                state::Vector{Float64}, 
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                observables::Union{Vector{String}, Vector{Symbol}},
                T::timings; 
                # timer::TimerOutput = TimerOutput(),
                on_failure_likelihood = -Inf,
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
            return on_failure_likelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
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
                        return on_failure_likelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
                    end

        try
            ℒ.ldiv!(λ[i], jacc_fact, x[i])
        catch
            if opts.verbose println("Inversion filter failed at step $i") end
            return on_failure_likelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
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
                return on_failure_likelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
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
                            inv(ℒ.svd(jacc[i]))'
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


@stable default_mode = "disable" begin

function calculate_inversion_filter_loglikelihood(::Val{:pruned_third_order},
                                                    state::Vector{Vector{R}}, 
                                                    𝐒::Vector{AbstractMatrix{R}}, 
                                                    data_in_deviations::Matrix{R}, 
                                                    observables::Union{Vector{String}, Vector{Symbol}},
                                                    T::timings;
                                                    # timer::TimerOutput = TimerOutput(), 
                                                    on_failure_likelihood::U = -Inf,
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)::R where {R <: AbstractFloat, U <: AbstractFloat}
    # @timeit_debug timer "Inversion filter" begin

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = @ignore_derivatives indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    s_in_s⁺ = @ignore_derivatives BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = @ignore_derivatives BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺ = @ignore_derivatives BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))

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
    𝐒³⁻ᵉ² = 𝐒[3][cond_var_idx,shockvar³2_idxs] |> collect
    𝐒³⁻ᵉ = 𝐒[3][cond_var_idx,shockvar³_idxs]
    𝐒³ᵉ  = 𝐒[3][cond_var_idx,shock³_idxs]
    𝐒⁻³  = 𝐒[3][T.past_not_future_and_mixed_idx,:]

    𝐒³⁻ᵛ    = nnz(𝐒³⁻ᵛ)    / length(𝐒³⁻ᵛ)  > .1 ? collect(𝐒³⁻ᵛ)    : 𝐒³⁻ᵛ
    𝐒³⁻ᵉ    = nnz(𝐒³⁻ᵉ)    / length(𝐒³⁻ᵉ)  > .1 ? collect(𝐒³⁻ᵉ)    : 𝐒³⁻ᵉ
    𝐒³ᵉ     = nnz(𝐒³ᵉ)     / length(𝐒³ᵉ)   > .1 ? collect(𝐒³ᵉ)     : 𝐒³ᵉ
    𝐒⁻³     = nnz(𝐒⁻³)     / length(𝐒⁻³)   > .1 ? collect(𝐒⁻³)     : 𝐒⁻³

    state[1] = state[1][T.past_not_future_and_mixed_idx]
    state[2] = state[2][T.past_not_future_and_mixed_idx]
    state[3] = state[3][T.past_not_future_and_mixed_idx]

    𝐒ⁱ = copy(𝐒¹ᵉ)

    jacc = copy(𝐒¹ᵉ)

    kron_buffer = zeros(T.nExo^2)

    kron_buffer² = zeros(T.nExo^3)

    II = ℒ.I(T.nExo^2)
    
    J = ℒ.I(T.nExo)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    kron_buffer3 = ℒ.kron(J, kron_buffer)
    
    kron_buffer4 = ℒ.kron(II, zeros(T.nExo))

    kron_buffer4sv = ℒ.kron(II, vcat(1,state[1]))

    kron_buffer2s = ℒ.kron(J, state[1])

    kron_buffer2sv = ℒ.kron(J, vcat(1,state[1]))

    kron_buffer2ss = ℒ.kron(state[1], state[1])

    kron_buffer2svsv = ℒ.kron(vcat(1,state[1]), vcat(1,state[1]))

    kron_buffer3svsv = ℒ.kron(kron_buffer2svsv, vcat(1,state[1]))

    kron_buffer3sv = ℒ.kron(kron_buffer2sv, vcat(1,state[1]))
    
    kron_aug_state₁ = zeros((T.nPast_not_future_and_mixed + 1 + T.nExo)^2)
    
    kron_kron_aug_state₁ = zeros((T.nPast_not_future_and_mixed + 1 + T.nExo)^3)

    state¹⁻ = state[1]

    state²⁻ = state[2]#[T.past_not_future_and_mixed_idx]

    state³⁻ = state[3]#[T.past_not_future_and_mixed_idx]

    # @timeit_debug timer "Loop" begin

    𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

    init_guess = zeros(size(𝐒ⁱ, 2))
    
    for i in axes(data_in_deviations,2)
        state¹⁻_vol = vcat(state¹⁻, 1)

        shock_independent = copy(data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒¹⁻, state²⁻, -1, 1)

        ℒ.mul!(shock_independent, 𝐒¹⁻, state³⁻, -1, 1)

        ℒ.kron!(kron_buffer2svsv, state¹⁻_vol, state¹⁻_vol)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_buffer2svsv, -1/2, 1)

        ℒ.kron!(kron_buffer2ss, state¹⁻, state²⁻)

        ℒ.mul!(shock_independent, 𝐒²⁻, kron_buffer2ss, -1, 1)

        ℒ.kron!(kron_buffer3svsv, kron_buffer2svsv, state¹⁻_vol)

        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, kron_buffer3svsv, -1/6, 1)   
        
        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(J, state¹⁻_vol) + 𝐒²⁻ᵛᵉ * ℒ.kron(J, state²⁻) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(J, state¹⁻_vol), state¹⁻_vol) / 2
        
        ℒ.kron!(kron_buffer2s, J, state²⁻)
    
        ℒ.mul!(𝐒ⁱ, 𝐒²⁻ᵛᵉ, kron_buffer2s)

        ℒ.kron!(kron_buffer2sv, J, state¹⁻_vol)
    
        ℒ.mul!(𝐒ⁱ, 𝐒²⁻ᵉ, kron_buffer2sv, 1, 1)

        ℒ.kron!(kron_buffer2sv, J, state¹⁻_vol)
    
        ℒ.kron!(kron_buffer3sv, kron_buffer2sv, state¹⁻_vol)
        
        ℒ.mul!(𝐒ⁱ, 𝐒³⁻ᵉ², kron_buffer3sv, 1/2, 1)

        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ)

        x_kron_II!(kron_buffer4sv, state¹⁻_vol)

        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * kron_buffer4sv / 2

        # x, jacc, matchd = find_shocks(Val(:fixed_point), state isa Vector{Float64} ? [state] : state, 𝐒, data_in_deviations[:,i], observables, T)

        init_guess *= 0

        # x² , matched = find_shocks(Val(filter_algorithm), 
        #                         init_guess,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 200
        #                         )
        #                         println(x²)

        # @timeit_debug timer "Find shocks" begin
        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
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
                                # max_iter = 200
                                )
        # end # timeit_debug
                                
                                # println(x)
        # println("$filter_algorithm: $matched; current x: $x, $(ℒ.norm(x))")
        # if !matched

        # backup_solver = :COBYLA

        # if filter_algorithm ≠ backup_solver
        #     x̂, matched2 = find_shocks(Val(backup_solver), 
        #                         zeros(size(𝐒ⁱ, 2)),
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 5000
        #                         )
        #     if ℒ.norm(x̂) * (1 - eps(Float32)) < ℒ.norm(x)
        #         x̄, matched3 = find_shocks(Val(filter_algorithm), 
        #                             x̂,
        #                             kron_buffer,
        #                             kron_buffer²,
        #                             kron_buffer2,
        #                             kron_buffer3,
        #                             kron_buffer4,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             𝐒ⁱ³ᵉ,
        #                             shock_independent,
        #                             # max_iter = 200
        #                             )
                              
        #         if matched3 && (!matched || ℒ.norm(x̄) * (1 - eps(Float32)) < ℒ.norm(x̂) || (matched && ℒ.norm(x̄) * (1 - eps(Float32)) < ℒ.norm(x)))
        #             # println("$i - $filter_algorithm restart - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̄
        #             matched = matched3
        #         elseif matched2
        #             # println("$i - $backup_solver - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̂
        #             matched = matched2
        #         # else
        #         #     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

        #         #     norm1 = ℒ.norm(y)

        #         #     norm2 = ℒ.norm(shock_independent)

        #             # println("$i - $filter_algorithm - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")#, residual norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
        #         end
        #     # else
        #     #     println("$i - $filter_algorithm ($matched) - $(ℒ.norm(x)), $backup_solver ($matched2) - $(ℒ.norm(x̂))")
        #     end
        # end

        if !matched
            if opts.verbose println("Inversion filter failed at step $i") end
            return on_failure_likelihood # it can happen that there is no solution. think of a = bx + cx² where a is negative, b is zero and c is positive 
        end 
            # println("COBYLA: $matched; current x: $x")
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer²,
            #                             kron_buffer2,
            #                             kron_buffer3,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             𝐒ⁱ³ᵉ,
            #                             shock_independent)
                # println("$filter_algorithm: $matched; current x: $x")
                # if !matched
                #     x, matched = find_shocks(Val(:COBYLA), 
                #                             x,
                #                             kron_buffer,
                #                             kron_buffer²,
                #                             kron_buffer2,
                #                             kron_buffer3,
                #                             J,
                #                             𝐒ⁱ,
                #                             𝐒ⁱ²ᵉ,
                #                             𝐒ⁱ³ᵉ,
                #                             shock_independent)
                # end
            # end
        # end

        # x2, mat = find_shocks(Val(:SLSQP), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
            
        # x3, mat2 = find_shocks(Val(:COBYLA), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # if mat
        #     println("SLSQP: $(ℒ.norm(x2-x) / max(ℒ.norm(x2), ℒ.norm(x))), $(ℒ.norm(x2)-ℒ.norm(x))")
        # elseif mat2
        #     println("COBYLA: $(ℒ.norm(x3-x) / max(ℒ.norm(x3), ℒ.norm(x))), $(ℒ.norm(x3)-ℒ.norm(x))")
        # end
        
        ℒ.kron!(kron_buffer2, J, x)

        ℒ.kron!(kron_buffer3, kron_buffer2, x)

        ℒ.mul!(jacc, 𝐒ⁱ²ᵉ, kron_buffer2)

        ℒ.mul!(jacc, 𝐒ⁱ³ᵉ, kron_buffer3, 3, 2)

        ℒ.axpby!(-1, 𝐒ⁱ, -1, jacc)

        # jacc = -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * kron_buffer2 + 3 * 𝐒ⁱ³ᵉ * kron_buffer3)

        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc))
            end

            shocks² += sum(abs2,x)
            
            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_likelihood
            end
        end

        aug_state₁ = [state¹⁻; 1; x]
        aug_state₁̂ = [state¹⁻; 0; x]
        aug_state₂ = [state²⁻; 0; zero(x)]
        aug_state₃ = [state³⁻; 0; zero(x)]
        
        # kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)
        ℒ.kron!(kron_aug_state₁, aug_state₁, aug_state₁)

        ℒ.kron!(kron_kron_aug_state₁, kron_aug_state₁, aug_state₁)
        # res = 𝐒[1][cond_var_idx,:] * aug_state₁   +   𝐒[1][cond_var_idx,:] * aug_state₂ + 𝐒[2][cond_var_idx,:] * kron_aug_state₁ / 2   +   𝐒[1][cond_var_idx,:] * aug_state₃ + 𝐒[2][cond_var_idx,:] * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒[3][cond_var_idx,:] * ℒ.kron(kron_aug_state₁,aug_state₁) / 6 - data_in_deviations[:,i]
        # println("Match with data: $res")
        
        # println(ℒ.norm(x))

        # state[1] = 𝐒⁻¹ * aug_state₁
        # state[2] = 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * kron_aug_state₁ / 2
        # state[3] = 𝐒⁻¹ * aug_state₃ + 𝐒⁻² * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒⁻³ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6
        
        ℒ.mul!(state¹⁻, 𝐒⁻¹, aug_state₁)

        ℒ.mul!(state²⁻, 𝐒⁻¹, aug_state₂)
        ℒ.mul!(state²⁻, 𝐒⁻², kron_aug_state₁, 1/2, 1)

        ℒ.mul!(state³⁻, 𝐒⁻¹, aug_state₃)

        ℒ.kron!(kron_aug_state₁, aug_state₁̂, aug_state₂)
        
        ℒ.mul!(state³⁻, 𝐒⁻², kron_aug_state₁, 1, 1)
        ℒ.mul!(state³⁻, 𝐒⁻³, kron_kron_aug_state₁, 1/6, 1)
    end

    # end # timeit_debug
    # end # timeit_debug

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    return -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end

end # dispatch_doctor


function rrule(::typeof(calculate_inversion_filter_loglikelihood),
                ::Val{:pruned_third_order},
                state::Vector{Vector{Float64}}, 
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                observables::Union{Vector{String}, Vector{Symbol}},
                T::timings; 
                # timer::TimerOutput = TimerOutput(),
                on_failure_likelihood = -Inf,
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
            return on_failure_likelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
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
                return on_failure_likelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
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
                            inv(ℒ.svd(jacc[i]))'
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


@stable default_mode = "disable" begin


function calculate_inversion_filter_loglikelihood(::Val{:third_order},
                                                    state::Vector{R}, 
                                                    𝐒::Vector{AbstractMatrix{R}}, 
                                                    data_in_deviations::Matrix{R}, 
                                                    observables::Union{Vector{String}, Vector{Symbol}},
                                                    T::timings; 
                                                    # timer::TimerOutput = TimerOutput(),
                                                    on_failure_likelihood::U = -Inf,
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)::R where {R <: AbstractFloat,U <: AbstractFloat}
    # @timeit_debug timer "3rd - Inversion filter" begin
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
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = nnz(𝐒²⁻)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    state = state[T.past_not_future_and_mixed_idx]

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

    kron_buffer = zeros(T.nExo^2)

    kron_buffer² = zeros(T.nExo^3)

    J = ℒ.I(T.nExo)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    kron_buffer3 = ℒ.kron(J, kron_buffer)

    kron_buffer4 = ℒ.kron(ℒ.kron(J, J), zeros(T.nExo))

    II = sparse(ℒ.I(T.nExo^2))

    # end # timeit_debug
    # @timeit_debug timer "Loop" begin
    
    for i in axes(data_in_deviations,2)
        state¹⁻ = state

        state¹⁻_vol = vcat(state¹⁻, 1)
        
        shock_independent = copy(data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
        
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
    
        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

        𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

        # x, jacc, matchd = find_shocks(Val(:fixed_point), state isa Vector{Float64} ? [state] : state, 𝐒, data_in_deviations[:,i], observables, T)

        init_guess = zeros(size(𝐒ⁱ, 2))

        # @timeit_debug timer "Find shocks" begin
        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
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
                                # max_iter = 200
                                )
        # end # timeit_debug
                                
        # println("$filter_algorithm: $matched; current x: $x, $(ℒ.norm(x))")
        # if !matched

        # backup_solver = :COBYLA

        # if filter_algorithm ≠ backup_solver
        #     x̂, matched2 = find_shocks(Val(backup_solver), 
        #                         zeros(size(𝐒ⁱ, 2)),
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 5000
        #                         )
        #     if ℒ.norm(x̂) * (1 - eps(Float32)) < ℒ.norm(x)
        #         x̄, matched3 = find_shocks(Val(filter_algorithm), 
        #                             x̂,
        #                             kron_buffer,
        #                             kron_buffer²,
        #                             kron_buffer2,
        #                             kron_buffer3,
        #                             kron_buffer4,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             𝐒ⁱ³ᵉ,
        #                             shock_independent,
        #                             # max_iter = 200
        #                             )
                              
        #         if matched3 && ℒ.norm(x̄) * (1 - eps(Float32)) < ℒ.norm(x̂)
        #             println("$i - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̄
        #             matched = matched3
        #         elseif matched2
        #             println("$i - $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̂
        #             matched = matched2
        #         else
        #             y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

        #             norm1 = ℒ.norm(y)

        #             norm2 = ℒ.norm(shock_independent)

        #             println("$i - $filter_algorithm ($matched) - $(ℒ.norm(x)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), residual norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
        #         end
        #     else
        #         println("$i - $filter_algorithm ($matched) - $(ℒ.norm(x)), $backup_solver ($matched2) - $(ℒ.norm(x̂))")
        #     end
        # end

        if !matched
            if opts.verbose println("Inversion filter failed at step $i") end
            return on_failure_likelihood # it can happen that there is no solution. think of a = bx + cx² where a is negative, b is zero and c is positive 
        end 
            # println("COBYLA: $matched; current x: $x")
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer²,
            #                             kron_buffer2,
            #                             kron_buffer3,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             𝐒ⁱ³ᵉ,
            #                             shock_independent)
            #     println("$filter_algorithm: $matched; current x: $x")
            #     if !matched
            #         x, matched = find_shocks(Val(:COBYLA), 
            #                                 x,
            #                                 kron_buffer,
            #                                 kron_buffer²,
            #                                 kron_buffer2,
            #                                 kron_buffer3,
            #                                 J,
            #                                 𝐒ⁱ,
            #                                 𝐒ⁱ²ᵉ,
            #                                 𝐒ⁱ³ᵉ,
            #                                 shock_independent)
            #         println("COBYLA: $matched; current x: $x")
            #     end
            # end
        # end

        # x2, mat = find_shocks(Val(:COBYLA), 
        #                         init_guess,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 200
        #                         )
            
        # x3, mat2 = find_shocks(Val(filter_algorithm), 
        #                         x2,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # # if mat
        #     println("COBYLA - $mat: $x2, $(ℒ.norm(x2))")
        # # end
        # # if mat2
        #     println("LagrangeNewton restart - $mat2: $x3, $(ℒ.norm(x3))")
        # # end

        jacc = -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x, x)))
    
        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc))
            end

            shocks² += sum(abs2,x)
            
            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_likelihood
            end
        end

        aug_state = [state; 1; x]

        # res = 𝐒[1][cond_var_idx, :] * aug_state + 𝐒[2][cond_var_idx, :] * ℒ.kron(aug_state, aug_state) / 2 + 𝐒[3][cond_var_idx, :] * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6 - data_in_deviations[:,i]
        # println("Match with data: $res")

        state = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
        # state = state_update(state, x)
    end

    # end # timeit_debug
    # end # timeit_debug

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    return -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end


end # dispatch_doctor

function rrule(::typeof(calculate_inversion_filter_loglikelihood),
                ::Val{:third_order},
                state::Vector{Float64}, 
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                observables::Union{Vector{String}, Vector{Symbol}},
                T::timings; 
                # timer::TimerOutput = TimerOutput(),
                on_failure_likelihood = -Inf,
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
            return on_failure_likelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
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
                return on_failure_likelihood, x -> NoTangent(), NoTangent(),  NoTangent(), NoTangent(), NoTangent(), NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
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
                            inv(ℒ.svd(jacc[i]))'
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


@stable default_mode = "disable" begin

function filter_data_with_model(𝓂::ℳ,
                                data_in_deviations::KeyedArray{Float64},
                                ::Val{:first_order}, # algo
                                ::Val{:inversion}; # filter
                                warmup_iterations::Int = 0,
                                smooth::Bool = true,
                                opts::CalculationOptions = merge_calculation_options())
    T = 𝓂.timings

    variables = zeros(T.nVars, size(data_in_deviations,2))
    shocks = zeros(T.nExo, size(data_in_deviations,2))
    
    decomposition = zeros(T.nVars, T.nExo + 2, size(data_in_deviations, 2))

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)

    if solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error)
        @error "No solution for these parameters."
        return variables, shocks, zeros(0,0), decomposition
    end

    state = zeros(T.nVars)

    initial_state = zeros(T.nVars)

    ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)# |> Matrix

    𝐒₁, qme_sol, solved = calculate_first_order_solution(∇₁; 
                                                        T = T, 
                                                        initial_guess = 𝓂.solution.perturbation.qme_solution, 
                                                        opts = opts)
    
    if solved 𝓂.solution.perturbation.qme_solution = qme_sol end

    if !solved 
        @error "No solution for these parameters."
        return variables, shocks, zeros(0,0), decomposition
    end

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    observables = get_and_check_observables(𝓂, data_in_deviations)

    cond_var_idx = indexin(observables, sort(union(T.aux,T.var,T.exo_present)))

    if warmup_iterations > 0
        if warmup_iterations >= 1
            jac = 𝐒₁[cond_var_idx,end-T.nExo+1:end]
            if warmup_iterations >= 2
                jac = hcat(𝐒₁[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒₁[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], jac)
                if warmup_iterations >= 3
                    Sᵉ = 𝐒₁[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
                    for e in 1:warmup_iterations-2
                        jac = hcat(𝐒₁[cond_var_idx,1:T.nPast_not_future_and_mixed] * Sᵉ * 𝐒₁[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], jac)
                        Sᵉ *= 𝐒₁[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
                    end
                end
            end
        end
    
        jacdecomp = ℒ.svd(jac)

        x = jacdecomp \ data_in_deviations[:,1]
    
        warmup_shocks = reshape(x, T.nExo, warmup_iterations)
    
        for i in 1:warmup_iterations-1
            ℒ.mul!(state, 𝐒₁, vcat(state[T.past_not_future_and_mixed_idx], warmup_shocks[:,i]))
            # state = state_update(state, warmup_shocks[:,i])
        end
    end

    y = zeros(length(cond_var_idx))
    x = zeros(T.nExo)

    jac = 𝐒₁[cond_var_idx, end-T.nExo+1:end]

    if T.nExo == length(observables)
        jacdecomp = ℒ.lu(jac, check = false)

        if !ℒ.issuccess(jacdecomp)
            @error "Inversion filter failed"
            return variables, shocks, zeros(0,0), decomposition
        end

        invjac = inv(jacdecomp)
    else
        # jacdecomp = ℒ.svd(jac)
        
        invjac = ℒ.pinv(jac)
    end

    for i in axes(data_in_deviations,2)
        @views ℒ.mul!(y, 𝐒₁[cond_var_idx,1:end-T.nExo], state[T.past_not_future_and_mixed_idx])
        @views ℒ.axpby!(1, data_in_deviations[:,i], -1, y)

        ℒ.mul!(x, invjac, y)

        ℒ.mul!(state, 𝐒₁, vcat(state[T.past_not_future_and_mixed_idx], x))

        shocks[:,i] .= x
        variables[:,i] .= state
        # state = 𝐒₁ * vcat(state[T.past_not_future_and_mixed_idx], x)
    end

    decomposition[:,end,:] .= variables

    for i in 1:T.nExo
        sck = zeros(T.nExo)
        sck[i] = shocks[i, 1]
        decomposition[:,i,1] .= 𝐒₁ * vcat(initial_state[T.past_not_future_and_mixed_idx], sck) # state_update(initial_state , sck)
    end

    decomposition[:, end - 1, 1] .= decomposition[:, end, 1] - sum(decomposition[:, 1:end-2, 1], dims=2)

    for i in 2:size(data_in_deviations,2)
        for ii in 1:T.nExo
            sck = zeros(T.nExo)
            sck[ii] = shocks[ii, i]
            decomposition[:, ii, i] .= 𝐒₁ * vcat(decomposition[T.past_not_future_and_mixed_idx, ii, i-1], sck) # state_update(decomposition[:,ii, i-1], sck)
        end

        decomposition[:, end - 1, i] .= decomposition[:, end, i] - sum(decomposition[:, 1:end-2, i], dims=2)
    end
    
    return variables, shocks, zeros(0,0), decomposition
end


function filter_data_with_model(𝓂::ℳ,
                                data_in_deviations::KeyedArray{Float64},
                                ::Val{:second_order}, # algo
                                ::Val{:inversion}; # filter
                                warmup_iterations::Int = 0,
                                filter_algorithm::Symbol = :LagrangeNewton,
                                smooth::Bool = true,
                                opts::CalculationOptions = merge_calculation_options())

    T = 𝓂.timings

    variables = zeros(T.nVars, size(data_in_deviations,2))
    shocks = zeros(T.nExo, size(data_in_deviations,2))

    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_second_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, opts = opts)

    if !converged || solution_error > opts.tol.NSSS_acceptance_tol
        @error "Could not find 2nd order stochastic steady state"
        return variables, shocks, zeros(0,0), zeros(0,0)
    end

    all_SS = expand_steady_state(SS_and_pars,𝓂)

    full_state = collect(sss) - all_SS

    observables = get_and_check_observables(𝓂, data_in_deviations)

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    # s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))
    
    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind
    
    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind
    
    shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind
    
    # tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    # var²_idxs = tmp.nzind
    
    𝐒⁻¹ = 𝐒₁[T.past_not_future_and_mixed_idx,:]
    # 𝐒¹⁻ = 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒₁[cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒₂[cond_var_idx,var_vol²_idxs]
    # 𝐒²⁻ = 𝐒₂[cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒₂[cond_var_idx,shockvar²_idxs]
    𝐒²ᵉ = 𝐒₂[cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒₂[T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    # 𝐒²⁻     = length(𝐒²⁻.nzval)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    state = full_state[T.past_not_future_and_mixed_idx]

    state¹⁻_vol = vcat(state, 1)

    aug_state = [zeros(T.nPast_not_future_and_mixed); 1; zeros(T.nExo)]

    kronaug_state = zeros((T.nPast_not_future_and_mixed + 1 + T.nExo)^2)

    kron_buffer = zeros(T.nExo^2)

    J = ℒ.I(T.nExo)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    kron_buffer3 = ℒ.kron(J, zeros(T.nPast_not_future_and_mixed + 1))

    shock_independent = zeros(size(data_in_deviations,1))

    kronstate¹⁻_vol = zeros((T.nPast_not_future_and_mixed + 1)^2)

    𝐒ⁱ = copy(𝐒¹ᵉ)

    jacc = copy(𝐒¹ᵉ)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

    init_guess = zeros(size(𝐒ⁱ, 2))

    for i in axes(data_in_deviations,2)
        # state¹⁻ = state#[T.past_not_future_and_mixed_idx]
        # state¹⁻_vol = vcat(state¹⁻, 1)
        
        copyto!(state¹⁻_vol, 1, state, 1)

        copyto!(shock_independent, data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)

        ℒ.kron!(kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol, -1/2, 1)
        # shock_independent = data_in_deviations[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)
        ℒ.kron!(kron_buffer3, J, state¹⁻_vol)

        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron_buffer3
        ℒ.mul!(𝐒ⁱ, 𝐒²⁻ᵉ, kron_buffer3)

        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ)

        init_guess *= 0

        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer2,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )

        # if !matched
        #     x, matched = find_shocks(Val(:COBYLA), 
        #                             zeros(size(𝐒ⁱ, 2)),
        #                             kron_buffer,
        #                             kron_buffer2,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             shock_independent,
        #                             # max_iter = 500
        #                             )
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer2,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             shock_independent)
                if !matched
                    @error "Inversion filter failed at step $i"
                    return variables, shocks, zeros(0,0), zeros(0,0)
                end 
            # end
        # end

        # x2, mat = find_shocks(Val(:SLSQP), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
            
        # x3, mat2 = find_shocks(Val(:COBYLA), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # if mat
        #     println("SLSQP: $(ℒ.norm(x2-x) / max(ℒ.norm(x2), ℒ.norm(x)))")
        # elseif mat2
        #     println("COBYLA: $(ℒ.norm(x3-x) / max(ℒ.norm(x3), ℒ.norm(x)))")
        # end

        # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x)
        ℒ.kron!(kron_buffer2, J, x)

        ℒ.mul!(jacc, 𝐒ⁱ²ᵉ, kron_buffer2)

        ℒ.axpby!(1, 𝐒ⁱ, 2, jacc)

        # aug_state = [state; 1; x]
        # aug_state[1:T.nPast_not_future_and_mixed] = state
        # aug_state[end-T.nExo+1:end] = x
        copyto!(aug_state, 1, state, 1)
        copyto!(aug_state, length(state) + 2, x, 1)

        # res = 𝐒[1][cond_var_idx, :] * aug_state + 𝐒[2][cond_var_idx, :] * ℒ.kron(aug_state, aug_state) / 2 - data_in_deviations[:,i]
        # println("Match with data: $res")

        # state = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
        ℒ.kron!(kronaug_state, aug_state, aug_state)
        ℒ.mul!(full_state, 𝐒₁, aug_state)
        ℒ.mul!(full_state, 𝐒₂, kronaug_state, 1/2 ,1)

        shocks[:,i] .= x
        variables[:,i] .= full_state

        state .= full_state[T.past_not_future_and_mixed_idx]
    end

    return variables, shocks, zeros(0,0), zeros(0,0)
end


function filter_data_with_model(𝓂::ℳ,
                                data_in_deviations::KeyedArray{Float64},
                                ::Val{:pruned_second_order}, # algo
                                ::Val{:inversion}; # filter
                                warmup_iterations::Int = 0,
                                filter_algorithm::Symbol = :LagrangeNewton,
                                smooth::Bool = true,
                                opts::CalculationOptions = merge_calculation_options())
    T = 𝓂.timings

    variables = zeros(T.nVars, size(data_in_deviations,2))
    shocks = zeros(T.nExo, size(data_in_deviations,2))
    decomposition = zeros(𝓂.timings.nVars, 𝓂.timings.nExo + 3, size(data_in_deviations, 2))

    observables = get_and_check_observables(𝓂, data_in_deviations)
    
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_second_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, pruning = true, opts = opts)

    if solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error)
        @error "No solution for these parameters."
        return variables, shocks, zeros(0,0), decomposition
    end

    if !converged
        @error "No solution for these parameters."
        return variables, shocks, zeros(0,0), decomposition
    end

    𝐒 = [𝐒₁, 𝐒₂]

    all_SS = expand_steady_state(SS_and_pars,𝓂)

    state = [zeros(𝓂.timings.nVars), collect(sss) - all_SS]
     
    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    s_in_s⁺  = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺  = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))
    
    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind
    
    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind
    
    shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind
    
    tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    var²_idxs = tmp.nzind
    
    𝐒⁻¹  = 𝐒[1][T.past_not_future_and_mixed_idx, :]
    𝐒¹⁻  = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ  = 𝐒[1][cond_var_idx, end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx, var_vol²_idxs]
    𝐒²⁻  = 𝐒[2][cond_var_idx, var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx, shockvar²_idxs]
    𝐒²ᵉ  = 𝐒[2][cond_var_idx, shock²_idxs]
    𝐒⁻²  = 𝐒[2][T.past_not_future_and_mixed_idx, :]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = nnz(𝐒²⁻)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    initial_state = copy(state)

    state₁ = state[1][T.past_not_future_and_mixed_idx]
    state₂ = state[2][T.past_not_future_and_mixed_idx]

    state¹⁻_vol = vcat(state₁, 1)

    aug_state₁ = [state₁; 1; ones(T.nExo)]
    aug_state₂ = [state₂; 0; zeros(T.nExo)]

    kronaug_state₁ = zeros(length(aug_state₁)^2)

    J = ℒ.I(T.nExo)

    kron_buffer = zeros(T.nExo^2)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    kron_buffer3 = ℒ.kron(J, zeros(T.nPast_not_future_and_mixed + 1))

    kronstate¹⁻_vol = zeros((T.nPast_not_future_and_mixed + 1)^2)

    shock_independent = zeros(size(data_in_deviations,1))

    𝐒ⁱ = copy(𝐒¹ᵉ)

    jacc = copy(𝐒¹ᵉ)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 
        
    init_guess = zeros(size(𝐒ⁱ, 2))

    for i in axes(data_in_deviations, 2)
        # state¹⁻ = state₁
        # state¹⁻_vol = vcat(state¹⁻, 1)
        # state²⁻ = state₂#[T.past_not_future_and_mixed_idx]

        copyto!(state¹⁻_vol, 1, state₁, 1)

        # shock_independent = data_in_deviations[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒¹⁻ * state²⁻ + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)
        copyto!(shock_independent, data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)

        ℒ.kron!(kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol, -1/2, 1)

        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)  
        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron_buffer3
        ℒ.kron!(kron_buffer3, J, state¹⁻_vol)

        ℒ.mul!(𝐒ⁱ, 𝐒²⁻ᵉ, kron_buffer3)

        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ)

        init_guess *= 0

        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer2,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )
                     
        # if matched println("$filter_algorithm: $matched; current x: $x") end      
        # if !matched
        #     x, matched = find_shocks(Val(:COBYLA), 
        #                             zeros(size(𝐒ⁱ, 2)),
        #                             kron_buffer,
        #                             kron_buffer2,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             shock_independent,
        #                             # max_iter = 500
        #                             )
            # println("COBYLA: $matched; current x: $x")
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer2,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             shock_independent)
                if !matched
                    @error "Inversion filter failed at step $i"
                    return variables, shocks, zeros(0,0), decomposition
                end
            # end
        # end

        # x2, mat = find_shocks(Val(:SLSQP), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
            
        # x3, mat2 = find_shocks(Val(:COBYLA), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # if mat
        #     println("SLSQP: $(ℒ.norm(x2-x) / max(ℒ.norm(x2), ℒ.norm(x)))")
        # elseif mat2
        #     println("COBYLA: $(ℒ.norm(x3-x) / max(ℒ.norm(x3), ℒ.norm(x)))")
        # end

        # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x)

        # aug_state₁ = [state₁; 1; x]
        # aug_state₂ = [state₂; 0; zero(x)]
        copyto!(aug_state₁, 1, state₁, 1)
        copyto!(aug_state₁, length(state₁) + 2, x, 1)
        copyto!(aug_state₂, 1, state₂, 1)

        # state₁, state₂ = [𝐒⁻¹ * aug_state₁, 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * ℒ.kron(aug_state₁, aug_state₁) / 2] # strictly following Andreasen et al. (2018)
        # ℒ.mul!(state₁, 𝐒⁻¹, aug_state₁)
        ℒ.mul!(state[1], 𝐒[1], aug_state₁)
        state₁ .= state[1][T.past_not_future_and_mixed_idx]

        ℒ.kron!(kronaug_state₁, aug_state₁, aug_state₁)
        # ℒ.mul!(state₂, 𝐒⁻¹, aug_state₂)
        # ℒ.mul!(state₂, 𝐒⁻², kronaug_state₁, 1/2, 1)
        ℒ.mul!(state[2], 𝐒[1], aug_state₂)
        ℒ.mul!(state[2], 𝐒[2], kronaug_state₁, 1/2, 1)
        state₂ .= state[2][T.past_not_future_and_mixed_idx]

        variables[:,i] .= sum(state)
        shocks[:,i] .= x
    end

    states = [initial_state for _ in 1:𝓂.timings.nExo + 1]

    decomposition[:, end, :] .= variables

    for i in 1:𝓂.timings.nExo
        sck = zeros(𝓂.timings.nExo)
        sck[i] = shocks[i, 1]

        aug_state₁ = [initial_state[1][T.past_not_future_and_mixed_idx]; 1; sck]
        aug_state₂ = [initial_state[2][T.past_not_future_and_mixed_idx]; 0; zero(sck)]

        states[i] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * ℒ.kron(aug_state₁, aug_state₁) / 2] # state_update(initial_state , sck)
        decomposition[:,i,1] = sum(states[i])
    end

    aug_state₁ = [initial_state[1][T.past_not_future_and_mixed_idx]; 1; shocks[:, 1]]
    aug_state₂ = [initial_state[2][T.past_not_future_and_mixed_idx]; 0; zero(shocks[:, 1])]

    states[end] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * ℒ.kron(aug_state₁, aug_state₁) / 2] # state_update(initial_state, shocks[:, 1])

    decomposition[:, end - 2, 1] = sum(states[end]) - sum(decomposition[:, 1:end - 3, 1], dims = 2)
    decomposition[:, end - 1, 1] .= decomposition[:, end, 1] - sum(decomposition[:, 1:end - 2, 1], dims = 2)

    for i in 2:size(data_in_deviations, 2)
        for ii in 1:𝓂.timings.nExo
            sck = zeros(𝓂.timings.nExo)
            sck[ii] = shocks[ii, i]
            
            aug_state₁ = [states[ii][1][T.past_not_future_and_mixed_idx]; 1; sck]
            aug_state₂ = [states[ii][2][T.past_not_future_and_mixed_idx]; 0; zero(sck)]
    
            states[ii] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * ℒ.kron(aug_state₁, aug_state₁) / 2] # state_update(states[ii] , sck)
            decomposition[:, ii, i] = sum(states[ii])
        end

        aug_state₁ = [states[end][1][T.past_not_future_and_mixed_idx]; 1; shocks[:, i]]
        aug_state₂ = [states[end][2][T.past_not_future_and_mixed_idx]; 0; zero(shocks[:, i])]
    
        states[end] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * ℒ.kron(aug_state₁, aug_state₁) / 2] # state_update(states[end] , shocks[:, i])

        decomposition[:, end - 2, i] = sum(states[end]) - sum(decomposition[:, 1:end - 3, i], dims = 2)
        decomposition[:, end - 1, i] .= decomposition[:, end, i] - sum(decomposition[:, 1:end - 2, i], dims = 2)
    end

    return variables, shocks, zeros(0,0), decomposition
end

function filter_data_with_model(𝓂::ℳ,
                                data_in_deviations::KeyedArray{Float64},
                                ::Val{:third_order}, # algo
                                ::Val{:inversion}; # filter
                                warmup_iterations::Int = 0,
                                filter_algorithm::Symbol = :LagrangeNewton,
                                smooth::Bool = true,
                                opts::CalculationOptions = merge_calculation_options())
    T = 𝓂.timings

    variables = zeros(T.nVars, size(data_in_deviations,2))
    shocks = zeros(T.nExo, size(data_in_deviations,2))
    
    observables = get_and_check_observables(𝓂, data_in_deviations)

    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_third_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, opts = opts) # timer = timer,

    if !converged || solution_error > opts.tol.NSSS_acceptance_tol
        @error "Could not find 3rd order stochastic steady state"
        return variables, shocks, zeros(0,0), zeros(0,0)
    end

    𝐒 = [𝐒₁, 𝐒₂, 𝐒₃]
    
    all_SS = expand_steady_state(SS_and_pars,𝓂)

    state = collect(sss) - all_SS


    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))

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
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = nnz(𝐒²⁻)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    state = state[T.past_not_future_and_mixed_idx]

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

    kron_buffer = zeros(T.nExo^2)

    kron_buffer² = zeros(T.nExo^3)

    J = ℒ.I(T.nExo)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    kron_buffer3 = ℒ.kron(J, kron_buffer)

    kron_buffer4 = ℒ.kron(ℒ.kron(J, J), zeros(T.nExo))

    II = sparse(ℒ.I(T.nExo^2))

    for i in axes(data_in_deviations,2)
        state¹⁻ = state

        state¹⁻_vol = vcat(state¹⁻, 1)
        
        shock_independent = collect(data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
        
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
    
        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

        𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

        # x, jacc, matchd = find_shocks(Val(:fixed_point), state isa Vector{Float64} ? [state] : state, 𝐒, data_in_deviations[:,i], observables, T)

        init_guess = zeros(size(𝐒ⁱ, 2))

        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
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
                                # max_iter = 200
                                )
                                
        # println("$filter_algorithm: $matched; current x: $x, $(ℒ.norm(x))")
        # if !matched

        # backup_solver = :COBYLA

        # if filter_algorithm ≠ backup_solver
        #     x̂, matched2 = find_shocks(Val(backup_solver), 
        #                         zeros(size(𝐒ⁱ, 2)),
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 5000
        #                         )
        #     if ℒ.norm(x̂) * (1 - eps(Float32)) < ℒ.norm(x)
        #         x̄, matched3 = find_shocks(Val(filter_algorithm), 
        #                             x̂,
        #                             kron_buffer,
        #                             kron_buffer²,
        #                             kron_buffer2,
        #                             kron_buffer3,
        #                             kron_buffer4,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             𝐒ⁱ³ᵉ,
        #                             shock_independent,
        #                             # max_iter = 200
        #                             )
                              
        #         if matched3 && ℒ.norm(x̄) * (1 - eps(Float32)) < ℒ.norm(x̂)
        #             println("$i - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̄
        #             matched = matched3
        #         elseif matched2
        #             println("$i - $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̂
        #             matched = matched2
        #         else
        #             y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

        #             norm1 = ℒ.norm(y)

        #             norm2 = ℒ.norm(shock_independent)

        #             println("$i - $filter_algorithm ($matched) - $(ℒ.norm(x)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), residual norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
        #         end
        #     else
        #         println("$i - $filter_algorithm ($matched) - $(ℒ.norm(x)), $backup_solver ($matched2) - $(ℒ.norm(x̂))")
        #     end
        # end
        if !matched
            @error "Inversion filter failed at step $i"
            return variables, shocks, zeros(0,0), zeros(0,0)
        end 
            # println("COBYLA: $matched; current x: $x")
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer²,
            #                             kron_buffer2,
            #                             kron_buffer3,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             𝐒ⁱ³ᵉ,
            #                             shock_independent)
            #     println("$filter_algorithm: $matched; current x: $x")
            #     if !matched
            #         x, matched = find_shocks(Val(:COBYLA), 
            #                                 x,
            #                                 kron_buffer,
            #                                 kron_buffer²,
            #                                 kron_buffer2,
            #                                 kron_buffer3,
            #                                 J,
            #                                 𝐒ⁱ,
            #                                 𝐒ⁱ²ᵉ,
            #                                 𝐒ⁱ³ᵉ,
            #                                 shock_independent)
            #         println("COBYLA: $matched; current x: $x")
            #     end
            # end
        # end

        # x2, mat = find_shocks(Val(:COBYLA), 
        #                         init_guess,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 200
        #                         )
            
        # x3, mat2 = find_shocks(Val(filter_algorithm), 
        #                         x2,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # # if mat
        #     println("COBYLA - $mat: $x2, $(ℒ.norm(x2))")
        # # end
        # # if mat2
        #     println("LagrangeNewton restart - $mat2: $x3, $(ℒ.norm(x3))")
        # # end

        aug_state = [state; 1; x]

        # res = 𝐒[1][cond_var_idx, :] * aug_state + 𝐒[2][cond_var_idx, :] * ℒ.kron(aug_state, aug_state) / 2 + 𝐒[3][cond_var_idx, :] * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6 - data_in_deviations[:,i]
        # println("Match with data: $res")

        # state = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
        full_state = 𝐒[1] * aug_state + 𝐒[2] * ℒ.kron(aug_state, aug_state) / 2 + 𝐒[3] * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
        # state = state_update(state, x)
        
        shocks[:,i] .= x
        variables[:,i] .= full_state

        state .= full_state[T.past_not_future_and_mixed_idx]
    end
    
    return variables, shocks, zeros(0,0), zeros(0,0)
end


function filter_data_with_model(𝓂::ℳ,
                                data_in_deviations::KeyedArray{Float64},
                                ::Val{:pruned_third_order}, # algo
                                ::Val{:inversion}; # filter
                                warmup_iterations::Int = 0,
                                filter_algorithm::Symbol = :LagrangeNewton,
                                smooth::Bool = true,
                                opts::CalculationOptions = merge_calculation_options())
    T = 𝓂.timings

    variables = zeros(T.nVars, size(data_in_deviations,2))
    shocks = zeros(T.nExo, size(data_in_deviations,2))
    decomposition = zeros(𝓂.timings.nVars, 𝓂.timings.nExo + 3, size(data_in_deviations, 2))
    
    observables = get_and_check_observables(𝓂, data_in_deviations)

    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_third_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, pruning = true, opts = opts) # timer = timer,

    if !converged || solution_error > opts.tol.NSSS_acceptance_tol
        @error "Could not find pruned 3rd order stochastic steady state"
        return variables, shocks, zeros(0,0), zeros(0,0)
    end

    𝐒 = [𝐒₁, 𝐒₂, 𝐒₃]

    all_SS = expand_steady_state(SS_and_pars,𝓂)

    state = [zeros(𝓂.timings.nVars), collect(sss) - all_SS, zeros(𝓂.timings.nVars)]

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

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

    initial_state = copy(state)

    state₁ = state[1][T.past_not_future_and_mixed_idx]
    state₂ = state[2][T.past_not_future_and_mixed_idx]
    state₃ = state[3][T.past_not_future_and_mixed_idx]

    kron_buffer = zeros(T.nExo^2)

    kron_buffer² = zeros(T.nExo^3)

    II = sparse(ℒ.I(T.nExo^2))
    
    J = ℒ.I(T.nExo)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    kron_buffer3 = ℒ.kron(J, kron_buffer)

    kron_buffer4 = ℒ.kron(ℒ.kron(J, J), zeros(T.nExo))

    for i in axes(data_in_deviations,2)
        # state¹⁻ = state₁

        state¹⁻_vol = vcat(state₁, 1)

        # state²⁻ = state₂#[T.past_not_future_and_mixed_idx]

        # state³⁻ = state₃#[T.past_not_future_and_mixed_idx]

        shock_independent = collect(data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)

        ℒ.mul!(shock_independent, 𝐒¹⁻, state₃, -1, 1)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
        
        ℒ.mul!(shock_independent, 𝐒²⁻, ℒ.kron(state₁, state₂), -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒²⁻ᵛᵉ * ℒ.kron(ℒ.I(T.nExo), state₂) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
    
        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

        𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

        # x, jacc, matchd = find_shocks(Val(:fixed_point), state isa Vector{Float64} ? [state] : state, 𝐒, data_in_deviations[:,i], observables, T)

        init_guess = zeros(size(𝐒ⁱ, 2))


        # x² , matched = find_shocks(Val(filter_algorithm), 
        #                         init_guess,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 200
        #                         )
        #                         println(x²)

        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
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
                                # max_iter = 200
                                )
                                
                                # println(x)
        # println("$filter_algorithm: $matched; current x: $x, $(ℒ.norm(x))")
        # if !matched

        # backup_solver = :COBYLA

        # if filter_algorithm ≠ backup_solver
        #     x̂, matched2 = find_shocks(Val(backup_solver), 
        #                         zeros(size(𝐒ⁱ, 2)),
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 5000
        #                         )
        #     if ℒ.norm(x̂) * (1 - eps(Float32)) < ℒ.norm(x)
        #         x̄, matched3 = find_shocks(Val(filter_algorithm), 
        #                             x̂,
        #                             kron_buffer,
        #                             kron_buffer²,
        #                             kron_buffer2,
        #                             kron_buffer3,
        #                             kron_buffer4,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             𝐒ⁱ³ᵉ,
        #                             shock_independent,
        #                             # max_iter = 200
        #                             )
                              
        #         if matched3 && (!matched || ℒ.norm(x̄) * (1 - eps(Float32)) < ℒ.norm(x̂) || (matched && ℒ.norm(x̄) * (1 - eps(Float32)) < ℒ.norm(x)))
        #             # println("$i - $filter_algorithm restart - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̄
        #             matched = matched3
        #         elseif matched2
        #             # println("$i - $backup_solver - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̂
        #             matched = matched2
        #         # else
        #         #     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

        #         #     norm1 = ℒ.norm(y)

        #         #     norm2 = ℒ.norm(shock_independent)

        #             # println("$i - $filter_algorithm - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")#, residual norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
        #         end
        #     # else
        #     #     println("$i - $filter_algorithm ($matched) - $(ℒ.norm(x)), $backup_solver ($matched2) - $(ℒ.norm(x̂))")
        #     end
        # end
        if !matched
            @error "Inversion filter failed at step $i"
            return variables, shocks, zeros(0,0), decomposition
        end
            # println("COBYLA: $matched; current x: $x")
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer²,
            #                             kron_buffer2,
            #                             kron_buffer3,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             𝐒ⁱ³ᵉ,
            #                             shock_independent)
                # println("$filter_algorithm: $matched; current x: $x")
                # if !matched
                #     x, matched = find_shocks(Val(:COBYLA), 
                #                             x,
                #                             kron_buffer,
                #                             kron_buffer²,
                #                             kron_buffer2,
                #                             kron_buffer3,
                #                             J,
                #                             𝐒ⁱ,
                #                             𝐒ⁱ²ᵉ,
                #                             𝐒ⁱ³ᵉ,
                #                             shock_independent)
                # end
            # end
        # end

        # x2, mat = find_shocks(Val(:SLSQP), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
            
        # x3, mat2 = find_shocks(Val(:COBYLA), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # if mat
        #     println("SLSQP: $(ℒ.norm(x2-x) / max(ℒ.norm(x2), ℒ.norm(x))), $(ℒ.norm(x2)-ℒ.norm(x))")
        # elseif mat2
        #     println("COBYLA: $(ℒ.norm(x3-x) / max(ℒ.norm(x3), ℒ.norm(x))), $(ℒ.norm(x3)-ℒ.norm(x))")
        # end

        aug_state₁ = [state₁; 1; x]
        aug_state₁̂ = [state₁; 0; x]
        aug_state₂ = [state₂; 0; zero(x)]
        aug_state₃ = [state₃; 0; zero(x)]
        
        kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

        # res = 𝐒[1][cond_var_idx,:] * aug_state₁   +   𝐒[1][cond_var_idx,:] * aug_state₂ + 𝐒[2][cond_var_idx,:] * kron_aug_state₁ / 2   +   𝐒[1][cond_var_idx,:] * aug_state₃ + 𝐒[2][cond_var_idx,:] * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒[3][cond_var_idx,:] * ℒ.kron(kron_aug_state₁,aug_state₁) / 6 - data_in_deviations[:,i]
        # println("Match with data: $res")
        
        # println(ℒ.norm(x))

        # state = [𝐒⁻¹ * aug_state₁, 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * kron_aug_state₁ / 2, 𝐒⁻¹ * aug_state₃ + 𝐒⁻² * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒⁻³ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6]
        state = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * kron_aug_state₁ / 2, 𝐒[1] * aug_state₃ + 𝐒[2] * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒[3] * ℒ.kron(kron_aug_state₁,aug_state₁) / 6]
        
        state₁ .= state[1][T.past_not_future_and_mixed_idx]
        state₂ .= state[2][T.past_not_future_and_mixed_idx]
        state₃ .= state[3][T.past_not_future_and_mixed_idx]

        variables[:,i] .= sum(state)
        shocks[:,i] .= x
    end

    states = [initial_state for _ in 1:𝓂.timings.nExo + 1]

    decomposition[:, end, :] .= variables

    for i in 1:𝓂.timings.nExo
        sck = zeros(𝓂.timings.nExo)
        sck[i] = shocks[i, 1]

        aug_state₁ = [initial_state[1][T.past_not_future_and_mixed_idx]; 1; sck]
        aug_state₁̂ = [initial_state[1][T.past_not_future_and_mixed_idx]; 0; sck]
        aug_state₂ = [initial_state[2][T.past_not_future_and_mixed_idx]; 0; zero(sck)]
        aug_state₃ = [initial_state[3][T.past_not_future_and_mixed_idx]; 0; zero(sck)]
        
        kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

        states[i] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * kron_aug_state₁ / 2, 𝐒[1] * aug_state₃ + 𝐒[2] * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒[3] * ℒ.kron(kron_aug_state₁,aug_state₁) / 6] # state_update(initial_state , sck)

        decomposition[:,i,1] = sum(states[i])
    end

    aug_state₁ = [initial_state[1][T.past_not_future_and_mixed_idx]; 1; shocks[:, 1]]
    aug_state₁̂ = [initial_state[1][T.past_not_future_and_mixed_idx]; 0; shocks[:, 1]]
    aug_state₂ = [initial_state[2][T.past_not_future_and_mixed_idx]; 0; zero(shocks[:, 1])]
    aug_state₃ = [initial_state[3][T.past_not_future_and_mixed_idx]; 0; zero(shocks[:, 1])]
    
    kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

    states[end] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * kron_aug_state₁ / 2, 𝐒[1] * aug_state₃ + 𝐒[2] * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒[3] * ℒ.kron(kron_aug_state₁,aug_state₁) / 6] # state_update(initial_state, shocks[:, 1])

    decomposition[:,end - 2, 1] = sum(states[end]) - sum(decomposition[:,1:end - 3, 1], dims = 2)
    decomposition[:,end - 1, 1] .= decomposition[:, end, 1] - sum(decomposition[:,1:end - 2, 1], dims = 2)

    for i in 2:size(data_in_deviations, 2)
        for ii in 1:𝓂.timings.nExo
            sck = zeros(𝓂.timings.nExo)
            sck[ii] = shocks[ii, i]

            aug_state₁ = [states[ii][1][T.past_not_future_and_mixed_idx]; 1; sck]
            aug_state₁̂ = [states[ii][1][T.past_not_future_and_mixed_idx]; 0; sck]
            aug_state₂ = [states[ii][2][T.past_not_future_and_mixed_idx]; 0; zero(sck)]
            aug_state₃ = [states[ii][3][T.past_not_future_and_mixed_idx]; 0; zero(sck)]
            
            kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

            states[ii] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * kron_aug_state₁ / 2, 𝐒[1] * aug_state₃ + 𝐒[2] * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒[3] * ℒ.kron(kron_aug_state₁,aug_state₁) / 6] # state_update(states[ii] , sck)

            decomposition[:, ii, i] = sum(states[ii])
        end

        aug_state₁ = [states[end][1][T.past_not_future_and_mixed_idx]; 1; shocks[:, i]]
        aug_state₁̂ = [states[end][1][T.past_not_future_and_mixed_idx]; 0; shocks[:, i]]
        aug_state₂ = [states[end][2][T.past_not_future_and_mixed_idx]; 0; zero(shocks[:, i])]
        aug_state₃ = [states[end][3][T.past_not_future_and_mixed_idx]; 0; zero(shocks[:, i])]
        
        kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

        states[end] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * kron_aug_state₁ / 2, 𝐒[1] * aug_state₃ + 𝐒[2] * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒[3] * ℒ.kron(kron_aug_state₁,aug_state₁) / 6] # state_update(states[end] , shocks[:, i])
        
        decomposition[:,end - 2, i] = sum(states[end]) - sum(decomposition[:,1:end - 3, i], dims = 2)
        decomposition[:,end - 1, i] .= decomposition[:, end, i] - sum(decomposition[:,1:end - 2, i], dims = 2)
    end

    return variables, shocks, zeros(0,0), decomposition
end

end # dispatch_doctor