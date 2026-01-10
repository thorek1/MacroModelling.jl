
function calculate_first_order_solution(∇₁::Matrix{ℱ.Dual{Z,S,N}}; 
                                        T::timings, 
                                        opts::CalculationOptions = merge_calculation_options(),
                                        initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0))::Tuple{Matrix{ℱ.Dual{Z,S,N}}, Matrix{Float64}, Bool} where {Z,S,N}
    ∇̂₁ = ℱ.value.(∇₁)

    expand = [ℒ.I(T.nVars)[T.future_not_past_and_mixed_idx,:], ℒ.I(T.nVars)[T.past_not_future_and_mixed_idx,:]] 

    A = ∇̂₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    B = ∇̂₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]

    𝐒₁, qme_sol, solved = calculate_first_order_solution(∇̂₁; T = T, opts = opts, initial_guess = initial_guess)

    if !solved 
        return ∇₁, qme_sol, false
    end

    X = 𝐒₁[:,1:end-T.nExo] * expand[2]
    
    AXB = A * X + B
    
    AXBfact = RF.lu(AXB, check = false)

    if !ℒ.issuccess(AXBfact)
        AXBfact = ℒ.svd(AXB)
    end

    invAXB = inv(AXBfact)

    AA = invAXB * A

    X² = X * X

    X̃ = zeros(length(𝐒₁[:,1:end-T.nExo]), N)

    p = zero(∇̂₁)

    initial_guess = zero(invAXB)

    # https://arxiv.org/abs/2011.11430  
    for i in 1:N
        p .= ℱ.partials.(∇₁, i)

        dA = p[:,1:T.nFuture_not_past_and_mixed] * expand[1]
        dB = p[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
        dC = p[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]
        
        CC = invAXB * (dA * X² + dC + dB * X)

        if ℒ.norm(CC) < eps() continue end

        dX, solved = solve_sylvester_equation(AA, -X, -CC, 
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

    Jm = @view(ℒ.diagm(ones(S,T.nVars))[T.past_not_future_and_mixed_idx,:])
    
    ∇₊ = ∇₁[:,1:T.nFuture_not_past_and_mixed] * ℒ.diagm(ones(S,T.nVars))[T.future_not_past_and_mixed_idx,:]
    ∇₀ = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    ∇ₑ = ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

    B = -((∇₊ * x * Jm + ∇₀) \ ∇ₑ)

    return hcat(x, B), qme_sol, solved
end 


function sparse_preallocated!(Ŝ::Matrix{ℱ.Dual{Z,S,N}}; ℂ::higher_order_caches{T,F} = Higher_order_caches()) where {Z,S,N,T <: Real, F <: AbstractFloat}
    sparse(Ŝ)
end



function calculate_second_order_stochastic_steady_state(::Val{:newton}, 
                                                        𝐒₁::Matrix{ℱ.Dual{Z,S,N}}, 
                                                        𝐒₂::AbstractSparseMatrix{ℱ.Dual{Z,S,N}}, 
                                                        x::Vector{ℱ.Dual{Z,S,N}},
                                                        𝓂::ℳ;
                                                        # timer::TimerOutput = TimerOutput(),
                                                        tol::AbstractFloat = 1e-14)::Tuple{Vector{ℱ.Dual{Z,S,N}}, Bool} where {Z,S,N}

    𝐒₁̂ = ℱ.value.(𝐒₁)
    𝐒₂̂ = ℱ.value.(𝐒₂)
    x̂ = ℱ.value.(x)
    
    nᵉ = 𝓂.timings.nExo

    s_in_s⁺ = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))
    s_in_s = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed ), zeros(Bool, nᵉ + 1)))
    
    kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
    
    kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)
    
    A = 𝐒₁̂[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
    B = 𝐒₂̂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂̂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
 
    ∂x̄  = zeros(S, length(x̂), N)
    
    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    for i in 1:max_iters
        ∂x = (A + B * ℒ.kron(vcat(x̂,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))

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

            ∂A = ∂𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
            ∂B̂ = ∂𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]

            tmp = ∂A * x̂ + ∂B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2

            TMP = A + B * ℒ.kron(vcat(x̂,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed)

            ∂x̄[:,i] = -TMP \ tmp
        end
    end
    
    return reshape(map(x̂, eachrow(∂x̄)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(x̂)), solved
end


function calculate_third_order_stochastic_steady_state(::Val{:newton}, 
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
    
    nᵉ = 𝓂.timings.nExo

    s_in_s⁺ = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))
    s_in_s = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed ), zeros(Bool, nᵉ + 1)))
    
    kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
    
    kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)
    
    kron_s⁺_s⁺_s⁺ = ℒ.kron(s_in_s⁺, kron_s⁺_s⁺)
    
    kron_s_s⁺_s⁺ = ℒ.kron(kron_s⁺_s⁺, s_in_s)
    
    A = 𝐒₁̂[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
    B = 𝐒₂̂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂̂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
    C = 𝐒₃̂[𝓂.timings.past_not_future_and_mixed_idx,kron_s_s⁺_s⁺]
    Ĉ = 𝐒₃̂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺]

    ∂x̄  = zeros(S, length(x̂), N)
    
    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    for i in 1:max_iters
        ∂x = (A + B * ℒ.kron(vcat(x̂,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) + C * ℒ.kron(ℒ.kron(vcat(x̂,1), vcat(x̂,1)), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) / 2 - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))

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

            ∂A = ∂𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
            ∂B̂ = ∂𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
            ∂Ĉ = ∂𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺]

            tmp = ∂A * x̂ + ∂B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 + ∂Ĉ * ℒ.kron(vcat(x̂,1), ℒ.kron(vcat(x̂,1), vcat(x̂,1))) / 6

            TMP = A + B * ℒ.kron(vcat(x̂,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) + C * ℒ.kron(ℒ.kron(vcat(x̂,1), vcat(x̂,1)), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) / 2 - ℒ.I(𝓂.timings.nPast_not_future_and_mixed)

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
                                cold_start::Bool = false)::Tuple{Vector{ℱ.Dual{Z,S,N}}, Tuple{S, Int}} where {Z, S <: AbstractFloat, N}
                                # timer::TimerOutput = TimerOutput(),
    parameter_values = ℱ.value.(parameter_values_dual)

    if !isnothing(𝓂.custom_steady_state_function)
        SS_and_pars = 𝓂.custom_steady_state_function(parameter_values)

        vars_in_ss_equations = sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))
        expected_length = length(vars_in_ss_equations) + length(𝓂.calibration_equations_parameters)

        if length(SS_and_pars) != expected_length
            throw(ArgumentError("Custom steady state function returned $(length(SS_and_pars)) values, expected $expected_length."))
        end

        residual = zeros(length(𝓂.ss_equations) + length(𝓂.calibration_equations))

        𝓂.SS_check_func(residual, 𝓂.parameter_values, SS_and_pars)

        solution_error = sum(abs, residual)
        
        iters = 0

        if !isfinite(solution_error) || solution_error > opts.tol.NSSS_acceptance_tol
            throw(ArgumentError("Custom steady state function failed steady state check: residual $solution_error > $(opts.tol.NSSS_acceptance_tol)."))
        end
        
        var_idx = indexin([vars_in_ss_equations...], [𝓂.var...,𝓂.calibration_equations_parameters...])

        calib_idx = indexin([𝓂.calibration_equations_parameters...], [𝓂.var...,𝓂.calibration_equations_parameters...])

        SS_and_pars_tmp = zeros(length(𝓂.var) + length(𝓂.calibration_equations_parameters))

        SS_and_pars_tmp[[var_idx..., calib_idx...]] = SS_and_pars

        SS_and_pars = SS_and_pars_tmp
    else
        SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameter_values, 𝓂, opts.tol, opts.verbose, cold_start, 𝓂.solver_parameters)
    end
    
    ∂SS_and_pars = zeros(S, length(SS_and_pars), N)

    if solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error)
        if opts.verbose println("Failed to find NSSS") end

        solution_error = S(10.0)
    else
        SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)
            
        SS_and_pars_names = vcat(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)
        
        SS_and_pars_names_no_exo = vcat(Symbol.(replace.(string.(sort(setdiff(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)

        # unknowns = union(setdiff(𝓂.vars_in_ss_equations, 𝓂.➕_vars), 𝓂.calibration_equations_parameters)
        unknowns = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))), 𝓂.calibration_equations_parameters))
        

        ∂ = parameter_values
        C = SS_and_pars[indexin(unique(SS_and_pars_names_no_exo), SS_and_pars_names_lead_lag)] # [dyn_ss_idx])

        if eltype(𝓂.∂SS_equations_∂parameters[1]) != eltype(parameter_values)
            if 𝓂.∂SS_equations_∂parameters[1] isa SparseMatrixCSC
                jac_buffer = similar(𝓂.∂SS_equations_∂parameters[1], eltype(parameter_values))
                jac_buffer.nzval .= 0
            else
                jac_buffer = zeros(eltype(parameter_values), size(𝓂.∂SS_equations_∂parameters[1]))
            end
        else
            jac_buffer = 𝓂.∂SS_equations_∂parameters[1]
        end

        𝓂.∂SS_equations_∂parameters[2](jac_buffer, ∂, C)

        ∂SS_equations_∂parameters = jac_buffer

        
        if eltype(𝓂.∂SS_equations_∂SS_and_pars[1]) != eltype(parameter_values)
            if 𝓂.∂SS_equations_∂SS_and_pars[1] isa SparseMatrixCSC
                jac_buffer = similar(𝓂.∂SS_equations_∂SS_and_pars[1], eltype(SS_and_pars))
                jac_buffer.nzval .= 0
            else
                jac_buffer = zeros(eltype(SS_and_pars), size(𝓂.∂SS_equations_∂SS_and_pars[1]))
            end
        else
            jac_buffer = 𝓂.∂SS_equations_∂SS_and_pars[1]
        end

        𝓂.∂SS_equations_∂SS_and_pars[2](jac_buffer, ∂, C)

        ∂SS_equations_∂SS_and_pars = jac_buffer

        ∂SS_equations_∂SS_and_pars_lu = RF.lu(∂SS_equations_∂SS_and_pars, check = false)

        if !ℒ.issuccess(∂SS_equations_∂SS_and_pars_lu)
            if opts.verbose println("Failed to calculate implicit derivative of NSSS") end
            
            solution_error = S(10.0)
        else
            JVP = -(∂SS_equations_∂SS_and_pars_lu \ ∂SS_equations_∂parameters)#[indexin(SS_and_pars_names, unknowns),:]

            jvp = zeros(length(SS_and_pars_names_lead_lag), length(𝓂.parameters))
            
            for (i,v) in enumerate(SS_and_pars_names)
                if v in unknowns
                    jvp[i,:] = JVP[indexin([v], unknowns),:]
                end
            end

            for i in 1:N
                parameter_values_partials = ℱ.partials.(parameter_values_dual, i)

                ∂SS_and_pars[:,i] = jvp * parameter_values_partials
            end
        end
    end
    
    return reshape(map(SS_and_pars, eachrow(∂SS_and_pars)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(SS_and_pars)), (solution_error, iters)
end



function run_kalman_iterations(A::Matrix{S}, 
                                𝐁::Matrix{S}, 
                                C::Matrix{Float64}, 
                                P::Matrix{S}, 
                                data_in_deviations::Matrix{S}; 
                                presample_periods::Int = 0,
                                on_failure_loglikelihood::U = -Inf,
                                # timer::TimerOutput = TimerOutput(),
                                verbose::Bool = false)::S where {S <: ℱ.Dual, U <: AbstractFloat}
    # @timeit_debug timer "Calculate Kalman filter - forward mode AD" begin
    u = zeros(S, size(C,2))

    z = C * u

    loglik = S(0.0)

    F = similar(C * C')

    K = similar(C')

    for t in 1:size(data_in_deviations, 2)
        if !all(isfinite.(z)) 
            if verbose println("KF not finite at step $t") end
            return on_failure_loglikelihood 
        end

        v = data_in_deviations[:, t] - z

        F = C * P * C'

        luF = ℒ.lu(F, check = false) ###

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

        invF = inv(luF) ###

        if t > presample_periods
            loglik += log(Fdet) + ℒ.dot(v, invF, v)###
        end

        K = P * C' * invF

        P = A * (P - K * C * P) * A' + 𝐁

        u = A * (u + K * v)

        z = C * u
    end

    # end # timeit_debug

    return -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 
end


function solve_sylvester_equation(  A::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    B::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    C::AbstractMatrix{ℱ.Dual{Z,S,N}};
                                    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                    sylvester_algorithm::Symbol = :doubling,
                                    acceptance_tol::AbstractFloat = 1e-10,
                                    𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                    tol::AbstractFloat = 1e-14,
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false)::Tuple{Matrix{ℱ.Dual{Z,S,N}}, Bool} where {Z,S,N}
    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    B̂ = ℱ.value.(B)
    Ĉ = ℱ.value.(C)

    P̂, solved = solve_sylvester_equation(Â, B̂, Ĉ, 
                                        sylvester_algorithm = sylvester_algorithm, 
                                        tol = tol, 
                                        𝕊ℂ = 𝕊ℂ,
                                        verbose = verbose, 
                                        initial_guess = initial_guess)

    Ã = copy(Â)
    B̃ = copy(B̂)
    C̃ = copy(Ĉ)
    
    P̃ = zeros(S, length(P̂), N)
    
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        B̃ .= ℱ.partials.(B, i)
        C̃ .= ℱ.partials.(C, i)

        X = Ã * P̂ * B̂ + Â * P̂ * B̃ + C̃
        
        if ℒ.norm(X) < eps() continue end

        P, slvd = solve_sylvester_equation(Â, B̂, X, 
                                            sylvester_algorithm = sylvester_algorithm, 
                                            𝕊ℂ = 𝕊ℂ,
                                            tol = tol, 
                                            verbose = verbose)

        solved = solved && slvd

        P̃[:,i] = vec(P)
    end
    
    return reshape(map(P̂, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂)), solved
end


function solve_quadratic_matrix_equation(A::AbstractMatrix{ℱ.Dual{Z,S,N}}, 
                                        B::AbstractMatrix{ℱ.Dual{Z,S,N}}, 
                                        C::AbstractMatrix{ℱ.Dual{Z,S,N}}, 
                                        T::timings; 
                                        initial_guess::AbstractMatrix{<:Real} = zeros(0,0),
                                        tol::AbstractFloat = 1e-8, 
                                        quadratic_matrix_equation_algorithm::Symbol = :schur, 
                                        # timer::TimerOutput = TimerOutput(),
                                        verbose::Bool = false) where {Z,S,N}
    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    B̂ = ℱ.value.(B)
    Ĉ = ℱ.value.(C)

    X, solved = solve_quadratic_matrix_equation(Â, B̂, Ĉ, 
                                                Val(quadratic_matrix_equation_algorithm), 
                                                T; 
                                                tol = tol,
                                                initial_guess = initial_guess,
                                                # timer = timer,
                                                verbose = verbose)

    AXB = Â * X + B̂
    
    AXBfact = ℒ.lu(AXB, check = false)

    if !ℒ.issuccess(AXBfact)
        AXBfact = ℒ.svd(AXB)
    end

    invAXB = inv(AXBfact)

    AA = invAXB * Â

    X² = X * X

    X̃ = zeros(length(X), N)

    # https://arxiv.org/abs/2011.11430  
    for i in 1:N
        dA = ℱ.partials.(A, i)
        dB = ℱ.partials.(B, i)
        dC = ℱ.partials.(C, i)
    
        CC = invAXB * (dA * X² + dB * X + dC)

        if ℒ.norm(CC) < eps() continue end
    
        dX, slvd = solve_sylvester_equation(AA, -X, -CC, sylvester_algorithm = :doubling)

        solved = Bool(solved) && Bool(slvd)

        X̃[:,i] = vec(dX)
    end
    
    return reshape(map(X, eachrow(X̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(X)), solved
end


function solve_lyapunov_equation(  A::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    C::AbstractMatrix{ℱ.Dual{Z,S,N}};
                                    lyapunov_algorithm::Symbol = :doubling,
                                    tol::AbstractFloat = 1e-14,
                                    acceptance_tol::AbstractFloat = 1e-12,
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false)::Tuple{Matrix{ℱ.Dual{Z,S,N}}, Bool} where {Z,S,N}
    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    Ĉ = ℱ.value.(C)

    P̂, solved = solve_lyapunov_equation(Â, Ĉ, lyapunov_algorithm = lyapunov_algorithm, tol = tol, verbose = verbose)

    Ã = copy(Â)
    C̃ = copy(Ĉ)
    
    P̃ = zeros(length(P̂), N)
    
    # https://arxiv.org/abs/2011.11430  
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        C̃ .= ℱ.partials.(C, i)

        X = Ã * P̂ * Â' + Â * P̂ * Ã' + C̃

        if ℒ.norm(X) < eps() continue end

        P, slvd = solve_lyapunov_equation(Â, X, lyapunov_algorithm = lyapunov_algorithm, tol = tol, verbose = verbose)
        
        solved = solved && slvd

        P̃[:,i] = vec(P)
    end
    
    return reshape(map(P̂, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂)), solved
end
