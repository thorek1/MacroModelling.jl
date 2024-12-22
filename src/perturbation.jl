function calculate_first_order_solution(∇₁::Matrix{Float64}; 
                                        T::timings, 
                                        opts::CalculationOptions = merge_calculation_options(),
                                        initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0))::Tuple{Matrix{Float64}, Matrix{Float64}, Bool}
    # @timeit_debug timer "Calculate 1st order solution" begin
    # @timeit_debug timer "Preprocessing" begin

    dynIndex = T.nPresent_only+1:T.nVars

    reverse_dynamic_order = indexin([T.past_not_future_idx; T.future_not_past_and_mixed_idx], T.present_but_not_only_idx)

    comb = union(T.future_not_past_and_mixed_idx, T.past_not_future_idx)
    sort!(comb)

    future_not_past_and_mixed_in_comb = indexin(T.future_not_past_and_mixed_idx, comb)
    past_not_future_and_mixed_in_comb = indexin(T.past_not_future_and_mixed_idx, comb)
  
    Ir = ℒ.I(length(comb))
    
    ∇₊ = ∇₁[:,1:T.nFuture_not_past_and_mixed]
    ∇₀ = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]    
    ∇₋ = ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]
    ∇ₑ = ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]
    
    # end # timeit_debug
    # @timeit_debug timer "Invert ∇₀" begin

    Q    = ℒ.qr!(∇₀[:,T.present_only_idx])

    A₊ = Q.Q' * ∇₊
    A₀ = Q.Q' * ∇₀
    A₋ = Q.Q' * ∇₋
    
    # end # timeit_debug
    # @timeit_debug timer "Sort matrices" begin

    Ã₊ = A₊[dynIndex,:] * Ir[future_not_past_and_mixed_in_comb,:]
    Ã₀ = A₀[dynIndex, comb]
    Ã₋ = A₋[dynIndex,:] * Ir[past_not_future_and_mixed_in_comb,:]

    # end # timeit_debug
    # @timeit_debug timer "Quadratic matrix equation solve" begin

    sol, solved = solve_quadratic_matrix_equation(Ã₊, Ã₀, Ã₋, T, 
                                                    initial_guess = initial_guess,
                                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                                    tol = opts.qme_tol,
                                                    acceptance_tol = opts.qme_acceptance_tol,
                                                    verbose = opts.verbose)

    if !solved
        if opts.verbose println("Quadratic matrix equation solution failed.") end
        return zeros(T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false
    end

    # end # timeit_debug
    # @timeit_debug timer "Postprocessing" begin
    # @timeit_debug timer "Setup matrices" begin

    sol_compact = sol[reverse_dynamic_order, past_not_future_and_mixed_in_comb]

    D = sol_compact[end - T.nFuture_not_past_and_mixed + 1:end, :]

    L = sol[indexin(T.past_not_future_and_mixed_idx, T.present_but_not_only_idx), past_not_future_and_mixed_in_comb]

    Ā₀ᵤ  = A₀[1:T.nPresent_only, T.present_only_idx]
    A₊ᵤ  = A₊[1:T.nPresent_only,:]
    Ã₀ᵤ  = A₀[1:T.nPresent_only, T.present_but_not_only_idx]
    A₋ᵤ  = A₋[1:T.nPresent_only,:]

    # end # timeit_debug
    # @timeit_debug timer "Invert Ā₀ᵤ" begin

    Ā̂₀ᵤ = ℒ.lu!(Ā₀ᵤ, check = false)

    if !ℒ.issuccess(Ā̂₀ᵤ)
        if opts.verbose println("Factorisation of Ā₀ᵤ failed") end
        return zeros(T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false
    end

    # A    = vcat(-(Ā̂₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ)), sol)
    if T.nPresent_only > 0
        ℒ.mul!(A₋ᵤ, Ã₀ᵤ, sol[:,past_not_future_and_mixed_in_comb], 1, 1)
        nₚ₋ =  A₊ᵤ * D
        ℒ.mul!(A₋ᵤ, nₚ₋, L, 1, 1)
        ℒ.ldiv!(Ā̂₀ᵤ, A₋ᵤ)
        ℒ.rmul!(A₋ᵤ, -1)
    end
    
    A    = vcat(A₋ᵤ, sol_compact)[T.reorder,:]

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Exogenous part solution" begin

    M = A[T.future_not_past_and_mixed_idx,:] * ℒ.I(T.nVars)[T.past_not_future_and_mixed_idx,:]

    ℒ.mul!(∇₀, ∇₁[:,1:T.nFuture_not_past_and_mixed], M, 1, 1)

    C = ℒ.lu!(∇₀, check = false)
    
    if !ℒ.issuccess(C)
        if opts.verbose println("Factorisation of ∇₀ failed") end
        return zeros(T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false
    end
    
    ℒ.ldiv!(C, ∇ₑ)
    ℒ.rmul!(∇ₑ, -1)

    # end # timeit_debug
    # end # timeit_debug

    return hcat(A, ∇ₑ), sol, true
end


function rrule(::typeof(calculate_first_order_solution), 
                ∇₁::Matrix{Float64};
                T::timings, 
                opts::CalculationOptions = merge_calculation_options(),
                initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0))
    # Forward pass to compute the output and intermediate values needed for the backward pass
    # @timeit_debug timer "Calculate 1st order solution" begin
    # @timeit_debug timer "Preprocessing" begin

    dynIndex = T.nPresent_only+1:T.nVars

    reverse_dynamic_order = indexin([T.past_not_future_idx; T.future_not_past_and_mixed_idx], T.present_but_not_only_idx)

    comb = union(T.future_not_past_and_mixed_idx, T.past_not_future_idx)
    sort!(comb)

    future_not_past_and_mixed_in_comb = indexin(T.future_not_past_and_mixed_idx, comb)
    past_not_future_and_mixed_in_comb = indexin(T.past_not_future_and_mixed_idx, comb)
    
    Ir = ℒ.I(length(comb))
    
    ∇₊ = ∇₁[:,1:T.nFuture_not_past_and_mixed]
    ∇₀ = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]    
    ∇₋ = ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]
    ∇̂ₑ = ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]
    
    # end # timeit_debug
    # @timeit_debug timer "Invert ∇₀" begin

    Q    = ℒ.qr!(∇₀[:,T.present_only_idx])

    A₊ = Q.Q' * ∇₊
    A₀ = Q.Q' * ∇₀
    A₋ = Q.Q' * ∇₋
    
    # end # timeit_debug
    # @timeit_debug timer "Sort matrices" begin

    Ã₊ = A₊[dynIndex,:] * Ir[future_not_past_and_mixed_in_comb,:]
    Ã₀ = A₀[dynIndex, comb]
    Ã₋ = A₋[dynIndex,:] * Ir[past_not_future_and_mixed_in_comb,:]

    # end # timeit_debug
    # @timeit_debug timer "Quadratic matrix equation solve" begin

    sol, solved = solve_quadratic_matrix_equation(Ã₊, Ã₀, Ã₋, T, 
                                                    initial_guess = initial_guess,
                                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                                    tol = opts.qme_tol,
                                                    acceptance_tol = opts.qme_acceptance_tol,
                                                    verbose = opts.verbose)

    if !solved
        return (zeros(T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false), x -> NoTangent(), NoTangent(), NoTangent()
    end

    # end # timeit_debug
    # @timeit_debug timer "Postprocessing" begin
    # @timeit_debug timer "Setup matrices" begin

    sol_compact = sol[reverse_dynamic_order, past_not_future_and_mixed_in_comb]

    D = sol_compact[end - T.nFuture_not_past_and_mixed + 1:end, :]

    L = sol[indexin(T.past_not_future_and_mixed_idx, T.present_but_not_only_idx), past_not_future_and_mixed_in_comb]

    Ā₀ᵤ  = A₀[1:T.nPresent_only, T.present_only_idx]
    A₊ᵤ  = A₊[1:T.nPresent_only,:]
    Ã₀ᵤ  = A₀[1:T.nPresent_only, T.present_but_not_only_idx]
    A₋ᵤ  = A₋[1:T.nPresent_only,:]

    # end # timeit_debug
    # @timeit_debug timer "Invert Ā₀ᵤ" begin

    Ā̂₀ᵤ = ℒ.lu!(Ā₀ᵤ, check = false)

    if !ℒ.issuccess(Ā̂₀ᵤ)
        return (zeros(T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false), x -> NoTangent(), NoTangent(), NoTangent()
    end

    # A    = vcat(-(Ā̂₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ)), sol)
    if T.nPresent_only > 0
        ℒ.mul!(A₋ᵤ, Ã₀ᵤ, sol[:,past_not_future_and_mixed_in_comb], 1, 1)
        nₚ₋ =  A₊ᵤ * D
        ℒ.mul!(A₋ᵤ, nₚ₋, L, 1, 1)
        ℒ.ldiv!(Ā̂₀ᵤ, A₋ᵤ)
        ℒ.rmul!(A₋ᵤ, -1)
    end

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Exogenous part solution" begin

    expand =   [ℒ.I(T.nVars)[T.future_not_past_and_mixed_idx,:],
                ℒ.I(T.nVars)[T.past_not_future_and_mixed_idx,:]] 

    𝐒ᵗ = vcat(A₋ᵤ, sol_compact)[T.reorder,:]

    𝐒̂ᵗ = 𝐒ᵗ * expand[2]

    ℒ.mul!(∇₀, ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1], 𝐒̂ᵗ, 1, 1)

    C = ℒ.lu!(∇₀, check = false)
    
    if !ℒ.issuccess(C)
        return (zeros(T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false), x -> NoTangent(), NoTangent(), NoTangent()
    end
    
    ℒ.ldiv!(C, ∇̂ₑ)
    ℒ.rmul!(∇̂ₑ, -1)

    # end # timeit_debug
    # end # timeit_debug
    
    M = inv(C)

    tmp2 = -M' * (∇₊ * expand[1])'
    
    ∇₊ = ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    ∇ₑ = ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

    function first_order_solution_pullback(∂𝐒) 
        ∂∇₁ = zero(∇₁)

        ∂𝐒ᵗ = ∂𝐒[1][:,1:T.nPast_not_future_and_mixed]
        ∂𝐒ᵉ = ∂𝐒[1][:,T.nPast_not_future_and_mixed + 1:end]

        ∂∇₁[:,T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1:end] .= -M' * ∂𝐒ᵉ

        ∂∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)] .= M' * ∂𝐒ᵉ * ∇ₑ' * M'

        ∂∇₁[:,1:T.nFuture_not_past_and_mixed] .= (M' * ∂𝐒ᵉ * ∇ₑ' * M' * expand[2]' * 𝐒ᵗ')[:,T.future_not_past_and_mixed_idx]

        ∂𝐒ᵗ .+= ∇₊' * M' * ∂𝐒ᵉ * ∇ₑ' * M' * expand[2]'

        tmp1 = M' * ∂𝐒ᵗ * expand[2]

        ss, solved = solve_sylvester_equation(tmp2, 𝐒̂ᵗ', -tmp1,
                                                sylvester_algorithm = opts.sylvester_algorithm²,
                                                tol = opts.sylvester_tol,
                                                acceptance_tol = opts.sylvester_acceptance_tol,
                                                verbose = opts.verbose)

        if !solved
            NoTangent(), NoTangent(), NoTangent()
        end

        ∂∇₁[:,1:T.nFuture_not_past_and_mixed] .+= (ss * 𝐒̂ᵗ' * 𝐒̂ᵗ')[:,T.future_not_past_and_mixed_idx]
        ∂∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)] .+= ss * 𝐒̂ᵗ'
        ∂∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] .+= ss[:,T.past_not_future_and_mixed_idx]

        return NoTangent(), ∂∇₁, NoTangent()
    end

    return (hcat(𝐒ᵗ, ∇̂ₑ), sol, solved), first_order_solution_pullback
end


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
                                                tol = opts.sylvester_tol,
                                                acceptance_tol = opts.sylvester_acceptance_tol,
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





function calculate_second_order_solution(∇₁::AbstractMatrix{S}, #first order derivatives
                                            ∇₂::SparseMatrixCSC{S}, #second order derivatives
                                            𝑺₁::AbstractMatrix{S},#first order solution
                                            M₂::second_order_auxilliary_matrices;  # aux matrices
                                            T::timings,
                                            initial_guess::AbstractMatrix{Float64} = zeros(0,0),
                                            opts::CalculationOptions = merge_calculation_options())::Tuple{<: AbstractMatrix{S}, Bool} where S <: Real
    # @timeit_debug timer "Calculate second order solution" begin

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
    
    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) ℒ.I(nₑ + 1)[1,:] zeros(nₑ + 1, nₑ)]# |> sparse
    # droptol!(𝐒₁₋╱𝟏ₑ,tol)
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 1.0)

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                ℒ.I(nₑ₋)[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]] #|> sparse
    # droptol!(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,tol)

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)]# |> sparse
    # droptol!(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,tol)

    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.I(n)[i₋,:] - ∇₁[:,range(1,n) .+ n₊]

    # end # timeit_debug

    # @timeit_debug timer "Invert matrix" begin

    ∇₁₊𝐒₁➕∇₁₀lu = ℒ.lu(∇₁₊𝐒₁➕∇₁₀, check = false)

    if !ℒ.issuccess(∇₁₊𝐒₁➕∇₁₀lu)
        if opts.verbose println("Second order solution: inversion failed") end
        return ∇₁₊𝐒₁➕∇₁₀, false
    end

    # spinv = inv(∇₁₊𝐒₁➕∇₁₀)
    # spinv = choose_matrix_format(spinv)

    # end # timeit_debug

    # @timeit_debug timer "Setup second order matrices" begin
    # @timeit_debug timer "A" begin

    ∇₁₊ = @views ∇₁[:,1:n₊] * ℒ.I(n)[i₊,:]

    A = ∇₁₊𝐒₁➕∇₁₀lu \ ∇₁₊
    
    # end # timeit_debug
    # @timeit_debug timer "C" begin

    # ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = ∇₂ * (ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔) * M₂.𝐂₂ 
    ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, M₂.𝐂₂) + mat_mult_kron(∇₂, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎, M₂.𝛔 * M₂.𝐂₂)
    
    C = ∇₁₊𝐒₁➕∇₁₀lu \ ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹

    # end # timeit_debug
    # @timeit_debug timer "B" begin

    # 𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0)

    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0)
    B = mat_mult_kron(M₂.𝐔₂, 𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ, M₂.𝐂₂) + M₂.𝐔₂ * M₂.𝛔 * M₂.𝐂₂

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Solve sylvester equation" begin

    𝐒₂, solved = solve_sylvester_equation(A, B, C, 
                                            initial_guess = initial_guess,
                                            sylvester_algorithm = opts.sylvester_algorithm²,
                                            tol = opts.sylvester_tol,
                                            acceptance_tol = opts.sylvester_acceptance_tol,
                                            verbose = opts.verbose) # timer = timer)

    # end # timeit_debug
    # # @timeit_debug timer "Refine sylvester equation" begin

    # # if !solved && !(sylvester_algorithm == :doubling)
    # #     𝐒₂, solved = solve_sylvester_equation(A, B, C, 
    # #                                             # init = 𝐒₂, 
    # #                                             # sylvester_algorithm = :gmres, 
    # #                                             initial_guess = initial_guess,
    # #                                             sylvester_algorithm = :doubling, 
    # #                                             verbose = verbose, 
    # #                                             # tol = tol, 
    # #                                             timer = timer)
    # # end

    # # end # timeit_debug
    # @timeit_debug timer "Post-process" begin

    # 𝐒₂ *= M₂.𝐔₂

    # 𝐒₂ = sparse(𝐒₂)

    if !solved
        return 𝐒₂, false
    end

    # end # timeit_debug
    # end # timeit_debug

    return 𝐒₂, true
end




function rrule(::typeof(calculate_second_order_solution), 
                    ∇₁::AbstractMatrix{<: Real}, #first order derivatives
                    ∇₂::SparseMatrixCSC{<: Real}, #second order derivatives
                    𝑺₁::AbstractMatrix{<: Real},#first order solution
                    M₂::second_order_auxilliary_matrices;  # aux matrices
                    T::timings,
                    initial_guess::AbstractMatrix{Float64} = zeros(0,0),
                    opts::CalculationOptions = merge_calculation_options())
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
        return (∇₁₊𝐒₁➕∇₁₀, solved), x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
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

    𝐒₂, solved = solve_sylvester_equation(A, B, C,  
                                            initial_guess = initial_guess,
                                            sylvester_algorithm = opts.sylvester_algorithm²,
                                            tol = opts.sylvester_tol,
                                            acceptance_tol = opts.sylvester_acceptance_tol,
                                            verbose = opts.verbose) # timer = timer)

    # end # timeit_debug
    # @timeit_debug timer "Post-process" begin

    if !solved
        return (𝐒₂, solved), x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    # end # timeit_debug
  
    # sp⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t = choose_matrix_format(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋', density_threshold = 1.0)

    # sp𝐒₁₊╱𝟎t = choose_matrix_format(𝐒₁₊╱𝟎', density_threshold = 1.0)

    𝛔t = choose_matrix_format(M₂.𝛔', density_threshold = 1.0)

    𝐔₂t = choose_matrix_format(M₂.𝐔₂', density_threshold = 1.0)

    𝐂₂t = choose_matrix_format(M₂.𝐂₂', density_threshold = 1.0)

    ∇₂t = choose_matrix_format(∇₂', density_threshold = 1.0)

    # end # timeit_debug

    function second_order_solution_pullback(∂𝐒₂_solved) 
        # @timeit_debug timer "Second order solution - pullback" begin
            
        # @timeit_debug timer "Preallocate" begin
        ∂∇₂ = zeros(size(∇₂))
        ∂∇₁ = zero(∇₁)
        ∂𝐒₁ = zero(𝐒₁)
        ∂spinv = zero(∇₁₊𝐒₁➕∇₁₀)
        ∂𝐒₁₋╱𝟏ₑ = zeros(size(𝐒₁₋╱𝟏ₑ))
        ∂𝐒₁₊╱𝟎 = zeros(size(𝐒₁₊╱𝟎))
        ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = zeros(size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋))

        # end # timeit_debug

        ∂𝐒₂ = ∂𝐒₂_solved[1]
        
        # ∂𝐒₂ *= 𝐔₂t

        # @timeit_debug timer "Sylvester" begin

        ∂C, solved = solve_sylvester_equation(A', B', ∂𝐒₂,
                                                sylvester_algorithm = opts.sylvester_algorithm²,
                                                tol = opts.sylvester_tol,
                                                acceptance_tol = opts.sylvester_acceptance_tol,
                                                verbose = opts.verbose)
        
        if !solved
            return (𝐒₂, solved), x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
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

        return NoTangent(), ∂∇₁, ∂∇₂, ∂𝑺₁, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    

    # return (sparse(𝐒₂ * M₂.𝐔₂), solved), second_order_solution_pullback
    return (𝐒₂, solved), second_order_solution_pullback
end



function calculate_third_order_solution(∇₁::AbstractMatrix{<: Real}, #first order derivatives
                                            ∇₂::SparseMatrixCSC{<: Real}, #second order derivatives
                                            ∇₃::SparseMatrixCSC{<: Real}, #third order derivatives
                                            𝑺₁::AbstractMatrix{<: Real}, #first order solution
                                            𝐒₂::SparseMatrixCSC{<: Real}, #second order solution
                                            M₂::second_order_auxilliary_matrices,  # aux matrices second order
                                            M₃::third_order_auxilliary_matrices;  # aux matrices third order
                                            T::timings,
                                            initial_guess::AbstractMatrix{Float64} = zeros(0,0),
                                            opts::CalculationOptions = merge_calculation_options())

    # @timeit_debug timer "Calculate third order solution" begin
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
        return (∇₁₊𝐒₁➕∇₁₀, solved), x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
        
    # spinv = inv(∇₁₊𝐒₁➕∇₁₀)
    # spinv = choose_matrix_format(spinv)

    # end # timeit_debug
    
    ∇₁₊ = @views ∇₁[:,1:n₊] * ℒ.I(n)[i₊,:]

    A = ∇₁₊𝐒₁➕∇₁₀lu \ ∇₁₊

    # @timeit_debug timer "Setup B" begin
    # @timeit_debug timer "Add tmpkron" begin

    tmpkron = ℒ.kron(𝐒₁₋╱𝟏ₑ, M₂.𝛔)
    kron𝐒₁₋╱𝟏ₑ = ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ)
    
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
    B = choose_matrix_format(M₃.𝐔₃ * B)

    # end # timeit_debug
    # @timeit_debug timer "3rd Kronecker power" begin
    # B += mat_mult_kron(M₃.𝐔₃, collect(𝐒₁₋╱𝟏ₑ), collect(ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ)), M₃.𝐂₃) # slower than direct compression
    B += compressed_kron³(𝐒₁₋╱𝟏ₑ)#, timer = timer)

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Setup C" begin
    # @timeit_debug timer "Initialise smaller matrices" begin

    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [(𝐒₂ * kron𝐒₁₋╱𝟏ₑ + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
            𝐒₂
            zeros(n₋ + nₑ, nₑ₋^2)];
            
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = choose_matrix_format(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, density_threshold = 0.0, min_length = 10)
        
    𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:] 
            zeros(n₋ + n + nₑ, nₑ₋^2)];

    aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋
    # aux = choose_matrix_format(aux, density_threshold = 1.0, min_length = 10)

    # end # timeit_debug
    # @timeit_debug timer "∇₃" begin   

    tmpkron = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔)

    𝐔∇₃ = ∇₃ * M₃.𝐔∇₃

    𝐗₃ = 𝐔∇₃ * tmpkron + 𝐔∇₃ * M₃.𝐏₁ₗ̂ * tmpkron * M₃.𝐏₁ᵣ̃ + 𝐔∇₃ * M₃.𝐏₂ₗ̂ * tmpkron * M₃.𝐏₂ᵣ̃
    
    # end # timeit_debug
    # @timeit_debug timer "∇₂ & ∇₁₊" begin

    𝐒₂₊╱𝟎 = choose_matrix_format(𝐒₂₊╱𝟎, density_threshold = 1.0, min_length = 10)

    tmpkron1 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎)
    tmpkron2 = ℒ.kron(M₂.𝛔, 𝐒₁₋╱𝟏ₑ)
    
    ∇₁₊ = choose_matrix_format(∇₁₊, density_threshold = 1.0, min_length = 10)

    𝐒₂₋╱𝟎 = [𝐒₂[i₋,:] ; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]

    # @timeit_debug timer "Step 1" begin
    out2 = mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎) # this help

    # end # timeit_debug
    # @timeit_debug timer "Step 2" begin

    out2 += ∇₂ * tmpkron1 * tmpkron2# |> findnz

    # end # timeit_debug  
    # @timeit_debug timer "Step 3" begin

    out2 += ∇₂ * tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ# |> findnz

    # end # timeit_debug
    # @timeit_debug timer "Step 4" begin

    # out2 += ∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₂₊╱𝟎 * M₂.𝛔)# |> findnz
    out2 += mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, collect(𝐒₂₊╱𝟎 * M₂.𝛔))# |> findnz

    # end # timeit_debug
    # @timeit_debug timer "Step 5" begin
        # out2 += ∇₁₊ * mat_mult_kron(𝐒₂, collect(𝐒₁₋╱𝟏ₑ), collect(𝐒₂₋╱𝟎))
        # out2 += mat_mult_kron(∇₁₊ * 𝐒₂, collect(𝐒₁₋╱𝟏ₑ), collect(𝐒₂₋╱𝟎))
        # out2 += ∇₁₊ * 𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0)
    out2 += ∇₁₊ * mat_mult_kron(𝐒₂, 𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)
    
    # end # timeit_debug
    # @timeit_debug timer "Mult" begin

    𝐗₃ += out2 * M₃.𝐏

    𝐗₃ *= M₃.𝐂₃

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "3rd Kronecker power" begin

    # 𝐗₃ += mat_mult_kron(∇₃, collect(aux), collect(ℒ.kron(aux, aux)), M₃.𝐂₃) # slower than direct compression
    𝐗₃ += ∇₃ * compressed_kron³(aux, rowmask = unique(findnz(∇₃)[2])) #, timer = timer)
    
    # end # timeit_debug
    # @timeit_debug timer "Mult 2" begin

    C = ∇₁₊𝐒₁➕∇₁₀lu \ 𝐗₃# * M₃.𝐂₃

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Solve sylvester equation" begin

    𝐒₃, solved = solve_sylvester_equation(A, B, C, 
                                            initial_guess = initial_guess,
                                            sylvester_algorithm = opts.sylvester_algorithm³,
                                            tol = opts.sylvester_tol,
                                            acceptance_tol = opts.sylvester_acceptance_tol,
                                            verbose = opts.verbose) # timer = timer)
    
    # end # timeit_debug
    # # @timeit_debug timer "Refine sylvester equation" begin

    # if !solved
    #     𝐒₃, solved = solve_sylvester_equation(A, B, C, 
    #                                             sylvester_algorithm = :doubling, 
    #                                             verbose = verbose, 
    #                                             initial_guess = initial_guess, 
    #                                             # tol = tol,
    #                                             timer = timer)
    # end

    if !solved
        return 𝐒₃, solved
    end

    # # end # timeit_debug
    # @timeit_debug timer "Post-process" begin

    # 𝐒₃ *= M₃.𝐔₃

    # 𝐒₃ = sparse(𝐒₃)

    # end # timeit_debug
    # end # timeit_debug

    return 𝐒₃, solved
end




function rrule(::typeof(calculate_third_order_solution), 
                ∇₁::AbstractMatrix{<: Real}, #first order derivatives
                ∇₂::SparseMatrixCSC{<: Real}, #second order derivatives
                ∇₃::SparseMatrixCSC{<: Real}, #third order derivatives
                𝑺₁::AbstractMatrix{<: Real}, #first order solution
                𝐒₂::SparseMatrixCSC{<: Real}, #second order solution
                M₂::second_order_auxilliary_matrices,  # aux matrices second order
                M₃::third_order_auxilliary_matrices;  # aux matrices third order
                T::timings,
                initial_guess::AbstractMatrix{Float64} = zeros(0,0),
                opts::CalculationOptions = merge_calculation_options())    

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
        return (∇₁₊𝐒₁➕∇₁₀, solved), x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    spinv = inv(∇₁₊𝐒₁➕∇₁₀lu)
    spinv = choose_matrix_format(spinv)

    # end # timeit_debug
    
    ∇₁₊ = @views ∇₁[:,1:n₊] * ℒ.I(n)[i₊,:]

    A = spinv * ∇₁₊

    # tmpkron = ℒ.kron(𝐒₁₋╱𝟏ₑ,M₂.𝛔)
    tmpkron = choose_matrix_format(ℒ.kron(𝐒₁₋╱𝟏ₑ,M₂.𝛔), density_threshold = 1.0)
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
    B = choose_matrix_format(M₃.𝐔₃ * B)

    # end # timeit_debug
    # @timeit_debug timer "3rd Kronecker power" begin

    B += compressed_kron³(𝐒₁₋╱𝟏ₑ) # , timer = timer)

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Setup C" begin
    # @timeit_debug timer "Initialise smaller matrices" begin

    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [(𝐒₂ * kron𝐒₁₋╱𝟏ₑ + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
            𝐒₂
            zeros(n₋ + nₑ, nₑ₋^2)];
            
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = choose_matrix_format(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, density_threshold = 0.0, min_length = 10)
        
    𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:] 
            zeros(n₋ + n + nₑ, nₑ₋^2)];

    aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋

    # end # timeit_debug
    # @timeit_debug timer "∇₃" begin

    tmpkron0 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)
    tmpkron22 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, tmpkron0 * M₂.𝛔)

    𝐔∇₃ = ∇₃ * M₃.𝐔∇₃

    𝐗₃ = 𝐔∇₃ * tmpkron22 + 𝐔∇₃ * M₃.𝐏₁ₗ̂ * tmpkron22 * M₃.𝐏₁ᵣ̃ + 𝐔∇₃ * M₃.𝐏₂ₗ̂ * tmpkron22 * M₃.𝐏₂ᵣ̃
    
    # end # timeit_debug
    # @timeit_debug timer "∇₂ & ∇₁₊" begin

    𝐒₂₊╱𝟎 = choose_matrix_format(𝐒₂₊╱𝟎, density_threshold = 1.0, min_length = 10)

    tmpkron1 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎)
    tmpkron2 = ℒ.kron(M₂.𝛔, 𝐒₁₋╱𝟏ₑ)
    
    ∇₁₊ = choose_matrix_format(∇₁₊, density_threshold = 1.0, min_length = 10)

    𝐒₂₋╱𝟎 = [𝐒₂[i₋,:] ; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]

    𝐒₂₋╱𝟎 = choose_matrix_format(𝐒₂₋╱𝟎, density_threshold = 1.0, min_length = 10)

    # @timeit_debug timer "Step 1" begin

    # tmpkron10 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)
    # out2 = ∇₂ * tmpkron10
    out2 = mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎) # this help  

    # end # timeit_debug
    # @timeit_debug timer "Step 2" begin

    out2 += ∇₂ * tmpkron1 * tmpkron2# |> findnz

    # end # timeit_debug  
    # @timeit_debug timer "Step 3" begin

    out2 += ∇₂ * tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ# |> findnz

    # end # timeit_debug
    # @timeit_debug timer "Step 4" begin

    𝐒₂₊╱𝟎𝛔 = 𝐒₂₊╱𝟎 * M₂.𝛔
    tmpkron11 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₂₊╱𝟎𝛔)
    out2 += ∇₂ * tmpkron11# |> findnz

    # end # timeit_debug
    # @timeit_debug timer "Step 5" begin

    tmpkron12 = ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)
    out2 += ∇₁₊ * 𝐒₂ * tmpkron12
    
    # end # timeit_debug
    # @timeit_debug timer "Mult" begin

    𝐗₃ += out2 * M₃.𝐏

    𝐗₃ *= M₃.𝐂₃

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "3rd Kronecker power aux" begin

    𝐗₃ += ∇₃ * compressed_kron³(aux, rowmask = unique(findnz(∇₃)[2])) # , timer = timer)
    𝐗₃ = choose_matrix_format(𝐗₃, density_threshold = 1.0, min_length = 10)

    # end # timeit_debug
    # @timeit_debug timer "Mult 2" begin

    C = spinv * 𝐗₃

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Solve sylvester equation" begin

    𝐒₃, solved = solve_sylvester_equation(A, B, C, 
                                            initial_guess = initial_guess,
                                            sylvester_algorithm = opts.sylvester_algorithm³,
                                            tol = opts.sylvester_tol,
                                            acceptance_tol = opts.sylvester_acceptance_tol,
                                            verbose = opts.verbose) # timer = timer)
    
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
        return (𝐒₃, solved), x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent() 
    end

    𝐒₃ = sparse(𝐒₃)

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

    tmpkron1t = choose_matrix_format(tmpkron1')# , density_threshold = 1.0)
    
    tmpkron2t = choose_matrix_format(tmpkron2')# , density_threshold = 1.0)
    
    tmpkron22t = choose_matrix_format(tmpkron22')# , density_threshold = 1.0)
    
    tmpkron12t = choose_matrix_format(tmpkron12')# , density_threshold = 1.0)
    
    𝐒₂t = choose_matrix_format(𝐒₂', density_threshold = 1.0) # this must be sparse otherwise tests fail
    
    kronaux = ℒ.kron(aux, aux)

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t = choose_matrix_format(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋')
    
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎t = choose_matrix_format(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎')
    
    tmpkron10t = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎t)

    # end # timeit_debug
    # end # timeit_debug
    
    function third_order_solution_pullback(∂𝐒₃_solved) 
        ∂∇₁ = zero(∇₁)
        ∂∇₂ = zero(∇₂)
        # ∂𝐔∇₃ = zero(𝐔∇₃)
        ∂∇₃ = zero(∇₃)
        ∂𝐒₁ = zero(𝐒₁)
        ∂𝐒₂ = zero(𝐒₂)
        ∂spinv = zero(spinv)
        ∂𝐒₁₋╱𝟏ₑ = zero(𝐒₁₋╱𝟏ₑ)
        ∂kron𝐒₁₋╱𝟏ₑ = zero(kron𝐒₁₋╱𝟏ₑ)
        ∂𝐒₁₊╱𝟎 = zero(𝐒₁₊╱𝟎)
        ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = zero(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋)
        ∂tmpkron = zero(tmpkron)
        ∂tmpkron22 = zero(tmpkron22)
        ∂kronaux = zero(kronaux)
        ∂aux = zero(aux)
        ∂tmpkron0 = zero(tmpkron0)
        ∂⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = zero(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)
        ∂𝐒₂₊╱𝟎 = zero(𝐒₂₊╱𝟎)
        ∂𝐒₂₊╱𝟎𝛔 = zero(𝐒₂₊╱𝟎𝛔)
        ∂∇₁₊ = zero(∇₁₊)
        ∂𝐒₂₋╱𝟎 = zero(𝐒₂₋╱𝟎)

        # @timeit_debug timer "Third order solution - pullback" begin

        # @timeit_debug timer "Solve sylvester equation" begin

        ∂𝐒₃ = ∂𝐒₃_solved[1]

        # ∂𝐒₃ *= 𝐔₃t
        
        ∂C, solved = solve_sylvester_equation(A', B', ∂𝐒₃,
                                                sylvester_algorithm = opts.sylvester_algorithm³,
                                                tol = opts.sylvester_tol,
                                                acceptance_tol = opts.sylvester_acceptance_tol,
                                                verbose = opts.verbose)

        if !solved
            return (𝐒₃, solved), x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent() 
        end

        ∂C = choose_matrix_format(∂C, density_threshold = 1.0)

        # end # timeit_debug
        # @timeit_debug timer "Step 0" begin

        ∂A = ∂C * B' * 𝐒₃'

        # ∂B = 𝐒₃' * A' * ∂C
        ∂B = choose_matrix_format(𝐒₃' * A' * ∂C, density_threshold = 1.0)

        # end # timeit_debug
        # @timeit_debug timer "Step 1" begin

        # C = spinv * 𝐗₃
        # ∂𝐗₃ = spinv' * ∂C * M₃.𝐂₃'
        ∂𝐗₃ = choose_matrix_format(spinv' * ∂C, density_threshold = 1.0)

        ∂spinv += ∂C * 𝐗₃'

        # 𝐗₃ = ∇₃ * compressed_kron³(aux, rowmask = unique(findnz(∇₃)[2]))
        # + (𝐔∇₃ * tmpkron22 
        # + 𝐔∇₃ * M₃.𝐏₁ₗ̂ * tmpkron22 * M₃.𝐏₁ᵣ̃ 
        # + 𝐔∇₃ * M₃.𝐏₂ₗ̂ * tmpkron22 * M₃.𝐏₂ᵣ̃
        # + ∇₂ * (tmpkron10 + tmpkron1 * tmpkron2 + tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ + tmpkron11) * M₃.𝐏
        # + ∇₁₊ * 𝐒₂ * tmpkron12 * M₃.𝐏) * M₃.𝐂₃

        # ∇₁₊ * 𝐒₂ * tmpkron12 * M₃.𝐏 * M₃.𝐂₃
        ∂∇₁₊ += ∂𝐗₃ * 𝐂₃t * 𝐏t * tmpkron12t * 𝐒₂t
        ∂𝐒₂ += ∇₁₊' * ∂𝐗₃ * 𝐂₃t * 𝐏t * tmpkron12t
        ∂tmpkron12 = 𝐒₂t * ∇₁₊' * ∂𝐗₃ * 𝐂₃t * 𝐏t

        # tmpkron12 = ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)
        fill_kron_adjoint!(∂𝐒₁₋╱𝟏ₑ, ∂𝐒₂₋╱𝟎, ∂tmpkron12, 𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)
        
        # end # timeit_debug
        # @timeit_debug timer "Step 2" begin
        
        # ∇₂ * (tmpkron10 + tmpkron1 * tmpkron2 + tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ + tmpkron11) * M₃.𝐏 * M₃.𝐂₃
        #improve this
        # ∂∇₂ += ∂𝐗₃ * 𝐂₃t * 𝐏t * (
        #    tmpkron10
        #  + tmpkron1 * tmpkron2
        #  + tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ
        #  + tmpkron11
        #  )'

        ∂∇₂ += ∂𝐗₃ * 𝐂₃t * 𝐏t * tmpkron10t
        # ∂∇₂ += mat_mult_kron(∂𝐗₃ * 𝐂₃t * 𝐏t, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎t)
        # ∂∇₂ += ∂𝐗₃ * 𝐂₃t * 𝐏t * (tmpkron1 * tmpkron2)'
        ∂∇₂ += ∂𝐗₃ * 𝐂₃t * 𝐏t * tmpkron2t * tmpkron1t

        # ∂∇₂ += ∂𝐗₃ * 𝐂₃t * 𝐏t * (tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ)'
        ∂∇₂ += ∂𝐗₃ * 𝐂₃t * 𝐏t * M₃.𝐏₁ᵣ' * tmpkron2t * M₃.𝐏₁ₗ' * tmpkron1t

        ∂∇₂ += ∂𝐗₃ * 𝐂₃t * 𝐏t * tmpkron11'

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
        fill_kron_adjoint!(∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ∂tmpkron0, ∂tmpkron22, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔)

        ∂kron𝐒₁₊╱𝟎 = ∂tmpkron0 * 𝛔t

        fill_kron_adjoint!(∂𝐒₁₊╱𝟎, ∂𝐒₁₊╱𝟎, ∂kron𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)

        # -∇₃ * ℒ.kron(ℒ.kron(aux, aux), aux)
        # ∂∇₃ += ∂𝐗₃ * ℒ.kron(ℒ.kron(aux', aux'), aux')
        # A_mult_kron_power_3_B!(∂∇₃, ∂𝐗₃, aux') # not a good idea because filling an existing matrix one by one is slow
        # ∂∇₃ += A_mult_kron_power_3_B(∂𝐗₃, aux') # this is slower somehow
        
        # end # timeit_debug
        # @timeit_debug timer "Step 5" begin
            
        # this is very slow
        ∂∇₃ += ∂𝐗₃ * compressed_kron³(aux', rowmask = unique(findnz(∂𝐗₃)[2])) # , timer = timer)
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

        return NoTangent(), ∂∇₁, ∂∇₂, ∂∇₃, ∂𝑺₁, ∂𝐒₂, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return (𝐒₃, solved), third_order_solution_pullback
end

