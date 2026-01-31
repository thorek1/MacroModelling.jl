@stable default_mode = "disable" begin

function calculate_first_order_solution(∇₁::Matrix{R},
                                        constants::constants,
                                        qme_ws::qme_workspace{R,S},
                                        sylv_ws::sylvester_workspace{R,S},
                                        first_order_ws::first_order_workspace{R};
                                        opts::CalculationOptions = merge_calculation_options(),
                                        initial_guess::AbstractMatrix{R} = zeros(0,0))::Tuple{Matrix{R}, Matrix{R}, Bool} where {R <: AbstractFloat, S <: Real}
    # @timeit_debug timer "Calculate 1st order solution" begin
    # @timeit_debug timer "Preprocessing" begin

    T = constants.post_model_macro
    idx_constants = ensure_first_order_constants!(constants)

    dynIndex = idx_constants.dyn_index
    reverse_dynamic_order = idx_constants.reverse_dynamic_order
    comb = idx_constants.comb
    future_not_past_and_mixed_in_comb = idx_constants.future_not_past_and_mixed_in_comb
    past_not_future_and_mixed_in_comb = idx_constants.past_not_future_and_mixed_in_comb
    Ir = idx_constants.Ir
    
    ∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
    ∇₀ = ∇₁[:,idx_constants.nabla_zero_cols]
    ∇₋ = @view ∇₁[:,idx_constants.nabla_minus_cols]
    ∇ₑ = ∇₁[:,idx_constants.nabla_e_start:end]
    
    # end # timeit_debug
    # @timeit_debug timer "Invert ∇₀" begin

    Q    = ℒ.qr!(∇₀[:,T.present_only_idx])

    # Use preallocated workspace matrices for Q.Q' * ∇
    A₊ = first_order_ws.A₊
    A₀ = first_order_ws.A₀
    A₋ = first_order_ws.A₋
    ℒ.mul!(A₊, Q.Q', ∇₊)
    ℒ.mul!(A₀, Q.Q', ∇₀)
    ℒ.mul!(A₋, Q.Q', ∇₋)
    
    # end # timeit_debug
    # @timeit_debug timer "Sort matrices" begin

    # Use direct (allocating) construction for sorted matrices to preserve correctness
    Ã₊ = @view(A₊[dynIndex,:]) * Ir[future_not_past_and_mixed_in_comb,:]
    Ã₀ = @view(A₀[dynIndex, comb])
    Ã₋ = @view(A₋[dynIndex,:]) * Ir[past_not_future_and_mixed_in_comb,:]

    # end # timeit_debug
    # @timeit_debug timer "Quadratic matrix equation solve" begin

    sol, solved = solve_quadratic_matrix_equation(Ã₊, Ã₀, Ã₋, constants, qme_ws;
                                                    initial_guess = initial_guess,
                                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                                    tol = opts.tol.qme_tol,
                                                    acceptance_tol = opts.tol.qme_acceptance_tol,
                                                    verbose = opts.verbose)

    if !solved
        if opts.verbose println("Quadratic matrix equation solution failed.") end
        return zeros(R, T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false
    end

    # end # timeit_debug
    # @timeit_debug timer "Postprocessing" begin
    # @timeit_debug timer "Setup matrices" begin

    sol_compact = @view sol[reverse_dynamic_order, past_not_future_and_mixed_in_comb]

    D = @view sol_compact[end - T.nFuture_not_past_and_mixed + 1:end, :]

    L = @view sol[T.past_not_future_and_mixed_in_present_but_not_only_idx, past_not_future_and_mixed_in_comb]

    Ā₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_only_idx]
    A₊ᵤ  = @view A₊[1:T.nPresent_only,:]
    Ã₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_but_not_only_idx]
    A₋ᵤ  = @view A₋[1:T.nPresent_only,:]

    # end # timeit_debug
    # @timeit_debug timer "Invert Ā₀ᵤ" begin

    Ā̂₀ᵤ = ℒ.lu!(Ā₀ᵤ, check = false)

    if !ℒ.issuccess(Ā̂₀ᵤ)
        if opts.verbose println("Factorisation of Ā₀ᵤ failed") end
        return zeros(R, T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false
    end

    # A    = vcat(-(Ā̂₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ)), sol)
    # Use preallocated workspace for nₚ₋ = A₊ᵤ * D
    nₚ₋ = first_order_ws.nₚ₋
    if T.nPresent_only > 0
        ℒ.mul!(@view(A₋[1:T.nPresent_only,:]), Ã₀ᵤ, @view(sol[:,past_not_future_and_mixed_in_comb]), 1, 1)
        ℒ.mul!(nₚ₋, A₊ᵤ, D)
        ℒ.mul!(@view(A₋[1:T.nPresent_only,:]), nₚ₋, L, 1, 1)
        ℒ.ldiv!(Ā̂₀ᵤ, @view(A₋[1:T.nPresent_only,:]))
        ℒ.rmul!(@view(A₋[1:T.nPresent_only,:]), -1)
    end
    
    # Use preallocated workspace for vcat: A = vcat(A₋ᵤ, sol_compact)[T.reorder,:]
    A_vcat = first_order_ws.A_vcat
    A_vcat[1:T.nPresent_only, :] .= A₋ᵤ
    A_vcat[T.nPresent_only+1:end, :] .= sol_compact
    # Apply reorder in-place to result
    result = first_order_ws.result
    @views result[:, 1:T.nPast_not_future_and_mixed] .= A_vcat[T.reorder, :]

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Exogenous part solution" begin

    # M = A[T.future_not_past_and_mixed_idx,:] * expand_past
    # Use preallocated M
    M = first_order_ws.M
    ℒ.mul!(M, @view(result[T.future_not_past_and_mixed_idx, 1:T.nPast_not_future_and_mixed]), idx_constants.expand_past)

    ℒ.mul!(∇₀, @view(∇₁[:,1:T.nFuture_not_past_and_mixed]), M, 1, 1)

    C = ℒ.lu!(∇₀, check = false)
    
    if !ℒ.issuccess(C)
        if opts.verbose println("Factorisation of ∇₀ failed") end
        return zeros(R, T.nVars,T.nPast_not_future_and_mixed + T.nExo), sol, false
    end
    
    ℒ.ldiv!(C, ∇ₑ)
    ℒ.rmul!(∇ₑ, -1)

    # end # timeit_debug
    # end # timeit_debug

    # Use preallocated result for hcat
    result[:, T.nPast_not_future_and_mixed+1:end] .= ∇ₑ

    return result, sol, true
end


function calculate_second_order_solution(∇₁::AbstractMatrix{S}, #first order derivatives
                                            ∇₂::SparseMatrixCSC{S}, #second order derivatives
                                            𝑺₁::AbstractMatrix{S},#first order solution
                                            constants::constants,
                                            workspaces::workspaces;
                                            initial_guess::AbstractMatrix{R} = zeros(0,0),
                                            opts::CalculationOptions = merge_calculation_options())::Union{Tuple{Matrix{S}, Bool}, Tuple{SparseMatrixCSC{S, Int}, Bool}} where {R <: Real, S <: Real}
    if !(eltype(workspaces.second_order.Ŝ) == S)
        workspaces.second_order = Higher_order_workspace(T = S)
    end
    ℂ = workspaces.second_order
    M₂ = constants.second_order
    T = constants.post_model_macro
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

    𝐒₂, solved = solve_sylvester_equation(A, B, C, ℂ.sylvester_workspace,
                                            initial_guess = initial_guess,
                                            sylvester_algorithm = opts.sylvester_algorithm²,
                                            tol = opts.tol.sylvester_tol,
                                            acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                            verbose = opts.verbose)

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

    𝐒₂ = choose_matrix_format(𝐒₂, multithreaded = false)

    # end # timeit_debug
    # end # timeit_debug

    return 𝐒₂, solved
end


function calculate_third_order_solution(∇₁::AbstractMatrix{S}, #first order derivatives
                                            ∇₂::SparseMatrixCSC{S}, #second order derivatives
                                            ∇₃::SparseMatrixCSC{S}, #third order derivatives
                                            𝑺₁::AbstractMatrix{S}, #first order solution
                                            𝐒₂::SparseMatrixCSC{S}, #second order solution
                                            constants::constants,
                                            workspaces::workspaces;
                                            initial_guess::AbstractMatrix{R} = zeros(0,0),
                                            opts::CalculationOptions = merge_calculation_options())::Union{Tuple{Matrix{S}, Bool}, Tuple{SparseMatrixCSC{S, Int}, Bool}}  where {S <: Real,R <: Real}
    if !(eltype(workspaces.third_order.Ŝ) == S)
        workspaces.third_order = Higher_order_workspace(T = S)
    end
    ℂ = workspaces.third_order
    M₂ = constants.second_order
    M₃ = constants.third_order
    T = constants.post_model_macro
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

    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                ℒ.I(nₑ₋)[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]] #|> sparse

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)]# |> sparse
    𝐒₁₊╱𝟎 = choose_matrix_format(𝐒₁₊╱𝟎, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.I(n)[i₋,:] - ∇₁[:,range(1,n) .+ n₊]

    # end # timeit_debug
    # @timeit_debug timer "Invert matrix" begin

    ∇₁₊𝐒₁➕∇₁₀lu = ℒ.lu(∇₁₊𝐒₁➕∇₁₀, check = false)

    if !ℒ.issuccess(∇₁₊𝐒₁➕∇₁₀lu)
        if opts.verbose println("Second order solution: inversion failed") end
        return (∇₁₊𝐒₁➕∇₁₀, false)#, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
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
    B = choose_matrix_format(M₃.𝐔₃ * B, tol = opts.tol.droptol, multithreaded = false)

    # end # timeit_debug
    # @timeit_debug timer "3rd Kronecker power" begin
    # B += mat_mult_kron(M₃.𝐔₃, collect(𝐒₁₋╱𝟏ₑ), collect(ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ)), M₃.𝐂₃) # slower than direct compression

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
    # aux = choose_matrix_format(aux, density_threshold = 1.0, min_length = 10)

    # end # timeit_debug
    # @timeit_debug timer "∇₃" begin   

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

    # tmpkron = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔)

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
    out2 += mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, collect(𝐒₂₊╱𝟎 * M₂.𝛔), sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc3)# |> findnz

    # end # timeit_debug
    # @timeit_debug timer "Step 5" begin
        # out2 += ∇₁₊ * mat_mult_kron(𝐒₂, collect(𝐒₁₋╱𝟏ₑ), collect(𝐒₂₋╱𝟎))
        # out2 += mat_mult_kron(∇₁₊ * 𝐒₂, collect(𝐒₁₋╱𝟏ₑ), collect(𝐒₂₋╱𝟎))
        # out2 += ∇₁₊ * 𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)

    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0, tol = opts.tol.droptol)
    out2 += ∇₁₊ * mat_mult_kron(𝐒₂, 𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc4)
    
    # end # timeit_debug
    # @timeit_debug timer "Mult" begin
    # ℒ.mul!(𝐗₃, out2, M₃.𝐏, 1, 1) # less memory but way slower; .+= also more memory and slower
    𝐗₃ += out2 * M₃.𝐏

    𝐗₃ *= M₃.𝐂₃

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "3rd Kronecker power" begin

    # 𝐗₃ += mat_mult_kron(∇₃, collect(aux), collect(ℒ.kron(aux, aux)), M₃.𝐂₃) # slower than direct compression
    𝐗₃ += ∇₃ * compressed_kron³(aux, rowmask = unique(findnz(∇₃)[2]), tol = opts.tol.droptol, sparse_preallocation = ℂ.tmp_sparse_prealloc5) #, timer = timer)
    
    # end # timeit_debug
    # @timeit_debug timer "Mult 2" begin

    C = ∇₁₊𝐒₁➕∇₁₀lu \ 𝐗₃# * M₃.𝐂₃

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
    #                                             verbose = verbose, 
    #                                             initial_guess = initial_guess, 
    #                                             # tol = tol,
    #                                             timer = timer)
    # end

    # if !solved
    #     return 𝐒₃, solved
    # end

    # # end # timeit_debug
    # @timeit_debug timer "Post-process" begin

    # 𝐒₃ *= M₃.𝐔₃

    𝐒₃ = choose_matrix_format(𝐒₃, multithreaded = false, tol = opts.tol.droptol)

    # end # timeit_debug
    # end # timeit_debug

    return 𝐒₃, solved
end

end # dispatch_doctor

