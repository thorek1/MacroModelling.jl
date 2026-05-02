

function calculate_first_order_solution(∇₁::Matrix{R},
                                        constants::constants,
                                        workspaces::workspaces,
                                        cache::caches;
                                        opts::CalculationOptions = merge_calculation_options(),
                                        use_fastlapack_qr::Bool = true,
                                        use_fastlapack_lu::Bool = true,
                                        initial_guess::AbstractMatrix{R} = zeros(0,0),
                                        parameter_values::AbstractVector{<:Real} = Float64[],
                                        caching::Bool = true)::Tuple{Matrix{R}, Matrix{R}, Bool} where {R <: AbstractFloat}
    # Cache hit: return cached first-order solution if valid for current parameters
    if caching && R === Float64 && !isempty(parameter_values) &&
       cache_valid_for_parameters(cache.valid_for.first_order_solution, parameter_values)
        S₁_cached = cache.first_order_solution_matrix
        qme_cached = cache.qme_solution
        if S₁_cached isa Matrix{R} && !isempty(S₁_cached) && qme_cached isa Matrix{R} && !isempty(qme_cached)
            return S₁_cached, qme_cached, true
        end
    end
    # @timeit_debug timer "Calculate 1st order solution" begin
    # @timeit_debug timer "Preprocessing" begin

    T = constants.post_model_macro
    idx_constants = ensure_first_order_constants!(constants)

    dynIndex = idx_constants.dyn_index
    reverse_dynamic_order = idx_constants.reverse_dynamic_order
    comb = idx_constants.comb
    future_not_past_and_mixed_in_comb = idx_constants.future_not_past_and_mixed_in_comb
    past_not_future_and_mixed_in_comb = idx_constants.past_not_future_and_mixed_in_comb
    past_not_future_and_mixed_in_present_but_not_only = idx_constants.past_not_future_and_mixed_in_present_but_not_only
    Ir = idx_constants.Ir

    qme_ws = workspaces.first_order

    ensure_first_order_workspace_buffers!(qme_ws, T, length(dynIndex), length(comb))

    ∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
    ∇₀ = qme_ws.∇₀
    copyto!(∇₀, @view(∇₁[:,idx_constants.nabla_zero_cols]))
    ∇₋ = @view ∇₁[:,idx_constants.nabla_minus_cols]
    ∇ₑ = qme_ws.∇ₑ
    copyto!(∇ₑ, @view(∇₁[:,idx_constants.nabla_e_start:end]))
    
    # end # timeit_debug
    # @timeit_debug timer "Invert ∇₀" begin

    A₊ = qme_ws.𝐀₊
    A₀ = qme_ws.𝐀₀
    A₋ = qme_ws.𝐀₋
    ∇₀_present = @view ∇₀[:, T.present_only_idx]
    # Old way (≤v0.1.42):
    #   Q = qr(∇₀[:, present_only_idx])
    #   A₊ = Q' * ∇₊;  A₀ = Q' * ∇₀;  A₋ = Q' * ∇₋
    # Current code reuses QR/ORM workspaces to avoid allocations.
    qr_factors, qr_ws = ensure_first_order_fast_qr_workspace!(qme_ws, ∇₀_present)
    Q = factorize_qr!(∇₀_present, qr_factors, qr_ws;                 # Q = qr(∇₀_present)
                        use_fastlapack_qr = use_fastlapack_qr)

    qme_ws.fast_qr_orm_ws_plus, qme_ws.fast_qr_orm_dims_plus = apply_qr_transpose_left!(A₊, ∇₊, Q,           # A₊ = Q' * ∇₊
                                                                                        qme_ws.fast_qr_orm_ws_plus,
                                                                                        qme_ws.fast_qr_orm_dims_plus,
                                                                                        qr_ws;
                                                                                        use_fastlapack_qr = use_fastlapack_qr)
    qme_ws.fast_qr_orm_ws_zero, qme_ws.fast_qr_orm_dims_zero = apply_qr_transpose_left!(A₀, ∇₀, Q,           # A₀ = Q' * ∇₀
                                                                                        qme_ws.fast_qr_orm_ws_zero,
                                                                                        qme_ws.fast_qr_orm_dims_zero,
                                                                                        qr_ws;
                                                                                        use_fastlapack_qr = use_fastlapack_qr)
    qme_ws.fast_qr_orm_ws_minus, qme_ws.fast_qr_orm_dims_minus = apply_qr_transpose_left!(A₋, ∇₋, Q,         # A₋ = Q' * ∇₋
                                                                                           qme_ws.fast_qr_orm_ws_minus,
                                                                                           qme_ws.fast_qr_orm_dims_minus,
                                                                                           qr_ws;
                                                                                           use_fastlapack_qr = use_fastlapack_qr)
    
    # end # timeit_debug
    # @timeit_debug timer "Sort matrices" begin

    Ã₊ = qme_ws.𝐀̃₊
    ℒ.mul!(Ã₊, @view(A₊[dynIndex,:]), @view(Ir[future_not_past_and_mixed_in_comb,:]))  # Ã₊ = A₊[dynIndex,:] * Ir

    Ã₀ = qme_ws.𝐀̃₀
    copyto!(Ã₀, @view(A₀[dynIndex, comb]))

    Ã₋ = qme_ws.𝐀̃₋
    ℒ.mul!(Ã₋, @view(A₋[dynIndex,:]), @view(Ir[past_not_future_and_mixed_in_comb,:]))  # Ã₋ = A₋[dynIndex,:] * Ir

    # end # timeit_debug
    # @timeit_debug timer "Quadratic matrix equation solve" begin

    sol, solved = solve_quadratic_matrix_equation(Ã₊, Ã₀, Ã₋, constants, workspaces, cache;
                                                    initial_guess = initial_guess,
                                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                                    use_fastlapack_lu = use_fastlapack_lu,
                                                    tol = opts.tol.first_order.qme,
                                                    verbose = opts.verbose,
                                                    caching = caching)

    if !solved
        if opts.verbose println("Quadratic matrix equation solution failed.") end
        return fill(R(NaN), T.nVars, T.nPast_not_future_and_mixed + T.nExo), sol, false
    end

    # Detect unit roots from QME solution eigenvalues when the Schur QME path
    # did not already set the flag (e.g. doubling solver was used).
    if caching && !cache.has_unit_roots
        detect_unit_roots_from_solution!(cache, sol)
    end

    # end # timeit_debug
    # @timeit_debug timer "Postprocessing" begin
    # @timeit_debug timer "Setup matrices" begin

    sol_compact = @view sol[reverse_dynamic_order, past_not_future_and_mixed_in_comb]

    n_dyn = length(reverse_dynamic_order)
    𝐃 = @view sol[@view(reverse_dynamic_order[n_dyn - T.nFuture_not_past_and_mixed + 1:n_dyn]), past_not_future_and_mixed_in_comb]

    L = @view sol[past_not_future_and_mixed_in_present_but_not_only, past_not_future_and_mixed_in_comb]

    Ā₀ᵤ = qme_ws.𝐀̄₀ᵤ
    copyto!(Ā₀ᵤ, @view(A₀[1:T.nPresent_only, T.present_only_idx]))

    A₊ᵤ = qme_ws.𝐀₊ᵤ
    copyto!(A₊ᵤ, @view(A₊[1:T.nPresent_only,:]))

    Ã₀ᵤ = qme_ws.𝐀̃₀ᵤ
    copyto!(Ã₀ᵤ, @view(A₀[1:T.nPresent_only, T.present_but_not_only_idx]))

    A₋ᵤ = qme_ws.𝐀₋ᵤ
    copyto!(A₋ᵤ, @view(A₋[1:T.nPresent_only,:]))

    # end # timeit_debug
    # @timeit_debug timer "Invert Ā₀ᵤ" begin

    qme_ws.fast_lu_ws_a0u, qme_ws.fast_lu_dims_a0u, solved_Ā₀ᵤ, Ā̂₀ᵤ = factorize_lu!(Ā₀ᵤ,
                                                                                       qme_ws.fast_lu_ws_a0u,
                                                                                       qme_ws.fast_lu_dims_a0u;
                                                                                       use_fastlapack_lu = use_fastlapack_lu)

    if !solved_Ā₀ᵤ
        if opts.verbose println("Factorisation of Ā₀ᵤ failed") end
        return fill(R(NaN), T.nVars, T.nPast_not_future_and_mixed + T.nExo), sol, false
    end

    # Old way (≤v0.1.42): A₋ᵤ = -(Ā₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol + A₋ᵤ))
    if T.nPresent_only > 0
        ℒ.mul!(A₋ᵤ, Ã₀ᵤ, @view(sol[:,past_not_future_and_mixed_in_comb]), 1, 1)  # A₋ᵤ = A₋ᵤ + Ã₀ᵤ * sol
        nₚ₋ = qme_ws.𝐧ₚ₋
        ℒ.mul!(nₚ₋, A₊ᵤ, 𝐃)                                                    # nₚ₋ = A₊ᵤ * D
        ℒ.mul!(A₋ᵤ, nₚ₋, L, 1, 1)                                                # A₋ᵤ = A₋ᵤ + nₚ₋ * L
        solve_lu_left!(Ā₀ᵤ, A₋ᵤ, qme_ws.fast_lu_ws_a0u, Ā̂₀ᵤ;                 # A₋ᵤ = Ā₀ᵤ \ A₋ᵤ
                       use_fastlapack_lu = use_fastlapack_lu)
        ℒ.rmul!(A₋ᵤ, -1)                                                       # A₋ᵤ = -A₋ᵤ
    end

    A = qme_ws.𝐀
    # Old way (≤v0.1.42): A = vcat(A₋ᵤ, sol_compact)[reorder, :]
    # Expanded loop below writes into preallocated A without temporary concatenation.
    n_cols = size(A, 2)
    
    for i in 1:T.nVars
        src = T.reorder[i]
        if src <= T.nPresent_only
            for j in 1:n_cols
                @inbounds A[i, j] = A₋ᵤ[src, j]
            end
        else
            src_idx = src - T.nPresent_only
            for j in 1:n_cols
                @inbounds A[i, j] = sol_compact[src_idx, j]
            end
        end
    end

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Exogenous part solution" begin

    M = qme_ws.𝐌
    # Old way (≤v0.1.42):
    #   M = A[future_idx, :] * expand_past
    #   ∇₀ = ∇₊ * M + ∇₀
    ℒ.mul!(M, @view(A[T.future_not_past_and_mixed_idx,:]), idx_constants.expand_past)  # M = A[future_idx,:] * expand_past

    ℒ.mul!(∇₀, @view(∇₁[:,1:T.nFuture_not_past_and_mixed]), M, 1, 1)                 # ∇₀ = ∇₊ * M + ∇₀

    # Old way (≤v0.1.42): C = lu(∇₀)
    qme_ws.fast_lu_ws_nabla0, qme_ws.fast_lu_dims_nabla0, solved_∇₀, C = factorize_lu!(∇₀,
                                                                                         qme_ws.fast_lu_ws_nabla0,
                                                                                         qme_ws.fast_lu_dims_nabla0;
                                                                                         use_fastlapack_lu = use_fastlapack_lu)

    if !solved_∇₀
        if opts.verbose println("Factorisation of ∇₀ failed") end
        return fill(R(NaN), T.nVars, T.nPast_not_future_and_mixed + T.nExo), sol, false
    end

    # Old way (≤v0.1.42): ∇ₑ = -(∇₀ \ ∇ₑ)
    solve_lu_left!(∇₀, ∇ₑ, qme_ws.fast_lu_ws_nabla0, C;                                # ∇ₑ = ∇₀ \ ∇ₑ
                   use_fastlapack_lu = use_fastlapack_lu)
    ℒ.rmul!(∇ₑ, -1)

    # end # timeit_debug
    # end # timeit_debug

    n_rows = size(A, 1)
    n_cols_A = size(A, 2)
    n_cols_ϵ = size(∇ₑ, 2)
    total_cols = n_cols_A + n_cols_ϵ

    S₁ = if caching
        S₁_existing = cache.first_order_solution_matrix
        if S₁_existing isa Matrix{R} && size(S₁_existing) == (n_rows, total_cols)
            copyto!(@view(S₁_existing[:, 1:n_cols_A]), A)
            copyto!(@view(S₁_existing[:, n_cols_A+1:total_cols]), ∇ₑ)
            S₁_existing
        else
            S₁_tmp = hcat(A, ∇ₑ)
            cache.first_order_solution_matrix = S₁_tmp
            S₁_tmp
        end
    else
        hcat(A, ∇ₑ)
    end

    # Stamp cache validity for current parameters
    if caching && !isempty(parameter_values)
        cache.valid_for.first_order_solution = Float64.(primal.(parameter_values))
    end

    return S₁, sol, true
end


@unstable function calculate_second_order_solution(∇₁::AbstractMatrix{S}, #first order derivatives
                                            ∇₂::SparseMatrixCSC{S}, #second order derivatives
                                            𝑺₁::AbstractMatrix{S},#first order solution
                                            constants::constants,
                                            workspaces::workspaces,
                                            cache::caches;
                                            initial_guess::AbstractMatrix{R} = zeros(0,0),
                                            opts::CalculationOptions = merge_calculation_options(),
                                            parameter_values::AbstractVector{<:Real} = Float64[],
                                            caching::Bool = true)::Union{Tuple{Matrix{S}, Bool}, Tuple{SparseMatrixCSC{S, Int}, Bool}} where {R <: Real, S <: Real}
    # Cache hit: return cached second-order solution if valid for current parameters
    if caching && S === Float64 && !isempty(parameter_values) &&
       cache_valid_for_parameters(cache.valid_for.second_order_solution, parameter_values)
        cached = cache.second_order_solution
        if cached isa Matrix{S} && !isempty(cached)
            return cached, true
        end
    end
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

    ensure_higher_order_solution_buffers!(ℂ, n, nₑ₋)

    initial_guess_sylv = if length(initial_guess) == 0
        zeros(S, 0, 0)
    elseif eltype(initial_guess) <: AbstractFloat
        initial_guess isa Matrix{S} ? initial_guess : Matrix{S}(initial_guess)
    else
        zeros(S, 0, 0)
    end

    # @timeit_debug timer "Setup matrices" begin

    # 1st order solution
    𝐒₁ = ℂ.𝐒₁::Matrix{S}
    copyto!(@view(𝐒₁[:,1:n₋]), @view(𝑺₁[:,1:n₋]))
    fill!(@view(𝐒₁[:,n₋+1]), zero(S))
    copyto!(@view(𝐒₁[:,n₋+2:end]), @view(𝑺₁[:,n₋+1:end]))
    # droptol!(𝐒₁,tol)
    
    𝐒₁₋╱𝟏ₑ = ℂ.𝐒₁₋╱𝟏ₑ::Matrix{S}
    copyto!(@view(𝐒₁₋╱𝟏ₑ[1:n₋,:]), @view(𝐒₁[i₋,:]))
    fill!(@view(𝐒₁₋╱𝟏ₑ[n₋+1:end,:]), zero(S))
    @inbounds 𝐒₁₋╱𝟏ₑ[n₋+1,n₋+1] = one(S)
    # droptol!(𝐒₁₋╱𝟏ₑ,tol)
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 1.0)

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                ℒ.I(nₑ₋)[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]] #|> sparse
    # droptol!(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,tol)

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)]# |> sparse
    # droptol!(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,tol)

    ∇₁₊𝐒₁➕∇₁₀ = collect(@views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * M₂.𝐈ₙ₋ - ∇₁[:,range(1,n) .+ n₊])

    # end # timeit_debug

    # @timeit_debug timer "Invert matrix" begin

    qme_ws = workspaces.first_order

    if S === Float64
        qme_ws.fast_lu_ws_nabla0, qme_ws.fast_lu_dims_nabla0, solved_∇lu, lu_handle =
            factorize_lu!(∇₁₊𝐒₁➕∇₁₀, qme_ws.fast_lu_ws_nabla0, qme_ws.fast_lu_dims_nabla0)

        if !solved_∇lu
            if opts.verbose println("Second order solution: inversion failed") end
            return ∇₁₊𝐒₁➕∇₁₀, false
        end
    else
        ∇₁₊𝐒₁➕∇₁₀lu = ℒ.lu(∇₁₊𝐒₁➕∇₁₀, check = false)

        if !ℒ.issuccess(∇₁₊𝐒₁➕∇₁₀lu)
            if opts.verbose println("Second order solution: inversion failed") end
            return ∇₁₊𝐒₁➕∇₁₀, false
        end
    end

    # spinv = inv(∇₁₊𝐒₁➕∇₁₀)
    # spinv = choose_matrix_format(spinv)

    # end # timeit_debug
    # @timeit_debug timer "Setup second order matrices" begin
    # @timeit_debug timer "A" begin

    ∇₁₊ = @views ∇₁[:,1:n₊] * M₂.𝐈ₙ₊

    if S === Float64
        A = ∇₁₊
        solve_lu_left!(∇₁₊𝐒₁➕∇₁₀, A, qme_ws.fast_lu_ws_nabla0, lu_handle) # A = ∇₁₊𝐒₁➕∇₁₀ \ ∇₁₊
    else
        A = ∇₁₊𝐒₁➕∇₁₀lu \ ∇₁₊
    end
    
    # end # timeit_debug
    # @timeit_debug timer "C" begin

    # Build first forcing term directly in compressed Hessian space:
    #   ∇₂ * compressed_kron²(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋)
    # This skips explicit right-compression by M₂.𝐂₂ for this term.
    kron_compressed = compressed_kron²(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,
                                        rowmask = M₂.∇₂_nonempty_col_as_kron_rowmask,
                                        sparse_preallocation = ℂ.tmp_sparse_prealloc2)

    term1 = ∇₂ * kron_compressed

    # Build second forcing term in compressed Hessian space with extra pruning.
    # We only keep compressed-kron columns that can survive right multiplication by σc₂.
    kron_sigma_compressed = compressed_kron²(𝐒₁₊╱𝟎,
                                            rowmask = M₂.∇₂_nonempty_col_as_kron_rowmask,
                                            colmask = M₂.𝛔𝐂₂_nonempty_row_as_kron_colmask,
                                            sparse_preallocation = ℂ.tmp_sparse_prealloc3)

    term2 = (∇₂ * kron_sigma_compressed) * M₂.𝛔c₂

    ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = term1 + term2

    if S === Float64
        C = collect(∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹)
        solve_lu_left!(∇₁₊𝐒₁➕∇₁₀, C, qme_ws.fast_lu_ws_nabla0, lu_handle)
    else
        C = ∇₁₊𝐒₁➕∇₁₀lu \ ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹
    end

    # end # timeit_debug
    # @timeit_debug timer "B" begin

    # 𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0)
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0)
    B = compressed_kron²(𝐒₁₋╱𝟏ₑ, sparse_preallocation = ℂ.tmp_sparse_prealloc1) + M₂.𝛔c₂

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Solve sylvester equation" begin

    𝐒₂, solved = solve_sylvester_equation(A, B, C, ℂ.sylvester_workspace,
                                            initial_guess = initial_guess_sylv,
                                            sylvester_algorithm = opts.sylvester_algorithm²,
                                            preconditioner = opts.sylvester_preconditioner,
                                            tol = opts.tol.second_order.sylvester,
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

    if solved && caching && S === Float64
        if 𝐒₂ isa Matrix{S} && cache.second_order_solution isa Matrix{S} && size(cache.second_order_solution) == size(𝐒₂)
            copyto!(cache.second_order_solution, 𝐒₂)
        elseif 𝐒₂ isa SparseMatrixCSC{S, Int} && cache.second_order_solution isa SparseMatrixCSC{S, Int} &&
               size(cache.second_order_solution) == size(𝐒₂) &&
               cache.second_order_solution.colptr == 𝐒₂.colptr &&
               cache.second_order_solution.rowval == 𝐒₂.rowval
            copyto!(cache.second_order_solution.nzval, 𝐒₂.nzval)
        else
            cache.second_order_solution = copy(𝐒₂)
        end
        if !isempty(parameter_values)
            cache.valid_for.second_order_solution = Float64.(primal.(parameter_values))
            cache.valid_for.pruned_second_order_solution = Float64[]
        end
    end

    return 𝐒₂, solved
end


@unstable function calculate_third_order_solution(∇₁::AbstractMatrix{S}, #first order derivatives
                                            ∇₂::SparseMatrixCSC{S}, #second order derivatives
                                            ∇₃::SparseMatrixCSC{S}, #third order derivatives
                                            𝑺₁::AbstractMatrix{S}, #first order solution
                                            𝐒₂::AbstractMatrix{S}, #second order solution (compressed)
                                            constants::constants,
                                            workspaces::workspaces,
                                            cache::caches;
                                            initial_guess::AbstractMatrix{R} = zeros(0,0),
                                            opts::CalculationOptions = merge_calculation_options(),
                                            parameter_values::AbstractVector{<:Real} = Float64[],
                                            caching::Bool = true)::Union{Tuple{Matrix{S}, Bool}, Tuple{SparseMatrixCSC{S, Int}, Bool}}  where {S <: Real,R <: Real}
    # Cache hit: return cached third-order solution if valid for current parameters
    if caching && S === Float64 && !isempty(parameter_values) &&
       cache_valid_for_parameters(cache.valid_for.third_order_solution, parameter_values)
        cached = cache.third_order_solution
        if cached isa Matrix{S} && !isempty(cached)
            return cached, true
        end
    end
    if !(eltype(workspaces.third_order.Ŝ) == S)
        workspaces.third_order = Higher_order_workspace(T = S)
    end
    ℂ = workspaces.third_order
    M₂ = constants.second_order
    M₃ = constants.third_order
    T = constants.post_model_macro
    # @timeit_debug timer "Calculate third order solution" begin

    # Expand compressed hessian to full space
    ∇₂ = ∇₂ * M₂.𝐔∇₂

    # Expand compressed second-order solution to full space
    𝐒₂ = sparse(𝐒₂ * M₂.𝐔₂)::SparseMatrixCSC{S, Int}

    # inspired by Levintal

    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx;
    i₋ = T.past_not_future_and_mixed_idx;

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo;
    n = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    ensure_higher_order_solution_buffers!(ℂ, n, nₑ₋)

    initial_guess_sylv = if length(initial_guess) == 0
        zeros(S, 0, 0)
    elseif eltype(initial_guess) <: AbstractFloat
        initial_guess isa Matrix{S} ? initial_guess : Matrix{S}(initial_guess)
    else
        zeros(S, 0, 0)
    end

    # @timeit_debug timer "Setup matrices" begin

    # 1st order solution
    𝐒₁ = ℂ.𝐒₁::Matrix{S}
    copyto!(@view(𝐒₁[:,1:n₋]), @view(𝑺₁[:,1:n₋]))
    fill!(@view(𝐒₁[:,n₋+1]), zero(S))
    copyto!(@view(𝐒₁[:,n₋+2:end]), @view(𝑺₁[:,n₋+1:end]))
    
    𝐒₁₋╱𝟏ₑ = ℂ.𝐒₁₋╱𝟏ₑ::Matrix{S}
    copyto!(@view(𝐒₁₋╱𝟏ₑ[1:n₋,:]), @view(𝐒₁[i₋,:]))
    fill!(@view(𝐒₁₋╱𝟏ₑ[n₋+1:end,:]), zero(S))
    @inbounds 𝐒₁₋╱𝟏ₑ[n₋+1,n₋+1] = one(S)

    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 1.0, min_length = 10, tol = opts.tol.third_order.droptol)

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                ℒ.I(nₑ₋)[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]] #|> sparse

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)]# |> sparse
    𝐒₁₊╱𝟎 = choose_matrix_format(𝐒₁₊╱𝟎, density_threshold = 1.0, min_length = 10, tol = opts.tol.third_order.droptol)

    ∇₁₊𝐒₁➕∇₁₀ = collect(@views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.I(n)[i₋,:] - ∇₁[:,range(1,n) .+ n₊])

    # end # timeit_debug
    # @timeit_debug timer "Invert matrix" begin

    qme_ws = workspaces.first_order

    if S === Float64
        qme_ws.fast_lu_ws_nabla0, qme_ws.fast_lu_dims_nabla0, solved_∇lu, lu_handle =
            factorize_lu!(∇₁₊𝐒₁➕∇₁₀, qme_ws.fast_lu_ws_nabla0, qme_ws.fast_lu_dims_nabla0)

        if !solved_∇lu
            if opts.verbose println("Second order solution: inversion failed") end
            return (∇₁₊𝐒₁➕∇₁₀, false)
        end
    else
        ∇₁₊𝐒₁➕∇₁₀lu = ℒ.lu(∇₁₊𝐒₁➕∇₁₀, check = false)

        if !ℒ.issuccess(∇₁₊𝐒₁➕∇₁₀lu)
            if opts.verbose println("Second order solution: inversion failed") end
            return (∇₁₊𝐒₁➕∇₁₀, false)
        end
    end

    # spinv = inv(∇₁₊𝐒₁➕∇₁₀)
    # spinv = choose_matrix_format(spinv)

    # end # timeit_debug

    ∇₁₊ = @views ∇₁[:,1:n₊] * M₂.𝐈ₙ₊

    if S === Float64
        A = copy(∇₁₊) # solve in-place into a buffer; ∇₁₊ is reused later
        solve_lu_left!(∇₁₊𝐒₁➕∇₁₀, A, qme_ws.fast_lu_ws_nabla0, lu_handle)
    else
        A = ∇₁₊𝐒₁➕∇₁₀lu \ ∇₁₊
    end

    # @timeit_debug timer "Setup B" begin
    # @timeit_debug timer "Add tmpkron" begin

    kron𝐒₁₋╱𝟏ₑ = ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ)

    # tmpkron = ℒ.kron(𝐒₁₋╱𝟏ₑ, M₂.𝛔)
    # B = tmpkron + M₃.𝐏₁ₗ̄ * tmpkron * M₃.𝐏₁ᵣ̃ + M₃.𝐏₂ₗ̄ * tmpkron * M₃.𝐏₂ᵣ̃
    # B *= M₃.𝐂₃
    # B = choose_matrix_format(M₃.𝐔₃ * B, tol = opts.tol.third_order.droptol, multithreaded = false)
        # println("size(𝐒₁₋╱𝟏ₑ) = ",size(𝐒₁₋╱𝟏ₑ))
    B = compressed_permuted_mixed_kron(𝐒₁₋╱𝟏ₑ, M₂.𝛔,
                                       sparse_preallocation = ℂ.tmp_sparse_prealloc7)#, timer = timer)
    # println("size(B) = ",size(B))
    # end # timeit_debug
    # @timeit_debug timer "3rd Kronecker power" begin
    # B += mat_mult_kron(M₃.𝐔₃, collect(𝐒₁₋╱𝟏ₑ), collect(ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ)), M₃.𝐂₃) # slower than direct compression

    B += compressed_kron³(𝐒₁₋╱𝟏ₑ, tol = opts.tol.third_order.droptol, sparse_preallocation = ℂ.tmp_sparse_prealloc1)#, timer = timer)

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Setup C" begin
    # @timeit_debug timer "Initialise smaller matrices" begin

    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [(𝐒₂ * kron𝐒₁₋╱𝟏ₑ + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
            𝐒₂
            zeros(n₋ + nₑ, nₑ₋^2)];
            
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = choose_matrix_format(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, density_threshold = 0.0, min_length = 10, tol = opts.tol.third_order.droptol)
        
    𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:] 
            zeros(n₋ + n + nₑ, nₑ₋^2)];

    aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋
    # aux = choose_matrix_format(aux, density_threshold = 1.0, min_length = 10)

    # end # timeit_debug
    # @timeit_debug timer "∇₃" begin   

    # if length(ℂ.tmpkron0) > 0 && eltype(ℂ.tmpkron0) == S
    #     ℒ.kron!(ℂ.tmpkron0, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)
    # else
    #     ℂ.tmpkron0 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)
    # end
    
    # if length(ℂ.tmpkron22) > 0 && eltype(ℂ.tmpkron22) == S
    #     ℒ.kron!(ℂ.tmpkron22, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℂ.tmpkron0 * M₂.𝛔)
    # else
    #     ℂ.tmpkron22 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℂ.tmpkron0 * M₂.𝛔)
    # end

    # # tmpkron = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔)

    # 𝐔∇₃ = ∇₃ * M₃.𝐔∇₃

    # 𝐗₃ = 𝐔∇₃ * ℂ.tmpkron22 + 𝐔∇₃ * M₃.𝐏₁ₗ̂ * ℂ.tmpkron22 * M₃.𝐏₁ᵣ̃ + 𝐔∇₃ * M₃.𝐏₂ₗ̂ * ℂ.tmpkron22 * M₃.𝐏₂ᵣ̃

    # end # timeit_debug
    # @timeit_debug timer "∇₂ & ∇₁₊" begin

    𝐒₂₊╱𝟎 = choose_matrix_format(𝐒₂₊╱𝟎, density_threshold = 1.0, min_length = 10, tol = opts.tol.third_order.droptol)

    ∇₁₊ = choose_matrix_format(∇₁₊, density_threshold = 1.0, min_length = 10, tol = opts.tol.third_order.droptol)

    𝐒₂₋╱𝟎 = [𝐒₂[i₋,:] ; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]

    # Terms (a)+(b): ∇₂ * kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎) * [tmpkron2 + 𝐏₁ₗ * tmpkron2 * 𝐏₁ᵣ] * 𝐏𝐂₃
    # Compute D_ab to avoid materializing kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎)
    tmpkron2_sp = ℒ.kron(M₂.𝛔, choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0, tol = opts.tol.third_order.droptol))
    D_ab = (tmpkron2_sp + M₃.𝐏₁ₗ * tmpkron2_sp * M₃.𝐏₁ᵣ) * M₃.𝐏𝐂₃

    𝐗₃ = mat_mult_kron(∇₂, collect(𝐒₁₊╱𝟎), collect(𝐒₂₊╱𝟎), D_ab, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc2)

    # Term (c): ∇₂ * kron(⎸𝐒₁..⎹, ⎸𝐒₂k..⎹) * 𝐏𝐂₃
    𝐗₃ += mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, M₃.𝐏𝐂₃, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc3)

    # Term (d): ∇₂ * kron(⎸𝐒₁..⎹, 𝐒₂₊╱𝟎*𝛔) * 𝐏𝐂₃
    𝐗₃ += mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, collect(𝐒₂₊╱𝟎 * M₂.𝛔), M₃.𝐏𝐂₃, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc4)

    # Term (e): ∇₁₊ * 𝐒₂ * kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎) * 𝐏𝐂₃
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0, tol = opts.tol.third_order.droptol)
    
    𝐗₃ += mat_mult_kron(∇₁₊ * 𝐒₂, 𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎, M₃.𝐏𝐂₃, sparse = true)

    if length(ℂ.tmpkron0) > 0 && eltype(ℂ.tmpkron0) == S
        ℒ.kron!(ℂ.tmpkron0, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)
    else
        ℂ.tmpkron0 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)
    end

    ℂ.tmpkron0 *= M₂.𝛔
    # ℒ.rmul!(ℂ.tmpkron0, M₂.𝛔)

    𝐗₃ += mul_compressed_permuted_mixed_kron(∇₃, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,
                                                ℂ.tmpkron0,
                                                sparse_preallocation = ℂ.tmp_sparse_prealloc6)
    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "3rd Kronecker power" begin

    # 𝐗₃ += mat_mult_kron(∇₃, collect(aux), collect(ℒ.kron(aux, aux)), M₃.𝐂₃) # slower than direct compression
    𝐗₃ += mul_compressed_kron³(∇₃, aux, tol = opts.tol.third_order.droptol, sparse_preallocation = ℂ.tmp_sparse_prealloc5) #, timer = timer)
    
    # end # timeit_debug
    # @timeit_debug timer "Mult 2" begin

    if S === Float64
        C = collect(𝐗₃)
        solve_lu_left!(∇₁₊𝐒₁➕∇₁₀, C, qme_ws.fast_lu_ws_nabla0, lu_handle)
    else
        C = ∇₁₊𝐒₁➕∇₁₀lu \ 𝐗₃# * M₃.𝐂₃
    end

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Solve sylvester equation" begin

    𝐒₃, solved = solve_sylvester_equation(A, B, C, ℂ.sylvester_workspace,
                                            initial_guess = initial_guess_sylv,
                                            sylvester_algorithm = opts.sylvester_algorithm³,
                                            preconditioner = opts.sylvester_preconditioner,
                                            tol = opts.tol.third_order.sylvester,
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

    𝐒₃ = choose_matrix_format(𝐒₃, multithreaded = false, tol = opts.tol.third_order.droptol)

    # end # timeit_debug
    # end # timeit_debug

    if solved && caching && S === Float64
        if 𝐒₃ isa Matrix{S} && cache.third_order_solution isa Matrix{S} && size(cache.third_order_solution) == size(𝐒₃)
            copyto!(cache.third_order_solution, 𝐒₃)
        elseif 𝐒₃ isa SparseMatrixCSC{S, Int} && cache.third_order_solution isa SparseMatrixCSC{S, Int} &&
               size(cache.third_order_solution) == size(𝐒₃) &&
               cache.third_order_solution.colptr == 𝐒₃.colptr &&
               cache.third_order_solution.rowval == 𝐒₃.rowval
            copyto!(cache.third_order_solution.nzval, 𝐒₃.nzval)
        else
            cache.third_order_solution = copy(𝐒₃)
        end
        if !isempty(parameter_values)
            cache.valid_for.third_order_solution = Float64.(primal.(parameter_values))
            cache.valid_for.pruned_third_order_solution = Float64[]
        end
    end

    return 𝐒₃, solved
end



# ── Compressed Kronecker & matrix utilities (moved from MacroModelling.jl) ──

# Extract unique nonzero row indices, column indices, and nnz count from a dense
# matrix without allocating a sparse copy.  Returns sorted unique indices.
function dense_nz_structure(â::AbstractMatrix{T}) where T
    nrows, ncols = size(â)
    row_has_nz = falses(nrows)
    col_has_nz = falses(ncols)
    lennz = 0
    @inbounds for j in 1:ncols
        for i in 1:nrows
            if !iszero(â[i, j])
                lennz += 1
                row_has_nz[i] = true
                col_has_nz[j] = true
            end
        end
    end
    ui = findall(row_has_nz)
    uj = findall(col_has_nz)
    return ui, uj, lennz
end

function create_second_order_auxiliary_matrices(constants::constants)
    T = constants.post_model_macro
    

    # Indices and number of variables
    n₋ = T.nPast_not_future_and_mixed
    n = T.nVars
    nₑ = T.nExo

    # setup compression matrices for hessian matrix
    nₑ₋ = T.nPast_not_future_and_mixed + T.nVars + T.nFuture_not_past_and_mixed + T.nExo
    colls2 = [nₑ₋ * (i-1) + k for i in 1:nₑ₋ for k in 1:i]
    𝐂∇₂ = sparse(colls2, 1:length(colls2), 1)
    𝐔∇₂ = 𝐂∇₂' * sparse([i <= k ? (k - 1) * nₑ₋ + i : (i - 1) * nₑ₋ + k for k in 1:nₑ₋ for i in 1:nₑ₋], 1:nₑ₋^2, 1)

    # set up vector to capture volatility effect
    nₑ₋ = n₋ + 1 + nₑ
    redu = sparsevec(nₑ₋ - nₑ + 1:nₑ₋, 1)
    redu_idxs = findnz(ℒ.kron(redu, redu))[1]
    𝛔 = @views sparse(redu_idxs[Int.(range(1,nₑ^2,nₑ))], fill(n₋ * (nₑ₋ + 1) + 1, nₑ), 1, nₑ₋^2, nₑ₋^2)
    # setup compression matrices for transition matrix
    colls2 = [nₑ₋ * (i-1) + k for i in 1:nₑ₋ for k in 1:i]
    𝐂₂ = sparse(colls2, 1:length(colls2), 1)
    𝐔₂ = 𝐂₂' * sparse([i <= k ? (k - 1) * nₑ₋ + i : (i - 1) * nₑ₋ + k for k in 1:nₑ₋ for i in 1:nₑ₋], 1:nₑ₋^2, 1)

    # Build symmetrised volatility: 𝛔_sym = 𝛔 + P_swap * 𝛔 * P_swap
    # P_swap is the commutation matrix swapping axes 1 and 2 in nₑ₋² space
    swap_rows = Vector{Int}(undef, nₑ₋^2)
    swap_cols = Vector{Int}(undef, nₑ₋^2)
    @inbounds for a in 1:nₑ₋, b in 1:nₑ₋
        idx = (a - 1) * nₑ₋ + b
        swap_rows[idx] = idx
        swap_cols[idx] = (b - 1) * nₑ₋ + a
    end
    P_swap = sparse(swap_rows, swap_cols, ones(Int, nₑ₋^2), nₑ₋^2, nₑ₋^2)
    𝛔_sym = 𝛔 + P_swap * 𝛔 * P_swap

    so = constants.second_order
    so.𝛔 = 𝛔
    so.𝛔_sym = 𝛔_sym
    so.𝛔c₂ = 𝐔₂ * 𝛔 * 𝐂₂
    so.𝛔𝐂₂ = 𝛔 * 𝐂₂
    so.𝐂₂ = 𝐂₂
    so.𝐔₂ = 𝐔₂
    so.𝐔∇₂ = 𝐔∇₂
    so.𝐈ₙ₊ = sparse(1:T.nFuture_not_past_and_mixed, T.future_not_past_and_mixed_idx, 1, T.nFuture_not_past_and_mixed, n)
    so.𝐈ₙ₋ = sparse(1:T.nPast_not_future_and_mixed, T.past_not_future_and_mixed_idx, 1, T.nPast_not_future_and_mixed, n)
    so.∇₂_nonempty_col_as_kron_rowmask = Int[]
    sigma_row_lookup = falses(size(so.𝛔c₂, 1))
    @inbounds for r in so.𝛔c₂.rowval
        sigma_row_lookup[r] = true
    end
    so.𝛔𝐂₂_nonempty_row_as_kron_colmask = findall(sigma_row_lookup)
    # Pre-transposed constants for rrule pullback (computed once)
    so.𝛔ᵀ = sparse(𝛔')
    so.𝐂₂ᵀ = sparse(𝐂₂')
    so.𝐔₂ᵀ = sparse(𝐔₂')
    so.𝐔∇₂ᵀ = sparse(𝐔∇₂')
    return so
end



function add_sparse_entries!(P, perm)
    n = size(P, 1)
    for i in 1:n
        P[perm[i], i] += 1.0
    end
end


function create_third_order_auxiliary_matrices(constants::constants, ∇₃_col_indices::Vector{Int})
    T = constants.post_model_macro
    

    # Indices and number of variables
    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    n = T.nVars
    nₑ = T.nExo

    n̄ = n₋ + n + n₊ + nₑ

    # compression matrices for third order derivatives matrix
    nₑ₋ = T.nPast_not_future_and_mixed + T.nVars + T.nFuture_not_past_and_mixed + T.nExo
    colls3 = [nₑ₋^2 * (i-1) + nₑ₋ * (k-1) + l for i in 1:nₑ₋ for k in 1:i for l in 1:k]
    𝐂∇₃ = sparse(colls3, 1:length(colls3) , 1.0)
    
    idxs = Int[]
    for k in 1:nₑ₋
        for j in 1:nₑ₋
            for i in 1:nₑ₋
                sorted_ids = sort([k,j,i])
                push!(idxs, (sorted_ids[3] - 1) * nₑ₋ ^ 2 + (sorted_ids[2] - 1) * nₑ₋ + sorted_ids[1])
            end
        end
    end
    
    𝐔∇₃ = 𝐂∇₃' * sparse(idxs,1:nₑ₋ ^ 3, 1)

    # compression matrices for third order transition matrix
    nₑ₋ = n₋ + 1 + nₑ
    colls3 = [nₑ₋^2 * (i-1) + nₑ₋ * (k-1) + l for i in 1:nₑ₋ for k in 1:i for l in 1:k]
    𝐂₃ = sparse(colls3, 1:length(colls3) , 1.0)
    
    idxs = Int[]
    for k in 1:nₑ₋
        for j in 1:nₑ₋
            for i in 1:nₑ₋
                sorted_ids = sort([k,j,i])
                push!(idxs, (sorted_ids[3] - 1) * nₑ₋ ^ 2 + (sorted_ids[2] - 1) * nₑ₋ + sorted_ids[1])
            end
        end
    end
    
    𝐔₃ = 𝐂₃' * sparse(idxs,1:nₑ₋ ^ 3, 1)
    
    # Precompute 𝐈₃
    𝐈₃ = Dict{Vector{Int}, Int}()
    idx = 1
    for i in 1:nₑ₋
        for k in 1:i 
            for l in 1:k
                𝐈₃[[i,k,l]] = idx
                idx += 1
            end
        end
    end

    # permutation matrices
    M = reshape(1:nₑ₋^3,1,nₑ₋,nₑ₋,nₑ₋)

    𝐏 = spzeros(nₑ₋^3, nₑ₋^3)  # Preallocate the sparse matrix

    # Create the permutations directly
    add_sparse_entries!(𝐏, PermutedDimsArray(M, (1, 4, 2, 3)))
    add_sparse_entries!(𝐏, PermutedDimsArray(M, (1, 2, 4, 3)))
    add_sparse_entries!(𝐏, PermutedDimsArray(M, (1, 2, 3, 4)))

    # 𝐏 = @views sparse(reshape(spdiagm(ones(nₑ₋^3))[:,PermutedDimsArray(M,[1, 4, 2, 3])],nₑ₋^3,nₑ₋^3)
    #                     + reshape(spdiagm(ones(nₑ₋^3))[:,PermutedDimsArray(M,[1, 2, 4, 3])],nₑ₋^3,nₑ₋^3)
    #                     + reshape(spdiagm(ones(nₑ₋^3))[:,PermutedDimsArray(M,[1, 2, 3, 4])],nₑ₋^3,nₑ₋^3))

    𝐏₁ₗ = sparse(spdiagm(ones(nₑ₋^3))[vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(2,1,3))),:])
    𝐏₁ᵣ = sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(2,1,3)))])

    𝐏₁ₗ̂  = @views sparse(spdiagm(ones(n̄^3))[vec(permutedims(reshape(1:n̄^3,n̄,n̄,n̄),(1,3,2))),:])
    𝐏₂ₗ̂  = @views sparse(spdiagm(ones(n̄^3))[vec(permutedims(reshape(1:n̄^3,n̄,n̄,n̄),(3,1,2))),:])

    𝐏₁ₗ̄ = @views sparse(spdiagm(ones(nₑ₋^3))[vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(1,3,2))),:])
    𝐏₂ₗ̄ = @views sparse(spdiagm(ones(nₑ₋^3))[vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(3,1,2))),:])


    𝐏₁ᵣ̃ = @views sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(1,3,2)))])
    𝐏₂ᵣ̃ = @views sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(3,1,2)))])

    ∇₃_col_indices_extended = findnz(sparse(ones(Int,length(∇₃_col_indices)),∇₃_col_indices,ones(Int,length(∇₃_col_indices)),1,size(𝐔∇₃,1)) * 𝐔∇₃)[2]

    nonnull_columns = Set{Int}()
    for i in 1:n̄ 
        for j in i:n̄ 
            for k in j:n̄ 
                if n̄^2 * (i - 1)  + n̄ * (j - 1) + k in ∇₃_col_indices_extended
                    push!(nonnull_columns,i)
                    push!(nonnull_columns,j)
                    push!(nonnull_columns,k)
                end
            end
        end
    end
    
    𝐒𝐏 = sparse(collect(nonnull_columns), collect(nonnull_columns), 1, n̄, n̄)

    to = constants.third_order
    to.𝐂₃ = 𝐂₃
    to.𝐔₃ = 𝐔₃
    to.𝐈₃ = 𝐈₃
    to.𝐂∇₃ = 𝐂∇₃
    to.𝐔∇₃ = 𝐔∇₃
    to.∇₃_rowmask = sort!(unique(∇₃_col_indices))
    to.𝐏 = 𝐏
    to.𝐏𝐂₃ = 𝐏 * 𝐂₃
    to.𝐏₁ₗ = 𝐏₁ₗ
    to.𝐏₁ᵣ = 𝐏₁ᵣ
    to.𝐏₁ₗ̂ = 𝐏₁ₗ̂
    to.𝐏₂ₗ̂ = 𝐏₂ₗ̂
    to.𝐏₁ₗ̄ = 𝐏₁ₗ̄
    to.𝐏₂ₗ̄ = 𝐏₂ₗ̄
    to.𝐏₁ᵣ̃ = 𝐏₁ᵣ̃
    to.𝐏₂ᵣ̃ = 𝐏₂ᵣ̃
    to.𝐒𝐏 = 𝐒𝐏
    # Pre-transposed constants for rrule pullback (computed once)
    to.𝐂₃ᵀ = sparse(𝐂₃')
    to.𝐔₃ᵀ = sparse(𝐔₃')
    to.𝐏𝐂₃ᵀ = sparse((to.𝐏𝐂₃)')
    to.𝐏₁ₗᵀ = sparse(𝐏₁ₗ')
    to.𝐏₁ᵣᵀ = sparse(𝐏₁ᵣ')
    to.𝐏₁ₗ̄ᵀ = sparse(𝐏₁ₗ̄')
    to.𝐏₂ₗ̄ᵀ = sparse(𝐏₂ₗ̄')
    to.𝐏₁ᵣ̃ᵀ = sparse(𝐏₁ᵣ̃')
    to.𝐏₂ᵣ̃ᵀ = sparse(𝐏₂ᵣ̃')
    return to
end

@unstable function mat_mult_kron(A::AbstractSparseMatrix{R},
                        B::AbstractMatrix{T},
                        C::AbstractMatrix{T},
                        D::AbstractMatrix{S};
                        sparse_preallocation::Tuple{Vector{Int}, Vector{Int}, Vector{T}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{T}} = (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        sparse::Bool = false) where {R <: Real, T <: Real, S <: Real}
    n_rowB = size(B,1)
    n_colB = size(B,2)

    n_rowC = size(C,1)
    n_colC = size(C,2)

    estimated_nnz = 0
    I = Vector{Int}()
    J = Vector{Int}()
    V = Vector{T}()
    X = zeros(T, 0, 0)
    reused_sparse_buffers = sparse && length(sparse_preallocation[1]) > 0

    if sparse
        nnzA = nnz(A)
        nnzB = sum(abs.(B) .> eps())
        nnzC = sum(abs.(C) .> eps())
        nnzD = sum(abs.(D) .> eps())

        p = Float64(nnzA) * Float64(nnzB) * Float64(nnzC) * Float64(nnzD) / (Float64(length(A)) * Float64(length(B)) * Float64(length(C)) * Float64(length(D)))

        if length(sparse_preallocation[1]) == 0
            estimated_nnz = Int(ceil((1 - (1 - p)^size(A,1)) * size(A,1) * size(D,2)))

            resize!(sparse_preallocation[1], estimated_nnz)
            resize!(sparse_preallocation[2], estimated_nnz)
            resize!(sparse_preallocation[3], estimated_nnz)

            I = sparse_preallocation[1]
            J = sparse_preallocation[2]
            V = sparse_preallocation[3]
        else
            estimated_nnz = length(sparse_preallocation[3])

            resize!(sparse_preallocation[1], estimated_nnz)
            resize!(sparse_preallocation[2], estimated_nnz)
            resize!(sparse_preallocation[3], estimated_nnz)

            I = sparse_preallocation[1]
            J = sparse_preallocation[2]
            V = sparse_preallocation[3]
        end
    else
        X = zeros(T, size(A,1), size(D,2))
    end

    Ā = zeros(T, n_rowC, n_rowB)
    ĀB = zeros(T, n_rowC, n_colB)
    CĀB = zeros(T, n_colC, n_colB)
    vCĀB = zeros(T, n_colB * n_colC)
    vCĀBD = zeros(T, size(D,2))

    # Linked-list row index: O(nnz_in_row) per row instead of O(nnz) for A[row,:]
    A_csc = A isa SparseMatrixCSC ? A : A.A
    A_rv   = SparseArrays.rowvals(A_csc)
    A_nzv  = nonzeros(A_csc)
    A_cp   = SparseArrays.getcolptr(A_csc)
    nnzA_ll = nnz(A_csc)
    n_cols_A = size(A_csc, 2)
    row_head = zeros(Int, size(A_csc, 1))
    row_next = zeros(Int, nnzA_ll)
    nz_col   = Vector{Int}(undef, nnzA_ll)
    @inbounds for col in n_cols_A:-1:1
        for idx in A_cp[col]:(A_cp[col + 1] - 1)
            r = A_rv[idx]
            row_next[idx] = row_head[r]
            row_head[r] = idx
            nz_col[idx] = col
        end
    end

    α = .7
    k = 0

    @inbounds for row in eachindex(row_head)
        row_head[row] == 0 && continue
        fill!(Ā, zero(T))
        p = row_head[row]
        while p != 0
            Ā[nz_col[p]] = T(A_nzv[p])
            p = row_next[p]
        end
        ℒ.mul!(ĀB, Ā, B)
        ℒ.mul!(CĀB, C', ĀB)
        copyto!(vCĀB, CĀB)
        ℒ.mul!(vCĀBD, D', vCĀB)

        if sparse
            for (i,v) in enumerate(vCĀBD)
                if abs(v) > eps()
                    k += 1

                    if k > estimated_nnz
                        increment = max(10000, Int(ceil((α - 1) * estimated_nnz + (1 - α) * size(A,1) * size(D,2))))
                        estimated_nnz += min(size(A,1) * size(D,2), increment)

                        resize!(I, estimated_nnz)
                        resize!(J, estimated_nnz)
                        resize!(V, estimated_nnz)
                    end

                    I[k] = row
                    J[k] = i
                    V[k] = v
                end
            end
        else
            @views copyto!(X[row,:], vCĀBD)
        end
    end

    if sparse
        resize!(I, k)
        resize!(J, k)
        resize!(V, k)

        klasttouch = sparse_preallocation[4]
        csrrowptr  = sparse_preallocation[5]
        csrcolval  = sparse_preallocation[6]
        csrnzval   = sparse_preallocation[7]

        resize!(klasttouch, size(D,2))
        resize!(csrrowptr, size(A, 1) + 1)
        resize!(csrcolval, length(I))
        resize!(csrnzval, length(I))

        if length(I) >= size(D,2) + 1
            out = sparse!(I, J, V, size(A, 1), size(D,2), +, klasttouch, csrrowptr, csrcolval, csrnzval, I, J, V)
        else
            out = SparseArrays.sparse(I, J, V, size(A, 1), size(D,2))
        end
        # if reused_sparse_buffers
        #     out = copy(out)
        # end
    else
        out = choose_matrix_format(X)
    end

    return out
end




@unstable function mat_mult_kron(A::DenseMatrix{R},
                        B::AbstractMatrix{T},
                        C::AbstractMatrix{T},
                        D::AbstractMatrix{S}) where {R <: Real, T <: Real, S <: Real}
    n_rowB = size(B,1)
    n_colB = size(B,2)

    n_rowC = size(C,1)
    n_colC = size(C,2)

    X = zeros(T, size(A,1), size(D,2))

    # vals = T[]
    # rows = Int[]
    # cols = Int[]

    Ā = zeros(T, n_rowC, n_rowB)
    ĀB = zeros(T, n_rowC, n_colB)
    CĀB = zeros(T, n_colC, n_colB)
    vCĀB = zeros(T, n_colB * n_colC)
    # vCĀBD = zeros(size(D,2))

    # rv = A isa SparseMatrixCSC ? A.rowval : A.A.rowval

    # Polyester.@batch threadlocal = (Vector{T}(), Vector{Int}(), Vector{Int}()) for row in rv |> unique
    r = 1
    @inbounds for row in eachrow(A)
        @views copyto!(Ā, row)
        ℒ.mul!(ĀB, Ā, B)
        ℒ.mul!(CĀB, C', ĀB)
        copyto!(vCĀB, CĀB)
        @views ℒ.mul!(X[row,:], D', vCĀB)
        r += 1
    end

    return choose_matrix_format(X)
    #     ℒ.mul!(vCĀBD, D', vCĀB)

    #     for (i,v) in enumerate(vCĀBD)
    #         if abs(v) > eps()
    #             push!(rows, row)
    #             push!(cols, i)
    #             push!(vals, v)
    #         end
    #     end
    # end

    # if VERSION >= v"1.10"
    #     return sparse!(rows, cols, vals, size(A,1), size(D,2))   
    # else
    #     return sparse(rows, cols, vals, size(A,1), size(D,2))   
    # end
end

@unstable function mat_mult_kron(A::AbstractSparseMatrix{R},
                        B::AbstractMatrix{T},
                        C::AbstractMatrix{T};
                        sparse_preallocation::Tuple{Vector{Int}, Vector{Int}, Vector{T}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{T}} = (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        sparse::Bool = false) where {R <: Real, T <: Real}
    n_rowB = size(B,1)
    n_colB = size(B,2)

    n_rowC = size(C,1)
    n_colC = size(C,2)

    estimated_nnz = 0
    I = Vector{Int}()
    J = Vector{Int}()
    V = Vector{T}()
    X = zeros(T, 0, 0)
    reused_sparse_buffers = sparse && length(sparse_preallocation[1]) > 0

    if sparse
        nnzA = nnz(A)
        nnzB = sum(abs.(B) .> eps())
        nnzC = sum(abs.(C) .> eps())

        p = nnzA * nnzB * nnzC / (length(A) * length(B) * length(C))
        
        if length(sparse_preallocation[1]) == 0
            estimated_nnz = Int(ceil((1-(1-p)^size(A,1))*size(A,1) * n_colB * n_colC))

            resize!(sparse_preallocation[1], estimated_nnz)
            resize!(sparse_preallocation[2], estimated_nnz)
            resize!(sparse_preallocation[3], estimated_nnz)

            I = sparse_preallocation[1]
            J = sparse_preallocation[2]
            V = sparse_preallocation[3]
        else
            estimated_nnz = length(sparse_preallocation[3])

            resize!(sparse_preallocation[1], estimated_nnz)
            resize!(sparse_preallocation[2], estimated_nnz)
            resize!(sparse_preallocation[3], estimated_nnz)

            I = sparse_preallocation[1]
            J = sparse_preallocation[2]
            V = sparse_preallocation[3]
        end
    else
        X = zeros(T, size(A,1), n_colB * n_colC)
    end

    Ā = zeros(T, n_rowC, n_rowB)
    ĀB = zeros(T, n_rowC, n_colB)
    CĀB = zeros(T, n_colC, n_colB)

    # Linked-list row index: O(nnz_in_row) per row instead of O(nnz) for A[row,:]
    A_csc = A isa SparseMatrixCSC ? A : A.A
    A_rv   = SparseArrays.rowvals(A_csc)
    A_nzv  = nonzeros(A_csc)
    A_cp   = SparseArrays.getcolptr(A_csc)
    nnzA_ll = nnz(A_csc)
    n_cols_A = size(A_csc, 2)
    row_head = zeros(Int, size(A_csc, 1))
    row_next = zeros(Int, nnzA_ll)
    nz_col   = Vector{Int}(undef, nnzA_ll)
    @inbounds for col in n_cols_A:-1:1
        for idx in A_cp[col]:(A_cp[col + 1] - 1)
            r = A_rv[idx]
            row_next[idx] = row_head[r]
            row_head[r] = idx
            nz_col[idx] = col
        end
    end

    α = .7 # speed of Vector increase
    k = 0

    @inbounds for row in eachindex(row_head)
        row_head[row] == 0 && continue
        fill!(Ā, zero(T))
        p = row_head[row]
        while p != 0
            Ā[nz_col[p]] = T(A_nzv[p])
            p = row_next[p]
        end
        ℒ.mul!(ĀB, Ā, B)
        ℒ.mul!(CĀB, C', ĀB)
        
        if sparse
            for (i,v) in enumerate(CĀB)
                if abs(v) > eps()
                    k += 1

                    if k > estimated_nnz
                        estimated_nnz += min(size(A,1) * n_colB * n_colC, max(10000, Int(ceil((α - 1) * estimated_nnz + (1 - α) * size(A,1) * n_colB * n_colC))))
                        
                        resize!(I, estimated_nnz)
                        resize!(J, estimated_nnz)
                        resize!(V, estimated_nnz)
                    end

                    I[k] = row
                    J[k] = i
                    V[k] = v
                end
            end
        else
            @views copyto!(X[row,:], CĀB)
        end
    end

    if sparse
        resize!(I, k)
        resize!(J, k)
        resize!(V, k)

        klasttouch = sparse_preallocation[4] # Vector{Ti}(undef, n)
        csrrowptr  = sparse_preallocation[5] # Vector{Ti}(undef, m + 1)
        csrcolval  = sparse_preallocation[6] # Vector{Ti}(undef, length(I))
        csrnzval   = sparse_preallocation[7] # Vector{Tv}(undef, length(I))

        resize!(klasttouch, n_colB * n_colC)
        resize!(csrrowptr, size(A, 1) + 1)
        resize!(csrcolval, length(I))
        resize!(csrnzval, length(I))

        if length(I) >= n_colB * n_colC + 1
            out = sparse!(I, J, V, size(A, 1), n_colB * n_colC, +, klasttouch, csrrowptr, csrcolval, csrnzval, I, J, V)
        else
            out = SparseArrays.sparse(I, J, V, size(A, 1), n_colB * n_colC)
        end
        # if reused_sparse_buffers
        #     out = copy(out)
        # end
        # out = sparse!(I, J, V, size(A, 1), n_colB * n_colC)   
    else
        out = choose_matrix_format(X)
    end
    
    return out
end




@unstable function mat_mult_kron(A::DenseMatrix{R},
                        B::AbstractMatrix{T},
                        C::AbstractMatrix{T}) where {R <: Real, T <: Real}
    n_rowB = size(B,1)
    n_colB = size(B,2)

    n_rowC = size(C,1)
    n_colC = size(C,2)

    X = zeros(T, size(A,1), n_colB * n_colC)

    # vals = T[]
    # rows = Int[]
    # cols = Int[]

    Ā = zeros(T, n_rowC, n_rowB)
    ĀB = zeros(T, n_rowC, n_colB)
    CĀB = zeros(T, n_colC, n_colB)

    # Polyester.@batch threadlocal = (Vector{T}(), Vector{Int}(), Vector{Int}()) for row in rv |> unique
    r = 1
    @inbounds for row in eachrow(A)
        @views copyto!(Ā, row)
        ℒ.mul!(ĀB, Ā, B)
        ℒ.mul!(CĀB, C', ĀB)
        
        @views copyto!(X[r,:], CĀB)
        r += 1
    end

    return choose_matrix_format(X)
    #     for (i,v) in enumerate(CĀB)
    #         if abs(v) > eps()
    #             push!(rows, row)
    #             push!(cols, i)
    #             push!(vals, v)
    #         end
    #     end
    # end

    # if VERSION >= v"1.10"
    #     return sparse!(rows,cols,vals,size(A,1),n_colB*n_colC)   
    # else
    #     return sparse(rows,cols,vals,size(A,1),n_colB*n_colC)   
    # end
end

# Loop-based compressed permuted mixed Kronecker product.
# Computes  U₃ * (kron(A,σ) + P₁ₗ̄*kron(A,σ)*P₁ᵣ̃ + P₂ₗ̄*kron(A,σ)*P₂ᵣ̃) * C₃
# directly in compressed (sorted-triple) space without forming any n³×n³ intermediates.
#
# A is nr×nc (may be rectangular),  σ is nr²×nc².
# Output is mr₃×mc₃ sparse where mr₃ = nr(nr+1)(nr+2)/6, mc₃ = nc(nc+1)(nc+2)/6.
#
# The uncompressed entry at row (i,j,k) col (a,b,c) of the sum is:
#   A[i,a]*σ[(j-1)*nr+k,(b-1)*nc+c]                   (identity)
# + A[j,b]*σ[(i-1)*nr+k,(a-1)*nc+c]                   (P₁: swap i↔j rows, a↔b cols)
# + A[j,b]*σ[(k-1)*nr+i,(c-1)*nc+a]                   (P₂: cycle (i,j,k)→(j,k,i), (a,b,c)→(b,c,a))
#
# Compression: U₃ sums over all row permutations that sort to (i₁≥j₁≥k₁);
#              C₃ selects the sorted column representative (α≥β≥γ).
function compressed_permuted_mixed_kron(A::AbstractMatrix{T}, σ::AbstractMatrix;
                    tol::AbstractFloat = eps(),
                    sparse_preallocation::Tuple{Vector{Int}, Vector{Int}, Vector{T}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{T}} = (Int[], Int[], T[], Int[], Int[], Int[], T[])) where T <: Real

    nr = size(A, 1)
    nc = size(A, 2)
    size(σ) == (nr^2, nc^2) || throw(DimensionMismatch("σ must be $(nr^2)×$(nc^2), got $(size(σ))"))

    # Sparse copies for support-aware iteration.
    As = A isa SparseMatrixCSC{T, Int} ? A : sparse(T.(A))
    σs = σ isa SparseMatrixCSC{T, Int} ? σ : sparse(T.(σ))

    rv_A = SparseArrays.rowvals(As)
    nzv_A = nonzeros(As)
    rv_σ = SparseArrays.rowvals(σs)
    nzv_σ = nonzeros(σs)

    ranges_A = Vector{UnitRange{Int}}(undef, nc)
    ranges_σ = Vector{UnitRange{Int}}(undef, nc^2)
    @inbounds for col in 1:nc
        ranges_A[col] = SparseArrays.nzrange(As, col)
    end
    @inbounds for col in 1:(nc^2)
        ranges_σ[col] = SparseArrays.nzrange(σs, col)
    end

    mr₃ = nr * (nr + 1) * (nr + 2) ÷ 6
    mc₃ = nc * (nc + 1) * (nc + 2) ÷ 6

    # --- sparse buffer management (same pattern as compressed_kron³) ---
    if length(sparse_preallocation[1]) == 0
        estimated_nnz = max(min(mr₃, mc₃), 10000)

        resize!(sparse_preallocation[1], estimated_nnz)
        resize!(sparse_preallocation[2], estimated_nnz)
        resize!(sparse_preallocation[3], estimated_nnz)
    else
        estimated_nnz = length(sparse_preallocation[3])

        resize!(sparse_preallocation[1], estimated_nnz)
        resize!(sparse_preallocation[2], estimated_nnz)
        resize!(sparse_preallocation[3], estimated_nnz)
    end

    II = sparse_preallocation[1]
    JJ = sparse_preallocation[2]
    VV = sparse_preallocation[3]

    cnt = 0   # non-zero counter

    # Iterate sorted output columns first (α ≥ β ≥ γ). For each column triple,
    # only traverse non-zero supports from the relevant A and σ columns.
    for α in 1:nc
        rng_Aα = ranges_A[α]
        for β in 1:α
            rng_Aβ = ranges_A[β]
            for γ in 1:β
                rng_Aγ = ranges_A[γ]

                σ_col_βγ = (β - 1) * nc + γ
                σ_col_αγ = (α - 1) * nc + γ
                σ_col_αβ = (α - 1) * nc + β

                rng_σβγ = ranges_σ[σ_col_βγ]
                rng_σαγ = ranges_σ[σ_col_αγ]
                rng_σαβ = ranges_σ[σ_col_αβ]

                has_t1 = !isempty(rng_Aα) && !isempty(rng_σβγ)
                has_t2 = !isempty(rng_Aβ) && !isempty(rng_σαγ)
                has_t3 = !isempty(rng_Aγ) && !isempty(rng_σαβ)

                (has_t1 || has_t2 || has_t3) || continue

                col = (α - 1) * α * (α + 1) ÷ 6 + (β - 1) * β ÷ 2 + γ

                # term 1: A[p, α] * σ[(q, r), (β, γ)]
                if has_t1
                    @inbounds for ia in rng_Aα
                        p = rv_A[ia]
                        a_val = nzv_A[ia]

                        for is in rng_σβγ
                            qr = rv_σ[is]
                            q = (qr - 1) ÷ nr + 1
                            r = qr - (q - 1) * nr

                            val = a_val * nzv_σ[is]
                            abs(val) > tol || continue

                            i1 = p
                            j1 = q
                            k1 = r

                            if i1 < j1
                                i1, j1 = j1, i1
                            end
                            if j1 < k1
                                j1, k1 = k1, j1
                            end
                            if i1 < j1
                                i1, j1 = j1, i1
                            end

                            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1

                            cnt += 1
                            if cnt > estimated_nnz
                                estimated_nnz += Int(ceil(max(1000, estimated_nnz * 0.1)))
                                estimated_nnz = min(mr₃ * mc₃, estimated_nnz)
                                resize!(II, estimated_nnz)
                                resize!(JJ, estimated_nnz)
                                resize!(VV, estimated_nnz)
                            end

                            II[cnt] = row
                            JJ[cnt] = col
                            VV[cnt] = val
                        end
                    end
                end

                # term 2: A[q, β] * σ[(p, r), (α, γ)]
                if has_t2
                    @inbounds for ia in rng_Aβ
                        q = rv_A[ia]
                        a_val = nzv_A[ia]

                        for is in rng_σαγ
                            pr = rv_σ[is]
                            p = (pr - 1) ÷ nr + 1
                            r = pr - (p - 1) * nr

                            val = a_val * nzv_σ[is]
                            abs(val) > tol || continue

                            i1 = p
                            j1 = q
                            k1 = r

                            if i1 < j1
                                i1, j1 = j1, i1
                            end
                            if j1 < k1
                                j1, k1 = k1, j1
                            end
                            if i1 < j1
                                i1, j1 = j1, i1
                            end

                            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1

                            cnt += 1
                            if cnt > estimated_nnz
                                estimated_nnz += Int(ceil(max(1000, estimated_nnz * 0.1)))
                                estimated_nnz = min(mr₃ * mc₃, estimated_nnz)
                                resize!(II, estimated_nnz)
                                resize!(JJ, estimated_nnz)
                                resize!(VV, estimated_nnz)
                            end

                            II[cnt] = row
                            JJ[cnt] = col
                            VV[cnt] = val
                        end
                    end
                end

                # term 3: A[r, γ] * σ[(p, q), (α, β)]
                if has_t3
                    @inbounds for ia in rng_Aγ
                        r = rv_A[ia]
                        a_val = nzv_A[ia]

                        for is in rng_σαβ
                            pq = rv_σ[is]
                            p = (pq - 1) ÷ nr + 1
                            q = pq - (p - 1) * nr

                            val = a_val * nzv_σ[is]
                            abs(val) > tol || continue

                            i1 = p
                            j1 = q
                            k1 = r

                            if i1 < j1
                                i1, j1 = j1, i1
                            end
                            if j1 < k1
                                j1, k1 = k1, j1
                            end
                            if i1 < j1
                                i1, j1 = j1, i1
                            end

                            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1

                            cnt += 1
                            if cnt > estimated_nnz
                                estimated_nnz += Int(ceil(max(1000, estimated_nnz * 0.1)))
                                estimated_nnz = min(mr₃ * mc₃, estimated_nnz)
                                resize!(II, estimated_nnz)
                                resize!(JJ, estimated_nnz)
                                resize!(VV, estimated_nnz)
                            end

                            II[cnt] = row
                            JJ[cnt] = col
                            VV[cnt] = val
                        end
                    end
                end
            end
        end
    end

    resize!(II, cnt)
    resize!(JJ, cnt)
    resize!(VV, cnt)

    # Assemble sparse matrix using preallocated CSR workspace
    klasttouch = sparse_preallocation[4]
    csrrowptr  = sparse_preallocation[5]
    csrcolval  = sparse_preallocation[6]
    csrnzval   = sparse_preallocation[7]

    resize!(klasttouch, mc₃)
    resize!(csrrowptr, mr₃ + 1)
    resize!(csrcolval, length(II))
    resize!(csrnzval, length(II))

    out = if length(II) >= mc₃ + 1
        sparse!(II, JJ, VV, mr₃, mc₃, +, klasttouch, csrrowptr, csrcolval, csrnzval, II, JJ, VV)
    else
        SparseArrays.sparse(II, JJ, VV, mr₃, mc₃)
    end

    if tol > 0
        droptol!(out, tol)
    end

    return out
end

# Fused  M * compressed_permuted_mixed_kron(A, σ)
# Computes the product without materializing the large mr₃×mc₃ intermediate.
# M is m × mr₃ sparse, A is nr × nc, σ is nr² × nc².  Output: m × mc₃ sparse.
function mul_compressed_permuted_mixed_kron(M::SparseMatrixCSC, A::AbstractMatrix{T}, σ::AbstractMatrix;
                    tol::AbstractFloat = eps(),
                    sparse_preallocation::Tuple{Vector{Int}, Vector{Int}, Vector{T}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{T}} = (Int[], Int[], T[], Int[], Int[], Int[], T[])) where T <: Real

    nr = size(A, 1)
    nc = size(A, 2)
    m  = size(M, 1)
    mr₃ = nr * (nr + 1) * (nr + 2) ÷ 6
    mc₃ = nc * (nc + 1) * (nc + 2) ÷ 6

    size(σ) == (nr^2, nc^2) || throw(DimensionMismatch("σ must be $(nr^2)×$(nc^2), got $(size(σ))"))
    size(M, 2) == mr₃ || throw(DimensionMismatch("M must have $mr₃ columns, got $(size(M, 2))"))

    # Sparse copies for support-aware iteration
    As = A isa SparseMatrixCSC{T, Int} ? A : sparse(T.(A))
    σs = σ isa SparseMatrixCSC{T, Int} ? σ : sparse(T.(σ))

    rv_A = SparseArrays.rowvals(As)
    nzv_A = nonzeros(As)
    rv_σ = SparseArrays.rowvals(σs)
    nzv_σ = nonzeros(σs)
    rv_M = SparseArrays.rowvals(M)
    nzv_M = nonzeros(M)

    ranges_A = Vector{UnitRange{Int}}(undef, nc)
    ranges_σ = Vector{UnitRange{Int}}(undef, nc^2)
    @inbounds for col in 1:nc
        ranges_A[col] = SparseArrays.nzrange(As, col)
    end
    @inbounds for col in 1:(nc^2)
        ranges_σ[col] = SparseArrays.nzrange(σs, col)
    end

    # Small result buffer (size m, not mr₃)
    result_col = zeros(T, m)

    # --- sparse IJV buffer management ---
    if length(sparse_preallocation[1]) == 0
        estimated_nnz = max(min(m * mc₃ ÷ 4, m * mc₃), 10000)
        resize!(sparse_preallocation[1], estimated_nnz)
        resize!(sparse_preallocation[2], estimated_nnz)
        resize!(sparse_preallocation[3], estimated_nnz)
    else
        estimated_nnz = length(sparse_preallocation[3])
        resize!(sparse_preallocation[1], estimated_nnz)
        resize!(sparse_preallocation[2], estimated_nnz)
        resize!(sparse_preallocation[3], estimated_nnz)
    end

    II = sparse_preallocation[1]
    JJ = sparse_preallocation[2]
    VV = sparse_preallocation[3]
    cnt = 0

    for α in 1:nc
        rng_Aα = ranges_A[α]
        for β in 1:α
            rng_Aβ = ranges_A[β]
            for γ in 1:β
                rng_Aγ = ranges_A[γ]

                σ_col_βγ = (β - 1) * nc + γ
                σ_col_αγ = (α - 1) * nc + γ
                σ_col_αβ = (α - 1) * nc + β

                rng_σβγ = ranges_σ[σ_col_βγ]
                rng_σαγ = ranges_σ[σ_col_αγ]
                rng_σαβ = ranges_σ[σ_col_αβ]

                has_t1 = !isempty(rng_Aα) && !isempty(rng_σβγ)
                has_t2 = !isempty(rng_Aβ) && !isempty(rng_σαγ)
                has_t3 = !isempty(rng_Aγ) && !isempty(rng_σαβ)

                (has_t1 || has_t2 || has_t3) || continue

                col = (α - 1) * α * (α + 1) ÷ 6 + (β - 1) * β ÷ 2 + γ

                fill!(result_col, zero(T))

                # term 1: A[p, α] * σ[(q,r), (β,γ)] — scatter through M
                if has_t1
                    @inbounds for ia in rng_Aα
                        p = rv_A[ia]
                        a_val = nzv_A[ia]
                        for is in rng_σβγ
                            qr = rv_σ[is]
                            q = (qr - 1) ÷ nr + 1
                            r = qr - (q - 1) * nr
                            val = a_val * nzv_σ[is]
                            abs(val) > tol || continue
                            i1 = p; j1 = q; k1 = r
                            if i1 < j1; i1, j1 = j1, i1; end
                            if j1 < k1; j1, k1 = k1, j1; end
                            if i1 < j1; i1, j1 = j1, i1; end
                            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1
                            rng_M = SparseArrays.nzrange(M, row)
                            for p_M in rng_M
                                result_col[rv_M[p_M]] += nzv_M[p_M] * val
                            end
                        end
                    end
                end

                # term 2: A[q, β] * σ[(p,r), (α,γ)] — scatter through M
                if has_t2
                    @inbounds for ia in rng_Aβ
                        q = rv_A[ia]
                        a_val = nzv_A[ia]
                        for is in rng_σαγ
                            pr = rv_σ[is]
                            p = (pr - 1) ÷ nr + 1
                            r = pr - (p - 1) * nr
                            val = a_val * nzv_σ[is]
                            abs(val) > tol || continue
                            i1 = p; j1 = q; k1 = r
                            if i1 < j1; i1, j1 = j1, i1; end
                            if j1 < k1; j1, k1 = k1, j1; end
                            if i1 < j1; i1, j1 = j1, i1; end
                            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1
                            rng_M = SparseArrays.nzrange(M, row)
                            for p_M in rng_M
                                result_col[rv_M[p_M]] += nzv_M[p_M] * val
                            end
                        end
                    end
                end

                # term 3: A[r, γ] * σ[(p,q), (α,β)] — scatter through M
                if has_t3
                    @inbounds for ia in rng_Aγ
                        r = rv_A[ia]
                        a_val = nzv_A[ia]
                        for is in rng_σαβ
                            pq = rv_σ[is]
                            p = (pq - 1) ÷ nr + 1
                            q = pq - (p - 1) * nr
                            val = a_val * nzv_σ[is]
                            abs(val) > tol || continue
                            i1 = p; j1 = q; k1 = r
                            if i1 < j1; i1, j1 = j1, i1; end
                            if j1 < k1; j1, k1 = k1, j1; end
                            if i1 < j1; i1, j1 = j1, i1; end
                            row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1
                            rng_M = SparseArrays.nzrange(M, row)
                            for p_M in rng_M
                                result_col[rv_M[p_M]] += nzv_M[p_M] * val
                            end
                        end
                    end
                end

                # Extract nonzeros into IJV
                @inbounds for i in 1:m
                    v = result_col[i]
                    if abs(v) > tol
                        cnt += 1
                        if cnt > estimated_nnz
                            estimated_nnz += Int(ceil(max(1000, estimated_nnz * 0.1)))
                            estimated_nnz = min(m * mc₃, estimated_nnz)
                            resize!(II, estimated_nnz)
                            resize!(JJ, estimated_nnz)
                            resize!(VV, estimated_nnz)
                        end
                        II[cnt] = i
                        JJ[cnt] = col
                        VV[cnt] = v
                    end
                end
            end
        end
    end

    resize!(II, cnt)
    resize!(JJ, cnt)
    resize!(VV, cnt)

    # Sparse assembly
    klasttouch = sparse_preallocation[4]
    csrrowptr  = sparse_preallocation[5]
    csrcolval  = sparse_preallocation[6]
    csrnzval   = sparse_preallocation[7]

    resize!(klasttouch, mc₃)
    resize!(csrrowptr, m + 1)
    resize!(csrcolval, length(II))
    resize!(csrnzval, length(II))

    out = if length(II) >= mc₃ + 1
        sparse!(II, JJ, VV, m, mc₃, +, klasttouch, csrrowptr, csrcolval, csrnzval, II, JJ, VV)
    else
        SparseArrays.sparse(II, JJ, VV, m, mc₃)
    end

    if tol > 0
        droptol!(out, tol)
    end

    return out
end

function compressed_kron³(a::AbstractMatrix{T};
                    rowmask::Vector{Int} = Int[],
                    colmask::Vector{Int} = Int[],
                    # timer::TimerOutput = TimerOutput(),
                    tol::AbstractFloat = eps(),
                    sparse_preallocation::Tuple{Vector{Int}, Vector{Int}, Vector{T}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{T}} = (Int[], Int[], T[], Int[], Int[], Int[], T[])) where T <: Real
    # @timeit_debug timer "Compressed 3rd kronecker power" begin
          
    # @timeit_debug timer "Preallocation" begin
    
    a_is_adjoint = typeof(a) <: ℒ.Adjoint{T,Matrix{T}}
    reused_sparse_buffers = length(sparse_preallocation[1]) > 0
    
    if a_is_adjoint
        â = copy(a')
        
        rmask = colmask
        colmask = rowmask
        rowmask = rmask
    elseif typeof(a) <: DenseMatrix{T}
        â = copy(a)
    else
        â = convert(Matrix, a)  # Convert to dense matrix for faster access
    end
    # Get the number of rows and columns
    n_rows, n_cols = size(â)
    
    # Calculate the number of unique triplet indices for rows and columns
    m3_rows = n_rows * (n_rows + 1) * (n_rows + 2) ÷ 6    # For rows: i ≤ j ≤ k
    m3_cols = n_cols * (n_cols + 1) * (n_cols + 2) ÷ 6    # For columns: i ≤ j ≤ k

    if rowmask == Int[0] || colmask == Int[0]
        if a_is_adjoint
            return spzeros(T, m3_cols, m3_rows)
        else
            return spzeros(T, m3_rows, m3_cols)
        end
    end
    # Extract unique nonzero row/col indices directly from dense matrix
    ui, uj, lennz = dense_nz_structure(â)

    m3_c = length(colmask) > 0 ? length(colmask) : m3_cols
    m3_r = length(rowmask) > 0 ? length(rowmask) : m3_rows

    m3_exp = (length(colmask) > 0 || length(rowmask) > 0) ? 3 : 4

    if length(sparse_preallocation[1]) == 0
        estimated_nnz = floor(Int, max(m3_r * m3_c * (lennz / length(â)) ^ m3_exp, 10000))

        resize!(sparse_preallocation[1], estimated_nnz)
        resize!(sparse_preallocation[2], estimated_nnz)
        resize!(sparse_preallocation[3], estimated_nnz)

        I = sparse_preallocation[1]
        J = sparse_preallocation[2]
        V = sparse_preallocation[3]
    else
        estimated_nnz = length(sparse_preallocation[3])

        resize!(sparse_preallocation[1], estimated_nnz)
        resize!(sparse_preallocation[2], estimated_nnz)
        resize!(sparse_preallocation[3], estimated_nnz)

        I = sparse_preallocation[1]
        J = sparse_preallocation[2]
        V = sparse_preallocation[3]
    end

    # k = Threads.Atomic{Int}(0)  # Counter for non-zero entries
    # k̄ = Threads.Atomic{Int}(0)  # effectively slower than the non-threaded version

    k = 0


    # @timeit_debug timer "Loop" begin
    # Triple nested loops for (i1 ≤ j1 ≤ k1) and (i2 ≤ j2 ≤ k2)
    # Polyester.@batch threadlocal=(Vector{Int}(), Vector{Int}(), Vector{T}()) for i1 in ui
    # Polyester.@batch minbatch = 10 for i1 in ui
    # Threads.@threads for i1 in ui
    norowmask = length(rowmask) == 0
    nocolmask = length(colmask) == 0
    rowmask_lookup = norowmask ? BitVector() : falses(m3_rows)
    colmask_lookup = nocolmask ? BitVector() : falses(m3_cols)

    if !norowmask && rowmask != Int[0]
        @inbounds for r in rowmask
            if 1 <= r <= m3_rows
                rowmask_lookup[r] = true
            end
        end
    end
    if !nocolmask && colmask != Int[0]
        @inbounds for c in colmask
            if 1 <= c <= m3_cols
                colmask_lookup[c] = true
            end
        end
    end

    n_ui = length(ui)
    n_uj = length(uj)

    for idx_i1 in 1:n_ui
        @inbounds i1 = ui[idx_i1]
        for idx_j1 in 1:idx_i1
            @inbounds j1 = ui[idx_j1]
            for idx_k1 in 1:idx_j1
                @inbounds k1 = ui[idx_k1]

                row = (i1-1) * i1 * (i1+1) ÷ 6 + (j1-1) * j1 ÷ 2 + k1

                if norowmask || rowmask_lookup[row]
                    # Divisor depends only on the row triple
                    if i1 == j1
                        divisor = i1 == k1 ? 6 : 2
                    else
                        divisor = (i1 ≠ k1 && j1 ≠ k1) ? 1 : 2
                    end

                    for idx_i2 in 1:n_uj
                        @inbounds i2 = uj[idx_i2]
                        # Hoist i2-dependent reads
                        @inbounds aii = â[i1, i2]
                        @inbounds aji = â[j1, i2]
                        @inbounds aki = â[k1, i2]

                        for idx_j2 in 1:idx_i2
                            @inbounds j2 = uj[idx_j2]
                            # Hoist j2-dependent reads
                            @inbounds aij = â[i1, j2]
                            @inbounds ajj = â[j1, j2]
                            @inbounds akj = â[k1, j2]

                            # Precompute sub-expressions for the k2 inner loop
                            p1 = aii * ajj + aij * aji  # coefficient of akk
                            p2 = aii * akj + aij * aki  # coefficient of ajk
                            p3 = aji * akj + ajj * aki  # coefficient of aik
                            col_partial = (i2-1) * i2 * (i2+1) ÷ 6 + (j2-1) * j2 ÷ 2

                            for idx_k2 in 1:idx_j2
                                @inbounds k2 = uj[idx_k2]
                                @inbounds aik = â[i1, k2]
                                @inbounds ajk = â[j1, k2]
                                @inbounds akk = â[k1, k2]

                                val = akk * p1 + ajk * p2 + aik * p3

                                if abs(val) > tol
                                    col = col_partial + k2

                                    if nocolmask || colmask_lookup[col]
                                        k += 1

                                        if k > estimated_nnz
                                            estimated_nnz += Int(ceil(max(1000, estimated_nnz * .1)))
                                            estimated_nnz = min(m3_cols * m3_rows, estimated_nnz)
                                            resize!(I, estimated_nnz)
                                            resize!(J, estimated_nnz)
                                            resize!(V, estimated_nnz)
                                        end

                                        I[k] = row
                                        J[k] = col
                                        V[k] = val / divisor
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    # end # timeit_debug

    # @timeit_debug timer "Resize" begin

    # out = map(fetch, threadlocal)

    # I = mapreduce(v -> v[1], vcat, out)
    # J = mapreduce(v -> v[2], vcat, out)
    # V = mapreduce(v -> v[3], vcat, out)

    # # Resize the index and value arrays to the actual number of entries
    # resize!(I, k̄[])
    # resize!(J, k̄[])
    # resize!(V, k̄[]) 
    resize!(I, k)
    resize!(J, k)
    resize!(V, k)

    # end # timeit_debug
    # end # timeit_debug

    # Create the sparse matrix from the collected indices and values
    if a_is_adjoint
        klasttouch = sparse_preallocation[4] # Vector{Ti}(undef, n)
        csrrowptr  = sparse_preallocation[5] # Vector{Ti}(undef, m + 1)
        csrcolval  = sparse_preallocation[6] # Vector{Ti}(undef, length(I))
        csrnzval   = sparse_preallocation[7] # Vector{Tv}(undef, length(I))

        resize!(klasttouch, m3_rows)
        resize!(csrrowptr, m3_cols + 1)
        resize!(csrcolval, length(J))
        resize!(csrnzval, length(J))

        out = sparse!(J, I, V, m3_cols, m3_rows, +, klasttouch, csrrowptr, csrcolval, csrnzval, J, I, V)
        # out = sparse!(J, I, V, m3_cols, m3_rows)
    else
        klasttouch = sparse_preallocation[4] # Vector{Ti}(undef, n)
        csrrowptr  = sparse_preallocation[5] # Vector{Ti}(undef, m + 1)
        csrcolval  = sparse_preallocation[6] # Vector{Ti}(undef, length(I))
        csrnzval   = sparse_preallocation[7] # Vector{Tv}(undef, length(I))

        resize!(klasttouch, m3_cols)
        resize!(csrrowptr, m3_rows + 1)
        resize!(csrcolval, length(I))
        resize!(csrnzval, length(I))

        out = sparse!(I, J, V, m3_rows, m3_cols, +, klasttouch, csrrowptr, csrcolval, csrnzval, I, J, V)
        # out = sparse!(I, J, V, m3_rows, m3_cols)
    end

    # if reused_sparse_buffers
    #     out = copy(out)
    # end

    return out
end

# Fused  M * compressed_kron³(a)
# Computes the product without materializing the large mr₃×mc₃ intermediate.
# M is m × mr₃ sparse, a is n_rows × n_cols.  Output: m × mc₃ sparse.
# Row-outer / col-inner with sorted bounded ranges + direct IJV scatter.
# nzrange(M, row) checked once per row triple — skips ALL col iterations.
# Duplicate (I,J) entries resolved by sparse!(+).
function mul_compressed_kron³(M::SparseMatrixCSC, a::AbstractMatrix{T};
                    tol::AbstractFloat = eps(),
                    sparse_preallocation::Tuple{Vector{Int}, Vector{Int}, Vector{T}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{T}} = (Int[], Int[], T[], Int[], Int[], Int[], T[])) where T <: Real

    if typeof(a) <: DenseMatrix{T}
        â = a
    else
        â = convert(Matrix, a)
    end

    n_rows, n_cols = size(â)
    m = size(M, 1)
    m3_rows = n_rows * (n_rows + 1) * (n_rows + 2) ÷ 6
    m3_cols = n_cols * (n_cols + 1) * (n_cols + 2) ÷ 6

    size(M, 2) == m3_rows || throw(DimensionMismatch("M must have $m3_rows columns, got $(size(M, 2))"))

    rv_M = SparseArrays.rowvals(M)
    nzv_M = nonzeros(M)

    # Extract unique nonzero row/col indices directly from dense matrix
    ui, uj, lennz = dense_nz_structure(â)
    n_ui = length(ui)
    n_uj = length(uj)

    # --- sparse IJV buffer management ---
    if length(sparse_preallocation[1]) == 0
        estimated_nnz = floor(Int, max(m * m3_cols * (lennz / length(â)) ^ 4, 10000))
        resize!(sparse_preallocation[1], estimated_nnz)
        resize!(sparse_preallocation[2], estimated_nnz)
        resize!(sparse_preallocation[3], estimated_nnz)
    else
        estimated_nnz = length(sparse_preallocation[3])
        resize!(sparse_preallocation[1], estimated_nnz)
        resize!(sparse_preallocation[2], estimated_nnz)
        resize!(sparse_preallocation[3], estimated_nnz)
    end

    I = sparse_preallocation[1]
    J = sparse_preallocation[2]
    V = sparse_preallocation[3]
    k = 0

    # Row-outer loop: row triples (i1 ≥ j1 ≥ k1) with bounded index ranges
    for idx_i1 in 1:n_ui
        @inbounds i1 = ui[idx_i1]
        for idx_j1 in 1:idx_i1                 # j1 ≤ i1 by construction
            @inbounds j1 = ui[idx_j1]
            for idx_k1 in 1:idx_j1             # k1 ≤ j1 by construction
                @inbounds k1 = ui[idx_k1]

                row = (i1 - 1) * i1 * (i1 + 1) ÷ 6 + (j1 - 1) * j1 ÷ 2 + k1

                # nzrange checked ONCE per row triple — skips ALL col iterations
                rng_M = SparseArrays.nzrange(M, row)
                isempty(rng_M) && continue

                # Divisor depends only on row triple
                if i1 == j1
                    divisor = i1 == k1 ? 6 : 2
                else
                    divisor = (i1 ≠ k1 && j1 ≠ k1) ? 1 : 2
                end

                # Col-inner loop: column triples (i2 ≥ j2 ≥ k2) with bounded ranges
                for idx_i2 in 1:n_uj
                    @inbounds i2 = uj[idx_i2]
                    # Hoist i2-dependent reads
                    @inbounds aii = â[i1, i2]
                    @inbounds aji = â[j1, i2]
                    @inbounds aki = â[k1, i2]

                    for idx_j2 in 1:idx_i2
                        @inbounds j2 = uj[idx_j2]
                        # Hoist j2-dependent reads
                        @inbounds aij = â[i1, j2]
                        @inbounds ajj = â[j1, j2]
                        @inbounds akj = â[k1, j2]

                        # Precompute sub-expressions for the k2 inner loop
                        p1 = aii * ajj + aij * aji
                        p2 = aii * akj + aij * aki
                        p3 = aji * akj + ajj * aki
                        col_partial = (i2 - 1) * i2 * (i2 + 1) ÷ 6 + (j2 - 1) * j2 ÷ 2

                        for idx_k2 in 1:idx_j2
                            @inbounds k2 = uj[idx_k2]
                            @inbounds aik = â[i1, k2]
                            @inbounds ajk = â[j1, k2]
                            @inbounds akk = â[k1, k2]

                            val = akk * p1 + ajk * p2 + aik * p3

                            if abs(val) > tol
                                scaled_val = val / divisor
                                col = col_partial + k2

                                # Direct IJV scatter through M[:, row]
                                for p_M in rng_M
                                    k += 1
                                    if k > estimated_nnz
                                        estimated_nnz = k + max(1000, k ÷ 10)
                                        resize!(I, estimated_nnz)
                                        resize!(J, estimated_nnz)
                                        resize!(V, estimated_nnz)
                                    end
                                    I[k] = @inbounds rv_M[p_M]
                                    J[k] = col
                                    V[k] = @inbounds(nzv_M[p_M]) * scaled_val
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    resize!(I, k)
    resize!(J, k)
    resize!(V, k)

    # Sparse assembly — sparse!(+) resolves duplicate (I,J) entries
    klasttouch = sparse_preallocation[4]
    csrrowptr  = sparse_preallocation[5]
    csrcolval  = sparse_preallocation[6]
    csrnzval   = sparse_preallocation[7]

    resize!(klasttouch, m3_cols)
    resize!(csrrowptr, m + 1)
    resize!(csrcolval, length(I))
    resize!(csrnzval, length(I))

    out = if length(I) >= m3_cols + 1
        sparse!(I, J, V, m, m3_cols, +, klasttouch, csrrowptr, csrcolval, csrnzval, I, J, V)
    else
        SparseArrays.sparse(I, J, V, m, m3_cols)
    end

    if tol > 0
        droptol!(out, tol)
    end

    return out
end

function compressed_kron²(a::AbstractMatrix{T};
                    rowmask::Vector{Int} = Int[],
                    colmask::Vector{Int} = Int[],
                    tol::AbstractFloat = eps(),
                    sparse_preallocation::Tuple{Vector{Int}, Vector{Int}, Vector{T}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{T}} = (Int[], Int[], T[], Int[], Int[], Int[], T[])) where T <: Real

    a_is_adjoint = typeof(a) <: ℒ.Adjoint{T,Matrix{T}}
    reused_sparse_buffers = length(sparse_preallocation[1]) > 0

    if a_is_adjoint
        â = copy(a')

        rmask = colmask
        colmask = rowmask
        rowmask = rmask
    elseif typeof(a) <: DenseMatrix{T}
        â = copy(a)
    else
        â = convert(Matrix, a)  # Convert to dense matrix for faster access
    end

    # Get the number of rows and columns
    n_rows, n_cols = size(â)

    # Calculate the number of unique pair indices for rows and columns
    m2_rows = n_rows * (n_rows + 1) ÷ 2    # For rows: i ≤ j
    m2_cols = n_cols * (n_cols + 1) ÷ 2    # For columns: i ≤ j

    if rowmask == Int[0] || colmask == Int[0]
        if a_is_adjoint
            return spzeros(T, m2_cols, m2_rows)
        else
            return spzeros(T, m2_rows, m2_cols)
        end
    end

    # Initialize arrays to collect indices and values
    # Extract unique nonzero row/col indices directly from dense matrix
    ui, uj, lennz = dense_nz_structure(â)

    m2_c = length(colmask) > 0 ? length(colmask) : m2_cols
    m2_r = length(rowmask) > 0 ? length(rowmask) : m2_rows

    m2_exp = (length(colmask) > 0 || length(rowmask) > 0) ? 2 : 3

    if length(sparse_preallocation[1]) == 0
        estimated_nnz = floor(Int, max(m2_r * m2_c * (lennz / length(â)) ^ m2_exp, 10000))

        resize!(sparse_preallocation[1], estimated_nnz)
        resize!(sparse_preallocation[2], estimated_nnz)
        resize!(sparse_preallocation[3], estimated_nnz)

        I = sparse_preallocation[1]
        J = sparse_preallocation[2]
        V = sparse_preallocation[3]
    else
        estimated_nnz = length(sparse_preallocation[3])

        resize!(sparse_preallocation[1], estimated_nnz)
        resize!(sparse_preallocation[2], estimated_nnz)
        resize!(sparse_preallocation[3], estimated_nnz)

        I = sparse_preallocation[1]
        J = sparse_preallocation[2]
        V = sparse_preallocation[3]
    end

    k = 0


    norowmask = length(rowmask) == 0
    nocolmask = length(colmask) == 0
    rowmask_lookup = norowmask ? BitVector() : falses(m2_rows)
    colmask_lookup = nocolmask ? BitVector() : falses(m2_cols)

    if !norowmask && rowmask != Int[0]
        @inbounds for r in rowmask
            if 1 <= r <= m2_rows
                rowmask_lookup[r] = true
            end
        end
    end
    if !nocolmask && colmask != Int[0]
        @inbounds for c in colmask
            if 1 <= c <= m2_cols
                colmask_lookup[c] = true
            end
        end
    end

    n_ui = length(ui)
    n_uj = length(uj)

    for idx_i1 in 1:n_ui
        @inbounds i1 = ui[idx_i1]
        for idx_j1 in 1:idx_i1
            @inbounds j1 = ui[idx_j1]

            row = (i1 - 1) * i1 ÷ 2 + j1

            if norowmask || rowmask_lookup[row]
                divisor = i1 == j1 ? 2 : 1

                for idx_i2 in 1:n_uj
                    @inbounds i2 = uj[idx_i2]
                    @inbounds aii = â[i1, i2]
                    @inbounds aji = â[j1, i2]

                    for idx_j2 in 1:idx_i2
                        @inbounds j2 = uj[idx_j2]
                        @inbounds aij = â[i1, j2]
                        @inbounds ajj = â[j1, j2]

                        val = aii * ajj + aij * aji

                        if abs(val) > tol
                            col = (i2 - 1) * i2 ÷ 2 + j2

                            if nocolmask || colmask_lookup[col]
                                k += 1

                                if k > estimated_nnz
                                    estimated_nnz += Int(ceil(max(1000, estimated_nnz * .1)))
                                    estimated_nnz = min(m2_cols * m2_rows, estimated_nnz)
                                    resize!(I, estimated_nnz)
                                    resize!(J, estimated_nnz)
                                    resize!(V, estimated_nnz)
                                end

                                I[k] = row
                                J[k] = col
                                V[k] = val / divisor
                            end
                        end
                    end
                end
            end
        end
    end

    resize!(I, k)
    resize!(J, k)
    resize!(V, k)

    # Create the sparse matrix from the collected indices and values
    if a_is_adjoint
        klasttouch = sparse_preallocation[4]
        csrrowptr  = sparse_preallocation[5]
        csrcolval  = sparse_preallocation[6]
        csrnzval   = sparse_preallocation[7]

        resize!(klasttouch, m2_rows)
        resize!(csrrowptr, m2_cols + 1)
        resize!(csrcolval, length(J))
        resize!(csrnzval, length(J))

        out = sparse!(J, I, V, m2_cols, m2_rows, +, klasttouch, csrrowptr, csrcolval, csrnzval, J, I, V)
    else
        klasttouch = sparse_preallocation[4]
        csrrowptr  = sparse_preallocation[5]
        csrcolval  = sparse_preallocation[6]
        csrnzval   = sparse_preallocation[7]

        resize!(klasttouch, m2_cols)
        resize!(csrrowptr, m2_rows + 1)
        resize!(csrcolval, length(I))
        resize!(csrnzval, length(I))

        out = sparse!(I, J, V, m2_rows, m2_cols, +, klasttouch, csrrowptr, csrcolval, csrnzval, I, J, V)
    end

    # if reused_sparse_buffers
    #     out = copy(out)
    # end

    return out
end

# Detect unit roots from QME solution without computing eigenvalues.
# If sol has an eigenvalue near 1, (I - sol) is nearly singular.
# Uses LU factorization: exactly singular (info > 0) or smallest absolute pivot < tol.
# Cost: O(n³/3) LU on the small nPfm × nPfm solution matrix.
function detect_unit_roots_from_solution!(cache::caches, sol::AbstractMatrix{R};
                                            tol::Float64 = 1e-8) where R <: AbstractFloat
    n = size(sol, 1)
    n == 0 && return nothing
    ImA = similar(sol)
    @inbounds for j in 1:n, i in 1:n
        ImA[i, j] = ifelse(i == j, one(R), zero(R)) - sol[i, j]
    end
    F = ℒ.lu!(ImA; check = false)
    if F.info > 0
        cache.has_unit_roots = true
        return nothing
    end
    # Diagonal of packed LU factors = diagonal of U (L has unit diagonal).
    # Smallest absolute pivot indicates near-singularity ↔ eigenvalue of sol near 1.
    min_abs_pivot = typemax(R)
    @inbounds for i in 1:n
        v = abs(F.factors[i, i])
        if v < min_abs_pivot
            min_abs_pivot = v
        end
    end
    if min_abs_pivot < tol
        cache.has_unit_roots = true
    end
    return nothing
end


