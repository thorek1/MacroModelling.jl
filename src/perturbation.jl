@stable default_mode = "disable" begin

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
    # Legacy readable flow (before allocation-focused refactor):
    #   Q = qr!(∇₀[:, T.present_only_idx])
    #   A₊ = Q.Q' * ∇₊;  A₀ = Q.Q' * ∇₀;  A₋ = Q.Q' * ∇₋
    # Current code performs the same transforms using reusable QR/ORM workspaces.
    qr_factors, qr_ws = ensure_first_order_fast_qr_workspace!(qme_ws, ∇₀_present)
    Q = factorize_qr!(∇₀_present, qr_factors, qr_ws;
                        use_fastlapack_qr = use_fastlapack_qr)

    qme_ws.fast_qr_orm_ws_plus, qme_ws.fast_qr_orm_dims_plus = apply_qr_transpose_left!(A₊, ∇₊, Q,
                                                                                        qme_ws.fast_qr_orm_ws_plus,
                                                                                        qme_ws.fast_qr_orm_dims_plus,
                                                                                        qr_ws;
                                                                                        use_fastlapack_qr = use_fastlapack_qr)
    qme_ws.fast_qr_orm_ws_zero, qme_ws.fast_qr_orm_dims_zero = apply_qr_transpose_left!(A₀, ∇₀, Q,
                                                                                        qme_ws.fast_qr_orm_ws_zero,
                                                                                        qme_ws.fast_qr_orm_dims_zero,
                                                                                        qr_ws;
                                                                                        use_fastlapack_qr = use_fastlapack_qr)
    qme_ws.fast_qr_orm_ws_minus, qme_ws.fast_qr_orm_dims_minus = apply_qr_transpose_left!(A₋, ∇₋, Q,
                                                                                           qme_ws.fast_qr_orm_ws_minus,
                                                                                           qme_ws.fast_qr_orm_dims_minus,
                                                                                           qr_ws;
                                                                                           use_fastlapack_qr = use_fastlapack_qr)
    
    # end # timeit_debug
    # @timeit_debug timer "Sort matrices" begin

    Ã₊ = qme_ws.𝐀̃₊
    ℒ.mul!(Ã₊, @view(A₊[dynIndex,:]), @view(Ir[future_not_past_and_mixed_in_comb,:]))

    Ã₀ = qme_ws.𝐀̃₀
    copyto!(Ã₀, @view(A₀[dynIndex, comb]))

    Ã₋ = qme_ws.𝐀̃₋
    ℒ.mul!(Ã₋, @view(A₋[dynIndex,:]), @view(Ir[past_not_future_and_mixed_in_comb,:]))

    # end # timeit_debug
    # @timeit_debug timer "Quadratic matrix equation solve" begin

    sol, solved = solve_quadratic_matrix_equation(Ã₊, Ã₀, Ã₋, constants, workspaces, cache;
                                                    initial_guess = initial_guess,
                                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                                    use_fastlapack_lu = use_fastlapack_lu,
                                                    tol = opts.tol.first_order.qme,
                                                    verbose = opts.verbose)

    if !solved
        if opts.verbose println("Quadratic matrix equation solution failed.") end
        return zeros(R, T.nVars, T.nPast_not_future_and_mixed + T.nExo), sol, false
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
        return zeros(R, T.nVars, T.nPast_not_future_and_mixed + T.nExo), sol, false
    end

    # A    = vcat(-(Ā̂₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ)), sol)
    if T.nPresent_only > 0
        ℒ.mul!(A₋ᵤ, Ã₀ᵤ, @view(sol[:,past_not_future_and_mixed_in_comb]), 1, 1)
        nₚ₋ = qme_ws.𝐧ₚ₋
        ℒ.mul!(nₚ₋, A₊ᵤ, 𝐃)
        ℒ.mul!(A₋ᵤ, nₚ₋, L, 1, 1)
        solve_lu_left!(Ā₀ᵤ, A₋ᵤ, qme_ws.fast_lu_ws_a0u, Ā̂₀ᵤ;
                       use_fastlapack_lu = use_fastlapack_lu)
        ℒ.rmul!(A₋ᵤ, -1)
    end

    A = qme_ws.𝐀
    # Legacy readable flow:
    #   A = vcat(A₋ᵤ, sol_compact)[T.reorder, :]
    # Expanded loop below writes into preallocated `A` without temporary concatenation.
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
    # Legacy readable flow:
    #   M = A[T.future_not_past_and_mixed_idx, :] * expand_past
    #   ∇₀ = ∇₁[:, 1:T.nFuture_not_past_and_mixed] * M + ∇₀
    ℒ.mul!(M, @view(A[T.future_not_past_and_mixed_idx,:]), idx_constants.expand_past)

    ℒ.mul!(∇₀, @view(∇₁[:,1:T.nFuture_not_past_and_mixed]), M, 1, 1)

    qme_ws.fast_lu_ws_nabla0, qme_ws.fast_lu_dims_nabla0, solved_∇₀, C = factorize_lu!(∇₀,
                                                                                         qme_ws.fast_lu_ws_nabla0,
                                                                                         qme_ws.fast_lu_dims_nabla0;
                                                                                         use_fastlapack_lu = use_fastlapack_lu)

    if !solved_∇₀
        if opts.verbose println("Factorisation of ∇₀ failed") end
        return zeros(R, T.nVars, T.nPast_not_future_and_mixed + T.nExo), sol, false
    end

    solve_lu_left!(∇₀, ∇ₑ, qme_ws.fast_lu_ws_nabla0, C;
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
        cache.valid_for.first_order_solution = Float64.(parameter_values)
    end

    return S₁, sol, true
end


function calculate_second_order_solution(∇₁::AbstractMatrix{S}, #first order derivatives
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

    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * M₂.𝐈ₙ₋ - ∇₁[:,range(1,n) .+ n₊]

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

    ∇₁₊ = @views ∇₁[:,1:n₊] * M₂.𝐈ₙ₊

    A = ∇₁₊𝐒₁➕∇₁₀lu \ ∇₁₊
    
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
    
    C = ∇₁₊𝐒₁➕∇₁₀lu \ ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹

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

    if solved && caching
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
            cache.valid_for.second_order_solution = Float64.(parameter_values)
            cache.valid_for.pruned_second_order_solution = Float64[]
        end
    end

    return 𝐒₂, solved
end


function calculate_third_order_solution(∇₁::AbstractMatrix{S}, #first order derivatives
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
    
    ∇₁₊ = @views ∇₁[:,1:n₊] * M₂.𝐈ₙ₊

    A = ∇₁₊𝐒₁➕∇₁₀lu \ ∇₁₊

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

    C = ∇₁₊𝐒₁➕∇₁₀lu \ 𝐗₃# * M₃.𝐂₃

    # end # timeit_debug
    # end # timeit_debug
    # @timeit_debug timer "Solve sylvester equation" begin

    𝐒₃, solved = solve_sylvester_equation(A, B, C, ℂ.sylvester_workspace,
                                            initial_guess = initial_guess_sylv,
                                            sylvester_algorithm = opts.sylvester_algorithm³,
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

    if solved && caching
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
            cache.valid_for.third_order_solution = Float64.(parameter_values)
            cache.valid_for.pruned_third_order_solution = Float64[]
        end
    end

    return 𝐒₃, solved
end

end # dispatch_doctor

